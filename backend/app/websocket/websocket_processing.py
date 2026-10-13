"""
SignBridge WebSocket Processing
================================

Contains CPU/model-heavy processing used by the WebSocket handler:

- MediaPipe initialization
- MediaPipe landmark drawing
- Frame preprocessing
- Model inference
- Optional test-video generation

This module does not know about Flask or WebSocket connections.

MediaPipe compatibility
-----------------------
Written for mediapipe 1.x (also works on late 0.10.x releases).

The legacy ``mp.solutions`` package and the protobuf classes under
``mediapipe.framework`` no longer ship in current wheels, so landmark drawing
now uses the Tasks drawing API:

    mediapipe.tasks.python.vision.drawing_utils
    mediapipe.tasks.python.vision.drawing_styles
    mediapipe.tasks.python.vision.PoseLandmarksConnections
    mediapipe.tasks.python.vision.HandLandmarksConnections

Landmarks from the detectors are passed to the drawing function directly;
no ``NormalizedLandmarkList`` protobuf conversion is needed any more.

Model files
-----------
The pose and hand ``.task`` models are looked up by absolute path, so the
server no longer depends on the directory it was started from. Search order:

    1. POSE_LANDMARKER_MODEL / HAND_LANDMARKER_MODEL   (full path to the file)
    2. MEDIAPIPE_MODEL_DIR                             (directory, optional)
    3. the folder containing this file, then its ``models/`` subfolder
    4. the current working directory, then its ``models/`` subfolder

If a model cannot be found the server keeps running with landmarks disabled
and logs exactly which paths were checked.
"""

import logging
import os
import time

import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80


@dataclass(slots=True)
class BufferedFrame:
    """A 224x224 frame plus its JPEG bytes, encoded once."""
    image: Image.Image
    jpeg: bytes
    features_134: np.ndarray | None = None
    hand_landmarks: list | None = None


def encode_jpeg(image: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = BytesIO()
    try:
        image.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    finally:
        buf.close()


def _as_images(frames: list) -> list:
    return [f.image if isinstance(f, BufferedFrame) else f for f in frames]


def _as_jpegs(frames: list) -> list:
    return [f.jpeg if isinstance(f, BufferedFrame) else f for f in frames]


# Rendering behaviour

# The pre-migration code copied only x/y/z into a protobuf, which dropped
# each landmark's visibility/presence scores, so EVERY detected joint was
# always drawn - even occluded or off-frame ones.
#
# The new drawing API honours visibility/presence and would silently skip
# joints below its threshold. That changes the pixels the model receives, so
# by default we keep the old behaviour by stripping the scores.
#
# Set to True only if the model is (re)trained on frames rendered this way.
RESPECT_LANDMARK_VISIBILITY = False


# MediaPipe

_ENABLE_MEDIAPIPE = os.environ.get("ENABLE_MEDIAPIPE", "1") == "1"

mp = None
mp_python = None
mp_vision = None
mp_drawing = None   # mediapipe.tasks.python.vision.drawing_utils
mp_styles = None    # mediapipe.tasks.python.vision.drawing_styles
mp_landmark = None  # mediapipe.tasks.python.components.containers.landmark
MEDIAPIPE_OK = False

# Filled in once MediaPipe imports successfully.
POSE_CONNECTIONS = None
HAND_CONNECTIONS = None
POSE_LANDMARK_STYLE = None
HAND_LANDMARK_STYLE = None
HAND_CONNECTION_STYLE = None


if _ENABLE_MEDIAPIPE:
    try:
        import mediapipe as mp
        try:
            from mediapipe.tasks.python.vision import drawing_styles as mp_styles
            from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
            from mediapipe.tasks.python import vision as mp_vision
            POSE_CONNECTIONS = mp_vision.PoseLandmarksConnections.POSE_LANDMARKS
            HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
        except (ImportError, AttributeError):
            from mediapipe.python.solutions import drawing_styles as mp_styles
            from mediapipe.python.solutions import drawing_utils as mp_drawing
            POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
            HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        try:
            from mediapipe.tasks.python.components.containers import (
                landmark as mp_landmark,
            )
        except (ImportError, AttributeError):
            mp_landmark = None

        POSE_LANDMARK_STYLE = mp_styles.get_default_pose_landmarks_style()
        HAND_LANDMARK_STYLE = mp_styles.get_default_hand_landmarks_style()
        HAND_CONNECTION_STYLE = mp_styles.get_default_hand_connections_style()

        MEDIAPIPE_OK = True

        logger.info(
            "[MediaPipe] Loaded successfully (version %s).",
            getattr(mp, "__version__", "unknown"),
        )

    except Exception as e:
        logger.warning(
            "[MediaPipe] Init failed. "
            "Landmarks will be disabled: %s",
            e,
        )

else:
    logger.info(
        "[MediaPipe] Disabled via ENABLE_MEDIAPIPE environment variable."
    )


# Model files

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

POSE_LANDMARKER_PATH = MODELS_DIR / "pose_landmarker_full.task"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"

POSE_MODEL_ENV = "POSE_LANDMARKER_MODEL"
HAND_MODEL_ENV = "HAND_LANDMARKER_MODEL"
MODEL_DIR_ENV = "MEDIAPIPE_MODEL_DIR"


def _model_search_dirs() -> list:
    """Directories searched for the .task files, in priority order."""

    dirs = []

    configured = os.environ.get(MODEL_DIR_ENV)

    if configured:
        dirs.append(os.path.abspath(os.path.expanduser(configured)))

    cwd = os.getcwd()

    dirs += [
        _MODULE_DIR,
        os.path.join(_MODULE_DIR, "models"),
        cwd,
        os.path.join(cwd, "models"),
    ]

    # Remove duplicates but keep the order.
    return list(dict.fromkeys(dirs))


def resolve_model_path(filename: str, env_var: str):
    """
    Locate a MediaPipe model file.

    Returns:
        Absolute path to the file, or None if it cannot be found.
        If ``env_var`` is set it is the only location checked.
    """

    explicit = os.environ.get(env_var)

    if explicit:
        explicit = os.path.abspath(os.path.expanduser(explicit))

        return explicit if os.path.isfile(explicit) else None

    for directory in _model_search_dirs():

        candidate = os.path.join(directory, filename)

        if os.path.isfile(candidate):
            return candidate

    return None


def _describe_missing(filename: str, env_var: str) -> str:
    """Human-readable reason a model file was not found."""

    explicit = os.environ.get(env_var)

    if explicit:
        return f"{filename} ({env_var}={explicit} does not exist)"

    return filename


def _log_missing_models(missing: list):
    """Log one clear error describing which models are missing."""

    searched = ", ".join(_model_search_dirs())

    logger.error(
        "MediaPipe model file(s) not found: %s. "
        "Landmarks are DISABLED (the model will receive frames without "
        "landmark overlays). Searched: %s. "
        "Place the files in one of those folders, or set %s / %s "
        "(file paths) or %s (folder).",
        ", ".join(str(path) for path in missing),
        searched,
        POSE_MODEL_ENV,
        HAND_MODEL_ENV,
        MODEL_DIR_ENV,
    )


# MediaPipe landmarkers

def build_landmarkers():
    """
    Create MediaPipe pose and hand landmark detectors.

    Returns:
        tuple:
            (pose_detector, hand_detector)

        If MediaPipe is unavailable or initialization fails:
            (None, None)
    """

    if not MEDIAPIPE_OK:
        logger.info(
            "MediaPipe not available. "
            "Landmarks will be disabled."
        )
        return None, None

    pose_path = resolve_model_path(POSE_LANDMARKER_PATH, POSE_MODEL_ENV)
    hand_path = resolve_model_path(HAND_LANDMARKER_PATH, HAND_MODEL_ENV)

    missing = [
        _describe_missing(name, env_var)
        for name, env_var, path in (
            (POSE_LANDMARKER_PATH, POSE_MODEL_ENV, pose_path),
            (HAND_LANDMARKER_PATH, HAND_MODEL_ENV, hand_path),
        )
        if path is None
    ]

    if missing:
        try:
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            hand = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            logger.info("[MediaPipe] Using built-in mp.solutions (pose + hands).")
            return pose, hand
        except Exception:
            _log_missing_models(missing)
            return None, None

    pose = None

    try:
        pose = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=pose_path
                )
            )
        )

        hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=hand_path
                ),
                num_hands=2,
            )
        )

        return pose, hand

    except Exception:
        logger.exception("Landmarker initialization failed.")

        # Don't leak the pose detector if the hand detector failed.
        if pose is not None:
            try:
                pose.close()
            except Exception:
                logger.exception(
                    "Failed to close pose detector after init failure."
                )

        return None, None


# Landmark processing

def _drawable(landmarks):
    """
    Return landmarks in the form passed to the drawing function.

    Unless RESPECT_LANDMARK_VISIBILITY is set, visibility/presence are
    dropped so that every joint is drawn (the pre-migration behaviour).
    """

    if RESPECT_LANDMARK_VISIBILITY:
        return landmarks
    
    return [
        mp_landmark.NormalizedLandmark(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z,
        )
        for landmark in landmarks
    ]


def apply_landmarks(
    image: Image.Image,
    pose_detector,
    hand_detector,
) -> tuple[Image.Image, np.ndarray, list]:
    """
    Run MediaPipe pose/hand detection, draw landmarks, and extract feature vectors.

    Returns:
        (annotated_image, features_134, hand_landmarks)
    """

    empty_features = np.zeros(134, dtype=np.float32)
    if pose_detector is None and hand_detector is None:
        return image, empty_features, []

    img_rgb = np.array(image)

    # Built-in mp.solutions detector (Pose / Hands) - only if not Tasks detector (.detect)
    is_tasks = (pose_detector and hasattr(pose_detector, "detect")) or (hand_detector and hasattr(hand_detector, "detect"))
    if not is_tasks and (hasattr(pose_detector, "process") or hasattr(hand_detector, "process")):
        pose_result = pose_detector.process(img_rgb) if pose_detector and hasattr(pose_detector, "process") else None
        hand_result = hand_detector.process(img_rgb) if hand_detector and hasattr(hand_detector, "process") else None

        has_pose = bool(pose_result and pose_result.pose_landmarks)
        has_hands = bool(hand_result and hand_result.multi_hand_landmarks)
        feats = _extract_134_features(pose_result, hand_result)
        raw_hands = list(hand_result.multi_hand_landmarks) if has_hands else []

        if not has_pose and not has_hands:
            del img_rgb
            return image, feats, raw_hands

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        del img_rgb

        if has_pose:
            mp_drawing.draw_landmarks(
                img_bgr,
                pose_result.pose_landmarks,
                POSE_CONNECTIONS,
                POSE_LANDMARK_STYLE,
            )

        if has_hands:
            for landmarks in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_bgr,
                    landmarks,
                    HAND_CONNECTIONS,
                    HAND_LANDMARK_STYLE,
                    HAND_CONNECTION_STYLE,
                )

        result_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        del img_bgr
        return result_image, feats, raw_hands

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
    )

    pose_result = (
        pose_detector.detect(mp_image)
        if pose_detector
        else None
    )

    hand_result = (
        hand_detector.detect(mp_image)
        if hand_detector
        else None
    )

    no_pose = (
        not pose_result
        or not pose_result.pose_landmarks
    )

    no_hands = (
        not hand_result
        or not hand_result.hand_landmarks
    )

    feats = np.zeros(134, dtype=np.float32)
    if pose_result and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
        for i, lm in enumerate(pose_result.pose_landmarks[0][:25]):
            feats[i * 2] = lm.x
            feats[i * 2 + 1] = lm.y
    raw_hands = []
    if hand_result and hand_result.hand_landmarks:
        raw_hands = list(hand_result.hand_landmarks)
        for h_idx, hand in enumerate(hand_result.hand_landmarks[:2]):
            base = 50 + (h_idx * 42)
            for i, lm in enumerate(hand[:21]):
                feats[base + i * 2] = lm.x
                feats[base + i * 2 + 1] = lm.y

    if no_pose and no_hands:
        del img_rgb
        del mp_image

        return image, feats, raw_hands

    # Convert RGB -> BGR for OpenCV drawing.
    img_bgr = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2BGR,
    )

    del img_rgb
    del mp_image

    # Pose landmarks
    if pose_result and pose_result.pose_landmarks:
        for landmarks in pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                img_bgr,
                _drawable(landmarks),
                POSE_CONNECTIONS,
                POSE_LANDMARK_STYLE,
            )

    # Hand landmarks
    if hand_result and hand_result.hand_landmarks:
        for landmarks in hand_result.hand_landmarks:
            mp_drawing.draw_landmarks(
                img_bgr,
                _drawable(landmarks),
                HAND_CONNECTIONS,
                HAND_LANDMARK_STYLE,
                HAND_CONNECTION_STYLE,
            )

    # Convert back to PIL RGB.
    result_image = Image.fromarray(
        cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB,
        )
    )

    del img_bgr

    return result_image, feats, raw_hands


# Async landmark wrapper

def process_frame(
    raw_image: Image.Image,
    pose_detector,
    hand_detector,
    landmarks_enabled: bool,
    resize_dim: int = 224,
    jpeg_quality: int = JPEG_QUALITY,
) -> BufferedFrame:
    """
    Process one incoming frame (runs inside the landmark ThreadPoolExecutor).

    Draw landmarks at the original resolution, resize, extract landmarks, and JPEG-encode once.
    """
    try:
        if landmarks_enabled:
            image, feats, h_lms = apply_landmarks(raw_image, pose_detector, hand_detector)
        else:
            image, feats, h_lms = raw_image, np.zeros(134, dtype=np.float32), []

        if image.size != (resize_dim, resize_dim):
            image = image.resize((resize_dim, resize_dim))

        return BufferedFrame(
            image=image,
            jpeg=encode_jpeg(image, jpeg_quality),
            features_134=feats,
            hand_landmarks=h_lms,
        )

    finally:
        del raw_image


# Test video generation

def save_test_video(
    frames: list,
    output_dir: str = "temp_videos",
) -> str:
    """
    Save PIL frames as a local MP4.

    Used only when SAVE_TEST_VIDEOS=1.
    """

    try:
        if not frames:
            return ""

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        timestamp = int(time.time() * 1000)

        filepath = os.path.abspath(
            os.path.join(
                output_dir,
                f"test_clip_{timestamp}.mp4",
            )
        )

        width, height = frames[0].size

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            filepath,
            fourcc,
            15.0,
            (width, height),
        )

        try:
            for frame in frames:

                frame_array = np.array(frame)

                frame_bgr = cv2.cvtColor(
                    frame_array,
                    cv2.COLOR_RGB2BGR,
                )

                out.write(frame_bgr)

                del frame_array
                del frame_bgr

        finally:
            out.release()

        logger.info(
            "Saved test video locally: %s",
            filepath,
        )

        return filepath

    except Exception:
        logger.exception(
            "Failed to save test video locally."
        )

        return ""


def _extract_134_features(pose_result, hand_result) -> np.ndarray:
    feats = np.zeros(134, dtype=np.float32)
    if pose_result and hasattr(pose_result, "pose_landmarks") and pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks.landmark[:25]):
            feats[i * 2] = lm.x
            feats[i * 2 + 1] = lm.y
    if hand_result and hasattr(hand_result, "multi_hand_landmarks") and hand_result.multi_hand_landmarks:
        for h_idx, hand in enumerate(hand_result.multi_hand_landmarks[:2]):
            base = 50 + (h_idx * 42)
            for i, lm in enumerate(hand.landmark[:21]):
                feats[base + i * 2] = lm.x
                feats[base + i * 2 + 1] = lm.y
    return feats


# Model inference

def run_inference(
    frames: list,
    mode: str,
    model_api,
    save_test_videos: bool = False,
) -> dict:
    """
    Run model inference for a buffered sequence.

    Supported modes:

        frames
        video
        hybrid
    """

    t0 = time.time()

    try:

        if save_test_videos:
            save_test_video(_as_images(frames))

        # 1. Local ISL Alphabet / Digits classifier (<1ms)
        # Scan buffered frames for detected hand landmarks
        try:
            from app.services.isl_alphabet_service import ISLAlphabetService
            _alpha_svc = ISLAlphabetService()
            if _alpha_svc.is_ready():
                best_alpha = None
                for f in reversed(frames):
                    if isinstance(f, BufferedFrame) and f.hand_landmarks:
                        a_res = _alpha_svc.predict_landmarks(f.hand_landmarks)
                        if a_res.get("confidence", 0.0) >= 0.50:
                            if best_alpha is None or a_res["confidence"] > best_alpha["confidence"]:
                                best_alpha = a_res
                                if best_alpha["confidence"] >= 0.75:
                                    break
                if best_alpha and best_alpha.get("confidence", 0.0) >= 0.50:
                    logger.info(
                        "[ISL Alphabet] Local match: %s (%.0f%%)",
                        best_alpha.get("prediction"),
                        best_alpha.get("confidence", 0.0) * 100,
                    )
                    best_alpha["total_latency_ms"] = round((time.time() - t0) * 1000, 2)
                    return best_alpha
        except Exception as a_err:
            logger.debug("ISL alphabet check skipped: %s", a_err)

        # 2. Local AI4Bharat INCLUDE-50 LSTM model (<15ms)
        try:
            from app.services.include_classifier_service import INCLUDEClassifierService
            _include_svc = INCLUDEClassifierService()
            if _include_svc.is_ready():
                lm_seq = [f.features_134 for f in frames if isinstance(f, BufferedFrame) and f.features_134 is not None]
                if lm_seq:
                    lm_arr = np.array(lm_seq, dtype=np.float32)
                    if np.count_nonzero(lm_arr) > 40:
                        inc_res = _include_svc.predict_sequence(lm_arr)
                        if inc_res.get("confidence", 0.0) >= 0.40:
                            logger.info(
                                "[INCLUDE Classifier] Local match: %s (%.0f%%)",
                                inc_res.get("prediction"),
                                inc_res.get("confidence", 0.0) * 100,
                            )
                            inc_res["total_latency_ms"] = round((time.time() - t0) * 1000, 2)
                            return inc_res
        except Exception as inc_err:
            logger.debug("INCLUDE inference check skipped: %s", inc_err)

        # 3. Gemini Multimodal Vision (Translates continuous ISL sentences & gestures from frames)
        use_gemini = os.environ.get("USE_GEMINI", "1") == "1" or mode == "gemini"
        if use_gemini and os.environ.get("GEMINI_API_KEY"):
            try:
                from app.services.gemini_isl import GeminiISLService
                gemini_service = GeminiISLService()
                gemini_res = gemini_service.predict_from_frames(_as_jpegs(frames))
                if "error" not in gemini_res and (gemini_res.get("prediction") or gemini_res.get("sentence")):
                    if gemini_res.get("confidence", 0.0) >= 0.50:
                        logger.info(
                            "[Gemini ISL] Recognized: %s | Sentence: '%s' (%.0f%%)",
                            gemini_res.get("prediction"),
                            gemini_res.get("sentence", ""),
                            gemini_res.get("confidence", 0.0) * 100,
                        )
                        gemini_res["total_latency_ms"] = round((time.time() - t0) * 1000, 2)
                        return gemini_res
            except Exception as g_err:
                logger.warning("Gemini Vision check skipped: %s", g_err)

        # 4. Swin3D remote API (Fallback - require >= 0.50 confidence to filter random noise)
        if mode == "frames":
            result = model_api.predict_from_frames(_as_jpegs(frames))
        elif mode == "video":
            result = model_api.predict(_as_images(frames))
        else:
            result = model_api.predict_from_frames(_as_jpegs(frames))
            if "error" in result:
                result = model_api.predict(_as_images(frames))

        # Reject low-confidence hallucinations from Swin3D (< 0.50)
        if result.get("confidence", 0.0) < 0.50:
            result["prediction"] = ""
            result["confidence"] = 0.0

        result["total_latency_ms"] = round(
            (time.time() - t0) * 1000,
            2,
        )

        return result

    finally:
        # Release the frames list held by this worker.
        del frames