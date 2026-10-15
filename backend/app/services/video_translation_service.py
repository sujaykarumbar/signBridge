"""
SignBridge Video Translation Service
===================================
Direct video file translation for continuous Indian Sign Language (ISL) video clips
(social media reels, instructional signing, dynamic multi-sign sentences).

Pipeline:
1. Frame Preprocessing & Subsampling:
   - Samples up to 12 evenly spaced frames, matching sentence-model training.
   - Keeps full frames for landmark detection, matching sentence-model training.
2. Motion & Boundary Segmentation:
   - SignBoundaryDetector separates holds vs. stroke phases in continuous signing.
3. Multimodal Vision Fallback:
   - Gemini ISL interprets continuous multi-sign sentences following ISL SOV->SVO grammar rules.
"""

import os
import cv2
import logging
import numpy as np

from app.services.keyframe_sampler import mask_lower_third_and_overlays

logger = logging.getLogger(__name__)

MODEL_FRAME_COUNT = 12


class VideoTranslationService:
    def translate_video_file(self, video_path: str) -> dict:
        """
        Process an ISL video file and return full English sentence translation
        with detected glosses, confidence, and sign type.
        """
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}

        cap = cv2.VideoCapture(video_path)
        src_fps = 25.0
        duration = 0.0
        if cap.isOpened():
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = round(total_frames / src_fps, 2) if src_fps > 0 else 0.0
        else:
            cap.release()
            return {"error": "Failed to open video file"}
        cap.release()

        if total_frames <= 0:
            return {"error": "No frames found in video"}

        sampled_indices = np.linspace(
            0,
            total_frames - 1,
            min(MODEL_FRAME_COUNT, total_frames),
            dtype=int,
        )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Failed to open video file"}

        sampled_raw = []
        try:
            for frame_index in sampled_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ret, frame = cap.read()
                if ret:
                    sampled_raw.append(frame)
        finally:
            cap.release()

        if not sampled_raw:
            return {"error": "No frames extracted from video"}

        logger.info(
            "[Video Translation] Sampled %d of %d frames (%.1fs @ %.1f FPS)",
            len(sampled_raw), total_frames, duration, src_fps
        )

        vision_jpegs = []
        for frame_bgr in sampled_raw:
            h, w = frame_bgr.shape[:2]
            target_dim = 720
            if max(h, w) > target_dim:
                scale = target_dim / max(h, w)
                resized = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                resized = frame_bgr

            masked = mask_lower_third_and_overlays(
                resized,
                lower_third_ratio=0.25,
            )
            masked_ret, masked_buf = cv2.imencode(
                ".jpg",
                masked,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if masked_ret:
                vision_jpegs.append(masked_buf.tobytes())

        if not vision_jpegs:
            return {"error": "Could not decode video frames"}

        # Only send sampled, caption-masked images: this excludes both audio transcripts
        # and the burned-in lower-third text from Gemini's translation input.
        translation_error = "Gemini API key is not configured."
        try:
            from app.services.gemini_isl import GeminiISLService
            gemini = GeminiISLService()
            if gemini.is_available():
                logger.info(
                    "[Video Translation] Sending %d caption-masked frames for visual ISL translation.",
                    len(vision_jpegs),
                )
                gem_res = gemini.predict_from_frames(vision_jpegs)
                if "error" not in gem_res and (
                    gem_res.get("translated_sentence") or gem_res.get("sentence")
                ):
                    gem_res["duration_sec"] = duration
                    gem_res["total_frames_analyzed"] = len(vision_jpegs)
                    gem_res["source_fps"] = src_fps
                    return gem_res
                translation_error = gem_res.get("error", "Gemini returned no translation")
                logger.warning("[Video Translation] Gemini visual request returned: %s", translation_error)
        except Exception as g_err:
            translation_error = "Gemini visual request failed"
            logger.warning("[Video Translation] Gemini visual request failed: %s", g_err)

        return {
            "error": (
                f"Visual ISL translation is unavailable: {translation_error.rstrip('. ')}. "
                "Check the Gemini API configuration, quota, and network connection, then try again."
            )
        }
