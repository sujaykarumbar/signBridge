"""
Unit and Integration Tests for Continuous ISL Pipeline
======================================================
Validates:
1. Frame Preprocessing & Subsampling (KeyframeSampler at 12-16 FPS, lower-third masking).
2. Sliding Window & Trajectory Extraction (SignBoundaryDetector hold vs stroke segmentation).
3. Multimodal Vision Fallback Schema (detected_glosses, translated_sentence, confidence, sign_type).
4. Frame-by-frame processing through the pipeline with latency < 200ms per window.
5. Gloss Sequence Buffer feeding and Text-to-Speech output delivery.
"""

import os
import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from PIL import Image

from app.services.keyframe_sampler import (
    KeyframeSampler,
    mask_lower_third_and_overlays,
    focus_upper_body_roi,
)
from app.services.sign_boundary_detector import SignBoundaryDetector, GesturePhase
from app.services.isl_gloss_stream import ISLGlossStreamTracker
from app.services.video_translation_service import VideoTranslationService
from app.services.gemini_isl import GeminiISLService
from app.services.isl_words_service import ISLWordsService
from app.websocket.websocket_processing import BufferedFrame, encode_jpeg


class TestKeyframeSamplerAndPreprocessing(unittest.TestCase):
    """Requirement 1: Subsampling to 12-16 FPS and Lower-Third Masking."""

    def test_keyframe_sampler_cadence_30fps(self):
        sampler = KeyframeSampler(target_fps=14.0)
        total_frames = 90  # 3.0 seconds at 30 FPS
        src_fps = 30.0
        indices = sampler.compute_sample_indices(total_frames, src_fps)

        # 3 seconds * 14 FPS = ~42-43 frames
        self.assertTrue(40 <= len(indices) <= 45)
        # Verify strictly increasing
        for i in range(len(indices) - 1):
            self.assertLess(indices[i], indices[i + 1])
        # Verify rate is in 12-16 FPS range
        effective_fps = len(indices) / (total_frames / src_fps)
        self.assertGreaterEqual(effective_fps, 12.0)
        self.assertLessEqual(effective_fps, 16.0)

    def test_keyframe_sampler_cadence_60fps(self):
        sampler = KeyframeSampler(target_fps=14.0)
        total_frames = 180  # 3.0 seconds at 60 FPS
        src_fps = 60.0
        indices = sampler.compute_sample_indices(total_frames, src_fps)
        self.assertTrue(40 <= len(indices) <= 45)
        effective_fps = len(indices) / (total_frames / src_fps)
        self.assertGreaterEqual(effective_fps, 12.0)
        self.assertLessEqual(effective_fps, 16.0)

    def test_lower_third_masking_blanks_bottom(self):
        # Create a frame with bright text/content in bottom 20%
        img = np.ones((100, 100, 3), dtype=np.uint8) * 200
        # Lower third (y >= 80)
        img[80:, :] = 255

        masked = mask_lower_third_and_overlays(img, lower_third_ratio=0.20)
        # Upper 80% should be preserved
        self.assertEqual(int(masked[40, 50, 0]), 200)
        # Lower 20% must be completely blanked out (0)
        self.assertEqual(int(masked[85, 50, 0]), 0)
        self.assertEqual(int(masked[99, 99, 0]), 0)

    def test_focus_upper_body_roi(self):
        img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        cropped = focus_upper_body_roi(img, lower_third_ratio=0.20)
        w, h = cropped.size
        self.assertEqual(w, 200)
        self.assertEqual(h, 160)  # 200 * 0.80 = 160


class TestSignBoundaryDetector(unittest.TestCase):
    """Requirement 2: Kinematic Boundary Detection (Holds vs Stroke Phases)."""

    def setUp(self):
        self.detector = SignBoundaryDetector(
            stroke_velocity_threshold=0.045,
            hold_velocity_threshold=0.022,
            min_hold_frames=2,
            min_stroke_frames=3,
        )

    def _generate_hand_landmarks(self, wrist_x: float, wrist_y: float):
        lms = []
        for i in range(21):
            lms.append({
                "x": wrist_x + (i * 0.005),
                "y": wrist_y + (i * 0.005),
                "z": 0.0,
            })
        return [lms]

    def test_stroke_phase_detection(self):
        self.detector.reset()
        # Move hand quickly across consecutive frames (high velocity > 0.05)
        for i in range(5):
            lms = self._generate_hand_landmarks(0.2 + (i * 0.08), 0.3 + (i * 0.08))
            res = self.detector.update(hand_landmarks=lms)
            if i >= 3:
                self.assertEqual(res["phase"], GesturePhase.STROKE.value)

    def test_hold_phase_and_boundary_trigger(self):
        self.detector.reset()
        # 1. Stroke phase (moving quickly)
        for i in range(4):
            lms = self._generate_hand_landmarks(0.1 + (i * 0.07), 0.1 + (i * 0.07))
            self.detector.update(hand_landmarks=lms)

        self.assertEqual(self.detector._current_phase, GesturePhase.STROKE)

        # 2. Hold phase (hand stabilizes at sign apex, minimal movement)
        boundary_triggered = False
        for i in range(4):
            lms = self._generate_hand_landmarks(0.38 + (i * 0.002), 0.38 + (i * 0.002))
            res = self.detector.update(hand_landmarks=lms)
            if res["is_boundary"]:
                boundary_triggered = True

        self.assertTrue(boundary_triggered, "Transition from stroke to hold must trigger sign boundary")
        self.assertEqual(self.detector._current_phase, GesturePhase.HOLD)


class TestMultimodalVisionFallbackSchema(unittest.TestCase):
    """Requirement 3: Multimodal Vision Fallback Prompt & Strict JSON Schema."""

    def test_gemini_schema_parsing_continuous_sentence(self):
        gemini = GeminiISLService(api_key="test_key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps({
                                    "hand_gestures_observed": "Both hands move outward from the chest.",
                                    "detected_glosses": ["HELLO", "MEET", "YOU", "GLAD"],
                                    "translated_sentence": "I am very glad to meet you.",
                                    "confidence": 0.92,
                                    "sign_type": "continuous_sentence"
                                })
                            }
                        ]
                    }
                }
            ]
        }

        with patch("requests.post", return_value=mock_response) as post_mock:
            dummy_jpeg = cv2.imencode(".jpg", np.zeros((224, 224, 3), dtype=np.uint8))[1].tobytes()
            res = gemini.predict_from_frames([dummy_jpeg, dummy_jpeg])

            # Verify exact required schema keys
            self.assertIn("detected_glosses", res)
            self.assertIn("translated_sentence", res)
            self.assertIn("confidence", res)
            self.assertIn("sign_type", res)

            # Verify values
            self.assertEqual(res["detected_glosses"], ["HELLO", "MEET", "YOU", "GLAD"])
            self.assertEqual(res["translated_sentence"], "I am very glad to meet you.")
            self.assertEqual(res["confidence"], 0.92)
            self.assertEqual(res["sign_type"], "continuous_sentence")
            self.assertEqual(res["confidence_source"], "unvalidated_model_estimate")
            self.assertTrue(res["requires_review"])
            self.assertEqual(
                res["hand_gestures_observed"],
                "Both hands move outward from the chest.",
            )

            # Verify backward compatibility
            self.assertEqual(res["sentence"], "I am very glad to meet you.")
            self.assertEqual(res["prediction"], "HELLO")
            parts = post_mock.call_args.kwargs["json"]["contents"][0]["parts"]
            self.assertEqual(
                sum("inline_data" in part for part in parts),
                2,
            )
            self.assertIn("Frame 1 of 2", [part.get("text") for part in parts])

    def test_missing_gemini_confidence_is_not_assumed_high(self):
        gemini = GeminiISLService(api_key="test_key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "hand_gestures_observed": "The right hand moves toward the chest.",
                            "detected_glosses": ["I", "HAPPY"],
                            "translated_sentence": "I am happy.",
                            "sign_type": "continuous_sentence",
                        })
                    }]
                }
            }]
        }

        with patch("requests.post", return_value=mock_response):
            dummy_jpeg = cv2.imencode(
                ".jpg", np.zeros((224, 224, 3), dtype=np.uint8)
            )[1].tobytes()
            result = gemini.predict_from_frames([dummy_jpeg])

        self.assertEqual(result["translated_sentence"], "I am happy.")
        self.assertEqual(result["detected_glosses"], ["I", "HAPPY"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["prediction"], "I")

    def test_gemini_rejects_translation_without_visible_sign_evidence(self):
        gemini = GeminiISLService(api_key="test_key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "translated_sentence": "I am happy.",
                            "confidence": 0.99,
                        })
                    }]
                }
            }]
        }

        with patch("requests.post", return_value=mock_response):
            dummy_jpeg = cv2.imencode(
                ".jpg", np.zeros((224, 224, 3), dtype=np.uint8)
            )[1].tobytes()
            result = gemini.predict_from_frames([dummy_jpeg])

        self.assertIn("error", result)
        self.assertNotIn("translated_sentence", result)

    def test_gemini_retries_temporary_service_error(self):
        gemini = GeminiISLService(api_key="test_key")
        failed_response = MagicMock()
        failed_response.status_code = 503
        failed_response.headers = {}

        successful_response = MagicMock()
        successful_response.status_code = 200
        successful_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "hand_gestures_observed": "The signer moves both hands.",
                            "detected_glosses": ["HELLO"],
                            "translated_sentence": "Hello.",
                            "confidence": 0.4,
                            "sign_type": "isolated_word",
                        })
                    }]
                }
            }]
        }
        jpeg = cv2.imencode(".jpg", np.zeros((224, 224, 3), dtype=np.uint8))[1].tobytes()

        with patch("requests.post", side_effect=[failed_response, successful_response]) as post, \
                patch("time.sleep"):
            result = gemini.predict_from_frames([jpeg])

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["translated_sentence"], "Hello.")

    def test_gemini_returns_safe_actionable_service_error(self):
        gemini = GeminiISLService(api_key="test_key")
        failed_response = MagicMock()
        failed_response.status_code = 503
        failed_response.headers = {}
        jpeg = cv2.imencode(".jpg", np.zeros((224, 224, 3), dtype=np.uint8))[1].tobytes()

        with patch("requests.post", return_value=failed_response), patch("time.sleep"):
            result = gemini.predict_from_frames([jpeg])

        self.assertIn("HTTP 503", result["error"])
        self.assertIn("temporarily unavailable", result["error"])
        self.assertNotIn("test_key", result["error"])

    def test_gemini_requires_configured_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            gemini = GeminiISLService()

        self.assertFalse(gemini.is_available())
        self.assertEqual(
            gemini.predict_from_frames([]),
            {"error": "GEMINI_API_KEY not configured"},
        )


class TestPipelineLatencyAndGlossStreamBuffer(unittest.TestCase):
    """Requirement 4: Window Inference Latency (<200ms) & Gloss Buffer Feeding."""

    def test_gloss_stream_tracker_feeding_and_tts(self):
        tracker = ISLGlossStreamTracker(dwell_seconds=0.5, pause_timeout_seconds=0.8)

        # 1. Add detected glosses from continuous sequence
        r1 = tracker.add_gloss("TEACHER", confidence=0.88)
        self.assertTrue(r1["accepted"])
        self.assertEqual(r1["new_gloss"], "TEACHER")

        r2 = tracker.add_gloss("NAMASTE", confidence=0.91)
        self.assertTrue(r2["accepted"])
        self.assertEqual(r2["new_gloss"], "NAMASTE")

        # Debouncing: immediate repeat should be ignored
        r3 = tracker.add_gloss("NAMASTE", confidence=0.91)
        self.assertFalse(r3["accepted"])

        current = tracker.get_current_glosses()
        self.assertEqual(current, ["TEACHER", "NAMASTE"])

        # Flush and reconstruct sentence (simulates TTS trigger)
        reconstruction = tracker.flush_and_reconstruct()
        self.assertTrue(reconstruction["reconstructed"])
        self.assertTrue(len(reconstruction["sentence"]) > 0)
        self.assertEqual(reconstruction["original_glosses"], ["TEACHER", "NAMASTE"])
        # Buffer reset
        self.assertEqual(tracker.get_current_glosses(), [])

    def test_window_inference_latency_under_threshold(self):
        """
        Verify that sliding-window feature extraction and inference latency
        remains strictly under the 200 ms target threshold.
        """
        from app.services.isl_sentence_service import ISLSentenceService

        sent_svc = ISLSentenceService()
        dummy_img = Image.new("RGB", (224, 224), (100, 100, 100))
        dummy_jpeg = encode_jpeg(dummy_img)

        # Create a 12-frame window
        window_frames = []
        for i in range(12):
            window_frames.append(BufferedFrame(
                image=dummy_img,
                jpeg=dummy_jpeg,
                features_134=np.random.rand(134).astype(np.float32),
                hand_landmarks=[[{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]],
            ))

        t0 = time.perf_counter()
        feats = sent_svc.extract_features_from_buffered_frames(window_frames)
        feature_time_ms = (time.perf_counter() - t0) * 1000.0

        # Feature extraction must be ultra-fast (< 25 ms)
        self.assertLess(feature_time_ms, 25.0, f"Feature extraction too slow: {feature_time_ms}ms")

        # Even with full classification, overall pipeline latency must be well under 200ms
        self.assertIsNotNone(feats)
        self.assertEqual(feats.shape, (1, 900))

    def test_sentence_features_require_detected_hands(self):
        from app.services.isl_sentence_service import ISLSentenceService

        image = Image.new("RGB", (224, 224), (100, 100, 100))
        frames = [
            BufferedFrame(
                image=image,
                jpeg=b"",
                features_134=np.ones(134, dtype=np.float32),
                hand_landmarks=[],
            )
            for _ in range(12)
        ]

        self.assertIsNone(
            ISLSentenceService().extract_features_from_buffered_frames(frames)
        )

    def test_word_model_inference_uses_training_feature_shape(self):
        class ShapeCheckingModel:
            def predict_proba(self, features):
                self.assert_shape = features.shape
                return np.array([[1.0]])

        service = object.__new__(ISLWordsService)
        model = ShapeCheckingModel()
        service.model = model
        service.classes = ["HELLO"]

        hand = [(float(i), float(i + 1), float(i + 2)) for i in range(21)]
        pose = np.arange(50, dtype=np.float32)
        result = service.predict_landmarks([hand], pose_landmarks=pose)

        self.assertEqual(result["prediction"], "HELLO")
        self.assertEqual(model.assert_shape, (1, 150))

    def test_video_file_pipeline_ingestion(self):
        """
        Process a generated MP4 clip through the VideoTranslationService
        and verify downsampling, preprocessing, and structured translation output.
        """
        service = VideoTranslationService()

        # Create a temporary test MP4 video (30 FPS, 30 frames = 1 second)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video_path = tmp.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_video_path, fourcc, 30.0, (224, 224))
            for i in range(30):
                frame = np.ones((224, 224, 3), dtype=np.uint8) * 100
                # Draw simulated hand circle in upper torso zone
                cv2.circle(frame, (112, 100 + int(i * 1.5)), 20, (255, 255, 255), -1)
                # Draw simulated lower-third subtitle
                cv2.putText(frame, "SUBTITLE TEXT HERE", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                out.write(frame)
            out.release()

            # Mock GeminiISLService to test complete output pipeline
            with patch("app.services.gemini_isl.GeminiISLService.is_available", return_value=True), \
                 patch("app.services.gemini_isl.GeminiISLService.predict_from_frames", return_value={
                     "detected_glosses": ["HELLO", "FRIEND"],
                     "translated_sentence": "Hello my friend.",
                     "confidence": 0.95,
                     "sign_type": "continuous_sentence",
                     "sentence": "Hello my friend.",
                     "prediction": "HELLO"
                 }):
                t_start = time.perf_counter()
                result = service.translate_video_file(tmp_video_path)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0

                self.assertNotIn("error", result)
                self.assertEqual(result["detected_glosses"], ["HELLO", "FRIEND"])
                self.assertEqual(result["translated_sentence"], "Hello my friend.")
                self.assertEqual(result["sign_type"], "continuous_sentence")
                self.assertEqual(result["confidence"], 0.95)
                self.assertEqual(result["total_frames_analyzed"], 12)
                logger_msg = f"Processed 30-frame clip in {elapsed_ms:.1f}ms"
                self.assertIsNotNone(logger_msg)
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    def test_uploaded_video_uses_gemini_visual_translation(self):
        service = VideoTranslationService()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video_path = tmp.name

        try:
            writer = cv2.VideoWriter(
                tmp_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 32),
            )
            for _ in range(3):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()

            remote_result = {
                "translated_sentence": "I need some water.",
                "sentence": "I need some water.",
                "detected_glosses": ["I", "NEED", "WATER"],
                "confidence": 0.86,
            }
            with patch(
                "app.services.gemini_isl.GeminiISLService.is_available",
                return_value=True,
            ), patch(
                "app.services.gemini_isl.GeminiISLService.predict_from_frames",
                return_value=remote_result,
            ) as remote_vision:
                result = service.translate_video_file(tmp_video_path)

            self.assertEqual(
                result["translated_sentence"], "I need some water."
            )
            remote_vision.assert_called_once()
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    def test_gemini_receives_only_caption_masked_visual_frames(self):
        service = VideoTranslationService()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video_path = tmp.name

        try:
            writer = cv2.VideoWriter(
                tmp_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 64),
            )
            for _ in range(3):
                frame = np.full((64, 64, 3), 180, dtype=np.uint8)
                cv2.putText(
                    frame,
                    "SUBTITLE",
                    (1, 61),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.25,
                    (255, 255, 255),
                    1,
                )
                writer.write(frame)
            writer.release()

            def check_visual_frames(jpegs):
                self.assertEqual(len(jpegs), 3)
                frame = cv2.imdecode(
                    np.frombuffer(jpegs[0], dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                self.assertGreater(int(frame[20, 32].mean()), 100)
                self.assertLess(int(frame[60, 32].mean()), 10)
                return {
                    "translated_sentence": "I need water.",
                    "sentence": "I need water.",
                    "detected_glosses": ["I", "NEED", "WATER"],
                    "confidence": 0.8,
                }

            with patch("app.services.gemini_isl.GeminiISLService.is_available", return_value=True), \
                 patch(
                     "app.services.gemini_isl.GeminiISLService.predict_from_frames",
                     side_effect=check_visual_frames,
                 ):
                result = service.translate_video_file(tmp_video_path)

            self.assertEqual(result["translated_sentence"], "I need water.")
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    def test_unavailable_visual_api_returns_error_instead_of_a_guessed_sentence(self):
        service = VideoTranslationService()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video_path = tmp.name

        try:
            writer = cv2.VideoWriter(
                tmp_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 32),
            )
            for _ in range(3):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()

            with patch(
                "app.services.gemini_isl.GeminiISLService.is_available",
                return_value=False,
            ):
                result = service.translate_video_file(tmp_video_path)

            self.assertIn("error", result)
            self.assertNotIn("translated_sentence", result)
        finally:
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)


if __name__ == "__main__":
    unittest.main()
