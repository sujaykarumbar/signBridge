"""
SignBridge Continuous ISL Sentence Recognition Service
======================================================
Evaluates live sliding window frames against the trained continuous
ISL sentence model (97 classes).
Inference speed: < 2ms on CPU.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import joblib

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "models" / "isl_sentence_model.joblib"
CLASSES_PATH = _PROJECT_ROOT / "models" / "isl_sentence_classes.json"

NUM_FRAME_FEATS = 150
NUM_TOTAL_FEATS = 900


class ISLSentenceService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ISLSentenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Path = MODEL_PATH):
        if getattr(self, "_initialized", False):
            return

        self.model_path = model_path
        self.model = None
        self.classes = []
        self._load_model()
        self._initialized = True

    def _load_model(self):
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.classes = data.get("classes", [])
                logger.info(
                    "[ISL Sentence] Loaded continuous sentence model (%d classes) from %s",
                    len(self.classes),
                    self.model_path.name
                )
            except Exception as e:
                logger.error("[ISL Sentence] Failed to load sentence model: %s", e)
                self.model = None
        else:
            logger.info("[ISL Sentence] Model not found at %s yet.", self.model_path)

    def is_ready(self) -> bool:
        if self.model is None and self.model_path.exists():
            self._load_model()
        return self.model is not None and len(self.classes) > 0

    def extract_features_from_buffered_frames(self, frames: list) -> np.ndarray | None:
        """
        Extract the 900-dim spatio-temporal feature vector from a list of BufferedFrame objects.
        """
        if not frames or len(frames) < 3:
            return None
        if sum(bool(getattr(frame, "hand_landmarks", None)) for frame in frames) < 3:
            return None

        frame_feats = []
        for f in frames:
            feat_150 = np.zeros(NUM_FRAME_FEATS, dtype=np.float32)

            # 1. Hands (up to 2 hands) -> 126 coordinates
            hand_lms = getattr(f, "hand_landmarks", None)
            if hand_lms:
                for h_idx, hand in enumerate(hand_lms[:2]):
                    base = h_idx * 63
                    lms = hand.landmark if hasattr(hand, "landmark") else hand
                    if len(lms) > 0:
                        wrist = lms[0]
                        w_x = getattr(wrist, "x", wrist[0] if isinstance(wrist, (list, tuple, np.ndarray)) else 0.0)
                        w_y = getattr(wrist, "y", wrist[1] if isinstance(wrist, (list, tuple, np.ndarray)) else 0.0)
                        w_z = getattr(wrist, "z", wrist[2] if isinstance(wrist, (list, tuple, np.ndarray)) and len(wrist) > 2 else 0.0)

                        for lm_idx, lm in enumerate(lms[:21]):
                            lx = getattr(lm, "x", lm[0] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                            ly = getattr(lm, "y", lm[1] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                            lz = getattr(lm, "z", lm[2] if isinstance(lm, (list, tuple, np.ndarray)) and len(lm) > 2 else 0.0)

                            feat_150[base + lm_idx * 3]     = lx - w_x
                            feat_150[base + lm_idx * 3 + 1] = ly - w_y
                            feat_150[base + lm_idx * 3 + 2] = lz - w_z

            # 2. Pose -> 24 coordinates
            # Check features_134 in BufferedFrame which has pose coords in first 50 entries
            f_134 = getattr(f, "features_134", None)
            if f_134 is not None and len(f_134) >= 50:
                nose_x = f_134[0]
                nose_y = f_134[1]
                pose_indices = [0, 11, 12, 13, 14, 15, 16, 23]
                base_pose = 126
                for p_i, p_idx in enumerate(pose_indices):
                    px = f_134[p_idx * 2]
                    py = f_134[p_idx * 2 + 1]
                    if px > 0 or py > 0:
                        feat_150[base_pose + p_i * 3]     = px - nose_x
                        feat_150[base_pose + p_i * 3 + 1] = py - nose_y
                        feat_150[base_pose + p_i * 3 + 2] = 0.0

            frame_feats.append(feat_150)

        if len(frame_feats) < 3:
            return None

        arr = np.array(frame_feats, dtype=np.float32)
        if np.count_nonzero(arr) < 15:
            return None

        mean_f = np.mean(arr, axis=0)
        std_f = np.std(arr, axis=0)
        min_f = np.min(arr, axis=0)
        max_f = np.max(arr, axis=0)
        delta_f = arr[-1] - arr[0]
        vel_f = np.mean(np.abs(np.diff(arr, axis=0)), axis=0) if len(arr) > 1 else np.zeros(NUM_FRAME_FEATS, dtype=np.float32)

        return np.concatenate([mean_f, std_f, min_f, max_f, delta_f, vel_f]).reshape(1, -1)

    def predict_window(self, frames: list) -> dict:
        """
        Predict full sentence from live buffered window frames.
        Returns top prediction, confidence, and engine name.
        """
        if not self.is_ready():
            return {"error": "Sentence model not loaded"}

        feats = self.extract_features_from_buffered_frames(frames)
        if feats is None:
            return {"sentence": "", "prediction": "", "confidence": 0.0}

        try:
            probs = self.model.predict_proba(feats)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            sentence = str(self.classes[top_idx])

            return {
                "sentence": sentence,
                "prediction": sentence,
                "confidence": top_conf,
                "engine": "isl-sentence-continuous"
            }
        except Exception as e:
            logger.exception("[ISL Sentence] Prediction failed: %s", e)
            return {"error": str(e)}
