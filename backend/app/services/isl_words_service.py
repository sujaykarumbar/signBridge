import os
import json
import logging
from pathlib import Path
import joblib
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _PROJECT_ROOT / "models" / "isl_words_model.joblib"
CLASSES_PATH = _PROJECT_ROOT / "models" / "isl_words_classes.json"

NUM_FEATURES = 150

# Map raw folder names to clean English glosses
GLOSS_CLEAN_MAP = {
    "HELLO_HI": "HELLO",
    "I_ME_MINE_MY": "I",
    "COLLEGE_SCHOOL": "COLLEGE",
    "LIKE_LOVE": "LIKE",
    "OLD_AGE": "OLD",
    "A LOT": "A LOT",
    "DON'T CARE": "DONT CARE",
    "MEAN IT": "MEAN IT",
    "ON THE WAY": "ON THE WAY",
    "SO MUCH": "SO MUCH",
    "SOME HOW": "SOMEHOW",
    "SOME ONE": "SOMEONE",
    "TAKE CARE": "TAKE CARE",
    "TAKE TIME": "TAKE TIME",
    "TURN ON": "TURN ON",
}


class ISLWordsService:
    """
    Stage 1: Real-Time Indian Sign Language Word Gesture Classifier (114 Classes).
    Trained directly on local ISL-CSLTR gesture landmark representations.
    Inference speed: < 3ms on CPU.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ISLWordsService, cls).__new__(cls)
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
                    "[ISL Words] Loaded 114-word gesture model (%d classes) from %s",
                    len(self.classes),
                    self.model_path.name
                )
            except Exception as e:
                logger.error("[ISL Words] Failed to load model: %s", e)
                self.model = None
        else:
            logger.info("[ISL Words] Model not found at %s yet.", self.model_path)

    def is_ready(self) -> bool:
        if self.model is None and self.model_path.exists():
            self._load_model()
        return self.model is not None and len(self.classes) > 0

    def predict_landmarks(self, hand_landmarks_list, pose_landmarks=None) -> dict:
        """
        Predict ISL word gesture from live MediaPipe hand and pose landmarks.
        Returns top prediction, confidence, and engine name.
        """
        if not self.is_ready():
            return {"error": "ISL Words model not loaded"}

        if not hand_landmarks_list:
            return {"prediction": "", "confidence": 0.0}

        try:
            feats = np.zeros((1, NUM_FEATURES), dtype=np.float32)

            # 1. Hands (up to 2 hands) -> 126 features
            for h_idx, hand in enumerate(hand_landmarks_list[:2]):
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

                        feats[0, base + lm_idx * 3]     = lx - w_x
                        feats[0, base + lm_idx * 3 + 1] = ly - w_y
                        feats[0, base + lm_idx * 3 + 2] = lz - w_z

            # 2. Pose (8 landmarks relative to the nose) -> 24 features.
            if pose_landmarks is not None:
                pose = np.asarray(pose_landmarks, dtype=np.float32).reshape(-1)
                pose_indices = [0, 11, 12, 13, 14, 15, 16, 23]
                if pose.size >= 48:
                    nose_x, nose_y = pose[0], pose[1]
                    for pose_idx, landmark_idx in enumerate(pose_indices):
                        x, y = pose[landmark_idx * 2:landmark_idx * 2 + 2]
                        base = 126 + pose_idx * 3
                        feats[0, base] = x - nose_x
                        feats[0, base + 1] = y - nose_y

            # Check for meaningful signal (at least a detected hand)
            if np.count_nonzero(feats[0, :126]) < 15:
                return {"prediction": "", "confidence": 0.0}

            probs = self.model.predict_proba(feats)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            raw_class = str(self.classes[top_idx])
            clean_class = GLOSS_CLEAN_MAP.get(raw_class, raw_class)

            return {
                "prediction": clean_class,
                "confidence": top_conf,
                "engine": "isl-words-114"
            }

        except Exception as e:
            logger.exception("[ISL Words] Inference error: %s", e)
            return {"error": str(e)}
