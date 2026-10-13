import os
import logging
import joblib
import numpy as np
from PIL import Image

from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = str(_PROJECT_ROOT / "models" / "isl_alphabet_model.joblib")

class ISLAlphabetService:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.classes = []
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.classes = data.get("classes", [])
                logger.info(
                    "[ISL Alphabet] Loaded classifier with %d classes from %s",
                    len(self.classes),
                    self.model_path
                )
            except Exception as e:
                logger.error("[ISL Alphabet] Failed to load model: %s", e)
        else:
            logger.info("[ISL Alphabet] Model not found at %s yet.", self.model_path)

    def is_ready(self) -> bool:
        if self.model is None and os.path.exists(self.model_path):
            self._load_model()
        return self.model is not None

    def predict_landmarks(self, hand_landmarks_list) -> dict:
        """
        Given a list of NormalizedLandmarkList (from MediaPipe hands detector),
        extract 126 normalized features and return classification result.
        """
        if not self.is_ready():
            return {"error": "ISL Alphabet model not loaded"}

        if not hand_landmarks_list:
            return {"prediction": "", "confidence": 0.0}

        try:
            features = np.zeros((1, 126), dtype=np.float32)
            
            for h_idx, hand_landmarks in enumerate(hand_landmarks_list[:2]):
                base_idx = h_idx * 63
                # Get wrist reference
                lms = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
                wrist_x = lms[0].x
                wrist_y = lms[0].y
                wrist_z = lms[0].z

                for lm_idx, lm in enumerate(lms):
                    features[0, base_idx + lm_idx * 3]     = lm.x - wrist_x
                    features[0, base_idx + lm_idx * 3 + 1] = lm.y - wrist_y
                    features[0, base_idx + lm_idx * 3 + 2] = lm.z - wrist_z

            probs = self.model.predict_proba(features)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            top_class = str(self.classes[top_idx])

            return {
                "prediction": top_class,
                "confidence": top_conf,
                "engine": "isl-alphabet-35"
            }
        except Exception as e:
            logger.exception("[ISL Alphabet] Prediction error: %s", e)
            return {"error": str(e)}
