import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LstmConfig:
    input_size: int = 134
    hidden_size: int = 256
    num_layers: int = 5
    batch_first: bool = True
    bidirectional: bool = True
    dropout: float = 0.2


class AI4BharatLSTM(nn.Module):
    """
    Official LSTM model architecture from AI4Bharat/INCLUDE paper.
    Processes sequences of 134-dim MediaPipe (Hands + Pose) keypoints.
    """
    def __init__(self, config: LstmConfig = LstmConfig(), n_classes: int = 50):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=config.batch_first,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        in_features = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.l1 = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 134)
        x, _ = self.lstm(x)
        x = torch.max(x, dim=1).values
        x = F.dropout(x, p=0.3)
        x = self.l1(x)
        return x


class INCLUDEClassifierService:
    """
    Stage 2: Official AI4Bharat/INCLUDE ISL Landmark Sequence Classifier.
    Evaluates temporal sign language gestures directly on MediaPipe landmarks.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(INCLUDEClassifierService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.model = None
        self.label_map = {}
        self.idx_to_label = {}
        self.seq_length = 60
        self._load_model()
        self._initialized = True

    def _load_model(self):
        root = Path(__file__).resolve().parents[2]
        weights_path = root / "models" / "include50_no_cnn_lstm.pth"
        labels_path = root / "models" / "label_map_include50.json"

        if not labels_path.exists():
            labels_path = root.parents[0] / "datasets" / "include" / "label_map_include50.json"

        if not labels_path.exists():
            logger.warning("INCLUDE label map not found at %s", labels_path)
            return

        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            self.idx_to_label = {v: k for k, v in self.label_map.items()}
            num_classes = len(self.label_map)

            self.model = AI4BharatLSTM(n_classes=num_classes)

            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=torch.device("cpu"), weights_only=False)
                state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                self.model.load_state_dict(state_dict)
                logger.info("Loaded official AI4Bharat pretrained weights (%s) for %d classes.", weights_path.name, num_classes)
            else:
                logger.warning("Pretrained weights not found at %s. Using initialized architecture.", weights_path)

            self.model.eval()
        except Exception as e:
            logger.error("Failed to initialize AI4Bharat INCLUDE model: %s", e)
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None and len(self.idx_to_label) > 0

    def predict_sequence(self, sequence: np.ndarray) -> dict:
        """
        Predict ISL gloss from temporal sequence of 134-dim landmarks.
        Input shape: (T, 134) or (1, T, 134)
        """
        if not self.is_ready():
            return {"error": "INCLUDE model not loaded"}

        try:
            # Handle shape
            if sequence.ndim == 2:
                # If feature dimension is 225 from 3D coords, extract 134 2D coords
                if sequence.shape[1] != 134 and sequence.shape[1] >= 134:
                    sequence = sequence[:, :134]
                elif sequence.shape[1] < 134:
                    padded = np.zeros((sequence.shape[0], 134), dtype=np.float32)
                    padded[:, :sequence.shape[1]] = sequence
                    sequence = padded
                tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            elif sequence.ndim == 3:
                if sequence.shape[2] != 134 and sequence.shape[2] >= 134:
                    sequence = sequence[:, :, :134]
                elif sequence.shape[2] < 134:
                    padded = np.zeros((sequence.shape[0], sequence.shape[1], 134), dtype=np.float32)
                    padded[:, :, :sequence.shape[2]] = sequence
                    sequence = padded
                tensor = torch.tensor(sequence, dtype=torch.float32)
            else:
                return {"error": f"Invalid sequence shape: {sequence.shape}"}

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                gloss = self.idx_to_label.get(pred_idx.item(), f"class_{pred_idx.item()}")
                confidence = float(conf.item())

                return {
                    "prediction": gloss,
                    "confidence": confidence,
                    "engine": "ai4bharat-include50-lstm"
                }

        except Exception as e:
            logger.exception("AI4Bharat INCLUDE sequence inference failed")
            return {"error": str(e)}
