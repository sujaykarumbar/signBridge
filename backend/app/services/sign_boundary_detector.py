"""
SignBridge Sign Boundary Detector
=================================
Kinematic motion segmentation for continuous Indian Sign Language (ISL).
Differentiates between:
  - Stroke phases: Active trajectory execution with high hand velocity.
  - Hold phases: Gestural climax/holds with localized stabilization and low velocity.
  - Transitions / Coarticulation: Inter-sign resets.

Enables accurate segmentation of continuous gesture sequences into discrete sign units.
"""

from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GesturePhase(str, Enum):
    IDLE = "idle"
    STROKE = "stroke"
    HOLD = "hold"
    TRANSITION = "transition"


class SignBoundaryDetector:
    """
    Analyzes frame-to-frame motion kinematics to detect sign boundaries
    between hold and stroke phases in continuous ISL signing.
    """

    def __init__(
        self,
        stroke_velocity_threshold: float = 0.045,
        hold_velocity_threshold: float = 0.022,
        min_hold_frames: int = 2,
        min_stroke_frames: int = 3,
    ):
        self.stroke_velocity_threshold = stroke_velocity_threshold
        self.hold_velocity_threshold = hold_velocity_threshold
        self.min_hold_frames = min_hold_frames
        self.min_stroke_frames = min_stroke_frames

        self._prev_landmarks = None
        self._velocity_history = []
        self._current_phase = GesturePhase.IDLE
        self._hold_frame_count = 0
        self._stroke_frame_count = 0

    def reset(self):
        """Reset internal trajectory states."""
        self._prev_landmarks = None
        self._velocity_history.clear()
        self._current_phase = GesturePhase.IDLE
        self._hold_frame_count = 0
        self._stroke_frame_count = 0

    def compute_frame_velocity(self, hand_landmarks=None, features_134=None) -> float:
        """
        Calculate instantaneous hand kinematic velocity between consecutive frames.
        Returns normalized average velocity (0.0 to 1.0).
        """
        curr_pts = []

        # 1. From MediaPipe hand landmarks (preferred if available)
        if hand_landmarks:
            for hand in hand_landmarks[:2]:
                lms = hand.landmark if hasattr(hand, "landmark") else hand
                for lm in lms[:21]:
                    if isinstance(lm, dict):
                        x = float(lm.get("x", 0.0))
                        y = float(lm.get("y", 0.0))
                        z = float(lm.get("z", 0.0))
                    elif isinstance(lm, (list, tuple, np.ndarray)):
                        x = float(lm[0]) if len(lm) > 0 else 0.0
                        y = float(lm[1]) if len(lm) > 1 else 0.0
                        z = float(lm[2]) if len(lm) > 2 else 0.0
                    else:
                        x = float(getattr(lm, "x", 0.0))
                        y = float(getattr(lm, "y", 0.0))
                        z = float(getattr(lm, "z", 0.0))
                    curr_pts.append((x, y, z))

        # 2. Fallback to features_134 vector (hands are entries 50:134)
        elif features_134 is not None and len(features_134) >= 134:
            hand_feats = features_134[50:134]
            # Hand features are (x, y) coordinates for 42 hand joints (2 hands * 21)
            for i in range(0, len(hand_feats), 2):
                x = hand_feats[i]
                y = hand_feats[i + 1]
                if x > 0 or y > 0:
                    curr_pts.append((x, y, 0.0))

        if not curr_pts:
            self._prev_landmarks = None
            return 0.0

        curr_arr = np.array(curr_pts, dtype=np.float32)

        if self._prev_landmarks is None or len(self._prev_landmarks) != len(curr_arr):
            self._prev_landmarks = curr_arr
            return 0.0

        # Mean Euclidean distance across tracked keypoints
        displacement = np.linalg.norm(curr_arr - self._prev_landmarks, axis=1)
        velocity = float(np.mean(displacement))
        self._prev_landmarks = curr_arr

        return velocity

    def update(self, hand_landmarks=None, features_134=None) -> dict:
        """
        Process a new frame and update phase state machine.
        Returns:
            dict containing:
                - velocity: float
                - phase: GesturePhase
                - is_boundary: bool (True when stroke-to-hold or sign completion occurs)
                - hold_detected: bool
        """
        velocity = self.compute_frame_velocity(hand_landmarks, features_134)
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 30:
            self._velocity_history.pop(0)

        is_boundary = False
        prev_phase = self._current_phase

        if velocity >= self.stroke_velocity_threshold:
            self._stroke_frame_count += 1
            self._hold_frame_count = 0
            if self._stroke_frame_count >= self.min_stroke_frames:
                self._current_phase = GesturePhase.STROKE

        elif velocity <= self.hold_velocity_threshold and velocity > 0.001:
            self._hold_frame_count += 1
            if self._hold_frame_count >= self.min_hold_frames:
                if self._current_phase == GesturePhase.STROKE:
                    # Stroke transitioned into a stabilized hold = sign climax boundary!
                    is_boundary = True
                self._current_phase = GesturePhase.HOLD
                self._stroke_frame_count = 0
        else:
            # Intermediate velocities
            if self._current_phase == GesturePhase.HOLD:
                self._current_phase = GesturePhase.TRANSITION

        return {
            "velocity": round(velocity, 4),
            "phase": self._current_phase.value,
            "previous_phase": prev_phase.value,
            "is_boundary": is_boundary,
            "is_hold": self._current_phase == GesturePhase.HOLD,
            "is_stroke": self._current_phase == GesturePhase.STROKE,
        }

    def segment_continuous_sequence(self, frames_features: list) -> list[dict]:
        """
        Segment a continuous sequence of frame feature sets into distinct sign segments.
        Each segment represents [start_frame_idx, end_frame_idx, apex_frame_idx, phase].
        """
        self.reset()
        segments = []
        seg_start = 0

        for idx, f_data in enumerate(frames_features):
            hand_lms = f_data.get("hand_landmarks") if isinstance(f_data, dict) else getattr(f_data, "hand_landmarks", None)
            f_134 = f_data.get("features_134") if isinstance(f_data, dict) else getattr(f_data, "features_134", None)

            res = self.update(hand_landmarks=hand_lms, features_134=f_134)
            if res["is_boundary"] and idx > seg_start + self.min_stroke_frames:
                segments.append({
                    "start_idx": seg_start,
                    "end_idx": idx,
                    "hold_idx": idx,
                    "phase": GesturePhase.HOLD.value,
                })
                seg_start = idx

        if seg_start < len(frames_features):
            segments.append({
                "start_idx": seg_start,
                "end_idx": len(frames_features) - 1,
                "hold_idx": len(frames_features) - 1,
                "phase": self._current_phase.value,
            })

        return segments
