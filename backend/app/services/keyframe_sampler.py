"""
SignBridge Keyframe Sampler & Frame Preprocessing
=================================================
Uniform keyframe downsampling and region-of-interest (ROI) focusing for
continuous Indian Sign Language (ISL) video clips (reels, social media, instructional signing).

- Downsamples video streams to 12-16 FPS (default: 14 FPS) to prevent frame buffer saturation.
- Ignores lower-third text (captions, subtitles, banners) and focuses landmark detection on
  the upper torso, hands, and face.
"""

import math
from typing import List, Tuple, Union
import numpy as np
import cv2
from PIL import Image


class KeyframeSampler:
    """
    Subsamples continuous video frames to a uniform cadence between 12-16 FPS.
    Prevents temporal jitter and frame buffer saturation across variable source frame rates.
    """

    def __init__(self, target_fps: float = 14.0, min_fps: float = 12.0, max_fps: float = 16.0):
        self.target_fps = max(min_fps, min(max_fps, float(target_fps)))

    def compute_sample_indices(self, total_frames: int, src_fps: float) -> List[int]:
        """
        Compute uniform frame indices corresponding to target_fps sampling.
        """
        if total_frames <= 0:
            return []
        if total_frames == 1 or src_fps <= 0:
            return [0]

        duration = total_frames / max(1.0, src_fps)
        num_target_frames = max(1, int(math.ceil(duration * self.target_fps)))

        # Uniform timestamps: t_k = k / target_fps
        indices = []
        last_idx = -1
        step = src_fps / self.target_fps

        for k in range(num_target_frames):
            idx = int(round(k * step))
            idx = min(total_frames - 1, max(0, idx))
            if idx != last_idx:
                indices.append(idx)
                last_idx = idx

        # Ensure last frame is included for sequence boundary integrity
        if indices and indices[-1] != (total_frames - 1) and (total_frames - 1) - indices[-1] <= int(round(step)):
            indices[-1] = total_frames - 1
        elif indices and indices[-1] < (total_frames - 1):
            indices.append(total_frames - 1)

        return indices

    def sample_frames(self, frames: list, src_fps: float = 25.0) -> list:
        """
        Downsample a list of raw frames to target_fps.
        """
        indices = self.compute_sample_indices(len(frames), src_fps)
        return [frames[i] for i in indices]


def mask_lower_third_and_overlays(
    image: Union[np.ndarray, Image.Image],
    lower_third_ratio: float = 0.20,
    fill_color: Union[int, Tuple[int, int, int]] = 0,
) -> Union[np.ndarray, Image.Image]:
    """
    Blanks out static or animated lower-third on-screen text, subtitles,
    and caption overlays so landmark tracking focuses strictly on the
    upper torso, hands, and face.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        arr = np.array(image).copy()
    else:
        arr = image.copy()

    h, w = arr.shape[:2]
    cutoff_y = int(h * (1.0 - max(0.05, min(0.40, lower_third_ratio))))

    if arr.ndim == 2:
        arr[cutoff_y:, :] = 0 if isinstance(fill_color, int) else 0
    elif arr.ndim == 3:
        arr[cutoff_y:, :] = fill_color if isinstance(fill_color, (list, tuple)) else (0, 0, 0)

    if is_pil:
        return Image.fromarray(arr)
    return arr


def focus_upper_body_roi(
    image: Union[np.ndarray, Image.Image],
    lower_third_ratio: float = 0.20,
) -> Union[np.ndarray, Image.Image]:
    """
    Crops image to focus strictly on the upper torso, face, and active gesture signing zone,
    discarding lower third text entirely.
    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        w, h = image.size
        cutoff_y = int(h * (1.0 - lower_third_ratio))
        return image.crop((0, 0, w, cutoff_y))
    else:
        h, w = image.shape[:2]
        cutoff_y = int(h * (1.0 - lower_third_ratio))
        return image[:cutoff_y, :]
