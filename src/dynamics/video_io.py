"""Minimal deterministic video I/O for selected RoboCasa frames."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def read_selected_rgb_frames(path: Path, indices: Iterable[int]) -> np.ndarray:
    requested = np.asarray(sorted(set(int(value) for value in indices)), dtype=np.int64)
    if requested.size == 0:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    if requested[0] < 0:
        raise ValueError("frame indices must be non-negative")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    result = []
    target_pos = 0
    frame_index = 0
    try:
        while target_pos < len(requested):
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index == requested[target_pos]:
                result.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                target_pos += 1
            frame_index += 1
    finally:
        capture.release()
    if target_pos != len(requested):
        raise RuntimeError(
            f"Video {path} ended before frame {int(requested[target_pos])}; "
            f"decoded {frame_index} frames"
        )
    return np.stack(result)
