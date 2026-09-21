from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.core.config import settings
from app.services.model_modules import DeepGuardModelStack

_model_stack = DeepGuardModelStack()


def _adaptive_sample_indices(total_frames: int, max_frames: int) -> np.ndarray:

    if total_frames <= 0:
        return np.array([], dtype=np.int32)

    if total_frames <= max_frames:
        return np.arange(total_frames, dtype=np.int32)

    step = total_frames / max_frames

    indices = []

    current = 0.0

    while int(current) < total_frames:
        indices.append(int(current))
        current += step

    return np.array(indices, dtype=np.int32)


def _laplacian_sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _get_dynamic_window(total_frames: int) -> int:

    if total_frames < 200:
        return 5

    elif total_frames < 600:
        return 8

    elif total_frames < 1000:
        return 10

    return 15


def _select_sharpest_per_window(
    frames: List[np.ndarray],
    window_size: int = 8
) -> List[np.ndarray]:

    if not frames:
        return []

    selected: List[np.ndarray] = []

    for i in range(0, len(frames), window_size):

        window = frames[i: i + window_size]

        if not window:
            continue

        sharpest = max(window, key=_laplacian_sharpness)

        selected.append(sharpest)

    return selected


def _read_video_frames(video_path: str) -> List[np.ndarray]:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to read video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = _adaptive_sample_indices(
        total,
        settings.max_frames
    )

    indices_set = set(indices.tolist())

    sampled: List[np.ndarray] = []

    current = 0

    while cap.isOpened():

        ok, frame = cap.read()

        if not ok:
            break

        if current in indices_set:
            sampled.append(frame)

        current += 1

    cap.release()

    if not sampled:
        raise ValueError("No frames sampled from video")

    return sampled


def analyze_video_pipeline(video_path: str) -> dict:

    sampled_frames = _read_video_frames(video_path)

    print(
        f"[DeepGuard-NR] Sampled Frames: "
        f"{len(sampled_frames)}"
    )

    window_size = _get_dynamic_window(
        len(sampled_frames)
    )

    frames = _select_sharpest_per_window(
        sampled_frames,
        window_size=window_size
    )

    print(
        f"[DeepGuard-NR] Selected Frames: "
        f"{len(frames)}"
    )

    if len(frames) < 3:
        frames = sampled_frames[:min(len(sampled_frames), 12)]

    verdict, confidence, features = _model_stack.infer(frames)

    heatmap_dir = Path(settings.heatmap_dir) / uuid.uuid4().hex
    heatmap_paths = _model_stack.gradcam.generate_overlay_sequence(
            frames,
            verdict,
            heatmap_dir,
        )

    module_summary = {
        "artifact_score": round(features.artifact_score, 4),
        "depth_score": round(features.depth_score, 4),
        "temporal_score": round(features.temporal_score, 4),
        "lighting_score": round(features.lighting_score, 4),
    }

    return {
        "verdict": verdict,
        "confidence": float(round(confidence, 4)),
        "artifact_score": module_summary["artifact_score"],
        "depth_score": module_summary["depth_score"],
        "temporal_score": module_summary["temporal_score"],
        "lighting_score": module_summary["lighting_score"],
        "heatmap_paths": heatmap_paths,
        "module_summary": module_summary,
    }
