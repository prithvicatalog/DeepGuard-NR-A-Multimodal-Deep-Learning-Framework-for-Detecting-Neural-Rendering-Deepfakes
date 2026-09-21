from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset

from app.services.pipeline_service import (
    _get_dynamic_window,
    _read_video_frames,
    _select_sharpest_per_window,
)


class VideoDataset(Dataset):
    """
    Wraps a manifest CSV of (video_path, label) pairs and reuses the exact
    same frame-sampling logic as the inference pipeline, so training and
    serving see identically preprocessed frames (no train/serve skew).

    Manifest CSV format, no header:
        /path/to/real_001.mp4,0
        /path/to/fake_001.mp4,1

    Label convention: 0 = REAL, 1 = FAKE (matches FusionClassifier's
    sigmoid output, which infer() reads as "fake_prob").
    """

    def __init__(self, manifest_path: str) -> None:
        self.samples: List[Tuple[str, int]] = []
        manifest = Path(manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found at {manifest_path}. "
                "Create a CSV of 'video_path,label' rows before training."
            )

        with open(manifest, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                path, label = row[0].strip(), int(row[1].strip())
                if label not in (0, 1):
                    raise ValueError(f"Label must be 0 or 1, got {label} for {path}")
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"Manifest at {manifest_path} is empty.")

    def __len__(self) -> int:
        return len(self.samples)

    def group_ids(self) -> List[str]:
        groups = []
        for path, _ in self.samples:
            stem = Path(path).stem
            groups.append(stem.split("_")[0])
        return groups

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        sampled_frames = _read_video_frames(video_path)
        window_size = _get_dynamic_window(len(sampled_frames))
        frames = _select_sharpest_per_window(sampled_frames, window_size=window_size)

        if len(frames) < 3:
            frames = sampled_frames[: min(len(sampled_frames), 12)]

        # frames is a List[np.ndarray] of variable length per video —
        # returned as-is; collate_fn in the DataLoader must NOT try to
        # stack these into a single tensor (see train.py's use of
        # batch_size=1, no custom collate needed).
        return frames, label
