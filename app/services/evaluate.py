from __future__ import annotations

import argparse
import logging

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from app.services.dataset import VideoDataset
from app.services.model_modules import DeepGuardModelStack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepguard_nr.evaluate")


def _collate_single(batch):
    frames, label = batch[0]
    return frames, label


@torch.no_grad()
def _module_scores(stack: DeepGuardModelStack, frames):
    """
    Returns each module's individual fake-probability score plus the fused
    verdict's probability, so we can report per-module ablation alongside
    the fused result. Reuses infer() rather than duplicating its logic,
    since this path is grad-free just like serving.
    """
    verdict, confidence, features = stack.infer(frames)
    fused_prob = confidence if verdict == "FAKE" else 1.0 - confidence
    return {
        "artifact": features.artifact_score,
        "depth": features.depth_score,
        "temporal": features.temporal_score,
        "lighting": features.lighting_score,
        "fused": fused_prob,
    }


def _report(name: str, y_true, y_prob, threshold: float = 0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in y_true — can't compute AUC

    logger.info(
        f"[{name:9s}] acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} "
        f"f1={f1:.4f} auc={auc:.4f}"
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def evaluate(manifest_path: str, checkpoint_path: str | None):
    stack = DeepGuardModelStack()
    if checkpoint_path:
        stack.save_checkpoint  # no-op reference, keeps import used
        checkpoint = torch.load(checkpoint_path, map_location=stack.device)
        stack.artifact.load_state_dict(checkpoint["artifact"])
        stack.depth.load_state_dict(checkpoint["depth"])
        stack.temporal.load_state_dict(checkpoint["temporal"])
        stack.lighting.load_state_dict(checkpoint["lighting"])
        stack.fusion.load_state_dict(checkpoint["fusion"])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning(
            "No --checkpoint given — evaluating with whatever weights "
            "DeepGuardModelStack loaded by default (checkpoint_path in "
            "config if present, otherwise untrained random weights)."
        )
    stack.eval_mode()

    dataset = VideoDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate_single)

    y_true = []
    scores = {"artifact": [], "depth": [], "temporal": [], "lighting": [], "fused": []}

    for i, (frames, label) in enumerate(loader):
        try:
            module_scores = _module_scores(stack, frames)
        except Exception as e:
            logger.error(f"Skipping sample {i} ({dataset.samples[i][0]}): {e}")
            continue

        y_true.append(label)
        for key in scores:
            scores[key].append(module_scores[key])

    if not y_true:
        raise RuntimeError("No samples evaluated successfully — check the manifest paths.")

    logger.info(f"Evaluated {len(y_true)} samples "
                f"({sum(y_true)} fake, {len(y_true) - sum(y_true)} real)")
    logger.info("--- Per-module ablation (each module's score alone vs. label) ---")
    for module_name in ["artifact", "depth", "temporal", "lighting"]:
        _report(module_name, y_true, scores[module_name])

    logger.info("--- Fused verdict (the actual production output) ---")
    fused_metrics = _report("fused", y_true, scores["fused"])

    y_pred = [1 if p >= 0.5 else 0 for p in scores["fused"]]
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion matrix (rows=true, cols=pred, order=[REAL,FAKE]):\n{cm}")

    return fused_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard-NR on a held-out test manifest")
    parser.add_argument("--manifest", required=True, help="Path to TEST manifest CSV (must not overlap train manifest)")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint .pt file")
    args = parser.parse_args()

    evaluate(manifest_path=args.manifest, checkpoint_path=args.checkpoint)
