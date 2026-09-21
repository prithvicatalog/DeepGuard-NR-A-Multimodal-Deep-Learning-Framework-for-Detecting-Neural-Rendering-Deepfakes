from __future__ import annotations

import argparse
import json
import logging
import os
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import DataLoader, random_split

from app.services.dataset import VideoDataset

from app.services.model_modules import DeepGuardModelStack

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepguard_nr.train")


def _collate_single(batch):
    # batch_size=1 by design (see dataset.py) — this just unwraps the
    # single-item list DataLoader gives us back into (frames, label).
    frames, label = batch[0]
    return frames, label


def _forward_pass(stack: DeepGuardModelStack, frames):
    """
    Mirrors DeepGuardModelStack.infer(), but WITHOUT @torch.no_grad(),
    so gradients flow. Kept as a separate function (not a reused method)
    because infer() is the serving path and must stay grad-free — do not
    merge these into one method.
    """
    spatial_seq = stack.extractor.spatial_sequence(frames)
    spatial_mean = spatial_seq.mean(dim=0, keepdim=True)
    freq = stack.extractor.frequency_vector(frames)
    texture_edge = stack.extractor.texture_edge_vector(frames)
    artifact_in = torch.cat([spatial_mean, freq, texture_edge], dim=1)

    artifact_feat, _ = stack.artifact(artifact_in)
    depth_feat, _ = stack.depth(frames, stack.device)
    temporal_feat, _ = stack.temporal(spatial_seq)
    lighting_feat, _ = stack.lighting(frames, stack.device)

    fused_in = torch.cat([artifact_feat, depth_feat, temporal_feat, lighting_feat], dim=1)
    logit = stack.fusion(fused_in).squeeze()
    return logit


def _save_resume_checkpoint(stack: DeepGuardModelStack, path: str, next_epoch: int) -> None:
    torch.save(
        {
            "artifact": stack.artifact.state_dict(),
            "depth": stack.depth.state_dict(),
            "temporal": stack.temporal.state_dict(),
            "lighting": stack.lighting.state_dict(),
            "fusion": stack.fusion.state_dict(),
            "epoch": next_epoch,
        },
        path,
    )


def train(manifest_path: str, epochs: int, lr: float, val_split: float, checkpoint_out: str, save_every: int = 20):
    stack = DeepGuardModelStack()

    start_epoch = 1
    resume_path = checkpoint_out.replace(".pt", "_resume.pt")
    if os.path.exists(resume_path):
        logger.info(f"Found resume checkpoint at {resume_path} — resuming training.")
        checkpoint = torch.load(resume_path, map_location=stack.device)
        stack.artifact.load_state_dict(checkpoint["artifact"])
        stack.depth.load_state_dict(checkpoint["depth"])
        stack.temporal.load_state_dict(checkpoint["temporal"])
        stack.lighting.load_state_dict(checkpoint["lighting"])
        stack.fusion.load_state_dict(checkpoint["fusion"])
        start_epoch = checkpoint.get("epoch", 1)
        logger.info(f"Resuming from epoch {start_epoch}")

    stack.train_mode()

    dataset = VideoDataset(manifest_path)
    groups = dataset.group_ids()
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(dataset.samples, groups=groups))
    if len(train_idx) < 1 or len(val_idx) < 1:
        raise ValueError(
            f"Dataset too small or too few identity groups for a {val_split:.0%} val split "
            f"({len(set(groups))} unique groups found)."
            )
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=_collate_single)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=_collate_single)

    optimizer = torch.optim.Adam(stack.trainable_parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    best_val_loss_path = checkpoint_out.replace(".pt", "_best_val_loss.txt")
    if os.path.exists(best_val_loss_path):
        with open(best_val_loss_path) as bf:
            best_val_loss = float(bf.read().strip())
        logger.info(f"Loaded prior best_val_loss={best_val_loss:.4f} from {best_val_loss_path}")
    else:
        best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs + 1):
        stack.train_mode()
        running_loss = 0.0

        for step, (frames, label) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            logit = _forward_pass(stack, frames)
            target = torch.tensor(float(label), device=stack.device)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 5 == 0 or step == 1:
                logger.info(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

            # Mid-epoch safety save — survives a crash/disconnect partway
            # through an epoch, not just between epochs.
            if step % save_every == 0:
                _save_resume_checkpoint(stack, resume_path, epoch)
                logger.info(f"  [safety checkpoint] epoch {epoch} step {step}")

        avg_train_loss = running_loss / max(1, len(train_loader))

        stack.eval_mode()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for frames, label in val_loader:
                logit = _forward_pass(stack, frames)
                target = torch.tensor(float(label), device=stack.device)
                val_loss += criterion(logit, target).item()
                predicted = 1 if logit.item() >= 0.5 else 0
                correct += int(predicted == label)

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = correct / max(1, len(val_loader))

        logger.info(
            f"Epoch {epoch}/{epochs} | train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.2%}"
        )

        metrics_path = checkpoint_out.replace(".pt", "_metrics.jsonl")
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps({
                "epoch": epoch, "train_loss": avg_train_loss,
                "val_loss": avg_val_loss, "val_acc": val_acc
            }) + "\n")

        # End-of-epoch resume checkpoint, regardless of val_loss improvement
        # — this is what "resume from where I left off" reads on restart.
        _save_resume_checkpoint(stack, resume_path, epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            saved_path = stack.save_checkpoint(checkpoint_out)
            with open(best_val_loss_path, "w") as bf:
                bf.write(str(best_val_loss))
            logger.info(f"New best val_loss — best checkpoint saved to {saved_path}")

    logger.info("Training complete.")
    if os.path.exists(resume_path):
        os.remove(resume_path)  # clean up — training finished normally


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGuard-NR module stack")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--checkpoint-out", default="models/deepguard_nr_checkpoint.pt")
    parser.add_argument("--save-every", type=int, default=20, help="Save a resume checkpoint every N training steps")
    args = parser.parse_args()

    train(
        manifest_path=args.manifest,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        checkpoint_out=args.checkpoint_out,
        save_every=args.save_every,
    )
