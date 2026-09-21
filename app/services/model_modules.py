from __future__ import annotations
from app.core.config import settings
from dataclasses import dataclass
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
import torch.hub
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from mamba_ssm import Mamba


def _to_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_frame(frame: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1)))
    return tensor


@dataclass
class MultimodalFeatures:
    artifact_feature: torch.Tensor
    depth_feature: torch.Tensor
    temporal_feature: torch.Tensor
    lighting_feature: torch.Tensor
    artifact_score: float
    depth_score: float
    temporal_score: float
    lighting_score: float


class SpatialFrequencyExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.DEFAULT
        )
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.device = _to_device()
        self.to(self.device)
        self.eval()

    def spatial_sequence(self, frames: List[np.ndarray]) -> torch.Tensor:
        tensors = torch.stack([_normalize_frame(frame) for frame in frames]).to(self.device)
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            feat_map = self.feature_extractor(tensors)
            pooled = self.pool(feat_map).flatten(1)
        return pooled.float()

    def frequency_vector(self, frames):
        vectors = []

        for frame in frames:

            gray = cv2.cvtColor(
                cv2.resize(frame, (224,224)),
                cv2.COLOR_BGR2GRAY
            ).astype(np.float32)/255.0

            dct = cv2.dct(gray)

            low = dct[:16,:16].flatten()

            mid = dct[16:48,16:48].flatten()

            stats = np.array([
                dct.mean(),
                dct.std(),
                np.max(dct),
                np.min(dct)
            ])

            feature = np.concatenate([
                low,
                mid,
                stats
            ])

            vectors.append(feature)

        freq = np.mean(np.stack(vectors),axis=0)

        return torch.tensor(
            freq,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
    def _texture_edge_stats(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

                # Edge gradient analysis — unstable rendering boundaries / artifact edges
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(gx, gy)
        edge_density = float((edge_mag > (edge_mag.mean() + edge_mag.std())).mean())

        orientation = np.arctan2(gy, gx)
        hist, _ = np.histogram(orientation, bins=16, range=(-np.pi, np.pi), density=True)
        hist = hist + 1e-8
        orientation_entropy = float(-(hist * np.log(hist)).sum())

                # Texture consistency — detect over-smoothed neural-rendering regions
        blur_fine = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_coarse = cv2.GaussianBlur(gray, (15, 15), 0)
        high_freq_energy = float(np.mean((gray - blur_fine) ** 2))
        texture_band_energy = float(np.mean((blur_fine - blur_coarse) ** 2))
        smoothness_ratio = high_freq_energy / (texture_band_energy + 1e-6)

        local_mean = cv2.blur(gray, (9, 9))
        local_var = cv2.blur((gray - local_mean) ** 2, (9, 9))

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)

        return np.array(
             [
                edge_density,
                float(edge_mag.mean()),
                float(edge_mag.std()),
                orientation_entropy,
                high_freq_energy,
                texture_band_energy,
                smoothness_ratio,
                float(local_var.mean()),
                float(local_var.std()),
                float(laplacian.var()),
             ], dtype=np.float32,
                )

    def texture_edge_vector(self, frames: List[np.ndarray]) -> torch.Tensor:
        stats = np.stack([self._texture_edge_stats(f) for f in frames], axis=0)
        summarized = np.mean(stats, axis=0)
        return torch.tensor(
            summarized, dtype=torch.float32, device=self.device
                ).unsqueeze(0)


class ArtifactModule(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.score = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        feat = self.net(x)
        score = float(self.score(feat).mean().detach().cpu().item())
        return feat, score


class DepthConsistencyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        self.proj = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def _midas_depth_stats(self, frame: np.ndarray, device: torch.device) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return np.array(
            [
                float(depth.mean()),
                float(depth.std()),
                float(depth.max()),
                float(depth.min()),
                float(np.percentile(depth, 25)),
                float(np.percentile(depth, 50)),
                float(np.percentile(depth, 75)),
                float(gray.mean()),
                float(gray.std()),
                float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            ],
            dtype=np.float32,
        )

    def forward(self, frames: List[np.ndarray], device: torch.device) -> tuple[torch.Tensor, float]:
        self.midas.to(device)
        stats = np.stack([self._midas_depth_stats(frame, device) for frame in frames], axis=0)
        stats = np.mean(stats, axis=0, keepdims=True)
        x = torch.tensor(stats, dtype=torch.float32, device=device)
        feat = self.proj(x)
        score = float(self.score_head(feat).mean().detach().cpu().item())
        return feat, score

    def _pseudo_depth_stats(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        depth = cv2.magnitude(grad_x, grad_y)
        return np.array(
            [
                float(depth.mean()),
                float(depth.std()),
                float(depth.max()),
                float(depth.min()),
                float(np.percentile(depth, 25)),
                float(np.percentile(depth, 50)),
                float(np.percentile(depth, 75)),
                float(gray.mean()),
                float(gray.std()),
                float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            ],
            dtype=np.float32,
        )

    def forward(self, frames: List[np.ndarray], device: torch.device) -> tuple[torch.Tensor, float]:
        stats = np.stack([self._pseudo_depth_stats(frame) for frame in frames], axis=0)
        stats = np.mean(stats, axis=0, keepdims=True)
        x = torch.tensor(stats, dtype=torch.float32, device=device)
        feat = self.proj(x)
        score = float(self.score_head(feat).mean().detach().cpu().item())
        return feat, score


class MambaTemporalModule(nn.Module):
    def __init__(self, dim: int = 1280, hidden: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim, hidden)
        self.mamba = Mamba(d_model=hidden, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, out_dim)
        self.score_head = nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid())

    def forward(self, seq_features: torch.Tensor) -> tuple[torch.Tensor, float]:
        # seq_features: (seq_len, dim) -> add batch dim -> (1, seq_len, dim)
        x = self.input_proj(seq_features).unsqueeze(0)
        x = self.mamba(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)  # temporal pooling -> (1, hidden)
        y = self.output_proj(pooled)
        score = float(self.score_head(y).mean().detach().cpu().item())
        return y, score


class LightingPhysicsModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def _lighting_stats(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        reflectance_proxy = gray / (np.maximum(cv2.GaussianBlur(gray, (21, 21), 0), 1.0))
        return np.array(
            [
                float(h.mean()),
                float(s.mean()),
                float(v.mean()),
                float(h.std()),
                float(s.std()),
                float(v.std()),
                float(lap.var()),
                float(np.abs(gx).mean()),
                float(np.abs(gy).mean()),
                float(reflectance_proxy.mean()),
                float(reflectance_proxy.std()),
                float(np.corrcoef(gray.flatten(), v.astype(np.float32).flatten())[0, 1] if gray.std() > 1e-6 else 0.0),
            ],
            dtype=np.float32,
        )

    def forward(self, frames: List[np.ndarray], device: torch.device) -> tuple[torch.Tensor, float]:
        stats = np.stack([self._lighting_stats(frame) for frame in frames], axis=0)
        mean_stats = np.mean(stats, axis=0)
        # Temporal std across frames — captures flicker/inconsistency in
            # lighting and reflectance, which is the actual 3DGS/NeRF-relevant
            # signal (splat popping, view-dependent specular instability).
            # A real video's lighting stats stay comparatively stable frame to
            # frame; neural-rendering artifacts show up as instability here.
        temporal_std_stats = np.std(stats, axis=0) if len(frames) > 1 else np.zeros_like(mean_stats)
        summarized = np.concatenate([mean_stats, temporal_std_stats])[None, :]
        feat = self.project(torch.tensor(summarized, dtype=torch.float32, device=device))
        score = float(self.score_head(feat).mean().detach().cpu().item())
        return feat, score


class FusionClassifier(nn.Module):
    def __init__(self, in_dim: int = 320) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GradCamPlusPlusGenerator:
    def __init__(self, extractor: SpatialFrequencyExtractor, artifact_module: "ArtifactModule") -> None:
        self.extractor = extractor
        self.artifact_module = artifact_module
        self.device = extractor.device

    def _cam_from_tensor(self, frame: np.ndarray) -> np.ndarray:
        tensor = _normalize_frame(frame).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)

        activations = self.extractor.feature_extractor(tensor)
        activations.retain_grad()
        pooled = self.extractor.pool(activations).flatten(1)

        # Frequency and texture/edge context are non-differentiable (cv2/numpy),
        # so they're held constant here — the gradient flows back only through
        # the spatial pathway, which is exactly what Grad-CAM++ needs to localize.
        freq = self.extractor.frequency_vector([frame]).detach()
        texture_edge = self.extractor.texture_edge_vector([frame]).detach()
        artifact_in = torch.cat([pooled, freq, texture_edge], dim=1)

        feat = self.artifact_module.net(artifact_in)
        fake_score = self.artifact_module.score(feat).squeeze()
        fake_score.backward()

        gradients = activations.grad
        if gradients is None:
            raise RuntimeError("Gradients unavailable for CAM generation")
        a = activations.detach()[0]
        g = gradients.detach()[0]

        alpha_num = g.pow(2)
        alpha_denom = 2 * g.pow(2) + (a * g.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)
        weights = (alpha * torch.relu(g)).sum(dim=(1, 2))

        cam = torch.relu((weights[:, None, None] * a).sum(dim=0)).cpu().numpy()
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def generate_overlay(self, frame: np.ndarray, verdict: str, output_path: Path) -> str:
        # `verdict` kept for signature compatibility with pipeline_service.py —
        # the heatmap always shows what pushed the artifact score toward FAKE,
        # which is the conventional forensic-heatmap reading regardless of verdict.
        try:
            cam = self._cam_from_tensor(frame)
        except Exception:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            cam = cv2.magnitude(gx, gy)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
        return str(output_path)

    def generate_overlay_sequence(self, frames: List[np.ndarray], verdict: str, output_dir: Path, max_frames: int = 5) -> List[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if len(frames) <= max_frames:
            chosen = list(enumerate(frames))
        else:
            step = len(frames) / max_frames
            chosen = [(int(i * step), frames[int(i * step)]) for i in range(max_frames)]

        paths = []
        for seq_idx, (frame_idx, frame) in enumerate(chosen):
            out_path = output_dir / f"heatmap_{seq_idx:02d}_frame{frame_idx}.jpg"
            saved = self.generate_overlay(frame, verdict, out_path)
            paths.append(saved)
        return paths


class DeepGuardModelStack:
    def __init__(self) -> None:
        torch.manual_seed(42)
        # ADD near the top of __init__
        logging.warning(f"[DeepGuard-NR] CUDA available: {torch.cuda.is_available()}")
        np.random.seed(42)
        self.extractor = SpatialFrequencyExtractor()
        self.device = self.extractor.device
        self.artifact = ArtifactModule(in_dim=1280 + 1284 + 10).to(self.device).eval()
        self.depth = DepthConsistencyModule().to(self.device).eval()
        self.temporal = MambaTemporalModule().to(self.device).eval()
        self.lighting = LightingPhysicsModule().to(self.device).eval()
        self.fusion = FusionClassifier().to(self.device).eval()
        self.gradcam = GradCamPlusPlusGenerator(self.extractor, self.artifact)
        self._load_checkpoint_if_available()

    def _load_checkpoint_if_available(self) -> None:
        checkpoint_path = Path(settings.checkpoint_path)
        if not checkpoint_path.exists():
            logging.warning(
                f"[DeepGuard-NR] No trained checkpoint found at {checkpoint_path} — "
                "running with untrained baseline weights."
            )
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.artifact.load_state_dict(checkpoint["artifact"])
        self.depth.load_state_dict(checkpoint["depth"])
        self.temporal.load_state_dict(checkpoint["temporal"])
        self.lighting.load_state_dict(checkpoint["lighting"])
        self.fusion.load_state_dict(checkpoint["fusion"])
        logging.info(f"[DeepGuard-NR] Loaded trained checkpoint from {checkpoint_path}")

    def save_checkpoint(self, path: str | None = None) -> str:
        checkpoint_path = Path(path or settings.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "artifact": self.artifact.state_dict(),
                "depth": self.depth.state_dict(),
                "temporal": self.temporal.state_dict(),
                "lighting": self.lighting.state_dict(),
                "fusion": self.fusion.state_dict(),
            },
            checkpoint_path,
        )
        return str(checkpoint_path)
    @torch.no_grad()
    def infer(self, frames: List[np.ndarray]) -> tuple[str, float, "MultimodalFeatures"]:
        spatial_seq = self.extractor.spatial_sequence(frames)
        spatial_mean = spatial_seq.mean(dim=0, keepdim=True)
        freq = self.extractor.frequency_vector(frames)
        texture_edge = self.extractor.texture_edge_vector(frames)
        artifact_in = torch.cat([spatial_mean, freq, texture_edge], dim=1)

        artifact_feat, artifact_score = self.artifact(artifact_in)
        print("[DeepGuard-NR] artifact module done")
        depth_feat, depth_score = self.depth(frames, self.device)
        print("[DeepGuard-NR] depth module done")
        temporal_feat, temporal_score = self.temporal(spatial_seq)
        print("[DeepGuard-NR] temporal (mamba) module done")
        lighting_feat, lighting_score = self.lighting(frames, self.device)
        print("[DeepGuard-NR] lighting module done")

        fused_in = torch.cat([artifact_feat, depth_feat, temporal_feat, lighting_feat], dim=1)
        fake_prob = float(self.fusion(fused_in).squeeze().item())

        verdict = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = fake_prob if verdict == "FAKE" else 1.0 - fake_prob

        features = MultimodalFeatures(
                artifact_feature=artifact_feat,
                depth_feature=depth_feat,
                temporal_feature=temporal_feat,
                lighting_feature=lighting_feat,
                artifact_score=artifact_score,
                depth_score=depth_score,
                temporal_score=temporal_score,
                lighting_score=lighting_score,
                )
        return verdict, confidence, features
    def train_mode(self) -> None:
        self.artifact.train()
        self.depth.train()
        self.temporal.train()
        self.lighting.train()
        self.fusion.train()

    def eval_mode(self) -> None:
            self.artifact.eval()
            self.depth.eval()
            self.temporal.eval()
            self.lighting.eval()
            self.fusion.eval()

    def trainable_parameters(self):
        modules = [self.artifact, self.depth.proj, self.depth.score_head,self.temporal, self.lighting, self.fusion]
        for module in modules:
            yield from module.parameters()
