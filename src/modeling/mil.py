"""Timm image encoder with target-specific attention pooling over study windows."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.core.config as config


class TimmAttentionMIL(nn.Module):
    """Encode valid slot windows and pool them with one attention distribution per finding."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        input_size: int = config.COATNET_INPUT_SIZE,
        encode_chunk_size: int = config.COATNET_ENCODE_CHUNK,
    ):
        super().__init__()
        if input_size <= 0 or encode_chunk_size <= 0:
            raise ValueError("input_size and encode_chunk_size must be positive")
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.input_size = int(input_size)
        self.encode_chunk_size = int(encode_chunk_size)
        self.model_type = "coatnet_mil"
        self.norm = nn.LayerNorm(self.feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, len(config.TARGETS)),
        )
        self.classifier = nn.Parameter(torch.empty(len(config.TARGETS), self.feature_dim))
        self.bias = nn.Parameter(torch.zeros(len(config.TARGETS)))
        nn.init.trunc_normal_(self.classifier, std=0.02)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        features = []
        parameter = next(self.backbone.parameters(), None)
        dtype = parameter.dtype if parameter is not None else images.dtype
        for start in range(0, len(images), self.encode_chunk_size):
            batch = images[start:start + self.encode_chunk_size]
            if batch.shape[-2:] != (self.input_size, self.input_size):
                batch = F.interpolate(
                    batch,
                    size=(self.input_size, self.input_size),
                    mode="bilinear",
                    align_corners=False,
                )
            batch = batch.to(dtype=dtype)
            batch = (batch - self.mean.to(batch.dtype)) / self.std.to(batch.dtype)
            if self.training:
                batch.requires_grad_(True)
            encoded = self.backbone(batch)
            if isinstance(encoded, (tuple, list)):
                encoded = encoded[0]
            if encoded.ndim > 2:
                encoded = encoded.flatten(2).mean(-1)
            if encoded.ndim != 2 or encoded.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"timm backbone returned {tuple(encoded.shape)}; expected "
                    f"[N, {self.feature_dim}] pooled features"
                )
            features.append(encoded)
        return torch.cat(features, dim=0)

    def forward(
        self,
        imgs: torch.Tensor,
        slot_mask: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        if imgs.ndim != 6:
            raise ValueError(f"expected images [B,S,W,3,H,W], received {tuple(imgs.shape)}")
        batch_size, slots, windows = imgs.shape[:3]
        valid = (slot_mask.unsqueeze(-1) > 0.5) & (window_mask > 0.5)
        flat_images = imgs.reshape(batch_size * slots * windows, *imgs.shape[3:])
        valid_indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()

        feature_dtype = self.norm.weight.dtype
        flat_features = imgs.new_zeros(
            (batch_size * slots * windows, self.feature_dim), dtype=feature_dtype
        )
        if valid_indices.numel():
            encoded_chunks = []
            index_chunks = []
            for start in range(0, valid_indices.numel(), self.encode_chunk_size):
                indices = valid_indices[start:start + self.encode_chunk_size]
                selected = flat_images.index_select(0, indices).float().div(255.0)
                encoded_chunks.append(self._encode(selected))
                index_chunks.append(indices)
            encoded = torch.cat(encoded_chunks, dim=0)
            indices = torch.cat(index_chunks, dim=0)
            flat_features = flat_features.index_copy(
                0, indices, encoded.to(feature_dtype)
            )

        features = self.norm(
            flat_features.view(batch_size, slots * windows, self.feature_dim)
        )
        valid_windows = valid.reshape(batch_size, slots * windows)
        scores = self.attention(features)
        scores = scores.masked_fill(~valid_windows.unsqueeze(-1), -10000.0)
        attention = scores.softmax(dim=1)
        pooled = torch.einsum("bnc,bnf->bcf", attention, features)
        return (pooled * self.classifier.unsqueeze(0)).sum(-1) + self.bias


def build_timm_attention_mil(
    arch: str,
    pretrained: bool = False,
    input_size: int = config.COATNET_INPUT_SIZE,
    encode_chunk_size: int = config.COATNET_ENCODE_CHUNK,
) -> TimmAttentionMIL:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("The timm package is required for the CoAtNet MIL model") from exc

    backbone = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
        in_chans=3,
    )
    if hasattr(backbone, "set_grad_checkpointing"):
        backbone.set_grad_checkpointing(True)
    feature_dim = getattr(backbone, "num_features", None)
    if not feature_dim:
        raise ValueError(f"timm model {arch!r} does not expose num_features")
    return TimmAttentionMIL(
        backbone,
        feature_dim,
        input_size=input_size,
        encode_chunk_size=encode_chunk_size,
    )
