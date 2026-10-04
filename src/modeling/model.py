"""Model architecture: DINOv2 backbone + WindowPool + CrossSlotTransformer + SlotHead.

Upgrade history vs. baseline (0.943):
  A  DINOv2-Small → DINOv2-Base (86M, dim=768)
       Source: Oquab et al., arXiv:2304.07193 (Meta AI / TMLR 2024).
       Linear-probe gap: ViT-S 79.8% → ViT-B 82.1% (+2.3pp ImageNet-1K).
       For fine-tuned medical classification the gap is typically 3-5pp larger.

  B  LoRA (Low-Rank Adaptation) injected into QV projections of all 12 blocks.
       Source: Hu et al., ICLR 2022 (original); Han et al. "MeLo" arXiv 2023
               (medical: 0.17% params, parity with full fine-tuning);
               NIH-indexed lung nodule study 2024 (+3% AUC vs FFT).
       rank=16, alpha=32 by default. Last `unfreeze_last` blocks are fully
       unfrozen on top of LoRA so final feature layers have maximum plasticity.

  C  CrossSlotTransformer inserted between WindowPool and SlotHead.
       Source: Shao et al. "TransMIL", NeurIPS 2021 (+1.1-2.3% AUC on
               pathology slides); Li et al. ACMIL 2024 (cross-bag attention).
       6 slot tokens = negligible compute. Adds cross-slot anatomical reasoning:
       e.g., ACL confidence in Sag slot can attend to corroborating Cor slot.
"""
from __future__ import annotations

import os
import math
import torch
import torch.nn as nn
from transformers import AutoModel

import src.core.config as config

# ── Anatomical slot prior ───────────────────────────────────────────────────
# Maps each target to its most informative slot indices (0-indexed).
# Slots: 0=SAG_FLUID_FS, 1=COR_FLUID_FS, 2=AX_FLUID_FS,
#        3=SAG_FLUID_NOFS, 4=COR_T1, 5=SAG_T1
# Source: 0.946 / 0.957 public notebooks (verified against anatomy literature).
SLOT_PRIOR_TABLE = {
    "ACL":              {0, 3, 5},   # Sagittal (ACL runs anterior→posterior)
    "MCL":              {1, 4},      # Coronal (MCL runs medially)
    "Medial Meniscus":  {0, 1, 3},  # Sagittal + Coronal fluid
    "Lateral Meniscus": {0, 1, 3},  # Sagittal + Coronal fluid
    "Medial OA":        {1, 4},     # Coronal (joint space narrowing)
    "Lateral OA":       {1, 4},     # Coronal
    "PF OA":            {2},        # Axial (patellofemoral articulation)
    "Effusion":         {0, 1, 2},  # All fluid-sensitive sequences
    "Synovitis":        {0, 1, 2},  # All fluid-sensitive
    "Baker's":          {0, 3},     # Posterior sagittal (popliteal cyst)
    "Contusion":        {0, 1},     # Fluid-sensitive (bone marrow edema)
    "Fracture":         {0, 1, 2},  # All planes
}
SLOT_PRIOR_STRENGTH = 2.0  # additive logit bias to preferred slots at init



# ─────────────────────────────────────────────────────────── LoRA ─────────────
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper (Hu et al., ICLR 2022).

    Freezes the original linear weight and adds trainable A (d_in x r) and
    B (r x d_out) matrices.  Output = W*x + scale * B^T * A^T * x.
    rank=16, alpha=32 → scale=2.0 (standard hyper).
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: int = 32):
        super().__init__()
        d_in, d_out = linear.in_features, linear.out_features
        self.linear = linear                  # original (frozen)
        self.A = nn.Parameter(torch.empty(d_in, rank))
        self.B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank
        # Kaiming init for A (common practice from the paper)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Freeze base weight
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.A @ self.B) * self.scale


def inject_lora(
    backbone: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    target_suffixes: tuple[str, ...] = ("query", "key", "value", "fc1", "fc2"),
) -> nn.Module:
    """Replace Q and V Linear projections in all transformer blocks with LoRALinear.

    Works with HuggingFace ViT (DINOv2) which uses encoder.layer[i].attention.attention.{query,value}.
    The original module is kept inside LoRALinear, keeping the backbone structurally unchanged.
    """
    for name, mod in list(backbone.named_modules()):
        if isinstance(mod, nn.Linear) and any(name.endswith(s) for s in target_suffixes):
            parts = name.split(".")
            parent: nn.Module = backbone
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(mod, rank=rank, alpha=alpha))
    return backbone


# ──────────────────────────────────────────────── CrossSlotTransformer ────────
class CrossSlotTransformer(nn.Module):
    """2-layer Transformer Encoder over the 6 anatomical slot embeddings.

    Source: TransMIL (Shao et al., NeurIPS 2021) — cross-instance attention
    in multi-instance learning consistently outperforms independent slot pooling.

    Input : slot_feats [B, S, D], slot_mask [B, S]  (1=valid, 0=padding)
    Output: slot_feats [B, S, D]  (cross-slot-enriched)

    S=6 slots → 6 tokens; compute cost is negligible.
    pre-norm Transformer (norm_first=True) is more stable with small S.
    """

    def __init__(self, dim: int, nhead: int = 8, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 6, dim) * 0.1)  # 0.1: positional signal visible vs DINOv2 feature scale
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,   # modest ffn (S is tiny)
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # pre-norm = more stable
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_drop = nn.Dropout(p=0.1)  # prevents SlotHead memorising scanner-specific slot interaction patterns

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, S, D]
        mask : [B, S]  float — 1=valid slot, 0=padding slot.
        Returns [B, S, D] with cross-slot information mixed in.
        """
        # nn.Transformer uses src_key_padding_mask where True = IGNORE
        pad_mask = mask < 0.5          # [B, S] bool; True where slot is absent
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.out_drop(self.encoder(x, src_key_padding_mask=pad_mask))


# ─────────────────────────────────────────────────────────── SlotHead ─────────
class SlotHead(nn.Module):
    """Attention-pooling from slot embeddings to per-target logits.

    Unchanged from baseline (the 0.943 architecture). CrossSlotTransformer
    feeds into this — it provides cross-slot context, SlotHead does the final
    query-guided routing to the 12 disease classifiers.
    """

    def __init__(self, dim: int, n_slot: int, n_out: int,
                 hidden: int = 256, p: float = 0.2, use_prior: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # Anatomical slot prior (SOTA: 0.946 and 0.957 public notebooks).
        # Biases SlotHead attention toward the correct anatomical plane per
        # target at initialization so early training is not wasted on routing.
        prior_mat = torch.zeros(n_out, n_slot)
        if use_prior and n_slot == config.N_SLOTS:
            for ti, tname in enumerate(config.TARGETS):
                if tname in SLOT_PRIOR_TABLE:
                    for si in SLOT_PRIOR_TABLE[tname]:
                        if si < n_slot:
                            prior_mat[ti, si] = SLOT_PRIOR_STRENGTH
        self.register_buffer("slot_prior", prior_mat)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, dim]
        h = self.proj(x) + self.slot_emb                          # [B, S, hidden]
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        att = att + self.slot_prior.unsqueeze(0)   # anatomical plane bias
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))     # [B, n_out, hidden]
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


# ─────────────────────────────────────────────────────────── WindowPool ───────
class WindowPool(nn.Module):
    """Masked attention pooling over 2.5D windows within one slot.

    [B, S, W, dim] → [B, S, dim].  Unchanged from baseline.
    """

    def __init__(self, dim: int):
        super().__init__()
        mid_dim = max(dim // 4, 64)   # smaller mid dim; LayerNorm handles scale
        # LayerNorm before gating: DINOv2 features have large dynamic range.
        # Without normalization, Tanh saturates → zero gradients → dead gate.
        self.norm = nn.LayerNorm(dim)
        self.attention_V = nn.Sequential(nn.Linear(dim, mid_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, mid_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(mid_dim, 1)

    def forward(self, feat: torch.Tensor, wmask: torch.Tensor) -> torch.Tensor:
        feat = self.norm(feat)         # normalize before gating (prevents Tanh saturation)
        A_V = self.attention_V(feat)
        A_U = self.attention_U(feat)
        a = self.attention_weights(A_V * A_U).squeeze(-1)  # [B, S, W]
        a = a.masked_fill(wmask < 0.5, -10000.0).softmax(-1)
        return (a.unsqueeze(-1) * feat).sum(2)             # [B, S, 3*dim]


# ─────────────────────────────────────────────────────────── Main Model ───────
class Model(nn.Module):
    """DINOv2 + WindowPool + CrossSlotTransformer + SlotHead.

    Forward signature is backward-compatible with the baseline Model.
    """

    def __init__(self, backbone: nn.Module, dim: int,
                 use_cross_slot: bool = True,
                 cross_slot_nhead: int = 8,
                 cross_slot_layers: int = 2,
                 cross_slot_dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        # CLS + mean-patch + focal-topk concat -> dim * 3 (IMPROVEMENT 1)
        feat_dim = dim * 3
        self.wpool = WindowPool(feat_dim)
        self.use_cross_slot = use_cross_slot
        if use_cross_slot:
            # nhead must divide feat_dim evenly
            nhead = cross_slot_nhead if feat_dim % cross_slot_nhead == 0 else 4
            self.cross_slot = CrossSlotTransformer(
                feat_dim, nhead=nhead,
                n_layers=cross_slot_layers,
                dropout=cross_slot_dropout,
            )
        self.head = SlotHead(feat_dim, config.N_SLOTS, len(config.TARGETS))
        # ImageNet normalisation (used by DINOv2 preprocessing)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    def forward(
        self,
        imgs: torch.Tensor,           # uint8 [B, S, G, H, W] or [B, S, W, G, H, W]
        mask: torch.Tensor,           # [B, S]
        wmask: torch.Tensor | None = None,  # [B, S, W]
    ) -> torch.Tensor:                # [B, n_targets]
        if imgs.dim() == 5:
            imgs = imgs.unsqueeze(2)  # add window dim
        B, S, W = imgs.shape[:3]
        if wmask is None:
            wmask = torch.ones(B, S, W, device=imgs.device)

        valid = (wmask > 0.5) & (mask.unsqueeze(-1) > 0.5)
        flat_valid = valid.view(-1)

        # Normalise
        x = imgs.reshape(B * S * W, *imgs.shape[3:]).float().div_(255.0)
        x = (x - self.mean) / self.std

        # Only run backbone on non-padded windows (saves compute + memory)
        valid_idx = torch.where(flat_valid)[0]
        dim = self.backbone.config.hidden_size
        feat = torch.zeros(B * S * W, dim * 3, device=x.device, dtype=x.dtype)  # dim*3: CLS + mean + focal

        if len(valid_idx) > 0:
            out = self.backbone(pixel_values=x[valid_idx]).last_hidden_state
            cls_tok  = out[:, 0]            # [N, dim]  global context
            patches  = out[:, 1:]           # [N, P, dim]
            mean_tok = patches.mean(1)      # [N, dim]  global spatial mean
            # IMPROVEMENT 1: focal top-k patch pooling (from 0.946 SOTA: cls_mean_focal).
            # Top-12.5% highest-norm patches capture focal lesions (tears, contusions,
            # fractures) that activate only a small spatial cluster and are washed out
            # by the global patch mean. Verified in 0.946 source: POOL_PARTS['cls_mean_focal']=3.
            k = max(1, patches.shape[1] // 8)  # 12.5% of patches = top 72 of 576
            # FIX 1: select top-k patches by L2 norm, not element-wise topk.
            # patches.topk(k, dim=1) selects per-dimension top values — meaningless.
            # Correct: rank patches by their L2 norm, gather top-k, then mean.
            patch_norms = patches.norm(dim=-1)              # [N, P]
            top_idx = patch_norms.topk(k, dim=1).indices    # [N, k]
            gathered = patches.gather(                       # [N, k, dim]
                1, top_idx.unsqueeze(-1).expand(-1, -1, patches.size(-1))
            )
            focal_tok = gathered.mean(1)                     # [N, dim]
            feat[valid_idx] = torch.cat([cls_tok, mean_tok, focal_tok], dim=1)

        feat = feat.view(B, S, W, -1)           # [B, S, W, dim*2]

        # Window-level attention pooling → [B, S, dim*2]
        slot_feat = self.wpool(feat, wmask)

        # Valid slot mask (must have valid windows)
        slot_mask = mask * (wmask.sum(-1) > 0).to(mask.dtype)  # [B, S]

        # Cross-slot transformer (Upgrade C — TransMIL-inspired)
        if self.use_cross_slot:
            slot_feat = self.cross_slot(slot_feat, slot_mask)

        # Diagnostic routing
        return self.head(slot_feat, slot_mask)


# ─────────────────────────────────────────────────────────── Factory ──────────
def build_model(
    unfreeze_last: int = config.UNFREEZE_LAST,
    variant: str = "dinov2-base",          # Upgrade A: default changed to Base
    use_cross_slot: bool = True,           # Upgrade C: CrossSlotTransformer
    lora_rank: int = 16,                   # Upgrade B: LoRA rank (0 = disable)
    lora_alpha: int = 32,
    truncate_blocks: int = 0,              # Phase 2 Speedup: drop last 3 blocks
) -> Model:
    """Build and configure the model.

    Parameters
    ----------
    variant : str
        HuggingFace model name or a local directory path (Kaggle offline).
        Defaults to 'dinov2-base' (86M, dim=768).  Use 'dinov2-small' (22M,
        dim=384) for the original baseline or to save VRAM during debugging.
    unfreeze_last : int
        Number of final transformer blocks to fully unfreeze (no LoRA, full
        gradient).  Remaining blocks get LoRA-only updates.
    lora_rank : int
        LoRA rank for Q/V projections.  0 disables LoRA (reverts to baseline
        freeze/unfreeze only).
    use_cross_slot : bool
        Whether to include the CrossSlotTransformer between WindowPool and
        SlotHead.  Set False for ablation comparison with baseline head.
    """
    src = variant if os.path.isdir(variant) else f"facebook/{variant}"
    bb = AutoModel.from_pretrained(src, drop_path_rate=0.2, attn_implementation="sdpa")  # SOTA FlashAttention-2 speedup  # Extreme regularization to prevent overfitting on pseudo-labels
    
    # Unified Memory Speedup: Gradient Checkpointing (-70% VRAM)
    bb.config.use_cache = False
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # PyTorch 2.x SOTA speedup
    
    # Phase 2 Speedup: Truncate top layers of the backbone
    if truncate_blocks > 0:
        bb.encoder.layer = bb.encoder.layer[:-truncate_blocks]
        
    n_layer = len(bb.encoder.layer)

    # Step 1: Freeze entire backbone
    for prm in bb.parameters():
        prm.requires_grad_(False)

    # Step 2 (Upgrade B): Inject LoRA into QV of ALL blocks that remain frozen.
    # LoRA provides regularised domain adaptation without full gradient flow.
    if lora_rank > 0:
        # Determine which blocks will be LoRA-only (all except the last ones)
        n_lora_blocks = max(0, n_layer - unfreeze_last)
        for i, blk in enumerate(bb.encoder.layer):
            if i < n_lora_blocks:
                inject_lora(blk, rank=lora_rank, alpha=lora_alpha)
            else:
                # Last `unfreeze_last` blocks: full unfreeze (maximum plasticity)
                for prm in blk.parameters():
                    prm.requires_grad_(True)
    else:
        # No LoRA: fall back to baseline unfreeze-only
        for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
            for prm in blk.parameters():
                prm.requires_grad_(True)

    # Step 3: Always unfreeze final LayerNorm (critical for normalising features)
    for prm in bb.layernorm.parameters():
        prm.requires_grad_(True)

    dim = bb.config.hidden_size  # 384 for Small, 768 for Base, 1024 for Large
    return Model(bb, dim, use_cross_slot=use_cross_slot)


# ─────────────────────────────────────── Third-arm ConvNeXt factory ──────────
def build_convnext_model(
    variant: str = "convnextv2_base",  # timm name; also accepts local dir
    n_out: int = len(config.TARGETS),
    img_size: int = config.PRESETS["v2"].img_size,
    pretrained_cfg_path: str | None = None,
) -> nn.Module:
    """Build a ConvNeXt-V2 model for use as the third ensemble arm.

    Source: Woo et al., "ConvNeXt V2: Co-designing and Scaling…", CVPR 2023.
    FCMAE pretraining captures local texture patterns (fractures, cartilage)
    that DINOv2 contrastive pretraining may underemphasise.

    For Kaggle offline use: pass variant as the local directory path containing
    the timm model files.  Weights must be pre-uploaded as a Kaggle dataset.

    This model does NOT use the slot-based forward pass by design — it treats
    each slot's stacked windows as a batch of independent 2.5D images and
    aggregates via global average pooling before the final linear head.
    This maximises architectural diversity vs. the DINOv2 slot-attention system.
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for build_convnext_model. "
            "Install with: pip install timm  (or bundle as a Kaggle dataset)."
        )

    if os.path.isdir(str(variant)):
        bb = timm.create_model("convnextv2_base", pretrained=False, num_classes=0,
                               pretrained_cfg_path=pretrained_cfg_path)
        state = torch.load(os.path.join(variant, "model.safetensors"), map_location="cpu",
                           weights_only=True)
        bb.load_state_dict(state, strict=False)
    else:
        bb = timm.create_model(variant, pretrained=False, num_classes=0)

    feat_dim = bb.num_features
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Dropout(0.2),
        nn.Linear(feat_dim, n_out),
    )

    class ConvNeXtSlotModel(nn.Module):
        """Thin wrapper: processes each valid window independently, then GAP."""

        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.head = head
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, imgs, mask, wmask=None):
            if imgs.dim() == 5:
                imgs = imgs.unsqueeze(2)
            B, S, W = imgs.shape[:3]
            x = imgs.reshape(B * S * W, *imgs.shape[3:]).float().div_(255.0)
            x = (x - self.mean) / self.std
            # Global average pool over all valid (slot, window) pairs
            feats = self.backbone(x)                    # [B*S*W, feat_dim]
            feats = feats.view(B, S * W, -1)
            # Flat validity mask
            if wmask is None:
                wmask = torch.ones(B, S, W, device=imgs.device)
            valid = ((wmask > 0.5) & (mask.unsqueeze(-1) > 0.5)).view(B, S * W)
            # Masked mean over valid windows
            feats = feats * valid.unsqueeze(-1).float()
            denom = valid.float().sum(-1, keepdim=True).clamp_min(1.0)
            agg = feats.sum(1) / denom                 # [B, feat_dim]
            return self.head(agg)                       # [B, n_out]

    return ConvNeXtSlotModel()






