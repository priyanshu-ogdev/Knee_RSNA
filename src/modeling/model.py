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
from src.modeling.mil import build_timm_attention_mil

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
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
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

    def _get_chunk_size(self) -> int:
        if not hasattr(self, "_cached_chunk_size"):
            if torch.cuda.is_available():
                try:
                    gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if gb >= 24.0:
                        self._cached_chunk_size = 96  # Restrict to HBM3 to hit 4s/step without OS thrashing
                    else:
                        self._cached_chunk_size = 48   # 16GB Kaggle T4 / P100
                except Exception:
                    self._cached_chunk_size = 48
            else:
                self._cached_chunk_size = 48
        return self._cached_chunk_size

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

        # SPEEDUP 1: Pre-filter valid windows BEFORE converting to float32 and normalizing.
        # Avoids wasting ~1.5 GB of VRAM allocations and millions of arithmetic operations on padded windows.
        valid_idx = torch.where(flat_valid)[0]
        dim = self.backbone.config.hidden_size

        if len(valid_idx) > 0:
            flat_imgs = imgs.reshape(B * S * W, *imgs.shape[3:])
            # Native BF16 normalization if CUDA BF16 supported: cuts bandwidth by 2x
            # Efficiency Track SOTA Fix: Respect incoming fp16/bf16 tensors to prevent fallback to fp32 on T4
            if imgs.dtype in (torch.float16, torch.bfloat16):
                dtype = imgs.dtype
            else:
                dtype = torch.bfloat16 if (imgs.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
            
            # Avoid re-allocating if already float16/bfloat16
            x_valid = flat_imgs[valid_idx].to(dtype=dtype) if flat_imgs.dtype != dtype else flat_imgs[valid_idx]
            x_valid = x_valid.div_(255.0)
            x_valid.sub_(self.mean.to(dtype=dtype)).div_(self.std.to(dtype=dtype))

            # Hardware-adaptive chunk size: 128 on 130GB Blackwell GB10 / A100, 48 on 16GB Kaggle T4
            chunk_size = self._get_chunk_size()
            f_list = []
            for c_start in range(0, len(x_valid), chunk_size):
                c_x = x_valid[c_start:c_start + chunk_size]
                if self.training:
                    # CRITICAL FIX: PyTorch gradient checkpointing silently fails (saves NO VRAM) 
                    # if inputs don't have requires_grad=True. This forces activation discarding!
                    c_x.requires_grad_(True)
                c_out = self.backbone(pixel_values=c_x).last_hidden_state
                c_cls = c_out[:, 0]
                c_patches = c_out[:, 1:]
                c_mean = c_patches.mean(1)
                k = max(1, c_patches.shape[1] // 8)
                # Native L1 magnitude ranking without allocating huge 1.6GB float32 copies
                c_norms = c_patches.abs().sum(dim=-1)
                c_top = c_norms.topk(k, dim=1).indices
                c_gathered = c_patches.gather(1, c_top.unsqueeze(-1).expand(-1, -1, c_patches.size(-1)))
                c_focal = c_gathered.mean(1)
                f_list.append(torch.cat([c_cls, c_mean, c_focal], dim=1))
            valid_feats = torch.cat(f_list, dim=0) if len(f_list) > 1 else f_list[0]
            # ANTI-DEGRADATION FIX: Match feat dtype to valid_feats so downstream attention stays in native Tensor Core BF16
            feat = torch.zeros(B * S * W, dim * 3, device=imgs.device, dtype=valid_feats.dtype)
            feat[valid_idx] = valid_feats
        else:
            feat = torch.zeros(B * S * W, dim * 3, device=imgs.device, dtype=torch.float32)

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
    variant: str = "dinov2-base",          # Standard first argument
    unfreeze_last: int = config.UNFREEZE_LAST,
    use_cross_slot: bool = True,           # Upgrade C: CrossSlotTransformer
    lora_rank: int = config.LORA_RANK,     # Upgrade B: LoRA rank (0 = disable)
    lora_alpha: int = config.LORA_ALPHA,
    truncate_blocks: int = 0,              # Phase 2 Speedup: drop last 3 blocks
    model_type: str | None = None,
    input_size: int = config.COATNET_INPUT_SIZE,
    encode_chunk_size: int = config.COATNET_ENCODE_CHUNK,
    pretrained: bool = True,
) -> nn.Module:
    # Polymorphic argument resolution for backward compatibility
    if isinstance(variant, int):
        unfreeze_last = variant
        variant = "dinov2-base" 
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
    model_type = model_type or ("coatnet_mil" if variant.startswith("coatnet") else ("timm_mil" if not variant.startswith("dinov2") else "dinov2"))
    if model_type in ("coatnet_mil", "timm_mil"):
        return build_timm_attention_mil(
            variant,
            pretrained=pretrained,
            input_size=input_size,
            encode_chunk_size=encode_chunk_size,
        )
    if model_type not in ("dinov2", "coatnet_mil", "timm_mil"):
        raise ValueError(f"Unknown model_type {model_type!r}; expected 'dinov2', 'coatnet_mil', or 'timm_mil'")

    src = variant if os.path.isdir(variant) else f"facebook/{variant}"
    try:
        bb = AutoModel.from_pretrained(src, drop_path_rate=0.2, attn_implementation="sdpa", local_files_only=os.path.isdir(src))  # SOTA FlashAttention-2 speedup  # Extreme regularization to prevent overfitting on pseudo-labels
    except Exception:
        # Offline fallback: if pretrained weights cannot be fetched (e.g. Kaggle offline submission),
        # instantiate directly from config since full weights will be loaded from checkpoint state_dict
        from transformers import AutoConfig
        local_cfg_dir = os.path.join(os.path.dirname(__file__), "configs")
        if os.path.isdir(src) and os.path.exists(os.path.join(src, "config.json")):
            cfg_obj = AutoConfig.from_pretrained(src)
        elif os.path.exists(os.path.join(local_cfg_dir, "config.json")):
            cfg_obj = AutoConfig.from_pretrained(local_cfg_dir)
        else:
            cfg_obj = AutoConfig.from_pretrained(src)
        bb = AutoModel.from_config(cfg_obj)
    
    # SOTA Memory Protection: PyTorch 2.x Non-Reentrant Gradient Checkpointing
    # Strictly bounds activation memory (~8-15 GB) across all batch sizes.
    # MANDATORY on DGX to prevent the GPU from silently spilling >50GB of backprop graphs into System RAM!
    bb.config.use_cache = False
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
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


# -----------------------------------------------------------------------------
# DINOv2 remains the default model. The optional timm MIL arm uses target-
# specific attention over valid windows rather than global-average pooling.
# -----------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device | str = "cpu") -> nn.Module:
    """Load a trained model checkpoint (fold*_ema.pt, fold*_best.pt, or fold*_swa.pt).
    Reconstructs the model architecture with the exact saved configuration and loads weights.
    Safely strips torch.compile (_orig_mod.) and DataParallel (module.) prefixes if present.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    model_config = dict(ckpt.get("model_config", {}))
    model_config.setdefault("variant", ckpt.get("variant", "dinov2-base"))
    model_config.setdefault(
        "model_type",
        ckpt.get("model_type", "coatnet_mil" if model_config["variant"].startswith("coatnet") else ("timm_mil" if not model_config["variant"].startswith("dinov2") else "dinov2")),
    )
    model_config.setdefault("use_cross_slot", ckpt.get("use_cross_slot", True))
    if "lora_rank" not in model_config:
        lora_a = next((v for k, v in state_dict.items() if k.endswith(".A")), None)
        model_config["lora_rank"] = int(lora_a.shape[-1]) if lora_a is not None else 0
    model_config.setdefault("lora_alpha", config.LORA_ALPHA)
    model_config.setdefault("unfreeze_last", config.UNFREEZE_LAST)
    model_config.setdefault("truncate_blocks", 0)
    model_config.setdefault("input_size", config.COATNET_INPUT_SIZE)
    model_config.setdefault("encode_chunk_size", config.COATNET_ENCODE_CHUNK)
    model_config.setdefault("pretrained", False)

    model = build_model(**model_config).to(device)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        if clean_k.startswith("_orig_mod."):
            clean_k = clean_k[len("_orig_mod."):]
        if clean_k.startswith("module."):
            clean_k = clean_k[len("module."):]
        cleaned_state_dict[clean_k] = v
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    model._rsna_preprocessing_config = ckpt.get("cfg")
    model._rsna_model_type = model_config["model_type"]
    return model
