import os
import torch
import torch.nn as nn
from transformers import AutoModel
from . import config


class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x, mask):
        # x: [B, S, dim]
        h = self.proj(x) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class WindowPool(nn.Module):
    """Masked attention pooling over the windows (2.5D triplets) of one slot: [B,S,W,dim] -> [B,S,dim]."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, feat, wmask):
        a = self.score(feat).squeeze(-1).masked_fill(wmask < 0.5, -10000.0).softmax(-1)
        return (a.unsqueeze(-1) * feat).sum(2)


class Model(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        # We concatenate cls token with mean patched tokens, so dim * 2
        self.wpool = WindowPool(dim * 2)
        self.head = SlotHead(dim * 2, config.N_SLOTS, len(config.TARGETS))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, wmask=None):
        """imgs: uint8 [B,S,G,H,W] (one window per slot, legacy) or [B,S,W,G,H,W]; mask [B,S]; wmask [B,S,W]."""
        if imgs.dim() == 5:
            imgs = imgs.unsqueeze(2)
        B, S, W = imgs.shape[:3]
        if wmask is None:
            wmask = torch.ones(B, S, W, device=imgs.device)
            
        valid = (wmask > 0.5) & (mask.unsqueeze(-1) > 0.5)
        flat_valid = valid.view(-1)
        
        x = imgs.reshape(B * S * W, *imgs.shape[3:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        
        valid_idx = torch.where(flat_valid)[0]
        dim = self.backbone.config.hidden_size
        feat = torch.zeros(B * S * W, dim * 2, device=x.device, dtype=x.dtype)
        
        if len(valid_idx) > 0:
            x_valid = x[valid_idx]
            out_valid = self.backbone(pixel_values=x_valid).last_hidden_state
            cls_token = out_valid[:, 0]
            patch_tokens = out_valid[:, 1:].mean(1)
            feat_valid = torch.cat([cls_token, patch_tokens], dim=1)
            feat[valid_idx] = feat_valid
            
        feat = feat.view(B, S, W, -1)
        slot_feat = self.wpool(feat, wmask)
        slot_mask = mask * (wmask.sum(-1) > 0).to(mask.dtype)
        return self.head(slot_feat, slot_mask)


def build_model(unfreeze_last=config.UNFREEZE_LAST, variant="dinov2-small"):
    # Note: Replace 'variant' with local path to weights if needed.
    # In the kaggle environment, this loads from local dataset directories.
    src = variant if os.path.isdir(variant) else f"facebook/{variant}"
    bb = AutoModel.from_pretrained(src)
    n_layer = len(bb.encoder.layer)

    # Freeze the entire backbone first
    for prm in bb.parameters():
        prm.requires_grad = False

    # Unfreeze the last unfreeze_last blocks
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True

    # Unfreeze layernorm
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True

    dim = bb.config.hidden_size
    return Model(bb, dim)
