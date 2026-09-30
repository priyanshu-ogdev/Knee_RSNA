import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import config

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

class Model(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.backbone = backbone
        # We concatenate cls token with mean patched tokens, so dim * 2
        self.head = SlotHead(dim * 2, config.N_SLOTS, len(config.TARGETS))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        
        # Pooling: Concatenate [CLS] and Mean of patch tokens
        cls_token = out[:, 0]
        patch_tokens = out[:, 1:].mean(1)
        feat = torch.cat([cls_token, patch_tokens], dim=1).reshape(B, S, -1)
        
        return self.head(feat, mask)

def build_model(unfreeze_last=config.UNFREEZE_LAST, variant="dinov2-small"):
    # Note: Replace 'variant' with local path to weights if needed.
    # In the kaggle environment, this loads from local dataset directories.
    bb = AutoModel.from_pretrained(f"facebook/{variant}")
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
