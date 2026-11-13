"""Standalone NCA mixer (inference: stochastic update off) with real trained weights.

Mirrors `NCAAttention` in Emergent-Attention1/src/models/nca_attention.py op for op
(same tensor layouts and copies), without the einops / hydra dependencies, so that the
"unfused" baseline we profile is exactly what the trained model runs.

    x (B, 1+N, D) -> LN(patches) -> grid (B, D, H, W)
    K times: perc = depthwise3x3(grid) (B, 3D, H, W) -> per-cell MLP 3D->HID->D -> grid += delta
    return cat([0, grid_final - grid_init])   # caller adds the residual
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = r"D:\Research\Emergent-Attention1\latest models\latest nca vit hybrid.pt"
D, HID, K, M = 192, 384, 4, 3


class NCABlockRef(nn.Module):
    def __init__(self, dim=D, hidden=HID, steps=K, m=M):
        super().__init__()
        self.dim, self.steps = dim, steps
        self.norm = nn.LayerNorm(dim)
        self.perc_w = nn.Parameter(torch.zeros(dim * m, 1, 3, 3))  # channel index = f*D + c
        self.linear1 = nn.Linear(dim * m, hidden)
        self.linear2 = nn.Linear(hidden, dim)

    def forward(self, x, grid_hw=None):
        B, n1, D_ = x.shape
        H = W = int(round((n1 - 1) ** 0.5)) if grid_hw is None else None
        if grid_hw is not None:
            H, W = grid_hw
        p = self.norm(x[:, 1:])
        grid = p.reshape(B, H, W, D_).permute(0, 3, 1, 2)       # 'b (h w) d -> b d h w' (a view)
        init = grid.clone()
        for _ in range(self.steps):
            perc = F.conv2d(grid, self.perc_w, padding=1, groups=D_)   # (B, 3D, H, W)
            perc = perc.permute(0, 2, 3, 1)                             # (B, H, W, 3D)
            delta = self.linear2(F.relu(self.linear1(perc)))            # (B, H, W, D)
            grid = grid + delta.permute(0, 3, 1, 2)
        delta = (grid - init).permute(0, 2, 3, 1).reshape(B, H * W, D_)
        return torch.cat([torch.zeros_like(x[:, :1]), delta], dim=1)


def load_state(which="model"):
    """Full state dict of the trained hybrid model. which: 'model' or 'ema_model'."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    return {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in ck[which].items()}


def load_block(idx=0, which="model", device="cuda", dtype=torch.float32):
    """NCA mixer `idx` (0..5) of the hybrid model, with trained weights."""
    sd = load_state(which)
    pre = f"nca_blocks.{idx}.nca_attn."
    blk = NCABlockRef()
    blk.load_state_dict({
        "norm.weight": sd[pre + "norm.weight"], "norm.bias": sd[pre + "norm.bias"],
        "perc_w": sd[pre + "perception.weight"],
        "linear1.weight": sd[pre + "linear1.weight"], "linear1.bias": sd[pre + "linear1.bias"],
        "linear2.weight": sd[pre + "linear2.weight"], "linear2.bias": sd[pre + "linear2.bias"],
    })
    return blk.to(device=device, dtype=dtype).eval()


if __name__ == "__main__":  # smoke test against the original module from the research repo
    import sys
    sys.path.insert(0, r"D:\Research\Emergent-Attention1")
    from src.models.hybrid_nca_vit import HybridNCAViT
    full = HybridNCAViT(learnable_filters=True, num_classes=100)
    full.load_state_dict(load_state()); full = full.cuda().eval()
    torch.manual_seed(0)
    x = torch.randn(8, 197, D, device="cuda")
    with torch.no_grad():
        for i in range(6):
            a = full.nca_blocks[i].nca_attn(x)
            b = load_block(i)(x)
            print(f"block {i}: max|orig - ref| = {(a - b).abs().max().item():.3e}  (max|out| {a.abs().max().item():.2f})")
