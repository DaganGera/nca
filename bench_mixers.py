"""M5 measurements: time per image of each token mixer vs resolution (224..1024 px, patch 16), fp16, this GPU.

    python bench_mixers.py  -> results/m5_mixer_timing.json
Mixers: NCA unfused (reference), NCA per-step fused, global attention materialised (N x N scores in DRAM, like
nn.MultiheadAttention with need_weights), global attention fused (SDPA/flash-style), window attention 7x7 (Swin-style, materialised per window).
Batch is scaled so every point has ~12.5k tokens (== 64 images at 224 px): B = max(2, 12544 // N). Random weights for attention (timing only);
real trained weights for the NCA. CUDA events, 20 warm-up / 100 timed, median + IQR.
"""
import math
import torch
import torch.nn.functional as F
from bench_utils import time_cuda, save
from reference import load_block
from fused import load_fused

D, HEADS, WIN = 192, 3, 7
RES = (224, 320, 448, 640, 1024)


class Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(D, 3 * D); self.proj = torch.nn.Linear(D, D)

    def heads(self, x):
        B, N, _ = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        return q, k, v

    def global_materialised(self, x):
        q, k, v = self.heads(x)
        a = (q @ k.transpose(-2, -1) * (D // HEADS) ** -0.5).softmax(-1)       # N x N scores live in DRAM
        return self.proj((a @ v).transpose(1, 2).reshape(x.shape))

    def global_fused(self, x):
        q, k, v = self.heads(x)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(x.shape))

    def window(self, x, g):
        B, N, _ = x.shape
        gp = WIN * math.ceil(g / WIN)                                           # pad the token grid to a multiple of 7
        x = F.pad(x.reshape(B, g, g, D), (0, 0, 0, gp - g, 0, gp - g))
        w = x.reshape(B, gp // WIN, WIN, gp // WIN, WIN, D).permute(0, 1, 3, 2, 4, 5).reshape(-1, WIN * WIN, D)
        q, k, v = self.heads(w)
        a = (q @ k.transpose(-2, -1) * (D // HEADS) ** -0.5).softmax(-1)
        o = self.proj((a @ v).transpose(1, 2).reshape(w.shape))
        o = o.reshape(B, gp // WIN, gp // WIN, WIN, WIN, D).permute(0, 1, 3, 2, 4, 5).reshape(B, gp, gp, D)
        return o[:, :g, :g].reshape(B, N, D)


if __name__ == "__main__":
    torch.manual_seed(0)
    ref, fus = load_block(0, dtype=torch.float16), load_fused(0)
    at = Attn().cuda().half().eval()
    out = []
    with torch.no_grad():
        for res in RES:
            g = res // 16; N = g * g; B = max(2, 12544 // N)
            x = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
            xc = torch.randn(B, N + 1, D, device="cuda", dtype=torch.float16)     # NCA blocks take CLS + patches
            fns = {"nca_unfused": lambda: ref(xc), "nca_fused": lambda: fus(xc),
                   "attn_global_materialised": lambda: at.global_materialised(x),
                   "attn_global_fused": lambda: at.global_fused(x), "attn_window7": lambda: at.window(x, g)}
            for name, fn in fns.items():
                try:
                    r = time_cuda(fn, warmup=20, iters=100)
                    rec = dict(mixer=name, res=res, grid=g, tokens=N, batch=B, us_per_image=r["median_ms"] * 1e3 / B, iqr_us_per_image=r["iqr_ms"] * 1e3 / B)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); rec = dict(mixer=name, res=res, grid=g, tokens=N, batch=B, us_per_image=None, note="OOM")
                out.append(rec); print(f"res {res:5d} N {N:5d} B {B:3d} {name:26s} {rec['us_per_image'] and round(rec['us_per_image'],1)} us/img", flush=True)
                save("m5_mixer_timing.json", out)
