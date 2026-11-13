"""Target program for Nsight Compute: runs ONE forward of an NCA block inside an NVTX range.

    ncu ... python profile_target.py --impl ref --dtype fp32 --batch 64
Warm-up forwards run outside the range (so the profiled one sees steady-state L2 and cached cuDNN plans).
"""
import argparse
import torch
from reference import load_block

ap = argparse.ArgumentParser()
ap.add_argument("--impl", default="ref")          # ref | fused (added in M2)
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--warmup", type=int, default=5)
a = ap.parse_args()

dt = dict(fp32=torch.float32, fp16=torch.float16)[a.dtype]
if a.impl == "ref":
    blk = load_block(0, dtype=dt)
    fwd = lambda x: blk(x)
else:
    from fused import load_fused
    fwd = load_fused(0, dtype=dt)
x = torch.randn(a.batch, 197, 192, device="cuda", dtype=dt)
with torch.no_grad():
    for _ in range(a.warmup):
        fwd(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()          # range for --replay-mode range (--profile-from-start off)
    torch.cuda.nvtx.range_push("nca_block")   # range for NVTX-filtered application replay
    fwd(x)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
