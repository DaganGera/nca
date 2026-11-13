"""Measured peaks of THIS GPU (used by the cost model and the roofline): copy bandwidth and matmul throughput.

    python peaks.py  -> results/peaks.json
Reported as the best of several size/shape configs (median of 30 timed runs each); never a datasheet number.
"""
import torch
from bench_utils import time_cuda, save, gpu_state

torch.backends.cuda.matmul.allow_tf32 = False       # fp32 peak = true fp32 FMA path
res = dict(copy=[], read=[], matmul=[])
for mb in (64, 256, 1024):
    n = mb * 2**20 // 4
    x, y = torch.empty(n, device="cuda"), torch.empty(n, device="cuda")
    x.normal_()
    r = time_cuda(lambda: y.copy_(x), warmup=10, iters=30)
    res["copy"].append(dict(mb=mb, gbps=2 * n * 4 / (r["median_ms"] * 1e6), ms=r["median_ms"]))       # read + write
    r = time_cuda(lambda: x.sum(), warmup=10, iters=30)
    res["read"].append(dict(mb=mb, gbps=n * 4 / (r["median_ms"] * 1e6), ms=r["median_ms"]))
    del x, y
for dt, name, sizes in ((torch.float16, "fp16", (4096, 6144, 8192)), (torch.float32, "fp32", (2048, 4096, 6144))):
    for s in sizes:
        a, b = torch.randn(s, s, device="cuda", dtype=dt), torch.randn(s, s, device="cuda", dtype=dt)
        r = time_cuda(lambda: a @ b, warmup=10, iters=30)
        res["matmul"].append(dict(dtype=name, n=s, tflops=2 * s**3 / (r["median_ms"] * 1e9), ms=r["median_ms"], gpu=r["gpu_after"]))
        del a, b
res["peak_copy_gbps"] = max(r["gbps"] for r in res["copy"])
res["peak_read_gbps"] = max(r["gbps"] for r in res["read"])
res["peak_fp16_tflops"] = max(r["tflops"] for r in res["matmul"] if r["dtype"] == "fp16")
res["peak_fp32_tflops"] = max(r["tflops"] for r in res["matmul"] if r["dtype"] == "fp32")
res["gpu_state_end"] = gpu_state()
save("peaks.json", res)
for k in ("peak_copy_gbps", "peak_read_gbps", "peak_fp16_tflops", "peak_fp32_tflops"):
    print(f"{k:20s} {res[k]:8.1f}")
for r in res["matmul"]:
    print(r["dtype"], r["n"], f"{r['tflops']:.2f} TFLOP/s", "sm_clock", r["gpu"]["sm_clock_mhz"])
for r in res["copy"] + res["read"]:
    print(r)
