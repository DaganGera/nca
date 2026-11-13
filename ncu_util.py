"""Per-kernel utilisation counters (what limits each kernel?): kernel replay, caches not flushed.

    python ncu_util.py --impl fused --dtype fp16 --batch 64   -> results/util_<impl>_<dtype>_B<batch>.json
Feeds the M3 go/no-go analysis and the roofline figure (M5/M6).
"""
import argparse, csv, io, json, subprocess, sys
from bench_utils import save
from run_ncu import NCU, num

METRICS = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",                                              # L2 traffic
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread", "launch__grid_size", "launch__block_size",
])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="fused"); ap.add_argument("--dtype", default="fp16"); ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    cmd = [NCU, "--replay-mode", "kernel", "--cache-control", "none", "--nvtx", "--nvtx-include", "nca_block/",
           "--metrics", METRICS, "--csv", "--page", "raw", "--print-units", "base",
           sys.executable, "profile_target.py", "--impl", a.impl, "--dtype", a.dtype, "--batch", str(a.batch)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if p.returncode:
        raise SystemExit(p.stdout[-1500:] + p.stderr[-1500:])
    lines = p.stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = [r for r in csv.DictReader(io.StringIO(chr(10).join(lines[start:]))) if r["ID"].isdigit()]
    keys = METRICS.split(",")
    out = [dict(kernel=r["Kernel Name"][:60], **{k: num(r[k]) for k in keys}) for r in rows]
    for o in out:
        print(f"{o['kernel']:44s} {o['gpu__time_duration.sum']/1e3:9.1f} us  DRAM {o['dram__throughput.avg.pct_of_peak_sustained_elapsed']:5.1f}%  "
              f"SM {o['sm__throughput.avg.pct_of_peak_sustained_elapsed']:5.1f}%  tensor {o['sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active']:5.1f}%  "
              f"warps {o['sm__warps_active.avg.pct_of_peak_sustained_active']:5.1f}%  regs {o['launch__registers_per_thread']:.0f}  "
              f"dram {(o['dram__bytes_read.sum']+o['dram__bytes_write.sum'])/1e6:7.2f} MB  L2 {o['lts__t_bytes.sum']/1e6:8.1f} MB")
    save(f"util_{a.impl}_{a.dtype}_B{a.batch}.json", out)
