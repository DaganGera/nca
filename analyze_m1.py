"""M1 summary: merge timing + DRAM traffic + analytic FLOPs -> results/m1_summary.json and a table.

    python analyze_m1.py
Numbers are all read from results/m1_timing.json and results/m1_dram_*.json (nothing typed in by hand).
FLOPs are analytic (PREREGISTRATION.md section 0): 470,550,528 per image per block (1 MAC = 2 FLOP).
"""
import glob, json
from bench_utils import save

FLOPS_PER_IMAGE = 4 * 196 * (576 * 9 * 2 + (576 * 384 + 384 * 192) * 2)

timing = json.load(open("results/m1_timing.json"))
t = {(r["dtype"], r["batch"]): r for r in timing["summary"]}
dram = [r for f in sorted(glob.glob("results/m1_dram_*.json")) for r in json.load(open(f))]

rows = []
for r in dram:
    key = (r["dtype"], r["batch"])
    if r["protocol"] == "range" and r["measured"] and key in t:
        ms = t[key]["median_of_medians_ms"]
        rows.append(dict(dtype=r["dtype"], batch=r["batch"], time_us_per_image=ms * 1e3 / r["batch"],
                         dram_mb_per_image=r["dram_bytes_per_image"] / 1e6, min_mb_per_image=r["min_bytes_per_image"] / 1e6,
                         R=r["R"], achieved_gflops=FLOPS_PER_IMAGE * r["batch"] / (ms * 1e-3) / 1e9,
                         flop_per_dram_byte=FLOPS_PER_IMAGE / r["dram_bytes_per_image"], kernels=t[key]["kernels_per_block"]))
cross = [r for r in dram if r["protocol"] in ("app", "flush") and r["measured"]]
print(f"{'dtype':5} {'B':>5} {'us/img':>8} {'DRAM MB/img':>11} {'min MB':>7} {'R':>6} {'GFLOP/s':>8} {'FLOP/B':>7}")
for r in rows:
    print(f"{r['dtype']:5} {r['batch']:5d} {r['time_us_per_image']:8.1f} {r['dram_mb_per_image']:11.3f} {r['min_mb_per_image']:7.3f} "
          f"{r['R']:6.2f} {r['achieved_gflops']:8.0f} {r['flop_per_dram_byte']:7.1f}")
print("cross-checks (not used for the main table):")
for r in cross:
    print(f"  {r['dtype']} B={r['batch']} {r['protocol']:5s} DRAM/img {r['dram_bytes_per_image']/1e6:.3f} MB  R={r['R']:.2f}")
save("m1_summary.json", dict(flops_per_image_per_block=FLOPS_PER_IMAGE, main=rows, cross_checks=cross,
                             kernels_note="36-37 CUDA kernels per block forward (ncu app replay) plus 12 memsets and 1 memcpy in the torch profiler count (49 events at fp32, 42 at fp16)"))
