"""M2 summary: fused per-step kernel vs same-precision (fp16) unfused reference, from results/*.json only.

    python analyze_m2.py   -> results/m2_summary.json
"""
import glob, json
from bench_utils import save

FLOPS = 4 * 196 * (576 * 9 * 2 + (576 * 384 + 384 * 192) * 2)
t_ref = {(r["dtype"], r["batch"]): r for r in json.load(open("results/m1_timing.json"))["summary"]}
t_fus = {r["batch"]: r for r in json.load(open("results/m2_timing.json"))["summary"]}
ref = {(r["dtype"], r["batch"]): r for f in glob.glob("results/m1_dram_*.json") for r in json.load(open(f))
       if r["protocol"] == "range" and r["measured"]}
fus = {r["batch"]: r for r in json.load(open("results/m2_dram_fused.json")) if r["protocol"] == "range" and r["measured"]}

rows = []
print(f"{'B':>5} | {'ref16 ms':>8} {'fused ms':>8} {'speedup':>7} {'vs fp32':>7} | {'ref16 MB/img':>12} {'fused MB/img':>12} {'reduction':>9} {'R fused':>7} | {'fused TFLOP/s':>13}")
for B in sorted(t_fus):
    a, f = t_ref[("fp16", B)]["median_of_medians_ms"], t_fus[B]["median_of_medians_ms"]
    a32 = t_ref[("fp32", B)]["median_of_medians_ms"]
    dr, df = ref[("fp16", B)], fus[B]
    row = dict(batch=B, ref_fp16_ms=a, ref_fp32_ms=a32, fused_ms=f, speedup_vs_fp16=a / f, speedup_vs_fp32=a32 / f,
               ref_fp16_dram_mb_per_image=dr["dram_bytes_per_image"] / 1e6, fused_dram_mb_per_image=df["dram_bytes_per_image"] / 1e6,
               dram_reduction=dr["dram_bytes_per_image"] / df["dram_bytes_per_image"], R_fused=df["R"], R_ref_fp16=dr["R"],
               fused_tflops=FLOPS * B / (f * 1e-3) / 1e12, fused_kernels_per_block=t_fus[1]["kernels_per_block"])
    rows.append(row)
    print(f"{B:5d} | {a:8.3f} {f:8.3f} {a/f:6.2f}x {a32/f:6.2f}x | {row['ref_fp16_dram_mb_per_image']:12.3f} {row['fused_dram_mb_per_image']:12.3f} "
          f"{row['dram_reduction']:8.1f}x {df['R']:7.2f} | {row['fused_tflops']:13.2f}")
save("m2_summary.json", dict(rows=rows, note="fused B=1/16 DRAM bytes are lower bounds: the whole batch state stays in the 24 MB L2 and dirty output lines "
                             "are not written back inside the profiled range. B>=64 (and especially 2048) is the reliable regime."))
