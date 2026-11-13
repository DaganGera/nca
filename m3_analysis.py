"""M3 go/no-go: temporal fusion of all K=4 NCA steps, analysed BEFORE writing any code (PREREGISTRATION.md H6).

    python m3_analysis.py -> results/m3_analysis.json
Footprints / redundancy / DRAM bytes are exact arithmetic (enumerated on the 14x14 grid). The time bound uses MEASURED
numbers (results/m2_timing.json, m2_dram_fused.json, peaks.json) and is deliberately generous to temporal fusion:
it assumes DRAM time is fully additive (no overlap with compute) and that the redundant halo compute is free.
"""
import json
from bench_utils import save

K, D, G, S = 4, 192, 14, 2                  # steps, channels, grid side, bytes (fp16)
SMEM_OPTIN = 101_376                         # measured: shared_memory_per_block_optin
CELL = D * S                                 # bytes per cell state
W_BYTES = 301_056 * S                        # weights, fp16

peaks = json.load(open("results/peaks.json"))
t2 = {r["batch"]: r for r in json.load(open("results/m2_timing.json"))["summary"]}
d2 = {r["batch"]: r for r in json.load(open("results/m2_dram_fused.json")) if r["protocol"] == "range"}
BW = peaks["peak_read_gbps"] * 1e9
FLOPS_IMG = 470_550_528


def tiles(th, tw):
    return [(r, c) for r in range(0, G, th) for c in range(0, G, tw)]


def halo_option(th, tw):
    """Option B: each program owns a th x tw tile, carries a halo of K cells, recomputes it redundantly."""
    upd = reads = 0
    max_in = 0
    for r0, c0 in tiles(th, tw):
        for s in range(1, K + 1):                      # step s updates cells within radius K-s of the tile
            rho = K - s
            h = min(G, r0 + th + rho) - max(0, r0 - rho)
            w = min(G, c0 + tw + rho) - max(0, c0 - rho)
            upd += h * w
        rho = K                                        # step 1 reads state within radius K (K-1 compute + 1 stencil)
        h = min(G, r0 + th + rho) - max(0, r0 - rho)
        w = min(G, c0 + tw + rho) - max(0, c0 - rho)
        reads += h * w
        max_in = max(max_in, h * w)
    return dict(tile=f"{th}x{tw}", programs_per_image=len(tiles(th, tw)),
                redundancy=upd / (K * G * G), smem_bytes_double_buffer=2 * max_in * CELL,
                fits_smem=2 * max_in * CELL <= SMEM_OPTIN,
                dram_bytes_per_image_B64=reads * CELL + G * G * CELL + W_BYTES / 64)


A = dict(state_bytes=G * G * CELL, double_buffer_bytes=2 * G * G * CELL,
         inplace_row_ring_bytes=G * G * CELL + 3 * G * CELL,
         dram_bytes_per_image_B64=2 * G * G * CELL + W_BYTES / 64, mma_rows_per_band=f"{G} -> padded to 16")
A["fits_double_buffer"] = A["double_buffer_bytes"] <= SMEM_OPTIN
A["fits_inplace_ring"] = A["inplace_row_ring_bytes"] <= SMEM_OPTIN
B = [halo_option(*t) for t in ((2, 2), (2, 14), (7, 7), (7, 14), (14, 14))]

bound = {}
for b in (64, 2048):
    t_ms = t2[b]["median_of_medians_ms"]
    cur = d2[b]["dram_bytes_per_image"]
    dram_ms_now = cur * b / BW * 1e3
    saved_A = (cur - A["dram_bytes_per_image_B64"]) * b / BW * 1e3
    bound[b] = dict(fused_ms=t_ms, fused_dram_mb_per_image=cur / 1e6, dram_time_at_peak_bw_ms=dram_ms_now,
                    dram_share_of_runtime=dram_ms_now / t_ms, compute_roof_ms=FLOPS_IMG * b / (peaks["peak_fp16_tflops"] * 1e12) * 1e3,
                    fraction_of_compute_roof=FLOPS_IMG * b / (peaks["peak_fp16_tflops"] * 1e12) * 1e3 / t_ms,
                    max_speedup_optionA_generous=t_ms / (t_ms - saved_A),
                    optionB_7x7_compute_work_factor=next(x for x in B if x["tile"] == "7x7")["redundancy"])
out = dict(optionA=A, optionB=B, bounds=bound, measured_peaks=dict(bw_gbps=peaks["peak_read_gbps"], fp16_tflops=peaks["peak_fp16_tflops"]))
save("m3_analysis.json", out)
print("Option A (whole image per program):", json.dumps(A, indent=1))
print("Option B (halo tiles):")
for x in B:
    print(f"  tile {x['tile']:6s} programs/img {x['programs_per_image']:3d}  redundancy {x['redundancy']:.2f}x  smem(2 buf) {x['smem_bytes_double_buffer']/1e3:7.1f} kB "
          f"{'fits' if x['fits_smem'] else 'NO FIT'}  DRAM/img@B64 {x['dram_bytes_per_image_B64']/1e3:7.1f} kB")
for b, v in bound.items():
    print(f"B={b}: fused {v['fused_ms']:.2f} ms; DRAM {v['fused_dram_mb_per_image']:.2f} MB/img = {v['dram_time_at_peak_bw_ms']:.2f} ms at peak BW "
          f"({100*v['dram_share_of_runtime']:.1f}% of runtime); at {100*v['fraction_of_compute_roof']:.0f}% of fp16 compute roof; "
          f"best-case speedup from removing DRAM traffic: {v['max_speedup_optionA_generous']:.2f}x")
