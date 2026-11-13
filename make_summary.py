"""results/summary.json (exactly the requested keys + GPU/software) and results/summary_details.json (definitions, other comparisons).

    python make_summary.py      -- every number is read from results/*.json; nothing typed in.
Headline comparisons are SAME-PRECISION (fp16 vs fp16) at batch 64 so precision is never credited to fusion.
"""
import json, platform, subprocess
import torch, triton
from bench_utils import save

J = lambda f: json.load(open(f"results/{f}"))
m2, m4, cm, peaks = J("m2_summary.json"), J("m4_throughput.json")["summary"], J("m5_cost_model.json"), J("peaks.json")
row = next(r for r in m2["rows"] if r["batch"] == 64)
t = {(r["variant"], r["batch"]): r["img_per_s"] for r in m4}
p = torch.cuda.get_device_properties(0)
ncu = subprocess.run([r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat", "--version"], capture_output=True, text=True).stdout.strip().splitlines()
smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

pm = cm["parity_model"]["nca_fused__vs__attn_global_fused"]
par, ideal = pm["t_cal"], pm["t_ideal"]
closest = min(cm["projection"]["nca_fused"][str(g)][str(r)]["t_cal"] / cm["projection"]["attn_global_fused"][str(g)][str(r)]["t_cal"]
              for g in range(6) for r in cm["res_fine"])
ideal_txt = ", ".join(f"gen+{g}: {v} px" if v else f"gen+{g}: none" for g, v in ideal.items())
parity = (("none: with the efficiencies measured for these kernels (calibrated at 224 px), per-step-fused NCA never matches flash-style global attention for 224-1024 px "
           f"over 5 hardware generations (closest model ratio {closest:.2f}x; measured on this GPU: none up to 1024 px). ")
          if all(v is None for v in par.values()) else f"model parity at {par}. ") +          f"Ideal-roofline model (every kernel at 100% of its roof): {ideal_txt}."

summary = dict(
    dram_mb_per_image_before=row["ref_fp16_dram_mb_per_image"],     # unfused fp16 reference block, B=64, ncu range replay
    dram_mb_per_image_after=row["fused_dram_mb_per_image"],         # per-step fused fp16 block, B=64
    throughput_before=t[("ref_fp16", 64)],                          # full hybrid model, original (unfused) NCA, fp16, B=64, img/s
    throughput_after=t[("fused_fp16", 64)],                         # full hybrid model, fused NCA, fp16, B=64, img/s
    parity_point=parity,
    gpu=f"{p.name} ({p.total_memory // 2**20} MiB, sm_{p.major}{p.minor}, {p.multi_processor_count} SMs, L2 {p.L2_cache_size // 2**20} MiB), driver {smi}",
    software=dict(python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda, triton_windows=triton.__version__,
                  nsight_compute=ncu[-1] if ncu else None, os=platform.platform()))
save("summary.json", summary)
save("summary_details.json", dict(
    definitions=dict(dram="one NCA block (LN + 4 update steps), MB per image, fp16, batch 64, Nsight Compute range replay, caches not flushed",
                     throughput="whole hybrid ViT on CIFAR-sized 224x224 input, images/s, batch 64, fp16, CUDA events, median of 3 process launches",
                     parity_point="smallest resolution from which NCA time <= attention time at every larger resolution, model calibrated at 224 px; compared mixer: per-step fused NCA vs flash-style global attention"),
    other_comparisons=dict(original_fp32_img_per_s_b64=t[("ref_fp32", 64)], speedup_vs_same_precision=t[("fused_fp16", 64)] / t[("ref_fp16", 64)],
                           speedup_vs_original_fp32_MIXED_PRECISION=t[("fused_fp16", 64)] / t[("ref_fp32", 64)],
                           deit_tiny_fp16_img_per_s_b64=t[("deit_fp16", 64)], deit_tiny_fp32_img_per_s_b64=t[("deit_fp32", 64)],
                           block_time_speedup_b64=row["speedup_vs_fp16"], dram_reduction_b64=row["dram_reduction"],
                           dram_mb_b2048=dict(before=m2["rows"][-1]["ref_fp16_dram_mb_per_image"], after=m2["rows"][-1]["fused_dram_mb_per_image"])),
    parity_model_all_pairs=cm["parity_model"], parity_measured_gen0=cm["parity_measured_gen0"], measured_peaks=peaks,
    energy="not measured: NVML energy/power counters unsupported on this GPU; model energy uses assumed constants"))
print(json.dumps(summary, indent=1))
