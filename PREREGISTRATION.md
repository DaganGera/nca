# Pre-registration

Written and committed **before any measurement** of the NCA block (git timestamp is the proof).
Nothing in this file is a result. Numbers below are either (a) arithmetic from the block definition
or (b) predictions that the measurements are allowed to falsify.

## 0. Facts known before measuring (not results)

| Item | Value | Source |
|---|---|---|
| GPU reported by `nvidia-smi` | NVIDIA GeForce RTX 4060, 8188 MiB, 115 W cap (Ada, sm_89) | `nvidia-smi` |
| Project brief said | "RTX 3050 Laptop (sm_86)" in one place, "RTX 4060" in another | brief is inconsistent; **we use what the hardware reports** |
| L2 size, max shared mem / block, SM count | queried at M1 via `torch.cuda.get_device_properties` and recorded, not assumed | M1 |
| NCA block | LN -> grid (D=192, 14x14) -> K=4 x [depthwise 3x3 x M=3 filters (zero pad) -> 576->384 ReLU ->192 -> residual] -> `grid_final - grid_init` | `nca_attention.py` |
| Block scope | The **mixer only** (`NCAAttention`). The per-block token MLP (`NCAViTBlock.mlp`) is identical in every baseline and is *not* part of the fused kernel | `nca_vit.py` |
| Hybrid MHSA blocks | use `nn.MultiheadAttention(..., need_weights=True)` (default), which materialises the N x N weights. timm DeiT-Tiny uses fused SDPA. This is a confound for the "585 vs 1672 img/s" comparison, so we decompose time per component before attributing any of the gap to the NCA | `hybrid_nca_vit.py` |

Perception weight layout: output channel index = `f*D + c` (f = filter, c = channel), so the 576-wide perception tensor is three contiguous 192-wide groups (sobel_x, sobel_y, identity). Filters are *learnable* in the trained config, so all 27 taps/channel are loaded from the checkpoint, not hard-coded.

### Arithmetic (derived by hand from the block definition; N=196 cells, D=192, H=384, M=3, K=4)

- Parameters per block: perception 192*3*9 = 5,184; W1 576*384 + 384 = 221,568; W2 384*192 + 192 = 73,920; LN 384. **Total 301,056.**
- FLOPs (1 MAC = 2 FLOP) per cell per step: perception 576*9*2 = 10,368; MLP (576*384 + 384*192)*2 = 589,824. Per image per block (K=4, 196 cells): **~470.6 MFLOP**. (LN, ReLU, residual adds are negligible and ignored.)
- Minimum DRAM bytes per image per block, if every intermediate stayed on-chip, with element size s bytes and batch B:
  `min(B, s) = 2*196*192*s + 301,056*s / B`
  (read patch tokens once, write the delta once, read weights once per launch and amortise over the batch; CLS token pass-through and the zero CLS row are excluded, an assumption that changes the figure by 0.5%).
  - fp32: B=1 -> 1.505 MB, B=16 -> 0.376 MB, B=64 -> 0.320 MB.
  - fp16: B=1 -> 0.753 MB, B=16 -> 0.188 MB, B=64 -> 0.160 MB.
- The unfused fp32 reference materialises, per step and image, at least: perception 196*576*4 = 451 KB, hidden 196*384*4 = 301 KB, delta 150 KB, plus any `rearrange`/contiguous copies and the grid itself. Written once and read once each -> several MB per step -> order of 10+ MB per image per block if nothing hits in L2.

## 1. Measurement protocol (fixed in advance)

- **Timing:** CUDA events, 20 warm-up runs, 100 timed runs, report **median and IQR**. Each configuration is run in 3 separate process launches; we report the median of medians and the spread across launches. Clocks are not locked (laptop/WDDM); we log GPU clock, temperature and power before and after every benchmark and flag any run where the SM clock moved by more than 10%.
- **Batch sizes:** 1, 16, 64 and the largest that fits in memory.
- **Precisions:** reference in **fp32** (what the original 585 img/s likely used) *and* **fp16** (what the fused kernel uses). Fusion speedups are headline-compared against the **same-precision** reference so that precision is never credited to fusion. fp32 reference vs fp16 fused is reported separately and labelled as mixed.
- **DRAM traffic:** Nsight Compute `dram__bytes_read.sum` and `dram__bytes_write.sum`. Primary protocol: `--replay-mode application --cache-control none` (kernels see the real L2 state of an actual run). Secondary protocol (reported as an upper bound): ncu default `--cache-control all`, which flushes L2 between kernels and therefore *overstates* DRAM traffic for L2-resident working sets. We never report the secondary number as the main result. Bytes are divided by B to get bytes per image.
- **Kernel count:** number of kernels launched in one block forward, counted from ncu / torch profiler.
- **Achieved FLOP/s:** analytic FLOPs above divided by measured median time.
- **Correctness:** fused vs reference at the same precision on random inputs, on real activations from CIFAR-100, and on held-out shapes (other batch sizes; 7x7 and 28x28 grids). Report max-abs and max-rel error. Pass bar: fp16 fused vs fp32 reference max-abs error no worse than fp16 reference vs fp32 reference (the fused kernel must be at least as accurate as simply running the reference in half precision).
- **Energy:** `nvmlDeviceGetTotalEnergyConsumption` deltas around timed loops of >= 5 s. If the driver does not support it on this GPU, we report "not measured" and do not substitute an estimate.
- **Everything measured is written raw to `results/` by a single command per script.**

## 2. Hypotheses and what falsifies them

Let `R(B) = measured DRAM bytes per image per block / min(B, s)` (same precision, primary ncu protocol).

**H1: the unfused NCA block is DRAM-traffic bound by its intermediates.**
Prediction: at B=64, fp32 reference has `R >= 3`.
- Supported: `R >= 3` at B=64.
- Weakly supported: `1.5 < R < 3`.
- **Falsified (GO/NO-GO 1, stop):** `R <= 1.5` at B=64. The project then becomes a launch-overhead / L2-absorption story and the resume bullet changes.
- Gate rule fixed now: the gate is evaluated at **B=64** (the throughput-relevant regime). Results at B=1/16/largest are reported, and a B=1-only pass of the 1.5x test is reported as a *batch-dependent L2 effect* and does **not** by itself stop the project.

**H2: traffic amplification is batch-dependent because of L2.**
Prediction: `R(1) < R(64)`. At B=1 the whole working set of one image (a few MB) can fit in L2, at B=64 it cannot. L2 size is queried, not assumed.
- Falsified if `R(1) >= R(64)`.

**H3: per-step fusion removes most of the intermediate traffic but not the inter-step traffic.**
The per-step kernel still reads the grid and writes the grid once per step, so its traffic floor is `4 * 2 * 196*192*s` (+weights) = ~4x min.
Prediction (fp16, B=64): fused `R` in **[1.5, 4.5]**, and at least **3x** lower DRAM bytes than the fp16 unfused reference.
- Falsified if fused `R > 6` (an unexpected spill, e.g. weights re-streamed from DRAM) or if the reduction vs the same-precision unfused reference is < 2x.

**H4: lower DRAM traffic converts to lower runtime for the NCA block.**
Prediction (B=64, fp16): per-step fused block time is **>= 1.5x faster** than the fp16 unfused reference.
- Falsified if speedup < 1.1x. If traffic falls but time does not, that is reported as "the block is not memory-bound at this size" and H1's causal story is rejected even if H1's traffic part holds.
- Secondary comparison (best effort, not blocking): `torch.compile` of the reference as the "compiler fusion" baseline.

**H5: end-to-end gain is bounded by Amdahl, and the NCA is not the only reason for the gap to DeiT-Tiny.**
- Measure time share `f` of the 6 NCA mixers in the full hybrid model (B=64, fp16) with per-module CUDA events.
- Prediction: `f >= 0.40`; end-to-end speedup of the fused model `>= 1.2x` vs the same-precision original; and the fused model is **still slower than DeiT-Tiny** at the same batch/precision.
- Falsified: `f < 0.25` (NCA mixers are not the dominant cost, the gap is elsewhere, e.g. `need_weights=True` MHSA) -> reported as such.
- Accuracy: fused-model CIFAR-100 top-1 within **0.1 points** of the same-machine reference run at the same precision. Larger delta -> fused kernel rejected.

**H6: temporal fusion (M3) is only worth coding if its on-chip footprint fits.**
Decision is made by arithmetic before coding (shared mem per block vs 196*192*2 = 75,264 B state, plus staging for the MLP tiles, plus halo redundancy factor). If neither option beats per-step fusion on predicted DRAM bytes *and* is implementable in Triton without spilling the state to global scratch memory, we stop and report per-step fusion as final.
Integrity rule fixed now: "keeps cellular-automaton state in on-chip SRAM across all update iterations through halo tiling" is only used if a measured kernel does exactly that (ncu shows DRAM bytes near `min`, not ~4x). Otherwise the resume wording is changed to what was measured.

**H7: memory-wall projection (cost model, M5).**
Time per image per layer = `max(FLOPs / peak_compute, bytes / bandwidth)`, with peaks **measured** on this GPU (matmul and copy micro-benchmarks). Mixers: global attention (DeiT-style, modelled both as *materialised* N x N like `nn.MultiheadAttention(need_weights=True)` and as *fused/flash-style*), window attention (7x7 windows), NCA (unfused, per-step fused, fully temporally fused if built). Resolution 224..1024 px (patch 16), Gholami scaling: compute 3.0x and bandwidth 1.6x per generation, 5 generations.
Hand-derived expectation (DeiT-Tiny width D=192): attention mixer FLOPs per token = 8D^2 + 4ND; NCA = ~2.40 MFLOP/token (K=4). They cross at N ~ 2.7k tokens (~840 px).
- Prediction A: at 224 px on today's GPU, fused NCA is **slower** than global attention (NCA does ~5x the FLOPs).
- Prediction B: parity vs *fused* global attention, if it occurs in range, occurs at >= 768 px; vs *materialised* attention it occurs at a lower resolution.
- Prediction C: scaling compute faster than bandwidth makes memory-bound mixers lose ground; NCA after fusion has high arithmetic intensity, so it should *not* lose ground relative to fused attention (both compute-bound), and should gain only against the materialised variant.
- Model error is reported per configuration against M1/M2/M4 measurements. If model error exceeds 25% for a configuration, projections that depend on it are flagged as unreliable.
- **If parity never occurs in the studied range, the result is "no parity in range" and that is what `summary.json` records.**
- Energy per op/byte uses Horowitz (ISSCC 2014, 45 nm), clearly labelled as an assumption, compared against NVML board energy where available.

## 3. Things we will not do

- Round in our favour, drop a configuration after seeing its result, or change a threshold above after seeing data. Any change is made in a new commit to a section titled "Amendments" at the bottom of this file, with the reason.
- Report an estimate where a measurement is specified. A missing measurement is written as "not measured".
- Compare a fused fp16 number to an unfused fp32 number without saying so.

## Amendments

_(none yet)_
