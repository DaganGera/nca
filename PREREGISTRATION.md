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

### Amendment 1 (2026-11-14): corrections found while building M2 and M1. No result had influenced any threshold.

1. **Perception channel layout (error in section 0).** Section 0 says the perception output index is `f*D + c`. That is wrong. `F.conv2d(..., groups=D)` assigns output channel `j` to input channel `j // 3`, so the index is `3c + m` and trained row `j` acts on channel `j//3`. Consequences: (a) the kernel permutes weight rows and W1 inputs once on the host (a relabelling; verified numerically by `test_fused.py`); (b) the original filter initialisation (`cat` of three tiled filters) therefore gives input channel `c` three copies of *one* filter type at init (channels 0-63 sobel_x, 64-127 sobel_y, 128-191 identity), not sobel_x/sobel_y/identity on every channel. This is a property of the original model code, not of this project; it is reported, not changed.
2. **NVML energy/power unsupported.** On this GPU `nvmlDeviceGetTotalEnergyConsumption` and `nvmlDeviceGetPowerUsage` return NotSupported. Per section 1, board energy is "not measured". The M5 energy model will be labelled as an assumption (Horowitz figures) with no measured counterpart.
3. **GPU is an RTX 4060 with 24 MB L2** (queried at M1, 24 SMs, 99 KB opt-in shared memory per block). H2 refers to this L2 size.
4. **"Largest batch that fits"** is the largest power of two whose peak allocation is below 70% of VRAM (Windows WDDM can silently spill to system RAM instead of raising OOM, which would invalidate timing). Result: 2048.
5. **M2 correctness bar.** Strict pre-registered bar (fused max-abs AND mean-abs error vs fp32 reference both <= fp16 reference's): **22 of 24 cases pass, 2 fail** (block 0 on 7x7 random: 3.53e-3 vs 3.14e-3; block 0 real CIFAR-100: 6.42e-3 vs 5.78e-3; ratios 1.12x and 1.11x; mean-abs error is lower than the fp16 reference in all 24). The strict result is reported as is. *Post-hoc* secondary criterion (added after seeing these numbers, so it is labelled exploratory): fused max-abs <= 1.25x the fp16 reference's. All 24 pass it. It does not replace the strict result.

### Amendment 2 (2026-11-14): ncu protocol, written after the B=16 fp32 smoke test, before the B=64 gate was measured

Section 1 called the cache-flushing protocol (`--cache-control all`, kernel replay) an *upper bound* on DRAM traffic. A smoke test (reference, fp32, B=16) showed that is wrong:

| protocol | read / image | write / image | total |
|---|---|---|---|
| primary: application replay, no flush (2 repeats) | 5.01 / 5.01 MB | 14.40 / 14.39 MB | 19.42 / 19.40 MB |
| secondary: kernel replay, L2 flushed per kernel | 13.24 MB | 1.44 MB | 14.69 MB |

The flushed protocol over-counts reads (every input starts cold) and under-counts writes (dirty lines are still in L2 when the kernel ends and are written back outside the measured window). Its total is therefore neither an upper nor a lower bound. **The primary protocol is the only one used for H1-H4 and for the GO/NO-GO gate**; the secondary is reported for B=64 only, as a sensitivity check. Residual caveat on the primary protocol: dirty lines still resident in L2 when the profiled range ends are not counted as writes (at most the 24 MB L2 capacity, i.e. at most 24 MB / B per image).

### Amendment 3 (2026-11-14): replay mode for the main DRAM protocol; GO/NO-GO 1 recorded

- Application-replay profiling (Amendment 2's "primary") ran in ~4 min at B=1 and B=16 but did not finish at B=64 after 26 min (`ncu.exe` CPU-bound, target process idle; stopped by me, **no result**, recorded as not measured). The cause was not investigated.
- The **main protocol is therefore range replay** (`--replay-mode range --cache-control none`, the whole block forward as one profiled range, caches not flushed). It returns one aggregate row, so per-kernel counts come from the torch profiler / app replay, not from it.
- Validation against app replay (fp32 / fp16 total DRAM per image): B=16 range 21.0 / 6.04 MB vs app 19.4 / 5.18 MB (8% / 17% apart, same conclusion). **B=1 disagrees strongly** (fp32 2.20 vs 3.94 MB; fp16 0.64 vs 1.33 MB) because range replay sees weights already in L2 from the preceding warm-up. B=1 results are reported under both protocols and the B=1 traffic ratio is called protocol-sensitive.
- `min(B, s)` is a **cold-start** floor (weights read from DRAM once). In a steady-state loop the weights (0.6 MB fp16) stay in the 24 MB L2, so R can be below 1 (fp16 B=1 range: 0.85). That is a property of the definition, not an error.
- **GO/NO-GO 1 (B=64): GO.** R = 50.5 (fp32) and 68.1 (fp16) against a stop threshold of 1.5 and an H1 support threshold of 3. H1 supported, H2 supported (R(1) < R(64) under both protocols). R is not monotone in B (fp16: 68.1 at B=64, 43.4 at B=2048); not investigated, so it is reported but not explained.

### Amendment 4 (2026-11-14): M2 outcome, kernel revision, H3/H4 verdicts (stated against the thresholds in section 2, unchanged)

- **Kernel revision.** The first working kernel was ~10x *slower* than the fp16 reference (674 us/img at B=64) because it reloaded the 9 perception taps once per filter and its autotune space contained configs that spilled thousands of bytes to local memory. It was restructured once (each tap loaded once, three filter accumulators, autotune restricted to 5 configs) and then measured. That is the only revision; the slow version's timings are not reported as a result because it was a bug, not a design. Correctness was re-run on the revised kernel: **strict pre-registered bar 23/24 pass** (the failure: block 0, 7x7 random, fused max-abs 3.53e-3 vs fp16-reference 3.14e-3, ratio 1.12x); mean-abs error <= the fp16 reference in 24/24; post-hoc (exploratory) 1.25x max-abs criterion 24/24. This supersedes the 22/24 figure in Amendment 1.
- **H4 (time), fp16, B=64:** fused 2.59 ms vs unfused fp16 4.57 ms = **1.76x** (threshold >= 1.5x): **supported**. Other batches: B=1 4.7x, B=16 2.1x, B=2048 2.9x. Kernel launches per block: 42 -> 6.
- **H3 (traffic), fp16:** at B=64 fused R = 5.06 and reduction vs unfused fp16 = 13.5x. The "reduction >= 3x" half is **supported**; the predicted bracket R in [1.5, 4.5] is **missed** (5.06), and the falsification line R > 6 is not crossed at B=64. At B=2048, R = 9.81 (> 6) and reduction is 4.4x: by the strict rule **H3 is falsified at B=2048**. Not all of that is the fused kernel: the surrounding wrapper (LayerNorm kernel, `torch.zeros` fill of the output, `init` re-read in the last step) has its own traffic. Floor of this implementation per image at fp16: reads 451,584 B + writes 451,968 B = 0.90 MB, so measured 1.48 MB is 1.64x the implementation's own floor.
- **Measurement limit for the fused kernel:** at B=1 and B=16 the whole batch state stays in the 24 MB L2 and output lines are still dirty when the profiled range ends, so DRAM *writes* are undercounted (0.000-0.006 MB/img). Those two numbers (R = 0.37, 0.43) are lower bounds and are not used in any claim. B >= 64 is the usable regime.

### Amendment 5 (2026-11-14): H6 decision, made by arithmetic before any M3 code (results/m3_analysis.json)

**Verdict: NO-GO. M3 is not built; per-step fusion is the final kernel.**
- Measured peaks on this GPU (peaks.py): 258 GB/s read bandwidth, 31.3 TFLOP/s fp16 tensor, 8.3 TFLOP/s fp32 (best-of; fp32 sags 2460 -> 1875 MHz under sustained load).
- The per-step fused kernel at B=64 spends 7.7% of its runtime on DRAM traffic at peak bandwidth (0.81 MB/img, 2.59 ms) and 13.6% at B=2048. It reaches 37% of the fp16 compute roof. Removing *all* DRAM traffic, assuming zero overlap, bounds the gain at **1.07x (B=64) / 1.14x (B=2048)**.
- Option A (one program per image): double-buffered state 150,528 B > 101,376 B opt-in shared memory; only an in-place 3-row ring (91,392 B) fits, which needs explicit shared-memory control Triton does not provide (nvcc 12.4 exists, no MSVC `cl` on PATH; hand-written CUDA is out of scope). DRAM would fall to 0.16 MB/img.
- Option B (halo tiles): only 7x7 tiles fit (92.9 kB double-buffered), with 1.50x redundant compute and 0.27 MB/img DRAM; 2x2 tiles are 5.9x redundant. It adds work to a kernel that is already latency/occupancy-bound (tensor pipe ~25%, warps active ~17% under `ncu` kernel replay, whose absolute times and DRAM bytes are distorted by clock locking and memory restore and are not used).
- **Integrity rule (H6): the resume bullet "keeps cellular-automaton state in on-chip SRAM across all update iterations through halo tiling" is NOT supported by anything built here and must not be used.** What was built and measured: a Triton kernel that fuses the 3x3 depthwise perception, the 576->384->192 per-cell MLP and the residual of **one** update step, keeping the perception and hidden tensors in registers (state is read and written once per step, K=4 launches per block).

### Amendment 6 (2026-11-14): M4 verdicts (thresholds from section 2, unchanged)

- Weights for M4 are `checkpoints/nca_vit_hybrid/cifar100/seed42/best.pt`, which is bit-identical in accuracy to the `ema_model` of `latest nca vit hybrid.pt` (81.27% top-1 / 95.44% top-5 on both). The 81.91% in the research repo's `results/nca_vit_hybrid_cifar100.json` for this checkpoint was **not reproduced** here (0.64 points lower with the original PIL-bicubic Resize(224) -> Normalize pipeline; training loop stored 81.29%). Cause not investigated. The M4 criterion is relative, so it is unaffected.
- **Accuracy (H5): fused fp16 81.27% = original fp32 81.27% = original fp16 81.27%; delta 0.00 points** (threshold 0.1). Prediction agreement fused vs fp32 100.00%, fused vs fp16 99.99%.
- **Throughput at B=64 (matches the original 585 img/s protocol):** original fp32 510 img/s (research repo file: 585), original fp16 1,589, fused fp16 2,201 (**1.39x** vs same-precision, 4.3x vs the fp32 original; threshold >= 1.2x: supported), DeiT-Tiny fp32 1,790 (file: 1,672) / fp16 5,271. The fused model remains slower than DeiT-Tiny at equal precision (0.42x at fp16), as predicted.
- **NCA mixer time share (H5, threshold >= 0.40, falsified below 0.25):** 72.8% (fp32 original), 67.8% (fp16 original), 55.0% after fusion. `nn.MultiheadAttention(need_weights=True)` blocks are 9.7-12.8% of the model, so that confound is real but small.

### Amendment 7 (2026-11-14): M5 verdicts (H7), thresholds unchanged

Model: `cost_model.py`; measured mixer timings: `bench_mixers.py` (fp16, ~12.5k tokens per point, 224-1024 px). Calibration: one efficiency factor per mixer fitted at 224 px only; 16 held-out points (4 resolutions x 4... 5 mixers x 4 resolutions = 20) used for validation.
- **Model error.** Uncalibrated roofline underpredicts time by 23-75% (no kernel reaches its roof). Calibrated at 224 px, held-out error: mean |error| 8.1%; three points exceed the pre-registered 25% flag: fused NCA @1024 px (-33%), flash attention @640 px (+26%), window attention @320 px (-27%). Analytic DRAM-traffic model vs ncu: within +/-15% in 2 of 6 block-level comparisons (unfused B=16 +7%, B=2048 -2%; fused B=64 +13%); off by -41% (unfused B=64), -39% (fused B=2048) and, where the data are L2-resident lower bounds, +1065% (fused B=16). Whole-model check (M4, B=64 fp16): -9% (original) / -13% (fused) once the mixer efficiencies are applied, -57% / -53% uncalibrated.
- **Prediction A (fused NCA slower than global attention at 224 px today): supported**, measured 40.8 us vs 4.8 us (flash) / 11.4 us (materialised) per image.
- **Prediction B (parity vs fused global attention, if any, at >= 768 px; vs materialised at lower resolution): supported.** Measured on this GPU: fused NCA beats materialised attention from 640 px (first measured point), never beats flash attention or 7x7 window attention up to 1024 px. Ideal-roofline model: parity with flash attention at 848 px for generations +0..+2 and none from +3.
- **Prediction C (reasoning partly wrong).** Expected "NCA after fusion does not lose ground vs fused attention because both are compute-bound". The model says otherwise about why: flash attention at 224 px is already memory-bound, NCA-fused is compute-bound, so NCA *gains* ground as compute outgrows bandwidth (ratio NCA/flash at 224 px: 8.4x -> 2.0x by generation +3) but then both are memory-bound and the ratio freezes at ~2.0x (calibrated) / 1.2x (ideal) = the ratio of bytes per token (12 vs 10 N*D*2 B). Prediction (does not lose ground) holds, the stated mechanism does not.
- **Headline:** no sustained parity of per-step-fused NCA with flash-style global attention anywhere in 224-1024 px over 5 generations, for the kernel efficiencies measured here. Conditional on those efficiencies. The model parity of fused NCA vs 7x7 window attention at 1024 px for generations >= +3 is an artifact of window padding at the last grid point (64 -> 70 tokens) and is not a result.
- **Energy:** not measured (NVML unsupported). Energy numbers are from assumed constants (DRAM 1-2 nJ per 8 B access, verified in the text of Horowitz ISSCC 2014; 1.5 pJ per fp16 MAC, *not verified*: the per-op table in that paper is an image that could not be machine-read; sensitivity 10 pJ, verified text) and are constant across generations by construction.
