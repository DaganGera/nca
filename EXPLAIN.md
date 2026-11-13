# EXPLAIN: a whiteboard walkthrough

Everything below uses numbers measured in this repo (RTX 4060: 258 GB/s, 31.3 TFLOP/s fp16). Draw as you read.

## 1. What an NCA block does, and what it costs

A 14x14 grid of cells, each holding a 192-number state. One *update step*:

1. **Perceive**: every cell looks at its 3x3 neighbourhood with 3 learnable filters per channel (depthwise), giving 3 x 192 = 576 numbers per cell.
2. **Think**: a tiny MLP per cell, 576 -> 384 -> 192 (ReLU in the middle).
3. **Update**: add the MLP output to the cell's state.

Repeat K=4 times. Each step is about 2.4 MFLOP per cell (the MLP is 98% of it), 196 cells, so ~470 MFLOP per image per block. Compare global attention at the same width: 8·N·D² + 4·N²·D FLOPs, i.e. ~87 MFLOP at N=196. **The NCA does ~5x more arithmetic than attention at 224 px.** It is only cheaper in FLOPs once N is large (N² grows, the NCA is linear in N): around 2,700 tokens, ~840 px. That is the whole reason the memory wall matters here.

## 2. Why "unfused" is slow: the intermediate tensors

PyTorch runs each operation as its own GPU kernel, and every kernel reads its input from and writes its output to DRAM (the big, slow memory). Per step the unfused code creates: the 576-channel perception tensor, the 384-wide hidden tensor, the 192-wide MLP output, then adds it. Per image per step that is several megabytes that exist only to be handed to the next kernel.

- Minimum DRAM traffic per image per block (read the input once, write the result once, weights amortised): **0.16 MB** (fp16, B=64).
- Measured (Nsight Compute): **10.9 MB**. 68x more.

**Fusion** = one kernel does perceive + think + update for a tile of cells, keeping the 576- and 384-wide tensors in registers. Now each step moves only the cell states: read 75 KB, write 75 KB per image. Measured: **0.81 MB** at B=64 (13.5x less), 1.76x faster.

## 3. Arithmetic intensity and the roofline

- **Arithmetic intensity** = FLOPs done per byte moved from DRAM.
- Your GPU can do 31.3 TFLOP/s but only move 258 GB/s. Dividing: **ridge point = 121 FLOP/byte**. A kernel below that is limited by memory bandwidth, above it by compute.
- **Roofline**: plot achieved FLOP/s against intensity; the ceiling is `min(peak compute, bandwidth x intensity)` (Fig 1).
- Unfused NCA fp16: 470 MFLOP / 10.9 MB = **43 FLOP/B**, well left of the ridge: memory-bound, so it can reach at most 43 x 258 GB/s = 11 TFLOP/s (it gets 6.6).
- Fused: 470 MFLOP / 0.81 MB = **580 FLOP/B**, far right of the ridge: no longer memory-bound. It reaches 11.6 TFLOP/s = 37% of peak. Now the limit is the kernel itself (occupancy, latency), not DRAM.

Fusion moves a kernel to the right on the roofline. That is the whole idea.

## 4. L2 cache: why small batches lie

The 4060 has a 24 MB L2 cache. With one image the entire working set fits in L2, so most "DRAM" traffic never reaches DRAM and the ratio to the minimum is small (1.5x measured, protocol-sensitive). At batch 64 the intermediates are tens of MB and spill: 50-68x. The lesson: traffic ratios depend on the batch, so always state it. It also bites the measurement: dirty cache lines still sitting in L2 when you stop profiling are not yet counted as DRAM writes, which is why fused-kernel numbers at B=1 and 16 are only lower bounds.

## 5. Temporal blocking: what we did *not* build, and why

Per-step fusion still writes the whole grid to DRAM after each of the 4 steps. **Temporal blocking** (as in stencil codes) keeps a tile on-chip and does all 4 steps before writing. A 3x3 stencil needs its neighbours' *previous* values, so a tile that wants to survive 4 steps needs a halo 4 cells thick, recomputed redundantly (7x7 tiles of a 14x14 grid: **1.5x extra work**). Alternatively one program owns the whole image: 196 cells x 192 x 2 B = 75 KB, but you need the old and new state at once (150 KB), more than the 99 KB of shared memory per block.

Numbers decided it before any code: this GPU's fused kernel spends only **7.7%** of its time on DRAM at B=64 (13.6% at 2048). Even deleting all DRAM traffic would give at most 1.07-1.14x. And the halo version adds 1.5x work to a kernel that is not memory-bound. Verdict: no-go. This is why the repo claims per-step fusion only.

## 6. The memory wall

Over two years, peak compute grows ~3.0x, DRAM bandwidth ~1.6x. So every generation the ridge point moves right by 1.875x: kernels that were compute-bound become memory-bound.

Our model: `time = max(FLOPs / compute, bytes / bandwidth)`, calibrated to measured kernels. Findings:

- At 224 px the NCA is compute-bound, flash attention is already memory-bound. As compute outgrows bandwidth, the NCA gains ground (ratio NCA/flash 8.4x at 224 px today -> 2.0x by generation +3).
- Then both are memory-bound and the ratio **freezes at the ratio of bytes per token**: per-step-fused NCA moves ~12·N·D·2 B, flash attention ~10·N·D·2 B. No parity.
- Only **temporal fusion** (2·N·D bytes) would break through, in the model, and it is the one thing the GPU analysis says is not worth building *today*. That tension is the honest conclusion.
- Attention that writes the N x N score matrix to DRAM *does* lose to the NCA: from 640 px today, from 224 px by generation +3. This is why fusing attention (flash) matters as much as fusing the NCA.

## 7. Glossary of traps

- **FLOP vs MAC**: one multiply-accumulate = 2 FLOPs. Our counts use FLOPs.
- **fp32 vs fp16**: fp16 halves bytes and uses tensor cores. Never credit fusion with a precision change: we compare fp16 to fp16 (the fp32 original is 4.3x slower end-to-end, but ~3x of that is precision, not fusion).
- **Amdahl**: the NCA mixers are 68% of the model; making them 1.7x faster speeds up the model by 1.39x, not 1.7x.
- **Calibration vs validation**: we fit one efficiency factor at 224 px and *test* at 320-1024 px. A model that is only checked where it was fitted proves nothing.

---

# Quiz: 10 questions a PhD would ask

Answer out loud first. Bring your answers back and I will grade them against the data.

1. Your unfused block moves 68x the minimum DRAM traffic yet fusing it gives only 1.76x speedup at batch 64. Why isn't the speedup closer to 68x, or even to 13.5x? What would you measure to split the gain between "fewer bytes" and "fewer launches / layout copies"?
2. At batch 1, the fused kernel appears to move *less* than the analytic minimum. Explain how a measured DRAM number can fall below a "minimum", and say precisely which of your reported numbers are therefore unusable.
3. Draw the roofline for this GPU with the ridge point. Where do the unfused and fused kernels sit, and what limits the fused kernel now if it is neither bandwidth- nor compute-bound?
4. Why did the first Triton kernel run ~10x slower than the PyTorch reference? What does "registers: 255, spills: thousands of bytes" mean physically, and why did autotuning still pick a bad config?
5. You claim temporal fusion cannot pay off on this GPU without writing it. What measurements and arithmetic support that, and what is the strongest objection to your argument (what would make you wrong)?
6. Explain halo (overlapped) tiling for K=4 steps of a 3x3 stencil on a 14x14 grid. Why is the redundancy 1.5x for 7x7 tiles and 5.9x for 2x2 tiles? What changes if the stencil radius or K doubles?
7. The cost model is calibrated at 224 px and validated at four other resolutions. Which points failed your own 25% criterion, why might those fail, and how does each failure bias the conclusion about parity?
8. Under 3x compute / 1.6x bandwidth per generation, why does the NCA-to-flash-attention time ratio fall from 8.4x to 2.0x and then stop? What single quantity determines the plateau, and what would it take to get below 1?
9. Your end-to-end accuracy is identical in fp32, fp16 and fused fp16 (81.27%), but in one of 24 kernel-level correctness cases (block 0, 7x7 grid) its max-abs error (3.53e-3) exceeded the fp16 reference's (3.14e-3). Is the "within 0.1 points" check enough to claim the kernel is correct? What else would you test, and how does that single strict-bar miss differ from a real bug (hint: the *first* version of the kernel had one, with mean error 1.3)?
10. Energy: NVML would not report it on this GPU, so the energy plot uses assumed constants and is flat across generations. What exactly can and cannot be concluded from it, and what is the single assumption you would test first if you had a power meter?
