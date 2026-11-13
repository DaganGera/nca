"""M2: per-step fused NCA kernel (Triton).

One launch per NCA step does   perception(3x3 depthwise x3 filters, zero pad) -> 576->384 ReLU -> 384->192 -> residual
for a tile of BM cells. The 576-wide perception tensor and the 384-wide hidden layer live only in
registers; DRAM sees just the cell state (read) and the new cell state (write).

Layout: tokens-last (B, N, D) fp16 -- the model's own token layout, so no rearrange copies exist.
Inside the kernel the perception index is f*D + c (filter-major). The trained model's index is 3c + f
(grouped-conv order); FusedNCABlock permutes the weights once so the two are equivalent.
Triton needs power-of-two dot shapes: D=192 is processed as 3 chunks of 64, the hidden 384 as 3 chunks of 128.
All dots are fp16 x fp16 -> fp32 accumulate (tl.dot).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import load_state

D, HID, K = 192, 384, 4
D_C, HID_C = tl.constexpr(D), tl.constexpr(HID)   # globals visible inside @triton.jit


@triton.autotune(
    configs=[triton.Config({"BM": bm}, num_warps=w, num_stages=s)
             for bm in (16, 32, 64) for w in (4, 8) for s in (1, 2)],
    key=["R"],
)
@triton.jit
def nca_step_kernel(x_ptr, out_ptr, init_ptr, wp_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
                    R, H, W, N, out_tokens, out_off,
                    LAST: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)          # flat cell index over (batch, y, x)
    rmask = rows < R
    n = rows % N
    y = n // W
    xx = n % W
    c64 = tl.arange(0, 64)
    k128 = tl.arange(0, 128)

    h0 = tl.zeros([BM, 128], tl.float32)
    h1 = tl.zeros([BM, 128], tl.float32)
    h2 = tl.zeros([BM, 128], tl.float32)
    for f in tl.static_range(3):                              # perception filter: sobel_x, sobel_y, identity
        for cc in tl.static_range(3):                         # 64-channel chunk
            ch = cc * 64 + c64
            p = tl.zeros([BM, 64], tl.float32)
            for t in tl.static_range(9):                      # 3x3 taps (zero padding via mask)
                dy = t // 3 - 1
                dx = t % 3 - 1
                ok = rmask & (y + dy >= 0) & (y + dy < H) & (xx + dx >= 0) & (xx + dx < W)
                xs = tl.load(x_ptr + (rows + dy * W + dx)[:, None] * D_C + ch[None, :], mask=ok[:, None], other=0.0)
                w = tl.load(wp_ptr + (f * D_C + ch) * 9 + t).to(tl.float32)
                p += xs.to(tl.float32) * w[None, :]
            pa = p.to(tl.float16)                             # perception chunk [BM, 64] stays in registers
            kk = (f * D_C + cc * 64 + c64)[:, None] * HID_C       # rows of W1^T for these 64 inputs
            h0 = tl.dot(pa, tl.load(w1_ptr + kk + (0 * 128 + k128)[None, :]), h0)
            h1 = tl.dot(pa, tl.load(w1_ptr + kk + (1 * 128 + k128)[None, :]), h1)
            h2 = tl.dot(pa, tl.load(w1_ptr + kk + (2 * 128 + k128)[None, :]), h2)

    h0 = tl.maximum(h0 + tl.load(b1_ptr + 0 * 128 + k128)[None, :].to(tl.float32), 0.0).to(tl.float16)
    h1 = tl.maximum(h1 + tl.load(b1_ptr + 1 * 128 + k128)[None, :].to(tl.float32), 0.0).to(tl.float16)
    h2 = tl.maximum(h2 + tl.load(b1_ptr + 2 * 128 + k128)[None, :].to(tl.float32), 0.0).to(tl.float16)

    b = rows // N
    orow = b * out_tokens + n + out_off
    for oc in tl.static_range(3):                             # output channel chunk
        co = oc * 64 + c64
        acc = tl.zeros([BM, 64], tl.float32)
        acc = tl.dot(h0, tl.load(w2_ptr + (0 * 128 + k128)[:, None] * D_C + co[None, :]), acc)
        acc = tl.dot(h1, tl.load(w2_ptr + (1 * 128 + k128)[:, None] * D_C + co[None, :]), acc)
        acc = tl.dot(h2, tl.load(w2_ptr + (2 * 128 + k128)[:, None] * D_C + co[None, :]), acc)
        acc += tl.load(b2_ptr + co)[None, :].to(tl.float32)
        cur = tl.load(x_ptr + rows[:, None] * D_C + co[None, :], mask=rmask[:, None], other=0.0)
        new = (cur.to(tl.float32) + acc).to(tl.float16)       # residual add; state stored fp16
        if LAST:                                               # emit grid_final - grid_init directly
            ini = tl.load(init_ptr + rows[:, None] * D_C + co[None, :], mask=rmask[:, None], other=0.0)
            new = new - ini
        tl.store(out_ptr + orow[:, None] * D_C + co[None, :], new, mask=rmask[:, None])


class FusedNCABlock:
    """Drop-in for NCABlockRef: (B, 1+N, D) fp16 -> (B, 1+N, D) fp16 (CLS delta = 0)."""

    def __init__(self, sd, idx, device="cuda"):
        pre = f"nca_blocks.{idx}.nca_attn."
        g = lambda k: sd[pre + k].to(device=device, dtype=torch.float16).contiguous()
        self.ln_w, self.ln_b = g("norm.weight"), g("norm.bias")
        # F.conv2d(groups=D) gives output channel j to input channel j // 3 (j = 3c + m), so the trained
        # perception row j acts on channel c=j//3. The kernel wants filter-major order f*D + c: permute
        # the perception rows and W1's input columns identically (a pure relabelling, mathematically equal).
        perm = (torch.arange(D)[None, :] * 3 + torch.arange(3)[:, None]).reshape(-1).to(device)  # perm[f*D+c] = 3c+f
        self.wp = g("perception.weight")[perm].contiguous()                  # (576,1,3,3), row f*D+c, tap = 3*ky+kx
        self.w1t = g("linear1.weight").t()[perm].contiguous()                # (576, 384)  K-major, rows permuted
        self.b1 = g("linear1.bias")
        self.w2t = g("linear2.weight").t().contiguous()       # (384, 192)
        self.b2 = g("linear2.bias")

    def __call__(self, x):
        B, n1, _ = x.shape
        N = n1 - 1
        H = W = int(round(N ** 0.5))
        assert H * W == N
        R = B * N
        state = F.layer_norm(x[:, 1:], (D,), self.ln_w, self.ln_b)   # (B, N, D) fp16, same as fp16 reference
        init = state
        bufs = [torch.empty_like(state), torch.empty_like(state)]
        out = torch.zeros(B, n1, D, device=x.device, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(R, meta["BM"]),)
        cur = state
        for s in range(K):
            last = s == K - 1
            dst = out if last else bufs[s % 2]
            nca_step_kernel[grid](cur, dst, init, self.wp, self.w1t, self.b1, self.w2t, self.b2,
                                  R, H, W, N, n1 if last else N, 1 if last else 0, LAST=last)
            cur = dst
        return out


def load_fused(idx=0, dtype=torch.float16, which="model"):
    assert dtype == torch.float16, "fused kernel is fp16 in / fp32 accumulate"
    return FusedNCABlock(load_state(which), idx)
