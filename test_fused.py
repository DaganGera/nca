"""M2 correctness: fused fp16 kernel vs reference.

Ground truth = fp32 reference. Pass bar (PREREGISTRATION.md): fused error vs fp32 reference must be no worse
than the fp16 reference's own error vs fp32 reference. Cases: random inputs, held-out batch sizes,
grid sizes 7x7 / 14x14 / 28x28, and real activations from CIFAR-100 test images.
One command:  python test_fused.py     -> results/m2_correctness.json
"""
import json, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F
from bench_utils import save
from reference import load_block, load_state
from fused import load_fused

DATA = r"D:\Research\Emergent-Attention1\data\cifar-100-python\test"


def real_activations(block_idx, n=64):
    """Input tokens of NCA block `block_idx` for n real CIFAR-100 test images (fp32 model, eval)."""
    sys.path.insert(0, r"D:\Research\Emergent-Attention1")
    from src.models.hybrid_nca_vit import HybridNCAViT
    m = HybridNCAViT(learnable_filters=True, num_classes=100)
    m.load_state_dict(load_state()); m = m.cuda().eval()
    d = pickle.load(open(DATA, "rb"), encoding="latin1")
    img = torch.tensor(d["data"][:n]).reshape(n, 3, 32, 32).float().div(255)
    img = F.interpolate(img, size=224, mode="bicubic", align_corners=False)   # val transform: bicubic resize
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1); std = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)
    img = ((img - mean) / std).cuda()
    with torch.no_grad():
        x = m.patch_embed(img).flatten(2).transpose(1, 2)
        x = torch.cat([m.cls_token.expand(n, -1, -1), x], 1) + m.pos_embed
        for b in m.nca_blocks[:block_idx]:
            x = b(x)
    return x


def compare(name, x32, blk32, blk16, fused):
    with torch.no_grad():
        ref = blk32(x32)[:, 1:]
        r16 = blk16(x32.half())[:, 1:].float()
        fu = fused(x32.half())[:, 1:].float()
    def err(a):
        d = (a - ref).abs()
        return dict(max_abs=d.max().item(), max_rel=(d / ref.abs().clamp_min(1e-2)).max().item(), mean_abs=d.mean().item())
    e16, efu = err(r16), err(fu)
    ok = efu["max_abs"] <= e16["max_abs"] and efu["mean_abs"] <= e16["mean_abs"]
    print(f"{name:34s} fp16-ref max_abs {e16['max_abs']:.3e} mean {e16['mean_abs']:.2e} | fused max_abs {efu['max_abs']:.3e} "
          f"mean {efu['mean_abs']:.2e} max_rel {efu['max_rel']:.2e}  {'PASS' if ok else 'FAIL'}")
    return dict(case=name, fp16_ref_vs_fp32=e16, fused_vs_fp32=efu, passed=bool(ok), ref_max_abs=ref.abs().max().item())


if __name__ == "__main__":
    torch.manual_seed(0)
    res = []
    for idx in (0, 3, 5):
        blk32, blk16 = load_block(idx), load_block(idx, dtype=torch.float16)
        fused = load_fused(idx)
        for B in (1, 3, 16, 64, 100):
            res.append(compare(f"block{idx} random B={B} 14x14", torch.randn(B, 197, 192, device="cuda"), blk32, blk16, fused))
        for g in (7, 28):
            res.append(compare(f"block{idx} random B=8 {g}x{g}", torch.randn(8, g * g + 1, 192, device="cuda"), blk32, blk16, fused))
        res.append(compare(f"block{idx} real CIFAR-100 B=64", real_activations(idx), blk32, blk16, fused))
    save("m2_correctness.json", res)
    print("ALL PASS" if all(r["passed"] for r in res) else "SOME FAIL")
