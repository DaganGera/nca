"""M4: end-to-end CIFAR-100 accuracy and throughput of the original vs fused hybrid model, and DeiT-Tiny.

    python e2e.py acc      -> results/m4_accuracy.json      (10,000 test images, original PIL-bicubic pipeline)
    python e2e.py bench    -> results/m4_throughput.json    (3 process launches; CUDA events; B in {1,16,64,256})
Variants: ref_fp32, ref_fp16 (original model), fused_fp16 (6 NCA mixers replaced), deit_fp32, deit_fp16 (timm DeiT-Tiny, random weights: timing only).
Weights: the checkpoint behind the 81.91% result (checkpoints/nca_vit_hybrid/cifar100/seed42/best.pt).
"""
import json, os, statistics, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
from bench_utils import time_cuda, save, gpu_state

SRC = r"D:\Research\Emergent-Attention1"
CKPT = SRC + r"\checkpoints\nca_vit_hybrid\cifar100\seed42\best.pt"
CACHE = r"E:\NCA\.cache\cifar100_test_224.npy"
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
sys.path.insert(0, SRC)


def test_set():
    """(10000,224,224,3) uint8 = PIL bicubic Resize(224) of the CIFAR-100 test set (== original val transform before ToTensor) and labels."""
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.CIFAR100(SRC + r"\data", train=False, download=False)
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        rs = T.Resize(224, interpolation=T.InterpolationMode.BICUBIC)
        np.save(CACHE, np.stack([np.asarray(rs(im)) for im, _ in ds]))
    return np.load(CACHE, mmap_mode="r"), torch.tensor(ds.targets)


class FusedMixer(nn.Module):
    def __init__(self, sd, idx):
        super().__init__()
        from fused import FusedNCABlock
        self.f = FusedNCABlock(sd, idx)

    def forward(self, x):
        return self.f(x)


def build(variant):
    from src.models.hybrid_nca_vit import HybridNCAViT
    dt = torch.float32 if variant.endswith("fp32") else torch.float16
    if variant.startswith("deit"):
        import timm
        return timm.create_model("deit_tiny_patch16_224", num_classes=100).cuda().to(dt).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = HybridNCAViT(learnable_filters=True, num_classes=100)
    m.load_state_dict(sd)
    m = m.cuda().to(dt).eval()
    if variant.startswith("fused"):
        for i, blk in enumerate(m.nca_blocks):
            blk.nca_attn = FusedMixer({k: v.cuda() for k, v in sd.items()}, i)
    return m


@torch.no_grad()
def evaluate(model, dt, bs=250):
    x8, y = test_set()
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1); std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    preds, top5 = [], 0
    for i in range(0, len(y), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x8[i:i + bs])).cuda().permute(0, 3, 1, 2).float().div(255)
        out = model(((xb - mean) / std).to(dt)).float()
        preds.append(out.argmax(1).cpu())
        top5 += (out.topk(5, 1).indices.cpu() == y[i:i + bs, None]).any(1).sum().item()
    p = torch.cat(preds)
    return p, (p == y).float().mean().item() * 100, top5 / len(y) * 100


def breakdown(model, x, iters=30):
    """Median GPU ms per component (CUDA-event hooks): NCA mixers, token MLPs, MHSA, everything else."""
    groups = {"nca_mixer": [b.nca_attn for b in model.nca_blocks],
              "nca_block_mlp": [b.mlp for b in model.nca_blocks],
              "mhsa": [b.attn for b in model.attn_blocks],
              "attn_block_mlp": [b.mlp for b in model.attn_blocks]}
    ev, hs = {g: [] for g in groups}, []
    for g, mods in groups.items():
        for m in mods:
            def pre(mod, inp, g=g):
                s = torch.cuda.Event(enable_timing=True); s.record(); ev[g].append([s, None])
            def post(mod, inp, out, g=g):
                e = torch.cuda.Event(enable_timing=True); e.record(); ev[g][-1][1] = e
            hs += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    with torch.no_grad():
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        for g in ev: ev[g].clear()
        tot = []
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); model(x); e.record(); tot.append((s, e))
        torch.cuda.synchronize()
    for h in hs: h.remove()
    total = statistics.median(s.elapsed_time(e) for s, e in tot)
    per = {g: sum(s.elapsed_time(e) for s, e in v) / iters for g, v in ev.items()}
    per["other"] = total - sum(per.values())
    per["total"] = total
    return per


def worker(variants, batches):
    out = []
    for v in variants:
        model = build(v)
        dt = torch.float32 if v.endswith("fp32") else torch.float16
        for B in batches:
            x = torch.randn(B, 3, 224, 224, device="cuda", dtype=dt)
            with torch.no_grad():
                r = time_cuda(lambda: model(x), warmup=20, iters=100)
            r.update(variant=v, batch=B, img_per_s=B / (r["median_ms"] * 1e-3))
            if not v.startswith("deit") and B == 64:
                r["breakdown_ms"] = breakdown(model, x)
            out.append(r)
            print(f"  {v:11s} B={B:4d} {r['median_ms']:8.2f} ms (IQR {r['iqr_ms']:.2f})  {r['img_per_s']:8.0f} img/s", flush=True)
        del model; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    variants = ["ref_fp32", "ref_fp16", "fused_fp16", "deit_fp32", "deit_fp16"]
    if mode == "acc":
        res, preds = {}, {}
        for v in variants[:3]:
            m = build(v)
            p, t1, t5 = evaluate(m, torch.float32 if v.endswith("fp32") else torch.float16)
            preds[v] = p; res[v] = dict(top1=t1, top5=t5, n=len(p)); print(v, f"top1 {t1:.2f}  top5 {t5:.2f}", flush=True)
            del m; torch.cuda.empty_cache()
        res["delta_top1_fused_vs_ref_fp16"] = res["fused_fp16"]["top1"] - res["ref_fp16"]["top1"]
        res["delta_top1_fused_vs_ref_fp32"] = res["fused_fp16"]["top1"] - res["ref_fp32"]["top1"]
        res["pred_agreement_fused_vs_ref_fp16"] = (preds["fused_fp16"] == preds["ref_fp16"]).float().mean().item()
        res["pred_agreement_fused_vs_ref_fp32"] = (preds["fused_fp16"] == preds["ref_fp32"]).float().mean().item()
        res["pred_agreement_ref_fp16_vs_ref_fp32"] = (preds["ref_fp16"] == preds["ref_fp32"]).float().mean().item()
        print(json.dumps({k: v for k, v in res.items() if k.startswith(("delta", "pred"))}, indent=1))
        save("m4_accuracy.json", res)
    elif mode == "worker":
        print("RESULT" + json.dumps(worker(variants, [1, 16, 64, 256])))
    elif mode == "bench":
        launches = []
        for i in range(3):
            print(f"launch {i}", flush=True)
            p = subprocess.run([sys.executable, __file__, "worker"], capture_output=True, text=True)
            sys.stdout.write("".join(l + "\n" for l in p.stdout.splitlines() if not l.startswith("RESULT")))
            if p.returncode: sys.stderr.write(p.stderr[-2000:]); sys.exit(1)
            launches.append(json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][0][6:]))
        summary = []
        for j, rec in enumerate(launches[0]):
            meds = [l[j]["median_ms"] for l in launches]
            summary.append(dict(variant=rec["variant"], batch=rec["batch"], median_of_medians_ms=statistics.median(meds), medians_ms_per_launch=meds,
                                iqr_ms_per_launch=[l[j]["iqr_ms"] for l in launches],
                                img_per_s=rec["batch"] / (statistics.median(meds) * 1e-3),
                                breakdown_ms_per_launch=[l[j].get("breakdown_ms") for l in launches]))
        save("m4_throughput.json", dict(summary=summary, launches=launches))
        for s in summary:
            print(f"{s['variant']:11s} B={s['batch']:4d} {s['median_of_medians_ms']:8.2f} ms {s['img_per_s']:8.0f} img/s")
