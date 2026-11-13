"""Timing of one NCA block: --impl ref (M1, fp32+fp16) or --impl fused (M2, fp16), B in {1,16,64,largest}.

One command:  python bench_block.py [--impl fused]   (spawns 3 independent process launches, merges them)
Writes results/m1_timing.json (ref) or results/m2_timing.json (fused)
"""
import argparse, json, subprocess, sys, statistics
import torch
from bench_utils import time_cuda, count_kernels, largest_batch, save, gpu_state
from reference import load_block

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def worker(batches, impl):
    out = []
    for name, dt in DTYPES.items():
        if impl == "fused":
            if name != "fp16":
                continue
            from fused import load_fused
            blk = load_fused(0)
        else:
            blk = load_block(0, dtype=dt)
        for B in batches:
            x = torch.randn(B, 197, 192, device="cuda", dtype=dt)
            with torch.no_grad():
                fn = lambda: blk(x)
                r = time_cuda(fn)
                kn = count_kernels(fn) if B == batches[0] else None
            r.update(dtype=name, batch=B, us_per_image=r["median_ms"] * 1e3 / B,
                     kernels_per_block=len(kn) if kn else None, kernel_names=kn)
            out.append(r)
            print(f"  {name} B={B:5d}  median {r['median_ms']:8.3f} ms  IQR {r['iqr_ms']:.3f}  {r['us_per_image']:.1f} us/img", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=str, default=None)
    ap.add_argument("--impl", default="ref", choices=["ref", "fused"])
    ap.add_argument("--launches", type=int, default=3)
    a = ap.parse_args()
    if a.worker:
        print("RESULT" + json.dumps(worker(json.loads(a.worker), a.impl)))
        return
    batches = [1, 16, 64, 2048] if a.impl == "fused" else None   # same batches as the M1 reference run
    blk = load_block(0)
    def probe(B):
        with torch.no_grad():
            blk(torch.randn(B, 197, 192, device="cuda"))
    if batches is None:
        big = largest_batch(probe)          # fp32 is the memory-hungry case
        batches = [1, 16, 64] + ([big] if big and big > 64 else [])
    del blk; torch.cuda.empty_cache()
    print("batches", batches, flush=True)
    launches = []
    for i in range(a.launches):
        print(f"launch {i}", flush=True)
        p = subprocess.run([sys.executable, __file__, "--impl", a.impl, "--worker", json.dumps(batches)],
                           capture_output=True, text=True)
        sys.stdout.write("".join(l + "\n" for l in p.stdout.splitlines() if not l.startswith("RESULT")))
        if p.returncode: sys.stderr.write(p.stderr[-2000:]); sys.exit(1)
        launches.append(json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT")][0][6:]))
    summary = []
    for j, rec in enumerate(launches[0]):
        meds = [l[j]["median_ms"] for l in launches]
        summary.append(dict(dtype=rec["dtype"], batch=rec["batch"], median_of_medians_ms=statistics.median(meds),
                            medians_ms_per_launch=meds, iqr_ms_per_launch=[l[j]["iqr_ms"] for l in launches],
                            us_per_image=statistics.median(meds) * 1e3 / rec["batch"],
                            kernels_per_block=rec["kernels_per_block"]))
    p = torch.cuda.get_device_properties(0)
    save("m1_timing.json" if a.impl == "ref" else "m2_timing.json", dict(gpu=p.name, L2_bytes=p.L2_cache_size, smem_optin_bytes=p.shared_memory_per_block_optin, sms=p.multi_processor_count,
                                batches=batches, summary=summary, launches=launches,
                                kernel_names=launches[0][0]["kernel_names"]))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
