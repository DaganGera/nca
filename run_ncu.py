"""DRAM traffic per image via Nsight Compute (protocol: PREREGISTRATION.md section 1).

    python run_ncu.py --impl ref --out m1_dram.json
Primary protocol  : --replay-mode application --cache-control none  (real L2 state of an actual run)
Secondary protocol: default kernel replay with --cache-control all   (L2 flushed per kernel; upper bound)
Writes results/<out> with bytes/image, kernel count and ratio R = measured / analytic minimum.
"""
import argparse, csv, io, json, subprocess, sys
from bench_utils import save

NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"
METRICS = "dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
PARAMS = 301_056   # per-block parameters (PREREGISTRATION.md section 0)


def min_bytes_per_image(B, s, n=196, d=192):
    return 2 * n * d * s + PARAMS * s / B


def run(impl, dtype, B, protocol):
    flags = (["--replay-mode", "application", "--cache-control", "none"] if protocol == "primary"
             else ["--replay-mode", "kernel", "--cache-control", "all"])
    cmd = [NCU, *flags, "--nvtx", "--nvtx-include", "nca_block/", "--metrics", METRICS,
           "--csv", "--page", "raw", "--print-units", "base",
           sys.executable, "profile_target.py", "--impl", impl, "--dtype", dtype, "--batch", str(B)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-1500:] + p.stderr[-1500:])
    lines = p.stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    rd = sum(float(r["dram__bytes_read.sum"]) for r in rows)
    wr = sum(float(r["dram__bytes_write.sum"]) for r in rows)
    s = 4 if dtype == "fp32" else 2
    mn = min_bytes_per_image(B, s)
    return dict(impl=impl, dtype=dtype, batch=B, protocol=protocol, kernels=len(rows),
                dram_read_bytes_per_image=rd / B, dram_write_bytes_per_image=wr / B,
                dram_bytes_per_image=(rd + wr) / B, min_bytes_per_image=mn, R=(rd + wr) / B / mn,
                gpu_time_us_sum=sum(float(r["gpu__time_duration.sum"]) for r in rows) / 1e3,
                kernel_names=[r["Kernel Name"] for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="ref")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--batches", default="1,16,64,4096")
    ap.add_argument("--out", default="m1_dram.json")
    a = ap.parse_args()
    out = []
    for dtype in a.dtypes.split(","):
        for B in map(int, a.batches.split(",")):
            for protocol in ("primary", "secondary"):
                r = run(a.impl, dtype, B, protocol); out.append(r)
                print(f"{a.impl} {dtype} B={B:5d} {protocol:9s} kernels={r['kernels']:3d} "
                      f"DRAM/img={r['dram_bytes_per_image']/1e6:8.3f} MB  min={r['min_bytes_per_image']/1e6:.3f} MB  R={r['R']:.2f}", flush=True)
    save(a.out, out)
