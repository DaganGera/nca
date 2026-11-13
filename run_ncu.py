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


def num(s):
    return float(s.replace(",", ""))


def min_bytes_per_image(B, s, n=196, d=192):
    return 2 * n * d * s + PARAMS * s / B


PROTOCOLS = {  # name -> ncu flags (see PREREGISTRATION.md section 1 and Amendments 2-3)
    "range": ["--replay-mode", "range", "--cache-control", "none"],              # whole forward as one range; main protocol
    "app":   ["--replay-mode", "application", "--cache-control", "none", "--nvtx", "--nvtx-include", "nca_block/"],
    "flush": ["--replay-mode", "kernel", "--cache-control", "all", "--nvtx", "--nvtx-include", "nca_block/"],
}


def run(impl, dtype, B, protocol, timeout=1200):
    cmd = [NCU, *PROTOCOLS[protocol], "--metrics", METRICS, "--csv", "--page", "raw", "--print-units", "base",
           sys.executable, "profile_target.py", "--impl", impl, "--dtype", dtype, "--batch", str(B)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:   # never estimate: record as not measured
        return dict(impl=impl, dtype=dtype, batch=B, protocol=protocol, measured=False, note=f"ncu timed out after {timeout}s")
    if p.returncode:
        raise RuntimeError(p.stdout[-1500:] + p.stderr[-1500:])
    lines = p.stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = [r for r in csv.DictReader(io.StringIO(chr(10).join(lines[start:]))) if r["ID"].isdigit()]  # drop units row
    rd = sum(num(r["dram__bytes_read.sum"]) for r in rows)
    wr = sum(num(r["dram__bytes_write.sum"]) for r in rows)
    s = 4 if dtype == "fp32" else 2
    mn = min_bytes_per_image(B, s)
    return dict(impl=impl, dtype=dtype, batch=B, protocol=protocol, measured=True,
                kernels=None if protocol == "range" else len(rows),   # range replay returns one aggregate row
                dram_read_bytes_per_image=rd / B, dram_write_bytes_per_image=wr / B,
                dram_bytes_per_image=(rd + wr) / B, min_bytes_per_image=mn, R=(rd + wr) / B / mn,
                kernel_names=None if protocol == "range" else [r["Kernel Name"] for r in rows])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="ref")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--batches", default="1,16,64,2048")
    ap.add_argument("--validate-batches", default="1,16", help="also profile with app replay (cross-check of range replay)")
    ap.add_argument("--flush-batches", default="16", help="also profile with the L2-flush protocol (sensitivity check)")
    ap.add_argument("--out", default="m1_dram.json")
    a = ap.parse_args()
    val, fl = (set(int(x) for x in s.split(",") if x) for s in (a.validate_batches, a.flush_batches))
    out = []
    for dtype in a.dtypes.split(","):
        for B in map(int, a.batches.split(",")):
            for protocol in ["range"] + (["app"] if B in val else []) + (["flush"] if B in fl else []):
                r = run(a.impl, dtype, B, protocol); out.append(r); save(a.out, out)   # incremental save
                msg = f"R={r['R']:.2f}  DRAM/img={r['dram_bytes_per_image']/1e6:8.3f} MB (rd {r['dram_read_bytes_per_image']/1e6:.3f} wr {r['dram_write_bytes_per_image']/1e6:.3f})  min={r['min_bytes_per_image']/1e6:.3f}" if r["measured"] else r["note"]
                print(f"{a.impl} {dtype} B={B:5d} {protocol:6s} {msg}", flush=True)
