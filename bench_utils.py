"""Timing + GPU-state helpers shared by every benchmark (protocol: PREREGISTRATION.md section 1)."""
import json, os, statistics, time
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # NVML optional; recorded as None
    _h = None


def _try(f, *a):
    try:
        return f(*a)
    except Exception:  # field unsupported on this GPU/driver -> recorded as None, never estimated
        return None


def gpu_state():
    if _h is None:
        return None
    pw = _try(pynvml.nvmlDeviceGetPowerUsage, _h)
    return dict(sm_clock_mhz=_try(pynvml.nvmlDeviceGetClockInfo, _h, pynvml.NVML_CLOCK_SM),
                mem_clock_mhz=_try(pynvml.nvmlDeviceGetClockInfo, _h, pynvml.NVML_CLOCK_MEM),
                temp_c=_try(pynvml.nvmlDeviceGetTemperature, _h, pynvml.NVML_TEMPERATURE_GPU),
                power_w=None if pw is None else pw / 1000)


def time_cuda(fn, warmup=20, iters=100):
    """CUDA-event timing of fn(). Returns ms stats (median, q1, q3, iqr) plus GPU state before/after."""
    before = gpu_state()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in ev)
    q = statistics.quantiles(t, n=4)
    return dict(median_ms=statistics.median(t), q1_ms=q[0], q3_ms=q[2], iqr_ms=q[2] - q[0],
                min_ms=t[0], max_ms=t[-1], iters=iters, warmup=warmup, gpu_before=before, gpu_after=gpu_state())


def count_kernels(fn):
    """Names of the CUDA kernels launched by one call of fn() (torch profiler)."""
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def largest_batch(make_and_run, start=64, cap=4096, vram_frac=0.7):
    """Largest power-of-two batch whose peak allocation stays under vram_frac of VRAM.

    Plain OOM-catching is not enough on Windows: WDDM can spill to system RAM instead of failing,
    which would give a meaningless (host-bandwidth) timing, so we also cap the peak allocation.
    """
    total = torch.cuda.get_device_properties(0).total_memory
    best, b = None, start
    while b <= cap:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            make_and_run(b); torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            break
        if torch.cuda.max_memory_allocated() > vram_frac * total:
            break
        best = b
        b *= 2
    torch.cuda.empty_cache()
    return best


def save(name, obj):
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", name), "w") as f:
        json.dump(obj, f, indent=2)
