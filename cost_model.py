"""M5: data-movement / energy cost model of token mixers (global attention, window attention, NCA) and its validation.

    python cost_model.py  -> results/m5_cost_model.json

time per image  = max(FLOPs / peak_compute, DRAM_bytes / peak_bandwidth)        (peaks MEASURED by peaks.py)
calibration     = one efficiency factor per mixer, eta = t_model(224 px) / t_measured(224 px)   (fit at 224 only)
validation      = predictions at 320/448/640/1024 px (held out) vs results/m5_mixer_timing.json, reported with and without eta
projection      = compute x3.0 and bandwidth x1.6 per 2-year generation (Gholami et al., arXiv 2403.14123), 5 generations
energy          = FLOPs/2 * E_MAC + DRAM bytes * E_BYTE      ASSUMPTIONS, see ENERGY below.  On-chip SRAM/L2 energy is NOT modelled.
All bytes are fp16, per image, with every intermediate tensor in DRAM (the large-batch regime). FLOP = 2 x MAC.
"""
import json, math
from bench_utils import save

D, HID, K, HEADS, WIN, S = 192, 384, 4, 3, 7, 2
PARAMS_NCA, PARAMS_ATTN = 301_056, 4 * D * D + 4 * D
peaks = json.load(open("results/peaks.json"))
C0, BW0 = peaks["peak_fp16_tflops"] * 1e12, peaks["peak_read_gbps"] * 1e9
GROWTH_C, GROWTH_BW, GENS = 3.0, 1.6, 5

# ENERGY (ASSUMPTIONS).  DRAM: Horowitz ISSCC 2014 text, "energy cost of a DRAM access (1 to 2 nJ)", taken per 8-byte access
# -> 125..250 pJ/B, centre 187.5.  E_MAC: ~1.5 pJ (fp16 multiply 1.1 + add 0.4 from the paper's Fig. 1.1.9, an image I could not
# machine-read, so UNVERIFIED); sensitivity: 10 pJ ("functional operation", verified text).  45 nm numbers, held constant over generations.
E_BYTE = dict(low=125e-12, mid=187.5e-12, high=250e-12)
E_MAC = dict(low=1.5e-12, high=10e-12)


def grid_pad(g):
    return WIN * math.ceil(g / WIN)


def flops(mixer, g):
    N = g * g
    if mixer.startswith("nca"):
        return K * N * (2 * 9 * 3 * D + 2 * (3 * D * HID + HID * D))
    if mixer.startswith("attn_global"):
        return 8 * N * D * D + 4 * N * N * D
    Np = grid_pad(g) ** 2                                            # window attention on the padded grid
    return 8 * Np * D * D + 4 * Np * WIN * WIN * D


def dram_bytes(mixer, g, B=64):
    N = g * g
    if mixer == "nca_unfused":      # per step: conv 4D, linear1 5D, relu 4D, linear2 3D, residual add 3D = 19 D; + LN/clone/sub/cat 9 D
        el = (19 * K + 9) * D * N + PARAMS_NCA / B
    elif mixer == "nca_fused":      # implementation floor: LN 2ND, 4 steps x 2ND, init re-read ND, output zero-fill+write 2ND -> 12 ND
        el = 12 * D * N + PARAMS_NCA / B
    elif mixer == "nca_temporal":   # HYPOTHETICAL (not built; M3 NO-GO): read input once, write delta once
        el = 2 * D * N + PARAMS_NCA / B
    elif mixer == "attn_global_materialised":   # qkv/proj/bmm traffic 10ND + scores written, read, softmax r/w, read: 4 H N^2
        el = 10 * D * N + 4 * HEADS * N * N + PARAMS_ATTN / B
    elif mixer == "attn_global_fused":          # flash-style: no N^2 term
        el = 10 * D * N + PARAMS_ATTN / B
    elif mixer == "attn_window7":               # as materialised, but N^2 -> 49 N; + window partition/merge copies 4ND
        Np = grid_pad(g) ** 2
        el = 14 * D * Np + 4 * HEADS * WIN * WIN * Np + PARAMS_ATTN / B
    return el * S


def t_model(mixer, g, gen=0, B=64):
    return max(flops(mixer, g) / (C0 * GROWTH_C ** gen), dram_bytes(mixer, g, B) / (BW0 * GROWTH_BW ** gen))


def energy(mixer, g, B=64, e_byte="mid", e_mac="low"):
    return flops(mixer, g) / 2 * E_MAC[e_mac] + dram_bytes(mixer, g, B) * E_BYTE[e_byte]


MEASURED = ["nca_unfused", "nca_fused", "attn_global_materialised", "attn_global_fused", "attn_window7"]
ALL = MEASURED + ["nca_temporal"]
meas = {(r["mixer"], r["res"]): r["us_per_image"] * 1e-6 for r in json.load(open("results/m5_mixer_timing.json")) if r["us_per_image"]}
RES = sorted({r for _, r in meas})
eta = {m: t_model(m, 224 // 16) / meas[(m, 224)] for m in MEASURED}       # fit at 224 px only

# ---- validation at held-out resolutions (Batch for weight amortisation = tokens-constant batch used in bench_mixers)
val = []
for m in MEASURED:
    for res in RES:
        g = res // 16; B = max(2, 12544 // (g * g))
        tm = t_model(m, g, 0, B); tc = tm / eta[m]; tt = meas[(m, res)]
        val.append(dict(mixer=m, res=res, held_out=res != 224, measured_us=tt * 1e6, model_uncalibrated_us=tm * 1e6, model_calibrated_us=tc * 1e6,
                        err_uncal=(tm - tt) / tt, err_cal=(tc - tt) / tt, bound="memory" if dram_bytes(m, g, B) / BW0 > flops(m, g) / C0 else "compute"))

# ---- M1/M2 block-level DRAM-traffic calibration (analytic bytes vs ncu-measured bytes at 224 px)
dram_meas = {}
for fn, key in (("results/m1_dram_fp32.json", None), ("results/m1_dram_fp16.json", None), ("results/m2_dram_fused.json", None)):
    for r in json.load(open(fn)):
        if r["protocol"] == "range" and r["measured"]:
            dram_meas[(r["impl"], r["dtype"], r["batch"])] = r["dram_bytes_per_image"]
traffic = []
for B in (16, 64, 2048):
    for impl, mixer in (("ref", "nca_unfused"), ("fused", "nca_fused")):
        k = (impl, "fp16", B)
        if k in dram_meas:
            a = dram_bytes(mixer, 14, B)
            traffic.append(dict(mixer=mixer, batch=B, model_bytes=a, measured_bytes=dram_meas[k], err=(a - dram_meas[k]) / dram_meas[k]))

# ---- M4: composite per-image model of the whole hybrid vs measured (B=64, fp16), eta from block-level benchmarks (mixers) only
def mlp_cost(N):          # token MLP: 16 N D^2 FLOPs; bytes (LN 2, fc1 5, GELU 8, fc2 5, residual 3) x N D
    return 16 * N * D * D, 23 * N * D * S + 8 * D * D * S / 64

N = 196
def t_comp(f, b): return max(f / C0, b / BW0)
m4 = json.load(open("results/m4_throughput.json"))["summary"]
m4 = {(r["variant"], r["batch"]): r for r in m4}
comp = {}
for variant, nca in (("ref_fp16", "nca_unfused"), ("fused_fp16", "nca_fused")):
    mix_nca = t_model(nca, 14); mix_att = t_model("attn_global_materialised", 14); mf, mb = mlp_cost(N)
    other = t_comp(2 * N * 768 * D + 2 * D * 100, 224 * 224 * 3 * S * 2)          # patch embed + head (small)
    uncal = 6 * mix_nca + 6 * mix_att + 12 * t_comp(mf, mb) + other
    cal = 6 * mix_nca / eta[nca] + 6 * mix_att / eta["attn_global_materialised"] + 12 * t_comp(mf, mb) + other
    tt = m4[(variant, 64)]["median_of_medians_ms"] / 64 * 1e-3
    comp[variant] = dict(measured_us=tt * 1e6, model_uncalibrated_us=uncal * 1e6, model_mixer_calibrated_us=cal * 1e6,
                         err_uncal=(uncal - tt) / tt, err_mixer_calibrated=(cal - tt) / tt)

# ---- projection: time and energy per image, 5 generations, 224..1024 px (every 16 px), calibrated and uncalibrated
RES_FINE = list(range(224, 1025, 16))
proj = {m: {gen: {res: dict(t_cal=t_model(m, res // 16, gen) / eta.get(m, 1.0), t_ideal=t_model(m, res // 16, gen))
                  for res in RES_FINE} for gen in range(GENS + 1)} for m in ALL}
for m in ALL:
    for res in RES_FINE:
        e = {k: energy(m, res // 16, e_byte=k) for k in E_BYTE}
        for gen in range(GENS + 1):
            proj[m][gen][res]["energy_J_mid"] = e["mid"]; proj[m][gen][res]["energy_J_low"] = e["low"]; proj[m][gen][res]["energy_J_high"] = e["high"]

def parity(nca, attn, gen, key):
    """Sustained parity: smallest resolution from which NCA time <= attention time at that and EVERY larger resolution in range.
    (First-crossing is fooled by window-attention padding at grids not divisible by 7.) None = no parity in range."""
    p = None
    for res in reversed(RES_FINE):
        if proj[nca][gen][res][key] <= proj[attn][gen][res][key]:
            p = res
        else:
            break
    return p

pairs = [(n, a) for n in ("nca_unfused", "nca_fused", "nca_temporal") for a in ("attn_global_materialised", "attn_global_fused", "attn_window7")]
par = {f"{n}__vs__{a}": {key: {gen: parity(n, a, gen, key) for gen in range(GENS + 1)} for key in ("t_cal", "t_ideal")} for n, a in pairs}
def parity_energy(n, a, key):
    pr = None
    for res in reversed(RES_FINE):
        if proj[n][0][res][key] <= proj[a][0][res][key]:
            pr = res
        else:
            break
    return pr


par_energy = {f"{n}__vs__{a}": {k: parity_energy(n, a, k) for k in ("energy_J_low", "energy_J_mid", "energy_J_high")} for n, a in pairs}
measured_parity = {}
for n in ("nca_unfused", "nca_fused"):
    for a in ("attn_global_materialised", "attn_global_fused", "attn_window7"):
        measured_parity[f"{n}__vs__{a}"] = next((r for r in RES if meas[(n, r)] <= meas[(a, r)]), None)

save("m5_cost_model.json", dict(
    peaks=dict(compute_flops=C0, bandwidth_Bps=BW0), growth=dict(compute=GROWTH_C, bandwidth=GROWTH_BW, generations=GENS),
    energy_assumptions=dict(E_BYTE_J=E_BYTE, E_MAC_J=E_MAC, note="45 nm Horowitz-style assumptions, constant across generations; on-chip energy not modelled"),
    eta=eta, validation=val, traffic_calibration=traffic, composite_m4=comp, parity_model=par, parity_energy_gen0_model=par_energy,
    parity_measured_gen0=measured_parity, projection={m: {str(g): {str(r): v for r, v in d.items()} for g, d in gs.items()} for m, gs in proj.items()},
    res_fine=RES_FINE))

print("eta (model/measured at 224):", {m: round(v, 3) for m, v in eta.items()})
print("\nheld-out validation, error = (model - measured)/measured")
print(f"{'mixer':26s} {'res':>5s} {'meas us':>9s} {'uncal err':>10s} {'cal err':>9s} bound")
for v in val:
    if v["held_out"]:
        print(f"{v['mixer']:26s} {v['res']:5d} {v['measured_us']:9.1f} {100*v['err_uncal']:9.0f}% {100*v['err_cal']:8.0f}% {v['bound']}")
print("\nDRAM-traffic model vs ncu:", [(t['mixer'], t['batch'], f"{100*t['err']:+.0f}%") for t in traffic])
print("composite M4:", {k: {a: round(b, 3) if a.startswith('err') else round(b, 1) for a, b in v.items()} for k, v in comp.items()})
print("\nmeasured parity at gen 0 (first res where NCA <= attention):", measured_parity)
for k, v in par.items():
    print(f"model parity {k:52s} calibrated: {v['t_cal']}")
