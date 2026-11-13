"""M6 figures, drawn only from results/*.json:  python plot_results.py -> results/fig1_roofline.png, fig2_dram.png, fig3_projection.png
Palette: validated reference categorical order (blue, orange, aqua, yellow, magenta); direct labels + line styles carry identity (contrast < 3:1 for aqua/yellow/magenta)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "font.size": 9, "legend.frameon": False})
def end_labels(ax, items, xpos, gap=0.075):
    """items: [(text, y_value, color)]; place labels at x=xpos, pushed apart (in log10 y) so none overlap."""
    items = sorted(items, key=lambda t: t[1]); ys = [np.log10(t[1]) for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.06 * (np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])))
    for (txt, y, col), yy in zip(items, ys):
        ax.annotate(txt, (xpos, 10 ** yy), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK2, va="center", annotation_clip=False)
        ax.plot([xpos - 12, xpos], [y, 10 ** yy], color=col, lw=0.6, alpha=0.6, clip_on=False)
J = lambda f: json.load(open(f"results/{f}"))
FLOPS = 470_550_528
m1, m2, peaks, cm = J("m1_summary.json"), J("m2_summary.json"), J("peaks.json"), J("m5_cost_model.json")
C16, C32, BW = peaks["peak_fp16_tflops"], peaks["peak_fp32_tflops"], peaks["peak_read_gbps"]

# ---------------------------------------------------------------- Fig 1: roofline
fig, ax = plt.subplots(figsize=(7.4, 4.8))
x = np.logspace(0, 4, 200)
for C, name, col in ((C16, "fp16 tensor roof", INK2), (C32, "fp32 roof", MUTED)):
    ax.plot(x, np.minimum(C, x * BW / 1e3), color=col, lw=1.2, ls="-" if "16" in name else "--")
ax.text(3.9, 1.75, f"bandwidth {BW:.0f} GB/s (measured)", rotation=52, color=MUTED, fontsize=8, ha="center")
ax.text(1400, C16 * 1.07, f"fp16 peak {C16:.1f} TFLOP/s", color=INK2, fontsize=8.5, ha="center")
ax.text(900, C32 * 0.84, f"fp32 peak {C32:.1f} TFLOP/s", color=MUTED, fontsize=8.5, ha="center")
pts = []
for r in m1["main"]:
    if r["batch"] in (64, 2048):
        pts.append((f"unfused {r['dtype']}", r["flop_per_dram_byte"], r["achieved_gflops"] / 1e3, r["batch"], BLUE if r["dtype"] == "fp16" else MAGENTA))
for r in m2["rows"]:
    if r["batch"] in (64, 2048):
        pts.append(("fused fp16", FLOPS / (r["fused_dram_mb_per_image"] * 1e6), r["fused_tflops"], r["batch"], ORANGE))
seen = set()
for lab, xi, yi, B, col in pts:
    ax.plot(xi, yi, "o" if B == 64 else "s", ms=9, color=col, mec=col, mew=1.6, mfc=col if B == 64 else SURF)
    ax.annotate(f"{lab}, B={B}", (xi, yi), xytext=(7, -3 if B == 64 else 7), textcoords="offset points", fontsize=8, color=INK)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 3000); ax.set_ylim(0.8, 60)
ax.set_xlabel("arithmetic intensity: FLOP per DRAM byte (ncu-measured bytes)"); ax.set_ylabel("achieved TFLOP/s")
ax.set_title("One NCA block on RTX 4060: unfused sits under the bandwidth slope, fused moves far right", loc="left")
ax.text(0.01, 0.02, "filled = B 64, hollow = B 2048. fp16 intensity at B=1/16 omitted: fused DRAM bytes there are L2-resident lower bounds.", transform=ax.transAxes, fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig("results/fig1_roofline.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: DRAM bytes per image
dram = {}
import glob
for f in glob.glob("results/m1_dram_*.json") + ["results/m2_dram_fused.json"]:
    for r in json.load(open(f)):
        if r["protocol"] == "range" and r["measured"]:
            dram[(r["impl"], r["dtype"], r["batch"])] = r
fig, ax = plt.subplots(figsize=(7.4, 4.4))
groups = [64, 2048]; w = 0.25
series = [("unfused fp32", ("ref", "fp32"), MAGENTA), ("unfused fp16", ("ref", "fp16"), BLUE), ("fused fp16", ("fused", "fp16"), ORANGE)]
for gi, B in enumerate(groups):
    for si, (lab, (impl, dt), col) in enumerate(series):
        r = dram[(impl, dt, B)]; xpos = gi + (si - 1) * (w + 0.02)
        ax.bar(xpos, r["dram_bytes_per_image"] / 1e6, w, color=col, label=lab if gi == 0 else None, zorder=3)
        ax.text(xpos, r["dram_bytes_per_image"] / 1e6 + 0.4, f"{r['dram_bytes_per_image']/1e6:.2f}", ha="center", fontsize=8.5, color=INK)
    mn = dram[("ref", "fp16", B)]["min_bytes_per_image"] / 1e6
    ax.plot([gi - 0.5, gi + 0.5], [mn, mn], color=INK2, lw=1, ls=":", zorder=4)
ax.set_xticks(range(2)); ax.set_xticklabels([f"batch {b}" for b in groups]); ax.set_ylabel("DRAM MB per image (one NCA block, ncu range replay)")
ax.set_title("Fusion cuts DRAM traffic 13x at B=64 and 4.4x at B=2048 (fp16, same precision)", loc="left"); from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], color=INK2, ls=":", lw=1)); l.append("analytic minimum, fp16 (0.16 / 0.15 MB)")
ax.legend(h, l, loc="upper left"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("results/fig2_dram.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: projection
P = cm["projection"]; RF = cm["res_fine"]
STY = {"nca_unfused": ("NCA unfused", BLUE, "-"), "nca_fused": ("NCA per-step fused (built)", ORANGE, "-"),
       "nca_temporal": ("NCA temporally fused (hypothetical, not built)", ORANGE, ":"),
       "attn_global_materialised": ("global attention, N x N in DRAM", AQUA, "-"), "attn_global_fused": ("global attention, flash-style", YELLOW, "-"),
       "attn_window7": ("window attention 7x7", MAGENTA, "-")}
meas = {(r["mixer"], r["res"]): r["us_per_image"] for r in J("m5_mixer_timing.json") if r["us_per_image"]}
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.4))
for ax, gen in ((axs[0, 0], 0), (axs[0, 1], 5)):
    labs = []
    for m, (lab, col, ls) in STY.items():
        t = [P[m][str(gen)][str(r)]["t_cal"] * 1e6 for r in RF]
        ax.plot(RF, t, color=col, ls=ls, lw=1.7); labs.append((lab.split(" (")[0], t[-1], col))
        if gen == 0 and m in cm["eta"]:
            rr = sorted(r for (mm, r) in meas if mm == m); ax.plot(rr, [meas[(m, r)] for r in rr], "o", ms=6.5, color=col, mec=SURF, mew=1.2)
    ax.set_yscale("log"); ax.set_xlim(224, 1180); ax.set_xlabel("input resolution (px, patch 16)"); ax.set_ylabel("token-mixer time per image (us)")
    ax.figure.canvas.draw(); end_labels(ax, labs, RF[-1] + 14)
    ax.set_title("Today's GPU: dots = measured, lines = model calibrated at 224 px" if gen == 0 else f"Generation +{gen} (compute x{3**gen}, bandwidth x{1.6**gen:.1f}): model projection", loc="left")
ax = axs[1, 0]
ratio = np.array([[P["nca_fused"][str(g)][str(r)]["t_cal"] / P["attn_global_fused"][str(g)][str(r)]["t_cal"] for r in RF] for g in range(6)])
norm = TwoSlopeNorm(vcenter=1.0, vmin=min(ratio.min(), 0.5), vmax=max(ratio.max(), 2.0))
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
im = ax.imshow(ratio, aspect="auto", origin="lower", cmap=cmap, norm=norm, extent=[RF[0] - 8, RF[-1] + 8, -0.5, 5.5])
ax.contour(RF, range(6), ratio, levels=[1.0], colors=INK, linewidths=1.2)
ax.set_xlabel("input resolution (px)"); ax.set_ylabel("hardware generation (+2 yr each)"); ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("time(NCA fused) / time(flash attention):  < 1 means NCA wins", fontsize=8)
ax.set_title(f"No parity vs flash attention in range (min ratio {ratio.min():.2f}, max {ratio.max():.2f})", loc="left")
ax = axs[1, 1]
labs = []
for m, (lab, col, ls) in STY.items():
    e = [P[m]["0"][str(r)]["energy_J_mid"] * 1e3 for r in RF]
    labs.append((lab.split(" (")[0], e[-1], col))
    lo = [P[m]["0"][str(r)]["energy_J_low"] * 1e3 for r in RF]; hi = [P[m]["0"][str(r)]["energy_J_high"] * 1e3 for r in RF]
    ax.fill_between(RF, lo, hi, color=col, alpha=0.15, lw=0); ax.plot(RF, e, color=col, ls=ls, lw=1.7)
ax.set_yscale("log"); ax.set_xlim(224, 1180); ax.set_xlabel("input resolution (px)"); ax.set_ylabel("modelled energy per image (mJ)")
ax.figure.canvas.draw(); end_labels(ax, labs, RF[-1] + 14)
ax.set_title("Energy per image: ASSUMED constants, no measurement", loc="left")
ax.text(0.01, 0.02, "45 nm Horowitz-style: DRAM 125-250 pJ/B (shaded), MAC 1.5 pJ. Same in every generation.", transform=ax.transAxes, fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig("results/fig3_projection.png", dpi=150); plt.close(fig)
print("figures written; ratio min/max", ratio.min(), ratio.max())
