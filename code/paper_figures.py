"""Figures for the revised paper, based only on frozen result files."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
with (ROOT / "results/final_case/final_case.json").open() as f:
    case = json.load(f)
with (ROOT / "results/最终主点性能实验/per_case.csv").open(encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
with (ROOT / "results/整理阻力剖面实验/profile.csv").open(encoding="utf-8-sig") as f:
    profiles = list(csv.DictReader(f))

def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"))
    plt.close(fig)

fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.7), layout="constrained")
for ax, variant, column, ylabel in [(a, "F_LS", "pure_data_profile_R", "Data misfit $R$"),
                                    (b, "F1_MAP", "objective", "Penalized objective $J$")]:
    group = sorted((r for r in profiles if r["variant"] == variant), key=lambda r: float(r["cd"]))
    assert len(group) == 26
    ax.plot([float(r["cd"]) for r in group], [float(r[column]) for r in group], color="#0072B2")
    ax.axvline(.22, color="black", ls="--", lw=1, label="Frozen $C_D=0.22$")
    ax.set(xlabel="Drag coefficient $C_D$", ylabel=ylabel, title=variant.replace("_", " "))
    ax.legend(fontsize=8, frameon=False)
save(fig, "drag_profile_readable")

# The marker and ellipse are intentionally labelled with their different conditions.
fig, ax = plt.subplots(figsize=(6.5, 3.3), layout="constrained")
ax.add_patch(Rectangle((-3.66, 0), 7.32, 2.44, fill=False, lw=1.4, color="black"))
region = case["spatial"]["calibrated"]
cov = np.asarray(region["covariance_m2"])
val, vec = np.linalg.eigh(cov)
order = np.argsort(val)[::-1]
val, vec = val[order], vec[:, order]
angle = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
axes = 2 * np.sqrt(region["threshold"] * val)
ax.add_patch(Ellipse(region["center_yz_m"], *axes, angle=angle, facecolor="#56B4E9",
                     alpha=.25, edgecolor="#0072B2", lw=1.5,
                     label="Calibrated region: direct crossing"))
ax.plot(case["primary_point"]["y_m"], case["primary_point"]["z_m"], "*", ms=12,
        color="#D55E00", label="Smoothed peak: inside goal")
ax.set(xlabel="Lateral position $Y$ (m)", ylabel="Height $Z$ (m)", xlim=(-4, 4), ylim=(-.1, 2.8))
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="lower left", frameon=True, fontsize=8)
save(fig, "case_result")

subset = [r for r in rows if float(r["sigma_m"]) == .01 and r["truth_inside"] == "True"
          and r["stable_inside"] == "True" and r["peak_error_m"] and r["g_error_m"]]
assert len(subset) == 49 and len({r["source_id"] for r in subset}) == 49
f = np.array([float(r["peak_error_m"]) for r in subset])
g = np.array([float(r["g_error_m"]) for r in subset])
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
for values, label, color, style in [(f, "F1", "#0072B2", "-"), (g, "Gravity G", "#D55E00", "--")]:
    a.step(np.sort(values), np.arange(1, len(values)+1)/len(values), where="post",
           label=label, color=color, ls=style)
a.set(xlabel="Goal-plane error (m)", ylabel="Empirical cumulative fraction", ylim=(0, 1.03))
a.legend(frameon=False)
b.scatter(g, f, color="#0072B2", alpha=.75, s=20)
limit = max(f.max(), g.max()) * 1.06
b.plot([0, limit], [0, limit], "k--", lw=1)
b.set(xlabel="Gravity G error (m)", ylabel="F1 error (m)", xlim=(0, limit), ylim=(0, limit))
b.set_aspect("equal", adjustable="box")
save(fig, "point_errors_1cm")

noise = [.005, .01, .02, .05]
inside, direct, covered = [], [], []
for sig in noise:
    level = [r for r in rows if float(r["sigma_m"]) == sig]
    x = [r for r in level if r["truth_inside"] == "True"]
    y = [r for r in level if r["truth_direct"] == "True"]
    inside.append(sum(r["stable_inside"] == "True" for r in x)/len(x))
    direct.append(sum(r["region_covered"] == "True" for r in y)/len(y))
    covered.append((len(x), len(y)))
assert covered == [(49, 77)] * 4
fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.8), layout="constrained", sharey=True)
a.plot(noise, inside, "o-", color="#0072B2")
b.plot(noise, direct, "s-", color="#D55E00")
b.axhline(.9, color="black", ls="--", lw=1, label="90% target")
for ax in (a, b):
    ax.set_xscale("log")
    ax.set(xlabel="Observation $\\sigma$ (m; log scale)", ylim=(0, 1.05), xticks=noise)
    ax.set_xticklabels(["0.005", "0.01", "0.02", "0.05"])
    ax.minorticks_off()
a.set(ylabel="Fraction", title="Stable point: 49 in-goal sources")
b.set_title("Region coverage: 77 crossing sources")
b.legend(frameon=False, loc="lower right")
save(fig, "coverage_stability")
