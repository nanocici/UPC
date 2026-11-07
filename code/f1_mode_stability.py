"""F1 专用众数稳定性：原始格网 + 固定带宽 KDE（64k × 4 seeds）。

口径：
- 条件先归一化，不因固定窗口删除候选后重新归一化；
- 窗口仅用于求值和搜索，记录窗口外质量与边界峰；
- 采样/ESS/事件质量诊断先通过，才判断峰位置稳定性；
- 四个种子各自条件归一化后等权合并，作为汇总峰；
- 历史原始格网失败另行保留，新 64k 结果独立报告。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter

from config import CASE, INFERENCE, PROJECT_ROOT
from uncertainty import UNCERTAINTY, grid_mode, sample_case

BASE_WIDTH = 2.44 / 24
WIDTHS = (BASE_WIDTH / 2, BASE_WIDTH, BASE_WIDTH * 2)
SHIFTS = (0.0, BASE_WIDTH / 2)
BANDWIDTHS = (0.05, 0.10, 0.20)
PREFIXES = (4000, 8000, 16000, 32000, 64000)
SEEDS = (20261109, 20261210, 20261311, 20261412)
F1_SETTINGS = replace(INFERENCE, vertical_force_prior_std=0.10)
OPTIONS = replace(UNCERTAINTY, base_samples=32000, max_sample_doublings=0)
TOL = 0.20
Y_RANGE = (1.5, 4.5)
Z_RANGE = (0.5, 3.0)
RES = 0.01
PEAK_RATIO = 0.8
PEAK_SEPARATION = 0.10


def run_seed(args):
    seed, output = args
    result = sample_case("F", settings=F1_SETTINGS, uncertainty=OPTIONS,
                         drag_coefficient=0.22, sigma_m=0.01, seed=seed)
    draws = result.pop("draws", [])
    path = Path(output) / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=False)
    with gzip.open(path / "candidates.jsonl.gz", "wt", encoding="utf-8") as handle:
        for draw in draws:
            handle.write(json.dumps(draw, ensure_ascii=False, allow_nan=False) + "\n")
    (path / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"seed": seed, "draws": len(draws), "summary": result}


def select(draws, condition):
    if condition == "direct_crossing":
        return [d for d in draws if d["weight"] > 0 and d["event"] in ("goal_inside", "goal_outside")]
    return [d for d in draws if d["weight"] > 0 and d["event"] == "goal_inside"]


def arrays(draws, condition):
    points = np.asarray([d["goal_yz_m"] for d in draws], float)
    weights = np.asarray([d["weight"] for d in draws], float)
    if not len(weights) or weights.sum() <= 0:
        raise ValueError("空条件")
    return points, weights / weights.sum()


def kde_window(points, weights, bandwidth):
    y_edges = np.arange(Y_RANGE[0], Y_RANGE[1] + RES / 2, RES)
    z_edges = np.arange(Z_RANGE[0], Z_RANGE[1] + RES / 2, RES)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(y_edges, z_edges), weights=weights)
    cell_area = RES ** 2
    density = gaussian_filter(hist, sigma=bandwidth / RES, mode="constant", truncate=4.0) / cell_area
    mass = float(density.sum() * cell_area)
    y = (y_edges[:-1] + y_edges[1:]) / 2
    z = (z_edges[:-1] + z_edges[1:]) / 2
    return y, z, density, mass


def peak_of(y, z, density, mass):
    iy, iz = np.unravel_index(np.argmax(density), density.shape)
    boundary = (iy < 10 or iy > len(y) - 11 or iz < 10 or iz > len(z) - 11)
    return {"y_m": float(y[iy]), "z_m": float(z[iz]),
            "density": float(density[iy, iz]), "mass_in_window": mass,
            "outside_mass": float(1 - mass), "boundary_peak": bool(boundary),
            "iy": int(iy), "iz": int(iz)}


def local_peaks(y, z, density):
    size = int(round(PEAK_SEPARATION / RES)) * 2 + 1
    lm = maximum_filter(density, size=size, mode="nearest")
    mask = (density == lm) & (density >= PEAK_RATIO * density.max())
    iy, iz = np.nonzero(mask)
    order = np.argsort(density[iy, iz])[::-1]
    peaks = []
    for index in order:
        py, pz = float(y[iy[index]]), float(z[iz[index]])
        if any(np.hypot(py - p["y_m"], pz - p["z_m"]) < PEAK_SEPARATION for p in peaks):
            continue
        mask2 = (y[:, None] - py) ** 2 + (z[None, :] - pz) ** 2 <= PEAK_SEPARATION ** 2
        peaks.append({"y_m": py, "z_m": pz,
                      "fraction_of_max": float(density[iy[index], iz[index]] / density.max()),
                      "mass_within_separation": float(density[mask2].sum() * RES ** 2)})
    return peaks


def high_density_regions(y, z, density):
    mask = density >= PEAK_RATIO * density.max()
    labels, count = label(mask, structure=np.ones((3, 3), int))
    regions = []
    for component in range(1, count + 1):
        selected = labels == component
        iy, iz = np.nonzero(selected)
        regions.append({"mass": float(density[selected].sum() * RES ** 2),
                        "area_m2": float(selected.sum() * RES ** 2),
                        "y_min_m": float(y[iy.min()]), "y_max_m": float(y[iy.max()]),
                        "z_min_m": float(z[iz.min()]), "z_max_m": float(z[iz.max()])})
    regions.sort(key=lambda item: item["mass"], reverse=True)
    return regions


def curvature(y, z, density, peak, bandwidth):
    iy, iz = peak["iy"], peak["iz"]
    if iy < 2 or iz < 2 or iy >= len(y) - 2 or iz >= len(z) - 2:
        return None
    logd = np.log(np.maximum(density, 1e-300))
    h = RES
    fyy = (logd[iy + 1, iz] - 2 * logd[iy, iz] + logd[iy - 1, iz]) / h ** 2
    fzz = (logd[iy, iz + 1] - 2 * logd[iy, iz] + logd[iy, iz - 1]) / h ** 2
    fyz = (logd[iy + 1, iz + 1] - logd[iy + 1, iz - 1] -
           logd[iy - 1, iz + 1] + logd[iy - 1, iz - 1]) / (4 * h ** 2)
    eigen = np.linalg.eigvalsh(np.array([[fyy, fyz], [fyz, fzz]]))
    radii = [float(1 / np.sqrt(-value)) if value < 0 else None for value in eigen]
    def halfwidth(axis):
        line = density[iy, iz:] if axis == "z" else density[iy:, iz]
        if axis == "z":
            vals = density[iy, iz:]
        else:
            vals = density[iy:, iz]
        out = {}
        for ratio in (0.8, 0.5):
            below = np.where(vals < ratio * density[iy, iz])[0]
            out[str(ratio)] = float(below[0] * RES) if len(below) else None
        return out
    return {"eigenvalues": [float(v) for v in eigen], "curvature_radii_m": radii,
            "profile_halfwidth_m": {"y": halfwidth("y"), "z": halfwidth("z")},
            "bandwidth_m": bandwidth}


def diagnostics_ok(result):
    summary = result.get("summary") or {}
    diag = result.get("diagnostics") or {}
    return bool(result.get("success")
                and diag.get("sampling_stable")
                and summary.get("effective_samples", 0) >= UNCERTAINTY.min_effective_samples
                and summary.get("unknown_event_mass", 1.0) <= UNCERTAINTY.unknown_mass_tolerance)


def analyze(records, output):
    result = {"seeds": list(SEEDS), "base_width_m": BASE_WIDTH,
              "widths_m": list(WIDTHS), "shifts_m": list(SHIFTS),
              "bandwidths_m": list(BANDWIDTHS), "prefixes": list(PREFIXES),
              "tolerance_m": TOL, "window": {"y": Y_RANGE, "z": Z_RANGE, "resolution_m": RES},
              "diagnostics": {}, "per_seed_kde": {}, "combined_kde": {},
              "stability": {}, "raw_grid": {}, "curvature": {}, "decision": {}}
    draws_by_seed = {}
    for record in records:
        seed = record["seed"]
        result["diagnostics"][str(seed)] = {
            "draws": record["draws"], "success": record["summary"].get("success"),
            "sampling_stable": (record["summary"].get("diagnostics") or {}).get("sampling_stable"),
            "effective_samples": (record["summary"].get("summary") or {}).get("effective_samples"),
            "unknown_event_mass": (record["summary"].get("summary") or {}).get("unknown_event_mass"),
            "gate_pass": diagnostics_ok(record["summary"]),
        }
        draws_by_seed[seed] = [json.loads(line) for line in
                               gzip.open(Path(output) / f"seed_{seed}" / "candidates.jsonl.gz", "rt", encoding="utf-8")]
    all_gate = all(item["gate_pass"] for item in result["diagnostics"].values())
    result["all_diagnostics_pass"] = all_gate
    kde_rows = []
    grid_rows = []
    for condition in ("direct_crossing", "inside_conditioned"):
        result["per_seed_kde"][condition] = {}
        result["combined_kde"][condition] = {}
        result["stability"][condition] = {}
        result["raw_grid"][condition] = {}
        seed_peaks = defaultdict(dict)
        seed_grid = defaultdict(dict)
        for seed in SEEDS:
            selected = select(draws_by_seed[seed], condition)
            if len(selected) < 10:
                continue
            for count in PREFIXES:
                prefix = select(draws_by_seed[seed][:count], condition)
                if len(prefix) < 10:
                    continue
                p, w = arrays(prefix, condition)
                for bandwidth in BANDWIDTHS:
                    y, z, density, mass = kde_window(p, w, bandwidth)
                    peak = peak_of(y, z, density, mass)
                    peak.update({"condition": condition, "seed": seed, "proposals": count,
                                 "bandwidth_m": bandwidth, "kind": "single"})
                    seed_peaks[count][(bandwidth, seed)] = peak
                    kde_rows.append(peak)
                for width in WIDTHS:
                    for sy in SHIFTS:
                        for sz in SHIFTS:
                            mode = grid_mode(p, w, CASE, width, shift=np.array((sy, sz)))
                            grid_rows.append({"condition": condition, "seed": seed,
                                              "proposals": count, "width_m": width,
                                              "shift_y_m": sy, "shift_z_m": sz,
                                              "mode_y_m": mode["center_yz_m"][0],
                                              "mode_z_m": mode["center_yz_m"][1],
                                              "cell_probability": mode["cell_probability"]})
                if count == 64000:
                    seed_grid[seed] = [row for row in grid_rows
                                       if row["condition"] == condition and row["seed"] == seed
                                       and row["proposals"] == 64000]
        # per-seed 64k and last doubling per bandwidth
        for bandwidth in BANDWIDTHS:
            full = []
            shifts = []
            for seed in SEEDS:
                a = seed_peaks.get(64000, {}).get((bandwidth, seed))
                b = seed_peaks.get(32000, {}).get((bandwidth, seed))
                if a:
                    full.append(a)
                if a and b:
                    shifts.append(float(np.hypot(a["y_m"] - b["y_m"], a["z_m"] - b["z_m"])))
            if full:
                pts = np.array([[p["y_m"], p["z_m"]] for p in full])
                diameter = float(np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)))
                result["stability"][condition][str(bandwidth)] = {
                    "cross_seed_diameter_m": diameter,
                    "max_last_doubling_shift_m": max(shifts) if shifts else None,
                    "pass": bool(all_gate and diameter <= TOL and (not shifts or max(shifts) <= TOL)),
                    "per_seed_full": [{"seed": p["seed"], "y_m": p["y_m"], "z_m": p["z_m"],
                                       "boundary_peak": p["boundary_peak"],
                                       "outside_mass": p["outside_mass"],
                                       "mass_in_window": p["mass_in_window"]} for p in full],
                }
        # combined: four seeds each normalized, equal weight
        for count in PREFIXES:
            parts_p, parts_w = [], []
            for seed in SEEDS:
                prefix = select(draws_by_seed[seed][:count], condition)
                if len(prefix) < 10:
                    continue
                p, w = arrays(prefix, condition)
                parts_p.append(p); parts_w.append(w / len(SEEDS))
            if not parts_p:
                continue
            p = np.vstack(parts_p); w = np.concatenate(parts_w); w /= w.sum()
            for bandwidth in BANDWIDTHS:
                y, z, density, mass = kde_window(p, w, bandwidth)
                peak = peak_of(y, z, density, mass)
                peak.update({"condition": condition, "seed": "combined", "proposals": count,
                             "bandwidth_m": bandwidth, "kind": "combined"})
                result["combined_kde"][condition][f"{count}_{bandwidth}"] = peak
                kde_rows.append(peak)
                if count == 64000 and abs(bandwidth - 0.10) < 1e-12:
                    peaks = local_peaks(y, z, density)
                    regions = high_density_regions(y, z, density)
                    result["curvature"][condition] = curvature(y, z, density, peak, bandwidth)
                    result["curvature"][condition]["competing_peaks"] = peaks
                    result["curvature"][condition]["high_density_regions"] = regions
        # raw grid summary
        for seed in SEEDS:
            rows = seed_grid.get(seed, [])
            if rows:
                pts = np.array([[r["mode_y_m"], r["mode_z_m"]] for r in rows])
                result["raw_grid"][condition][str(seed)] = {
                    "grid_diameter_m": float(np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1))),
                    "rows": rows,
                }
    for condition in ("direct_crossing", "inside_conditioned"):
        rows = [row for row in grid_rows if row["condition"] == condition and row["proposals"] == 64000]
        base_cross, within, last = [], [], []
        for seed in SEEDS:
            seed_rows = [row for row in rows if row["seed"] == seed]
            if not seed_rows:
                continue
            pts = np.array([[row["mode_y_m"], row["mode_z_m"]] for row in seed_rows])
            within.append(float(np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1))))
            base = [row for row in seed_rows if abs(row["width_m"] - BASE_WIDTH) < 1e-12
                    and row["shift_y_m"] == 0 and row["shift_z_m"] == 0]
            base32 = [row for row in grid_rows if row["condition"] == condition
                      and row["seed"] == seed and row["proposals"] == 32000
                      and abs(row["width_m"] - BASE_WIDTH) < 1e-12
                      and row["shift_y_m"] == 0 and row["shift_z_m"] == 0]
            if base:
                base_cross.append([base[0]["mode_y_m"], base[0]["mode_z_m"]])
            if base and base32:
                last.append(float(np.hypot(base[0]["mode_y_m"] - base32[0]["mode_y_m"],
                                           base[0]["mode_z_m"] - base32[0]["mode_z_m"])))
        cross = 0.0
        if len(base_cross) > 1:
            pts = np.array(base_cross)
            cross = float(np.max(np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)))
        result["raw_grid"][condition] = {
            "within_seed_grid_diameter_m": max(within) if within else None,
            "base_cross_seed_m": cross,
            "base_last_doubling_m": max(last) if last else None,
            "pass": bool(all_gate and within and max(within) <= TOL and cross <= TOL
                         and (not last or max(last) <= TOL)),
        }
        result["decision"][condition] = {
            "smooth_0.10_pass": result["stability"][condition].get("0.1", {}).get("pass", False),
            "raw_grid_pass": result["raw_grid"][condition]["pass"],
        }
    smooth_certified = bool(all_gate and result["stability"]["inside_conditioned"].get("0.1", {}).get("pass", False))
    result["decision"]["smooth_mode_certified"] = smooth_certified
    result["decision"]["raw_grid_certified"] = result["raw_grid"]["inside_conditioned"]["pass"]
    result["final_single_point"] = None
    if smooth_certified:
        peak = result["combined_kde"]["inside_conditioned"]["64000_0.1"]
        result["final_single_point"] = {
            "source": "combined equal-weight KDE, inside-conditioned, bandwidth 0.10 m, 64k",
            "y_m": peak["y_m"], "z_m": peak["z_m"],
            "mass_in_window": peak["mass_in_window"],
            "outside_mass": peak["outside_mass"],
            "boundary_peak": peak["boundary_peak"],
            "note": "Fixed-bandwidth smoothed peak; not a unique continuous mode.",
        }
    result["decision"]["note"] = ("mean is auxiliary, not the most likely point; "
                                   "raw-grid and smoothed-mode decisions are separate")
    with (output / "kde_peaks.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "seed", "proposals", "bandwidth_m",
                                                    "kind", "y_m", "z_m", "density",
                                                    "mass_in_window", "outside_mass", "boundary_peak"])
        writer.writeheader()
        for row in kde_rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    with (output / "grid_modes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid_rows[0]))
        writer.writeheader(); writer.writerows(grid_rows)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="F1 专用众数稳定性：64k × 4 seeds。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "f1_mode_stability")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在；保留旧结果，换一个目录。")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_seed, (seed, args.output)) for seed in SEEDS]
        for future in as_completed(futures):
            records.append(future.result())
    result = analyze(records, args.output)
    print(json.dumps({"output": str(args.output), "diagnostics": result["diagnostics"],
                      "stability": result["stability"], "combined": result["combined_kde"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
