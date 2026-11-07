"""C 检查：复用现有众数候选，分开数值抖动、假设位移与高密度区。

不生成新样本、不改变 B 预注册门槛。固定格网、KDE 带宽和竞争峰定义
见 docs/superpowers/specs/2026-11-07-round2-prereg.md 第 4 节。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, maximum_filter

from config import CASE, PROJECT_ROOT
from uncertainty import grid_mode

BASE_WIDTH = 2.44 / 24
WIDTHS = (BASE_WIDTH / 2, BASE_WIDTH, BASE_WIDTH * 2)
SHIFTS = (0.0, BASE_WIDTH / 2)
KDE_BANDWIDTHS = (0.10, 0.05, 0.20)
Y_RANGE = (1.5, 4.5)
Z_RANGE = (0.5, 3.0)
KDE_RESOLUTION = 0.01
PEAK_RATIO = 0.8
PEAK_SEPARATION = 0.10
PREFIXES = (4000, 8000, 16000)


def read_candidates(path):
    draws = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["event"] in ("goal_inside", "goal_outside") and record["weight"] > 0:
                draws.append(record)
    if not draws:
        raise ValueError(f"没有可用的直接过线候选: {path}")
    return draws


def weighted_points(draws):
    points = np.asarray([d["goal_yz_m"] for d in draws], float)
    weights = np.asarray([d["weight"] for d in draws], float)
    weights = weights / weights.sum()
    return points, weights


def prefix(draws, count):
    selected = [d for d in draws[:count] if d["event"] in ("goal_inside", "goal_outside") and d["weight"] > 0]
    if not selected:
        raise ValueError(f"前缀 {count} 没有正权重直接过线候选")
    return selected


def grid_variants(draws):
    """12 种格网：3 格宽 × 4 平移。"""
    points, weights = weighted_points(draws)
    rows = []
    for width in WIDTHS:
        for shift_y in SHIFTS:
            for shift_z in SHIFTS:
                mode = grid_mode(points, weights, CASE, width,
                                 shift=np.array((shift_y, shift_z)))
                rows.append({"grid_width_m": width, "shift_y_m": shift_y,
                             "shift_z_m": shift_z, **mode})
    return rows


def grid_diameter(rows):
    centers = np.asarray([r["center_yz_m"] for r in rows], float)
    if len(centers) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)))


def kde(points, weights, bandwidth):
    """加权二维各向同性高斯核，返回归一化密度和网格。"""
    y_edges = np.arange(Y_RANGE[0], Y_RANGE[1] + KDE_RESOLUTION / 2, KDE_RESOLUTION)
    z_edges = np.arange(Z_RANGE[0], Z_RANGE[1] + KDE_RESOLUTION / 2, KDE_RESOLUTION)
    hist, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=(y_edges, z_edges), weights=weights)
    density = gaussian_filter(hist, sigma=bandwidth / KDE_RESOLUTION, mode="constant", truncate=4.0)
    cell_area = KDE_RESOLUTION ** 2
    total = float(density.sum() * cell_area)
    if total <= 0 or not np.isfinite(total):
        raise ValueError("KDE 密度无效")
    density = density / total
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    return y_centers, z_centers, density


def local_peaks(y, z, density, ratio=PEAK_RATIO, separation=PEAK_SEPARATION):
    """按固定阈值与间距筛选竞争局部峰。"""
    size = int(round(separation / KDE_RESOLUTION)) * 2 + 1
    local_max = maximum_filter(density, size=size, mode="nearest")
    peak_mask = (density == local_max) & (density >= ratio * density.max())
    iy, iz = np.nonzero(peak_mask)
    order = np.argsort(density[iy, iz])[::-1]
    peaks = []
    cell_area = KDE_RESOLUTION ** 2
    for index in order:
        py, pz, value = float(y[iy[index]]), float(z[iz[index]]), float(density[iy[index], iz[index]])
        if py <= y[0] + separation or py >= y[-1] - separation:
            continue
        if pz <= z[0] + separation or pz >= z[-1] - separation:
            continue
        if any(np.hypot(py - p["y_m"], pz - p["z_m"]) < separation for p in peaks):
            continue
        mask = (y[:, None] - py) ** 2 + (z[None, :] - pz) ** 2 <= separation ** 2
        peaks.append({"y_m": py, "z_m": pz, "density": value,
                      "fraction_of_max": value / float(density.max()),
                      "mass_within_separation": float(density[mask].sum() * cell_area)})
    return peaks


def high_density_regions(y, z, density, ratio=PEAK_RATIO):
    """密度不低于全局最大值 ratio 的水平集连通部分。"""
    mask = density >= ratio * density.max()
    labels, count = label(mask, structure=np.ones((3, 3), int))
    cell_area = KDE_RESOLUTION ** 2
    regions = []
    for component in range(1, count + 1):
        selected = labels == component
        mass = float(density[selected].sum() * cell_area)
        iy, iz = np.nonzero(selected)
        peak_index = np.argmax(density[selected])
        regions.append({"component": component, "mass": mass,
                        "area_m2": float(selected.sum() * cell_area),
                        "y_min_m": float(y[iy.min()]), "y_max_m": float(y[iy.max()]),
                        "z_min_m": float(z[iz.min()]), "z_max_m": float(z[iz.max()]),
                        "peak_y_m": float(y[iy[peak_index]]), "peak_z_m": float(z[iz[peak_index]]),
                        "peak_density": float(density[selected].max())})
    regions.sort(key=lambda item: item["mass"], reverse=True)
    return regions


def analyze(draws, model, seed):
    points, weights = weighted_points(draws)
    result = {"model": model, "seed": seed, "count": len(draws)}
    result["grids"] = grid_variants(draws)
    result["grid_diameter_m"] = grid_diameter(result["grids"])
    base = [r for r in result["grids"] if abs(r["grid_width_m"] - BASE_WIDTH) < 1e-12
            and r["shift_y_m"] == 0 and r["shift_z_m"] == 0][0]
    result["base_mode"] = base
    result["prefix_modes"] = {}
    for count in PREFIXES:
        selected = prefix(draws, count)
        points_n, weights_n = weighted_points(selected)
        result["prefix_modes"][str(count)] = grid_mode(points_n, weights_n, CASE, BASE_WIDTH)
    result["last_doubling_shift_m"] = float(np.linalg.norm(
        np.array(result["prefix_modes"]["16000"]["center_yz_m"])
        - np.array(result["prefix_modes"]["8000"]["center_yz_m"])))
    result["prefix_4000_to_8000_m"] = float(np.linalg.norm(
        np.array(result["prefix_modes"]["8000"]["center_yz_m"])
        - np.array(result["prefix_modes"]["4000"]["center_yz_m"])))
    result["kde"] = {}
    for bandwidth in KDE_BANDWIDTHS:
        y, z, density = kde(points, weights, bandwidth)
        peaks = local_peaks(y, z, density)
        regions = high_density_regions(y, z, density)
        result["kde"][str(bandwidth)] = {
            "bandwidth_m": bandwidth,
            "global_peak": {"y_m": float(y[np.unravel_index(np.argmax(density), density.shape)[0]]),
                            "z_m": float(z[np.unravel_index(np.argmax(density), density.shape)[1]]),
                            "density": float(density.max())},
            "peaks": peaks,
            "regions": regions,
            "mass_in_window": float(density.sum() * KDE_RESOLUTION ** 2),
        }
    result["points"] = points
    result["weights"] = weights
    return result


def combined_results(results):
    points = np.concatenate([r["points"] for r in results], axis=0)
    weights = np.concatenate([r["weights"] / len(results) for r in results], axis=0)
    weights = weights / weights.sum()
    output = {"model": results[0]["model"], "seed": "combined", "count": len(points)}
    output["grids"] = []
    for width in WIDTHS:
        for shift_y in SHIFTS:
            for shift_z in SHIFTS:
                output["grids"].append({"grid_width_m": width, "shift_y_m": shift_y,
                                        "shift_z_m": shift_z,
                                        **grid_mode(points, weights, CASE, width,
                                                    shift=np.array((shift_y, shift_z)))})
    output["grid_diameter_m"] = grid_diameter(output["grids"])
    output["base_mode"] = [r for r in output["grids"] if abs(r["grid_width_m"] - BASE_WIDTH) < 1e-12
                           and r["shift_y_m"] == 0 and r["shift_z_m"] == 0][0]
    output["kde"] = {}
    for bandwidth in KDE_BANDWIDTHS:
        y, z, density = kde(points, weights, bandwidth)
        output["kde"][str(bandwidth)] = {
            "bandwidth_m": bandwidth,
            "global_peak": {"y_m": float(y[np.unravel_index(np.argmax(density), density.shape)[0]]),
                            "z_m": float(z[np.unravel_index(np.argmax(density), density.shape)[1]]),
                            "density": float(density.max())},
            "peaks": local_peaks(y, z, density),
            "regions": high_density_regions(y, z, density),
            "mass_in_window": float(density.sum() * KDE_RESOLUTION ** 2),
        }
    return output


def sweep(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: sweep(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sweep(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description="复用现有众数候选做 C 检查；不生成新样本。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "mode_cause_check")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在；保留旧结果，换一个目录。")
    root = PROJECT_ROOT / "results" / "mode_stability_example"
    models = {}
    for model in ("F", "S"):
        paths = sorted(root.glob(f"{model}_nominal_seed*/candidates.jsonl.gz"))
        if not paths:
            raise FileNotFoundError(f"缺少 {model} 名义候选")
        results = [analyze(read_candidates(path), model, path.parent.name.split("seed")[-1])
                   for path in paths]
        results.append(combined_results(results))
        models[model] = results
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"settings": {"base_width_m": BASE_WIDTH, "widths_m": list(WIDTHS),
                            "shifts_m": list(SHIFTS), "kde_bandwidths_m": list(KDE_BANDWIDTHS),
                            "y_range_m": Y_RANGE, "z_range_m": Z_RANGE,
                            "kde_resolution_m": KDE_RESOLUTION, "peak_ratio": PEAK_RATIO,
                            "peak_separation_m": PEAK_SEPARATION, "prefixes": list(PREFIXES)},
               "models": {}}
    for model, results in models.items():
        summary["models"][model] = []
        for result in results:
            summary["models"][model].append({k: sweep(v) for k, v in result.items()
                                             if k not in ("points", "weights")})
    (args.output / "summary.json").write_text(
        json.dumps(sweep(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    with (args.output / "mode_cells.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "grid_width_m", "shift_y_m", "shift_z_m",
                         "mode_y_m", "mode_z_m", "cell_probability"])
        for model, results in models.items():
            for result in results:
                for row in result["grids"]:
                    writer.writerow([model, result["seed"], f"{row['grid_width_m']:.9f}",
                                     f"{row['shift_y_m']:.9f}", f"{row['shift_z_m']:.9f}",
                                     f"{row['center_yz_m'][0]:.6f}", f"{row['center_yz_m'][1]:.6f}",
                                     f"{row['cell_probability']:.10f}"])

    with (args.output / "kde_peaks.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "bandwidth_m", "rank", "peak_y_m", "peak_z_m",
                         "fraction_of_max", "mass_within_separation"])
        for model, results in models.items():
            for result in results:
                for bandwidth, payload in result["kde"].items():
                    for rank, peak in enumerate(payload["peaks"], start=1):
                        writer.writerow([model, result["seed"], f"{float(bandwidth):.4f}", rank,
                                         f"{peak['y_m']:.4f}", f"{peak['z_m']:.4f}",
                                         f"{peak['fraction_of_max']:.4f}",
                                         f"{peak['mass_within_separation']:.6f}"])
    print(json.dumps({"output": str(args.output), "models": list(models),
                      "runs": sum(len(v) for v in models.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
