"""Reproduce current F1 case-mode stability, retaining samples and publication figures.

Uses the existing stability experiment unchanged; FINAL settings are checked first.
--analyze-only regenerates figures/tables from this run's saved raw samples.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from config import CASE, ENVIRONMENT, PROJECT_ROOT, SOLVER
from final_config import FINAL
import f1_mode_stability as experiment


DEFAULT_OUTPUT = PROJECT_ROOT/"results"/"算例稳定性实验"/"重新运行_20261108"


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def check_settings():
    expected, actual = FINAL.inference_settings(), experiment.F1_SETTINGS
    def effective(settings):
        d = asdict(settings)
        for name in ("vertical_force_prior_std", "side_force_prior_std"):
            d[name] = d[name] if d[name] is not None else d["transverse_prior_std"]
        return d
    if effective(expected) != effective(actual):
        raise ValueError("Stability experiment's effective prior differs from frozen FINAL.")
    if FINAL.drag_coefficient != .22 or FINAL.sigma_m != .01 or FINAL.mode_bandwidth_m != .10:
        raise ValueError("Existing experiment constants differ from frozen FINAL.")
    if len(experiment.SEEDS) != 4 or experiment.OPTIONS.base_samples*2 != 64000:
        raise ValueError("Expected four seeds, 64000 proposals each.")


def load_draws(output, seed):
    with gzip.open(output/f"seed_{seed}"/"candidates.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def records_from_disk(output):
    records = []
    for seed in experiment.SEEDS:
        summary = json.loads((output/f"seed_{seed}"/"summary.json").read_text())
        draws = load_draws(output, seed)
        if len(draws) != 64000:
            raise ValueError(f"Seed {seed}: expected 64000 candidates, got {len(draws)}")
        records.append({"seed": seed, "draws": len(draws), "summary": summary})
    return records


def plot_tables(output, summary):
    with (output/"kde_peaks.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if r["condition"] == "inside_conditioned"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    combined = summary["combined_kde"]["inside_conditioned"]["64000_0.1"]
    for seed in experiment.SEEDS:
        rr = [r for r in rows if r["seed"] == str(seed) and float(r["bandwidth_m"]) == .1]
        rr.sort(key=lambda r: int(r["proposals"]))
        n = [int(r["proposals"]) for r in rr]
        axes[0].plot(n, [float(r["y_m"]) for r in rr], "o-", label=str(seed))
        axes[1].plot(n, [float(r["z_m"]) for r in rr], "o-", label=str(seed))
    for ax, coordinate in zip(axes, ("y", "z")):
        ax.axhline(combined[f"{coordinate}_m"], color="black", linestyle="--", label="Combined 64k peak")
        ax.set(xlabel="Proposals per seed", ylabel=f"Peak {coordinate.upper()} (m)", xscale="log",
               title="Inside-conditioned, bandwidth 0.10 m")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(output/"sampling_convergence.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for seed in experiment.SEEDS:
        rr = [r for r in rows if r["seed"] == str(seed) and int(r["proposals"]) == 64000]
        rr.sort(key=lambda r: float(r["bandwidth_m"]))
        for ax, coordinate in zip(axes, ("y", "z")):
            ax.plot([float(r["bandwidth_m"]) for r in rr], [float(r[f"{coordinate}_m"]) for r in rr], "o-", label=str(seed))
    for ax, coordinate in zip(axes, ("y", "z")):
        rr = [r for r in rows if r["kind"] == "combined" and int(r["proposals"]) == 64000]
        rr.sort(key=lambda r: float(r["bandwidth_m"]))
        ax.plot([float(r["bandwidth_m"]) for r in rr], [float(r[f"{coordinate}_m"]) for r in rr], "k*--", label="Combined")
        ax.set(xlabel="Fixed bandwidth (m)", ylabel=f"Peak {coordinate.upper()} (m)", title="64k proposals per seed")
        ax.set_xticks(experiment.BANDWIDTHS)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(output/"bandwidth_sensitivity.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for seed in experiment.SEEDS:
        rr = [r for r in rows if r["seed"] == str(seed) and int(r["proposals"]) == 64000 and float(r["bandwidth_m"]) == .1]
        ax.scatter([float(r["y_m"]) for r in rr], [float(r["z_m"]) for r in rr], s=75, label=str(seed))
    ax.scatter(combined["y_m"], combined["z_m"], marker="*", c="black", s=150, label="Combined")
    ax.set(xlabel="Peak Y (m)", ylabel="Peak Z (m)", title="Cross-seed peaks: 64k, bandwidth 0.10 m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(output/"cross_seed_peaks.png", dpi=180)
    plt.close(fig)


def plot_density(output, summary):
    # Each seed is normalized within its event condition before equal mixing.
    parts = {c: [[], []] for c in ("direct_crossing", "inside_conditioned")}
    for seed in experiment.SEEDS:
        draws = load_draws(output, seed)
        for condition, (points, weights) in parts.items():
            p, w = experiment.arrays(experiment.select(draws, condition), condition)
            points.append(p)
            weights.append(w/len(experiment.SEEDS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    arrays_to_save = {}
    for ax, (condition, (points, weights)) in zip(axes, parts.items()):
        p, w = np.vstack(points), np.concatenate(weights)
        y, z, density, mass = experiment.kde_window(p, w, .10)
        peak = experiment.peak_of(y, z, density, mass)
        saved = summary["combined_kde"][condition]["64000_0.1"]
        if not np.allclose([peak["y_m"], peak["z_m"]], [saved["y_m"], saved["z_m"]], atol=1e-12):
            raise ValueError("Density recomputation disagrees with recorded combined peak")
        mesh = ax.pcolormesh(y, z, density.T, shading="auto", cmap="viridis")
        ax.contour(y, z, density.T, levels=[.5*density.max(), .8*density.max()], colors="white", linewidths=.8)
        ax.plot(peak["y_m"], peak["z_m"], "r*", markersize=12, label="Smoothed peak")
        ax.axvline(CASE.goal_y_bounds_m[1], color="orange", linestyle="--", label="Goal boundary")
        ax.axhline(CASE.goal_z_bounds_m[1], color="orange", linestyle="--")
        ax.set(xlabel="Y (m)", ylabel="Z (m)", title=condition.replace("_", " ")+"; bandwidth 0.10 m")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower left")
        fig.colorbar(mesh, ax=ax, label="Smoothed density (1/m²)", shrink=.8)
        arrays_to_save[condition+"_density"] = density
    arrays_to_save.update(y_m=y, z_m=z)
    np.savez_compressed(output/"kde_density.npz", **arrays_to_save)
    fig.savefig(output/"kde_density.png", dpi=180)
    plt.close(fig)


def report(output, summary, wall_s):
    stability = summary["stability"]["inside_conditioned"]
    combined = summary["combined_kde"]["inside_conditioned"]
    old_path = output.parent/"summary.json"
    previous = json.loads(old_path.read_text()) if old_path.exists() else {}
    lines = ["# 当前 F1 算例稳定性实验：重新采样与完整留档", "", "日期：2026-11-08（Asia/Shanghai）。", "",
             "本次按当前冻结F1重新生成候选；模型、先验、Cd、误差假设、带宽与判据均未调优。",
             "仅使用题目指定的两帧坐标，不访问开发、校准、测试真值。", "",
             "F1先验by/bz尺度0.10/0.20，Cd=0.22，sigma=0.01m。四个预先指定种子，每种子64k提议。",
             "主带宽0.10m；0.05/0.20m仅作敏感性；门槛沿用0.20m。direct crossing和inside-conditioned分开。",
             "所有原始提议均保留，包括零权重、拒绝和未决候选；每种子条件归一化后等权合并。",
             "搜索窗口与原实验一致；不因窗口截断重新归一化。密度图为固定带宽平滑估计，边界附近会有核平滑越界。", "",
             "## 采样诊断", "", "| 种子 | 提议数 | ESS | 未决质量 | 通过 |", "|---|---:|---:|---:|---|"]
    for seed, d in summary["diagnostics"].items():
        lines.append(f"| {seed} | {d['draws']} | {d['effective_samples']:.1f} | {d['unknown_event_mass']:.3g} | {d['gate_pass']} |")
    lines += ["", "## 主点与带宽敏感性", "", "| 带宽/m | 合并峰(Y,Z)/m | 跨种子直径/m | 32k→64k最大移动/m | 通过 |",
              "|---|---|---:|---:|---|"]
    for b in experiment.BANDWIDTHS:
        s = stability[str(b)]
        p = combined[f"64000_{b}"]
        lines.append(f"| {b:.2f} | ({p['y_m']:.3f}, {p['z_m']:.3f}) | {s['cross_seed_diameter_m']:.4f} | {s['max_last_doubling_shift_m']:.4f} | {s['pass']} |")
    lines += ["", "## 原始格网对照", "", "```json", json.dumps(summary["raw_grid"]["inside_conditioned"], ensure_ascii=False, indent=2), "```", "",
              "## 与旧结果比较", ""]
    old_stab = previous.get("stability", {}).get("inside_conditioned", {}).get("0.1")
    if old_stab:
        now = stability["0.1"]
        lines.append(f"旧主带宽跨种子直径 {old_stab['cross_seed_diameter_m']:.4f}m，新值 {now['cross_seed_diameter_m']:.4f}m；旧加倍移动 {old_stab['max_last_doubling_shift_m']:.4f}m，新值 {now['max_last_doubling_shift_m']:.4f}m。旧记录保留在上级目录。")
    else:
        lines.append("未找到上级目录旧汇总，不作数值比较。")
    p = combined["64000_0.1"]
    curve = summary["curvature"]["inside_conditioned"]
    lines += ["", "## 结论与范围", "",
              f"- 主带宽合并峰为({p['y_m']:.3f}, {p['z_m']:.3f})m；稳定性通过：{stability['0.1']['pass']}。",
              f"- 峰是否位于搜索边缘：{p['boundary_peak']}；平滑密度窗口外质量约{p['outside_mass']:.3g}。",
              f"- 原有阈值下检测竞争峰数：{len(curve['competing_peaks'])}；高密度连通区域数：{len(curve['high_density_regions'])}。未发现竞争峰不能证明连续密度全局唯一。",
              "- 固定随机种子与设置用于复现实验，不通过挑种子或带宽追求更好结果。",
              "- 支持的是指定算例、当前模型和固定平滑尺度下的数值稳定性；不证明真实预测准确，也不证明其他射门均稳定。",
              "- 4k/8k/16k行是同一64k固定提议分布的前缀诊断，不等于分别以不同预算重新构造提议运行。",
              f"- 本次采样、分析与绘图耗时约{wall_s:.1f}s。", "",
              "## 图表", "", "![采样量稳定性](sampling_convergence.png)", "", "![跨种子峰位置](cross_seed_peaks.png)", "",
              "![带宽敏感性](bandwidth_sensitivity.png)", "", "![两种条件密度](kde_density.png)", "",
              "## 文件与复现", "",
              "seed_*/candidates.jsonl.gz为逐条原始候选；seed_*/summary.json为各次诊断；kde_peaks.csv/grid_modes.csv为完整统计；kde_density.npz为密度图数值。",
              "run_config.json保存输入及源码指纹；source/保留源码；artifact_manifest.json校验所有文件。旧结果未覆盖。", "",
              "从项目根目录运行（新目录必须不存在）：", "", "```bash",
              ".venv/bin/python code/final_mode_stability.py --output results/算例稳定性实验/再次运行 --workers 4", "```", "",
              "仅从已保存候选重新分析与绘图：", "", "```bash",
              f".venv/bin/python code/final_mode_stability.py --output '{output.relative_to(PROJECT_ROOT)}' --analyze-only", "```"]
    (output/"STABILITY_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(output, workers=4, analyze_only=False):
    check_settings()
    started = time.perf_counter()
    if not analyze_only:
        output.mkdir(parents=True, exist_ok=False)
        (output/"source").mkdir()
        names = ("final_mode_stability.py", "f1_mode_stability.py", "uncertainty.py", "inverse.py", "physics.py", "config.py", "final_config.py", "data.py", "evaluate.py")
        hashes = {}
        for name in names:
            p = Path(__file__).with_name(name)
            shutil.copy2(p, output/"source"/name)
            hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        save_json(output/"run_config.json", {
            "date": "2026-11-08 Asia/Shanghai", "case": asdict(CASE), "environment": asdict(ENVIRONMENT),
            "solver": asdict(SOLVER), "inference": asdict(FINAL.inference_settings()), "uncertainty": asdict(experiment.OPTIONS),
            "seeds": list(experiment.SEEDS), "prefixes": list(experiment.PREFIXES), "bandwidths_m": list(experiment.BANDWIDTHS),
            "sigma_m": FINAL.sigma_m, "drag_coefficient": FINAL.drag_coefficient, "tolerance_m": experiment.TOL,
            "window": {"y": experiment.Y_RANGE, "z": experiment.Z_RANGE, "resolution_m": experiment.RES},
            "source_sha256": hashes, "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
            "workers": workers, "note": "Current F1 settings checked; original experiment method and thresholds retained."})
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(experiment.run_seed, (seed, output)) for seed in experiment.SEEDS]
            for future in as_completed(futures):
                record = future.result()
                print(f"Seed {record['seed']}: {record['draws']} raw candidates saved", flush=True)
    else:
        cfg = json.loads((output/"run_config.json").read_text())
        for name, digest in cfg["source_sha256"].items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed since sampling: {name}; use the saved source snapshot.")
    records = records_from_disk(output)
    summary = experiment.analyze(records, output)
    plot_tables(output, summary)
    plot_density(output, summary)
    elapsed = time.perf_counter()-started
    report(output, summary, elapsed)
    save_json(output/"execution.json", {"elapsed_s": elapsed, "analyze_only": analyze_only,
                                      "candidate_count": sum(r["draws"] for r in records), "all_diagnostics_pass": summary["all_diagnostics_pass"]})
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    save_json(output/"artifact_manifest.json", hashes)
    print(json.dumps({"output": str(output), "elapsed_s": elapsed, "decision": summary["decision"], "peak": summary["final_single_point"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.workers, args.analyze_only)
