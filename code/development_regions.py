"""开发集预测区域检验：先预测并封存，再读取 development 真值。

前 20 条观测只用于计时，配置不因试跑结果调整；复用其结果完成全开发组。
不打开校准/测试文件，不调先验、校准尺度或采样阈值。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import DATA_DIR, ENVIRONMENT, INFERENCE, PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR, SOLVER
from data import load_observations
from evaluate import (evaluate_predictions, generate_baseline_predictions, load_predictions,
                      point_prediction, read_jsonl, save_evaluation)
from uncertainty import UNCERTAINTY, uncertainty_observation


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_development(observations):
    if not observations or any(o["split"] != "development" for o in observations):
        raise ValueError("本入口只接受非空 development 观测，不能处理校准或测试组。")
    if len({o["case_id"] for o in observations}) != len(observations):
        raise ValueError("开发观测 case_id 重复。")


def predict_task(observation, index, model, output, *, base_samples=2000, starts=3,
                 seed=RANDOM_SEED, drag=INFERENCE.fixed_drag_coefficient):
    """子进程只取得一条公开观测与冻结设置，不取得 dataset 路径或真值。"""
    validate_development([observation])
    directory = Path(output)/f"case_{index:04d}_{model}"
    directory.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    options = replace(UNCERTAINTY, base_samples=base_samples)
    prediction, result = uncertainty_observation(observation, model, uncertainty=options,
                         seed=seed+index, starts=starts, drag_coefficient=drag)
    fit = result.get("fit")
    status = "unavailable" if not observation["observation_available"] else "failure"
    map_prediction = point_prediction(observation, f"{model}_cd{drag:g}_map",
        fit["simulation"] if fit and fit["success"] else None,
        status="ok" if fit and fit["success"] else status, point_kind="map_trajectory",
        message=fit["message"] if fit else result["message"])
    draws = result.get("draws", [])
    with gzip.open(directory/"candidates.jsonl.gz", "wt", encoding="utf-8") as file:
        for draw in draws:
            file.write(json.dumps(draw, ensure_ascii=False, allow_nan=False)+"\n")
    record = {"observation_index": index, "case_id": observation["case_id"], "source_id": observation["source_id"],
              "model": model, "seed": seed+index, "sigma_m": observation["measurement"]["sigma_m"],
              "prediction": prediction, "map_prediction": map_prediction,
              "draw_count": len(draws), "wall_time_s": time.perf_counter()-begun,
              "result": {k: v for k, v in result.items() if k != "draws"}}
    write_json(directory/"summary.json", record)
    return record


def assemble_predictions(observations, records, models):
    """完整性关口：缺任务、重复或错配不得在打开真值前被忽略。"""
    validate_development(observations)
    expected = {(i, model) for i in range(len(observations)) for model in models}
    lookup = {}
    for record in records:
        key = (record["observation_index"], record["model"])
        if key not in expected or key in lookup or record["case_id"] != observations[key[0]]["case_id"]:
            raise ValueError("预测任务重复、越界或与输入错配。")
        lookup[key] = record
    if set(lookup) != expected:
        raise ValueError("预测任务不完整，禁止读取真值评分。")
    predictions = generate_baseline_predictions(observations)
    for key in sorted(lookup):
        predictions.extend((lookup[key]["prediction"], lookup[key]["map_prediction"]))
    return predictions


def paired_region_comparison(rows, *, bootstrap_samples=1000, seed=RANDOM_SEED):
    """F-S：覆盖差以全部真实过线为分母；面积与点误差使用各自共同有效子集。"""
    methods = {row["method"] for row in rows if row["region_mass"] is not None}
    s_methods = sorted(m for m in methods if m.startswith("S_"))
    f_methods = sorted(m for m in methods if m.startswith("F_"))
    if not s_methods or not f_methods:
        return None
    if len(s_methods) != 1 or len(f_methods) != 1:
        raise ValueError("配对区域比较只接受一个 S 和一个 F 概率方法。")
    a, b = s_methods[0], f_methods[0]
    lookup = {(r["case_id"], r["method"]): r for r in rows}
    pairs = [(r, lookup[(r["case_id"], b)]) for r in rows if r["method"] == a]
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[0]["source_id"]].append(pair)

    def metrics(selected):
        direct = [(s, f) for s, f in selected if s["true_direct_goal_plane"]]
        available = [(s, f) for s, f in direct if s["observation_available"]]
        point = [(s, f) for s, f in direct if s["position_error_m"] is not None and f["position_error_m"] is not None]
        area = [(s, f) for s, f in direct if s["region_valid"] and f["region_valid"]]
        return {
            "coverage_F_minus_S": float(np.mean([int(f["covered"])-int(s["covered"]) for s, f in direct])) if direct else None,
            "available_coverage_F_minus_S": float(np.mean([int(f["covered"])-int(s["covered"]) for s, f in available])) if available else None,
            "median_point_error_F_minus_S_m": float(np.median([f["position_error_m"] for _, f in point])
                -np.median([s["position_error_m"] for s, _ in point])) if point else None,
            "p90_point_error_F_minus_S_m": float(np.quantile([f["position_error_m"] for _, f in point], .9)
                -np.quantile([s["position_error_m"] for s, _ in point], .9)) if point else None,
            "median_area_F_minus_S_m2": float(np.median([f["region_area_m2"] for _, f in area])
                -np.median([s["region_area_m2"] for s, _ in area])) if area else None,
            "failure_rate_F_minus_S": float(np.mean([int(f["status"] in ("failure", "missing"))
                -int(s["status"] in ("failure", "missing")) for s, f in selected])),
        }

    estimate = metrics(pairs)
    distribution = defaultdict(list)
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    for _ in range(bootstrap_samples):
        selected = [pair for i in rng.integers(0, len(keys), len(keys)) for pair in groups[keys[int(i)]]]
        for name, value in metrics(selected).items():
            if value is not None:
                distribution[name].append(value)
    intervals = {name: {"lower": float(np.quantile(values, .025)), "upper": float(np.quantile(values, .975)),
                        "valid_resamples": len(values)} for name, values in distribution.items()}
    common_points = [(s, f) for s, f in pairs if s["position_error_m"] is not None and f["position_error_m"] is not None]
    common_areas = [(s, f) for s, f in pairs if s["true_direct_goal_plane"] and s["region_valid"] and f["region_valid"]]
    return {"method_S": a, "method_F": b, "case_count": len(pairs), "source_count": len(keys),
            "true_direct_count": sum(s["true_direct_goal_plane"] for s, _ in pairs),
            "common_point_count": len(common_points), "common_point_sources": len({s["source_id"] for s, _ in common_points}),
            "common_region_count": len(common_areas), "common_region_sources": len({s["source_id"] for s, _ in common_areas}),
            "estimates": estimate, "cluster_bootstrap_95_intervals": intervals,
            "bootstrap_samples": bootstrap_samples,
            "note": "F-S；误差/面积/失败率负值利于 F，覆盖差正值利于 F。仅开发数据，不自动冻结模型。"}


def score_development(dataset, observations, predictions_path, expected_hash, output, *, bootstrap_samples=1000, seed=RANDOM_SEED):
    validate_development(observations)
    if file_hash(predictions_path) != expected_hash:
        raise ValueError("预测文件封存后发生变化，禁止读取真值评分。")
    predictions = load_predictions(predictions_path)
    # 真值仅在全部预测保存和校验之后由父进程打开；其余划分无读取路径。
    truth_path = Path(dataset)/"truth"/"development.jsonl"
    truths = read_jsonl(truth_path)
    report = evaluate_predictions(observations, truths, predictions, bootstrap_samples=bootstrap_samples, seed=seed)
    save_evaluation(report, output, predictions)
    comparison = paired_region_comparison(report["rows"], bootstrap_samples=bootstrap_samples, seed=seed+1)
    write_json(Path(output)/"paired_regions.json", comparison)
    write_json(Path(output)/"truth_provenance.json", {"path": str(truth_path.resolve()), "sha256": file_hash(truth_path),
        "opened_after_predictions_sha256": expected_hash, "split": "development"})
    return report, comparison


def plot_development(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(report["methods"])
    aliases = {m: (m.split("_")[0]+(" mean" if m.endswith("_posterior") else " MAP")) if "_" in m else m for m in methods}
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for offset, field, label in ((-.18, "median", "Median"), (.18, "p90", "90th percentile")):
        axes[0].bar(x+offset, [report["methods"][m]["position_error_m"][field] for m in methods], width=.36, label=label)
    axes[0].set(ylabel="Goal-plane error (m)", title="Own scoreable subsets; see paired comparisons")
    axes[0].legend()
    axes[1].bar(x, [report["methods"][m]["direct_point_prediction_rate"] for m in methods])
    axes[1].set(ylabel="Point provision on all true direct crossings", ylim=(0, 1.05), title="Effective prediction rate")
    for ax in axes:
        ax.set_xticks(x, [aliases[m] for m in methods])
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output/"point_comparison.png", dpi=180)
    plt.close(fig)

    region_methods = [m for m in methods if report["methods"][m]["region_mass"] is not None]
    xr = np.arange(len(region_methods))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    all_cov = [report["methods"][m]["coverage"]["rate"] for m in region_methods]
    avail = [report["methods"][m]["coverage_available_inputs"]["rate"] for m in region_methods]
    axes[0].bar(xr-.18, all_cov, width=.36, label="All true direct crossings")
    axes[0].bar(xr+.18, avail, width=.36, label="Available observations")
    for i, m in enumerate(region_methods):
        ci = report["methods"][m]["cluster_bootstrap_95_intervals"].get("coverage_rate")
        if ci:
            axes[0].vlines(i-.18, ci["lower"], ci["upper"], color="black", linewidth=1)
    axes[0].axhline(INFERENCE.prediction_region_mass, color="red", linestyle="--", label="Nominal 90%")
    axes[0].set(ylabel="Coverage fraction", ylim=(0, 1.05), title="Uncalibrated regions")
    axes[0].legend(fontsize=7)
    for i, m in enumerate(region_methods):
        values = [r["region_area_m2"] for r in report["rows"] if r["method"] == m and r["true_direct_goal_plane"] and r["region_valid"]]
        if values:
            axes[1].boxplot(values, positions=[i], widths=.5)
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Valid region area (m2, log scale)", title="True direct crossings with valid regions")
    bottom = np.zeros(len(region_methods))
    for status in ("ok", "failure", "unavailable", "missing"):
        values = [report["methods"][m]["status_counts"].get(status, 0) for m in region_methods]
        axes[2].bar(xr, values, bottom=bottom, label=status)
        bottom += values
    axes[2].set(ylabel="Observation cases", title="All cases retained")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(xr, [aliases[m] for m in region_methods])
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output/"coverage_area.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for m in region_methods:
        strata = report["strata"][m]["noise_sigma_m"]
        levels = sorted(strata, key=float)
        for ax, getter in zip(axes, (lambda s: s["coverage"]["rate"], lambda s: s["position_error_m"]["median"],
                                     lambda s: s["region_area_on_true_direct_m2"]["median"])):
            ax.plot([float(k) for k in levels], [getter(strata[k]) for k in levels], "o-", label=aliases[m])
    axes[0].axhline(.9, color="red", linestyle="--")
    axes[0].set(ylabel="Coverage on all true direct crossings", ylim=(0, 1.05))
    axes[1].set(ylabel="Own-subset median point error (m)")
    axes[2].set(ylabel="Valid region median area (m2)")
    for ax in axes:
        ax.set(xlabel="Observation sigma (m)")
        ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(output/"noise_strata.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="仅 development 的 S/F 预测区域检验，先计时再完成同配置全组。")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR/"synthetic_demo")
    parser.add_argument("--model", choices=("S", "F", "both"), default="both")
    parser.add_argument("--samples", type=int, default=UNCERTAINTY.base_samples, help="基础 N；每任务实际提议 2N")
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--drag", type=float, default=INFERENCE.fixed_drag_coefficient)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--pilot-cases", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR/"development_regions")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        if (args.samples < 2 or not 1 <= args.starts <= 9 or args.workers < 1 or args.pilot_cases < 1
                or args.bootstrap < 0 or args.seed < 0 or not np.isfinite(args.drag)
                or not INFERENCE.drag_bounds[0] <= args.drag <= INFERENCE.drag_bounds[1]):
            raise ValueError("样本/起点/进程/试跑数量、种子、重采样数或阻力无效。")
        if args.output.exists():
            raise ValueError("输出目录已存在，请指定新的 --output。")
        observation_path = args.dataset/"observations"/"development.jsonl"
        observations = load_observations(observation_path)
        validate_development(observations)
    except ValueError as error:
        parser.error(str(error))
    models = ("S", "F") if args.model == "both" else (args.model,)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/"cases").mkdir()
    started = time.perf_counter()
    pilot_count = min(args.pilot_cases, len(observations))
    config = {"created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "split": "development", "observation_path": str(observation_path.resolve()),
        "observations_sha256": file_hash(observation_path), "case_count": len(observations),
        "source_count": len({o["source_id"] for o in observations}), "models": models,
        "inference": asdict(INFERENCE), "default_environment": asdict(ENVIRONMENT), "solver": asdict(SOLVER),
        "uncertainty": asdict(replace(UNCERTAINTY, base_samples=args.samples)), "drag": args.drag,
        "starts": args.starts, "seed_rule": "base seed + observation file index, same for S/F",
        "base_seed": args.seed, "workers": args.workers, "bootstrap_samples": args.bootstrap,
        "pilot_case_count": pilot_count, "pilot_selection": "first cases in development observation file; timing only",
        "pilot_source_count": len({o["source_id"] for o in observations[:pilot_count]}),
        "versions": {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in ("numpy", "scipy", "matplotlib")}},
        "source_sha256": {name: file_hash(PROJECT_ROOT/"code"/name) for name in
            ("development_regions.py", "config.py", "uncertainty.py", "inverse.py", "physics.py", "data.py", "evaluate.py")},
        "protocol": "No tuning/retries based on pilot scores; same fixed configuration for full development. "
            "No truth opened until all predictions are sealed. Failures/unavailable retained; score conditional means, "
            "uncalibrated nominal regions, MAP/G/A controls. No model freezing, calibration or test access."}
    write_json(output/"run_config.json", config)
    records = []
    options = dict(base_samples=args.samples, starts=args.starts, seed=args.seed, drag=args.drag)

    def run_stage(pool, indices, stage):
        pending = {pool.submit(predict_task, observations[i], i, model, output/"cases", **options): (i, model)
                   for i in indices for model in models}
        total = len(pending)
        for count, future in enumerate(as_completed(pending), 1):
            record = future.result()
            records.append(record)
            with (output/"task_log.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps({"stage": stage, "index": record["observation_index"], "model": record["model"],
                    "case_id": record["case_id"], "status": record["prediction"]["status"],
                    "wall_time_s": record["wall_time_s"], "draw_count": record["draw_count"]}, allow_nan=False)+"\n")
            if count % 5 == 0 or count == total:
                print(f"{stage}: {count}/{total} tasks; latest {record['model']} {record['case_id']}", file=sys.stderr, flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pilot_start = time.perf_counter()
        run_stage(pool, range(pilot_count), "pilot")
        elapsed = time.perf_counter()-pilot_start
        timing = {"observation_count": pilot_count, "source_count": config["pilot_source_count"],
                  "task_count": len(records), "workers": args.workers, "wall_time_s": elapsed,
                  "mean_task_compute_time_s": float(np.mean([r["wall_time_s"] for r in records])),
                  "remaining_task_count": (len(observations)-pilot_count)*len(models),
                  "estimated_remaining_wall_time_s": elapsed/pilot_count*(len(observations)-pilot_count),
                  "note": "计时包含已列入试跑的所有观测与各模型；含相关噪声副本，非独立样本数。仅粗略耗时外推，未打开真值。"}
        write_json(output/"pilot_timing.json", timing)
        print(f"Pilot completed in {elapsed:.1f}s; estimated remaining {timing['estimated_remaining_wall_time_s']:.1f}s. Continuing frozen settings.",
              file=sys.stderr, flush=True)
        run_stage(pool, range(pilot_count, len(observations)), "development")
    predictions = assemble_predictions(observations, records, models)
    prediction_path = output/"predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(p, ensure_ascii=False, allow_nan=False)+"\n" for p in predictions), encoding="utf-8")
    seal = {"prediction_count": len(predictions), "task_count": len(records), "sha256": file_hash(prediction_path),
            "sealed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "observations_sha256": file_hash(observation_path), "note": "All predictions saved before reading development truth."}
    if seal["observations_sha256"] != config["observations_sha256"]:
        raise RuntimeError("观测文件在预测期间发生变化，停止评分。")
    write_json(output/"prediction_manifest.json", seal)
    print("All predictions sealed. Opening development truth for evaluation.", file=sys.stderr, flush=True)
    report, comparison = score_development(args.dataset, observations, prediction_path, seal["sha256"], output/"evaluation",
                                           bootstrap_samples=args.bootstrap, seed=args.seed+100000)
    if args.plot:
        plot_development(report, output/"evaluation")
    summary = {"status": "completed", "split": "development", "case_count": len(observations),
        "source_count": config["source_count"], "task_count": len(records), "prediction_count": len(predictions),
        "posterior_status_counts": {m: dict(Counter(r["prediction"]["status"] for r in records if r["model"] == m)) for m in models},
        "actual_proposal_count": sum(r["draw_count"] for r in records), "wall_time_s": time.perf_counter()-started,
        "paired_regions": comparison,
        "note": "开发示例的未校准区域检验，不是独立测试或实测性能；众数未稳定标记保留，评分点为条件均值。"}
    write_json(output/"run_summary.json", summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
