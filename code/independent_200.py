"""独立测试：200 源，100 calibration / 100 test；F1 固定 Cd。

空间区域在 calibration 上做 split-conformal 阈值校准；
时间使用已冻结的全局 additive q；test 只评价一次。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT
from data import build_dataset, load_observations, save_dataset
from development_regions import file_hash, write_json
from model_diagnosis import write_jsonl
from round2_selection import METHODS, VARIANTS, run, stable_seed

DATA_SEED = 20261110
METHOD = "F1_round2"
TIME_Q = 0.015188910308210842


def prepare(dataset, output):
    if dataset.exists() or output.exists():
        raise FileExistsError("数据或结果目录已存在；保留旧结果。")
    built = build_dataset(200, seed=DATA_SEED, split_fractions=(1, 0, 0))
    by_mechanism = defaultdict(list)
    for truth in built["truth"]:
        by_mechanism[truth["parameters"]["mechanism"]].append(truth["source_id"])
    split_map = {}
    for mechanism, ids in sorted(by_mechanism.items()):
        ids = sorted(ids)
        if len(ids) != 40:
            raise ValueError(f"{mechanism} 源数不是 40")
        for source in ids[:20]:
            split_map[source] = "calibration"
        for source in ids[20:]:
            split_map[source] = "test"
    for observation in built["observations"]:
        observation["split"] = split_map[observation["source_id"]]
    for truth in built["truth"]:
        truth["split"] = split_map[truth["source_id"]]
    built["manifest"]["source_counts_by_split"] = {"development": 0, "calibration": 100, "test": 100}
    save_dataset(built, dataset)
    observations = load_observations(dataset / "observations" / "calibration.jsonl") + \
                   load_observations(dataset / "observations" / "test.jsonl")
    tasks = [{"index": i, "case_id": row["case_id"], "source_id": row["source_id"],
              "variant": "F1", "method": METHODS["F1"],
              "seed": stable_seed(row["case_id"])} for i, row in enumerate(observations)]
    if len(observations) != 800 or len(tasks) != 800:
        raise ValueError("观测/任务数不是 800")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "observations.jsonl", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "run_config.json", {
        "dataset": str(dataset), "data_seed": DATA_SEED, "method": METHOD,
        "task_count": len(tasks), "variants": VARIANTS,
        "split": {"calibration": 100, "test": 100},
        "time_q_s": TIME_Q, "notes": "Independent test; time q frozen; spatial calibration only on calibration split.",
    })
    print(json.dumps({"dataset": str(dataset), "observations": len(observations),
                      "tasks": len(tasks)}, ensure_ascii=False))


def conformal_threshold(values, mass=0.90):
    values = np.sort(np.asarray(values, float))
    n = len(values)
    rank = int(np.ceil((n + 1) * mass)) - 1
    rank = min(max(rank, 0), n - 1)
    return float(values[rank]), rank, n


def region_r2(region, truth_yz):
    center = np.asarray(region["center_yz_m"], float)
    cov = np.asarray(region["covariance_m2"], float)
    delta = np.asarray(truth_yz, float) - center
    return float(delta @ np.linalg.inv(cov) @ delta)


def evaluate(records):
    calibration = [r for r in records if r["split"] == "calibration"]
    test = [r for r in records if r["split"] == "test"]
    r2_cal = [r["r2"] for r in calibration if np.isfinite(r["r2"])]
    q_cal, rank, n_cal = conformal_threshold(r2_cal, 0.90)
    def summary(rows, threshold=None):
        direct = [r for r in rows if r["direct"]]
        covered = [r["r2"] <= (r["threshold"] if threshold is None else threshold) for r in direct]
        if threshold is None:
            areas = [r["area"] for r in direct if r["area"] is not None]
        else:
            areas = [float(np.pi * threshold * r["sqrt_det"]) for r in direct if r["sqrt_det"] > 0]
        points = [r["error_m"] for r in direct if r["error_m"] is not None]
        time = [r["time_half"] for r in direct if r["time_half"] is not None]
        time_cov = [r["time_error"] <= np.sqrt(r["time_half"] ** 2 + TIME_Q ** 2)
                    for r in direct if r["time_half"] is not None and r["time_error"] is not None]
        return {
            "n": len(direct), "coverage": float(np.mean(covered)) if covered else None,
            "median_area_m2": float(np.median(areas)) if areas else None,
            "median_error_m": float(np.median(points)) if points else None,
            "p90_error_m": float(np.quantile(points, 0.9)) if points else None,
            "time_coverage": float(np.mean(time_cov)) if time_cov else None,
            "median_time_width_ms": float(np.median([2 * np.sqrt(h ** 2 + TIME_Q ** 2) for h in time]) * 1000) if time else None,
            "median_time_error_ms": float(np.median([r["time_error"] for r in direct if r["time_error"] is not None]) * 1000) if any(r["time_error"] is not None for r in direct) else None,
        }
    by_mechanism = {}
    for mechanism in sorted({r["mechanism"] for r in test}):
        subset = [r for r in test if r["mechanism"] == mechanism]
        by_mechanism[mechanism] = {"baseline": summary(subset), "calibrated": summary(subset, q_cal)}
    return {"q_cal": q_cal, "calibration_rank": rank, "calibration_n": n_cal,
            "baseline_calibration": summary(calibration), "baseline_test": summary(test),
            "calibrated_calibration": summary(calibration, q_cal),
            "calibrated_test": summary(test, q_cal),
            "by_mechanism_test": by_mechanism,
            "time_q_s": TIME_Q,
            "failure_counts_test": dict(Counter(r["failure_class"] for r in test)),
            "test_sources": len({r["source_id"] for r in test}),
            "calibration_sources": len({r["source_id"] for r in calibration})}


def score(output, dataset):
    observations = {r["case_id"]: r for r in load_observations(output / "observations.jsonl")}
    truths = {}
    for split in ("calibration", "test"):
        for line in (dataset / "truth" / f"{split}.jsonl").open():
            record = json.loads(line); truths[record["source_id"]] = record
    records = []
    for line in (output / "predictions.jsonl").open():
        prediction = json.loads(line)
        if prediction["method"] != METHOD:
            continue
        observation = observations[prediction["case_id"]]
        truth_record = truths[prediction["source_id"]]
        truth = truth_record["truth"]
        if truth["event"] != "goal":
            continue
        region = prediction.get("region")
        if region is None:
            r2, q, sqrt_det, area = np.inf, 0.0, 0.0, None
        else:
            r2 = region_r2(region, truth["event_position_m"][1:3])
            q = float(region["threshold"])
            cov = np.asarray(region["covariance_m2"], float)
            sqrt_det = float(np.sqrt(np.linalg.det(cov)))
            area = float(np.pi * q * sqrt_det)
        point = prediction.get("point_yz_m")
        error = None if point is None else float(np.linalg.norm(np.asarray(point) - np.asarray(truth["event_position_m"][1:3])))
        interval = prediction.get("arrival_time_90_interval_s")
        time_half = None if interval is None else (interval[1] - interval[0]) / 2
        time_error = None if interval is None or prediction.get("arrival_time_s") is None else abs(prediction["arrival_time_s"] - truth["event_time_s"])
        records.append({
            "case_id": prediction["case_id"], "source_id": prediction["source_id"],
            "split": truth_record["split"], "mechanism": truth_record["parameters"]["mechanism"],
            "direct": True,
            "sigma": observation["measurement"]["sigma_m"], "r2": r2, "threshold": q,
            "sqrt_det": sqrt_det, "area": area,
            "error_m": error, "time_half": time_half, "time_error": time_error,
            "failure_class": ("none" if prediction["status"] == "ok" and region is not None
                              else "missing_region" if prediction["status"] == "ok" else prediction["status"]),
        })
    result = evaluate(records)
    write_json(output / "independent_summary.json", result)
    with (output / "per_case.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="独立测试：200 源，F1 固定 Cd。")
    parser.add_argument("stage", choices=("prepare", "run", "score", "all"))
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "independent_200")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "independent_200")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.stage in ("prepare", "all") and not args.output.exists():
        prepare(args.dataset, args.output)
    if args.stage in ("run", "all"):
        run(args.output, workers=args.workers)
    if args.stage in ("score", "all"):
        score(args.output, args.dataset)


if __name__ == "__main__":
    main()
