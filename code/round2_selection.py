"""第二轮 F0/F1 选择实验：数据准备、固定预算运行、封存评分。

只读取新数据集的 development 划分；calibration/test 不打开。
判定规则见 docs/superpowers/specs/2026-11-07-round2-prereg.md。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import INFERENCE, NOISE_SCENARIOS_M, PROJECT_ROOT
from data import build_dataset, load_observations, save_dataset
from development_regions import file_hash, write_json
from evaluate import point_prediction
from model_diagnosis import diagnostic_rows, write_jsonl
from uncertainty import UNCERTAINTY, uncertainty_observation

B_PATH = PROJECT_ROOT / "docs" / "superpowers" / "specs" / "2026-11-07-round2-prereg.md"
B_SHA = "4fee25d48f0f274cf8fd18bf47bd37c0ab12f07018e91c29ab458fa8ca07199d"
DATA_SEED = 20261108
BASE_SEED = 20261108
VARIANTS = {
    "F0": {"model": "F", "vertical_prior_std": None},
    "F1": {"model": "F", "vertical_prior_std": 0.10},
}
METHODS = {"F0": "F0_round2", "F1": "F1_round2"}
CLEAR_WORSE = {"coverage": -0.10, "area_ratio": 1.10, "median_error": 0.05,
               "p90_error": 0.10, "failure": 0.05}


def stable_seed(case_id, base=BASE_SEED):
    digest = hashlib.sha256(f"{base}|{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def verify_prereg():
    if not B_PATH.exists() or file_hash(B_PATH) != B_SHA:
        raise RuntimeError("B 预注册缺失或哈希不符；拒绝生成/运行新源实验。")


def prepare(dataset, output):
    verify_prereg()
    if dataset.exists():
        raise FileExistsError(f"数据集已存在，保留旧数据: {dataset}")
    if output.exists():
        raise FileExistsError(f"结果目录已存在，保留旧结果: {output}")
    built = build_dataset(100, seed=DATA_SEED, noise_sigmas_m=NOISE_SCENARIOS_M,
                          split_fractions=(1, 0, 0))
    save_dataset(built, dataset)
    observations = load_observations(dataset / "observations" / "development.jsonl")
    mechanisms = Counter()
    sources = set()
    source_mechanism = {row["source_id"]: row["parameters"]["mechanism"] for row in built["truth"]}
    for row in observations:
        sources.add(row["source_id"])
        if row["measurement"]["sigma_m"] == NOISE_SCENARIOS_M[0]:
            mechanisms[source_mechanism[row["source_id"]]] += 1
    if len(sources) != 100 or len(observations) != 400:
        raise ValueError(f"数据分母不符: {len(sources)} sources, {len(observations)} observations")
    if any(count != 20 for count in mechanisms.values()):
        raise ValueError(f"机制计数不是每类 20: {dict(mechanisms)}")
    tasks = []
    for index, observation in enumerate(observations):
        for variant in VARIANTS:
            tasks.append({"index": index, "case_id": observation["case_id"],
                          "source_id": observation["source_id"], "variant": variant,
                          "method": METHODS[variant], "seed": stable_seed(observation["case_id"])})
    if len(tasks) != 800:
        raise ValueError(f"F0/F1 任务数应为 800，实际 {len(tasks)}")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "observations.jsonl", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "run_config.json", {
        "prereg": str(B_PATH), "prereg_sha256": B_SHA,
        "dataset": str(dataset), "data_seed": DATA_SEED,
        "source_count": len(sources), "observation_count": len(observations),
        "mechanism_counts": dict(mechanisms), "variants": VARIANTS,
        "methods": METHODS, "task_count": len(tasks),
        "inference": asdict(INFERENCE),
        "uncertainty": asdict(replace(UNCERTAINTY, base_samples=2000, max_sample_doublings=0)),
        "starts": 3, "proposals_per_task": 4000,
        "notes": "Development only. F0/F1 selection; S structural control deferred. No calibration/test read.",
    })
    print(json.dumps({"dataset": str(dataset), "sources": len(sources),
                      "observations": len(observations), "tasks": len(tasks),
                      "mechanism_counts": dict(mechanisms)}, ensure_ascii=False, indent=2))


def worker(observation, task, output):
    path = output / "cases" / f"{task['index']:04d}_{task['variant']}"
    if (path / "summary.json").exists():
        return json.loads((path / "summary.json").read_text())
    path.mkdir(parents=True, exist_ok=False)
    cfg = VARIANTS[task["variant"]]
    settings = replace(INFERENCE, vertical_force_prior_std=cfg["vertical_prior_std"])
    options = replace(UNCERTAINTY, base_samples=2000, max_sample_doublings=0)
    started = time.perf_counter()
    try:
        prediction, result = uncertainty_observation(
            observation, cfg["model"], settings=settings, uncertainty=options,
            seed=task["seed"], starts=3)
    except Exception as error:  # 保留失败，不伪造预测。
        prediction = point_prediction(observation, task["method"], status="failure",
                                      message=str(error))
        result = {"success": False, "message": str(error), "draws": []}
    prediction["method"] = task["method"]
    interval = (result.get("summary") or {}).get("direct_crossing", {}).get("arrival_time_90_interval_s")
    if interval is not None:
        prediction["arrival_time_90_interval_s"] = interval
    draws = result.pop("draws", [])
    with gzip.open(path / "candidates.jsonl.gz", "wt", encoding="utf-8") as handle:
        for draw in draws:
            handle.write(json.dumps(draw, ensure_ascii=False, allow_nan=False) + "\n")
    record = {"index": task["index"], "variant": task["variant"], "method": task["method"],
              "case_id": task["case_id"], "source_id": task["source_id"],
              "seed": task["seed"], "prediction": prediction, "result": result,
              "draw_count": len(draws), "wall_s": time.perf_counter() - started}
    write_json(path / "summary.json", record)
    return record


def run(output, workers=6, limit=None):
    observations = {row["case_id"]: row for row in load_observations(
        output / "observations.jsonl")}
    tasks = json.loads((output / "tasks.json").read_text())
    if limit is not None:
        tasks = tasks[:limit]
    records = []
    begun = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        for task in tasks:
            path = output / "cases" / f"{task['index']:04d}_{task['variant']}" / "summary.json"
            if path.exists():
                records.append(json.loads(path.read_text()))
            else:
                pending.append(pool.submit(worker, observations[task["case_id"]], task, output))
        for future in as_completed(pending):
            records.append(future.result())
            if len(records) % 20 == 0:
                print(f"round2: {len(records)}/{len(tasks)}, elapsed {time.perf_counter()-begun:.1f}s",
                      flush=True)
    if len(records) != len(tasks):
        raise RuntimeError(f"任务数不符: {len(records)} != {len(tasks)}")
    predictions = [record["prediction"] for record in sorted(
        records, key=lambda item: (item["case_id"], item["variant"]))]
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(output / "prediction_manifest.json", {
        "task_count": len(records),
        "predictions_sha256": file_hash(output / "predictions.jsonl"),
        "observations_sha256": file_hash(output / "observations.jsonl"),
        "actual_proposals": sum(record["draw_count"] for record in records),
        "wall_s": time.perf_counter() - begun,
        "notes": "Sealed before truth read. F0/F1 only; S deferred.",
    })
    print(json.dumps({"records": len(records), "proposals": sum(r["draw_count"] for r in records),
                      "wall_s": time.perf_counter() - begun}, ensure_ascii=False))


def _safe_median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _safe_p90(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.quantile(values, 0.90)) if values else None


def _method_summary(rows):
    direct = [r for r in rows if r["direct"]]
    provided = [r for r in direct if r["provided"]]
    points = [r for r in direct if r["point_valid"]]
    times = [r for r in direct if r["time_valid"]]
    time_covered = [r for r in times if r["time_covered"] is True]
    failure = Counter(r["failure_class"] for r in direct)
    return {
        "direct_n": len(direct), "provided_n": len(provided), "point_n": len(points),
        "coverage_all": sum(r["covered_all"] for r in direct) / len(direct) if direct else None,
        "coverage_provided": sum(r["covered"] for r in provided) / len(provided) if provided else None,
        "publishable_rate": len(provided) / len(direct) if direct else None,
        "axis_y_all": sum(r["axis_y_covered_all"] for r in direct) / len(direct) if direct else None,
        "axis_z_all": sum(r["axis_z_covered_all"] for r in direct) / len(direct) if direct else None,
        "axis_y_provided": sum(r["axis_y_covered_provided"] for r in provided) / len(provided) if provided else None,
        "axis_z_provided": sum(r["axis_z_covered_provided"] for r in provided) / len(provided) if provided else None,
        "median_error_m": _safe_median([r.get("error_m") for r in points]),
        "p90_error_m": _safe_p90([r.get("error_m") for r in points]),
        "median_area_m2": _safe_median([r.get("area_m2") for r in provided]),
        "median_time_error_s": _safe_median([r.get("time_error_s") for r in times]),
        "p90_time_error_s": _safe_p90([r.get("time_error_s") for r in times]),
        "time_coverage_all": len(time_covered) / len(direct) if direct else None,
        "time_coverage_provided": len(time_covered) / len(times) if times else None,
        "time_provision_rate": len(times) / len(direct) if direct else None,
        "median_time_width_s": _safe_median([r.get("time_width_s") for r in times]),
        "failure_counts": dict(failure),
        "primary_failure_rate": sum(r["primary_failure"] for r in direct) / len(direct) if direct else None,
        "misfit_count": sum(r["failure_class"] == "misfit" for r in direct),
    }


def _augment(rows, observations, results):
    obs = {row["case_id"]: row for row in observations}
    augmented = []
    for row in rows:
        obs_row = obs[row["case_id"]]
        entry = results.get((row["case_id"], row["method"])) or {}
        result = entry.get("result") or {}
        fit = result.get("fit") or {}
        prediction = entry.get("prediction") or {}
        summary = result.get("summary") or {}
        interval = (summary.get("direct_crossing") or {}).get("arrival_time_90_interval_s")
        if interval is None:
            interval = prediction.get("arrival_time_90_interval_s")
        sigma = float(obs_row["measurement"]["sigma_m"])
        direct = row["truth_event"] == "goal"
        provided = "normalized_radius" in row
        point_valid = "error_m" in row
        radius = row.get("normalized_radius")
        covered = bool(row.get("covered")) if direct and provided else False
        axis_y = bool(direct and provided and not row.get("outside_y_projection", True))
        axis_z = bool(direct and provided and not row.get("outside_z_projection", True))
        rms = fit.get("rms_residual_m")
        chi2 = fit.get("data_chi_square")
        dof = max(1, 6 - len(fit.get("parameter_names", [])))
        chi2_dof = float(chi2 / dof) if chi2 is not None else None
        misfit = bool(fit.get("success") and ((rms is not None and rms > 3 * sigma)
                                              or (chi2_dof is not None and chi2_dof > 4)))
        truth_time = row.get("true_time")
        time_valid = bool(direct and interval is not None and len(interval) == 2)
        time_covered = bool(time_valid and interval[0] <= truth_time <= interval[1]) if time_valid else False
        time_width = float(interval[1] - interval[0]) if time_valid else None
        time_error = abs(prediction.get("arrival_time_s") - truth_time) if (
            time_valid and prediction.get("arrival_time_s") is not None) else None
        if not obs_row["observation_available"]:
            failure_class = "unavailable"
        elif not fit.get("success"):
            failure_class = "infeasible"
        elif misfit:
            failure_class = "misfit"
        elif not result.get("success", False):
            failure_class = "sampling"
        elif row["status"] != "ok":
            failure_class = "other"
        elif direct and (not point_valid or not provided):
            failure_class = "incomplete"
        else:
            failure_class = "none"
        augmented.append({**row, "direct": direct, "provided": provided,
                          "point_valid": point_valid, "covered": covered,
                          "covered_all": covered if direct else None,
                          "axis_y_covered_all": axis_y if direct else None,
                          "axis_z_covered_all": axis_z if direct else None,
                          "axis_y_covered_provided": axis_y if direct and provided else None,
                          "axis_z_covered_provided": axis_z if direct and provided else None,
                          "primary_failure": bool(direct and (not point_valid or not provided)),
                          "failure_class": failure_class,
                          "fit_rms_m": rms, "fit_chi2_dof": chi2_dof, "misfit": misfit,
                          "time_valid": time_valid, "time_covered": time_covered if time_valid else None,
                          "time_width_s": time_width, "time_error_s": time_error,
                          "prediction": prediction})
    return augmented


def _pairs(rows):
    by_case = defaultdict(dict)
    for row in rows:
        by_case[(row["case_id"], row["method"])] = row
    pairs = []
    for case_id in {key[0] for key in by_case}:
        a = by_case.get((case_id, METHODS["F0"]))
        b = by_case.get((case_id, METHODS["F1"]))
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def _bootstrap_metrics(pairs):
    direct = [(a, b) for a, b in pairs if a["direct"]]
    if not direct:
        return None
    def cov(row): return bool(row["covered_all"])
    def fail(row): return bool(row["primary_failure"])
    area = [(a, b) for a, b in direct if a["provided"] and b["provided"]]
    points = [(a, b) for a, b in direct if a["point_valid"] and b["point_valid"]]
    values = {
        "coverage_diff": np.mean([cov(b) for a, b in direct]) - np.mean([cov(a) for a, b in direct]),
        "failure_diff": np.mean([fail(b) for a, b in direct]) - np.mean([fail(a) for a, b in direct]),
        "area_ratio": (np.median([b["area_m2"] for a, b in area]) /
                       np.median([a["area_m2"] for a, b in area])) if area else None,
        "median_error_diff": (np.median([b["error_m"] for a, b in points]) -
                              np.median([a["error_m"] for a, b in points])) if points else None,
        "p90_error_diff": (np.quantile([b["error_m"] for a, b in points], 0.9) -
                           np.quantile([a["error_m"] for a, b in points], 0.9)) if points else None,
    }
    return values


def _stratified_bootstrap(rows, draws=10000, seed=20261108):
    sources = defaultdict(list)
    cases = defaultdict(list)
    mechanisms = {}
    for row in rows:
        if row["method"] != METHODS["F0"]:
            continue
        sources[row["scenario"]].append(row["source_id"])
        mechanisms[row["source_id"]] = row["scenario"]
    for row in rows:
        cases[row["source_id"]].append(row["case_id"])
    by_mechanism = {mech: sorted(set(ids)) for mech, ids in sources.items()}
    pairs = _pairs(rows)
    pair_by_case = {a["case_id"]: (a, b) for a, b in pairs}
    pairs_by_source = defaultdict(list)
    for case_id, pair in pair_by_case.items():
        pairs_by_source[pair[0]["source_id"]].append(pair)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    invalid = Counter()
    for _ in range(draws):
        selected = []
        for mech, ids in by_mechanism.items():
            chosen = rng.choice(ids, size=len(ids), replace=True)
            for source in chosen:
                selected.extend(pairs_by_source[source])
        metrics = _bootstrap_metrics(selected)
        if metrics is None:
            invalid["all"] += 1
            for key in ("coverage_diff", "failure_diff", "area_ratio",
                        "median_error_diff", "p90_error_diff"):
                invalid[key] += 1
            continue
        for key, value in metrics.items():
            if value is None or not np.isfinite(value):
                invalid[key] += 1
            else:
                samples[key].append(float(value))
    bounds = {}
    for key, values in samples.items():
        if key == "coverage_diff":
            bounds[key] = [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))]
        else:
            bounds[key] = [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))]
    return {"draws": draws, "seed": seed, "bounds": bounds,
            "invalid_counts": dict(invalid), "valid_counts": {k: len(v) for k, v in samples.items()}}


def _mechanism_guard(rows):
    table = {}
    sources = defaultdict(set)
    for row in rows:
        if row["method"] == METHODS["F0"]:
            sources[row["scenario"]].add(row["source_id"])
    pairs = _pairs(rows)
    by_mechanism = defaultdict(list)
    for a, b in pairs:
        by_mechanism[a["scenario"]].append((a, b))
    rng = np.random.default_rng(20261108)
    for mech, items in sorted(by_mechanism.items()):
        direct = [(a, b) for a, b in items if a["direct"]]
        cov_diff = (np.mean([b["covered_all"] for a, b in direct]) -
                    np.mean([a["covered_all"] for a, b in direct])) if direct else None
        fail_diff = (np.mean([b["primary_failure"] for a, b in direct]) -
                     np.mean([a["primary_failure"] for a, b in direct])) if direct else None
        interval = {}
        ids = sorted(sources[mech])
        for _ in range(2000):
            chosen = rng.choice(ids, size=len(ids), replace=True)
            sample = [pair for source in chosen for pair in items
                      if pair[0]["source_id"] == source]
            sample = [(a, b) for a, b in sample if a["direct"]]
            if not sample:
                continue
            for key in ("coverage_diff", "failure_diff"):
                if key == "coverage_diff":
                    value = (np.mean([b["covered_all"] for a, b in sample]) -
                             np.mean([a["covered_all"] for a, b in sample]))
                else:
                    value = (np.mean([b["primary_failure"] for a, b in sample]) -
                             np.mean([a["primary_failure"] for a, b in sample]))
                interval.setdefault(key, []).append(float(value))
        table[mech] = {
            "direct_n": len(direct), "coverage_diff": float(cov_diff) if cov_diff is not None else None,
            "failure_diff": float(fail_diff) if fail_diff is not None else None,
            "coverage_diff_95": [float(np.quantile(interval["coverage_diff"], 0.025)),
                                  float(np.quantile(interval["coverage_diff"], 0.975))]
                                 if "coverage_diff" in interval else None,
            "failure_diff_95": [float(np.quantile(interval["failure_diff"], 0.025)),
                                 float(np.quantile(interval["failure_diff"], 0.975))]
                                if "failure_diff" in interval else None,
        }
    return table


def score(output):
    manifest = json.loads((output / "prediction_manifest.json").read_text())
    if file_hash(output / "predictions.jsonl") != manifest["predictions_sha256"]:
        raise RuntimeError("预测文件在封存后被修改。")
    if file_hash(output / "observations.jsonl") != manifest["observations_sha256"]:
        raise RuntimeError("观测文件在封存后被修改。")
    observations = load_observations(output / "observations.jsonl")
    config = json.loads((output / "run_config.json").read_text())
    dataset = Path(config["dataset"])
    truth = [json.loads(line) for line in
             (dataset / "truth" / "development.jsonl").open()]
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").open()]
    if len(predictions) != manifest["task_count"]:
        raise RuntimeError("预测数与封存清单不一致。")
    if {p["method"] for p in predictions} != set(METHODS.values()):
        raise RuntimeError("预测方法不完整；本轮要求 F0/F1。")
    results = {}
    for path in sorted((output / "cases").glob("*/summary.json")):
        record = json.loads(path.read_text())
        results[(record["case_id"], record["method"])] = {
            "result": record["result"], "prediction": record["prediction"]}
    rows = diagnostic_rows(predictions, observations, truth)
    rows = _augment(rows, observations, results)
    per_method = {METHODS["F0"]: [r for r in rows if r["method"] == METHODS["F0"]],
                  METHODS["F1"]: [r for r in rows if r["method"] == METHODS["F1"]]}
    summary = {method: _method_summary(items) for method, items in per_method.items()}
    bootstrap = _stratified_bootstrap(rows)
    mechanisms = _mechanism_guard(rows)
    point = _bootstrap_metrics(_pairs(rows))
    criteria = {
        "coverage": {"estimate": point["coverage_diff"],
                     "lower": bootstrap["bounds"].get("coverage_diff", [None, None])[0],
                     "threshold": -0.05,
                     "pass": (bootstrap["bounds"].get("coverage_diff", [None])[0] is not None
                              and bootstrap["bounds"]["coverage_diff"][0] >= -0.05
                              and bootstrap["invalid_counts"].get("coverage_diff", 0) == 0)},
        "area_ratio": {"estimate": point["area_ratio"],
                       "upper": bootstrap["bounds"].get("area_ratio", [None, None])[1],
                       "threshold": 0.85,
                       "pass": (bootstrap["bounds"].get("area_ratio", [None, None])[1] is not None
                                and bootstrap["bounds"]["area_ratio"][1] <= 0.85
                                and bootstrap["invalid_counts"].get("area_ratio", 0) == 0)},
        "median_error": {"estimate": point["median_error_diff"],
                         "upper": bootstrap["bounds"].get("median_error_diff", [None, None])[1],
                         "threshold": 0.02,
                         "pass": (bootstrap["bounds"].get("median_error_diff", [None, None])[1] is not None
                                  and bootstrap["bounds"]["median_error_diff"][1] <= 0.02
                                  and bootstrap["invalid_counts"].get("median_error_diff", 0) == 0)},
        "p90_error": {"estimate": point["p90_error_diff"],
                      "upper": bootstrap["bounds"].get("p90_error_diff", [None, None])[1],
                      "threshold": 0.05,
                      "pass": (bootstrap["bounds"].get("p90_error_diff", [None, None])[1] is not None
                               and bootstrap["bounds"]["p90_error_diff"][1] <= 0.05
                               and bootstrap["invalid_counts"].get("p90_error_diff", 0) == 0)},
        "failure": {"estimate": point["failure_diff"],
                    "upper": bootstrap["bounds"].get("failure_diff", [None, None])[1],
                    "threshold": 0.02,
                    "pass": (bootstrap["bounds"].get("failure_diff", [None, None])[1] is not None
                             and bootstrap["bounds"]["failure_diff"][1] <= 0.02
                             and bootstrap["invalid_counts"].get("failure_diff", 0) == 0)},
    }
    veto = [name for name, item in mechanisms.items()
            if (item["coverage_diff"] is not None and item["coverage_diff"] < -0.10)
            or (item["failure_diff"] is not None and item["failure_diff"] > 0.05)]
    clear_worse = (point["coverage_diff"] < CLEAR_WORSE["coverage"]
                   or (point["area_ratio"] is not None and point["area_ratio"] > CLEAR_WORSE["area_ratio"])
                   or (point["median_error_diff"] is not None and point["median_error_diff"] > CLEAR_WORSE["median_error"])
                   or (point["p90_error_diff"] is not None and point["p90_error_diff"] > CLEAR_WORSE["p90_error"])
                   or point["failure_diff"] > CLEAR_WORSE["failure"])
    all_pass = all(item["pass"] for item in criteria.values()) and not veto
    any_invalid = any(bootstrap["invalid_counts"].get(key, 0) > 0 for key in criteria)
    favorable = (criteria["coverage"]["estimate"] >= -0.05
                 and criteria["area_ratio"]["estimate"] is not None
                 and criteria["area_ratio"]["estimate"] <= 0.85
                 and criteria["median_error"]["estimate"] is not None
                 and criteria["median_error"]["estimate"] <= 0.02
                 and criteria["p90_error"]["estimate"] is not None
                 and criteria["p90_error"]["estimate"] <= 0.05
                 and criteria["failure"]["estimate"] <= 0.02)
    if all_pass:
        decision = "采纳 F1"
        reason = "五项主条件全部通过，且未触发机制保护。"
    elif veto or clear_worse:
        decision = "不采纳 F1"
        reason = "触发机制保护或出现明确恶化： " + ", ".join(veto) if veto else "至少一项点估计越过明确恶化阈值。"
    elif any_invalid:
        decision = "证据不足"
        reason = "存在无法计算的指标或无效重采样；按预注册不采纳 F1。"
    elif favorable:
        decision = "证据不足"
        reason = "点估计均达到门槛，但至少一项置信界未过；按预注册不采纳 F1。"
    else:
        decision = "尚不能确定整体改进"
        reason = "不同指标收益与代价冲突，未达到明确恶化或机制否决。"
    report = {"summary": summary, "criteria": criteria, "mechanisms": mechanisms,
              "decision": decision, "reason": reason, "veto": veto,
              "clear_worse": bool(clear_worse), "bootstrap": bootstrap,
              "manifest": manifest, "prereg_sha256": B_SHA}
    write_json(output / "summary.json", _jsonable(report))
    _write_csv(output / "per_case.csv", rows)
    _write_csv(output / "mechanisms.csv", [
        {"mechanism": name, **item} for name, item in mechanisms.items()])
    _write_report(output / "report.md", report, rows)
    print(json.dumps({"decision": decision, "reason": reason,
                      "F0": summary[METHODS["F0"]], "F1": summary[METHODS["F1"]]},
                     ensure_ascii=False, indent=2))
    return report


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
                             for key, value in row.items()})


def _fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(path, report, rows):
    s = report["summary"]
    lines = ["# 第二轮 F0/F1 选择报告", "",
             "范围：新 100 个 development 源、400 条观测、F0/F1 共 800 个任务；"
             "calibration/test 未读取；S 结构对照本轮暂缓。", "",
             "## 方法汇总", "",
             "| 指标 | F0 | F1 |", "|---|---:|---:|"]
    keys = ["direct_n", "provided_n", "coverage_all", "coverage_provided",
            "publishable_rate", "axis_y_all", "axis_z_all", "median_error_m",
            "p90_error_m", "median_area_m2", "time_coverage_all", "median_time_error_s",
            "primary_failure_rate", "misfit_count"]
    for key in keys:
        lines.append(f"| {key} | {_fmt(s[METHODS['F0']].get(key))} | {_fmt(s[METHODS['F1']].get(key))} |")
    lines += ["", "## 五项主条件（F1 − F0）", "",
              "| 指标 | 点估计 | 单侧95%界 | 门槛 | 通过 |", "|---|---:|---:|---:|---:|"]
    for key, item in report["criteria"].items():
        bound = item.get("lower", item.get("upper"))
        lines.append(f"| {key} | {_fmt(item['estimate'])} | {_fmt(bound)} | {_fmt(item['threshold'])} | {item['pass']} |")
    lines += ["", "## 机制分层", "",
              "| 机制 | 真过线 | 覆盖差 F1−F0 | 覆盖差95%区间 | 失败差 | 失败差95%区间 |",
              "|---|---:|---:|---|---:|---|"]
    for name, item in report["mechanisms"].items():
        lines.append(f"| {name} | {item['direct_n']} | {_fmt(item['coverage_diff'])} | "
                     f"{_fmt(item['coverage_diff_95'])} | {_fmt(item['failure_diff'])} | "
                     f"{_fmt(item['failure_diff_95'])} |")
    lines += ["", "## 失败分类", "",
              "| 方法 | 类别 | 数量 |", "|---|---|---:|"]
    for method in (METHODS["F0"], METHODS["F1"]):
        for name, count in sorted(s[method]["failure_counts"].items()):
            lines.append(f"| {method} | {name} | {count} |")
    lines += ["", "## 决策", "",
              f"**{report['decision']}**：{report['reason']}", "",
              f"- B 预注册 SHA256：`{report['prereg_sha256']}`",
              f"- bootstrap：{report['bootstrap']['draws']} 次，种子 {report['bootstrap']['seed']}",
              f"- 无效重采样：`{report['bootstrap']['invalid_counts']}`",
              "- 本报告是 development 验证，不是校准/测试；S 对照本轮暂缓，F1 判定不受影响。",
              "", "## 说明", "",
              "- 覆盖率(全)分母为所有真实直接过线案例；缺区域计漏覆盖。",
              "- 覆盖率(已提供)分母为有有效区域的案例；另列可发布区域比例。",
              "- 轴覆盖为 90% 椭圆坐标投影，不是边际 90% 概率。",
              "- 主失败率为“无有效点或无有效区域”；misfit/ sampling 等仅作诊断分类。",
              "- 所有失败、不可用和未决任务保留，不删除、不替换。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="第二轮 F0/F1 选择实验；只读 development。")
    parser.add_argument("stage", choices=("prepare", "run", "score", "all"))
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "round2_development")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "round2_selection")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.stage in ("prepare", "all") and not args.output.exists():
        prepare(args.dataset, args.output)
    if args.stage in ("run", "all"):
        run(args.output, workers=args.workers, limit=args.limit)
    if args.stage in ("score", "all"):
        score(args.output)


if __name__ == "__main__":
    main()
