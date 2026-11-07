"""路线 2：F-fixed 的 200 源时间误差模型实验。

生成 200 个源，按机制各 20 个 calibration / 20 个 test；只跑 F1 固定 Cd。
在 calibration 上拟合全局/分机制时间误差项，在 test 上评估覆盖与宽度。
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

DATA_SEED = 20261109
METHOD = "F1_round2"


def prepare(dataset, output):
    if dataset.exists() or output.exists():
        raise FileExistsError("数据或结果目录已存在；保留旧结果。")
    built = build_dataset(200, seed=DATA_SEED, split_fractions=(1, 0, 0))
    by_mechanism = defaultdict(list)
    for truth in built["truth"]:
        by_mechanism[truth["parameters"]["mechanism"]].append(truth["source_id"])
    split_map = {}
    for mechanism, ids in sorted(by_mechanism.items()):
        if len(ids) != 40:
            raise ValueError(f"{mechanism} 源数不是 40: {len(ids)}")
        ids = sorted(ids)
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
    tasks = []
    for index, observation in enumerate(observations):
        tasks.append({"index": index, "case_id": observation["case_id"],
                      "source_id": observation["source_id"], "variant": "F1",
                      "method": METHODS["F1"], "seed": stable_seed(observation["case_id"])})
    if len(observations) != 800 or len(tasks) != 800:
        raise ValueError("观测/任务数不是 800")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "observations.jsonl", observations)
    write_json(output / "tasks.json", tasks)
    write_json(output / "run_config.json", {
        "dataset": str(dataset), "data_seed": DATA_SEED, "method": METHOD,
        "task_count": len(tasks), "variants": VARIANTS,
        "source_counts": {"calibration": 100, "test": 100},
        "notes": "F1 fixed Cd=0.22 only; time error model calibration/test; no original calibration/test.",
    })
    print(json.dumps({"dataset": str(dataset), "observations": len(observations),
                      "tasks": len(tasks), "split_counts": {"calibration": 100, "test": 100}},
                     ensure_ascii=False, indent=2))


def load_records(output, dataset):
    truths = {}
    for split in ("calibration", "test"):
        for line in (dataset / "truth" / f"{split}.jsonl").open():
            record = json.loads(line)
            truths[record["source_id"]] = record
    records = []
    for line in (output / "predictions.jsonl").open():
        prediction = json.loads(line)
        if prediction["method"] != METHOD:
            continue
        truth = truths[prediction["source_id"]]["truth"]
        if truth["event"] != "goal":
            continue
        interval = prediction.get("arrival_time_90_interval_s")
        if interval is None or prediction.get("arrival_time_s") is None:
            continue
        records.append({
            "case_id": prediction["case_id"], "source_id": prediction["source_id"],
            "split": truth["split"] if "split" in truth else truths[prediction["source_id"]]["split"],
            "mechanism": truths[prediction["source_id"]]["parameters"]["mechanism"],
            "sigma": float(prediction.get("sigma", 0.0)) if prediction.get("sigma") is not None else None,
            "truth_time": truth["event_time_s"], "pred_time": prediction["arrival_time_s"],
            "half": (interval[1] - interval[0]) / 2,
            "residual": prediction["arrival_time_s"] - truth["event_time_s"],
        })
    return records


def quantile(values, q):
    return float(np.quantile(np.asarray(values, float), q)) if values else None


def evaluate(records):
    calibration = [r for r in records if r["split"] == "calibration"]
    test = [r for r in records if r["split"] == "test"]
    q_global_conformal = quantile([abs(r["residual"]) for r in calibration], 0.90)
    q_mech_conformal = {}
    q_global_add = quantile([np.sqrt(max(r["residual"] ** 2 - r["half"] ** 2, 0.0))
                            for r in calibration], 0.90)
    q_mech_add = {}
    for mechanism in sorted({r["mechanism"] for r in calibration}):
        selected = [r for r in calibration if r["mechanism"] == mechanism]
        q_mech_conformal[mechanism] = quantile([abs(r["residual"]) for r in selected], 0.90)
        q_mech_add[mechanism] = quantile([np.sqrt(max(r["residual"] ** 2 - r["half"] ** 2, 0.0))
                                         for r in selected], 0.90)
    methods = {}
    for name in ("baseline", "global_conformal", "mechanism_conformal",
                 "global_additive", "mechanism_additive"):
        details = []
        for row in test:
            if name == "baseline":
                half = row["half"]
            elif name == "global_conformal":
                half = q_global_conformal
            elif name == "mechanism_conformal":
                half = q_mech_conformal.get(row["mechanism"], q_global_conformal)
            elif name == "global_additive":
                half = np.sqrt(row["half"] ** 2 + q_global_add ** 2)
            else:
                q = q_mech_add.get(row["mechanism"], q_global_add)
                half = np.sqrt(row["half"] ** 2 + q ** 2)
            details.append({"case_id": row["case_id"], "mechanism": row["mechanism"],
                            "sigma": row["sigma"], "covered": bool(abs(row["residual"]) <= half),
                            "width": float(2 * half)})
        methods[name] = {"coverage": float(np.mean([d["covered"] for d in details])),
                         "median_width_ms": float(np.median([d["width"] for d in details]) * 1000),
                         "n": len(details), "details": details}
    by_mechanism = {}
    for mechanism in sorted({r["mechanism"] for r in test}):
        by_mechanism[mechanism] = {}
        for name, payload in methods.items():
            selected = [d for d in payload["details"] if d["mechanism"] == mechanism]
            by_mechanism[mechanism][name] = {
                "n": len(selected),
                "coverage": float(np.mean([d["covered"] for d in selected])) if selected else None,
                "median_width_ms": float(np.median([d["width"] for d in selected]) * 1000) if selected else None,
            }
    return {"calibration_n": len(calibration), "test_n": len(test),
            "q_global_conformal": q_global_conformal,
            "q_mechanism_conformal": q_mech_conformal,
            "q_global_additive": q_global_add,
            "q_mechanism_additive": q_mech_add,
            "methods": methods, "by_mechanism": by_mechanism}


def score(output, dataset):
    records = load_records(output, dataset)
    result = evaluate(records)
    write_json(output / "time_model_summary.json", {
        "calibration_n": result["calibration_n"], "test_n": result["test_n"],
        "q_global_conformal": result["q_global_conformal"],
        "q_mechanism_conformal": result["q_mechanism_conformal"],
        "q_global_additive": result["q_global_additive"],
        "q_mechanism_additive": result["q_mechanism_additive"],
        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                    for k, v in result["methods"].items()},
        "by_mechanism": result["by_mechanism"],
    })
    print(json.dumps({"calibration_n": result["calibration_n"], "test_n": result["test_n"],
                      "q_global_conformal": result["q_global_conformal"],
                      "q_global_additive": result["q_global_additive"],
                      "methods": {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                                  for k, v in result["methods"].items()}},
                     ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="F-fixed 时间误差模型：200 源 calibration/test。")
    parser.add_argument("stage", choices=("prepare", "run", "score", "all"))
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "time_model_200")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "time_model_200")
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
