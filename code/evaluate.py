"""预测与真值的独立评价，默认仅在开发集运行 G/A 流程示例。

预测器只接触 observations；本模块在预测生成后读取 truth 判卷。
不选择模型、不拟合校准尺度，也不实现两帧气动力反演。
预测 JSONL 的接口及各指标分母见 code/README.md。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from math import isfinite, pi
from pathlib import Path

import numpy as np

from config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, Environment
from data import SCHEMA_VERSION, SPLITS, load_observations, observation_case

EVENTS = ("goal", "ground", "boundary", "unresolved", "failure")
STATUSES = ("ok", "failure", "unavailable")
PREDICTION_FIELDS = {
    "schema_version", "case_id", "source_id", "split", "method", "status", "event",
    "point_kind", "point_yz_m", "arrival_time_s", "region_mass", "region",
    "goal_probability", "message",
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _finite_vector(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 形状应为 {shape} 且所有数值有限。")
    return array


def _validate_prediction(row):
    if set(row) != PREDICTION_FIELDS or row["schema_version"] != SCHEMA_VERSION:
        raise ValueError("预测字段与约定不符，见 code/README.md 的预测接口。")
    for field in ("case_id", "source_id", "method", "point_kind"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{field} 须为非空字符串。")
    if row["split"] not in SPLITS or row["status"] not in STATUSES:
        raise ValueError("未知数据划分或预测状态。")
    if row["event"] is not None and row["event"] not in EVENTS:
        raise ValueError("未知预测事件。")
    if row["point_yz_m"] is not None:
        _finite_vector(row["point_yz_m"], (2,), "门线二维评分点")
    if row["arrival_time_s"] is not None and not isfinite(row["arrival_time_s"]):
        raise ValueError("门线时间须有限或为 None。")
    for key in ("region_mass", "goal_probability"):
        value = row[key]
        if value is not None and (not isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{key} 须在 [0,1] 内或为 None。")
    if row["region_mass"] in (0, 1):
        raise ValueError("区域名义概率须严格在 (0,1) 内。")
    if row["region"] is not None and row["region_mass"] is None:
        raise ValueError("提供区域时须声明 region_mass。")
    if row["status"] != "ok" and any(row[k] is not None for k in
            ("point_yz_m", "arrival_time_s", "region", "goal_probability")):
        raise ValueError("失败/不可用记录不能携带有效预测；region_mass 可保留以统计漏覆盖。")


def load_predictions(path):
    rows = read_jsonl(path)
    for row in rows:
        _validate_prediction(row)
    return rows


def point_prediction(observation, method, simulation=None, *, status="ok", message="",
                     point_kind="point_estimate"):
    """把 physics 的单条结果包装为共同预测格式；未来 inverse 可复用。"""
    event = simulation["event"] if simulation is not None else None
    if event == "failure":
        status = "failure"
    point = simulation["event_position_m"][1:3] if (
        status == "ok" and event == "goal") else None
    return {
        "schema_version": SCHEMA_VERSION, "case_id": observation["case_id"],
        "source_id": observation["source_id"], "split": observation["split"],
        "method": method, "status": status, "event": event,
        # 单条拟合轨迹的交点不自动等于门线后验均值或众数。
        "point_kind": point_kind, "point_yz_m": point,
        "arrival_time_s": simulation["event_time_s"] if point is not None else None,
        "region_mass": None, "region": None, "goal_probability": None,
        "message": message or (simulation["message"] if simulation else ""),
    }


def generate_baseline_predictions(observations, models=("G", "A")):
    """只读取公开观测，独立于真值产生 G/A 点预测；不伪造概率区域。"""
    from physics import predict_baseline

    if not models or any(model not in ("G", "A") for model in models):
        raise ValueError("内置演示只支持 G/A 基线。")
    output = []
    for observation in observations:
        for model in models:
            if not observation["observation_available"]:
                output.append(point_prediction(observation, model, status="unavailable",
                                               message="没有完整观测，保留不可用记录。"))
                continue
            case = observation_case(observation)
            known = dict(observation["known_environment"])
            known["wind_m_s"] = tuple(known["wind_m_s"])
            environment = Environment(**known)
            try:
                simulation = predict_baseline(model, case=case, environment=environment)
                prediction = point_prediction(observation, model, simulation)
            except (ValueError, FloatingPointError, RuntimeError) as error:
                prediction = point_prediction(observation, model, status="failure", message=str(error))
            output.append(prediction)
    return output


def _region_details(region, true_yz):
    """区域为 (g-center)' covariance^-1 (g-center) <= threshold。"""
    if not isinstance(region, dict) or set(region) != {"center_yz_m", "covariance_m2", "threshold"}:
        raise ValueError("区域必须包含 center_yz_m、covariance_m2、threshold。")
    center = _finite_vector(region["center_yz_m"], (2,), "区域中心")
    covariance = _finite_vector(region["covariance_m2"], (2, 2), "协方差")
    threshold = region["threshold"]
    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("区域 threshold 须为有限正数。")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("区域协方差须对称。")
    cholesky = np.linalg.cholesky(covariance)
    area = float(pi * threshold * np.prod(np.diag(cholesky)))
    if not isfinite(area) or area <= 0:
        raise ValueError("区域面积必须是有限正数。")
    covered = None
    if true_yz is not None:
        offset = np.linalg.solve(cholesky, np.asarray(true_yz)-center)
        distance = float(np.dot(offset, offset))
        covered = bool(distance <= threshold + 32*np.finfo(float).eps*max(1, threshold))
    return area, covered


def _aligned_rows(observations, truths, predictions):
    if not observations:
        raise ValueError("观测集不能为空。")
    observation_map, source_splits = {}, {}
    splits = {row["split"] for row in observations}
    if len(splits) != 1 or not splits.issubset(SPLITS):
        raise ValueError("一次只评价一个明确的数据划分，不能混合开发/校准/测试。")
    for row in observations:
        if row["case_id"] in observation_map:
            raise ValueError("重复观测 case_id。")
        observation_map[row["case_id"]] = row
        source_splits[row["source_id"]] = row["split"]
    truth_map = {}
    for row in truths:
        if row["source_id"] in truth_map or row["split"] not in splits:
            raise ValueError("重复真值 source_id 或真值划分与观测不一致。")
        truth = row["truth"]
        if truth["event"] not in EVENTS:
            raise ValueError("未知真值事件。")
        if truth["event"] == "goal":
            _finite_vector(truth["event_position_m"], (3,), "真实门线位置")
            if not isfinite(truth["event_time_s"]) or not isinstance(truth["inside_goal_opening"], bool):
                raise ValueError("真实过线需要有限时间和明确入框标记。")
        truth_map[row["source_id"]] = row
    if set(truth_map) != set(source_splits):
        raise ValueError("真值与观测的源射门集合不一致，不能丢弃缺失答案。")
    lookup, metadata = {}, {}
    for prediction in predictions:
        _validate_prediction(prediction)
        observation = observation_map.get(prediction["case_id"])
        if observation is None or any(prediction[key] != observation[key] for key in ("source_id", "split")):
            raise ValueError("预测 case_id/source_id/split 与观测不匹配。")
        key = (prediction["method"], prediction["case_id"])
        if key in lookup:
            raise ValueError("同一方法同一 case_id 不得重复提交预测。")
        lookup[key] = prediction
        info = (prediction["point_kind"], prediction["region_mass"])
        if prediction["method"] in metadata and metadata[prediction["method"]] != info:
            raise ValueError("同一方法的评分点定义与区域名义概率必须一致；不同版本另起 method 名称。")
        metadata[prediction["method"]] = info
    if not metadata:
        raise ValueError("至少需要一条预测来声明待评价方法。")

    rows = []
    for method, (point_kind, region_mass) in sorted(metadata.items()):
        for observation in observations:
            truth_record = truth_map[observation["source_id"]]
            truth = truth_record["truth"]
            direct = truth["event"] == "goal"
            true_yz = truth["event_position_m"][1:3] if direct else None
            prediction = lookup.get((method, observation["case_id"]))
            row = {
                "method": method, "case_id": observation["case_id"],
                "source_id": observation["source_id"], "split": observation["split"],
                "observation_available": observation["observation_available"],
                "mechanism": truth_record.get("parameters", {}).get("mechanism", "unknown"),
                "noise_sigma_m": observation["measurement"]["sigma_m"],
                "noise_mode": observation["measurement"]["mode"],
                "truth_event": truth["event"], "true_direct_goal_plane": direct,
                "true_inside_goal": truth["inside_goal_opening"] if direct else
                                    (False if truth["event"] == "ground" else None),
                "status": "missing" if prediction is None else prediction["status"],
                "predicted_event": prediction["event"] if prediction else None,
                "point_kind": point_kind, "region_mass": region_mass,
                "position_error_m": None, "arrival_time_absolute_error_s": None,
                "region_valid": False, "region_invalid": False,
                "region_area_m2": None, "covered": False if direct and region_mass is not None else None,
                "brier_score": None, "message": prediction["message"] if prediction else "未提交预测。",
            }
            if prediction is not None and prediction["status"] == "ok":
                if direct and prediction["point_yz_m"] is not None:
                    row["position_error_m"] = float(np.linalg.norm(
                        np.asarray(prediction["point_yz_m"])-true_yz))
                if direct and prediction["arrival_time_s"] is not None:
                    row["arrival_time_absolute_error_s"] = abs(prediction["arrival_time_s"]-truth["event_time_s"])
                if prediction["region"] is not None:
                    try:
                        area, covered = _region_details(prediction["region"], true_yz)
                        row.update(region_valid=True, region_area_m2=area,
                                   covered=covered if direct else None)
                    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                        row.update(region_invalid=True, message=row["message"]+f" 区域无效：{error}")
                if row["true_inside_goal"] is not None and prediction["goal_probability"] is not None:
                    row["brier_score"] = (prediction["goal_probability"]-float(row["true_inside_goal"]))**2
            rows.append(row)
    return rows


def _distribution(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p90": float(np.quantile(values, .9)) if values else None}


def _summary(rows):
    direct = [r for r in rows if r["true_direct_goal_plane"]]
    available_direct = [r for r in direct if r["observation_available"]]
    mass = rows[0]["region_mass"]
    position_errors = [r["position_error_m"] for r in rows]
    valid_point_n = sum(x is not None for x in position_errors)
    known_outcomes = sum(r["true_inside_goal"] is not None for r in rows)
    return {
        "case_count": len(rows), "source_count": len({r["source_id"] for r in rows}),
        "observation_unavailable_count": sum(not r["observation_available"] for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "predicted_event_counts": dict(Counter(r["predicted_event"] or "no_event_label" for r in rows)),
        "truth_event_counts": dict(Counter(r["truth_event"] for r in rows)),
        "failure_or_missing_rate": sum(r["status"] in ("failure", "missing") for r in rows)/len(rows),
        "true_direct_case_count": len(direct), "true_direct_available_input_count": len(available_direct),
        "direct_point_prediction_rate": valid_point_n/len(direct) if direct else None,
        "position_error_m": _distribution(position_errors),
        "arrival_time_absolute_error_s": _distribution(r["arrival_time_absolute_error_s"] for r in rows),
        "region_mass": mass,
        "valid_region_count": sum(r["region_valid"] for r in rows),
        "invalid_region_count": sum(r["region_invalid"] for r in rows),
        "coverage": {"denominator": len(direct),
            "covered_count": sum(r["covered"] is True for r in direct),
            "rate": sum(r["covered"] is True for r in direct)/len(direct)
                    if direct and mass is not None else None},
        "coverage_available_inputs": {"denominator": len(available_direct),
            "rate": sum(r["covered"] is True for r in available_direct)/len(available_direct)
                    if available_direct and mass is not None else None},
        # 主面积口径：真实直接过线案例中成功提供的区域，和覆盖率联合解读。
        "region_area_on_true_direct_m2": _distribution(
            r["region_area_m2"] for r in direct if r["region_valid"]),
        "brier_score": _distribution(r["brier_score"] for r in rows),
        "brier_known_outcome_count": known_outcomes,
        "probability_missing_on_known_outcome_count": sum(
            r["true_inside_goal"] is not None and r["brier_score"] is None for r in rows),
    }


def _cluster_intervals(rows, samples, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    keys = list(groups)
    values = defaultdict(list)
    for _ in range(samples):
        selected = [row for i in rng.integers(0, len(keys), size=len(keys)) for row in groups[keys[int(i)]]]
        summary = _summary(selected)
        estimates = {
            "position_error_median_m": summary["position_error_m"]["median"],
            "position_error_p90_m": summary["position_error_m"]["p90"],
            "coverage_rate": summary["coverage"]["rate"],
            "region_area_median_m2": summary["region_area_on_true_direct_m2"]["median"],
            "failure_or_missing_rate": summary["failure_or_missing_rate"],
            "brier_mean": summary["brier_score"]["mean"],
        }
        for key, value in estimates.items():
            if value is not None:
                values[key].append(value)
    return {key: {"lower": float(np.quantile(value, .025)),
                  "upper": float(np.quantile(value, .975)), "valid_resamples": len(value)}
            for key, value in values.items() if value}


def _paired_comparisons(rows, samples, rng):
    methods = sorted({row["method"] for row in rows})
    lookup = {(row["method"], row["case_id"]): row for row in rows}
    output = []
    for a, b in combinations(methods, 2):
        pairs = []
        for row in rows:
            if row["method"] != a:
                continue
            other = lookup[(b, row["case_id"])]
            if row["position_error_m"] is not None and other["position_error_m"] is not None:
                pairs.append((row["source_id"], row["position_error_m"], other["position_error_m"]))
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair[0]].append(pair)
        keys = list(groups)
        def difference(selected):
            return float(np.median([p[1] for p in selected])-np.median([p[2] for p in selected]))
        estimates = []
        if keys:
            for _ in range(samples):
                selected = [p for i in rng.integers(0, len(keys), size=len(keys)) for p in groups[keys[int(i)]]]
                estimates.append(difference(selected))
        point_kinds = {row["method"]: row["point_kind"] for row in rows}
        output.append({"method_a": a, "method_b": b, "common_case_count": len(pairs),
            "common_source_count": len(keys), "point_kind_a": point_kinds[a], "point_kind_b": point_kinds[b],
            "median_error_a_minus_b_m": difference(pairs) if pairs else None,
            "cluster_bootstrap_95_interval": [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))]
                if estimates else None,
            "note": "负值表示 A 在共同可评分样本上的中位误差更小；须同时检查缺失/失败及评分点口径。"})
    return output


def evaluate_predictions(observations, truths, predictions, *, bootstrap_samples=300, seed=RANDOM_SEED):
    """按全部观测清单对齐，缺预测保留；区间按源射门而非噪声副本重采样。"""
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples 须为非负整数；0 表示不计算区间。")
    rows = _aligned_rows(observations, truths, predictions)
    rng = np.random.default_rng(seed)
    summaries, strata = {}, {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summaries[method] = _summary(selected)
        summaries[method]["point_kind"] = selected[0]["point_kind"]
        summaries[method]["cluster_bootstrap_95_intervals"] = _cluster_intervals(selected, bootstrap_samples, rng)
        strata[method] = {}
        for axis in ("noise_sigma_m", "mechanism"):
            groups = defaultdict(list)
            for row in selected:
                groups[str(row[axis])].append(row)
            strata[method][axis] = {key: _summary(group) for key, group in sorted(groups.items())}
    return {
        "schema_version": SCHEMA_VERSION, "split": observations[0]["split"],
        "bootstrap_samples": bootstrap_samples, "seed": seed,
        "methods": summaries, "strata": strata,
        "paired_comparisons": _paired_comparisons(rows, bootstrap_samples, rng),
        "rows": rows,
        "definitions": {
            "position_error": "门线 (Y,Z) 欧氏距离；仅真值直接过平面且提供评分点的共同可定义样本。",
            "coverage": "所有真值直接过平面案例为分母；声明概率区域的方法缺失/失败/无效区域均漏覆盖。另报可用输入子集。",
            "no_probability_output": "未声明 region_mass 的点预测基线覆盖率为 null，不伪造区域或概率。",
            "area": "真实直接过线案例中有效区域的面积，须连同覆盖率、区域提供数与失败率解读。",
            "brier": "直接入框概率的均方误差；真值过线或先接地时标签明确，边界/未决/失败不评分，缺概率计数另报。",
            "bootstrap": "按 source_id 整组有放回重采样；95% 百分位区间，噪声副本不是独立源样本。分层表未额外计算区间。",
            "scope": "本模块评价已有预测，不选择模型、不校准区域、不证明模拟结果适用于实测。",
        },
    }


def save_evaluation(report, output_dir, predictions=()):
    """保存汇总、逐案例明细与预测；新目录写入，禁止覆盖。"""
    output = Path(output_dir)
    summary = {key: value for key, value in report.items() if key != "rows"}
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    predictions_text = "".join(json.dumps(p, ensure_ascii=False, allow_nan=False)+"\n" for p in predictions)
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(summary_text+"\n", encoding="utf-8")
    with (output/"per_case.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    with (output/"metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("method", "case_count", "source_count", "point_error_n", "median_error_m",
                         "p90_error_m", "direct_point_prediction_rate", "failure_or_missing_rate",
                         "region_mass", "coverage_denominator", "coverage", "median_region_area_m2"))
        for method, s in report["methods"].items():
            writer.writerow((method, s["case_count"], s["source_count"], s["position_error_m"]["n"],
                s["position_error_m"]["median"], s["position_error_m"]["p90"], s["direct_point_prediction_rate"],
                s["failure_or_missing_rate"], s["region_mass"], s["coverage"]["denominator"],
                s["coverage"]["rate"], s["region_area_on_true_direct_m2"]["median"]))
    if predictions_text:
        (output/"predictions.jsonl").write_text(predictions_text, encoding="utf-8")
    return output


def plot_evaluation(report, output_path):
    """导出静态 PNG；默认不调用，Matplotlib 只在要求画图时导入。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(report["methods"])
    summaries = [report["methods"][method] for method in methods]
    x = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for offset, statistic, label in ((-.18, "median", "Median"), (.18, "p90", "90th percentile")):
        values = [s["position_error_m"][statistic] for s in summaries]
        axes[0].bar(x+offset, [v if v is not None else np.nan for v in values], width=.36, label=label)
    axes[0].set(ylabel="Goal-plane distance error (m)", title="Conditional on available point predictions")
    axes[0].legend()
    rates = [s["direct_point_prediction_rate"] for s in summaries]
    axes[1].bar(x, [v if v is not None else np.nan for v in rates], width=.6)
    axes[1].set(ylabel="Fraction with point prediction", ylim=(0, 1.05),
                title="Denominator: all true direct crossings")
    for ax in axes:
        ax.set_xticks(x, methods)
    figure.suptitle(f"{report['split']} | See CSV/JSON for denominators and source counts")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="评价已有预测；默认演示开发集上的 G/A 基线。")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR/"synthetic_demo")
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--predictions", type=Path, nargs="+", help="一个或多个共同格式的预测 JSONL")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR/"evaluation_demo")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--plot", action="store_true", help="另存 PNG，需要 Matplotlib")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在，请指定新的 --output。")
    observations = load_observations(args.dataset/"observations"/f"{args.split}.jsonl")
    if args.predictions:
        predictions = [p for path in args.predictions for p in load_predictions(path)]
        note = "已提交预测的评价；方法选择和冻结须由研究流程另行记录。"
    else:
        # 先产生全部预测，再加载答案，防止基线预测函数访问真值。
        predictions = generate_baseline_predictions(observations)
        note = "仅 G/A 简单基线流程演示，未实现 S/F 反演，也未生成概率区域。"
    truths = read_jsonl(args.dataset/"truth"/f"{args.split}.jsonl")
    report = evaluate_predictions(observations, truths, predictions,
                                  bootstrap_samples=args.bootstrap, seed=args.seed)
    output = save_evaluation(report, args.output, predictions)
    if args.plot:
        plot_evaluation(report, output/"point_metrics.png")
    print(json.dumps({"output": str(output.resolve()), "split": report["split"], "note": note,
        "methods": {name: {"cases": s["case_count"], "sources": s["source_count"],
            "point_error_n": s["position_error_m"]["n"], "median_error_m": s["position_error_m"]["median"],
            "coverage": s["coverage"]["rate"]} for name, s in report["methods"].items()}},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
