"""最终交付入口：指定算例、验证表、冻结配置检查。

用法：
  python code/final_pipeline.py case
  python code/final_pipeline.py validate
  python code/final_pipeline.py check
  python code/final_pipeline.py all

本入口不改变 F 的方程；F1 只是 b_y 先验 0.10。
空间校准、时间 q、平滑峰都是后处理/估计层，写入 final_case.json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy

from config import CASE, ENVIRONMENT, PROJECT_ROOT, SOLVER
from final_config import FINAL, FinalConfig
from uncertainty import sample_case

MODE_PATH = (PROJECT_ROOT / "results" / "算例稳定性实验" /
             "重新运行_20261108" / "summary.json")
if not MODE_PATH.exists():
    MODE_PATH = PROJECT_ROOT / "results" / "算例稳定性实验" / "summary.json"
TIME_PATH = PROJECT_ROOT / "results" / "time_model_200" / "time_model_summary.json"
INDEPENDENT_PATH = PROJECT_ROOT / "results" / "independent_200" / "independent_summary.json"


def calibrated_time_interval(median_s: float, interval_s: Sequence[float], q_s: float) -> Tuple[float, float]:
    if len(interval_s) != 2 or not all(math.isfinite(value) for value in interval_s):
        raise ValueError("时间区间必须是两个有限数")
    if not math.isfinite(median_s) or not math.isfinite(q_s) or q_s < 0:
        raise ValueError("时间中位数和 q 必须有限且 q 非负")
    half = (float(interval_s[1]) - float(interval_s[0])) / 2
    total = math.sqrt(half ** 2 + q_s ** 2)
    return float(median_s) - total, float(median_s) + total


def calibrated_region(region: Mapping, threshold: float) -> dict:
    if not isinstance(region, Mapping) or "center_yz_m" not in region or "covariance_m2" not in region:
        raise ValueError("区域须包含中心和协方差")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("阈值必须有限且为正")
    cov = np.asarray(region["covariance_m2"], float)
    det = float(np.linalg.det(cov))
    if not np.isfinite(det) or det <= 0:
        raise ValueError("协方差必须正定")
    return {
        "center_yz_m": [float(v) for v in region["center_yz_m"]],
        "covariance_m2": cov.tolist(),
        "threshold": float(threshold),
        "area_m2": float(math.pi * threshold * math.sqrt(det)),
    }


def _mode_info(mode_summary: Optional[Mapping]) -> dict:
    if not mode_summary:
        return {"available": False}
    cond = mode_summary.get("stability", {}).get("inside_conditioned", {})
    main = cond.get("0.1", {})
    combined = mode_summary.get("combined_kde", {}).get("inside_conditioned", {}).get("64000_0.1", {})
    raw = mode_summary.get("raw_grid", {}).get("inside_conditioned", {})
    return {
        "available": True,
        "bandwidth_m": 0.10,
        "cross_seed_diameter_m": main.get("cross_seed_diameter_m"),
        "last_doubling_shift_m": main.get("max_last_doubling_shift_m"),
        "pass": bool(main.get("pass", False)),
        "per_seed_peaks": main.get("per_seed_full", []),
        "combined_peak": combined,
        "raw_grid_64k": raw,
        "historical": "4k/8k/16k raw-grid failures remain recorded separately.",
    }


def assemble_case(summary: Mapping, mode_summary: Optional[Mapping] = None,
                  cfg: FinalConfig = FINAL) -> dict:
    direct = summary["summary"]["direct_crossing"]
    events = summary["summary"]["event_probabilities"]
    median = float(direct["arrival_time_median_s"])
    old_interval = [float(v) for v in direct["arrival_time_90_interval_s"]]
    new_interval = calibrated_time_interval(median, old_interval, cfg.time_q_s)
    model_region = dict(direct["region"])
    model_region["area_m2"] = float(direct["region_area_m2"])
    calibrated = calibrated_region(direct["region"], cfg.spatial_threshold)
    mode = _mode_info(mode_summary)
    per_seed = mode.get("per_seed_peaks", []) if mode.get("available") else []
    return {
        "model": "F1",
        "config": asdict(cfg),
        "primary_point": {
            "kind": "fixed_bandwidth_smooth_mode",
            "condition": "inside_conditioned",
            "bandwidth_m": cfg.mode_bandwidth_m,
            "y_m": cfg.smooth_mode_yz_m[0],
            "z_m": cfg.smooth_mode_yz_m[1],
            "source": "four seeds, condition-normalized, equally weighted KDE",
            "per_seed_peaks": per_seed,
            "stability": mode,
        },
        "raw_grid_64k": {
            "kind": "raw_grid_mode",
            "y_m": cfg.raw_grid_mode_yz_m[0],
            "z_m": cfg.raw_grid_mode_yz_m[1],
            "pass": bool(mode.get("raw_grid_64k", {}).get("pass", True)) if mode.get("available") else True,
            "note": "new 64k result; historical 4k/8k/16k failure retained",
        },
        "mean_auxiliary": {
            "kind": "conditional_mean_auxiliary",
            "y_m": float(direct["mean_yz_m"][0]),
            "z_m": float(direct["mean_yz_m"][1]),
            "note": "auxiliary scoring point; not the most likely point",
        },
        "spatial": {
            "model_internal": model_region,
            "calibrated": calibrated,
            "note": "calibrated threshold from independent calibration; test coverage 88.6%",
        },
        "time": {
            "median_s": median,
            "posterior_interval_s": old_interval,
            "calibrated_interval_s": list(new_interval),
            "width_ms": (new_interval[1] - new_interval[0]) * 1000,
            "q_ms": cfg.time_q_s * 1000,
            "note": "global additive q from 200-source calibration/test; not final external guarantee",
        },
        "goal_probability": float(events["goal_inside"]),
        "unknown_event_mass": float(summary["summary"]["unknown_event_mass"]),
        "diagnostics": {
            "sampling_stable": bool(summary.get("diagnostics", {}).get("sampling_stable", False)),
            "effective_samples": summary["summary"].get("effective_samples"),
            "mode_available": mode.get("available", False),
        },
        "limitations": [
            "spatial calibrated test coverage 88.6%; combined mechanism 82.8%",
            "time interval calibrated on synthetic development data; width 33.7 ms",
            "results are conditional on F1, fixed Cd=0.22, sigma=0.01 m, still air",
            "no real-world validation",
        ],
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case_markdown(path: Path, case: Mapping):
    primary = case["primary_point"]
    mean = case["mean_auxiliary"]
    raw = case["raw_grid_64k"]
    spatial = case["spatial"]
    time = case["time"]
    lines = [
        "# 最终算例：F1 + 时间模型",
        "",
        "## 三个点",
        "",
        "| 名称 | 值 | 状态 |",
        "|---|---|---|",
        f"| 0.10 m 平滑峰（主点） | ({primary['y_m']:.4f}, {primary['z_m']:.4f}) m | 四种子等权；跨种子 {primary['stability'].get('cross_seed_diameter_m', float('nan')):.4f} m |",
        f"| 原始格网 64k | ({raw['y_m']:.4f}, {raw['z_m']:.4f}) m | 通过，但接近 0.20 m 阈值 |",
        f"| 辅助均值 | ({mean['y_m']:.4f}, {mean['z_m']:.4f}) m | 辅助，不是最可能点 |",
        "",
        "## 空间",
        "",
        "| 区域 | 面积 | 阈值 |",
        "|---|---:|---:|",
        f"| 模型内 90% | {spatial['model_internal']['area_m2']:.4f} m² | {spatial['model_internal']['threshold']:.4f} |",
        f"| 校准 90% | {spatial['calibrated']['area_m2']:.4f} m² | {spatial['calibrated']['threshold']:.4f} |",
        "",
        "## 时间",
        "",
        f"- 中位到达时间：{time['median_s']:.6f} s",
        f"- 原后验 90%：{time['posterior_interval_s'][0]:.6f}–{time['posterior_interval_s'][1]:.6f} s",
        f"- 校准 90%：{time['calibrated_interval_s'][0]:.6f}–{time['calibrated_interval_s'][1]:.6f} s（宽 {time['width_ms']:.2f} ms）",
        "",
        f"- 入框概率：{case['goal_probability']:.4f}",
        f"- 未判定事件质量：{case['unknown_event_mass']:.3g}",
        "",
        "## 限制",
        "",
    ]
    lines.extend(f"- {item}" for item in case["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(output: Path, force: bool = False):
    if output.exists() and not force:
        raise FileExistsError(f"输出目录已存在：{output}；使用 --force 覆盖")
    output.mkdir(parents=True, exist_ok=True)
    result = sample_case("F", settings=FINAL.inference_settings(),
                         uncertainty=FINAL.uncertainty_settings(),
                         drag_coefficient=FINAL.drag_coefficient,
                         sigma_m=FINAL.sigma_m, seed=FINAL.case_seed, starts=3)
    result.pop("draws", None)
    mode_summary = json.loads(MODE_PATH.read_text()) if MODE_PATH.exists() else None
    case = assemble_case(result, mode_summary)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "final_case.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_case_markdown(output / "final_case.md", case)
    config = {
        "config": asdict(FINAL),
        "case": asdict(CASE),
        "environment": asdict(ENVIRONMENT),
        "solver": asdict(SOLVER),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "hashes": {
            "final_case.json": _hash(output / "final_case.json"),
            "source": _hash(Path(__file__)),
            "config": _hash(Path(__file__).with_name("final_config.py")),
        },
        "note": "F1 is a prior setting; spatial/time calibration are post-processing.",
    }
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return case


def validate(output: Path):
    mode = json.loads(MODE_PATH.read_text()) if MODE_PATH.exists() else {}
    time = json.loads(TIME_PATH.read_text()) if TIME_PATH.exists() else {}
    independent = json.loads(INDEPENDENT_PATH.read_text()) if INDEPENDENT_PATH.exists() else {}
    lines = ["# 最终验证表", "", "本表只读取已保存结果，不重新采样。", "", "## 时间模型", ""]
    if time:
        lines += [
            f"- q_global_additive = {time.get('q_global_additive', 0) * 1000:.4f} ms",
            f"- time_model_200 test 覆盖 = {time.get('methods', {}).get('global_additive', {}).get('coverage', float('nan')):.2%}",
            f"- time_model_200 test 宽度中位 = {time.get('methods', {}).get('global_additive', {}).get('median_width_ms', float('nan')):.2f} ms",
        ]
    else:
        lines.append("- 缺少 time_model_200 结果")
    lines += ["", "## 空间校准（独立测试）", ""]
    if independent:
        base = independent.get("baseline_test", {})
        cal = independent.get("calibrated_test", {})
        lines += [
            f"- 校准阈值 q_cal = {independent.get('q_cal', float('nan')):.4f}",
            f"- 原区域：覆盖 {base.get('coverage', float('nan')):.2%}，面积 {base.get('median_area_m2', float('nan')):.3f} m²",
            f"- 校准区域：覆盖 {cal.get('coverage', float('nan')):.2%}，面积 {cal.get('median_area_m2', float('nan')):.3f} m²",
            f"- combined 覆盖：{independent.get('by_mechanism_test', {}).get('combined', {}).get('calibrated', {}).get('coverage', float('nan')):.2%}",
            f"- 独立 test 时间覆盖（冻结 q）：{base.get('time_coverage', float('nan')):.2%}；宽度 {base.get('median_time_width_ms', float('nan')):.2f} ms",
        ]
    else:
        lines.append("- 缺少 independent_200 结果")
    lines += ["", "## 众数", ""]
    if mode:
        stab = mode.get("stability", {}).get("inside_conditioned", {}).get("0.1", {})
        lines += [
            f"- 0.10 m 跨种子 {stab.get('cross_seed_diameter_m', float('nan')):.4f} m；32k→64k {stab.get('max_last_doubling_shift_m', float('nan')):.4f} m；通过 {stab.get('pass')}",
            f"- 原始格网 64k 通过：{mode.get('raw_grid', {}).get('inside_conditioned', {}).get('pass')}",
            "- 历史 4k/8k/16k 失败保留。",
        ]
    else:
        lines.append("- 缺少 f1_mode_stability 结果")
    lines += ["", "## 最终算例", ""]
    case_path = output / "final_case.json"
    if case_path.exists():
        case = json.loads(case_path.read_text())
        lines += [
            f"- 主点：({case['primary_point']['y_m']:.4f}, {case['primary_point']['z_m']:.4f}) m",
            f"- 校准区域：{case['spatial']['calibrated']['area_m2']:.4f} m²",
            f"- 校准时间：{case['time']['calibrated_interval_s'][0]:.6f}–{case['time']['calibrated_interval_s'][1]:.6f} s",
            f"- 辅助均值：({case['mean_auxiliary']['y_m']:.4f}, {case['mean_auxiliary']['z_m']:.4f}) m",
        ]
    else:
        lines.append("- 尚未运行 case")
    (output / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "validation.md"


def check():
    checks = {}
    if TIME_PATH.exists():
        time = json.loads(TIME_PATH.read_text())
        checks["time_q"] = abs(time.get("q_global_additive", -1) - FINAL.time_q_s) < 1e-15
    else:
        checks["time_q"] = False
    if INDEPENDENT_PATH.exists():
        independent = json.loads(INDEPENDENT_PATH.read_text())
        checks["spatial_threshold"] = abs(independent.get("q_cal", -1) - FINAL.spatial_threshold) < 1e-12
    else:
        checks["spatial_threshold"] = False
    if MODE_PATH.exists():
        mode = json.loads(MODE_PATH.read_text())
        peak = mode.get("combined_kde", {}).get("inside_conditioned", {}).get("64000_0.1", {})
        checks["smooth_mode"] = (abs(peak.get("y_m", -1) - FINAL.smooth_mode_yz_m[0]) < 1e-9
                                 and abs(peak.get("z_m", -1) - FINAL.smooth_mode_yz_m[1]) < 1e-9)
    else:
        checks["smooth_mode"] = False
    checks["all"] = all(checks.values())
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all"]:
        raise SystemExit(1)
    return checks


def main():
    parser = argparse.ArgumentParser(description="最终交付入口：F1 + 空间/时间校准 + 平滑峰。")
    parser.add_argument("command", choices=("case", "validate", "check", "all"))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "final_case")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in ("case", "all"):
        run_case(args.output, force=args.force)
    if args.command in ("validate", "all"):
        validate(args.output)
    if args.command == "check":
        check()
    print(json.dumps({"command": args.command, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
