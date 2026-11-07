"""最小统一入口：调用已有命令行模块，记录设置、日志与结果。

默认仅运行题目 F 算例。development 模式需显式选择，且只处理开发组。
本文件不增加模型、调参、校准或测试集实验。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import (CASE, DATA_DIR, ENVIRONMENT, INFERENCE, PROJECT_ROOT,
                    RANDOM_SEED, RESULTS_DIR, SOLVER, validate_config)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def build_steps(output, *, task="example", model="F", samples=2000,
                starts=3, seed=RANDOM_SEED, drag=None, sigma=None, dataset=DATA_DIR/"synthetic_demo", plot=False):
    """命令都使用当前 Python 和绝对脚本路径，以列表传递，不使用 shell。"""
    if task not in ("example", "development") or model not in ("S", "F", "both"):
        raise ValueError("未知任务或模型。")
    output, dataset = Path(output).resolve(), Path(dataset).resolve()
    python = sys.executable
    common = ["--model", model, "--starts", str(starts)]
    if drag is not None:
        common += ["--drag", str(drag)]
    if sigma is not None:
        common += ["--sigma", str(sigma)]
    inverse = [python, str(PROJECT_ROOT/"code/inverse.py"), *common]
    if task == "example":
        return [
            ("inverse", [*inverse, "--output", str(output/"inverse")]),
            ("uncertainty", [python, str(PROJECT_ROOT/"code/uncertainty.py"), *common,
                "--samples", str(samples), "--seed", str(seed), "--output", str(output/"uncertainty"),
                *(["--plot"] if plot else [])]),
        ]
    # 第一次整合仅提供已有点预测的开发模式，不自动生成数据、不访问校准/测试组。
    development = ["--dataset", str(dataset), "--split", "development"]
    return [
        ("inverse", [*inverse, *development, "--output", str(output/"inverse")]),
        ("baselines", [python, str(PROJECT_ROOT/"code/evaluate.py"), *development,
            "--seed", str(seed), "--output", str(output/"baselines")]),
        ("evaluation", [python, str(PROJECT_ROOT/"code/evaluate.py"), *development,
            "--predictions", str(output/"inverse/predictions.jsonl"), str(output/"baselines/predictions.jsonl"),
            "--seed", str(seed), "--output", str(output/"evaluation"), *(["--plot"] if plot else [])]),
    ]


def execute_step(command, log_path):
    """stdout/stderr 写同一 UTF-8 日志，Windows 下也显式指定子进程编码。"""
    environment = dict(os.environ)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with Path(log_path).open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def example_review(output):
    """复述模块已有的诊断，不额外改变判定规则；题目无真值，不计算定位误差。"""
    fits = json.loads((output/"inverse/example.json").read_text(encoding="utf-8"))
    posteriors = json.loads((output/"uncertainty/summary.json").read_text(encoding="utf-8"))
    review = []
    models = []
    for fit in fits:
        if not fit["success"]:
            review.append(f"{fit['model']} 参数反演未收敛。")
    for posterior in posteriors:
        model = posterior.get("model")
        direct = (posterior.get("summary") or {}).get("direct_crossing")
        info = {"model": model, "sampling_diagnostics_passed": posterior["success"],
                "mean_yz_m": direct["mean_yz_m"] if direct else None,
                "mode_stable": direct.get("mode_stable") if direct else None,
                "message": posterior["message"]}
        models.append(info)
        if not posterior["success"]:
            review.append(f"{model} 不确定性诊断未通过，查看 uncertainty/summary.json。")
        elif direct and not direct.get("mode_stable", False):
            review.append(f"{model} 格网众数尚不稳定；均值/区域通过不代表最可能点稳定。")
    return {"models": models, "review_items": review,
            "note": "题目单算例没有门线真值，本次仅检查流程接通，不进行误差评价或最终结论判定。"}


def run_pipeline(output, *, task="example", model="F", samples=2000,
                 starts=3, seed=RANDOM_SEED, drag=None, sigma=None,
                 dataset=DATA_DIR/"synthetic_demo", plot=False):
    validate_config()
    if not isinstance(samples, int) or samples < 2 or not isinstance(starts, int) or not 1 <= starts <= 9:
        raise ValueError("samples 至少为 2，starts 在 1–9 之间。")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 须为非负整数。")
    output = Path(output).resolve()
    steps = build_steps(output, task=task, model=model, samples=samples, starts=starts,
                        seed=seed, drag=drag, sigma=sigma, dataset=dataset, plot=plot)
    if task == "development":
        for kind in ("observations", "truth"):
            if not (Path(dataset)/kind/"development.jsonl").is_file():
                raise ValueError("开发数据缺失，请先用 data.py 准备所选目录。")
    # 记录配置及本次脚本指纹，便于核对不同运行；不覆盖已有结果目录。
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = {"task": task, "model": model, "samples": samples, "starts": starts,
        "seed": seed, "drag_override": drag, "sigma_override_m": sigma, "plot": plot,
        "dataset": str(Path(dataset).resolve()) if task == "development" else None,
        "split": "development" if task == "development" else None,
        "case": asdict(CASE), "environment": asdict(ENVIRONMENT), "solver": asdict(SOLVER),
        "inference": asdict(INFERENCE), "python_executable": sys.executable,
        "versions": versions, "steps": [{"name": name, "command": command} for name, command in steps],
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((PROJECT_ROOT/"code").glob("*.py"))},
        "purpose": "流程运行与记录；不执行模型选择、敏感性扫描、校准或独立测试。"}
    # 先序列化，非法数值不写入半份设置。
    json.dumps(config, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    (output/"logs").mkdir()
    _write_json(output/"run_config.json", config)
    manifest = {"status": "running", "task": task, "output": str(output), "steps": [], "review_items": []}
    _write_json(output/"run_summary.json", manifest)
    start = time.perf_counter()
    for index, (name, command) in enumerate(steps, 1):
        log_path = output/"logs"/f"{index:02d}_{name}.log"
        print(f"[{index}/{len(steps)}] {name} ... log: {log_path}", flush=True)
        stage_start = time.perf_counter()
        try:
            returncode = execute_step(command, log_path)
            error = None
        except OSError as exception:
            returncode, error = -1, str(exception)
            log_path.write_text(str(exception)+"\n", encoding="utf-8")
        manifest["steps"].append({"name": name, "returncode": returncode,
            "wall_time_s": time.perf_counter()-stage_start, "log": str(log_path.relative_to(output)),
            "status": "completed" if returncode == 0 else "failed", "error": error})
        manifest["wall_time_s"] = time.perf_counter()-start
        if returncode != 0:
            manifest.update(status="failed", message=f"{name} 执行失败，后续步骤未运行，查看对应日志。")
            _write_json(output/"run_summary.json", manifest)
            print(manifest["message"], file=sys.stderr)
            return manifest
        _write_json(output/"run_summary.json", manifest)
        print(f"[{index}/{len(steps)}] {name}: completed", flush=True)
    try:
        if task == "example":
            manifest.update(example_review(output))
        else:
            evaluation = json.loads((output/"evaluation/summary.json").read_text(encoding="utf-8"))
            manifest.update(methods=list(evaluation["methods"]),
                            note="开发组点预测流程；概率区域的批量评价由后续实验安排。")
    except (OSError, KeyError, ValueError, TypeError) as error:
        manifest.update(status="failed", message=f"输出汇总失败：{error}")
        _write_json(output/"run_summary.json", manifest)
        return manifest
    manifest.update(status="completed", wall_time_s=time.perf_counter()-start,
                    diagnostics_status="review_required" if manifest["review_items"] else "no_reported_issues")
    _write_json(output/"run_summary.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="最小统一入口，默认仅跑题目 F 算例。")
    parser.add_argument("--task", choices=("example", "development"), default="example")
    parser.add_argument("--model", choices=("S", "F", "both"), default="F")
    parser.add_argument("--samples", type=int, default=2000, help="不确定性基础 N，实际提议数为 2N")
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--drag", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--dataset", type=Path, default=DATA_DIR/"synthetic_demo")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    required = ("numpy", "scipy")+(("matplotlib",) if args.plot else ())
    if any(importlib.util.find_spec(package) is None for package in required):
        parser.error("缺少依赖，请先用当前 Python 执行 -m pip install -r requirements.txt。")
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or RESULTS_DIR/f"run_{args.task}_{timestamp}"
    try:
        result = run_pipeline(output, task=args.task, model=args.model, samples=args.samples,
            starts=args.starts, seed=args.seed, drag=args.drag, sigma=args.sigma, dataset=args.dataset, plot=args.plot)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
