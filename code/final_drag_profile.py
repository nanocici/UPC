"""Current-model drag refit profile for the statement case; no dataset/truth access.

Run from project root: .venv/bin/python code/final_drag_profile.py
An existing output directory is never overwritten. All results and source snapshots
are saved together under results/final_drag_profile by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from config import CASE, ENVIRONMENT, INFERENCE, PROJECT_ROOT, SOLVER
from final_config import FINAL
from inverse import fit_case
from physics import simulate


def variants():
    # S retains its current default marginal prior (matched to F0, not F1).
    return {"F_LS": ("F", "ls", INFERENCE),
            "S_LS": ("S", "ls", INFERENCE),
            "F0_MAP": ("F", "map", INFERENCE),
            "F1_MAP": ("F", "map", FINAL.inference_settings()),
            "S_MAP": ("S", "map", INFERENCE)}


def drag_grid():
    return sorted(set(np.linspace(*INFERENCE.drag_bounds, 25).tolist()
                      + [FINAL.drag_coefficient]))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                    encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def goal_values(simulation):
    if not simulation or simulation.get("event") != "goal":
        return {"y_m": None, "z_m": None, "time_s": None}
    p = simulation["event_position_m"]
    return {"y_m": p[1], "z_m": p[2], "time_s": simulation["event_time_s"]}


def fit_layer(variant, cd, strict=False):
    model, mode, settings = variants()[variant]
    solver = strict_solver() if strict else SOLVER
    fit = fit_case(model, settings=settings, solver=solver, mode=mode,
                   drag_coefficient=cd, sigma_m=FINAL.sigma_m, starts=9 if strict else 3)
    return {"variant": variant, "cd": cd, "strict": strict, "fit": fit}


def strict_solver():
    return replace(SOLVER, relative_tolerance=SOLVER.relative_tolerance/10,
                   position_absolute_tolerance_m=SOLVER.position_absolute_tolerance_m/10,
                   velocity_absolute_tolerance_m_s=SOLVER.velocity_absolute_tolerance_m_s/10,
                   max_step_s=SOLVER.max_step_s/2)


def flat_row(record):
    fit = record["fit"]
    diag = fit.get("diagnostics") or {}
    sim = fit.get("simulation") or {}
    row = {"variant": record["variant"], "cd": record["cd"], "success": fit["success"],
           "objective": fit["objective"], "data_chi_square": fit["data_chi_square"],
           "prior_penalty": fit["prior_penalty"],
           "pure_data_profile_R": fit["data_chi_square"] if fit["mode"] == "ls" else None,
           "rms_residual_m": fit["rms_residual_m"], "event": sim.get("event", "fit_failure"),
           **goal_values(sim), "boundary_hits": ";".join(diag.get("boundary_hits", [])),
           "optimality_residual": diag.get("optimality_residual"),
           "converged_starts": diag.get("converged_start_count", 0),
           "wall_s": fit["wall_time_s"]}
    p = fit.get("parameters") or {}
    for i, key in enumerate(("vx0_m_s", "vy0_m_s", "vz0_m_s")):
        row[key] = p["initial_velocity_m_s"][i] if p else None
    row.update(b_y=p.get("b_y"), b_z=p.get("b_z"))
    return row


def verification_indices(records):
    indices = {0, len(records)-1}
    indices.update(i for i, r in enumerate(records) if r["cd"] == FINAL.drag_coefficient)
    rows = [flat_row(r) for r in records]
    for key in ("objective", "y_m", "z_m", "time_s"):
        available = [i for i, r in enumerate(rows) if r[key] is not None]
        if available:
            indices.add(min(available, key=lambda i: rows[i][key]))
            indices.add(max(available, key=lambda i: rows[i][key]))
    for i, row in enumerate(rows):
        if not row["success"] or row["boundary_hits"] or row["event"] != "goal":
            indices.add(i)
        if i:
            prev = rows[i-1]
            if row["event"] != prev["event"]:
                indices.update((i-1, i))
            if row["y_m"] is not None and prev["y_m"] is not None:
                if np.hypot(row["y_m"]-prev["y_m"], row["z_m"]-prev["z_m"]) > .1:
                    indices.update((i-1, i))
    return sorted(indices)


def compare_fits(base, strict):
    a, b = base["fit"], strict["fit"]
    result = {"both_success": a["success"] and b["success"], "same_event": False,
              "objective_difference": None, "goal_shift_m": None, "time_shift_s": None,
              "pass": False}
    if not result["both_success"]:
        return result
    sa, sb = a["simulation"], b["simulation"]
    result["same_event"] = sa["event"] == sb["event"]
    difference = abs(a["objective"]-b["objective"])
    result["objective_difference"] = difference
    passed = result["same_event"] and difference <= 1e-5*max(1, abs(a["objective"]))
    if sa["event"] == sb["event"] == "goal":
        shift = float(np.linalg.norm(np.array(sa["event_position_m"])[1:]-np.array(sb["event_position_m"])[1:]))
        dt = abs(sa["event_time_s"]-sb["event_time_s"])
        result.update(goal_shift_m=shift, time_shift_s=dt)
        passed = passed and shift <= .001 and dt <= 1e-5
    else:
        # This experiment's goal-coordinate verification requires a confirmed crossing.
        passed = False
    result["pass"] = bool(passed)
    return result


def forward_scan(records, grid):
    output = []
    for variant in ("F0_MAP", "F1_MAP", "S_MAP"):
        nominal = next(r for r in records if r["variant"] == variant and r["cd"] == FINAL.drag_coefficient)
        for cd in grid:
            p = nominal["fit"].get("parameters")
            params = dict(p, drag_coefficient=cd) if p else None
            sim = simulate(variants()[variant][0], params, CASE.times_s) if params else None
            chi = None
            if sim and all(pos is not None for pos in sim["positions_m"]):
                chi = float(np.sum(((np.asarray(sim["positions_m"])-CASE.positions_m)/FINAL.sigma_m)**2))
            output.append({"variant": variant, "cd": cd, "nominal_cd": FINAL.drag_coefficient,
                           "parameters": params, "simulation": sim, "data_chi_square": chi,
                           "note": "No refit; all parameters except Cd held at nominal MAP."})
    return output


def summarize(records, checks, scans):
    groups = {}
    for variant in variants():
        rs = [r for r in records if r["variant"] == variant]
        rows = [flat_row(r) for r in rs]
        valid = [r for r in rows if r["success"]]
        ranges = {}
        for key in ("y_m", "z_m", "time_s", "data_chi_square", "objective"):
            vals = [r[key] for r in valid if r[key] is not None]
            ranges[key] = {"min": min(vals), "max": max(vals), "span": max(vals)-min(vals)} if vals else None
        best = min(valid, key=lambda r: r["objective"]) if valid else None
        nominal = next(r for r in rows if r["cd"] == FINAL.drag_coefficient)
        vv = [r for r in checks if r["variant"] == variant]
        groups[variant] = {"count": len(rs), "success_count": len(valid),
                           "events": dict(Counter(r["event"] for r in rows)),
                           "boundary_count": sum(bool(r["boundary_hits"]) for r in rows),
                           "ranges": ranges, "grid_minimum_cd": best["cd"] if best else None,
                           "nominal": nominal, "verification_count": len(vv),
                           "verification_pass_count": sum(r["comparison"]["pass"] for r in vv)}
    # LS optimized residual should not exceed residual of any feasible fixed-parameter trajectory.
    comparisons = []
    for scan in scans:
        if scan["data_chi_square"] is None:
            continue
        label = "S_LS" if scan["variant"] == "S_MAP" else "F_LS"
        ls = next(r for r in records if r["variant"] == label and r["cd"] == scan["cd"])
        if ls["fit"]["success"]:
            delta = ls["fit"]["data_chi_square"]-scan["data_chi_square"]
            comparisons.append({"variant": scan["variant"], "cd": scan["cd"],
                                "ls_minus_forward_chi_square": delta, "pass": delta <= 1e-5})
    return {"groups": groups, "fit_count": len(records), "forward_count": len(scans),
            "verification_count": len(checks),
            "verification_pass_count": sum(r["comparison"]["pass"] for r in checks),
            "ls_vs_forward_checks": comparisons,
            "ls_vs_forward_all_pass": all(r["pass"] for r in comparisons)}


def plot_results(records, scans, output):
    rows = [flat_row(r) for r in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for variant in variants():
        rr = [r for r in rows if r["variant"] == variant]
        x = [r["cd"] for r in rr]
        if variant.endswith("LS"):
            axes[0].plot(x, [r["data_chi_square"] for r in rr], label=variant)
        else:
            axes[1].plot(x, [r["objective"] for r in rr], label=variant)
            axes[2].plot(x, [.5*r["data_chi_square"] if r["success"] else np.nan for r in rr], label=variant+" data/2")
            axes[2].plot(x, [r["prior_penalty"] for r in rr], "--", label=variant+" prior")
    for ax, title in zip(axes, ("Pure data profile (LS)", "Penalized profile (MAP)", "MAP objective components")):
        ax.set(xlabel="Drag coefficient Cd", title=title)
        ax.axvline(FINAL.drag_coefficient, color="gray", linestyle=":")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(output/"fit_profiles.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for variant in variants():
        rr = [r for r in rows if r["variant"] == variant]
        for ax, key in zip(axes, ("y_m", "z_m", "time_s")):
            line, = ax.plot([r["cd"] for r in rr], [r[key] for r in rr], label=variant+" refit")
            ss = [r for r in scans if r["variant"] == variant]
            if ss:
                ax.plot([r["cd"] for r in ss], [goal_values(r["simulation"])[key] for r in ss],
                        "--", color=line.get_color(), alpha=.65, label=variant+" fixed")
    for ax, label in zip(axes, ("Goal-plane Y (m)", "Goal-plane Z (m)", "Arrival time (s)")):
        ax.set(xlabel="Drag coefficient Cd", ylabel=label)
        ax.axvline(FINAL.drag_coefficient, color="gray", linestyle=":")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(output/"goal_profiles.png", dpi=180)
    plt.close(fig)


def write_report(summary, output):
    lines = ["# 当前模型阻力重拟合剖面实验", "", "日期：2026-11-08（Asia/Shanghai）。", "",
             "仅使用题面指定算例，不读取开发、校准、测试数据，不修改模型或校准参数。",
             "沿用原方案：Cd∈[0.08,0.55] 的25个等距节点，加0.22，共26层；基础每层3起点。", "",
             "## 模型与口径", "",
             "- F0：现有默认F，by/bz先验尺度0.20/0.20；F1：现有最终F，0.10/0.20。",
             "- S：现有默认S，by=0，使用F0圆盘高斯的bz边缘先验；不是另造一个匹配F1的S。",
             "- F0/F1的LS数学目标相同，统一运行F_LS；S另跑S_LS；MAP运行F0/F1/S。",
             "- LS数据卡方才是纯数据剖面R(Cd)。MAP目标=数据卡方/2+先验罚项；不包含Cd先验，也不估计Cd。",
             "- 固定参数前向对照：固定各模型Cd=0.22的MAP初速度和横向参数，只改变Cd，不重新拟合。",
             "- 所有门线点均为拟合轨迹点，不是KDE平滑峰；event=goal表示过平面，不等于入框。", "",
             "## 重拟合结果", "",
             "| 组别 | 成功/总数 | 网格目标极小Cd | Y范围/m | Z范围/m | T范围/s | 数据卡方范围 |",
             "|---|---:|---:|---|---|---|---|"]
    def interval(v):
        return "缺失" if v is None else f"{v['min']:.6f}–{v['max']:.6f}"
    for label, group in summary["groups"].items():
        rr = group["ranges"]
        lines.append(f"| {label} | {group['success_count']}/{group['count']} | {group['grid_minimum_cd']} | "+
                     " | ".join(interval(rr[k]) for k in ("y_m", "z_m", "time_s", "data_chi_square"))+" |")
    lines += ["", "## 名义Cd=0.22", "",
              "| 组别 | Y/m | Z/m | T/s | 数据卡方 | 先验罚项 |", "|---|---:|---:|---:|---:|---:|"]
    for label, group in summary["groups"].items():
        r = group["nominal"]
        lines.append("| "+label+" | "+" | ".join("缺失" if r[k] is None else f"{r[k]:.6f}" for k in
                     ("y_m", "z_m", "time_s", "data_chi_square", "prior_penalty"))+" |")
    lines += ["", "## 数值复核", "",
              "按端点、名义层、目标及Y/Z/T极值、失败/边界、事件变化和相邻门线跳变>0.1m选层。",
              "复核每层9起点，积分相对/绝对容差收紧10倍、最大步长减半；保持当前优化认证门槛。",
              "通过标准：两次拟合均成功、均确认过线；目标差≤1e-5·max(1,|基础目标|)，门线差≤0.001m，时间差≤1e-5s。",
              f"- 基础拟合：{summary['fit_count']}次；固定参数对照：{summary['forward_count']}条。",
              f"- 加严复核：{summary['verification_pass_count']}/{summary['verification_count']}通过；未通过记录保留，未替换基础结果。",
              f"- LS残差不劣于固定参数前向对照：{summary['ls_vs_forward_all_pass']}。", "",
              "## 结论与限制", ""]
    f1 = summary["groups"]["F1_MAP"]["ranges"]
    if all(f1[k] for k in ("y_m", "z_m", "time_s")):
        lines.append(f"- 当前F1在整个扫描范围重拟合后的Y跨度{f1['y_m']['span']:.6f}m，Z跨度{f1['z_m']['span']:.6f}m，时间跨度{1000*f1['time_s']['span']:.3f}ms。这是条件敏感性，不是总预测误差或置信区间。")
    lines += ["- 离散网格目标极小点只说明本范围、本目标的网格最优层，不代表真实阻力已被识别；先验罚项下的MAP最小点不能冒充纯数据识别。",
              "- 若LS数据剖面较平、多个Cd拟合相近，说明观测对Cd约束有限；本实验不构造Cd置信区间，也不能证明Cd=0.22是真实值。",
              "- 固定Cd是条件建模取舍；本实验只量化该取舍的敏感性，不能独立证明模型优于其他预测方法。",
              "- 不将扫描层等权混合作为后验；不把全扫描范围当90%区域；不评价空间覆盖率或最终KDE主点误差。",
              "- 数值复核支持有限层的计算稳定性，不证明全局唯一、真实精度或通用性能。", "",
              "## 图表", "", "![拟合剖面](fit_profiles.png)", "", "![门线剖面](goal_profiles.png)", "",
              "## 复现", "", "从项目根目录运行（输出目录须不存在）：", "",
              "```bash", ".venv/bin/python code/final_drag_profile.py --output results/final_drag_profile_rerun", "```", "",
              "源码快照位于source/；输入、全部变体设置、阈值及SHA256见run_config.json。"]
    (output/"FINAL_DRAG_PROFILE_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    grid = drag_grid()
    source = output/"source"
    source.mkdir()
    hashes = {}
    for name in ("final_drag_profile.py", "config.py", "final_config.py", "inverse.py", "physics.py", "data.py"):
        path = Path(__file__).with_name(name)
        shutil.copy2(path, source/name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output/"run_config.json", {
        "date": "2026-11-08 Asia/Shanghai", "case": asdict(CASE), "environment": asdict(ENVIRONMENT),
        "solver": asdict(SOLVER), "strict_solver": asdict(strict_solver()), "sigma_m": FINAL.sigma_m,
        "grid": grid, "base_starts": 3, "verification_starts": 9,
        "variants": {k: {"model": m, "mode": mode, "settings": asdict(s)} for k, (m, mode, s) in variants().items()},
        "verification": {"objective_relative_max1": 1e-5, "goal_shift_m": .001, "time_shift_s": 1e-5,
                         "selection": "endpoints, nominal, objective/Y/Z/T extrema, failures/boundaries, event changes, adjacent shifts >0.1m"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "source_sha256": hashes, "no_dataset_access": True,
        "S_definition": "current default S; marginal matched to F0"})
    records, checks = [], []
    for variant in variants():
        group = []
        for cd in grid:
            record = fit_layer(variant, cd)
            append_json(output/"fits.jsonl", record)
            group.append(record)
        records += group
        print(f"{variant}: {sum(r['fit']['success'] for r in group)}/{len(group)} base fits", flush=True)
        for i in verification_indices(group):
            strict = fit_layer(variant, group[i]["cd"], strict=True)
            record = {"variant": variant, "cd": group[i]["cd"], "strict_result": strict,
                      "comparison": compare_fits(group[i], strict)}
            append_json(output/"verification.jsonl", record)
            checks.append(record)
    scans = forward_scan(records, grid)
    for record in scans:
        append_json(output/"forward_scan.jsonl", record)
    rows = [flat_row(r) for r in records]
    with (output/"profile.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(records, checks, scans)
    plot_results(records, scans, output)
    summary["wall_s"] = time.perf_counter()-begun
    write_json(output/"summary.json", summary)
    write_report(summary, output)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    write_json(output/"artifact_manifest.json", artifacts)
    print(json.dumps({"output": str(output), "fits": len(records), "verification_pass": summary["verification_pass_count"],
                      "verification_total": len(checks), "wall_s": summary["wall_s"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT/"results"/"final_drag_profile")
    args = parser.parse_args()
    run(args.output)
