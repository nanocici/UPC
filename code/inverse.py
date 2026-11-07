"""固定阻力的两帧反演：调用 physics，不读取生成真值。

fit_case 为单案例接口，fit_observation 将结果包装成 evaluate 的预测格式。
默认使用 MAP；--mode ls 明确去掉正则项。概率传播与阻力积分另行实现。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from math import isfinite, sqrt
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import erf

from config import (CASE, ENVIRONMENT, INFERENCE, SOLVER, DATA_DIR, RESULTS_DIR,
                    F_PARAMETER_NAMES, S_PARAMETER_NAMES, Environment)
from data import load_observations, observation_case, SPLITS
from physics import gravity_initial_velocity, simulate


def observation_covariance(case, sigma_m=None, frame_correlation=0.0, covariance_m2=None):
    """坐标按 [x1,y1,z1,x2,y2,z2,...] 排列，返回正定协方差。"""
    n = len(case.times_s)
    if covariance_m2 is None:
        sigma = INFERENCE.position_sigma_m if sigma_m is None else float(sigma_m)
        if not isfinite(sigma) or sigma <= 0 or not isfinite(frame_correlation) or not 0 <= frame_correlation < 1:
            raise ValueError("高斯误差须有正标准差，跨帧相关系数须在 [0,1) 内。")
        frame_matrix = (1-frame_correlation)*np.eye(n)+frame_correlation*np.ones((n, n))
        covariance = sigma**2*np.kron(frame_matrix, np.eye(3))
    else:
        covariance = np.asarray(covariance_m2, dtype=float)
    if (covariance.shape != (3*n, 3*n) or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-16)):
        raise ValueError("观测协方差尺寸不符、非有限或不对称。")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("观测协方差须正定；奇异精确约束暂不支持。") from error
    return covariance


def transverse_prior_penalty(model, transverse, settings=INFERENCE):
    """负对数先验相对各自零点的增量；不用于计算模型证据。

    F 是圆盘截断的二维各向同性高斯；S 用其 b_z 边缘先验，
    而不是条件于 b_y=0 的一维截断高斯。
    """
    transverse = np.asarray(transverse, dtype=float)
    expected = (1,) if model == "S" else (2,)
    if model not in ("S", "F") or transverse.shape != expected or not np.all(np.isfinite(transverse)):
        raise ValueError("横向参数形状或模型不符。")
    radius, scale = settings.transverse_norm_max, settings.transverse_prior_std
    norm = float(np.linalg.norm(transverse))
    if norm > radius or (model == "S" and norm >= radius):
        return float("inf")
    penalty = .5*float(np.dot(transverse, transverse))/scale**2
    if model == "S":
        width = sqrt(max(0.0, radius**2-transverse[0]**2))
        # 常数选择使零点罚项为 0；它不改变该模型中的 MAP。
        penalty += np.log(erf(radius/(sqrt(2)*scale)))-np.log(erf(width/(sqrt(2)*scale)))
    return float(penalty)


def _parameter_dict(model, vector, drag_coefficient):
    return {"initial_velocity_m_s": tuple(float(x) for x in vector[:3]),
            "drag_coefficient": float(drag_coefficient),
            "b_y": float(vector[3]) if model == "F" else 0.0,
            "b_z": float(vector[-1])}


def _finite_jacobian(function, q, bounds, base):
    """对缩放后的参数差分，边界/不相容候选采用可用单侧差分。"""
    columns = []
    for j in range(len(q)):
        h = 1e-5*max(1.0, abs(q[j]))
        plus, minus = q.copy(), q.copy()
        plus[j] = min(q[j]+h, bounds[j][1])
        minus[j] = max(q[j]-h, bounds[j][0])
        a = function(plus) if plus[j] > q[j] else None
        b = function(minus) if minus[j] < q[j] else None
        if a is not None and b is not None:
            column = (a-b)/(plus[j]-minus[j])
        elif a is not None:
            column = (a-base)/(plus[j]-q[j])
        elif b is not None:
            column = (base-b)/(q[j]-minus[j])
        else:
            return None
        columns.append(column)
    return np.column_stack(columns)


def fit_case(
    model="F", case=CASE, environment=ENVIRONMENT, solver=SOLVER, settings=INFERENCE,
    *, drag_coefficient=None, mode="map", sigma_m=None, frame_correlation=0.0,
    covariance_m2=None, starts=3, initial_guesses=None,
) -> dict:
    """返回参数、残差、目标分解、局部数据雅可比、各起点状态及最终球路。

    初值向量按 S_PARAMETER_NAMES/F_PARAMETER_NAMES；Cd 固定，不加入向量。
    优化阶段只积分到最后一帧，避免每次迭代计算全部球路；最终再传播至事件。
    """
    wall_start = time.perf_counter()
    if model not in ("S", "F") or mode not in ("map", "ls"):
        raise ValueError("反演模型用 S/F，目标函数模式用 map/ls。")
    if not isinstance(starts, int) or isinstance(starts, bool) or not 1 <= starts <= 9:
        raise ValueError("starts 须为 1 到 9 的整数。")
    cd = settings.fixed_drag_coefficient if drag_coefficient is None else float(drag_coefficient)
    if not isfinite(cd) or not settings.drag_bounds[0] <= cd <= settings.drag_bounds[1]:
        raise ValueError("固定阻力须在配置的搜索范围内。")
    observed = np.asarray(case.positions_m, dtype=float)
    if len(case.times_s) < 2 or observed.shape != (len(case.times_s), 3) or not np.all(np.isfinite(observed)):
        raise ValueError("反演需要至少两帧有限的三维观测。")
    if not all(isfinite(x) and x > 0 for x in (settings.transverse_prior_std, settings.transverse_norm_max)):
        raise ValueError("横向先验尺度和搜索半径须为有限正数。")
    if not isinstance(settings.max_iterations_per_start, int) or settings.max_iterations_per_start <= 0:
        raise ValueError("每起点的最大迭代数须为正整数。")
    velocity_guess = np.asarray(gravity_initial_velocity(case, environment))
    covariance = observation_covariance(case, settings.position_sigma_m if sigma_m is None else sigma_m,
                                        frame_correlation, covariance_m2)
    lower_factor = np.linalg.cholesky(covariance)
    names = F_PARAMETER_NAMES if model == "F" else S_PARAMETER_NAMES
    radius, bscale = settings.transverse_norm_max, settings.transverse_prior_std
    bounds = [settings.vx_bounds_m_s, settings.vy_bounds_m_s, settings.vz_bounds_m_s]
    for pair in bounds:
        if len(pair) != 2 or not all(isfinite(x) for x in pair) or pair[0] >= pair[1]:
            raise ValueError("初速度搜索界限须有限且递增。")
    if bounds[0][1] <= 0:
        raise ValueError("正向发射需允许 vx>0。")
    bounds[0] = (max(bounds[0][0], 1e-6), bounds[0][1])
    # S 的边缘先验在端点为零；留极小数值余量，避免 log(0)。
    b_limit = radius*(1-1e-8) if model == "S" else radius
    bounds += [(-b_limit, b_limit)]*(len(names)-3)
    scales = np.array([max(1.0, np.linalg.norm(velocity_guess))]*3+[bscale]*(len(names)-3))
    scaled_bounds = [(lo/s, hi/s) for (lo, hi), s in zip(bounds, scales)]
    elapsed = case.times_s[-1]-case.initial_time_s
    duration = float(np.nextafter(elapsed, np.inf))
    window_solver = replace(solver, initial_horizon_s=duration, maximum_horizon_s=duration)
    # 用物理程序验证环境、几何、时间窗等结构；错误输入直接报错。
    simulate(model, _parameter_dict(model, np.r_[velocity_guess, [0.]*(len(names)-3)], cd),
             case.times_s, case, environment, window_solver)
    cache = {}
    counters = {"forward_calls": 0, "incompatible_calls": 0, "numerical_failure_calls": 0}

    def residual(q):
        key = tuple(q)
        if key in cache:
            return cache[key]
        parameters = _parameter_dict(model, q*scales, cd)
        result = simulate(model, parameters, case.times_s, case, environment, window_solver)
        counters["forward_calls"] += 1
        if result["event"] == "failure":
            counters["numerical_failure_calls"] += 1
            value = None
        elif any(position is None for position in result["positions_m"]):
            counters["incompatible_calls"] += 1
            value = None
        else:
            error = (np.asarray(result["positions_m"])-observed).ravel()
            value = np.linalg.solve(lower_factor, error)
        if len(cache) > 10000:
            cache.clear()
        cache[key] = value
        return value

    def objective(q):
        r = residual(q)
        if r is None:
            return 1e12  # 只用于拒绝无似然候选，绝不作为可接受拟合。
        prior = transverse_prior_penalty(model, (q*scales)[3:], settings) if mode == "map" else 0.0
        return .5*float(np.dot(r, r))+prior if isfinite(prior) else 1e12

    def gradient(q):
        # 对目标作固定尺度的差分，避免零横向初值使用过小差分步长。
        base = np.array([objective(q)])
        jac = _finite_jacobian(lambda z: np.array([objective(z)]), q, scaled_bounds, base)
        return jac[0] if jac is not None else np.zeros_like(q)

    constraints = []
    if model == "F":
        constraints = [{"type": "ineq", "fun": lambda q: 1-np.sum((q[3:]*bscale/radius)**2),
                        "jac": lambda q: np.r_[np.zeros(3), -2*q[3:]*(bscale/radius)**2]}]
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    if initial_guesses is None:
        coefficient_seeds = ([0.0], [-.15], [.15], [-.3], [.3], [-.05], [.05], [-.4], [.4]) if model == "S" else (
            [0., 0.], [-.1, -.15], [.1, .15], [.15, -.15], [-.15, .15],
            [0., -.3], [0., .3], [-.3, 0.], [.3, 0.])
        guesses = [np.r_[velocity_guess, np.asarray(seed)*min(1., radius/.5)]
                   for seed in coefficient_seeds[:starts]]
    else:
        guesses = [np.asarray(guess, dtype=float) for guess in initial_guesses]
        if not guesses:
            raise ValueError("initial_guesses 不能为空。")
        if any(g.shape != (len(names),) or not np.all(np.isfinite(g)) for g in guesses):
            raise ValueError("初值须为对应参数顺序的有限向量。")
    runs, accepted = [], []
    for index, guess in enumerate(guesses):
        start = np.clip(guess, lower, upper)
        if model == "F" and np.linalg.norm(start[3:]) >= radius:
            start[3:] *= radius*.95/np.linalg.norm(start[3:])
        # 若噪声导致重力初值早落地，仅修复起点以找到可计算的观测窗。
        for boost in (0., .5, 2., 5.):
            repaired = start.copy()
            repaired[2] = min(upper[2], max(lower[2]+1e-6, start[2]+boost))
            q0 = repaired/scales
            if residual(q0) is not None:
                break
        else:
            runs.append({"start": index, "success": False, "message": "无法构造观测窗相容初值。"})
            continue
        optimized = minimize(objective, q0, method="SLSQP", jac=gradient,
            bounds=scaled_bounds, constraints=constraints,
            options={"maxiter": settings.max_iterations_per_start, "ftol": 1e-9})
        vector = optimized.x*scales
        r = residual(optimized.x)
        feasible = (r is not None and np.all(vector >= lower-1e-8)
                    and np.all(vector <= upper+1e-8) and np.linalg.norm(vector[3:]) <= radius+1e-8
                    and isfinite(objective(optimized.x)) and objective(optimized.x) < 1e11)
        successful = bool(optimized.success and feasible)
        runs.append({"start": index, "success": successful, "optimizer_success": bool(optimized.success),
                     "message": str(optimized.message), "iterations": int(optimized.nit),
                     "objective": float(objective(optimized.x)),
                     "parameters": _parameter_dict(model, vector, cd)})
        if successful:
            accepted.append((float(objective(optimized.x)), optimized.x.copy()))

    output = {"model": model, "mode": mode, "drag_coefficient": cd,
        "parameter_names": list(names), "parameter_scales": scales.tolist(),
        "observation_covariance_m2": covariance.tolist(), "optimizer_runs": runs,
        "prior_definition": "F_disk_gaussian__S_matching_marginal" if mode == "map" else "none",
        "success": False, "parameters": None, "simulation": None,
        "objective": None, "data_chi_square": None, "prior_penalty": None,
        "residuals_m": None, "rms_residual_m": None, "diagnostics": None,
        "message": "没有收敛且观测窗相容的候选。"}
    if accepted:
        value, q = min(accepted, key=lambda item: item[0])
        vector = q*scales
        parameters = _parameter_dict(model, vector, cd)
        r = residual(q)
        jacobian = _finite_jacobian(residual, q, scaled_bounds, r)
        diagnostics = {"data_jacobian_scaled": None, "data_jacobian_rank": None,
                       "data_jacobian_singular_values": None, "data_jacobian_condition": None,
                       "boundary_hits": [], "converged_start_count": len(accepted)}
        if jacobian is not None:
            singular = np.linalg.svd(jacobian, compute_uv=False)
            diagnostics.update(data_jacobian_scaled=jacobian.tolist(),
                data_jacobian_rank=int(np.linalg.matrix_rank(jacobian)),
                data_jacobian_singular_values=singular.tolist(),
                data_jacobian_condition=float(singular[0]/singular[-1]) if singular[-1] > 0 else None)
        diagnostics["boundary_hits"] = [name for name, v, (lo, hi) in zip(names, vector, bounds)
            if min(abs(v-lo), abs(v-hi)) <= 1e-5*max(1, hi-lo)]
        if radius-np.linalg.norm(vector[3:]) <= 1e-5:
            diagnostics["boundary_hits"].append("transverse_norm")
        physical_error = (lower_factor@r).reshape(observed.shape)
        final_simulation = simulate(model, parameters, case.times_s, case, environment, solver)
        output.update(success=True, parameters=parameters, simulation=final_simulation,
            objective=value, data_chi_square=float(np.dot(r, r)),
            prior_penalty=transverse_prior_penalty(model, vector[3:], settings) if mode == "map" else 0.0,
            residuals_m=physical_error.tolist(), rms_residual_m=float(np.sqrt(np.mean(physical_error**2))),
            diagnostics=diagnostics, message="观测窗反演收敛；最终事件和预测能力须分别检查。")
    output.update(counters, wall_time_s=time.perf_counter()-wall_start)
    return output


def fit_observation(record, model="F", *, mode="map", drag_coefficient=None,
                    sigma_override_m=None, starts=3):
    """只使用公开观测，返回 (共同预测记录, 拟合日志)。"""
    from evaluate import point_prediction

    cd = INFERENCE.fixed_drag_coefficient if drag_coefficient is None else drag_coefficient
    method = f"{model}_cd{cd:g}_{mode}"
    if not record["observation_available"]:
        return point_prediction(record, method, status="unavailable", message="观测不完整。",
                                point_kind=f"{mode}_trajectory"), {
            "case_id": record["case_id"], "method": method, "success": False, "message": "观测不完整。"}
    try:
        case = observation_case(record)
        known = dict(record["known_environment"])
        known["wind_m_s"] = tuple(known["wind_m_s"])
        environment = Environment(**known)
        measurement = record["measurement"]
        if sigma_override_m is None and measurement["mode"] != "gaussian":
            raise ValueError("本版只实现正定高斯似然；舍入/无误差数据需显式 --sigma 工作假设。")
        sigma = measurement["sigma_m"] if sigma_override_m is None else sigma_override_m
        correlation = measurement["frame_correlation"] if measurement["mode"] == "gaussian" else 0.0
        fit = fit_case(model, case, environment, mode=mode, drag_coefficient=drag_coefficient,
                       sigma_m=sigma, frame_correlation=correlation, starts=starts)
        fit.update(case_id=record["case_id"], method=method,
                   sigma_override_m=sigma_override_m)
        if fit["success"]:
            prediction = point_prediction(record, method, fit["simulation"],
                                           point_kind=f"{mode}_trajectory")
        else:
            prediction = point_prediction(record, method, status="failure", message=fit["message"],
                                           point_kind=f"{mode}_trajectory")
        return prediction, fit
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return point_prediction(record, method, status="failure", message=str(error),
                                point_kind=f"{mode}_trajectory"), {
            "case_id": record["case_id"], "method": method, "success": False, "message": str(error)}


def main():
    parser = argparse.ArgumentParser(description="固定阻力的 S/F 两帧反演。默认计算题目算例。")
    parser.add_argument("--model", choices=("S", "F", "both"), default="both")
    parser.add_argument("--mode", choices=("map", "ls"), default="map")
    parser.add_argument("--drag", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None, help="显式覆盖观测标准差，单位 m")
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=None, help="提供后批量处理该目录的 observations")
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR/"inverse_example")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，请指定新的 --output 目录。")
    models = ("S", "F") if args.model == "both" else (args.model,)
    fits, predictions = [], []
    if args.dataset is None:
        for model in models:
            fits.append(fit_case(model, mode=args.mode, drag_coefficient=args.drag,
                                 sigma_m=args.sigma, starts=args.starts))
    else:
        records = load_observations(args.dataset/"observations"/f"{args.split}.jsonl")
        for i, record in enumerate(records):
            for model in models:
                prediction, fit = fit_observation(record, model, mode=args.mode,
                    drag_coefficient=args.drag, sigma_override_m=args.sigma, starts=args.starts)
                predictions.append(prediction)
                fits.append(fit)
            if (i+1) % 10 == 0 or i+1 == len(records):
                print(f"Processed {i+1}/{len(records)} observation cases", file=sys.stderr, flush=True)
    run_config = {"mode": args.mode, "models": models, "drag": args.drag,
        "sigma_override_m": args.sigma, "starts": args.starts,
        "dataset": str(args.dataset.resolve()) if args.dataset else None,
        "split": args.split if args.dataset else None,
        "case": asdict(CASE) if args.dataset is None else None,
        "settings": asdict(INFERENCE), "solver": asdict(SOLVER),
        "default_environment": asdict(ENVIRONMENT),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "note": "固定阻力条件下的参数点估计，不是门线后验众数或概率区域。"}
    # 先确认全部可序列化，避免 NaN/Infinity 被静默写入。
    config_text = json.dumps(run_config, ensure_ascii=False, indent=2, allow_nan=False)
    fits_text = "".join(json.dumps(fit, ensure_ascii=False, allow_nan=False)+"\n" for fit in fits)
    predictions_text = "".join(json.dumps(p, ensure_ascii=False, allow_nan=False)+"\n" for p in predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"run_config.json").write_text(config_text+"\n", encoding="utf-8")
    (args.output/"fits.jsonl").write_text(fits_text, encoding="utf-8")
    if args.dataset is None:
        (args.output/"example.json").write_text(json.dumps(fits, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    else:
        (args.output/"predictions.jsonl").write_text(predictions_text, encoding="utf-8")
    summary = {"output": str(args.output.resolve()), "fit_count": len(fits),
               "converged_fit_count": sum(fit["success"] for fit in fits), "note": run_config["note"]}
    if args.dataset is None:
        summary["example"] = [{"model": f["model"], "success": f["success"],
            "rms_residual_m": f["rms_residual_m"],
            "event": f["simulation"]["event"] if f["simulation"] else None,
            "event_position_m": f["simulation"]["event_position_m"] if f["simulation"] else None}
            for f in fits]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
