"""共同的运动计算器：参数 -> 两帧位置、首个门线/地面事件。

simulate("F"/"S", params, times_s) 是甲交给乙、丙的主要接口。
predict_baseline("G"/"A") 提供两种从观测拟合的简单对照。
本模块不反演气动参数，不计算预测概率，也不模拟接地后的运动。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from config import (
    CASE, ENVIRONMENT, INFERENCE, SOLVER, DynamicsParameters, Environment,
    ObservationCase, SimulationResult, SolverSettings, Vector3,
)


def _vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是三个有限数值，按 (x,y,z) 排列。")
    return result


def _tuple3(value) -> Vector3:
    return tuple(float(x) for x in value)  # type: ignore[return-value]


def _parameters(model: str, params: Mapping) -> tuple[np.ndarray, float, float, float]:
    if model not in ("G", "S", "F"):
        raise ValueError('simulate 支持 G/S/F；A 请使用 predict_baseline("A")。')
    velocity = _vector(params["initial_velocity_m_s"], "初速度")
    if model == "G":
        return velocity, 0.0, 0.0, 0.0
    cd, by, bz = (float(params[key]) for key in ("drag_coefficient", "b_y", "b_z"))
    if not all(isfinite(x) for x in (cd, by, bz)) or cd < 0:
        raise ValueError("阻力须非负，气动力系数须有限。")
    if model == "S" and by != 0:
        raise ValueError("S 模型要求 b_y=0，请显式设置，不能静默忽略。")
    return velocity, cd, by, bz


def _check_environment(environment: Environment) -> None:
    positive = (environment.mass_kg, environment.radius_m,
                environment.kinematic_viscosity_m2_s)
    nonnegative = (environment.gravity_m_s2, environment.air_density_kg_m3)
    if not all(isfinite(x) and x > 0 for x in positive):
        raise ValueError("质量、半径和运动黏度须为有限正数。")
    if not all(isfinite(x) and x >= 0 for x in nonnegative):
        raise ValueError("重力和空气密度须为有限非负数；零值可用于极限检查。")
    _vector(environment.wind_m_s, "风速")


def _air_acceleration(velocity, cd, by, bz, environment):
    ux, uy, uz = velocity - np.asarray(environment.wind_m_s)
    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    factor = environment.aerodynamic_k_per_m * speed
    # (0,by,bz) x (ux,uy,uz)；与相对速度垂直。
    return np.array((
        factor * (-cd * ux + by * uz - bz * uy),
        factor * (-cd * uy + bz * ux),
        -environment.gravity_m_s2 + factor * (-cd * uz - by * ux),
    ))


def acceleration(
    velocity_m_s: Vector3,
    params: DynamicsParameters,
    environment: Environment = ENVIRONMENT,
    model: str = "F",
) -> np.ndarray:
    """给定速度计算加速度，供物理检查或独立诊断使用，单位 m/s²。"""
    _check_environment(environment)
    _, cd, by, bz = _parameters(model, params)
    return _air_acceleration(_vector(velocity_m_s, "速度"), cd, by, bz, environment)


def is_in_goal_opening(position_m: Vector3, case: ObservationCase = CASE) -> bool:
    """只检查球心的 y/z 严格开区间；调用者另须确认 event == 'goal'。"""
    _, y, z = _vector(position_m, "位置")
    return bool(case.goal_y_bounds_m[0] < y < case.goal_y_bounds_m[1]
                and case.goal_z_bounds_m[0] < z < case.goal_z_bounds_m[1])


def _propagate(
    velocity: np.ndarray,
    acceleration_function: Callable,
    times_s: Sequence[float],
    case: ObservationCase,
    environment: Environment,
    solver: SolverSettings,
) -> SimulationResult:
    """所有模型共用的积分、事件、失败和位置采样逻辑。"""
    _check_environment(environment)
    position = _vector(case.initial_position_m, "起点")
    t0 = float(case.initial_time_s)
    times = tuple(float(t) for t in times_s)
    if not isfinite(t0) or not all(isfinite(t) and t >= t0 for t in times):
        raise ValueError("请求时刻必须有限，且不早于起始时刻。")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("请求时刻须严格递增（允许空列表，只查询事件）。")
    for bounds in (case.goal_y_bounds_m, case.goal_z_bounds_m):
        if len(bounds) != 2 or not all(isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("门框边界须为有限、递增的两端点。")
    if not isfinite(case.goal_x_m) or position[0] > case.goal_x_m:
        raise ValueError("起点不能已位于门线之后。")
    if position[2] < environment.radius_m:
        raise ValueError("起始球心不能位于地面之下。")
    settings = (solver.relative_tolerance, solver.position_absolute_tolerance_m,
                solver.velocity_absolute_tolerance_m_s, solver.max_step_s,
                solver.initial_horizon_s, solver.maximum_horizon_s,
                solver.event_time_tolerance_s)
    if not all(isfinite(x) and x > 0 for x in settings):
        raise ValueError("积分时间窗、容差和最大步长须为有限正数。")
    if solver.initial_horizon_s > solver.maximum_horizon_s:
        raise ValueError("初始积分时间窗不能超过最大时间窗。")
    if solver.method not in ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"):
        raise ValueError("未知的积分方法。")
    maximum_time = t0 + solver.maximum_horizon_s
    if times and times[-1] > maximum_time:
        raise ValueError("请求时刻超出最大积分时间窗，请先扩大 maximum_horizon_s。")

    result: SimulationResult = {
        "times_s": times, "positions_m": tuple(None for _ in times),
        "event": "unresolved", "event_time_s": None,
        "event_position_m": None, "message": "",
    }
    samples: list[Optional[Vector3]] = [None for _ in times]
    for i, t in enumerate(times):
        if t == t0:
            samples[i] = _tuple3(position)

    on_ground = position[2] == environment.radius_m
    initial_ground = on_ground and velocity[2] <= 0
    initial_goal = position[0] == case.goal_x_m and velocity[0] > 0
    if initial_ground or initial_goal:
        result.update(
            event="boundary" if initial_ground and initial_goal else
                  ("ground" if initial_ground else "goal"),
            event_time_s=t0, event_position_m=_tuple3(position),
            positions_m=tuple(samples), message="起始位置已触发事件。",
        )
        return result

    def rhs(t, state):
        return np.concatenate((state[3:], acceleration_function(t, state[3:])))

    def goal_event(t, state):
        return state[0] - case.goal_x_m

    def ground_event(t, state):
        gap = state[2] - environment.radius_m
        # 除去离地时已知的 t=t0 零点；在 t0 的连续极限是初始竖直速度。
        # 与简单删除首个事件相比，这也能识别很短的离地后再次接地。
        if on_ground:
            return velocity[2] if t == t0 else gap / (t - t0)
        return gap

    goal_event.direction, goal_event.terminal = 1, True
    ground_event.direction, ground_event.terminal = -1, True
    state = np.concatenate((position, velocity))
    start = t0
    end = t0 + solver.initial_horizon_s
    atol = np.array([solver.position_absolute_tolerance_m] * 3
                    + [solver.velocity_absolute_tolerance_m_s] * 3)

    while True:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                solution = solve_ivp(
                    rhs, (start, end), state, method=solver.method,
                    dense_output=True, events=(goal_event, ground_event),
                    rtol=solver.relative_tolerance, atol=atol,
                    max_step=solver.max_step_s,
                )
        except (FloatingPointError, OverflowError, RuntimeError) as error:
            result.update(event="failure", message=f"数值求解异常：{error}",
                          positions_m=tuple(samples))
            return result

        actual_end = float(solution.t[-1])
        # 只允许机器精度范围的时间舍入；不采样首个事件之后的球路。
        time_roundoff = 16 * np.finfo(float).eps * max(1.0, abs(actual_end))
        for i, t in enumerate(times):
            if samples[i] is None and start <= t <= actual_end + time_roundoff:
                sample = solution.sol(min(t, actual_end))[:3]
                if np.all(np.isfinite(sample)):
                    samples[i] = _tuple3(sample)
        result["positions_m"] = tuple(samples)
        if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
            result.update(event="failure", message=str(solution.message))
            return result

        hits = [(float(events[0]), name) for events, name in
                zip(solution.t_events, ("goal", "ground")) if len(events)]
        if hits:
            event_time, name = min(hits)
            event_state = solution.sol(event_time)
            x, _, z, vx, _, vz = event_state
            # terminal 事件会让第二类事件停止搜索，使用另一表面的距离/
            # 当前法向速度检查近同时事件；另加机器舍入和位置绝对容差。
            position_roundoff = max(
                10 * solver.position_absolute_tolerance_m,
                32 * np.finfo(float).eps * max(1.0, abs(x), abs(z)),
            )
            near_ground = vz < 0 and abs(z - environment.radius_m) <= (
                abs(vz) * solver.event_time_tolerance_s + position_roundoff)
            near_goal = vx > 0 and abs(x - case.goal_x_m) <= (
                abs(vx) * solver.event_time_tolerance_s + position_roundoff)
            boundary = (name == "goal" and near_ground) or (name == "ground" and near_goal)
            result.update(
                event="boundary" if boundary else name,
                event_time_s=event_time,
                event_position_m=_tuple3(event_state[:3]),
                message="门线与接地在数值容差内重合。" if boundary else
                        ("首次正向过门线；入框需另检查 y/z。" if name == "goal"
                         else "先接地，停止自由飞行；不计算反弹。"),
            )
            return result
        if end >= maximum_time:
            result["message"] = "最大积分时间窗内未确认门线或接地事件。"
            return result
        start, state = actual_end, solution.y[:, -1]
        end = maximum_time  # 无事件时，从初始时间窗继续至最大时间窗。


def simulate(
    model: str,
    params: DynamicsParameters,
    times_s: Sequence[float],
    case: ObservationCase = CASE,
    environment: Environment = ENVIRONMENT,
    solver: SolverSettings = SOLVER,
) -> SimulationResult:
    """计算 F/S（或给定初速度的 G）球路，返回 config 约定的字典。

    初态在 case.initial_time_s，所有请求时刻使用同一时间坐标。
    输入格式或物理定义不合法时抛出 ValueError/KeyError；数值失败返回
    event='failure'。研究先验边界由 inverse.py 处理，本函数不强加。
    """
    velocity, cd, by, bz = _parameters(model, params)
    return _propagate(
        velocity, lambda t, v: _air_acceleration(v, cd, by, bz, environment),
        times_s, case, environment, solver,
    )


def _observation_arrays(case):
    elapsed = np.asarray(case.times_s, dtype=float) - case.initial_time_s
    positions = np.asarray(case.positions_m, dtype=float)
    if (elapsed.ndim != 1 or len(elapsed) == 0
            or positions.shape != (len(elapsed), 3)
            or not np.all(np.isfinite(elapsed)) or not np.all(np.isfinite(positions))
            or np.any(elapsed <= 0) or np.any(np.diff(elapsed) <= 0)):
        raise ValueError("基线拟合需要一一对应、有限且递增的初始时刻后观测。")
    return elapsed, positions - _vector(case.initial_position_m, "起点")


def gravity_initial_velocity(
    case: ObservationCase = CASE, environment: Environment = ENVIRONMENT,
) -> Vector3:
    """G：固定起点、重力，两帧同方差位置最小二乘初速度。"""
    _check_environment(environment)
    elapsed, displacement = _observation_arrays(case)
    adjusted = displacement.copy()
    adjusted[:, 2] += 0.5 * environment.gravity_m_s2 * elapsed**2
    return _tuple3(np.sum(elapsed[:, None] * adjusted, axis=0) / np.dot(elapsed, elapsed))


def constant_acceleration_parameters(case: ObservationCase = CASE) -> tuple[Vector3, Vector3]:
    """A：固定起点，拟合初速度和恒加速度；两帧时等于三点二次插值。"""
    elapsed, displacement = _observation_arrays(case)
    design = np.column_stack((elapsed, 0.5 * elapsed**2))
    coefficients, _, rank, _ = np.linalg.lstsq(design, displacement, rcond=None)
    if rank != 2:
        raise ValueError("恒加速度基线至少需要两个不同的正观测时刻。")
    return _tuple3(coefficients[0]), _tuple3(coefficients[1])


def predict_baseline(
    model: str,
    times_s: Optional[Sequence[float]] = None,
    case: ObservationCase = CASE,
    environment: Environment = ENVIRONMENT,
    solver: SolverSettings = SOLVER,
) -> SimulationResult:
    """从 case 观测拟合 G/A，再使用相同事件规则向门线传播。"""
    requested = case.times_s if times_s is None else times_s
    if model == "G":
        params = {"initial_velocity_m_s": gravity_initial_velocity(case, environment),
                  "drag_coefficient": 0.0, "b_y": 0.0, "b_z": 0.0}
        return simulate("G", params, requested, case, environment, solver)
    if model == "A":
        velocity, constant_a = constant_acceleration_parameters(case)
        a = np.asarray(constant_a)
        return _propagate(np.asarray(velocity), lambda t, v: a,
                          requested, case, environment, solver)
    raise ValueError("predict_baseline 只支持 G/A。")


def main() -> None:
    parser = argparse.ArgumentParser(description="UPC 球路正向计算示例，非最终反演答案。")
    parser.add_argument("--model", choices=("F", "S", "G", "A"), default="F")
    args = parser.parse_args()
    if args.model in ("G", "A"):
        result = predict_baseline(args.model)
        note = "简单基线：由题面观测拟合，不是最终气动力反演。"
    else:
        # 仅用于确认程序可运行；不是根据两帧反演得到的参数。
        params = {"initial_velocity_m_s": (30.0, 9.0, 7.0),
                  "drag_coefficient": INFERENCE.fixed_drag_coefficient,
                  "b_y": 0.05 if args.model == "F" else 0.0, "b_z": -0.15}
        result = simulate(args.model, params, CASE.times_s)
        note = "演示参数：人为指定，仅检查运动计算接口，不是题目预测结果。"
    inside = (is_in_goal_opening(result["event_position_m"])
              if result["event"] == "goal" else None)
    print(json.dumps({"model": args.model, "note": note,
                      "result": result, "inside_goal_opening": inside},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
