"""独立的合成测试数据生成器，以及有明确单位的 CSV 轨迹读取器。

H 方程在本文件独立实现，不导入 physics 或 inverse。经验曲线及参数
仅为明确的模拟压力设置，不是已标定的某款足球。输入与答案分文件保存。
在项目根目录运行：python code/data.py --sources 60
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from math import isfinite, pi
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

from config import (CASE, DATA_DIR, ENVIRONMENT, NOISE_SCENARIOS_M, RANDOM_SEED,
                    SOLVER, Environment, ObservationCase, SolverSettings, Vector3)

SCHEMA_VERSION = 1
MECHANISMS = ("constant", "variable_drag", "spin", "wind", "combined")
SPLITS = ("development", "calibration", "test")


@dataclass(frozen=True)
class TruthParameters:
    """生成端真值参数，绝不作为预测器输入。旋转角速度单位 rad/s。"""

    initial_velocity_m_s: Vector3 = (30.0, 5.0, 6.0)
    mechanism: str = "combined"
    drag_coefficient: float = 0.22
    transverse_vector: Vector3 = (0.0, 0.05, -0.15)
    spin_rad_s: Vector3 = (0.0, 0.0, -40.0)
    drag_low: float = 0.15
    drag_high: float = 0.45
    transition_re: float = 3e5
    transition_width_re: float = 5e4
    magnus_slope: float = 1.2
    magnus_cap: float = 0.35


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须是三个有限数值。")
    return array


def _tuple3(value):
    return tuple(float(x) for x in value)


def _validate_parameters(parameters, environment):
    if parameters.mechanism not in MECHANISMS:
        raise ValueError(f"未知生成机制：{parameters.mechanism}")
    for vector in (parameters.initial_velocity_m_s, parameters.transverse_vector,
                   parameters.spin_rad_s, environment.wind_m_s):
        _vector(vector, "速度/横向系数/自旋/风")
    for value in (parameters.drag_coefficient, parameters.drag_low,
                  parameters.drag_high, parameters.magnus_slope,
                  environment.gravity_m_s2, environment.air_density_kg_m3):
        if not isfinite(value) or value < 0:
            raise ValueError("阻力、Magnus 斜率、重力和密度须为有限非负数。")
    for value in (parameters.transition_re, parameters.transition_width_re,
                  parameters.magnus_cap, environment.mass_kg,
                  environment.radius_m, environment.kinematic_viscosity_m2_s):
        if not isfinite(value) or value <= 0:
            raise ValueError("过渡尺度、幅值上限、质量、半径、黏度须为有限正数。")
    if parameters.drag_low > parameters.drag_high:
        raise ValueError("低阻力系数不能高于高阻力系数。")


def _acceleration(velocity, parameters, environment):
    u = velocity - np.asarray(environment.wind_m_s)
    speed = float(np.linalg.norm(u))
    gravity = np.array((0.0, 0.0, -environment.gravity_m_s2))
    if speed == 0:
        return gravity
    re = 2 * environment.radius_m * speed / environment.kinematic_viscosity_m2_s
    cd = parameters.drag_coefficient
    if parameters.mechanism in ("variable_drag", "combined"):
        # 明确的压力测试曲线，不能写作某球型的实验拟合公式。
        cd = parameters.drag_low + (parameters.drag_high - parameters.drag_low) * expit(
            (parameters.transition_re - re) / parameters.transition_width_re)
    direction = u / speed
    if parameters.mechanism in ("spin", "combined"):
        omega = np.asarray(parameters.spin_rad_s)
        omega_norm = float(np.linalg.norm(omega))
        lateral = np.zeros(3)
        if omega_norm > 0:
            spin_ratio = environment.radius_m * omega_norm / speed
            cm = parameters.magnus_cap * np.tanh(
                parameters.magnus_slope * spin_ratio / parameters.magnus_cap)
            # Cm 乘以未归一化的叉积方向；幅值系数含 sin(alpha)。
            lateral = cm * np.cross(omega / omega_norm, direction)
    else:
        lateral = np.cross(np.asarray(parameters.transverse_vector), direction)
    # k*speed² 为 m/s²；所有系数无量纲。
    k = environment.air_density_kg_m3 * pi * environment.radius_m**2 / (2*environment.mass_kg)
    return gravity + k * speed**2 * (-cd * direction + lateral)


def truth_acceleration(velocity_m_s, parameters, environment=ENVIRONMENT):
    """独立 H 加速度接口，供零自旋、方向和功平衡检查。"""
    _validate_parameters(parameters, environment)
    return _acceleration(_vector(velocity_m_s, "速度"), parameters, environment)


def generate_truth(
    parameters: TruthParameters,
    case: ObservationCase = CASE,
    environment: Environment = ENVIRONMENT,
    solver: SolverSettings = SOLVER,
    trajectory_step_s: float = 0.01,
) -> dict:
    """生成首个事件前的完整状态和干净观测，保留接地/未决/失败。"""
    _validate_parameters(parameters, environment)
    r0 = _vector(case.initial_position_m, "起点")
    velocity = _vector(parameters.initial_velocity_m_s, "初速度")
    t0 = float(case.initial_time_s)
    times = tuple(float(t) for t in case.times_s)
    positive = (trajectory_step_s, solver.relative_tolerance,
                solver.position_absolute_tolerance_m, solver.velocity_absolute_tolerance_m_s,
                solver.max_step_s, solver.maximum_horizon_s, solver.event_time_tolerance_s)
    if not all(isfinite(x) and x > 0 for x in positive):
        raise ValueError("采样间隔、积分容差、步长与时间窗须为有限正数。")
    if not isfinite(t0) or not all(isfinite(t) and t > t0 for t in times):
        raise ValueError("观测时刻须有限且晚于起始时刻。")
    if not times or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("至少需要一个观测时刻，且须严格递增。")
    if times[-1] > t0 + solver.maximum_horizon_s:
        raise ValueError("观测时刻超出生成时间窗。")
    if r0[2] < environment.radius_m or not isfinite(case.goal_x_m) or r0[0] >= case.goal_x_m:
        raise ValueError("生成起点应在地面上或上方、且在门线之前。")
    for bounds in (case.goal_y_bounds_m, case.goal_z_bounds_m):
        if len(bounds) != 2 or not all(isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("门框边界应是递增的有限端点。")

    initial_state = np.concatenate((r0, velocity))
    clean = [None] * len(times)
    result = {
        "event": "unresolved", "event_time_s": None, "event_position_m": None,
        "inside_goal_opening": None, "message": "", "clean_positions_m": clean,
        "trajectory_times_s": [t0], "trajectory_states": [initial_state.tolist()],
        "observation_available": False,
    }
    if r0[2] == environment.radius_m and velocity[2] <= 0:
        result.update(event="ground", event_time_s=t0, event_position_m=r0.tolist(),
                      message="起始即接触地面，不生成接触后的自由飞行。")
        return result

    def rhs(t, state):
        return np.concatenate((state[3:], _acceleration(state[3:], parameters, environment)))

    def goal(t, state):
        return state[0] - case.goal_x_m

    def ground(t, state):
        gap = state[2] - environment.radius_m
        if r0[2] == environment.radius_m:
            return velocity[2] if t == t0 else gap / (t - t0)
        return gap

    goal.direction, goal.terminal = 1, True
    ground.direction, ground.terminal = -1, True
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = solve_ivp(
                rhs, (t0, t0 + solver.maximum_horizon_s), initial_state,
                method="DOP853", events=(goal, ground), dense_output=True,
                rtol=solver.relative_tolerance/10,
                atol=[solver.position_absolute_tolerance_m/10]*3
                     + [solver.velocity_absolute_tolerance_m_s/10]*3,
                max_step=solver.max_step_s/2,
            )
    except (FloatingPointError, OverflowError, RuntimeError) as error:
        result.update(event="failure", message=f"生成求解异常：{error}")
        return result

    stop = float(solution.t[-1])
    roundoff = 16*np.finfo(float).eps*max(1.0, abs(stop))
    for i, t in enumerate(times):
        if t <= stop + roundoff:
            position = solution.sol(min(t, stop))[:3]
            if np.all(np.isfinite(position)):
                clean[i] = position.tolist()
    # 真值轨迹只保存首个事件前状态，另显式加入请求时刻与末端。
    grid = sorted(set([float(t) for t in np.arange(t0, stop, trajectory_step_s)]
                      + [t for t in times if t <= stop] + [stop]))
    states = solution.sol(grid).T
    if np.all(np.isfinite(states)):
        result.update(trajectory_times_s=grid, trajectory_states=states.tolist())
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        result.update(event="failure", message=str(solution.message))
        return result

    result["observation_available"] = all(p is not None for p in clean)
    hits = [(float(events[0]), name) for events, name in
            zip(solution.t_events, ("goal", "ground")) if len(events)]
    if not hits:
        result["message"] = "最大时间窗内无已确认事件。"
        return result
    t, event = min(hits)
    state = solution.sol(t)
    x, y, z, vx, vy, vz = state
    position_tolerance = max(10*solver.position_absolute_tolerance_m,
                             32*np.finfo(float).eps*max(1.0, abs(x), abs(z)))
    near_ground = vz < 0 and abs(z-environment.radius_m) <= (
        abs(vz)*solver.event_time_tolerance_s+position_tolerance)
    near_goal = vx > 0 and abs(x-case.goal_x_m) <= (
        abs(vx)*solver.event_time_tolerance_s+position_tolerance)
    if (event == "goal" and near_ground) or (event == "ground" and near_goal):
        event = "boundary"
    inside = bool(case.goal_y_bounds_m[0] < y < case.goal_y_bounds_m[1]
                  and case.goal_z_bounds_m[0] < z < case.goal_z_bounds_m[1]) if event == "goal" else None
    result.update(event=event, event_time_s=t, event_position_m=state[:3].tolist(),
                  inside_goal_opening=inside, message="独立生成端的首次事件。")
    return result


def apply_measurement_error(
    clean_positions_m, rng, mode="gaussian", sigma_m=0.01,
    rounding_step_m=0.01, frame_correlation=0.0,
) -> list:
    """同一轴跨帧等相关高斯误差；舍入与高斯模式分别使用，不裁剪坐标。"""
    if mode not in ("gaussian", "rounding", "none"):
        raise ValueError("误差模式应为 gaussian、rounding 或 none。")
    if not isfinite(sigma_m) or sigma_m < 0 or not 0 <= frame_correlation <= 1:
        raise ValueError("误差标准差须非负，跨帧相关系数须在 [0,1] 内。")
    if not isfinite(rounding_step_m) or rounding_step_m <= 0:
        raise ValueError("舍入步长须为有限正数。")
    valid = [(i, _vector(p, "干净位置")) for i, p in enumerate(clean_positions_m) if p is not None]
    output = [None]*len(clean_positions_m)
    shared = rng.normal(size=3) if mode == "gaussian" else np.zeros(3)
    for i, p in valid:
        if mode == "gaussian":
            noise = sigma_m*(np.sqrt(frame_correlation)*shared
                             + np.sqrt(1-frame_correlation)*rng.normal(size=3))
            observed = p + noise
        elif mode == "rounding":
            observed = np.round(p/rounding_step_m)*rounding_step_m
        else:
            observed = p
        output[i] = observed.tolist()
    return output


def assign_splits(source_ids, rng, fractions=(0.6, 0.2, 0.2)) -> dict:
    """按源射门划分，所有噪声副本共享同一 split。"""
    if (len(fractions) != 3 or not all(isfinite(x) and x > 0 for x in fractions)
            or not np.isclose(sum(fractions), 1.0)):
        raise ValueError("三组比例须为正数且总和为 1。")
    ids = list(source_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("source_id 必须唯一。")
    order = rng.permutation(len(ids))
    first = int(len(ids)*fractions[0])
    second = first + int(len(ids)*fractions[1])
    return {ids[int(index)]: SPLITS[0 if j < first else (1 if j < second else 2)]
            for j, index in enumerate(order)}


def _draw_parameters(rng, mechanism):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    spin = axis*(2*pi*rng.uniform(0, 12))
    parameters = TruthParameters(
        initial_velocity_m_s=(float(rng.uniform(22, 36)), float(rng.uniform(-10, 10)),
                              float(rng.uniform(0.2, 10))),
        mechanism=mechanism, drag_coefficient=float(rng.uniform(.15, .4)),
        transverse_vector=(0.0, float(rng.uniform(-.12, .12)), float(rng.uniform(-.25, .25))),
        spin_rad_s=_tuple3(spin),
    )
    wind = ((float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3)), float(rng.uniform(-.5, .5)))
            if mechanism in ("wind", "combined") else (0.0, 0.0, 0.0))
    return parameters, replace(ENVIRONMENT, wind_m_s=wind)


def build_dataset(
    source_count=60, seed=RANDOM_SEED, noise_sigmas_m=NOISE_SCENARIOS_M,
    noise_mode="gaussian", frame_correlation=0.0, wind_known=False,
    split_fractions=(0.6, 0.2, 0.2), mechanisms=MECHANISMS,
) -> dict:
    """在内存生成数据；一次求真值，多种噪声副本，不按入框或成功筛选。"""
    if not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < 1:
        raise ValueError("源射门数须为正整数。")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("随机种子须为非负整数。")
    mechanisms = tuple(mechanisms)
    if not mechanisms or any(m not in MECHANISMS for m in mechanisms):
        raise ValueError("生成机制列表不能为空且须受支持。")
    sigmas = tuple(float(s) for s in noise_sigmas_m)
    if not sigmas or not all(isfinite(s) and s >= 0 for s in sigmas):
        raise ValueError("至少给出一个有限非负误差档位。")
    # 预先验证误差设置，避免运行生成后才报告配置错误。
    apply_measurement_error([], np.random.default_rng(0), mode=noise_mode,
                            sigma_m=sigmas[0], frame_correlation=frame_correlation)
    if noise_mode != "gaussian":
        sigmas = (0.0,)  # 舍入和无噪声模式只生成一个副本。
    children = np.random.SeedSequence(seed).spawn(source_count+1)
    source_ids = [f"source_{i:05d}" for i in range(source_count)]
    split_map = assign_splits(source_ids, np.random.default_rng(children[0]), split_fractions)
    observations, truths = [], []
    for i, source_id in enumerate(source_ids):
        source_seed = int(children[i+1].generate_state(1)[0])
        rng = np.random.default_rng(source_seed)
        parameters, true_environment = _draw_parameters(rng, mechanisms[i % len(mechanisms)])
        truth = generate_truth(parameters, environment=true_environment)
        split = split_map[source_id]
        noise_seeds = []
        for j, sigma in enumerate(sigmas):
            noise_seed = int(rng.integers(0, 2**32))
            noise_seeds.append(noise_seed)
            observed = apply_measurement_error(truth["clean_positions_m"],
                np.random.default_rng(noise_seed), noise_mode, sigma,
                frame_correlation=frame_correlation)
            case_dict = asdict(CASE)
            case_dict["positions_m"] = observed
            known_environment = asdict(true_environment if wind_known else
                                       replace(true_environment, wind_m_s=(0.0, 0.0, 0.0)))
            observations.append({
                "schema_version": SCHEMA_VERSION, "source_id": source_id,
                "case_id": f"{source_id}_noise_{j:02d}", "split": split,
                "observation_available": truth["observation_available"],
                "case": case_dict, "known_environment": known_environment,
                "wind_known": bool(wind_known),
                "measurement": {"mode": noise_mode, "sigma_m": sigma,
                    "frame_correlation": frame_correlation if noise_mode == "gaussian" else 0.0,
                    "rounding_step_m": 0.01 if noise_mode == "rounding" else None},
            })
        truths.append({"schema_version": SCHEMA_VERSION, "source_id": source_id,
                       "split": split, "source_seed": source_seed,
                       "noise_seeds": noise_seeds, "parameters": asdict(parameters),
                       "true_environment": asdict(true_environment), "truth": truth})
    source_counts = dict(Counter(split_map.values()))
    manifest = {
        "schema_version": SCHEMA_VERSION, "seed": seed,
        "source_count": source_count, "observation_count": len(observations),
        "source_counts_by_split": {s: source_counts.get(s, 0) for s in SPLITS},
        "event_counts": dict(Counter(t["truth"]["event"] for t in truths)),
        "unavailable_source_count": sum(not t["truth"]["observation_available"] for t in truths),
        "mechanisms": list(mechanisms), "noise_mode": noise_mode,
        "noise_sigmas_m": list(sigmas), "frame_correlation": frame_correlation,
        "wind_known": bool(wind_known), "split_fractions": list(split_fractions),
        "geometry": {k: v for k, v in asdict(CASE).items() if k != "positions_m"},
        "solver": {"method": "DOP853", "rtol": SOLVER.relative_tolerance/10,
            "position_atol_m": SOLVER.position_absolute_tolerance_m/10,
            "velocity_atol_m_s": SOLVER.velocity_absolute_tolerance_m_s/10,
            "max_step_s": SOLVER.max_step_s/2,
            "maximum_horizon_s": SOLVER.maximum_horizon_s,
            "trajectory_step_s": 0.01, "event_time_tolerance_s": SOLVER.event_time_tolerance_s},
        "sampling_assumptions": {
            "status": "synthetic_pressure_settings_not_empirically_calibrated",
            "velocity_ranges_m_s": [[22, 36], [-10, 10], [0.2, 10]],
            "drag_coefficient_range": [.15, .4],
            "transverse_ranges": [[0, 0], [-.12, .12], [-.25, .25]],
            "spin_rps_range": [0, 12], "spin_axis": "isotropic",
            "wind_ranges_m_s": [[-3, 3], [-3, 3], [-.5, .5]],
            "variable_drag": {"low": .15, "high": .45, "transition_re": 3e5, "width_re": 5e4},
            "magnus": {"slope": 1.2, "cap": .35, "definition": "Cm times cross(unit_spin, unit_relative_velocity)"},
        },
        "notes": ["No filtering by goal outcome; failures and unavailable observations retained.",
                  "Truth and source/noise seeds must not be supplied to the predictor.",
                  "Development chooses methods; calibration fixes regions; test evaluates frozen methods.",
                  "Default 60 sources is a small demonstration, not sufficient calibration evidence."],
    }
    return {"manifest": manifest, "observations": observations, "truth": truths}


def save_dataset(dataset: dict, output_dir) -> Path:
    """只写入新目录，避免覆盖已冻结数据；每行一个 JSON 对象。"""
    output = Path(output_dir)
    # 先序列化验证，拒绝 NaN/Infinity，不创建半套非法数据。
    manifest_text = json.dumps(dataset["manifest"], ensure_ascii=False, indent=2, allow_nan=False)
    files = {}
    for split in SPLITS:
        for kind, key in (("observations", "observations"), ("truth", "truth")):
            files[(kind, split)] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n"
                for row in dataset[key] if row["split"] == split)
    output.mkdir(parents=True, exist_ok=False)
    for kind in ("observations", "truth"):
        (output/kind).mkdir()
        for split in SPLITS:
            (output/kind/f"{split}.jsonl").write_text(files[(kind, split)], encoding="utf-8")
    (output/"manifest.json").write_text(manifest_text+"\n", encoding="utf-8")
    return output


def load_observations(path) -> list[dict]:
    """只接受预测输入记录；误传真值文件时立即拒绝。不可用样本仍保留。"""
    rows, case_ids = [], set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = {"schema_version", "source_id", "case_id", "split", "observation_available",
                        "case", "known_environment", "wind_known", "measurement"}
            if set(row) != expected or row["schema_version"] != SCHEMA_VERSION:
                raise ValueError(f"第 {line_number} 行不是受支持的观测输入格式；禁止加载真值。")
            if row["case_id"] in case_ids or row["split"] not in SPLITS:
                raise ValueError("重复 case_id 或未知数据划分。")
            case_ids.add(row["case_id"])
            if row["observation_available"]:
                observation_case(row)
            rows.append(row)
    return rows


def observation_case(record: dict) -> ObservationCase:
    """将一条可用输入转为 inverse.py 使用的共同 case；不读取答案。"""
    if not record["observation_available"]:
        raise ValueError("该源射门未提供全部观测，须计入不可用样本，不能静默丢弃。")
    data = record["case"]
    fields = {"initial_time_s", "initial_position_m", "times_s", "positions_m",
              "goal_x_m", "goal_y_bounds_m", "goal_z_bounds_m"}
    if set(data) != fields:
        raise ValueError("观测 case 字段不符合共同接口。")
    case = ObservationCase(
        initial_time_s=float(data["initial_time_s"]), initial_position_m=_tuple3(_vector(data["initial_position_m"], "起点")),
        times_s=tuple(float(t) for t in data["times_s"]),
        positions_m=tuple(_tuple3(_vector(p, "观测位置")) for p in data["positions_m"]),
        goal_x_m=float(data["goal_x_m"]), goal_y_bounds_m=tuple(data["goal_y_bounds_m"]),
        goal_z_bounds_m=tuple(data["goal_z_bounds_m"]),
    )
    if len(case.times_s) != len(case.positions_m) or not case.times_s:
        raise ValueError("观测时刻与位置必须一一对应且非空。")
    times = (case.initial_time_s,)+case.times_s
    if not all(isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("观测时间须有限且严格递增。")
    return case


def load_trajectory_csv(path, *, length_unit: str, time_unit: str, fps: Optional[float] = None) -> dict:
    """读取一条已重建的轨迹：CSV 必须有 time,x,y,z 列，单位由调用者明确给出。

    time_unit='frame' 时必须给 fps。读取不自动识别起飞、对齐球门或划分数据，
    这些条件仍须逐条核查。后段轨迹仅供真值端使用。
    """
    scales = {"m": 1.0, "dm": .1, "cm": .01, "mm": .001}
    if length_unit not in scales or time_unit not in ("s", "ms", "frame"):
        raise ValueError("长度单位用 m/dm/cm/mm；时间单位用 s/ms/frame。")
    if time_unit == "frame" and (fps is None or not isfinite(fps) or fps <= 0):
        raise ValueError("帧号必须配套有限正 FPS。")
    time_scale = 1/fps if time_unit == "frame" else (.001 if time_unit == "ms" else 1.0)
    times, positions = [], []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"time", "x", "y", "z"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV 必须有 time,x,y,z 列。")
        for row in reader:
            times.append(float(row["time"])*time_scale)
            positions.append(_tuple3(_vector([float(row[c])*scales[length_unit] for c in ("x", "y", "z")], "CSV 位置")))
    if len(times) < 2 or not all(isfinite(t) for t in times) or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("CSV 至少需两个有限、严格递增的时间点，缺测需先核查。")
    return {"times_s": tuple(times), "positions_m": tuple(positions),
            "source_path": str(Path(path)), "length_unit_input": length_unit,
            "time_unit_input": time_unit, "fps": fps}


def main():
    parser = argparse.ArgumentParser(description="生成独立 H 测试球路；不是题目落点预测。")
    parser.add_argument("--sources", type=int, default=60, help="独立源射门数，不含噪声副本")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DATA_DIR/"synthetic_demo")
    parser.add_argument("--noise-mode", choices=("gaussian", "rounding", "none"), default="gaussian")
    parser.add_argument("--correlation", type=float, default=0.0)
    parser.add_argument("--wind-known", action="store_true", help="显式向预测器提供生成风速")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在，请使用新的 --output 目录，保留原数据。")
    dataset = build_dataset(args.sources, args.seed, noise_mode=args.noise_mode,
                            frame_correlation=args.correlation, wind_known=args.wind_known)
    output = save_dataset(dataset, args.output)
    print(json.dumps({"output": str(output.resolve()), "sources": args.sources,
                      "observation_cases": dataset["manifest"]["observation_count"],
                      "source_counts_by_split": dataset["manifest"]["source_counts_by_split"],
                      "event_counts": dataset["manifest"]["event_counts"],
                      "unavailable_sources": dataset["manifest"]["unavailable_source_count"],
                      "note": "合成压力测试数据；真值文件仅供评价，不交给预测器。"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
