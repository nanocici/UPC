"""UPC Problem B 的共享设置。所有长度用 m，时间用 s，质量用 kg。

三人从这里导入共同设置，避免在自己的文件中另写一份数值。
题面输入放在 CASE；其余默认参数均为建模报告中的工作假设。
在项目目录运行 python code/config.py，可检查设置并查看内容。
本文件不计算球路，不包含任何拟合结果。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite, pi
from pathlib import Path
from typing import Literal, Optional, TypedDict


# 顺序是三个人共同遵守的约定。
Vector3 = tuple[float, float, float]
COORDINATE_ORDER = ("x", "y", "z")
STATE_ORDER = ("x", "y", "z", "vx", "vy", "vz")
ModelId = Literal["G", "A", "S", "F", "H"]
EventId = Literal["goal", "ground", "boundary", "unresolved", "failure"]
MODEL_IDS = ("G", "A", "S", "F", "H")
PREDICTOR_MODELS = ("G", "A", "S", "F")  # H 是独立测试生成机制。


@dataclass(frozen=True)
class ObservationCase:
    """题面给定的数据；positions_m 的第 i 项对应 times_s 的第 i 项。"""

    initial_time_s: float = 0.0
    initial_position_m: Vector3 = (0.0, 0.0, 0.11)
    times_s: tuple[float, ...] = (0.05, 0.10)
    positions_m: tuple[Vector3, ...] = (
        (1.50, 0.45, 0.44),
        (2.96, 0.86, 0.74),
    )
    goal_x_m: float = 11.0
    goal_y_bounds_m: tuple[float, float] = (-3.66, 3.66)
    goal_z_bounds_m: tuple[float, float] = (0.0, 2.44)


@dataclass(frozen=True)
class Environment:
    """外部给定的物理情景，不由两帧同时拟合。"""

    mass_kg: float = 0.43
    radius_m: float = 0.11
    gravity_m_s2: float = 9.81
    air_density_kg_m3: float = 1.20
    kinematic_viscosity_m2_s: float = 1.5e-5
    wind_m_s: Vector3 = (0.0, 0.0, 0.0)

    @property
    def area_m2(self) -> float:
        return pi * self.radius_m**2

    @property
    def aerodynamic_k_per_m(self) -> float:
        """动力学中的 k = rho*A/(2*m)，单位 1/m。"""
        return self.air_density_kg_m3 * self.area_m2 / (2.0 * self.mass_kg)


@dataclass(frozen=True)
class SolverSettings:
    """数值积分的起步设置；收敛性仍需 physics.py 检查。"""

    method: str = "RK45"
    relative_tolerance: float = 1e-7
    position_absolute_tolerance_m: float = 1e-9
    velocity_absolute_tolerance_m_s: float = 1e-9
    max_step_s: float = 0.01
    initial_horizon_s: float = 2.0
    maximum_horizon_s: float = 4.0


@dataclass(frozen=True)
class InferenceSettings:
    """工作先验、误差和搜索范围；不是已测得的球参数。"""

    position_sigma_m: float = 0.01  # 六坐标独立同方差高斯情景。
    rounding_half_width_m: float = 0.005  # 舍入界限，与高斯情景分开。
    fixed_drag_coefficient: float = 0.22
    drag_bounds: tuple[float, float] = (0.08, 0.55)
    drag_prior_mean: float = 0.22
    drag_prior_std: float = 0.10  # 在 drag_bounds 内截断。
    transverse_prior_std: float = 0.20
    transverse_norm_max: float = 0.50  # F 中 sqrt(by**2+bz**2) 的上限。
    vx_bounds_m_s: tuple[float, float] = (0.0, 60.0)  # 下界严格：vx > 0。
    vy_bounds_m_s: tuple[float, float] = (-30.0, 30.0)
    vz_bounds_m_s: tuple[float, float] = (0.0, 30.0)
    max_iterations_per_start: int = 100
    prediction_region_mass: float = 0.90


CASE = ObservationCase()
ENVIRONMENT = Environment()
SOLVER = SolverSettings()
INFERENCE = InferenceSettings()
NOISE_SCENARIOS_M = (0.005, 0.010, 0.020, 0.050)
RANDOM_SEED = 20261107  # 复现用；每次实验另存实际种子。

# 固定阻力时，乙的优化向量必须使用此顺序。
S_PARAMETER_NAMES = ("vx0", "vy0", "vz0", "b_z")
F_PARAMETER_NAMES = ("vx0", "vy0", "vz0", "b_y", "b_z")
# 若以后释放 C_D，在上述向量最后增加 "drag_coefficient"。
# S/F 主要比较采用相同阻力处理；S 的 b_z 先验须与 F 边缘先验匹配。

# 用文件位置确定目录，Windows 和 Mac 均适用；导入时不创建文件夹。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


class DynamicsParameters(TypedDict):
    """传给甲的 F/S 运动程序的参数字典。S 必须设置 b_y=0。"""

    initial_velocity_m_s: Vector3
    drag_coefficient: float
    b_y: float
    b_z: float


class SimulationResult(TypedDict):
    """甲返回给乙、丙的共同格式；TypedDict 只定义格式，不运行求解。"""

    times_s: tuple[float, ...]
    # 与 times_s 等长；首个事件后或求解失败时未取得的位置用 None。
    positions_m: tuple[Optional[Vector3], ...]
    event: EventId
    # unresolved/failure 若无已确认事件，两项均为 None。
    event_time_s: Optional[float]
    event_position_m: Optional[Vector3]
    message: str


def validate_config() -> None:
    """只检查配置本身的基本一致性；不等于通过物理或模型验证。"""
    if len(CASE.times_s) != len(CASE.positions_m) or not CASE.times_s:
        raise ValueError("每个观测时刻必须对应一个三维位置。")
    times = (CASE.initial_time_s,) + CASE.times_s
    if not all(isfinite(t) for t in times):
        raise ValueError("时间必须是有限数值。")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("初始和观测时刻必须严格递增。")
    for vector in (CASE.initial_position_m, *CASE.positions_m, ENVIRONMENT.wind_m_s):
        if len(vector) != 3 or not all(isfinite(x) for x in vector):
            raise ValueError("位置和风必须按 (x,y,z) 给出三个有限数值。")
    for bounds in (CASE.goal_y_bounds_m, CASE.goal_z_bounds_m):
        if not all(isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError("球门边界必须有限，且下界小于上界。")
    positive = (
        ENVIRONMENT.mass_kg, ENVIRONMENT.radius_m, ENVIRONMENT.gravity_m_s2,
        ENVIRONMENT.air_density_kg_m3, ENVIRONMENT.kinematic_viscosity_m2_s,
        SOLVER.relative_tolerance, SOLVER.position_absolute_tolerance_m,
        SOLVER.velocity_absolute_tolerance_m_s, SOLVER.max_step_s,
        INFERENCE.position_sigma_m, INFERENCE.rounding_half_width_m,
        INFERENCE.drag_prior_std, INFERENCE.transverse_prior_std,
        INFERENCE.transverse_norm_max,
    )
    if not all(isfinite(x) and x > 0 for x in positive):
        raise ValueError("物理量、误差尺度和数值容差必须为有限正数。")
    if CASE.initial_position_m[2] < ENVIRONMENT.radius_m:
        raise ValueError("球心起始高度不能低于球半径。")
    if not CASE.initial_position_m[0] < CASE.goal_x_m:
        raise ValueError("球门必须位于起点的 +x 方向。")
    lo, hi = INFERENCE.drag_bounds
    if not 0 <= lo < hi or not lo <= INFERENCE.fixed_drag_coefficient <= hi:
        raise ValueError("名义阻力必须位于非负阻力搜索区间内。")
    if not lo <= INFERENCE.drag_prior_mean <= hi:
        raise ValueError("当前工作先验的阻力中心应在搜索区间内。")
    if not 0 < INFERENCE.prediction_region_mass < 1:
        raise ValueError("预测区域的概率质量必须在 0 和 1 之间。")
    if not CASE.times_s[-1] < SOLVER.initial_horizon_s <= SOLVER.maximum_horizon_s:
        raise ValueError("积分时间窗必须覆盖观测，最大时间窗不得更小。")
    if not all(isfinite(x) and x > 0 for x in NOISE_SCENARIOS_M):
        raise ValueError("噪声测试档位必须为有限正数。")


if __name__ == "__main__":
    validate_config()
    print("Config check passed. Settings only; no trajectory has been calculated.")
    print(json.dumps({
        "case": asdict(CASE),
        "environment_working_assumptions": asdict(ENVIRONMENT),
        "solver": asdict(SOLVER),
        "inference_working_assumptions": asdict(INFERENCE),
        "noise_scenarios_m": NOISE_SCENARIOS_M,
        "random_seed": RANDOM_SEED,
        "project_root": str(PROJECT_ROOT),
    }, indent=2))
