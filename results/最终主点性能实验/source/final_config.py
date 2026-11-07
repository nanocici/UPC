"""最终交付的冻结常量；不改动物理方程或旧结果。

F1 = F 模型，b_y 先验 0.10；空间校准阈值与时间 q 来自最终验证实验。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

from config import INFERENCE


@dataclass(frozen=True)
class FinalConfig:
    model: str = "F"
    b_y_prior_std: float = 0.10
    b_z_prior_std: float = 0.20
    drag_coefficient: float = 0.22
    sigma_m: float = 0.01
    time_q_s: float = 0.015188910308210842
    spatial_threshold: float = 3.083525450931334
    mode_bandwidth_m: float = 0.10
    smooth_mode_yz_m: Tuple[float, float] = (2.875, 1.925)
    raw_grid_mode_yz_m: Tuple[float, float] = (2.8975, 1.8808)
    mean_auxiliary_yz_m: Tuple[float, float] = (2.9057531544942057, 1.9285065612977923)
    case_seed: int = 20261109
    base_samples: int = 2000
    total_proposals: int = 4000
    mode_seed_count: int = 4

    def inference_settings(self):
        """F1：只改 b_y 先验，其他共享设置不变。"""
        return replace(INFERENCE, vertical_force_prior_std=self.b_y_prior_std,
                       side_force_prior_std=self.b_z_prior_std)

    def uncertainty_settings(self):
        from uncertainty import UNCERTAINTY
        return replace(UNCERTAINTY, base_samples=self.base_samples,
                       max_sample_doublings=0)


FINAL = FinalConfig()
