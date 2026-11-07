"""固定 Cd 的条件后验重要性采样与门线传播。

拟合/先验与 inverse.py 一致。在缩放参数坐标中建立含宽尾分量的提议，
用真实似然和先验修正权重，不把观测扰动重拟合当作贝叶斯后验。
不包含 Cd 积分、模型平均、区域外部校准或风情景频率混合。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from math import isfinite, pi
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from config import (CASE, DATA_DIR, ENVIRONMENT, INFERENCE, RANDOM_SEED,
                    RESULTS_DIR, SOLVER, Environment)
from data import SPLITS, load_observations, observation_case
from inverse import fit_case, transverse_prior_penalty
from physics import is_in_goal_opening, simulate


@dataclass(frozen=True)
class UncertaintySettings:
    base_samples: int = 2000  # 实际固定提议数为 2N，比较前 N 与全部 2N。
    min_effective_samples: float = 300.0
    min_conditional_effective_samples: float = 100.0
    event_probability_tolerance: float = 0.02
    mean_shift_tolerance_m: float = 0.05
    relative_area_tolerance: float = 0.15
    unknown_mass_tolerance: float = 1e-6
    grid_width_m: float = 2.44/24
    mode_shift_tolerance_m: float = 0.20
    student_df: float = 5.0
    broad_component_weight: float = 0.15
    broad_standard_deviation_factor: float = 3.0
    max_sample_doublings: int = 0  # Revision experiments explicitly allow up to two extensions.


UNCERTAINTY = UncertaintySettings()
EVENT_CLASSES = ("goal_inside", "goal_outside", "ground", "boundary", "unresolved", "failure")


def weighted_quantile(values, weights, probability):
    """加权经验 CDF 的广义逆，不做高斯假设。"""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if (values.ndim != 1 or values.shape != weights.shape or not len(values)
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0 or not 0 <= probability <= 1):
        raise ValueError("加权分位数需要有限值、非负且非零权重和 [0,1] 概率。")
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])/weights.sum()
    index = min(int(np.searchsorted(cumulative, probability, side="left")), len(order)-1)
    return float(values[order[index]])


def normalize_log_weights(log_weights):
    values = np.asarray(log_weights, float)
    if values.ndim != 1 or np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("对数权重须是一维有限值或负无穷。")
    if not np.any(np.isfinite(values)):
        raise ValueError("全部候选权重为零，不能构成后验。")
    return np.exp(values-logsumexp(values))


def effective_sample_size(weights):
    weights = np.asarray(weights, float)
    if not len(weights) or np.sum(weights) <= 0:
        return 0.0
    normalized = weights/np.sum(weights)
    return float(1/np.dot(normalized, normalized))


def _parameters(model, vector, cd):
    return {"initial_velocity_m_s": tuple(float(x) for x in vector[:3]),
            "drag_coefficient": cd, "b_y": float(vector[3]) if model == "F" else 0.0,
            "b_z": float(vector[-1])}


def _vector(model, parameters):
    return np.array([*parameters["initial_velocity_m_s"],
                     *([parameters["b_y"]] if model == "F" else []), parameters["b_z"]])


def _retry_solver(solver):
    return replace(solver, relative_tolerance=solver.relative_tolerance/10,
                   position_absolute_tolerance_m=solver.position_absolute_tolerance_m/10,
                   velocity_absolute_tolerance_m_s=solver.velocity_absolute_tolerance_m_s/10,
                   max_step_s=solver.max_step_s/2)


class _Posterior:
    def __init__(self, model, fit, case, environment, solver, settings):
        self.model, self.fit, self.case = model, fit, case
        self.environment, self.solver, self.settings = environment, solver, settings
        self.scales = np.asarray(fit["parameter_scales"])
        self.center = _vector(model, fit["parameters"])/self.scales
        self.cd = fit["drag_coefficient"]
        self.cholesky = np.linalg.cholesky(np.asarray(fit["observation_covariance_m2"]))
        duration = float(np.nextafter(case.times_s[-1]-case.initial_time_s, np.inf))
        self.window_solver = replace(solver, initial_horizon_s=duration, maximum_horizon_s=duration)
        self.cache = {}
        self.retry_count = 0
        self.failure_count = 0

    def prior(self, q):
        vector = np.asarray(q)*self.scales
        bounds = (self.settings.vx_bounds_m_s, self.settings.vy_bounds_m_s, self.settings.vz_bounds_m_s)
        if not np.all(np.isfinite(vector)) or vector[0] < max(1e-6, bounds[0][0]):
            return float("inf")
        if any(not lo <= v <= hi for v, (lo, hi) in zip(vector[:3], bounds)):
            return float("inf")
        return transverse_prior_penalty(self.model, vector[3:], self.settings)

    def log_density(self, q):
        """q 坐标中的未归一化后验；固定缩放雅可比为常数，可共同约去。"""
        key = tuple(q)
        if key in self.cache:
            return self.cache[key]
        prior = self.prior(q)
        if not isfinite(prior):
            return -np.inf
        params = _parameters(self.model, np.asarray(q)*self.scales, self.cd)
        result = simulate(self.model, params, self.case.times_s, self.case,
                          self.environment, self.window_solver)
        if result["event"] == "failure":
            self.retry_count += 1
            result = simulate(self.model, params, self.case.times_s, self.case,
                              self.environment, _retry_solver(self.window_solver))
        if result["event"] == "failure":
            self.failure_count += 1
            raise RuntimeError("候选在观测窗数值失败，重试后仍无法计算似然；停止发布概率。")
        if any(p is None for p in result["positions_m"]):
            value = -np.inf  # 与自由飞行观测窗不相容，确切零似然。
        else:
            residual = (np.asarray(result["positions_m"])-self.case.positions_m).ravel()
            whitened = np.linalg.solve(self.cholesky, residual)
            value = -.5*float(np.dot(whitened, whitened))-prior
        self.cache[key] = value
        return value


def _hessian(function, center, step=1e-3):
    """缩放坐标中的完整数值曲率；含真实似然的非线性项及先验。"""
    dimension = len(center)
    base = function(center)
    if not isfinite(base):
        return None
    h = step*np.maximum(1, np.abs(center))
    result = np.zeros((dimension, dimension))
    for i in range(dimension):
        ei = np.zeros(dimension)
        ei[i] = h[i]
        plus, minus = function(center+ei), function(center-ei)
        if not all(isfinite(v) for v in (plus, minus)):
            return None
        result[i, i] = (plus-2*base+minus)/h[i]**2
        for j in range(i):
            ej = np.zeros(dimension)
            ej[j] = h[j]
            values = [function(center+ei+ej), function(center+ei-ej),
                      function(center-ei+ej), function(center-ei-ej)]
            if not all(isfinite(v) for v in values):
                return None
            result[i, j] = result[j, i] = (values[0]-values[1]-values[2]+values[3])/(4*h[i]*h[j])
    return result


def proposal_covariance(posterior):
    """优先完整后验曲率；GN+先验退路只作为提议形状，不冒充高斯后验。"""
    precision = _hessian(lambda q: -posterior.log_density(q), posterior.center)
    kind = "full_posterior_hessian"
    if precision is None or np.min(np.linalg.eigvalsh(precision)) <= 0:
        jac = posterior.fit["diagnostics"]["data_jacobian_scaled"]
        prior_hessian = _hessian(posterior.prior, posterior.center)
        if jac is None or prior_hessian is None:
            raise ValueError("局部曲率不可用，需参数剖面/其他提议，不能伪造协方差。")
        precision = np.asarray(jac).T@np.asarray(jac)+prior_hessian
        kind = "gauss_newton_plus_prior_proposal_only"
    precision = (precision+precision.T)/2
    np.linalg.cholesky(precision)
    covariance = np.linalg.inv(precision)
    covariance = (covariance+covariance.T)/2
    np.linalg.cholesky(covariance)
    return covariance, kind


def _proposal(posterior, covariance, settings):
    centers = [posterior.center]
    # 保留多起点发现的不同高权重参数区，未发现不等于证明没有其他模式。
    best_log = posterior.log_density(posterior.center)
    for run in posterior.fit["optimizer_runs"]:
        if not run.get("success"):
            continue
        q = _vector(posterior.model, run["parameters"])/posterior.scales
        if posterior.log_density(q) < best_log-30:
            continue
        if min(float((q-c)@np.linalg.solve(covariance, q-c)) for c in centers) > 4:
            centers.append(q)
    distributions, probabilities = [], []
    # scipy 的 t.shape 不等于协方差；df>2 时乘 (df-2)/df 作转换。
    shape = covariance*(settings.student_df-2)/settings.student_df
    for center in centers:
        for factor, probability in ((1., 1-settings.broad_component_weight),
                                    (settings.broad_standard_deviation_factor**2, settings.broad_component_weight)):
            distributions.append(multivariate_t(loc=center, shape=shape*factor, df=settings.student_df))
            probabilities.append(probability/len(centers))
    return distributions, np.asarray(probabilities)


def draw_proposal(distributions, probabilities, count, rng):
    probabilities = np.asarray(probabilities, float)
    if (len(distributions) != len(probabilities) or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1)):
        raise ValueError("提议分量概率须为正且总和为 1。")
    labels = rng.choice(len(distributions), size=count, p=probabilities)
    points = np.empty((count, len(np.asarray(distributions[0].loc))))
    for i, distribution in enumerate(distributions):
        mask = labels == i
        if np.any(mask):
            points[mask] = np.asarray(distribution.rvs(size=int(mask.sum()), random_state=rng)).reshape(
                int(mask.sum()), points.shape[1])
    log_q = logsumexp(np.vstack([np.log(p)+d.logpdf(points)
                                for p, d in zip(probabilities, distributions)]), axis=0)
    return points, np.asarray(log_q)


def grid_mode(points, weights, case=CASE, width=None, shift=0.0):
    width = UNCERTAINTY.grid_width_m if width is None else width
    points, weights = np.asarray(points, float), np.asarray(weights, float)
    if (points.ndim != 2 or points.shape[1] != 2 or weights.shape != (len(points),)
            or not len(points) or not isfinite(width) or width <= 0
            or not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("众数格网需有效二维点、非负权重和正格宽。")
    anchor = np.array((case.goal_y_bounds_m[0], case.goal_z_bounds_m[0]))+shift
    indices = np.floor((points-anchor)/width).astype(np.int64)
    cells, inverse = np.unique(indices, axis=0, return_inverse=True)
    masses = np.bincount(inverse, weights=weights/weights.sum())
    best = int(np.argmax(masses))
    return {"center_yz_m": (anchor+(cells[best]+.5)*width).tolist(),
            "cell_probability": float(masses[best]), "grid_width_m": width,
            "grid_shift_m": shift}


def _conditional_summary(draws, case, settings, mass):
    weights = np.array([d["weight"] for d in draws])
    total_mass = float(weights.sum())
    if total_mass <= 0:
        return None
    weights /= total_mass
    points = np.asarray([d["goal_yz_m"] for d in draws])
    times = np.asarray([d["arrival_time_s"] for d in draws])
    mean = weights@points
    centered = points-mean
    covariance = (centered*weights[:, None]).T@centered
    ess = effective_sample_size(weights)
    mode = grid_mode(points, weights, case, settings.grid_width_m)
    variants = [grid_mode(points, weights, case, settings.grid_width_m/2),
                grid_mode(points, weights, case, settings.grid_width_m, settings.grid_width_m/2)]
    max_mode_shift = max(float(np.linalg.norm(np.array(v["center_yz_m"])-mode["center_yz_m"])) for v in variants)
    region, area = None, None
    if ess >= settings.min_conditional_effective_samples and len(points) >= 3:
        try:
            factor = np.linalg.cholesky(covariance)
            distances = np.sum(np.linalg.solve(factor, centered.T)**2, axis=0)
            threshold = weighted_quantile(distances, weights, mass)
            area = float(pi*threshold*np.prod(np.diag(factor)))
            if not isfinite(area) or area <= 0:
                raise ValueError("预测区域面积无效。")
            region = {"center_yz_m": mean.tolist(), "covariance_m2": covariance.tolist(), "threshold": threshold}
        except (np.linalg.LinAlgError, ValueError):
            region, area = None, None
    return {"posterior_mass": total_mass, "effective_samples": ess, "mean_yz_m": mean.tolist(),
            "most_likely_cell": mode, "grid_variants": variants,
            "mode_grid_stable": max_mode_shift <= settings.mode_shift_tolerance_m,
            "max_grid_mode_shift_m": max_mode_shift, "region": region, "region_area_m2": area,
            "arrival_time_median_s": weighted_quantile(times, weights, .5),
            "arrival_time_90_interval_s": [weighted_quantile(times, weights, .05), weighted_quantile(times, weights, .95)]}


def summarize_candidates(draws, case=CASE, settings=UNCERTAINTY, region_mass=.90):
    """归一化全部已知后验质量；先接地及传播失败不从分母删除。"""
    if not draws or not 0 < region_mass < 1:
        raise ValueError("需要候选及有效的区域概率。")
    weights = np.array([d["weight"] for d in draws], float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("候选权重无效。")
    normalized = [{**d, "weight": float(w/weights.sum())} for d, w in zip(draws, weights)]
    probabilities = {event: sum(d["weight"] for d in normalized if d["event"] == event)
                     for event in EVENT_CLASSES}
    positive = [d for d in normalized if d["weight"] > 0]
    if any(d["event"] not in EVENT_CLASSES for d in positive):
        raise ValueError("正权重候选的事件类别未知。")
    # 仅修正有限加法舍入，使单个概率不会因 1+epsilon 被评价器拒绝。
    class_total = sum(probabilities.values())
    probabilities = {event: value/class_total for event, value in probabilities.items()}
    direct = [d for d in positive if d["event"] in ("goal_inside", "goal_outside")]
    inside = [d for d in positive if d["event"] == "goal_inside"]
    unknown = probabilities["boundary"]+probabilities["unresolved"]+probabilities["failure"]
    return {"effective_samples": effective_sample_size(weights), "event_probabilities": probabilities,
            "unknown_event_mass": unknown,
            "direct_inside_probability_bounds": [probabilities["goal_inside"], min(1., probabilities["goal_inside"]+unknown)],
            "direct_crossing": _conditional_summary(direct, case, settings, region_mass),
            "inside_conditioned": _conditional_summary(inside, case, settings, region_mass)}


def _convergence(first, full, settings):
    delta = max(abs(first["event_probabilities"][e]-full["event_probabilities"][e]) for e in EVENT_CLASSES)
    a, b = first["direct_crossing"], full["direct_crossing"]
    mean_shift, relative_area, mode_shift = None, None, None
    mean_stable = a is None and b is None
    area_stable = a is None and b is None
    if a is not None and b is not None:
        mean_shift = float(np.linalg.norm(np.array(a["mean_yz_m"])-b["mean_yz_m"]))
        mode_shift = float(np.linalg.norm(np.array(a["most_likely_cell"]["center_yz_m"])
                                         - b["most_likely_cell"]["center_yz_m"]))
        mean_stable = mean_shift <= settings.mean_shift_tolerance_m
        if a["region_area_m2"] is not None and b["region_area_m2"] is not None:
            relative_area = abs(a["region_area_m2"]-b["region_area_m2"])/b["region_area_m2"]
            area_stable = relative_area <= settings.relative_area_tolerance
    stable = delta <= settings.event_probability_tolerance and mean_stable and area_stable
    return {"sampling_stable": stable, "max_event_probability_change": delta,
            "conditional_mean_shift_m": mean_shift, "relative_region_area_change": relative_area,
            "conditional_mode_cell_shift_m": mode_shift,
            "mode_sampling_stable": mode_shift <= settings.mode_shift_tolerance_m if mode_shift is not None else a is None and b is None}


def certify_outputs(first, full, settings):
    """Separate certificates; unresolved mass remains in the full posterior."""
    conv = _convergence(first, full, settings)
    enough = full['effective_samples'] >= settings.min_effective_samples
    known = full['unknown_event_mass'] <= settings.unknown_mass_tolerance
    event = bool(enough and full['unknown_event_mass'] == 0 and
                 conv['max_event_probability_change'] <= settings.event_probability_tolerance)
    a, b = first['direct_crossing'], full['direct_crossing']
    mean = bool(enough and known and a is not None and b is not None and
                b['effective_samples'] >= settings.min_conditional_effective_samples and
                conv['conditional_mean_shift_m'] is not None and
                conv['conditional_mean_shift_m'] <= settings.mean_shift_tolerance_m)
    region = bool(mean and b['region'] is not None and
                  conv['relative_region_area_change'] is not None and
                  conv['relative_region_area_change'] <= settings.relative_area_tolerance)
    return dict(event_probability=event, conditional_mean=mean, prediction_region=region)


def sample_case(
    model="F", case=CASE, environment=ENVIRONMENT, solver=SOLVER, settings=INFERENCE,
    *, uncertainty=UNCERTAINTY, drag_coefficient=None, sigma_m=None, frame_correlation=0.,
    covariance_m2=None, seed=RANDOM_SEED, starts=3, progress=None,
):
    """固定 Cd 后验，固定分布抽 2N 次；第一 N 次仅用于加倍稳定性比较。"""
    wall_start = time.perf_counter()
    if not isinstance(uncertainty.base_samples, int) or uncertainty.base_samples < 2:
        raise ValueError("base_samples 必须为不小于 2 的整数。")
    if (not isinstance(uncertainty.max_sample_doublings, int) or isinstance(uncertainty.max_sample_doublings, bool)
            or not 0 <= uncertainty.max_sample_doublings <= 2):
        raise ValueError("追加预算加倍次数须为0、1或2。")
    positive_settings = (uncertainty.min_effective_samples, uncertainty.min_conditional_effective_samples,
        uncertainty.event_probability_tolerance, uncertainty.mean_shift_tolerance_m,
        uncertainty.relative_area_tolerance, uncertainty.grid_width_m, uncertainty.mode_shift_tolerance_m)
    if (not all(isfinite(x) and x > 0 for x in positive_settings)
            or not isfinite(uncertainty.student_df) or uncertainty.student_df <= 2
            or not 0 < uncertainty.broad_component_weight < 1
            or not isfinite(uncertainty.broad_standard_deviation_factor)
            or uncertainty.broad_standard_deviation_factor <= 1
            or not 0 <= uncertainty.unknown_mass_tolerance < 1):
        raise ValueError("采样阈值、t 分布和混合提议设置无效。")
    result = {"model": model, "success": False, "message": "", "fit": None,
              "summary": None, "diagnostics": {}, "draws": []}
    fit = fit_case(model, case, environment, solver, settings, mode="map",
                   drag_coefficient=drag_coefficient, sigma_m=sigma_m,
                   frame_correlation=frame_correlation, covariance_m2=covariance_m2, starts=starts)
    result["fit"] = fit
    if not fit["success"]:
        result["message"] = "MAP 反演未收敛，无法构造条件后验提议。"
        result["wall_time_s"] = time.perf_counter()-wall_start
        return result
    posterior = _Posterior(model, fit, case, environment, solver, settings)
    try:
        covariance, kind = proposal_covariance(posterior)
        distributions, mixture_weights = _proposal(posterior, covariance, uncertainty)
        rng = np.random.default_rng(seed)
        draws, all_targets, all_proposals, stages = [], [], [], []
        for extension in range(uncertainty.max_sample_doublings+1):
            count = 2*uncertainty.base_samples*2**extension
            points, log_q = draw_proposal(distributions, mixture_weights, count-len(draws), rng)
            targets = np.array([posterior.log_density(q) for q in points])
            for q, target, proposal in zip(points, targets, log_q):
                params = _parameters(model, q*posterior.scales, posterior.cd)
                draw = {"parameters": (q*posterior.scales).tolist(), "weight": 0.,
                        "event": "rejected", "goal_yz_m": None, "arrival_time_s": None,
                        "log_target": float(target) if isfinite(target) else None,
                        "log_proposal": float(proposal)}
                if isfinite(target):
                    trajectory = simulate(model, params, case.times_s, case, environment, solver)
                    if trajectory["event"] == "failure":
                        trajectory = simulate(model, params, case.times_s, case, environment, _retry_solver(solver))
                    event = trajectory["event"]
                    if event != "failure" and any(p is None for p in trajectory["positions_m"]):
                        event = "failure"
                    if event == "goal":
                        draw.update(event="goal_inside" if is_in_goal_opening(trajectory["event_position_m"], case)
                                    else "goal_outside", goal_yz_m=list(trajectory["event_position_m"][1:]),
                                    arrival_time_s=trajectory["event_time_s"])
                    else:
                        draw["event"] = event
                draws.append(draw)
                if progress is not None and (len(draws)%500 == 0 or len(draws)==count):
                    progress(len(draws),count)
            all_targets.extend(targets.tolist()); all_proposals.extend(log_q.tolist())
            log_target=np.asarray(all_targets); log_q=np.asarray(all_proposals)
            weights=normalize_log_weights(log_target-log_q)
            for draw,weight in zip(draws,weights): draw['weight']=float(weight)
            summary=summarize_candidates(draws,case,uncertainty,settings.prediction_region_mass)
            half=count//2
            prefix_weights=normalize_log_weights((log_target-log_q)[:half])
            prefix=[{**d,"weight":float(w)} for d,w in zip(draws[:half],prefix_weights)]
            first=summarize_candidates(prefix,case,uncertainty,settings.prediction_region_mass)
            convergence=_convergence(first,summary,uncertainty)
            validity=certify_outputs(first,summary,uncertainty)
            stages.append(dict(total_proposals=count, effective_samples=summary['effective_samples'],
                               unknown_event_mass=summary['unknown_event_mass'],
                               output_validity=validity, convergence=convergence))
            # Same joint stopping rule as baseline; output release is separately certified.
            if (summary['effective_samples'] >= uncertainty.min_effective_samples and
                convergence['sampling_stable'] and summary['unknown_event_mass'] <= uncertainty.unknown_mass_tolerance):
                break
        for condition in ("direct_crossing", "inside_conditioned"):
            a,b=first[condition],summary[condition]
            if b is not None:
                shift=(float(np.linalg.norm(np.array(a["most_likely_cell"]["center_yz_m"])-
                       b["most_likely_cell"]["center_yz_m"])) if a is not None else None)
                b['mode_sampling_shift_m']=shift
                b['mode_stable']=bool(b['mode_grid_stable'] and shift is not None and
                    shift <= uncertainty.mode_shift_tolerance_m and
                    b['effective_samples'] >= uncertainty.min_conditional_effective_samples)
        # 检查局部高斯的参数边界截断率，独立随机流，不改变重要性样本。
        probes = np.random.default_rng(seed+1).multivariate_normal(posterior.center, covariance, size=256)
        outside = float(np.mean([not isfinite(posterior.prior(q)) for q in probes]))
        local_valid = (kind == "full_posterior_hessian" and not fit["diagnostics"]["boundary_hits"] and outside <= .05)
        success = (summary["effective_samples"] >= uncertainty.min_effective_samples
                   and convergence["sampling_stable"]
                   and summary["unknown_event_mass"] <= uncertainty.unknown_mass_tolerance)
        result.update(success=bool(success), summary=summary, draws=draws, output_validity=validity,
            message="条件后验采样通过本轮诊断；区域尚未经外部校准。" if success else
                    "采样/事件诊断未通过，保留诊断与候选，不发布正式概率预测。",
            diagnostics={**convergence, "sampling_stages": stages, "output_validity": validity,
                "max_sample_doublings": uncertainty.max_sample_doublings, "base_samples": uncertainty.base_samples,
                "total_proposals": count, "positive_weight_count": int(np.sum(weights > 0)),
                "proposal_component_count": len(distributions), "proposal_probabilities": mixture_weights.tolist(),
                "proposal_covariance_q": covariance.tolist(), "curvature_source": kind,
                "local_gaussian_boundary_rejection_fraction": outside,
                "local_gaussian_basic_checks_passed": bool(local_valid),
                "posterior_assumed_gaussian": False, "likelihood_retries": posterior.retry_count,
                "unresolved_likelihood_failures": posterior.failure_count})
        if success and summary["direct_crossing"] is not None and not summary["direct_crossing"]["mode_stable"]:
            result["message"] = "均值、区域和事件采样通过诊断；众数格尚不稳定，不能宣称峰值位置已稳定。"
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        result["message"] = str(error)
        result["diagnostics"] = {"likelihood_retries": posterior.retry_count,
                                  "unresolved_likelihood_failures": posterior.failure_count}
    result["wall_time_s"] = time.perf_counter()-wall_start
    return result


def uncertainty_observation(record, model="F", *, uncertainty=UNCERTAINTY,
                            drag_coefficient=None, sigma_override_m=None, seed=RANDOM_SEED, starts=3, settings=INFERENCE):
    """返回 evaluate 格式的条件均值/区域/概率，以及完整采样日志。"""
    from evaluate import point_prediction
    cd = INFERENCE.fixed_drag_coefficient if drag_coefficient is None else drag_coefficient
    method = f"{model}_cd{cd:g}_posterior"
    prediction = point_prediction(record, method, status="failure", point_kind="mean")
    prediction["region_mass"] = INFERENCE.prediction_region_mass
    if not record["observation_available"]:
        prediction.update(status="unavailable", message="没有完整观测。")
        return prediction, {"success": False, "message": prediction["message"], "draws": []}
    try:
        case = observation_case(record)
        known = dict(record["known_environment"])
        known["wind_m_s"] = tuple(known["wind_m_s"])
        measurement = record["measurement"]
        if sigma_override_m is None and measurement["mode"] != "gaussian":
            raise ValueError("非高斯误差暂需显式 sigma 高斯工作假设。")
        result = sample_case(model, case, Environment(**known), settings=settings, uncertainty=uncertainty,
            drag_coefficient=drag_coefficient,
            sigma_m=measurement["sigma_m"] if sigma_override_m is None else sigma_override_m,
            frame_correlation=measurement["frame_correlation"] if measurement["mode"] == "gaussian" else 0.,
            seed=seed, starts=starts)
        prediction["message"] = result["message"]
        validity = result.get('output_validity', {})
        if result.get('summary') is not None and any(validity.values()):
            summary=result['summary']; direct=summary['direct_crossing']
            prediction['status']='ok'
            if validity.get('conditional_mean'):
                prediction.update(point_yz_m=direct['mean_yz_m'],arrival_time_s=direct['arrival_time_median_s'])
            if validity.get('prediction_region'):
                prediction['region']=direct['region']
            if validity.get('event_probability'):
                prediction['goal_probability']=summary['event_probabilities']['goal_inside']
            if not result['success']:
                prediction['message']='部分输出通过独立认证；缺失项及联合失败原因见诊断。'
        return prediction, result
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        prediction["message"] = str(error)
        return prediction, {"success": False, "message": str(error), "draws": []}


def _save_run(output, records, predictions, config):
    compact = [{k: v for k, v in record.items() if k != "draws"} for record in records]
    summary_text = json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False)
    prediction_text = "".join(json.dumps(p, ensure_ascii=False, allow_nan=False)+"\n" for p in predictions)
    config_text = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    (output/"summary.json").write_text(summary_text+"\n", encoding="utf-8")
    (output/"run_config.json").write_text(config_text+"\n", encoding="utf-8")
    if predictions:
        (output/"predictions.jsonl").write_text(prediction_text, encoding="utf-8")
    for index, record in enumerate(records):
        draws = record.get("draws", [])
        if draws:
            with (output/f"candidates_{index:04d}.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(draws[0]))
                writer.writeheader()
                writer.writerows(draws)


def plot_distribution(result, output_path, case=CASE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle
    figure, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    draws = [d for d in result["draws"] if d["weight"] > 0 and d["goal_yz_m"] is not None]
    if draws:
        points = np.asarray([d["goal_yz_m"] for d in draws])
        weights = np.asarray([d["weight"] for d in draws])
        ax.scatter(points[:, 0], points[:, 1], s=3+35*weights/weights.max(), alpha=.25, label="Weighted candidates")
    ax.add_patch(Rectangle((case.goal_y_bounds_m[0], case.goal_z_bounds_m[0]),
                          case.goal_y_bounds_m[1]-case.goal_y_bounds_m[0],
                          case.goal_z_bounds_m[1]-case.goal_z_bounds_m[0], fill=False, color="black"))
    direct = result["summary"]["direct_crossing"] if result.get("summary") else None
    if direct:
        mean = direct["mean_yz_m"]
        ax.scatter(*mean, marker="x", s=90, color="red", label="Conditional mean")
        ax.scatter(*direct["most_likely_cell"]["center_yz_m"], marker="+", s=90, color="green", label="Mode grid cell")
        region = direct["region"]
        if region:
            values, vectors = np.linalg.eigh(region["covariance_m2"])
            angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
            ax.add_patch(Ellipse(mean, 2*np.sqrt(values[1]*region["threshold"]),
                                2*np.sqrt(values[0]*region["threshold"]), angle=angle,
                                fill=False, color="red", label="Model-based region (uncalibrated)"))
    ax.set(xlabel="Y (m)", ylabel="Z (m)", title=f"{result.get('model', '')}: fixed-drag conditional posterior")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="固定阻力条件后验；默认题目算例 F。")
    parser.add_argument("--model", choices=("S", "F", "both"), default="F")
    parser.add_argument("--samples", type=int, default=UNCERTAINTY.base_samples, help="基础 N，实际提议数为 2N")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--drag", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR/"uncertainty_example")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在，请指定新的 --output。")
    models = ("S", "F") if args.model == "both" else (args.model,)
    options = replace(UNCERTAINTY, base_samples=args.samples)
    records, predictions = [], []
    if args.dataset is None:
        for model in models:
            result = sample_case(model, uncertainty=options, drag_coefficient=args.drag, sigma_m=args.sigma,
                seed=args.seed, starts=args.starts,
                progress=lambda i, n: print(f"Propagated {i}/{n}", file=sys.stderr, flush=True))
            records.append(result)
    else:
        observations = load_observations(args.dataset/"observations"/f"{args.split}.jsonl")
        for index, observation in enumerate(observations):
            for model in models:
                prediction, result = uncertainty_observation(observation, model, uncertainty=options,
                    drag_coefficient=args.drag, sigma_override_m=args.sigma,
                    seed=args.seed+index, starts=args.starts)
                result.update(case_id=observation["case_id"], source_id=observation["source_id"], model=model)
                predictions.append(prediction)
                records.append(result)
            print(f"Processed {index+1}/{len(observations)}", file=sys.stderr, flush=True)
    config = {"uncertainty": asdict(options), "inference": asdict(INFERENCE), "solver": asdict(SOLVER),
        "default_environment": asdict(ENVIRONMENT), "starts": args.starts,
        "models": models, "drag_override": args.drag, "sigma_override_m": args.sigma, "seed": args.seed,
        "dataset": str(args.dataset.resolve()) if args.dataset else None,
        "case": asdict(CASE) if args.dataset is None else None, "split": args.split if args.dataset else None,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "note": "固定 Cd 的条件后验；区域未经外部校准。尚未整合 Cd、未知环境与模型形式不确定性。"}
    _save_run(args.output, records, predictions, config)
    if args.plot and args.dataset is None:
        for result in records:
            plot_distribution(result, args.output/f"distribution_{result['model']}.png")
    print(json.dumps({"output": str(args.output.resolve()), "case_count": len(records),
        "passed_diagnostics": sum(r["success"] for r in records),
        "results": [{"model": r.get("model"), "success": r["success"], "message": r["message"],
                     "effective_samples": r["summary"]["effective_samples"] if r.get("summary") else None}
                    for r in records] if args.dataset is None else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
