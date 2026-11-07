"""Numerically validated acceleration of current F1 importance sampling.

MAP, parameter scaling, mixture-t proposal, likelihood, prior, event definitions
and weights match inverse.py/uncertainty.py. Only ODE evaluation is batched in C.
"""
from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path
import numpy as np

from config import SOLVER
from inverse import fit_case
from uncertainty import (_Posterior, proposal_covariance, _proposal, draw_proposal,
                         normalize_log_weights, effective_sample_size, weighted_quantile,
                         grid_mode, UNCERTAINTY, _convergence, certify_outputs)

_LIB = None
EVENTS = {0: "rejected", 1: "goal_inside", 2: "goal_outside", 3: "ground",
          4: "boundary", 5: "unresolved", 6: "failure"}


def native_library():
    global _LIB
    if _LIB is None:
        source = Path(__file__).with_name("batch_f_dynamics.c")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        target = source.parent/"__pycache__"/f"batch_f_{digest}.dylib"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            subprocess.run(["clang", "-O3", "-shared", "-fPIC", str(source), "-o", str(target)], check=True)
        _LIB = ctypes.CDLL(str(target))
        arr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        _LIB.propagate_batch.argtypes = [ctypes.c_size_t, arr, arr, arr, ctypes.c_double, arr, ints, arr]
    return _LIB


def propagate(parameters, case, environment, cd=.22, step=.005):
    if case.initial_time_s != 0 or tuple(case.times_s) != (.05, .10):
        raise ValueError("Batch integrator requires the statement observation times.")
    p = np.ascontiguousarray(parameters, dtype=float)
    env = np.array([environment.aerodynamic_k_per_m, environment.gravity_m_s2,
                    *environment.wind_m_s, cd], float)
    geometry = np.array([*case.initial_position_m, case.goal_x_m, environment.radius_m,
                         *case.goal_y_bounds_m, *case.goal_z_bounds_m, SOLVER.maximum_horizon_s], float)
    obs = np.empty((len(p), 6)); event = np.empty(len(p), np.int32); yzt = np.empty((len(p), 3))
    native_library().propagate_batch(len(p), p, env, geometry, step, obs, event, yzt)
    return obs, event, yzt


def prepare(case, environment, settings, sigma, correlation=0):
    fit = fit_case("F", case, environment, settings=settings, mode="map",
                   drag_coefficient=settings.fixed_drag_coefficient, sigma_m=sigma,
                   frame_correlation=correlation, starts=3)
    if not fit["success"]:
        return fit, None, None, None
    post = _Posterior("F", fit, case, environment, SOLVER, settings)
    cov, kind = proposal_covariance(post)
    distributions, probabilities = _proposal(post, cov, UNCERTAINTY)
    return fit, post, distributions, probabilities


def sample(post, distributions, probabilities, seed, count=64000, step=.005):
    q, logq = draw_proposal(distributions, probabilities, count, np.random.default_rng(seed))
    physical = q*post.scales
    settings = post.settings
    valid = np.all(np.isfinite(physical), axis=1)
    for j, bounds in enumerate((settings.vx_bounds_m_s, settings.vy_bounds_m_s, settings.vz_bounds_m_s)):
        valid &= (physical[:, j] >= max(1e-6, bounds[0]) if j == 0 else physical[:, j] >= bounds[0]) & (physical[:, j] <= bounds[1])
    valid &= np.sum(physical[:, 3:]**2, axis=1) <= settings.transverse_norm_max**2
    logtarget = np.full(count, -np.inf)
    event = np.zeros(count, np.int32)
    yzt = np.full((count, 3), np.nan)
    observations = np.full((count, 6), np.nan)
    indices = np.flatnonzero(valid)
    if len(indices):
        obs, ev, pt = propagate(physical[indices], post.case, post.environment, post.cd, step)
        observations[indices] = obs; event[indices] = ev; yzt[indices] = pt
        compatible = np.all(np.isfinite(obs), axis=1)
        eligible = indices[compatible]
        error = obs[compatible]-np.asarray(post.case.positions_m).ravel()
        white = np.linalg.solve(post.cholesky, error.T).T
        sy = settings.vertical_force_prior_std or settings.transverse_prior_std
        sz = settings.side_force_prior_std or settings.transverse_prior_std
        penalty = .5*np.sum((physical[eligible, 3:]/[sy, sz])**2, axis=1)
        logtarget[eligible] = -.5*np.sum(white**2, axis=1)-penalty
        event[indices[~compatible]] = 0
    weights = normalize_log_weights(logtarget-logq)
    return dict(parameters=physical, log_target=logtarget, log_proposal=logq,
                weight=weights, event=event, goal_yz_m=yzt[:, :2], arrival_time_s=yzt[:, 2],
                observation_positions_m=observations)


def conditional(data, mask, case):
    w = data["weight"][mask]; total = float(w.sum())
    if total <= 0:
        return None
    w = w/total; p = data["goal_yz_m"][mask]; times = data["arrival_time_s"][mask]
    mean = w@p; centered = p-mean
    cov = (centered*w[:, None]).T@centered
    ess = effective_sample_size(w)
    region, area = None, None
    if ess >= UNCERTAINTY.min_conditional_effective_samples and len(p) >= 3:
        try:
            factor = np.linalg.cholesky(cov)
            r2 = np.sum(np.linalg.solve(factor, centered.T)**2, axis=0)
            threshold = weighted_quantile(r2, w, .9)
            area = float(np.pi*threshold*np.sqrt(np.linalg.det(cov)))
            if area > 0:
                region = dict(center_yz_m=mean.tolist(), covariance_m2=cov.tolist(), threshold=threshold)
        except np.linalg.LinAlgError:
            pass
    return dict(posterior_mass=total, effective_samples=ess, mean_yz_m=mean.tolist(),
                covariance_m2=cov.tolist(), region=region, region_area_m2=area,
                most_likely_cell=grid_mode(p, w, case),
                arrival_time_median_s=weighted_quantile(times,w,.5),
                arrival_time_90_interval_s=[weighted_quantile(times,w,.05),weighted_quantile(times,w,.95)])


def summarize(data, case, prefix=None):
    if prefix is not None:
        data = {k:v[:prefix] for k,v in data.items()}
    w = data["weight"]; w = w/w.sum()
    data = dict(data, weight=w)
    probabilities = {name:float(w[data["event"] == code].sum()) for code,name in EVENTS.items() if code != 0}
    total = sum(probabilities.values())
    probabilities = {k:v/total for k,v in probabilities.items()}
    positive = w > 0
    return dict(effective_samples=effective_sample_size(w), event_probabilities=probabilities,
                unknown_event_mass=sum(probabilities[k] for k in ("boundary","unresolved","failure")),
                direct_crossing=conditional(data, positive & np.isin(data["event"],[1,2]),case),
                inside_conditioned=conditional(data, positive & (data["event"] == 1),case))


def diagnose(data, case):
    first, full = summarize(data, case, 32000), summarize(data, case)
    conv = _convergence(first, full, UNCERTAINTY)
    valid = certify_outputs(first, full, UNCERTAINTY)
    passed = (full["effective_samples"] >= UNCERTAINTY.min_effective_samples and
              conv["sampling_stable"] and full["unknown_event_mass"] <= UNCERTAINTY.unknown_mass_tolerance)
    return dict(summary=full, first_half=first, diagnostics=conv, output_validity=valid, success=bool(passed))
