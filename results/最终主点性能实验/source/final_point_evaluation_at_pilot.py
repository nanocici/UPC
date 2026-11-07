"""Frozen F1: per-shot mode stability first, sealed predictions then truth scoring.

Run: python code/final_point_evaluation.py all --workers 4
Prepare/run never load truth. Resume uses completed per-case files without resampling.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

import batch_f_sampler as sampler
from config import PROJECT_ROOT, Environment, SOLVER
from data import load_observations, observation_case
from final_config import FINAL
from physics import predict_baseline
from uncertainty import UNCERTAINTY, grid_mode, weighted_quantile

OUT=PROJECT_ROOT/'results'/'最终主点性能实验'
DATA=PROJECT_ROOT/'data'/'independent_200'
CONDITIONS=('direct_crossing','inside_conditioned')
BW=(.05,.10,.20)
TOL=.20
RES=.01
SEED_OFFSETS=(0,101,202,303)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def jsonl(path,rows):
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def prepare(output):
    if output.exists():raise FileExistsError(output)
    rows=load_observations(DATA/'observations'/'test.jsonl')
    assert len(rows)==400 and len({r['source_id'] for r in rows})==100
    output.mkdir(parents=True);(output/'cases').mkdir();(output/'source').mkdir()
    jsonl(output/'observations.jsonl',rows)
    hashes={}
    for name in ('final_point_evaluation.py','batch_f_sampler.py','batch_f_dynamics.c','config.py',
                 'final_config.py','physics.py','inverse.py','uncertainty.py','data.py'):
        p=Path(__file__).with_name(name);shutil.copy2(p,output/'source'/name);hashes[name]=digest(p)
    save(output/'run_config.json',dict(date='2026-11-08 Asia/Shanghai',model=asdict(FINAL),
         inference=asdict(FINAL.inference_settings()),solver=asdict(SOLVER),uncertainty=asdict(UNCERTAINTY),
         observation_sha256=digest(output/'observations.jsonl'),source_sha256=hashes,
         proposals_per_seed=64000,seed_offsets=SEED_OFFSETS,comparison_prefix=32000,
         bandwidths_m=BW,main_bandwidth_m=.1,peak_tolerance_m=TOL,resolution_m=RES,
         batch_step_s=.005,search_window='inside: goal bounds plus 0.60m; direct: predicted mean +/-6std plus 0.60m',
         scope='Existing test data, frozen-method supplementary evaluation; not a new blind test.'))


def selected(data,condition,count=64000):
    e=data['event'][:count];w=data['weight'][:count]
    mask=(w>0)&(np.isin(e,[1,2]) if condition=='direct_crossing' else e==1)
    if not np.any(mask):return None
    return data['goal_yz_m'][:count][mask],w[mask]/w[mask].sum()


def window(samples,summaries,condition,case):
    if condition=='inside_conditioned':
        return ((case.goal_y_bounds_m[0]-.6,case.goal_y_bounds_m[1]+.6),
                (case.goal_z_bounds_m[0]-.6,case.goal_z_bounds_m[1]+.6))
    available=[s['summary'][condition] for s in summaries if s['summary'][condition] is not None]
    if not available:return None
    lo=np.min([np.array(s['mean_yz_m'])-6*np.sqrt(np.diag(s['covariance_m2'])) for s in available],axis=0)-.6
    hi=np.max([np.array(s['mean_yz_m'])+6*np.sqrt(np.diag(s['covariance_m2'])) for s in available],axis=0)+.6
    return tuple((float(np.floor(l/RES)*RES),float(np.ceil(h/RES)*RES)) for l,h in zip(lo,hi))


def density_peak(points,weights,bounds,bw):
    y=np.arange(bounds[0][0],bounds[0][1]+RES/2,RES)
    z=np.arange(bounds[1][0],bounds[1][1]+RES/2,RES)
    if (len(y)-1)*(len(z)-1)>4_000_000:raise ValueError('Search window exceeds fixed 4M-cell resource limit')
    hist,_,_=np.histogram2d(points[:,0],points[:,1],bins=(y,z),weights=weights)
    density=gaussian_filter(hist,sigma=bw/RES,mode='constant',truncate=4)/RES**2
    peak=peak_from_density(density,bounds)
    return peak,density


def peak_from_density(d,bounds):
    iy,iz=np.unravel_index(np.argmax(d),d.shape)
    return dict(point_yz_m=[float(bounds[0][0]+(iy+.5)*RES),float(bounds[1][0]+(iz+.5)*RES)],
                boundary=bool(iy<10 or iz<10 or iy>=d.shape[0]-10 or iz>=d.shape[1]-10),
                window_mass=float(d.sum()*RES**2))


def diameter(points):
    p=np.array(points);return float(np.linalg.norm(p[:,None]-p[None,:],axis=-1).max())


def mode_analysis(samples,summaries,condition,case):
    bounds=window(samples,summaries,condition,case)
    records=[];combined={};fullpoints=[];shifts=[];reasons=[];gridpoints=[];grid_diameters=[]
    if bounds is None:return dict(stable=False,reasons=['no_condition_candidates'],raw_peak=None)
    for i,(data,summary) in enumerate(zip(samples,summaries)):
        bykey={}
        cond=summary['summary'][condition]
        if not summary['success']:reasons.append(f'seed_{i}:sampling_diagnostics')
        if cond is None or cond['effective_samples']<UNCERTAINTY.min_conditional_effective_samples:
            reasons.append(f'seed_{i}:conditional_ess')
        half=summary['first_half'][condition]
        if half is None or half['effective_samples']<UNCERTAINTY.min_conditional_effective_samples:
            reasons.append(f'seed_{i}:half_conditional_ess')
        for count,bw in ((32000,.1),(64000,.05),(64000,.1),(64000,.2)):
            part=selected(data,condition,count)
            if part is None:continue
            peak,d=density_peak(*part,bounds,bw)
            record=dict(seed_index=i,proposals=count,bandwidth_m=bw,**peak)
            records.append(record);bykey[(count,bw)]=peak
            if count==64000:combined[bw]=combined.get(bw,0)+d/4
        if (64000,.1) in bykey and (32000,.1) in bykey:
            a,b=bykey[(64000,.1)],bykey[(32000,.1)]
            fullpoints.append(a['point_yz_m']);shifts.append(float(np.linalg.norm(np.array(a['point_yz_m'])-b['point_yz_m'])))
            if a['boundary'] or b['boundary']:reasons.append(f'seed_{i}:boundary_peak')
        else:reasons.append(f'seed_{i}:missing_peak')
        part=selected(data,condition)
        if part is not None:
            pts=[]
            width=2.44/24
            for factor in (.5,1,2):
                for sy in (0,width/2):
                    for sz in (0,width/2):
                        pts.append(grid_mode(*part,case,width*factor,shift=np.array([sy,sz]))['center_yz_m'])
            grid_diameters.append(diameter(pts))
            gridpoints.append(grid_mode(*part,case,width)['center_yz_m'])
    if len(fullpoints)!=4:reasons.append('incomplete_seed_peaks')
    cross=diameter(fullpoints) if fullpoints else None
    last=max(shifts) if shifts else None
    if cross is not None and cross>TOL:reasons.append('cross_seed_shift')
    if last is not None and last>TOL:reasons.append('sample_doubling_shift')
    peaks={str(bw):peak_from_density(d,bounds) for bw,d in combined.items()}
    main=peaks.get('0.1')
    if main is None or main['boundary']:reasons.append('combined_missing_or_boundary')
    competing=None
    if .1 in combined:
        d=combined[.1];m=maximum_filter(d,size=21,mode='nearest')
        competing=int(np.sum((m==d)&(d>=.8*d.max())))
    return dict(stable=not reasons,reasons=reasons,raw_peak=main['point_yz_m'] if main else None,
                point_yz_m=main['point_yz_m'] if main and not reasons else None,
                cross_seed_diameter_m=cross,last_doubling_shift_m=last,
                search_bounds=bounds,combined_peaks=peaks,per_seed_peaks=records,
                competing_peak_count=competing,
                raw_grid=dict(cross_seed_m=diameter(gridpoints) if gridpoints else None,
                              within_seed_max_m=max(grid_diameters) if grid_diameters else None,
                              pass_=len(gridpoints)==4 and max(grid_diameters)<=TOL and diameter(gridpoints)<=TOL))


def pooled_region(samples,summaries,case):
    if not all(s['output_validity']['prediction_region'] for s in summaries):return None
    parts=[selected(a,'direct_crossing') for a in samples]
    if any(p is None for p in parts):return None
    p=np.vstack([a[0] for a in parts]);w=np.concatenate([a[1]/4 for a in parts])
    mean=w@p;c=p-mean;cov=(c*w[:,None]).T@c
    try:
        chol=np.linalg.cholesky(cov);r2=np.sum(np.linalg.solve(chol,c.T)**2,axis=0)
    except np.linalg.LinAlgError:return None
    q=weighted_quantile(r2,w,.9)
    return dict(center_yz_m=mean.tolist(),covariance_m2=cov.tolist(),internal_threshold=q,
                threshold=FINAL.spatial_threshold,area_m2=float(np.pi*FINAL.spatial_threshold*np.sqrt(np.linalg.det(cov))))


def worker(row,output):
    path=Path(output)/'cases'/row['case_id'];path.mkdir(exist_ok=True)
    checkpoint=path/'prediction.json'
    if checkpoint.exists():return json.loads(checkpoint.read_text())
    start=time.perf_counter()
    prediction=dict(case_id=row['case_id'],source_id=row['source_id'],sigma_m=row['measurement']['sigma_m'],
                    observation_available=row['observation_available'],status='unavailable',conditions={},region=None,g_point=None)
    if not row['observation_available']:
        save(checkpoint,prediction);return prediction
    case=observation_case(row);env=Environment(**dict(row['known_environment'],wind_m_s=tuple(row['known_environment']['wind_m_s'])))
    try:
        g=predict_baseline('G',case=case,environment=env)
        prediction['g_event']=g['event']
        if g['event']=='goal':prediction['g_point']=list(g['event_position_m'][1:])
        fit,post,dist,prob=sampler.prepare(case,env,FINAL.inference_settings(),row['measurement']['sigma_m'],row['measurement']['frame_correlation'])
        save(path/'fit.json',fit)
        if post is None:
            prediction['status']='fit_failure';save(checkpoint,prediction);return prediction
        base=int(hashlib.sha256(row['case_id'].encode()).hexdigest()[:8],16)
        seeds=[base+x for x in SEED_OFFSETS];samples=[];summaries=[]
        for seed in seeds:
            raw=path/f'candidates_{seed}.npz';sp=path/f'summary_{seed}.json'
            if raw.exists() and sp.exists():
                with np.load(raw) as f:data={k:f[k] for k in f.files}
                summary=json.loads(sp.read_text())
            else:
                data=sampler.sample(post,dist,prob,seed)
                np.savez_compressed(raw,**data)
                summary=sampler.diagnose(data,case);save(sp,summary)
            samples.append(data);summaries.append(summary)
        prediction.update(status='computed',seeds=seeds,proposal_count=256000,
                          seed_diagnostics=[dict(success=s['success'],effective_samples=s['summary']['effective_samples'],
                                                 unknown_mass=s['summary']['unknown_event_mass']) for s in summaries])
        for condition in CONDITIONS:
            prediction['conditions'][condition]=mode_analysis(samples,summaries,condition,case)
            cs=[s['summary'][condition] for s in summaries]
            prediction['conditions'][condition]['mean_auxiliary']=(np.mean([c['mean_yz_m'] for c in cs],axis=0).tolist() if all(c is not None for c in cs) else None)
        prediction['region']=pooled_region(samples,summaries,case)
        prediction['inside_probability']=float(np.mean([s['summary']['event_probabilities']['goal_inside'] for s in summaries]))
    except Exception as e:
        prediction.update(status='failure',exception=f'{type(e).__name__}: {e}')
    prediction['wall_s']=time.perf_counter()-start
    save(checkpoint,prediction)
    return prediction


def run(output,workers=4,limit=None):
    config=json.loads((output/'run_config.json').read_text())
    for name,sha in config['source_sha256'].items():
        if digest(Path(__file__).with_name(name))!=sha:raise RuntimeError(f'Source changed: {name}')
    rows=load_observations(output/'observations.jsonl')
    tasks=rows[:limit] if limit else rows
    start=time.perf_counter();done=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(worker,row,str(output)) for row in tasks]
        for f in as_completed(fs):
            p=f.result();done.append(p)
            if len(done)%5==0 or len(done)==len(tasks):
                print(f'{len(done)}/{len(tasks)} cases complete; elapsed {time.perf_counter()-start:.1f}s; last={p["status"]}',flush=True)
    if len(tasks)==len(rows):
        done.sort(key=lambda x:x['case_id']);jsonl(output/'predictions.jsonl',done)
        seal=dict(predictions_sha256=digest(output/'predictions.jsonl'),observations_sha256=digest(output/'observations.jsonl'),
                  count=len(done),sealed_before_truth_scoring=True)
        save(output/'prediction_manifest.json',seal)
    save(output/('pilot_run.json' if limit else 'execution.json'),dict(cases=len(tasks),workers=workers,wall_s=time.perf_counter()-start,status_counts=dict(Counter(p['status'] for p in done))))


def score(output):
    # Implemented below; called only after complete predictions are sealed.
    raise NotImplementedError


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','run','score','all'))
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage in ('prepare','all') and not a.output.exists():prepare(a.output)
    if a.stage in ('run','all'):run(a.output,a.workers,a.limit)
    if a.stage in ('score','all') and not a.limit:score(a.output)
