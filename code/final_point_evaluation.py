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
    seal=json.loads((output/'prediction_manifest.json').read_text())
    assert digest(output/'predictions.jsonl')==seal['predictions_sha256']
    assert digest(output/'observations.jsonl')==seal['observations_sha256']
    predictions=lines(output/'predictions.jsonl')
    if len(predictions)!=400:raise ValueError('Scoring requires all 400 sealed predictions')
    truths={r['source_id']:r for r in lines(DATA/'truth'/'test.jsonl')}
    rows=[]
    for p in predictions:
        t=truths[p['source_id']];truth=t['truth'];direct=truth['event']=='goal'
        inside=direct and truth['inside_goal_opening'];target=truth['event_position_m'][1:] if direct else None
        ci=p['conditions'].get('inside_conditioned',{});cd=p['conditions'].get('direct_crossing',{})
        peak=ci.get('point_yz_m');g=p.get('g_point');mean=ci.get('mean_auxiliary')
        def error(point):return float(np.linalg.norm(np.array(point)-target)) if direct and point is not None else None
        region=p.get('region');covered=False;area=None
        if direct and region:
            delta=np.array(target)-region['center_yz_m'];cov=np.array(region['covariance_m2'])
            covered=bool(delta@np.linalg.solve(cov,delta)<=region['threshold']);area=region['area_m2']
        rows.append(dict(case_id=p['case_id'],source_id=p['source_id'],sigma_m=p['sigma_m'],
             mechanism=t['parameters']['mechanism'],observation_available=p['observation_available'],
             status=p['status'],truth_direct=direct,truth_inside=bool(inside),
             stable_inside=bool(ci.get('stable',False)),stable_direct=bool(cd.get('stable',False)),
             failure_reason=';'.join(ci.get('reasons',[])) or (p['status'] if not peak else ''),
             peak_error_m=error(peak) if inside else None,g_error_m=error(g) if inside else None,
             mean_error_m=error(mean) if inside else None,
             raw_peak_error_m=error(ci.get('raw_peak')) if inside else None,
             direct_peak_error_m=error(cd.get('point_yz_m')) if direct else None,
             direct_mean_error_m=error(cd.get('mean_auxiliary')) if direct else None,
             region_available=region is not None,region_covered=covered,region_area_m2=area,
             cross_seed_m=ci.get('cross_seed_diameter_m'),last_doubling_m=ci.get('last_doubling_shift_m'),
             inside_probability=p.get('inside_probability')))
    with (output/'per_case.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    overall=metrics(rows);primary=metrics([r for r in rows if r['sigma_m']==.01])
    noise={str(s):metrics([r for r in rows if r['sigma_m']==s]) for s in sorted({r['sigma_m'] for r in rows})}
    mechanism={s:metrics([r for r in rows if r['mechanism']==s]) for s in sorted({r['mechanism'] for r in rows})}
    boot=bootstrap(rows);primary_boot=bootstrap([r for r in rows if r['sigma_m']==.01])
    result=dict(overall=overall,primary_sigma_001=primary,by_noise=noise,by_mechanism=mechanism,
                source_bootstrap=boot,primary_source_bootstrap=primary_boot,
                status_counts=dict(Counter(r['status'] for r in rows)),
                truth_sha256=digest(DATA/'truth'/'test.jsonl'),prediction_sha256=seal['predictions_sha256'],
                limitations=['Existing test data previously inspected; supplementary frozen-method evaluation.',
                             'Synthetic H settings not calibrated to a real football.',
                             'Coverage is empirical, not an unconditional conformal guarantee.',
                             'Point metrics condition on true inside goal; region metrics condition on direct crossing.'])
    save(output/'summary.json',result)
    plot_results(rows,result,output)
    write_report(result,output)
    print(json.dumps(dict(overall=overall,primary=primary,paired_ci=boot),ensure_ascii=False,indent=2),flush=True)


def errors(rows,key):
    a=[r[key] for r in rows if r[key] is not None]
    return dict(n=len(a),median_m=float(np.median(a)) if a else None,
                p90_m=float(np.quantile(a,.9)) if a else None,
                rmse_m=float(np.sqrt(np.mean(np.array(a)**2))) if a else None)


def metrics(rows):
    ins=[r for r in rows if r['truth_inside']];direct=[r for r in rows if r['truth_direct']]
    paired=[r for r in ins if r['peak_error_m'] is not None and r['g_error_m'] is not None]
    areas=[r['region_area_m2'] for r in direct if r['region_area_m2'] is not None]
    return dict(observations=len(rows),sources=len({r['source_id'] for r in rows}),
                available=sum(r['observation_available'] for r in rows),
                true_inside_observations=len(ins),true_inside_sources=len({r['source_id'] for r in ins}),
                true_direct_observations=len(direct),true_direct_sources=len({r['source_id'] for r in direct}),
                stable_inside_count=sum(r['stable_inside'] for r in ins),
                stable_inside_rate=float(np.mean([r['stable_inside'] for r in ins])) if ins else None,
                stable_inside_missing=sum(not r['stable_inside'] for r in ins),
                peak=errors(ins,'peak_error_m'),g_all_eligible=errors(ins,'g_error_m'),mean_auxiliary=errors(ins,'mean_error_m'),
                paired_peak=errors(paired,'peak_error_m'),paired_g=errors(paired,'g_error_m'),
                raw_peak_exploratory=errors(ins,'raw_peak_error_m'),
                stable_and_error_below_05_rate=sum(r['peak_error_m'] is not None and r['peak_error_m']<=.5 for r in ins)/len(ins) if ins else None,
                region_coverage=float(np.mean([r['region_covered'] for r in direct])) if direct else None,
                region_provided=sum(r['region_available'] for r in direct),
                region_median_area_m2=float(np.median(areas)) if areas else None,
                direct_peak=errors(direct,'direct_peak_error_m'),direct_mean=errors(direct,'direct_mean_error_m'),
                failure_reasons=dict(Counter(r['failure_reason'] for r in ins if not r['stable_inside'])))


def bootstrap(rows,n=2000):
    # Resample independent sources within mechanisms; retain every noise replicate.
    rng=np.random.default_rng(20261108);by_source=defaultdict(list)
    for r in rows:by_source[r['source_id']].append(r)
    strata=defaultdict(list)
    for sid,rs in by_source.items():strata[rs[0]['mechanism']].append(sid)
    shifts=[];coverage=[];release=[]
    for _ in range(n):
        selected=[r for ids in strata.values() for sid in rng.choice(ids,size=len(ids),replace=True) for r in by_source[sid]]
        pairs=[r for r in selected if r['truth_inside'] and r['peak_error_m'] is not None and r['g_error_m'] is not None]
        if pairs:shifts.append(float(np.median([r['peak_error_m'] for r in pairs])-np.median([r['g_error_m'] for r in pairs])))
        direct=[r for r in selected if r['truth_direct']];inside=[r for r in selected if r['truth_inside']]
        if direct:coverage.append(float(np.mean([r['region_covered'] for r in direct])))
        if inside:release.append(float(np.mean([r['stable_inside'] for r in inside])))
    def interval(a):return np.quantile(a,[.025,.975]).tolist() if a else None
    return dict(replicates=n,paired_median_difference_f1_minus_g_ci95_m=interval(shifts),
                region_coverage_ci95=interval(coverage),stable_inside_rate_ci95=interval(release))


def plot_results(rows,result,output):
    paired=[r for r in rows if r['truth_inside'] and r['peak_error_m'] is not None and r['g_error_m'] is not None]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for key,label in [('peak_error_m','F1 stable smoothed peak'),('g_error_m','G, same cases')]:
        vals=np.sort([r[key] for r in paired])
        if len(vals):axes[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label)
    axes[0].set(xlabel='2D point error (m)',ylabel='Empirical CDF',title='True-inside, paired available cases')
    if paired:
        axes[1].scatter([r['g_error_m'] for r in paired],[r['peak_error_m'] for r in paired],s=12,alpha=.5)
        high=max(max(r['g_error_m'],r['peak_error_m']) for r in paired);axes[1].plot([0,high],[0,high],'k--')
    axes[1].set(xlabel='G error (m)',ylabel='F1 peak error (m)',title='Below diagonal: F1 smaller error')
    axes[0].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(output/'point_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    sigmas=list(result['by_noise']);x=np.arange(len(sigmas))
    axes[0].plot(x,[result['by_noise'][s]['region_coverage'] for s in sigmas],'o-',label='Direct-crossing region coverage')
    axes[0].plot(x,[result['by_noise'][s]['stable_inside_rate'] for s in sigmas],'s-',label='True-inside stable output rate')
    axes[0].axhline(.9,color='gray',ls='--',label='Nominal region mass 0.90')
    axes[0].set_xticks(x,sigmas);axes[0].set(xlabel='Observation sigma (m)',ylabel='Fraction',ylim=(0,1.05))
    names=list(result['by_mechanism']);axes[1].bar(np.arange(len(names)),[result['by_mechanism'][n]['stable_inside_rate'] or 0 for n in names])
    axes[1].set_xticks(np.arange(len(names)),names,rotation=25,ha='right');axes[1].set(ylabel='Stable output rate on true-inside shots',ylim=(0,1.05))
    axes[0].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2,axis='y')
    fig.savefig(output/'coverage_and_stability.png',dpi=180);plt.close(fig)


def write_report(result,output):
    o=result['overall'];main=result['primary_sigma_001'];ci=result['source_bootstrap']['paired_median_difference_f1_minus_g_ci95_m']
    def f(v):return '缺失' if v is None else f'{v:.4f}'
    rows=['# 冻结 F1：批量主点稳定性与最终主点性能实验','',
          '日期：2026-11-08（Asia/Shanghai）。模型、先验、Cd、带宽、区域阈值未修改。',
          '使用已有independent_200测试部分100源、400观测；这是已使用测试数据上的冻结方法补充评价，不是新盲测。',
          '每条可用观测四种子各64k提议；先保存候选和稳定性状态，全部预测封存后才由评分程序读真值。','',
          '## 数值实现与检查','',
          'MAP与混合t提议沿用inverse/uncertainty；同一动力学采用经原积分器对照及步长减半验收的C批量RK4（步长0.005s、Hermite事件定位）。不把加速实现当成新物理模型。',
          '原始候选以每种子NPZ保存，包含物理参数、log_target、log_proposal、权重、事件、过线位置/时间及观测窗位置；拒绝候选保留。','',
          '## 稳定性口径','',
          '主点为入框条件下0.10m固定带宽平滑峰，网格0.01m；四种子分别条件归一化后等权合并。32k与64k是同一提议分布前缀。',
          '四种子均须通过原采样诊断，条件ESS在32k和64k均≥100；主带宽跨种子直径及最大加倍移动均≤0.20m，且峰不在搜索边缘。',
          '整门框加0.60m扩边用于入框条件搜索；直接过线窗口由预测均值±6标准差再扩0.60m确定，不读真值。',
          '另保存0.05/0.20m峰与原始格网诊断，原始格网未通过不冒充平滑峰未通过。未发布的原始峰只作探索性记录，不混入正式点误差。','',
          '## 样本与输出','',
          f"- 总源数{o['sources']}、观测数{o['observations']}、完整可用观测{o['available']}。",
          f"- 真实入框：{o['true_inside_sources']}源/{o['true_inside_observations']}观测；稳定主点{o['stable_inside_count']}，未通过{o['stable_inside_missing']}，比例{f(o['stable_inside_rate'])}。",
          f"- 真实直接过线：{o['true_direct_sources']}源/{o['true_direct_observations']}观测；区域提供{o['region_provided']}。",'',
          '## 入框条件点误差','',
          '| 方法与样本口径 | n | 中位误差/m | P90/m | RMSE/m |','|---|---:|---:|---:|---:|']
    for label,key in [('F1稳定平滑峰','peak'),('G，所有可用入框案例','g_all_eligible'),('F1，共同成功子集','paired_peak'),('G，共同成功子集','paired_g'),('F1入框条件均值，辅助','mean_auxiliary')]:
        a=o[key];rows.append(f"| {label} | {a['n']} | {f(a['median_m'])} | {f(a['p90_m'])} | {f(a['rmse_m'])} |")
    rows += ['',f'共同成功子集的中位误差差F1−G，按机制分层、按源整组配对bootstrap 2000次，95%区间：{ci}m。负值支持F1误差较小；区间跨零则不支持明确优势。',
             f"全体真实入框观测中，同时稳定且误差≤0.5m的比例：{f(o['stable_and_error_below_05_rate'])}；失败保留在该分母。",'',
             '## 空间区域','',
             f"冻结阈值{FINAL.spatial_threshold:.6f}用于新四种子合并的直接过线后验协方差；缺失区域计漏覆盖。实际覆盖{f(o['region_coverage'])}，95%源级区间{result['source_bootstrap']['region_coverage_ci95']}，面积中位{f(o['region_median_area_m2'])}m²。",
             '该区域是经验校准区域，不宣称严格conformal保证。其样本条件与入框峰误差不同。','',
             '## 主要噪声层：sigma=0.01m','',
             f"真实入框{main['true_inside_observations']}观测，稳定比例{f(main['stable_inside_rate'])}；共同成功子集F1中位误差{f(main['paired_peak']['median_m'])}m，G为{f(main['paired_g']['median_m'])}m；区域覆盖{f(main['region_coverage'])}，面积中位{f(main['region_median_area_m2'])}m²。",'',
             '## 噪声分层','',
             '| sigma/m | 入框n | 稳定输出率 | 配对F1中位误差/m | 配对G中位误差/m | 过线区域覆盖 | 面积/m² |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label,a in result['by_noise'].items():
        rows.append(f"| {label} | {a['true_inside_observations']} | {f(a['stable_inside_rate'])} | {f(a['paired_peak']['median_m'])} | {f(a['paired_g']['median_m'])} | {f(a['region_coverage'])} | {f(a['region_median_area_m2'])} |")
    rows += ['','## 机制分层','','| 机制 | 入框n | 稳定输出率 | 配对F1中位误差/m | 配对G中位误差/m | 过线区域覆盖 |','|---|---:|---:|---:|---:|---:|']
    for label,a in result['by_mechanism'].items():
        rows.append(f"| {label} | {a['true_inside_observations']} | {f(a['stable_inside_rate'])} | {f(a['paired_peak']['median_m'])} | {f(a['paired_g']['median_m'])} | {f(a['region_coverage'])} |")
    rows += ['','## 未通过原因','','```json',json.dumps(o['failure_reasons'],ensure_ascii=False,indent=2),'```','',
             '## 结论边界','',
             ('本批数据的配对区间支持F1平滑峰相对G的中位误差改善。' if ci and ci[1]<0 else
              '本批数据未给出F1平滑峰相对G中位误差明确改善的证据，应保留简单基线并如实报告差异。'),
             '稳定输出比例与条件误差必须同时报告，不能只展示通过案例的误差。不能将数值稳定性解释为真实精度、唯一众数或全局最优。',
             'H的气动机制为合成压力设置，未匹配具体实测球型；没有实测外部验证。四档副本不是独立源，bootstrap始终整源重采样。','',
             '![点误差比较](point_errors.png)','','![覆盖与稳定性](coverage_and_stability.png)','',
             '## 复现','','```bash','.venv/bin/python code/final_point_evaluation.py run --workers 4',
             '.venv/bin/python code/final_point_evaluation.py score','```',
             'run可复用已完成案例及原始候选，不重复采样。source/与run_config.json固定实现和输入；prediction_manifest.json记录评分前封存的完整预测。']
    (output/'FINAL_POINT_EVALUATION_REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','run','score','all'))
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage in ('prepare','all') and not a.output.exists():prepare(a.output)
    if a.stage in ('run','all'):run(a.output,a.workers,a.limit)
    if a.stage in ('score','all') and not a.limit:score(a.output)
