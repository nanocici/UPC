"""Read-only development diagnosis and new controlled development experiments.
No calibration/test paths are accepted. Predictor settings remain unchanged.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from dataclasses import asdict, replace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from config import CASE, ENVIRONMENT, INFERENCE, PROJECT_ROOT
from data import TruthParameters, generate_truth, observation_case
from inverse import fit_case
from physics import simulate
from development_regions import (write_json, file_hash, predict_task,
                                 validate_development)

SEED = 20261117
SCENARIOS = ('baseline_S', 'vertical_force', 'drag_offset', 'unknown_wind', 'variable_drag')

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')

def region_diagnostics(region, truth):
    v = np.asarray(region['covariance_m2'], float)
    q = float(region['threshold'])
    eigen, vectors = np.linalg.eigh(v)
    if np.min(eigen) <= 0 or q <= 0:
        raise ValueError('invalid ellipse')
    delta = np.asarray(truth)-np.asarray(region['center_yz_m'])
    principal = vectors.T@delta/np.sqrt(eigen*q)
    marginal = delta/np.sqrt(np.diag(v)*q)
    return dict(covered=bool(np.dot(principal, principal) <= 1),
        normalized_radius=float(np.linalg.norm(principal)),
        minor_semiaxis_m=float(np.sqrt(eigen[0]*q)), major_semiaxis_m=float(np.sqrt(eigen[1]*q)),
        major_angle_deg=float(np.degrees(np.arctan2(vectors[1,1], vectors[0,1]))%180),
        miss_minor_component=bool(abs(principal[0]) > 1),
        outside_y_projection=bool(abs(marginal[0]) > 1),
        outside_z_projection=bool(abs(marginal[1]) > 1),
        area_m2=float(np.pi*q*np.sqrt(np.linalg.det(v))))

def diagnostic_rows(predictions, observations, truths):
    validate_development(observations)
    if any(t['split'] != 'development' for t in truths):
        raise ValueError('development truth only')
    obs = {o['case_id']:o for o in observations}
    truth = {t['source_id']:t for t in truths}
    rows=[]
    for p in predictions:
        o=obs[p['case_id']]; t=truth[o['source_id']]; tr=t['truth']
        row=dict(case_id=o['case_id'], source_id=o['source_id'], method=p['method'],
            scenario=t.get('scenario',t['parameters']['mechanism']), sigma=o['measurement']['sigma_m'],
            status=p['status'], truth_event=tr['event'], covered=False,
            true_time=tr['event_time_s'], point_kind=p['point_kind'])
        if tr['event']=='goal' and p.get('point_yz_m') is not None:
            d=np.asarray(p['point_yz_m'])-np.asarray(tr['event_position_m'])[1:]
            row.update(error_y_m=float(d[0]),error_z_m=float(d[1]),error_m=float(np.linalg.norm(d)))
        if tr['event']=='goal' and p.get('region') is not None:
            row.update(region_diagnostics(p['region'],tr['event_position_m'][1:]))
        rows.append(row)
    return rows

def aggregate(rows, seed=SEED, bootstrap=1000):
    def metrics(rs):
        direct=[r for r in rs if r['truth_event']=='goal']
        valid=[r for r in direct if 'normalized_radius' in r]
        points=[r for r in direct if 'error_m' in r]
        def med(key, subset):
            return float(np.median([r[key] for r in subset])) if subset else None
        return dict(n=len(rs), source_count=len({r['source_id'] for r in rs}), direct_n=len(direct),
            coverage=float(np.mean([r['covered'] for r in direct])) if direct else None,
            provided=len(valid), failures=sum(r['status']=='failure' for r in rs),
            unavailable=sum(r['status']=='unavailable' for r in rs),
            bias_y_m=float(np.mean([r['error_y_m'] for r in points])) if points else None,
            bias_z_m=float(np.mean([r['error_z_m'] for r in points])) if points else None,
            median_abs_y_m=med('abs_y', [dict(r,abs_y=abs(r['error_y_m'])) for r in points]),
            median_abs_z_m=med('abs_z', [dict(r,abs_z=abs(r['error_z_m'])) for r in points]),
            median_error_m=med('error_m',points),
            p90_error_m=float(np.quantile([r['error_m'] for r in points],.9)) if points else None,
            median_area_m2=med('area_m2',valid), median_minor_m=med('minor_semiaxis_m',valid),
            median_major_m=med('major_semiaxis_m',valid), median_radius=med('normalized_radius',valid),
            outside_y=sum(r.get('outside_y_projection',False) for r in valid),
            outside_z=sum(r.get('outside_z_projection',False) for r in valid),
            miss_minor=sum(r.get('miss_minor_component',False) for r in valid))
    result={}
    for grouping in ('overall','noise','mechanism','mechanism_noise'):
        groups=defaultdict(list)
        for r in rows:
            key=(r['method'],)+( (r['sigma'],) if grouping=='noise' else
                (r['scenario'],) if grouping=='mechanism' else
                (r['scenario'],r['sigma']) if grouping=='mechanism_noise' else ())
            groups[str(key)].append(r)
        result[grouping]={}
        for key,rs in groups.items():
            m=metrics(rs)
            direct=[r for r in rs if r['truth_event']=='goal']
            if direct and len({r['source_id'] for r in direct})==len(direct):
                n=len(direct); ph=m['coverage']; z=1.959963984540054
                center=(ph+z*z/(2*n))/(1+z*z/n)
                half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
                m['coverage_wilson_95_independent_sources']=[float(center-half),float(center+half)]
            if grouping=='overall' and bootstrap:
                by_source=defaultdict(list)
                for r in rs: by_source[r['source_id']].append(r)
                keys=list(by_source); rng=np.random.default_rng(seed)
                samples=defaultdict(list)
                for _ in range(bootstrap):
                    draw=[r for k in rng.choice(keys,len(keys),replace=True) for r in by_source[k]]
                    for field,value in metrics(draw).items():
                        if field in ('coverage','bias_y_m','bias_z_m','median_error_m') and value is not None:
                            samples[field].append(value)
                m['source_bootstrap_95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
            result[grouping][key]=m
    return result

def diagnose_existing(output):
    output.mkdir(parents=True,exist_ok=False)
    root=PROJECT_ROOT
    op=root/'data/synthetic_demo/observations/development.jsonl'
    tp=root/'data/synthetic_demo/truth/development.jsonl'
    pp=root/'results/development_regions/predictions.jsonl'
    rows=diagnostic_rows(read_jsonl(pp),read_jsonl(op),read_jsonl(tp))
    write_jsonl(output/'per_case.jsonl',rows)
    summary=aggregate(rows)
    # Attach raw fit diagnostics and MAP-to-posterior displacement; never infer LS from MAP.
    details=[]
    for path in sorted((root/'results/development_regions/cases').glob('*/summary.json')):
        rec=json.loads(path.read_text()); fit=rec['result'].get('fit')
        if not fit or not fit.get('success'): continue
        p=rec['prediction']; mp=rec['map_prediction']
        shift=None
        if p.get('point_yz_m') is not None and mp.get('point_yz_m') is not None:
            shift=float(np.linalg.norm(np.asarray(p['point_yz_m'])-mp['point_yz_m']))
        details.append(dict(case_id=rec['case_id'],model=rec['model'],parameters=fit['parameters'],
            diagnostics=fit['diagnostics'],data_chi_square=fit['data_chi_square'],
            prior_penalty=fit['prior_penalty'],map_mean_shift_m=shift))
    write_jsonl(output/'fit_diagnostics.jsonl',details)
    write_json(output/'summary.json',summary)
    write_json(output/'provenance.json',{str(p.relative_to(root)):file_hash(p) for p in (op,tp,pp)})
    ls_existing(output)
    extra_existing(output)
    failure_inventory(root/'results/development_regions/cases',output)
    return summary

def observation(truth, source, sigma, noise):
    case=asdict(CASE)
    case['positions_m']=[None if p is None else (np.asarray(p)+sigma*noise[i]).tolist()
                         for i,p in enumerate(truth['clean_positions_m'])]
    return dict(schema_version=1,source_id=source,case_id=f'{source}_sigma{sigma:g}',split='development',
        observation_available=truth['observation_available'],case=case,known_environment=asdict(ENVIRONMENT),
        wind_known=False,measurement=dict(mode='gaussian',sigma_m=sigma,frame_correlation=0.,rounding_step_m=None))

def draw_disk(rng):
    while True:
        b=rng.normal(0,INFERENCE.transverse_prior_std,2)
        if np.linalg.norm(b)<INFERENCE.transverse_norm_max: return b

def prepare(output, count=30):
    output.mkdir(parents=True,exist_ok=False)
    config=dict(seed=SEED,source_parameter_groups=count,scenarios=SCENARIOS,sigmas=[.005,.02],
        base_samples=2000,starts=3,workers=6,prior_predictive_sources_per_model=count,
        notes='Exploratory controlled development only; 30 matched groups, not 150 independent groups. '
              'Ablation uses engineering ranges. Prior-predictive uses the exact inference prior; '
              'S b_z is the F disk marginal. Ground and unavailable retained. '
              'Noise is shared across scenarios and scaled across sigmas; MC random streams are not paired.',
        settings=asdict(INFERENCE),
        hashes={p.name:file_hash(p) for p in (Path(__file__),PROJECT_ROOT/'code/uncertainty.py',
                PROJECT_ROOT/'code/data.py',PROJECT_ROOT/'code/inverse.py',PROJECT_ROOT/'code/physics.py')})
    write_json(output/'experiment_config.json',config)
    obs=[]; truths=[]; tasks=[]; recovery=[]
    for i in range(count):
        rng=np.random.default_rng(SEED+i)
        velocity=(float(rng.uniform(22,36)),float(rng.uniform(-10,10)),float(rng.uniform(.2,10)))
        by=float(rng.uniform(-.12,.12)); bz=float(rng.uniform(-.25,.25))
        cd=float(rng.uniform(.15,.4)); wind=tuple(rng.uniform([-3,-3,-.5],[3,3,.5]))
        noise=rng.normal(size=(2,3))
        for scenario in SCENARIOS:
            # Each differs from baseline in exactly one generating mechanism.
            p=TruthParameters(initial_velocity_m_s=velocity,mechanism='variable_drag' if scenario=='variable_drag' else 'constant',
                drag_coefficient=cd if scenario=='drag_offset' else .22,
                transverse_vector=(0.,by if scenario=='vertical_force' else 0.,bz))
            env=replace(ENVIRONMENT,wind_m_s=wind) if scenario=='unknown_wind' else ENVIRONMENT
            tr=generate_truth(p,environment=env); source=f'group{i:03d}_{scenario}'
            truths.append(dict(source_id=source,group_id=f'group{i:03d}',split='development',scenario=scenario,
                parameters=asdict(p),true_environment=asdict(env),truth=tr))
            for sigma in (.005,.02):
                o=observation(tr,source,sigma,noise); index=len(obs); obs.append(o)
                for model in ('S','F'): tasks.append((index,model))
            if scenario in ('baseline_S','vertical_force'):
                recovery.append(dict(source_id=source,model='S' if scenario=='baseline_S' else 'F',
                    observation=observation(tr,source,1e-4,np.zeros((2,3))),truth=tr,parameters=asdict(p)))
        for model in ('S','F'):
            prng=np.random.default_rng(SEED+10000+i*2+(model=='F'))
            b=draw_disk(prng)
            velocity_prior=(float(prng.uniform(1e-6,60)),float(prng.uniform(-30,30)),float(prng.uniform(0,30)))
            p=TruthParameters(initial_velocity_m_s=velocity_prior,mechanism='constant',drag_coefficient=.22,
                transverse_vector=(0.,float(b[0]) if model=='F' else 0.,float(b[1])))
            tr=generate_truth(p); source=f'prior_{model}_{i:03d}'
            truths.append(dict(source_id=source,group_id=source,split='development',scenario=f'prior_{model}',
                parameters=asdict(p),true_environment=asdict(ENVIRONMENT),truth=tr))
            # One noisy observation per independent prior draw.
            o=observation(tr,source,.01,prng.normal(size=(2,3))); index=len(obs); obs.append(o)
            tasks.append((index,model))
    write_jsonl(output/'observations.jsonl',obs); write_jsonl(output/'truth_development.jsonl',truths)
    write_json(output/'tasks.json',tasks); write_jsonl(output/'recovery_inputs.jsonl',recovery)

def recovery_check(output):
    rows=[]
    for item in read_jsonl(output/'recovery_inputs.jsonl'):
        o=item['observation']; tr=item['truth']; model=item['model']
        row=dict(source_id=item['source_id'],model=model,available=o['observation_available'])
        if o['observation_available']:
            case=observation_case(o)
            # Exact observations: sigma only rescales pure LS, it is not added sensor noise.
            fit=fit_case(model,case,mode='ls',sigma_m=.01,starts=3)
            row['ls_objective_scale_m']=.01
            row.update(fit_success=fit['success'],rms_m=fit.get('rms_residual_m'))
            truep=item['parameters']
            params=dict(initial_velocity_m_s=truep['initial_velocity_m_s'],drag_coefficient=.22,
                b_y=truep['transverse_vector'][1],b_z=truep['transverse_vector'][2])
            forward=simulate(model,params,case.times_s,case)
            row['forward_position_max_difference_m']=float(np.max(np.abs(np.asarray(forward['positions_m'])-np.asarray(tr['clean_positions_m']))))
            row['truth_event']=tr['event']; row['fit_event']=fit['simulation']['event'] if fit['success'] else None
            if fit['success'] and tr['event']=='goal' and fit['simulation']['event']=='goal':
                row['goal_error_m']=float(np.linalg.norm(np.asarray(fit['simulation']['event_position_m'])[1:]-np.asarray(tr['event_position_m'])[1:]))
        rows.append(row)
    write_jsonl(output/'recovery_results.jsonl',rows)

def run(output,workers=6):
    obs=read_jsonl(output/'observations.jsonl'); tasks=json.loads((output/'tasks.json').read_text())
    validate_development(obs)
    begun=time.perf_counter(); records=[]; cases=output/'cases'; cases.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={}
        for index,model in tasks:
            path=cases/f'case_{index:04d}_{model}'/'summary.json'
            if path.exists():
                r=json.loads(path.read_text())
                if r['case_id']!=obs[index]['case_id']: raise ValueError('resume mismatch')
                records.append(r)
            else:
                f=pool.submit(predict_task,obs[index],index,model,cases,seed=SEED+50000)
                futures[f]=(index,model)
        for f in as_completed(futures):
            records.append(f.result())
            if len(records)%20==0: print(f'{len(records)}/{len(tasks)} tasks, elapsed {time.perf_counter()-begun:.1f}s',flush=True)
    expected={(i,m) for i,m in tasks}
    if len(records)!=len(expected) or {(r['observation_index'],r['model']) for r in records}!=expected:
        raise ValueError('incomplete or duplicate tasks')
    predictions=[p for r in sorted(records,key=lambda r:(r['observation_index'],r['model']))
                 for p in (r['prediction'],r['map_prediction'])]
    write_jsonl(output/'predictions.jsonl',predictions)
    write_json(output/'prediction_manifest.json',dict(predictions_sha256=file_hash(output/'predictions.jsonl'),
        observation_sha256=file_hash(output/'observations.jsonl'),tasks=len(records),
        actual_draws=sum(r['draw_count'] for r in records),wall_s=time.perf_counter()-begun))
    summarize(output)

def summarize(output):
    manifest=json.loads((output/'prediction_manifest.json').read_text())
    if file_hash(output/'predictions.jsonl')!=manifest['predictions_sha256']: raise ValueError('seal changed')
    if file_hash(output/'observations.jsonl')!=manifest['observation_sha256']: raise ValueError('observations changed')
    rows=diagnostic_rows(read_jsonl(output/'predictions.jsonl'),read_jsonl(output/'observations.jsonl'),read_jsonl(output/'truth_development.jsonl'))
    # Restore paired source grouping across all five mechanisms and noise replicas.
    for r in rows:
        if r['source_id'].startswith('group'): r['source_id']=r['source_id'].split('_')[0]
    write_jsonl(output/'per_case.jsonl',rows); write_json(output/'summary.json',aggregate(rows))
    effects={}; rng=np.random.default_rng(SEED)
    for model in ('S','F'):
        for sigma in (.005,.02):
            subset=[r for r in rows if r['method'].startswith(model+'_') and r['point_kind']=='mean' and r['sigma']==sigma]
            base={r['source_id']:r for r in subset if r['scenario']=='baseline_S'}
            for scenario in SCENARIOS[1:]:
                pairs=[(base[r['source_id']],r) for r in subset if r['scenario']==scenario and r['source_id'] in base]
                common=[(a,b) for a,b in pairs if 'error_m' in a and 'error_m' in b]
                values=np.array([b['error_m']-a['error_m'] for a,b in common])
                # Conditional comparison keeps only common true direct + valid point, explicitly counted.
                effects[f'{model}_{sigma}_{scenario}']=dict(total_pairs=len(pairs),common_point_pairs=len(common),
                    median_error_change_m=float(np.median(values)) if len(values) else None,
                    source_bootstrap_95=np.quantile([np.median(rng.choice(values,len(values))) for _ in range(1000)],[.025,.975]).tolist() if len(values) else None)
    write_json(output/'paired_effects.json',effects)
    balanced_effects(output)

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['diagnose','prepare','recovery','run','summarize']); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=30); p.add_argument('--workers',type=int,default=6); a=p.parse_args()
    if a.action=='diagnose': diagnose_existing(a.output)
    elif a.action=='prepare': prepare(a.output,a.count)
    elif a.action=='recovery': recovery_check(a.output)
    elif a.action=='run': run(a.output,a.workers)
    else: summarize(a.output)

def ls_existing(output):
    """Offline diagnostic counterpart of saved MAP fits; no predictor setting changes."""
    from inverse import fit_observation
    obs=read_jsonl(PROJECT_ROOT/'data/synthetic_demo/observations/development.jsonl')
    validate_development(obs)
    predictions=[]; details=[]
    from evaluate import point_prediction
    for o in obs:
        for model in ('S','F'):
            _,fit=fit_observation(o,model,mode='ls',starts=3)
            predictions.append(point_prediction(o,model+'_ls_diagnostic',
                fit['simulation'] if fit.get('success') else None,
                status='ok' if fit.get('success') else ('unavailable' if not o['observation_available'] else 'failure'),
                point_kind='ls_trajectory'))
            details.append(dict(case_id=o['case_id'],model=model,fit=fit))
    write_jsonl(output/'ls_fits.jsonl',details)
    write_jsonl(output/'ls_predictions.jsonl',predictions)
    rows=diagnostic_rows(predictions,obs,read_jsonl(PROJECT_ROOT/'data/synthetic_demo/truth/development.jsonl'))
    write_jsonl(output/'ls_per_case.jsonl',rows); write_json(output/'ls_summary.json',aggregate(rows))


def extra_existing(output):
    rows=read_jsonl(output/'per_case.jsonl')+read_jsonl(output/'ls_per_case.jsonl')
    extra={}; paired={}
    for model in ('S','F'):
        rs=[r for r in rows if r['method'].startswith(model+'_') and r['point_kind']=='mean' and 'error_m' in r]
        by_source=defaultdict(list)
        for r in rs: by_source[r['source_id']].append(r)
        a=np.array([(v[0]['true_time'],np.mean([r['error_m'] for r in v]),np.mean([abs(r['error_z_m']) for r in v])) for v in by_source.values()])
        extra[model]=dict(n_source=len(a),descriptive_pearson_time_error=float(np.corrcoef(a[:,0],a[:,1])[0,1]),
            descriptive_pearson_time_abs_z=float(np.corrcoef(a[:,0],a[:,2])[0,1]),time_bins=[])
        for lo,hi in ((0,.35),(.35,.45),(.45,10)):
            b=[r for r in rs if lo<=r['true_time']<hi]
            extra[model]['time_bins'].append(dict(bounds_s=[lo,hi],n=len(b),source_count=len({r['source_id'] for r in b}),
                median_error_m=float(np.median([r['error_m'] for r in b])) if b else None))
        for kind in ('ls_trajectory','map_trajectory'):
            ref={r['case_id']:r for r in rows if r['method'].startswith(model+'_') and r['point_kind']==kind and 'error_m' in r}
            pairs=[(r,ref[r['case_id']]) for r in rs if r['case_id'] in ref]; groups=defaultdict(list)
            for pair in pairs: groups[pair[0]['source_id']].append(pair)
            keys=list(groups); rng=np.random.default_rng(SEED)
            def metric(ps):return float(np.median([b['error_m'] for a,b in ps])-np.median([a['error_m'] for a,b in ps]))
            dist=[metric([p for k in rng.choice(keys,len(keys)) for p in groups[k]]) for _ in range(1000)]
            paired[model+'_'+kind+'_minus_mean']=dict(n=len(pairs),sources=len(keys),median_error_difference_m=metric(pairs),
                source_bootstrap_95=np.quantile(dist,[.025,.975]).tolist())
    write_json(output/'time_dependence.json',extra);write_json(output/'point_comparison.json',paired)


def failure_inventory(cases, output):
    from collections import Counter
    rows=[]
    for path in sorted(cases.glob('*/summary.json')):
        rec=json.loads(path.read_text()); result=rec['result']; diag=result.get('diagnostics',{}); summary=result.get('summary') or {}
        reasons=[]
        if rec['prediction']['status']=='failure':
            if summary.get('effective_samples',0)<300: reasons.append('overall_ESS_or_missing_summary')
            if diag.get('max_event_probability_change',0)>.02: reasons.append('event_probability_doubling')
            if (diag.get('conditional_mean_shift_m') or 0)>.05: reasons.append('mean_doubling')
            if (diag.get('relative_region_area_change') or 0)>.15: reasons.append('area_doubling')
            if diag.get('relative_region_area_change') is None and not diag.get('sampling_stable',False): reasons.append('conditional_region_or_condition_change')
            if summary.get('unknown_event_mass',0)>1e-6: reasons.append('unknown_event_mass')
            if not reasons: reasons.append('other_inspect_message')
        rows.append(dict(case_id=rec['case_id'],model=rec['model'],status=rec['prediction']['status'],reasons=reasons,
            effective_samples=summary.get('effective_samples'),diagnostics={k:diag.get(k) for k in ('max_event_probability_change','conditional_mean_shift_m','relative_region_area_change')},
            message=result.get('message')))
    write_jsonl(output/'failure_inventory.jsonl',rows)
    write_json(output/'failure_counts.json',{model:dict(Counter(reason for r in rows if r['model']==model for reason in r['reasons'])) for model in ('S','F')})


def balanced_effects(output):
    rows=read_jsonl(output/'per_case.jsonl'); result={}
    for model in ('S','F'):
        for sigma in (.005,.02):
            selected=[r for r in rows if r['method']==f'{model}_cd0.22_posterior' and r['sigma']==sigma and r['scenario'] in SCENARIOS]
            groups=defaultdict(dict)
            for r in selected: groups[r['source_id']][r['scenario']]=r
            direct=[v for v in groups.values() if all(s in v and v[s]['truth_event']=='goal' for s in SCENARIOS)]
            points=[v for v in direct if all('error_m' in v[s] for s in SCENARIOS)]
            result[f'{model}_{sigma}']=dict(all_scenario_true_crossing_sources=len(direct),all_scenario_point_sources=len(points),effects={})
            for scenario in SCENARIOS:
                differences=np.array([v[scenario]['error_m']-v['baseline_S']['error_m'] for v in points])
                rng=np.random.default_rng(SEED)
                result[f'{model}_{sigma}']['effects'][scenario]=dict(
                    coverage_on_common_true_crossings=float(np.mean([v[scenario]['covered'] for v in direct])) if direct else None,
                    median_error_change_on_common_points_m=float(np.median(differences)) if len(differences) else None,
                    source_bootstrap_95=np.quantile([np.median(rng.choice(differences,len(differences))) for _ in range(1000)],[.025,.975]).tolist() if len(differences) else None)
    write_json(output/'balanced_effects.json',result)

if __name__=='__main__': main()
