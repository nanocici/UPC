"""Independent saved-candidate/reference-integrator audit; never reads truth."""
import json
import sys
from pathlib import Path
import numpy as np
from config import Environment
from data import load_observations,observation_case
from final_config import FINAL
from physics import simulate,is_in_goal_opening
from batch_f_sampler import prepare,propagate,EVENTS


def audit(output):
    rows=load_observations(output/'observations.jsonl')
    indices=np.linspace(0,len(rows)-1,32,dtype=int)
    maxima=dict(observation_error_m=0.,event_position_error_m=0.,event_time_error_s=0.,log_target_error=0.,step_halving_position_error_m=0.)
    checked=[];skipped=[];candidate_count=0
    for index in indices:
        row=rows[index];directory=output/'cases'/row['case_id']
        path=directory/'prediction.json'
        if not path.exists():raise ValueError('Prediction run must finish before final audit')
        prediction=json.loads(path.read_text())
        if prediction['status']!='computed':skipped.append(row['case_id']);continue
        case=observation_case(row);env=Environment(**dict(row['known_environment'],wind_m_s=tuple(row['known_environment']['wind_m_s'])))
        _,post,_,_=prepare(case,env,FINAL.inference_settings(),row['measurement']['sigma_m'])
        with np.load(directory/f"candidates_{prediction['seeds'][0]}.npz") as f:
            data={k:f[k] for k in f.files}
        valid=np.flatnonzero(np.isfinite(data['log_target']))
        selected=valid[np.linspace(0,len(valid)-1,min(16,len(valid)),dtype=int)]
        for i in selected:
            p=data['parameters'][i]
            ref=simulate('F',dict(initial_velocity_m_s=p[:3],b_y=p[3],b_z=p[4],drag_coefficient=FINAL.drag_coefficient),case.times_s,case,env)
            expected=('goal_inside' if is_in_goal_opening(ref['event_position_m'],case) else 'goal_outside') if ref['event']=='goal' else ref['event']
            if EVENTS[data['event'][i]]!=expected:raise AssertionError((row['case_id'],i,'event mismatch'))
            err=float(np.max(np.abs(np.asarray(ref['positions_m']).ravel()-data['observation_positions_m'][i])))
            maxima['observation_error_m']=max(maxima['observation_error_m'],err)
            if ref['event_time_s'] is not None:
                maxima['event_time_error_s']=max(maxima['event_time_error_s'],abs(ref['event_time_s']-data['arrival_time_s'][i]))
                maxima['event_position_error_m']=max(maxima['event_position_error_m'],float(np.linalg.norm(np.array(ref['event_position_m'])[1:]-data['goal_yz_m'][i])))
            log=post.log_density(p/post.scales)
            maxima['log_target_error']=max(maxima['log_target_error'],abs(log-data['log_target'][i]))
            candidate_count+=1
        fine=propagate(data['parameters'][selected],case,env,step=.0025)
        np.testing.assert_array_equal(fine[1],data['event'][selected])
        maxima['step_halving_position_error_m']=max(maxima['step_halving_position_error_m'],float(np.max(np.abs(fine[2][:,:2]-data['goal_yz_m'][selected]))))
        checked.append(row['case_id'])
    checks=dict(observation_accuracy=maxima['observation_error_m']<=1e-6,
                event_position_accuracy=maxima['event_position_error_m']<=1e-4,
                event_time_accuracy=maxima['event_time_error_s']<=1e-6,
                log_target_accuracy=maxima['log_target_error']<=1e-5,
                step_halving_accuracy=maxima['step_halving_position_error_m']<=1e-4)
    maxima={k:float(v) for k,v in maxima.items()}
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(checked_case_ids=checked,skipped_noncomputed_case_ids=skipped,candidate_count=candidate_count,maxima=maxima,checks=checks,all_pass=all(checks.values()),truth_access=False)
    (output/'numerical_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not result['all_pass']:raise AssertionError(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':audit(Path(sys.argv[1]))
