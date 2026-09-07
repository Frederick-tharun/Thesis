"""Synthetic pilot and gated final evaluation; no final-target tuning path."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from chapter2.cross_regime import load_training_prefix, recursive_forecast, save_model_bundle, load_model_bundle
from chapter2.esn_data import FixedCurrentTrajectory, NumpyStandardScaler, StateCurrentScalers, file_sha256
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_optimisation import atomic_write_json, load_strict_json
from chapter2.cross_regime_numerics import evaluate_predictions
from chapter2.esn_step8 import atomic_save_npz
from chapter2.config_ch2 import HR_PARAMETERS, DT
from chapter2.hr_data_ch2 import simulate_fixed_current, HRTrajectory
from chapter2.dynamics_analysis_ch2 import analyze_spikes_and_bursts

from .config import *
from .optimisation import current_scaler, scale_current, model_config, evaluate_candidate, prepare_fold, optimise_all
from .protection import (write_new, verify_protected, assert_esn, output_path,
                         source_hashes, require_production, stage_lock, selection_access)

BENCHMARK_PROTOCOL = ROOT / 'FRESH_BENCHMARK_PROTOCOL.json'


def selection():
    path=RESULTS/'optimisation/selection.json'
    if not path.is_file():
        raise RuntimeError('selection must be locked')
    value=load_strict_json(path)
    if value['status']!='LOCKED' or value['sources']!=source_hashes() or set(value['models'])!=set(SCENARIOS):
        raise RuntimeError('selection source/design lock mismatch')
    for scenario, item in value['models'].items():
        if file_sha256(RESULTS/'optimisation'/f'{scenario}_history.json')!=item['history_sha256']:
            raise RuntimeError('selection history provenance mismatch')
    return value, file_sha256(path)


def _resume(path, initial, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        value=load_strict_json(path)
        for key in initial:
            if key not in ('status','models','records') and value.get(key)!=initial[key]:
                raise RuntimeError('resume provenance mismatch: '+key)
        return value
    write_new(path,initial)
    return initial


def final_training(scenario, loader=load_training_prefix):
    allocation=FINAL_ALLOCATIONS[scenario]
    blocks={c:loader(c,n) for c,n in allocation.items()}
    scaler=StateCurrentScalers(NumpyStandardScaler.fit(np.concatenate([blocks[c].states[:n] for c,n in allocation.items()])),current_scaler())
    sequences=tuple(TrainingSequence(
        scaler.transform_inputs(np.column_stack((blocks[c].states[:n],blocks[c].current_values[:n]))),
        scaler.transform_targets(blocks[c].states[1:n+1])) for c,n in allocation.items())
    if sum(len(s.inputs)-WASHOUT for s in sequences)!=130000:
        raise RuntimeError('final training budget mismatch')
    return scaler,sequences


def train_final(resume=False):
    chosen,sha=selection()
    if (RESULTS/'benchmark/manifest.json').exists() or (RESULTS/'evaluation.json').exists():
        return verified_models()  # Target access permanently closes the training stage.
    path=RESULTS/'model_manifest.json'
    manifest=_resume(path,{'schema':'parameter_transfer_models_v1','status':'training',
        'selection_sha256':sha,'sources':source_hashes(),'models':[]},resume)
    for scenario in SCENARIOS:
        params=chosen['models'][scenario]['hyperparameters']
        scaler,sequences=final_training(scenario)
        for seed in SEEDS:
            if any(m['scenario']==scenario and m['seed']==seed for m in manifest['models']):
                continue
            out=output_path(RESULTS/'models'/f'{scenario}__{seed}.npz')
            if out.exists():
                raise FileExistsError('orphan model requires inspection: '+str(out))
            model=EchoStateNetwork(model_config(params,seed));model.fit(sequences,washout=WASHOUT)
            metadata={'scenario':scenario,'seed':seed,'selection_sha256':sha,
                'hyperparameters':params,'training_currents':list(SCENARIOS[scenario]),
                'effective_samples':130000,'current_affine':[I_CENTRE,I_SCALE],
                'sources':source_hashes()}
            save_model_bundle(out,model,scaler,metadata)
            _,loaded,meta=load_model_bundle(out)
            if meta!=metadata or not np.array_equal(loaded.current.scale,scaler.current.scale):
                raise RuntimeError('model bundle round-trip mismatch')
            manifest['models'].append({'scenario':scenario,'seed':seed,'path':str(out),
                'sha256':file_sha256(out),'metadata':metadata})
            atomic_write_json(path,manifest)
    manifest['status']='complete';atomic_write_json(path,manifest)
    return verified_models()


def verified_models():
    chosen,sha=selection()
    path=RESULTS/'model_manifest.json'
    if not path.exists():
        raise RuntimeError('all ten models must be locked before fresh benchmark access')
    manifest=load_strict_json(path)
    expected={(s,k) for s in SCENARIOS for k in SEEDS}
    if manifest['status']!='complete' or manifest['selection_sha256']!=sha or len(manifest['models'])!=10 or {(m['scenario'],m['seed']) for m in manifest['models']}!=expected:
        raise RuntimeError('model lock incomplete')
    for item in manifest['models']:
        model,scaler,meta=load_model_bundle(Path(item['path']))
        params=chosen['models'][item['scenario']]['hyperparameters']
        if (file_sha256(item['path'])!=item['sha256'] or meta!=item['metadata'] or
            meta['selection_sha256']!=sha or meta['hyperparameters']!=params or
            meta['training_currents']!=list(SCENARIOS[item['scenario']]) or
            asdict(model.config)!=asdict(model_config(params,item['seed'])) or
            not np.array_equal(scaler.current.mean,[I_CENTRE]) or
            not np.array_equal(scaler.current.scale,[I_SCALE])):
            raise RuntimeError('model hash/configuration/provenance mismatch')
    return manifest


def benchmark_plan():
    plan=load_strict_json(BENCHMARK_PROTOCOL)
    # Validate provenance against the established alternate HR state, not final metrics.
    from config import HR_PARAMETER_SETS
    existing=HR_PARAMETER_SETS['periodic_spiking']
    if plan['initial_state']!=existing['x0'] or plan['parameters']!=asdict(HR_PARAMETERS):
        raise RuntimeError('benchmark initial-state/physics provenance mismatch')
    if (tuple(plan['currents'])!=CURRENTS or plan['dt']!=DT or
        plan['windows']!=json.loads(json.dumps(FINAL_WINDOWS)) or
        plan['reservoir_seeds']!=list(SEEDS) or plan['retained_samples']!=100000 or
        plan['transient_steps']!=100000 or plan['continuous_schedules']):
        raise RuntimeError('benchmark protocol changed')
    return plan


def freeze_benchmark_manifest():
    """Lock design before simulator/prediction access; exclusive creation."""
    verified_models()
    plan=benchmark_plan()
    value={'schema':'parameter_transfer_benchmark_v1','status':'design_locked','plan':plan,
           'plan_sha256':file_sha256(BENCHMARK_PROTOCOL),
           'selection_sha256':selection()[1],
           'models_sha256':file_sha256(RESULTS/'model_manifest.json')}
    write_new(RESULTS/'benchmark/manifest.json',value)
    return value


def benchmark_manifest():
    verified_models()
    value=load_strict_json(RESULTS/'benchmark/manifest.json')
    if (value['plan']!=benchmark_plan() or value['plan_sha256']!=file_sha256(BENCHMARK_PROTOCOL)
        or value['models_sha256']!=file_sha256(RESULTS/'model_manifest.json')
        or value['selection_sha256']!=selection()[1]):
        raise RuntimeError('fresh benchmark immutable manifest mismatch')
    return value


def fresh_data(resume=False):
    verified_models()
    if not (RESULTS/'benchmark/manifest.json').exists():
        freeze_benchmark_manifest()
    manifest=benchmark_manifest();plan=manifest['plan']
    inventory=RESULTS/'benchmark/datasets.json'
    if inventory.exists():
        if not resume:
            raise FileExistsError(inventory)
        datasets=load_strict_json(inventory)
        if datasets['manifest_sha256']!=file_sha256(RESULTS/'benchmark/manifest.json'):
            raise RuntimeError('fresh dataset manifest mismatch')
    else:
        datasets={'manifest_sha256':file_sha256(RESULTS/'benchmark/manifest.json'),'datasets':[]}
        write_new(inventory,datasets)
    for c in CURRENTS:
        if any(d['current']==c for d in datasets['datasets']):
            continue
        path=output_path(RESULTS/'benchmark'/f'I_{c:.2f}.npz')
        if path.exists():
            raise FileExistsError('orphan fresh dataset requires inspection: '+str(path))
        t=simulate_fixed_current(c,retained_samples=plan['retained_samples'],
            transient_steps=plan['transient_steps'],initial_state=plan['initial_state'],dt=plan['dt'])
        atomic_save_npz(path,time=t.t,states=t.state,current=t.I)
        datasets['datasets'].append({'current':c,'path':str(path),'sha256':file_sha256(path)})
        atomic_write_json(inventory,datasets)
    result={}
    for d in datasets['datasets']:
        if file_sha256(d['path'])!=d['sha256']:
            raise RuntimeError('fresh trajectory hash mismatch')
        with np.load(d['path'],allow_pickle=False) as a:
            result[d['current']]=FixedCurrentTrajectory(d['current'],a['time'],a['states'],a['current'])
    if set(result)!=set(CURRENTS):
        raise RuntimeError('fresh current matrix incomplete')
    return result


def strict_values(value):
    if isinstance(value,dict):return {k:strict_values(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [strict_values(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def climate(states, times, current):
    if not np.isfinite(states).all():
        return {'defined':False,'reason':'nonfinite trajectory; no climate claim'}
    trajectory=HRTrajectory(times,*states.T,current)
    events=analyze_spikes_and_bursts(trajectory)
    duration=len(times)*DT
    return strict_values({'defined':True,'mean':states.mean(0),'std':states.std(0),
        'min':states.min(0),'max':states.max(0),'range':np.ptp(states,axis=0),
        'spike_frequency':len(events.spike_indices)/duration,
        'burst_frequency':events.burst_count/duration if events.burst_count is not None else None,
        'spike_burst_statistics':asdict(events)})


def expected_ids():
    return {f'{s}__{k}__{c:.2f}__{w}' for s in SCENARIOS for k in SEEDS for c in CURRENTS for w,_,_ in FINAL_WINDOWS}


def final_evaluation(resume=False):
    models=verified_models();trajectories=fresh_data(resume)
    path=RESULTS/'evaluation.json'
    raw=_resume(path,{'schema':'parameter_transfer_final_v1','status':'evaluating',
        'selection_sha256':selection()[1],'model_manifest_sha256':file_sha256(RESULTS/'model_manifest.json'),
        'benchmark_manifest_sha256':file_sha256(RESULTS/'benchmark/manifest.json'),
        'datasets_sha256':file_sha256(RESULTS/'benchmark/datasets.json'),'records':[]},resume)
    for item in models['models']:
        model,scaler,_=load_model_bundle(Path(item['path']))
        for current,t in trajectories.items():
            for window,warm,scored in FINAL_WINDOWS:
                identifier=f"{item['scenario']}__{item['seed']}__{current:.2f}__{window}"
                if any(r['record_id']==identifier for r in raw['records']):continue
                out=output_path(RESULTS/'raw_arrays'/(identifier+'.npz'))
                if out.exists():raise FileExistsError('orphan rollout requires inspection: '+str(out))
                pred,step,reason=recursive_forecast(model,scaler,t.states,t.current_values,warmup_range=warm,forecast_range=scored)
                begin,stop=scored;target=t.states[begin+1:stop+1];time=t.time[begin+1:stop+1];curr=t.current_values[begin:stop]
                fields,_,error=evaluate_predictions(pred,target,scaler.state.scale)
                atomic_save_npz(out,predictions=pred,targets=target,time=time,current=curr,pointwise_normalised_error=error)
                raw['records'].append(dict(record_id=identifier,scenario=item['scenario'],seed=item['seed'],
                    current=current,matrix_code=matrix_code(item['scenario'],current),window=window,
                    family='fixed_long' if window=='long' else 'fixed_short',warmup=list(warm),forecast=list(scored),
                    generation_failure_step=step,generation_failure_reason=reason,
                    raw_arrays_path=str(out),raw_arrays_sha256=file_sha256(out),model_sha256=item['sha256'],
                    secondary={'prediction':climate(pred,time,curr),'truth':climate(target,time,curr),
                        'prediction_interpretation':'descriptive only; inspect primary divergence before interpreting'},**fields))
                atomic_write_json(path,raw)
    if len(raw['records'])!=200 or {r['record_id'] for r in raw['records']}!=expected_ids():
        raise RuntimeError('final record matrix incomplete/duplicated')
    raw['status']='complete';atomic_write_json(path,raw)
    return audit_final()


def audit_final():
    models=verified_models();benchmark_manifest();raw=load_strict_json(RESULTS/'evaluation.json')
    if raw['status']!='complete' or len(raw['records'])!=200 or {r['record_id'] for r in raw['records']}!=expected_ids():
        raise RuntimeError('final results not complete')
    for key,path in (('selection_sha256','optimisation/selection.json'),('model_manifest_sha256','model_manifest.json'),
                     ('benchmark_manifest_sha256','benchmark/manifest.json'),('datasets_sha256','benchmark/datasets.json')):
        if raw[key]!=file_sha256(RESULTS/path):raise RuntimeError('final provenance mismatch')
    hashes={(m['scenario'],m['seed']):m['sha256'] for m in models['models']}
    for r in raw['records']:
        if file_sha256(r['raw_arrays_path'])!=r['raw_arrays_sha256'] or r['model_sha256']!=hashes[(r['scenario'],r['seed'])]:
            raise RuntimeError('final record hash mismatch')
    return raw


def synthetic(current,n=70):
    t=np.arange(n+1)*DT
    s=np.column_stack((np.sin(t)+current/10,np.cos(t),np.sin(t/3)))
    return FixedCurrentTrajectory(current,t,s,np.full(n+1,current))


class FeedbackSpy:
    def __init__(self):self.inputs=[]
    def teacher_forced_warmup(self,inputs,reset=True):pass
    def predict_one_step(self,value):
        self.inputs.append(value.copy());return value[:3]+0.01
    def reset_reservoir(self):pass


def run_pilot(output=None,check_protection=True):
    before=verify_protected() if check_protection else {'not_run':'test fixture'}
    params=dict(zip(PARAMETERS,(100,0.1,0.3,0.7,1e-6,0.6)))
    windows=((1,(40,50),(50,65)),);reports={}
    checks={'scaling_exact':bool(np.allclose(current_scaler().transform(np.array(CURRENTS)[:,None])[:,0],scale_current(CURRENTS))),
            'scaling_bounded':bool(np.all(np.abs(scale_current(CURRENTS))<=1)),
            'dimensions':model_config(params,42).input_dimension==4 and model_config(params,42).output_dimension==3}
    for scenario in SCENARIOS:
        data={c:synthetic(c) for c in SCENARIOS[scenario]}
        result=evaluate_candidate(scenario,params,42,data,fit_stop=40,windows=windows,washout=5)
        reports[scenario]=result
        checks[scenario+'_unseen']=all(r['current'] not in r['training_currents'] for r in result['rollouts'])
        checks[scenario+'_rollout']=all(r['fit_failure'] is None and not r['numerical_failure'] for r in result['rollouts'])
    unit=StateCurrentScalers(NumpyStandardScaler(np.zeros(3),np.ones(3)),current_scaler())
    states=np.arange(30,dtype=float).reshape(10,3);currents=np.linspace(I_MIN,I_MAX,10);spy=FeedbackSpy()
    first,_,_=recursive_forecast(spy,unit,states,currents,warmup_range=(0,2),forecast_range=(2,9))
    states[3:]=-999
    second,_,_=recursive_forecast(FeedbackSpy(),unit,states,currents,warmup_range=(0,2),forecast_range=(2,9))
    checks['future_truth_not_used']=bool(np.array_equal(first,second))
    checks['current_supplied']=bool(np.allclose(np.array(spy.inputs)[:,3],scale_current(currents[2:9])))
    try:
        with selection_access('regular_parameter_transfer'):
            (PROJECT/'chapter2/cross_regime_improvement/results/evaluation_records.json').read_text()
    except PermissionError:checks['historical_io_blocked']=True
    else:checks['historical_io_blocked']=False
    if check_protection:assert verify_protected()==before
    report={'schema':'parameter_transfer_synthetic_pilot_v1','passed':all(checks.values()),
            'checks':checks,'protection':before,'scenarios':reports,'synthetic_only':True}
    write_new(output or RESULTS/'pilot_report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for mode in ('pilot','optimise-all','final'):group.add_argument('--'+mode,action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args(argv)
    if args.pilot:
        report=run_pilot();print('PILOT PASSED' if report['passed'] else 'PILOT FAILED');return 0 if report['passed'] else 1
    require_production();verify_protected()
    if args.optimise_all:
        optimise_all(args.resume)
    else:
        with stage_lock('final'):
            train_final(args.resume);final_evaluation(args.resume)
            from .figures import generate_final
            generate_final()
    verify_protected();return 0


if __name__=='__main__':raise SystemExit(main())
