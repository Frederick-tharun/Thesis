"""Leakage, reproducibility, immutability and smoke contracts for the follow-up."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from chapter2.esn_data import NumpyStandardScaler, StateCurrentScalers
from chapter2.esn_optimisation import atomic_write_json
from chapter2.cross_regime import recursive_forecast
from chapter2.cross_regime_parameter_transfer import config as c
from chapter2.cross_regime_parameter_transfer import optimisation as o
from chapter2.cross_regime_parameter_transfer import experiment as e
from chapter2.cross_regime_parameter_transfer import protection as p
from chapter2.cross_regime_parameter_transfer import figures as f

PARAMS=dict(zip(c.PARAMETERS,(100,0.1,0.3,0.7,1e-6,0.6)))
WINDOWS=((1,(40,50),(50,65)),)


@pytest.fixture
def isolated_output(monkeypatch,tmp_path):
    for module in (p,o,e,f):monkeypatch.setattr(module,'RESULTS',tmp_path)
    return tmp_path


def data(scenario):return {x:e.synthetic(x) for x in c.SCENARIOS[scenario]}


def test_esn_source_and_dimensions():
    p.assert_esn()
    config=o.model_config(PARAMS,42)
    assert config.input_dimension==4 and config.output_dimension==3
    assert config.bias_scaling==0.1 and config.regularise_bias is False


def test_regimes_and_search_capacity():
    assert c.REGULAR==(1.67,3.29,3.50) and c.CHAOTIC==(3.20,3.34)
    assert c.SIZES==(100,200,300,500,800)
    assert tuple(o.dimensions()[0].categories)==c.SIZES
    for size in c.SIZES:
        assert o.model_config({**PARAMS,'reservoir_size':size},42).reservoir_size==size
    with pytest.raises(ValueError):o.model_config({**PARAMS,'reservoir_size':801},42)


def test_exact_fixed_current_scaling_and_no_fit(monkeypatch):
    monkeypatch.setattr(NumpyStandardScaler,'fit',lambda *_:pytest.fail('current scaler must not fit'))
    values=np.array(c.CURRENTS)
    np.testing.assert_array_equal(o.scale_current(values),2*(values-1.67)/(3.50-1.67)-1)
    actual=o.current_scaler().transform(values[:,None])[:,0]
    np.testing.assert_allclose(actual,o.scale_current(values),atol=1e-15)
    assert np.isfinite(actual).all() and np.all(actual>=-1-1e-15) and np.all(actual<=1+1e-15)
    np.testing.assert_array_equal(o.current_scaler().mean,o.current_scaler().mean)


@pytest.mark.parametrize('scenario,definition',[(s,d) for s,ds in c.FOLDS.items() for d in ds])
def test_each_held_out_current_excluded_from_readout_and_state_scaler(scenario,definition):
    blocks=data(scenario);held=definition[2]
    before=o.prepare_fold(scenario,definition,blocks,40,WINDOWS)
    blocks[held].states[:]+=1e8
    after=o.prepare_fold(scenario,definition,blocks,40,WINDOWS)
    assert held not in after.training_currents
    np.testing.assert_array_equal(before.scalers.state.mean,after.scalers.state.mean)
    np.testing.assert_array_equal(before.scalers.state.scale,after.scalers.state.scale)
    for old,new in zip(before.sequences,after.sequences):
        np.testing.assert_array_equal(old.inputs,new.inputs)
        np.testing.assert_array_equal(old.targets,new.targets)
    expected=np.concatenate([blocks[x].states[:40] for x in definition[1]])
    np.testing.assert_allclose(after.scalers.state.mean,expected.mean(axis=0))
    assert len(after.sequences)==len(definition[1])
    for x,seq in zip(definition[1],after.sequences):
        np.testing.assert_allclose(seq.targets,after.scalers.transform_targets(blocks[x].states[1:41]))


def test_validation_chronological_and_state_scaler_ignores_training_suffix():
    scenario='regular_parameter_transfer';blocks=data(scenario);definition=c.FOLDS[scenario][0]
    before=o.prepare_fold(scenario,definition,blocks,40,WINDOWS)
    for x in definition[1]:blocks[x].states[40:]+=1e7
    after=o.prepare_fold(scenario,definition,blocks,40,WINDOWS)
    np.testing.assert_array_equal(before.scalers.state.mean,after.scalers.state.mean)
    with pytest.raises(ValueError,match='chronological'):
        o.prepare_fold(scenario,definition,blocks,40,((1,(39,50),(50,65)),))
    with pytest.raises(ValueError,match='chronological'):
        o.prepare_fold(scenario,definition,blocks,40,((1,(40,51),(50,65)),))


@pytest.mark.parametrize('scenario,current',[('regular_parameter_transfer',3.20),('chaotic_parameter_transfer',1.67)])
def test_opposite_regime_rejected_before_loader(scenario,current):
    with pytest.raises(PermissionError):o.authorised_load(scenario,current,loader=lambda *_:pytest.fail('I/O must not occur'))


@pytest.mark.parametrize('name',['evaluation_records.json','aggregate_results.json'])
def test_historical_json_io_is_blocked(name):
    path=c.PROJECT/'chapter2/cross_regime_improvement/results'/name
    with p.selection_access('regular_parameter_transfer'):
        with pytest.raises(PermissionError):path.read_text()
        with pytest.raises(PermissionError):open(path,'rb')


def test_opposite_regime_file_and_fresh_output_io_blocked():
    from chapter2.esn_config import FIXED_DATASETS
    path=next(x.path for x in FIXED_DATASETS if x.current==3.20)
    with p.selection_access('regular_parameter_transfer'):
        with pytest.raises(PermissionError):path.read_bytes()
        with pytest.raises(PermissionError):(c.RESULTS/'evaluation.json').read_text()


def test_guard_active_inside_candidate(monkeypatch):
    def forbidden(*a,**k):
        (c.PROJECT/'chapter2/cross_regime_improvement/results/aggregate_results.json').read_text()
    monkeypatch.setattr(o,'prepare_fold',forbidden)
    with pytest.raises(PermissionError):o.evaluate_candidate('regular_parameter_transfer',PARAMS,42,data('regular_parameter_transfer'),40,WINDOWS,5)


def test_future_truth_not_used_and_current_supplied():
    states=np.arange(30,dtype=float).reshape(10,3);current=np.linspace(1.67,3.50,10)
    scaler=StateCurrentScalers(NumpyStandardScaler(np.zeros(3),np.ones(3)),o.current_scaler());spy=e.FeedbackSpy()
    first,_,_=recursive_forecast(spy,scaler,states,current,warmup_range=(0,2),forecast_range=(2,9))
    states[3:]=-1e9
    second,_,_=recursive_forecast(e.FeedbackSpy(),scaler,states,current,warmup_range=(0,2),forecast_range=(2,9))
    np.testing.assert_array_equal(first,second)
    np.testing.assert_allclose(np.array(spy.inputs)[:,3],o.scale_current(current[2:9]))


def ranked(fail=0,div=0,score=1,leak=.6):
    return {'aggregate':{'numerical_failure_count':fail,'divergence_count':div,
        'worst_nrmse':score,'median_nrmse':score,'mean_nrmse':score,'median_vpt':1,'mean_vpt':1},
        'hyperparameters':{**PARAMS,'leak_rate':leak}}


def test_numerical_failure_priority():
    assert o.candidate_rank_key(ranked(div=9,score=100))<o.candidate_rank_key(ranked(fail=1,score=.001))


def test_divergence_priority():
    assert o.candidate_rank_key(ranked(score=100))<o.candidate_rank_key(ranked(div=1,score=.001))


def test_deterministic_ranking_tie_break():
    a,b=ranked(leak=.1),ranked(leak=.2)
    assert sorted([b,a],key=o.candidate_rank_key)==[a,b]


def test_five_seed_eligibility_rejects_unstable_or_missing_seed():
    item={'seed_results':[{'seed':k,'aggregate':ranked()['aggregate']} for k in c.SEEDS]}
    assert o.eligible(item)
    bad=copy.deepcopy(item);bad['seed_results'][3]['aggregate']['divergence_count']=1
    assert not o.eligible(bad)
    bad=copy.deepcopy(item);bad['seed_results'][0]['aggregate']['numerical_failure_count']=1
    assert not o.eligible(bad)
    item['seed_results'].pop();assert not o.eligible(item)


def test_no_eligible_candidate_cannot_lock(isolated_output):
    for scenario in c.SCENARIOS:
        p.write_new(o.history_path(scenario),{'status':'complete','trials':[{}]*40,'confirmations':[]})
    with pytest.raises(RuntimeError,match='no five-seed stable'):o.write_selection()
    assert not (isolated_output/'optimisation/selection.json').exists()


def test_deterministic_gp_replay_and_tamper_detection():
    scenario='regular_parameter_transfer';a=o.make_optimizer(scenario);trials=[]
    for i in range(2):
        point=[o.parameters(a.ask())[k] for k in c.PARAMETERS];a.tell(point,float(i+1))
        trials.append({'trial_index':i+1,'point':point,'objective':float(i+1)})
    b=o.make_optimizer(scenario);o.replay(b,trials)
    np.testing.assert_allclose(a.ask(),b.ask(),rtol=1e-15,atol=0)
    trials[0]['point'][1]+=.1
    with pytest.raises(ValueError,match='replay'):o.replay(o.make_optimizer(scenario),trials)


def test_search_cannot_resume_after_selection(isolated_output):
    p.write_new(isolated_output/'optimisation/selection.json',{'status':'LOCKED'})
    with pytest.raises(RuntimeError,match='already locked'):o.search('regular_parameter_transfer',True)


def test_fresh_benchmark_cannot_open_without_selection(isolated_output,monkeypatch):
    monkeypatch.setattr(e,'simulate_fixed_current',lambda *a,**k:pytest.fail('fresh generator called'))
    with pytest.raises(RuntimeError,match='locked'):e.fresh_data()
    with pytest.raises(RuntimeError,match='locked'):e.final_evaluation()


def test_benchmark_manifest_exclusive_and_bound_to_models(isolated_output,monkeypatch):
    monkeypatch.setattr(e,'verified_models',lambda:{})
    monkeypatch.setattr(e,'selection',lambda:({},'selection-sha'))
    p.write_new(isolated_output/'model_manifest.json',{'models':[]})
    first=e.freeze_benchmark_manifest()
    assert first['plan']['initial_state']==[.1,0,0]
    assert e.benchmark_manifest()==first
    with pytest.raises(FileExistsError):e.freeze_benchmark_manifest()
    atomic_write_json(isolated_output/'model_manifest.json',{'models':[1]})
    with pytest.raises(RuntimeError,match='immutable'):e.benchmark_manifest()


def test_no_training_after_benchmark_access(isolated_output,monkeypatch):
    monkeypatch.setattr(e,'selection',lambda:({},'sha'))
    monkeypatch.setattr(e,'verified_models',lambda:{'already_locked':True})
    monkeypatch.setattr(e,'final_training',lambda *_:pytest.fail('retraining forbidden'))
    p.write_new(isolated_output/'benchmark/manifest.json',{})
    assert e.train_final(True)=={'already_locked':True}


def test_output_overwrite_path_escape_and_strict_json(isolated_output):
    path=isolated_output/'a.json';p.write_new(path,{'a':1})
    with pytest.raises(FileExistsError):p.write_new(path,{'a':2})
    assert json.loads(path.read_text())=={'a':1}
    with pytest.raises(PermissionError):p.write_new(isolated_output.parent/'escape.json',{})
    with pytest.raises(ValueError):p.write_new(isolated_output/'bad.json',{'x':float('nan')})


def test_concurrent_stage_lock_rejected(isolated_output):
    with p.stage_lock('test'):
        with pytest.raises(FileExistsError):
            with p.stage_lock('test'):pass
    assert not (isolated_output/'test.running').exists()


def test_fresh_protocol_uses_established_alternative_and_same_physics():
    plan=e.benchmark_plan()
    assert plan['initial_state']==[.1,0,0]
    assert plan['dt']==.01 and plan['parameters']['r']==.006
    assert plan['expected_records']==200 and len(e.expected_ids())==200


def test_secondary_metrics_real_utilities():
    t=e.synthetic(1.67)
    result=e.climate(t.states,t.time,t.current_values)
    assert result['defined'] and 'interspike_intervals' in result['spike_burst_statistics']
    json.dumps(result,allow_nan=False)
    t.states[1]=np.nan
    assert not e.climate(t.states,t.time,t.current_values)['defined']


def test_historical_figure_loader_requires_complete_fresh_results(isolated_output):
    with pytest.raises(RuntimeError,match='locked'):f.historical_records()


def test_all_thirteen_figure_groups_smoke(isolated_output):
    records=[]
    for scenario in c.SCENARIOS:
        for current in c.CURRENTS:
            records.append({'matrix_code':c.matrix_code(scenario,current),'seed':42,'current':current,
                'window':'short_1','family':'fixed_short','metrics':{'nrmse_state':.2,'valid_prediction_time':1.,'diverged':False}})
    t=e.synthetic(1.67);arrays=lambda _:dict(time=t.time,targets=t.states,predictions=t.states)
    h={s:{'trials':[{'objective':1.},{'objective':.5}], 'confirmations':[{'seed_results':[{'seed':k,'aggregate':{'worst_nrmse':.2}} for k in c.SEEDS]}]} for s in c.SCENARIOS}
    paths=f.design_figures()+f.optimisation_figures(h)+f.prediction_figures(records,arrays)
    paths+=f.metric_figures(records)+f.phase_figure(records,arrays)+f.comparison_figure(records,records)
    assert len(paths)==26 and all(Path(x).stat().st_size>1000 for x in paths)


def test_tiny_pilot_and_all_folds(isolated_output):
    report=e.run_pilot(check_protection=False)
    assert report['passed'] and all(report['checks'].values())
    assert len(report['scenarios']['regular_parameter_transfer']['rollouts'])==3
    assert len(report['scenarios']['chaotic_parameter_transfer']['rollouts'])==2
    assert not (isolated_output/'benchmark').exists()


def test_all_previous_scientific_and_improvement_artifacts_unchanged():
    assert p.verify_protected()['hashes_verified']>=1412


def test_confirmation_repeats_all_folds_for_all_five_seeds(isolated_output,monkeypatch):
    calls=[]
    def evaluate(scenario,params,seed,blocks):
        calls.append((scenario,seed))
        result=o.evaluate_candidate_original(scenario,params,seed,blocks,40,WINDOWS,5)
        assert {r['fold'] for r in result['rollouts']}=={d[0] for d in c.FOLDS[scenario]}
        return result
    monkeypatch.setattr(o,'evaluate_candidate_original',o.evaluate_candidate,raising=False)
    monkeypatch.setattr(o,'evaluate_candidate',evaluate)
    monkeypatch.setattr(o,'load_data',lambda scenario:data(scenario))
    for scenario in c.SCENARIOS:
        metadata=json.loads(json.dumps(o.design(scenario)))
        base=o.evaluate_candidate_original(scenario,PARAMS,42,data(scenario),40,WINDOWS,5)
        trial={'hyperparameters':PARAMS,**base}
        p.write_new(o.history_path(scenario),{'design':metadata,'status':'search_complete','trials':[trial]*40,'confirmations':[]})
        h=o.confirm(scenario)
        assert h['status']=='complete'
        seed_results=h['confirmations'][0]['seed_results']
        assert [r['seed'] for r in seed_results]==list(c.SEEDS)
        assert h['confirmations'][0]['aggregate']['rollout_count']==len(c.FOLDS[scenario])*5
        assert all(r['current'] not in r['training_currents'] for s in seed_results for r in s['rollouts'])
    assert calls==[(s,k) for s in c.SCENARIOS for k in c.SEEDS if k!=42]


def test_resume_cannot_ignore_changed_design(isolated_output,monkeypatch):
    monkeypatch.setattr(o,'design',lambda scenario:{'expected':'current'})
    p.write_new(o.history_path('regular_parameter_transfer'),{'design':{'expected':'old'},'status':'searching'})
    with pytest.raises(ValueError,match='resume design'):o.search('regular_parameter_transfer',True)
