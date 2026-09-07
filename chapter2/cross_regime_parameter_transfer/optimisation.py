"""Leave-one-current-out fitting and deterministic, resumable GP search."""
from dataclasses import dataclass
from functools import wraps
import json
import statistics

import numpy as np
from skopt import Optimizer
from skopt.space import Categorical, Real

from chapter2.cross_regime import load_training_prefix, recursive_forecast
from chapter2.cross_regime_numerics import evaluate_predictions
from chapter2.esn_config import ESNModelConfig, FIXED_DATASETS
from chapter2.esn_model import EchoStateNetwork, TrainingSequence
from chapter2.esn_data import NumpyStandardScaler, StateCurrentScalers, file_sha256
from chapter2.esn_optimisation import atomic_write_json, load_strict_json, software_versions
from chapter2.cross_regime_improvement.optimisation import aggregate_rollouts, candidate_rank_key, stability_objective
from .config import *
from .protection import assert_esn, selection_access, source_hashes, write_new, stage_lock, digest


def isolated(function):
    @wraps(function)
    def call(scenario, *args, **kwargs):
        with selection_access(scenario):
            return function(scenario, *args, **kwargs)
    return call


def current_scaler():
    """Fixed affine constants; no trajectory argument or fit method is used."""
    return NumpyStandardScaler(np.array([I_CENTRE]), np.array([I_SCALE]))


def scale_current(values):
    return 2 * (np.asarray(values, dtype=float) - I_MIN) / (I_MAX - I_MIN) - 1


def parameters(point):
    if len(point) != 6 or point[0] not in SIZES:
        raise ValueError('invalid search point')
    result = dict(zip(PARAMETERS, [int(point[0])] + [float(x) for x in point[1:]]))
    if any(not lo <= result[key] <= hi for key, (lo, hi) in zip(PARAMETERS[1:], RANGES)):
        raise ValueError('parameter outside prespecified bounds')
    return result


def model_config(params, seed):
    parameters([params[k] for k in PARAMETERS])
    return ESNModelConfig(**params, seed=int(seed), input_dimension=4, output_dimension=3,
                          bias_scaling=0.1, regularise_bias=False)


def dimensions():
    return [Categorical(list(SIZES), name=PARAMETERS[0])] + [
        Real(lo, hi, prior='log-uniform' if name=='ridge_regularisation' else 'uniform', name=name)
        for name, (lo, hi) in zip(PARAMETERS[1:], RANGES)]


@dataclass(frozen=True)
class Fold:
    name: str
    training_currents: tuple
    validation_current: float
    scalers: StateCurrentScalers
    sequences: tuple
    validation: object
    windows: tuple


@isolated
def load_data(scenario, loader=load_training_prefix):
    return {c: authorised_load(scenario, c, loader=loader) for c in SCENARIOS[scenario]}


def authorised_load(scenario, current, loader=load_training_prefix):
    if scenario not in SCENARIOS or current not in SCENARIOS[scenario]:
        raise PermissionError('target regime unavailable to optimisation')
    return loader(current, 70000)


def prepare_fold(scenario, definition, data, fit_stop=FIT_STOP, windows=WINDOWS):
    name, training, held = definition
    if definition not in FOLDS[scenario] or set(data) != set(SCENARIOS[scenario]):
        raise ValueError('fold/scenario membership mismatch')
    if held in training or not set(training).issubset(SCENARIOS[scenario]):
        raise PermissionError('held-out current enters training')
    if not windows or fit_stop <= 0:
        raise ValueError('empty fitting/validation')
    previous = fit_stop
    for _, warm, scored in windows:
        if not (previous <= warm[0] < warm[1] == scored[0] < scored[1]):
            raise ValueError('validation must be chronological, disjoint and adjacent to warmup')
        previous = scored[1]
    for c, trajectory in data.items():
        if trajectory.current != c or len(trajectory.states) <= max(fit_stop, windows[-1][2][1]):
            raise ValueError('trajectory identity/length mismatch')
    # Only fitting inputs of training blocks participate in learned preprocessing.
    state = NumpyStandardScaler.fit(np.concatenate([data[c].states[:fit_stop] for c in training]))
    scalers = StateCurrentScalers(state, current_scaler())
    sequences = tuple(TrainingSequence(
        scalers.transform_inputs(np.column_stack((data[c].states[:fit_stop], data[c].current_values[:fit_stop]))),
        scalers.transform_targets(data[c].states[1:fit_stop+1])) for c in training)
    return Fold(name, training, held, scalers, sequences, data[held], tuple(windows))


def aggregate(records):
    result = aggregate_rollouts(records)
    result['per_fold'] = {f: aggregate_rollouts([r for r in records if r['fold']==f]) for f in sorted({r['fold'] for r in records})}
    return result


@isolated
def evaluate_candidate(scenario, params, seed, data=None, fit_stop=FIT_STOP, windows=WINDOWS, washout=WASHOUT):
    assert_esn()
    if seed not in SEEDS:
        raise ValueError('seed outside frozen protocol')
    data = load_data(scenario) if data is None else data
    records = []
    for definition in FOLDS[scenario]:
        fold = prepare_fold(scenario, definition, data, fit_stop, windows)
        failure = None
        try:
            model = EchoStateNetwork(model_config(params, seed))
            model.fit(fold.sequences, washout=washout)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            failure = 'fit:' + type(error).__name__
        for number, warm, scored in fold.windows:
            targets = fold.validation.states[scored[0]+1:scored[1]+1]
            predictions = np.full_like(targets, np.nan)
            reason = failure
            if failure is None:
                try:
                    predictions, _, reason = recursive_forecast(model, fold.scalers,
                        fold.validation.states, fold.validation.current_values,
                        warmup_range=warm, forecast_range=scored)
                except (ArithmeticError, RuntimeError, ValueError) as error:
                    reason = 'rollout:' + type(error).__name__
            fields, _, _ = evaluate_predictions(predictions, targets, fold.scalers.state.scale)
            records.append(dict(fold=fold.name, current=fold.validation_current,
                training_currents=list(fold.training_currents), seed=seed, window=number,
                warmup=list(warm), scored=list(scored), fit_failure=failure,
                generation_failure=reason, **fields))
    total = aggregate(records)
    return {'rollouts': records, 'aggregate': total, 'objective': stability_objective(total)}


def history_path(scenario):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    return RESULTS / 'optimisation' / (scenario + '_history.json')


def design(scenario):
    return {'schema': 'parameter_transfer_search_v1', 'scenario': scenario,
        'currents': list(SCENARIOS[scenario]), 'folds': FOLDS[scenario],
        'fitting_stop': FIT_STOP, 'washout': WASHOUT, 'windows': WINDOWS,
        'current_affine': [I_CENTRE, I_SCALE], 'sizes': SIZES, 'ranges': RANGES,
        'calls': CALLS, 'initial_calls': INITIAL_CALLS, 'candidate_seed': 42,
        'optimiser_seed': OPTIMISER_SEEDS[scenario], 'confirmation_seeds': SEEDS,
        'sources': source_hashes(), 'software': software_versions(),
        'threads': {k: __import__('os').environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'dataset_hashes': {str(p.path): file_sha256(p.path) for p in FIXED_DATASETS if p.current in SCENARIOS[scenario]},
        'selection_uses_historical_results': False, 'selection_uses_final_targets': False,
        'thresholds': [VPT_THRESHOLD, DIVERGENCE_THRESHOLD, COLLAPSE_THRESHOLD, FAILURE_SCORE]}


def make_optimizer(scenario):
    return Optimizer(dimensions(), base_estimator='GP', n_initial_points=INITIAL_CALLS,
        initial_point_generator='random', acq_func='EI', acq_optimizer='auto',
        random_state=OPTIMISER_SEEDS[scenario], n_jobs=1, avoid_duplicates=True)


def replay(optimizer, trials):
    for number, trial in enumerate(trials, 1):
        proposed = [parameters(optimizer.ask())[k] for k in PARAMETERS]
        saved = trial['point']
        if trial['trial_index'] != number or proposed[0] != saved[0] or not np.allclose(proposed[1:], saved[1:], rtol=1e-15, atol=0):
            raise ValueError('deterministic optimiser replay mismatch')
        optimizer.tell(saved, trial['objective'])


@isolated
def search(scenario, resume=False):
    if (RESULTS/'optimisation/selection.json').exists():
        raise RuntimeError('selection already locked; no further search')
    metadata = json.loads(json.dumps(design(scenario)))
    path = history_path(scenario)
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        history = load_strict_json(path)
        if history['design'] != metadata:
            raise ValueError('resume design/source/runtime mismatch')
        if history['status']=='complete':
            return history
    else:
        history = {'design': metadata, 'status': 'searching', 'trials': [], 'confirmations': []}
        write_new(path, history)
    opt = make_optimizer(scenario)
    replay(opt, history['trials'])
    data = load_data(scenario)
    while len(history['trials']) < CALLS:
        params = parameters(opt.ask()); point = [params[k] for k in PARAMETERS]
        evaluated = evaluate_candidate(scenario, params, 42, data)
        opt.tell(point, evaluated['objective'])
        history['trials'].append(dict(trial_index=len(history['trials'])+1,
            point=point, hyperparameters=params, **evaluated))
        atomic_write_json(path, history)
        print(scenario, 'trial', len(history['trials']), 'of', CALLS, flush=True)
    history['status'] = 'search_complete'
    atomic_write_json(path, history)
    return history


def top_trials(history):
    selected, seen = [], set()
    for trial in sorted(history['trials'], key=candidate_rank_key):
        key = digest(trial['hyperparameters'])
        if key not in seen:
            seen.add(key); selected.append(trial)
        if len(selected)==TOP_COUNT:
            break
    return selected


def confirmed_summary(seed_results):
    result = aggregate([r for s in seed_results for r in s['rollouts']])
    result['per_seed'] = {str(s['seed']): s['aggregate'] for s in seed_results}
    result['seed_count'] = len(seed_results)
    return result


def eligible(confirmation):
    seeds = confirmation['seed_results']
    return (sorted(s['seed'] for s in seeds)==sorted(SEEDS) and
        all(s['aggregate']['numerical_failure_count']==s['aggregate']['divergence_count']==0 for s in seeds))


@isolated
def confirm(scenario):
    if (RESULTS/'optimisation/selection.json').exists():
        raise RuntimeError('selection already locked')
    path = history_path(scenario); history = load_strict_json(path)
    if len(history['trials'])!=CALLS or history['design']!=json.loads(json.dumps(design(scenario))):
        raise ValueError('incomplete search or changed confirmation design')
    data = load_data(scenario)
    for trial in top_trials(history):
        item = next((x for x in history['confirmations'] if x['hyperparameters']==trial['hyperparameters']), None)
        if item is None:
            item = {'hyperparameters':trial['hyperparameters'], 'seed_results':[]}
            history['confirmations'].append(item)
        for seed in SEEDS:
            if any(s['seed']==seed for s in item['seed_results']):
                continue
            result = ({k:trial[k] for k in ('rollouts','aggregate','objective')} if seed==42 else
                evaluate_candidate(scenario, item['hyperparameters'], seed, data))
            item['seed_results'].append({'seed':seed, **result})
            history['status']='confirming'
            atomic_write_json(path, history)
        item['aggregate']=confirmed_summary(item['seed_results'])
        item['eligible']=eligible(item)
        atomic_write_json(path, history)
    history['status']='complete'
    atomic_write_json(path, history)
    return history


def write_selection():
    models={}
    for scenario in SCENARIOS:
        h=load_strict_json(history_path(scenario))
        if h['status']!='complete' or len(h['trials'])!=CALLS:
            raise RuntimeError('search/confirmation incomplete')
        candidates=[c for c in h['confirmations'] if eligible(c)]
        if not candidates:
            raise RuntimeError(scenario + ': no five-seed stable candidate; selection NOT locked')
        chosen=min(candidates,key=candidate_rank_key)
        models[scenario]={'hyperparameters':chosen['hyperparameters'], 'aggregate':chosen['aggregate'],
                          'history_sha256':file_sha256(history_path(scenario))}
    value={'schema':'parameter_transfer_selection_v1','status':'LOCKED',
           'models':models,'sources':source_hashes(),'historical_results_used':False,
           'final_targets_used':False}
    write_new(RESULTS/'optimisation/selection.json',value)
    return value


def optimise_all(resume=False):
    with stage_lock('optimisation'):
        for scenario in SCENARIOS:
            search(scenario,resume=resume)
            confirm(scenario)
        return write_selection()
