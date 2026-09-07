"""Read-only historical inventory, strict isolated writers and selection I/O guard."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from chapter2.esn_data import file_sha256
from chapter2.esn_optimisation import atomic_write_json, load_strict_json
from .config import ROOT, PROJECT, RESULTS, START, ESN_SHA, SCENARIOS

_selection_scope = ContextVar('parameter_transfer_selection_scope', default=None)


def _audit(event, args):
    scenario = _selection_scope.get()
    if scenario is None or event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    forbidden = [PROJECT / 'chapter2' / p for p in (
        'cross_regime_improvement/results', 'cross_regime_results', 'cross_regime_models',
        'final_results', 'final_models', 'optimisation_results')]
    forbidden += [RESULTS / p for p in ('benchmark', 'models', 'evaluation.json', 'model_manifest.json')]
    if any(path == p or p in path.parents for p in forbidden):
        raise PermissionError('historical/final evidence unavailable during selection')
    from chapter2.esn_config import FIXED_DATASETS
    if any(path == p.path.resolve() and p.current not in SCENARIOS[scenario] for p in FIXED_DATASETS):
        raise PermissionError('opposite-regime trajectory unavailable during selection')


sys.addaudithook(_audit)


@contextmanager
def selection_access(scenario):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    token = _selection_scope.set(scenario)
    try:
        yield
    finally:
        _selection_scope.reset(token)


def source_hashes():
    paths = list(ROOT.glob('*.py')) + [ROOT / 'PARAMETER_TRANSFER_PROTOCOL.md',
        ROOT / 'FRESH_BENCHMARK_PROTOCOL.json', ROOT / 'protection_inventory.json']
    return {str(p.relative_to(PROJECT)): file_sha256(p) for p in sorted(paths)}


def assert_esn():
    if file_sha256(PROJECT / 'chapter2/esn_model.py') != ESN_SHA:
        raise RuntimeError('protected ESN source changed')


def verify_protected():
    data = load_strict_json(ROOT / 'protection_inventory.json')
    for relative, expected in data['hashes'].items():
        path = PROJECT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError('protected artifact changed: ' + relative)
    assert_esn()
    return {'hashes_verified': len(data['hashes'])}


def output_path(path):
    path = Path(path).resolve()
    if RESULTS.resolve() not in path.parents:
        raise PermissionError('output outside isolated results directory')
    return path


def write_new(path, value):
    path = output_path(path)
    if path.exists():
        raise FileExistsError(path)
    atomic_write_json(path, value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def stage_lock(name):
    """Exclusive stage marker; stale locks require inspection, never blind removal."""
    path = output_path(RESULTS / (name + '.running'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def require_production():
    from .config import BRANCH
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('production requires Slurm')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT, text=True).strip()
    if branch != BRANCH:
        raise RuntimeError('wrong branch')
    for args in (['git', 'diff', '--quiet'], ['git', 'diff', '--cached', '--quiet']):
        subprocess.run(args, cwd=PROJECT, check=True)
    # New implementation must be committed before production; old generated outputs are allowed.
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', str(ROOT)], cwd=PROJECT, text=True)
    if any('/results/' not in p for p in untracked.splitlines()):
        raise RuntimeError('commit reviewed implementation before production')
