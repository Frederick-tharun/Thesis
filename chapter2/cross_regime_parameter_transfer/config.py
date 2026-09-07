"""Prespecified follow-up constants; never inferred from final results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RESULTS = ROOT / 'results'
BRANCH = 'chapter2-cross-regime-parameter-transfer'
START = 'd5cc55f6d2ea4210f6c06be6d79398ec90749907'
REGULAR = (1.67, 3.29, 3.50)
CHAOTIC = (3.20, 3.34)
CURRENTS = tuple(sorted(REGULAR + CHAOTIC))
SCENARIOS = {'regular_parameter_transfer': REGULAR, 'chaotic_parameter_transfer': CHAOTIC}
FOLDS = {
    'regular_parameter_transfer': (('R1', (3.29, 3.50), 1.67),
                                   ('R2', (1.67, 3.50), 3.29),
                                   ('R3', (1.67, 3.29), 3.50)),
    'chaotic_parameter_transfer': (('C1', (3.20,), 3.34), ('C2', (3.34,), 3.20)),
}
SEEDS = (42, 123, 456, 789, 2026)
SIZES = (100, 200, 300, 500, 800)
OPTIMISER_SEEDS = dict(zip(SCENARIOS, (41101, 41102)))
CALLS, INITIAL_CALLS, TOP_COUNT = 40, 10, 5
FIT_STOP, WASHOUT = 40000, 2000
WINDOWS = ((1, (40000, 42000), (42000, 50000)),
           (2, (50000, 52000), (52000, 60000)),
           (3, (60000, 62000), (62000, 70000)))
VPT_THRESHOLD, DIVERGENCE_THRESHOLD, COLLAPSE_THRESHOLD = 0.4, 5.0, 0.05
FAILURE_SCORE = 1000000.0
I_MIN, I_MAX = 1.67, 3.50
I_CENTRE, I_SCALE = (I_MIN + I_MAX) / 2, (I_MAX - I_MIN) / 2
ESN_SHA = '130c0f0f1753c7429a37bf14dbfd49de5bc8a0e04741893ef0b360126d8edcd3'
PARAMETERS = ('reservoir_size', 'reservoir_connectivity', 'input_scaling',
              'spectral_radius', 'ridge_regularisation', 'leak_rate')
RANGES = ((0.01, 1.0), (0.01, 3.0), (0.01, 3.0), (1e-10, 1e-2), (0.01, 1.0))
FINAL_ALLOCATIONS = {
    'regular_parameter_transfer': {1.67: 45334, 3.29: 45333, 3.50: 45333},
    'chaotic_parameter_transfer': {3.20: 67000, 3.34: 67000},
}
FINAL_WINDOWS = (('short_1', (70000, 72000), (72000, 80000)),
                 ('short_2', (80000, 82000), (82000, 90000)),
                 ('short_3', (89999, 91999), (91999, 99999)),
                 ('long', (70000, 72000), (72000, 99999)))


def matrix_code(scenario, current):
    if scenario not in SCENARIOS or current not in CURRENTS:
        raise ValueError('unknown scenario/current')
    return ('R' if scenario.startswith('regular') else 'C') + ('R' if current in REGULAR else 'C')
