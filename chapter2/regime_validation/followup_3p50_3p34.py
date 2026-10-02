"""Follow-up checks for the two near-misses in current_regimes_validation.json.

I = 3.50 (periodic spiking) missed the one-interval cycle limit (0.54% vs
0.5%) after a 2,000-unit transient. This script shows the alternation
between consecutive ISIs decays and passes after a 10,000-unit transient.

I = 3.34 (chaotic bursting) missed the quarter-to-quarter drift limit
(0.40 vs 0.35) over 1,500 retained time units. This script shows the drift
falls to about 0.07 over 3,000 and 5,000 time units. The Lyapunov values
passed to the validator for I = 3.34 are the 5,000-unit estimate and its
convergence spread from current_regimes_validation.json.

Run on a node of the `work` partition from the repository root:

    python chapter2/regime_validation/followup_3p50_3p34.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import _rk4_hr
from hr_regime_validation import extract_inter_spike_intervals, validate_hr_trajectory

OUTPUT = REPO_ROOT / "chapter2" / "regime_validation" / "followup_3p50_3p34.json"

B=dict(a=1.0,b=3.0,c=1.0,d=5.0,r=0.006,s=4.0,xr=-1.6)
out={}
# 3.50: does the lag-1 alternation decay with a longer transient?
for trans in (200_000, 1_000_000):
    t=_rk4_hr((-1.0,-3.0,3.0), trans+150_000, 0.01, dict(B,I=3.50))[trans:]
    _,isi=extract_inter_spike_intervals(t[:,0],0.01); m=isi.mean()
    e1=np.max(np.abs(isi[1:]-isi[:-1]))/m; e2=np.max(np.abs(isi[2:]-isi[:-2]))/m
    alt=np.abs(np.diff(isi))/m
    out[f'3.50_transient_{trans//100}']=dict(lag1=e1,lag2=e2,alt_first5=alt[:5].tolist(),alt_last5=alt[-5:].tolist(),isi_first4=isi[:4].tolist())
    v=validate_hr_trajectory(t,dt=0.01,expected_regime='periodic_spiking',expected_cycle_length=1,transient_steps=trans,largest_lyapunov_exponent=0.0,lyapunov_tail_std=1.0)
    out[f'3.50_transient_{trans//100}']['passed']=v['passed']
# 3.34: drift with longer retained windows
for keep in (150_000, 300_000, 500_000):
    t=_rk4_hr((-1.0,-3.0,3.0), 400_000+keep, 0.01, dict(B,I=3.34))[400_000:]
    v=validate_hr_trajectory(t,dt=0.01,expected_regime='chaotic_bursting',expected_cycle_length=None,transient_steps=400_000,largest_lyapunov_exponent=0.0106,lyapunov_tail_std=0.0005)
    out[f'3.34_retained_{keep//100}']=dict(mean_shift=v['stationarity_diagnostics']['maximum_normalized_mean_shift'],std_shift=v['stationarity_diagnostics']['maximum_normalized_std_shift'],passed=v['passed'])
OUTPUT.write_text(json.dumps(out, indent=1) + "\n")
print(f"saved {OUTPUT}")
