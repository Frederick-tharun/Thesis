# Chapter 2 Part 2 - long-horizon cross-regime results (280 time units)

| Direction | Role | Median NRMSE | Median VPT (tu) | Diverged | Spike count within 10% |
|---|---|---:|---:|---:|---:|
| RR | control | 0.511 | 35.6 | 0/3 | 3/3 |
| RC | **cross-regime result** | 0.293 | 53.0 | 0/3 | 3/3 |
| CC | control | 0.628 | 58.8 | 0/3 | 2/3 |
| CR | **cross-regime result** | 0.567 | 21.2 | 1/3 | 2/3 |

Reference - Chapter 2 Part 1, parameter-aware ESN, genuinely unseen currents:

| Part 1 family | Median NRMSE | Mean VPT (tu) | Divergence |
|---|---:|---:|---:|
| unseen_short (80 tu) | 0.586 | 17.0 | 6/30 |
| unseen_long (280 tu) | 0.754 | 14.6 | 2/10 |
