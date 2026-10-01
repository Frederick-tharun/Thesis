# Chapter 1 multiseed evaluation

Seeds evaluated: 42, 123, 456, 789, 2026

## Prediction

| Metric | Mean | Std | Median | Min | Max | n |
|---|---:|---:|---:|---:|---:|---:|
| rmse_recursive_x | 0.189782 | 0.0669054 | 0.205541 | 0.0811625 | 0.254968 | 5 |
| nrmse_recursive_x | 0.392047 | 0.138211 | 0.424601 | 0.167663 | 0.526705 | 5 |
| rmse_recursive_all_states | 0.257349 | 0.0894912 | 0.276408 | 0.113293 | 0.347309 | 5 |
| nrmse_recursive_all_states | 0.273657 | 0.0965317 | 0.295316 | 0.117741 | 0.369353 | 5 |

## Controller success

| Controller | Successful | Attempted | Success rate |
|---|---:|---:|---:|
| finite_time | 5 | 5 | 100.0% |
| linear_feedback | 5 | 5 | 100.0% |
| pyragas | 5 | 5 | 100.0% |

## Representative seed

Representative seed: **456** using the median prediction-NRMSE rule.
