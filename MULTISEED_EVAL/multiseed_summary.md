# Chapter 1 multiseed evaluation

Seeds evaluated: 42, 123, 456, 789, 2026

## Prediction

| Metric | Mean | Std | Median | Min | Max | n |
|---|---:|---:|---:|---:|---:|---:|
| rmse_recursive_x | 0.00113242 | 0.000439613 | 0.00112259 | 0.000552447 | 0.0017368 | 5 |
| nrmse_recursive_x | 0.00229513 | 0.000890988 | 0.00227522 | 0.00111968 | 0.00352006 | 5 |
| rmse_recursive_all_states | 0.00156634 | 0.000607848 | 0.00154171 | 0.000772024 | 0.00240705 | 5 |
| nrmse_recursive_all_states | 0.00154472 | 0.000599842 | 0.00152762 | 0.000755556 | 0.00237103 | 5 |

## Controller success

| Controller | Successful | Attempted | Success rate |
|---|---:|---:|---:|
| finite_time | 5 | 5 | 100.0% |
| linear_feedback | 5 | 5 | 100.0% |
| pyragas | 5 | 5 | 100.0% |

## Representative seed

Representative seed: **42** using the median prediction-NRMSE rule.
