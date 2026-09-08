# Chapter 2 cross-regime thesis summary

This folder is the presentation-only summary of the completed Chapter 2 cross-regime ESN evaluation. It answers two questions: whether a regular-trained ESN predicts chaotic targets (RC), and whether a chaotic-trained ESN predicts regular targets (CR). RR and CC are the within-regime controls.

No model was trained, optimised, or evaluated to create this package. The script only validates and plots previously saved final-evaluation records and arrays.

## Open these four figures

1. `figures/01_cross_regime_task_setup.{png,pdf}` — task and current groups.
2. `figures/02_regular_trained_results.{png,pdf}` — representative RR and RC trajectories.
3. `figures/03_chaotic_trained_results.{png,pdf}` — representative CC and CR trajectories.
4. `figures/04_cross_regime_summary_metrics.{png,pdf}` — four-direction metric summary.

The concise numeric table is available as `tables/01_cross_regime_summary.csv` and `tables/01_cross_regime_summary.md`.

## Frozen sources used

Main results and validation:

- `chapter2/cross_regime_improvement/results/evaluation_records.json` — complete 345-record evaluation; the four summary cells use its 225 fixed-short records.
- `chapter2/cross_regime_improvement/results/aggregate_results.json` — stored RR/RC/CC/CR aggregate values checked against values recomputed from the records.
- `chapter2/cross_regime_improvement/results/train_test_summary.csv` — tested-current membership and per-current record-count cross-check.
- The four selected NPZ files listed below — saved time, target, and prediction arrays. Each is checked against the SHA-256 stored in its evaluation record.

Visual reference only:

- `chapter2/final_results/figures_thesis/` — earlier Chapter 2 thesis figures used as inspiration for typography, trajectory colours, light backgrounds, and uncluttered layouts.

Backup follow-up sources:

- `chapter2/cross_regime_parameter_transfer/results/optimisation/regular_parameter_transfer_history.json`
- `chapter2/cross_regime_parameter_transfer/results/optimisation/chaotic_parameter_transfer_history.json`
- `chapter2/cross_regime_parameter_transfer/results/pilot_report.json`

## Deterministic representative selection

Selection is restricted to the same `fixed_short` final-evaluation records used by the stored RR/RC/CC/CR matrix. For each direction, the script computes the direction-wide median aggregate NRMSE and selects the record with the smallest absolute distance from that median. An exact-distance tie is broken by lexicographically sorted record ID.

This equals the literal median record for the 45-record RR and CR cells. The 30-record RC and CC cells have arithmetic medians between two observations, so nearest-to-median is the documented deterministic equivalent.

| Direction | Selected record | Current | Seed | Window | Record NRMSE | Record VPT | Diverged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RR | `fixed_short__regular_to_chaotic__seed_2026__I_3p50__window_1` | 3.50 | 2026 | 1 | 0.00373481625442 | 80.00 | No |
| RC | `fixed_short__regular_to_chaotic__seed_123__I_3p20__window_2` | 3.20 | 123 | 2 | 1.02343764366 | 38.61 | No |
| CC | `fixed_short__chaotic_to_regular__seed_42__I_3p20__window_3` | 3.20 | 42 | 3 | 0.00452811502270 | 80.00 | No |
| CR | `fixed_short__chaotic_to_regular__seed_789__I_1p67__window_3` | 1.67 | 789 | 3 | 5.38750923456 | 0.01 | Yes |

The trajectory figures display the selected record's metrics. The summary figure and tables display direction-wide medians/rates, so the RC representative VPT, for example, is not the RC group median VPT.

Selected arrays:

- `chapter2/cross_regime_improvement/results/raw_arrays/fixed_short__regular_to_chaotic__seed_2026__I_3p50__window_1.npz`
- `chapter2/cross_regime_improvement/results/raw_arrays/fixed_short__regular_to_chaotic__seed_123__I_3p20__window_2.npz`
- `chapter2/cross_regime_improvement/results/raw_arrays/fixed_short__chaotic_to_regular__seed_42__I_3p20__window_3.npz`
- `chapter2/cross_regime_improvement/results/raw_arrays/fixed_short__chaotic_to_regular__seed_789__I_1p67__window_3.npz`

## Main quantitative result

All values below aggregate the fixed-short final-evaluation rollouts. IQR is Q3 minus Q1. The maximum fixed-short forecast horizon is 80 time units.

| Direction | Train → test | Records | Median NRMSE | IQR NRMSE | Median VPT | Divergence | Numerical failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RR | Regular → Regular | 45 | 0.00373481625442 | 0.0113255563716 | 80.00 | 0.0% | 0.0% |
| RC | Regular → Chaotic | 30 | 1.02536558582 | 0.449377637789 | 12.50 | 0.0% | 0.0% |
| CC | Chaotic → Chaotic | 30 | 0.00388366155871 | 0.00501994093672 | 80.00 | 0.0% | 0.0% |
| CR | Chaotic → Regular | 45 | 5.38750923456 | 13.2526017653 | 0.09 | 80.0% | 0.0% |

The direct conclusion is that both within-regime controls work well. Regular-to-chaotic transfer is substantially worse and useful only over a shorter horizon, while chaotic-to-regular transfer is the clear failure case.

## What is intentionally omitted

Mixed-shuffled results, continuous schedules, optimisation histories, heatmaps, numerical-failure plots, divergence heatmaps, phase portraits, and audit/provenance plots are omitted from the four-figure story for clarity. They remain intact in their historical result locations; nothing was deleted.

The parameter-transfer follow-up is also backup context rather than a main figure. It used fixed physical current scaling, leave-one-current-out validation, reservoir sizes through 800, and five-seed confirmation. Regular-current transfer produced one stable but only partially accurate candidate; chaotic-current transfer produced no five-seed-stable candidate. Because both scenario selections were required, no selection lock or fresh benchmark was opened.

## Rebuild or validate

From the repository root:

```bash
python3 chapter2/thesis_cross_regime_summary/scripts/build_cross_regime_summary_figures.py
```

For read-only source and selection checks without rewriting presentation outputs:

```bash
python3 chapter2/thesis_cross_regime_summary/scripts/build_cross_regime_summary_figures.py --check-only
```

The builder imports no training, optimisation, model, or evaluation module. It validates source schemas, record counts, recomputed aggregates, tested-current membership, and selected-array hashes before creating presentation files.
