# Validated Chaotic-Control Results

## Provenance

- Source commit: `ae50ce2a434114902c72d0f76895fba9d73da0c1`
- HR regime: `chaotic_bursting`
- Input current: `I = 3.25`
- State: full `[x, y, z]`
- Samples: `150000`; sample interval: `0.01`
- Controller rollout: `45000` samples
- Control start: index `9000` of the rollout (`t = 1140.0`)
- ESN selector for the controller comparison: `dummy`
- ESN reservoir size: `527`; washout: `446`
- Controller-comparison ESN recursive x NRMSE: `0.2658309801`

The linear and finite-time archives contain identical ESN parameter and
prediction-metric files. The Pyragas run log records the same deterministic
optimizer sequence and prediction metrics.

## Final Parameters And Outcomes

| Controller | Objective | Parameters | Main result | Stable |
|---|---|---|---:|:---:|
| Linear feedback | Rest-state stabilization | `K=1.0` | state RMSE `0.0`; spike reduction `100%`; energy `4.1306577e-06` | Yes |
| Finite-time | Rest-state stabilization | `K=0.4582142857142857`, `s=0.8` | state RMSE `7.9564192e-04`; spike reduction `100%`; energy `7.1687762e-06` | Yes |
| Pyragas | Periodic-spiking stabilization | `K=0.8`, delay `2400`, sign `-1` | period `2400` steps (`24.0` time units); rhythm CV `0.0` | Yes |

Pyragas diagnostics:

- 12 evaluated peaks/cycles, distributed `[3, 3, 3, 3]` across four windows
- normalized recurrence error: `0.0029471549`
- recurrence correlation: `0.9999956646`
- normalized tail-closure error: `0.0026937966`
- x-amplitude ratio versus uncontrolled ESN: `0.9890851752`
- normalized tail feedback RMS: `0.0158581681`
- quality status: `PASS`, with no listed quality issues

## Evidence Archives

| Archive | SHA-256 |
|---|---|
| `final_K0p8_delay2400_sign_-1_1715049.zip` | `0897a8cb998697494a45506dfbb2c9877832ad96da310e7d8199a484e2d76943` |
| `final_linear_finite_1715156.zip` | `0c920b86646993844192a1af90b74ea528be6f139573f8582f483315816219db` |

Every file listed by each archive's internal `SHA256SUMS` manifest was checked
successfully. Key metrics were also independently recomputed from the archived
rollout CSV files.

## Claim Boundaries

- These experiments control the trained ESN digital twin, not the original HR
  differential equations directly.
- Linear and finite-time control target a rest state. Pyragas targets a
  sustained periodic orbit, so its rest-state RMSE and spike-reduction values
  are not success criteria.
- `K=1.0` makes the linear corrected input equal the target algebraically after
  control starts; its exact zero tracking error should be reported explicitly.
- Pyragas `PASS` is an empirical finite-horizon result under the implemented
  quality thresholds. It is not a mathematical proof of asymptotic stability.
- The Pyragas tail feedback is small on average but not identically zero; avoid
  claiming perfectly non-invasive control.
- The controller-comparison ESN is not the separate best prediction benchmark.
  Do not combine its `0.265831` x NRMSE with prediction results from a different
  ESN as though they came from the same model.
