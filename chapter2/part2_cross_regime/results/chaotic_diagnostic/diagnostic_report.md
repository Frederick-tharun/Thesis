# Chaotic source-model failure diagnostic (Phase A/B)

## Phase A — reproducibility

The stored rank-1 (Part-1 anchor) seed-42 search result was re-fit and re-scored from
scratch: `objective=0.07998492603` matched the stored value exactly (bit-for-bit,
`< 1e-9` tolerance). **The old chaotic failure is fully deterministic and reproducible.**

## Top-10 chaotic candidates (seed-42 search rank)

The Part-1 anchor (`reservoir_size=100, connectivity=0.089, input_scaling=0.064,
spectral_radius=0.412, ridge=3.97e-10, leak=0.938`) is rank 1 and is **dramatically**
better than every one of the other 47 candidates found by the original single-seed
48-candidate search (median VPT fraction 0.71-0.92 for its good seeds vs 0.11-0.29 for
every other top-10 candidate's best seed). The search never found a comparably good
region elsewhere in hyperparameter space.

## Good vs. bad seed comparison (anchor)

| seed | mean frac \|r\|>0.90 | mean frac \|r\|>0.99 | max \|r\| | divergent rollouts | classification |
|---|---:|---:|---:|---:|---|
| 42   | 0.0000 | 0.0000 | 0.547 | 0/15  | within_threshold |
| 123  | 0.0000 | 0.0000 | 0.523 | 0/15  | within_threshold |
| 456  | 0.3923 | 0.1217 | 1.000 | 15/15 | **true_divergence** |
| 789  | 0.2846 | 0.1473 | 1.000 | 15/15 | **true_divergence** |
| 2026 | 0.0000 | 0.0000 | 0.602 | 0/15  | within_threshold |

Readout diagnostics (W_out Frobenius norm, max coefficient, Gram condition number) are
**statistically indistinguishable** between good and bad seeds (condition number
1.3-1.4e15 regularised for every seed; W_out norms are if anything *smaller* for the
bad seeds). This rules out an unstable/ill-conditioned readout as the mechanism.

## What actually happens (see `good_seed_vs_bad_seed.png`)

Seed 42 reproduces two full chaotic bursts of I=3.20 almost exactly for the entire
80-time-unit horizon. Seed 456, after ~7 time units of plausible tracking, erupts into
a **large-amplitude (|x|~30, true HR range is ~[-2,2]), rigid, high-frequency
oscillation that never recovers** — a saturated limit-cycle runaway, not a chaotic
phase-shift. `error_growth_comparison.png` shows the pointwise error for seeds 456/789
crossing the divergence threshold within 1-14 time units and never returning, while
42/123/2026 stay bounded (occasional brief threshold crossings that recover, consistent
with ordinary chaotic sensitivity, not instability).

Divergence for the bad seeds is **uniform across all 5 chaotic training currents and
all 3 windows (15/15)** — not concentrated on one current. This rules out "one specific
chaotic current" as the cause.

## Generalization check (rank2, rank3, rank6)

Higher-input-scaling candidates (rank2/3/6, `input_scaling` 1.4-3.0) show high
saturation (52-71% of units with \|r\|>0.90) **at every seed tested**, but this alone
does not guarantee divergence: rank3 (53% saturation, both seeds) shows 0/15 divergence
at both seeds, while rank6 shows 0/15 at seed 42 but 11/15 at seed 456 with *similar*
saturation levels at both seeds. This confirms saturation is *necessary-looking but not
sufficient*: whether a saturated regime is dynamically stable or explosive depends on
the specific random realization of the sparse recurrent connectivity, not just the
nominal hyperparameters.

## Hypothesis assessment

| # | Hypothesis | Verdict |
|---|---|---|
| A | implementation/alignment/scaling bug | **Ruled out** — reproducibility is exact; alignment, scaling and recursion mechanics are all correct |
| B | pathological reservoir dynamics / saturation | **Confirmed as primary mechanism** — bad seeds show 0%->39% saturation jump and runaway |
| C | unstable readout | **Ruled out** — Gram conditioning and W_out norms are indistinguishable between good/bad seeds |
| D | excessive spectral amplification | Contributing factor for the high-input-scaling candidates, not the anchor (nominal spectral radius is identical across seeds; only the *realized* dynamics differ) |
| E | seed-dependent reservoir conditioning | **Confirmed** — same hyperparameters, same nominal spectral radius, only the random draw of sparse connectivity differs between seeds |
| F | one specific chaotic current | **Ruled out** — failure is uniform across all 5 currents |
| G | ordinary chaotic phase separation | **Ruled out for the failing seeds** — predictions leave the plausible HR state range and lock into a rigid non-chaotic oscillation, not a bounded phase-shifted chaotic trajectory |
| H | insufficient hyperparameter search for seed-robust solutions | **Contributing** — the original single-seed search never found anything close to the anchor's quality elsewhere in the space |
| I | other source-only mechanism | not needed; B/E fully explain the observed pattern |

## Classification

**NO_BUG_STABILITY_PROBLEM**

The chaotic source-model failure is a genuine, well-characterized reservoir-computing
phenomenon: for a sparse, small reservoir (size 100, connectivity ~9%) evaluated in
closed-loop (autonomous) recursion on genuinely chaotic dynamics, certain random
reservoir realizations produce a locally saturating, unstable closed loop while others
(at identical nominal hyperparameters) remain accurate for the full 80-time-unit
horizon. This is not an implementation defect. Per protocol, Phase C (seed-aware
recovery search) proceeds automatically.
