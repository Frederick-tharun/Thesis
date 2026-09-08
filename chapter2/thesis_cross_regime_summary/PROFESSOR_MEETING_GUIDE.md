# Professor meeting guide

## Open in this order

1. `figures/01_cross_regime_task_setup.pdf` — define the regular and chaotic currents and the two transfer questions.
2. `figures/02_regular_trained_results.pdf` — show RR as the control, then RC as the first transfer direction.
3. `figures/03_chaotic_trained_results.pdf` — show CC as the control, then the strong CR failure.
4. `figures/04_cross_regime_summary_metrics.pdf` — close with the four-direction quantitative comparison.
5. Open `tables/01_cross_regime_summary.md` only if exact values are requested.

## 60–90 second pitch

“This is the clean cross-regime test. The regular group contains currents 1.67, 3.29, and 3.50; the chaotic group contains 3.20 and 3.34. I trained nothing new for these slides—these figures only re-present the completed final evaluation. The within-regime controls are strong: RR and CC both have median NRMSE around 0.004, reach the full 80-time-unit median prediction horizon, and show no divergence. Transfer across regimes is much weaker. A regular-trained ESN tested on chaotic currents has median NRMSE 1.03 and median valid prediction time 12.5, so it captures only a limited initial horizon before losing the trajectory. The reverse direction is much worse: chaotic-to-regular has median NRMSE 5.39, median valid prediction time 0.09, and 80 percent divergence. The conclusion is therefore asymmetric but clear: training within the target regime works, regular-to-chaotic transfer is partial and short-lived, and chaotic-to-regular transfer fails.”

## Backup: parameter-transfer follow-up

The follow-up kept the same ESN architecture but made the transfer test stricter: physical current scaling was fixed rather than fitted, validation left out one current at a time, reservoir sizes up to 800 were searched, and the top candidates were checked over five seeds. Regular-current transfer produced one stability-eligible 800-unit candidate, but its aggregate validation quality was only partial (median NRMSE 0.667 and median VPT 3.24). All five chaotic-transfer candidates failed the zero-divergence five-seed eligibility rule. Since both scenarios had to pass before locking a selection, no selection was locked and no fresh benchmark was generated or opened.

## Likely questions

### Why these particular trajectories?

They were selected deterministically, not by visual appeal: within each fixed-short direction, choose the record nearest the direction's median aggregate NRMSE; break exact-distance ties by sorted record ID. The README records the exact current, seed, window, and record ID.

### Why use only the fixed-short results for the main summary?

They are the comparable three-window, five-seed records used by the completed RR/RC/CC/CR train–test matrix. Each rollout has the same 80-time-unit maximum forecast horizon. Long and continuous records answer different questions and would blur this specific comparison.

### How can CR have 80% divergence but 0% numerical failure?

Divergence means the normalized-error threshold was reached; numerical failure means non-finite or otherwise unusable numerical output. A prediction can remain finite while being dynamically very wrong, which is what occurs in CR.

### Why is NRMSE shown on a log scale?

The medians span more than three orders of magnitude, from about 0.004 to 5.39. A direct axis would make both successful within-regime bars nearly invisible. The panel title and y-axis explicitly mark the log scale, and every bar is numerically annotated.

### Were mixed-shuffled or follow-up results removed because they were unfavorable?

No. They remain untouched in the historical result folders. Mixed-shuffled, continuous-schedule, heatmap, and follow-up material is omitted only to keep the professor-facing story focused on the two requested transfer directions and their within-regime controls.
