# Thesis controller-sweep figures (run 1752614)

These figures were regenerated directly from the saved full controller-validation CSV sweeps. No ESN training, Bayesian optimization, controller rollout, or held-out controller-test evaluation was rerun.

Important interpretation:

- These are validation-sweep figures used to show how the controller gain K was selected.
- They are not held-out controller-test figures. Use the final controller-test metrics and rollout figures from FINAL_THESIS_RUN_1752614 for claims about final generalization/performance.
- The combined overview uses controller-specific selection scores. Score magnitudes are not comparable between regulation controllers and Pyragas control.
- “Empirical quiet-reference RMSE” is deliberate wording: the regulation target is data-derived and is not claimed to be an exact Hindmarsh–Rose equilibrium.
- The finite-time controller is the global piecewise law: fractional-power feedback for ||e|| <= 1 and a linear branch for ||e|| > 1.

See figure_manifest.json for exact source CSVs, point counts, selected gains, and commit provenance.
