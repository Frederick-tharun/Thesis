# Cross-regime ESN improvement extension

This package implements the scenario-specific, stability-aware extension in
`IMPROVEMENT_PROTOCOL.md`. It reuses the existing Chapter 2 ESN, data integrity
checks, recursive forecast, corrected numerical classification, metrics, and
model bundle format.

```bash
python -m chapter2.cross_regime_improvement.experiment --pilot
sbatch run_chapter2_cross_regime_improvement_optimisation.slurm
sbatch run_chapter2_cross_regime_improvement_final.slurm
```

Production histories and final evaluation are resumable. The final job cannot
proceed until `results/optimisation/selection.json` exists. Expected products
are 15 models, 345 isolated evaluation records/raw arrays, the six-cell summary
table, continuous-schedule summaries, and fourteen figure groups in PNG/PDF.
