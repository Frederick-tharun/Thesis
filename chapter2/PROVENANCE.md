# Chapter 2 provenance

## Experimental lineage

Step 7 used only fitting and validation transitions at the three training
currents. It selected the parameter-aware source trial 4 and ordinary-baseline
source trial 12 using validation results only. The selection is labelled
`VALIDATION-SELECTED — BENCHMARKS NOT OPENED`.

Before Step 8 accessed any held-out, unseen-current, or continuous numerical
values, it wrote the frozen selection lock and evaluation manifest and saved all
ten final model bundles. Step 8 then performed the one final held-out
evaluation. The frozen release contains 210 benchmark records: 105 records for
each model class across five reservoir seeds. It also contains ten final models,
one for each model class/seed combination.

## Post-benchmark correction

A later audit found six continuous-current interval subrecords whose event
metrics remained marked as defined even though the corresponding interval was
divergent. The deterministic correction changed only those event-validity
metadata fields. It did not change forecasts, target arrays, state-error
metrics, model bundles, scalers, optimisation histories, selections,
aggregates, seeds, or the scientific conclusion. The correction manifest and
pre/post snapshots are preserved under
`chapter2/final_results/post_benchmark_event_correction/`.

## Step 8 source identity

The selection lock records the Step 8 execution source as:

`2202e02d9d54dcca38a56b775b5bbc8a533680e9a5979feb85761a18c264880b`

A read-only recovery search covered all reachable Git refs, reflog-visible
objects, 23,571 plausible reachable/unreachable Git blobs, repository backup
and archive candidates, 676 small repository files, 125 VS Code remote-history
entries, similarly named project files, and available Thesis-associated
snapshot locations. No exact match was found; the checked HPC snapshot
locations were absent. The source must therefore not be reconstructed or
described as recovered. Only its hash and the behavior evidenced by frozen
artifacts survive.

The corrected source immediately before submission hardening had SHA-256:

`847d5ba8828973503a01aebf53e52b8be735ed5bedea8dd197ad4e82ae2d7999`

Its exact bytes are preserved, without a `.py` suffix and never to be
executed, at
`chapter2/provenance/esn_step8_post_correction_pre_hardening_sha256_847d5ba8828973503a01aebf53e52b8be735ed5bedea8dd197ad4e82ae2d7999.py.snapshot`.

The current `esn_step8.py` is a post-evaluation release source. Its added
fail-closed resume checks were not used to train a model or produce any
published prediction, metric, table, or figure. Its exact release hash is
recorded in `chapter2/release/release_manifest.json`.

## Audit and figure history

The saved
`chapter2/final_results/step8_seed_stability_audit.json` is a protected
historical artifact. It was produced by an earlier audit-source state that
recorded 19 figure artifacts under the now-removed
`final_results/figures_final/` directory. The current audit source separates
figure generation and records no figure artifacts, so it cannot be claimed to
be the exact producer of that saved audit. Submission hardening only made
future derived JSON/CSV audit writes atomic; it did not run the audit or rewrite
the saved audit. Any future audit must use a new versioned output filename and
record the producing `audit_step8.py` SHA-256.

The Step 8 verification also contains historical references to 20 files under
the superseded `final_results/figures/` directory. Those missing legacy files
and the audit's missing `figures_final/` files are deliberately not restored,
and their protected JSON references are not rewritten.

The only official thesis figures are the four PDF/PNG pairs plus manifest under
`chapter2/final_results/figures_thesis/`. Their release hashes are enumerated
in the new release manifest.
