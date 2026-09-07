# Parameter-transfer follow-up

Read PARAMETER_TRANSFER_PROTOCOL.md first. This changes preprocessing,
validation and capacity search, not the existing EchoStateNetwork architecture.
Generated output belongs only under this package's results directory.

This implementation stage runs tests and a tiny synthetic pilot only:

    python -m chapter2.cross_regime_parameter_transfer.experiment --pilot

Future authorised Slurm stages: --optimise-all --resume, then --final --resume.
Do not submit production before review and a subsequent instruction.
Two scenario selections and ten final models must lock before fresh target access.
No five-seed stable candidate means no selection lock and no final run.

The fresh protocol reuses the repository HR alternative state (0.1,0,0), not
that old configuration's current. Different deterministic initial conditions
do not guarantee statistical independence or distinct attractors. Tests and
pilot generate no fresh benchmark trajectories.

Resume verifies sources, design, software, thread settings and GP ask/tell replay.
Concurrent stage writers are blocked. Orphan binary files or stale locks require
inspection, never automatic deletion/overwrite. Final evaluation has 150 short
and 50 long records, RR/RC/CC/CR only, and thirteen PNG/PDF figure groups.

The historical protection inventory stores hashes, not parsed result values.
Selection I/O blocks historical/final paths and opposite-regime datasets.
This is an accidental-access guard, not a security sandbox. Climate diagnostics
remain supplementary. Historical comparisons involve different trajectories
and are descriptive, not paired tests.
