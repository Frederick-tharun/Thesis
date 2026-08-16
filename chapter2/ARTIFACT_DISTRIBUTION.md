# Chapter 2 artifact distribution

## Current inventory

The pre-cleanup Git status contained exactly 288 modified/untracked paths. They
were inspected and classified as follows:

| Category | Paths | Policy |
|---|---:|---|
| Source code | 9 | ordinary Git |
| Tests | 7 | ordinary Git |
| Documentation | 2 | ordinary Git |
| Launch scripts | 5 | ordinary Git |
| Small reproducibility artifacts | 7 | ordinary Git |
| Final summaries, figures, and manifests | 38 | ordinary Git |
| Large binary scientific artifacts newly visible in status | 220 | GitHub Release; do not stage |
| Caches, logs, or clutter | 0 | ignore if later created |
| Unrelated Chapter 1 or user changes | 0 | do not stage as Chapter 2 work |

The 220 large binaries newly visible in that status are 210 raw rollout NPZ
files and ten final model NPZ bundles. Six frozen dataset NPZ files were
already tracked and therefore were not part of the 288-path status baseline.
The complete external binary package contains all 226 files. The
release-hardening work adds source, tests, documentation, provenance, and
release metadata after the original baseline.

Current disk measurements are:

| Artifact group | Files | Filesystem allocation | Apparent bytes |
|---|---:|---:|---:|
| Raw rollout arrays | 210 | approximately 682 MiB | 355,396,721 bytes (338.9 MiB) |
| Final models including manifest | 11 | approximately 7.3 MiB | approximately 3.5 MiB |
| Step 7 histories/selections | 6 | approximately 22 MiB | approximately 11 MiB |
| Frozen input/output tree | 15 | approximately 46 MiB | approximately 23 MiB |
| Official thesis figures | 9 | approximately 2.6 MiB | approximately 1.2 MiB |
| Final tables | 5 | approximately 37 KiB | approximately 9.4 KiB |
| Correction archive | 11 | approximately 5.6 MiB | approximately 2.7 MiB |
| Entire Chapter 2 tree | 312 before new release/docs | approximately 769 MiB | varies after hardening |

The larger allocated size reflects this HPC filesystem's accounting; the
apparent byte count is the transferable file content.

## Distribution policy

Keep source, tests, Markdown, SLURM launchers, strict JSON/CSV summaries,
manifests, and the eight official thesis figure files in ordinary Git. Their
small size makes review and integrity verification practical.

Do not add the 210 raw rollout NPZ files or ten model bundles to ordinary Git
history. The selected distribution is a GitHub Release named
`chapter2-v1.0.0`, not Git LFS. Its archive also carries the six frozen input
datasets so every scientific binary can be obtained and checked as one
versioned asset. The archive has been prepared locally but has not been
uploaded; release creation remains a separately approved publication step.

The raw-result JSON records a SHA-256 for each rollout NPZ, the model manifest
records all ten model hashes, the frozen protocol records dataset hashes, and
the release manifest records official figure/table and reference hashes.
These manifests stay in ordinary Git even when payloads are external.

## Prepared GitHub Release assets

The persistent preparation directory, outside this Git repository, is:

`/home/hpc/rlvl/rlvl177v/chapter2_release_assets/chapter2-v1.0.0`

The release payload is one deterministic, uncompressed TAR. NPZ payloads are
already compressed, so an additional compression layer would add cost without
meaningfully reducing the upload. The asset is below GitHub's 2 GiB per-file
limit.

| Release archive | Size (bytes) | SHA-256 |
|---|---:|---|
| `chapter2-v1.0.0-scientific-binaries.tar` | 383,119,360 | `2f8319624ff41f7890044943743d288c031eccba1183efbb5191803304be4a36` |

The TAR contains all 226 scientific binaries at their repository-relative
paths: 210 raw rollouts, ten model bundles, and six frozen datasets, totaling
381,534,011 bytes. It also contains the existing raw-results, model,
selection-lock, and diagnostic manifests; `README.md`;
`chapter2-v1.0.0-scientific-binaries.json`; and
`chapter2-v1.0.0-scientific-binaries.sha256`.

Prepared sidecars are:

| Sidecar | Size (bytes) | SHA-256 |
|---|---:|---|
| `chapter2-v1.0.0-archives.sha256` | 106 | `d087433d668a3dca9c71b7d809c86e07b5de91711e9bc690fcc4157f77f60445` |
| `chapter2-v1.0.0-scientific-binaries.json` | 60,199 | `4b8562f1cb2c2d64319662647eea6e75153942fd343aeabba9f37b2f243c6ac6` |
| `chapter2-v1.0.0-scientific-binaries.sha256` | 35,186 | `1781ffda8bf1089b4518f8e35427c609612aef553981bcc6013b2291b5560f47` |
| `README.md` | 683 | `26c3e995256a43f6f9be92035b4e03219e84133e4a53b7b9a736c9b481f1b01f` |
| `package_report.json` | 909 | `463acca8ae5ef5a46baf29f43c9a631b59d28a51d716ac4e22e452dac5695ca1` |

No download URL is recorded because the GitHub Release does not yet exist.

## Safe publication and retrieval workflow

1. Commit and push the selectively staged repository files only after review.
2. Create the `chapter2-v1.0.0` GitHub Release only after explicit approval.
3. Upload the TAR and checksum/manifest sidecars from the persistent external
   preparation directory; do not upload `package_report.json` unless desired
   as an additional machine-readable packaging log.
4. Verify the uploaded asset against `chapter2-v1.0.0-archives.sha256`.
5. In a clean checkout, extract the TAR at the repository root so its preserved
   paths restore the binary payloads.
6. Run `sha256sum -c chapter2-v1.0.0-scientific-binaries.sha256`, followed by
   `python3 -m chapter2.verify_release`.
7. Add the real immutable release URL to documentation only after publication.

Do not broadly ignore `*.npz`; the scientific binaries must remain visible in
`git status` until the distribution decision is complete. Repository
ZIP/TAR exports are local transport products and must not be committed.
