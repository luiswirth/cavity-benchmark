# cavity-benchmark

Local-only cross-validation harness for the cavity reaction-operator benchmark.
Generates the shared config both solvers read, owns the analytic sphere
reference, and aggregates + plots the operators the two solvers produce. It does
not solve or run on Euler -- that is `cavity-epgp` and `cavity-bem`. uv-managed;
always `uv run`. Depends on `cavity-epgp` for the shared dipole physics.

## Entry points

- `gen-config` -- write the shared config (Lambda points, tangent frames, k,
  semi-axes) for a geometry. Single source of truth for both solvers.
- `aggregate` -- read the solver operators + manifests and compute `results.csv`
  (recip, self-convergence, err vs reference, cond, secs, mem). Ellipse is
  cross-validated against the BEM operator named by `BEM_REFERENCE`
  (benchmark.py); sphere against the exact analytic operator (`sphere.py`).
- `make-figures` -- aggregate + all convergence/rate/field figures.

## Two geometries

- `ellipse` (4,4,6): no closed form; EP-GP cross-validated against the BEM
  reference (`cavity-bem`).
- `sphere` (4,4,4): exact analytic reference (`sphere.py`); EP-GP measured
  directly against it.

## Data flow

`gen-config` -> shared config copied into `cavity-epgp/res` and `cavity-bem/res`
-> each solver computes its grid on Euler -> `out/{shape}/` -> pulled here by
copying each solver's `out/` into `out/{solver}/` (so `out/{epgp,bem}/{shape}/`) ->
pulled back here -> `aggregate` cross-validates -> `make-figures` -> figures/CSVs
copied into `epgp-thesis/res/` (manual; thesis tables/figures read them there).
