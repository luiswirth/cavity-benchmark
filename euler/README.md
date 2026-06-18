# Running on Euler

EP-GP convergence grids on ETH Euler, cross-validated against the BEM reference
(`cavity-bem/euler`). Both run on 16 cores of an EPYC_7742 node.

Jobs run under the `ls_math` shareholder account (`#SBATCH --account=ls_math`).
Omitting it falls back to the `public` guest account, capped at 48 cores /
128 GiB.

## Setup (login node, once)

SSH-agent forwarding must reach GitHub (for cloning this repo and the
`maxwellgp` dependency): check with `ssh -T git@github.com`.

    git clone git@github.com:luiswirth/cavity-dipoles.git   # or git pull
    curl -LsSf https://astral.sh/uv/install.sh | sh         # if uv is missing
    cd ~/cavity-dipoles
    uv sync

For the GPU backend (optional, throughput only): `uv sync --extra gpu` on a GPU
node and submit to a GPU partition; JAX uses the GPU automatically. The
BEM-matched fairness comparison stays on CPU (Bembel is CPU-only).

## Run

One task per grid point (job-level parallelism). EP-GP does not scale past
~8-16 cores, so this, not more cores per task, is what makes a study finish fast.
`run.sbatch` runs any `--index`-aware entry point; each task writes a per-point
fragment. Submit the array, then merge the fragments locally with `--collect`
(done alongside aggregate, which also needs the BEM reference). This mirrors the
BEM side: submit the array, pull results, post-process locally.

    sbatch --array=0-59 euler/run.sbatch epgp-convergence --geometry ellipse
    sbatch --array=0-59 euler/run.sbatch epgp-convergence --geometry sphere
    # BEM side: sbatch --array=1-20 euler/run.sbatch    (in cavity-bem)

The grid records accuracy and memory; both are contention-independent (peak RSS
is per-process, and accuracy doesn't depend on the node), so the grid runs on
shared nodes. The reference (the high corner for now, or a separate finer run)
is set by BEM_REFERENCE in benchmark.py and used by aggregate.

Threads are pinned to the allocation via `srun --cpu-bind=cores` (else JAX
oversubscribes the node's full core count).

Results land in `out/<geom>_epgp/` (per-grid-point operators T_epgp_*.npy +
`manifest.csv` + `provenance.json`). The manifest records only what cannot be
reconstructed from the saved operators: dofs, secs, cond, maxrss (log_noise is a
fixed input -> provenance.json, not a per-row value).
Derived quantities (norm, recip, selfconv, err) are computed in post-processing
by results.aggregate from the saved T's. Pull the directory back; aggregation
(which also needs the BEM reference) is a local step.
