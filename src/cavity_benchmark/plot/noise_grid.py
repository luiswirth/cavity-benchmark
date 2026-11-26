"""Reference error against assumed noise, one curve per N_s, for both geometries.

Reads out/epgp/noise-grid/{geometry}/ln{log_noise}/T_ns{ns}_nb{nb}.npy and writes
the figure and a CSV of every point. A failed solve (non-finite operator) is
left out of the plot and kept in the CSV as nan.
"""

import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cavity_maxwellgp import load_config

from ..benchmark import config_path, reference_operator
from .common import FIGS, setup_style

_LN = re.compile(r"ln(-?\d+(?:\.\d+)?)$")
_T = re.compile(r"T_ns(\d+)_nb(\d+)\.npy$")
LABEL = {"sphere": "sphere, against analytic", "ellipse": "ellipsoid, against BEM"}


def noise_grid_runs(geometry):
    k, _, pts, e1, e2 = load_config(config_path(geometry))
    T_ref = reference_operator(geometry, k, pts, e1, e2)
    nref = np.linalg.norm(T_ref)
    rows = []
    for d in glob.glob(os.path.join("out", "epgp", "noise-grid", geometry, "ln*")):
        ln = float(_LN.search(d).group(1))
        for path in glob.glob(os.path.join(d, "T_ns*_nb*.npy")):
            ns, nb = map(int, _T.search(path).groups())
            T = np.load(path)
            err = np.linalg.norm(T - T_ref) / nref if np.all(np.isfinite(T)) else np.nan
            rows.append((ns, nb, ln, float(err)))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometries", nargs="+", default=["sphere", "ellipse"])
    ap.add_argument("--thesis-log-noise", type=float, default=-12.0)
    ap.add_argument("--out", default=os.path.join(FIGS, "noise_grid.svg"))
    args = ap.parse_args()

    setup_style()
    fig, axes = plt.subplots(1, len(args.geometries), figsize=(6.0 * len(args.geometries), 4.6),
                             sharey=True, squeeze=False)
    os.makedirs(FIGS, exist_ok=True)
    table = os.path.splitext(args.out)[0] + ".csv"
    with open(table, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["geometry", "n_spectral", "n_boundary", "log_noise", "sigma_n", "err"])
        for ax, geometry in zip(axes[0], args.geometries):
            rows = noise_grid_runs(geometry)
            if not rows:
                raise SystemExit(f"no runs in out/epgp/noise-grid/{geometry}")
            for ns, nb, ln, err in rows:
                w.writerow([geometry, ns, nb, ln, f"{np.exp(ln / 2):.6e}", f"{err:.6e}"])
            cmap = plt.get_cmap("viridis")
            series = sorted({(ns, nb) for ns, nb, _, _ in rows})
            for i, (ns, nb) in enumerate(series):
                pts = [(np.exp(ln / 2), e) for n, b, ln, e in rows
                       if (n, b) == (ns, nb) and np.isfinite(e)]
                xs, ys = zip(*sorted(pts))
                ax.loglog(xs, ys, "o-", color=cmap(i / max(len(series) - 1, 1)),
                          label=fr"$N_s = {ns}$, $N_b = {nb}$")
            ax.axvline(np.exp(args.thesis_log_noise / 2), color="0.5", ls="--", lw=1.2)
            ax.set_xlabel(r"assumed noise level $\sigma_n$")
            ax.set_title(LABEL[geometry])
        axes[0][0].set_ylabel(r"reference error $\epsilon$")
        axes[0][0].legend()

    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out} and {table}")


if __name__ == "__main__":
    main()
