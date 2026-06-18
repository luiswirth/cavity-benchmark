import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from ..benchmark import BEM_REFERENCE
from .common import FIGS, save, setup_style

ELLIPSE_BEM = os.path.join("out", "bem", "ellipse")
ELLIPSE_EPGP = os.path.join("out", "epgp", "ellipse")
SPHERE_EPGP = os.path.join("out", "epgp", "sphere")

# Reference-error marker style, and the reciprocity-error axis label.
ERR = dict(color="#d62728", marker="D", ms=8)
L_RHO = r"$\rho = \|\mathbf{T}-\mathbf{T}^{\!\top}\|/\|\mathbf{T}\|$"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _grid(ax):
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.margins(x=0.04, y=0.08)


def _epgp_conv_fig(rows, savename, fmt):
    """Single-panel reciprocity error vs n_spectral, one line per n_boundary.
    n_boundary is a sufficiency parameter, not a refinement axis, so it appears
    only as a family of lines showing that the result is stable once n_bnd is
    large enough."""
    nbs = sorted({int(r["n_boundary"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")

    for i, nb in enumerate(nbs):
        rs = sorted((r for r in rows if int(r["n_boundary"]) == nb and float(r["recip"]) > 0),
                    key=lambda r: int(r["n_spectral"]))
        if len(rs) < 2:
            continue
        ax.loglog([int(r["n_spectral"]) for r in rs], [float(r["recip"]) for r in rs],
                  "D-", color=cmap(i / max(len(nbs) - 1, 1)), mec="white", mew=0.8,
                  label=fr"$n_\mathrm{{bnd}}={nb}$")
    ax.set_xlabel(r"$n_\mathrm{spec}$"); ax.set_ylabel(L_RHO)
    ax.legend(frameon=False, ncol=2); _grid(ax)
    save(fig, savename, fmt)


def fig_sphere_epgp_convergence(fmt="svg"):
    path = os.path.join(SPHERE_EPGP, "results.csv")
    if not os.path.exists(path):
        return
    _epgp_conv_fig(read_csv(path), "epgp_sphere_convergence", fmt)


def fig_ellipse_epgp_convergence(fmt="svg"):
    path = os.path.join(ELLIPSE_EPGP, "results.csv")
    if not os.path.exists(path):
        return
    _epgp_conv_fig(read_csv(path), "epgp_ellipse_convergence", fmt)


def fig_ellipse_bem_convergence(bem, fmt="svg"):
    # BEM is a two-parameter (p, m) study, fundamentally unlike the EPGP sweeps,
    # so it keeps its own h- and p-refinement panels rather than the shared
    # single-axis template. Reciprocity rho is the per-run quantity; delta is in
    # the table.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    cmap = plt.get_cmap("viridis")

    ms_all = sorted({r["m"] for r in bem})
    ps_all = sorted({r["p"] for r in bem})

    for i, p in enumerate(ps_all):
        rows = sorted((r for r in bem if r["p"] == p and r["recip"] > 0),
                      key=lambda r: r["m"])
        if len(rows) < 2:
            continue
        ax[0].semilogy([r["m"] for r in rows], [r["recip"] for r in rows],
                       "D-", color=cmap(i / max(len(ps_all) - 1, 1)),
                       mec="white", mew=0.8, label=f"$p={p}$")
    ax[0].set_xlabel(r"mesh level $m$"); ax[0].set_ylabel(L_RHO)
    ax[0].set_title(r"$h$-refinement"); ax[0].legend(frameon=False, ncol=2)
    ax[0].set_xticks(ms_all)
    _grid(ax[0])

    for i, m in enumerate(ms_all):
        rows = sorted((r for r in bem if r["m"] == m and r["recip"] > 0),
                      key=lambda r: r["p"])
        if len(rows) < 2:
            continue
        ax[1].semilogy([r["p"] for r in rows], [r["recip"] for r in rows],
                       "D-", color=cmap(i / max(len(ms_all) - 1, 1)),
                       mec="white", mew=0.8, label=f"$m={m}$")
    ax[1].set_xlabel(r"polynomial degree $p$"); ax[1].set_ylabel(L_RHO)
    ax[1].set_title(r"$p$-refinement"); ax[1].legend(frameon=False)
    ax[1].set_xticks(ps_all)
    _grid(ax[1])

    save(fig, "bem_ellipse_convergence", fmt)


def _rate_fig(path, errcol, label, savename, fmt):
    """Convergence-rate view: the reference error on a log axis against
    sqrt(n_spec) on a linear axis. Root-exponential convergence,
    error ~ exp(-c sqrt(n_spec)), is a straight line here. The plateau and the
    numerical floor are masked so only the spectral descent is shown, and a
    least-squares guide line is overlaid to judge linearity."""
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    # The grid has many n_boundary; the spectral-rate view uses the best
    # (highest n_boundary) slice so the boundary error doesn't mask the spectrum.
    nbmax = max(int(r["n_boundary"]) for r in rows)
    rows = sorted((r for r in rows if int(r["n_boundary"]) == nbmax),
                  key=lambda r: int(r["n_spectral"]))
    ns = np.array([int(r["n_spectral"]) for r in rows], float)
    err = np.array([float(r[errcol]) for r in rows], float)

    floor = err[err > 0].min()
    mask = (err > 3 * floor) & (err < 0.5)          # spectral descent only
    x, y = np.sqrt(ns[mask]), err[mask]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.semilogy(x, y, ERR["marker"] + "-", color=ERR["color"], mec="white",
                mew=1.0, markersize=ERR["ms"], label=label)
    if len(x) >= 2:
        a, b = np.polyfit(x, np.log10(y), 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.semilogy(xx, 10 ** (a * xx + b), "--", color="0.4", lw=1.4,
                    label="root-exponential fit")
    ax.set_xlabel(r"$\sqrt{n_\mathrm{spec}}$")
    ax.set_ylabel("relative error")
    ax.legend(frameon=False)
    _grid(ax)
    save(fig, savename, fmt)


def fig_sphere_epgp_rate(fmt="svg"):
    _rate_fig(os.path.join(SPHERE_EPGP, "results.csv"), "err_vs_analytic",
              r"$\varepsilon$", "epgp_sphere_rate", fmt)


def fig_ellipse_epgp_rate(fmt="svg"):
    _rate_fig(os.path.join(ELLIPSE_EPGP, "results.csv"), "err_vs_bem_ref",
              r"$\varepsilon$", "epgp_ellipse_rate", fmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["svg", "png"], default="svg")
    fmt = ap.parse_args().format

    setup_style()
    os.makedirs(FIGS, exist_ok=True)
    bem = [{"p": int(r["p"]), "m": int(r["m"]), "dofs": int(r["dofs"]),
            "recip": float(r["recip"]), "selfconv_vs_ref": float(r["selfconv_vs_ref"])}
           for r in read_csv(os.path.join(ELLIPSE_BEM, "results.csv"))]

    fig_sphere_epgp_convergence(fmt)
    fig_ellipse_epgp_convergence(fmt)
    fig_ellipse_bem_convergence(bem, fmt)
    fig_sphere_epgp_rate(fmt)
    fig_ellipse_epgp_rate(fmt)
    stale = ("h_convergence", "p_convergence", "reciprocity", "svd_spectrum",
             "bem_validity", "bem_self_convergence", "preview", "all_preview",
             "epgp_convergence", "operator_spectrum",
             "bem_reciprocity", "epgp_reciprocity", "epgp_vs_bem",
             "bem_convergence", "ellipse_convergence", "sphere_convergence",
             "epgp_ksweep", "sphere_ksweep", "sphere_multipole",
             # old shape_solver order
             "ellipse_bem_convergence", "ellipse_epgp_convergence",
             "sphere_epgp_convergence", "ellipse_epgp_rate", "sphere_epgp_rate",
             "ellipse_epgp_field_real", "ellipse_epgp_field_phase",
             "ellipse_epgp_field_lic", "ellipse_epgp_field_real_anim",
             "ellipse_epgp_field_phase_anim", "ellipse_epgp_field_lic_anim",
             "sphere_epgp_field_real", "sphere_epgp_field_phase",
             "sphere_epgp_field_lic", "sphere_epgp_field_real_anim",
             "sphere_epgp_field_phase_anim", "sphere_epgp_field_lic_anim")
    for f in os.listdir(FIGS):
        if f.rsplit(".", 1)[0] in stale:
            os.remove(os.path.join(FIGS, f))
    print(f"wrote figures to {FIGS}  (BEM ref p{BEM_REFERENCE[0]} m{BEM_REFERENCE[1]})")


if __name__ == "__main__":
    main()
