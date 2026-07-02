"""Uncertainty-vs-normal-component investigation on the ellipsoidal cavity.

Tests the conjecture that the posterior-uncertainty lobes coincide with where the
boundary field has a large normal component. The EPGP posterior variance depends
only on the conditioning geometry (the Cholesky factor of the weight-space
precision), not on the transmitter data, so the uncertainty is identical for
every transmitter. We verify this numerically and compare the (fixed) lobe
pattern against the (transmitter-dependent) boundary normal component for two
different transmitters.

Run (from the cavity-benchmark checkout, with maxwellgp resolvable):

    uv run python -m cavity_benchmark.uncertainty_normal \
        --config ../cavity-epgp/res/config_ellipse.txt \
        --n-spectral 512 --n-boundary 4096 --out out/figs/ellipse_uncertainty_normal.svg
"""
import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from cavity_epgp.operators import (
    GPConfig, boundary_collocation, fit, load_config, tangential_trace,
)
from cavity_epgp.analytic import incident_field_batch

from .plot.common import setup_style, save


def _fit(cfg, semiaxes, k, ns, z, p):
    bp, bn = boundary_collocation(semiaxes, cfg.n_boundary)
    y = jnp.asarray(tangential_trace(incident_field_batch(bp, z, k, p), bn).reshape(-1, 1))
    return fit(cfg, semiaxes, k, y, ns)


def _field_std(model, post, pts, batch=1500):
    pts = np.asarray(pts)
    mc, vc = [], []
    for i in range(0, len(pts), batch):
        phi = model.kernel.feature_map.full(jnp.asarray(pts[i:i + batch]))
        mc.append(np.asarray(post.mean(phi)))
        vc.append(np.asarray(post.var(phi)))
    mean = np.concatenate(mc).reshape(-1, 6)
    var = np.concatenate(vc).reshape(-1, 6)
    return mean[:, :3], np.sqrt(np.clip(var[:, :3], 0.0, None))


def _corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def run(args):
    k, semiaxes, *_ = load_config(args.config)
    a, b, c = semiaxes
    cfg = GPConfig(n_boundary=args.n_boundary)
    ns = args.n_spectral

    # two structurally different transmitters
    srcA = (np.array([0., 0., 1.]), np.array([1., 0., 0.]))
    srcB = (np.array([0.7, 0., 0.]), np.array([0., 0., 1.]))
    mA, pA = _fit(cfg, semiaxes, k, ns, *srcA)
    mB, pB = _fit(cfg, semiaxes, k, ns, *srcB)

    # posterior-std slice in the xz-plane, both transmitters
    ng = args.ngrid
    xs = np.linspace(-1.02 * a, 1.02 * a, ng)
    zs = np.linspace(-1.02 * c, 1.02 * c, ng)
    XX, ZZ = np.meshgrid(xs, zs)
    grid = np.stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()], axis=1)
    mask = (grid[:, 0] ** 2 / a ** 2 + grid[:, 2] ** 2 / c ** 2) <= 1.0
    _, sA = _field_std(mA, pA, grid); stdA = np.linalg.norm(sA, axis=1)
    _, sB = _field_std(mB, pB, grid); stdB = np.linalg.norm(sB, axis=1)
    reldiff = np.linalg.norm((stdA - stdB)[mask]) / np.linalg.norm(stdA[mask])
    img = np.where(mask, stdA, np.nan).reshape(ng, ng)
    vmax = np.nanpercentile(img, 99)

    # per-direction lobe intensity (max std along the ray) vs |E.n| at the wall
    nring = 480
    t = np.linspace(0, 2 * np.pi, nring, endpoint=False)
    wall = np.stack([a * np.cos(t), np.zeros_like(t), c * np.sin(t)], axis=1)
    g = np.stack([wall[:, 0] / a ** 2, np.zeros_like(t), wall[:, 2] / c ** 2], axis=1)
    nrm = g / np.linalg.norm(g, axis=1, keepdims=True)
    ss = np.linspace(0.15, 0.97, 60)
    lobe = np.array([
        np.linalg.norm(_field_std(mA, pA, ss[:, None] * w[None, :])[1], axis=1).max()
        for w in wall
    ])

    def normal_at_wall(model, post, z, p):
        Es, _ = _field_std(model, post, wall)
        Etot = incident_field_batch(wall, z, k, p) + Es
        return np.abs(np.sum(Etot * nrm, axis=1))

    EnA = normal_at_wall(mA, pA, *srcA)
    EnB = normal_at_wall(mB, pB, *srcB)
    cA, cB, cAB = _corr(lobe, EnA), _corr(lobe, EnB), _corr(EnA, EnB)
    print(f"rel diff of std between transmitters (interior) = {reldiff:.3e}")
    print(f"corr(lobes, |E.n|_A) = {cA:+.3f}")
    print(f"corr(lobes, |E.n|_B) = {cB:+.3f}")
    print(f"corr(|E.n|_A, |E.n|_B) = {cAB:+.3f}")

    setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})
    im = ax[0].pcolormesh(xs, zs, img, shading="gouraud", cmap="magma",
                          vmin=0.0, vmax=vmax, rasterized=True)
    ax[0].add_patch(mpatches.Ellipse((0, 0), 2 * a, 2 * c, fill=False, edgecolor="black", lw=3.2))
    ax[0].plot([srcA[0][0]], [srcA[0][2]], "o", color="lime", ms=7, mec="black", mew=0.8)
    ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"$z$")
    ax[0].set_aspect("equal"); ax[0].grid(False)
    ax[0].set_title(r"$\sigma[\mathbf{E}^\mathrm{s}]$ (identical for every transmitter)")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    deg = np.degrees(t)
    n = lambda x: x / np.max(x)
    ax[1].plot(deg, n(lobe), color="black", lw=2.4, label="uncertainty lobes (any transmitter)")
    ax[1].plot(deg, n(EnA), color="tab:blue", lw=1.5, ls="--", label=r"$|\mathbf{E}\cdot\mathbf{n}|$, transmitter A")
    ax[1].plot(deg, n(EnB), color="tab:orange", lw=1.5, ls=":", label=r"$|\mathbf{E}\cdot\mathbf{n}|$, transmitter B")
    ax[1].set_xlabel(r"boundary direction in the $xz$-plane [deg]")
    ax[1].set_ylabel("normalized magnitude")
    ax[1].set_xlim(0, 360); ax[1].set_xticks([0, 90, 180, 270, 360])
    ax[1].legend(frameon=False, loc="upper center")
    ax[1].set_title(rf"corr(lobes, $|E{{\cdot}}n|$): A ${cA:+.2f}$, B ${cB:+.2f}$")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="../cavity-epgp/res/config_ellipse.txt")
    ap.add_argument("--n-spectral", type=int, default=512)
    ap.add_argument("--n-boundary", type=int, default=4096)
    ap.add_argument("--ngrid", type=int, default=240)
    ap.add_argument("--out", default="out/figs/ellipse_uncertainty_normal.svg")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
