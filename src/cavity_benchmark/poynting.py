"""Poynting-vector row for the spherical-cavity field figure.

Computes the time-averaged Poynting vector S = 1/2 Re(E x H*) of the incident,
scattered, and total field on the xz-slice, with H obtained from Maxwell,
H = curl E / (i k), via a thin three-plane finite difference. The total field is
a pure standing wave, so its Poynting vector vanishes; where the flux is only
numerical noise it is set to exactly zero and drawn as a uniform "S = 0" panel.

Run (from the cavity-benchmark checkout, with maxwellgp/cavity-epgp resolvable):

    uv run python -m cavity_benchmark.poynting \
        --config ../cavity-epgp/res/config_sphere.txt \
        --out out/figs/epgp_sphere_field_poynting.png
"""
import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from cavity_epgp.operators import (
    GPConfig, boundary_collocation, fit, load_config, tangential_trace,
)
from cavity_epgp.analytic import incident_field_batch

from .plot.common import setup_style, decorate


def _fit(cfg, semiaxes, k, ns, z, p):
    bp, bn = boundary_collocation(semiaxes, cfg.n_boundary)
    y = jnp.asarray(tangential_trace(incident_field_batch(bp, z, k, p), bn).reshape(-1, 1))
    return fit(cfg, semiaxes, k, y, ns)


def run(args):
    k, semiaxes, *_ = load_config(args.config)
    R = float(semiaxes[0]); a = c = R
    z = np.array([0.0, 0.0, 1.0]); p = np.array([1.0, 0.0, 0.0])
    model, post = _fit(GPConfig(n_boundary=args.n_boundary), semiaxes, k, args.n_spectral, z, p)

    ng = args.ngrid
    xs = np.linspace(-1.05 * R, 1.05 * R, ng)
    zs = np.linspace(-1.05 * R, 1.05 * R, ng)
    XX, ZZ = np.meshgrid(xs, zs)
    h = 0.02

    def pts(yv):
        return np.stack([XX.ravel(), np.full(XX.size, yv), ZZ.ravel()], 1)

    def e_scat(P, b=1500):
        o = []
        for i in range(0, len(P), b):
            o.append(np.asarray(post.mean(model.kernel.feature_map.full(jnp.asarray(P[i:i + b])))))
        return np.concatenate(o).reshape(-1, 6)[:, :3]

    def e_inc(P):
        return incident_field_batch(P, z, k, p)

    def field_and_H(efun):
        dx, dz = xs[1] - xs[0], zs[1] - zs[0]
        E0 = efun(pts(0.0)).reshape(ng, ng, 3)
        Ep = efun(pts(h)).reshape(ng, ng, 3)
        Em = efun(pts(-h)).reshape(ng, ng, 3)
        dEdz = np.gradient(E0, dz, axis=0)
        dEdx = np.gradient(E0, dx, axis=1)
        dEdy = (Ep - Em) / (2 * h)
        curl = np.empty_like(E0)
        curl[..., 0] = dEdy[..., 2] - dEdz[..., 1]
        curl[..., 1] = dEdz[..., 0] - dEdx[..., 2]
        curl[..., 2] = dEdx[..., 1] - dEdy[..., 0]
        return E0, curl / (1j * k)

    Ei, Hi = field_and_H(e_inc)
    Es, Hs = field_and_H(e_scat)
    poynt = lambda E, H: 0.5 * np.real(np.cross(E, np.conj(H)))
    Si, Ss, St = poynt(Ei, Hi), poynt(Es, Hs), poynt(Ei + Es, Hi + Hs)

    r = np.sqrt(XX ** 2 + ZZ ** 2)
    inside = r < 0.997 * R
    srcmask = np.sqrt(XX ** 2 + (ZZ - 1) ** 2) > 0.35
    mag = lambda S: np.hypot(S[..., 0], S[..., 2])
    vmax = np.nanpercentile(mag(Si)[inside & srcmask], 94)
    noise = 1e-6 * mag(Si)[inside].max()

    setup_style()
    plt.rcParams["savefig.bbox"] = "tight"
    fig, ax = plt.subplots(1, 3, figsize=(15, 6.2))
    for a_, (S, title) in zip(ax, [(Si, r"$\mathbf{S}^{\mathrm{i}}$"),
                                   (Ss, r"$\mathbf{S}^{\mathrm{s}}$"),
                                   (St, r"$\mathbf{S}^{\mathrm{tot}}$")]):
        Sx, Sz, M = S[..., 0].copy(), S[..., 2].copy(), mag(S)
        zero = M.max() < noise
        if zero:
            Sx[:] = Sz[:] = 0.0
            M = M * 0.0
        a_.pcolormesh(xs, zs, np.where(inside, M, np.nan), shading="gouraud",
                      cmap="magma", vmin=0, vmax=vmax, rasterized=True)
        if zero:
            a_.text(0, 0, r"$\mathbf{S}=\mathbf{0}$", ha="center", va="center",
                    color="white", fontsize=17)
        else:
            m = ~(inside & srcmask)
            Sx[m] = 0.0; Sz[m] = 0.0
            lw = 0.4 + 2.0 * np.clip(np.hypot(Sx, Sz) / vmax, 0, 1)
            a_.streamplot(xs, zs, Sx, Sz, color="white", density=1.15,
                          linewidth=lw, arrowsize=0.9)
        decorate(a_, a, c, z, title)
        a_.grid(False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="../cavity-epgp/res/config_sphere.txt")
    ap.add_argument("--n-spectral", type=int, default=768)
    ap.add_argument("--n-boundary", type=int, default=6000)
    ap.add_argument("--ngrid", type=int, default=200)
    ap.add_argument("--out", default="out/figs/epgp_sphere_field_poynting.png")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
