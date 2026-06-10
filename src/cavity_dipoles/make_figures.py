import argparse
import os
import subprocess
import sys

FIELD_DIR = os.path.join("out", "field")
FIGS = os.path.join("out", "figs")
MODES = ["real", "phase", "lic"]

# Field slices to render: (npz file, output prefix, required). The ellipsoidal
# cavity is the default benchmark; the spherical cavity is rendered too when its
# slice exists.
FIELD_SLICES = [
    (os.path.join(FIELD_DIR, "ellipse_field_slice.npz"), "ellipse_epgp_field", True),
    (os.path.join(FIELD_DIR, "sphere_field_slice.npz"), "sphere_epgp_field", False),
]

AGG = "cavity_dipoles.results.aggregate"
CONV = "cavity_dipoles.plot.convergence"
FLD = "cavity_dipoles.plot.field"

DATA = [[AGG]]
BENCH = [[CONV]]


def field_steps(npz, prefix, animate):
    ext = "webp" if animate else "png"
    anim = ["--animate"] if animate else []
    suffix = "_anim" if animate else ""
    return [[FLD, npz, "--mode", m, *anim,
             "--out", os.path.join(FIGS, f"{prefix}_{m}{suffix}.{ext}")]
            for m in MODES]


def run(step):
    print(f"\n=== {' '.join(step)} ===", flush=True)
    subprocess.run([sys.executable, "-m", step[0], *step[1:]], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-anim", action="store_true")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--png", action="store_true",
                    help="also emit PNG versions of the line plots (default: SVG only)")
    args = ap.parse_args()

    steps = DATA + BENCH
    if args.png:
        steps += [[CONV, "--format", "png"]]
    if not args.skip_field:
        for npz, prefix, required in FIELD_SLICES:
            if not os.path.exists(npz):
                if required:
                    print(f"! {npz} missing -- generate it with "
                          "'python -m cavity_dipoles.epgp.operators field', or pass --skip-field",
                          file=sys.stderr)
                    sys.exit(1)
                continue
            steps += field_steps(npz, prefix, animate=False)
            if not args.skip_anim:
                steps += field_steps(npz, prefix, animate=True)

    for s in steps:
        run(s)
    print(f"\nall figures written to {os.path.join('out', 'figs')}")


if __name__ == "__main__":
    main()
