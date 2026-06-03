import os

import numpy as np

K = 2.0
SEMIAXES = (4.0, 4.0, 6.0)
N = 12
OUT = "out/config.txt"


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )


def tangent_frame(n):
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def main():
    points = fibonacci_sphere(N)  # unit sphere Lambda
    rows = []
    for x in points:
        n = x
        e1, e2 = tangent_frame(n)
        rows.append(np.concatenate([x, e1, e2]))
    rows = np.stack(rows)

    a, b, c = SEMIAXES

    os.makedirs("out", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# k a b c N\n")
        f.write(f"{K} {a} {b} {c} {N}\n")
        f.write("# x y z  e1x e1y e1z  e2x e2y e2z\n")
        for r in rows:
            f.write(" ".join(f"{v:.15g}" for v in r) + "\n")

    print(f"wrote {OUT}: {N} points")


if __name__ == "__main__":
    main()
