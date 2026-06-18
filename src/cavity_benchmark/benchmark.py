"""Benchmark geometries and their reference reaction operators.

Two benchmarks share one EPGP convergence study, differing only in the reference
the EPGP operator is compared against:
  * ellipse -- semi-axes (4, 4, 6); reference is the deterministic BEM operator
    at BEM_REFERENCE (the high corner of the (p,m) grid for now);
  * sphere  -- semi-axes (R, R, R); reference is the exact analytic operator of
    sphere.reaction_operator_sphere.

Generation (assembling EPGP operators) and analysis (comparing to the reference)
are kept separate, as forced on the BEM side: epgp.convergence generates the
operators and a raw manifest for a given geometry, and results.aggregate then
compares them to the reference returned here.
"""

import os

from .results.compare import load_bem
from .sphere import reaction_operator_sphere

GEOMETRIES = {"ellipse": (4.0, 4.0, 6.0), "sphere": (4.0, 4.0, 4.0)}

# The single, explicit declaration of the BEM reference for the ellipse
# cross-validation, as the (p, m) of the run whose operator is the reference.
# Every consumer (benchmark.reference_operator, results.aggregate, the figures)
# reads this -- nothing picks the reference implicitly. The high corner of the
# (p, m) grid for now; bump to a finer (p, m) when a finer reference is computed.
BEM_REFERENCE = (5, 4)


def bem_reference_path():
    p, m = BEM_REFERENCE
    return os.path.join("out", "bem", "ellipse", f"T_p{p}_m{m}.dat")


def semiaxes(name):
    return GEOMETRIES[name]


def config_path(name):
    return os.path.join("res", f"config_{name}.txt")


def out_dir(name):
    return os.path.join("out", "epgp", name)


def reference_operator(name, k, points, e1, e2):
    """Reference reaction operator for the named benchmark at wavenumber k."""
    if name == "sphere":
        return reaction_operator_sphere(k, float(max(GEOMETRIES[name])), points, e1, e2)
    if name == "ellipse":
        return load_bem(bem_reference_path())
    raise ValueError(f"unknown benchmark {name!r}")
