"""Generate the reference for devtools/checkkick.js's fixture leg (MUSIC_PLAN M1).

    python webgpu/gen_kickfixture.py        # writes webgpu/devtools/fixtures/kick_offline.json

The WebGPU `blobKick` kernel evaluates a gaussian blob's transform ANALYTICALLY in k space.
The offline reference implementation of the same kick -- `make_kick` in
examples/music_turbulence.py -- builds the gaussians in REAL space (minimum image), takes
taranis' own rfft2, applies taranis' own dealias mask and zeroes k = 0. This file runs THAT
function, unmodified, at fp64 on a 32^2 grid and records what it adds to a zero state, so
the gate compares the kernel against the code the offline movies were made with rather
than against a re-derivation.

Every sigma here is <= 0.5 on purpose: `make_kick`'s gaussian is MINIMUM-IMAGE, the kernel's
analytic form is the fully periodic one, and the two differ by exp(-(L/2)^2/2 sigma^2) of the
peak -- 3e-9 at 0.5, 3e-6 at 0.625 (the music map's widest blob is 0.45).

Recorded per case: the blob list in the solver's `kick(list)` convention
({x, y, sigma, amp, pol}; pol < 0 = z-) and the resulting (phi, psi) modes, flattened as the
WebGPU fields buffer is: the phi half then the psi half, mode m = ix*nky + iy, (re, im)
interleaved. Deterministic: no RNG.
"""
import json
import os
import sys

os.environ["TARANIS_PRECISION"] = "64"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

import numpy as np
import jax.numpy as jnp

import taranis as jr
import music_turbulence as mt

N = 32
CASES = [
    ("one z+ blob, offset (0.75, 1.5)",
     [dict(x=0.75, y=1.5, sigma=0.375, amp=0.25, pol=1)]),
    ("a different blob in each channel",
     [dict(x=0.75, y=1.5, sigma=0.5, amp=0.25, pol=1),
      dict(x=2.5, y=0.25, sigma=0.375, amp=-0.5, pol=-1)]),
    ("three per channel, both signs",
     [dict(x=0.75, y=1.5, sigma=0.5, amp=0.25, pol=1),
      dict(x=4.25, y=5.0, sigma=0.375, amp=0.75, pol=1),
      dict(x=1.25, y=0.75, sigma=0.4375, amp=-0.625, pol=1),
      dict(x=5.5, y=3.25, sigma=0.5, amp=0.375, pol=-1),
      dict(x=3.0, y=4.5, sigma=0.375, amp=-0.5, pol=-1),
      dict(x=0.25, y=6.0, sigma=0.5, amp=1.0, pol=-1)]),
]


# MUSIC_EFFECTS_PLAN E1: the anisotropic `sheet` kick, which make_kick builds analytically
# in k space (potential_k) -- the same closed form the page's sheetKick kernel evaluates, so
# these are fully periodic at any width. `ratio` = length / width, `theta` = the angle of
# the long axis from x. Recorded under their own key: `cases` above is unchanged.
SHEET_CASES = [
    ("one z+ sheet, ratio 4 at 0.6 rad",
     [dict(x=0.75, y=1.5, sigma=0.25, amp=0.25, pol=1, ratio=4.0, theta=0.6)]),
    ("a different sheet in each channel",
     [dict(x=0.75, y=1.5, sigma=0.25, amp=0.25, pol=1, ratio=4.0, theta=0.6),
      dict(x=2.5, y=0.25, sigma=0.3125, amp=-0.5, pol=-1, ratio=2.5, theta=2.2)]),
    ("three per channel, both signs, one of ratio 1",
     [dict(x=0.75, y=1.5, sigma=0.25, amp=0.25, pol=1, ratio=4.0, theta=0.6),
      dict(x=4.25, y=5.0, sigma=0.375, amp=0.75, pol=1, ratio=1.0, theta=0.0),
      dict(x=1.25, y=0.75, sigma=0.3125, amp=-0.625, pol=1, ratio=3.0, theta=1.9),
      dict(x=5.5, y=3.25, sigma=0.25, amp=0.375, pol=-1, ratio=4.0, theta=3.0),
      dict(x=3.0, y=4.5, sigma=0.375, amp=-0.5, pol=-1, ratio=2.0, theta=0.1),
      dict(x=0.25, y=6.0, sigma=0.3125, amp=1.0, pol=-1, ratio=5.0, theta=1.2)]),
]


# MUSIC_STAGE_PLAN S2: the same two kicks in a RECTANGULAR box -- box_for(aspect), short
# side 2 pi -- on a rectangular grid, landscape and portrait. The grids are 64 x 32 and
# 32 x 64 so that both spacings stay at the square fixture's 2 pi / 32 or below (make_kick's
# blob is sampled in real space: a coarser axis would alias its tail into the retained
# modes at a level the fp32 comparison sees). Positions use the whole long side. Recorded
# under their own key, `rect`: `cases` and `sheet_cases` above are unchanged.
RECT = [
    dict(tag="landscape 16:9 on 64 x 32", aspect=16.0 / 9.0, nx=64, ny=32,
         blobs=("three per channel across the long side, both signs",
                [dict(x=0.75, y=1.5, sigma=0.5, amp=0.25, pol=1),
                 dict(x=9.25, y=5.0, sigma=0.375, amp=0.75, pol=1),
                 dict(x=6.25, y=0.75, sigma=0.4375, amp=-0.625, pol=1),
                 dict(x=10.5, y=3.25, sigma=0.5, amp=0.375, pol=-1),
                 dict(x=3.0, y=4.5, sigma=0.375, amp=-0.5, pol=-1),
                 dict(x=0.25, y=6.0, sigma=0.5, amp=1.0, pol=-1)]),
         sheets=("sheets and a ratio-1 blob, both channels",
                 [dict(x=8.75, y=1.5, sigma=0.25, amp=0.25, pol=1, ratio=4.0, theta=0.6),
                  dict(x=4.25, y=5.0, sigma=0.375, amp=0.75, pol=1, ratio=1.0, theta=0.0),
                  dict(x=10.5, y=3.25, sigma=0.25, amp=0.375, pol=-1, ratio=4.0, theta=3.0),
                  dict(x=0.25, y=6.0, sigma=0.3125, amp=1.0, pol=-1, ratio=5.0, theta=1.2)])),
    dict(tag="portrait 9:16 on 32 x 64", aspect=9.0 / 16.0, nx=32, ny=64,
         blobs=("two per channel up the long side",
                [dict(x=0.75, y=9.5, sigma=0.5, amp=0.25, pol=1),
                 dict(x=4.25, y=5.0, sigma=0.375, amp=-0.75, pol=1),
                 dict(x=5.5, y=10.75, sigma=0.5, amp=0.375, pol=-1),
                 dict(x=3.0, y=0.5, sigma=0.4375, amp=-0.5, pol=-1)]),
         sheets=("a sheet in each channel",
                 [dict(x=0.75, y=8.5, sigma=0.25, amp=0.25, pol=1, ratio=4.0, theta=0.6),
                  dict(x=2.5, y=10.25, sigma=0.3125, amp=-0.5, pol=-1, ratio=2.5, theta=2.2)])),
]


def rect_rows():
    rows = []
    for R in RECT:
        nx, ny = R["nx"], R["ny"]
        Lx, Ly = mt.box_for(R["aspect"])
        params = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=0.4, dims=2,
                               comm_backend="serial", eqpars={"diss": (0.0, 0.0), "hyper": 2})
        kick, _ = mt.make_kick(params, jr.setup_kgrids(params))
        nky = ny // 2 + 1
        row = dict(tag=R["tag"], aspect=R["aspect"], nx=nx, ny=ny, Lx=float(Lx), Ly=float(Ly))
        for key, ncol in (("blobs", 4), ("sheets", 6)):
            tag, blobs = R[key]
            cols = ("x", "y", "sigma", "amp", "ratio", "theta")[:ncol]
            chan = lambda sgn: np.array([[b[c] for c in cols] for b in blobs
                                         if (b["pol"] < 0) == (sgn < 0)]).reshape(-1, ncol)
            zero = jnp.zeros((2, 1, nx, nky), dtype=jnp.complex128)
            f = np.asarray(kick(zero, mt.pad(chan(+1)), mt.pad(chan(-1))))
            flat = np.stack([f.real, f.imag], axis=-1).reshape(-1)
            assert flat.shape == (2 * nx * nky * 2,)
            row[key] = dict(tag=tag, list=blobs, fields=[float(v) for v in flat])
        rows.append(row)
    return rows


def main():
    params = jr.Parameters(nx=N, ny=N, Lx=mt.LBOX, Ly=mt.LBOX, cfl_safety=0.4, dims=2,
                           comm_backend="serial", eqpars={"diss": (0.0, 0.0), "hyper": 2})
    kgrid = jr.setup_kgrids(params)
    kick, _ = mt.make_kick(params, kgrid)
    nky = N // 2 + 1
    out = dict(nx=N, ny=N, Lx=float(mt.LBOX), Ly=float(mt.LBOX),
               source="examples/music_turbulence.py make_kick, TARANIS_PRECISION=64",
               cases=[])
    for tag, blobs in CASES:
        chan = lambda sgn: np.array([[b["x"], b["y"], b["sigma"], b["amp"]]
                                     for b in blobs if (b["pol"] < 0) == (sgn < 0)]).reshape(-1, 4)
        zero = jnp.zeros((2, 1, N, nky), dtype=jnp.complex128)
        f = np.asarray(kick(zero, mt.pad(chan(+1)), mt.pad(chan(-1))))    # (2, 1, nx, nky)
        flat = np.stack([f.real, f.imag], axis=-1).reshape(-1)           # phi half, psi half
        assert flat.shape == (2 * N * nky * 2,)
        out["cases"].append(dict(tag=tag, list=blobs, fields=[float(v) for v in flat]))
    out["sheet_cases"] = []
    for tag, blobs in SHEET_CASES:
        chan = lambda sgn: np.array([[b["x"], b["y"], b["sigma"], b["amp"], b["ratio"], b["theta"]]
                                     for b in blobs if (b["pol"] < 0) == (sgn < 0)]).reshape(-1, 6)
        zero = jnp.zeros((2, 1, N, nky), dtype=jnp.complex128)
        f = np.asarray(kick(zero, mt.pad(chan(+1)), mt.pad(chan(-1))))
        flat = np.stack([f.real, f.imag], axis=-1).reshape(-1)
        out["sheet_cases"].append(dict(tag=tag, list=blobs, fields=[float(v) for v in flat]))
    out["rect"] = rect_rows()
    dst = os.path.join(HERE, "devtools", "fixtures", "kick_offline.json")
    with open(dst, "w") as fh:
        json.dump(out, fh)
    print("wrote", dst, os.path.getsize(dst), "bytes,", len(out["cases"]), "cases")


if __name__ == "__main__":
    main()
