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
    dst = os.path.join(HERE, "devtools", "fixtures", "kick_offline.json")
    with open(dst, "w") as fh:
        json.dump(out, fh)
    print("wrote", dst, os.path.getsize(dst), "bytes,", len(out["cases"]), "cases")


if __name__ == "__main__":
    main()
