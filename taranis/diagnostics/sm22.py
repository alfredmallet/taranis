# Read-only observers for the SM22 growth rule (physics/sm22.py). All real-space, pointwise
# where it matters (aw_papers CLAUDE.md rule 7: convergence audits are POINTWISE).
#
# The fields are exactly band-limited (2/3 mask), so the reference code's tail_max (content
# above N/3) is identically zero here. The two honest unresolved burdens are instead
#   * Berr_out: the out-of-band part of |B|^2 -- the constraint is imposed Galerkin-wise on
#     the band only, so this is what the discretization does not control. THE convergence-audit
#     burden (A. Mallet, 2026-09-29); Berr_out_max is its pointwise form (rule 7: an rms can
#     hide a dilute defect -- draining with N = honest, pinned = weak member);
#   * edge_*: B's content in the top shell of the retained band (max-norm mode index in
#     [edge_frac, 1) x the band edge), pointwise max and L8, the analogue of tail_max/tail8.
import jax
import jax.numpy as jnp
from .. import grids
from ..physics import sm22 as _sm22


def _modeindex_maxnorm(kgrid, params):
    # max_i |k_i| / k_edge_i: 1 at the (per-axis) 2/3 band edge
    kx, ky, kz = _sm22._kvec(kgrid, params)
    r = jnp.maximum(jnp.abs(kx) / (2*jnp.pi*(params.nx/3)/params.Lx),
                    jnp.abs(ky) / (2*jnp.pi*(params.ny/3)/params.Ly))
    if kz is not None:
        r = jnp.maximum(r, jnp.abs(kgrid.kz) / (2*jnp.pi*(params.nz/3)/params.Lz))
    return r


def diagnostics(state, kgrid, params, edge_frac=0.75):
    fk = state.fields
    B = jnp.stack([grids.ifft(fk[i], params) for i in range(3)])
    axes = tuple(range(1, B.ndim))
    Bbar = jnp.mean(B, axis=axes, keepdims=True)
    B0 = jnp.sqrt(jnp.sum(Bbar**2))
    dB = B - Bbar
    Bsq = jnp.sum(B*B, axis=0)
    kx, ky, kz = _sm22._kvec(kgrid, params)
    ks = (kx, ky) if kz is None else (kx, ky, kz)
    G2 = sum(grids.ifft(1j*k*fk[i], params)**2 for i in range(3) for k in ks) / Bsq
    Bsq_k = grids.fft(Bsq, params)
    out = grids.ifft(jnp.where(kgrid.dealias, 0.0, 1.0) * Bsq_k, params)
    edge = (_modeindex_maxnorm(kgrid, params) >= edge_frac) & kgrid.dealias
    Be = jnp.stack([grids.ifft(edge * fk[i], params) for i in range(3)])
    e2 = jnp.sum(Be*Be, axis=0)
    mBsq = jnp.mean(Bsq)
    return dict(A=jnp.sqrt(jnp.mean(jnp.sum(dB*dB, axis=0))) / B0,
                Berr=jnp.sqrt(jnp.mean((Bsq - mBsq)**2)) / mBsq,
                Berr_out=jnp.sqrt(jnp.mean(out**2)) / mBsq,
                Berr_out_max=jnp.max(jnp.abs(out)) / mBsq,
                maxgrad=jnp.sqrt(jnp.max(G2)),
                lp8=jnp.mean(G2**4)**0.125,
                edge_max=jnp.sqrt(jnp.max(e2)),
                edge8=jnp.mean(e2**4)**0.125,
                edge_rms=jnp.sqrt(jnp.mean(e2)),
                divB=jnp.max(jnp.abs(sum(grids.ifft(1j*k*fk[i], params)
                                         for i, k in enumerate(ks)))),
                wmin=jnp.min(B[2] / jnp.sqrt(Bsq)), wmax=jnp.max(B[2] / jnp.sqrt(Bsq)))


def amplitude(state, params):
    fk = state.fields
    B = jnp.stack([grids.ifft(fk[i], params) for i in range(3)])
    axes = tuple(range(1, B.ndim))
    Bbar = jnp.mean(B, axis=axes, keepdims=True)
    return jnp.sqrt(jnp.mean(jnp.sum((B - Bbar)**2, axis=0))) / jnp.sqrt(jnp.sum(Bbar**2))
