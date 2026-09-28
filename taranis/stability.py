# Linear stability about a frozen state x0, by autodiff (plans/AUTODIFF_PLAN.md, rung 0).
# construct_rhs returns N(f) only -- the k-local linear part L lives in kgrid.lin -- so the
# Jacobian of dt f = L f + N(f) is
#       J v = kgrid.lin.apply_L(v) + jvp(N, (x0,), (v,))[1]
# N' is jax's derivative of the solver's own RHS; no linearized equations are written.
# Read-only: nothing here mutates Parameters, the kgrid or a state, and the solver never
# imports this module. Unforced, single-process, non-sharded backends only.
#
# Three vector spaces:
#   fields       the solver's (nfields, nz, nkx, nky) complex array. J is only REAL-linear on
#                it: irfft2 discards the anti-Hermitian part of the self-conjugate rows
#                (ky = 0 and Nyquist).
#   ky column    for a ky-independent x0 and 0 < iky < ny/2, J maps the dealias-kept
#                (field, z, kx) entries of one ky column to themselves complex-linearly:
#                ky_block_matrix / ky_block_operator, an ordinary complex matrix.
#   real coords  the dealias-kept modes of a REAL field, each conjugate pair counted once,
#                as (Re, Im) coordinates (RealCoords): the space for a general x0, where J
#                is a real matrix (real_matrix / real_operator).
# Dealias-masked modes are left out of every block: there J is L alone (N is masked), which
# only adds spurious damped eigenvalues.
import functools
import warnings
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from . import _precision
from .physics import construct_rhs, equation_registry
from .types import SimulationState

# per-probe working-set estimate (in state-sized arrays) that sizes the vmapped probe chunks
_PROBE_ARRAYS = 40
_PROBE_BUDGET_BYTES = 1e9


def _check_supported(params):
    if params.forcing:
        raise ValueError("stability: the harness linearizes the UNFORCED RHS; build the "
                         "Parameters with forcing=False")
    if params.size != 1 or params.comm_backend == "jax":
        raise ValueError("stability: single-process, non-sharded backends only (size == 1, "
                         f"comm_backend != 'jax'); got size={params.size}, "
                         f"comm_backend={params.comm_backend!r}")
    if params.comm_backend == "mpi4jax" and params.spatial_dimensions == 3 and not params.z_spectral:
        # the FD-z halo is an mpi4jax sendrecv, which has no jvp rule
        raise ValueError("stability: finite-difference z needs comm_backend='serial' (mpi4jax's "
                         "sendrecv halo cannot be differentiated); at size 1 it is the same "
                         "operator")


def _fields(x0, params):
    f = x0.fields if isinstance(x0, SimulationState) else x0
    f = jnp.asarray(f, dtype=_precision.ctype)
    shape = (params.nfields, params.nz, params.nx, params.ny//2 + 1)
    if f.shape != shape:
        raise ValueError(f"stability: x0 has shape {f.shape}, expected the fields shape {shape}")
    return f


def _nonlinear(params):
    # N(fields, kgrid): the RHS terms construct_rhs sums, on a throwaway unforced state
    rhs = construct_rhs(equation_registry[params.eqtype])
    nkx, nky = params.nx, params.ny//2 + 1

    def N(fields, kgrid):
        state = SimulationState(
            t=jnp.float64(0.0), fields=fields,
            forcing_state=jnp.zeros((params.n_ou, 2, nkx, nky), dtype=_precision.ctype),
            forcing_key=jax.random.key(0),
            forcing_scale=jnp.zeros((params.n_ou,), dtype=_precision.ftype))
        return rhs(state, kgrid, params)[0]
    return N


def _jacobian_action(params):
    # (x0, v, kgrid) -> J v
    N = _nonlinear(params)

    def jv(x0, v, kgrid):
        nv = jax.jvp(lambda f: N(f, kgrid), (x0,), (v,))[1]
        return (kgrid.lin.apply_L(v) + nv).astype(_precision.ctype)
    return jv


def _adjoint_action(params):
    # (x0, u, kgrid) -> J^H u, the adjoint under Re<u, w> = Re sum(conj(u)*w): jax's
    # cotangent map is the plain transpose, so conjugate on the way in and out
    jv = _jacobian_action(params)

    def jhu(x0, u, kgrid):
        _, pullback = jax.vjp(lambda v: jv(x0, v, kgrid), jnp.zeros_like(u))
        return jnp.conj(pullback(jnp.conj(u))[0]).astype(_precision.ctype)
    return jhu


@functools.lru_cache(maxsize=16)
def _jitted(params, adjoint):
    # one compiled J v (or J^H u) per Parameters (identity-hashed; x0, kgrid are traced args)
    return jax.jit(_adjoint_action(params) if adjoint else _jacobian_action(params))


def jvp_operator(x0, kgrid, params):
    # matrix-free v -> J v on the fields array (jitted; x0 a fields array or a state)
    _check_supported(params)
    x0 = _fields(x0, params)
    jv = _jitted(params, False)
    return lambda v: jv(x0, jnp.asarray(v, dtype=_precision.ctype), kgrid)


def transpose_operator(x0, kgrid, params):
    # matrix-free u -> J^H u on the fields array, by vjp of one RHS evaluation. The adjoint
    # is under the real inner product Re<u, w>; on a ky column, where J is complex-linear,
    # that is the conjugate transpose.
    _check_supported(params)
    x0 = _fields(x0, params)
    jhu = _jitted(params, True)
    return lambda u: jhu(x0, jnp.asarray(u, dtype=_precision.ctype), kgrid)


# ---------------------------------------------------------------- one ky column

def _kept(kgrid, params):
    # dealias mask as (nz, nkx, nky) (the 2D/FD-z mask is z-independent)
    return np.broadcast_to(np.asarray(kgrid.dealias), (params.nz, params.nx, params.ny//2 + 1))


def ky_block_index(kgrid, params, iky):
    # (n, 3) int array of the (field, iz, ikx) entries of ky column iky that the dealias
    # mask keeps, in C order: the row/column order of every ky block
    if not (isinstance(iky, (int, np.integer)) and 0 < iky < params.ny/2):
        raise ValueError(f"stability: iky must be an int with 0 < iky < ny/2 = {params.ny/2} "
                         f"(ky = 0 and Nyquist are self-conjugate rows, where J is not "
                         f"complex-linear); got {iky!r}")
    col = _kept(kgrid, params)[..., iky]
    if not col.any():
        raise ValueError(f"stability: ky column {iky} is entirely dealias-masked")
    return np.argwhere(np.broadcast_to(col, (params.nfields,) + col.shape))


def _assert_ky_independent(x0):
    a = np.asarray(x0)
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    tol = 1e3*np.finfo(a.real.dtype).eps*scale
    worst = float(np.max(np.abs(a[..., 1:]))) if a.shape[-1] > 1 else 0.0
    if worst > tol:
        raise ValueError(f"stability: a ky block needs a ky-independent x0 (J is then "
                         f"block-diagonal in ky); x0's ky != 0 content is {worst:.3e} against "
                         f"max |x0| = {scale:.3e}")


def _column_maps(params, idx, iky):
    # embed: (..., n) block vector -> fields array;  gather: fields array -> (n,) block vector
    shape = (params.nfields, params.nz, params.nx, params.ny//2 + 1)
    f, z, x = (jnp.asarray(idx[:, i]) for i in range(3))

    def embed(u):
        return jnp.zeros(shape, dtype=_precision.ctype).at[f, z, x, iky].set(u)

    def gather(g):
        return g[f, z, x, iky]
    return embed, gather


def _probe_chunk(params, chunk):
    if chunk is not None:
        return int(chunk)
    per = (_PROBE_ARRAYS*params.nfields*params.nz*params.nx*params.ny
           * np.dtype(_precision.ctype).itemsize)
    return int(np.clip(_PROBE_BUDGET_BYTES//per, 1, 512))


def _dense_from_probes(probe, n, chunk, dtype):
    # probe(cols) -> (len(cols), n) images of the unit vectors e_cols; returns the n x n matrix
    M = np.empty((n, n), dtype=dtype)
    for j0 in range(0, n, chunk):
        cols = np.arange(j0, j0 + chunk)
        cols = np.minimum(cols, n - 1)          # pad the last chunk: one compile
        out = np.asarray(probe(jnp.asarray(cols)))
        j1 = min(j0 + chunk, n)
        M[:, j0:j1] = out[:j1 - j0].T
    return M


def ky_block_matrix(x0, kgrid, params, iky, chunk=None):
    # dense J restricted to the kept entries of ky column iky (order: ky_block_index), for a
    # ky-independent x0: one jvp per basis vector, vmapped in chunks. Returns
    # (J_block complex128 (n, n), idx).
    _check_supported(params)
    x0 = _fields(x0, params)
    _assert_ky_independent(x0)
    idx = ky_block_index(kgrid, params, iky)
    n = len(idx)
    embed, gather = _column_maps(params, idx, iky)
    jv = _jacobian_action(params)

    @jax.jit
    def probe(cols, x0, kgrid):
        def one(j):
            return gather(jv(x0, embed(jax.nn.one_hot(j, n, dtype=_precision.ctype)), kgrid))
        return jax.vmap(one)(cols)

    B = _dense_from_probes(lambda cols: probe(cols, x0, kgrid), n,
                           min(_probe_chunk(params, chunk), n), np.complex128)
    return B, idx


def ky_block_operator(x0, kgrid, params, iky):
    # matrix-free form of ky_block_matrix: a complex scipy LinearOperator (matvec J,
    # rmatvec J^H) on the kept entries of ky column iky. Returns (op, idx).
    _check_supported(params)
    x0 = _fields(x0, params)
    _assert_ky_independent(x0)
    idx = ky_block_index(kgrid, params, iky)
    n = len(idx)
    embed, gather = _column_maps(params, idx, iky)
    jv = _jacobian_action(params)
    jhu = _adjoint_action(params)
    mv = jax.jit(lambda u, x0, kgrid: gather(jv(x0, embed(u), kgrid)))
    rmv = jax.jit(lambda u, x0, kgrid: gather(jhu(x0, embed(u), kgrid)))

    def wrap(fn):
        def apply(u):
            u = np.asarray(u).reshape(-1)
            return np.asarray(fn(jnp.asarray(u, dtype=_precision.ctype), x0, kgrid),
                              dtype=np.complex128)
        return apply
    op = spla.LinearOperator((n, n), matvec=wrap(mv), rmatvec=wrap(rmv), dtype=np.complex128)
    return op, idx


# ---------------------------------------------------------------- real coordinates

class RealCoords(NamedTuple):
    # (Re, Im) coordinates of the dealias-kept modes of a real field, one entry per conjugate
    # pair on the ky = 0 row (mirror kx -> -kx, and kz -> -kz under z_spectral), no Im at a
    # self-conjugate mode. All index arrays are into the flattened fields array.
    shape: tuple
    re: np.ndarray           # entries carrying a Re coordinate
    im: np.ndarray           # entries carrying an Im coordinate
    mirror_src: np.ndarray   # canonical entries whose conjugate partner is also stored...
    mirror_dst: np.ndarray   # ...at these entries

    @property
    def n(self):
        return len(self.re) + len(self.im)

    def pack(self, fields):
        g = jnp.reshape(fields, (-1,))
        return jnp.concatenate([jnp.real(g[self.re]), jnp.imag(g[self.im])])

    def unpack(self, u):
        nre = len(self.re)
        g = jnp.zeros(int(np.prod(self.shape)), dtype=_precision.ctype)
        g = g.at[self.re].set(u[:nre].astype(_precision.ftype))
        g = g.at[self.im].add(1j*u[nre:].astype(_precision.ftype))
        g = g.at[self.mirror_dst].set(jnp.conj(g[self.mirror_src]))
        return jnp.reshape(g, self.shape)


def real_coords(kgrid, params):
    nf, nz, nkx, nky = params.nfields, params.nz, params.nx, params.ny//2 + 1
    iz, ix, iy = np.meshgrid(np.arange(nz), np.arange(nkx), np.arange(nky), indexing="ij")
    kept = _kept(kgrid, params)
    # ky = 0 is the only kept self-conjugate row: the 2/3 mask always drops the Nyquist row
    if params.ny % 2 == 0 and kept[..., nky - 1].any():
        raise AssertionError("stability.real_coords: the ky Nyquist row is dealias-kept")
    self_row = iy == 0
    mz = (-iz) % nz if params.z_spectral else iz
    flat = np.ravel_multi_index((iz, ix, iy), (nz, nkx, nky))
    mflat = np.ravel_multi_index((mz, (-ix) % nkx, iy), (nz, nkx, nky))
    canon = kept & (~self_row | (flat <= mflat))
    selfconj = self_row & (flat == mflat)
    paired = canon & self_row & ~selfconj
    per = nz*nkx*nky
    offs = np.arange(nf)[:, None]*per

    def lift(sel):
        return (offs + flat[sel][None, :]).reshape(-1)
    return RealCoords(shape=(nf, nz, nkx, nky), re=lift(canon), im=lift(canon & ~selfconj),
                      mirror_src=lift(paired),
                      mirror_dst=(offs + mflat[paired][None, :]).reshape(-1))


def _real_action(x0, kgrid, params, coords):
    jv = _jacobian_action(params)

    def ju(u, x0, kgrid):
        return coords.pack(jv(x0, coords.unpack(u), kgrid))
    return ju


def real_operator(x0, kgrid, params, coords=None):
    # matrix-free J in real coordinates, for any x0: a real scipy LinearOperator (matvec J,
    # rmatvec J^T by vjp). Returns (op, coords).
    _check_supported(params)
    x0 = _fields(x0, params)
    coords = real_coords(kgrid, params) if coords is None else coords
    ju = _real_action(x0, kgrid, params, coords)
    mv = jax.jit(ju)

    @jax.jit
    def rmv(w, x0, kgrid):
        _, pullback = jax.vjp(lambda u: ju(u, x0, kgrid), jnp.zeros_like(w))
        return pullback(w)[0]

    def wrap(fn):
        def apply(u):
            u = np.asarray(u, dtype=np.float64).reshape(-1)
            return np.asarray(fn(jnp.asarray(u, dtype=_precision.ftype), x0, kgrid),
                              dtype=np.float64)
        return apply
    op = spla.LinearOperator((coords.n, coords.n), matvec=wrap(mv), rmatvec=wrap(rmv),
                             dtype=np.float64)
    return op, coords


def real_matrix(x0, kgrid, params, coords=None, chunk=None):
    # dense J in real coordinates (small grids): one jvp per coordinate. Returns (M, coords).
    _check_supported(params)
    x0 = _fields(x0, params)
    coords = real_coords(kgrid, params) if coords is None else coords
    ju = _real_action(x0, kgrid, params, coords)
    n = coords.n

    @jax.jit
    def probe(cols, x0, kgrid):
        return jax.vmap(lambda j: ju(jax.nn.one_hot(j, n, dtype=_precision.ftype),
                                     x0, kgrid))(cols)

    M = _dense_from_probes(lambda cols: probe(cols, x0, kgrid), n,
                           min(_probe_chunk(params, chunk), n), np.float64)
    return M, coords


# ---------------------------------------------------------------- eigensolvers

# shift_invert raises when a pivot of the LU of J - sigma I is below this fraction of
# ||J|| + |sigma|: sigma is then (numerically) an eigenvalue, or deep in the pseudospectrum,
# and the solves carry no digits in the directions ARPACK needs for the other pairs
_PIVOT_TOL = 1e-12
# shift_invert's default residual_tol, in units of max(tol, eps)
_RESIDUAL_FACTOR = 1e3
_NORM_PROBES = 4


class EigResult(NamedTuple):
    values: np.ndarray                # eigenvalues
    right: np.ndarray                 # right eigenvectors, one per column
    left: Optional[np.ndarray]        # left eigenvectors w (J^H w = conj(lambda) w), or None
    residuals: np.ndarray             # ||J v - lambda v|| / (max(|lambda|, scale) ||v||)
    scale: float                      # the ||J|| scale the residuals are measured against


def _dense_norm(J):
    # sqrt(||J||_1 ||J||_inf), an O(n^2) upper bound on ||J||_2
    return float(np.sqrt(np.linalg.norm(J, 1)*np.linalg.norm(J, np.inf)))


def _probe_norm(mv, n):
    # max ||J r|| / ||r|| over a few seeded random complex r: a lower estimate of ||J||_2
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(_NORM_PROBES):
        r = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        best = max(best, float(np.linalg.norm(mv(r))/np.linalg.norm(r)))
    return best


def _residuals(apply, values, V, scale):
    # backward error: relative to max(|lambda|, ||J||), so a zero eigenvalue is measured
    # against the operator, not against itself
    R = apply(V) - V*values[None, :]
    den = np.maximum(np.abs(values), scale)*np.linalg.norm(V, axis=0)
    num = np.linalg.norm(R, axis=0)
    return np.where(den > 0, num/np.where(den > 0, den, 1.0), num)


def eig_dense(J):
    # full spectrum of a dense matrix, sorted by decreasing real part, with unit-norm right
    # AND left eigenvectors (left: w^H J = lambda w^H) and the residual of each right pair
    J = np.asarray(J)
    w, vl, vr = scipy.linalg.eig(J, left=True, right=True)
    order = np.argsort(-w.real, kind="stable")
    w, vl, vr = w[order], vl[:, order], vr[:, order]
    scale = _dense_norm(J)
    return EigResult(w, vr, vl, _residuals(lambda V: J @ V, w, vr, scale), scale)


def _complexify(op):
    # a real operator acting on complex vectors: A(a + ib) = A a + i A b
    if np.issubdtype(op.dtype, np.complexfloating):
        return op.matvec

    def mv(z):
        z = np.asarray(z).reshape(-1)
        return op.matvec(z.real) + 1j*op.matvec(z.imag)
    return mv


def shift_invert(J, sigma, k=6, *, tol=1e-12, arpack_maxiter=300, ncv=None, v0=None,
                 gmres_rtol=1e-12, restart=None, maxiter=100, M=None, residual_tol=None):
    """The k eigenvalues nearest sigma, by ARPACK on (J - sigma I)^-1.

    J is a dense array (one LU factorisation) or a scipy LinearOperator -- real
    (real_operator) or complex (ky_block_operator) -- whose shifted solves are GMRES to
    gmres_rtol, optionally preconditioned by M ~ (J - sigma I)^-1, in at most maxiter
    restarts. ARPACK runs to tol in at most arpack_maxiter restarts.

    Guaranteed on return: every pair satisfies
        ||J v - lambda v|| <= residual_tol * max(|lambda|, scale) * ||v||,
    residual_tol defaulting to 1e3*max(tol, eps), with scale = sqrt(||J||_1 ||J||_inf)
    (dense, >= ||J||_2) or the largest ||J r||/||r|| over a few random r (operator). The
    values are sorted by |lambda - sigma|. Not guaranteed: that ARPACK found the k NEAREST
    eigenvalues (compare against eig_dense where the block fits).

    Raises RuntimeError instead of returning otherwise: a dense J - sigma I with a pivot
    below 1e-12 (||J|| + |sigma|) (sigma is numerically an eigenvalue: move it), a GMRES
    solve short of gmres_rtol or non-finite, ARPACK non-convergence (typically k cutting
    through a near-degenerate cluster, e.g. the ~0 eigenvalues of an undamped nu = 0
    field), or any pair over the residual bound. Returns an EigResult (left=None).
    """
    sigma = complex(sigma)
    n = J.shape[0]
    if isinstance(J, np.ndarray):
        scale = _dense_norm(J)
        A = J.astype(np.complex128)             # the one n x n working copy
        A[np.diag_indices(n)] -= sigma
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            lu = scipy.linalg.lu_factor(A, overwrite_a=True, check_finite=False)
        pivot = float(np.abs(np.diag(lu[0])).min())
        if not pivot > _PIVOT_TOL*(scale + abs(sigma)):
            raise RuntimeError(
                f"stability.shift_invert: J - sigma I is numerically singular at sigma="
                f"{sigma} (smallest LU pivot {pivot:.3e} against ||J|| + |sigma| = "
                f"{scale + abs(sigma):.3e}): sigma is (numerically) an eigenvalue; move it")
        solve = lambda b: scipy.linalg.lu_solve(lu, np.asarray(b).reshape(-1))
        apply = lambda V: J @ V
    else:
        mv = _complexify(J)
        scale = _probe_norm(mv, n)
        shifted = spla.LinearOperator((n, n), matvec=lambda z: mv(z) - sigma*np.asarray(z).reshape(-1),
                                      dtype=np.complex128)

        def solve(b):
            x, info = spla.gmres(shifted, np.asarray(b).reshape(-1), rtol=gmres_rtol, atol=0.0,
                                 restart=restart, maxiter=maxiter, M=M)
            if info != 0 or not np.all(np.isfinite(x)):
                raise RuntimeError(f"stability.shift_invert: GMRES did not reach rtol="
                                   f"{gmres_rtol:g} (info={info}); loosen gmres_rtol, raise "
                                   f"maxiter/restart, or pass a preconditioner M -- or sigma "
                                   f"is (numerically) an eigenvalue: move it")
            return x

        def apply(V):
            return np.stack([mv(V[:, i]) for i in range(V.shape[1])], axis=1)
    if residual_tol is None:
        residual_tol = _RESIDUAL_FACTOR*max(tol, np.finfo(np.float64).eps)
    inv = spla.LinearOperator((n, n), matvec=solve, dtype=np.complex128)
    try:
        mu, V = spla.eigs(inv, k=k, which="LM", tol=tol, v0=v0, ncv=ncv, maxiter=arpack_maxiter)
    except spla.ArpackNoConvergence as err:
        found = sigma + 1.0/np.asarray(err.eigenvalues) if len(err.eigenvalues) else []
        raise RuntimeError(
            f"stability.shift_invert: ARPACK did not converge {k} eigenvalues to tol={tol:g} in "
            f"{arpack_maxiter} restarts ({len(found)} converged: {found}). Most likely k "
            f"reaches into a (near-)degenerate cluster -- lower k, move sigma, or raise "
            f"ncv") from err
    values = sigma + 1.0/mu
    order = np.argsort(np.abs(values - sigma), kind="stable")
    values, V = values[order], V[:, order]
    res = _residuals(apply, values, V, scale)
    if not np.all(res <= residual_tol):
        raise RuntimeError(
            f"stability.shift_invert: {int(np.sum(~(res <= residual_tol)))} of {k} returned "
            f"pairs are not eigenpairs to residual_tol={residual_tol:.1e} (residuals {res}, "
            f"values {values}, ||J|| scale {scale:.3e}). sigma={sigma} is likely too close to "
            f"an eigenvalue (move it), or the GMRES solves are too loose (tighten gmres_rtol)")
    return EigResult(values, V, None, res, scale)
