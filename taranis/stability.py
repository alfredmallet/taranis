# Linear stability about a frozen state x0, by autodiff (plans/AUTODIFF_PLAN.md, rungs 0/0b).
# construct_rhs returns N(f) only -- the k-local linear part L lives in kgrid.lin -- so the
# Jacobian of dt f = L f + N(f) is
#       J v = kgrid.lin.apply_L(v) + jvp(N, (x0,), (v,))[1]
# N' is jax's derivative of the solver's own RHS; no linearized equations are written.
# Equation-agnostic: any registered recipe (RMHD, GDI, CMHD in both density variables) in any
# geometry it supports (2D, FD-z, z_spectral). Read-only: nothing here mutates Parameters,
# the kgrid or a state, and the solver never imports this module. Unforced, single-process,
# non-sharded backends only; the CMHD expanding box is rejected (its RHS depends on t, so
# there is no autonomous J).
#
# Three vector spaces:
#   fields       the solver's (nfields, nz, nkx, nky) complex array. J is only REAL-linear on
#                it: irfft2 discards the anti-Hermitian part of the self-conjugate rows
#                (ky = 0 and Nyquist).
#   mode block   for an x0 invariant along y (and optionally x, and z under z_spectral), J is
#                block-diagonal in the wavevector along the invariant directions. For
#                0 < iky < ny/2 the block of one ky column -- or of one (kx, ky) or
#                (kx, ky, kz) mode when ikx / iz are fixed too -- maps its dealias-kept
#                entries to themselves complex-linearly: ky_block_matrix / ky_block_operator,
#                an ordinary complex matrix. The invariance is asserted, never assumed.
#   real coords  the dealias-kept modes of a REAL field, each conjugate pair counted once,
#                as (Re, Im) coordinates (RealCoords): the space for a general x0, where J
#                is a real matrix (real_matrix / real_operator).
# Dealias-masked modes are left out of every block: there J is L alone (N is masked), which
# only adds spurious damped eigenvalues.
#
# Eigensolvers: eig_dense (a dense block), shift_invert (eigenvalues nearest sigma; dense LU,
# or GMRES with an optional preconditioner factory -- diagonal_preconditioner,
# ky_averaged_preconditioner), and propagator_eigs, the default for the FASTEST-GROWING modes
# of a general x0: ARPACK on v -> exp(J T) v, integrated by the solver's own stepper.
import functools
import time
import warnings
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla

from . import _precision, propagators, timestepping
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
    if params.eqtype == "CMHD" and "expansion" in params.eqpars:
        # the EBM RHS reads a(t): J depends on t and exp(J T) is not a semigroup
        raise ValueError("stability: the CMHD expanding box is time-dependent (a(t) enters the "
                         "RHS), so there is no autonomous Jacobian to linearize; drop "
                         "eqpars['expansion']")


def _fields(x0, params):
    f = x0.fields if isinstance(x0, SimulationState) else x0
    f = jnp.asarray(f, dtype=_precision.ctype)
    shape = (params.nfields, params.nz, params.nx, params.ny//2 + 1)
    if f.shape != shape:
        raise ValueError(f"stability: x0 has shape {f.shape}, expected the fields shape {shape}")
    return f


def _state(params, fields):
    # a throwaway unforced state at t = 0 carrying `fields`
    nkx, nky = params.nx, params.ny//2 + 1
    return SimulationState(
        t=jnp.float64(0.0), fields=fields,
        forcing_state=jnp.zeros((params.n_ou, 2, nkx, nky), dtype=_precision.ctype),
        forcing_key=jax.random.key(0),
        forcing_scale=jnp.zeros((params.n_ou,), dtype=_precision.ftype))


def _nonlinear(params):
    # N(fields, kgrid): the RHS terms construct_rhs sums, on a throwaway unforced state
    rhs = construct_rhs(equation_registry[params.eqtype])

    def N(fields, kgrid):
        return rhs(_state(params, fields), kgrid, params)[0]
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


def _is_index(i):
    # a genuine integer: bool is an int subclass (True would silently mean index 1)
    return isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))


def _check_block_args(params, iky, ikx, iz):
    if not (_is_index(iky) and 0 < iky < params.ny/2):
        raise ValueError(f"stability: iky must be an int with 0 < iky < ny/2 = {params.ny/2} "
                         f"(ky = 0 and Nyquist are self-conjugate rows, where J is not "
                         f"complex-linear); got {iky!r}")
    if ikx is not None and not (_is_index(ikx) and 0 <= ikx < params.nx):
        raise ValueError(f"stability: ikx must be an int index in [0, nx); got {ikx!r}")
    if iz is not None:
        if not (params.spatial_dimensions == 3 and params.z_spectral):
            # under finite-difference z (or in 2D) axis 1 is real-space z: a z-invariant x0
            # commutes with z shifts but does not decouple the z points
            raise ValueError("stability: a fixed iz (a kz mode) needs dims=3 with "
                             "z_spectral=True, where axis 1 is kz")
        if not (_is_index(iz) and 0 <= iz < params.nz):
            raise ValueError(f"stability: iz must be an int index in [0, nz); got {iz!r}")


def ky_block_index(kgrid, params, iky, *, ikx=None, iz=None):
    # (n, 3) int array of the (field, iz, ikx) entries of ky column iky that the dealias
    # mask keeps, in C order: the row/column order of every block. ikx and/or iz (z_spectral:
    # a kz index) restrict it to one wavevector along those axes too.
    _check_block_args(params, iky, ikx, iz)
    col = np.array(_kept(kgrid, params)[..., iky])
    if ikx is not None:
        col[:, np.arange(params.nx) != ikx] = False
    if iz is not None:
        col[np.arange(params.nz) != iz, :] = False
    if not col.any():
        raise ValueError(f"stability: block (iky={iky}, ikx={ikx}, iz={iz}) is entirely "
                         f"dealias-masked")
    return np.argwhere(np.broadcast_to(col, (params.nfields,) + col.shape))


def _nonzero_content(a, axis):
    # (largest |x0| off the zero wavenumber along axis, the round-off allowance, max |x0|)
    scale = float(np.max(np.abs(a))) if a.size else 0.0
    tol = 1e3*np.finfo(a.real.dtype).eps*scale
    if a.shape[axis] == 1:
        return 0.0, tol, scale
    return float(np.max(np.abs(np.take(a, np.arange(1, a.shape[axis]), axis=axis)))), tol, scale


def _assert_invariant(x0, ikx=None, iz=None):
    # a block needs x0 invariant along y, plus x when ikx is fixed and z (kz axis) when iz is:
    # x0 may then carry only the zero wavenumber along each of those axes
    a = np.asarray(x0)
    for name, axis, fixed in (("ky", 3, True), ("kx", 2, ikx is not None),
                              ("kz", 1, iz is not None)):
        if not fixed:
            continue
        worst, tol, scale = _nonzero_content(a, axis)
        if worst > tol:
            raise ValueError(f"stability: this block needs a {name}-independent x0 (J is "
                             f"then block-diagonal in {name}); x0's {name} != 0 content is "
                             f"{worst:.3e} against max |x0| = {scale:.3e}")


# Index arrays ride into the compiled functions as TRACED arguments, so one compile per
# (Parameters, space kind, block size) serves every block and sigma.

def _block_args(idx, iky):
    # traced index arrays (field, iz, ikx, iky) of a block's entries
    return tuple(jnp.asarray(a) for a in (idx[:, 0], idx[:, 1], idx[:, 2],
                                          np.full(len(idx), iky)))


def _fields_shape(params):
    return (params.nfields, params.nz, params.nx, params.ny//2 + 1)


def _block_embed(params):
    shape = _fields_shape(params)

    def embed(u, args):
        return jnp.zeros(shape, dtype=_precision.ctype).at[args].set(u)
    return embed


def _block_gather(g, args):
    return g[args]


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


class _Fns(NamedTuple):
    # the compiled maps of one (Parameters, space kind); every one takes (..., x0, kgrid, args)
    embed: object            # (u, args) -> fields (not jitted)
    gather: object           # (g, args) -> (n,) (not jitted)
    mv: object               # J
    rmv: object              # J^H (block) / J^T (real coords)
    probe: object            # (cols, ...) -> images of the unit vectors e_cols
    batch: object            # (U (b, n), ...) -> images of the rows of U
    dtype: object            # the space's jax dtype


@functools.lru_cache(maxsize=32)
def _space_fns(params, kind):
    # kind "block": complex (field, iz, ikx) entries of one ky column; "real": RealCoords
    jv = _jacobian_action(params)
    if kind == "block":
        embed, gather, dt = _block_embed(params), _block_gather, _precision.ctype
    else:
        embed, gather, dt = _real_unpack(params), _real_pack, _precision.ftype

    def act(u, x0, kgrid, args):
        return gather(jv(x0, embed(u, args), kgrid), args)

    if kind == "block":
        jhu = _adjoint_action(params)

        def ract(u, x0, kgrid, args):
            return gather(jhu(x0, embed(u, args), kgrid), args)
    else:
        def ract(w, x0, kgrid, args):
            _, pullback = jax.vjp(lambda u: act(u, x0, kgrid, args), jnp.zeros_like(w))
            return pullback(w)[0]

    def probe(cols, x0, kgrid, args):
        n = _space_size(kind, args)
        return jax.vmap(lambda j: act(jax.nn.one_hot(j, n, dtype=dt), x0, kgrid, args))(cols)

    batch = jax.vmap(act, in_axes=(0, None, None, None))
    return _Fns(embed, gather, jax.jit(act), jax.jit(ract), jax.jit(probe), jax.jit(batch), dt)


def _space_size(kind, args):
    return args[0].shape[0] if kind == "block" else args[0].shape[0] + args[1].shape[0]


def ky_block_matrix(x0, kgrid, params, iky, chunk=None, *, ikx=None, iz=None):
    # dense J restricted to the kept entries of ky column iky (order: ky_block_index), for a
    # ky-independent x0 -- or of one (kx, ky[, kz]) mode for an x0 independent of x [and z]:
    # one jvp per basis vector, vmapped in chunks. Returns (J_block complex128 (n, n), idx).
    _check_supported(params)
    x0 = _fields(x0, params)
    idx = ky_block_index(kgrid, params, iky, ikx=ikx, iz=iz)
    _assert_invariant(x0, ikx, iz)
    args = _block_args(idx, iky)
    fns = _space_fns(params, "block")
    B = _dense_from_probes(lambda cols: fns.probe(cols, x0, kgrid, args), len(idx),
                           min(_probe_chunk(params, chunk), len(idx)), np.complex128)
    return B, idx


def _host_op(fns, n, np_dtype, x0, kgrid, args, adjoint=True):
    # scipy LinearOperator over the compiled maps (matvec J, rmatvec J^H / J^T)
    def wrap(fn):
        def apply(u):
            u = np.asarray(u).reshape(-1)
            return np.array(fn(jnp.asarray(u, dtype=fns.dtype), x0, kgrid, args), dtype=np_dtype)
        return apply
    return spla.LinearOperator((n, n), matvec=wrap(fns.mv),
                               rmatvec=wrap(fns.rmv) if adjoint else None, dtype=np_dtype)


def ky_block_operator(x0, kgrid, params, iky, *, ikx=None, iz=None):
    # matrix-free form of ky_block_matrix: a complex scipy LinearOperator (matvec J,
    # rmatvec J^H) on the kept entries of the block. Returns (op, idx).
    _check_supported(params)
    x0 = _fields(x0, params)
    idx = ky_block_index(kgrid, params, iky, ikx=ikx, iz=iz)
    _assert_invariant(x0, ikx, iz)
    op = _host_op(_space_fns(params, "block"), len(idx), np.complex128, x0, kgrid,
                  _block_args(idx, iky))
    return op, idx


# ---------------------------------------------------------------- real coordinates

def _real_unpack(params):
    # (u, (re, im, mirror_src, mirror_dst)) -> the fields array of the real field u
    shape = _fields_shape(params)
    size = int(np.prod(shape))

    def unpack(u, args):
        re, im, msrc, mdst = args
        nre = re.shape[0]
        g = jnp.zeros(size, dtype=_precision.ctype)
        g = g.at[re].set(u[:nre].astype(_precision.ftype))
        g = g.at[im].add(1j*u[nre:].astype(_precision.ftype))
        g = g.at[mdst].set(jnp.conj(g[msrc]))
        return jnp.reshape(g, shape)
    return unpack


def _real_pack(fields, args):
    re, im = args[0], args[1]
    g = jnp.reshape(fields, (-1,))
    return jnp.concatenate([jnp.real(g[re]), jnp.imag(g[im])])


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

    @property
    def args(self):
        # the index arrays as traced arguments of the compiled maps
        return tuple(jnp.asarray(a) for a in (self.re, self.im, self.mirror_src,
                                              self.mirror_dst))

    def pack(self, fields):
        return _real_pack(fields, (self.re, self.im))

    def unpack(self, u):
        size = int(np.prod(self.shape))
        nre = len(self.re)
        g = jnp.zeros(size, dtype=_precision.ctype)
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


def real_operator(x0, kgrid, params, coords=None):
    # matrix-free J in real coordinates, for any x0: a real scipy LinearOperator (matvec J,
    # rmatvec J^T by vjp). Returns (op, coords).
    _check_supported(params)
    x0 = _fields(x0, params)
    coords = real_coords(kgrid, params) if coords is None else coords
    op = _host_op(_space_fns(params, "real"), coords.n, np.float64, x0, kgrid, coords.args)
    return op, coords


def real_matrix(x0, kgrid, params, coords=None, chunk=None):
    # dense J in real coordinates (small grids): one jvp per coordinate. Returns (M, coords).
    _check_supported(params)
    x0 = _fields(x0, params)
    coords = real_coords(kgrid, params) if coords is None else coords
    fns, args = _space_fns(params, "real"), coords.args
    M = _dense_from_probes(lambda cols: fns.probe(cols, x0, kgrid, args), coords.n,
                           min(_probe_chunk(params, chunk), coords.n), np.float64)
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
    stats: Optional[dict] = None      # shift_invert's solve counts (GMRES iterations)


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
    restarts. M is a LinearOperator, or a factory M(sigma) -> LinearOperator
    (diagonal_preconditioner, ky_averaged_preconditioner). ARPACK runs to tol in at most
    arpack_maxiter restarts. result.stats counts the solves and the GMRES inner iterations.

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
    stats = dict(solves=0, gmres_iters=0)
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
        def solve(b):
            stats["solves"] += 1
            return scipy.linalg.lu_solve(lu, np.asarray(b).reshape(-1))
        apply = lambda V: J @ V
    else:
        mv = _complexify(J)
        scale = _probe_norm(mv, n)
        shifted = spla.LinearOperator((n, n), matvec=lambda z: mv(z) - sigma*np.asarray(z).reshape(-1),
                                      dtype=np.complex128)
        if M is not None and not isinstance(M, (spla.LinearOperator, np.ndarray)):
            M = M(sigma)

        def count(_):
            stats["gmres_iters"] += 1

        def solve(b):
            stats["solves"] += 1
            x, info = spla.gmres(shifted, np.asarray(b).reshape(-1), rtol=gmres_rtol, atol=0.0,
                                 restart=restart, maxiter=maxiter, M=M, callback=count,
                                 callback_type="pr_norm")
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
    return EigResult(values, V, None, res, scale, stats)


# ---------------------------------------------------------------- spaces

class _Space(NamedTuple):
    # a coordinate space for J: a mode block or the real coordinates
    n: int
    kind: str                # "block" or "real"
    np_dtype: type           # np.complex128 (block) or np.float64 (real coords)
    args: tuple              # the traced index arrays
    index: object            # the block's idx array, or the RealCoords


def _space(x0, kgrid, params, iky, ikx, iz, coords):
    # the mode block (iky given; asserts x0's invariance when x0 is given) or real coordinates
    if iky is None:
        if ikx is not None or iz is not None:
            raise ValueError("stability: ikx / iz select a mode block and need iky too")
        coords = real_coords(kgrid, params) if coords is None else coords
        return _Space(coords.n, "real", np.float64, coords.args, coords)
    idx = ky_block_index(kgrid, params, iky, ikx=ikx, iz=iz)
    if x0 is not None:
        _assert_invariant(x0, ikx, iz)
    return _Space(len(idx), "block", np.complex128, _block_args(idx, iky), idx)


# ---------------------------------------------------------------- propagator Arnoldi

# convergence order of each IF-LSRK/RK scheme (the IMEX tableaux carry their own .order)
_SCHEME_ORDER = {"rk44": 4, "lsrk33": 3, "lsrk54": 4}

# propagator_eigs' default residual_tol: the largest scale-relative residual on J
# (||J v - lambda v|| / (max(|lambda|, scale) ||v||)) a returned pair may carry. That residual
# IS the stepper's O(dt^p) time error -- not round-off -- so the bound is a statement about dt,
# not about tol. Measured 2026-09-28 over every propagator_eigs call in
# tests/test_stability_general.py: 1.2e-15 (N' = 0, exact IF) to 1.7e-6 (RMHD cos x tearing,
# lsrk54 dt = 0.5) at fp64, 1.1e-6 worst at fp32 (CMHD mode block, dt = 0.02). The broken
# cases of the rung-0b review sit at 2e-3 and 1.4e-2 (IMEX on the +-i kz wave L at dt = 0.5,
# 1 -- now also rejected outright, _check_scheme_on_L) and 0.38 (lsrk54 past its explicit
# stability limit on CMHD's N', omega_fast dt = 3.9; 0.48 in test_propagator_loud_default).
# 1e-4 leaves ~60x over the worst legitimate call and ~20x under the mildest broken one; at
# fp32 it is ~1e3 eps, so round-off never trips it.
_PROPAGATOR_RESIDUAL_TOL = 1e-4


def _scheme_order(name):
    _, scheme = timestepping.get_scheme(name)
    return int(getattr(scheme, "order", None) or _SCHEME_ORDER[name])


class PropagatorInfo(NamedTuple):
    T: float                 # integration time: exp(J T) is approximated
    dt: float                # the fixed step actually used, T/nsteps
    nsteps: int
    scheme: str
    order: int               # the scheme's order p: lambda carries an O(dt^p) error


def _propagator_info(T, dt, scheme):
    if not (T > 0 and dt > 0):
        raise ValueError(f"stability: T and dt must be > 0, got T={T!r}, dt={dt!r}")
    nsteps = max(1, int(np.ceil(T/dt - 1e-9)))
    return PropagatorInfo(float(T), float(T)/nsteps, nsteps, scheme, _scheme_order(scheme))


def _check_scheme_on_L(kgrid, name):
    # CB-IMEX treats ALL of L implicitly with an L-stable solve, which artificially damps an
    # oscillatory L at |omega| dt >~ 1 (CLAUDE.md: never IMEX a wave-dominated L) -- in
    # propagator_eigs that reorders the spectrum silently (a faster wave mode damped out of
    # the k fastest). Allowed only on a REAL diagonal L (pure dissipation: 2D and FD-z RMHD,
    # CMHD), where the implicit solve is the legitimate stiff-dissipation path. Nothing is
    # lost by the rule: an IF scheme applies any L exactly.
    _, scheme = timestepping.get_scheme(name)
    if not isinstance(scheme, timestepping.IMEX_Scheme):
        return
    lin = kgrid.lin
    if isinstance(lin, propagators.IdentityOperator):
        return
    if isinstance(lin, propagators.DiagonalOperator) and not np.any(np.imag(np.asarray(lin.L))):
        return
    raise ValueError(f"stability: the IMEX scheme {name!r} treats L implicitly, and this L "
                     f"({type(lin).__name__}) is not a real diagonal (pure dissipation): it "
                     f"carries oscillatory (wave / drift) terms, which an L-stable solve "
                     f"damps artificially at |omega| dt >~ 1, reordering the spectrum. Use an "
                     f"IF scheme (lsrk54, rk44, lsrk33), which applies L exactly")


@functools.lru_cache(maxsize=32)
def _flow_fn(params, kind, info):
    # compiled (u, x0, kgrid, args) -> exp(J T) u: the solver's stepper run info.nsteps times
    # at the fixed info.dt on the linearized system df/dt = L f + N'(x0) f. The stepper's rhs
    # is the frozen tangent map of N (one jax.linearize, outside the step scan); the stepper
    # applies L itself, exactly as in a solver run (hoisted exp ops when the backend is).
    stepper, scheme = timestepping.get_scheme(info.scheme)
    N = _nonlinear(params)
    fns = _space_fns(params, kind)
    dt = info.dt

    def flow(u, x0, kgrid, args):
        _, dN = jax.linearize(lambda g: N(g, kgrid), x0)

        def rhs(state, kgrid_, params_):
            return dN(state.fields).astype(_precision.ctype), None
        exp_ops = timestepping.stage_exp_ops(kgrid, params, scheme, stepper, dt)

        def body(s, _):
            return stepper(s, kgrid, params, rhs, None, scheme, dt_override=dt,
                           exp_ops=exp_ops), None
        s, _ = jax.lax.scan(body, _state(params, fns.embed(u, args)), None, length=info.nsteps)
        return fns.gather(s.fields, args)
    return jax.jit(flow)


def propagator_operator(x0, kgrid, params, *, T, dt, scheme="lsrk54", iky=None, ikx=None,
                        iz=None, coords=None):
    """exp(J T) as a scipy LinearOperator, integrated by the solver's own stepper.

    The space is the mode block (iky [, ikx, iz]) or, with iky None, the real coordinates.
    dt is rounded down so that nsteps*dt = T exactly. Forcing is off and there are no
    particles (the harness rejects both). An IMEX scheme is accepted only on a real diagonal
    L (pure dissipation), ValueError otherwise (_check_scheme_on_L). Returns (op, info,
    index) with info a PropagatorInfo and index the block's idx or the RealCoords.
    """
    _check_supported(params)
    x0 = _fields(x0, params)
    sp = _space(x0, kgrid, params, iky, ikx, iz, coords)
    info = _propagator_info(T, dt, scheme)
    _check_scheme_on_L(kgrid, scheme)
    flow = _flow_fn(params, sp.kind, info)
    jdt = _space_fns(params, sp.kind).dtype

    def apply(u):
        u = np.asarray(u).reshape(-1)
        return np.array(flow(jnp.asarray(u, dtype=jdt), x0, kgrid, sp.args), dtype=sp.np_dtype)
    op = spla.LinearOperator((sp.n, sp.n), matvec=apply, dtype=sp.np_dtype)
    return op, info, sp.index


class PropagatorEigResult(NamedTuple):
    values: np.ndarray       # lambda = log(mu)/T, branch fixed by the Rayleigh quotient
    right: np.ndarray        # eigenvectors (columns), in the block / real-coordinate space
    left: Optional[np.ndarray]    # always None
    residuals: np.ndarray    # ||J v - lambda v|| / (max(|lambda|, scale) ||v||), on J itself
    scale: float             # the ||J|| estimate the residuals are relative to
    mu: np.ndarray           # the eigenvalues of exp(J T) the values come from
    rayleigh: np.ndarray     # v^H J v / v^H v
    aliased: np.ndarray      # bool: |Im lambda| T within branch_margin*pi of pi, or past it
    info: PropagatorInfo
    applications: int        # exp(J T) applications (each one is info.nsteps linear steps)
    seconds: float           # wall time of the eigensolve (compile included)
    index: object            # the block's idx or the RealCoords


def propagator_eigs(x0, kgrid, params, *, T, dt, k=6, scheme="lsrk54", iky=None, ikx=None,
                    iz=None, coords=None, tol=1e-10, ncv=None, maxiter=300, v0=None,
                    branch_margin=0.1, residual_tol=_PROPAGATOR_RESIDUAL_TOL):
    """The k fastest-growing eigenvalues of J (largest Re lambda), by ARPACK on exp(J T).

    The default solver for the fastest modes of a general x0. exp(J T) is nsteps = T/dt
    fixed steps of the solver's stepper `scheme` on the linearized system
    (propagator_operator): no linear solve, and the stiffness of L never enters an IF scheme
    (L is applied exactly). N' is stepped EXPLICITLY, so dt must respect the stepper's
    stability limit on N' -- for CMHD the waves live in N, so that is the fast-wave CFL.
    IMEX schemes are rejected unless L is a real diagonal (propagator_operator).
    The largest |mu| of exp(J T) are the largest Re lambda, lambda = log(mu)/T.

    Accuracy: the one-step map is R(dt) = exp(J dt) + O(dt^(p+1)), so lambda carries an
    O(dt^p) error (p = info.order) that does not depend on T; the residuals are measured on
    J itself and ARE that error (for a normal J they bound it), not round-off. At x0 with
    N' = 0 an IF scheme is exact. T sets the separation |mu_k/mu_(k+1)| ARPACK works with,
    and the branch: arg(mu) = Im(lambda) T mod 2 pi. The branch is chosen by the Rayleigh
    quotient v^H J v / v^H v; `aliased` flags |Im lambda| T within branch_margin*pi of pi or
    beyond (shorten T). When k >= n - 1 (ARPACK's limit) exp(J T) is formed on the n basis
    vectors and diagonalised densely, keeping the k largest |mu|; k > n is a ValueError.

    Loud by default: RuntimeError when any returned pair's residual on J exceeds
    residual_tol (default 1e-4, _PROPAGATOR_RESIDUAL_TOL: ~60x over the worst O(dt^p) error
    of the gate suite, far under what a dt past the explicit limit on N' or a branch
    collision produces). A residual over it means the pairs are not J's to that accuracy:
    reduce dt (or change T when two modes share one mu). residual_tol=np.inf is the explicit
    opt-out, for studying the discrete propagator itself; None means the default.

    Krylov limitation: a single start vector spans ONE direction of a semisimple multiple
    eigenvalue, so an exactly repeated mu is returned once and the next distinct eigenvalue
    takes the other slot -- the k = 0 means (all at lambda = 0), and, on a ky column under
    z_spectral with a z-INDEPENDENT x0 (J block-diagonal in kz), any eigenvalue shared by the
    kz and -kz blocks: every eigenvalue of an equation whose J depends on kz only through kz^2
    (GDI's D_par kz^2), the real eigenvalues of RMHD (whose -kz spectrum is the conjugate of
    the kz one). Such an eigenvalue may be listed fewer times than its multiplicity, with
    eigenvectors mixing the kz blocks; that case is detected (an eigenvector of a
    z-independent x0 spread over m kz, its eigenvalue returned fewer than m times) and warned
    about. Pass iz to take one kz block at a time instead. The dense branch lists every
    copy. Degeneracies with no block structure to reveal them (e.g. +-kx at x0 = 0) are not
    detected.

    Raises RuntimeError on ARPACK non-convergence and on any residual over residual_tol.
    Sorted by decreasing Re lambda.
    """
    t0 = time.perf_counter()
    if residual_tol is None:
        residual_tol = _PROPAGATOR_RESIDUAL_TOL
    op, info, index = propagator_operator(x0, kgrid, params, T=T, dt=dt, scheme=scheme,
                                          iky=iky, ikx=ikx, iz=iz, coords=coords)
    x0 = _fields(x0, params)
    sp = _space(None, kgrid, params, iky, ikx, iz, index if iky is None else None)
    n = sp.n
    if not (_is_index(k) and 1 <= k <= n):
        raise ValueError(f"stability.propagator_eigs: k must be an int with 1 <= k <= n = {n} "
                         f"(the size of the space); got {k!r}")
    dense = k >= n - 1
    if dense:
        E = np.stack([op.matvec(e) for e in np.eye(n, dtype=sp.np_dtype)], axis=1)
        mu, V = np.linalg.eig(E)
        keep = np.argsort(-np.abs(mu), kind="stable")[:k]
        mu, V = mu[keep], V[:, keep]
        apps = n
    else:
        if v0 is None:
            rng = np.random.default_rng(0)
            v0 = rng.standard_normal(n)
            if sp.np_dtype == np.complex128:
                v0 = v0 + 1j*rng.standard_normal(n)
        counter = [0]

        def mv(u):
            counter[0] += 1
            return op.matvec(u)
        cop = spla.LinearOperator((n, n), matvec=mv, dtype=sp.np_dtype)
        try:
            mu, V = spla.eigs(cop, k=k, which="LM", tol=tol, v0=v0, ncv=ncv, maxiter=maxiter)
        except spla.ArpackNoConvergence as err:
            raise RuntimeError(
                f"stability.propagator_eigs: ARPACK did not converge {k} eigenvalues of "
                f"exp(J T) to tol={tol:g} in {maxiter} restarts ({len(err.eigenvalues)} "
                f"converged). Most likely k cuts through a cluster of equal |mu| -- change k, "
                f"lengthen T to separate the moduli, or raise ncv") from err
        apps = counter[0]
    Jc = _complexify(_host_op(_space_fns(params, sp.kind), n, sp.np_dtype, x0, kgrid, sp.args,
                              adjoint=False))
    JV = np.stack([Jc(V[:, i]) for i in range(V.shape[1])], axis=1)
    ray = np.einsum("ij,ij->j", V.conj(), JV)/np.einsum("ij,ij->j", V.conj(), V)
    with np.errstate(divide="ignore"):
        lam = np.log(mu.astype(np.complex128))/info.T
    wrap = np.round((ray.imag - lam.imag)*info.T/(2*np.pi))
    lam = lam + 2j*np.pi*wrap/info.T
    aliased = np.abs(lam.imag)*info.T > np.pi*(1.0 - branch_margin)
    scale = _probe_norm(Jc, n)
    den = np.maximum(np.abs(lam), scale)*np.linalg.norm(V, axis=0)
    res = np.linalg.norm(JV - V*lam[None, :], axis=0)/den
    order = np.argsort(-lam.real, kind="stable")
    out = PropagatorEigResult(lam[order], V[:, order], None, res[order], scale, mu[order],
                              ray[order], aliased[order], info, apps,
                              time.perf_counter() - t0, index)
    if not np.all(out.residuals <= residual_tol):
        bad = ~(out.residuals <= residual_tol)
        raise RuntimeError(
            f"stability.propagator_eigs: {int(bad.sum())} of {len(bad)} returned pairs have a "
            f"residual on J over residual_tol={residual_tol:.1e} (residuals {out.residuals}, "
            f"values {out.values}; scheme {info.scheme}, dt={info.dt:.4g}, T={info.T:.4g}). "
            f"They are not eigenpairs of J to that accuracy: the O(dt^{info.order}) time error "
            f"dominates, or dt is past the stepper's explicit stability limit on N' (reduce "
            f"dt), or two modes share one mu (change T). residual_tol=np.inf opts out")
    if (not dense and sp.kind == "block" and iz is None and params.spatial_dimensions == 3
            and params.z_spectral):
        _warn_collapsed_kz(out, index, x0)
    return out


# an Arnoldi eigenvector of a kz-block-diagonal J whose weight on a kz, relative to its dominant
# kz, exceeds this fraction (in norm) counts that kz as spanned
_KZ_MIX_TOL = 1e-3
# returned values within this fraction of max(|lambda|, scale) are copies of one eigenvalue
_KZ_COPY_TOL = 1e-6


def _warn_collapsed_kz(out, idx, x0):
    # z-independent x0 under z_spectral: J is block-diagonal in kz, so a simple eigenvalue's
    # eigenvector lives on one kz. An eigenvector spread over m kz belongs to an eigenvalue
    # shared by those m blocks (to numerical precision); if it was returned fewer than m
    # times, Krylov collapsed copies of it (propagator_eigs' docstring) -- warn
    worst, tol, _ = _nonzero_content(np.asarray(x0), 1)
    izs = np.unique(idx[:, 1])
    if worst > tol or len(izs) < 2:
        return
    W = np.stack([np.linalg.norm(out.right[idx[:, 1] == i], axis=0) for i in izs])
    spans = W > _KZ_MIX_TOL*np.max(W, axis=0, keepdims=True)
    lam = out.values
    ctol = _KZ_COPY_TOL*np.maximum(np.abs(lam), out.scale)
    short = []
    for j in np.flatnonzero(spans.sum(axis=0) > 1):
        copies = np.abs(lam - lam[j]) <= ctol[j]
        nkz = int(np.any(spans[:, copies], axis=1).sum())
        if copies.sum() < nkz:
            short.append((lam[j], int(copies.sum()), nkz))
    if short:
        listed = "; ".join(f"{v:.6g} returned {c}x, spans {m} kz" for v, c, m in short)
        warnings.warn(
            f"stability.propagator_eigs: x0 is z-independent, so J is block-diagonal in kz, "
            f"but these eigenvectors spread over several kz blocks and their eigenvalue was "
            f"returned fewer times than the blocks it spans ({listed}): each is (numerically) "
            f"an eigenvalue shared by those kz blocks (e.g. the +-kz pair), and a single-start "
            f"Krylov space missed copies of it -- the k returned are short of its "
            f"multiplicity. Pass iz to take one kz block at a time", RuntimeWarning,
            stacklevel=3)


# ---------------------------------------------------------------- preconditioners
# Factories sigma -> LinearOperator ~ (J - sigma I)^-1 in a space, for shift_invert's M.
# On real coordinates J is a real matrix acting on complex vectors z = a + i b. For a map
# that acts on each canonical entry f as f -> A f (complex-linear: L, and a ky > 0 column),
# the complexification is diagonal in f+ = z_re + i z_im, f- = z_re - i z_im, acting as A
# on f+ and conj(A) on f-; so (. - sigma)^-1 is (A - sigma)^-1 on f+ and
# conj((A - conj(sigma))^-1 conj(.)) on f-, and z_re = (F+ + F-)/2, z_im = (F+ - F-)/(2i).


@functools.lru_cache(maxsize=32)
def _diag_fn(params, kind):
    # compiled (z, sigma, lin, args) -> (L - sigma)^-1 z in the space, z complex
    shape = _fields_shape(params)
    size = int(np.prod(shape))

    def solve(lin, f, sigma):
        # (L - sigma)^-1 = -(1/sigma) (I - L/sigma)^-1
        return -lin.solve_shifted(f, 1.0/sigma)/sigma

    if kind == "block":
        def fn(z, sigma, lin, args):
            f = jnp.zeros(shape, dtype=_precision.ctype).at[args].set(z)
            return solve(lin, f, sigma)[args]
        return jax.jit(fn)

    def place(re, im, zr, zi):
        g = jnp.zeros(size, dtype=_precision.ctype).at[re].set(zr)
        return jnp.reshape(g.at[im].add(1j*zi), shape)

    def fn(z, sigma, lin, args):
        re, im = args[0], args[1]
        nre = re.shape[0]
        zr, zi = z[:nre], z[nre:]
        fp = solve(lin, place(re, im, zr, zi), sigma).reshape(-1)
        fm = jnp.conj(solve(lin, jnp.conj(place(re, im, zr, -zi)), jnp.conj(sigma))).reshape(-1)
        return jnp.concatenate([(fp[re] + fm[re])/2, (fp[im] - fm[im])/2j])
    return jax.jit(fn)


def diagonal_preconditioner(kgrid, params, *, iky=None, ikx=None, iz=None, coords=None):
    """Factory sigma -> (L - sigma I)^-1 (L = kgrid.lin, k-local) in the block (iky [, ikx,
    iz]) or the real coordinates, through the operator's solve_shifted. Exact where N' is
    small against L - sigma; no help where L ~ 0 (nu = 0) or N' dominates."""
    sp = _space(None, kgrid, params, iky, ikx, iz, coords)
    fn = _diag_fn(params, sp.kind)

    def factory(sigma):
        sigma = complex(sigma)
        if sigma == 0:
            raise ValueError("stability.diagonal_preconditioner: sigma = 0 (L is singular at "
                             "k = 0)")
        s = jnp.asarray(sigma, dtype=_precision.ctype)

        def apply(z):
            z = np.asarray(z).reshape(-1)
            return np.array(fn(jnp.asarray(z, dtype=_precision.ctype), s, kgrid.lin, sp.args),
                            dtype=np.complex128)
        return spla.LinearOperator((sp.n, sp.n), matvec=apply, dtype=np.complex128)
    return factory


class KyAveragedPreconditioner:
    """Factory sigma -> (J(xbar) - sigma I)^-1 on the real coordinates, xbar the y-average
    (ky = 0 part) of x0. J(xbar) is block-diagonal in ky: a dense complex block per ky > 0
    column and one dense real block for the ky = 0 row, all found together by probing the
    j-th entry of every block at once (max block size probes, not their sum). Exact for a
    ky-independent x0; for a 2D x0 it drops the ky coupling. Each sigma costs one LU per
    block (two per ky > 0 column: sigma and conj(sigma)).
    """

    def __init__(self, x0, kgrid, params, coords=None, chunk=None):
        _check_supported(params)
        x0 = _fields(x0, params)
        xbar = x0.at[..., 1:].set(0.0)
        coords = real_coords(kgrid, params) if coords is None else coords
        self.coords, self.n = coords, coords.n
        size, nre = int(np.prod(coords.shape)), len(coords.re)
        pos = np.full(size, -1)
        pos[coords.re] = np.arange(nre)
        pos_im = np.full(size, -1)
        pos_im[coords.im] = nre + np.arange(len(coords.im))
        self.cols = []
        kept = _kept(kgrid, params)
        for iky in range(1, params.ny//2 + 1):
            if iky < params.ny/2 and kept[..., iky].any():
                idx = ky_block_index(kgrid, params, iky)
                flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2],
                                             np.full(len(idx), iky)), coords.shape)
                re_c, im_c = pos[flat], pos_im[flat]
                if (re_c < 0).any() or (im_c < 0).any():
                    raise AssertionError("stability: a ky > 0 entry lacks a real coordinate")
                self.cols.append((re_c, im_c))
        iy_re = np.unravel_index(coords.re, coords.shape)[3]
        iy_im = np.unravel_index(coords.im, coords.shape)[3]
        self.row0 = np.concatenate([np.flatnonzero(iy_re == 0),
                                    nre + np.flatnonzero(iy_im == 0)])
        covered = len(self.row0) + 2*sum(len(r) for r, _ in self.cols)
        if covered != self.n:
            raise AssertionError(f"stability: blocks cover {covered} of {self.n} coordinates")
        nprobe = max([len(self.row0)] + [len(r) for r, _ in self.cols])
        fns, args = _space_fns(params, "real"), coords.args
        chunk = min(_probe_chunk(params, chunk), nprobe)
        self.blocks = [np.empty((len(r), len(r)), dtype=np.complex128) for r, _ in self.cols]
        self.M0 = np.empty((len(self.row0), len(self.row0)))
        for j0 in range(0, nprobe, chunk):
            js = np.arange(j0, j0 + chunk)
            U = np.zeros((chunk, self.n))
            for re_c, _ in self.cols:
                ok = js < len(re_c)
                U[np.flatnonzero(ok), re_c[js[ok]]] = 1.0
            ok = js < len(self.row0)
            U[np.flatnonzero(ok), self.row0[js[ok]]] = 1.0
            out = np.asarray(fns.batch(jnp.asarray(U, dtype=_precision.ftype), xbar, kgrid,
                                       args), dtype=np.float64)
            for b, (re_c, im_c) in enumerate(self.cols):
                for i, j in enumerate(js):
                    if j < len(re_c):
                        self.blocks[b][:, j] = out[i, re_c] + 1j*out[i, im_c]
            for i, j in enumerate(js):
                if j < len(self.row0):
                    self.M0[:, j] = out[i, self.row0]

    def __call__(self, sigma):
        sigma = complex(sigma)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            eye = lambda m: np.eye(m.shape[0])
            lu0 = scipy.linalg.lu_factor(self.M0 - sigma*eye(self.M0))
            lus = [(scipy.linalg.lu_factor(B - sigma*eye(B)),
                    scipy.linalg.lu_factor(B - np.conj(sigma)*eye(B))) for B in self.blocks]

        def apply(z):
            z = np.asarray(z, dtype=np.complex128).reshape(-1)
            out = np.empty(self.n, dtype=np.complex128)
            out[self.row0] = scipy.linalg.lu_solve(lu0, z[self.row0])
            for (re_c, im_c), (lu, lub) in zip(self.cols, lus):
                zr, zi = z[re_c], z[im_c]
                Fp = scipy.linalg.lu_solve(lu, zr + 1j*zi)
                Fm = np.conj(scipy.linalg.lu_solve(lub, np.conj(zr - 1j*zi)))
                out[re_c] = (Fp + Fm)/2
                out[im_c] = (Fp - Fm)/2j
            return out
        return spla.LinearOperator((self.n, self.n), matvec=apply, dtype=np.complex128)


def ky_averaged_preconditioner(x0, kgrid, params, coords=None, chunk=None):
    # KyAveragedPreconditioner(x0, ...): the factory, blocks built once for every sigma
    return KyAveragedPreconditioner(x0, kgrid, params, coords=coords, chunk=chunk)
