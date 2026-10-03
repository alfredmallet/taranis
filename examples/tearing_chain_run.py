# 2D equilibria through the autodiff harness: the Fadeev / Kelvin-Stuart island chain
# (plans/AUTODIFF_PLAN.md rung 2b item 3) -- data generation for examples/tearing-chain.ipynb,
# gated by tests/test_tearing_chain.py (which imports this module).
#
# EQUILIBRIUM, ORIENTATION. taranis's 1D tearing code puts the sheet normal along x (B_y(x),
# ky blocks), so the chain runs along y:
#     psi0 = ln(cosh x + eps cos y)          (the paper form ln(cosh y + eps cos x) with x <-> y)
# b_perp = z x grad psi, so B_y -> tanh x as eps -> 0 (the Harris sheet, half-width a = 1,
# v_A = 1, S = 1/eta) and
#     lap psi0 = (1 - eps^2) e^(-2 psi0) = F(psi0)   =>  {psi0, lap psi0} = 0,
# exactly steady in ideal RMHD (and phi0 = G(psi0) keeps it steady for any G). Chain period
# 2 pi: O-points at y = pi (mod 2 pi), X-points at y = 0. The box Ly = 4 pi holds TWO chain
# periods, so the period-doubling (subharmonic, coalescence) modes are in it.
#
# PERIODIC IN x: two sheets. psi0 = psi_pair(x) + sum_images [h(x - Lx/4, y) - h(x - 3Lx/4, y)],
# h(u, y) = ln(1 + eps cos y/cosh u) (the chain minus its Harris part, decaying like
# 2 eps e^-|u| cos y, harmonic at leading order) and psi_pair the EXACT tanh pair of rung
# 1b-ref (T.psi0_tanhpair: B_y = tanh(x - Lx/4) near sheet 1, -tanh(x - 3Lx/4) near sheet 2).
# Near each sheet psi0 is the chain (sheet 2: minus the chain) up to O(eps e^-Lx/2): the
# discrete N(x0) is set by that and by the truncation of the (non-band-limited) chain spectrum,
# both measured (n_of_x0, the gate file).
#
# SYMMETRIES (exact; the gate file checks J S = S J on random vectors), on (phi, psi):
#   T  (Bloch)   y -> y + 2 pi:            (phi, psi) -> (phi, psi)(x, y - 2 pi);  on the rfft2
#                array (-1)^iky. T = +1: the chain's own period (harmonic sector, iky even);
#                T = -1: PERIOD-DOUBLING (subharmonic sector, iky odd) -- coalescence lives here.
#   R  (mirror)  y -> -y:                  (phi, psi) -> (-phi, psi)(x, -y)  (the bracket flips
#                sign under y -> -y). psi0 even in y => a symmetry at alpha = 0. R = +1 with
#                T = -1: psi1 even, phi1 odd about the X-point y = 0 -- the islands at y = +-pi
#                move TOGETHER (coalescence); R = -1: psi1 odd about the X-point, i.e. even
#                about the O-points, one island grows while its neighbours shrink ("alternating").
#   P  (pair)    x -> x + Lx/2, psi -> -psi: (phi, psi) -> (phi, -psi)(x - Lx/2, y)  (RMHD is
#                even in psi). The two sheets' modes split into P = +-1 (in phase / anti-phase),
#                the near-degenerate pair of every two-sheet box, split ~e^(-k Lx/2).
#   XR (rotation by pi about the sheet-1 X-point) (phi, psi)(x, y) -> (phi, psi)(Lx/2 - x, -y):
#                a symmetry for ANY alpha. At alpha = 0 it is R times the x-reflection X about
#                the sheet centres, and the coalescence mode (R+, tearing parity: psi1 even,
#                phi1 odd in x) has XR = +1.
#   With field-aligned flow phi0 = alpha psi0 only T, P.R and XR survive (each of P, R maps
#   alpha -> -alpha); sectors are then labelled (T, PR, XR).
# The sector of a solve is named by its signs; sector_basis builds an orthonormal real basis E
# of it in real coordinates (stability.real_coords restricted to the T-sector's iky), and every
# solve works with E^T (.) E -- exact, since J commutes with the maps.
#
# SOLVERS. (i) sector_eigs: ARPACK on E^T exp(J T) E, exp(J T) by stability.propagator_operator
# (the solver's own IF stepper on the linearized system; rung 0b's 2D workhorse). Its checks
# mirror stability.propagator_eigs (Rayleigh-quotient branch, residual on J itself, RuntimeError
# above residual_tol = 1e-4) -- with E = the T-sector identity it IS propagator_eigs (gated).
# (ii) sector_dense: eig_dense of the dense E^T J E (left vectors too) for small grids: the
# independent route, and the input of the eigenvalue sensitivities (sector_sensitivity:
# stability.dj_operator along dx0 = d x0/d eps or d x0/d alpha, or dparams={"diss": (0, 1)} for
# d/d eta through the overrides seam; stability.eigenvalue_sensitivity).
#
# FROZEN EQUILIBRIUM. With eta > 0, eta lap psi0 != 0 (as in 1D): x0 is frozen, J = L + N'(x0)
# is the instantaneous linearization (valid for gamma >> eta).
#
# `python examples/tearing_chain_run.py [wall_s] [phase ...]` runs make_data (resumable; every
# solve is one cache file under DATA_ROOT/solves, written atomically).
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import hashlib
import json
import sys
import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import jax.numpy as jnp

import taranis as jr
from taranis import stability as st

import tearing_eigen_run as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "tearing-chain")

TWO_PI = 2*np.pi
LX = 40.0                   # sheet-normal box: sheets 20 apart, pair coupling ~e^(-k Lx/2)
LY = 4*np.pi                # two chain periods
RESIDUAL_TOL = 1e-4         # = stability._PROPAGATOR_RESIDUAL_TOL (never relaxed here)


# ============================================================ equilibrium

def chain_h(u, y, eps):
    # ln(cosh u + eps cos y) - ln cosh u, overflow-free
    return np.log1p(eps*np.cos(y)/np.cosh(u))


def chain_dh(u, y, eps):
    # d/d eps of chain_h: cos y/(cosh u + eps cos y)
    return np.cos(y)/(np.cosh(u) + eps*np.cos(y))


def psi_chain(x, y, Lx, eps, deriv=False):
    # the periodic two-sheet chain on the (nx, ny) grid (module note); deriv=True: d psi0/d eps
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="ij")
    f = chain_dh if deriv else chain_h
    p = np.zeros_like(X) if deriv else T.psi0_tanhpair(np.asarray(x, float), Lx)[:, None] + 0*Y
    for m in range(-2, 3):
        p = p + f(X - Lx/4 - m*Lx, Y, eps) - f(X - 3*Lx/4 - m*Lx, Y, eps)
    return p


def chain_current(x, y, eps):
    # the isolated chain's exact current lap psi = (1 - eps^2)/(cosh x + eps cos y)^2
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="ij")
    return (1 - eps*eps)/(np.cosh(X) + eps*np.cos(Y))**2


def make_params(S, nx, ny, Lx=LX, Ly=LY):
    # dims = 2, nu = 0, eta = 1/S, hyper = 1, unforced, fp64
    return jr.Parameters(nx=int(nx), ny=int(ny), Lx=float(Lx), Ly=float(Ly), dims=2,
                         cfl_safety=0.5, forcing=False,
                         eqpars={"diss": (0.0, 1.0/float(S)), "hyper": 1})


def chain_periods(params):
    m = params.Ly/TWO_PI
    if abs(m - round(m)) > 1e-12 or round(m) < 1:
        raise ValueError(f"Ly = {params.Ly} is not a whole number of chain periods 2 pi")
    return int(round(m))


def grid(params):
    return (np.arange(params.nx)*params.Lx/params.nx, np.arange(params.ny)*params.Ly/params.ny)


def _to_fields(params, phi, psi):
    # real (nx, ny) phi, psi -> the rfft2 fields array (dealiased by initialize), projected onto
    # the chain harmonics (iky a multiple of the number of chain periods): exact T-invariance
    p = jnp.stack([jnp.asarray(phi)[None], jnp.asarray(psi)[None]])
    f = np.array(jr.initialize(lambda X, Y: p, params).fields)
    m = chain_periods(params)
    f[..., np.arange(f.shape[-1]) % m != 0] = 0
    return jnp.asarray(f)


def make_x0(params, eps, alpha=0.0, direction=None):
    # the equilibrium (phi0, psi0) = (alpha psi0, psi0) as a fields array; direction "eps":
    # d x0/d eps = (alpha, 1) d psi0/d eps; "alpha": d x0/d alpha = (psi0, 0)
    x, y = grid(params)
    if direction == "eps":
        d = psi_chain(x, y, params.Lx, eps, deriv=True)
        return _to_fields(params, alpha*d, d)
    p = psi_chain(x, y, params.Lx, eps)
    if direction == "alpha":
        return _to_fields(params, p, 0*p)
    if direction is not None:
        raise ValueError(direction)
    return _to_fields(params, alpha*p, p)


def n_of_x0(params, x0, kgrid=None):
    # max |N(x0)| (the solver's own dealiased RHS, eta not included) over max |x0|, in k space
    kg = jr.setup_kgrids(params) if kgrid is None else kgrid
    n = st._nonlinear(params)(jnp.asarray(x0), kg)
    return float(jnp.abs(n).max()/jnp.abs(x0).max())


# ============================================================ symmetries

def _k_indices(params):
    ikx = np.fft.fftfreq(params.nx, 1.0/params.nx).astype(int)
    iky = np.arange(params.ny//2 + 1)
    return ikx, iky


def map_T(f, params):
    # y -> y + Lx_chain = 2 pi (one chain period): the rfft2 coefficient picks up e^(i ky 2 pi)
    m = chain_periods(params)
    _, iky = _k_indices(params)
    ph = np.exp(2j*np.pi*iky/m)
    return jnp.asarray(f)*ph


def map_R(f, params):
    # y -> -y, (phi, psi) -> (-phi, psi): g(kx, ky) = conj(f(-kx, ky))
    f = jnp.asarray(f)
    mirror = (-np.arange(params.nx)) % params.nx
    g = jnp.conj(f[:, :, mirror, :])
    return g*jnp.asarray(np.array([-1.0, 1.0]).reshape(2, 1, 1, 1))


def map_P(f, params):
    # x -> x + Lx/2, psi -> -psi: (phi, psi)(x) -> (phi, -psi)(x - Lx/2): (-1)^ikx
    ikx, _ = _k_indices(params)
    s = ((-1.0)**ikx).reshape(1, 1, -1, 1)
    return jnp.asarray(f)*s*jnp.asarray(np.array([1.0, -1.0]).reshape(2, 1, 1, 1))


def map_PR(f, params):
    return map_P(map_R(f, params), params)


def map_XR(f, params):
    # the rotation by pi about the sheet-1 X-point (x, y) = (Lx/4, 0):
    # (phi, psi)(x, y) -> (phi, psi)(Lx/2 - x, -y) (two reflections: the bracket keeps its sign),
    # g(kx, ky) = (-1)^ikx conj(f(kx, ky)). x0 is invariant for ANY alpha (both sheets are even
    # about their centres, the chain even in y)
    ikx, _ = _k_indices(params)
    return jnp.conj(jnp.asarray(f))*((-1.0)**ikx).reshape(1, 1, -1, 1)


MAPS = {"T": map_T, "R": map_R, "P": map_P, "PR": map_PR, "XR": map_XR}


def bloch_coords(kgrid, params, T_sign):
    # real coordinates of the T sector: the iky with e^(i ky 2 pi) = T_sign. The ky = 0 row
    # (T = +1 only) keeps its mirror entries.
    co = st.real_coords(kgrid, params)
    m = chain_periods(params)
    if m == 1:
        if T_sign != 1:
            raise ValueError("one chain period: there is no T = -1 sector")
        return co
    if m != 2:
        raise ValueError("bloch_coords handles Ly = 2 pi or 4 pi")
    want = 0 if T_sign == 1 else 1

    def keep(idx):
        return idx[np.unravel_index(idx, co.shape)[3] % 2 == want]
    return st.RealCoords(co.shape, keep(co.re), keep(co.im), keep(co.mirror_src),
                         keep(co.mirror_dst))


def signed_perm(coords, fmap, params):
    # the map as a signed permutation of the coordinates: (perm, sign) with M e_i = sign_i
    # e_perm_i. Raises unless it IS one (it is for R, P, PR on any sector, T on a Bloch sector)
    n = coords.n
    u = np.arange(1, n + 1, dtype=float)
    w = np.asarray(coords.pack(fmap(coords.unpack(jnp.asarray(u)), params)))
    a = np.rint(np.abs(w)).astype(int)
    if (np.abs(np.abs(w) - a).max() > 1e-9 or sorted(a) != list(range(1, n + 1))):
        raise ValueError("signed_perm: the map is not a signed permutation of these coordinates")
    perm = np.empty(n, dtype=int)
    sign = np.empty(n)
    perm[a - 1] = np.arange(n)
    sign[a - 1] = np.sign(w)
    return perm, sign


def sector_basis(coords, params, signs):
    # orthonormal real basis (sparse n x m) of the joint eigenspace {M v = s v} of the commuting
    # involutions signs = {"R": +-1, "P": +-1, "PR": +-1}, within coords (a Bloch sector)
    n = coords.n
    Pm = sps.identity(n, format="csr")
    for name, s in sorted(signs.items()):
        perm, sign = signed_perm(coords, MAPS[name], params)
        M = sps.csr_matrix((sign, (perm, np.arange(n))), shape=(n, n))
        Pm = Pm @ ((sps.identity(n) + s*M)*0.5)
    Pm = sps.csc_matrix(Pm)
    Pm.eliminate_zeros()
    cols = []
    for i in range(n):
        rows = Pm.indices[Pm.indptr[i]:Pm.indptr[i + 1]]
        vals = Pm.data[Pm.indptr[i]:Pm.indptr[i + 1]]
        keep = np.abs(vals) > 1e-12
        if keep.any() and rows[keep].min() == i:
            cols.append((rows[keep], vals[keep]/np.linalg.norm(vals[keep])))
    r = np.concatenate([c[0] for c in cols])
    v = np.concatenate([c[1] for c in cols])
    j = np.concatenate([np.full(len(c[0]), k) for k, c in enumerate(cols)])
    return sps.csc_matrix((v, (r, j)), shape=(n, len(cols)))


class Sector:
    # one symmetry sector: coords (the Bloch sector's RealCoords), E (n x m basis; None = the
    # whole Bloch sector), and the labels
    def __init__(self, kgrid, params, T_sign=-1, signs=None):
        self.params, self.kgrid = params, kgrid
        self.T_sign, self.signs = int(T_sign), dict(signs or {})
        self.coords = bloch_coords(kgrid, params, T_sign)
        self.E = sector_basis(self.coords, params, self.signs) if self.signs else None
        self.m = self.coords.n if self.E is None else self.E.shape[1]

    def lift(self, v):
        # sector coordinates -> real coordinates of the Bloch sector
        return v if self.E is None else self.E @ v

    def restrict(self, u):
        return u if self.E is None else self.E.T @ u

    def fields(self, v):
        # a sector vector (real or complex) as a complex fields array, real and imaginary parts
        # unpacked separately (each is a real field)
        u = self.lift(np.asarray(v))
        re = np.asarray(self.coords.unpack(jnp.asarray(np.real(u))))
        im = np.asarray(self.coords.unpack(jnp.asarray(np.imag(u))))
        return re, im

    def label(self):
        s = "".join(f"{k}{'+' if v > 0 else '-'}" for k, v in sorted(self.signs.items()))
        return f"T{'+' if self.T_sign > 0 else '-'}{s}"


def _wrap(op, sec):
    # a real-coordinate LinearOperator restricted to the sector
    if sec.E is None:
        return op
    E = sec.E
    return spla.LinearOperator((sec.m, sec.m), matvec=lambda v: E.T @ op.matvec(E @ np.asarray(v).reshape(-1)),
                               dtype=np.float64)


def _cplx(op):
    def mv(z):
        z = np.asarray(z).reshape(-1)
        return op.matvec(z.real) + 1j*op.matvec(z.imag)
    return mv


def _probe_scale(mv, n):
    # max ||J r||/||r|| over four seeded random complex r (as stability._probe_norm)
    rng = np.random.default_rng(0)
    best = 0.0
    for _ in range(4):
        r = rng.standard_normal(n) + 1j*rng.standard_normal(n)
        best = max(best, float(np.linalg.norm(mv(r))/np.linalg.norm(r)))
    return best


def sector_eigs(x0, sec, *, T, dt, k=4, scheme="lsrk54", tol=1e-10, ncv=None, maxiter=300,
                residual_tol=RESIDUAL_TOL, overrides=None):
    """The k fastest-growing eigenvalues of J in the sector, by ARPACK on E^T exp(J T) E.

    exp(J T): stability.propagator_operator on the sector's Bloch coordinates. Branch from the
    Rayleigh quotient, residuals ||J v - lam v||/(max(|lam|, scale)||v||) on J itself (the
    sector-restricted real_operator), RuntimeError above residual_tol -- as
    stability.propagator_eigs, of which this is the sector-restricted copy. Returns a dict
    (values, vectors (sector coordinates), residuals, scale, mu, applications, seconds, info)
    sorted by decreasing Re."""
    t0 = time.perf_counter()
    if not residual_tol <= RESIDUAL_TOL:
        raise ValueError("sector_eigs: residual_tol may not exceed 1e-4")
    op, info, _ = st.propagator_operator(x0, sec.kgrid, sec.params, T=T, dt=dt, scheme=scheme,
                                         coords=sec.coords, overrides=overrides)
    A = _wrap(op, sec)
    n = sec.m
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n)
    count = [0]

    def mv(u):
        count[0] += 1
        return A.matvec(u)
    try:
        mu, V = spla.eigs(spla.LinearOperator((n, n), matvec=mv, dtype=np.float64), k=k,
                          which="LM", tol=tol, v0=v0, ncv=ncv, maxiter=maxiter)
    except spla.ArpackNoConvergence as err:
        raise RuntimeError(f"sector_eigs: ARPACK did not converge {k} eigenvalues "
                           f"({len(err.eigenvalues)} converged)") from err
    Jop, _ = st.real_operator(x0, sec.kgrid, sec.params, coords=sec.coords, overrides=overrides)
    Jc = _cplx(_wrap(Jop, sec))
    JV = np.stack([Jc(V[:, i]) for i in range(V.shape[1])], axis=1)
    ray = np.einsum("ij,ij->j", V.conj(), JV)/np.einsum("ij,ij->j", V.conj(), V)
    lam = np.log(mu.astype(complex))/info.T
    lam = lam + 2j*np.pi*np.round((ray.imag - lam.imag)*info.T/(2*np.pi))/info.T
    scale = _probe_scale(Jc, n)
    res = np.linalg.norm(JV - V*lam[None, :], axis=0)/(np.maximum(np.abs(lam), scale)
                                                       * np.linalg.norm(V, axis=0))
    order = np.argsort(-lam.real, kind="stable")
    out = dict(values=lam[order], vectors=V[:, order], residuals=res[order], scale=scale,
               mu=mu[order], rayleigh=ray[order], applications=count[0], nsteps=info.nsteps,
               dt=info.dt, T=info.T, seconds=time.perf_counter() - t0, n=n)
    if not np.all(out["residuals"] <= residual_tol):
        raise RuntimeError(f"sector_eigs: residuals {out['residuals']} over {residual_tol:.0e} "
                           f"(values {out['values']}, dt={info.dt:.4g}, T={info.T:.4g})")
    return out


def sector_matrix(x0, sec, dx0=None, dparams=None, overrides=None, block=64):
    # dense E^T J E (dx0 = dparams = None) or E^T dJ E (along dx0 and/or dparams) -- one
    # matvec per basis column, in blocks
    if dx0 is None and dparams is None:
        op, _ = st.real_operator(x0, sec.kgrid, sec.params, coords=sec.coords, overrides=overrides)
    else:
        op, _ = st.dj_operator(x0, dx0, sec.kgrid, sec.params, coords=sec.coords,
                               dparams=dparams, overrides=overrides)
    n = sec.m
    M = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        M[:, j] = sec.restrict(op.matvec(sec.lift(e)))
    return M


def sector_dense(x0, sec, overrides=None, nkeep=12):
    # the whole sector spectrum (eig_dense: right AND left vectors), top nkeep by Re kept
    t0 = time.perf_counter()
    J = sector_matrix(x0, sec, overrides=overrides)
    r = st.eig_dense(J)
    return dict(J=J, values=r.values, right=r.right, left=r.left, residuals=r.residuals,
                scale=r.scale, seconds=time.perf_counter() - t0, n=sec.m)


def sector_sensitivity(x0, sec, dense, dx0=None, dparams=None, which=0, overrides=None):
    # d lambda/ds of dense eigenvalue `which` along (dx0, dparams), by
    # stability.eigenvalue_sensitivity with the full sector spectrum (complete isolation test)
    dJ = sector_matrix(x0, sec, dx0=dx0, dparams=dparams, overrides=overrides)
    lam = dense["values"][which]
    return st.eigenvalue_sensitivity(dense["J"], dJ, lam, dense["right"][:, which],
                                     dense["left"][:, which], spectrum=dense["values"])


# ============================================================ eigenfunctions

def mode_fields(sec, v):
    # a complex sector eigenvector -> real-space (phi, psi) on the grid, phase-normalised so
    # the larger of the real and imaginary parts (each a real eigen-field for real lambda) wins
    re, im = sec.fields(v)
    f = re if np.linalg.norm(re) >= np.linalg.norm(im) else im
    p = sec.params
    phi = np.fft.irfft2(f[0, 0], s=(p.nx, p.ny))
    psi = np.fft.irfft2(f[1, 0], s=(p.nx, p.ny))
    s = np.max(np.abs(psi))
    return phi/s, psi/s


def parity(sec_from, v, name):
    # <v, M v>/<v, v> for a map M (sector-coordinate vector of sec_from): +-1 for an eigenvector
    re, im = sec_from.fields(v)
    num = den = 0.0
    for f in (re, im):
        g = np.asarray(MAPS[name](f, sec_from.params))
        a, b = sec_from.coords.pack(jnp.asarray(f)), sec_from.coords.pack(jnp.asarray(g))
        num += float(np.dot(a, b))
        den += float(np.dot(a, a))
    return num/den


# ============================================================ caching

def _key(**cfg):
    def norm(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if isinstance(v, dict):
            return {k: norm(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v
    txt = json.dumps(norm(cfg), sort_keys=True)
    return hashlib.sha1(txt.encode()).hexdigest()[:16], txt


def _save(path, out):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def cached(compute, root=None, recompute=False, **cfg):
    # one solve per file, DATA_ROOT/solves/<hash>.npz, written atomically; a solve that raised
    # is cached as its message (<hash>.failed) and re-raised on every later call.
    # compute=None reads only (KeyError when absent).
    root = root or DATA_ROOT
    d = os.path.join(root, "solves")
    os.makedirs(d, exist_ok=True)
    h, txt = _key(**cfg)
    path, failed = os.path.join(d, h + ".npz"), os.path.join(d, h + ".failed")
    if os.path.exists(path) and not recompute:
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if os.path.exists(failed) and not recompute:
        with open(failed) as fh:
            raise RuntimeError(fh.read())
    if compute is None:
        raise KeyError(f"not cached: {txt}")
    try:
        out = compute()
    except RuntimeError as e:
        with open(failed, "w") as fh:
            fh.write(f"(cached failure) {txt}: {e}")
        raise
    out = dict(out, cfg=txt)
    _save(path, out)
    return out


# ============================================================ configuration

S_LIST = (1e2, 1e3)
GRID = {1e2: (256, 64), 1e3: (256, 128)}      # dense baseline (Lx = 20): converged to ~1e-3 (CONV)
LX0 = 20.0                  # production box: P+/P- split ~5-10%, their MEAN good to ~1e-4 (BOX)
EPS = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75)
EPS_ALT = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3)    # the R- (alternating) sector
EPS_HARM = (0.3, 0.6, 0.75)                   # the T+ (chain-period) sector
EPS_SMALL = (1e-3, 3e-3, 1e-2, 3e-2)          # the linear-splitting regime ...
SMALL_GRID = {1e3: (512, 32), 1e4: (1024, 32)}    # ... (dense; few y harmonics matter)
S_SPLIT = (1e2, 1e3, 1e4, 1e5)                # d lam/d eps at eps = 0 (dense, ny = 16)
SPLIT_NX = {1e2: 512, 1e3: 512, 1e4: 1024, 1e5: 2048}
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)
ALPHA_S = (1e3, 1e2)
EPS_ALPHA = 0.3
# matrix-free (sector_eigs, k = 1) convergence ladders: (S, eps) -> grids; dt from the CFL
CONV = {(1e3, 0.1): ((256, 128), (512, 128), (256, 256), (512, 256)),
        (1e3, 0.3): ((256, 128), (512, 128), (256, 256), (512, 256)),
        (1e3, 0.6): ((256, 128), (512, 256)),
        (1e4, 0.3): ((512, 256), (768, 256), (512, 384)),
        (1e4, 0.1): ((512, 256),),
        (1e4, 0.6): ((512, 256),)}
HARM_BOX = ((20.0, (256, 64)), (30.0, (384, 64)), (40.0, (512, 64)))   # the T+ R- P- mode vs Lx, eps = 0.6
DCONV = ((1e3, 0.01, (512, 64)), (1e3, 0.03, (512, 64)), (1e3, 0.1, (512, 64)))   # dense, small eps
CONV_ALPHA = ((1e3, 0.3, 0.8, (512, 256)), (1e4, 0.3, 0.6, (512, 256)))
BOX = ((1e3, 0.3, 30.0, (384, 128)), (1e3, 0.3, 40.0, (512, 128)))
PROP_T = 20.0
FD_POINTS = ((1e3, 0.3, 0.0), (1e3, 0.1, 0.0), (1e3, 0.3, 0.6))   # (S, eps, alpha), R+ P+ / PR+
FD_H = (2e-3, 1e-3)


def prop_dt(nx, ny, Lx=LX0, Ly=LY):
    # lsrk54 on N': |k|max |b| dt <= ~2 with |b| <~ 1.2 on these chains -- conservative
    kx = TWO_PI/Lx*(nx//3)
    ky = TWO_PI/Ly*(ny//3)
    return float(min(0.05, 1.6/np.hypot(kx, ky)))


def _sig(signs):
    return {k: int(v) for k, v in sorted(signs.items())}


# ============================================================ cached solves

def dense_point(S, eps, signs, alpha=0.0, T_sign=-1, grid_=None, Lx=LX0, root=None,
                sens=("eps", "eta"), compute=True, nkeep=12):
    # one dense sector solve (top nkeep eigenvalues, the leading mode's fields) plus the
    # sensitivities of the leading eigenvalue along `sens` (eps / alpha: state directions; eta:
    # dparams through the overrides seam). Cached.
    nx, ny = grid_ or GRID[S]
    sens = tuple(sens)

    def comp():
        p = make_params(S, nx, ny, Lx=Lx)
        kg = jr.setup_kgrids(p)
        sec = Sector(kg, p, T_sign, signs)
        x0 = make_x0(p, eps, alpha)
        d = sector_dense(x0, sec)
        lam = d["values"][0]
        out = dict(values=d["values"][:nkeep], residuals=d["residuals"][:nkeep], n=d["n"],
                   seconds=d["seconds"], scale=d["scale"],
                   n_grow=int(np.sum(d["values"].real > 1e-9)))
        phi, psi = mode_fields(sec, d["right"][:, 0])
        out.update(phi=phi.astype(np.float32), psi=psi.astype(np.float32))
        for s in sens:
            try:
                if s == "eta":
                    r = sector_sensitivity(x0, sec, d, dparams={"diss": (0.0, 1.0)})
                else:
                    r = sector_sensitivity(x0, sec, d, dx0=make_x0(p, eps, alpha, direction=s))
                out[f"d_{s}"] = r.dlam
                out[f"kappa_{s}"] = r.kappa
            except RuntimeError as e:        # a refused (non-isolated) eigenvalue: recorded
                out[f"d_{s}"] = np.nan + 0j
                out[f"refused_{s}"] = str(e)[:300]
        return out
    return cached(comp if compute else None, root=root, kind="dense", S=S, eps=eps,
                  alpha=alpha, T_sign=T_sign, signs=_sig(signs), nx=nx, ny=ny, Lx=Lx,
                  sens=list(sens))


def prop_point(S, eps, signs, grid_, alpha=0.0, Lx=LX0, T=PROP_T, dt=None, root=None,
               compute=True):
    # one matrix-free sector solve (k = 1). Cached.
    nx, ny = grid_
    dt = prop_dt(nx, ny, Lx) if dt is None else dt

    def comp():
        p = make_params(S, nx, ny, Lx=Lx)
        kg = jr.setup_kgrids(p)
        sec = Sector(kg, p, -1, signs)
        x0 = make_x0(p, eps, alpha)
        r = sector_eigs(x0, sec, T=T, dt=dt, k=1, ncv=12)
        return dict(values=r["values"], residuals=r["residuals"], applications=r["applications"],
                    nsteps=r["nsteps"], dt=r["dt"], seconds=r["seconds"], n=r["n"])
    return cached(comp if compute else None, root=root, kind="prop", S=S, eps=eps, alpha=alpha,
                  signs=_sig(signs), nx=nx, ny=ny, Lx=Lx, T=T, dt=dt)


def split_point(S, root=None, compute=True, ny=16, Lx=LX0):
    # eps = 0: the degenerate R+/R- pair and its first-order splitting d lam/d eps, P+ and P-
    def comp():
        p = make_params(S, SPLIT_NX[S], ny, Lx=Lx)
        kg = jr.setup_kgrids(p)
        dx = make_x0(p, 0.0, direction="eps")
        x0 = make_x0(p, 0.0)
        out = {}
        for R in (1, -1):
            for P in (1, -1):
                sec = Sector(kg, p, -1, {"R": R, "P": P})
                d = sector_dense(x0, sec)
                s = sector_sensitivity(x0, sec, d, dx0=dx)
                out[f"lam_{R:+d}{P:+d}"] = d["values"][0]
                out[f"d_{R:+d}{P:+d}"] = s.dlam
        return out
    return cached(comp if compute else None, root=root, kind="split", S=S, nx=SPLIT_NX[S],
                  ny=ny, Lx=Lx)


def fd_point(S, eps, alpha, root=None, compute=True):
    # eigenvalue finite differences against the autodiff sensitivities (R+ P+ at alpha = 0,
    # PR+ XR+ otherwise): d/d eps, d/d eta (overrides), d/d alpha (alpha > 0)
    signs = {"R": 1, "P": 1} if alpha == 0 else {"PR": 1, "XR": 1}
    sens = ("eps", "eta") + (("alpha",) if alpha else ())
    base = dense_point(S, eps, signs, alpha, root=root, sens=sens, compute=compute)

    def comp():
        nx, ny = GRID[S]
        p = make_params(S, nx, ny, Lx=LX0)
        kg = jr.setup_kgrids(p)
        sec = Sector(kg, p, -1, signs)
        out = {}
        for h in FD_H:
            for s in sens:
                v = []
                for sg in (1, -1):
                    if s == "eps":
                        d = sector_dense(make_x0(p, eps + sg*h, alpha), sec)
                    elif s == "alpha":
                        d = sector_dense(make_x0(p, eps, alpha + sg*h), sec)
                    else:
                        d = sector_dense(make_x0(p, eps, alpha), sec,
                                         overrides={"diss": (0.0, 1.0/S + sg*h/S)})
                    v.append(d["values"][0])
                hh = h/S if s == "eta" else h
                out[f"fd_{s}_{h:g}"] = (v[0] - v[1])/(2*hh)
        return out
    fd = cached(comp if compute else None, root=root, kind="fd", S=S, eps=eps, alpha=alpha,
                h=list(FD_H))
    return base, fd


# ============================================================ independent route: the nonlinear solver

IVP = dict(S=1e4, eps=0.3, signs={"R": 1, "P": -1}, grid=(256, 128), Lx=LX0, dt=0.02, amp=1e-9,
           t_end=90.0, every=50, window=(40.0, 90.0))


IVP_S = (1e4, np.inf)       # S = inf: eta = 0, x0 exactly steady (no resistive drift of the base state)


def ivp_growth(S, eps, signs, grid_, Lx=LX0, dt=0.02, amp=1e-9, t_end=90.0, every=50,
               scheme="lsrk54", seed=0):
    # the growth rate of x(t; x0 + amp v) - x(t; x0) in the FULL NONLINEAR solver
    # (run.block_of_steps, fixed dt): no jvp, no stability.py operator, no Arnoldi. v is a seeded
    # random vector of the sector (so one eigenvalue dominates). Both runs see the same
    # resistive drift of x0, which cancels from the difference to first order. Returns
    # (t, |delta|) at every `every` steps.
    nx, ny = grid_
    p = jr.Parameters(nx=int(nx), ny=int(ny), Lx=float(Lx), Ly=LY, dims=2, cfl_safety=0.5,
                      forcing=False, adaptive_timestep=False, dt=float(dt),
                      eqpars={"diss": (0.0, 1.0/float(S)), "hyper": 1})
    kg = jr.setup_kgrids(p)
    sec = Sector(kg, p, -1, signs)
    x0 = make_x0(p, eps)
    v = np.random.default_rng(seed).standard_normal(sec.m)
    pert, _ = sec.fields(v)
    pert = pert/np.abs(pert).max()*float(np.abs(np.asarray(x0)).max())*amp
    zero = jnp.zeros((2, 1, nx, ny), dtype=jnp.float64)
    base = jr.initialize(lambda X, Y: zero, p)
    sa = base._replace(fields=jnp.asarray(x0))
    sb = base._replace(fields=jnp.asarray(x0) + jnp.asarray(pert))
    stepper, sch = jr.timestepping.get_scheme(scheme)
    ts, ds = [0.0], [float(jnp.linalg.norm(sb.fields - sa.fields))]
    nblocks = int(round(t_end/dt/every))
    for _ in range(nblocks):
        sa = jr.run.block_of_steps(sa, kg, p, every, sch, stepper)
        sb = jr.run.block_of_steps(sb, kg, p, every, sch, stepper)
        ts.append(float(sa.t))
        ds.append(float(jnp.linalg.norm(sb.fields - sa.fields)))
    return np.array(ts), np.array(ds)


def ivp_point(root=None, compute=True, **cfg):
    c = dict(IVP, **cfg)

    def comp():
        t, d = ivp_growth(c["S"], c["eps"], c["signs"], c["grid"], c["Lx"], c["dt"], c["amp"],
                          c["t_end"], c["every"])
        sel = (t >= c["window"][0]) & (t <= c["window"][1])
        slope = np.polyfit(t[sel], np.log(d[sel]), 1)[0]
        return dict(t=t, delta=d, gamma=slope)
    c2 = dict(c, signs=_sig(c["signs"]), grid=list(c["grid"]), window=list(c["window"]))
    return cached(comp if compute else None, root=root, kind="ivp", **c2)


# ============================================================ theory and readers

def island_halfwidth(eps):
    # separatrix psi = ln(1 + eps) through the X-point (0, 0); on the O-point line y = pi:
    # cosh x - eps = 1 + eps  =>  w = acosh(1 + 2 eps) (-> 2 sqrt(eps) as eps -> 0)
    return float(np.arccosh(1.0 + 2.0*eps))


def current_O(eps):
    # lap psi0 at the O-point (0, pi): (1 + eps)/(1 - eps); at the X-point (1 - eps)/(1 + eps)
    return (1.0 + eps)/(1.0 - eps)


def gamma_harris(S, ka=0.5):
    # rung 1a's exact layer relation for an isolated Harris sheet (Delta' a = 2(1/ka - ka),
    # |f'| = 1): the eps = 0 anchor
    return float(T.gamma_disp(ka, 1.0/S, 1.0, T.dprime_harris_local(ka)))


def ideal_extrapolation(g1, g2, e2, ratio=10.0):
    # gamma(S) = g_inf + a S^-q from gamma at S1 and S2 = ratio S1 and the AUTODIFF exponent
    # e2 = d ln gamma/d ln S at S2 (= -(eta/gamma) d gamma/d eta): a S2^-q = -e2 g2/q and
    # g1 - g2 = a S2^-q (ratio^q - 1) fix q (bisection), then g_inf. Returns (g_inf, q,
    # e1_pred) -- e1_pred, the exponent this model implies at S1, is the consistency check
    # against the independently computed e1.
    from scipy.optimize import brentq
    A = -e2*g2
    if not (A > 0 and g1 > g2):
        return np.nan, np.nan, np.nan
    f = lambda q: A*(ratio**q - 1.0)/q - (g1 - g2)
    lo, hi = 1e-6, 5.0
    if f(lo)*f(hi) > 0:
        return np.nan, np.nan, np.nan
    q = brentq(f, lo, hi, xtol=1e-14)
    return g2 - A/q, q, -A*ratio**q/g1


def _try(fn):
    try:
        return fn()
    except (KeyError, RuntimeError):
        return None


def _pair(S, eps, signs_pair, root=None, **kw):
    # the two pair members' records (or None)
    return [_try(lambda sg=sg: dense_point(S, eps, sg, compute=False, root=root, **kw)) for sg in signs_pair]


def eps_table(S, eps_list=None, grid_=None, sens=("eps", "eta"), root=None):
    # rows (eps, lam P+, lam P-, mean, d mean/d eps, d mean/d eta) of the R+ (coalescence) mode
    rows = []
    for e in eps_list or EPS:
        a, b = _pair(S, e, RP, root=root, grid_=grid_, sens=sens)
        if a is None or b is None:
            continue
        la, lb = a["values"][0], b["values"][0]
        row = dict(eps=e, lam_p=la, lam_m=lb, mean=0.5*(la + lb).real, split=abs(la - lb)/abs(la + lb)*2,
                   n_grow=(int(a["n_grow"]), int(b["n_grow"])), phi=a["phi"], psi=a["psi"])
        for s_ in sens:
            row[f"d_{s_}"] = 0.5*(a[f"d_{s_}"] + b[f"d_{s_}"]).real
            row[f"kappa_{s_}"] = (float(a[f"kappa_{s_}"]), float(b[f"kappa_{s_}"]))
        rows.append(row)
    return rows


def alt_table(S, root=None):
    rows = []
    for e in EPS_ALT:
        a, b = _pair(S, e, RM, root=root, sens=())
        if a is not None and b is not None:
            rows.append(dict(eps=e, lam_p=a["values"][0], lam_m=b["values"][0],
                             mean=0.5*(a["values"][0] + b["values"][0]).real))
    return rows


def harm_table(root=None):
    rows = []
    for e in EPS_HARM:
        recs = [_try(lambda sg=sg: dense_point(1e3, e, sg, T_sign=1, sens=(), compute=False, root=root)) for sg in RP + RM]
        if all(r is not None for r in recs):
            rows.append(dict(eps=e, tops=[r["values"][0] for r in recs], vals=[r["values"] for r in recs]))
    return rows


def small_table(S, root=None):
    g = SMALL_GRID[S]
    rows = []
    for e in (0.0,) + EPS_SMALL:
        a, b = _pair(S, e, RP, root=root, grid_=g, sens=("eps",))
        c, d = _pair(S, e, RM, root=root, grid_=g, sens=())
        if None in (a, b, c, d):
            continue
        rows.append(dict(eps=e, plus=0.5*(a["values"][0] + b["values"][0]).real,
                         minus=0.5*(c["values"][0] + d["values"][0]).real,
                         d_plus=0.5*(a["d_eps"] + b["d_eps"]).real))
    return rows


def split_table(root=None):
    rows = []
    for S in S_SPLIT:
        r = _try(lambda S=S: split_point(S, compute=False, root=root))
        if r is not None:
            rows.append(dict(S=S, lam0=0.5*(r["lam_+1+1"] + r["lam_+1-1"]).real,
                             c=0.5*(r["d_+1+1"] + r["d_+1-1"]).real,
                             c_minus=0.5*(r["d_-1+1"] + r["d_-1-1"]).real, rec=r))
    return rows


def alpha_table(root=None, S=1e3):
    rows = []
    for a_ in ALPHAS:
        r = [_try(lambda sg=sg: dense_point(S, EPS_ALPHA, sg, alpha=a_, sens=("alpha", "eta"), compute=False, root=root))
             for sg in PR]
        if None in r:
            continue
        la, lb = r[0]["values"][0], r[1]["values"][0]
        rows.append(dict(alpha=a_, lam_p=la, lam_m=lb, mean=0.5*(la + lb).real,
                         d_alpha=0.5*(r[0]["d_alpha"] + r[1]["d_alpha"]).real,
                         d_eta=0.5*(r[0]["d_eta"] + r[1]["d_eta"]).real,
                         im=max(abs(la.imag), abs(lb.imag))))
    return rows


def fd_table(root=None):
    rows = []
    for S, e, a_ in FD_POINTS:
        r = _try(lambda: fd_point(S, e, a_, compute=False, root=root))
        if r is not None:
            rows.append(dict(S=S, eps=e, alpha=a_, base=r[0], fd=r[1]))
    return rows


def conv_table(root=None):
    out = {}
    for (S, e), grids in CONV.items():
        for g in grids:
            r = [_try(lambda sg=sg: prop_point(S, e, sg, g, compute=False, root=root)) for sg in RP]
            if None not in r:
                out[(S, e, g)] = dict(lam_p=r[0]["values"][0], lam_m=r[1]["values"][0],
                                      mean=0.5*(r[0]["values"][0] + r[1]["values"][0]).real,
                                      res=max(r[0]["residuals"][0], r[1]["residuals"][0]),
                                      seconds=r[0]["seconds"] + r[1]["seconds"],
                                      apps=(int(r[0]["applications"]), int(r[1]["applications"])))
    for S, e, a_, g in CONV_ALPHA:
        r = [_try(lambda sg=sg: prop_point(S, e, sg, g, alpha=a_, compute=False, root=root)) for sg in PR]
        if None not in r:
            out[(S, e, g, a_)] = dict(lam_p=r[0]["values"][0], lam_m=r[1]["values"][0],
                                      mean=0.5*(r[0]["values"][0] + r[1]["values"][0]).real,
                                      res=max(r[0]["residuals"][0], r[1]["residuals"][0]),
                                      seconds=r[0]["seconds"] + r[1]["seconds"])
    return out


def harm_box_table(root=None):
    out = {}
    for Lx, g in HARM_BOX:
        r = _try(lambda: dense_point(1e3, 0.6, {"R": -1, "P": -1}, T_sign=1, grid_=g, Lx=Lx, sens=(),
                                     compute=False, root=root))
        if r is not None:
            out[Lx] = r
    return out


def dconv_table(root=None):
    out = {}
    for S, e, g in DCONV:
        r = _pair(S, e, RP, root=root, grid_=g, sens=())
        if None not in r:
            out[(S, e)] = 0.5*(r[0]["values"][0] + r[1]["values"][0]).real
    return out


def box_table(root=None):
    out = {}
    for S, e, Lx, g in BOX:
        r = _pair(S, e, RP, root=root, grid_=g, Lx=Lx, sens=())
        if None not in r:
            out[Lx] = dict(lam_p=r[0]["values"][0], lam_m=r[1]["values"][0],
                           mean=0.5*(r[0]["values"][0] + r[1]["values"][0]).real)
    return out


# ============================================================ phases

RP = ({"R": 1, "P": 1}, {"R": 1, "P": -1})          # coalescence (R+), both pair members
RM = ({"R": -1, "P": 1}, {"R": -1, "P": -1})        # alternating (R-)
PR = ({"PR": 1, "XR": 1}, {"PR": -1, "XR": 1})      # with flow (coalescence: XR = +1)


def _jobs(name):
    # (label, callable) per solve of a phase
    out = []
    if name == "eps":
        for S in S_LIST:
            for e in EPS:
                for sg in RP:
                    out.append((f"eps S={S:g} eps={e} {sg}", lambda S=S, e=e, sg=sg: dense_point(S, e, sg)))
    elif name == "alt":
        for S in S_LIST:
            for e in EPS_ALT:
                for sg in RM:
                    out.append((f"alt S={S:g} eps={e} {sg}", lambda S=S, e=e, sg=sg: dense_point(S, e, sg, sens=())))
    elif name == "harm":
        for e in EPS_HARM:
            for sg in RP + RM:
                out.append((f"harm eps={e} {sg}", lambda e=e, sg=sg: dense_point(1e3, e, sg, T_sign=1, sens=())))
    elif name == "small":
        for S, g in SMALL_GRID.items():
            for e in (0.0,) + EPS_SMALL:
                for sg in RP + RM:
                    out.append((f"small S={S:g} eps={e} {sg}",
                                lambda S=S, e=e, sg=sg, g=g: dense_point(S, e, sg, grid_=g, sens=("eps",) if sg["R"] > 0 else ())))
    elif name == "split":
        for S in S_SPLIT:
            out.append((f"split S={S:g}", lambda S=S: split_point(S)))
    elif name == "alpha":
        for S in ALPHA_S:
            for a in ALPHAS:
                for sg in PR:
                    out.append((f"alpha S={S:g} alpha={a} {sg}",
                                lambda S=S, a=a, sg=sg: dense_point(S, EPS_ALPHA, sg, alpha=a, sens=("alpha", "eta"))))
    elif name == "fd":
        for S, e, a in FD_POINTS:
            out.append((f"fd S={S:g} eps={e} alpha={a}", lambda S=S, e=e, a=a: fd_point(S, e, a)))
    elif name == "conv":
        for (S, e), grids in CONV.items():
            for g in grids:
                for sg in RP:
                    out.append((f"conv S={S:g} eps={e} {g} {sg}", lambda S=S, e=e, g=g, sg=sg: prop_point(S, e, sg, g)))
    elif name == "harm_box":
        for Lx, g in HARM_BOX:
            out.append((f"harm_box Lx={Lx:g}", lambda Lx=Lx, g=g: dense_point(1e3, 0.6, {"R": -1, "P": -1}, T_sign=1,
                                                                              grid_=g, Lx=Lx, sens=())))
    elif name == "dconv":
        for S, e, g in DCONV:
            for sg in RP:
                out.append((f"dconv S={S:g} eps={e} {g} {sg}", lambda S=S, e=e, g=g, sg=sg: dense_point(S, e, sg, grid_=g, sens=())))
    elif name == "conv_alpha":
        for S, e, a, g in CONV_ALPHA:
            for sg in PR:
                out.append((f"conv_alpha alpha={a} {g} {sg}", lambda S=S, e=e, a=a, g=g, sg=sg: prop_point(S, e, sg, g, alpha=a)))
    elif name == "box":
        for S, e, Lx, g in BOX:
            for sg in RP:
                out.append((f"box Lx={Lx:g} {sg}", lambda S=S, e=e, Lx=Lx, g=g, sg=sg: dense_point(S, e, sg, grid_=g, Lx=Lx, sens=())))
    elif name == "ivp":
        for S in IVP_S:
            out.append((f"ivp S={S:g}", lambda S=S: ivp_point(S=S)))
            out.append((f"ivp dense S={S:g}", lambda S=S: dense_point(S, IVP["eps"], IVP["signs"], grid_=IVP["grid"], sens=())))
    else:
        raise ValueError(name)
    return out


PHASES = ("eps", "fd", "split", "small", "alt", "harm", "alpha", "box", "ivp", "dconv", "harm_box", "conv", "conv_alpha")


def make_data(wall_budget=3*3600.0, phases=None, log=print, stride=None):
    # resumable: every solve is cached; stride=(i, n) takes every n-th job from offset i (two
    # processes split a phase without overlap)
    deadline = time.time() + wall_budget
    for name in phases or PHASES:
        t0 = time.time()
        ok = True
        jobs = _jobs(name)
        if stride is not None:
            jobs = jobs[stride[0]::stride[1]]
        for label, fn in jobs:
            if time.time() > deadline:
                ok = False
                break
            t1 = time.time()
            try:
                r = fn()
            except RuntimeError as e:
                ok = False
                log(f"  FAILED {label}: {str(e)[:200]}")
                continue
            if isinstance(r, tuple):
                r = r[0]
            v = r["values"][0] if "values" in r else (r["gamma"] if "gamma" in r else r.get("lam_+1+1"))
            log(f"  {label}: {v} ({time.time() - t1:.0f} s)")
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    args = sys.argv[1:]
    wall = float(args.pop(0)) if args and args[0].replace(".", "").isdigit() else 3*3600.0
    stride = None
    if args and args[0].startswith("stride="):
        i, n = args.pop(0)[7:].split("/")
        stride = (int(i), int(n))
    make_data(wall, args or None, log=lambda *a: print(*a, flush=True), stride=stride)
