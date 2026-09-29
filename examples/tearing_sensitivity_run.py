# Eigenvalue sensitivities and resumable data generation for examples/tearing-sensitivity.ipynb:
# plans/AUTODIFF_PLAN.md rung 1b (science half) -- d lambda/d p for resistive tearing with
# field-aligned shear flow, from the harness's first-order perturbation theory
#     d lambda/ds = w^H (dJ/ds) v / (w^H v)
# (taranis/stability.py: dj_operator = forward-over-forward jvp, left_eigenvector = shift-invert
# on J^H, eigenvalue_sensitivity = the formula with its refusal criteria). No derivative is
# formed by differencing eigenvalues except in the FD gates that check the formula.
#
# Setup exactly as rung 1b-ref (tearing_shear_run.py): dims=2, nu = 0 (diss = (0, eta)),
# hyper = 1, S = 1/eta, ky block (Ly = 2 pi/k, ny = 8, iky = 1), fp64; Psi0 = sech^2 x with its
# two nearest periodic images, Phi0 = alpha Psi0, Lx = max(15/k, 20).
#
# PARAMETERS THROUGH x0 ONLY (the harness's scope):
#   * alpha: dx0/dalpha = (Psi0 in phi, 0 in psi). N is quadratic, so J is AFFINE in x0:
#     J(alpha) = J(0) + alpha dJ exactly, which the band preconditioner exploits (the band of
#     J(0) and of dJ are probed once per (S, ka, nx) and combined for every alpha).
#   * the sheet width a at FIXED k and eta (marginal Newton): the family keeps the FIELD
#     amplitude fixed, B_y(x; a) = f(x/a), i.e. Psi0(x; a) = a Psi0^(x/a), so the outer
#     equation psi'' = (k^2 + f''/f/a^2) psi makes Delta' a a function of ka ONLY (sech^2:
#     Delta'a = 2(5 - k^2a^2)(k^2a^2 + 3)/(k^2a^2 sqrt(k^2a^2 + 4)), marginal at ka = sqrt 5; tanh:
#     2(1/ka - ka), marginal at ka = 1). dx0/da is analytic:
#     sech^2: d/da [a sech^2(x/a)] = sech^2 u (1 + 2u tanh u), u = x/a (times alpha in phi).
#     S = 1/eta is fixed along the family, so the sheet's own Lundquist number a/eta varies
#     with a; the ideal marginal point does not depend on it.
#
# RUNNING EXPONENT: p(alpha) = d ln gamma / d ln(1 - alpha^2) = (dgamma/dalpha)(1 - alpha^2)/
# (-2 alpha gamma). The paper's asymptotes (Mallet et al. JPP 2025; the Julia goodguess):
# constant-psi gamma ~ S^-1/2 (1-alpha^2)^(1/2) k^-1/2 -> p = 1/2; nonconstant-psi gamma ~
# (1-alpha^2)^(2/3) S^-1/3 k^2/3 -> p = 2/3; transition K_tr ~ S^-1/7 (1-alpha^2)^-1/7 and
# gamma_max ~ (1-alpha^2)^(4/7) S^-3/7 -> p = 4/7 on the ridge. On the ridge the envelope theorem
# gives d gamma_max/d alpha = (d gamma/d alpha) at k_max, so the ridge exponent needs no k
# derivative: p is interpolated across ka to the k_max(alpha) of a parabola in ln k.
#
# SOLVER (per point, all matrix-free on the ky block):
#   right: stability.shift_invert(k=1) with rung 1a's band-LU preconditioned GMRES;
#   left : stability.left_eigenvector at the SAME shift, preconditioned by the adjoint solve of
#          the same LU (splu trans='H');
#   dJ   : stability.dj_operator along dx0;
#   sens : stability.eigenvalue_sensitivity, residual-gated (1e-8) and isolation-gated.
#   Its `spectrum` (the gap) comes from a DENSE spectrum of the same block on a coarse seed grid:
#   shift_invert with k >= 2 does not converge here (measured: ARPACK 300 restarts, 1 of 2
#   converged, S = 1e4, ka = 0.5, alpha = 0.8, nx = 4096) because the next-nearest eigenvalues
#   are the nu = 0 NULL CLUSTER at lambda = 0 (|Re| ~ 1e-16, |Im| <= 1e-10 in eig_dense at every
#   nx tried), a near-degenerate cluster ARPACK cannot split. The dense spectrum certifies that
#   the nearest other eigenvalue is that cluster, |lambda| away; the fine-grid lambda plus the
#   coarse grid's OTHER eigenvalues is what is passed (gap = min distance, ~ |lambda|).
#
# `python examples/tearing_sensitivity_run.py [wall_seconds] [phase ...]` runs make_data to
# completion (resumable; every solve is one cache file under DATA_ROOT/solves).
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import functools
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jax.numpy as jnp

import taranis as jr
from taranis import stability as st

import tearing_eigen_run as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "tearing-sensitivity")
SHEAR_ROOT = os.path.join(HERE, "data", "tearing-shear")
SQRT5 = np.sqrt(5.0)

# running-exponent families: (S, ka) at fixed k; alphas chosen on ln(1 - alpha^2)
S_EXP = (1e3, 1e4, 1e5)
S_EXP_CONST = (1e3, 1e4, 1e5, 1e6)        # S = 1e6: seeded and gridded from rung 1a (alpha = 0)
KA_NONCONST = 0.1
KA_CONST = 1.0
KA_CONST2 = 1.5                          # deeper in constant-psi (ka/K_tr = 7.8 at S = 1e5)
ALPHAS = (0.1, 0.3, 0.45, 0.6, 0.7, 0.8, 0.87, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998,
          0.999)
ALPHAS_TAIL = (0.9995, 0.9998, 0.9999)   # S >= 1e5 only: 1 - alpha^2 down to 2e-4
# ka = 0.1 at S = 1e5 (Lx = 150, nx = 8192: 10-15 min per point, and per failed shift, on the
# loaded laptop): a subset, at nx_hi only (the grid check nx_hi/2 is kept at alpha = 0.1).
# alpha = 0.995 and 0.998 were dropped after four and two GMRES-stalled shifts (~12 min each) --
# out of budget, not refused
ALPHAS_NC_1E5 = (0.1, 0.6, 0.8, 0.92, 0.97, 0.99)
# ridge: a fixed geometric ka grid (bands shared across alpha); 5 ka around the predicted k_max
S_RIDGE = (1e4, 1e5)
RIDGE_KA = tuple(np.round(0.2*1.15**np.arange(16), 4))
RIDGE_ALPHAS = (0.1, 0.3, 0.6, 0.8, 0.92, 0.95, 0.98, 0.99, 0.995, 0.998)
FD_POINTS = ((1e4, 0.1, 0.8), (1e4, 1.0, 0.95), (1e5, 0.1, 0.99), (1e5, 1.0, 0.6))
FD_H = (1e-2, 3e-3, 1e-3)
FD_H_NEAR1 = (3e-3, 1e-3, 3e-4)
# kappa against nx at fixed (S, ka, alpha): the eigenvalue converges, kappa does not
KAPPA_NX_POINTS = ((1e4, 1.0, 0.3), (1e4, 1.0, 0.95), (1e3, 1.0, 0.8))
KAPPA_NX = (512, 1024, 2048, 4096, 8192, 16384)
NX_DENSE = 1024


# ============================================================ equilibria

def _sech2(u):
    e = np.exp(-2.0*np.abs(u))
    return 4.0*e/(1.0 + e)**2


def psi0_sech2_a(x, Lx, a=1.0):
    # a sech^2(x/a) about Lx/2 plus its two nearest periodic images (B_y amplitude fixed)
    return sum(a*_sech2((x - Lx/2 - m*Lx)/a) for m in (-1, 0, 1))


def dpsi0_sech2_da(x, Lx, a=1.0):
    # d/da [a sech^2(x/a)] = sech^2 u (1 + 2 u tanh u), u = x/a
    out = 0.0
    for m in (-1, 0, 1):
        u = (x - Lx/2 - m*Lx)/a
        out = out + _sech2(u)*(1.0 + 2.0*u*np.tanh(u))
    return out


def _fft_integral(by, Lx):
    # Psi0 = int B_y dx by FFT (B_y analytic with zero mean): spectrally exact
    n = len(by)
    kx = 2*np.pi*np.fft.fftfreq(n, d=Lx/n)
    bh = np.fft.fft(by)
    ph = np.zeros_like(bh)
    ph[1:] = bh[1:]/(1j*kx[1:])
    return np.real(np.fft.ifft(ph))


def by_tanhpair_a(x, Lx, a=1.0):
    # rung 1b-ref's exact-tanh pair with width a: B_y = tanh((x - Lx/4)/a) near the first
    # sheet, -tanh((x - 3Lx/4)/a) near the second (field amplitude 1 for every a)
    return -1.0 + sum(np.tanh((x - Lx/4 - m*Lx)/a) - np.tanh((x - 3*Lx/4 - m*Lx)/a)
                      for m in range(-2, 3))


def dby_tanhpair_da(x, Lx, a=1.0):
    # d/da tanh((x - c)/a) = -u sech^2(u)/a, u = (x - c)/a
    out = 0.0
    for m in range(-2, 3):
        for c, sg in ((Lx/4 + m*Lx, 1.0), (3*Lx/4 + m*Lx, -1.0)):
            u = (x - c)/a
            out = out - sg*u*_sech2(u)/a
    return out


def make_params(S, k, nx, Lx):
    return jr.Parameters(nx=int(nx), ny=8, Lx=float(Lx), Ly=2*np.pi/k, dims=2, cfl_safety=0.5,
                         forcing=False, eqpars={"diss": (0.0, 1.0/S), "hyper": 1})


def _fields(params, phi, psi):
    phi = jnp.asarray(phi).reshape(1, -1, 1)
    psi = jnp.asarray(psi).reshape(1, -1, 1)
    return jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields


def equilibrium(params, alpha, a=1.0, direction="alpha", case="sech2"):
    # (x0, dx0): case "sech2" or "tanhpair"; direction "alpha" (dx0 = (Psi0, 0)) or "a" (width)
    x = np.arange(params.nx)*params.Lx/params.nx
    if case == "sech2":
        p0 = psi0_sech2_a(x, params.Lx, a)
    else:
        p0 = _fft_integral(by_tanhpair_a(x, params.Lx, a), params.Lx)
    x0 = _fields(params, alpha*p0, p0)
    if direction == "alpha":
        dx0 = _fields(params, p0, 0*p0)
    elif direction == "a":
        dp = dpsi0_sech2_da(x, params.Lx, a) if case == "sech2" else \
            _fft_integral(dby_tanhpair_da(x, params.Lx, a), params.Lx)
        dx0 = _fields(params, alpha*dp, dp)
    else:
        raise ValueError(direction)
    return x0, dx0


# ============================================================ band preconditioner

def band_matrix(op, idx, nx, band):
    # the banded part of a ky-block operator (J or dJ) from coloured probes (rung 1a's
    # T._band_preconditioner, returning the sparse matrix instead of its shifted LU)
    n = len(idx)
    P = 1
    while P < 2*band + 1:
        P *= 2
    P = min(P, nx)
    lut = -np.ones((2, nx), dtype=np.int64)
    lut[idx[:, 0], idx[:, 2]] = np.arange(n)
    fld, ikx = idx[:, 0], idx[:, 2]
    rows, cols, vals = [], [], []
    for f in (0, 1):
        for c in range(P):
            sel = (fld == f) & (ikx % P == c)
            if not sel.any():
                continue
            v = np.zeros(n, dtype=complex)
            v[sel] = 1.0
            img = op.matvec(v)
            d = (c - ikx) % P
            d = np.where(d > P//2, d - P, d)
            cand = lut[f, (ikx + d) % nx]
            ok = (cand >= 0) & (np.abs(d) <= band) & (img != 0)
            rows.append(np.nonzero(ok)[0])
            cols.append(cand[ok])
            vals.append(img[ok])
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n, n))


def _lu_ops(A, sigma):
    n = A.shape[0]
    lu = spla.splu((A - sigma*sp.identity(n, format="csc")).tocsc(), permc_spec="COLAMD")
    M = spla.LinearOperator((n, n), matvec=lambda b: lu.solve(np.asarray(b).reshape(-1)),
                            dtype=complex)
    MH = spla.LinearOperator((n, n), matvec=lambda b: lu.solve(np.asarray(b).reshape(-1),
                                                               trans="H"), dtype=complex)
    return M, MH


@functools.lru_cache(maxsize=4)
def _alpha_bands(S, k, nx, Lx):
    # (band of J at alpha = 0, band of dJ/dalpha): J(alpha) = J(0) + alpha dJ exactly
    params = make_params(S, k, nx, Lx)
    kg = jr.setup_kgrids(params)
    x0, dx0 = equilibrium(params, 0.0)
    op, idx = st.ky_block_operator(x0, kg, params, 1)
    dop, _ = st.dj_operator(x0, dx0, kg, params, iky=1)
    band = T.band_width("sech2", Lx)
    return band_matrix(op, idx, nx, band), band_matrix(dop, idx, nx, band)


# ============================================================ parity sectors (tanh pair)
# The pair profile is symmetric under reflection about the midpoint between its sheets; in the
# ky block that is f(kx) -> f(-kx) on both fields (checked: it commutes with J to 5e-14 of
# ||J||). Its two sectors hold the two pair members (even / odd about the midpoint), which are
# split only by the e^(-k Lx/2) image coupling -- far too close for the isolation test on the
# full block, exactly separated on the sectors. E: orthonormal sparse basis of a sector.

def sector_basis(idx, nx, sector):
    lut = {(int(f), int(i)): j for j, (f, _, i) in enumerate(idx)}
    rows, cols, vals, m = [], [], [], 0
    for j, (f, _, i) in enumerate(idx):
        jm = lut[(int(f), (-int(i)) % nx)]
        if jm < j:
            continue
        if jm == j:                    # self-mirrored mode (kx = 0): symmetric sector only
            if sector > 0:
                rows.append(j); cols.append(m); vals.append(1.0); m += 1
            continue
        rows += [j, jm]; cols += [m, m]; vals += [2**-0.5, sector*2**-0.5]; m += 1
    return sp.csc_matrix((vals, (rows, cols)), shape=(len(idx), m))


def reduce_op(op, E):
    EH = E.conj().T.tocsc()
    m = E.shape[1]
    mv = lambda u: EH @ op.matvec(E @ np.asarray(u).reshape(-1))
    rmv = (lambda u: EH @ op.rmatvec(E @ np.asarray(u).reshape(-1))) \
        if hasattr(op, "rmatvec") else None
    return spla.LinearOperator((m, m), matvec=mv, rmatvec=rmv, dtype=complex)


# ============================================================ one sensitivity solve

def _eigfun_window(params, idx, v, ctr, half=3.0, nmax=1500):
    i0 = int(round(ctr/params.Lx*params.nx)) % params.nx
    x, psi, phi, _ = T._eigfun(params, idx, v, i0)
    xs = (x - ctr + params.Lx/2) % params.Lx - params.Lx/2
    o = np.argsort(xs)
    x, psi, phi, xs = x[o], psi[o], phi[o], xs[o]
    sel = np.nonzero(np.abs(xs) <= half)[0]
    sel = sel[::max(1, len(sel)//nmax)]
    return xs[sel], psi[sel], phi[sel]


@functools.lru_cache(maxsize=64)
def _dense_others_cached(S, k, a, alpha, Lx, nx, case="sech2", sector=0):
    # eig_dense of the block (or its parity sector) on a coarse grid: every eigenvalue EXCEPT
    # the fastest (the tearing mode on that grid), and the number growing
    params = make_params(S, k, nx, Lx)
    kg = jr.setup_kgrids(params)
    x0, _ = equilibrium(params, alpha, a, case=case)
    B, idx = st.ky_block_matrix(x0, kg, params, 1)
    if sector:
        E = sector_basis(idx, nx, sector).toarray()
        B = E.conj().T @ B @ E
    r = st.eig_dense(B)
    return r.values[1:], complex(r.values[0]), int(np.sum(r.values.real > 1e-10*r.scale))


def sens_solve(S, k, a, alpha, nx, Lx, sigma, direction="alpha", lam_hint=None, dense_nx=None,
               case="sech2", sector=0, rtol_mult=1.0):
    # one right/left/sensitivity solve; returns a record dict (raises RuntimeError on any
    # harness refusal -- the caller caches that as a failure). sector = +-1: the tanh pair's
    # parity sectors (the operators restricted by E^H . E)
    t0 = time.time()
    params = make_params(S, k, nx, Lx)
    kg = jr.setup_kgrids(params)
    x0, dx0 = equilibrium(params, alpha, a, direction, case)
    op_full, idx = st.ky_block_operator(x0, kg, params, 1)
    dop_full, _ = st.dj_operator(x0, dx0, kg, params, iky=1)
    if direction == "alpha" and a == 1.0 and case == "sech2":
        A0, A1 = _alpha_bands(S, k, nx, Lx)
        A = A0 + alpha*A1
    else:
        A = band_matrix(op_full, idx, nx, T.band_width("sech2", Lx))
    if sector:
        E = sector_basis(idx, nx, sector)
        op, dop = reduce_op(op_full, E), reduce_op(dop_full, E)
        A = (E.conj().T @ A @ E).tocsc()
        vec = lambda u: E @ u
    else:
        op, dop, vec = op_full, dop_full, (lambda u: u)
    M, MH = _lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    tb = time.time() - t0
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
    # inner GMRES tolerance: the attainable accuracy of a shifted solve of a NORMAL operator,
    # ~eps ||J||/|lambda - sigma|; strongly non-normal blocks (kappa >~ 1e5) stall above it, and
    # the retries loosen it by rtol_mult (shift_invert's own residual gate still guards the pair)
    grtol = float(np.clip(300*rtol_mult*np.finfo(float).eps*normA/max(dist, 1e-300), 1e-10,
                          1e-6*max(rtol_mult, 1.0)))
    kw = dict(tol=1e-12, gmres_rtol=grtol, restart=30 if rtol_mult == 1.0 else 60, maxiter=20)
    right = st.shift_invert(op, sigma, k=1, M=M, **kw)
    lam = complex(right.values[0])
    v = right.right[:, 0]
    tr = time.time() - t0
    left = st.left_eigenvector(op, lam, sigma=sigma, M=MH, **kw)
    w = left.left
    tl = time.time() - t0
    others, lam_dense, npos = _dense_others_cached(S, k, a, alpha, Lx, dense_nx or NX_DENSE,
                                                   case, sector)
    spectrum = np.concatenate([[lam], others])
    s = st.eigenvalue_sensitivity(op, dop, lam, v, w, spectrum=spectrum)
    dv = dop.matvec(v)
    naive = complex(np.vdot(v, dv)/np.vdot(v, v))
    ctr = params.Lx/2 if case == "sech2" else params.Lx/4
    xr, psir, phir = _eigfun_window(params, idx, vec(v), ctr)
    _, psil, phil = _eigfun_window(params, idx, vec(w), ctr)
    return dict(S=S, k=k, a=a, ka=k*a, alpha=alpha, nx=nx, Lx=Lx, sigma=complex(sigma),
                direction=direction, case=case, sector=sector, rtol_mult=rtol_mult,
                lam=lam, dlam=s.dlam,
                kappa=s.kappa, wv=s.wv, gap=s.gap,
                rel_err=s.rel_err, res_r=s.residual_right, res_l=s.residual_left, scale=s.scale,
                naive=naive, lam_dense=lam_dense, npos_dense=npos, dense_nx=dense_nx or NX_DENSE,
                gmres_rtol=grtol, right_iters=right.stats["gmres_iters"],
                left_iters=left.stats["gmres_iters"], t_build=tb, t_right=tr - tb,
                t_left=tl - tr, seconds=time.time() - t0, xr=xr, psi_r=psir, phi_r=phir,
                psi_l=psil, phi_l=phil)


def eig_only(S, k, a, alpha, nx, Lx, sigma, lam_hint=None):
    # the eigenvalue alone (FD gates): same preconditioned shift-invert
    params = make_params(S, k, nx, Lx)
    kg = jr.setup_kgrids(params)
    x0, _ = equilibrium(params, alpha, a)
    op, idx = st.ky_block_operator(x0, kg, params, 1)
    if a == 1.0:
        A0, A1 = _alpha_bands(S, k, nx, Lx)
        A = A0 + alpha*A1
    else:
        A = band_matrix(op, idx, nx, T.band_width("sech2", Lx))
    M, _ = _lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
    grtol = float(np.clip(300*np.finfo(float).eps*normA/max(dist, 1e-300), 1e-10, 1e-6))
    r = st.shift_invert(op, sigma, k=1, M=M, tol=1e-12, gmres_rtol=grtol, restart=30, maxiter=20)
    return dict(lam=complex(r.values[0]), res=float(r.residuals[0]))


# ============================================================ cache

def cached(kind, root=None, compute=None, **cfg):
    # one solve, cached as DATA_ROOT/solves/<hash>.npz keyed by (kind, cfg); a solve that
    # raised is cached as its message and re-raised on every later read
    root = root or DATA_ROOT
    os.makedirs(os.path.join(root, "solves"), exist_ok=True)
    h, txt = T._key(kind=kind, **cfg)
    path = os.path.join(root, "solves", h + ".npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: (d[k][()] if d[k].ndim == 0 else d[k]) for k in d.files}
    failed = os.path.join(root, "solves", h + ".failed")
    if os.path.exists(failed):
        with open(failed) as fh:
            raise RuntimeError(fh.read())
    if compute is None:
        raise KeyError(f"not cached: {txt}")
    try:
        out = compute()
    except RuntimeError as e:
        with open(failed, "w") as fh:
            fh.write(f"(cached failure) {e}")
        raise
    out = dict(out, cfg=txt, kind=kind)
    T._save(path, out)
    return {k: v for k, v in out.items()}


def _pos_real(lam, tol=1e-6, floor=1e-11):
    # a real growing eigenvalue: |Im| at round-off, which for these non-normal blocks is
    # ~kappa eps ||J|| (measured 1e-11 at kappa = 2e5), not eps ||J||; and Re above the nu = 0
    # null cluster (|lambda| <= 1e-10, Re ~ 1e-16: a shift below an under-resolved rung once
    # landed on a member at 2e-17) -- the smallest tearing rate here is ~1e-7
    lam = complex(lam)
    return lam.real > floor and abs(lam.imag) <= tol*abs(lam) + 1e-12


def point(S, ka, alpha, nx, Lx, sigma, lam_hint=None, root=None, log=print, a=1.0,
          direction="alpha", k=None, compute=True, tag=None, case="sech2", sector=0):
    # a cached sensitivity solve; retried at moved shifts (like T.ladder) when the harness
    # refuses or the eigenvalue found is not a real growing mode
    k = ka/a if k is None else k
    cfg = dict(S=S, k=k, a=a, alpha=alpha, nx=nx, Lx=Lx, direction=direction)
    if tag:
        cfg["tag"] = tag
    if case != "sech2":                 # sech^2 keys unchanged
        cfg.update(case=case, sector=sector)
    last = None
    # (shift offset, inner-rtol multiplier): attempts 4-6 loosen GMRES for non-normal blocks
    for attempt, (off, rm) in enumerate(((0.0, 1.0), (0.1, 1.0), (0.5, 1.0), (-0.05, 1.0),
                                         (0.0, 100.0), (0.1, 100.0), (0.0, 1e4))):
        sig = complex(sigma) + off*abs(sigma)
        try:
            rec = cached("sens", root, (lambda sig=sig, rm=rm: sens_solve(
                S, k, a, alpha, nx, Lx, sig, direction, lam_hint, case=case, sector=sector,
                rtol_mult=rm)) if compute else None,
                attempt=attempt, **cfg)
        except RuntimeError as e:
            last = str(e)
            log(f"    sens failed (S={S:.0e} ka={ka:g} alpha={alpha} nx={nx} attempt {attempt}): "
                f"{last[:160]}")
            continue
        if _pos_real(rec["lam"]):
            return rec
        log(f"    rejected {complex(rec['lam']):.4e} (S={S:.0e} ka={ka:g} alpha={alpha})")
    raise RuntimeError(f"no sensitivity at S={S}, ka={ka}, alpha={alpha}, nx={nx}: {last}")


def exponent(rec):
    # d ln gamma / d ln(1 - alpha^2) from a sensitivity record
    al, g, dg = rec["alpha"], complex(rec["lam"]).real, complex(rec["dlam"]).real
    return dg*(1 - al*al)/(-2*al*g)


# ============================================================ 1b-ref seeds and grids

@functools.lru_cache(maxsize=1)
def shear_tables():
    # 1b-ref's converged sech^2 ladders: {(ka, alpha): {S: res}} (main + extra ka), all cached
    import tearing_shear_run as R
    out = {}
    for (w, ka, al), fam in R.main_results(which=(2,)).items():
        out[(ka, al)] = fam
    for (ka, al), fam in R.extra_results().items():
        out[(ka, al)] = fam
    return out


@functools.lru_cache(maxsize=1)
def eigen_tables():
    # rung 1a's converged alpha = 0 ladders {ka: {S: res}} (S up to 1e6)
    return T.main_results()


def ref_seed(S, ka, alpha):
    # (gamma, final nx) of 1b-ref at the nearest tabulated (ka, alpha) with alpha <= given;
    # S = 1e6 (beyond 1b-ref): rung 1a's alpha = 0 value
    if S not in (1e3, 1e4, 1e5):
        r = eigen_tables()[ka][S]
        return complex(r["value"]).real, int(r["nx"][-1])
    tab = shear_tables()
    kas = sorted({k for k, _ in tab})
    als = sorted({a for _, a in tab})
    kk = min(kas, key=lambda q: abs(np.log(q/ka)))
    aa = max([a for a in als if a <= alpha + 1e-12] or [0.0])
    r = tab[(kk, aa)][S]
    return complex(r["value"]).real, int(r["nx"][-1])


def nx_for(S, ka, alpha):
    # 1b-ref's final ladder nx (its previous rung was already within 1e-7): nx_hi is that
    # previous rung, nx_lo its half. ka below the table's smallest -> the smallest ka's nx.
    if S not in (1e3, 1e4, 1e5):
        return max(int(eigen_tables()[ka][S]["nx"][-1])//2, 2048)
    tab = shear_tables()
    kas = sorted({k for k, _ in tab})
    lower = [q for q in kas if q <= ka + 1e-12] or [kas[0]]
    kk = lower[-1]
    # the most demanding alphas are the smallest (the flow widens the layer): max over 0, 0.3
    nx = max(int(tab[(kk, a_)][S]["nx"][-1]) for a_ in (0.0, 0.3))
    return max(nx//2, 2048)


def lx_of(ka):
    return T.lx_for(ka)


# ============================================================ phases

def alphas_for(S):
    return ALPHAS + (ALPHAS_TAIL if S >= 1e5 else ())


def alpha_family(S, ka, alphas=None, root=None, log=print, deadline=np.inf, levels=(1, 2),
                 compute=True):
    # sensitivity at every alpha (ascending), continuation in ln(1 - alpha^2) with the running
    # exponent itself: gamma_next ~ gamma (u_next/u)^p. levels: nx_hi/level for the grid check.
    # Returns ({level: {alpha: rec}}, complete)
    alphas = alphas or alphas_for(S)
    Lx = lx_of(ka)
    nx_hi = nx_for(S, ka, alphas[0])
    out = {lv: {} for lv in levels}
    g0, _ = ref_seed(S, ka, alphas[0])
    a_ref = max([a for a in (0.0, 0.3, 0.6, 0.8, 0.95) if a <= alphas[0] + 1e-12])
    g0 *= ((1 - alphas[0]**2)/(1 - a_ref**2))**0.55      # a first seed between table alphas
    pred, p_prev, u_prev, prev_g = g0, None, None, g0
    for al in alphas:
        if time.time() > deadline:
            return out, False
        u = 1 - al*al
        if p_prev is not None:
            pred = prev_g*(u/u_prev)**p_prev
        if al in (0.3, 0.6, 0.8, 0.95) and S in (1e3, 1e4, 1e5):
            try:
                pred = ref_seed(S, ka, al)[0]
            except KeyError:
                pass
        for lv in levels:
            nx = nx_hi//lv
            try:
                rec = point(S, ka, al, nx, Lx, pred*1.03, lam_hint=pred, root=root, log=log,
                            compute=compute)
            except KeyError:              # read-back of a point never computed
                log(f"  S={S:.0e} ka={ka} alpha={al} nx={nx}: NOT COMPUTED")
                continue
            except RuntimeError as e:
                log(f"  S={S:.0e} ka={ka} alpha={al} nx={nx}: FAILED {str(e)[:200]}")
                continue
            out[lv][al] = rec
            if lv == levels[0]:
                prev_g, p_prev, u_prev = complex(rec["lam"]).real, exponent(rec), u
                log(f"  S={S:.0e} ka={ka:g} alpha={al:<6g} nx={nx:6d} gamma={prev_g:.8e} "
                    f"p={p_prev:.5f} kappa={rec['kappa']:.3e} rel_err={rec['rel_err']:.1e} "
                    f"({rec['seconds']:.0f}s)")
    return out, True


def kmax_pred(S, alpha):
    # 1b-ref's k_max(alpha) (T.peak_fit over its 10 ka), extrapolated past 0.95 by the asymptotic
    # (1 - alpha^2)^(-1/7)
    tab = shear_tables()
    kas = sorted({k for k, _ in tab})
    def km(al):
        gs = [complex(tab[(k, al)][S]["value"]).real for k in kas]
        return T.peak_fit(kas, gs)[0]
    als = (0.0, 0.3, 0.6, 0.8, 0.95)
    if alpha <= 0.95:
        ks = [km(a) for a in als]
        return float(np.exp(np.interp(alpha, als, np.log(ks))))
    return km(0.95)*((1 - alpha**2)/(1 - 0.95**2))**(-1/7)


def ridge_kas(S, alpha, n=5):
    kp = kmax_pred(S, alpha)
    grid = np.array(RIDGE_KA)
    i = int(np.argmin(np.abs(np.log(grid/kp))))
    i = min(max(i, n//2), len(grid) - 1 - n//2)
    return tuple(float(q) for q in grid[i - n//2:i + n//2 + 1])


def ridge(S, alphas=RIDGE_ALPHAS, root=None, log=print, deadline=np.inf, compute=True):
    # {alpha: {ka: rec}} at the 5 grid ka around the predicted k_max(alpha); seeds from the
    # previous alpha at the same ka (continuation as in alpha_family), else from 1b-ref
    out = {}
    last = {}                     # ka -> (alpha, gamma, p)
    for al in alphas:
        out[al] = {}
        for ka in ridge_kas(S, al):
            if time.time() > deadline:
                return out, False
            if ka in last:
                a0, g0, p0 = last[ka]
                pred = g0*((1 - al*al)/(1 - a0*a0))**p0
            else:
                pred = _theory_seed(S, ka, al)
            nx = nx_for(S, ka, al)
            try:
                rec = point(S, ka, al, nx, lx_of(ka), pred*1.03, lam_hint=pred, root=root,
                            log=log, compute=compute)
            except KeyError:
                log(f"  ridge S={S:.0e} ka={ka} alpha={al}: NOT COMPUTED")
                continue
            except RuntimeError as e:
                log(f"  ridge S={S:.0e} ka={ka} alpha={al}: FAILED {str(e)[:200]}")
                continue
            out[al][ka] = rec
            last[ka] = (al, complex(rec["lam"]).real, exponent(rec))
            log(f"  ridge S={S:.0e} alpha={al:<6g} ka={ka:<7g} nx={nx:6d} gamma="
                f"{complex(rec['lam']).real:.8e} p={exponent(rec):.5f} ({rec['seconds']:.0f}s)")
    return out, True


def _theory_seed(S, ka, alpha):
    # a growth-rate guess at a (ka, alpha) 1b-ref never computed: interpolate its table in ln ka
    # at the nearest lower alpha, times the alpha-scaling (u/u_ref)^(4/7)
    tab = shear_tables()
    kas = np.array(sorted({k for k, _ in tab}))
    als = sorted({a for _, a in tab})
    aa = max([a for a in als if a <= alpha + 1e-12] or [0.0])
    gs = np.array([complex(tab[(k, aa)][S]["value"]).real for k in kas])
    g = float(np.exp(np.interp(np.log(ka), np.log(kas), np.log(gs))))
    return g*((1 - alpha**2)/(1 - aa**2))**(4/7)


def fd_gate(S, ka, alpha, root=None, log=print, hs=None, compute=True):
    # central differences of converged-grid eigenvalues in alpha against the sensitivity, at the
    # sensitivity's own nx (the discrete derivative is what the formula computes). h must stay
    # well inside 1 - alpha (at alpha = 0.99, h = 0.01 steps onto alpha = 1): FD_H_NEAR1 there
    if hs is None:
        hs = FD_H if 1 - alpha >= 0.05 - 1e-12 else FD_H_NEAR1
    fam, _ = alpha_family(S, ka, (alpha,), root=root, log=log, levels=(1,), compute=compute)
    rec = fam[1][alpha]
    lam = complex(rec["lam"])
    rows = []
    for h in hs:
        vals = []
        for sgn in (1, -1):
            al = alpha + sgn*h
            r = None
            for attempt, off in enumerate((0.03, 0.1, 0.5)):
                sig = lam*(1 + off)
                try:
                    r = cached("eig", root, (lambda al=al, sig=sig: eig_only(
                        S, ka, 1.0, al, int(rec["nx"]), float(rec["Lx"]), sig, lam_hint=lam))
                        if compute else None, S=S, k=ka, a=1.0, alpha=al, nx=int(rec["nx"]),
                        Lx=float(rec["Lx"]), **({"attempt": attempt} if attempt else {}))
                    break
                except RuntimeError as e:
                    log(f"    FD eig failed at alpha={al} (attempt {attempt}): {str(e)[:120]}")
            if r is None:
                raise RuntimeError(f"FD eigenvalue at alpha = {al} failed at every shift")
            vals.append(complex(r["lam"]))
        fd = (vals[0] - vals[1])/(2*h)
        rows.append((h, fd, abs(fd - complex(rec["dlam"]))))
        log(f"  FD S={S:.0e} ka={ka} alpha={alpha} h={h:g}: FD {fd.real:.10e}  sens "
            f"{complex(rec['dlam']).real:.10e}  |diff| {rows[-1][2]:.2e}")
    return rec, rows


def kappa_nx(S, ka, alpha, root=None, log=print, compute=True, nxs=KAPPA_NX):
    # {nx: rec} at one point; seeded by the family's converged eigenvalue
    g, _ = ref_seed(S, ka, alpha)
    out = {}
    for nx in nxs:
        try:
            out[nx] = point(S, ka, alpha, nx, lx_of(ka), g*1.03, lam_hint=g, root=root, log=log,
                            compute=compute)
        except (KeyError, RuntimeError) as e:
            log(f"  kappa_nx S={S:.0e} ka={ka} alpha={alpha} nx={nx}: {str(e)[:150]}")
            continue
        r = out[nx]
        log(f"  kappa_nx S={S:.0e} ka={ka} alpha={alpha} nx={nx:6d} gamma="
            f"{complex(r['lam']).real:.10e} dgamma={complex(r['dlam']).real:.10e} "
            f"kappa={float(r['kappa']):.4e} ({r['seconds']:.0f}s)")
    return out


# ============================================================ marginal Newton (width a)
# fixed k and eta; the root is a = 1: sech^2 at k = sqrt 5 (Lx = 20, rung 1a's box), the tanh
# pair at k = 1 (Lx = 30: sheet spacing 15/k) in each parity sector (+1: even member, -1: odd)
MARG = {"sech2": dict(k=SQRT5, Lx=20.0, a0=2.0/SQRT5),
        "tanhpair": dict(k=1.0, Lx=30.0, a0=0.8)}
MARG_S = (1e3, 1e4)
MARG_NXCAP = 32768
MARG_ITERS = 14


def marginal_ladder(S, a, nx0, sigma, root=None, log=print, compute=True, tol=1e-4,
                    case="sech2", sector=0):
    # the width-direction sensitivity at nx0, 2 nx0, ... until gamma and dgamma/da agree to tol
    # (relative) between rungs, or the cap. sigma continued at 0.8 x the previous rung (the
    # eigenvalue FALLS with nx near marginality, measured in rung 1a)
    c = MARG[case]
    rungs = []
    nx = int(nx0)
    while nx <= MARG_NXCAP:
        # the eigenvalue may fall (sech^2) or rise (tanh pair, S = 1e4) between rungs by a factor
        # ~3 while the layer is under-resolved: shifts at 1, 2.5 and 0.4 x the previous rung
        # (distinct cache tags), the first real growing mode above the null cluster accepted
        rec, err = None, None
        for mult in (1.0, 2.5, 0.4):
            try:
                rec = point(S, c["k"]*a, 0.0, nx, c["Lx"], sigma*mult, lam_hint=None, root=root,
                            log=log, a=a, direction="a", k=c["k"], compute=compute, case=case,
                            sector=sector, tag=None if mult == 1.0 else f"shift{mult:g}")
                break
            except RuntimeError as e:
                err = str(e)
            except KeyError:              # read-back of a rung never computed
                return rungs, "not computed"
        if rec is None:
            log(f"    marginal {case}{sector or ''} a={a:.10f} nx={nx}: {err[:300]}")
            return rungs, err
        rungs.append(rec)
        g, dg = complex(rec["lam"]).real, complex(rec["dlam"]).real
        log(f"    {case}{sector or ''} a={a:.10f} (ka={c['k']*a:.8f}) nx={nx:6d} gamma={g:.8e} "
            f"dgamma/da={dg:.6e} kappa={rec['kappa']:.3e} rel_err={rec['rel_err']:.1e} "
            f"({rec['seconds']:.0f}s)")
        if len(rungs) >= 2:
            g1, dg1 = complex(rungs[-2]["lam"]).real, complex(rungs[-2]["dlam"]).real
            if abs(g - g1) <= tol*abs(g) and abs(dg - dg1) <= tol*abs(dg):
                return rungs, None
        sigma = g
        nx *= 2
    return rungs, "nx cap"


def marginal_newton(S, root=None, log=print, deadline=np.inf, compute=True, case="sech2",
                    sector=0, iters=MARG_ITERS):
    # plain Newton on Re lambda(a) = 0 at fixed k and eta: a <- a - gamma/(dgamma/da), every
    # iterate converged in nx (gamma and dgamma/da to 1e-4). Returns (iterates, stop reason)
    c = MARG[case]
    its = []
    a = c["a0"]
    if compute:               # the seed shift (read-back never solves: sigma is not in the key)
        top, _ = _dense_top(S, c["k"], a, c["Lx"], case=case, sector=sector)
        sigma = top.real*1.05
    else:
        sigma = 1.0
    nx0 = 2048
    for n in range(iters):
        if time.time() > deadline:
            return its, "deadline"
        rungs, why = marginal_ladder(S, a, nx0, sigma, root, log, compute, case=case,
                                     sector=sector)
        if not rungs or why is not None:
            its.append(dict(a=a, rungs=rungs, stop=why or "no rung"))
            return its, why or "no rung"
        rec = rungs[-1]
        g, dg = complex(rec["lam"]).real, complex(rec["dlam"]).real
        a_new = a - g/dg
        its.append(dict(a=a, gamma=g, dgamma=dg, step=-g/dg, rec=rec, rungs=rungs,
                        nx=int(rec["nx"])))
        log(f"  Newton {case}{sector or ''} S={S:.0e} it {n}: a={a:.10f} gamma={g:.4e} -> "
            f"a={a_new:.10f} (1 - a = {1 - a_new:.3e})")
        # next shift: the power-law prediction gamma ~ (a* - a)^p, p from the last two iterates
        # (1/p = the slope of gamma/gamma' in a), default p = 1
        p = 1.0
        if len(its) >= 2:
            q0, q1 = its[-2]["gamma"]/its[-2]["dgamma"], g/dg
            slope = (q1 - q0)/(a - its[-2]["a"])
            if slope > 0:
                p = 1.0/slope
        ratio = max(1.0 - 1.0/p, 0.05)**p if p > 1 else 0.3
        sigma = g*ratio*1.1
        nx0 = max(2048, int(rec["nx"])//2)
        a = a_new
    return its, "iterations"


def _dense_top(S, k, a, Lx, nx=NX_DENSE, case="sech2", sector=0):
    _, top, npos = _dense_others_cached(S, k, a, 0.0, Lx, nx, case, sector)
    return top, npos


def newton_table(its):
    # per iterate: a, gamma, gamma/gamma', the multiplicity estimate p = 1/slope(gamma/gamma')
    # from consecutive iterates, and the multiplicity-corrected root a - p gamma/gamma'
    rows = []
    for i, it in enumerate(its):
        if "gamma" not in it:
            continue
        q = it["gamma"]/it["dgamma"]
        p, astar = np.nan, np.nan
        if i >= 1 and "gamma" in its[i - 1]:
            q0 = its[i - 1]["gamma"]/its[i - 1]["dgamma"]
            slope = (q - q0)/(it["a"] - its[i - 1]["a"])
            p = 1.0/slope
            astar = it["a"] - p*q
        rows.append(dict(a=it["a"], gamma=it["gamma"], dgamma=it["dgamma"], q=q, p=p,
                         astar=astar, nx=it["nx"], kappa=float(it["rec"]["kappa"]),
                         rel_err=float(it["rec"]["rel_err"])))
    return rows


# ============================================================ make_data

PHASES = ("fd", "const", "const2", "nonconst", "ridge", "kappa_nx", "marginal", "marginal_tanh")


def make_data(root=None, wall_budget=3*3600.0, phases=None, log=print, S_list=None):
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    done = True
    for name in phases or PHASES:
        t0 = time.time()
        ok = True
        if name == "fd":
            for (S, ka, al) in FD_POINTS:
                fd_gate(S, ka, al, root, log)
        elif name in ("const", "const2", "nonconst"):
            ka = {"const": KA_CONST, "const2": KA_CONST2, "nonconst": KA_NONCONST}[name]
            for S in (S_list or (S_EXP if name == "nonconst" else S_EXP_CONST)):
                if name == "nonconst" and S >= 1e5:
                    alpha_family(S, ka, (0.1,), root=root, log=log, deadline=deadline)
                    _, ok1 = alpha_family(S, ka, ALPHAS_NC_1E5, root=root, log=log,
                                          deadline=deadline, levels=(1,))
                    ok &= ok1
                    continue
                _, ok1 = alpha_family(S, ka, root=root, log=log, deadline=deadline)
                ok &= ok1
        elif name == "ridge":
            for S in (S_list or S_RIDGE):
                _, ok1 = ridge(S, root=root, log=log, deadline=deadline)
                ok &= ok1
        elif name == "fd_near1":             # the extra smallest h of the alpha = 0.99 gate
            fd_gate(1e5, KA_NONCONST, 0.99, root, log, hs=(3e-4,))
        elif name.startswith("nc5:"):                # a slice of the ka = 0.1, S = 1e5 family
            als = tuple(float(v) for v in name[4:].split(","))
            _, ok = alpha_family(1e5, KA_NONCONST, als, root=root, log=log, deadline=deadline,
                                 levels=(1,))
        elif name == "kappa_nx":
            for pt in KAPPA_NX_POINTS:
                kappa_nx(*pt, root=root, log=log)
        elif name == "marginal":
            for S in (S_list or MARG_S):
                its, why = marginal_newton(S, root, log, deadline)
                log(f"== marginal sech2 S={S:.0e}: {len(its)} iterates, stop: {why}")
        elif name == "marginal_tanh":
            for S in (S_list or MARG_S):
                for sector in (1, -1):
                    its, why = marginal_newton(S, root, log, deadline, case="tanhpair",
                                               sector=sector)
                    log(f"== marginal tanhpair sector {sector} S={S:.0e}: {len(its)} iterates, "
                        f"stop: {why}")
        else:
            raise ValueError(name)
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")
        done &= ok
    return bool(done)


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def family_results(S, ka):
    # {level: {alpha: rec}} of a fixed-k family (level 1 = nx_hi, 2 = nx_hi/2)
    if ka == KA_NONCONST and S >= 1e5:
        out = alpha_family(S, ka, ALPHAS_NC_1E5, root=DATA_ROOT, log=_quiet, compute=False,
                           levels=(1,))[0]
        out[2] = alpha_family(S, ka, (0.1,), root=DATA_ROOT, log=_quiet, compute=False)[0][2]
        return out
    return alpha_family(S, ka, root=DATA_ROOT, log=_quiet, compute=False)[0]


def ridge_results(S):
    return ridge(S, root=DATA_ROOT, log=_quiet, compute=False)[0]


def fd_results():
    out = {}
    for pt in FD_POINTS:
        try:
            out[pt] = fd_gate(*pt, root=DATA_ROOT, log=_quiet, compute=False)
        except (KeyError, RuntimeError):     # not (yet) computed
            pass
    return out


def marginal_results(S, case="sech2", sector=0):
    return marginal_newton(S, DATA_ROOT, _quiet, compute=False, case=case, sector=sector)


def ridge_fit(rows):
    # rows: {ka: rec} at one alpha. Quadratic in ln k through the 5 points for gamma (-> k_max,
    # gamma_max) and for dgamma/dalpha (-> its value at k_max: the envelope theorem makes that
    # d gamma_max/d alpha), plus the cubic-vs-quadratic spread as the fit error; and
    # dk_max/dalpha = -gamma_ka/gamma_kk (from the ln-k derivative of the dgamma/dalpha fit).
    kas = np.array(sorted(rows))
    lk = np.log(kas)
    g = np.array([complex(rows[k]["lam"]).real for k in kas])
    dg = np.array([complex(rows[k]["dlam"]).real for k in kas])
    al = float(rows[kas[0]]["alpha"])
    cg = np.polyfit(lk, g, 2)
    lkm = -cg[1]/(2*cg[0])
    gm = np.polyval(cg, lkm)
    cd = np.polyfit(lk, dg, 2)
    dgm = np.polyval(cd, lkm)
    c3 = np.polyfit(lk, g, 3)
    xs = np.linspace(lk[0], lk[-1], 20001)
    j = int(np.argmax(np.polyval(c3, xs)))
    # d lnk_max/d alpha = -(d/dlnk dgamma/dalpha)/(d2gamma/dlnk2)
    dlkm = -np.polyval(np.polyder(cd), lkm)/(2*cg[0])
    u = 1 - al*al
    return dict(alpha=al, kmax=float(np.exp(lkm)), gmax=float(gm), dgmax=float(dgm),
                p=float(dgm*u/(-2*al*gm)), inside=bool(lk[0] < lkm < lk[-1]),
                kmax_err=float(abs(np.exp(xs[j]) - np.exp(lkm))),
                gmax_err=float(abs(np.polyval(c3, xs[j]) - gm)),
                pk=float(dlkm*u/(-2*al)))


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3*3600.0
    args = sys.argv[2:]
    S_list = None
    if args and args[0].startswith("S="):
        S_list = tuple(float(s) for s in args[0][2:].split(","))
        args = args[1:]

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    make_data(wall_budget=budget, phases=tuple(args) or None, log=log, S_list=S_list)
