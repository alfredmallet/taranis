# Oblique (3D) resistive tearing with and without shear flow -- theory, eigen-solves and resumable
# data generation for examples/tearing-oblique.ipynb (plans/AUTODIFF_PLAN.md rung 2b item 4).
#
# SETUP. RMHD in normalized Alfven units: guide field B0 = 1 along z (v_A = 1 along z), a 1D
# equilibrium b0 = f(x) y^ (B_y = d psi0/dx), flow u0 = u(x) y^ (u_y = d phi0/dx), S = 1/eta,
# nu = Pm/S, hyper = 1, fp64. A perturbation exp(i ky y + i kz z) is resonant where
#       ky f(x_s) + kz = 0,   i.e.   f(x_s) = c := -kz/ky
# (the sign is taranis' and is GATED: tests/test_tearing_oblique.py locates the current peak).
#
# THE EXACT REDUCTION (derived in the notebook). In RMHD the perpendicular Laplacian has no kz,
# dissipation is perpendicular (z_diss_k = 0) and the equilibrium enters J only through
# ky f(x) + kz = ky (f - c), ky f'' and ky u. So the (ky, kz) block for (f, u) IS the kz = 0 block
# for the profile f - c (a uniform B_y = -c added) with the same flow. For FIELD-ALIGNED flow
# u = alpha f, u = alpha (f - c) + alpha c: the shifted problem is again field-aligned, plus a
# uniform flow alpha c, which only Doppler shifts. Hence, exactly,
#       lambda(ky, kz; f, alpha f) = lambda_2D(ky; f - c, alpha (f - c)) - i ky alpha c,
# alpha is the same at every surface, and the oblique question is the 2D question on the
# ASYMMETRIC-magnitude sheet f - c (|B_y| -> 1 - c and 1 + c for tanh). Non-aligned flows break
# this: alpha_loc(x_s) = u'(x_s)/f'(x_s) varies with c (the "tanh" flow with R != 1 below).
#
# OUTER THEORY (alpha = 0): psi'' = [ky^2 + f''/(f - c)] psi, singular at x_s, where
# psi = A [1 + r s ln|s| + ...] + B [s + ...] (s = x - x_s, r = f''(x_s)/f'(x_s)): the log term
# has the same coefficient on both sides, so Delta' = B+/A - B-/A is convention-free. dprime_iso
# shoots it with a second-order Frobenius start (Richardson-checked in eps). Closed forms:
# c = 0 tanh: Delta' = 2(1/k - k); small k, any profile with f(+-inf) = f+-:
#       Delta' -> (f'(x_s)^2/k) [1/(f+ - c)^2 + 1/(f- - c)^2]       (tanh: 2(1 + c^2)/k).
# Layer: the exact nonconstant-psi relation (rung 1a, T.gamma_disp) with k|f'(0)| ->
# ky |f'(x_s)| and Delta' -> Delta'(ky, c). With Delta' = A/k the maximum over k is self-similar:
#       gamma_max = G_MAX (A |f'(x_s)| eta)^(1/2),   k_max = KAP_MAX A^(3/4) eta^(1/4) |f'(x_s)|^(-1/4),
# so for tanh gamma_max(c)/gamma_max(0) -> (1 - c^4)^(1/2) as S -> inf: the k_z = 0 mode is the
# fastest, by a margin QUARTIC in c. For an asymmetric profile (ASYM below) the maximiser of
# f'(x_s)^(3/2) [1/(f+ - c)^2 + 1/(f- - c)^2]^(1/2) is oblique.
#
# NUMERICS. dims = 3, z_spectral = True, nz = 4, ny = 8, comm_backend "serial", the block
# ky_block_*(iky = 1, iz = IZ) of a z-independent x0: iz = 1 is kz = +2 pi/Lz, iz = 3 is
# kz = -2 pi/Lz (nz = 4 keeps |iz| = 1 under the 2/3 kz cut), iz = 0 is kz = 0. ONE Parameters per
# (nx, Lx) (make_params: placeholders Ly = Lz = 2 pi, diss = (0, 1e-6)); every operator is built at
# overrides = {"Ly": 2 pi/k, "Lz": 2 pi/|kz|, "diss": (Pm/S, 1/S)} (overrides()). Profiles:
#   "pair"  rung 1b-ref's exact tanh pair (tearing_ofman_run.by_pair; sheets at Lx/4, 3Lx/4, B_y =
#           tanh(x - Lx/4) and -tanh(x - 3Lx/4) locally). Flows "none" and "tanh" (u = V by_pair(R),
#           R = 1 is u = V b0 bitwise). Two sheets -> an even/odd pair (about the midpoint), split
#           ~e^(-k Lx/2); with c != 0 sheet 1 resonates at Lx/4 + atanh(c), sheet 2 at 3Lx/4 -
#           atanh(c) (the profile is even about 0 and Lx/2, so the parity is still defined). We
#           report the member of larger Re and store its parity psi(x_s2)/psi(x_s1).
#   "sech2" rung 1a/2's single sheet Psi0 = sech^2 (U.psi0_sech2_a), aligned flow phi = alpha psi.
#           f = -2 tanh sech^2 is NOT monotonic: for 0 < |c| < 4/(3 sqrt 3) = 0.770 f = c has a
#           CENTRAL root (|f'| ~ 2) and a TAIL root on the far side (|f'| ~ 2|c|): two surfaces.
# Solver: tearing_ofman_run's ladder -- a dense seed (np.linalg.eigvals of the block) at NX_SEED,
# then band-LU-preconditioned shift-invert rungs (k = 2 eigenvalues) doubling nx until two agree to
# CONV_TOL. The band of J(kz) is J(kz0)'s plus i (kz - kz0) sigma_x EXACTLY (L's kz part is k-local),
# so one band probe serves a whole c scan (band_for; the exact identity is gated). Residuals are
# always the harness's own, against the exact 3D operator.
#
# INDEPENDENT ROUTE (no taranis): fd_oblique, tearing_ofman_run.fd_single_sheet's one-sheet
# finite-difference eigencode with k F -> k F + kz (walls, stretched grid, sparse LU).
#
# `python examples/tearing_oblique_run.py [wall_s] [phase ...]` runs make_data (resumable; every
# solve is one cache file under DATA_ROOT/solves). TARANIS_OBL_PROFILE=smoke|full (default full).
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import functools
import sys
import time

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

import jax.numpy as jnp

import taranis as jr
from taranis import stability as st

import tearing_eigen_run as T
import tearing_sensitivity_run as U
import tearing_ofman_run as O

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FULL = os.path.join(HERE, "data", "tearing-oblique")
DATA_ROOT_SMOKE = os.path.join(HERE, "data", "tearing-oblique-smoke")
PROFILE = os.environ.get("TARANIS_OBL_PROFILE", "full")
if PROFILE not in ("full", "smoke"):
    raise ValueError(f"TARANIS_OBL_PROFILE must be 'full' or 'smoke', got {PROFILE!r}")
DATA_ROOT = DATA_ROOT_SMOKE if PROFILE == "smoke" else DATA_ROOT_FULL

TWO_PI = 2*np.pi
NZ = 4
IZ_POS, IZ_NEG = 1, 3            # kz = +2 pi/Lz, -2 pi/Lz at nz = 4
CASES = ("pair", "sech2")
FLOWS = ("none", "tanh")
DIFF_PLACEHOLDER = (0.0, 1e-6)
SECH2_FMAX = 4.0/(3.0*np.sqrt(3.0))   # max |f| of f = -2 tanh sech^2


# ============================================================ profiles (theory side)

class Profile:
    # an isolated sheet B_y = f(x): f, f', f'', f''' (vectorised) and f(+-inf)
    def __init__(self, name, f, f1, f2, f3, fplus, fminus):
        self.name, self.f, self.f1, self.f2, self.f3 = name, f, f1, f2, f3
        self.fplus, self.fminus = fplus, fminus


def _sech2(x):
    return 1.0/np.cosh(x)**2


TANH = Profile("tanh", np.tanh, _sech2,
               lambda x: -2*np.tanh(x)*_sech2(x),
               lambda x: _sech2(x)*(4*np.tanh(x)**2 - 2*_sech2(x)), 1.0, -1.0)

ASYM_BETA = 0.3


def _asym(beta):
    # f = tanh x + beta sech^2 x (|beta| < 1/2 keeps f' = sech^2 (1 - 2 beta tanh) > 0): the same
    # asymptotic fields +-1, an asymmetric current sheet (peak off x = 0)
    t = np.tanh
    return Profile(f"asym{beta:g}",
                   lambda x: t(x) + beta*_sech2(x),
                   lambda x: _sech2(x)*(1 - 2*beta*t(x)),
                   lambda x: _sech2(x)*(-2*t(x) - 2*beta*(1 - 3*t(x)**2)),
                   lambda x: _sech2(x)*(4*t(x)**2 - 2*_sech2(x)
                                        + beta*(12*t(x)*_sech2(x) + 4*t(x)*(1 - 3*t(x)**2))),
                   1.0, -1.0)


ASYM = _asym(ASYM_BETA)


def x_res(c, prof=TANH, lo=-40.0, hi=40.0):
    # the resonant surface f(x_s) = c of a monotonic profile
    if not prof.fminus < c < prof.fplus:
        raise ValueError(f"c = {c} outside ({prof.fminus}, {prof.fplus}): no resonant surface")
    if prof is TANH:
        return float(np.arctanh(c))
    return float(brentq(lambda x: prof.f(x) - c, lo, hi, xtol=1e-15, rtol=1e-15))


def fprime_s(c, prof=TANH):
    # |f'(x_s)|: the local shear of the resonant surface (tanh: 1 - c^2)
    return float(abs(prof.f1(x_res(c, prof))))


def frob_coeffs(k, c, prof=TANH):
    # (x_s, r, q): psi'' = (r/s + q + O(s)) psi near x_s, r = f''/f', q = k^2 + f'''/f' - r^2/2
    xs = x_res(c, prof)
    f1, f2, f3 = prof.f1(xs), prof.f2(xs), prof.f3(xs)
    r = f2/f1
    return xs, float(r), float(k*k + f3/f1 - 0.5*r*r)


def _frob_basis(s, r, q):
    # (u_a, u_a', u_b, u_b') at s = x - x_s: u_b = s + r s^2/2 + (r^2/2 + q) s^3/6 (regular),
    # u_a = 1 + r u_b ln|s| + (q - 3 r^2/2) s^2/2 (the convention: no pure-s term in u_a)
    b3 = (0.5*r*r + q)/6.0
    ub = s + 0.5*r*s*s + b3*s**3
    ubp = 1.0 + r*s + 3*b3*s*s
    d2 = 0.5*(q - 1.5*r*r)
    L = np.log(abs(s))
    ua = 1.0 + r*ub*L + d2*s*s
    uap = r*ubp*L + r*ub/s + 2*d2*s
    return ua, uap, ub, ubp


def _shoot(k, c, prof, x0, y0, xe):
    # integrate psi'' = (k^2 + f''/(f - c)) psi from x0 (state y0) to xe
    k2 = float(k)**2
    rhs = lambda x, y: [y[1], (k2 + prof.f2(x)/(prof.f(x) - c))*y[0]]
    sol = solve_ivp(rhs, [x0, xe], y0, rtol=1e-13, atol=1e-300, method="DOP853")
    if sol.status != 0:
        raise RuntimeError(f"dprime shooting failed: {sol.message}")
    return sol.y[0, -1], sol.y[1, -1]


def _ab(psi, dpsi, s, r, q):
    ua, uap, ub, ubp = _frob_basis(s, r, q)
    det = ua*ubp - uap*ub
    return (psi*ubp - dpsi*ub)/det, (ua*dpsi - uap*psi)/det


def dprime_iso(k, c, prof=TANH, X=30.0, eps=1e-4):
    # Delta'(k, c) of an ISOLATED sheet (physical units, a = 1): psi ~ e^(-k|x|) at +-X (the
    # potential f''/(f - c) decays like e^(-2|x|)), shot to x_s +- eps, matched to the Frobenius
    # pair: Delta' = B+/A+ - B-/A-
    xs, r, q = frob_coeffs(k, c, prof)
    pr = _shoot(k, c, prof, xs + X, [1.0, -k], xs + eps)
    pl = _shoot(k, c, prof, xs - X, [1.0, k], xs - eps)
    ar, br = _ab(*pr, eps, r, q)
    al, bl = _ab(*pl, -eps, r, q)
    return float(br/ar - bl/al)


def dprime_pair(k, c, Lx, parity):
    # Delta'(k, c) at sheet 1 of the periodic tanh PAIR (box Lx), for the mode even (+1) or odd
    # (-1) about the midpoint Lx/2. The pair B_y is even about Lx/2 AND about 0 (one reflection
    # times the other is the period), so the mode has the same parity about both: shoot from Lx/2
    # and from 0 to sheet 1's surface Lx/4 + atanh(c), local coordinate u = x - Lx/4
    xs, r, q = frob_coeffs(k, c, TANH)
    y0 = [1.0, 0.0] if parity > 0 else [0.0, 1.0]
    pair = Profile("pair", lambda u: O.by_pair(u + Lx/4, Lx),
                   None, lambda u: _pair_f2(u + Lx/4, Lx), None, 1.0, -1.0)
    eps = 1e-4
    pr = _shoot(k, c, pair, Lx/4, y0, xs + eps)
    pl = _shoot(k, c, pair, -Lx/4, y0, xs - eps)
    ar, br = _ab(*pr, eps, r, q)
    al, bl = _ab(*pl, -eps, r, q)
    return float(br/ar - bl/al)


def _pair_f2(x, Lx):
    # d^2/dx^2 of by_pair: sum of -2 tanh sech^2 with the pair's signs
    f2 = lambda u: -2*np.tanh(u)*_sech2(u)
    return O.sheet_sum(x, Lx, f2)


def dprime_smallk_A(c, prof=TANH):
    # the small-k limit k Delta' -> A(c) = f'(x_s)^2 [1/(f+ - c)^2 + 1/(f- - c)^2] (tanh: 2(1 + c^2))
    f1 = fprime_s(c, prof)
    return f1*f1*(1.0/(prof.fplus - c)**2 + 1.0/(prof.fminus - c)**2)


def _gk(lk):
    # the self-similar growth Lam(kappa) kappa^(2/3) at Delta' delta_eta = kappa^(-4/3)
    kap = np.exp(lk)
    return T.lam_of(kap**(-4.0/3.0))*kap**(2.0/3.0)


@functools.lru_cache(maxsize=1)
def selfsimilar_max():
    # (KAP_MAX, G_MAX): with Delta' = A/k, gamma = G(kappa) (A f' eta)^(1/2) at
    # k = kappa A^(3/4) eta^(1/4) f'^(-1/4); the maximum over kappa
    r = minimize_scalar(lambda lk: -_gk(lk), bounds=(np.log(0.05), np.log(20.0)),
                        method="bounded", options=dict(xatol=1e-12, maxiter=500))
    return float(np.exp(r.x)), float(-r.fun)


def gamma_max_asym(c, S, prof=TANH):
    # the S -> inf asymptote: (k_max, gamma_max) with Delta' = A(c)/k exactly
    kap, g = selfsimilar_max()
    A, f1, eta = dprime_smallk_A(c, prof), fprime_s(c, prof), 1.0/S
    return kap*A**0.75*eta**0.25*f1**-0.25, g*(A*f1*eta)**0.5


def asym_rule(c, prof=TANH):
    # the large-S ranking function f'(x_s)^(3/2) [1/(f+ - c)^2 + 1/(f- - c)^2]^(1/2)
    #   (gamma_max proportional to its square root times f'^0 ... = (A f')^(1/2))
    return (dprime_smallk_A(c, prof)*fprime_s(c, prof))**0.5


def gamma_layer(k, c, S, prof=TANH, Lx=None, parity=1, dprime=None):
    # the exact resistive layer relation at the shifted surface: k|f'(0)| -> k|f'(x_s)|,
    # Delta' -> Delta'(k, c) (isolated, or the pair's at box Lx and parity); nan if Delta' <= 0
    if dprime is None:
        dprime = dprime_iso(k, c, prof) if Lx is None else dprime_pair(k, c, Lx, parity)
    return float(T.gamma_disp(float(k), 1.0/S, fprime_s(c, prof), dprime))


def theory_kmax(c, S, prof=TANH, Lx=None, parity=1, bracket=(0.02, 0.99)):
    # (k_max, gamma_max) of gamma_layer at fixed c (bounded maximisation in ln k)
    f = lambda lk: -np.nan_to_num(gamma_layer(np.exp(lk), c, S, prof, Lx, parity), nan=0.0)
    r = minimize_scalar(f, bounds=np.log(bracket), method="bounded",
                        options=dict(xatol=1e-7, maxiter=300))
    return float(np.exp(r.x)), float(-r.fun)


def kmarg(c, prof=TANH):
    # the marginal wavenumber Delta'(k_m, c) = 0 (isolated sheet)
    return float(brentq(lambda k: dprime_iso(k, c, prof), 0.05, 1.5, xtol=1e-12))


def current_peak(prof=ASYM):
    # the position of the current maximum, argmax f'
    r = minimize_scalar(lambda x: -prof.f1(x), bounds=(-3.0, 3.0), method="bounded",
                        options=dict(xatol=1e-12))
    return float(r.x)


def asym_cstar(prof=ASYM, bracket=(-0.9, 0.9)):
    # the large-S fastest surface: argmax_c asym_rule(c)
    r = minimize_scalar(lambda c: -asym_rule(c, prof), bounds=bracket, method="bounded",
                        options=dict(xatol=1e-10))
    return float(r.x), float(-r.fun)


# ============================================================ equilibria and operators

def block_spec(k, c):
    # (iz, Lz) of the block with ky = k, kz = -c k: kz = 0 is iz = 0 (Lz a placeholder), kz > 0
    # iz = IZ_POS and kz < 0 iz = IZ_NEG, with Lz = 2 pi/|kz|
    kz = -float(c)*float(k)
    if kz == 0.0:
        return 0, TWO_PI
    return (IZ_POS if kz > 0 else IZ_NEG), TWO_PI/abs(kz)


def overrides(k, c, S, Pm):
    # the parameter point of block (k, c): Ly, Lz and diss = (Pm/S, 1/S). Every block carries the
    # same three keys, so one compile serves every point of a (nx, Lx)
    _, Lz = block_spec(k, c)
    return {"Ly": TWO_PI/float(k), "Lz": float(Lz), "diss": (float(Pm)/float(S), 1.0/float(S))}


@functools.lru_cache(maxsize=16)
def make_params(nx, Lx):
    # ONE Parameters per (nx, Lx): Ly, Lz and diss are placeholders, every operator is built at
    # overrides(k, c, S, Pm)
    return jr.Parameters(nx=int(nx), ny=8, nz=NZ, Lx=float(Lx), Ly=TWO_PI, Lz=TWO_PI, dims=3,
                         z_spectral=True, comm_backend="serial", cfl_safety=0.5, forcing=False,
                         eqpars={"diss": DIFF_PLACEHOLDER, "hyper": 1})


@functools.lru_cache(maxsize=16)
def _kgrid(nx, Lx):
    return jr.setup_kgrids(make_params(nx, Lx))


def fields3(params, phi, psi):
    # the z-independent fields array of 1D profiles phi(x), psi(x)
    a = jnp.asarray(phi).reshape(1, -1, 1)
    b = jnp.asarray(psi).reshape(1, -1, 1)
    return jr.initialize(lambda X, Y, Z: jnp.stack([a + 0*Y + 0*Z, b + 0*Y + 0*Z]), params).fields


def equilibrium_1d(case, nx, Lx, flow, V_, R):
    # (phi, psi) on the grid. pair: tearing_ofman_run's tanh pair and its flows (zero mean);
    # sech2: Psi0 = sech^2 (+ images), phi = V psi (aligned; flow "tanh" needs R = 1 here)
    if case == "pair":
        phi, psi, um = O.equilibrium_fields(nx, Lx, flow, V_, R)
        if um != 0.0:
            raise ValueError("only zero-mean flows here")
        return phi, psi
    if case == "sech2":
        if flow == "tanh" and R != 1.0:
            raise ValueError("sech2 takes the aligned flow only (R = 1)")
        x = np.arange(nx)*Lx/nx
        psi = U.psi0_sech2_a(x, Lx)
        return (V_*psi if flow == "tanh" else 0.0*psi), psi
    raise ValueError(f"case must be one of {CASES}, got {case!r}")


def x0_of(case, nx, Lx, flow, V_, R):
    params = make_params(nx, Lx)
    return fields3(params, *equilibrium_1d(case, nx, Lx, flow, V_, R))


def sheet_index(case, nx):
    return nx//4 if case == "pair" else nx//2


BAND_Q = 10.0          # the preconditioner's band in PHYSICAL wavenumber (rung 1a used 21, for a 5e-9
                       # preconditioner; 10 leaves GMRES iteration counts unchanged here, measured)


def band_of(case, Lx, flow, R):
    # kx index band of the preconditioner: BAND_Q Lx/(2 pi), widened by R for a narrow tanh flow
    b = int(np.ceil(BAND_Q*Lx/TWO_PI)) + 2
    return b if (case == "sech2" or flow == "none" or R <= 1.0) else int(np.ceil(R*(b - 2))) + 2


PROBE_BATCH = 32


def _batched(case, S, Pm, k, c, flow, V_, R, Lx, nx):
    # (U (b, n) -> J U^T rows, idx): the block's compiled batch map (stability's own vmapped
    # J; ~3x cheaper per probe than LinearOperator.matvec calls)
    params = make_params(nx, Lx)
    iz, _ = block_spec(k, c)
    kg = st._kgrid_at(_kgrid(nx, Lx), params, overrides(k, c, S, Pm))
    x0 = st._fields(x0_of(case, nx, Lx, flow, V_, R), params)
    idx = st.ky_block_index(kg, params, 1, iz=iz)
    st._assert_invariant(x0, None, iz)
    args = st._block_args(idx, 1)
    fns = st._space_fns(params, "block")
    return (lambda Um: np.asarray(fns.batch(jnp.asarray(Um, dtype=jnp.complex128), x0, kg, args))), idx


def band_matrix_batched(batch, idx, nx, band, chunk=PROBE_BATCH):
    # U.band_matrix (coloured probes of the banded part), the probes applied in batches
    n = len(idx)
    P = 1
    while P < 2*band + 1:
        P *= 2
    P = min(P, nx)
    lut = -np.ones((2, nx), dtype=np.int64)
    lut[idx[:, 0], idx[:, 2]] = np.arange(n)
    fld, ikx = idx[:, 0], idx[:, 2]
    colours = [(f, cc) for f in (0, 1) for cc in range(P) if np.any((fld == f) & (ikx % P == cc))]
    rows, cols, vals = [], [], []
    for j0 in range(0, len(colours), chunk):
        part = colours[j0:j0 + chunk]
        Um = np.zeros((chunk, n), dtype=complex)
        for i, (f, cc) in enumerate(part):
            Um[i, (fld == f) & (ikx % P == cc)] = 1.0
        imgs = batch(Um)
        for i, (f, cc) in enumerate(part):
            img = imgs[i]
            d = (cc - ikx) % P
            d = np.where(d > P//2, d - P, d)
            cand = lut[f, (ikx + d) % nx]
            ok = (cand >= 0) & (np.abs(d) <= band) & (img != 0)
            rows.append(np.nonzero(ok)[0])
            cols.append(cand[ok])
            vals.append(img[ok])
    return sps.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))


def block_matrix(case, S, Pm, k, c, flow, V_, R, Lx, nx):
    # the dense (k, c) block of the 3D Jacobian and its idx
    params = make_params(nx, Lx)
    iz, _ = block_spec(k, c)
    return st.ky_block_matrix(x0_of(case, nx, Lx, flow, V_, R), _kgrid(nx, Lx), params, 1, iz=iz,
                              overrides=overrides(k, c, S, Pm))


def block_operator(case, S, Pm, k, c, flow, V_, R, Lx, nx):
    params = make_params(nx, Lx)
    iz, _ = block_spec(k, c)
    return st.ky_block_operator(x0_of(case, nx, Lx, flow, V_, R), _kgrid(nx, Lx), params, 1,
                                iz=iz, overrides=overrides(k, c, S, Pm))


def dense_top(case, S, Pm, k, c, flow, V_, R, Lx, nx, nkeep=8):
    B, _ = block_matrix(case, S, Pm, k, c, flow, V_, R, Lx, nx)
    vals = np.linalg.eigvals(B)
    return vals[np.argsort(-vals.real, kind="stable")][:nkeep]


def sigma_x(n):
    # the phi <-> psi coupling of the block (entries ordered field-major: the first n/2 are phi,
    # the next n/2 psi at the same ikx): L's kz part is i kz sigma_x
    h = n//2
    I = sps.identity(h, format="csc")
    return sps.bmat([[None, I], [I, None]], format="csc")


_BAND_MEMO = {}


def band_for(case, S, Pm, k, c, flow, V_, R, Lx, nx, op, idx):
    # the band of J at (k, c): probed once per (case, S, Pm, k, flow, V, R, Lx, nx) and moved to
    # another c by the exact identity J(kz) = J(kz0) + i (kz - kz0) sigma_x
    key = (case, float(S), float(Pm), float(k), flow, float(V_), float(R), float(Lx), int(nx))
    kz = -float(c)*float(k)
    if key not in _BAND_MEMO:
        if len(_BAND_MEMO) > 2:
            _BAND_MEMO.clear()
        batch, idx_b = _batched(case, S, Pm, k, c, flow, V_, R, Lx, nx)
        assert np.array_equal(idx_b, idx)
        _BAND_MEMO[key] = (band_matrix_batched(batch, idx, nx, band_of(case, Lx, flow, R)), kz)
    A0, kz0 = _BAND_MEMO[key]
    return A0 if kz == kz0 else A0 + 1j*(kz - kz0)*sigma_x(len(idx))


def _parity(params, idx, v, case, k, c):
    # pair: psi at sheet 2's surface over psi at sheet 1's (+1 even, -1 odd about the midpoint);
    # sech2: nan
    if case != "pair":
        return complex(np.nan)
    nx, Lx = params.nx, params.Lx
    xs = x_res(c) if c else 0.0
    i1 = int(round((Lx/4 + xs)/(Lx/nx))) % nx
    i2 = int(round((3*Lx/4 - xs)/(Lx/nx))) % nx
    psi = T._eigfun(params, idx, v, i1)[1]
    return complex(psi[i2]/psi[i1])


def solve(case, S, Pm, k, c, flow, V_, R, Lx, nx, sigma, nev=2, rtol_mult=1.0):
    # one band-LU-preconditioned shift-invert of block (k, c): the nev eigenvalues nearest sigma,
    # residuals, parities, the current peak of the leading vector; RuntimeError on any refusal
    t0 = time.time()
    params = make_params(nx, Lx)
    op, idx = block_operator(case, S, Pm, k, c, flow, V_, R, Lx, nx)
    A = band_for(case, S, Pm, k, c, flow, V_, R, Lx, nx, op, idx)
    M, _ = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    tb = time.time() - t0
    grtol = float(np.clip(300*rtol_mult*np.finfo(float).eps*normA/(0.03*abs(sigma)), 1e-10,
                          1e-6*max(rtol_mult, 1.0)))
    big = rtol_mult != 1.0
    r = st.shift_invert(op, sigma, k=nev, tol=1e-12, M=M, gmres_rtol=grtol,
                        restart=60 if big else 30, maxiter=40 if big else 20)
    par = np.array([_parity(params, idx, r.right[:, j], case, k, c) for j in range(len(r.values))])
    j0 = int(np.argmax(np.asarray(r.values).real))
    x, psi, phi, d2 = T._eigfun(params, idx, r.right[:, j0], sheet_index(case, nx))
    ipk = int(np.argmax(np.abs(d2)))
    return dict(values=np.asarray(r.values), residuals=np.asarray(r.residuals), parity=par,
                sigma=complex(sigma), nx=nx, n=len(idx), gmres_rtol=grtol,
                gmres_iters=r.stats["gmres_iters"], scale=r.scale, t_build=tb,
                x_peak=float(x[ipk]), seconds=time.time() - t0)


# ============================================================ independent route: one sheet, finite differences

def fd_oblique(S, Pm, k, c, sigma, N, prof="tanh", flow="none", V_=0.0, R=1.0, L=20.0, nev=1,
               stretch=6.2):
    # tearing_ofman_run.fd_single_sheet with the oblique resonance: k F -> k F + kz = k (F - c) in
    # the coupling terms (F'' unchanged), the TRUE flow G in the advection (no frame shift). One
    # sheet on [-L, L], psi = W = Z = 0 at the walls, second-order FD on a stretched grid (refined
    # at x = 0; x_s = atanh(c) is within 0.7 for |c| <= 0.6), sparse LU shift-invert. prof "tanh"
    # (F = tanh x) or "sech2" (F = -2 tanh sech^2, the B_y of Psi0 = sech^2); flow "none" or
    # "tanh" (G = V tanh(R x); for "sech2" the aligned G = V F needs R = 1).
    # Returns the nev eigenvalues nearest sigma, by decreasing Re.
    xi_ = np.linspace(-1.0, 1.0, N + 1)
    x = L*np.sinh(stretch*xi_)/np.sinh(stretch)
    hm, hp = np.diff(x)[:-1], np.diff(x)[1:]
    xi = x[1:-1]
    n = len(xi)
    D2 = sps.diags([(2/(hm*(hm + hp)))[1:], -2/(hm*hp), (2/(hp*(hm + hp)))[:-1]], [-1, 0, 1],
                   format="csc")
    I = sps.identity(n, format="csc")
    Z0 = sps.csc_matrix((n, n))
    lap = D2 - k*k*I
    dg = lambda v: sps.diags(v, 0, format="csc")
    eta, nu = 1.0/S, Pm/S
    if prof == "tanh":
        F, Fpp = np.tanh(xi), -2*np.tanh(xi)*_sech2(xi)
    elif prof == "sech2":
        # f = -2 t s2 (t = tanh, s2 = sech^2): f' = s2 (4 t^2 - 2 s2), f'' = s2 (16 t s2 - 8 t^3)
        t, s2 = np.tanh(xi), _sech2(xi)
        F, Fpp = -2*t*s2, s2*(16*t*s2 - 8*t**3)
    else:
        raise ValueError(prof)
    if flow == "none":
        G, Gpp = 0*xi, 0*xi
    elif flow == "tanh":
        if prof == "sech2":
            if R != 1.0:
                raise ValueError("sech2: aligned flow only")
            G, Gpp = V_*F, V_*Fpp
        else:
            G, Gpp = V_*np.tanh(R*xi), -2*V_*R*R*np.tanh(R*xi)*_sech2(R*xi)
    else:
        raise ValueError(flow)
    ik = 1j*k
    Fc = F - c
    A = sps.bmat([[-ik*dg(G) + eta*lap, ik*dg(Fc), Z0],
                  [ik*(dg(Fc) @ lap) - ik*dg(Fpp), ik*dg(Gpp), -ik*dg(G) + nu*lap],
                  [Z0, lap, -I]], format="csc")
    M = sps.bmat([[I, Z0, Z0], [Z0, Z0, I], [Z0, Z0, Z0]], format="csc")
    perm = np.arange(3*n).reshape(3, n).T.reshape(-1)
    P = sps.csc_matrix((np.ones(3*n), (np.arange(3*n), perm)), shape=(3*n, 3*n))
    lu = spla.splu((P @ (A - sigma*M) @ P.T).tocsc(), permc_spec="NATURAL", diag_pivot_thresh=1.0)
    op = spla.LinearOperator(A.shape, matvec=lambda v: P.T @ lu.solve(P @ (M @ v)), dtype=complex)
    mu = spla.eigs(op, k=nev, which="LM", tol=1e-13, maxiter=3000, return_eigenvectors=False)
    lam = sigma + 1/mu
    return lam[np.argsort(-lam.real, kind="stable")]


# ============================================================ the ladder

NX_MAX = 32768
CONV_TOL = 1e-6
GROW_FLOOR = 1e-9
SEED_K, LADDER_K = 4, 2
ATTEMPTS = ((0.03, 1.0), (0.13, 1.0), (0.5, 1.0), (-0.03, 1.0), (0.03, 100.0), (0.13, 100.0),
            (0.03, 1e4))          # (relative shift offset, inner-rtol multiplier), as rung 2


def cfg_of(case="pair", S=1e4, Pm=0.0, k=0.5, c=0.0, flow="none", V_=0.0, R=1.0, Lx=None):
    # canonical configuration tuple (case, S, Pm, k, c, flow, V, R, Lx); no flow has V = 0, R = 1;
    # Lx defaults to box_of(case, k)
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    if flow not in FLOWS:
        raise ValueError(f"flow must be one of {FLOWS}, got {flow!r}")
    if flow == "none":
        V_, R = 0.0, 1.0
    Lx = box_of(case, k) if Lx is None else Lx
    return (case, float(S), float(Pm), float(np.round(k, 10)), float(np.round(c, 10)), flow,
            float(V_), float(R), float(Lx))


def _keys(cf):
    case, S, Pm, k, c, flow, V_, R, Lx = cf
    return dict(case=case, S=S, Pm=Pm, k=k, c=c, flow=flow, V=V_, R=R, Lx=Lx)


def _pow2_at_least(n):
    return int(2**np.ceil(np.log2(n - 1e-9)))


def box_of(case, k):
    # pair: Lx = max(40, 16/k) (sheet spacing >= 8/k: image coupling e^(-k Lx/2) <= 3e-4; the MEASURED
    # relative pair split is ~2.5x that, 6e-4 to 1.4e-3 on the kmax0/kmaxA scans, so the reported even
    # member sits ~half a split (3e-4 to 7e-4) above an isolated sheet -- review 2026-10-03, against an
    # independent Chebyshev one-sheet code; box-doubling checks in BOX);
    # sech2: rung 2's Lx = 20/k (one sheet)
    return float(np.round(max(40.0, 16.0/k) if case == "pair" else 20.0/k, 6))


def dx0_of(S, Pm=0.0, flow="none", k=0.5):
    # the first rung's grid spacing: ~ the resistive layer width (eta/k)^(1/3) over 5 (measured: at
    # S = 1e4, k = 0.5, Lx = 40 the rungs nx = 4096 and 8192 agree to 6e-12), halved for an inviscid
    # shear layer (narrower critical layer)
    dx = 0.08*(1e3/float(S))**(1.0/3.0)*(0.5/float(k))**(1.0/3.0)
    return dx/2 if (flow != "none" and Pm == 0.0) else dx


def nxs_for(cf, cap=None):
    case, S, Pm, k, c, flow, V_, R, Lx = cf
    n0 = max(1024, _pow2_at_least(Lx/dx0_of(S, Pm, flow, k)))
    out = []
    n = n0
    while n <= NX_MAX and (cap is None or n <= cap):
        out.append(n)
        n *= 2
    return tuple(out)


def nx_seed_for(cf):
    return min(2048, max(512, _pow2_at_least(1024*cf[8]/40.0)))


def pick(values, floor=GROW_FLOOR):
    vals = [complex(v) for v in np.atleast_1d(values) if complex(v).real > floor]
    return max(vals, key=lambda z: z.real) if vals else None


def seed(cf, root=None, compute=True):
    nx = nx_seed_for(cf)
    rec = U.cached("obl_seed", root or DATA_ROOT,
                   (lambda: dict(top=dense_top(*cf, nx))) if compute else None, nx=nx, **_keys(cf))
    return np.atleast_1d(rec["top"])


def rung(cf, nx, hint, nev, root=None, log=print, compute=True):
    last = None
    for attempt, (off, rm) in enumerate(ATTEMPTS):
        sig = complex(hint.real*(1.0 + off), hint.imag)
        try:
            rec = U.cached("obl", root or DATA_ROOT,
                           (lambda sig=sig, rm=rm: solve(*cf, nx, sig, nev, rm)) if compute else None,
                           attempt=attempt, nx=nx, nev=nev, **_keys(cf))
        except RuntimeError as e:
            last = str(e)
            log(f"    failed {cf} nx={nx} attempt {attempt}: {last[:140]}")
            continue
        lam = pick(rec["values"])
        if lam is not None:
            return rec, lam
        log(f"    no growing eigenvalue {cf} nx={nx} attempt {attempt}: {rec['values']}")
    raise RuntimeError(f"no growing eigenvalue at {cf}, nx={nx}: {last}")


def nev_for(cf, i):
    # eigenvalues per shift-invert on rung i: SEED_K then LADDER_K (the pair needs both members); ONE for the
    # inviscid single sheet (sech2, Pm = 0), whose undamped ~0 cluster makes ARPACK fail for k > 1 (measured at
    # alpha = 1: three 20-minute failures at k = 4, the k = 1 value converged every time)
    if cf[0] == "sech2" and cf[2] == 0.0:
        return 1
    return SEED_K if i == 0 else LADDER_K


def ladder(cf, root=None, log=print, compute=True, cap=None):
    # dense seed, then the nx ladder until CONV_TOL: a dict with every rung and the result
    # (lam, nx, conv, converged, split to the pair partner, parity, x_peak)
    nxs = nxs_for(cf, cap)
    top = seed(cf, root, compute)
    hint = pick(top)
    if hint is None:
        raise RuntimeError(f"no growing dense seed at {cf}: {top}")
    rows = []
    for i, nx in enumerate(nxs):
        try:
            rec, lam = rung(cf, nx, hint, nev_for(cf, i), root, log, compute)
        except RuntimeError as e:
            if not rows:
                raise
            log(f"  ladder {cf} stops at nx={nx}: {str(e)[:120]}")
            break
        vals = np.atleast_1d(rec["values"])
        j = int(np.argmin(np.abs(vals - lam)))
        others = [complex(v) for i2, v in enumerate(vals) if i2 != j]
        partner = min(others, key=lambda z: abs(z - lam)) if others else complex(np.nan)
        rows.append(dict(nx=nx, lam=lam, partner=partner, parity=complex(rec["parity"][j]),
                         residual=float(np.atleast_1d(rec["residuals"])[j]),
                         seconds=float(rec["seconds"]), x_peak=float(rec["x_peak"])))
        hint = lam
        if len(rows) >= 2 and abs(rows[-1]["lam"] - rows[-2]["lam"]) <= CONV_TOL*abs(lam):
            break
    conv = (abs(rows[-1]["lam"] - rows[-2]["lam"])/abs(rows[-1]["lam"])
            if len(rows) >= 2 else np.nan)
    return dict(cfg=cf, seed=hint if not rows else pick(top), rows=rows, lam=rows[-1]["lam"],
                nx=rows[-1]["nx"], conv=float(conv), converged=bool(conv <= CONV_TOL),
                split=float(abs(rows[-1]["partner"] - rows[-1]["lam"])/abs(rows[-1]["lam"])),
                parity=rows[-1]["parity"], x_peak=rows[-1]["x_peak"],
                seconds=float(sum(r["seconds"] for r in rows)))


def gamma(rec):
    return complex(rec["lam"]).real


# ============================================================ k scans (fastest mode at fixed c)

KFAC = (0.7, 0.85, 1.0, 1.18, 1.4)
KA_COARSE = tuple(np.round(np.geomspace(0.08, 1.2, 9), 5))


def coarse_scan(case, S, Pm, c, flow, V_, R, root=None, compute=True, kas=KA_COARSE):
    # {k: fastest dense eigenvalue} on the seed grids (cheap, coarse): only to centre the fine scan
    out = {}
    for k in kas:
        top = seed(cfg_of(case, S, Pm, k, c, flow, V_, R), root, compute)
        out[float(k)] = pick(top)
    return out


def coarse_kstar(scan):
    ks = [k for k, v in scan.items() if v is not None]
    if not ks:
        raise RuntimeError(f"nothing grows on the coarse scan: {scan}")
    return max(ks, key=lambda k: scan[k].real)


def parabola_max(ks, gs):
    # (k*, gamma*) of the parabola in (ln k, ln gamma) through the best point and its neighbours
    ks, gs = np.asarray(ks, float), np.asarray(gs, float)
    o = np.argsort(ks)
    ks, gs = ks[o], gs[o]
    i = int(np.argmax(gs))
    if i == 0 or i == len(ks) - 1:
        return float(ks[i]), float(gs[i]), False
    x, y = np.log(ks[i - 1:i + 2]), np.log(gs[i - 1:i + 2])
    a, b, cc = np.polyfit(x, y, 2)
    if a >= 0:
        return float(ks[i]), float(gs[i]), False
    xm = -b/(2*a)
    return float(np.exp(xm)), float(np.exp(a*xm*xm + b*xm + cc)), True


def kscan(case, S, Pm, c, flow, V_, R, k0, root=None, log=print, compute=True, facs=KFAC,
          max_ext=3):
    # the ladder at k0 * facs, extended outward (by the outer ratio) while the maximum sits on an
    # end; returns dict(rows={k: ladder record}, kstar, gstar, interior)
    facs = list(facs)
    rows = {}
    for _ in range(max_ext + 1):
        for f in facs:
            k = float(np.round(k0*f, 6))
            if k in rows:
                continue
            try:
                rows[k] = ladder(cfg_of(case, S, Pm, k, c, flow, V_, R), root, log, compute)
            except RuntimeError as e:
                log(f"  kscan point failed k={k}: {str(e)[:120]}")
        ks = sorted(rows)
        gs = [gamma(rows[k]) for k in ks]
        if not ks:
            break
        i = int(np.argmax(gs))
        if 0 < i < len(ks) - 1:
            break
        ratio = facs[-1]/facs[-2]
        facs = facs + ([facs[-1]*ratio] if i == len(ks) - 1 else [facs[0]/ratio])
        facs.sort()
    ks = sorted(rows)
    if not ks:
        raise RuntimeError(f"kscan: no point solved ({case}, S={S}, Pm={Pm}, c={c}, {flow} {V_} {R})")
    kstar, gstar, interior = parabola_max(ks, [gamma(rows[k]) for k in ks])
    return dict(rows=rows, kstar=kstar, gstar=gstar, interior=interior)


# ============================================================ the campaign

# A. alpha = 0, the pair: the layer relation at the shifted surface, and the fastest mode vs c
LAYER_K, LAYER_LX = 0.5, 40.0
LAYER_C = (0.0, 0.3, 0.6)
LAYER_S = (1e3, 1e4, 1e5)
KMAX0_S = (1e3, 1e4)
SCAN_C = (0.0, 0.2, 0.4, 0.6)
# B. field-aligned shear u = alpha b0 (pair), S = 1e4
SHEAR_S = 1e4
SHEAR_AC = tuple((0.6, c) for c in SCAN_C) + ((0.9, 0.0), (0.9, 0.4))   # (alpha, c) k scans
SYM = ((0.0, 0.4), (0.6, 0.4))          # (alpha, c): the +-c blocks at LAYER_K, SHEAR_S
# C. non-aligned flows u = V tanh(R x): alpha_loc(x_s) = V R sech^2(R x_s)/sech^2(x_s) varies with c
LOCAL = ((0.6, 0.5), (0.3, 2.0))        # (V, R) at LAYER_K, SHEAR_S
LOCAL_C = (0.0, 0.3, 0.6)
# D. the Pm >> 1 destabilisation (sech2, as tearing-viscosity.ipynb): S = 1e4, Pm = 100
VISC_S, VISC_PM = 1e4, 100.0
VISC_ALPHA = (0.0, 0.6)
VISC_C = (0.0, 0.4)
# E. the alpha -> 1 low-k mode (sech2, tearing-lowk-branch.ipynb): S = 1e4, nu = 0, alpha = 1
LOWK_S, LOWK_ALPHA = 1e4, 1.0
LOWK_C = (0.0,)          # the k scan at c = 0 only: obliquely the branch does not survive as a k-scan target
# continuation in c at the c = 0 k* (shift-invert, one eigenvalue, seeded by the previous c): where does the mode go?
LOWK_K = 0.265
LOWK_TRACK_C = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)
LOWK_TRACK_NX = (2048, 4096)
LOWK_CHECK = ((0.05, 0.265, 4096), (0.2, 0.265, 4096), (0.4, 0.265, 4096), (0.05, 0.60976, 2048), (0.05, 0.60976, 4096),
              (0.4, 0.8554, 2048), (0.4, 0.8554, 4096))   # (c, k, nx): dense spectra, finer than the seeds
# the review's continuations (lowk_paths): the c = 0.05, k ~ 0.61 mode followed in c to 0 and in k to LOWK_K
LOWK_PATH_ANCHOR = (0.05, 0.60976, 4096)
LOWK_PATH_C = (0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.0)
LOWK_PATH_KS = (0.60976, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.265)
LOWK_PATH_CONV = ((8192, 1.0), (16384, 1.0), (8192, 2.0))   # (nx, box factor) at the anchor
# box checks: (case, S, Pm, k, c, flow, V, R) at box_of and at twice it
BOX = (("pair", 1e4, 0.0, 0.5, 0.6, "none", 0.0, 1.0), ("pair", 1e4, 0.0, 0.5, 0.4, "tanh", 0.6, 1.0))
# the independent FD route: (name, case/prof, S, Pm, k, c, flow, V, R)
FD_NS = (8000, 16000)
FD = (("a0c6", "tanh", 1e4, 0.0, 0.5, 0.6, "none", 0.0, 1.0),
      ("a6c4", "tanh", 1e4, 0.0, 0.5, 0.4, "tanh", 0.6, 1.0),
      ("loc2", "tanh", 1e4, 0.0, 0.5, 0.3, "tanh", 0.3, 2.0))
if PROFILE == "smoke":
    NX_MAX = 4096
    LAYER_S, KMAX0_S, SCAN_C = (1e3,), (1e3,), (0.0, 0.4)
    SHEAR_S, SHEAR_AC = 1e3, ((0.6, 0.0), (0.6, 0.4))
    SYM = ((0.6, 0.4),)
    LOCAL, LOCAL_C = ((0.3, 2.0),), (0.0, 0.3)
    VISC_S, VISC_PM, VISC_ALPHA, VISC_C = 1e3, 10.0, (0.0, 0.6), (0.0, 0.4)
    LOWK_S, LOWK_C, LOWK_CHECK = 1e3, (0.0,), ((0.4, 0.4, 1024),)
    LOWK_K, LOWK_TRACK_C, LOWK_TRACK_NX = 0.4, (0.0, 0.01), (1024,)
    LOWK_PATH_ANCHOR, LOWK_PATH_C, LOWK_PATH_KS, LOWK_PATH_CONV = (0.4, 0.4, 1024), (0.4, 0.3), (0.4, 0.35), ((2048, 1.0),)
    BOX = (("pair", 1e3, 0.0, 0.5, 0.6, "none", 0.0, 1.0),)
    FD_NS = (4000, 8000)
    FD = (("a6c4", "tanh", 1e3, 0.0, 0.5, 0.4, "tanh", 0.6, 1.0),)


def alpha_loc(c, V_, R):
    # u'(x_s)/f'(x_s) of u = V tanh(R x) on f = tanh x
    xs = x_res(c) if c else 0.0
    return float(V_*R*_sech2(R*xs)/_sech2(xs))


def doppler_of(c, k, flow, V_, R):
    # -k u(x_s): the Doppler frequency of the surface (tanh flows on the pair's sheet 1)
    if flow == "none" or c == 0:
        return 0.0
    return float(-k*V_*np.tanh(R*x_res(c)))


def layer_points():
    return [cfg_of("pair", S, 0.0, LAYER_K, c, Lx=LAYER_LX) for S in LAYER_S for c in LAYER_C]


def sym_points():
    out = []
    for al, c in SYM:
        fl = "tanh" if al else "none"
        out += [cfg_of("pair", SHEAR_S, 0.0, LAYER_K, s*c, fl, al, 1.0, Lx=LAYER_LX) for s in (1, -1)]
    return out


def local_points():
    out = []
    for V_, R in LOCAL:
        for c in LOCAL_C:
            out.append(cfg_of("pair", SHEAR_S, 0.0, LAYER_K, c, "tanh", V_, R, Lx=LAYER_LX))
            out.append(cfg_of("pair", SHEAR_S, 0.0, LAYER_K, c, "tanh",
                              float(np.round(alpha_loc(c, V_, R), 8)), 1.0, Lx=LAYER_LX))
    return out


def box_points():
    out = []
    for case, S, Pm, k, c, fl, V_, R in BOX:
        L0 = box_of(case, k)
        out += [cfg_of(case, S, Pm, k, c, fl, V_, R, Lx=L) for L in (L0, 2*L0)]
    return out


def scans():
    # every k scan of the campaign: (tag, case, S, Pm, c, flow, V, R, k0 or None = coarse-centred)
    out = []
    for S in KMAX0_S:
        for c in SCAN_C:
            out.append(("kmax0", "pair", S, 0.0, c, "none", 0.0, 1.0,
                        float(np.round(theory_kmax(c, S)[0], 4))))
    for al, c in SHEAR_AC:
        out.append(("kmaxA", "pair", SHEAR_S, 0.0, c, "tanh", al, 1.0, None))
    for al in VISC_ALPHA:
        for c in VISC_C:
            out.append(("visc", "sech2", VISC_S, VISC_PM, c, "tanh" if al else "none", al, 1.0, None))
    for c in LOWK_C:
        out.append(("lowk", "sech2", LOWK_S, 0.0, c, "tanh", LOWK_ALPHA, 1.0, None))
    return out


def run_scan(sc, root=None, log=print, compute=True):
    tag, case, S, Pm, c, flow, V_, R, k0 = sc
    if k0 is None:
        k0 = coarse_kstar(coarse_scan(case, S, Pm, c, flow, V_, R, root, compute))
    return kscan(case, S, Pm, c, flow, V_, R, k0, root, log, compute)


def lowk_check(root=None, compute=True):
    # {(c, k, nx): the 8 fastest dense eigenvalues} of the alpha = 1 sech2 block at LOWK_S on a finer grid than the
    # seeds (whether ANY growing mode is left where the low-k branch was)
    out = {}
    for c, k, nx in LOWK_CHECK:
        cf = cfg_of("sech2", LOWK_S, 0.0, k, c, "tanh", LOWK_ALPHA, 1.0)
        rec = U.cached("obl_lowk_check", root or DATA_ROOT,
                       (lambda cf=cf, nx=nx: dict(top=dense_top(*cf, nx))) if compute else None, nx=nx, **_keys(cf))
        out[(c, k, nx)] = np.atleast_1d(rec["top"])
    return out


def lowk_track(root=None, compute=True, log=print):
    # {nx: [(c, lambda, residual, x_peak)]}: the alpha = 1 low-k mode at k = LOWK_K followed in c by shift-invert
    # continuation (sigma 3% above the previous eigenvalue in Re, same Im shifted by the Doppler change -k dc),
    # one eigenvalue per solve; stops at the first c where no growing eigenvalue is found
    out = {}
    for nx in LOWK_TRACK_NX:
        rows = []
        lam = None
        for c in LOWK_TRACK_C:
            cf = cfg_of("sech2", LOWK_S, 0.0, LOWK_K, c, "tanh", LOWK_ALPHA, 1.0)
            if lam is None:
                lam = pick(seed(cf, root, compute))
                if lam is None:
                    break
            else:
                lam = complex(lam.real, lam.imag - LOWK_K*(c - rows[-1][0]))
            try:
                rec = U.cached("obl_lowk_track", root or DATA_ROOT,
                               (lambda cf=cf, sig=complex(1.03*lam.real, lam.imag): solve(*cf, nx, sig, 1, 100.0))
                               if compute else None, nx=nx, **_keys(cf))
            except RuntimeError as e:
                log(f"  lowk_track nx={nx} c={c}: {str(e)[:120]}")
                break
            v = complex(np.atleast_1d(rec["values"])[0])
            rows.append((c, v, float(np.atleast_1d(rec["residuals"])[0]), float(rec["x_peak"] - cf[8]/2)))
            if v.real <= GROW_FLOOR:
                break
            lam = v
        out[nx] = rows
    return out


def lowk_paths(root=None, compute=True, log=print):
    # (added by the review, 2026-10-03) the identity of the oblique alpha = 1 mode at LOWK_PATH_ANCHOR = (c, k, nx), a
    # LOWK_CHECK entry whose fastest dense eigenvalue seeds three continuations, one eigenvalue per shift-invert, each
    # seeded by the previous value (Im scaled by the change in c or k: the Doppler shift is ~ -k c):
    #   "c":    c from the anchor's c down to 0 at the anchor's k (LOWK_PATH_C)
    #   "k":    k from the anchor's k through LOWK_PATH_KS at the anchor's c
    #   "conv": the anchor point at (nx, box factor) in LOWK_PATH_CONV (resolution and box)
    # {"c": [(c, lam, res, x_peak - Lx/2)], "k": [(k, ...)], "conv": [(nx, Lx, lam, res)]}; a path stops at its first
    # refusal or non-growing value
    c0, k0, nx0 = LOWK_PATH_ANCHOR
    seed_ = pick(lowk_check(root, compute)[LOWK_PATH_ANCHOR])
    out = {"c": [], "k": [], "conv": []}
    if seed_ is None:
        return out

    def one(k, c, nx, Lx, sig):
        cf = cfg_of("sech2", LOWK_S, 0.0, k, c, "tanh", LOWK_ALPHA, 1.0, Lx=Lx)
        rec = U.cached("obl_lowk_path", root or DATA_ROOT,
                       (lambda: solve(*cf, nx, complex(1.03*sig.real, sig.imag), 1, 100.0)) if compute else None,
                       nx=nx, **_keys(cf))
        return (complex(np.atleast_1d(rec["values"])[0]), float(np.atleast_1d(rec["residuals"])[0]),
                float(rec["x_peak"] - cf[8]/2))

    for name, path, par0 in (("c", LOWK_PATH_C, c0), ("k", LOWK_PATH_KS, k0)):
        lam, prev = seed_, par0
        for p in path:
            k, c = (k0, p) if name == "c" else (p, c0)
            sig = complex(lam.real, lam.imag*(p/prev if prev else 1.0))
            try:
                v, res, xp = one(k, c, nx0, box_of("sech2", k), sig)
            except RuntimeError as e:
                log(f"  lowk_paths {name}={p}: {str(e)[:120]}")
                break
            out[name].append((p, v, res, xp))
            if v.real <= GROW_FLOOR:
                break
            lam, prev = v, p
    for nx, f in LOWK_PATH_CONV:
        Lx = f*box_of("sech2", k0)
        try:
            v, res, _ = one(k0, c0, nx, Lx, seed_)
        except RuntimeError as e:
            log(f"  lowk_paths conv nx={nx} Lx={Lx}: {str(e)[:120]}")
            continue
        out["conv"].append((nx, Lx, v, res))
    return out


def fd_check(root=None, compute=True):
    # {name: {N: eigenvalues}} of the FD route, each shifted 3% above its ladder eigenvalue
    out = {}
    for name, prof, S, Pm, k, c, flow, V_, R in FD:
        lad = ladder(cfg_of("pair" if prof == "tanh" else "sech2", S, Pm, k, c, flow, V_, R,
                            Lx=LAYER_LX if prof == "tanh" else None), root, _quiet, compute=False)
        hint = complex(lad["lam"])
        sig = complex(1.03*hint.real, hint.imag)
        out[name] = {}
        for N in FD_NS:
            rec = U.cached("obl_fd", root or DATA_ROOT,
                           (lambda: dict(vals=fd_oblique(S, Pm, k, c, sig, N, prof, flow, V_, R)))
                           if compute else None, name=name, prof=prof, S=S, Pm=Pm, k=k, c=c,
                           flow=flow, V=V_, R=R, N=N)
            out[name][N] = np.atleast_1d(rec["vals"])
    return out


def fd_results(root=None):
    # {name: (FD eigenvalue nearest the ladder's at the finest N, its change from the previous N,
    #         the ladder's eigenvalue)}
    try:
        d = fd_check(root, compute=False)
    except (KeyError, RuntimeError):
        return {}
    out = {}
    for name, prof, S, Pm, k, c, flow, V_, R in FD:
        lad = complex(ladder(cfg_of("pair" if prof == "tanh" else "sech2", S, Pm, k, c, flow, V_,
                                    R, Lx=LAYER_LX if prof == "tanh" else None),
                             root, _quiet, compute=False)["lam"])
        near = [complex(v[np.argmin(np.abs(v - lad))]) for v in (d[name][N] for N in FD_NS)]
        out[name] = (near[-1], abs(near[-1] - near[-2])/abs(near[-1]), lad)
    return out


PHASES = ("layer", "sym", "local", "box", "fd", "kmax0", "kmaxA", "visc", "lowk", "lowk_track", "lowk_check",
          "lowk_paths")
POINTS = dict(layer=layer_points, sym=sym_points, local=local_points, box=box_points)


def make_data(root=None, wall_budget=3*3600.0, phases=None, log=print):
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    done = True
    for name in phases or PHASES:
        t0 = time.time()
        ok = True
        if name in POINTS:
            for cf in POINTS[name]():
                if time.time() > deadline:
                    ok = False
                    break
                try:
                    r = ladder(cf, root, log)
                except RuntimeError as e:
                    ok = False
                    log(f"  FAILED {cf}: {str(e)[:160]}")
                    continue
                log(f"  {name} {cf}: lambda = {r['lam']:.8e} (nx {r['nx']}, conv {r['conv']:.1e}, "
                    f"parity {r['parity'].real:+.3f}, split {r['split']:.1e}, x_peak {r['x_peak']:.3f}, "
                    f"{r['seconds']:.0f} s)")
        elif name == "lowk_track":
            for nx, rows in lowk_track(root, log=log).items():
                for c, v, res, xp in rows:
                    log(f"  lowk_track nx={nx} c={c}: {v:.8e} (residual {res:.1e}, x_peak {xp:+.3f})")
        elif name == "lowk_check":
            for key, top in lowk_check(root).items():
                log(f"  lowk_check {key}: {np.round(top[:4], 6)}")
        elif name == "lowk_paths":
            for key, rows in lowk_paths(root, log=log).items():
                for row in rows:
                    log(f"  lowk_paths {key}: {row}")
        elif name == "fd":
            fd_check(root)
            for nm, (lam, ch, lad) in fd_results(root).items():
                log(f"  fd {nm}: {lam:.8e} (change {ch:.1e}) vs ladder {lad:.8e}")
        elif name in ("kmax0", "kmaxA", "visc", "lowk"):
            for sc in scans():
                if sc[0] != name:
                    continue
                if time.time() > deadline:
                    ok = False
                    break
                try:
                    r = run_scan(sc, root, log)
                except RuntimeError as e:
                    ok = False
                    log(f"  FAILED scan {sc}: {str(e)[:160]}")
                    continue
                log(f"  {name} {sc[1:8]}: k* = {r['kstar']:.5f}, gamma* = {r['gstar']:.6e} "
                    f"(interior {r['interior']}, {len(r['rows'])} points)")
        else:
            raise ValueError(name)
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")
        done &= ok
    return bool(done)


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def results(points, root=None):
    out = {}
    for cf in points:
        try:
            out[cf] = ladder(cf, root or DATA_ROOT, _quiet, compute=False)
        except (KeyError, RuntimeError):
            pass
    return out


def scan_results(tag, root=None):
    # {(S, Pm, c, flow, V, R): kscan record} for every cached scan of this tag
    out = {}
    for sc in scans():
        if sc[0] != tag:
            continue
        try:
            out[tuple(sc[2:8])] = run_scan(sc, root or DATA_ROOT, _quiet, compute=False)
        except (KeyError, RuntimeError):
            pass
    return out


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3*3600.0

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    log(f"profile {PROFILE}, root {DATA_ROOT}")
    make_data(wall_budget=budget, phases=tuple(sys.argv[2:]) or None, log=log)
