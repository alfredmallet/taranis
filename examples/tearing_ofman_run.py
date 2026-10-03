# Comparison with Ofman, Chen, Morrison & Steinolfson 1991, "Resistive tearing mode instability
# with shear flow and viscosity", Phys. Fluids B 3, 1364 -- data generation for the last section of
# examples/tearing-viscosity.ipynb (plans/AUTODIFF_PLAN.md rung 2b item 1). Promoted from the
# 2026-10-02 scratch study (examples/data/tearing-ofman/scratch-2026-10-02/, untracked).
#
# THEIR SETUP, OURS: a Harris sheet F = tanh(x) (a = v_A = 1, |F'(0)| = 1), S = 1e6 = 1/eta,
# ka = 0.5, S_v = S/Pm (nu = Pm/S), flows (their eq. 7) G = V tanh(R x) ("tanh") and
# G = V[sech(R x) - 1] ("sech" jet, zero AT the sheet). dims = 2, hyper = 1, ny = 8, the ky block
# iky = 1, fp64. Periodic box: the EXACT tanh pair of rung 1b-ref -- B_y = tanh(x - Lx/4) near the
# first sheet, -tanh(x - 3Lx/4) near the second (by_pair), Lx = 40, so the sheets are 20 apart
# and couple through the outer region at ~e^(-k Lx/2) = 4.5e-5.
#   * tanh flow: u_y = V by_pair(x; R), the SAME pair construction with tanh(R u): it follows the
#     sign of B at each sheet, so at R = 1 it is u_0 = V b_0 EXACTLY (phi = V Psi0 bitwise, gated)
#     -- the field-aligned shear of our JPP paper and of tearing-viscosity.ipynb, alpha = V.
#   * sech jet: the same-sign jet V sech(R(x - x_s)) at both sheets (F -> -F at fixed G is a
#     symmetry of the linear problem, so the two sheets are equivalent), minus its value at the
#     sheet, so G(x_s) = 0 as in their eq. 7b. That profile has a NONZERO mean, which a periodic
#     stream function cannot carry: phi is built from u - <u>, and the uniform <u> is restored
#     EXACTLY as a Doppler shift, J -> J - i k <u> I (doppler()). The scratch removed the mean
#     and left the shift in, Im lambda ~ -0.32 at R = 0.44, which also wrecked its GMRES
#     tolerance (set from |sigma|) -- the inviscid R = 0.44 jet never solved. The sign of the
#     shift is gated by box independence (Lx changes <u>, not the physics).
#   * which pair member: shift_invert returns k = 2 eigenvalues, the two near-degenerate pair
#     modes (even/odd about the midpoint between the sheets, split ~5e-4 relative). We report
#     the one of LARGER Re lambda; its parity (psi(second sheet)/psi(first sheet) = +-1) and the
#     split are stored with every rung.
#
# THE OVERRIDES SEAM: one Parameters per (nx, Lx) (make_params: placeholder Ly = 2 pi and
# diss = (0, 1e-6)) serves every S, Pm and ka: every operator is built at
# overrides = {"Ly": 2 pi/ka, "diss": (Pm/S, 1/S)} (V.overrides). J there == J of a Parameters
# built with that Ly and diss: the independent-route gate in tests/test_tearing_ofman.py.
#
# SOLVER: rung 1a's band-LU-preconditioned shift-invert on the matrix-free block (U.band_matrix,
# U._lu_ops), the band widened by max(1, R) for the narrow profiles (the equilibrium spectrum of
# tanh(R x)/sech(R x) decays like e^(-pi q/(2R))). A LADDER per point: a dense seed at NX_SEED
# (np.linalg.eigvals of ky_block_matrix), then shift-invert at NXS (k = SEED_K on the first rung,
# LADDER_K after), each rung seeded 3% above the previous eigenvalue (V.ATTEMPTS's shift / inner
# tolerance retries), stopping when two successive rungs agree to CONV_TOL or at the cap. The
# reported convergence is the relative change between the last two rungs -- per point, honest:
# the inviscid shear points need nx ~ 1e5 (their critical layer is ~ gamma/(k G') wide).
#
# THEORY (no flow; the exact visco-resistive layer relation of tearing_viscosity_run, which is
# profile-independent once written with Delta' and f'): Harris Delta' a = 2(1/ka - ka) = 3 at
# ka = 0.5, f' = 1. Porcelli's viscous constant-psi limit gamma = Delta' eta^(5/6) nu^(-1/6)
# (k f')^(1/3)/I_V. With flow: their eq. 13c (from Chen & Morrison 1990b),
#     gamma(g)/gamma(0) = (1 - g^2)^(1/3) [1 + g^2 (1 - Pm)^2/(4 Pm)]^(-1/6),   g = G'/F',
# and, at Pm = 0, Chen & Morrison 1989 as printed in our JPP paper (eq. 1.6),
#     gamma = (2 pi)^(-1/2) Delta'^(1/2) [g (1 - g^2)]^(1/2) S^(-1/2).
#
# `python examples/tearing_ofman_run.py [wall_s] [cap=NX] [phase ...]` runs make_data (resumable;
# every solve is one cache file under DATA_ROOT/solves; cap=NX stops every ladder at NX so a
# second process can take the cheap points without a second nx >= 65536 solve).
# TARANIS_OFMAN_PROFILE=smoke|full (default full); smoke writes to DATA_ROOT_SMOKE.
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import functools
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

import taranis as jr
from taranis import stability as st

import tearing_eigen_run as T
import tearing_sensitivity_run as U
import tearing_viscosity_run as V

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FULL = os.path.join(HERE, "data", "tearing-ofman")
DATA_ROOT_SMOKE = os.path.join(HERE, "data", "tearing-ofman-smoke")
PROFILE = os.environ.get("TARANIS_OFMAN_PROFILE", "full")
if PROFILE not in ("full", "smoke"):
    raise ValueError(f"TARANIS_OFMAN_PROFILE must be 'full' or 'smoke', got {PROFILE!r}")
DATA_ROOT = DATA_ROOT_SMOKE if PROFILE == "smoke" else DATA_ROOT_FULL

TWO_PI = 2*np.pi
FLOWS = ("none", "tanh", "sech")
FPRIME = 1.0                 # |F'(0)| of F = tanh x
KA = 0.5
LX = 40.0
DIFF_PLACEHOLDER = (0.0, 1e-6)    # make_params' own diss: never used (always overridden)

# ---- configuration (full profile; smoke below)
S0 = 1e6
NXS = (8192, 16384, 32768, 65536, 131072)     # the ladder at Lx = 40 (scaled with Lx)
NX_MAX = 131072              # no rung above this (memory: one 131072 solve is ~2.4 GB RSS) ...
NX_HI = 262144               # ... except one more rung for the HI points (inviscid shear, slowest to converge)
HI = ((0.0, "tanh", 0.6, 1.0), (0.0, "tanh", 0.8, 1.0), (0.0, "tanh", 1.0, 0.9))   # (Pm, flow, V, R) at S0, KA, LX
NX_SEED = 2048
CONV_TOL = 1e-6              # stop when successive rungs agree to this (relative)
SEED_K, LADDER_K = 4, 2      # eigenvalues per shift-invert: first rung, later rungs
GROW_FLOOR = 1e-9            # a growing eigenvalue: Re lambda above this (smallest here ~2e-5)
PM_NF = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)    # fig 7 (no flow), S_v = 1e8 .. 1e2
PM_FA = (0.0, 1.0, 100.0)                               # field-aligned shear (R = 1, V = alpha)
ALPHA_FA = (0.2, 0.4, 0.6, 0.8)
R_INV = (0.1, 0.3, 0.5, 0.73, 0.9)          # fig 6, tanh, V = 1, Pm = 0
R_VISC = (0.6, 0.73, 0.85, 0.9, 0.95, 0.98)  # fig 8, tanh, V = 1, Pm = 1 (S = S_v)
S_SLOPE = (1e4, 1e5)                        # fig 5a with S0: tanh V = 1, R = 0.73, Pm = 0
R_SLOPE = 0.73
JET = ((0.0, 0.44), (0.0, 2.5), (0.0, 6.0), (1.0, 0.44), (10.0, 0.44), (100.0, 0.44),
       (100.0, 6.0), (1e3, 6.0), (1e4, 6.0))   # (Pm, R) sech jet, V = 1: figs 6 and 9
BOX_LX = 60.0                               # a second box for these (Pm, flow, V, R):
BOX = ((100.0, "none", 0.0, 1.0), (100.0, "tanh", 0.6, 1.0), (1.0, "sech", 1.0, 0.44))
CMK = ((0.25, 80.0, 0.2),)                  # (ka, Lx, alpha): C&M eq. 1.6's k-dependence, Pm = 0
OSC_PM = (0.0, 100.0, 1e3, 3e3, 1e4, 3e4)   # sech R = 6: the oscillating branch vs nu, dense at OSC_NX
OSC_NX, OSC_IM = 1024, 0.3                  # ... |Im lambda| > OSC_IM separates it from the KH branch
DENSE = ((100.0, 0.6),)                     # (Pm, alpha): independent dense route at NX_INDEP
NX_INDEP = 4096
if PROFILE == "smoke":
    S0, NXS, NX_SEED, NX_MAX, NX_HI = 1e4, (1024, 2048, 4096), 512, 8192, 8192
    PM_NF, PM_FA, ALPHA_FA = (0.0, 1.0, 100.0), (1.0, 100.0), (0.6,)
    R_INV, R_VISC, S_SLOPE = (0.73,), (0.73,), (1e3,)
    JET = ((1.0, 0.44),)
    BOX = ((100.0, "tanh", 0.6, 1.0),)
    CMK = ()
    DENSE, NX_INDEP = ((100.0, 0.6),), 1024
    OSC_PM, OSC_NX = (0.0, 1e4), 512


# ============================================================ theory

def dprime_a(ka):
    # Harris: an isolated tanh sheet, Delta' a = 2(1/ka - ka) (T.dprime_harris_local)
    return float(T.dprime_harris_local(ka))


def theory_root(Pm):
    # the viscosity notebook's tables where it has them (Pm in V.PM_CHAIN), ours otherwise
    if os.path.exists(V._table_path(float(Pm), True, V.DATA_ROOT_FULL)):
        return V.DATA_ROOT_FULL
    return DATA_ROOT_FULL


def gamma_layer(S, Pm, ka=KA, fprime=FPRIME):
    # the exact visco-resistive layer relation Delta' delta_eta = F(Lam, Pm) (Pm = 0: B&L) for
    # the Harris Delta', no flow: gamma = Lam eta^(1/3) (k f')^(2/3)
    eta, kb = 1.0/S, ka*fprime
    dd = dprime_a(ka)*(eta/kb)**(1.0/3.0)
    lam = T.lam_of(dd) if Pm == 0 else V.lam_of_visc(dd, float(Pm), theory_root(Pm))
    return float(lam*eta**(1.0/3.0)*kb**(2.0/3.0))


def gamma_porcelli(S, Pm, ka=KA, fprime=FPRIME):
    # viscous constant-psi (Porcelli 1987): Delta' eta^(5/6) nu^(-1/6) (k f')^(1/3)/I_V
    eta, nu = 1.0/S, Pm/S
    return float(dprime_a(ka)*eta**(5/6)*nu**(-1/6)*(ka*fprime)**(1/3)/V.I_V)


def ratio_13c(g, Pm):
    # their eq. 13c over its g = 0 value: (1 - g^2)^(1/3) [1 + g^2 (1 - Pm)^2/(4 Pm)]^(-1/6)
    # (gamma ~ S^(-2/3) Q^(1/3) [4 S/S_v + g^2 (1 - S/S_v)^2]^(-1/6), Q = ka F'(1 - g^2))
    g, Pm = float(g), float(Pm)
    return (1.0 - g*g)**(1/3)*(1.0 + g*g*(1.0 - Pm)**2/(4.0*Pm))**(-1/6)


def gamma_cm89(g, S, ka=KA):
    # Chen & Morrison 1989 (constant psi, Pm = 0) as printed in Mallet et al. JPP 2025 eq. (1.6):
    # gamma a/v_Ay = (2 pi)^(-1/2) (Delta' a)^(1/2) [g (1 - g^2)]^(1/2) S^(-1/2) -- no explicit k
    return float((2*np.pi)**-0.5*dprime_a(ka)**0.5*(g*(1.0 - g*g))**0.5*S**-0.5)


def q_of(V_, R, ka=KA, fprime=FPRIME):
    # their Q = ka F'[1 - (G'/F')^2], G'(0) = V R for the tanh flow
    return float(ka*fprime*(1.0 - (V_*R/fprime)**2))


def local_slopes(x, y):
    # d ln y/d ln x between neighbours, at the geometric midpoints
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.sqrt(x[1:]*x[:-1]), np.diff(np.log(y))/np.diff(np.log(x))


def lsq_slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])


# ============================================================ equilibrium

def sheet_sum(x, Lx, f, sign=-1.0):
    # sum_m [f(x - Lx/4 - m Lx) + sign f(x - 3 Lx/4 - m Lx)], m = -2..2: sign -1 follows B's
    # reversal at the second sheet, +1 repeats the profile there
    return sum(f(x - Lx/4 - m*Lx) + sign*f(x - 3*Lx/4 - m*Lx) for m in range(-2, 3))


def by_pair(x, Lx, R=1.0):
    # -1 + the sign-following pair of tanh(R u): B_y at R = 1 (== U.by_tanhpair_a(x, Lx, 1)),
    # the tanh flow's shape at any R
    return -1.0 + sheet_sum(x, Lx, lambda u: np.tanh(R*u))


def _sech(u):
    # cosh overflows to inf for |u| > 710, where 1/inf = 0 is the right value
    with np.errstate(over="ignore"):
        return 1.0/np.cosh(u)


def flow_profile(x, Lx, flow, V_, R):
    # u_y(x) in the sheets' frame: tanh -> V by_pair(x; R); sech -> V [jet(x) - jet(x_s)],
    # jet = the same-sign pair of sech(R u), so u(x_s) = 0 at both sheets
    x = np.asarray(x, float)
    if flow == "none":
        return np.zeros_like(x)
    if flow == "tanh":
        return V_*by_pair(x, Lx, R)
    if flow == "sech":
        jet = lambda z: sheet_sum(z, Lx, lambda u: _sech(R*u), sign=1.0)
        return V_*(jet(x) - jet(np.float64(Lx/4)))
    raise ValueError(f"flow must be one of {FLOWS}, got {flow!r}")


def equilibrium_fields(nx, Lx, flow, V_, R):
    # (phi, psi, u_mean) on the grid: psi = int B_y, phi = int (u - u_mean) (FFT, spectrally
    # exact), u_mean the uniform part (nonzero only for the jet), restored by doppler(). The tanh
    # flow is integrated as V * int(by_pair(R)) so that R = 1 gives phi == V psi bitwise
    x = np.arange(nx)*Lx/nx
    psi = U._fft_integral(by_pair(x, Lx), Lx)
    if flow == "none":
        return 0.0*psi, psi, 0.0
    if flow == "tanh":
        return V_*U._fft_integral(by_pair(x, Lx, R), Lx), psi, 0.0
    u = flow_profile(x, Lx, flow, V_, R)
    um = float(np.mean(u))
    return U._fft_integral(u - um, Lx), psi, um


def band(Lx, flow, R):
    # the preconditioner's kx band: rung 1a's tanh-pair band, widened for the narrow flows
    b = T.band_width("tanhpair", Lx)
    return b if flow == "none" or R <= 1.0 else int(np.ceil(R*(b - 2))) + 2


def doppler(op, ka, u_mean):
    # J of the true profile u = (u - u_mean) + u_mean from J of the periodic part: a uniform
    # u_y = U advects e^(iky) as d/dt = -i k U, so J_true = J - i k U I
    if u_mean == 0.0:
        return op
    c = -1j*float(ka)*float(u_mean)
    return spla.LinearOperator(op.shape, matvec=lambda v: op.matvec(v) + c*np.asarray(v).reshape(-1),
                               dtype=np.complex128)


# ============================================================ operators

@functools.lru_cache(maxsize=16)
def make_params(nx, Lx):
    # ONE Parameters per (nx, Lx): Ly and diss are placeholders, every operator is built at
    # V.overrides(ka, S, Pm) = {"Ly": 2 pi/ka, "diss": (Pm/S, 1/S)}
    return jr.Parameters(nx=int(nx), ny=8, Lx=float(Lx), Ly=TWO_PI, dims=2, cfl_safety=0.5,
                         forcing=False, eqpars={"diss": DIFF_PLACEHOLDER, "hyper": 1})


@functools.lru_cache(maxsize=16)
def _kgrid(nx, Lx):
    return jr.setup_kgrids(make_params(nx, Lx))


def _x0(params, flow, V_, R):
    phi, psi, um = equilibrium_fields(params.nx, params.Lx, flow, V_, R)
    return U._fields(params, phi, psi), um


def block_matrix(S, Pm, ka, flow, V_, R, Lx, nx):
    # the dense block (Doppler included) and its idx -- the independent dense route
    params = make_params(nx, Lx)
    x0, um = _x0(params, flow, V_, R)
    B, idx = st.ky_block_matrix(x0, _kgrid(nx, Lx), params, 1, overrides=V.overrides(ka, S, Pm))
    if um != 0.0:
        B = B - 1j*ka*um*np.eye(B.shape[0])
    return B, idx


def dense_top(S, Pm, ka, flow, V_, R, Lx, nx, nkeep=6):
    # the nkeep eigenvalues of largest real part (np.linalg.eigvals of the dense block)
    B, _ = block_matrix(S, Pm, ka, flow, V_, R, Lx, nx)
    vals = np.linalg.eigvals(B)
    return vals[np.argsort(-vals.real, kind="stable")][:nkeep]


def _parity(params, idx, v):
    # psi(second sheet)/psi(first sheet) of an eigenvector: +1 even, -1 odd about the midpoint
    nx = params.nx
    psi = T._eigfun(params, idx, v, nx//4)[1]
    return complex(psi[(nx//4 + nx//2) % nx]/psi[nx//4])


def solve(S, Pm, ka, flow, V_, R, Lx, nx, sigma, k=LADDER_K, rtol_mult=1.0):
    # one band-LU-preconditioned shift-invert at the overridden (Ly, diss): k eigenvalues nearest
    # sigma, their residuals and parities; RuntimeError on any refusal
    t0 = time.time()
    params = make_params(nx, Lx)
    x0, um = _x0(params, flow, V_, R)
    op, idx = st.ky_block_operator(x0, _kgrid(nx, Lx), params, 1,
                                   overrides=V.overrides(ka, S, Pm))
    op = doppler(op, ka, um)
    A = U.band_matrix(op, idx, nx, band(Lx, flow, R))
    M, _ = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    tb = time.time() - t0
    grtol = float(np.clip(300*rtol_mult*np.finfo(float).eps*normA/(0.03*abs(sigma)), 1e-10,
                          1e-6*max(rtol_mult, 1.0)))
    big = rtol_mult != 1.0
    r = st.shift_invert(op, sigma, k=k, tol=1e-12, M=M, gmres_rtol=grtol,
                        restart=60 if big else 30, maxiter=40 if big else 20)
    par = np.array([_parity(params, idx, r.right[:, j]) for j in range(len(r.values))])
    return dict(values=np.asarray(r.values), residuals=np.asarray(r.residuals), parity=par,
                u_mean=um, sigma=complex(sigma), nx=nx, n=len(idx), gmres_rtol=grtol,
                gmres_iters=r.stats["gmres_iters"], scale=r.scale, t_build=tb,
                seconds=time.time() - t0)


# ============================================================ the ladder

def cfg_of(S=None, Pm=0.0, ka=KA, flow="none", V_=0.0, R=1.0, Lx=LX):
    # a canonical configuration tuple (S, Pm, ka, flow, V, R, Lx): no flow has V = 0, R = 1
    if flow not in FLOWS:
        raise ValueError(f"flow must be one of {FLOWS}, got {flow!r}")
    if flow == "none":
        V_, R = 0.0, 1.0
    return (float(S0 if S is None else S), float(Pm), float(ka), flow, float(V_), float(R),
            float(Lx))


def _keys(c):
    S, Pm, ka, flow, V_, R, Lx = c
    return dict(S=S, Pm=Pm, ka=ka, flow=flow, V=V_, R=R, Lx=Lx)


def _pow2_at_least(n):
    # the band preconditioner's colouring misbehaves off powers of two (measured: nx = 3072,
    # Lx = 60 fails GMRES at every shift where 4096 converges in 147 iterations)
    return int(2**np.ceil(np.log2(n - 1e-9)))


def is_hi(c):
    # a HI point: its ladder may go one rung further, to NX_HI
    return c[0] == S0 and c[2] == KA and c[6] == LX and (c[1], c[3], c[4], c[5]) in HI


def nxs_for(Lx, cap=None, hi=False):
    # the ladder at this box: NXS scaled by Lx/LX (dx kept or refined), rounded up to powers of
    # two, at most NX_MAX (NX_HI for a HI point), capped
    top = NX_HI if hi else NX_MAX
    out = tuple(_pow2_at_least(n*Lx/LX) for n in NXS + ((NX_HI,) if hi else ()))
    return tuple(n for n in out if n <= top and (cap is None or n <= cap))


def nx_seed_for(Lx):
    return _pow2_at_least(NX_SEED*Lx/LX)


def seed(c, root=None, compute=True):
    # dense eigenvalues of largest Re on the coarse seed grid (cached)
    nx = nx_seed_for(c[6])
    rec = U.cached("ofman_seed", root or DATA_ROOT,
                   (lambda: dict(top=dense_top(*c, nx))) if compute else None, nx=nx, **_keys(c))
    return np.atleast_1d(rec["top"])


def pick(values, floor=GROW_FLOOR):
    # the growing eigenvalue of largest real part (None if none grows)
    vals = [complex(v) for v in np.atleast_1d(values) if complex(v).real > floor]
    return max(vals, key=lambda z: z.real) if vals else None


def rung(c, nx, hint, k, root=None, log=print, compute=True):
    # one ladder rung: the cached solve at shifts moved off `hint` (V.ATTEMPTS), first growing
    # result accepted; (record, lambda). The key is the configuration, never sigma
    last = None
    for attempt, (off, rm) in enumerate(V.ATTEMPTS):
        sig = complex(hint.real*(1.0 + off), hint.imag)
        try:
            rec = U.cached("ofman", root or DATA_ROOT,
                           (lambda sig=sig, rm=rm: solve(*c, nx, sig, k, rm)) if compute else None,
                           attempt=attempt, nx=nx, k=k, **_keys(c))
        except RuntimeError as e:
            last = str(e)
            log(f"    failed {c} nx={nx} attempt {attempt}: {last[:140]}")
            continue
        lam = pick(rec["values"])
        if lam is not None:
            return rec, lam
        log(f"    no growing eigenvalue {c} nx={nx} attempt {attempt}: {rec['values']}")
    raise RuntimeError(f"no growing eigenvalue at {c}, nx={nx}: {last}")


def ladder(c, root=None, log=print, compute=True, cap=None, nxs=None):
    # the point: dense seed, then the nx ladder until CONV_TOL. Returns a dict with every rung
    # (nx, lambda, its pair partner and parity, residual, seconds), conv = |last - previous|/|last|
    # and converged = conv <= CONV_TOL. KeyError if not cached (compute=False)
    nxs = nxs or nxs_for(c[6], cap, is_hi(c))
    top = seed(c, root, compute)
    hint = pick(top)
    if hint is None:
        raise RuntimeError(f"no growing dense seed at {c}: {top}")
    rows = []
    for i, nx in enumerate(nxs):
        try:
            rec, lam = rung(c, nx, hint, SEED_K if i == 0 else LADDER_K, root, log, compute)
        except RuntimeError as e:
            if not rows:
                raise
            log(f"  ladder {c} stops at nx={nx}: {str(e)[:120]}")
            break
        vals = np.atleast_1d(rec["values"])
        j = int(np.argmin(np.abs(vals - lam)))
        others = [complex(v) for i2, v in enumerate(vals) if i2 != j]
        partner = min(others, key=lambda z: abs(z - lam)) if others else complex(np.nan)
        rows.append(dict(nx=nx, lam=lam, partner=partner, parity=complex(rec["parity"][j]),
                         residual=float(np.atleast_1d(rec["residuals"])[j]),
                         seconds=float(rec["seconds"]), u_mean=float(rec["u_mean"])))
        hint = lam
        if len(rows) >= 2 and abs(rows[-1]["lam"] - rows[-2]["lam"]) <= CONV_TOL*abs(lam):
            break
    conv = (abs(rows[-1]["lam"] - rows[-2]["lam"])/abs(rows[-1]["lam"])
            if len(rows) >= 2 else np.nan)
    out = dict(cfg=c, seed=pick(top), rows=rows, lam=rows[-1]["lam"],
               nx=rows[-1]["nx"], conv=float(conv), converged=bool(conv <= CONV_TOL),
               split=float(abs(rows[-1]["partner"] - rows[-1]["lam"])/abs(rows[-1]["lam"])),
               parity=rows[-1]["parity"], seconds=float(sum(r["seconds"] for r in rows)))
    return out


def gamma(rec):
    return complex(rec["lam"]).real


# ============================================================ the campaign's points

def nf_points():
    return [cfg_of(Pm=p) for p in PM_NF]


def fa_points():
    return [cfg_of(Pm=p, flow="tanh", V_=a, R=1.0) for p in PM_FA for a in ALPHA_FA]


def rinv_points():
    return [cfg_of(Pm=0.0, flow="tanh", V_=1.0, R=r) for r in R_INV]


def rvisc_points():
    return [cfg_of(Pm=1.0, flow="tanh", V_=1.0, R=r) for r in R_VISC]


def slope_points():
    return [cfg_of(S=s, Pm=0.0, flow="tanh", V_=1.0, R=R_SLOPE) for s in S_SLOPE + (S0,)]


def jet_points():
    return [cfg_of(Pm=p, flow="sech", V_=1.0, R=r) for p, r in JET]


def box_points():
    out = []
    for Pm, flow, V_, R in BOX:
        out += [cfg_of(Pm=Pm, flow=flow, V_=V_, R=R, Lx=Lx) for Lx in (LX, BOX_LX)]
    return out


def cmk_points():
    out = []
    for ka, Lx, al in CMK:
        out += [cfg_of(Pm=0.0, ka=ka, flow=f, V_=v, Lx=Lx) for f, v in (("none", 0.0), ("tanh", al))]
    return out


PHASES = ("theory", "nf", "fa", "rvisc", "jet", "box", "rinv", "slope", "cmk", "dense", "osc")
POINTS = dict(nf=nf_points, fa=fa_points, rinv=rinv_points, rvisc=rvisc_points, slope=slope_points,
              jet=jet_points, box=box_points, cmk=cmk_points)


def theory_pms():
    return tuple(p for p in sorted(set(PM_NF) | set(PM_FA) | {p for p, _ in JET} |
                                   {b[0] for b in BOX}) if p > 0)


def dense_check(Pm, alpha, root=None, compute=True, nx=None):
    # the independent dense route: np.linalg.eigvals of the dense block at NX_INDEP, at alpha and
    # at alpha = 0, so the shear ratio is recomputed with no shift-invert, band LU or ladder
    nx = nx or NX_INDEP

    def comp():
        out = {}
        for a in sorted({0.0, float(alpha)}):
            c = cfg_of(Pm=Pm, flow="tanh" if a else "none", V_=a, R=1.0)
            out[f"top_{a}"] = dense_top(*c, nx)
        return out
    return U.cached("ofman_dense", root or DATA_ROOT, comp if compute else None, Pm=float(Pm),
                    alpha=float(alpha), nx=nx, S=S0, ka=KA, Lx=LX)


def split_branches(vals, im=None):
    # (fastest Re among |Im| > im, fastest Re among |Im| <= im): the oscillating branch vs the rest
    im = OSC_IM if im is None else im
    vals = np.asarray(vals)
    hi, lo = vals[np.abs(vals.imag) > im], vals[np.abs(vals.imag) <= im]
    top = lambda v: complex(v[np.argmax(v.real)]) if len(v) else complex(np.nan, np.nan)
    return top(hi), top(lo)


def osc_scan(root=None, compute=True):
    # the sech R = 6 jet's spectrum vs Pm on a coarse dense grid: {Pm: (oscillating branch,
    # the rest)} -- where the |Im| ~ 0.8 mode that dominates at S_v = 1e2 appears

    def comp():
        out = {}
        for i, Pm in enumerate(OSC_PM):
            B, _ = block_matrix(S0, Pm, KA, "sech", 1.0, 6.0, LX, OSC_NX)
            out[f"vals_{i}"] = np.linalg.eigvals(B)
        return out
    rec = U.cached("ofman_osc", root or DATA_ROOT, comp if compute else None, Pm=list(OSC_PM),
                   nx=OSC_NX, S=S0, ka=KA, Lx=LX)
    return {Pm: split_branches(rec[f"vals_{i}"]) for i, Pm in enumerate(OSC_PM)}


def make_data(root=None, wall_budget=3*3600.0, phases=None, log=print, cap=None):
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    done = True
    for name in phases or PHASES:
        t0 = time.time()
        ok = True
        if name == "theory":
            for p in theory_pms():
                V.theory_table(float(p), True, theory_root(p))
                log(f"  theory table Pm = {p:g}: {theory_root(p)}")
        elif name == "dense":
            for Pm, al in DENSE:
                if time.time() > deadline:
                    ok = False
                    break
                dense_check(Pm, al, root)
                log(f"  dense check Pm={Pm:g} alpha={al}: done")
        elif name == "osc":
            for Pm, (h, l) in osc_scan(root).items():
                log(f"  osc Pm={Pm:g}: oscillating branch {h:.4e}, rest {l:.4e}")
        elif name in POINTS:
            for c in POINTS[name]():
                if time.time() > deadline:
                    ok = False
                    break
                try:
                    r = ladder(c, root, log, cap=cap)
                except RuntimeError as e:
                    ok = False
                    log(f"  FAILED {c}: {str(e)[:160]}")
                    continue
                full = cap is None or nxs_for(c[6], None, is_hi(c)) == nxs_for(c[6], cap, is_hi(c))
                ok &= r["converged"] or full
                log(f"  {name} S={c[0]:.0e} Pm={c[1]:g} ka={c[2]} {c[3]} V={c[4]} R={c[5]} Lx={c[6]:g}: "
                    f"lambda = {r['lam']:.8e} (nx {r['nx']}, conv {r['conv']:.1e}, parity "
                    f"{r['parity'].real:+.3f}, split {r['split']:.1e}, {r['seconds']:.0f} s)")
        else:
            raise ValueError(name)
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")
        done &= ok
    return bool(done)


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def results(points, root=None):
    # {cfg: ladder record} for every cached point (uncached or failed points skipped)
    out = {}
    for c in points:
        try:
            out[c] = ladder(c, root or DATA_ROOT, _quiet, compute=False)
        except (KeyError, RuntimeError):
            pass
    return out


def nf_results(root=None):
    return {c[1]: r for c, r in results(nf_points(), root).items()}


def fa_results(root=None):
    # {(Pm, alpha): record}, alpha = 0 from the no-flow points
    out = {(c[1], c[4]): r for c, r in results(fa_points(), root).items()}
    for Pm, r in nf_results(root).items():
        if Pm in PM_FA:
            out[(Pm, 0.0)] = r
    return out


def rinv_results(root=None):
    return {c[5]: r for c, r in results(rinv_points(), root).items()}


def rvisc_results(root=None):
    return {c[5]: r for c, r in results(rvisc_points(), root).items()}


def slope_results(root=None):
    return {c[0]: r for c, r in results(slope_points(), root).items()}


def jet_results(root=None):
    return {(c[1], c[5]): r for c, r in results(jet_points(), root).items()}


def box_results(root=None):
    return {(c[1], c[3], c[4], c[5], c[6]): r for c, r in results(box_points(), root).items()}


def cmk_results(root=None):
    return {(c[2], c[3], c[4]): r for c, r in results(cmk_points(), root).items()}


def dense_results(root=None):
    out = {}
    for Pm, al in DENSE:
        try:
            d = dense_check(Pm, al, root, compute=False)
        except (KeyError, RuntimeError):
            continue
        out[(Pm, al)] = {a: pick(d[f"top_{a}"]) for a in sorted({0.0, float(al)})}
    return out


def shear_ratios(fa=None, root=None):
    # {(Pm, alpha): gamma(alpha)/gamma(0)} from the field-aligned table
    fa = fa_results(root) if fa is None else fa
    return {(Pm, a): gamma(r)/gamma(fa[(Pm, 0.0)]) for (Pm, a), r in fa.items()
            if (Pm, 0.0) in fa and a > 0}


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3*3600.0
    args = sys.argv[2:]
    cap = None
    if args and args[0].startswith("cap="):
        cap = int(args[0][4:])
        args = args[1:]

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    log(f"profile {PROFILE}, root {DATA_ROOT}, cap {cap}")
    make_data(wall_budget=budget, phases=tuple(args) or None, log=log, cap=cap)
