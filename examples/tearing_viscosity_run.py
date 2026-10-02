# Visco-resistive tearing with field-aligned shear flow: theory, sensitivities, k_max by Newton
# and resumable data generation for examples/tearing-viscosity.ipynb -- plans/AUTODIFF_PLAN.md
# rung 2b item 1 ("Viscosity: gamma(Pm, alpha, k), diss = (nu, eta)"), the paper's stated future
# work (Mallet, Eriksson, Swisdak & Juno, JPP 91 E146 (2025) sec. 2).
#
# Setup exactly as rung 2 (tearing_rung2_run.py) but nu > 0: dims=2, hyper = 1, a = v_Ay = 1,
# Psi0 = sech^2 x (+ its two nearest periodic images), Phi0 = alpha Psi0, the ky block iky = 1
# (ny = 8), fp64, S = 1/eta, Pm = nu/eta, so diss = (nu, eta) = (Pm/S, 1/S).
#
# BOTH k AND diss ARE OVERRIDES. One Parameters per (S, nx, Lx) -- rung 2's R.make_params, whose
# own diss is the placeholder (0, 1/S) -- serves every k, Pm and alpha: the operators are always
# built at overrides={"Ly": 2 pi/k, "diss": (Pm/S, 1/S)} (J there == J of a Parameters built with
# that Ly and diss: the independent-route gate in tests/test_tearing_viscosity.py).
#
# FOUR SENSITIVITIES PER SOLVE from one right/left pair (stability.eigenvalue_sensitivity):
#   eta    dparams={"diss": (0, 1)}  (nu held)       nu  dparams={"diss": (1, 0)}  (eta held)
#   k      dparams={"Ly": -Ly/k}                     alpha  dx0 = (Psi0, 0)
# Local exponents (one solve each, no sweep):
#   q_Pm  = d ln gamma/d ln Pm |_S          =  (nu/gamma) gamma_nu
#   q_S   = d ln gamma/d ln S  |_Pm         = -(eta gamma_eta + nu gamma_nu)/gamma
#   q_Snu = d ln gamma/d ln S  |_nu         = -(eta/gamma) gamma_eta   (= q_S + q_Pm)
#   q_k   = (k/gamma) gamma_k,  q_alpha = d ln gamma/d ln(1 - alpha^2) (rung 1b)
#
# THEORY (alpha = 0; independent of taranis). The visco-resistive inner layer (f ~ f'x, |f'(0)| =
# 2, Porcelli 1987 Phys. Fluids 30, 1734 for the limits) in s = x/delta_eta, delta_eta =
# (eta/(k|f'|))^(1/3), Lam = gamma/(eta^(1/3)(k|f'|)^(2/3)) -- B&L 2018's variables (T._inner_bvp)
# plus the viscous term; derivation in the notebook:
#     psi'' = Lam psi - s u,      Pm u'''' = Lam u'' + s psi''          (Pm = nu/eta)
# psi even, u odd; far field the ideal branch u = Lam psi/s (two conditions at s_max kill the two
# growing viscous exponentials); Delta' delta_eta = F(Lam, Pm) = 2 c1/c0 of psi -> c0 + c1 s,
# Richardson-extrapolated in s_max like T.inner_F_numeric. Pm -> 0 is B&L's exact relation
# (T.inner_F; the deviation is linear in Pm). Pm -> inf at fixed Lam Pm^(1/3): the inertia-free
# layer, F -> Pm^(-1/6) F_v(Lam Pm^(1/3)), F_v from the same BVP without the Lam u'' term. Its
# limits, n = 2 (Delta' a ~ 15/(ka)^2):
#   viscous constant-psi (Lam -> 0): F_v -> I_V L, I_V = pi 6^(-1/3) Gamma(5/6)/Gamma(7/6)
#       (derived in Fourier space; the m = 1 member of the same formula is FKR's C_FKR):
#       gamma = Delta' eta^(5/6) nu^(-1/6) (k f')^(1/3)/I_V ~ S^(-2/3) Pm^(-1/6) (ka)^(-5/3)
#   viscous nonconstant-psi (F_v's pole L* = C_VC):
#       gamma = C_VC eta^(2/3) nu^(-1/3) (k f')^(2/3)         ~ S^(-1/3) Pm^(-1/3) (ka)^(2/3)
#   their crossing (k_max): k_max ~ S^(-1/7) Pm^(1/14), gamma_max ~ S^(-3/7) Pm^(-2/7)
#   (n = 1: S^(-1/4) Pm^(1/8), S^(-1/2) Pm^(-1/4)). The S-exponents at fixed Pm are the
#   inviscid n = 2 ones; the Pm-exponents discriminate. Valid for Pm >> 1 (inertia negligible:
#   gamma delta_v^2/nu = Lam Pm^(-2/3) << 1) and delta_v/a = Pm^(1/6) delta_eta << 1.
# The F(Lam, Pm) relation is TABULATED per Pm (theory_table: c. 64 BVP solves, cached under
# DATA_ROOT/theory) and inverted by a spline; d ln F/d ln Pm from tables at Pm e^(+-PM_DLN).
#
# SHEAR (alpha > 0): no theory with viscosity exists; the scaling (notebook): shear changes the
# viscous layer once the growth rate falls below the shear rate across it, gamma/(alpha k f'
# delta_v) ~ C_VC Pm^(-1/2)/alpha < 1 -- so at large Pm the alpha-dependence should set in at
# alpha ~ Pm^(-1/2) instead of alpha ~ 1.
#
# K_MAX by Newton on dgamma/dk = 0 exactly as rung 2 (R.newton_kmax: secant curvature, step cap,
# relative stop; then k*(1 +- CURV_H) for gamma_kk and the mixed derivatives; implicit slopes in
# S at fixed Pm, in Pm at fixed S and in alpha). Boxes: rung 2's R.kmax_box(S) (one per S, shared
# by every Pm and alpha) and R.disp_box(S, ka). Continuation chains: in Pm (alpha fixed) seeded
# from this theory, in alpha (Pm fixed) starting from the Pm chain's alpha = 0 solution; seeds
# clipped by R._trust (SEED_TRUST) so a branch switch (the alpha >~ 0.95 low-k maximum of rung 2)
# shows as a jump in k*, never a crash. branch_scan: coarse dense gamma(k) to see both maxima.
# alpha_c (phase "alpha_c", added in review 2026-10-02): the viscous stability boundary by bisection in
# alpha on the sign of max_k of the largest REAL dense eigenvalue (max_real_growth), plus a linear root
# and a 2 nx re-evaluation of the final bracket.
#
# `python examples/tearing_viscosity_run.py [wall_s] [S=..] [P=..] [A=..] [phase ...]` runs
# make_data (resumable; every solve is one cache file under DATA_ROOT/solves).
# TARANIS_VISC_PROFILE=smoke|full (default full); smoke writes to DATA_ROOT_SMOKE.
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import functools
import sys
import time

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import gamma as Gfun

from taranis import stability as st

import tearing_eigen_run as T
import tearing_sensitivity_run as U
import tearing_rung2_run as R

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FULL = os.path.join(HERE, "data", "tearing-viscosity")
DATA_ROOT_SMOKE = os.path.join(HERE, "data", "tearing-viscosity-smoke")
PROFILE = os.environ.get("TARANIS_VISC_PROFILE", "full")
if PROFILE not in ("full", "smoke"):
    raise ValueError(f"TARANIS_VISC_PROFILE must be 'full' or 'smoke', got {PROFILE!r}")
DATA_ROOT = DATA_ROOT_SMOKE if PROFILE == "smoke" else DATA_ROOT_FULL

TWO_PI = 2*np.pi
ETA_DIR = (0.0, 1.0)          # d/d eta of diss = (nu, eta), nu held
NU_DIR = (1.0, 0.0)           # d/d nu, eta held
CURV_H = R.CURV_H
NX_DENSE = R.NX_DENSE

# ---- configuration (full profile; smoke below)
PM_CHAIN = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)    # k_max(Pm) continuation
S_CHAIN = (1e3, 1e4, 1e5)                                     # ... at alpha = 0, these S
CHAIN_SHEAR = ((1e4, 0.9),)                                   # ... and (S, alpha) with shear
PM_DISP = (0.1, 1.0, 10.0, 100.0)
S_DISP = (1e3, 1e4, 1e5)
KA_DISP = (0.15, 0.3, 0.4, 0.6, 0.85, 1.25, 1.5, 2.0)    # a subset of rung 2's KA_DISP
DISP_SHEAR = ((1e4, 0.9),)                                    # (S, alpha): PM_DISP sweeps, nx only
PM_KA = {1e3: (1.0, 10.0, 100.0), 1e4: (0.1, 1.0, 10.0, 100.0)}   # k_max(alpha): S -> Pm list
ALPHA_KA = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99)
KA_HI = {1e5: ((10.0,), (0.0, 0.3, 0.6, 0.8, 0.9))}           # S: (Pm list, alpha list)
GRID2_POINTS = ((1e4, 10.0, 0.9), (1e5, 100.0, 0.0))        # (S, Pm, alpha): Newton at 2 nx
FD_POINTS = ((1e3, 0.5, 0.0, 1.0), (1e4, 0.3, 0.8, 10.0), (1e4, 1.0, 0.6, 0.1))  # S, ka, al, Pm
FD_H = (1e-2, 3e-3, 1e-3)
ALPHA_C2 = (0.02, 0.04)       # c2 = -gamma_alpha/(2 alpha gamma) at k*(alpha=0), Richardson in alpha^2
PM_C2 = (0.0,) + PM_CHAIN     # ... at every S_CHAIN (Pm = 0: rung 2's k*(0) on the same box)
ALPHA_C_POINTS = ((1e3, 10.0, 0.95, 0.99), (1e4, 100.0, 0.95, 0.99), (1e3, 100.0, 0.6, 0.8),
                  (1e4, 1000.0, 0.6, 0.8))   # (S, Pm, alpha growing, alpha stable): alpha_c brackets
AC_NBIS = 7                   # bisection steps (bracket / 2^7)
KA_AC = tuple(np.round(np.geomspace(0.15, 1.6, 12), 5))     # max_real_growth's coarse k grid
BRANCH = ((1e3, 0.95, (0.0, 0.1, 1.0, 10.0, 100.0)), (1e3, 0.99, (0.0, 0.1, 1.0, 10.0, 100.0)),
          (1e4, 0.95, (0.0, 10.0, 100.0)), (1e4, 0.99, (0.0, 1.0, 10.0, 100.0)))  # (S, alpha, Pms)
KA_BRANCH = tuple(np.round(np.geomspace(0.1, 1.6, 21), 5))
if PROFILE == "smoke":
    PM_CHAIN, S_CHAIN, CHAIN_SHEAR = (0.1, 1.0, 10.0), (1e3,), ()
    PM_DISP, S_DISP, KA_DISP, DISP_SHEAR = (1.0, 10.0), (1e3,), (0.3, 0.6, 1.0), ((1e3, 0.8),)
    PM_KA, ALPHA_KA, KA_HI = {1e3: (1.0,)}, (0.0, 0.5, 0.8), {}
    GRID2_POINTS = ((1e3, 1.0, 0.5),)
    FD_POINTS = ((1e3, 0.5, 0.3, 1.0),)
    BRANCH = ((1e3, 0.95, (0.0, 10.0)),)
    PM_C2 = (0.0, 1.0)
    KA_BRANCH = (0.2, 0.35, 0.5, 0.7, 0.9)
    ALPHA_C_POINTS, AC_NBIS = ((1e3, 100.0, 0.6, 0.8),), 2


def visc_exponents(n):
    # large-Pm asymptotes for Delta' a ~ (ka)^(-n): viscous constant-psi gamma ~ Delta' S^(-2/3)
    # Pm^(-1/6) k^(1/3), viscous nonconstant-psi S^(-1/3) Pm^(-1/3) k^(2/3), and their crossing
    d = 3*n + 1
    return dict(vfkr_S=-2/3, vfkr_Pm=-1/6, vfkr_k=1/3 - n, vcoppi_S=-1/3, vcoppi_Pm=-1/3,
                vcoppi_k=2/3, kmax_S=-1/d, kmax_Pm=1/(2*d), gmax_S=-1/3 - 2/(3*d),
                gmax_Pm=-1/3 + 1/(3*d))


PRED = visc_exponents(2)      # n = 2: -1/7, 1/14, -3/7, -2/7


# ============================================================ theory: the inner layer

def const_psi_integral(m):
    # I_m = int (1 - s U) ds for U^(2m) + (-1)^m ... the constant-psi layer integral
    # (m = 1: inertia, U'' = s(sU - 1); m = 2: viscosity, U'''' = s(1 - sU)), from the Fourier
    # solution sqrt|kappa| K_nu(|kappa|^(m+1)/(m+1)), nu = 1/(2m+2):
    #   I_m = pi (2m+2)^(-2 nu) Gamma(1 - nu)/Gamma(1 + nu)
    nu = 1.0/(2*m + 2)
    return float(np.pi*(2*m + 2)**(-2*nu)*Gfun(1 - nu)/Gfun(1 + nu))


I_V = const_psi_integral(2)    # 2.10359...; const_psi_integral(1) == T.C_FKR


def _inner_bvp_visc(lam, Pm, smax, inertia=True, tol=1e-9):
    # y = (psi, psi', u, u', u'', u'''); see the header. inertia=False drops Lam u'' (the
    # Pm -> inf layer, used with Pm = 1 in its own variables)
    lin = lam if inertia else 0.0

    def ode(s, y):
        d2psi = lam*y[0] - s*y[2]
        return np.vstack([y[1], d2psi, y[3], y[4], y[5], (lin*y[4] + s*d2psi)/Pm])

    def bc(ya, yb):
        return np.array([ya[1], ya[2], ya[4], ya[0] - 1.0,
                         yb[2] - lam*yb[0]/smax,
                         yb[3] - lam*yb[1]/smax + lam*yb[0]/smax**2])
    w = min(1.0, np.sqrt(Pm/max(lam, 1e-3))) if inertia else 1.0   # viscous sublayer
    if Pm < 1e-2 and inertia:
        # the decaying viscous far-field exponentials exp(-s^(3/2)/(1.5 Pm^(1/4))) are steep at
        # small Pm: a uniform mesh resolving them (solve_bvp's refinement runs out of nodes)
        far = np.linspace(5*w, smax, int(min(40000, 2*smax**1.5/Pm**0.25)))
    else:
        far = np.geomspace(5*w, smax, 400)
    s = np.unique(np.concatenate([np.linspace(0.0, 5*w, 200), far]))
    y0 = np.zeros((6, s.size))
    y0[0] = 1.0
    y0[2] = lam*s/(1.0 + s*s)
    y0[3] = lam*(1.0 - s*s)/(1.0 + s*s)**2
    r = solve_bvp(ode, bc, s, y0, tol=tol, max_nodes=300000)
    if r.status != 0:
        raise RuntimeError(f"viscous inner BVP failed at Lam={lam}, Pm={Pm}, smax={smax}: "
                           f"{r.message}")
    y = r.sol(smax)
    c1 = y[1]
    return y[0] - smax*c1, c1          # (c0, c1) of the far field psi = c0 + c1 s


def _smaxes(Pm, inertia):
    sc = max(1.0, Pm**(1.0/6.0)) if inertia else 1.0      # the viscous layer is Pm^(1/6) wider
    return 40.0*sc, 80.0*sc


def inner_invF_visc(lam, Pm, inertia=True):
    # 1/F = c0/(2 c1), Richardson-extrapolated in s_max (O(1/s_max^2)); smooth through F's pole
    s0, s1 = _smaxes(Pm, inertia)
    v = [c0/(2*c1) for c0, c1 in (_inner_bvp_visc(lam, Pm, s, inertia) for s in (s0, s1))]
    return (s1**2*v[1] - s0**2*v[0])/(s1**2 - s0**2)


def inner_F_visc(lam, Pm, inertia=True):
    # Delta' delta_eta = F(Lam, Pm); Pm = 0 is B&L's exact relation (T.inner_F)
    if Pm == 0:
        return float(T.inner_F(lam))
    s0, s1 = _smaxes(Pm, inertia)
    v = [2*c1/c0 for c0, c1 in (_inner_bvp_visc(lam, Pm, s, inertia) for s in (s0, s1))]
    return float((s1**2*v[1] - s0**2*v[0])/(s1**2 - s0**2))


def pole_visc(Pm, inertia=True):
    # the Lam where F -> inf (the nonconstant-psi limit): the root of 1/F. Pm = 0: exactly 1
    if Pm == 0:
        return 1.0
    f = lambda l: inner_invF_visc(l, Pm, inertia)
    hi = 1.05 if not inertia else 1.05*min(1.0, 1.53*Pm**(-1.0/3.0))
    while f(hi) > 0:
        hi *= 1.25
    lo = 0.6*hi
    while f(lo) < 0:
        lo *= 0.7
    return float(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13))


@functools.lru_cache(maxsize=1)
def c_vcoppi():
    # C_VC: the pole of the inertia-free layer F_v (viscous nonconstant-psi prefactor)
    return pole_visc(1.0, inertia=False)


# r = Lam/Lam_pole grid, in x = ln r: geometric toward 0, then toward the pole 1 - 1e-5
X_TAB = np.concatenate([np.log(np.geomspace(1e-5, 0.9, 40)),
                        np.log1p(-np.geomspace(0.1, 1e-5, 25)[1:])])
PM_DLN = 0.01                 # d ln F/d ln Pm from the tables at Pm e^(+-PM_DLN)


def theory_root(root=None):
    return os.path.join(root or DATA_ROOT, "theory")


def _table_path(Pm, inertia, root):
    return os.path.join(theory_root(root), f"F_{'i' if inertia else 'v'}_Pm{Pm:.12e}.npz")


@functools.lru_cache(maxsize=64)
def theory_table(Pm, inertia=True, root=None, compute=True):
    # (ln Lam_pole, X_TAB, g) with g = ln F + ln(1 - r) on X_TAB (smooth through the pole);
    # cached as one npz per (Pm, inertia)
    Pm = float(Pm)
    path = _table_path(Pm, inertia, root)
    if os.path.exists(path):
        d = np.load(path)
        if np.array_equal(d["X"], X_TAB):
            return float(d["lnpole"]), d["X"], d["g"]
    if not compute:
        raise KeyError(f"theory table not cached: Pm={Pm}, inertia={inertia}")
    lp = pole_visc(Pm, inertia)
    r = np.exp(X_TAB)
    g = np.array([np.log(inner_F_visc(lp*ri, Pm, inertia)) + np.log1p(-ri) for ri in r])
    os.makedirs(theory_root(root), exist_ok=True)
    T._save(path, dict(lnpole=np.log(lp), X=X_TAB, g=g, Pm=Pm, inertia=inertia))
    return float(np.log(lp)), X_TAB.copy(), g


@functools.lru_cache(maxsize=64)
def _spline(Pm, inertia=True, root=None):
    lp, X, g = theory_table(Pm, inertia, root)
    return lp, CubicSpline(X, g)


def F_table(lam, Pm, inertia=True, root=None):
    # (ln F, d ln F/d ln Lam) from the table; nan outside it
    if Pm == 0:
        h = 1e-6
        lf = lambda l: np.log(T.inner_F(l))
        return float(lf(lam)), float((lf(lam*np.exp(h)) - lf(lam*np.exp(-h)))/(2*h))
    lp, sp = _spline(float(Pm), inertia, root)
    x = np.log(lam) - lp
    if not (X_TAB[0] <= x <= X_TAB[-1]):
        return np.nan, np.nan
    r = np.exp(x)
    return float(sp(x) - np.log1p(-r)), float(sp(x, 1) + r/(1.0 - r))


def lam_of_visc(Ddelta, Pm, root=None):
    # invert F(., Pm) = Delta' delta_eta on (0, pole); nan beyond the table
    if Pm == 0:
        return T.lam_of(Ddelta)
    lp, sp = _spline(float(Pm), True, root)
    target = np.log(Ddelta)
    f = lambda x: sp(x) - np.log1p(-np.exp(x)) - target
    a, b = X_TAB[0], X_TAB[-1]
    if not (f(a) < 0 < f(b)):
        return np.nan
    return float(np.exp(lp + brentq(f, a, b, xtol=1e-14, rtol=1e-14)))


def _lam_th(ka, S, Pm, root=None):
    Dp = float(T.dprime_sech2(ka))
    if Dp <= 0:
        return np.nan
    return lam_of_visc(Dp*((1.0/S)/(ka*T.FPRIME))**(1.0/3.0), Pm, root)


def gamma_theory_visc(ka, S, Pm, root=None):
    # the exact visco-resistive layer relation, alpha = 0 (a = 1, eta = 1/S, nu = Pm/S)
    lam = _lam_th(ka, S, Pm, root)
    return lam*(1.0/S)**(1.0/3.0)*(ka*T.FPRIME)**(2.0/3.0)


def theory_exponents(ka, S, Pm, root=None):
    # analytic-from-the-relation local exponents at fixed k: with m = d ln F/d ln Lam,
    # mu = d ln F/d ln Pm |_Lam:  q_Pm = -mu/m, q_S (fixed Pm) = -1/3 - 1/(3m),
    # q_k = 2/3 + (d ln Delta'/d ln k - 1/3)/m, q_Snu = q_S + q_Pm. mu is taken at FIXED
    # r = Lam/Lam_pole (the tables' variable, so it stays defined next to the pole, which moves
    # with Pm): ln F = g(x, ln Pm) - ln(1 - e^x), x = ln Lam - ln Lam_pole(Pm), so
    # mu = g_lnPm|x - m d ln Lam_pole/d ln Pm (both central differences over Pm e^(+-PM_DLN))
    lam = _lam_th(ka, S, Pm, root)
    if Pm == 0:
        qS, qk = R.theory_s_exponent(ka, S), R.theory_k_exponent(ka, S)
        return dict(lam=lam, m=np.nan, mu=0.0, qPm=0.0, qS=qS, qk=qk, qSnu=qS)
    _, m = F_table(lam, Pm, root=root)
    lp0, _ = _spline(float(Pm), True, root)
    lpp, spp = _spline(float(Pm*np.exp(PM_DLN)), True, root)
    lpm, spm = _spline(float(Pm*np.exp(-PM_DLN)), True, root)
    x = np.log(lam) - lp0
    mu = float((spp(x) - spm(x))/(2*PM_DLN) - m*(lpp - lpm)/(2*PM_DLN))
    qS = -1.0/3.0 - 1.0/(3.0*m)
    qPm = -mu/m
    return dict(lam=lam, m=m, mu=mu, qPm=qPm, qS=qS,
                qk=2.0/3.0 + (R.dlnDprime_dlnk(ka) - 1.0/3.0)/m, qSnu=qS + qPm)


def theory_kmax_visc(S, Pm, root=None):
    # (k_max, gamma_max) of the relation: the root of q_k
    if Pm == 0:
        return R.theory_kmax(S)
    f = lambda lk: theory_exponents(np.exp(lk), S, Pm, root)["qk"]
    k = float(np.exp(brentq(f, np.log(0.05), np.log(2.2), xtol=1e-13, rtol=1e-13)))
    return k, float(gamma_theory_visc(k, S, Pm, root))


def theory_kmax_slopes_visc(S, Pm, h=1e-3, root=None):
    # d ln k_max/d ln S |_Pm (central difference of the root), d ln k_max/d ln Pm |_S (root at
    # the Pm e^(+-PM_DLN) tables), and the envelope exponents q_S, q_Pm at k_max
    k0 = theory_kmax_visc(S, Pm, root)[0]
    kp, km = theory_kmax_visc(S*np.exp(h), Pm, root)[0], theory_kmax_visc(S*np.exp(-h), Pm, root)[0]
    out = dict(k_S=float(np.log(kp/km)/(2*h)))
    if Pm > 0:
        kpp = theory_kmax_visc(S, Pm*np.exp(PM_DLN), root)[0]
        kpm = theory_kmax_visc(S, Pm*np.exp(-PM_DLN), root)[0]
        out["k_Pm"] = float(np.log(kpp/kpm)/(2*PM_DLN))
    else:
        out["k_Pm"] = 0.0
    e = theory_exponents(k0, S, Pm, root)
    out.update(g_S=e["qS"], g_Pm=e["qPm"])
    return out


def gamma_vfkr(ka, S, Pm):
    # viscous constant-psi: Delta' eta^(5/6) nu^(-1/6) (k f')^(1/3)/I_V
    eta = 1.0/S
    return float(T.dprime_sech2(ka))*eta**(5/6)*(Pm*eta)**(-1/6)*(ka*T.FPRIME)**(1/3)/I_V


def gamma_vcoppi(ka, S, Pm):
    # viscous nonconstant-psi: C_VC eta^(2/3) nu^(-1/3) (k f')^(2/3)
    eta = 1.0/S
    return c_vcoppi()*eta**(2/3)*(Pm*eta)**(-1/3)*(ka*T.FPRIME)**(2/3)


def theory_pms():
    # every Pm the campaign's theory needs, with its +-PM_DLN neighbours
    base = sorted(set(PM_CHAIN) | set(PM_DISP) | {p for v in PM_KA.values() for p in v}
                  | {p for v in KA_HI.values() for p in v[0]} | {p for _, p, _ in GRID2_POINTS}
                  | {p for *_, p in FD_POINTS} - {0.0})
    return tuple(float(p*np.exp(s*PM_DLN)) if s else float(p) for p in base if p > 0
                 for s in (0, 1, -1))


def build_theory(root=None, log=print):
    t0 = time.time()
    for p in theory_pms():
        theory_table(p, True, root)
    theory_table(1.0, False, root)
    log(f"  theory tables: {len(theory_pms()) + 1} ({time.time() - t0:.0f} s); "
        f"C_VC = {c_vcoppi():.10f}, I_V = {I_V:.10f}")


# ============================================================ one solve

def diss_of(S, Pm):
    return (float(Pm)/float(S), 1.0/float(S))


def overrides(k, S, Pm):
    return {"Ly": TWO_PI/float(k), "diss": diss_of(S, Pm)}


@functools.lru_cache(maxsize=256)
def _dense_others(S, Pm, k, alpha, Lx, nx):
    # the dense block's spectrum on a coarse grid: (every eigenvalue but the fastest, the
    # fastest, the number growing) -- the gap spectrum and the sigma seed. Eigenvalues only
    # (np.linalg.eigvals: half of eig_dense's cost, which is most of a solve on the loaded M1);
    # the growth threshold is relative to the block's inf-norm
    params = R.make_params(S, nx, Lx)
    x0, _ = U.equilibrium(params, alpha)
    B, _ = st.ky_block_matrix(x0, R._kgrid(S, nx, Lx), params, 1, overrides=overrides(k, S, Pm))
    vals = np.linalg.eigvals(B)
    vals = vals[np.argsort(-vals.real, kind="stable")]
    scale = float(np.abs(B).sum(axis=1).max())
    return vals[1:], complex(vals[0]), int(np.sum(vals.real > 1e-10*scale))


def sens_solve(S, Pm, k, alpha, nx, Lx, sigma, lam_hint=None, rtol_mult=1.0, dense_nx=None):
    # one right/left pair (band-LU-preconditioned shift_invert + left_eigenvector at the
    # overridden (Ly, diss)) and the four sensitivities; raises RuntimeError on any refusal
    t0 = time.time()
    params = R.make_params(S, nx, Lx)
    kg = R._kgrid(S, nx, Lx)
    x0, dx0 = U.equilibrium(params, alpha)
    ov = overrides(k, S, Pm)
    op, idx = st.ky_block_operator(x0, kg, params, 1, overrides=ov)
    A = U.band_matrix(op, idx, nx, T.band_width("sech2", Lx))
    M, MH = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    tb = time.time() - t0
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
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
    dnx = dense_nx or min(NX_DENSE, nx)
    others, lam_dense, npos = _dense_others(S, Pm, float(k), alpha, Lx, dnx)
    spectrum = np.concatenate([[lam], others])
    sens = {}
    for name, dxx, dp in (("eta", None, {"diss": ETA_DIR}), ("nu", None, {"diss": NU_DIR}),
                          ("k", None, R.k_direction(k)), ("alpha", dx0, None)):
        d, _ = st.dj_operator(x0, dxx, kg, params, iky=1, dparams=dp, overrides=ov)
        sens[name] = st.eigenvalue_sensitivity(op, d, lam, v, w, spectrum=spectrum)
    s_eta = sens["eta"]
    x, psi, phi, d2 = T._eigfun(params, idx, v, params.nx//2)
    prof = T._profile(x, psi, phi, d2, params.nx//2, Lx, half=1.5)
    return dict(S=S, Pm=Pm, k=float(k), alpha=alpha, nx=nx, Lx=Lx, sigma=complex(sigma),
                rtol_mult=rtol_mult, lam=lam, dlam_eta=s_eta.dlam, dlam_nu=sens["nu"].dlam,
                dlam_k=sens["k"].dlam, dlam_alpha=sens["alpha"].dlam, kappa=s_eta.kappa,
                wv=s_eta.wv, gap=s_eta.gap, rel_err=s_eta.rel_err, res_r=s_eta.residual_right,
                res_l=s_eta.residual_left, scale=s_eta.scale, lam_dense=lam_dense,
                npos_dense=npos, dense_nx=dnx, width=T.width_quarter(prof["xn"], prof["d2n"]),
                gmres_rtol=grtol, right_iters=right.stats["gmres_iters"],
                left_iters=left.stats["gmres_iters"], t_build=tb, t_right=tr - tb,
                t_left=tl - tr, seconds=time.time() - t0)


def eig_only(S, Pm, k, alpha, nx, Lx, sigma, lam_hint=None, eta=None, nu=None):
    # the eigenvalue alone (FD gates): eta / nu move through the diss override, k through Ly
    params = R.make_params(S, nx, Lx)
    kg = R._kgrid(S, nx, Lx)
    x0, _ = U.equilibrium(params, alpha)
    nu0, eta0 = diss_of(S, Pm)
    ov = {"Ly": TWO_PI/float(k), "diss": (float(nu0 if nu is None else nu),
                                           float(eta0 if eta is None else eta))}
    op, idx = st.ky_block_operator(x0, kg, params, 1, overrides=ov)
    A = U.band_matrix(op, idx, nx, T.band_width("sech2", Lx))
    M, _ = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
    grtol = float(np.clip(300*np.finfo(float).eps*normA/max(dist, 1e-300), 1e-10, 1e-6))
    r = st.shift_invert(op, sigma, k=1, M=M, tol=1e-12, gmres_rtol=grtol, restart=30, maxiter=20)
    return dict(lam=complex(r.values[0]), res=float(r.residuals[0]))


ATTEMPTS = ((0.03, 1.0), (0.13, 1.0), (0.5, 1.0), (-0.03, 1.0), (0.03, 100.0), (0.13, 100.0),
            (0.03, 1e4))           # (relative shift offset, inner-rtol multiplier), as rung 2


def point(S, Pm, k, alpha, nx, Lx, pred, root=None, log=print, compute=True, tag=None):
    # a cached sensitivity solve (kind "visc"), retried at moved shifts / looser inner solves
    # on a refusal or an eigenvalue that is not a real growing mode. The key is the
    # configuration, never sigma
    cfg = dict(S=S, Pm=Pm, k=float(k), alpha=alpha, nx=nx, Lx=Lx)
    if tag:
        cfg["tag"] = tag
    last = None
    for attempt, (off, rm) in enumerate(ATTEMPTS):
        sig = pred*(1.0 + off)
        try:
            rec = U.cached("visc", root or DATA_ROOT,
                           (lambda sig=sig, rm=rm: sens_solve(S, Pm, k, alpha, nx, Lx, sig, pred,
                                                              rtol_mult=rm))
                           if compute else None, attempt=attempt, **cfg)
        except RuntimeError as e:
            last = str(e)
            log(f"    visc failed (S={S:.0e} Pm={Pm:g} k={k:.8g} alpha={alpha} nx={nx} attempt "
                f"{attempt}): {last[:160]}")
            continue
        if U._pos_real(rec["lam"]):
            return rec
        log(f"    rejected {complex(rec['lam']):.4e} (S={S:.0e} Pm={Pm:g} k={k:.8g} "
            f"alpha={alpha})")
    raise RuntimeError(f"no sensitivity at S={S}, Pm={Pm}, k={k}, alpha={alpha}, nx={nx}: {last}")


# ============================================================ exponents from a record

def _g(rec, key="lam"):
    return complex(rec[key]).real


def q_Pm(rec):
    # d ln gamma/d ln Pm at fixed S = (nu/gamma) dgamma/dnu
    return float(rec["Pm"])/float(rec["S"])*_g(rec, "dlam_nu")/_g(rec)


def q_S(rec):
    # d ln gamma/d ln S at fixed Pm = -(eta gamma_eta + nu gamma_nu)/gamma
    nu, eta = diss_of(rec["S"], rec["Pm"])
    return -(eta*_g(rec, "dlam_eta") + nu*_g(rec, "dlam_nu"))/_g(rec)


def q_Snu(rec):
    # d ln gamma/d ln S at fixed nu = -(eta/gamma) gamma_eta
    return -_g(rec, "dlam_eta")/float(rec["S"])/_g(rec)


def q_k(rec):
    return R.k_exponent(rec)


def q_alpha(rec):
    return R.alpha_exponent(rec)


def nu_eta_ratio(rec):
    # gamma_nu/gamma_eta: -1/(5 Pm) for viscous FKR, -1/(2 Pm) viscous Coppi; a positive O(1)
    # value means nu acts like eta (both widen the layer)
    return _g(rec, "dlam_nu")/_g(rec, "dlam_eta")


# ============================================================ k_max

def implicit_slopes(S, Pm, alpha, kstar, gstar, rec0, rp, rm, h):
    # from the solve at k* (rec0) and at k*(1 +- h): central differences of the sensitivities in
    # k, implicit derivatives of k* (dk*/dp = -gamma_kp/gamma_kk) and the envelope ones of gamma*
    dk = 2*h*kstar
    d = lambda key: (_g(rp, key) - _g(rm, key))/dk
    gkk, gke, gkn, gka = d("dlam_k"), d("dlam_eta"), d("dlam_nu"), d("dlam_alpha")
    ge, gn, ga = _g(rec0, "dlam_eta"), _g(rec0, "dlam_nu"), _g(rec0, "dlam_alpha")
    nu, eta = diss_of(S, Pm)
    u = 1 - alpha*alpha
    out = dict(gkk=gkk, gketa=gke, gknu=gkn, gkalpha=gka,
               dk_deta=-gke/gkk, dk_dnu=-gkn/gkk, dk_dalpha=-gka/gkk,
               slope_k_S=(eta*gke + nu*gkn)/(kstar*gkk),      # d ln k*/d ln S |_Pm
               slope_g_S=-(eta*ge + nu*gn)/gstar,              # d ln gamma*/d ln S |_Pm
               slope_k_Pm=-nu*gkn/(kstar*gkk),                 # d ln k*/d ln Pm |_S
               slope_g_Pm=nu*gn/gstar,                         # d ln gamma*/d ln Pm |_S
               gkk_from_gamma=(_g(rp) - 2*_g(rec0) + _g(rm))/(h*kstar)**2)
    if alpha > 0:
        out["slope_k_u"] = (-gka/gkk)*u/(-2*alpha*kstar)
        out["slope_g_u"] = ga*u/(-2*alpha*gstar)
    else:
        out["slope_k_u"] = out["slope_g_u"] = np.nan
    return out


def kmax_solve(S, Pm, alpha, k0, g0, c0, nx, Lx, root=None, log=print, compute=True,
               deadline=np.inf, tag=None):
    # Newton (R.newton_kmax) at fixed (S, Pm, alpha, nx, Lx), the curvature pair, the polish
    # step and the grid check at nx/2. Returns a dict or raises RuntimeError
    def solve(k, pred):
        return point(S, Pm, k, alpha, nx, Lx, pred, root=root, log=log, compute=compute, tag=tag)
    its, ks, gs, why = R.newton_kmax(solve, k0, g0, c0, log=log, deadline=deadline)
    if why is not None:
        raise RuntimeError(f"Newton k_max at S={S}, Pm={Pm}, alpha={alpha}, nx={nx}: {why}")
    rec0 = its[-1]["rec"]
    rp = solve(ks*(1 + CURV_H), gs)
    rm = solve(ks*(1 - CURV_H), gs)
    sl = implicit_slopes(S, Pm, alpha, ks, gs, rec0, rp, rm, CURV_H)
    k_pol = its[-1]["k"] - its[-1]["gk"]/sl["gkk"]
    out = dict(S=S, Pm=Pm, alpha=alpha, nx=nx, Lx=Lx, kmax=ks, gmax=gs, k_polished=k_pol,
               iterations=len(its), its=[dict(k=i["k"], gamma=i["gamma"], gk=i["gk"],
                                              curv=i["curv"], step=i["step"]) for i in its],
               k0=float(k0), rec=rec0, rec_p=rp, rec_m=rm, **sl)
    try:
        rl = point(S, Pm, its[-1]["k"], alpha, nx//2, Lx, gs, root=root, log=log,
                   compute=compute, tag=tag)
        out.update(grid_dk=-(_g(rl, "dlam_k") - its[-1]["gk"])/sl["gkk"],
                   grid_dgamma=_g(rl) - _g(rec0), rec_lo=rl)
    except (RuntimeError, KeyError) as e:
        log(f"    grid check at nx/2 failed: {str(e)[:150]}")
        out.update(grid_dk=np.nan, grid_dgamma=np.nan)
    log(f"  k_max S={S:.0e} Pm={Pm:<6g} alpha={alpha:<5g} nx={nx:6d}: k* = {ks:.10f} gamma* = "
        f"{gs:.10e} ({len(its)} its; grid dk {out['grid_dk']:+.1e}) dlnk*/dlnPm = "
        f"{sl['slope_k_Pm']:+.5f} dlng*/dlnPm = {sl['slope_g_Pm']:+.5f} dlng*/dlnS = "
        f"{sl['slope_g_S']:+.5f} kappa = {float(rec0['kappa']):.2e}")
    return out


def curvature_theory(S, Pm, k, h=1e-3, root=None):
    g = [gamma_theory_visc(k*(1 + s*h), S, Pm, root) for s in (1, 0, -1)]
    return (g[0] - 2*g[1] + g[2])/(k*h)**2


def seed_theory(S, Pm, alpha, root=None):
    # (k, gamma, curvature) seed: the alpha = 0 viscous theory maximum, times rung 2's rough shear
    # factors (1 - alpha^2)^(-1/7), (1 - alpha^2)^(4/7); curvature rescaled as gamma/k^2
    kth, gth = theory_kmax_visc(S, Pm, root)
    u = 1 - alpha*alpha
    k0, g0 = kth*u**(-1/7), gth*u**(4/7)
    return k0, g0, curvature_theory(S, Pm, kth, root=root)*(g0/gth)*(kth/k0)**2


def _sigma_seed(S, Pm, k, alpha, Lx, g, compute):
    # the coarse dense spectrum's fastest real mode at k (sets sigma only, never a cache key)
    if not compute:
        return g
    top = _dense_others(S, Pm, float(k), alpha, Lx, NX_DENSE)[1]
    if U._pos_real(top) and (g is None or not np.isfinite(g) or 1/3 < top.real/g < 3):
        return top.real
    return g


def _fail(log, what, e):
    log(f"  {what}: FAILED {str(e)[:220]}")


def kmax_Pm(S, alpha, Pms=None, root=None, log=print, compute=True, deadline=np.inf):
    # k_max(Pm) at fixed (S, alpha) by continuation in ln Pm: seed k* (Pm'/Pm)^(d ln k*/d ln Pm),
    # gamma* likewise, curvature rescaled; the first Pm from the theory. ({Pm: result}, complete)
    Pms = Pms or PM_CHAIN
    Lx, nx = R.kmax_box(S)
    out, prev = {}, None
    for Pm in Pms:
        if time.time() > deadline:
            return out, False
        if prev is None:
            k0, g0, c0 = seed_theory(S, Pm, alpha, root)
            g0 = _sigma_seed(S, Pm, k0, alpha, Lx, g0, compute)
        else:
            r_ = np.log(Pm/prev["Pm"])
            k0 = R._trust(prev["kmax"]*np.exp(prev["slope_k_Pm"]*r_), prev["kmax"])
            g0 = prev["gmax"]*np.exp(prev["slope_g_Pm"]*r_)
            c0 = prev["gkk"]*(g0/prev["gmax"])*(prev["kmax"]/k0)**2
        try:
            r = kmax_solve(S, Pm, alpha, k0, g0, c0, nx, Lx, root, log, compute, deadline)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"k_max S={S:.0e} Pm={Pm} alpha={alpha}", e)
            continue
        out[Pm] = r
        prev = r
    return out, True


def _pms_upto(Pm):
    return tuple(p for p in PM_CHAIN if p <= Pm*(1 + 1e-12))


def kmax_alpha(S, Pm, alphas=None, root=None, log=print, compute=True, deadline=np.inf):
    # k_max(alpha) at fixed (S, Pm) by continuation in alpha (rung 2's seeds, R._trust). alpha = 0
    # is the alpha = 0 Pm chain's solution when Pm is on it (replayed, so shared), else a theory
    # seeded solve
    alphas = alphas or ALPHA_KA
    Lx, nx = R.kmax_box(S)
    out, prev = {}, None
    for al in alphas:
        if time.time() > deadline:
            return out, False
        r = None
        if prev is None and al == 0.0 and Pm in PM_CHAIN:
            r = kmax_Pm(S, 0.0, _pms_upto(Pm), root, _quiet if not compute else log,
                        compute, deadline)[0].get(Pm)
            if r is None:
                _fail(log, f"k_max S={S:.0e} Pm={Pm} alpha=0 (Pm chain)", RuntimeError("missing"))
                continue
        else:
            if prev is None:
                k0, g0, c0 = seed_theory(S, Pm, al, root)
                g0 = _sigma_seed(S, Pm, k0, al, Lx, g0, compute)
            else:
                dal = al - prev["alpha"]
                k0 = R._trust(prev["kmax"] + prev["dk_dalpha"]*dal, prev["kmax"])
                g0 = prev["gmax"] + _g(prev["rec"], "dlam_alpha")*dal
                g0 = g0 if g0 > 0 else prev["gmax"]
                c0 = prev["gkk"]*(g0/prev["gmax"])
            try:
                r = kmax_solve(S, Pm, al, k0, g0, c0, nx, Lx, root, log, compute, deadline)
            except (RuntimeError, KeyError) as e:
                _fail(log, f"k_max S={S:.0e} Pm={Pm} alpha={al}", e)
                continue
        out[al] = r
        prev = r
    return out, True


def _alphas_upto(alpha, alphas):
    return tuple(a for a in alphas if a <= alpha + 1e-12)


def kmax_grid2(S, Pm, alpha, root=None, log=print, compute=True, deadline=np.inf):
    # the same k_max solve at 2 nx, started from the nx solution (the nx-doubling check)
    Lx, nx = R.kmax_box(S)
    if alpha == 0.0:
        base = kmax_Pm(S, 0.0, _pms_upto(Pm), root, _quiet, compute=False)[0].get(Pm)
    else:
        al_list = ALPHA_KA if S in PM_KA else KA_HI.get(S, ((), ALPHA_KA))[1]
        base = kmax_alpha(S, Pm, _alphas_upto(alpha, al_list), root, _quiet,
                          compute=False)[0].get(alpha)
    if base is None:
        raise KeyError(f"grid2 needs the nx solution at S={S}, Pm={Pm}, alpha={alpha}")
    return kmax_solve(S, Pm, alpha, base["kmax"], base["gmax"], base["gkk"], 2*nx, Lx, root, log,
                      compute, deadline)


def small_alpha_c2(S, Pm, root=None, log=print, compute=True):
    # the onset of the shear dependence: gamma(k*(0), alpha) = gamma0 (1 - c2 alpha^2 + O(alpha^4))
    # (gamma is even in alpha), so c2(alpha) = -gamma_alpha/(2 alpha gamma) -> c2 + O(alpha^2),
    # Richardson-extrapolated over ALPHA_C2. At fixed k = k*(0): gamma_max(alpha) differs from
    # gamma(k*(0), alpha) only at O(alpha^4) (envelope), so c2 is also gamma_max's coefficient.
    # Pm = 0 takes rung 2's alpha = 0 solution (same box, same nx). Returns a dict
    Lx, nx = R.kmax_box(S)
    if Pm == 0.0:
        r0 = R.kmax_alpha(S, (0.0,), log=_quiet, compute=False)[0].get(0.0)
    else:
        r0 = kmax_Pm(S, 0.0, _pms_upto(Pm), root, _quiet, compute=False)[0].get(Pm)
    if r0 is None:
        raise KeyError(f"c2 needs the alpha = 0 k_max at S={S}, Pm={Pm}")
    c2 = []
    recs = []
    for al in ALPHA_C2:
        rec = point(S, Pm, r0["kmax"], al, nx, Lx, r0["gmax"], root=root, log=log,
                    compute=compute)
        recs.append(rec)
        c2.append(-_g(rec, "dlam_alpha")/(2*al*_g(rec)))
    a1, a2 = ALPHA_C2
    c2x = (a2*a2*c2[0] - a1*a1*c2[1])/(a2*a2 - a1*a1)
    out = dict(S=S, Pm=Pm, k=r0["kmax"], gamma0=r0["gmax"], c2_alpha=tuple(c2), c2=c2x,
               recs=recs)
    if compute:
        log(f"  c2 S={S:.0e} Pm={Pm:g}: k*(0) = {r0['kmax']:.6f}, c2({a1}) = {c2[0]:.6f}, "
            f"c2({a2}) = {c2[1]:.6f}, extrapolated {c2x:.6f}")
    return out


# ============================================================ sweeps, FD gate, branch scan

def dispersion(S, Pm, alpha, kas=None, root=None, log=print, compute=True, deadline=np.inf,
               levels=(1, 2)):
    # gamma(k) and its four sensitivities along a ka sweep at fixed (S, Pm, alpha), rung 1a's
    # per-k box (R.disp_box), levels = nx/level. Seeds: the theory at alpha = 0; with shear the
    # previous ka's gamma + gamma_k dk (first: the dense top). ({level: {ka: rec}}, complete)
    kas = kas or KA_DISP
    out = {lv: {} for lv in levels}
    prev = None
    for ka in kas:
        if time.time() > deadline:
            return out, False
        Lx, nx = R.disp_box(S, ka)
        pred = None
        if not compute:
            pred = 1.0                  # read-back: sigma is never part of a key
        elif alpha == 0.0:
            pred = gamma_theory_visc(ka, S, Pm, root)
        if (pred is None or not np.isfinite(pred)) and prev is not None:
            pred = _g(prev) + _g(prev, "dlam_k")*(ka - prev["k"])
            pred = max(pred, 0.3*_g(prev))
        if alpha != 0.0:
            pred = _sigma_seed(S, Pm, ka, alpha, Lx, pred, compute)
        if pred is None or not np.isfinite(pred):
            log(f"  disp S={S:.0e} Pm={Pm} alpha={alpha} ka={ka}: no seed")
            continue
        for lv in levels:
            try:
                rec = point(S, Pm, ka, alpha, nx//lv, Lx, pred, root=root, log=log,
                            compute=compute)
            except KeyError:
                log(f"  disp S={S:.0e} Pm={Pm} alpha={alpha} ka={ka} nx={nx//lv}: NOT COMPUTED")
                continue
            except RuntimeError as e:
                _fail(log, f"disp S={S:.0e} Pm={Pm} alpha={alpha} ka={ka} nx={nx//lv}", e)
                continue
            out[lv][ka] = rec
            if lv == levels[0]:
                prev = rec
                log(f"  disp S={S:.0e} Pm={Pm:<5g} alpha={alpha:<4g} ka={ka:<5g} nx={nx:6d} "
                    f"gamma={_g(rec):.8e} qPm={q_Pm(rec):+.5f} qS={q_S(rec):+.5f} "
                    f"qk={q_k(rec):+.5f} kappa={float(rec['kappa']):.2e} "
                    f"({float(rec['seconds']):.0f}s)")
    return out, True


def fd_gate(S, ka, alpha, Pm, root=None, log=print, compute=True, hs=FD_H):
    # central differences of eigenvalues in eta, nu (relative steps, through the diss override)
    # and in k (Ly), at the sensitivity's own (nx, Lx), against gamma_eta, gamma_nu, gamma_k
    Lx, nx = R.disp_box(S, ka)
    pred = 1.0                          # read-back: sigma is never part of a key
    if compute:
        pred = _sigma_seed(S, Pm, ka, alpha, Lx,
                           gamma_theory_visc(ka, S, Pm, root)*(1 - alpha*alpha)**0.55, compute)
    rec = point(S, Pm, ka, alpha, nx, Lx, pred, root=root, log=log, compute=compute)
    lam = complex(rec["lam"])
    nu0, eta0 = diss_of(S, Pm)
    rows = {"eta": [], "nu": [], "k": []}
    for which in ("eta", "nu", "k"):
        for h in hs:
            vals = []
            for sgn in (1, -1):
                kk = ka*(1 + sgn*h) if which == "k" else ka
                ee = eta0*(1 + sgn*h) if which == "eta" else eta0
                nn = nu0*(1 + sgn*h) if which == "nu" else nu0
                r = None
                for attempt, off in enumerate((0.03, 0.1, 0.5)):
                    sig = lam*(1 + off)
                    try:
                        r = U.cached("eigv", root or DATA_ROOT,
                                     (lambda kk=kk, ee=ee, nn=nn, sig=sig: eig_only(
                                         S, Pm, kk, alpha, nx, Lx, sig, lam_hint=lam, eta=ee,
                                         nu=nn)) if compute else None,
                                     S=S, Pm=Pm, k=float(kk), alpha=alpha, nx=nx, Lx=Lx,
                                     eta=float(ee), nu=float(nn), attempt=attempt)
                        break
                    except RuntimeError as e:
                        log(f"    FD eig failed ({which}, h={h}, attempt {attempt}): "
                            f"{str(e)[:120]}")
                if r is None:
                    raise RuntimeError(f"FD eigenvalue ({which}, h={h}) failed at every shift")
                vals.append(complex(r["lam"]))
            step = 2*h*{"k": ka, "eta": eta0, "nu": nu0}[which]
            fd = (vals[0] - vals[1])/step
            ref = complex(rec["dlam_" + which])
            rows[which].append((h, fd, abs(fd - ref)))
            log(f"  FD S={S:.0e} ka={ka} alpha={alpha} Pm={Pm} d/d{which} h={h:g}: FD "
                f"{fd.real:+.10e}  sens {ref.real:+.10e}  |diff| {rows[which][-1][2]:.2e}")
    return rec, rows


def branch_scan(S, Pm, alpha, kas=None, root=None, log=print, compute=True, nx=NX_DENSE):
    # the fastest real growing eigenvalue of eig_dense on the k_max box at a coarse nx, over a k
    # grid: shows whether gamma(k) has one maximum or two (rung 2's low-k branch), or none.
    # Pm = 0 too. Returns (ks, gamma (nan where the top is not a real growing mode), the
    # second-largest real part, npos, the top's real part whatever it is)
    kas = kas or KA_BRANCH
    Lx = R.kmax_box(S)[0]
    rows = []
    for ka in kas:
        def comp(ka=ka):
            others, top, npos = _dense_others(S, Pm, float(ka), alpha, Lx, nx)
            re = np.sort(np.concatenate([[top], others]).real)[::-1]
            return dict(top=top, second=float(re[1]), npos=npos)
        rec = U.cached("scanv", root or DATA_ROOT, comp if compute else None, S=S, Pm=Pm,
                       k=float(ka), alpha=alpha, nx=nx, Lx=Lx)
        top = complex(rec["top"])
        rows.append((float(ka), top.real if U._pos_real(top) else np.nan, float(rec["second"]),
                     int(rec["npos"]), top.real))
    a = np.array(rows)
    if compute:
        g = a[:, 1]
        i = np.nanargmax(g) if np.isfinite(g).any() else -1
        log(f"  branch scan S={S:.0e} alpha={alpha} Pm={Pm}: global max on the grid at ka = "
            f"{a[i, 0] if i >= 0 else np.nan:.4g}, local maxima at {local_maxima(a[:, 0], g)}")
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(int), a[:, 4]


def local_maxima(ks, g):
    # interior grid maxima of g(k) (the endpoints count when they are the larger neighbour)
    ks, g = np.asarray(ks), np.asarray(g)
    out = []
    for i in range(len(g)):
        if not np.isfinite(g[i]):
            continue
        lo = g[i - 1] if i > 0 and np.isfinite(g[i - 1]) else -np.inf
        hi = g[i + 1] if i < len(g) - 1 and np.isfinite(g[i + 1]) else -np.inf
        if g[i] > lo and g[i] > hi:
            out.append(float(ks[i]))
    return out


def _top_real(vals):
    # the largest real part among the numerically REAL eigenvalues (complex pairs excluded)
    vals = np.asarray(vals)
    real = vals[np.abs(vals.imag) <= 1e-6*np.abs(vals) + 1e-12]
    return float(real.real.max()) if real.size else -np.inf


def max_real_growth(S, Pm, alpha, nx=NX_DENSE, root=None, compute=True, kas=None):
    # max over k of the largest REAL eigenvalue of the dense block on the k_max box (coarse KA_AC
    # grid, then bounded Brent in ln k around the best point), cached (kind "acv"). Its sign is
    # what the alpha_c bisection reads: below the boundary the maximum is the (decaying) top of
    # the null cluster, ~ -nu k^2, never 0
    from scipy.optimize import minimize_scalar
    kas = kas or KA_AC
    Lx = R.kmax_box(S)[0]

    def comp():
        f = lambda ka: _top_real(np.concatenate([[_dense_others(S, Pm, float(ka), alpha, Lx, nx)[1]],
                                                 _dense_others(S, Pm, float(ka), alpha, Lx, nx)[0]]))
        g = [f(k) for k in kas]
        i = int(np.argmax(g))
        lo, hi = kas[max(i - 1, 0)], kas[min(i + 1, len(kas) - 1)]
        r = minimize_scalar(lambda lk: -f(np.exp(lk)), bounds=(np.log(lo), np.log(hi)),
                            method="bounded", options=dict(xatol=2e-3))
        gm, km = (-float(r.fun), float(np.exp(r.x))) if -r.fun >= g[i] else (g[i], float(kas[i]))
        return dict(gmax=gm, kmax=km, coarse=np.array(g))
    return U.cached("acv", root or DATA_ROOT, comp if compute else None, S=S, Pm=Pm,
                    alpha=float(alpha), nx=nx, Lx=Lx, kas=list(map(float, kas)))


def alpha_c_bisect(S, Pm, a_grow, a_stable, nbis=None, root=None, log=print, compute=True,
                   nx=NX_DENSE):
    # the viscous stability boundary: bisection in alpha on the sign of max_real_growth, from a
    # bracket (growing, stable); alpha_c_lin extrapolates the two growing evaluations nearest the
    # boundary linearly to gamma = 0 (gamma_max(alpha) is ~linear there). The bracket ends are
    # re-evaluated at 2 nx (grid check of the signs). Returns a dict; RuntimeError on a bad bracket
    nbis = AC_NBIS if nbis is None else nbis
    G = lambda a, n=nx: float(max_real_growth(S, Pm, a, n, root, compute)["gmax"])
    lo, hi = float(a_grow), float(a_stable)
    ev = {lo: G(lo), hi: G(hi)}
    if not (ev[lo] > 0 >= ev[hi]):
        raise RuntimeError(f"alpha_c bracket at S={S}, Pm={Pm}: G({lo}) = {ev[lo]:.3e}, "
                           f"G({hi}) = {ev[hi]:.3e}")
    for _ in range(nbis):
        mid = 0.5*(lo + hi)
        ev[mid] = G(mid)
        if ev[mid] > 0:
            lo = mid
        else:
            hi = mid
    grow = sorted(a for a in ev if ev[a] > 0)[-2:]
    lin = np.nan
    if len(grow) == 2:
        (a1, a2), (g1, g2) = grow, (ev[grow[0]], ev[grow[1]])
        lin = a2 - g2*(a2 - a1)/(g2 - g1)
    out = dict(S=S, Pm=Pm, lo=lo, hi=hi, alpha_c=0.5*(lo + hi), alpha_c_lin=float(lin),
               evals=dict(sorted(ev.items())), g_lo_2nx=G(lo, 2*nx), g_hi_2nx=G(hi, 2*nx))
    if compute:
        log(f"  alpha_c S={S:.0e} Pm={Pm:g} (nu = {Pm/S:g}): [{lo:.5f}, {hi:.5f}], linear "
            f"{lin:.5f}, 1 - alpha_c^2 = {1 - out['alpha_c']**2:.4f}; at 2 nx G(lo) = "
            f"{out['g_lo_2nx']:+.3e}, G(hi) = {out['g_hi_2nx']:+.3e}")
    return out


# ============================================================ make_data

PHASES = ("theory", "fd", "disp", "kmax_Pm", "kmax_alpha", "branch", "kmax_grid2", "c2", "alpha_c")


def _sel(vals, flt):
    return [v for v in vals if not flt or v in flt]


def make_data(root=None, wall_budget=3*3600.0, phases=None, log=print, S_list=None,
              P_list=None, A_list=None):
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    done = True
    for name in phases or PHASES:
        t0 = time.time()
        ok = True
        if name == "theory":
            build_theory(root, log)
        elif name == "fd":
            for pt in FD_POINTS:
                if (S_list and pt[0] not in S_list) or (P_list and pt[3] not in P_list):
                    continue
                try:
                    fd_gate(*pt, root=root, log=log)
                except RuntimeError as e:
                    _fail(log, f"FD {pt}", e)
        elif name == "disp":
            for S in _sel(S_DISP, S_list):
                for Pm in _sel(PM_DISP, P_list):
                    if not A_list or 0.0 in A_list:
                        ok &= dispersion(S, Pm, 0.0, root=root, log=log, deadline=deadline)[1]
            for (S, al) in DISP_SHEAR:
                if (S_list and S not in S_list) or (A_list and al not in A_list):
                    continue
                for Pm in _sel(PM_DISP, P_list):
                    ok &= dispersion(S, Pm, al, root=root, log=log, deadline=deadline,
                                     levels=(1,))[1]
        elif name == "kmax_Pm":
            for S in _sel(S_CHAIN, S_list):
                if not A_list or 0.0 in A_list:
                    ok &= kmax_Pm(S, 0.0, root=root, log=log, deadline=deadline)[1]
            for (S, al) in CHAIN_SHEAR:
                if (S_list and S not in S_list) or (A_list and al not in A_list):
                    continue
                ok &= kmax_Pm(S, al, root=root, log=log, deadline=deadline)[1]
        elif name == "kmax_alpha":
            for S in _sel(PM_KA, S_list):
                for Pm in _sel(PM_KA[S], P_list):
                    ok &= kmax_alpha(S, Pm, root=root, log=log, deadline=deadline)[1]
            for S, (pms, als) in KA_HI.items():
                if S_list and S not in S_list:
                    continue
                for Pm in _sel(pms, P_list):
                    ok &= kmax_alpha(S, Pm, als, root=root, log=log, deadline=deadline)[1]
        elif name == "branch":
            for (S, al, pms) in BRANCH:
                if (S_list and S not in S_list) or (A_list and al not in A_list):
                    continue
                for Pm in _sel(pms, P_list):
                    branch_scan(S, Pm, al, root=root, log=log)
        elif name == "c2":
            for S in _sel(S_CHAIN, S_list):
                for Pm in _sel(PM_C2, P_list):
                    try:
                        small_alpha_c2(S, Pm, root, log)
                    except (RuntimeError, KeyError) as e:
                        _fail(log, f"c2 S={S:.0e} Pm={Pm}", e)
        elif name == "alpha_c":
            for (S, Pm, a1, a2) in ALPHA_C_POINTS:
                if (S_list and S not in S_list) or (P_list and Pm not in P_list):
                    continue
                try:
                    alpha_c_bisect(S, Pm, a1, a2, root=root, log=log)
                except (RuntimeError, KeyError) as e:
                    _fail(log, f"alpha_c S={S:.0e} Pm={Pm}", e)
        elif name == "kmax_grid2":
            for (S, Pm, al) in GRID2_POINTS:
                if S_list and S not in S_list:
                    continue
                try:
                    kmax_grid2(S, Pm, al, root, log, deadline=deadline)
                except (RuntimeError, KeyError) as e:
                    _fail(log, f"grid2 S={S:.0e} Pm={Pm} alpha={al}", e)
        else:
            raise ValueError(name)
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")
        done &= ok
    return bool(done)


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def disp_results(S, Pm, alpha=0.0, root=None):
    levels = (1, 2) if alpha == 0.0 else (1,)
    return dispersion(S, Pm, alpha, root=root or DATA_ROOT, log=_quiet, compute=False,
                      levels=levels)[0]


def kmax_Pm_results(S, alpha=0.0, root=None):
    return kmax_Pm(S, alpha, root=root or DATA_ROOT, log=_quiet, compute=False)[0]


def kmax_alpha_results(S, Pm, root=None):
    als = KA_HI[S][1] if S in KA_HI and S not in PM_KA else ALPHA_KA
    return kmax_alpha(S, Pm, als, root=root or DATA_ROOT, log=_quiet, compute=False)[0]


def grid2_results(root=None):
    out = {}
    for (S, Pm, al) in GRID2_POINTS:
        try:
            out[(S, Pm, al)] = kmax_grid2(S, Pm, al, root or DATA_ROOT, _quiet, compute=False)
        except (RuntimeError, KeyError):
            pass
    return out


def fd_results(root=None):
    out = {}
    for pt in FD_POINTS:
        try:
            out[pt] = fd_gate(*pt, root=root or DATA_ROOT, log=_quiet, compute=False)
        except (RuntimeError, KeyError):
            pass
    return out


def c2_results(root=None):
    out = {}
    for S in S_CHAIN:
        for Pm in PM_C2:
            try:
                out[(S, Pm)] = small_alpha_c2(S, Pm, root or DATA_ROOT, _quiet, compute=False)
            except (RuntimeError, KeyError):
                pass
    return out


def alpha_c_results(root=None):
    out = {}
    for (S, Pm, a1, a2) in ALPHA_C_POINTS:
        try:
            out[(S, Pm)] = alpha_c_bisect(S, Pm, a1, a2, root=root or DATA_ROOT, log=_quiet,
                                          compute=False)
        except (RuntimeError, KeyError):
            pass
    return out


def branch_results(root=None):
    out = {}
    for (S, al, pms) in BRANCH:
        for Pm in pms:
            try:
                out[(S, al, Pm)] = branch_scan(S, Pm, al, root=root or DATA_ROOT, log=_quiet,
                                               compute=False)
            except (RuntimeError, KeyError):
                pass
    return out


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3*3600.0
    args = sys.argv[2:]
    flt = {"S": None, "P": None, "A": None}
    while args and args[0][:2] in ("S=", "P=", "A="):
        flt[args[0][0]] = tuple(float(s) for s in args[0][2:].split(","))
        args = args[1:]

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    log(f"profile {PROFILE}, root {DATA_ROOT}")
    make_data(wall_budget=budget, phases=tuple(args) or None, log=log, S_list=flt["S"],
              P_list=flt["P"], A_list=flt["A"])
