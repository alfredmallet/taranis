# Parameter sensitivities, k_max by Newton, and resumable data generation for
# examples/tearing-kmax.ipynb: plans/AUTODIFF_PLAN.md rung 2 (science) -- d lambda/d eta and
# d lambda/d k of resistive tearing with field-aligned shear flow, through the overrides seam
# (taranis/overrides.py; stability.dj_operator(..., dparams=, overrides=)).
#
# Setup exactly as rungs 1b-ref / 1b (tearing_shear_run.py, tearing_sensitivity_run.py): dims=2,
# nu = 0 (diss = (0, eta)), hyper = 1, a = v_Ay = 1, S = 1/eta, fp64; Psi0 = sech^2 x (+ its two
# nearest periodic images), Phi0 = alpha Psi0; the ky block iky = 1 (ny = 8).
#
# THE WAVENUMBER IS AN OVERRIDE. Every Parameters here has Ly = 2 pi (k = 1, a placeholder); the
# block's wavenumber k enters as overrides={"Ly": 2 pi/k}, so one Parameters (one compile) serves
# every k of a (S, nx, Lx) -- and a Newton solve moves k with the box (Lx, nx) FIXED (below).
# J at an overridden Ly is J of a Parameters built with that Ly (0.0 difference at fp64: rung-2
# gate, and the notebook re-checks it).
#
# THREE SENSITIVITIES PER SOLVE, one right/left pair (w^H dJ v / w^H v, eigenvalue_sensitivity):
#   eta   dparams={"diss": (0.0, 1.0)}: diss is the (nu, eta) pair, so this tangent moves eta
#         ONLY (nu stays 0). S-exponent q_S = d ln gamma/d ln S = -(eta/gamma) dgamma/deta.
#   k     dparams={"Ly": -Ly/k} = -2 pi/k^2 (a block is an INDEX: ky = 2 pi/Ly, so dLy/dk =
#         -2 pi/k^2 = -Ly/k). k-exponent q_k = d ln gamma/d ln k = (k/gamma) dgamma/dk.
#   alpha dx0 = (Psi0, 0) (rung 1b's state direction), at the overridden kgrid.
#
# THEORY (alpha = 0, independent of taranis; rung 1a's exact inner-layer relation, Boldyrev &
# Loureiro 2018 eq. 60 in tearing_eigen_run): Delta'(k) delta_eta = F(Lam), gamma = Lam eta^(1/3)
# (k|f'|)^(2/3). Differentiated ANALYTICALLY (digamma), with m(Lam) = d ln F/d ln Lam =
# 5/4 + (3/8) b [psi((b-1)/4) - psi((b+5)/4)], b = Lam^(3/2):
#   d ln gamma/d ln S = -1/3 - 1/(3 m)      (FKR m = 5/4: -3/5; Coppi m -> inf: -1/3)
#   d ln gamma/d ln k = 2/3 + (d ln Delta'/d ln k - 1/3)/m
# (the notebook checks both against central differences of gamma_theory). k_max(S): the root of
# the second. The paper (Mallet et al. JPP 2025 eqs. 5.11-5.14, 8.1), alpha close to 1: shear
# constant-psi gamma ~ (1-alpha^2)^(1/2) S^(-1/2) (ka)^(-1/2) (n = 2) -> q_S = -1/2, q_k = -1/2;
# shear nonconstant-psi (1-alpha^2)^(2/3) (ka)^(2/3) S^(-1/3) -> q_S = -1/3, q_k = 2/3; their
# crossing k_tr a ~ (alpha/(1-alpha^2))^(1/7) S^(-1/7) is k_max, gamma_max ~ alpha^(3/7)
# (1-alpha^2)^(4/7) S^(-3/7). For THIS n = 2 sheet the alpha = 0 exponents are the same -1/7,
# -3/7 (FKR/Coppi crossing), so the S-scalings do not distinguish the regimes; the shear regime
# needs 1 - alpha^2 << alpha AND delta_in/a = (alpha/(1-alpha^2))^(1/3)(kaS)^(-1/3) (eq. 5.15)
# << 1, which the paper reaches at S = 1e8-1e16 (its fig 4, 1 - alpha^2 = 0.01).
#
# K_MAX BY NEWTON ON dgamma/dk = 0, the curvature by a SECANT on the iterates' own dgamma/dk (one
# eigen-solve per iterate; superlinear, order 1.618, against 3 solves per iterate for a
# central-difference d2gamma/dk2), seeded by a curvature estimate (theory at alpha = 0, the
# previous solution in a continuation). Converged when the step is below NEWTON_TOL k. Then TWO
# more solves at k*(1 +- CURV_H) give the central differences
#   gamma_kk, gamma_k eta, gamma_k alpha   (O(CURV_H^2))
# and by implicit differentiation of dgamma/dk(k*(p), p) = 0:
#   d ln k*/d ln S = eta gamma_keta/(k* gamma_kk),   d ln gamma*/d ln S = -(eta/gamma*) gamma_eta
#   (envelope theorem: d gamma_max/dp = dgamma/dp at k*), and the same in alpha.
# BOX RULE (one per S, shared by every alpha and by both k_max families, so k_max(alpha) and
# k_max(S) are evaluations of ONE discrete operator family per S): Lx = BOX_C/k_th(S) with
# k_th(S) the alpha = 0 theory maximum (k_max(alpha) >= 0.85 k_th everywhere measured, so
# Lx >= 17/k); nx from rung 1a's converged ladders (dx needed at alpha = 0 and 1.5 k_th, the
# finest the family needs: the flow widens the layer). Grid check: every k* is re-evaluated at
# nx/2, and its k_max shift is -dgamma/dk(nx/2, k*)/gamma_kk; a subset is re-solved at 2 nx.
#
# `python examples/tearing_rung2_run.py [wall_seconds] [S=...] [A=...] [phase ...]` runs
# make_data (resumable; every solve is one cache file under DATA_ROOT/solves). The profile
# (TARANIS_R2_PROFILE=smoke|full, default full) picks the configuration lists; smoke writes to
# DATA_ROOT_SMOKE.
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import functools
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma

import taranis as jr
from taranis import stability as st

import tearing_eigen_run as T
import tearing_sensitivity_run as U

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FULL = os.path.join(HERE, "data", "tearing-rung2")
DATA_ROOT_SMOKE = os.path.join(HERE, "data", "tearing-rung2-smoke")
PROFILE = os.environ.get("TARANIS_R2_PROFILE", "full")
if PROFILE not in ("full", "smoke"):
    raise ValueError(f"TARANIS_R2_PROFILE must be 'full' or 'smoke', got {PROFILE!r}")
DATA_ROOT = DATA_ROOT_SMOKE if PROFILE == "smoke" else DATA_ROOT_FULL

TWO_PI = 2*np.pi
ETA_DIR = (0.0, 1.0)          # d/d eta of diss = (nu, eta): nu does NOT move
BOX_C = 20.0
NEWTON_TOL = 1e-7             # |step| <= NEWTON_TOL k*
NEWTON_MAXIT = 16            # a switch of branch (the local maximum followed by continuation
                             # ceases to exist, e.g. alpha = 0.995 S = 5e4) walks ~10 capped steps
MAX_STEP = 0.25               # |step| <= MAX_STEP k (a trust region for the first steps)
SEED_TRUST = 2.0              # continuation seeds within [k*/SEED_TRUST, SEED_TRUST k*]
CURV_H = 0.02                 # relative k step of the curvature/mixed central differences
NX_DENSE = 1024              # the gap spectrum's coarse grid (rung 1b: 1024 for every box, Lx <= 150)

# ---- configuration (full profile; smoke below)
S_DISP = (1e3, 1e4, 1e5)
ALPHA_DISP = (0.0, 0.3, 0.6, 0.8)
KA_DISP = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.2)
S_KA = (1e3, 1e4, 1e5)                                  # k_max(alpha) at these S
ALPHA_KMAX = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99,
              0.995)
S_KS = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5)              # k_max(S) at these alpha
ALPHA_KS = (0.0, 0.8, 0.95, 0.995)
S_KS_HI = (2e5, 5e5, 1e6)                               # optional (phase kmax_S_hi), alpha = 0
GRID2_POINTS = ((1e4, 0.0), (1e5, 0.0), (1e5, 0.8), (1e5, 0.99))    # full Newton at 2 nx
FD_POINTS = ((1e3, 0.5, 0.0), (1e4, 0.2, 0.6), (1e4, 1.5, 0.3), (1e5, 0.5, 0.8))
FD_H = (1e-2, 3e-3, 1e-3)     # relative steps in eta and in k
if PROFILE == "smoke":
    S_DISP, ALPHA_DISP, KA_DISP = (1e3,), (0.0, 0.8), (0.2, 0.5, 1.0, 2.0)
    S_KA, ALPHA_KMAX = (1e3,), (0.0, 0.5, 0.8)
    S_KS, ALPHA_KS, S_KS_HI = (1e3, 2e3), (0.0,), ()
    GRID2_POINTS = ((1e3, 0.0),)
    FD_POINTS = ((1e3, 0.5, 0.0),)

PAPER = dict(fkr=-3/5, coppi=-1/3, shear_const=-1/2, shear_nonconst=-1/3, gmax=-3/7,
             kmax=-1/7, k_fkr=-6/5, k_coppi=2/3, k_shear_const=-1/2)


# ============================================================ theory (alpha = 0, analytic)

def dlnF_dlnLam(lam):
    # m(Lam) = d ln F/d ln Lam of T.inner_F: 5/4 + (3/8) b [digamma((b-1)/4) - digamma((b+5)/4)]
    lam = np.asarray(lam, dtype=float)
    b = lam**1.5
    return 1.25 + 0.375*b*(digamma((b - 1.0)/4.0) - digamma((b + 5.0)/4.0))


def dlnDprime_dlnk(ka):
    # d ln Delta'/d ln k of T.dprime_sech2 (q = k^2): 2 [-q/(5-q) + q/(q+3) - 1 - q/(2(q+4))]
    q = np.asarray(ka, dtype=float)**2
    return 2.0*(-q/(5.0 - q) + q/(q + 3.0) - 1.0 - q/(2.0*(q + 4.0)))


def _lam_th(ka, S):
    Dp = float(T.dprime_sech2(ka))
    return T.lam_of(Dp*((1.0/S)/(ka*T.FPRIME))**(1.0/3.0))


def theory_s_exponent(ka, S):
    # d ln gamma/d ln S of the exact relation at fixed k (analytic)
    return float(-1.0/3.0 - 1.0/(3.0*dlnF_dlnLam(_lam_th(ka, S))))


def theory_k_exponent(ka, S):
    # d ln gamma/d ln k of the exact relation at fixed S (analytic)
    return float(2.0/3.0 + (dlnDprime_dlnk(ka) - 1.0/3.0)/dlnF_dlnLam(_lam_th(ka, S)))


@functools.lru_cache(maxsize=256)
def theory_kmax(S):
    # (k_max, gamma_max) of the exact relation: the root of d ln gamma/d ln k
    k = brentq(lambda lk: theory_k_exponent(np.exp(lk), S), np.log(1e-3), np.log(2.2),
               xtol=1e-14, rtol=1e-14)
    k = float(np.exp(k))
    return k, float(T.gamma_theory(k, S))


def theory_kmax_slopes(S, h=1e-3):
    # (d ln k_max/d ln S, d ln gamma_max/d ln S) of the exact relation: the first by a central
    # difference of the root in ln S, the second analytic (envelope theorem: q_S at k_max)
    kp, km = theory_kmax(S*np.exp(h))[0], theory_kmax(S*np.exp(-h))[0]
    k0 = theory_kmax(S)[0]
    return float((np.log(kp) - np.log(km))/(2*h)), theory_s_exponent(k0, S)


# ============================================================ exponents from a record

def s_exponent(rec):
    # d ln gamma/d ln S = -(eta/gamma) dgamma/deta
    S = float(rec["S"])
    return -(1.0/S)*complex(rec["dlam_eta"]).real/complex(rec["lam"]).real


def k_exponent(rec):
    # d ln gamma/d ln k = (k/gamma) dgamma/dk
    return float(rec["k"])*complex(rec["dlam_k"]).real/complex(rec["lam"]).real


def alpha_exponent(rec):
    # d ln gamma/d ln(1 - alpha^2) (rung 1b's running exponent)
    return U.exponent({"alpha": rec["alpha"], "lam": rec["lam"], "dlam": rec["dlam_alpha"]})


def delta_in(S, ka, alpha):
    # the paper's shear-layer width (eq. 5.15, prefactor 1): (alpha/(1-alpha^2))^(1/3) (kaS)^(-1/3)
    return (alpha/(1.0 - alpha*alpha))**(1.0/3.0)*(ka*S)**(-1.0/3.0)


# ============================================================ grids

@functools.lru_cache(maxsize=1)
def _dx_table():
    # rung 1a's converged ladders (alpha = 0): dx = Lx/nx of the PREVIOUS-to-last rung, which
    # already agreed with the last (dx/2) to 1e-7 (rung 1b's nx_hi rule), on a (ln S, ln ka) grid
    tab = U.eigen_tables()
    kas = sorted(tab)
    Ss = (1e3, 1e4, 1e5, 1e6)
    dx = np.array([[2*T.lx_for(ka)/tab[ka][S]["nx"][-1] for ka in kas] for S in Ss])
    return np.log(Ss), np.log(kas), np.log(dx)


def dx_needed(S, ka):
    # bilinear interpolation of ln dx in (ln S, ln ka), clamped to the table
    lS, lk, ld = _dx_table()
    x = float(np.clip(np.log(S), lS[0], lS[-1]))
    y = float(np.clip(np.log(ka), lk[0], lk[-1]))
    i = min(np.searchsorted(lS, x, side="right") - 1, len(lS) - 2)
    j = min(np.searchsorted(lk, y, side="right") - 1, len(lk) - 2)
    tx, ty = (x - lS[i])/(lS[i + 1] - lS[i]), (y - lk[j])/(lk[j + 1] - lk[j])
    v = (1 - tx)*(1 - ty)*ld[i, j] + tx*(1 - ty)*ld[i + 1, j] + (1 - tx)*ty*ld[i, j + 1] \
        + tx*ty*ld[i + 1, j + 1]
    return float(np.exp(v))


def _pow2ceil(n):
    return int(2**int(np.ceil(np.log2(n) - 1e-9)))


def nx_rule(S, ka, Lx):
    # 10 % slack before the next power of 2: dx_needed is already a converged rung (1e-7), and
    # every point is grid-checked at nx/2
    return max(1024, _pow2ceil(0.9*Lx/dx_needed(S, ka)))


def kmax_box(S):
    # the k_max families' box at this S: Lx = BOX_C/k_th(S), nx for alpha = 0 at 1.5 k_th
    kth = theory_kmax(S)[0]
    Lx = float(np.round(BOX_C/kth, 6))
    return Lx, nx_rule(S, 1.5*kth, Lx)


def disp_box(S, ka):
    # the dispersion sweeps' box: rung 1a's per-k box Lx = max(15/k, 20), nx from its ladders
    Lx = T.lx_for(ka)
    return Lx, nx_rule(S, ka, Lx)


# ============================================================ one solve

@functools.lru_cache(maxsize=64)
def make_params(S, nx, Lx):
    # Ly = 2 pi is a placeholder: the block's k is ALWAYS the override Ly = 2 pi/k. Cached, so
    # one Parameters (one set of compiled operators) serves every k and alpha of a (S, nx, Lx)
    return jr.Parameters(nx=int(nx), ny=8, Lx=float(Lx), Ly=TWO_PI, dims=2, cfl_safety=0.5,
                         forcing=False, eqpars={"diss": (0.0, 1.0/S), "hyper": 1})


def k_override(k):
    return {"Ly": TWO_PI/float(k)}


def k_direction(k):
    # d/dk of the block's wavenumber: ky = 2 pi/Ly, dLy/dk = -Ly/k (CLAUDE.md rung-2 rule)
    Ly = TWO_PI/float(k)
    return {"Ly": -Ly/float(k)}


@functools.lru_cache(maxsize=16)
def _kgrid(S, nx, Lx):
    return jr.setup_kgrids(make_params(S, nx, Lx))


@functools.lru_cache(maxsize=256)
def _dense_others(S, k, alpha, Lx, nx):
    # eig_dense of the block on a coarse grid: every eigenvalue except the fastest (the gap
    # spectrum, as rung 1b: the nearest other eigenvalue is the nu = 0 null cluster), the
    # fastest, and the number growing
    params = make_params(S, nx, Lx)
    x0, _ = U.equilibrium(params, alpha)
    B, _ = st.ky_block_matrix(x0, _kgrid(S, nx, Lx), params, 1, overrides=k_override(k))
    r = st.eig_dense(B)
    return r.values[1:], complex(r.values[0]), int(np.sum(r.values.real > 1e-10*r.scale))


def _operators(S, k, alpha, nx, Lx):
    params = make_params(S, nx, Lx)
    kg = _kgrid(S, nx, Lx)
    x0, dx0 = U.equilibrium(params, alpha)
    ov = k_override(k)
    op, idx = st.ky_block_operator(x0, kg, params, 1, overrides=ov)
    return params, kg, x0, dx0, ov, op, idx


def sens_solve(S, k, alpha, nx, Lx, sigma, lam_hint=None, rtol_mult=1.0, dense_nx=None):
    # one right/left pair and the three sensitivities (eta, k, alpha); raises RuntimeError on any
    # harness refusal (cached as a failure by the caller)
    t0 = time.time()
    params, kg, x0, dx0, ov, op, idx = _operators(S, k, alpha, nx, Lx)
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
    others, lam_dense, npos = _dense_others(S, float(k), alpha, Lx, dnx)
    spectrum = np.concatenate([[lam], others])
    d_eta, _ = st.dj_operator(x0, None, kg, params, iky=1, dparams={"diss": ETA_DIR},
                              overrides=ov)
    d_k, _ = st.dj_operator(x0, None, kg, params, iky=1, dparams=k_direction(k), overrides=ov)
    d_al, _ = st.dj_operator(x0, dx0, kg, params, iky=1, overrides=ov)
    s_eta = st.eigenvalue_sensitivity(op, d_eta, lam, v, w, spectrum=spectrum)
    s_k = st.eigenvalue_sensitivity(op, d_k, lam, v, w, spectrum=spectrum)
    s_al = st.eigenvalue_sensitivity(op, d_al, lam, v, w, spectrum=spectrum)
    x, psi, phi, d2 = T._eigfun(params, idx, v, params.nx//2)
    prof = T._profile(x, psi, phi, d2, params.nx//2, Lx, half=1.5)
    return dict(S=S, k=float(k), alpha=alpha, nx=nx, Lx=Lx, sigma=complex(sigma),
                rtol_mult=rtol_mult, lam=lam, dlam_eta=s_eta.dlam, dlam_k=s_k.dlam,
                dlam_alpha=s_al.dlam, kappa=s_eta.kappa, wv=s_eta.wv, gap=s_eta.gap,
                rel_err=s_eta.rel_err, res_r=s_eta.residual_right, res_l=s_eta.residual_left,
                scale=s_eta.scale, lam_dense=lam_dense, npos_dense=npos, dense_nx=dnx,
                width=T.width_quarter(prof["xn"], prof["d2n"]), gmres_rtol=grtol,
                right_iters=right.stats["gmres_iters"], left_iters=left.stats["gmres_iters"],
                t_build=tb, t_right=tr - tb, t_left=tl - tr, seconds=time.time() - t0)


def eig_only(S, k, alpha, nx, Lx, sigma, lam_hint=None, eta=None):
    # the eigenvalue alone (FD gates), same preconditioned shift-invert; eta moves the (nu, eta)
    # pair's eta through overrides, nu stays 0
    params = make_params(S, nx, Lx)
    kg = _kgrid(S, nx, Lx)
    x0, _ = U.equilibrium(params, alpha)
    ov = k_override(k)
    if eta is not None:
        ov["diss"] = (0.0, float(eta))
    op, idx = st.ky_block_operator(x0, kg, params, 1, overrides=ov)
    A = U.band_matrix(op, idx, nx, T.band_width("sech2", Lx))
    M, _ = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
    grtol = float(np.clip(300*np.finfo(float).eps*normA/max(dist, 1e-300), 1e-10, 1e-6))
    r = st.shift_invert(op, sigma, k=1, M=M, tol=1e-12, gmres_rtol=grtol, restart=30, maxiter=20)
    return dict(lam=complex(r.values[0]), res=float(r.residuals[0]))


def point(S, k, alpha, nx, Lx, pred, root=None, log=print, compute=True, tag=None):
    # a cached sensitivity solve (kind "sens2"), retried at moved shifts / looser inner solves
    # when the harness refuses or the eigenvalue found is not a real growing mode. pred: the
    # expected gamma (sigma = 1.03 pred). The key is the configuration, NOT sigma.
    cfg = dict(S=S, k=float(k), alpha=alpha, nx=nx, Lx=Lx)
    if tag:
        cfg["tag"] = tag
    last = None
    for attempt, (off, rm) in enumerate(((0.03, 1.0), (0.13, 1.0), (0.5, 1.0), (-0.03, 1.0),
                                         (0.03, 100.0), (0.13, 100.0), (0.03, 1e4))):
        sig = pred*(1.0 + off)
        try:
            rec = U.cached("sens2", root or DATA_ROOT,
                           (lambda sig=sig, rm=rm: sens_solve(S, k, alpha, nx, Lx, sig, pred,
                                                              rtol_mult=rm))
                           if compute else None, attempt=attempt, **cfg)
        except RuntimeError as e:
            last = str(e)
            log(f"    sens2 failed (S={S:.0e} k={k:.8g} alpha={alpha} nx={nx} attempt "
                f"{attempt}): {last[:160]}")
            continue
        if U._pos_real(rec["lam"]):
            return rec
        log(f"    rejected {complex(rec['lam']):.4e} (S={S:.0e} k={k:.8g} alpha={alpha})")
    raise RuntimeError(f"no sensitivity at S={S}, k={k}, alpha={alpha}, nx={nx}: {last}")


# ============================================================ Newton on dgamma/dk = 0

def newton_kmax(solve, k0, g0, c0, tol=NEWTON_TOL, maxit=NEWTON_MAXIT, max_step=MAX_STEP,
                log=print, deadline=np.inf):
    # Newton on g(k) = dgamma/dk = 0 with the curvature gamma_kk from a SECANT on the iterates'
    # g (c0: the seed curvature for the first step). solve(k, pred_gamma) -> record (lam,
    # dlam_k). A step is taken only toward a MAXIMUM: if the curvature estimate is not negative
    # the step goes uphill by max_step k; every step is capped at max_step k. Returns
    # (iterates, k*, gamma*, stop): k* = the last iterate plus its (converged, < tol k) step,
    # gamma* = gamma + g step/2 (the quadratic model's maximum). stop None when converged.
    its = []
    k, pred, c = float(k0), float(g0), float(c0)
    for n in range(maxit):
        if time.time() > deadline:
            return its, np.nan, np.nan, "deadline"
        rec = solve(k, pred)
        g, gk = complex(rec["lam"]).real, complex(rec["dlam_k"]).real
        if its:
            kp, gkp = its[-1]["k"], its[-1]["gk"]
            if k != kp:
                c = (gk - gkp)/(k - kp)
        step = -gk/c if c < 0 else np.sign(gk)*max_step*k
        if abs(step) > max_step*k:
            step = np.sign(step)*max_step*k
        its.append(dict(k=k, gamma=g, gk=gk, curv=c, step=step, rec=rec))
        log(f"    Newton it {n}: k = {k:.10f}  gamma = {g:.10e}  dgamma/dk = {gk:+.3e}  "
            f"curv = {c:.4e}  step = {step:+.3e}")
        if abs(step) <= tol*k:
            return its, k + step, g + 0.5*gk*step, None
        pred = g + gk*step + 0.5*c*step*step if c < 0 else g
        k = k + step
    return its, np.nan, np.nan, "iterations"


def implicit_slopes(S, alpha, kstar, gstar, rec0, rp, rm, h):
    # from the solve at k* (rec0) and at k*(1 +- h) (rp, rm): central differences of the three
    # sensitivities in k, and the implicit derivatives of k* and gamma* = gamma(k*)
    dk = 2*h*kstar
    gkk = (complex(rp["dlam_k"]).real - complex(rm["dlam_k"]).real)/dk
    gke = (complex(rp["dlam_eta"]).real - complex(rm["dlam_eta"]).real)/dk
    gka = (complex(rp["dlam_alpha"]).real - complex(rm["dlam_alpha"]).real)/dk
    ge, ga = complex(rec0["dlam_eta"]).real, complex(rec0["dlam_alpha"]).real
    eta = 1.0/S
    u = 1 - alpha*alpha
    out = dict(gkk=gkk, gketa=gke, gkalpha=gka,
               dk_deta=-gke/gkk, dk_dalpha=-gka/gkk,
               slope_k_S=eta*gke/(kstar*gkk),                  # d ln k*/d ln S
               slope_g_S=-eta*ge/gstar,                        # d ln gamma*/d ln S
               # gamma_kk from the three gammas: the curvature check (O(h^2) too)
               gkk_from_gamma=(complex(rp["lam"]).real - 2*complex(rec0["lam"]).real
                               + complex(rm["lam"]).real)/(h*kstar)**2)
    if alpha > 0:
        out["slope_k_u"] = (-gka/gkk)*u/(-2*alpha*kstar)     # d ln k*/d ln(1 - alpha^2)
        out["slope_g_u"] = ga*u/(-2*alpha*gstar)             # d ln gamma*/d ln(1 - alpha^2)
    else:
        out["slope_k_u"] = out["slope_g_u"] = np.nan
    return out


def kmax_solve(S, alpha, k0, g0, c0, nx, Lx, root=None, log=print, compute=True,
               deadline=np.inf, tag=None):
    # the full k_max solution at fixed (S, alpha, nx, Lx): Newton, then the curvature pair, then
    # the grid check at nx/2 (one solve at k*). Returns a dict (or raises RuntimeError)
    def solve(k, pred):
        return point(S, k, alpha, nx, Lx, pred, root=root, log=log, compute=compute, tag=tag)
    its, ks, gs, why = newton_kmax(solve, k0, g0, c0, log=log, deadline=deadline)
    if why is not None:
        raise RuntimeError(f"Newton k_max at S={S}, alpha={alpha}, nx={nx}: {why}")
    rec0 = its[-1]["rec"]
    rp = solve(ks*(1 + CURV_H), gs)
    rm = solve(ks*(1 - CURV_H), gs)
    sl = implicit_slopes(S, alpha, ks, gs, rec0, rp, rm, CURV_H)
    # polish: one Newton step from the last iterate with the central-difference curvature (the
    # secant's k* against it is the Newton convergence error)
    k_pol = its[-1]["k"] - its[-1]["gk"]/sl["gkk"]
    out = dict(S=S, alpha=alpha, nx=nx, Lx=Lx, kmax=ks, gmax=gs, k_polished=k_pol,
               iterations=len(its), its=[dict(k=i["k"], gamma=i["gamma"], gk=i["gk"],
                                              curv=i["curv"], step=i["step"]) for i in its],
               rec=rec0, rec_p=rp, rec_m=rm, **sl)
    try:
        rl = point(S, its[-1]["k"], alpha, nx//2, Lx, gs, root=root, log=log, compute=compute,
                   tag=tag)
        gk_lo = complex(rl["dlam_k"]).real
        out.update(grid_dk=-(gk_lo - its[-1]["gk"])/sl["gkk"],
                   grid_dgamma=complex(rl["lam"]).real - complex(rec0["lam"]).real,
                   rec_lo=rl)
    except (RuntimeError, KeyError) as e:
        log(f"    grid check at nx/2 failed: {str(e)[:150]}")
        out.update(grid_dk=np.nan, grid_dgamma=np.nan)
    log(f"  k_max S={S:.0e} alpha={alpha:<6g} nx={nx:6d}: k* = {ks:.10f} gamma* = {gs:.10e} "
        f"({len(its)} its; polished {k_pol - ks:+.1e}; grid dk {out['grid_dk']:+.1e}) "
        f"dlnk*/dlnS = {sl['slope_k_S']:+.5f} dlng*/dlnS = {sl['slope_g_S']:+.5f} "
        f"kappa = {float(rec0['kappa']):.2e}")
    return out


# ============================================================ seeds

def curvature_theory(S, k, h=1e-3):
    # d2 gamma/dk2 of the exact alpha = 0 relation at k (central difference)
    g = [T.gamma_theory(k*(1 + s*h), S) for s in (1, 0, -1)]
    return (g[0] - 2*g[1] + g[2])/(k*h)**2


def seed_1bref(S, alpha):
    # (k, gamma) seed at a 1b-ref (S, alpha): its peak fit (U.kmax_pred for k), gamma from its
    # table interpolated in ln k at the nearest lower tabulated alpha, scaled by (u/u_ref)^(4/7)
    k = U.kmax_pred(S, alpha)
    return k, U._theory_seed(S, k, alpha)


def seed_theory(S, alpha):
    # (k, gamma) seed anywhere: the alpha = 0 theory maximum scaled by the asymptotic shear
    # factors (1 - alpha^2)^(-1/7) and (1 - alpha^2)^(4/7) -- rough, Newton fixes k
    k0, g0 = theory_kmax(S)
    u = 1 - alpha*alpha
    return k0*u**(-1/7), g0*u**(4/7)


def seed(S, alpha):
    if S in (1e3, 1e4, 1e5) and alpha <= 0.95:
        return seed_1bref(S, alpha)
    return seed_theory(S, alpha)


def _sigma_seed(S, k, alpha, Lx, g, compute):
    # the first solve's expected gamma of a chain: the coarse dense spectrum's fastest real mode
    # at the seed k (it sets sigma only -- never the Newton path, whose first step uses the
    # table/theory curvature seed, so cache keys stay deterministic); read-back: unused
    if not compute:
        return g
    top = _dense_others(S, float(k), alpha, Lx, NX_DENSE)[1]
    return top.real if U._pos_real(top) else g


def curvature_seed(S, k, g):
    kth, gth = theory_kmax(S)
    return curvature_theory(S, kth)*(g/gth)*(kth/k)**2


# ============================================================ phases

def _fail(log, what, e):
    log(f"  {what}: FAILED {str(e)[:220]}")


def _trust(k0, kprev):
    # a continuation seed stays within a factor SEED_TRUST of the previous k*: where the maximum
    # is nearly flat (gamma_kk -> 0, e.g. near a switch between two local maxima) the implicit
    # slope -gamma_k./gamma_kk blows up and the raw extrapolation can land at k < 0. Wide on
    # purpose: it only stops runaways (every seed of the first full campaign is inside it, so
    # their cache keys are unchanged); Newton's own steps are capped at MAX_STEP k anyway
    return float(np.clip(k0, kprev/SEED_TRUST, kprev*SEED_TRUST))


def kmax_alpha(S, alphas=None, root=None, log=print, compute=True, deadline=np.inf):
    # k_max(alpha) at fixed S by continuation in alpha: the seed k is the previous k* extrapolated
    # with the previous dk*/dalpha (implicit differentiation), gamma likewise, the curvature the
    # previous gamma_kk. Returns ({alpha: result}, complete)
    alphas = alphas or ALPHA_KMAX
    Lx, nx = kmax_box(S)
    out, prev = {}, None
    for al in alphas:
        if time.time() > deadline:
            return out, False
        if prev is None:
            k0, g0 = seed(S, al)
            c0 = curvature_seed(S, k0, g0)
            g0 = _sigma_seed(S, k0, al, Lx, g0, compute)
        else:
            dal = al - prev["alpha"]
            k0 = _trust(prev["kmax"] + prev["dk_dalpha"]*dal, prev["kmax"])
            g0 = prev["gmax"] + complex(prev["rec"]["dlam_alpha"]).real*dal
            g0 = g0 if g0 > 0 else prev["gmax"]
            c0 = prev["gkk"]*(g0/prev["gmax"])
        try:
            r = kmax_solve(S, al, k0, g0, c0, nx, Lx, root, log, compute, deadline)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"k_max S={S:.0e} alpha={al}", e)
            continue
        out[al] = r
        prev = r
    return out, True


def kmax_S(alpha, S_list=None, root=None, log=print, compute=True, deadline=np.inf):
    # k_max(S) at fixed alpha by continuation in S: seed k* (S'/S)^(d ln k*/d ln S), gamma*
    # likewise (the implicit slopes of the previous solution)
    S_list = S_list or S_KS
    out, prev = {}, None
    for S in S_list:
        if time.time() > deadline:
            return out, False
        Lx, nx = kmax_box(S)
        if prev is None:
            k0, g0 = seed(S, alpha)
            c0 = curvature_seed(S, k0, g0)
            g0 = _sigma_seed(S, k0, alpha, Lx, g0, compute)
        else:
            r_ = np.log(S/prev["S"])
            k0 = _trust(prev["kmax"]*np.exp(prev["slope_k_S"]*r_), prev["kmax"])
            g0 = prev["gmax"]*np.exp(prev["slope_g_S"]*r_)
            c0 = prev["gkk"]*(g0/prev["gmax"])*(prev["kmax"]/k0)**2
        try:
            r = kmax_solve(S, alpha, k0, g0, c0, nx, Lx, root, log, compute, deadline)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"k_max S={S:.0e} alpha={alpha}", e)
            continue
        out[S] = r
        prev = r
    return out, True


def kmax_grid2(S, alpha, root=None, log=print, compute=True, deadline=np.inf):
    # the same k_max solve at 2 nx, started from the nx solution (the nx-doubling check)
    Lx, nx = kmax_box(S)
    base = kmax_alpha(S, _alphas_upto(alpha), root, _quiet, compute=False)[0].get(alpha)
    if base is None:
        raise KeyError(f"grid2 needs the k_max(alpha) solution at S={S}, alpha={alpha}")
    return kmax_solve(S, alpha, base["kmax"], base["gmax"], base["gkk"], 2*nx, Lx, root, log,
                      compute, deadline)


def _alphas_upto(alpha):
    # the kmax_alpha continuation chain up to alpha (read-back must replay the same chain)
    return tuple(a for a in ALPHA_KMAX if a <= alpha + 1e-12)


def dispersion(S, alpha, kas=None, root=None, log=print, compute=True, deadline=np.inf,
               levels=(1, 2)):
    # gamma(k) and its three sensitivities along a ka sweep at fixed (S, alpha), per-k box
    # (rung 1a's Lx = max(15/k, 20)), levels = nx/level. Seeds: 1b-ref / rung 1a tables where
    # they exist, else the previous ka's gamma + dgamma/dk dk. Returns ({level: {ka: rec}},
    # complete)
    kas = kas or KA_DISP
    out = {lv: {} for lv in levels}
    prev = None
    for ka in kas:
        if time.time() > deadline:
            return out, False
        Lx, nx = disp_box(S, ka)
        pred = _table_gamma(S, ka, alpha)
        if pred is None and prev is not None:
            pred = complex(prev["lam"]).real + complex(prev["dlam_k"]).real*(ka - prev["k"])
            pred = max(pred, 0.3*complex(prev["lam"]).real)
        if pred is None:
            pred = T.gamma_theory(ka, S)*(1 - alpha*alpha)**0.55
        for lv in levels:
            try:
                rec = point(S, ka, alpha, nx//lv, Lx, pred, root=root, log=log, compute=compute)
            except KeyError:
                log(f"  disp S={S:.0e} alpha={alpha} ka={ka} nx={nx//lv}: NOT COMPUTED")
                continue
            except RuntimeError as e:
                _fail(log, f"disp S={S:.0e} alpha={alpha} ka={ka} nx={nx//lv}", e)
                continue
            out[lv][ka] = rec
            if lv == levels[0]:
                prev = rec
                log(f"  disp S={S:.0e} alpha={alpha:<4g} ka={ka:<5g} nx={nx:6d} gamma="
                    f"{complex(rec['lam']).real:.8e} qS={s_exponent(rec):+.5f} "
                    f"qk={k_exponent(rec):+.5f} kappa={float(rec['kappa']):.2e} "
                    f"({float(rec['seconds']):.0f}s)")
    return out, True


def _table_gamma(S, ka, alpha):
    # a converged gamma from 1b-ref (alpha in its table, ka in its table) or rung 1a (alpha = 0)
    try:
        tab = U.shear_tables()
        if (ka, alpha) in tab and S in tab[(ka, alpha)]:
            return complex(tab[(ka, alpha)][S]["value"]).real
        if alpha == 0.0:
            et = U.eigen_tables()
            if ka in et and S in et[ka]:
                return complex(et[ka][S]["value"]).real
    except (KeyError, FileNotFoundError, RuntimeError):
        pass
    return None


def fd_gate(S, ka, alpha, root=None, log=print, compute=True, hs=FD_H):
    # central differences of eigenvalues in eta (relative steps, through overrides) and in k
    # (overrides Ly) at the sensitivity's own (nx, Lx), against dgamma/deta and dgamma/dk
    Lx, nx = disp_box(S, ka)
    pred = _table_gamma(S, ka, alpha) or T.gamma_theory(ka, S)
    rec = point(S, ka, alpha, nx, Lx, pred, root=root, log=log, compute=compute)
    lam = complex(rec["lam"])
    eta = 1.0/S
    rows = {"eta": [], "k": []}
    for which in ("eta", "k"):
        for h in hs:
            vals = []
            for sgn in (1, -1):
                kk = ka*(1 + sgn*h) if which == "k" else ka
                ee = eta*(1 + sgn*h) if which == "eta" else None
                r = None
                for attempt, off in enumerate((0.03, 0.1, 0.5)):
                    sig = lam*(1 + off)
                    try:
                        r = U.cached("eig2", root or DATA_ROOT,
                                     (lambda kk=kk, ee=ee, sig=sig: eig_only(
                                         S, kk, alpha, nx, Lx, sig, lam_hint=lam, eta=ee))
                                     if compute else None,
                                     S=S, k=float(kk), alpha=alpha, nx=nx, Lx=Lx,
                                     eta=float(ee if ee is not None else eta),
                                     **({"attempt": attempt} if attempt else {}))
                        break
                    except RuntimeError as e:
                        log(f"    FD eig failed ({which}, h={h}, attempt {attempt}): "
                            f"{str(e)[:120]}")
                if r is None:
                    raise RuntimeError(f"FD eigenvalue ({which}, h={h}) failed at every shift")
                vals.append(complex(r["lam"]))
            step = 2*h*(ka if which == "k" else eta)
            fd = (vals[0] - vals[1])/step
            ref = complex(rec["dlam_k" if which == "k" else "dlam_eta"])
            rows[which].append((h, fd, abs(fd - ref)))
            log(f"  FD S={S:.0e} ka={ka} alpha={alpha} d/d{which} h={h:g}: FD {fd.real:+.10e}"
                f"  sens {ref.real:+.10e}  |diff| {rows[which][-1][2]:.2e}")
    return rec, rows


# ============================================================ make_data

PHASES = ("fd", "disp", "kmax_alpha", "kmax_S", "kmax_grid2")


def make_data(root=None, wall_budget=3*3600.0, phases=None, log=print, S_list=None,
              A_list=None):
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    done = True
    for name in phases or PHASES:
        t0 = time.time()
        ok = True
        if name == "fd":
            for pt in FD_POINTS:
                if S_list and pt[0] not in S_list:
                    continue
                try:
                    fd_gate(*pt, root=root, log=log)
                except RuntimeError as e:
                    _fail(log, f"FD {pt}", e)
        elif name == "disp":
            for S in (S_list or S_DISP):
                for al in (A_list or ALPHA_DISP):
                    ok &= dispersion(S, al, root=root, log=log, deadline=deadline)[1]
        elif name == "kmax_alpha":
            for S in (S_list or S_KA):
                ok &= kmax_alpha(S, root=root, log=log, deadline=deadline)[1]
        elif name in ("kmax_S", "kmax_S_hi"):
            for al in (A_list or (ALPHA_KS if name == "kmax_S" else (0.0,))):
                Ss = S_KS if name == "kmax_S" else S_KS + S_KS_HI
                ok &= kmax_S(al, Ss, root=root, log=log, deadline=deadline)[1]
        elif name == "kmax_grid2":
            for (S, al) in GRID2_POINTS:
                if S_list and S not in S_list:
                    continue
                try:
                    kmax_grid2(S, al, root, log, deadline=deadline)
                except (RuntimeError, KeyError) as e:
                    _fail(log, f"grid2 S={S:.0e} alpha={al}", e)
        else:
            raise ValueError(name)
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")
        done &= ok
    return bool(done)


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def disp_results(S, alpha, root=None):
    return dispersion(S, alpha, root=root or DATA_ROOT, log=_quiet, compute=False)[0]


def kmax_alpha_results(S, root=None):
    return kmax_alpha(S, root=root or DATA_ROOT, log=_quiet, compute=False)[0]


def kmax_S_results(alpha, hi=False, root=None):
    Ss = S_KS + (S_KS_HI if hi else ())
    return kmax_S(alpha, Ss, root=root or DATA_ROOT, log=_quiet, compute=False)[0]


def grid2_results(root=None):
    out = {}
    for (S, al) in GRID2_POINTS:
        try:
            out[(S, al)] = kmax_grid2(S, al, root or DATA_ROOT, _quiet, compute=False)
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


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3*3600.0
    args = sys.argv[2:]
    S_list = A_list = None
    while args and args[0][:2] in ("S=", "A="):
        vals = tuple(float(s) for s in args[0][2:].split(","))
        if args[0].startswith("S="):
            S_list = vals
        else:
            A_list = vals
        args = args[1:]

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    log(f"profile {PROFILE}, root {DATA_ROOT}")
    make_data(wall_budget=budget, phases=tuple(args) or None, log=log, S_list=S_list,
              A_list=A_list)
