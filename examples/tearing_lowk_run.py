# Identification of the LOW-k maximum of gamma(k) found by the rung-2 campaign
# (examples/tearing-kmax.ipynb sections 3-4): resumable data generation for
# examples/tearing-lowk-branch.ipynb.
#
# Setup exactly as rungs 1b-ref / 1b / 2 (tearing_rung2_run.py): dims=2, hyper = 1, a = v_Ay = 1,
# S = 1/eta, fp64; Psi0 = sech^2 x (+ its two nearest periodic images), Phi0 = alpha Psi0; the ky
# block iky = 1 with the wavenumber AND the dissipation pair as overrides,
#     overrides = {"Ly": 2 pi/k, "diss": (nu, eta)},
# so one Parameters (R.make_params(S, nx, Lx): diss (0, 1/S), Ly = 2 pi placeholders) serves every
# k, alpha and nu of a (S, nx, Lx). nu = 0 everywhere except the Pm checks.
#
# ONE SOLVE (`lk_solve`) = rung 2's band-LU shift-invert right/left pair, plus FOUR sensitivities
# from it (w^H dJ v / w^H v): eta (diss tangent (0, 1)), nu (diss tangent (1, 0)), k (Ly tangent
# -Ly/k), alpha (dx0 = (Psi0, 0)); plus, from the right eigenvector:
#   * the ENERGY BUDGET of the eigenmode (y-averaged, per unit length in y),
#       2 gamma E = R + M + C - D_nu - D_eta,   E = int <|u1|^2 + |b1|^2>/2 dx,
#       R = -int U' <u1x u1y>   (Reynolds stress against the flow shear),
#       M = +int U' <b1x b1y>   (Maxwell stress against the flow shear),
#       C = +int B' <u1y b1x - u1x b1y>   (work of the equilibrium current: the tearing drive),
#       D_nu = nu int <|w1|^2>, D_eta = eta int <|j1|^2>,
#     U = alpha B = alpha Psi0'(x) (u = z x grad phi, b = z x grad psi); the closure
#     |2 gamma E - sum| is recorded (it is exact for an exact eigenpair: a gate);
#   * Elsasser energies of the perturbation, E+- = int <|grad(phi +- psi)|^2>/4 (background
#     Z+- = (alpha +- 1) Psi0, so Z- = 0 at alpha = 1);
#   * structure: |psi(0)|/max|psi|, the antiunitary symmetry residual (with flow, the
#     equilibrium is invariant under the rotation (x, y) -> (-x, -y), which on a ky block is
#     f(x) -> conj f(-x): a real eigenvalue's eigenvector is invariant up to phase), the RMS
#     extent of the energy density and of the C density, the paper's quarter-max layer width,
#     and the eigenfunction profile (T._profile).
#
# WHY alpha = 1 IS SPECIAL (Elsasser form, derivation in the notebook): with Z- = 0 the
# linearized z- equation is d_t dz- + (Z+.grad) dz- = -grad p + (nu+eta)/2 lap dz- +
# (nu-eta)/2 lap dz+, so at nu = eta > 0, ||dz-|| decays monotonically and an eigenmode with
# dz- = 0 has dz+ purely diffusing: NO growing EIGENMODE at alpha = 1, Pm = 1, for any profile, k
# or S. A modal statement: ||dz+|| (and the total energy) is driven by (dz-.grad)Z+ and can grow
# transiently. A growing mode at alpha = 1 therefore needs nu != eta. Gated in
# tests/test_tearing_lowk.py.
#
# PHASES (`python examples/tearing_lowk_run.py [wall_s] [phase ...]`; TARANIS_LK_PROFILE =
# smoke|full, default full; data under DATA_ROOT/solves, one cache file per solve):
#   box       gamma and the sensitivities at fixed (S, alpha, k) against Lx (factors BOX_FAC of the
#             rung-2 k_max box, dx fixed) and nx (factors at the rung-2 box) -- the artifact test
#   julia     the same points with Alfred's Julia eigencode (open Robin boundary at xlim, a
#             nonuniform grid: an INDEPENDENT discretization and boundary), xlim = 10, 20, 40
#   scan      gamma(k) on a ka grid at S_SCAN x ALPHA_SCAN (dense eig, coarse grid, cached):
#             the two-hump picture, continued in alpha through 1
#   ref       the coarse dense gamma at the Julia reference npz's (S = 1e3, ka, alpha = 0.95)
#   cutoff    gamma across the ideal marginal point ka = sqrt 5 (Delta' = 0) at alpha = 0, 0.995, 1
#   alpha1    alpha = 1 (exact Alfvenic state): k_max by Newton at S_A1; gamma(S) at fixed KA_GS
#             for ALPHA_GS = 0.995, 1, 1.02, 1.1 (alpha > 1: ideal Kelvin-Helmholtz), 2nx checks
#   pm        gamma at Pm = nu/eta in PM_LIST at the low-k and high-k maxima (and alpha = 1)
#   pmscan    gamma(k) at Pm in PM_SCAN_LIST (dense, scan grid)
#   boundary  alpha_c(S): where the low-k maximum's gamma* equals the high-k maximum's, as the
#             root in u = ln(1 - alpha^2) of ln(gamma*_lo/gamma*_hi) (bracketed_root), each maximum
#             a bounded Brent search in ln k on EIGENVALUES ALONE (gpoint). The sensitivity route
#             (Newton on dgamma/dk, kept as alpha_c for the S = 1e4 check) is REFUSED by the harness
#             near alpha_c at S >= 2e4 (kappa ~ 2e5: kappa delta/gap ~ 1e-3), and the refusal is
#             respected. From BOUNDARY_DENSE_S on, shift-invert fails in GMRES at every shift and the
#             boundary runs on the dense spectrum (nx = 1024)
#   bcheck    the alpha_c grid checks (full nx, or dense at 2 x 1024) and the S = 1e4 Newton route
#   bbox      the alpha_c BOX checks (both maxima, dense, Lx x 2 at fixed dx) and the box-corrected
#             alpha_c by the envelope theorem (boundary_box_check; added in review)
#   struct    both branches' k_max solutions (sensitivities, budget, eigenfunctions) at STRUCT_POINTS
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import functools
import sys
import time

import numpy as np

from taranis import stability as st

import tearing_rung2_run as R
import tearing_shear_run as H

T, U = R.T, R.U

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FULL = os.path.join(HERE, "data", "tearing-lowk")
DATA_ROOT_SMOKE = os.path.join(HERE, "data", "tearing-lowk-smoke")
PROFILE = os.environ.get("TARANIS_LK_PROFILE", "full")
if PROFILE not in ("full", "smoke"):
    raise ValueError(f"TARANIS_LK_PROFILE must be 'full' or 'smoke', got {PROFILE!r}")
DATA_ROOT = DATA_ROOT_SMOKE if PROFILE == "smoke" else DATA_ROOT_FULL

TWO_PI = 2*np.pi
NX_DENSE = 1024
ETA_DIR, NU_DIR = (0.0, 1.0), (1.0, 0.0)

# ---- configuration (full profile; smoke below)
# box points: (S, alpha, ka). The two low-k maxima named by the rung-2 campaign, alpha = 1, and
# the high-k tearing maximum at S = 1e4 alpha = 0.99 as the control
BOX_POINTS = ((1e4, 0.995, 0.179), (1e3, 0.995, 0.36), (1e4, 1.0, 0.18), (1e4, 0.99, 0.713))
BOX_FAC = (0.5, 1.0, 2.0)            # Lx / (rung-2 k_max box), dx fixed
BOX_FAC4 = 1                         # the first BOX_FAC4 points also at Lx x4 (nx 16384: costly)
NX_FAC = (0.5, 1.0, 2.0)             # nx / (rung-2 nx), at the rung-2 box
JL_XLIM = (10.0, 20.0, 40.0)
JL_LEVELS = (1.0, 0.5, 0.25)
S_SCAN = (1e3, 1e4)
ALPHA_SCAN = (0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 1.0, 1.01, 1.05)
KA_SCAN = tuple(np.round(np.geomspace(0.05, 1.6, 22), 4))
S_A1 = (1e3, 3e3, 1e4, 3e4, 1e5)
ALPHA_GS = (0.995, 1.0, 1.02, 1.1)   # gamma(S) at fixed k
KA_GS = 0.2
S_GS = (1e3, 3e3, 1e4, 3e4, 1e5)          # S >= 3e5 at nx 4096: every shift refused (GMRES)
PM_LIST = (0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
PM_SCAN = ((1e4, 0.995), (1e4, 1.0))  # gamma(k) at Pm in PM_SCAN_LIST (dense, scan grid)
PM_SCAN_LIST = (0.0, 0.1, 1.0, 10.0)
PM_POINTS = (("low", 1e4, 0.995), ("high", 1e4, 0.99), ("low", 1e3, 0.995), ("alpha1", 1e4, 1.0))
S_BOUNDARY = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5)
REF_POINTS = tuple((1e3, ka, 0.95) for ka in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0))   # in the Julia npz
CUTOFF_ALPHA = (0.0, 0.995, 1.0)         # gamma across the ideal marginal point ka = sqrt 5
CUTOFF_KA = (1.6, 2.0, 2.2, 2.23, 2.24, 2.3, 2.6)
CUTOFF_S = 1e3
STRUCT_POINTS = ((1e4, 0.9916), (1e4, 0.995), (1e4, 0.99), (1e4, 1.0))   # 0.9916: both maxima (alpha_c)
if PROFILE == "smoke":
    BOX_POINTS = ((1e3, 0.995, 0.36),)
    BOX_FAC, NX_FAC, BOX_FAC4 = (1.0, 2.0), (0.5, 1.0), 0
    JL_XLIM, JL_LEVELS = (10.0, 20.0), (1.0, 0.5)
    S_SCAN, ALPHA_SCAN = (1e3,), (0.99, 1.0)
    KA_SCAN = (0.1, 0.36, 1.0)
    S_A1, ALPHA_GS, S_GS = (1e3, 3e3), (1.0,), (1e3, 3e3)
    PM_LIST, PM_POINTS = (0.0, 1.0), (("alpha1", 1e3, 1.0),)
    PM_SCAN, PM_SCAN_LIST = ((1e3, 1.0),), (0.0, 1.0)
    S_BOUNDARY, STRUCT_POINTS = (1e3,), ((1e3, 0.995),)
    REF_POINTS, CUTOFF_ALPHA, CUTOFF_KA = REF_POINTS[2:4], (1.0,), (2.2, 2.3)


# ============================================================ operators

def overrides(k, S, nu=0.0):
    return {"Ly": TWO_PI/float(k), "diss": (float(nu), 1.0/float(S))}


def box(S, fac=1.0, nxfac=1.0):
    # the rung-2 k_max box at this S scaled by fac with dx fixed, times nxfac in nx
    Lx, nx = R.kmax_box(S)
    return float(np.round(Lx*fac, 6)), int(round(nx*fac*nxfac))


def _setup(S, k, alpha, nx, Lx, nu):
    params = R.make_params(S, nx, Lx)
    kg = R._kgrid(S, nx, Lx)
    x0, dx0 = U.equilibrium(params, alpha)
    return params, kg, x0, dx0, overrides(k, S, nu)


def dense_block(S, k, alpha, nx, Lx, nu=0.0):
    # (params, idx, EigResult) of the ky block on this grid (eig_dense)
    params, kg, x0, _, ov = _setup(S, k, alpha, nx, Lx, nu)
    B, idx = st.ky_block_matrix(x0, kg, params, 1, overrides=ov)
    return params, idx, st.eig_dense(B)


@functools.lru_cache(maxsize=256)
def _dense_others(S, k, alpha, Lx, nx, nu):
    r = dense_block(S, k, alpha, nx, Lx, nu)[2]
    return r.values[1:], complex(r.values[0]), int(np.sum(r.values.real > 1e-10*r.scale))


def dense_top(S, k, alpha, nx, Lx, nu=0.0, root=None, compute=True, ntop=8):
    # the top of the dense spectrum (cached kind "lkdense"): values, residuals, # growing
    def run():
        r = dense_block(S, k, alpha, nx, Lx, nu)[2]
        return dict(values=r.values[:ntop], residuals=r.residuals[:ntop], scale=r.scale,
                    npos=int(np.sum(r.values.real > 1e-10*r.scale)))
    return U.cached("lkdense", root or DATA_ROOT, run if compute else None, S=S, k=float(k),
                    alpha=alpha, nx=nx, Lx=Lx, nu=float(nu))


# ============================================================ eigenfunction diagnostics

def _dx(f, Lx, n=1):
    kx = TWO_PI*np.fft.fftfreq(len(f), d=Lx/len(f))
    return np.fft.ifft((1j*kx)**n*np.fft.fft(f))


def fields_of(params, idx, v):
    # (x centred on the sheet, phi(x), psi(x)) of a block eigenvector, phase-normalized so that
    # psi(0) is real and positive (if psi(0) = 0: at max |psi|), scaled to max(|phi|,|psi|) = 1
    x, psi, phi, _ = T._eigfun(params, idx, v, params.nx//2)
    if abs(psi[params.nx//2]) < 1e-12*np.abs(psi).max():
        i = int(np.argmax(np.abs(psi)))
        ph = np.conj(psi[i])/abs(psi[i])
        psi, phi = psi*ph, phi*ph
    s = max(np.abs(phi).max(), np.abs(psi).max())
    return x - params.Lx/2, phi/s, psi/s


def budget(params, idx, v, k, alpha, nu, eta):
    # energy-budget densities and integrals of the block eigenvector v (see the header)
    Lx = params.Lx
    x, phi, psi = fields_of(params, idx, v)
    xg = np.arange(params.nx)*Lx/params.nx
    P0 = U.psi0_sech2_a(xg, Lx)
    Bp = np.real(_dx(P0, Lx, 2))
    Up = alpha*Bp
    ux, uy, bx, by = -1j*k*phi, _dx(phi, Lx), -1j*k*psi, _dx(psi, Lx)
    w = _dx(phi, Lx, 2) - k*k*phi
    j = _dx(psi, Lx, 2) - k*k*psi

    def av(a, b):
        return 0.5*np.real(a*np.conj(b))
    # Elsasser: zeta+- = phi +- psi, dz+- = z x grad zeta+-, background Z+-_y = (alpha +- 1) B,
    # Ep/Em = <|dz+-|^2>/4 (E = Ep + Em). d_t Ep = Tp + (Dpp + X)/2... per field:
    # 2 gamma Ep = Tp - Dpp + Xp, 2 gamma Em = Tm - Dmm + Xm with the shear transfers
    # Tp = -(1/2) Z+' <dz+_y dz-_x>, Tm = -(1/2) Z-' <dz-_y dz+_x> and the dissipation
    # Dpp = (nu+eta)/4 <|w+|^2>, Xp = -(nu-eta)/4 <w+ w->, w+- = lap zeta+- (same for -)
    zp, zm = phi + psi, phi - psi
    zpx, zpy, zmx, zmy = -1j*k*zp, _dx(zp, Lx), -1j*k*zm, _dx(zm, Lx)
    wp, wm = w + j, w - j
    Zpp, Zmp = (alpha + 1.0)*Bp, (alpha - 1.0)*Bp
    dens = dict(E=0.5*(av(ux, ux) + av(uy, uy) + av(bx, bx) + av(by, by)),
                R=-Up*av(ux, uy), M=Up*av(bx, by), C=Bp*(av(uy, bx) - av(ux, by)),
                Dnu=nu*av(w, w), Deta=eta*av(j, j),
                Ep=0.25*(av(zpx, zpx) + av(zpy, zpy)), Em=0.25*(av(zmx, zmx) + av(zmy, zmy)),
                Tp=-0.5*Zpp*av(zpy, zmx), Tm=-0.5*Zmp*av(zmy, zpx),
                Dpp=0.25*(nu + eta)*av(wp, wp), Dmm=0.25*(nu + eta)*av(wm, wm),
                Xp=-0.25*(nu - eta)*av(wp, wm), Xm=-0.25*(nu - eta)*av(wm, wp))
    h = Lx/params.nx
    tot = {key: float(np.sum(d)*h) for key, d in dens.items()}
    return x, dens, tot


def structure(params, idx, v, lam, k, alpha, nu, eta):
    # scalar structure measures + the budget (normalized by 2 gamma E) of one eigenvector
    x, phi, psi = fields_of(params, idx, v)
    xb, dens, tot = budget(params, idx, v, k, alpha, nu, eta)
    g = complex(lam).real
    norm = 2*g*tot["E"]
    i0 = params.nx//2
    # antiunitary symmetry f(x) -> conj f(-x): x_i -> x_{-i} about the sheet index nx/2
    mir = (2*i0 - np.arange(params.nx)) % params.nx
    sym = max(np.abs(psi - np.conj(psi[mir])).max(), np.abs(phi - np.conj(phi[mir])).max())
    E = dens["E"]
    out = dict(psi0=float(abs(psi[i0])/np.abs(psi).max()),
               phi_over_psi=float(np.abs(phi).max()/np.abs(psi).max()),
               sym_residual=float(sym),
               x_rms_E=float(np.sqrt(np.sum(x*x*E)/np.sum(E))),
               x_rms_C=(float(np.sqrt(np.sum(x*x*np.abs(dens["C"]))/np.sum(np.abs(dens["C"]))))
                        if np.any(dens["C"] != 0) else np.nan),
               closure=float(abs(norm - (tot["R"] + tot["M"] + tot["C"] - tot["Dnu"]
                                          - tot["Deta"]))/abs(norm)),
               em_frac=float(tot["Em"]/(tot["Ep"] + tot["Em"])),
               closure_p=float(abs(2*g*tot["Ep"] - (tot["Tp"] - tot["Dpp"] + tot["Xp"]))/abs(norm)),
               closure_m=float(abs(2*g*tot["Em"] - (tot["Tm"] - tot["Dmm"] + tot["Xm"]))/abs(norm)))
    for key in ("R", "M", "C", "Dnu", "Deta", "Tp", "Tm", "Dpp", "Dmm", "Xp", "Xm"):
        out["b_" + key] = tot[key]/norm
    return out


def lk_solve(S, k, alpha, nx, Lx, sigma, nu=0.0, lam_hint=None, rtol_mult=1.0):
    # right/left pair, four sensitivities, structure + budget, profile. Raises RuntimeError on any
    # harness refusal (cached as a failure by the caller)
    t0 = time.time()
    params, kg, x0, dx0, ov = _setup(S, k, alpha, nx, Lx, nu)
    op, idx = st.ky_block_operator(x0, kg, params, 1, overrides=ov)
    A = U.band_matrix(op, idx, nx, T.band_width("sech2", Lx))
    M, MH = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
    grtol = float(np.clip(300*rtol_mult*np.finfo(float).eps*normA/max(dist, 1e-300), 1e-10,
                          1e-6*max(rtol_mult, 1.0)))
    kw = dict(tol=1e-12, gmres_rtol=grtol, restart=30 if rtol_mult == 1.0 else 60, maxiter=20)
    right = st.shift_invert(op, sigma, k=1, M=M, **kw)
    lam = complex(right.values[0])
    v = right.right[:, 0]
    w = st.left_eigenvector(op, lam, sigma=sigma, M=MH, **kw).left
    others = _dense_others(S, float(k), alpha, Lx, min(NX_DENSE, nx), float(nu))[0]
    spectrum = np.concatenate([[lam], others])
    sens = {}
    for name, dx0_, dp in (("eta", None, {"diss": ETA_DIR}), ("nu", None, {"diss": NU_DIR}),
                           ("k", None, R.k_direction(k)), ("alpha", dx0, None)):
        dJ, _ = st.dj_operator(x0, dx0_, kg, params, iky=1, dparams=dp, overrides=ov)
        s = st.eigenvalue_sensitivity(op, dJ, lam, v, w, spectrum=spectrum)
        sens["dlam_" + name] = s.dlam
        if name == "eta":
            sens.update(kappa=s.kappa, gap=s.gap, rel_err=s.rel_err, res_r=s.residual_right,
                        res_l=s.residual_left)
    stc = structure(params, idx, v, lam, k, alpha, nu, 1.0/S)
    x, psi, phi, d2 = T._eigfun(params, idx, v, params.nx//2)
    prof = T._profile(x, psi, phi, d2, params.nx//2, Lx, half=1.5)
    return dict(S=S, k=float(k), alpha=alpha, nu=float(nu), nx=nx, Lx=Lx, sigma=complex(sigma),
                lam=lam, **sens, **stc, width=T.width_quarter(prof["xn"], prof["d2n"]),
                xf=prof["xf"], psif=prof["psif"], phif=prof["phif"], xn=prof["xn"],
                psin=prof["psin"], phin=prof["phin"], seconds=time.time() - t0)


LADDER = ((0.03, 1.0), (0.13, 1.0), (0.5, 1.0), (-0.03, 1.0), (0.03, 100.0), (0.13, 100.0),
          (0.03, 1e4))                       # (relative shift offset, GMRES rtol multiplier)
LADDER_LOOSE = ((0.03, 100.0), (0.13, 100.0), (0.5, 100.0), (-0.03, 100.0), (0.03, 1e4),
                (0.03, 1.0))
LOOSE_ALPHA = 0.99


def point(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True):
    # a cached lk_solve (kind "lk"), retried at moved shifts / looser inner solves when the harness
    # refuses or the eigenvalue found is not a real growing mode; the key never holds sigma
    cfg = dict(S=S, k=float(k), alpha=alpha, nx=nx, Lx=Lx, nu=float(nu))
    ladder = LADDER
    if alpha >= LOOSE_ALPHA:
        # near alpha = 1 the shifted block is strongly non-normal and GMRES at the tight inner
        # rtol fails at nearly every point (4 failed attempts per solve in the first runs); the
        # pair is residual-gated by shift_invert either way, so start loose (key marks it)
        ladder = LADDER_LOOSE
        cfg["ladder"] = "loose"
    last = None
    for attempt, (off, rm) in enumerate(ladder):
        sig = pred*(1.0 + off)
        try:
            rec = U.cached("lk", root or DATA_ROOT,
                           (lambda sig=sig, rm=rm: lk_solve(S, k, alpha, nx, Lx, sig, nu, pred,
                                                            rtol_mult=rm))
                           if compute else None, attempt=attempt, **cfg)
        except RuntimeError as e:
            last = str(e)
            log(f"    lk failed (S={S:.0e} k={k:.8g} alpha={alpha} nu={nu:g} nx={nx} attempt "
                f"{attempt}): {last[:160]}")
            continue
        if U._pos_real(rec["lam"]):
            return rec
        log(f"    rejected {complex(rec['lam']):.4e} (S={S:.0e} k={k:.8g} alpha={alpha})")
    raise RuntimeError(f"no growing mode at S={S}, k={k}, alpha={alpha}, nu={nu}, nx={nx}: {last}")


def gamma_solve(S, k, alpha, nx, Lx, sigma, nu=0.0, rtol_mult=100.0):
    # the eigenvalue ALONE (right pair, residual-gated by shift_invert): the boundary phase, where
    # near alpha = 1 the sensitivities are REFUSED (kappa ~ 2e5: kappa delta/gap ~ 1e-3) but the
    # eigenvalue itself is reproducible to ~1e-11 across shifts
    t0 = time.time()
    params, kg, x0, _, ov = _setup(S, k, alpha, nx, Lx, nu)
    op, idx = st.ky_block_operator(x0, kg, params, 1, overrides=ov)
    A = U.band_matrix(op, idx, nx, T.band_width("sech2", Lx))
    M, _ = U._lu_ops(A, sigma)
    normA = float(abs(A).sum(axis=0).max())
    grtol = float(np.clip(300*rtol_mult*np.finfo(float).eps*normA/(0.03*abs(sigma)), 1e-10,
                          1e-6*max(rtol_mult, 1.0)))
    r = st.shift_invert(op, sigma, k=1, M=M, tol=1e-12, gmres_rtol=grtol,
                        restart=30 if rtol_mult == 1.0 else 60, maxiter=20)
    return dict(S=S, k=float(k), alpha=alpha, nu=float(nu), nx=nx, Lx=Lx, sigma=complex(sigma),
                lam=complex(r.values[0]), res=float(r.residuals[0]), seconds=time.time() - t0)


def gpoint(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True,
           dense_first=False):
    # a cached gamma_solve (kind "lkg") over LADDER_LOOSE, then the dense fallback; the key never
    # holds sigma. dense_first: straight to the dense spectrum (where every shift fails anyway)
    cfg = dict(S=S, k=float(k), alpha=alpha, nx=nx, Lx=Lx, nu=float(nu))
    last = None
    for attempt, (off, rm) in enumerate(() if dense_first else LADDER_LOOSE):
        sig = pred*(1.0 + off)
        try:
            rec = U.cached("lkg", root or DATA_ROOT,
                           (lambda sig=sig, rm=rm: gamma_solve(S, k, alpha, nx, Lx, sig, nu, rm))
                           if compute else None, attempt=attempt, **cfg)
        except RuntimeError as e:
            last = str(e)
            log(f"    lkg failed (S={S:.0e} k={k:.8g} alpha={alpha} nx={nx} attempt {attempt}): "
                f"{last[:120]}")
            continue
        if U._pos_real(rec["lam"]):
            return rec
        log(f"    rejected {complex(rec['lam']):.4e} (S={S:.0e} k={k:.8g} alpha={alpha})")
    if nx <= GPOINT_DENSE_NX or dense_first:
        # every shift failed (GMRES on the strongly non-normal block): the dense spectrum of the
        # same block, its fastest real growing eigenvalue (eig_dense residuals recorded)
        def dense():
            r = dense_block(S, k, alpha, nx, Lx, nu)[2]
            g = top_real(r.values)
            i = int(np.argmin(np.abs(r.values - g)))
            return dict(S=S, k=float(k), alpha=alpha, nu=float(nu), nx=nx, Lx=Lx,
                        lam=complex(r.values[i]), res=float(r.residuals[i]), dense=True)
        rec = U.cached("lkg", root or DATA_ROOT, dense if compute else None, attempt="dense",
                       **cfg)
        if U._pos_real(rec["lam"]):
            return rec
    raise RuntimeError(f"no growing mode at S={S}, k={k}, alpha={alpha}, nu={nu}, nx={nx}: {last}")


GPOINT_DENSE_NX = 2048                     # gpoint's dense fallback up to this nx


def kmax_brent(S, alpha, k0, nx, Lx, ratio=1.5, xatol=2e-4, root=None, log=print, compute=True,
               dense_first=False):
    # the local maximum of gamma(k) on [k0/ratio, k0 ratio] by bounded Brent in ln k on
    # eigenvalues alone (gpoint). The maximum EXISTS only if it is interior: a maximizer within
    # 2 xatol (in ln k) of an end means gamma rises out of the window, and None is returned
    # with the end point. Returns (dict(kmax, gmax, evals) or None, end-point dict)
    from scipy.optimize import minimize_scalar
    lo, hi = np.log(k0/ratio), np.log(k0*ratio)
    seen = {}
    state = dict(pred=_pred(S, k0, alpha, Lx, compute=compute, fallback=1e-3)
                 if compute else 1e-3)

    def f(lk):
        k = float(np.exp(lk))
        r = gpoint(S, k, alpha, nx, Lx, state["pred"], root=root, log=log, compute=compute,
                   dense_first=dense_first)
        g = growth(r)
        state["pred"] = g
        seen[k] = g
        return -g
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options=dict(xatol=xatol))
    ks = float(np.exp(res.x))
    out = dict(S=S, alpha=alpha, kmax=ks, gmax=float(-res.fun), evals=len(seen), nx=nx, Lx=Lx,
               k0=k0)
    if res.x - lo < 2*xatol or hi - res.x < 2*xatol:
        log(f"    brent from k0={k0:.4f} (S={S:.0e} alpha={alpha:.8f}): maximizer at the window "
            f"end k = {ks:.5f}: no interior maximum")
        return None, out
    log(f"    brent S={S:.0e} alpha={alpha:.8f}: k* = {ks:.6f} gamma* = {out['gmax']:.8e} "
        f"({len(seen)} evals)")
    return out, out


def growth(rec):
    return complex(rec["lam"]).real


def s_exponent(rec):
    # d ln gamma/d ln S at fixed nu (= -(eta/gamma) dgamma/deta)
    return -(1.0/float(rec["S"]))*complex(rec["dlam_eta"]).real/growth(rec)


def pm_exponent(rec):
    # d gamma/d nu in units of eta: (dgamma/dnu)*eta/gamma, the response to Pm at fixed eta
    return complex(rec["dlam_nu"]).real/(float(rec["S"])*growth(rec))


def _pred(S, k, alpha, Lx, nu=0.0, compute=True, fallback=None):
    # the expected gamma for sigma: the dense coarse grid's fastest real mode
    if not compute:
        return fallback if fallback is not None else 1.0
    top = _dense_others(S, float(k), alpha, Lx, NX_DENSE, float(nu))[1]
    if U._pos_real(top):
        return top.real
    if fallback is not None:
        return fallback
    raise RuntimeError(f"no growing mode on the coarse grid at S={S}, k={k}, alpha={alpha}, "
                       f"nu={nu}: top {top}")


# ============================================================ phases: box / grid

def box_facs(S, alpha, ka):
    # the Lx factors of a box point: BOX_FAC, plus 4 for the first BOX_FAC4 points
    return BOX_FAC + ((4.0,) if (S, alpha, ka) in BOX_POINTS[:BOX_FAC4] else ())


def box_study(S, alpha, ka, root=None, log=print, compute=True):
    # {("Lx", fac): rec} at dx fixed, {("nx", fac): rec} at the rung-2 box
    out = {}
    L1, _ = box(S)
    pred = None
    for kind, facs in (("nx", NX_FAC), ("Lx", box_facs(S, alpha, ka))):
        for f in facs:
            Lx, nx = box(S, f, 1.0) if kind == "Lx" else box(S, 1.0, f)
            if pred is None:
                pred = _pred(S, ka, alpha, L1, compute=compute)
            try:
                rec = point(S, ka, alpha, nx, Lx, pred, root=root, log=log, compute=compute)
            except (RuntimeError, KeyError) as e:
                log(f"  box S={S:.0e} alpha={alpha} ka={ka} {kind}={f}: {str(e)[:150]}")
                continue
            out[(kind, f)] = rec
            log(f"  box S={S:.0e} alpha={alpha} ka={ka} {kind} x{f:<4g} Lx={Lx:8.3f} nx={nx:6d} "
                f"gamma={growth(rec):.10e} qS={s_exponent(rec):+.5f} "
                f"({float(rec['seconds']):.0f} s)")
    return out


def julia_jobs(S, alpha, ka, gamma, xlims=None, levels=None):
    # Julia eigencode jobs (f = -2 tanh sech^2): the reference's grid rule dxmin =
    # min(delta_noflow, delta_in)/50, eps = 0.02, refined jointly by `levels`, at each xlim
    dn = (S*ka)**(-1/3) if ka < S**-0.25 else S**-0.4*ka**-0.6
    di = (alpha/(1 - alpha*alpha))**(1/3)*(ka*S)**(-1/3) if 0 < alpha < 1 else np.inf
    dxmin = min(dn, di)/50.0
    return [(2, S, ka, alpha, dxmin*f, xl, 0.02*f, complex(gamma))
            for xl in (xlims or JL_XLIM) for f in (levels or JL_LEVELS)]


def julia_table(S, alpha, ka, gamma, root=None, log=print, compute=True):
    # {xlim: [gamma per level]} from the Julia eigencode (cached in root/julia.tsv)
    jobs = julia_jobs(S, alpha, ka, gamma)
    path = os.path.join(root or DATA_ROOT, "julia.tsv")
    if not compute:
        have = H.julia_study([], path, log)
        if any(H._jl_key(j) not in have for j in jobs):
            raise KeyError(f"julia not cached at S={S}, alpha={alpha}, ka={ka}")
    else:
        have = H.julia_study(jobs, path, log)
    out = {}
    for j in jobs:
        out.setdefault(j[5], []).append(have[H._jl_key(j)][0])
    return out


# ============================================================ phases: scan

def scan(S, alpha, kas=None, nx=None, Lx=None, root=None, log=print, compute=True, nu=0.0):
    # gamma(k): the top real eigenvalue on the dense coarse grid, {ka: top-of-spectrum record}
    Lx0, nx0 = box(S)
    Lx, nx = Lx or Lx0, nx or min(NX_DENSE, nx0)
    out = {}
    for ka in (kas or KA_SCAN):
        try:
            out[ka] = dense_top(S, ka, alpha, nx, Lx, nu, root=root, compute=compute)
        except KeyError:
            continue
    if compute:
        log(f"  scan S={S:.0e} alpha={alpha}: " + " ".join(
            f"{ka:g}:{complex(r['values'][0]).real:.3e}" for ka, r in out.items()))
    return out


def scan_maxima(rows):
    # local maxima of the top real growth rate over a ka grid: [(ka, gamma)] (interior points
    # whose gamma exceeds both neighbours, plus an end point that is the global maximum)
    kas = sorted(rows)
    g = np.array([top_real(rows[ka]["values"]) for ka in kas])
    out = [(kas[i], g[i]) for i in range(1, len(kas) - 1) if g[i] > g[i - 1] and g[i] > g[i + 1]]
    if len(kas) > 1 and g[0] >= g.max():
        out.insert(0, (kas[0], g[0]))
    if len(kas) > 1 and g[-1] >= g.max():
        out.append((kas[-1], g[-1]))
    return out


def top_real(values, tol=1e-6):
    # the largest real growth rate among values (|Im| <= tol |lambda|); 0 if none grows
    vals = [complex(z) for z in np.atleast_1d(values)
            if abs(complex(z).imag) <= tol*abs(complex(z)) + 1e-12 and complex(z).real > 1e-11]
    return max((z.real for z in vals), default=0.0)


# ============================================================ phases: k_max of a branch

def kmax_branch(S, alpha, k0, nu=0.0, root=None, log=print, compute=True, Lx=None, nx=None,
                c0=None):
    # Newton on dgamma/dk = 0 (R.newton_kmax, secant curvature) from k0 at the rung-2 box (or the
    # given one). The curvature seed (unless c0 is given): a quadratic through the coarse dense
    # gamma at k0(1 +- 0.1) (nx = NX_DENSE on the same Lx). Returns dict(kmax, gmax, rec) or
    # raises RuntimeError
    L0, n0 = box(S)
    Lx, nx = Lx or L0, nx or n0
    # seeds from CACHED coarse dense tops (lkdense), so a read-back replays the same Newton path
    gs = [top_real(dense_top(S, k0*f, alpha, NX_DENSE, Lx, nu, root=root, compute=compute)
                   ["values"]) for f in (0.9, 1.0, 1.1)]
    g0 = gs[1]
    if not g0 > 0:
        raise RuntimeError(f"no growing mode on the coarse grid at S={S}, k={k0}, alpha={alpha}")
    if c0 is None:
        c0 = (gs[0] - 2*gs[1] + gs[2])/(0.1*k0)**2
    c0 = c0 if c0 < 0 else -abs(g0)/k0**2

    def solve(k, pred):
        return point(S, k, alpha, nx, Lx, pred, nu=nu, root=root, log=log, compute=compute)
    its, ks, gs, why = R.newton_kmax(solve, k0, g0, c0, log=log)
    if why is not None:
        raise RuntimeError(f"Newton k_max at S={S}, alpha={alpha}, nu={nu}: {why}")
    return dict(S=S, alpha=alpha, nu=nu, kmax=ks, gmax=gs, rec=its[-1]["rec"],
                iterations=len(its), k0=k0, c0=c0, Lx=Lx, nx=nx)


# ============================================================ phases: dominance boundary

def bracketed_root(f, ua, ub, tol=1e-3, maxit=20, log=print):
    # root of f(u) in [ua, ub] where f(ua), f(ub) have opposite signs; f returns (value, slope)
    # and the value may be +-inf (a branch missing: the sign is still known). Newton with the
    # slope (a secant on the last finite value when the slope is nan) when the step stays
    # inside the bracket, else bisection.
    # Converged when the bracket is narrower than tol. Returns (u*, [(u, value, slope)])
    fa, fb = f(ua)[0], f(ub)[0]
    its = []
    if not np.sign(fa)*np.sign(fb) < 0:
        raise RuntimeError(f"no sign change on [{ua}, {ub}]: {fa}, {fb}")
    prev = None
    for (uu, ff) in ((ua, fa), (ub, fb)):
        if np.isfinite(ff):
            prev = (uu, ff)
    u = 0.5*(ua + ub)
    for _ in range(maxit):
        fv, fd = f(u)
        its.append((u, fv, fd))
        if fv == 0:
            return u, its
        if np.sign(fv) == np.sign(fa):
            ua, fa = u, fv
        else:
            ub, fb = u, fv
        if abs(ub - ua) <= tol:
            break
        if not (np.isfinite(fd) and fd != 0) and np.isfinite(fv) and prev is not None:
            fd = (fv - prev[1])/(u - prev[0])      # no slope: secant on the last finite point
        if np.isfinite(fv):
            prev = (u, fv)
        un = u - fv/fd if np.isfinite(fv) and np.isfinite(fd) and fd != 0 else np.nan
        if np.isfinite(un) and abs(un - u) < 0.25*tol:
            return un, its                   # the Newton step itself is below tol/4
        if np.isfinite(un) and min(ua, ub) < un < max(ua, ub) and abs(un - u) < 0.5*abs(ub - ua):
            u = un
        else:
            u = 0.5*(ua + ub)
    else:
        raise RuntimeError(f"bracketed_root: no convergence ({its[-3:]})")
    # interpolate inside the final bracket where both ends are finite
    if np.isfinite(fa) and np.isfinite(fb) and fa != fb:
        return ua - fa*(ub - ua)/(fb - fa), its
    return 0.5*(ua + ub), its


def alpha_of_u(u):
    return float(np.sqrt(1.0 - np.exp(u)))


def branch_gap(S, alpha, k_low, k_high, root=None, log=print, compute=True, sep=0.75):
    # (ln gamma*_low - ln gamma*_high, its derivative in u = ln(1 - alpha^2), low, high).
    # Envelope theorem: d ln gamma*/d alpha = (d gamma/d alpha at k*)/gamma*, and
    # d alpha/du = -(1 - alpha^2)/(2 alpha). A branch whose maximum does not exist at this alpha
    # (its Newton walks onto the other maximum, or does not converge) is None, and the gap is
    # then -inf (low missing) or +inf (high missing)
    def one(k0):
        try:
            return kmax_branch(S, alpha, k0, root=root, log=log, compute=compute)
        except RuntimeError as e:
            log(f"    branch from k0={k0:.4f} at S={S:.0e} alpha={alpha:.8f}: {str(e)[:120]}")
            return None
    lo, hi = one(k_low), one(k_high)
    mid = np.sqrt(k_low*k_high)
    if lo is not None and lo["kmax"] > mid:
        lo = None if hi is None or abs(lo["kmax"] - hi["kmax"]) < (1 - sep)*hi["kmax"] else lo
    if hi is not None and hi["kmax"] < mid:
        hi = None if lo is None or abs(lo["kmax"] - hi["kmax"]) < (1 - sep)*lo["kmax"] else hi
    if lo is not None and hi is not None and not lo["kmax"] < sep*hi["kmax"]:
        raise RuntimeError(f"branches merged at S={S}, alpha={alpha}: k* {lo['kmax']:.4f}, "
                           f"{hi['kmax']:.4f}")
    if lo is None and hi is None:
        raise RuntimeError(f"no maximum found at S={S}, alpha={alpha}")
    if lo is None:
        return -np.inf, np.nan, None, hi
    if hi is None:
        return np.inf, np.nan, lo, None
    da = -(1 - alpha*alpha)/(2*alpha)
    dl = complex(lo["rec"]["dlam_alpha"]).real/lo["gmax"]
    dh = complex(hi["rec"]["dlam_alpha"]).real/hi["gmax"]
    return float(np.log(lo["gmax"]/hi["gmax"])), float((dl - dh)*da), lo, hi


def alpha_c(S, u_a, u_b, k_low, k_high, tol=2e-3, root=None, log=print, compute=True):
    # alpha_c(S): the root in u = ln(1 - alpha^2) of branch_gap on the bracket [u_a, u_b], each
    # branch's k* seeded from its last solution
    state = dict(kl=k_low, kh=k_high)
    rows = []

    def f(u):
        al = alpha_of_u(u)
        g, d, lo, hi = branch_gap(S, al, state["kl"], state["kh"], root, log, compute)
        if lo is not None:
            state["kl"] = lo["kmax"]
        if hi is not None:
            state["kh"] = hi["kmax"]
        rows.append(dict(u=u, alpha=al, gap=g, dgap=d, lo=lo, hi=hi))
        log(f"  alpha_c S={S:.0e}: alpha={al:.8f} (1-a^2={np.exp(u):.5e}) ln(g_lo/g_hi)={g:+.4e}"
            + (f" k*_lo={lo['kmax']:.5f} g*_lo={lo['gmax']:.6e}" if lo else " (no low max)")
            + (f" k*_hi={hi['kmax']:.5f} g*_hi={hi['gmax']:.6e}" if hi else " (no high max)"))
        return g, d
    u, its = bracketed_root(f, u_a, u_b, tol=tol, log=log)
    return dict(S=S, u=u, alpha=alpha_of_u(u), rows=rows)


BOUNDARY_DENSE_S = 5e4                     # from here the boundary runs on the dense spectrum


def boundary_nx(S):
    # the boundary phase's grid: the rung-2 box with nx/4 (>= 1024); the box study shows nx/2
    # moves gamma by <= 3e-10 at S = 1e4 on both branches. From BOUNDARY_DENSE_S on, every shift
    # of the shift-invert ladder fails near alpha_c (GMRES on the strongly non-normal block), so
    # the boundary runs on the DENSE spectrum at nx = 1024 (both layers are thick there:
    # delta_in/a ~ 0.3). Returns (Lx, nx, dense). alpha_c is grid-checked (boundary_check)
    Lx, nx = box(S)
    if S >= BOUNDARY_DENSE_S:
        return Lx, 1024, True
    return Lx, max(1024, nx//4), False


def branch_gap_brent(S, alpha, k_low, k_high, root=None, log=print, compute=True):
    # branch_gap on eigenvalues alone: each branch's maximum by kmax_brent in a window around its
    # seed; a branch without an interior maximum is None and the gap is -inf (low missing) or
    # +inf (high missing). The slope is not available (nan): the root finder takes secants
    Lx, nx, dense = boundary_nx(S)
    mid = np.sqrt(k_low*k_high)

    def one(k0, side):
        # the window at the seed, re-centred once on the end the maximizer ran into; accepted
        # only if interior AND on the branch's own side of the seeds' geometric mean
        res, end = kmax_brent(S, alpha, k0, nx, Lx, root=root, log=log, compute=compute,
                              dense_first=dense)
        if res is None:
            res, end = kmax_brent(S, alpha, end["kmax"], nx, Lx, root=root, log=log,
                                  compute=compute, dense_first=dense)
        if res is None or (res["kmax"] - mid)*side < 0:
            return None
        return res
    lo, hi = one(k_low, -1.0), one(k_high, +1.0)
    if lo is None and hi is None:
        raise RuntimeError(f"no interior maximum at S={S}, alpha={alpha}")
    if lo is not None and hi is not None and not lo["kmax"] < 0.75*hi["kmax"]:
        raise RuntimeError(f"branches merged at S={S}, alpha={alpha}: k* {lo['kmax']:.4f}, "
                           f"{hi['kmax']:.4f}")
    if lo is None:
        return -np.inf, np.nan, None, hi
    if hi is None:
        return np.inf, np.nan, lo, None
    return float(np.log(lo["gmax"]/hi["gmax"])), np.nan, lo, hi


def alpha_c_brent(S, u_a, u_b, k_low, k_high, tol=0.02, root=None, log=print, compute=True):
    # alpha_c(S) from eigenvalues alone (see branch_gap_brent); seeds continue each branch's k*
    state = dict(kl=k_low, kh=k_high)
    rows = []

    def f(u):
        al = alpha_of_u(u)
        g, d, lo, hi = branch_gap_brent(S, al, state["kl"], state["kh"], root, log, compute)
        if lo is not None:
            state["kl"] = lo["kmax"]
        if hi is not None:
            state["kh"] = hi["kmax"]
        rows.append(dict(u=u, alpha=al, gap=g, lo=lo, hi=hi))
        log(f"  alpha_c S={S:.0e}: alpha={al:.8f} (1-a^2={np.exp(u):.5e}) ln(g_lo/g_hi)={g:+.4e}"
            + (f" k*_lo={lo['kmax']:.5f} g*_lo={lo['gmax']:.6e}" if lo else " (no low max)")
            + (f" k*_hi={hi['kmax']:.5f} g*_hi={hi['gmax']:.6e}" if hi else " (no high max)"))
        return g, d
    u, its = bracketed_root(f, u_a, u_b, tol=tol, log=log)
    return dict(S=S, u=u, alpha=alpha_of_u(u), rows=rows)


# ============================================================ make_data

PHASES = ("box", "julia", "scan", "ref", "cutoff", "alpha1", "pm", "pmscan", "boundary", "bcheck",
          "bbox", "struct")


def _fail(log, what, e):
    log(f"  {what}: FAILED {str(e)[:220]}")


def make_data(root=None, wall_budget=6*3600.0, phases=None, log=print, S_list=None):
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    for name in phases or PHASES:
        if time.time() > deadline:
            log("deadline")
            return False
        t0 = time.time()
        if name == "box":
            for (S, al, ka) in BOX_POINTS:
                box_study(S, al, ka, root, log)
        elif name == "julia":
            for (S, al, ka) in BOX_POINTS:
                try:
                    g = growth(box_study(S, al, ka, root, log, compute=False)[("nx", 1.0)])
                    julia_table(S, al, ka, g, root, log)
                except (RuntimeError, KeyError) as e:
                    _fail(log, f"julia {S, al, ka}", e)
        elif name == "scan":
            for S in S_SCAN:
                for al in ALPHA_SCAN:
                    scan(S, al, root=root, log=log)
        elif name == "ref":
            ref_results(root, log, compute=True)
        elif name == "cutoff":
            cutoff_results(root, log, compute=True)
        elif name == "alpha1":
            alpha1_results(root, log, compute=True)
        elif name == "pm":
            pm_results(root, log, compute=True)
        elif name == "pmscan":
            pmscan_results(root, log, compute=True)
        elif name == "boundary":
            boundary_results(root, log, compute=True, S_list=S_list)
        elif name == "bcheck":
            for S, res in boundary_results(root, S_list=S_list).items():
                try:
                    boundary_check(res, root, log, compute=True)
                except (RuntimeError, KeyError, IndexError) as e:
                    _fail(log, f"alpha_c grid check S={S:.0e}", e)
            boundary_newton_results(root, log, compute=True)
        elif name == "bbox":
            for S, res in boundary_results(root, S_list=S_list).items():
                if not any(r["lo"] is not None and r["hi"] is not None for r in res["rows"]):
                    continue
                try:
                    boundary_box_check(res, root, log, compute=True)
                except (RuntimeError, KeyError, IndexError) as e:
                    _fail(log, f"alpha_c box check S={S:.0e}", e)
        elif name == "struct":
            struct_results(root, log, compute=True)
        else:
            raise ValueError(name)
        log(f"== phase {name} ({time.time() - t0:.0f} s)")
    return True


# ============================================================ phase bodies (compute / read-back)

def _quiet(*a, **k):
    pass


def alpha1_results(root=None, log=_quiet, compute=False):
    # k_max at alpha = 1 along S_A1 (continued in S), and gamma(S) at fixed KA_GS for ALPHA_GS
    root = root or DATA_ROOT
    out = dict(kmax={}, gs={})
    k0 = 0.4
    for S in S_A1:
        try:
            r = kmax_branch(S, 1.0, k0, root=root, log=log, compute=compute)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"alpha=1 k_max S={S:.0e}", e)
            continue
        out["kmax"][S] = r
        k0 = r["kmax"]*(3.0**-0.3)
        log(f"  alpha=1 S={S:.0e}: k* = {r['kmax']:.8f} gamma* = {r['gmax']:.8e} "
            f"qS = {s_exponent(r['rec']):+.4f}")
    for al in ALPHA_GS:
        for S in S_GS:
            Lx, nx = gs_box(S)
            try:
                pred = _pred(S, KA_GS, al, Lx, compute=compute)
                rec = point(S, KA_GS, al, nx, Lx, pred, root=root, log=log, compute=compute)
            except (RuntimeError, KeyError) as e:
                _fail(log, f"gamma(S) alpha={al} S={S:.0e}", e)
                continue
            out["gs"][(al, S)] = rec
            log(f"  gamma(S) alpha={al} S={S:.0e} ka={KA_GS}: gamma = {growth(rec):.8e} "
                f"qS = {s_exponent(rec):+.4f}")
    out["gs_check"] = {}
    for al in ALPHA_GS:
        for S in S_GS[-2:]:
            if (al, S) not in out["gs"]:
                continue
            Lx, nx = gs_box(S)
            try:
                rec = point(S, KA_GS, al, 2*nx, Lx, growth(out["gs"][(al, S)]), root=root,
                            log=log, compute=compute)
            except (RuntimeError, KeyError) as e:
                _fail(log, f"gamma(S) grid check alpha={al} S={S:.0e}", e)
                continue
            out["gs_check"][(al, S)] = rec
            log(f"  gamma(S) grid check alpha={al} S={S:.0e}: 2nx/nx - 1 = "
                f"{growth(rec)/growth(out['gs'][(al, S)]) - 1:+.2e}")
    return out


def ref_results(root=None, log=_quiet, compute=False):
    # the coarse dense top at the Julia reference's (S, ka, alpha = 0.95) points, rung-2 box
    out = {}
    for (S, ka, al) in REF_POINTS:
        Lx, nx = box(S)
        try:
            out[(S, ka, al)] = dense_top(S, ka, al, min(NX_DENSE, nx), Lx, root=root,
                                         compute=compute)
        except KeyError:
            continue
        log(f"  ref S={S:.0e} ka={ka} alpha={al}: {top_real(out[(S, ka, al)]['values']):.6e}")
    return out


def cutoff_results(root=None, log=_quiet, compute=False):
    # gamma across ka = sqrt 5 (Delta' = 0) at CUTOFF_S, rung-2 box, dense
    out = {}
    Lx, nx = box(CUTOFF_S)
    for al in CUTOFF_ALPHA:
        for ka in CUTOFF_KA:
            try:
                out[(al, ka)] = dense_top(CUTOFF_S, ka, al, min(NX_DENSE, nx), Lx, root=root,
                                          compute=compute)
            except KeyError:
                continue
            log(f"  cutoff alpha={al} ka={ka}: top {complex(out[(al, ka)]['values'][0]):.4e}")
    return out


def gs_box(S):
    # the fixed-k gamma(S) sweep's box: rung 1a's Lx = 15/KA_GS (the box study puts the image
    # error at ~e^(-k Lx)), nx = a quarter of the rung-1a alpha = 0 rule (the low-k mode is grid
    # converged far coarser: nx/2 moves gamma by <= 3e-10 in the box study), >= 2048; the two
    # largest S are grid-checked at 2 nx (gs_check)
    Lx = 15.0/KA_GS
    return Lx, max(2048, R.nx_rule(S, KA_GS, Lx)//4)


def pm_results(root=None, log=_quiet, compute=False):
    # gamma at Pm in PM_LIST at each PM_POINTS branch maximum (k fixed at the nu = 0 k*)
    root = root or DATA_ROOT
    out = {}
    for (which, S, al) in PM_POINTS:
        try:
            k = pm_k(which, S, al, root, log, compute)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"pm {which} S={S:.0e} alpha={al}", e)
            continue
        Lx, nx = box(S)
        for pm in PM_LIST:
            nu = pm/S
            try:
                top = dense_top(S, k, al, min(NX_DENSE, nx), Lx, nu, root=root, compute=compute)
            except KeyError:
                continue
            g = top_real(top["values"])
            rec = None
            if g > 0:
                try:
                    rec = point(S, k, al, nx, Lx, g, nu=nu, root=root, log=log, compute=compute)
                except (RuntimeError, KeyError) as e:
                    _fail(log, f"pm {which} Pm={pm}", e)
            out[(which, S, al, pm)] = dict(k=k, dense=top, rec=rec)
            log(f"  Pm={pm:<5g} {which:6s} S={S:.0e} alpha={al} k={k:.5f}: dense top "
                f"{complex(top['values'][0]):.4e}" +
                (f", gamma = {growth(rec):.8e}" if rec is not None else ""))
    return out


def pm_k(which, S, al, root, log, compute):
    # the nu = 0 k* of the named branch at (S, alpha)
    seeds = dict(low=0.2*(1e4/S)**0.3, high=R.theory_kmax(S)[0]*max(1 - al*al, 1e-3)**(-1/7),
                 alpha1=0.2*(1e4/S)**0.3)
    return kmax_branch(S, al, seeds[which], root=root, log=log, compute=compute)["kmax"]


# alpha_c brackets in 1 - alpha^2: (low-k side, high-k side). From the rung-2 campaign (S = 1e4:
# alpha 0.99 high-k global, 0.995 low-k global; alpha = 0.995: low-k for S <= 2e4, high-k at 5e4)
# and the scaling estimate 1 - alpha_c^2 ~ 0.015 (S/1e4)^(-0.64)
BOUNDARY_BRACKETS = {1e3: (0.02, 0.1), 2e3: (0.02, 0.1), 5e3: (0.01, 0.05), 1e4: (0.00998, 0.0199),
                     2e4: (0.004, 0.0199), 5e4: (0.0015, 0.00998), 1e5: (0.0008, 0.008)}


BOUNDARY_NEWTON_S = (1e4,)          # alpha_c also by the Newton/sensitivity route (a check)


def boundary_seeds(S):
    # (u_a, u_b, k_low, k_high): the bracket and each branch's seed (k_low from the rung-2
    # alpha = 0.995 low-k chain's k* ~ 0.35 (S/1e3)^-0.3, k_high from the alpha = 0 theory
    # maximum times the paper's (1 - alpha^2)^(-1/7), 0.85 x)
    ua, ub = (float(np.log(v)) for v in BOUNDARY_BRACKETS[S])
    kl = 0.35*(1e3/S)**0.3
    kh = R.theory_kmax(S)[0]*np.exp(0.5*(ua + ub))**(-1/7)*0.85
    return ua, ub, kl, kh


def boundary_newton_results(root=None, log=_quiet, compute=False):
    out = {}
    for S in BOUNDARY_NEWTON_S:
        try:
            out[S] = alpha_c(S, *boundary_seeds(S), tol=0.02, root=root or DATA_ROOT, log=log,
                             compute=compute)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"alpha_c (Newton) S={S:.0e}", e)
    return out


def boundary_check(res, root=None, log=_quiet, compute=False):
    # grid check of an alpha_c solution: both branches' gamma at the last evaluation's k*
    # re-solved at the FULL rung-2 nx (the boundary ran at boundary_nx), or for the dense route
    # by the dense spectrum at twice its nx
    S = res["S"]
    Lx, nx = box(S)
    _, nx_b, dense = boundary_nx(S)
    if dense:
        nx = 2*nx_b                  # the dense route's check: dense at twice its nx
    row = [r for r in res["rows"] if r["lo"] is not None and r["hi"] is not None][-1]
    out = dict(alpha=row["alpha"], nx=nx, nx_b=nx_b)
    for which in ("lo", "hi"):
        b = row[which]
        r = gpoint(S, b["kmax"], row["alpha"], nx, Lx, b["gmax"], root=root, log=log,
                   compute=compute, dense_first=dense)
        out[which] = (b["gmax"], growth(r))
    log(f"  alpha_c grid check S={S:.0e}: " + " ".join(
        f"{w}: {a:.10e} (nx {out['nx_b']}) vs {b:.10e} (nx {nx})" for w, (a, b) in
        ((w, out[w]) for w in ("lo", "hi"))))
    return out


BOX_CHECK_FAC = 2.0                 # boundary_box_check: Lx x this at fixed dx


def boundary_box_check(res, root=None, log=_quiet, compute=False, fac=None):
    # BOX check of an alpha_c solution (the boundary runs on the rung-2 box, where the low-k
    # maximum carries an image error ~ e^(-k* Lx): 5e-4 at S = 1e4, ~1e-2 at S = 1e5). Both
    # branches' gamma at the last both-branch row's (k*, alpha), dense, on the rung-2 box at
    # nx = NX_DENSE and on Lx x fac at the SAME dx (nx = fac NX_DENSE). The box-corrected root by
    # the envelope theorem (k* moves at second order): u' = u_c - dgap/(d gap/du), dgap the change
    # of ln(gamma_lo/gamma_hi) with the box, d gap/du the secant of the two both-branch rows
    # closest to the root
    fac = fac or BOX_CHECK_FAC
    S = res["S"]
    Lx, _ = box(S)
    both = [r for r in res["rows"] if r["lo"] is not None and r["hi"] is not None
            and np.isfinite(r["gap"])]
    row = both[-1]
    out = dict(alpha=row["alpha"], fac=fac, nx=NX_DENSE, nx_big=int(round(NX_DENSE*fac)),
               Lx=Lx, Lx_big=float(np.round(Lx*fac, 6)))
    for which in ("lo", "hi"):
        b = row[which]
        g1 = growth(gpoint(S, b["kmax"], row["alpha"], NX_DENSE, Lx, b["gmax"], root=root,
                           log=log, compute=compute, dense_first=True))
        g2 = growth(gpoint(S, b["kmax"], row["alpha"], out["nx_big"], out["Lx_big"], g1,
                           root=root, log=log, compute=compute, dense_first=True))
        out[which] = (g1, g2)
    dgap = float(np.log(out["lo"][1]/out["lo"][0]) - np.log(out["hi"][1]/out["hi"][0]))
    near = sorted(both, key=lambda r: abs(r["gap"]))[:2]
    slope = (float((near[0]["gap"] - near[1]["gap"])/(near[0]["u"] - near[1]["u"]))
             if len(near) == 2 and near[0]["u"] != near[1]["u"] else np.nan)
    u = res["u"] - dgap/slope
    out.update(dgap=dgap, slope=slope, u=float(u), alpha_c=alpha_of_u(u) if np.isfinite(u)
               else np.nan)
    log(f"  alpha_c box check S={S:.0e}: gamma(Lx x{fac:g})/gamma(Lx) - 1 = "
        f"{out['lo'][1]/out['lo'][0] - 1:+.2e} (lo), {out['hi'][1]/out['hi'][0] - 1:+.2e} (hi); "
        f"1 - alpha_c^2 {np.exp(res['u']):.5e} -> {np.exp(u):.5e}")
    return out


def boundary_results(root=None, log=_quiet, compute=False, S_list=None):
    root = root or DATA_ROOT
    out = {}
    for S in (S_list or S_BOUNDARY):
        try:
            res = alpha_c_brent(S, *boundary_seeds(S), tol=0.02, root=root, log=log,
                                compute=compute)
        except (RuntimeError, KeyError) as e:
            _fail(log, f"alpha_c S={S:.0e}", e)
            continue
        out[S] = res
        log(f"  alpha_c(S={S:.0e}) = {res['alpha']:.8f}, 1 - alpha_c^2 = {np.exp(res['u']):.5e}")
    return out


STRUCT_KSPLIT = 0.4          # S = 1e4: low-k maxima at ka 0.17-0.27, high-k at 0.6-0.75


def struct_results(root=None, log=_quiet, compute=False):
    # both branches' k_max solutions at STRUCT_POINTS (whichever exist), for eigenfunctions
    root = root or DATA_ROOT
    out = {}
    for (S, al) in STRUCT_POINTS:
        for which, k0 in (("low", 0.2*(1e4/S)**0.3),
                          ("high", R.theory_kmax(S)[0]*max(1 - al*al, 1e-3)**(-1/7))):
            try:
                r = kmax_branch(S, al, k0, root=root, log=log, compute=compute)
            except (RuntimeError, KeyError) as e:
                _fail(log, f"struct {which} S={S:.0e} alpha={al}", e)
                continue
            if (r["kmax"] - STRUCT_KSPLIT)*(1 if which == "high" else -1) < 0:
                log(f"  struct {which} S={S:.0e} alpha={al}: Newton reached the other maximum "
                    f"(k* = {r['kmax']:.4f}); this branch has no maximum here")
                continue
            out[(S, al, which)] = r
            rec = r["rec"]
            log(f"  struct {which:4s} S={S:.0e} alpha={al}: k* = {r['kmax']:.6f} gamma* = "
                f"{r['gmax']:.6e} psi0 = {float(rec['psi0']):.3f} x_rms_E = "
                f"{float(rec['x_rms_E']):.3f} R/M/C = {float(rec['b_R']):+.3f}/"
                f"{float(rec['b_M']):+.3f}/{float(rec['b_C']):+.3f}")
    return out


def box_results(root=None):
    return {pt: box_study(*pt, root=root or DATA_ROOT, log=_quiet, compute=False)
            for pt in BOX_POINTS}


def julia_results(root=None):
    out = {}
    for pt, rows in box_results(root).items():
        if ("nx", 1.0) not in rows:
            continue
        try:
            out[pt] = julia_table(*pt, growth(rows[("nx", 1.0)]), root=root, log=_quiet,
                                  compute=False)
        except KeyError:
            pass
    return out


def pmscan_results(root=None, log=_quiet, compute=False):
    # {(S, alpha, Pm): scan rows}: gamma(k) with viscosity nu = Pm/S
    return {(S, al, pm): scan(S, al, root=root or DATA_ROOT, log=log, compute=compute, nu=pm/S)
            for (S, al) in PM_SCAN for pm in PM_SCAN_LIST}


def scan_results(root=None):
    return {(S, al): scan(S, al, root=root or DATA_ROOT, log=_quiet, compute=False)
            for S in S_SCAN for al in ALPHA_SCAN}


if __name__ == "__main__":
    wall = float(sys.argv[1]) if len(sys.argv) > 1 else 6*3600.0
    args = sys.argv[2:]
    S_list = None
    if args and args[0].startswith("S="):          # S filter (boundary phase only)
        S_list = tuple(float(s) for s in args[0][2:].split(","))
        args = args[1:]

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    log(f"profile {PROFILE}, root {DATA_ROOT}")
    make_data(wall_budget=wall, phases=tuple(args) or None, log=log, S_list=S_list)
