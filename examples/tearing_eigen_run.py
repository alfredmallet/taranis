# Theory, eigen-solves and resumable data generation for examples/tearing-eigen.ipynb:
# plans/AUTODIFF_PLAN.md rung 1a -- no-shear resistive tearing of Psi0 = sech^2(x) against
# EXACT theory, through the autodiff eigen-harness taranis/stability.py (rungs 0/0b).
#
# Setup (every number below): dims=2, nu = 0 (diss = (0, eta)), hyper = 1, a = v_Ay = 1,
# S = 1/eta, Psi0 = sech^2(x - Lx/2) (periodic images summed, so the box sees no kink),
# Phi0 = 0, Ly = 2*pi/k, ny = 8, iky = 1, fp64. b_perp = z x grad psi, so B_y = dPsi0/dx =
# f(x) = -2 tanh(x) sech^2(x), f'(0) = -2: every layer formula uses k|f'(0)| = 2k.
#
# THEORY (all of it independent of taranis; derivations and checks in the notebook):
#   * outer region: psi'' = (k^2 + f''/f) psi = (k^2 + 4 - 12 sech^2 x) psi (Poschl-Teller
#     l = 3), Delta'(k) = 2(5 - k^2)(k^2 + 3)/(k^2 sqrt(k^2 + 4)) -- verified by shooting
#     (dprime_shoot), marginal at k = sqrt(5).
#   * inner layer: the exact (nonconstant-psi) resistive dispersion relation of Coppi,
#     Galvao, Pellat, Rosenbluth & Rutherford 1976 / Ara et al. 1978, in the form of
#     Boldyrev & Loureiro 2018 (J. Phys. Conf. Ser. 1100, 012003; arXiv:1809.00453) eq. (60):
#         -(pi/8) beta^(3/2) Gamma((beta-1)/4)/Gamma((beta+5)/4) = lambda Delta',
#     lambda = gamma/(k|f'|), eta~ = eta/(k|f'|), beta^2 = lambda^3/eta~ (their eqs. 6-7 with
#     f'(0) = 1; k -> k|f'(0)| is the exact rescaling of their inner eqs. 26-27 for f ~ f'x).
#     With Lam = gamma/(eta^(1/3)(k|f'|)^(2/3)) (so beta = Lam^(3/2)) it reads
#         Delta' delta_eta = F(Lam) = -(pi/8) Lam^(5/4) Gamma((Lam^(3/2)-1)/4)/Gamma((Lam^(3/2)+5)/4),
#     delta_eta = (eta/(k|f'|))^(1/3). F is verified here against an independent numerical
#     solution of the inner-layer ODEs (inner_F_numeric); its FKR (Lam -> 0) and Coppi
#     (Lam -> 1) limits are verified numerically in the notebook.
#
# SOLVER: the ky = k column of J = L + N'(x0) (stability.ky_block_matrix / ky_block_operator).
# Small grids: eig_dense (the whole spectrum -- what certifies "fastest mode"). Large grids:
# stability.shift_invert (k=1; k=2 for the two-sheet pair) on the MATRIX-FREE block, its GMRES
# solves preconditioned by a sparse LU of the block's band (the sech^2 spectrum decays like
# e^(-pi q/2): couplings beyond ~3.4*Lx kx indices are below 1e-12 and are dropped from the
# PRECONDITIONER only); the band is recovered from ~2P operator probes by colouring, no dense
# matrix is ever formed, and every returned pair passes the harness's own residual gate
# against the exact operator. The ladders (ladder, _ladder_family) double nx until successive
# eigenvalues agree to 1e-7, sigma continued 10% above the previous rung.
#
# make_data() is resumable and wall-clock bounded like examples/gdi_2d_run.py: every
# eigen-solve is cached as one small .npz under DATA_ROOT, keyed by its configuration and its
# TARGET (not by sigma: a target names one eigenvalue), so a repeated call only computes what
# is missing. `python examples/tearing_eigen_run.py` runs it to completion: ~2.6 h of solves
# (three processes on a loaded M1 laptop took ~2.5 h wall), 75 MiB; the notebook then re-reads
# everything in ~10 s.
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import hashlib
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gamma as Gfun

import jax.numpy as jnp

import taranis as jr
from taranis import stability as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "tearing-eigen")
FPRIME = 2.0          # |f'(0)| for Psi0 = sech^2 x
SQRT5 = np.sqrt(5.0)


# ============================================================ theory: outer region

def dprime_sech2(ka):
    # exact Delta'a of Psi0 = sech^2 x (Poschl-Teller l = 3 ladder solution)
    q = np.asarray(ka, dtype=float)**2
    return 2.0*(5.0 - q)*(q + 3.0)/(q*np.sqrt(q + 4.0))


def fpp_over_f_sech2(x):
    # f''/f for f = -2 tanh sech^2 -- 4 - 12 sech^2 x (checked numerically in the notebook)
    return 4.0 - 12.0/np.cosh(x)**2


def dprime_shoot(ka, X=40.0, pot=fpp_over_f_sech2):
    # Delta' by shooting the outer equation psi'' = (k^2 + V(x)) psi inward from x = X, where
    # psi = exp(-sqrt(k^2 + V(X)) x), to x = 0+: Delta' = 2 psi'(0+)/psi(0) (tearing parity:
    # psi even about the sheet, the jump in psi' is the whole of Delta')
    k2 = float(ka)**2
    kap = np.sqrt(k2 + pot(X))
    sol = solve_ivp(lambda x, y: [y[1], (k2 + pot(x))*y[0]], [X, 0.0], [1.0, -kap],
                    rtol=1e-13, atol=1e-300, method="DOP853")
    return 2.0*sol.y[1, -1]/sol.y[0, -1]


# ============================================================ theory: inner layer

def inner_F(lam):
    # Boldyrev & Loureiro 2018 eq. (60) in Lam = gamma/(eta^(1/3)(k f')^(2/3)):
    # Delta' delta_eta = F(Lam); positive and increasing on 0 < Lam < 1, -> inf as Lam -> 1
    lam = np.asarray(lam, dtype=float)
    b = lam**1.5
    return -(np.pi/8.0)*lam**1.25*Gfun((b - 1.0)/4.0)/Gfun((b + 5.0)/4.0)


C_FKR = (np.pi/2.0)*Gfun(0.75)/Gfun(1.25)       # F(Lam) -> C_FKR Lam^(5/4) as Lam -> 0 (2.1236)


def _inner_bvp(Q, smax):
    # the inner layer in s = x/delta_eta (derived in the notebook from the linearized RMHD
    # layer equations; the same system as B&L eqs. 26-27 with eta = 1):
    #   psi'' = Q psi - s u,   u'' = -s psi + s^2 u/Q,
    # psi even (psi'(0) = 0), u odd (u(0) = 0), psi(0) = 1, and at s = smax the algebraic
    # (non-exponential) branch u = Q psi/s. Far field psi -> c0 + c1 s.
    def ode(s, y):
        return np.vstack([y[1], Q*y[0] - s*y[2], y[3], -s*y[0] + s*s*y[2]/Q])

    def bc(ya, yb):
        return np.array([ya[1], ya[2], ya[0] - 1.0, yb[2] - Q*yb[0]/smax])
    s = np.linspace(0.0, smax, 4001)
    y0 = np.zeros((4, s.size))
    y0[0] = 1.0
    y0[2] = Q*s/(1.0 + s*s)
    r = solve_bvp(ode, bc, s, y0, tol=1e-10, max_nodes=400000)
    if r.status != 0:
        raise RuntimeError(f"inner BVP failed at Q={Q}, smax={smax}: {r.message}")
    return r


def inner_F_numeric(Q, smaxes=(40.0, 80.0)):
    # Delta' delta_eta from the numerical inner solution: 2 c1/c0 of the far-field
    # psi = c0 + c1 s. The truncation error is O(1/smax^2) (measured), removed by Richardson
    # extrapolation over the two smax values. Returns (extrapolated, raw values).
    vals = []
    for smax in smaxes:
        r = _inner_bvp(Q, smax)
        y = r.sol(smax)
        c1 = y[1]
        c0 = y[0] - smax*c1
        vals.append(2.0*c1/c0)
    s0, s1 = smaxes
    ext = (s1**2*vals[1] - s0**2*vals[0])/(s1**2 - s0**2)
    return ext, vals


def inner_profile(Q, smax=40.0, s=None):
    # the exact inner solution (psi, psi'') on s >= 0, psi(0) = 1
    r = _inner_bvp(Q, smax)
    s = np.linspace(0.0, smax, 20001) if s is None else s
    y = r.sol(s)
    return s, y[0], Q*y[0] - s*y[2]


def lam_of(Ddelta):
    # invert F: the Lam in (0, 1) with F(Lam) = Delta' delta_eta (> 0)
    return brentq(lambda l: inner_F(l) - Ddelta, 1e-14, 1.0 - 1e-15, xtol=1e-16, rtol=1e-15)


def gamma_theory(ka, S, fprime=FPRIME, dprime=dprime_sech2):
    # growth rate of the exact dispersion relation (a = 1, eta = 1/S); nan if Delta' <= 0
    ka = float(ka)
    Dp = float(dprime(ka))
    if Dp <= 0:
        return np.nan
    kb = ka*fprime
    eta = 1.0/S
    return lam_of(Dp*(eta/kb)**(1.0/3.0))*eta**(1.0/3.0)*kb**(2.0/3.0)


def gamma_fkr(ka, S, fprime=FPRIME, dprime=dprime_sech2):
    # FKR (constant-psi) limit: [Gamma(1/4)/(2 pi Gamma(3/4))]^(4/5) Delta'^(4/5) eta^(3/5) (k f')^(2/5)
    return C_FKR**-0.8*dprime(ka)**0.8*S**-0.6*(ka*fprime)**0.4


def gamma_coppi(ka, S, fprime=FPRIME):
    # Coppi (Delta' -> inf) limit: eta^(1/3) (k f')^(2/3)
    return S**(-1.0/3.0)*(ka*fprime)**(2.0/3.0)


def theory_max(S, fprime=FPRIME, dprime=dprime_sech2, bracket=(1e-3, SQRT5 - 1e-6)):
    # (k_max, gamma_max) of the exact dispersion relation, by bounded maximisation in log k
    f = lambda lk: -gamma_theory(np.exp(lk), S, fprime, dprime)
    r = minimize_scalar(f, bounds=np.log(bracket), method="bounded",
                        options=dict(xatol=1e-10, maxiter=500))
    return float(np.exp(r.x)), float(-r.fun)


def delta_fkr(gam, ka, S, fprime=FPRIME):
    # constant-psi inner width (B&L eq. 58 with k -> k f'): (gamma eta/(k f')^2)^(1/4)
    return (gam/S/(ka*fprime)**2)**0.25


def delta_coppi(gam, ka, fprime=FPRIME):
    # nonconstant-psi inner width (B&L eq. 68): gamma/(k f')
    return gam/(ka*fprime)


def width_quarter(x, d2):
    # the paper's (JPP 2025 §7) layer width: full width of the region around the sheet where
    # Re psi'' >= max/4 (the contiguous interval containing the maximum; linear interpolation)
    y = np.real(d2)
    i = int(np.argmax(y))
    thr = 0.25*y[i]
    j = i
    while j < len(y) - 1 and y[j + 1] >= thr:
        j += 1
    k = i
    while k > 0 and y[k - 1] >= thr:
        k -= 1
    xr = x[j] + (x[j + 1] - x[j])*(y[j] - thr)/(y[j] - y[j + 1]) if j < len(y) - 1 else x[j]
    xl = x[k] - (x[k] - x[k - 1])*(y[k] - thr)/(y[k] - y[k - 1]) if k > 0 else x[k]
    return xr - xl


def theory_width(ka, S, fprime=FPRIME, dprime=dprime_sech2, smax=40.0):
    # the quarter-maximum width of psi'' in the EXACT inner solution at this (k, S)
    gam = gamma_theory(ka, S, fprime, dprime)
    eta = 1.0/S
    Q = gam/(eta**(1.0/3.0)*(ka*fprime)**(2.0/3.0))
    s, _, d2 = inner_profile(Q, smax)
    xs = np.concatenate([-s[::-1], s[1:]])
    d2s = np.concatenate([d2[::-1], d2[1:]])
    return width_quarter(xs, d2s)*(eta/(ka*fprime))**(1.0/3.0)


def gamma_disp(k, eta, fprime, Dprime):
    # the exact dispersion relation in physical units (any profile): k, eta, |f'(x_s)|, Delta'
    if not Dprime > 0:
        return np.nan
    kb = k*fprime
    return lam_of(Dprime*(eta/kb)**(1.0/3.0))*eta**(1.0/3.0)*kb**(2.0/3.0)


# ============================================================ the Harris-like periodic family

def harris_fprime(a):
    # |f'(0)| of B_y = tanh(sin x/a)/tanh(1/a)
    return 1.0/(a*np.tanh(1.0/a))


def harris_pot(x, a):
    # f''/f for f = tanh(sin x/a)/tanh(1/a) (regular at x = 0, where it is -(2/a^2 + 1))
    u = np.sin(x)/a
    if abs(u) < 1e-8:
        return -(2.0/a**2 + 1.0)
    return -(1.0/np.cosh(u)**2)*(2*np.cos(x)**2/a**2 + np.sin(x)/(a*np.tanh(u)))


def dprime_harris(k, a, parity):
    # Delta' (physical units) of the periodic two-sheet profile for the tearing-parity mode
    # (psi even about the sheet at x = 0) that is EVEN (parity=+1) or ODD (-1) about the
    # midpoint x = pi/2 (the profile is symmetric about it): shoot psi'' = (k^2 + f''/f) psi
    # from pi/2 to 0+
    y0 = [1.0, 0.0] if parity > 0 else [0.0, 1.0]
    sol = solve_ivp(lambda x, y: [y[1], (k*k + harris_pot(x, a))*y[0]], [np.pi/2, 0.0], y0,
                    rtol=1e-12, atol=1e-14, method="DOP853")
    return 2.0*sol.y[1, -1]/sol.y[0, -1]


def dprime_harris_local(ka):
    # an isolated Harris sheet: Delta'a = 2(1/ka - ka)
    return 2.0*(1.0/ka - ka)


# ============================================================ equilibria

def lx_for(ka, lmin=20.0):
    # box: Lx >= 15/k (the phi far field decays like e^(-k|x|); measured in the notebook)
    return max(15.0/ka, lmin)


def psi0_sech2(x, Lx):
    # sech^2 centred at Lx/2 plus its two nearest periodic images: smooth and periodic
    # (written as 4 e^-2|u|/(1 + e^-2|u|)^2, which cannot overflow)
    def sech2(u):
        e = np.exp(-2.0*np.abs(u))
        return 4.0*e/(1.0 + e)**2
    return sum(sech2(x - Lx/2 - m*Lx) for m in (-1, 0, 1))


def psi0_harrislike(x, Lx, a):
    # B_y = tanh(sin x/a)/tanh(1/a) (period 2 pi, sheets at 0 and pi), Psi0 = int B_y dx by
    # FFT (B_y is analytic and has zero mean, so this is spectrally exact)
    n = len(x)
    by = np.tanh(np.sin(2*np.pi*x/Lx)/a)/np.tanh(1.0/a)
    kx = 2*np.pi*np.fft.fftfreq(n, d=Lx/n)
    bh = np.fft.fft(by)
    ph = np.zeros_like(bh)
    ph[1:] = bh[1:]/(1j*kx[1:])
    return np.real(np.fft.ifft(ph))


def make_params(S, ka, nx, Lx, a=1.0):
    # eta = a/S: S is the Lundquist number on the sheet half-width a (v_Ay = 1); k = ka/a
    k = ka/a
    return jr.Parameters(nx=int(nx), ny=8, Lx=float(Lx), Ly=2*np.pi/k, dims=2, cfl_safety=0.5,
                         forcing=False, eqpars={"diss": (0.0, a/S), "hyper": 1})


def make_x0(params, case, a=1.0, amp=1.0):
    # the equilibrium as a fields array: psi = amp*Psi0(x), phi = 0
    x = np.arange(params.nx)*params.Lx/params.nx
    if case == "sech2":
        p0 = psi0_sech2(x, params.Lx)
    elif case == "harris":
        p0 = psi0_harrislike(x, params.Lx, a)
    elif case == "cosx":                  # examples/tearing-mode-2D, tearing-growth-vs-k
        p0 = np.cos(2*np.pi*x/params.Lx)
    else:
        raise ValueError(case)
    p0 = jnp.asarray(amp*p0).reshape(1, -1, 1)
    return jr.initialize(lambda X, Y: jnp.stack([0*p0 + 0*Y, p0 + 0*Y]), params).fields


def sheet_index(case, nx):
    # grid index of the (first) resonant surface: sech^2 at Lx/2, Harris-like/cos x at 0
    return nx//2 if case == "sech2" else 0


# ============================================================ eigen-solves

def _eigfun(params, idx, v, i0=None):
    # eigenvector (kept entries of the ky column) -> (x, psi(x), phi(x), psi''(x)) in real
    # space, phase-normalized so psi is real and positive at index i0 (default: at max |psi|)
    nx = params.nx
    out = np.zeros((2, nx), dtype=complex)
    out[idx[:, 0], idx[:, 2]] = v
    kx = 2*np.pi*np.fft.fftfreq(nx, d=params.Lx/nx)
    psi, phi = np.fft.ifft(out[1]), np.fft.ifft(out[0])
    d2 = np.fft.ifft(-(kx**2)*out[1])
    i = int(np.argmax(np.abs(psi))) if i0 is None else i0
    ph = np.conj(psi[i])/abs(psi[i]) if abs(psi[i]) > 0 else 1.0
    x = np.arange(nx)*params.Lx/nx
    return x, psi*ph, phi*ph, d2*ph


def dense_spectrum(case, S, ka, nx, Lx, a=1.0, nkeep=16, amp=1.0):
    # full spectrum of the ky block (eig_dense); returns a dict with the top nkeep eigenvalues
    # by real part, their residuals, and the leading eigenfunction
    params = make_params(S, ka, nx, Lx, a)
    kg = jr.setup_kgrids(params)
    x0 = make_x0(params, case, a, amp)
    t0 = time.time()
    B, idx = st.ky_block_matrix(x0, kg, params, 1)
    r = st.eig_dense(B)
    x, psi, phi, d2 = _eigfun(params, idx, r.right[:, 0], sheet_index(case, nx))
    return dict(values=r.values[:nkeep], residuals=r.residuals[:nkeep], scale=r.scale,
                n=B.shape[0], seconds=time.time() - t0, x=x, psi=psi, phi=phi, d2=d2,
                all_values=r.values, vectors=r.right[:, :nkeep], idx=idx)


def _band_preconditioner(op, idx, nx, sigma, band):
    # the sparse banded part of (J - sigma I) from operator probes (J couples kx indices only
    # within circular distance `band`, up to entries below round-off), LU-factorised.
    # Columns are coloured by (field, ikx mod P) with P | nx and P >= 2 band + 1, so each row
    # of a probe's image is fed by exactly one column of that colour.
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
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    normA = float(abs(A).sum(axis=0).max())
    A = A - sigma*sp.identity(n, format="csc")
    lu = spla.splu(A.tocsc(), permc_spec="COLAMD")
    return spla.LinearOperator((n, n), matvec=lambda b: lu.solve(np.asarray(b).reshape(-1)),
                               dtype=complex), A.nnz, normA


def band_width(case, Lx, a=1.0):
    # kx index distance beyond which J's couplings are negligible for a PRECONDITIONER (the
    # exact operator is always what the residuals are measured against). The equilibrium
    # spectrum decays like exp(-w q); measured for sech^2 (S=1e4, ka=0.1, nx=2048, Lx=150):
    # dropping couplings beyond q = 21.4 leaves ||I - M(J - sigma)|| = 5e-9. The Harris-like
    # profile's nearest complex singularity is at Im x = asinh(pi a/2) -> w ~ that.
    if case == "cosx":
        return int(np.ceil(Lx/(2*np.pi))) + 2     # cos x couples kx to kx +- 1 only
    qcut = 21.0 if case == "sech2" else 30.0/np.arcsinh(np.pi*a/2)
    return int(np.ceil(qcut*Lx/(2*np.pi))) + 2


def si_solve(case, S, ka, nx, Lx, sigma, a=1.0, k=1, tol=1e-12, amp=1.0, lam_hint=None):
    # shift-invert about sigma on the matrix-free ky block, band-LU-preconditioned GMRES.
    # The GMRES tolerance is set from the attainable accuracy of a shifted solve, whose
    # relative residual cannot go below ~eps ||J||/|lambda - sigma| (lam_hint: the expected
    # eigenvalue; default |lambda - sigma| ~ 0.03|sigma|).
    params = make_params(S, ka, nx, Lx, a)
    kg = jr.setup_kgrids(params)
    x0 = make_x0(params, case, a, amp)
    t0 = time.time()
    op, idx = st.ky_block_operator(x0, kg, params, 1)
    M, nnz, normA = _band_preconditioner(op, idx, nx, sigma, band_width(case, Lx, a))
    tb = time.time() - t0
    dist = abs(sigma - lam_hint) if lam_hint is not None else 0.03*abs(sigma)
    gmres_rtol = float(np.clip(300*np.finfo(float).eps*normA/max(dist, 1e-300), 1e-10, 1e-6))
    r = st.shift_invert(op, sigma, k=k, tol=tol, M=M, gmres_rtol=gmres_rtol, restart=30,
                        maxiter=20)
    # the plain relative residual ||J v - lam v||/(|lam| ||v||), next to the harness's
    v = r.right[:, 0]
    rel = np.linalg.norm(op.matvec(v) - r.values[0]*v)/(abs(r.values[0])*np.linalg.norm(v))
    x, psi, phi, d2 = _eigfun(params, idx, v, sheet_index(case, nx))
    # two-sheet cases: psi(pi)/psi(0) of every returned vector (+1 even, -1 odd about pi/2)
    ratios = [complex(_eigfun(params, idx, r.right[:, j], 0)[1][nx//2]) for j in range(k)] \
        if case != "sech2" else [np.nan]*k
    return dict(values=r.values, residuals=r.residuals, rel_residual=rel, scale=r.scale,
                n=len(idx), seconds=time.time() - t0, build_seconds=tb, nnz=nnz,
                gmres_rtol=gmres_rtol, sheet_ratio=np.asarray(ratios),
                stats=r.stats, x=x, psi=psi, phi=phi, d2=d2)


# ============================================================ cache

def _key(**cfg):
    txt = json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) and
                          not isinstance(v, bool) else v) for k, v in sorted(cfg.items())},
                     sort_keys=True)
    return hashlib.sha1(txt.encode()).hexdigest()[:16], txt


def _profile(x, psi, phi, d2, i0, Lx, half=1.5, nmax=3001):
    # the eigenfunction near the sheet (|x - x_s| <= half) at native resolution (<= nmax
    # points), plus the whole box decimated to <= 2048 points; widths are measured here
    nx = len(x)
    xs = x[i0]
    dx = (x - xs + Lx/2) % Lx - Lx/2
    near = np.nonzero(np.abs(dx) <= half)[0]
    near = near[np.argsort(dx[near])]
    step = max(1, len(near)//nmax)
    near = near[::step]
    dec = np.arange(0, nx, max(1, nx//2048))
    order = np.argsort(dx[dec])
    return dict(xn=dx[near], psin=psi[near], phin=phi[near], d2n=d2[near],
                xf=dx[dec][order], psif=psi[dec][order], phif=phi[dec][order])


def _save(path, out):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def cached(kind, case, S, ka, nx, Lx, sigma=None, a=1.0, amp=1.0, target="lead", root=None,
           nkeep=16, lam_hint=None, nev=1, attempt=0):
    # The key is the configuration and the TARGET, not sigma: a target names one eigenvalue.
    # attempt > 0 (a retry with a moved sigma after a rejected pair) gets its own entry.
    # one eigen-solve, computed once and cached as DATA_ROOT/solves/<hash>.npz.
    # kind "dense" (eig_dense of the whole block) or "si" (shift_invert, k=1, about sigma).
    root = root or DATA_ROOT
    os.makedirs(os.path.join(root, "solves"), exist_ok=True)
    extra_key = {} if nev == 1 else {"nev": nev}
    if attempt:
        extra_key["attempt"] = attempt
    h, txt = _key(kind=kind, case=case, S=S, ka=ka, nx=nx, Lx=Lx, a=a, amp=amp, target=target,
                  **extra_key)
    path = os.path.join(root, "solves", h + ".npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    t0 = time.time()
    if kind == "dense":
        r = dense_spectrum(case, S, ka, nx, Lx, a, nkeep, amp)
        values, residuals, rel = r["values"], r["residuals"], np.nan
        extra = dict(n_pos=int(np.sum(r["all_values"].real > 1e-10*r["scale"])))
    else:
        r = si_solve(case, S, ka, nx, Lx, sigma, a, k=nev, amp=amp, lam_hint=lam_hint)
        values, residuals, rel = r["values"], r["residuals"], r["rel_residual"]
        extra = dict(sigma=complex(sigma), nnz=r["nnz"], build_seconds=r["build_seconds"],
                     solves=r["stats"]["solves"], gmres_iters=r["stats"]["gmres_iters"],
                     gmres_rtol=r["gmres_rtol"], sheet_ratio=r["sheet_ratio"])
    i0 = sheet_index(case, nx)
    prof = _profile(r["x"], r["psi"], r["phi"], r["d2"], i0, Lx,
                    half=1.5 if case == "sech2" else (0.6 if case == "harris" else 1.0))
    width = width_quarter(prof["xn"], prof["d2n"])
    out = dict(cfg=txt, kind=kind, case=case, S=S, ka=ka, nx=nx, Lx=Lx, a=a, amp=amp,
               target=target, values=np.asarray(values), residuals=np.asarray(residuals),
               rel_residual=rel, scale=r["scale"], n=r["n"], seconds=time.time() - t0,
               width=width, **prof, **extra)
    _save(path, out)
    return out


# ============================================================ ladders

def _accept(lam, target, sigma):
    if target in ("lead", "tearing", "pair"):
        # real (Im at round-off: relative, or absolute for the tiniest near-marginal modes)
        return abs(lam.imag) <= 1e-8*abs(lam) + 1e-12 and lam.real > 0
    return True


def ladder(case, S, ka, Lx, nx0, sigma0, nxcap=32768, target="tearing", a=1.0, amp=1.0,
           tol=1e-7, root=None, log=print, nev=1):
    # shift-invert at nx0, 2 nx0, ... (sigma continued from the previous rung, 10% above it)
    # until successive eigenvalues agree to tol (relative) or nx passes nxcap. Returns a dict:
    # the rung eigenvalues, the final value, its convergence error |lam_N - lam_{N-1}|, the
    # converged flag and the final rung's record (eigenfunction, residuals).
    rungs, recs = [], []
    sigma, nx, hint = complex(sigma0), int(nx0), None
    while nx <= nxcap:
        rec = None
        for attempt, off in enumerate((0.0, 0.1, 0.5, -0.05)):
            sig = sigma + off*abs(sigma)
            try:
                rec = cached("si", case, S, ka, nx, Lx, sig, a, amp, target, root,
                             lam_hint=hint, nev=nev, attempt=attempt)
            except RuntimeError as e:
                log(f"    si failed at nx={nx}, sigma={sig:.4g}: {str(e)[:120]}")
                continue
            # a pair whose plain relative residual is O(1) is a round-off member of the
            # nu = 0 null cluster, never the target
            if _accept(complex(rec["values"][0]), target, sig) and rec["rel_residual"] < 1e-3:
                break
            log(f"    rejected {complex(rec['values'][0]):.4e} at nx={nx} (target {target})")
            rec = None
        if rec is None:
            break
        lam = complex(rec["values"][0])
        rungs.append((nx, lam))
        recs.append(rec)
        log(f"    {case} S={S:.0e} ka={ka:g} nx={nx}: {lam:.10e}  res={rec['rel_residual']:.1e}"
            f"  ({rec['seconds']:.0f}s)" + (f"  all: {rec['values']}" if nev > 1 else ""))
        if len(rungs) >= 2 and abs(lam - rungs[-2][1]) <= tol*abs(lam) and (
                nev == 1 or np.max(np.abs(np.sort_complex(rec["values"])
                                          - np.sort_complex(recs[-2]["values"])))
                <= tol*abs(lam)):
            break
        sigma, hint = lam + 0.1*abs(lam), lam
        nx *= 2
    if not rungs:
        return dict(ok=False, nx=[], lam=[], value=np.nan, err=np.inf, converged=False, rec=None)
    err = abs(rungs[-1][1] - rungs[-2][1]) if len(rungs) >= 2 else np.inf
    return dict(ok=True, nx=[r[0] for r in rungs], lam=[r[1] for r in rungs],
                value=rungs[-1][1], err=err, converged=err <= tol*abs(rungs[-1][1]),
                rec=recs[-1])


# ============================================================ configurations

KA_MAIN = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.2)
S_MAIN = (1e3, 1e4, 1e5, 1e6)
S_FINE = (3e3, 3e4, 3e5)          # extra S points for KA_FINE
KA_FINE = (0.3, 1.0)
S_MAX = (1e3, 1e4, 1e5, 1e6)
KMAX_FACTORS = (0.7, 0.85, 1.0, 1.18, 1.4)
S_MARG = (1e3, 1e4, 1e5)
KA_MARG = (2.10, 2.15, 2.18, 2.20, 2.21, 2.22, 2.225, 2.23, 2.233, 2.235)
KA_ABOVE = (2.24, 2.26)                  # above sqrt 5: dense spectra only
COSX_KMARG = (0.97, 0.98, 0.99, 0.995, 0.998)
NX_CAP = 32768
DENSE_SEED_KA = 1.9
HARRIS_A = 0.25
HARRIS_S = (1e3, 1e4)
HARRIS_KA = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9)
LX_KA = (0.1, 0.3, 1.0)
LX_C = (6.0, 9.0, 12.0, 15.0, 20.0, 30.0)     # Lx = c/k (and >= 10)


def _nx_dense(Lx):
    return 1024 if Lx <= 40 else 2048


def _ladder_family(S_list, ka, root, log, deadline, nxcap=NX_CAP, target="tearing",
                   stop_unconverged=True):
    # ascending-S ladders at one ka: S = S_list[0] seeded by a dense spectrum, each later S by
    # the previous converged value scaled with the theory ratio, starting at its final nx/2
    Lx = lx_for(ka)
    out = {}
    prev = None
    for S in S_list:
        if time.time() > deadline:
            return out, False
        if prev is None or ka >= DENSE_SEED_KA:
            # near the marginal point the finite-S suppression is too strong for the theory
            # scaling below to seed sigma: every S there starts from a dense spectrum
            nxd = _nx_dense(Lx)
            d = cached("dense", "sech2", S, ka, nxd, Lx, root=root)
            sigma0 = complex(d["values"][0])
            sigma0 = sigma0 + 0.03*abs(sigma0)
            nx0 = nxd
        else:
            Sp, res = prev
            gt, gtp = gamma_theory(ka, S), gamma_theory(ka, Sp)
            scale = gt/gtp if np.isfinite(gt) and np.isfinite(gtp) else (Sp/S)**0.5
            sigma0 = res["value"].real*scale*1.03
            nx0 = max(_nx_dense(Lx), res["nx"][-1]//2)
        res = ladder("sech2", S, ka, Lx, nx0, sigma0, nxcap=nxcap, target=target, root=root,
                     log=log)
        out[S] = res
        if not res["converged"] and stop_unconverged:
            log(f"  ka={ka}: not converged at S={S:.0e} (nx cap {nxcap}); stopping this ka")
            break
        prev = (S, res)
    return out, True


def make_data(root=None, wall_budget=6*3600.0, phases=None, log=print):
    # resumable: every eigen-solve is cached; returns True when every phase is complete
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    phases = phases or ("main", "max", "marginal", "lx", "surprise", "cosx", "harris")
    done = True
    if "main" in phases:
        for ka in KA_MAIN:
            Ss = sorted(S_MAIN + (S_FINE if ka in KA_FINE else ()))
            _, ok = _ladder_family(Ss, ka, root, log, deadline)
            done &= ok
    if "max" in phases:
        for S in S_MAX:
            k0, _ = theory_max(S)
            for fac in KMAX_FACTORS:
                ka = round(k0*fac, 4)
                chain = [s_ for s_ in S_MAX if s_ <= S]
                _, ok = _ladder_family(chain, ka, root, log, deadline)
                done &= ok
    if "marginal" in phases:
        done &= _marginal(root, log, deadline)[1]
    for name, fn in (("lx", _lx), ("surprise", _surprise), ("cosx", _cosx),
                     ("harris", _harris)):
        if name in phases:
            done &= fn(root, log, deadline)[1]
    return bool(done) and time.time() <= deadline


# ============================================================ the surprise mode, cos x, Harris

def lx_points(ka):
    return sorted({max(c/ka, 10.0) for c in LX_C})


def _marginal(root, log, deadline):
    # (a) the sech^2 tearing branch on a fine ka grid below sqrt 5 (kept going past an
    # unconverged S: near Delta' = 0 the required nx grows without bound); (b) dense spectra
    # above sqrt 5; (c) the same branch for the cos x pair below its marginal k = 1, continued
    # in k (its exact k = 1 mode is checked separately, cosx_exact_mode).
    out = {"sech2": {}, "above": {}, "cosx": {}}
    for ka in KA_MARG:
        if time.time() > deadline:
            return out, False
        out["sech2"][ka] = _ladder_family(S_MARG, ka, root, log, deadline,
                                          stop_unconverged=False)[0]
    for S in (1e3, 1e4):
        for ka in KA_ABOVE:
            if time.time() > deadline:
                return out, False
            out["above"][(S, ka)] = cached("dense", "sech2", S, ka, 2048, 20.0, root=root)
    for S in S_MARG:
        prev = []                 # (k, converged gamma): the branch is continued in k
        for k in COSX_KMARG:
            if time.time() > deadline:
                return out, False
            if not prev:
                d = cached("dense", "cosx", S, k, 1024, 2*np.pi, root=root)
                guess, nx0 = complex(d["values"][0]).real, 1024
            else:
                # start where the previous k converged: a coarse first rung under-resolves
                # the shrinking layer and hands the continuation a wrong sigma
                (k1, g1), nx0 = prev[-1], max(1024, out["cosx"][(S, prev[-1][0])]["nx"][-1]//2)
                guess = g1 + (g1 - prev[-2][1])*(k - k1)/(k1 - prev[-2][0]) if len(prev) > 1 \
                    else 0.6*g1
                guess = max(guess, 0.2*g1)
            r = ladder("cosx", S, k, 2*np.pi, nx0, guess*1.1, nxcap=16384, target="tearing",
                       root=root, log=log)
            out["cosx"][(S, k)] = r
            if not r["ok"]:
                break
            prev.append((k, r["value"].real))
    return out, True


def _lx(root, log, deadline):
    # box-size study at S = 1e3: the same mode at Lx = c/k, each converged in nx.
    # Returns ({ka: [(Lx, ladder result), ...]}, complete)
    out = {}
    for ka in LX_KA:
        ref, rows = None, []
        for Lx in lx_points(ka):
            if time.time() > deadline:
                return out, False
            nxd = 1024 if Lx <= 40 else 2048
            if ref is None:
                d = cached("dense", "sech2", 1e3, ka, nxd, Lx, root=root)
                sig = complex(d["values"][0])
            else:
                sig = ref
            r = ladder("sech2", 1e3, ka, Lx, nxd, sig + 0.03*abs(sig), nxcap=16384, root=root,
                       log=log)
            rows.append((Lx, r))
            ref = r["value"] if r["ok"] else ref
        out[ka] = rows
    return out, True


SURPRISE_GUESS = 6.8e-4 + 0.0203j      # rung 0 / Julia: S = 1e3, ka = 0.1, alpha = 0


def _surprise(root, log, deadline):
    # the oscillatory mode at S = 1e3, ka = 0.1: its box-size and grid convergence, then the
    # same mode followed in S by continuation; dense spectra at ka = 0.15, 0.2.
    # Returns ({"lx": {Lx: res}, "S": {S: res}, "dense": {ka: rec}}, complete)
    out = {"lx": {}, "S": {}, "dense": {}}
    for Lx in (50.0, 75.0, 150.0, 300.0):
        if time.time() > deadline:
            return out, False
        out["lx"][Lx] = ladder("sech2", 1e3, 0.1, Lx, 1024 if Lx <= 75 else 2048,
                               SURPRISE_GUESS, nxcap=16384, target="surprise", root=root, log=log)
    base = out["lx"][150.0]
    out["S"][1e3] = base
    for S in (3e3, 1e4, 3e4):
        if time.time() > deadline or not base["ok"]:
            return out, False
        base = ladder("sech2", S, 0.1, 150.0, base["nx"][-1]//2 if base["nx"][-1] > 2048 else
                      2048, base["value"], nxcap=16384, target="surprise", root=root, log=log)
        out["S"][S] = base
    for ka, Lx in ((0.15, 100.0), (0.2, 75.0)):
        if time.time() > deadline:
            return out, False
        out["dense"][ka] = cached("dense", "sech2", 1e3, ka, 2048, Lx, root=root)
    return out, True


# examples/tearing-growth-vs-k.ipynb (eta = 1e-3, Ly = 16 pi, 128 x 1024, fp64): mode m at
# k = m/8, fitted over [t_lo, t_nl = 286.5]; examples/tearing-mode-2D.ipynb (k = 1/2, Ly = 4 pi,
# 128 x 256, fp32). Both inviscid, psi0 = cos x in Lx = 2 pi, no equilibrium sustainment.
GVK_T_LO = (72.3, 45.9, 51.1, 67.0, 99.4, 167.8, 25.0)
GVK_T_NL = 286.5
GVK_GAMMA = (0.02065, 0.02901, 0.02932, 0.02465, 0.01832, 0.01195, 0.00581)
GVK_SYS = (0.00062, 0.00100, 0.00095, 0.00065, 0.00038, 0.00015, 0.00018)
TM2D_ETA = (5e-4, 1e-3, 2e-3, 4e-3)
TM2D_WIN = ((45.0, 312.0), (36.0, 222.0), (31.0, 164.0), (26.0, 127.0))
TM2D_GAMMA = (0.0174, 0.0249, 0.0339, 0.0437)


def cosx_points():
    # (label, eta, k, amplitude B'(tbar) = exp(-eta tbar), measured gamma, its error)
    pts = []
    for m in range(1, 8):
        tbar = 0.5*(GVK_T_LO[m - 1] + GVK_T_NL)
        pts.append((f"gvk m={m}", 1e-3, m/8.0, np.exp(-1e-3*tbar), GVK_GAMMA[m - 1],
                    GVK_SYS[m - 1]))
    for eta, (t0, t1), g in zip(TM2D_ETA, TM2D_WIN, TM2D_GAMMA):
        pts.append((f"tm2d eta={eta:g}", eta, 0.5, np.exp(-eta*0.5*(t0 + t1)), g, np.nan))
    return pts


def _cosx(root, log, deadline):
    # returns ({label: (res at B' = 1, res at B'(tbar))}, complete)
    out = {}
    for label, eta, k, amp, g, _ in cosx_points():
        pair = []
        for A in (1.0, amp):
            if time.time() > deadline:
                return out, False
            S = 1.0/eta
            d = cached("dense", "cosx", S, k, 128, 2*np.pi, amp=A, root=root)
            lam = complex(d["values"][0])
            pair.append(ladder("cosx", S, k, 2*np.pi, 128, lam + 0.03*abs(lam), nxcap=4096,
                               amp=A, target="tearing", root=root, log=log))
        out[label] = tuple(pair)
    return out, True


def _harris(root, log, deadline):
    # the two-sheet periodic family: a dense spectrum locates the mode pair, then ONE
    # shift-invert ladder with k = 2 about a sigma just above it converges both together (the
    # pair can be split by less than the continuation step at large k, where the sheets
    # decouple). Returns ({(S, ka): (dense record, ladder result)}, complete)
    a = HARRIS_A
    out = {}
    for S in HARRIS_S:
        for ka in HARRIS_KA:
            if time.time() > deadline:
                return out, False
            d = cached("dense", "harris", S, ka, 1024, 2*np.pi, a=a, root=root)
            lam = complex(d["values"][0])
            res = ladder("harris", S, ka, 2*np.pi, 1024, lam + 0.1*abs(lam), nxcap=16384, a=a,
                         target="pair", root=root, log=log, nev=2)
            out[(S, ka)] = (d, res)
    return out, True


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def main_results(root=None):
    # {ka: {S: ladder result}} for the main S x ka grid
    out = {}
    for ka in KA_MAIN:
        Ss = sorted(S_MAIN + (S_FINE if ka in KA_FINE else ()))
        out[ka] = _ladder_family(Ss, ka, root or DATA_ROOT, _quiet, np.inf)[0]
    return out


def max_results(root=None):
    # {S: [(ka, ladder result), ...]} around the theory k_max(S)
    out = {}
    for S in S_MAX:
        k0, _ = theory_max(S)
        rows = []
        for fac in KMAX_FACTORS:
            ka = round(k0*fac, 4)
            fam = _ladder_family([s_ for s_ in S_MAX if s_ <= S], ka, root or DATA_ROOT, _quiet,
                                 np.inf)[0]
            if S in fam:
                rows.append((ka, fam[S]))
        out[S] = rows
    return out


def marginal_results(root=None):
    # {"sech2": {ka: {S: res}}, "above": {(S, ka): dense}, "cosx": {(S, k): res}}
    return _marginal(root or DATA_ROOT, _quiet, np.inf)[0]


def cosx_exact_mode(S, nx=1024):
    # the exact k = 1 eigenmode of psi0 = cos x: psi = e^{iky} (the kx = 0 entry of the psi
    # field), phi = 0, lambda = -eta k^2. Returns (Rayleigh quotient, ||J e - lambda e||) for the
    # unit vector e on that entry, straight from the harness's matrix-free ky block.
    params = make_params(S, 1.0, nx, 2*np.pi)
    kg = jr.setup_kgrids(params)
    op, idx = st.ky_block_operator(make_x0(params, "cosx"), kg, params, 1)
    e = np.zeros(len(idx), dtype=complex)
    e[np.nonzero((idx[:, 0] == 1) & (idx[:, 2] == 0))[0][0]] = 1.0
    Je = op.matvec(e)
    lam = np.vdot(e, Je)
    return lam, float(np.linalg.norm(Je - lam*e))


def lx_results(root=None):
    return _lx(root or DATA_ROOT, _quiet, np.inf)[0]


def surprise_results(root=None):
    return _surprise(root or DATA_ROOT, _quiet, np.inf)[0]


def cosx_results(root=None):
    return _cosx(root or DATA_ROOT, _quiet, np.inf)[0]


def harris_results(root=None):
    return _harris(root or DATA_ROOT, _quiet, np.inf)[0]


def peak_fit(ks, gs):
    # (k_max, gamma_max) from a quadratic in ln k through the three largest-gamma points, and
    # the spread against a cubic through all points (the fit error)
    ks, gs = np.asarray(ks, float), np.asarray(gs, float)
    i = int(np.argmax(gs))
    i = min(max(i, 1), len(gs) - 2)
    sl = slice(i - 1, i + 2)

    def peak(c, lo, hi):
        pp = np.poly1d(c)
        xs = np.linspace(lo, hi, 20001)
        j = int(np.argmax(pp(xs)))
        return np.exp(xs[j]), pp(xs[j])
    lk = np.log(ks)
    k2, g2 = peak(np.polyfit(lk[sl], gs[sl], 2), lk[0], lk[-1])
    k3, g3 = peak(np.polyfit(lk, gs, 3), lk[0], lk[-1])
    return k2, g2, abs(k3 - k2), abs(g3 - g2)


if __name__ == "__main__":
    import sys
    while not make_data(wall_budget=float(sys.argv[1]) if len(sys.argv) > 1 else 6*3600.0):
        pass
