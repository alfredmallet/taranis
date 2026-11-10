# plans/AUTODIFF_PLAN.md rung 2 (science) gate: PARAMETER sensitivities of resistive tearing
# through the overrides seam -- d lambda/d eta (diss = (nu, eta) tangent (0, 1)), d lambda/d k
# (dparams={"Ly": -Ly/k}) -- and k_max by Newton on d gamma/dk = 0. The science lives in
# examples/tearing-kmax.ipynb (data: examples/tearing_rung2_run.py); this file is its fast
# standing gate and IMPORTS the data script (examples/ is put on sys.path; not a package).
# Setup as rungs 1b-ref/1b: dims=2, nu = 0, hyper = 1, S = 1/eta, Psi0 = sech^2 (+ 2 images),
# Phi0 = alpha Psi0, ky block, fp64. Small grids throughout.
#   (i)   theory helpers (numpy/scipy only): the ANALYTIC local exponents of the exact alpha = 0
#         dispersion relation, d ln gamma/d ln S and d ln gamma/d ln k, equal central differences
#         of gamma_theory; their FKR (-3/5, -6/5) and Coppi (-1/3, 2/3) limits; theory_kmax is
#         T.theory_max's maximum and a zero of the k-exponent; theory_kmax_slopes against a
#         difference of T.theory_max in ln S; the paper's exponent table from the n = 2 formulas.
#   (ii)  the one-point solver `sens_solve` (overrides Ly, matrix-free, band-LU shift-invert, left
#         vector, three sensitivities) against INDEPENDENT routes on the dense block: lambda equal
#         to eig_dense of a Parameters built with Ly = 2 pi/k (no overrides); dgamma/deta against
#         eigenvalues of Parameters with diss = (0, eta(1 +- h)) at O(h^2) -- and dgamma/dnu from
#         the same FD differs by far more than the tolerance (so an eta-only tangent is what
#         passes); dgamma/dk against Parameters with Ly = 2 pi/(k(1 +- h)); dgamma/dalpha against
#         equilibria alpha +- h. The exponent helpers of the record are the formulas applied to it.
#   (iii) pure helpers on synthetic inputs: s/k/alpha exponents on exact power laws; newton_kmax
#         on analytic gamma(k) (converges to the maximum superlinearly, uphill on a wrong-sign
#         curvature seed, step cap, a RELATIVE stop -- the same maximum at k0 = 1e-5 --, and
#         gamma* the quadratic model's maximum under a loose stop); kmax_solve's polish step on
#         the same analytic gamma; point's retry safeguard (refusals, null-cluster members and
#         complex pairs skipped, the key never sigma); curvature_seed's g/k^2 rescaling
#         predicts another S's theory curvature; implicit_slopes on an analytic
#         gamma(k, eta, alpha) with known k*(eta, alpha), dk*/dalpha included; fd_gate's control flow (steps, directions, the sensitivity's own
#         grid) with the solves replaced by an analytic lambda(k, eta); the continuation seeds of
#         kmax_alpha / kmax_S (k, gamma and curvature seeds); the box rule (per-S,
#         alpha-independent) and the nx rule (rung 1a's converged previous rungs); the dispersion
#         sweep's per-k box, its two levels nx and nx/2, and its seeds; eig_only (the campaign FD
#         gate's eigen-solver) moves eta ONLY through its overrides, against a Parameters there.
#   (iv)  kmax_solve end to end on a small grid (S = 1e3, nx = 512): k* equals the maximum of
#         eig_dense eigenvalues over k found by bounded Brent (no sensitivities anywhere); the
#         implicit slopes d ln k*/d ln S and d ln gamma*/d ln S equal central differences of that
#         brute-force maximum in ln S; every solve of the Newton sits on the SAME (nx, Lx), the
#         grid check at nx/2.
# pytest: `pytest tests/test_tearing_rung2.py`; script: `python tests/test_tearing_rung2.py`.
from _rmhd_testing import bootstrap, checks, fit_order

bootstrap()

import importlib
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import taranis as jr
from taranis import stability

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _run_script(name):
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    return importlib.import_module(name)


class _patched:
    # temporarily replace module attributes (restored on exit, also on failure)
    def __init__(self, mod, **attrs):
        self.mod, self.attrs, self.saved = mod, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.mod, k, v)
        return False


# ============================================================ (i) theory

def test_theory_exponents_match_fd_and_limits():
    R = _run_script("tearing_rung2_run")
    T = R.T
    h = 1e-4
    with checks() as c:
        for ka in (0.1, 0.5, 1.0, 2.0):
            for S in (1e3, 1e5):
                gp = lambda s_: np.log(T.gamma_theory(ka, S*np.exp(s_)))
                fdS = (gp(h) - gp(-h))/(2*h)
                gk = lambda s_: np.log(T.gamma_theory(ka*np.exp(s_), S))
                fdk = (gk(h) - gk(-h))/(2*h)
                qS, qk = R.theory_s_exponent(ka, S), R.theory_k_exponent(ka, S)
                c.check(f"theory d ln gamma/d ln S at ka={ka}, S={S:.0e}: {qS:.8f} == FD",
                        abs(qS - fdS) < 1e-7, f"{qS} vs {fdS}")
                c.check(f"theory d ln gamma/d ln k at ka={ka}, S={S:.0e}: {qk:.6f} == FD",
                        abs(qk - fdk) < 1e-6*max(1, abs(qk)), f"{qk} vs {fdk}")
        for ka in (0.3, 1.0, 1.9):
            dp = lambda s_: np.log(T.dprime_sech2(ka*np.exp(s_)))
            c.check(f"d ln Delta'/d ln k at ka={ka} == FD",
                    abs(R.dlnDprime_dlnk(ka) - (dp(h) - dp(-h))/(2*h))
                    < 1e-6*abs(R.dlnDprime_dlnk(ka)))
        # limits: FKR (Lam -> 0: ka -> sqrt 5 or S -> inf at fixed ka) and Coppi (Lam -> 1)
        c.check("m(Lam) -> 5/4 as Lam -> 0", abs(R.dlnF_dlnLam(1e-6) - 1.25) < 1e-6)
        c.check("m(Lam) -> inf as Lam -> 1", R.dlnF_dlnLam(1 - 1e-9) > 1e7)
        c.check("q_S -> -3/5 near the marginal point (FKR)",
                abs(R.theory_s_exponent(2.23, 1e8) + 0.6) < 1e-4, f"{R.theory_s_exponent(2.23, 1e8)}")
        c.check("q_S -> -1/3 at small ka (Coppi)",
                abs(R.theory_s_exponent(0.01, 1e3) + 1/3) < 1e-3, f"{R.theory_s_exponent(0.01, 1e3)}")
        c.check("q_k -> 2/3 at small ka (Coppi)",
                abs(R.theory_k_exponent(0.01, 1e3) - 2/3) < 2e-3, f"{R.theory_k_exponent(0.01, 1e3)}")
        qk_fkr = R.theory_k_exponent(0.3, 1e14)          # FKR on the n = 2 branch: -6/5
        c.check(f"q_k -> -6/5 in FKR at small ka ({qk_fkr:.4f})", abs(qk_fkr + 1.2) < 1e-2)
        for S in (1e3, 1e5):
            k0, g0 = R.theory_kmax(S)
            k1, g1 = T.theory_max(S)
            c.check(f"theory_kmax({S:.0e}) == T.theory_max", abs(k0/k1 - 1) < 1e-7
                    and abs(g0/g1 - 1) < 1e-12, f"{k0} {k1}")
            c.check(f"theory_kmax({S:.0e}) is a zero of the k-exponent",
                    abs(R.theory_k_exponent(k0, S)) < 1e-10)
            hh = 1e-2
            fk = (np.log(T.theory_max(S*np.exp(hh))[0]) - np.log(T.theory_max(S*np.exp(-hh))[0]))/(2*hh)
            fg = (np.log(T.theory_max(S*np.exp(hh))[1]) - np.log(T.theory_max(S*np.exp(-hh))[1]))/(2*hh)
            sk, sg = R.theory_kmax_slopes(S)
            c.check(f"theory d ln k_max/d ln S at {S:.0e}: {sk:.5f} == FD of theory_max {fk:.5f}",
                    abs(sk - fk) < 1e-4)
            c.check(f"theory d ln gamma_max/d ln S at {S:.0e}: {sg:.6f} == FD {fg:.6f} (envelope)",
                    abs(sg - fg) < 1e-5)
        n = 2                     # Delta'a ~ (ka)^-n: FKR gamma ~ Delta'^(4/5) S^(-3/5) k^(2/5),
        want = dict(kmax=-1/(3*n + 1), gmax=-(n + 1)/(3*n + 1), fkr=-3/5, coppi=-1/3,
                    shear_const=-1/2, shear_nonconst=-1/3, k_fkr=2/5 - 4*n/5, k_coppi=2/3,
                    k_shear_const=1/2 - n/2)       # shear const-psi: (Delta' k)^(1/2) S^(-1/2)
        c.check("paper exponents for the n = 2 sheet",
                set(R.PAPER) == set(want) and all(abs(R.PAPER[q] - want[q]) < 1e-15 for q in want),
                f"{R.PAPER}")
        c.check("delta_in is eq. 5.15 with unit prefactor",
                abs(R.delta_in(1e4, 0.5, 0.8) - (0.8/0.36)**(1/3)*(5e3)**(-1/3)) < 1e-15)


# ============================================================ (ii) one-point solver

def _sech2(u):
    e = np.exp(-2.0*np.abs(u))
    return 4.0*e/(1.0 + e)**2


def _dense_lead(S, k, nx, Lx, alpha, diss=None):
    # the leading eigenvalue of the ky block of a Parameters built DIRECTLY (Ly = 2 pi/k, no
    # overrides), the sech^2 equilibrium built here (not the script's)
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/k, dims=2, cfl_safety=0.5,
                           forcing=False, eqpars={"diss": diss or (0.0, 1.0/S), "hyper": 1})
    kg = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    p0 = sum(_sech2(x - Lx/2 - m*Lx) for m in (-1, 0, 1))
    import jax.numpy as jnp
    phi, psi = jnp.asarray(alpha*p0).reshape(1, -1, 1), jnp.asarray(p0).reshape(1, -1, 1)
    x0 = jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields
    B, _ = stability.ky_block_matrix(x0, kg, params, 1)
    return stability.eig_dense(B).values[0]


# ============================================================ (iii) pure helpers

def test_exponent_helpers():
    R = _run_script("tearing_rung2_run")
    with checks() as c:
        for S, k, al, qS, qk, qa in ((1e3, 0.3, 0.8, -3/5, -6/5, 0.5), (1e5, 1.2, 0.95, -1/3, 2/3, 2/3)):
            C = 0.07
            g = C*S**qS*k**qk*(1 - al*al)**qa
            rec = {"S": S, "k": k, "alpha": al, "lam": g + 1e-13j,
                   "dlam_eta": -qS*g*S + 0j,                          # dgamma/deta = -S qS gamma... (eta = 1/S)
                   "dlam_k": qk*g/k - 1e-14j,
                   "dlam_alpha": qa*g/(1 - al*al)*(-2*al) + 0j}
            c.check(f"s_exponent recovers {qS:.4f}", abs(R.s_exponent(rec) - qS) < 1e-12,
                    f"{R.s_exponent(rec)}")
            c.check(f"k_exponent recovers {qk:.4f}", abs(R.k_exponent(rec) - qk) < 1e-12)
            c.check(f"alpha_exponent recovers {qa:.4f}", abs(R.alpha_exponent(rec) - qa) < 1e-12)


def _analytic_gamma(k, G=0.05, k0=0.47, A=0.3, B=0.1):
    # a smooth single maximum at k0: gamma = G - A (ln k - ln k0)^2 + B (ln k - ln k0)^3
    x = np.log(k) - np.log(k0)
    g = G - A*x*x + B*x**3
    dg = (-2*A*x + 3*B*x*x)/k
    return g, dg


def test_newton_kmax_on_analytic_gamma():
    R = _run_script("tearing_rung2_run")
    k0 = 0.47

    def solve(k, pred):
        g, dg = _analytic_gamma(k)
        return {"lam": g + 0j, "dlam_k": dg + 0j}
    q = lambda k: -2*0.3/k**2            # the exact curvature at k0 (seed)
    with checks() as c:
        its, ks, gs, why = R.newton_kmax(solve, 0.60, 0.05, q(0.60), log=lambda *a: None)
        errs = [abs(i["k"] - k0) for i in its]
        c.check(f"Newton converges ({why}) to the maximum k0 = {k0} ({ks})",
                why is None and abs(ks - k0) < 1e-9 and abs(gs - 0.05) < 1e-15, f"{ks}, {gs}")
        c.check(f"... in <= 7 iterates ({len(its)})", len(its) <= 7)
        rates = [np.log(errs[i + 1])/np.log(errs[i]) for i in range(1, len(errs) - 1)
                 if 0 < errs[i + 1] < errs[i] < 1e-3]
        c.check(f"superlinear: error exponents {np.round(rates, 2)} > 1.3 (secant 1.618)",
                len(rates) >= 1 and min(rates) > 1.3, f"{errs}")
        c.check("the curvature after the first step is the secant of the iterates' dgamma/dk",
                all(abs(its[i]["curv"] - (its[i]["gk"] - its[i - 1]["gk"])/(its[i]["k"] - its[i - 1]["k"]))
                    < 1e-12*abs(its[i]["curv"]) for i in range(1, len(its))))
        # a wrong-sign curvature seed: the first step goes UPHILL by the cap, never downhill
        its2, ks2, _, why2 = R.newton_kmax(solve, 0.3, 0.05, +1.0, log=lambda *a: None)
        c.check("positive curvature seed: first step uphill by max_step k",
                abs(its2[0]["step"] - R.MAX_STEP*0.3) < 1e-15 and why2 is None
                and abs(ks2 - k0) < 1e-9, f"{its2[0]['step']}, {ks2}")
        its3, _, _, _ = R.newton_kmax(solve, 1.5, 0.05, -1e-6, log=lambda *a: None)
        c.check("a tiny curvature seed: the step is capped at max_step k",
                abs(abs(its3[0]["step"]) - R.MAX_STEP*1.5) < 1e-15 and its3[0]["step"] < 0)
        its4, ks4, _, why4 = R.newton_kmax(solve, 0.9, 0.05, q(0.9), maxit=2, log=lambda *a: None)
        c.check("maxit exhausted -> 'iterations', no k*", why4 == "iterations" and np.isnan(ks4))
        c.check(f"stop criterion |step| <= NEWTON_TOL k ({R.NEWTON_TOL})",
                abs(its[-1]["step"]) <= R.NEWTON_TOL*its[-1]["k"]
                and all(abs(i["step"]) > R.NEWTON_TOL*i["k"] for i in its[:-1]))
        # gamma* is the quadratic model's maximum, gamma + gk step/2 (not + gk step): visible only
        # with a loose stop, where it must beat the last iterate's gamma by the model's order
        # (pure quadratic in ln k, B = 0, so the model's own error is not the cubic's)
        def solve_q(k, pred):
            g, dg = _analytic_gamma(k, B=0.0)
            return {"lam": g + 0j, "dlam_k": dg + 0j}
        its6, _, gs6, _ = R.newton_kmax(solve_q, 0.60, 0.05, q(0.60), tol=1e-2, log=lambda *a: None)
        e_last, e_model = abs(its6[-1]["gamma"] - 0.05), abs(gs6 - 0.05)
        c.check(f"loose stop: gamma* model error {e_model:.1e} << last-iterate error {e_last:.1e}",
                e_model < 0.05*e_last, f"{e_model} vs {e_last}")
        # the stop is RELATIVE: the same maximum at k0 = 1e-5 (gamma depends on ln k only) must
        # converge to the same relative accuracy -- an absolute |step| <= tol stops at once
        ksm = 1e-5

        def solve_small(k, pred):
            g, dg = _analytic_gamma(k, k0=ksm)
            return {"lam": g + 0j, "dlam_k": dg + 0j}
        its5, ks5, _, why5 = R.newton_kmax(solve_small, 0.60*ksm/k0, 0.05, q(0.60)*(k0/ksm)**2,
                                           log=lambda *a: None)
        c.check(f"scale-free stop: k0 = {ksm:g} found to 1e-9 relative ({ks5}, {len(its5)} its)",
                why5 is None and abs(ks5 - ksm) < 1e-9*ksm and len(its5) == len(its),
                f"{ks5}, {len(its5)} vs {len(its)}")


def test_implicit_slopes_on_analytic_gamma():
    # gamma(k; eta, alpha) = G(eta, alpha) - A (ln k - L(eta, alpha))^2, L = ln c + a ln eta +
    # b ln(1 - alpha^2): k* = c eta^a (1-alpha^2)^b exactly, so d ln k*/d ln S = -a,
    # d ln k*/d ln(1-alpha^2) = b; G = g0 eta^p (1-alpha^2)^r: d ln gamma*/d ln S = -p, ... = r
    R = _run_script("tearing_rung2_run")
    A, cc, a, b, g0, p, r = 0.02, 2.0, 1/7, -1/7, 0.9, 3/7, 4/7

    def rec(k, S, al):
        eta, u = 1.0/S, 1 - al*al
        L = np.log(cc) + a*np.log(eta) + b*np.log(u)
        G = g0*eta**p*u**r
        x = np.log(k) - L
        dL_deta, dL_dal = a/eta, b*(-2*al)/u
        dG_deta, dG_dal = p*G/eta, r*G*(-2*al)/u
        return {"lam": G - A*x*x + 0j, "dlam_k": -2*A*x/k + 0j,
                "dlam_eta": dG_deta + 2*A*x*dL_deta + 0j,
                "dlam_alpha": dG_dal + 2*A*x*dL_dal + 0j}
    with checks() as c:
        for S, al in ((1e4, 0.8), (1e5, 0.3)):
            eta, u = 1.0/S, 1 - al*al
            ks = cc*eta**a*u**b
            G = g0*eta**p*u**r
            errs = []
            for h in (4e-2, 2e-2, 1e-2):
                sl = R.implicit_slopes(S, al, ks, G, rec(ks, S, al), rec(ks*(1 + h), S, al),
                                       rec(ks*(1 - h), S, al), h)
                errs.append(abs(sl["slope_k_S"] + a))
            c.check(f"S={S:.0e} alpha={al}: d ln k*/d ln S -> -a = {-a:.6f} at O(h^2) "
                    f"({sl['slope_k_S']:.8f}; order {fit_order((4e-2, 2e-2, 1e-2), errs):.2f})",
                    errs[-1] < 1e-4 and 1.8 < fit_order((4e-2, 2e-2, 1e-2), errs) < 2.2, f"{errs}")
            c.check(f"S={S:.0e} alpha={al}: d ln gamma*/d ln S == -p (envelope, exact at k*)",
                    abs(sl["slope_g_S"] + p) < 1e-12, f"{sl['slope_g_S']}")
            c.check(f"S={S:.0e} alpha={al}: d ln k*/d ln(1-alpha^2) -> b = {b:.6f}",
                    abs(sl["slope_k_u"] - b) < 1e-4, f"{sl['slope_k_u']}")
            c.check(f"S={S:.0e} alpha={al}: d ln gamma*/d ln(1-alpha^2) == r",
                    abs(sl["slope_g_u"] - r) < 1e-12, f"{sl['slope_g_u']}")
            c.check(f"S={S:.0e} alpha={al}: gamma_kk from dgamma/dk and from gamma agree, and "
                    f"dk*/deta = -gamma_keta/gamma_kk", abs(sl["gkk"]/sl["gkk_from_gamma"] - 1) < 2e-3
                    and abs(sl["dk_deta"] + sl["gketa"]/sl["gkk"]) < 1e-15*abs(sl["dk_deta"])
                    and abs(sl["dk_deta"]/(a*ks/eta) - 1) < 1e-3, f"{sl}")
            # dk*/dalpha on its own (the alpha-continuation seed reads it, not slope_k_u)
            ref_da = ks*b*(-2*al)/u
            c.check(f"S={S:.0e} alpha={al}: dk*/dalpha = {ref_da:.6e} ({sl['dk_dalpha']:.6e})",
                    abs(sl["dk_dalpha"]/ref_da - 1) < 1e-3, f"{sl['dk_dalpha']}")
        sl0 = R.implicit_slopes(1e4, 0.0, 0.5, 0.03, rec(0.5, 1e4, 0.0), rec(0.51, 1e4, 0.0),
                                rec(0.49, 1e4, 0.0), 0.02)
        c.check("alpha = 0: the (1-alpha^2) exponents are undefined (nan)",
                np.isnan(sl0["slope_k_u"]) and np.isnan(sl0["slope_g_u"]))


def test_fd_gate_helper():
    R = _run_script("tearing_rung2_run")
    C, qS, qk = 0.05, -0.45, 0.3

    def lam_of(k, eta):
        return C*eta**(-qS)*k**qk + 0j            # gamma ~ S^qS k^qk
    S, ka, al = 1e4, 0.4, 0.3
    eta = 1.0/S
    calls = []

    def fake_point(S_, k, alpha, nx, Lx, pred, **kw):
        return {"lam": lam_of(k, eta), "dlam_eta": -qS*lam_of(k, eta)/eta,
                "dlam_k": qk*lam_of(k, eta)/k, "nx": nx, "Lx": Lx}

    def fake_cached(kind, root=None, compute=None, **cfg):
        return compute()

    def fake_eig(S_, k, alpha, nx, Lx, sigma, lam_hint=None, eta=None):
        calls.append((k, eta, nx, Lx))
        return {"lam": lam_of(k, eta if eta is not None else 1.0/S_), "res": 0.0}
    Lx, nx = R.disp_box(S, ka)
    with _patched(R, point=fake_point, eig_only=fake_eig), _patched(R.U, cached=fake_cached), \
            checks() as c:
        rec, rows = R.fd_gate(S, ka, al, log=lambda *a: None)
        for which, ref in (("eta", -qS*C*eta**(-qS - 1)*ka**qk), ("k", qk*C*eta**(-qS)*ka**(qk - 1))):
            hs = [r[0] for r in rows[which]]
            errs = [r[2] for r in rows[which]]
            c.check(f"{which}: the h set {R.FD_H}", tuple(hs) == tuple(R.FD_H))
            if which == "eta":
                fd_ref = [(lam_of(ka, eta*(1 + h)) - lam_of(ka, eta*(1 - h)))/(2*h*eta) for h in hs]
            else:
                fd_ref = [(lam_of(ka*(1 + h), eta) - lam_of(ka*(1 - h), eta))/(2*h*ka) for h in hs]
            c.check(f"{which}: fd = central difference with RELATIVE steps",
                    all(abs(r[1] - f) <= 1e-12*abs(f) for r, f in zip(rows[which], fd_ref)),
                    f"{[r[1] for r in rows[which]]} vs {fd_ref}")
            order = fit_order(hs, errs)
            c.check(f"{which}: |fd - sens| is O(h^2) ({order:.3f}) against the exact {ref:.6e}",
                    1.9 < order < 2.1)
        c.check("FD eigenvalues at the sensitivity's own (nx, Lx)",
                all(n_ == nx and L_ == Lx for _, _, n_, L_ in calls))
        c.check("eta steps move eta only (k fixed), k steps move k only (eta = 1/S)",
                sorted((k_, e_) for k_, e_, _, _ in calls if k_ == ka)
                == sorted((ka, eta*(1 + s*h)) for h in R.FD_H for s in (1, -1))
                and sorted(k_ for k_, e_, _, _ in calls if e_ is None)
                == sorted(ka*(1 + s*h) for h in R.FD_H for s in (1, -1)))


def test_continuation_seeds_and_box_rule():
    R = _run_script("tearing_rung2_run")
    seen = []

    def fake_kmax_solve(S, alpha, k0, g0, c0, nx, Lx, root=None, log=None, compute=True,
                        deadline=np.inf, tag=None):
        seen.append(dict(S=S, alpha=alpha, k0=k0, g0=g0, c0=c0, nx=nx, Lx=Lx))
        # a fake solution: k* = 0.5 (1 + alpha) S^-1/7, gamma* = 0.02 (1 - alpha^2)
        return dict(S=S, alpha=alpha, kmax=0.5*(1 + alpha)*S**(-1/7), gmax=0.02*(1 - alpha**2),
                    gkk=-0.3, dk_dalpha=0.5*S**(-1/7), slope_k_S=-1/7, slope_g_S=-3/7,
                    rec={"dlam_alpha": -0.04*alpha - 0.01 + 0j})
    with _patched(R, kmax_solve=fake_kmax_solve, seed=lambda S, a: (0.44, 0.03),
                  _sigma_seed=lambda S, k, a, Lx, g, compute: g), checks() as c:
        out, ok = R.kmax_alpha(1e4, alphas=(0.0, 0.3, 0.6), log=lambda *a: None)
        Lx, nx = R.kmax_box(1e4)
        c.check("kmax_alpha: one box (Lx, nx) = kmax_box(S) for every alpha",
                all(s_["Lx"] == Lx and s_["nx"] == nx for s_ in seen))
        s1 = seen[1]
        k_prev = 0.5*1e4**(-1/7)
        c.check("kmax_alpha: k seed = previous k* + dk*/dalpha dalpha",
                abs(s1["k0"] - (k_prev + 0.5*1e4**(-1/7)*0.3)) < 1e-15, f"{s1['k0']}")
        c.check("kmax_alpha: gamma seed = previous gamma* + dgamma/dalpha dalpha, curvature "
                "rescaled", abs(s1["g0"] - (0.02 - 0.01*0.3)) < 1e-15
                and abs(s1["c0"] + 0.3*s1["g0"]/0.02) < 1e-15, f"{s1}")
        c.check("kmax_alpha: first alpha seeded by seed()", seen[0]["k0"] == 0.44)
        seen.clear()
        out, ok = R.kmax_S(0.8, (1e3, 1e4), log=lambda *a: None)
        k3 = 0.5*1.8*1e3**(-1/7)
        c.check("kmax_S: k seed = k* (S'/S)^(d ln k*/d ln S)",
                abs(seen[1]["k0"] - k3*10**(-1/7)) < 1e-15, f"{seen[1]['k0']}")
        g3 = 0.02*(1 - 0.64)
        c.check("kmax_S: gamma seed = gamma* (S'/S)^(d ln gamma*/d ln S)",
                abs(seen[1]["g0"] - g3*10**(-3/7)) < 1e-15)
        k4 = seen[1]["k0"]
        c.check("kmax_S: curvature seed = previous gamma_kk (g0/g*) (k*/k0)^2",
                abs(seen[1]["c0"] + 0.3*(seen[1]["g0"]/g3)*(k3/k4)**2) < 1e-15, f"{seen[1]['c0']}")
        c.check("kmax_S: each S in its own kmax_box",
                all((s_["Lx"], s_["nx"]) == R.kmax_box(s_["S"]) for s_ in seen))

    # a nearly flat maximum (gamma_kk -> 0): the implicit slopes blow up and the raw
    # extrapolation lands at k < 0 (the S = 1e3, alpha 0.95 -> 0.97 crash of the first full
    # campaign: Ly = 2 pi/k < 0 refused by the overrides seam). The seed is held to
    # [k*/SEED_TRUST, SEED_TRUST k*], and a non-positive gamma seed falls back to gamma*
    def flat_kmax_solve(S, alpha, k0, g0, c0, nx, Lx, root=None, log=None, compute=True,
                        deadline=np.inf, tag=None):
        seen.append(dict(S=S, alpha=alpha, k0=k0, g0=g0, c0=c0))
        return dict(S=S, alpha=alpha, kmax=0.5, gmax=0.02, gkk=-1e-3, dk_dalpha=-50.0,
                    slope_k_S=+5.0, slope_g_S=-3/7, rec={"dlam_alpha": -10.0 + 0j})
    seen.clear()
    with _patched(R, kmax_solve=flat_kmax_solve, seed=lambda S, a: (0.44, 0.03),
                  _sigma_seed=lambda S, k, a, Lx, g, compute: g), checks() as c:
        R.kmax_alpha(1e3, alphas=(0.95, 0.97), log=lambda *a: None)
        c.check(f"kmax_alpha: runaway k seed clipped to k*/SEED_TRUST ({seen[1]['k0']})",
                seen[1]["k0"] == 0.5/R.SEED_TRUST)
        c.check(f"kmax_alpha: negative gamma seed -> previous gamma* ({seen[1]['g0']})",
                seen[1]["g0"] == 0.02)
        seen.clear()
        R.kmax_S(0.95, (1e3, 1e4), log=lambda *a: None)
        c.check(f"kmax_S: runaway k seed clipped to SEED_TRUST k* ({seen[1]['k0']})",
                seen[1]["k0"] == 0.5*R.SEED_TRUST)
    with checks() as c:
        for S in (1e3, 1e4, 1e5):
            Lx, nx = R.kmax_box(S)
            c.check(f"kmax_box({S:.0e}): Lx = BOX_C/k_th (alpha-independent)",
                    abs(Lx - R.BOX_C/R.theory_kmax(S)[0]) < 1e-5)
        tab = R.U.eigen_tables()
        for ka in (0.3, 1.0, 2.0):
            for S in (1e3, 1e4, 1e5):
                c.check(f"nx_rule reproduces rung 1a's converged previous rung at ka={ka}, "
                        f"S={S:.0e}", R.nx_rule(S, ka, R.T.lx_for(ka))
                        == max(1024, tab[ka][S]["nx"][-1]//2), f"{R.nx_rule(S, ka, R.T.lx_for(ka))}")
        c.check("disp_box: rung 1a's per-k box", R.disp_box(1e4, 0.1)[0] == 150.0
                and R.disp_box(1e4, 1.5)[0] == 20.0)


def test_dispersion_helper_boxes_levels_and_seeds():
    R = _run_script("tearing_rung2_run")
    seen = []

    def fake_point(S, k, alpha, nx, Lx, pred, **kw):
        seen.append(dict(k=k, nx=nx, Lx=Lx, pred=pred))
        return {"lam": 0.03 + 0.01*k + 0j, "dlam_k": 0.01 + 0j, "dlam_eta": -3.0 + 0j,
                "k": k, "kappa": 5.0, "seconds": 0.0, "S": S}
    with _patched(R, point=fake_point, _table_gamma=lambda S, ka, al: 0.05 if ka == 0.2 else None), \
            checks() as c:
        out, ok = R.dispersion(1e4, 0.6, kas=(0.2, 0.5), log=lambda *a: None)
        c.check("both levels computed", set(out[1]) == set(out[2]) == {0.2, 0.5} and ok)
        for s_ in seen:
            Lx, nx = R.disp_box(1e4, s_["k"])
            c.check(f"ka={s_['k']}: rung 1a's per-k box, nx or nx/2",
                    s_["Lx"] == Lx and s_["nx"] in (nx, nx//2))
        c.check("level 2 is nx/2", sorted(s_["nx"] for s_ in seen if s_["k"] == 0.5)
                == sorted((R.disp_box(1e4, 0.5)[1], R.disp_box(1e4, 0.5)[1]//2)))
        c.check("tabulated seed used where it exists",
                all(s_["pred"] == 0.05 for s_ in seen if s_["k"] == 0.2))
        c.check("elsewhere gamma + dgamma/dk dk from the previous ka (level 1)",
                all(abs(s_["pred"] - (0.032 + 0.01*0.3)) < 1e-15 for s_ in seen if s_["k"] == 0.5))


@pytest.mark.fp64
def test_eig_only_eta_override_moves_eta_only():
    # the campaign's FD-gate eigenvalue solver: eta through overrides (nu stays 0), k through Ly
    R = _run_script("tearing_rung2_run")
    S, alpha, nx, Lx = 1e3, 0.6, 512, 30.0
    eta2 = 1.3e-3
    ref = _dense_lead(S, 0.55, nx, Lx, alpha, (0.0, eta2))
    got = R.eig_only(S, 0.55, alpha, nx, Lx, ref*1.03, lam_hint=ref, eta=eta2)["lam"]
    ref0 = _dense_lead(S, 0.55, nx, Lx, alpha)
    got0 = R.eig_only(S, 0.55, alpha, nx, Lx, ref0*1.03, lam_hint=ref0)["lam"]
    with checks() as c:
        c.check(f"eig_only(k=0.55, eta={eta2}) == eig_dense of Parameters(Ly = 2 pi/0.55, diss = "
                f"(0, eta)) ({got:.12f})", abs(got - ref) < 1e-10*abs(ref), f"{got} vs {ref}")
        c.check("eig_only without eta is at the Parameters' own eta",
                abs(got0 - ref0) < 1e-10*abs(ref0), f"{got0} vs {ref0}")


def test_curvature_seed_is_self_similar_in_k_over_kstar():
    # curvature_seed(S, k, g) rescales the alpha = 0 theory gamma_kk at k_th(S) as g/k^2 (a
    # maximum self-similar in k/k*): moved to another S's theory maximum it must predict THAT
    # maximum's curvature (measured within 5 %; without the (k_th/k)^2 factor 1.8x / 3.5x off)
    R = _run_script("tearing_rung2_run")
    with checks() as c:
        for S2 in (1e4, 1e5):
            k2, g2 = R.theory_kmax(S2)
            ratio = R.curvature_seed(1e3, k2, g2)/R.curvature_theory(S2, k2)
            c.check(f"seed from S = 1e3 at S = {S2:.0e}'s maximum: curvature ratio {ratio:.4f}",
                    abs(ratio - 1) < 0.15)


def test_kmax_solve_polish_on_analytic_gamma():
    # kmax_solve's polish k_pol = k_last - gk/gamma_kk (central-difference curvature) is a Newton
    # step toward the maximum: with a loose Newton stop (so the step is visible) it lands far
    # closer to k0 than the last iterate. The solves are the analytic gamma, B = 0
    R = _run_script("tearing_rung2_run")
    k0 = 0.47

    def fake_point(S, k, alpha, nx, Lx, pred, **kw):
        g, dg = _analytic_gamma(k, B=0.0)
        return {"lam": g + 0j, "dlam_k": dg + 0j, "dlam_eta": 0j, "dlam_alpha": 0j,
                "kappa": 1.0}
    newton = R.newton_kmax

    def loose(*a, **kw):
        kw["tol"] = 1e-2
        return newton(*a, **kw)
    with _patched(R, point=fake_point, newton_kmax=loose), checks() as c:
        out = R.kmax_solve(1e3, 0.0, 0.60, 0.05, -2*0.3/0.60**2, 512, 30.0, log=lambda *a: None)
        e_last = abs(out["its"][-1]["k"] - k0)
        e_pol = abs(out["k_polished"] - k0)
        c.check(f"polished k* error {e_pol:.1e} << last iterate's {e_last:.1e}",
                e_pol < 0.05*e_last, f"{out['k_polished']} vs {out['its'][-1]['k']}")


def test_point_retries_refusals_and_rejects_non_growing_modes():
    # point's safeguard: a harness refusal (RuntimeError) or an eigenvalue that is not a real
    # growing mode (a null-cluster member, a complex pair) moves on to the next attempt with a
    # moved shift; the first real growing record is returned; nothing acceptable -> RuntimeError
    R = _run_script("tearing_rung2_run")
    pred = 0.05
    seq = {0: RuntimeError("near-singular"), 1: 1e-17 + 0j, 2: 0.04 + 0.01j, 3: 0.051 + 0j}
    calls = []

    def fake_cached(kind, root=None, compute=None, attempt=None, **cfg):
        calls.append((attempt, cfg))
        out = seq.get(attempt, 0.049 + 0j)
        if isinstance(out, Exception):
            raise out
        return {"lam": out, "attempt": attempt}

    def fake_sens(S, k, alpha, nx, Lx, sigma, lam_hint=None, rtol_mult=1.0, dense_nx=None):
        return {"sigma": sigma, "rtol_mult": rtol_mult}
    with _patched(R.U, cached=fake_cached), _patched(R, sens_solve=fake_sens), checks() as c:
        rec = R.point(1e3, 0.6, 0.3, 512, 30.0, pred, log=lambda *a: None)
        c.check("refusal, null member and complex pair skipped: attempt 3 returned",
                rec["attempt"] == 3 and [a for a, _ in calls] == [0, 1, 2, 3], f"{calls}")
        c.check("the cache key is the configuration, never sigma",
                all(cfg == dict(S=1e3, k=0.6, alpha=0.3, nx=512, Lx=30.0) for _, cfg in calls))
        seq.clear(); seq.update({a: -0.01 + 0j for a in range(20)}); calls.clear()
        with pytest.raises(RuntimeError, match="no sensitivity"):
            R.point(1e3, 0.6, 0.3, 512, 30.0, pred, log=lambda *a: None)
        c.check(f"all attempts tried before giving up ({len(calls)})", len(calls) == 7)


# ============================================================ (iv) kmax_solve end to end

def _brute_kmax(S, alpha, nx, Lx):
    # the maximum over k of eig_dense eigenvalues: bounded Brent in ln k. One Parameters per S,
    # k through overrides={"Ly": 2 pi/k} (== a Parameters with that Ly: test (ii)), so nothing
    # recompiles; no sensitivity is used
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi, dims=2, cfl_safety=0.5,
                           forcing=False, eqpars={"diss": (0.0, 1.0/S), "hyper": 1})
    kg = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    p0 = sum(_sech2(x - Lx/2 - m*Lx) for m in (-1, 0, 1))
    import jax.numpy as jnp
    phi, psi = jnp.asarray(alpha*p0).reshape(1, -1, 1), jnp.asarray(p0).reshape(1, -1, 1)
    x0 = jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields

    def f(lk):
        B, _ = stability.ky_block_matrix(x0, kg, params, 1, overrides={"Ly": 2*np.pi/np.exp(lk)})
        return -stability.eig_dense(B).values[0].real
    r = minimize_scalar(f, bounds=(np.log(0.45), np.log(0.85)), method="bounded",
                        options=dict(xatol=1e-9))
    return float(np.exp(r.x)), float(-r.fun)


@pytest.mark.fp64
def test_kmax_solve_matches_brute_force_maximum(tmp_path):
    R = _run_script("tearing_rung2_run")
    S, alpha, nx = 1e3, 0.6, 512
    Lx = R.kmax_box(S)[0]
    k0, g0 = 0.62, 0.06
    res = R.kmax_solve(S, alpha, k0, g0, R.curvature_seed(S, k0, g0), nx, Lx,
                       root=str(tmp_path), log=print)
    kb, gb = _brute_kmax(S, alpha, nx, Lx)
    h = 0.05
    kp, gp = _brute_kmax(S*np.exp(h), alpha, nx, Lx)
    km, gm = _brute_kmax(S*np.exp(-h), alpha, nx, Lx)
    sk, sg = (np.log(kp) - np.log(km))/(2*h), (np.log(gp) - np.log(gm))/(2*h)
    recs = [res["rec"], res["rec_p"], res["rec_m"]]
    print(f"  Newton k* = {res['kmax']:.10f} gamma* = {res['gmax']:.12e} ({res['iterations']} its)"
          f"; brute force k = {kb:.10f} gamma = {gb:.12e}")
    print(f"  d ln k*/d ln S: implicit {res['slope_k_S']:+.6f}, brute FD {sk:+.6f}; "
          f"d ln gamma*/d ln S: {res['slope_g_S']:+.7f}, {sg:+.7f}")
    with checks() as c:
        c.check(f"Newton k* == brute-force argmax to 2e-6 ({abs(res['kmax']/kb - 1):.1e})",
                abs(res["kmax"]/kb - 1) < 2e-6)
        c.check(f"gamma* == brute-force max to 1e-10 ({abs(res['gmax']/gb - 1):.1e})",
                abs(res["gmax"]/gb - 1) < 1e-10)
        c.check("implicit d ln k*/d ln S == FD of the brute-force argmax (h = 0.05 in ln S)",
                abs(res["slope_k_S"] - sk) < 2e-3, f"{res['slope_k_S']} vs {sk}")
        c.check("envelope d ln gamma*/d ln S == FD of the brute-force max",
                abs(res["slope_g_S"] - sg) < 2e-4, f"{res['slope_g_S']} vs {sg}")
        c.check("converged in <= 6 iterates", res["iterations"] <= 6)
        c.check("every Newton / curvature solve on the SAME (nx, Lx): the box does not follow k",
                all(int(r["nx"]) == nx and float(r["Lx"]) == Lx for r in recs))
        c.check("curvature pair at k*(1 +- CURV_H)",
                abs(float(res["rec_p"]["k"]) - res["kmax"]*(1 + R.CURV_H)) < 1e-15
                and abs(float(res["rec_m"]["k"]) - res["kmax"]*(1 - R.CURV_H)) < 1e-15)
        c.check("grid check at nx/2 on the same box", int(res["rec_lo"]["nx"]) == nx//2
                and float(res["rec_lo"]["Lx"]) == Lx and np.isfinite(res["grid_dk"]))
        c.check("polished k* (central-difference curvature) agrees with the secant k*",
                abs(res["k_polished"] - res["kmax"]) < 1e-7*res["kmax"])


@pytest.mark.fp64
def test_sens_solve_parameter_sensitivities_match_independent_fd():
    R = _run_script("tearing_rung2_run")
    S, k, alpha, nx, Lx = 1e3, 0.5, 0.6, 512, 30.0
    eta = 1.0/S
    lam = _dense_lead(S, k, nx, Lx, alpha)
    rec = R.sens_solve(S, k, alpha, nx, Lx, lam*1.03, lam_hint=lam, dense_nx=nx)
    hs = (1e-2, 3e-3, 1e-3)
    fd = {"eta": [], "k": [], "alpha": [], "nu": []}
    for h in hs:
        fd["eta"].append((_dense_lead(S, k, nx, Lx, alpha, (0.0, eta*(1 + h)))
                          - _dense_lead(S, k, nx, Lx, alpha, (0.0, eta*(1 - h))))/(2*h*eta))
        fd["k"].append((_dense_lead(S, k*(1 + h), nx, Lx, alpha)
                        - _dense_lead(S, k*(1 - h), nx, Lx, alpha))/(2*h*k))
        fd["alpha"].append((_dense_lead(S, k, nx, Lx, alpha + h)
                            - _dense_lead(S, k, nx, Lx, alpha - h))/(2*h))
    hnu = eta*1e-3
    dnu = (_dense_lead(S, k, nx, Lx, alpha, (hnu, eta))
           - _dense_lead(S, k, nx, Lx, alpha, (-hnu, eta)))/(2*hnu)
    sens = {"eta": complex(rec["dlam_eta"]), "k": complex(rec["dlam_k"]),
            "alpha": complex(rec["dlam_alpha"])}
    with checks() as c:
        c.check(f"sens_solve lambda (overrides Ly) == eig_dense of a Parameters with Ly = 2 pi/k "
                f"({complex(rec['lam']):.12f})", abs(complex(rec["lam"]) - lam) < 1e-10*abs(lam))
        for key in ("eta", "k", "alpha"):
            errs = [abs(f - sens[key]) for f in fd[key]]
            order = fit_order(hs, errs)
            print(f"  d gamma/d {key}: sens {sens[key].real:+.10e}; FD errs {errs[0]:.1e} -> "
                  f"{errs[-1]:.1e} (order {order:.2f})")
            c.check(f"d gamma/d {key} == independent eigenvalue FD at O(h^2) (order {order:.2f})",
                    1.85 < order < 2.15 and errs[-1] < 2e-5*abs(sens[key]), f"{errs}")
        print(f"  d gamma/d nu (FD) = {dnu.real:+.6e} against d gamma/d eta {sens['eta'].real:+.6e}")
        c.check("the eta tangent does not move nu: dgamma/dnu differs from dgamma/deta by > 10 %",
                abs(dnu - sens["eta"]) > 0.1*abs(sens["eta"])
                and abs(dnu + sens["eta"]) > 0.1*abs(sens["eta"]))
        g = complex(rec["lam"]).real
        c.check("s_exponent(rec) == -(eta/gamma) dgamma/deta",
                abs(R.s_exponent(rec) + eta*sens["eta"].real/g) < 1e-14)
        c.check("k_exponent(rec) == (k/gamma) dgamma/dk",
                abs(R.k_exponent(rec) - k*sens["k"].real/g) < 1e-14)
        c.check("k_direction is d/dk: -Ly/k = -2 pi/k^2",
                abs(R.k_direction(k)["Ly"] + 2*np.pi/k**2) < 1e-15
                and abs(R.k_override(k)["Ly"] - 2*np.pi/k) < 1e-15)
        c.check("ETA_DIR moves eta only", tuple(R.ETA_DIR) == (0.0, 1.0))
        c.check("residuals and isolation printed and small",
                float(rec["res_r"]) < 1e-9 and float(rec["res_l"]) < 1e-9
                and float(rec["rel_err"]) < 1e-3, f"{rec['res_r']}, {rec['res_l']}, {rec['rel_err']}")



if __name__ == "__main__":
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
