# plans/AUTODIFF_PLAN.md rung 2b item 1 (viscosity) gate: visco-resistive tearing with
# field-aligned shear flow, diss = (nu, eta) = (Pm/S, 1/S) through the overrides seam. The science
# lives in examples/tearing-viscosity.ipynb (data: examples/tearing_viscosity_run.py); this file
# is its fast standing gate and IMPORTS the data script (examples/ is put on sys.path).
# Setup as rung 2: dims=2, hyper = 1, S = 1/eta, Psi0 = sech^2 (+ 2 images), Phi0 = alpha Psi0,
# ky block, fp64. Small grids throughout.
# Order: (iii) first (seconds), then (i), (ii), (iv).
#   (i)   theory (numpy/scipy only): the constant-psi layer integral I_m (m = 1 is FKR's C_FKR,
#         m = 2 the viscous I_V) against an INDEPENDENT numerical solution of the viscous
#         constant-psi layer; the visco-resistive inner BVP's limits -- B&L's exact relation as
#         Pm -> 0 (deviation linear in Pm), the inertia-free layer Pm^(-1/6) F_v(Lam Pm^(1/3)) as
#         Pm -> inf, F_v(L)/L -> I_V, the pole C_VC; the tables against direct BVP solves;
#         gamma_theory_visc solves the relation; the analytic exponents against central
#         differences of gamma_theory_visc (S at fixed Pm, Pm at fixed S, k); the large-Pm
#         asymptotes (viscous FKR -1/6 / -2/3, viscous Coppi -1/3); theory_kmax_visc is the
#         maximum of gamma_theory_visc, its slopes FD-consistent; visc_exponents(n).
#   (ii)  the one-point solver against INDEPENDENT routes: J at overrides {Ly, diss} == J of a
#         Parameters built there; sens_solve's lambda == eig_dense of that Parameters (dense vs
#         band-LU shift-invert); dgamma/dnu, dgamma/deta, dgamma/dk, dgamma/dalpha against central
#         differences of those dense eigenvalues at O(h^2); eig_only moves eta and nu only.
#   (iii) pure helpers on synthetic inputs: the record exponents on exact power laws;
#         implicit_slopes on an analytic gamma(k; eta, nu, alpha); the continuation seeds of
#         kmax_Pm and kmax_alpha (alpha = 0 from the Pm chain, R._trust clipping); kmax_grid2;
#         dispersion's boxes, levels and seeds; fd_gate's control flow; point's retry safeguard;
#         branch_scan / local_maxima.
#   (iv)  kmax_solve end to end (S = 1e3, Pm = 10, alpha = 0.6, nx = 512): k* and gamma* equal the
#         bounded-Brent maximum of dense eigenvalues; the implicit d ln k*/d ln Pm, d ln k*/d ln S
#         (fixed Pm) and the envelope d ln gamma*/d ln Pm equal central differences of that
#         brute-force maximum.
# pytest: `pytest tests/test_tearing_viscosity.py`; script: `python tests/test_tearing_viscosity.py`.
from _rmhd_testing import bootstrap, checks, fit_order

bootstrap()

import functools
import importlib
import os
import sys
import tempfile

import numpy as np
import pytest
from scipy.integrate import solve_bvp
from scipy.optimize import minimize_scalar

import taranis as jr
from taranis import stability

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _run_script(name="tearing_viscosity_run"):
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    return importlib.import_module(name)


@functools.lru_cache(maxsize=1)
def _theory_root():
    # one scratch root for the theory tables, shared by the tests of this module
    return tempfile.mkdtemp(prefix="taranis_visc_theory_")


class _patched:
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


# ============================================================ (iii) pure helpers

def test_record_exponents_on_power_laws():
    V = _run_script()
    with checks() as c:
        for S, Pm, k, al, p, q, qk, qa in ((1e3, 2.0, 0.3, 0.8, 0.6, -0.15, -1.2, 0.5),
                                           (1e5, 30.0, 1.2, 0.95, 0.4, 0.2, 2/3, 2/3)):
            nu, eta = Pm/S, 1/S
            C = 0.07
            g = C*eta**p*nu**q*k**qk*(1 - al*al)**qa       # gamma = C eta^p nu^q ...
            rec = {"S": S, "Pm": Pm, "k": k, "alpha": al, "lam": g + 1e-13j,
                   "dlam_eta": p*g/eta + 0j, "dlam_nu": q*g/nu + 0j, "dlam_k": qk*g/k + 0j,
                   "dlam_alpha": qa*g/(1 - al*al)*(-2*al) + 0j}
            c.check(f"q_Pm = q ({q})", abs(V.q_Pm(rec) - q) < 1e-12, f"{V.q_Pm(rec)}")
            c.check(f"q_S|Pm = -(p + q) ({-(p + q)})", abs(V.q_S(rec) + p + q) < 1e-12)
            c.check(f"q_S|nu = -p ({-p})", abs(V.q_Snu(rec) + p) < 1e-12)
            c.check("q_k, q_alpha", abs(V.q_k(rec) - qk) < 1e-12 and abs(V.q_alpha(rec) - qa) < 1e-12)
            c.check("nu_eta_ratio = (q/nu)/(p/eta)", abs(V.nu_eta_ratio(rec) - (q/nu)/(p/eta)) < 1e-12*abs(q/nu/(p/eta)))


def test_implicit_slopes_on_analytic_gamma():
    # gamma = G - A (ln k - L)^2, L = ln c + a ln eta + b ln nu + e ln(1 - alpha^2), G = g0 eta^p
    # nu^q (1-alpha^2)^r: k* = c eta^a nu^b u^e exactly. With eta = 1/S, nu = Pm/S:
    # d ln k*/d ln S|Pm = -(a + b), d ln k*/d ln Pm|S = b, d ln gamma*/d ln S|Pm = -(p + q),
    # d ln gamma*/d ln Pm|S = q
    V = _run_script()
    A, cc, a, b, e, g0, p, q, r = 0.02, 2.0, 1/7, 1/14, -1/7, 0.9, 3/7, -2/7, 4/7

    def rec(k, S, Pm, al):
        eta, nu, u = 1.0/S, Pm/S, 1 - al*al
        L = np.log(cc) + a*np.log(eta) + b*np.log(nu) + e*np.log(u)
        G = g0*eta**p*nu**q*u**r
        x = np.log(k) - L
        return {"lam": G - A*x*x + 0j, "dlam_k": -2*A*x/k + 0j,
                "dlam_eta": p*G/eta + 2*A*x*a/eta + 0j, "dlam_nu": q*G/nu + 2*A*x*b/nu + 0j,
                "dlam_alpha": (r*G + 2*A*x*e)*(-2*al)/u + 0j}
    with checks() as c:
        for S, Pm, al in ((1e4, 10.0, 0.8), (1e3, 0.3, 0.3)):
            eta, nu, u = 1.0/S, Pm/S, 1 - al*al
            ks = cc*eta**a*nu**b*u**e
            G = g0*eta**p*nu**q*u**r
            errs = []
            for h in (4e-2, 2e-2, 1e-2):
                sl = V.implicit_slopes(S, Pm, al, ks, G, rec(ks, S, Pm, al), rec(ks*(1 + h), S, Pm, al),
                                       rec(ks*(1 - h), S, Pm, al), h)
                errs.append(abs(sl["slope_k_Pm"] - b))
            o = fit_order((4e-2, 2e-2, 1e-2), errs)
            c.check(f"S={S:.0e} Pm={Pm}: d ln k*/d ln Pm -> b at O(h^2) ({sl['slope_k_Pm']:.8f}, order {o:.2f})",
                    errs[-1] < 1e-4 and 1.8 < o < 2.2, f"{errs}")
            c.check(f"d ln k*/d ln S|Pm -> -(a + b) ({sl['slope_k_S']:.6f})", abs(sl["slope_k_S"] + a + b) < 1e-4)
            c.check(f"d ln gamma*/d ln S|Pm == -(p + q) (envelope)", abs(sl["slope_g_S"] + p + q) < 1e-12)
            c.check(f"d ln gamma*/d ln Pm|S == q", abs(sl["slope_g_Pm"] - q) < 1e-12)
            c.check("dk*/dnu = -gamma_knu/gamma_kk == b k*/nu", abs(sl["dk_dnu"]/(b*ks/nu) - 1) < 1e-3
                    and abs(sl["dk_deta"]/(a*ks/eta) - 1) < 1e-3)
            c.check(f"d ln k*/d ln(1-alpha^2) -> e, d ln gamma*/d ln(1-alpha^2) == r",
                    abs(sl["slope_k_u"] - e) < 1e-4 and abs(sl["slope_g_u"] - r) < 1e-12)
            c.check("dk*/dalpha", abs(sl["dk_dalpha"]/(ks*e*(-2*al)/u) - 1) < 1e-3)
            c.check("gamma_kk from gamma agrees", abs(sl["gkk"]/sl["gkk_from_gamma"] - 1) < 2e-3)
        sl0 = V.implicit_slopes(1e4, 1.0, 0.0, 0.5, 0.03, rec(0.5, 1e4, 1.0, 0.0), rec(0.51, 1e4, 1.0, 0.0),
                                rec(0.49, 1e4, 1.0, 0.0), 0.02)
        c.check("alpha = 0: (1-alpha^2) exponents nan", np.isnan(sl0["slope_k_u"]) and np.isnan(sl0["slope_g_u"]))


def _fake_solution(S, Pm, alpha, kmax, gmax, **kw):
    out = dict(S=S, Pm=Pm, alpha=alpha, kmax=kmax, gmax=gmax, gkk=-0.3, dk_dalpha=0.4,
               slope_k_Pm=1/14, slope_g_Pm=-2/7, slope_k_S=-1/7, slope_g_S=-3/7,
               rec={"dlam_alpha": -0.02 + 0j})
    out.update(kw)
    return out


def test_continuation_seeds_and_chains():
    V = _run_script()
    R = V.R
    seen = []

    def fake_kmax_solve(S, Pm, alpha, k0, g0, c0, nx, Lx, root=None, log=None, compute=True,
                        deadline=np.inf, tag=None):
        seen.append(dict(S=S, Pm=Pm, alpha=alpha, k0=k0, g0=g0, c0=c0, nx=nx, Lx=Lx))
        return _fake_solution(S, Pm, alpha, 0.5*(1 + alpha)*Pm**(1/14), 0.02*Pm**(-2/7)*(1 - alpha**2))
    seed = lambda S, Pm, a, root=None: (0.44, 0.03, -0.2)
    with _patched(V, kmax_solve=fake_kmax_solve, seed_theory=seed,
                  _sigma_seed=lambda S, Pm, k, a, Lx, g, compute: g), checks() as c:
        out, ok = V.kmax_Pm(1e4, 0.0, (1.0, 10.0, 100.0), log=lambda *a: None)
        Lx, nx = R.kmax_box(1e4)
        c.check("kmax_Pm: one box per S", all((s_["Lx"], s_["nx"]) == (Lx, nx) for s_ in seen))
        c.check("kmax_Pm: first Pm seeded by seed_theory", seen[0]["k0"] == 0.44 and seen[0]["c0"] == -0.2)
        k1, g1 = 0.5, 0.02
        c.check("kmax_Pm: k seed = k* (Pm'/Pm)^(d ln k*/d ln Pm)", abs(seen[1]["k0"] - k1*10**(1/14)) < 1e-15)
        c.check("kmax_Pm: gamma seed = gamma* (Pm'/Pm)^(d ln gamma*/d ln Pm)",
                abs(seen[1]["g0"] - g1*10**(-2/7)) < 1e-15)
        c.check("kmax_Pm: curvature seed = gamma_kk (g0/g*) (k*/k0)^2",
                abs(seen[1]["c0"] + 0.3*(seen[1]["g0"]/g1)*(k1/seen[1]["k0"])**2) < 1e-15)
        seen.clear()
        out, ok = V.kmax_alpha(1e4, 10.0, (0.0, 0.3, 0.6), log=lambda *a: None)
        c.check("kmax_alpha: alpha = 0 is the Pm chain's solution (replayed up to Pm = 10)",
                [s_["Pm"] for s_ in seen if s_["alpha"] == 0.0] == list(V._pms_upto(10.0)))
        a_ = [s_ for s_ in seen if s_["alpha"] > 0]
        kst = 0.5*10**(1/14)
        c.check("kmax_alpha: k seed = k* + dk*/dalpha dalpha", abs(a_[0]["k0"] - (kst + 0.4*0.3)) < 1e-15)
        c.check("kmax_alpha: gamma seed = gamma* + gamma_alpha dalpha, curvature rescaled",
                abs(a_[0]["g0"] - (0.02*10**(-2/7) - 0.02*0.3)) < 1e-15
                and abs(a_[0]["c0"] + 0.3*a_[0]["g0"]/(0.02*10**(-2/7))) < 1e-15)
        c.check("kmax_alpha: the (S, Pm) box", all(s_["Lx"] == Lx and s_["nx"] == nx for s_ in a_))
        c.check("_pms_upto(10) = the chain through 10", V._pms_upto(10.0) == tuple(p for p in V.PM_CHAIN if p <= 10.0))
        seen.clear()
        out, ok = V.kmax_alpha(1e4, 2.5, (0.0, 0.3), log=lambda *a: None)
        c.check("kmax_alpha off the Pm chain: alpha = 0 seeded by the theory",
                seen[0]["alpha"] == 0.0 and seen[0]["Pm"] == 2.5 and seen[0]["k0"] == 0.44)

    # a flat maximum: runaway extrapolation clipped by R._trust (as rung 2)
    def flat(S, Pm, alpha, k0, g0, c0, nx, Lx, root=None, log=None, compute=True, deadline=np.inf, tag=None):
        seen.append(dict(Pm=Pm, alpha=alpha, k0=k0, g0=g0))
        return _fake_solution(S, Pm, alpha, 0.5, 0.02, gkk=-1e-3, dk_dalpha=-50.0, slope_k_Pm=40.0,
                              rec={"dlam_alpha": -10.0 + 0j})
    seen.clear()
    with _patched(V, kmax_solve=flat, seed_theory=seed,
                  _sigma_seed=lambda S, Pm, k, a, Lx, g, compute: g), checks() as c:
        V.kmax_alpha(1e3, 2.5, (0.0, 0.95), log=lambda *a: None)
        c.check(f"kmax_alpha: runaway k seed clipped to k*/SEED_TRUST ({seen[1]['k0']})",
                seen[1]["k0"] == 0.5/R.SEED_TRUST)
        c.check("kmax_alpha: negative gamma seed -> gamma*", seen[1]["g0"] == 0.02)
        seen.clear()
        V.kmax_Pm(1e3, 0.0, (1.0, 10.0), log=lambda *a: None)
        c.check(f"kmax_Pm: runaway k seed clipped to SEED_TRUST k* ({seen[1]['k0']})",
                seen[1]["k0"] == 0.5*R.SEED_TRUST)

    # grid2: the nx solution's k*, gamma*, gamma_kk at 2 nx on the same box
    base = _fake_solution(1e4, 10.0, 0.9, 0.61, 0.012, gkk=-0.25)
    calls = []

    def fake_solve2(S, Pm, alpha, k0, g0, c0, nx, Lx, *a, **kw):
        calls.append((S, Pm, alpha, k0, g0, c0, nx, Lx))
        return {}
    with _patched(V, kmax_alpha=lambda *a, **kw: ({0.9: base}, True), kmax_solve=fake_solve2), \
            checks() as c:
        V.kmax_grid2(1e4, 10.0, 0.9)
        Lx, nx = R.kmax_box(1e4)
        c.check("kmax_grid2: Newton at 2 nx from the nx solution", calls[0] == (1e4, 10.0, 0.9, 0.61, 0.012,
                                                                                -0.25, 2*nx, Lx), f"{calls}")


def test_small_alpha_c2_helper():
    # gamma(k*(0), alpha) = g0 (1 - c2 alpha^2 + c4 alpha^4): c2(alpha) = -gamma_alpha/(2 alpha
    # gamma) = c2 + O(alpha^2), the Richardson extrapolation over ALPHA_C2 removes the alpha^2 term;
    # the solves sit at the alpha = 0 chain's k* on the k_max box
    V = _run_script()
    R = V.R
    c2, c4, g0 = 3.0, 5.0, 0.02
    seen = []

    def fake_point(S, Pm, k, alpha, nx, Lx, pred, **kw):
        seen.append((S, Pm, k, alpha, nx, Lx))
        g = g0*(1 - c2*alpha**2 + c4*alpha**4)
        return {"lam": g + 0j, "dlam_alpha": g0*(-2*c2*alpha + 4*c4*alpha**3) + 0j}
    chain = {10.0: dict(kmax=0.47, gmax=g0)}
    with _patched(V, point=fake_point, kmax_Pm=lambda *a, **k: (chain, True)), checks() as c:
        out = V.small_alpha_c2(1e4, 10.0, log=lambda *a: None)
        exact = [(2*c2 - 4*c4*a*a)/(2*(1 - c2*a*a + c4*a**4)) for a in V.ALPHA_C2]
        c.check("c2(alpha) = -gamma_alpha/(2 alpha gamma)", np.allclose(out["c2_alpha"], exact, rtol=1e-14))
        c.check(f"extrapolated c2 {out['c2']:.6f} closer to {c2} than either raw value",
                abs(out["c2"] - c2) < 0.1*min(abs(x - c2) for x in out["c2_alpha"]))
        Lx, nx = R.kmax_box(1e4)
        c.check("solves at k*(0) on the k_max box, at ALPHA_C2",
                [s_[3] for s_ in seen] == list(V.ALPHA_C2)
                and all(s_[2] == 0.47 and s_[4:] == (nx, Lx) and s_[1] == 10.0 for s_ in seen))


def test_kmax_solve_polish_on_analytic_gamma():
    # kmax_solve's polish k_pol = k_last - gk/gamma_kk (central-difference curvature) is a Newton
    # step toward the maximum: with a loose Newton stop (so the step is visible) it lands far closer
    # to k0 than the last iterate. gamma = G - A (ln k - ln k0)^2, the solves analytic
    V = _run_script()
    R = V.R
    k0, G, A = 0.47, 0.05, 0.3

    def fake_point(S, Pm, k, alpha, nx, Lx, pred, **kw):
        x = np.log(k) - np.log(k0)
        return {"lam": G - A*x*x + 0j, "dlam_k": -2*A*x/k + 0j, "dlam_eta": 0j, "dlam_nu": 0j,
                "dlam_alpha": 0j, "kappa": 1.0, "nx": nx, "Lx": Lx, "k": k}
    newton = R.newton_kmax

    def loose(*a, **kw):
        kw["tol"] = 1e-2
        return newton(*a, **kw)
    with _patched(V, point=fake_point), _patched(R, newton_kmax=loose), checks() as c:
        out = V.kmax_solve(1e3, 10.0, 0.0, 0.60, G, -2*A/0.60**2, 512, 30.0, log=lambda *a: None)
        e_last, e_pol = abs(out["its"][-1]["k"] - k0), abs(out["k_polished"] - k0)
        c.check(f"polished k* error {e_pol:.1e} << last iterate's {e_last:.1e}", e_pol < 0.05*e_last)


def test_sigma_seed():
    # the coarse dense top sets sigma only where it is a real growing mode within a factor 3 of the
    # expectation (or there is none); read-back never computes it
    V = _run_script()
    tops = {0.3: 0.012 + 0j, 0.4: -2e-4 + 0j, 0.5: 0.05 + 0j, 0.6: 0.011 + 0.004j}
    with _patched(V, _dense_others=lambda S, Pm, k, a, Lx, nx: (np.array([]), tops[k], 1)), checks() as c:
        c.check("growing top within x3 -> the top", V._sigma_seed(1e3, 1.0, 0.3, 0.5, 30.0, 0.01, True) == 0.012)
        c.check("no expectation -> the top", V._sigma_seed(1e3, 1.0, 0.3, 0.5, 30.0, None, True) == 0.012)
        c.check("decaying top -> the expectation", V._sigma_seed(1e3, 1.0, 0.4, 0.5, 30.0, 0.01, True) == 0.01)
        c.check("top 5x the expectation -> the expectation",
                V._sigma_seed(1e3, 1.0, 0.5, 0.5, 30.0, 0.01, True) == 0.01)
        c.check("complex top -> the expectation", V._sigma_seed(1e3, 1.0, 0.6, 0.5, 30.0, 0.01, True) == 0.01)
        c.check("read-back: the expectation, untouched", V._sigma_seed(1e3, 1.0, 0.3, 0.5, 30.0, 0.02, False) == 0.02)


def test_dispersion_fd_gate_point_and_branch_helpers():
    V = _run_script()
    R = V.R
    seen = []

    def fake_point(S, Pm, k, alpha, nx, Lx, pred, **kw):
        seen.append(dict(k=k, nx=nx, Lx=Lx, pred=pred, alpha=alpha))
        return {"lam": 0.03 + 0.01*k + 0j, "dlam_k": 0.01 + 0j, "dlam_eta": -3.0 + 0j,
                "dlam_nu": 0.5 + 0j, "k": k, "kappa": 5.0, "seconds": 0.0, "S": S, "Pm": Pm}
    with _patched(V, point=fake_point, gamma_theory_visc=lambda ka, S, Pm, root=None: 0.05 if ka == 0.3 else np.nan,
                  _sigma_seed=lambda S, Pm, k, a, Lx, g, compute: (0.07 if k == 0.3 else g) if compute else g), \
            checks() as c:
        out, ok = V.dispersion(1e4, 10.0, 0.0, kas=(0.3, 0.6), log=lambda *a: None)
        c.check("alpha = 0: both levels", set(out[1]) == set(out[2]) == {0.3, 0.6} and ok)
        for s_ in seen:
            Lx, nx = R.disp_box(1e4, s_["k"])
            c.check(f"ka={s_['k']}: rung 1a's per-k box, nx or nx/2", s_["Lx"] == Lx and s_["nx"] in (nx, nx//2))
        c.check("alpha = 0: theory seed", all(s_["pred"] == 0.05 for s_ in seen if s_["k"] == 0.3))
        c.check("theory nan -> previous gamma + gamma_k dk",
                all(abs(s_["pred"] - (0.033 + 0.01*0.3)) < 1e-15 for s_ in seen if s_["k"] == 0.6))
        seen.clear()
        out, ok = V.dispersion(1e4, 10.0, 0.9, kas=(0.3, 0.6), log=lambda *a: None, levels=(1,))
        c.check("alpha > 0: the dense top seeds sigma where it is a growing mode",
                [s_["pred"] for s_ in seen][0] == 0.07)
        seen.clear()
        V.dispersion(1e4, 10.0, 0.9, kas=(0.3,), log=lambda *a: None, compute=False, levels=(1,))
        c.check("read-back needs no seed (pred 1.0, sigma is not a key)", seen[0]["pred"] == 1.0)

    # fd_gate: relative steps in eta, nu (k fixed) and k, at the sensitivity's own grid
    C, p, q, qk = 0.05, 0.45, -0.2, 0.3
    S, ka, al, Pm = 1e4, 0.4, 0.3, 3.0
    nu0, eta0 = Pm/S, 1/S
    lam_of = lambda k, eta, nu: C*eta**p*nu**q*k**qk + 0j
    calls = []

    def fp(S_, Pm_, k, alpha, nx, Lx, pred, **kw):
        return {"lam": lam_of(k, eta0, nu0), "dlam_eta": p*lam_of(k, eta0, nu0)/eta0,
                "dlam_nu": q*lam_of(k, eta0, nu0)/nu0, "dlam_k": qk*lam_of(k, eta0, nu0)/k, "nx": nx, "Lx": Lx}

    def fe(S_, Pm_, k, alpha, nx, Lx, sigma, lam_hint=None, eta=None, nu=None):
        calls.append((k, eta, nu, nx, Lx))
        return {"lam": lam_of(k, eta, nu), "res": 0.0}
    with _patched(V, point=fp, eig_only=fe, gamma_theory_visc=lambda *a, **k: 0.05,
                  _sigma_seed=lambda *a: 0.05), \
            _patched(V.U, cached=lambda kind, root=None, compute=None, **cfg: compute()), checks() as c:
        rec, rows = V.fd_gate(S, ka, al, Pm, log=lambda *a: None)
        Lx, nx = R.disp_box(S, ka)
        for which, x0, ref in (("eta", eta0, p*C*eta0**(p - 1)*nu0**q*ka**qk),
                               ("nu", nu0, q*C*eta0**p*nu0**(q - 1)*ka**qk),
                               ("k", ka, qk*C*eta0**p*nu0**q*ka**(qk - 1))):
            hs = [r[0] for r in rows[which]]
            order = fit_order(hs, [r[2] for r in rows[which]])
            c.check(f"{which}: h set, O(h^2) against the exact {ref:.4e} ({order:.3f})",
                    tuple(hs) == tuple(V.FD_H) and 1.9 < order < 2.1)
        c.check("FD eigenvalues at the sensitivity's (nx, Lx)", all(n_ == nx and L_ == Lx for *_, n_, L_ in calls))
        c.check("eta steps move eta only, nu steps nu only, k steps k only",
                sorted(e_ for k_, e_, n_, *_ in calls if e_ != eta0) == sorted(eta0*(1 + s*h) for h in V.FD_H for s in (1, -1))
                and all(k_ == ka and n_ == nu0 for k_, e_, n_, *_ in calls if e_ != eta0)
                and sorted(n_ for k_, e_, n_, *_ in calls if n_ != nu0) == sorted(nu0*(1 + s*h) for h in V.FD_H for s in (1, -1))
                and all(k_ == ka and e_ == eta0 for k_, e_, n_, *_ in calls if n_ != nu0)
                and sorted(k_ for k_, e_, n_, *_ in calls if k_ != ka) == sorted(ka*(1 + s*h) for h in V.FD_H for s in (1, -1)))

    # point: refusal, null member, complex pair skipped; never-growing -> RuntimeError; key = config
    seq = {0: RuntimeError("near-singular"), 1: 1e-17 + 0j, 2: 0.04 + 0.01j, 3: 0.051 + 0j}
    cl = []

    def fake_cached(kind, root=None, compute=None, attempt=None, **cfg):
        cl.append((kind, attempt, cfg))
        out = seq.get(attempt, 0.049 + 0j)
        if isinstance(out, Exception):
            raise out
        return {"lam": out, "attempt": attempt}
    with _patched(V.U, cached=fake_cached), _patched(V, sens_solve=lambda *a, **k: {}), checks() as c:
        rec = V.point(1e3, 10.0, 0.6, 0.3, 512, 30.0, 0.05, log=lambda *a: None)
        c.check("attempt 3 returned after a refusal, a null member and a complex pair",
                rec["attempt"] == 3 and [a for _, a, _ in cl] == [0, 1, 2, 3])
        c.check("kind 'visc', key = (S, Pm, k, alpha, nx, Lx), never sigma",
                all(k_ == "visc" and cfg == dict(S=1e3, Pm=10.0, k=0.6, alpha=0.3, nx=512, Lx=30.0)
                    for k_, _, cfg in cl))
        seq.clear(); seq.update({a: -0.01 + 0j for a in range(20)}); cl.clear()
        with pytest.raises(RuntimeError, match="no sensitivity"):
            V.point(1e3, 10.0, 0.6, 0.3, 512, 30.0, 0.05, log=lambda *a: None)
        c.check(f"every attempt tried ({len(cl)})", len(cl) == len(V.ATTEMPTS) == 7)

    # branch_scan / local_maxima
    tops = {0.2: 0.010, 0.35: 0.012, 0.5: 0.011, 0.7: 0.0115, 0.9: 0.004}
    with _patched(V, _dense_others=lambda S, Pm, k, a, Lx, nx: (np.array([-1e-3 + 0j, -2.0 + 0j]),
                                                                 complex(tops[k]), 1)), \
            _patched(V.U, cached=lambda kind, root=None, compute=None, **cfg: compute()), checks() as c:
        ks, g, second, npos, top = V.branch_scan(1e3, 10.0, 0.95, kas=tuple(tops), log=lambda *a: None)
        c.check("branch_scan: the top real growing eigenvalue per k",
                np.allclose(g, list(tops.values())) and np.allclose(second, -1e-3) and (npos == 1).all()
                and np.allclose(top, g))
        c.check("local_maxima finds both maxima", V.local_maxima(ks, g) == [0.35, 0.7])
    with _patched(V, _dense_others=lambda S, Pm, k, a, Lx, nx: (np.array([-1e-3 + 0j]), -2e-4 + 0j, 0)), \
            _patched(V.U, cached=lambda kind, root=None, compute=None, **cfg: compute()), checks() as c:
        ks, g, second, npos, top = V.branch_scan(1e3, 10.0, 0.99, kas=(0.3, 0.5), log=lambda *a: None)
        c.check("branch_scan: a decaying top is no growth rate (nan) but is reported",
                np.isnan(g).all() and np.allclose(top, -2e-4) and (npos == 0).all()
                and V.local_maxima(ks, g) == [])
    c2 = V.local_maxima([0.1, 0.2, 0.3], [np.nan, 1.0, 0.5])
    assert c2 == [0.2], c2


# ============================================================ (i) theory

def _viscous_const_psi_integral_numeric(smax=12.0):
    # INDEPENDENT of the script: U'''' + s^2 U = s on s >= 0, U odd (U(0) = U''(0) = 0), and at
    # smax the algebraic branch U = 1/s - 24/s^7 + O(s^-13); I = 2 int_0^inf (1 - sU)
    def ode(s, y):
        return np.vstack([y[1], y[2], y[3], s - s*s*y[0]])

    def bc(ya, yb):
        return np.array([ya[0], ya[2], yb[0] - 1/smax + 24/smax**7, yb[1] + 1/smax**2 - 168/smax**8])
    s = np.linspace(0, smax, 3001)
    y0 = np.zeros((4, s.size))
    y0[0] = s/(1 + s*s)
    r = solve_bvp(ode, bc, s, y0, tol=1e-11, max_nodes=200000)
    assert r.status == 0, r.message
    ss = np.linspace(0, smax, 200001)
    f = 1 - ss*r.sol(ss)[0]
    tail = 2*24/(5*smax**5)            # int_smax^inf (1 - sU) = int 24/s^6 = 24/(5 smax^5)
    return 2*np.trapezoid(f, ss) + tail


def test_const_psi_integral_and_inner_layer_limits():
    V = _run_script()
    T = V.T
    with checks() as c:
        c.check(f"I_1 (inertia) == FKR's C_FKR ({V.const_psi_integral(1):.15f})",
                abs(V.const_psi_integral(1) - T.C_FKR) < 1e-14)
        Inum = _viscous_const_psi_integral_numeric()
        c.check(f"I_V = pi 6^(-1/3) G(5/6)/G(7/6) = {V.I_V:.10f} == independent ODE {Inum:.10f}",
                abs(V.I_V/Inum - 1) < 1e-7, f"{V.I_V} vs {Inum}")
        # the inertia-free layer: F_v(L)/L -> I_V (viscous constant-psi) as L -> 0
        rs = [V.inner_F_visc(L, 1.0, inertia=False)/L for L in (1e-2, 1e-3, 1e-4)]
        c.check(f"F_v(L)/L -> I_V: {np.round(rs, 7)}", abs(rs[-1]/V.I_V - 1) < 1e-4
                and abs(rs[-1] - V.I_V) < abs(rs[0] - V.I_V), f"{rs}")
        # Pm -> 0: B&L's exact relation, the deviation linear in Pm
        # (the linear range is Pm << Lam^2-ish: smaller Pm at smaller Lam)
        for lam, pms in ((0.1, (1e-3, 3e-4)), (0.5, (1e-2, 3e-3))):
            d = [V.inner_F_visc(lam, p)/T.inner_F(lam) - 1 for p in pms]
            c.check(f"Lam={lam}: F(Lam, Pm) -> B&L as Pm -> 0, linearly ({d[0]:.3e}, {d[1]:.3e})",
                    abs(d[1]) < 1e-2 and abs(d[0]/d[1]/(pms[0]/pms[1]) - 1) < 0.1, f"{d}")
        c.check("inner_F_visc(Pm=0) is T.inner_F", V.inner_F_visc(0.4, 0.0) == float(T.inner_F(0.4)))
        # the s_max -> inf limit: the Richardson value against a raw solve at s_max = 320 (truncation
        # O(1/s_max^2): 4e-5 at s_max = 80, ~2.5e-6 at 320 for Lam = 0.5, Pm = 1)
        c0_, c1_ = V._inner_bvp_visc(0.5, 1.0, 320.0)
        dR = V.inner_F_visc(0.5, 1.0)/(2*c1_/c0_) - 1
        c.check(f"inner_F_visc is the s_max -> inf limit (vs s_max = 320: {dR:.1e})", abs(dR) < 1e-5)
        # Pm -> inf at fixed L = Lam Pm^(1/3): F Pm^(1/6) -> F_v(L)
        Fv = V.inner_F_visc(0.5, 1.0, inertia=False)
        errs = [abs(V.inner_F_visc(0.5*p**(-1/3), p)*p**(1/6)/Fv - 1) for p in (10.0, 100.0, 1000.0, 1e4)]
        c.check(f"F(L Pm^(-1/3), Pm) Pm^(1/6) -> F_v(L): errors {np.round(errs, 7)} decrease like 1/Pm",
                errs[3] < errs[2] < errs[1] < errs[0] and errs[2] < 2e-3
                and 5 < errs[2]/errs[3] < 20, f"{errs}")
        cvc = V.c_vcoppi()
        c.check(f"C_VC = {cvc:.8f} is the pole of F_v (1/F_v there ~ 0)",
                abs(V.inner_invF_visc(cvc, 1.0, inertia=False)) < 1e-9 and 1.4 < cvc < 1.7)
        p3 = [V.pole_visc(p)*p**(1/3) for p in (100.0, 1000.0)]
        c.check(f"pole(Pm) Pm^(1/3) -> C_VC ({p3})", abs(p3[1] - cvc) < abs(p3[0] - cvc)
                and abs(p3[1]/cvc - 1) < 1e-2)
        c.check("pole(0) = 1 (B&L's Coppi limit)", V.pole_visc(0.0) == 1.0)
        n1, n2 = V.visc_exponents(1), V.visc_exponents(2)
        c.check("visc_exponents(n=1): k_max S^-1/4 Pm^1/8, gamma_max S^-1/2 Pm^-1/4",
                np.allclose([n1["kmax_S"], n1["kmax_Pm"], n1["gmax_S"], n1["gmax_Pm"]],
                            [-1/4, 1/8, -1/2, -1/4], atol=1e-15))
        c.check("visc_exponents(n=2) = PRED: -1/7, 1/14, -3/7, -2/7; regimes -2/3, -1/6, -5/3, -1/3",
                np.allclose([n2["kmax_S"], n2["kmax_Pm"], n2["gmax_S"], n2["gmax_Pm"], n2["vfkr_S"],
                             n2["vfkr_Pm"], n2["vfkr_k"], n2["vcoppi_Pm"]],
                            [-1/7, 1/14, -3/7, -2/7, -2/3, -1/6, -5/3, -1/3], atol=1e-15)
                and V.PRED == n2)
        # the n-general crossing reproduces the regime formulas' own crossing exponents
        for n in (1, 2, 3):
            e = V.visc_exponents(n)
            # gamma_vfkr ~ k^(1/3 - n) S^(-2/3) Pm^(-1/6) = gamma_vcoppi ~ k^(2/3) S^(-1/3) Pm^(-1/3)
            ks, kp = -1/3/(n + 1/3), (1/6)/(n + 1/3)
            c.check(f"n={n}: k_max exponents from the regime crossing",
                    abs(e["kmax_S"] - ks) < 1e-15 and abs(e["kmax_Pm"] - kp) < 1e-15
                    and abs(e["gmax_S"] - (-1/3 + 2/3*ks)) < 1e-15
                    and abs(e["gmax_Pm"] - (-1/3 + 2/3*kp)) < 1e-15)


def test_theory_tables_relation_and_exponents():
    V = _run_script()
    T = V.T
    root = _theory_root()
    h = V.PM_DLN
    for p in (1.0, 100.0):
        for s in (0, 1, -1):
            V.theory_table(float(p*np.exp(s*h)) if s else p, True, root)
    with checks() as c:
        for Pm, lams in ((1.0, (3e-3, 0.11, 0.37, 0.61)), (100.0, (1e-3, 0.05, 0.2))):
            for lam in lams:
                lf, m = V.F_table(lam, Pm, root=root)
                direct = V.inner_F_visc(lam, Pm)
                hh = 1e-4
                md = (np.log(V.inner_F_visc(lam*np.exp(hh), Pm))
                      - np.log(V.inner_F_visc(lam*np.exp(-hh), Pm)))/(2*hh)
                # the campaign's tables (40 + 24 points in ln r): ~1e-6 in F, i.e. ~1e-6/m in gamma
                c.check(f"table F(Lam={lam}, Pm={Pm}) == direct BVP to 3e-6 ({np.exp(lf)/direct - 1:.1e})",
                        abs(np.exp(lf)/direct - 1) < 3e-6)
                c.check(f"table d ln F/d ln Lam ({m:.7f}) == FD of the BVP ({md:.7f})",
                        abs(m/md - 1) < 1e-5, f"{m} {md}")
        for ka, S in ((0.3, 1e4), (1.2, 1e5)):
            g = V.gamma_theory_visc(ka, S, 1.0, root)
            lam = g/((1/S)**(1/3)*(2*ka)**(2/3))
            res_ = V.inner_F_visc(lam, 1.0)/(T.dprime_sech2(ka)*(1/(S*2*ka))**(1/3)) - 1
            c.check(f"gamma_theory_visc(ka={ka}, S={S:.0e}, Pm=1) solves Delta' delta_eta = F(Lam, Pm) "
                    f"(direct BVP, {res_:.1e})", abs(res_) < 3e-6)
            e = V.theory_exponents(ka, S, 1.0, root)
            hs = 1e-3
            fS = (np.log(V.gamma_theory_visc(ka, S*np.exp(hs), 1.0, root))
                  - np.log(V.gamma_theory_visc(ka, S*np.exp(-hs), 1.0, root)))/(2*hs)
            fk = (np.log(V.gamma_theory_visc(ka*np.exp(hs), S, 1.0, root))
                  - np.log(V.gamma_theory_visc(ka*np.exp(-hs), S, 1.0, root)))/(2*hs)
            fP = (np.log(V.gamma_theory_visc(ka, S, float(np.exp(h)), root))
                  - np.log(V.gamma_theory_visc(ka, S, float(np.exp(-h)), root)))/(2*h)
            c.check(f"ka={ka} S={S:.0e}: q_S|Pm {e['qS']:.7f} == FD {fS:.7f}", abs(e["qS"] - fS) < 1e-5)
            c.check(f"ka={ka} S={S:.0e}: q_k {e['qk']:.6f} == FD {fk:.6f}", abs(e["qk"] - fk) < 1e-4)
            c.check(f"ka={ka} S={S:.0e}: q_Pm {e['qPm']:.6f} == FD across the Pm tables {fP:.6f}",
                    abs(e["qPm"] - fP) < 2e-4)
            c.check("q_Snu = q_S + q_Pm", abs(e["qSnu"] - e["qS"] - e["qPm"]) < 1e-15)
        # large-Pm asymptotes (Pm = 100, S = 1e8: inertia Lam Pm^(-2/3) negligible)
        gv = V.gamma_theory_visc(2.2, 1e8, 100.0, root)
        e = V.theory_exponents(2.2, 1e8, 100.0, root)
        c.check(f"viscous constant-psi: gamma/gamma_vfkr = {gv/V.gamma_vfkr(2.2, 1e8, 100.0):.5f} -> 1",
                abs(gv/V.gamma_vfkr(2.2, 1e8, 100.0) - 1) < 5e-3)
        c.check(f"viscous constant-psi exponents q_Pm {e['qPm']:.4f} -> -1/6, q_S {e['qS']:.4f} -> -2/3",
                abs(e["qPm"] + 1/6) < 0.01 and abs(e["qS"] + 2/3) < 0.01)
        gc = V.gamma_theory_visc(0.01, 1e8, 100.0, root)
        e = V.theory_exponents(0.01, 1e8, 100.0, root)
        c.check(f"viscous nonconstant-psi: gamma/gamma_vcoppi = {gc/V.gamma_vcoppi(0.01, 1e8, 100.0):.4f} -> 1",
                abs(gc/V.gamma_vcoppi(0.01, 1e8, 100.0) - 1) < 3e-2)
        c.check(f"viscous nonconstant-psi exponents q_Pm {e['qPm']:.4f} -> -1/3, q_k {e['qk']:.4f} -> 2/3",
                abs(e["qPm"] + 1/3) < 0.03 and abs(e["qk"] - 2/3) < 0.03)
        # the Newton seeds: curvature_theory is d2 gamma/dk2 of the relation; seed_theory scales the
        # alpha = 0 maximum by (1 - alpha^2)^(-1/7), (1 - alpha^2)^(4/7) and the curvature as gamma/k^2
        k0_, hh = 0.5, 4e-3
        gq = [V.gamma_theory_visc(k0_*(1 + s_*hh), 1e4, 1.0, root) for s_ in (1, 0, -1)]
        cref = (gq[0] - 2*gq[1] + gq[2])/(k0_*hh)**2
        cth = V.curvature_theory(1e4, 1.0, k0_, root=root)
        c.check(f"curvature_theory = d2gamma/dk2 ({cth:.6e} vs {cref:.6e})", abs(cth/cref - 1) < 1e-3)
        kth, gth = V.theory_kmax_visc(1e4, 1.0, root)
        ks_, gs_, cs_ = V.seed_theory(1e4, 1.0, 0.8, root)
        u_ = 1 - 0.64
        c.check("seed_theory: (k_th u^(-1/7), g_th u^(4/7), curvature g/k^2-rescaled)",
                abs(ks_ - kth*u_**(-1/7)) < 1e-14 and abs(gs_ - gth*u_**(4/7)) < 1e-15
                and abs(cs_ - V.curvature_theory(1e4, 1.0, kth, root=root)*(gs_/gth)*(kth/ks_)**2) < 1e-12*abs(cs_))
        for S in (1e3, 1e5):
            k0, g0 = V.theory_kmax_visc(S, 1.0, root)
            r = minimize_scalar(lambda lk: -V.gamma_theory_visc(np.exp(lk), S, 1.0, root),
                                bounds=(np.log(0.1), np.log(1.5)), method="bounded",
                                options=dict(xatol=1e-10))
            c.check(f"theory_kmax_visc({S:.0e}, Pm=1) = {k0:.8f} is the maximum ({np.exp(r.x):.8f})",
                    abs(k0/np.exp(r.x) - 1) < 1e-5 and abs(g0/-r.fun - 1) < 1e-10)
            sl = V.theory_kmax_slopes_visc(S, 1.0, root=root)
            hh = 1e-2
            fk = (np.log(V.theory_kmax_visc(S*np.exp(hh), 1.0, root)[0])
                  - np.log(V.theory_kmax_visc(S*np.exp(-hh), 1.0, root)[0]))/(2*hh)
            fg = (np.log(V.theory_kmax_visc(S*np.exp(hh), 1.0, root)[1])
                  - np.log(V.theory_kmax_visc(S*np.exp(-hh), 1.0, root)[1]))/(2*hh)
            fgp = (np.log(V.theory_kmax_visc(S, float(np.exp(h)), root)[1])
                   - np.log(V.theory_kmax_visc(S, float(np.exp(-h)), root)[1]))/(2*h)
            c.check(f"S={S:.0e}: d ln k_max/d ln S {sl['k_S']:.5f} == FD {fk:.5f}", abs(sl["k_S"] - fk) < 1e-4)
            c.check(f"S={S:.0e}: envelope d ln gamma_max/d ln S {sl['g_S']:.6f} == FD {fg:.6f}",
                    abs(sl["g_S"] - fg) < 1e-5)
            c.check(f"S={S:.0e}: envelope d ln gamma_max/d ln Pm {sl['g_Pm']:.6f} == FD {fgp:.6f}",
                    abs(sl["g_Pm"] - fgp) < 1e-4)
        # reviewer gates: d ln k_max/d ln Pm (the notebook's theory column) against the root at the
        # Pm e^(+-h) tables, positive at Pm = 100 (k_max rises with Pm there); 1/F Richardson-
        # extrapolated is the reciprocal of F (raw s_max = 80: 1e-5..1e-4 off); the table is
        # usable down to its first node
        sl = V.theory_kmax_slopes_visc(1e5, 100.0, root=root)
        kp_ = V.theory_kmax_visc(1e5, float(100*np.exp(h)), root)[0]
        km_ = V.theory_kmax_visc(1e5, float(100*np.exp(-h)), root)[0]
        c.check(f"k_Pm(S=1e5, Pm=100) = {sl['k_Pm']:.5f} == root FD, > 0",
                abs(sl["k_Pm"] - np.log(kp_/km_)/(2*h)) < 1e-12 and 0.04 < sl["k_Pm"] < 0.08)
        for lam in (0.3, 0.8):
            c.check(f"inner_invF_visc(Lam={lam}) == 1/inner_F_visc",
                    abs(V.inner_invF_visc(lam, 1.0)*V.inner_F_visc(lam, 1.0) - 1) < 1e-6)
        lp_ = V.theory_table(1.0, True, root)[0]
        lf_, m_ = V.F_table(np.exp(lp_ + 0.5*(V.X_TAB[0] + V.X_TAB[1])), 1.0, root=root)
        c.check("F_table defined between the first two nodes", np.isfinite(lf_) and np.isfinite(m_))


# ============================================================ (ii) one-point solver

def _sech2(u):
    e = np.exp(-2.0*np.abs(u))
    return 4.0*e/(1.0 + e)**2


def _direct(S, k, nx, Lx, alpha, diss):
    # a Parameters built DIRECTLY at (Ly = 2 pi/k, diss), the sech^2 equilibrium built here
    import jax.numpy as jnp
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/k, dims=2, cfl_safety=0.5,
                           forcing=False, eqpars={"diss": tuple(diss), "hyper": 1})
    kg = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    p0 = sum(_sech2(x - Lx/2 - m*Lx) for m in (-1, 0, 1))
    phi, psi = jnp.asarray(alpha*p0).reshape(1, -1, 1), jnp.asarray(p0).reshape(1, -1, 1)
    x0 = jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields
    B, _ = stability.ky_block_matrix(x0, kg, params, 1)
    return B


def _dense_lead(S, k, nx, Lx, alpha, diss):
    return stability.eig_dense(_direct(S, k, nx, Lx, alpha, diss)).values[0]


@pytest.mark.fp64
def test_sens_solve_matches_independent_routes():
    V = _run_script()
    R = V.R
    S, Pm, k, alpha, nx, Lx = 1e3, 3.0, 0.5, 0.6, 512, 30.0
    nu, eta = Pm/S, 1/S
    # J at the overrides {Ly, diss} vs J of a Parameters built there
    p = R.make_params(S, nx, Lx)
    x0 = V.U.equilibrium(p, alpha)[0]
    Jo, _ = stability.ky_block_matrix(x0, jr.setup_kgrids(p), p, 1, overrides=V.overrides(k, S, Pm))
    Jp = _direct(S, k, nx, Lx, alpha, (nu, eta))
    lam = stability.eig_dense(Jp).values[0]
    rec = V.sens_solve(S, Pm, k, alpha, nx, Lx, lam*1.03, lam_hint=lam, dense_nx=nx)
    hs = (1e-2, 3e-3, 1e-3)
    fd = {key: [] for key in ("eta", "nu", "k", "alpha")}
    for h in hs:
        fd["eta"].append((_dense_lead(S, k, nx, Lx, alpha, (nu, eta*(1 + h)))
                          - _dense_lead(S, k, nx, Lx, alpha, (nu, eta*(1 - h))))/(2*h*eta))
        fd["nu"].append((_dense_lead(S, k, nx, Lx, alpha, (nu*(1 + h), eta))
                         - _dense_lead(S, k, nx, Lx, alpha, (nu*(1 - h), eta)))/(2*h*nu))
        fd["k"].append((_dense_lead(S, k*(1 + h), nx, Lx, alpha, (nu, eta))
                        - _dense_lead(S, k*(1 - h), nx, Lx, alpha, (nu, eta)))/(2*h*k))
        fd["alpha"].append((_dense_lead(S, k, nx, Lx, alpha + h, (nu, eta))
                            - _dense_lead(S, k, nx, Lx, alpha - h, (nu, eta)))/(2*h))
    with checks() as c:
        c.check(f"J(overrides Ly, diss=(nu, eta)) == J(Parameters there): max diff "
                f"{np.abs(Jo - Jp).max():.1e} of {np.abs(Jp).max():.1e}",
                np.abs(Jo - Jp).max() <= 1e-14*np.abs(Jp).max())
        c.check(f"sens_solve lambda (band-LU shift-invert) == eig_dense ({complex(rec['lam']):.12f})",
                abs(complex(rec["lam"]) - lam) < 1e-10*abs(lam), f"{rec['lam']} vs {lam}")
        for key in ("eta", "nu", "k", "alpha"):
            sens = complex(rec["dlam_" + key])
            errs = [abs(f - sens) for f in fd[key]]
            order = fit_order(hs, errs)
            print(f"  d gamma/d {key}: sens {sens.real:+.10e}; FD errs {errs[0]:.1e} -> {errs[-1]:.1e} "
                  f"(order {order:.2f})")
            c.check(f"d gamma/d {key} == eigenvalue FD at O(h^2) (order {order:.2f})",
                    1.85 < order < 2.15 and errs[-1] < 2e-5*abs(sens), f"{errs}")
        g = complex(rec["lam"]).real
        ge, gn = complex(rec["dlam_eta"]).real, complex(rec["dlam_nu"]).real
        c.check("q_Pm(rec) == (nu/gamma) gamma_nu", abs(V.q_Pm(rec) - nu*gn/g) < 1e-14)
        c.check("q_S(rec) == -(eta gamma_eta + nu gamma_nu)/gamma",
                abs(V.q_S(rec) + (eta*ge + nu*gn)/g) < 1e-14)
        c.check("q_Snu(rec) == q_S + q_Pm == -(eta/gamma) gamma_eta",
                abs(V.q_Snu(rec) + eta*ge/g) < 1e-14
                and abs(V.q_Snu(rec) - V.q_S(rec) - V.q_Pm(rec)) < 1e-13)
        c.check("nu_eta_ratio(rec) == gamma_nu/gamma_eta", abs(V.nu_eta_ratio(rec) - gn/ge) < 1e-15)
        c.check("ETA_DIR/NU_DIR are the (nu, eta) unit tangents",
                tuple(V.ETA_DIR) == (0.0, 1.0) and tuple(V.NU_DIR) == (1.0, 0.0))
        c.check("overrides(k, S, Pm) = {Ly: 2 pi/k, diss: (Pm/S, 1/S)}",
                V.overrides(0.5, 1e3, 3.0) == {"Ly": 2*np.pi/0.5, "diss": (3e-3, 1e-3)})
        # the layer width (JPP 2025 sec. 7 quarter-max of Re psi'') of the dense eigenvector, centred
        # on the sheet at nx/2
        T = V.T
        idx = stability.ky_block_index(jr.setup_kgrids(p), p, 1)
        xe, pe, fe, de = T._eigfun(p, idx, stability.eig_dense(Jp).right[:, 0], nx//2)
        pr = T._profile(xe, pe, fe, de, nx//2, Lx, half=1.5)
        wref = T.width_quarter(pr["xn"], pr["d2n"])
        c.check(f"width {float(rec['width']):.6f} == the dense eigenvector's {wref:.6f}",
                abs(float(rec["width"])/wref - 1) < 1e-6 and 0 < wref < 1.5)
        c.check("residuals small and isolated", float(rec["res_r"]) < 1e-9
                and float(rec["res_l"]) < 1e-9 and float(rec["rel_err"]) < 1e-3)
    # eig_only (the campaign FD gate's solver) moves eta / nu only, and k through Ly
    ref = _dense_lead(S, 0.55, nx, Lx, alpha, (nu*1.2, eta*0.9))
    got = V.eig_only(S, Pm, 0.55, alpha, nx, Lx, ref*1.03, lam_hint=ref, eta=eta*0.9, nu=nu*1.2)
    ref0 = _dense_lead(S, 0.55, nx, Lx, alpha, (nu, eta))
    got0 = V.eig_only(S, Pm, 0.55, alpha, nx, Lx, ref0*1.03, lam_hint=ref0)
    with checks() as c:
        c.check(f"eig_only(eta, nu moved) == eig_dense of Parameters there ({got['lam']:.12f})",
                abs(got["lam"] - ref) < 1e-10*abs(ref))
        c.check("eig_only default is (Pm/S, 1/S)", abs(got0["lam"] - ref0) < 1e-10*abs(ref0))


# ============================================================ (iv) kmax_solve end to end

def _brute_kmax(S, Pm, alpha, nx, Lx, lo=0.4, hi=0.9):
    # max over k of eig_dense eigenvalues of a Parameters built at diss = (nu, eta) (k by the Ly
    # override, rung 2's established route), bounded Brent in ln k; no sensitivity
    import jax.numpy as jnp
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi, dims=2, cfl_safety=0.5, forcing=False,
                           eqpars={"diss": (Pm/S, 1.0/S), "hyper": 1})
    kg = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    p0 = sum(_sech2(x - Lx/2 - m*Lx) for m in (-1, 0, 1))
    phi, psi = jnp.asarray(alpha*p0).reshape(1, -1, 1), jnp.asarray(p0).reshape(1, -1, 1)
    x0 = jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields

    def f(lk):
        B, _ = stability.ky_block_matrix(x0, kg, params, 1, overrides={"Ly": 2*np.pi/np.exp(lk)})
        return -stability.eig_dense(B).values[0].real
    r = minimize_scalar(f, bounds=(np.log(lo), np.log(hi)), method="bounded", options=dict(xatol=1e-9))
    return float(np.exp(r.x)), float(-r.fun)


@pytest.mark.fp64
def test_kmax_solve_matches_brute_force_maximum():
    V = _run_script()
    S, Pm, alpha, nx = 1e3, 10.0, 0.6, 512
    Lx = V.R.kmax_box(S)[0]
    root = tempfile.mkdtemp(prefix="taranis_visc_kmax_")
    k0, g0 = 0.6, 0.04
    res = V.kmax_solve(S, Pm, alpha, k0, g0, -2*0.3*g0/0.05/k0**2, nx, Lx, root=root, log=print)
    kb, gb = _brute_kmax(S, Pm, alpha, nx, Lx)
    h = 0.05
    kPp, gPp = _brute_kmax(S, Pm*np.exp(h), alpha, nx, Lx)
    kPm, gPm = _brute_kmax(S, Pm*np.exp(-h), alpha, nx, Lx)
    kSp, _ = _brute_kmax(S*np.exp(h), Pm, alpha, nx, Lx)
    kSm, _ = _brute_kmax(S*np.exp(-h), Pm, alpha, nx, Lx)
    skP, sgP = np.log(kPp/kPm)/(2*h), np.log(gPp/gPm)/(2*h)
    skS = np.log(kSp/kSm)/(2*h)
    print(f"  Newton k* = {res['kmax']:.10f} gamma* = {res['gmax']:.12e} ({res['iterations']} its); "
          f"brute k = {kb:.10f} gamma = {gb:.12e}")
    print(f"  d ln k*/d ln Pm: {res['slope_k_Pm']:+.6f} vs {skP:+.6f}; d ln g*/d ln Pm: "
          f"{res['slope_g_Pm']:+.7f} vs {sgP:+.7f}; d ln k*/d ln S|Pm: {res['slope_k_S']:+.6f} vs {skS:+.6f}")
    with checks() as c:
        c.check(f"Newton k* == brute-force argmax ({abs(res['kmax']/kb - 1):.1e})", abs(res["kmax"]/kb - 1) < 2e-6)
        c.check(f"gamma* == brute-force max ({abs(res['gmax']/gb - 1):.1e})", abs(res["gmax"]/gb - 1) < 1e-10)
        c.check("implicit d ln k*/d ln Pm == FD of the brute-force argmax", abs(res["slope_k_Pm"] - skP) < 2e-3)
        c.check("envelope d ln gamma*/d ln Pm == FD of the brute-force max", abs(res["slope_g_Pm"] - sgP) < 2e-4)
        c.check("implicit d ln k*/d ln S (fixed Pm) == FD", abs(res["slope_k_S"] - skS) < 2e-3)
        recs = [res["rec"], res["rec_p"], res["rec_m"]]
        c.check("every solve on the same (nx, Lx) and at this Pm",
                all(int(r["nx"]) == nx and float(r["Lx"]) == Lx and float(r["Pm"]) == Pm for r in recs))
        c.check("curvature pair at k*(1 +- CURV_H)",
                abs(float(res["rec_p"]["k"]) - res["kmax"]*(1 + V.CURV_H)) < 1e-15
                and abs(float(res["rec_m"]["k"]) - res["kmax"]*(1 - V.CURV_H)) < 1e-15)
        c.check("grid check at nx/2", int(res["rec_lo"]["nx"]) == nx//2 and np.isfinite(res["grid_dk"]))
        c.check("polished k* agrees with the secant k*", abs(res["k_polished"] - res["kmax"]) < 1e-7*res["kmax"])


# ============================================================ (iii') reviewer gates: the helpers the
# campaign and the notebook use that the tests above do not reach (each found by a surviving one-line
# mutation in the 2026-10-01 adversarial review)

def test_dense_others_ordering_gap_and_growth_count():
    # _dense_others on a stub block with known eigenvalues: others = every eigenvalue but the
    # fastest, in decreasing real part; npos counts Re > 1e-10 * (inf-norm), so round-off-positive
    # members of the null cluster are NOT growing
    V = _run_script()
    vals = np.array([-3.0, 1e-13, 0.02 + 0.01j, -1e-4, 0.05, 0.02 - 0.01j])

    class _St:
        @staticmethod
        def ky_block_matrix(x0, kg, params, iky, overrides=None):
            return np.diag(vals), None
    with _patched(V, st=_St), _patched(V.R, make_params=lambda *a: None, _kgrid=lambda *a: None), \
            _patched(V.U, equilibrium=lambda p, al: (None, None)), checks() as c:
        others, top, npos = V._dense_others.__wrapped__(1e3, 1.0, 0.5, 0.3, 30.0, 64)
        c.check("top = the largest real part", top == 0.05)
        c.check("others = the rest, decreasing real part",
                np.allclose(others.real, sorted(vals.real, reverse=True)[1:]) and len(others) == 5)
        c.check(f"npos = 3 (1e-13 is below 1e-10 * |B|_inf), got {npos}", npos == 3)


def test_kmax_solve_grid_check_signs():
    # the nx/2 grid check: grid_dk = k*(nx/2) - k*(nx) from one Newton step, grid_dgamma =
    # gamma(nx/2) - gamma(nx), on an analytic gamma whose maximum moves with nx
    V = _run_script()
    R = V.R
    k0, G, A, nx = 0.47, 0.05, 0.3, 512
    k_lo, G_lo = k0*(1 + 2e-3), G*(1 + 1e-4)

    def fake_point(S, Pm, k, alpha, nx_, Lx, pred, **kw):
        kk, GG = (k0, G) if nx_ == nx else (k_lo, G_lo)
        x = np.log(k) - np.log(kk)
        return {"lam": GG - A*x*x + 0j, "dlam_k": -2*A*x/k + 0j, "dlam_eta": 0j, "dlam_nu": 0j,
                "dlam_alpha": 0j, "kappa": 1.0, "nx": nx_, "Lx": Lx, "k": k}
    with _patched(V, point=fake_point), checks() as c:
        out = V.kmax_solve(1e3, 10.0, 0.0, 0.50, G, -2*A/0.5**2, nx, 30.0, log=lambda *a: None)
        kl = out["its"][-1]["k"]
        c.check(f"grid_dk {out['grid_dk']:.3e} ~ k*(nx/2) - k*(nx) = {k_lo - k0:.3e}",
                abs(out["grid_dk"]/(k_lo - k0) - 1) < 0.05)
        ref = (G_lo - A*np.log(kl/k_lo)**2) - (G - A*np.log(kl/k0)**2)
        c.check(f"grid_dgamma = gamma(nx/2) - gamma(nx) ({out['grid_dgamma']:.3e} vs {ref:.3e})",
                abs(out["grid_dgamma"] - ref) < 1e-15)


def test_point_retry_ladder_shifts():
    # each retry moves sigma to pred (1 + offset) and loosens the inner solve as ATTEMPTS says
    V = _run_script()
    got = []

    def fake_sens(S, Pm, k, alpha, nx, Lx, sig, pred, rtol_mult=1.0, **kw):
        got.append((sig, rtol_mult))
        if len(got) < len(V.ATTEMPTS):
            raise RuntimeError("refused")
        return {"lam": 0.05 + 0j}
    with _patched(V.U, cached=lambda kind, root=None, compute=None, **cfg: compute()), \
            _patched(V, sens_solve=fake_sens), checks() as c:
        V.point(1e3, 10.0, 0.6, 0.3, 512, 30.0, 0.05, log=lambda *a: None)
        c.check("sigma = pred (1 + offset), rtol multiplier per ATTEMPTS",
                np.allclose([g[0] for g in got], [0.05*(1 + o) for o, _ in V.ATTEMPTS], rtol=1e-15)
                and [g[1] for g in got] == [m for _, m in V.ATTEMPTS])


def test_sens_solve_inner_tolerance_ladder():
    # the loosened inner GMRES of a retry: gmres_rtol capped at 1e-6 rtol_mult, restart 60
    V = _run_script()
    seen = {}

    class _Stop(Exception):
        pass

    def stop(op, sigma, **kw):
        seen.update(kw)
        raise _Stop
    S, Pm, k, al, nx, Lx = 1e3, 10.0, 0.5, 0.3, 256, 30.0
    with _patched(V.st, shift_invert=stop):
        for rm in (1.0, 1e4):
            seen.clear()
            with pytest.raises(_Stop):
                V.sens_solve(S, Pm, k, al, nx, Lx, 0.05, lam_hint=0.0499, rtol_mult=rm)
            p = V.R.make_params(S, nx, Lx)
            x0 = V.U.equilibrium(p, al)[0]
            op, idx = stability.ky_block_operator(x0, V.R._kgrid(S, nx, Lx), p, 1,
                                                  overrides=V.overrides(k, S, Pm))
            A_ = V.U.band_matrix(op, idx, nx, V.T.band_width("sech2", Lx))
            normA = float(abs(A_).sum(axis=0).max())
            ref = float(np.clip(300*rm*np.finfo(float).eps*normA/1e-4, 1e-10, 1e-6*max(rm, 1.0)))
            assert abs(seen["gmres_rtol"]/ref - 1) < 1e-12, (rm, seen["gmres_rtol"], ref)
            assert seen["restart"] == (30 if rm == 1.0 else 60), seen


def test_helper_boxes_lists_and_branches():
    V = _run_script()
    R = V.R
    # dispersion: the next ka's seed comes from the FIRST level (nx), not nx/2
    seen = []

    def fake_point(S, Pm, k, alpha, nx, Lx, pred, **kw):
        seen.append(dict(k=k, nx=nx, pred=pred))
        lo = 0.0 if nx == R.disp_box(S, k)[1] else 0.5
        return {"lam": 0.03 + 0.01*k + lo + 0j, "dlam_k": 0.01 + 0j, "dlam_eta": -3.0 + 0j,
                "dlam_nu": 0.5 + 0j, "k": k, "kappa": 5.0, "seconds": 0.0, "S": S, "Pm": Pm}
    with _patched(V, point=fake_point, gamma_theory_visc=lambda ka, S, Pm, root=None: 0.05 if ka == 0.3 else np.nan,
                  _sigma_seed=lambda S, Pm, k, a, Lx, g, compute: g), checks() as c:
        V.dispersion(1e4, 10.0, 0.0, kas=(0.3, 0.6), log=lambda *a: None)
        c.check("dispersion: seed from the nx record", all(abs(s_["pred"] - 0.036) < 1e-15
                                                          for s_ in seen if s_["k"] == 0.6))
    # branch_scan: the k_max box at the coarse NX_DENSE, kind 'scanv'
    cfgs = []

    def rec_cached(kind, root=None, compute=None, **cfg):
        cfgs.append((kind, cfg))
        return compute()
    with _patched(V, _dense_others=lambda S, Pm, k, a, Lx, nx: (np.array([-1e-3 + 0j]), 0.01 + 0j, 1)), \
            _patched(V.U, cached=rec_cached), checks() as c:
        V.branch_scan(1e4, 10.0, 0.95, kas=(0.3, 0.5), log=lambda *a: None)
        c.check("branch_scan on (kmax_box(S), NX_DENSE)", all(
            k_ == "scanv" and cf["Lx"] == R.kmax_box(1e4)[0] and cf["nx"] == V.NX_DENSE for k_, cf in cfgs))
    # kmax_grid2 replays the right nx chain: the S's alpha list up to alpha, or the Pm chain
    calls = []

    def fake_ka(S, Pm, alphas, root=None, log=None, compute=True, deadline=np.inf):
        calls.append(("alpha", S, Pm, tuple(alphas), compute))
        return {alphas[-1]: _fake_solution(S, Pm, alphas[-1], 0.6, 0.01)}, True

    def fake_kp(S, alpha, Pms, root=None, log=None, compute=True, deadline=np.inf):
        calls.append(("Pm", S, alpha, tuple(Pms), compute))
        return {Pms[-1]: _fake_solution(S, Pms[-1], alpha, 0.6, 0.01)}, True
    with _patched(V, kmax_alpha=fake_ka, kmax_Pm=fake_kp, kmax_solve=lambda *a, **k: a), checks() as c:
        V.kmax_grid2(1e4, 10.0, 0.6)
        V.kmax_grid2(1e5, 10.0, 0.6)
        V.kmax_grid2(1e5, 100.0, 0.0)
        c.check(f"kmax_grid2 chains {calls}", calls == [
            ("alpha", 1e4, 10.0, (0.0, 0.3, 0.6), False),
            ("alpha", 1e5, 10.0, tuple(a for a in V.KA_HI[1e5][1] if a <= 0.6), False),
            ("Pm", 1e5, 0.0, V._pms_upto(100.0), False)])
    # kmax_Pm's first sigma seed is taken at the chain's own alpha
    sig = []
    with _patched(V, seed_theory=lambda S, Pm, a, root=None: (0.44, 0.03, -0.2),
                  _sigma_seed=lambda S, Pm, k, a, Lx, g, compute: sig.append(a) or g,
                  kmax_solve=lambda *a, **k: _fake_solution(a[0], a[1], a[2], 0.5, 0.02)):
        V.kmax_Pm(1e4, 0.9, (1.0,), log=lambda *a: None)
        assert sig == [0.9], sig
    # small_alpha_c2 at Pm = 0 reads rung 2's alpha = 0 k_max at the SAME S
    rs = []
    with _patched(R, kmax_alpha=lambda S, als, **kw: rs.append((S, als)) or ({0.0: dict(kmax=0.5, gmax=0.02)}, True)), \
            _patched(V, point=lambda S, Pm, k, al, nx, Lx, pred, **kw: rs.append((S, Pm, k)) or
                     {"lam": 0.02 + 0j, "dlam_alpha": -0.02*al + 0j}):
        V.small_alpha_c2(1e5, 0.0, log=lambda *a: None)
        assert rs[0] == (1e5, (0.0,)) and all(r_ == (1e5, 0.0, 0.5) for r_ in rs[1:]), rs
    # read-back helpers: levels, alpha and the full (S, Pm) set
    rb = []
    with _patched(V, dispersion=lambda S, Pm, al, root=None, log=None, compute=True, levels=(1, 2), **k:
                  rb.append(("disp", al, levels, compute)) or ({}, True),
                  kmax_Pm=lambda S, al, Pms=None, root=None, log=None, compute=True, **k:
                  rb.append(("kp", S, al, compute)) or ({}, True),
                  small_alpha_c2=lambda S, Pm, root=None, log=None, compute=True:
                  rb.append(("c2", S, Pm, compute)) or {}), checks() as c:
        V.disp_results(1e4, 10.0); V.disp_results(1e4, 10.0, 0.9)
        V.kmax_Pm_results(1e4, 0.9)
        V.c2_results()
        c.check("disp_results: (1, 2) at alpha = 0, (1,) with shear, read-only",
                rb[:2] == [("disp", 0.0, (1, 2), False), ("disp", 0.9, (1,), False)])
        c.check("kmax_Pm_results keeps alpha", rb[2] == ("kp", 1e4, 0.9, False))
        c.check("c2_results: every S_CHAIN x PM_C2 (Pm = 0 included)",
                rb[3:] == [("c2", S, p, False) for S in V.S_CHAIN for p in V.PM_C2])
    # make_data's filters select by the right coordinate
    hit = []
    with _patched(V, fd_gate=lambda *pt, **k: hit.append(("fd",) + pt),
                  kmax_Pm=lambda S, al, **k: hit.append(("kp", S, al)) or ({}, True),
                  dispersion=lambda S, Pm, al, **k: hit.append(("disp", S, Pm, al)) or ({}, True)):
        root = tempfile.mkdtemp(prefix="taranis_visc_md_")
        V.make_data(root=root, phases=("fd",), P_list=(10.0,), log=lambda *a: None)
        V.make_data(root=root, phases=("kmax_Pm",), A_list=(0.9,), log=lambda *a: None)
        V.make_data(root=root, phases=("disp",), S_list=(1e4,), P_list=(1.0,), A_list=(0.0,),
                    log=lambda *a: None)
        assert hit == ([("fd",) + pt for pt in V.FD_POINTS if pt[3] == 10.0]
                       + [("kp", S, al) for S, al in V.CHAIN_SHEAR if al == 0.9]
                       + [("disp", 1e4, 1.0, 0.0)]), hit
    # theory_pms carries the +-PM_DLN neighbours of every campaign Pm
    tp = np.array(V.theory_pms())
    for p in set(V.PM_CHAIN) | set(V.PM_DISP):
        for s in (0, 1, -1):
            assert np.isclose(tp, p*np.exp(s*V.PM_DLN), rtol=1e-14, atol=0).any(), (p, s)


def test_theory_table_cache_and_pm0_branches():
    V = _run_script()
    T = V.T
    root = tempfile.mkdtemp(prefix="taranis_visc_stale_")
    os.makedirs(V.theory_root(root))
    np.savez(V._table_path(3.0, True, root), lnpole=0.0, X=V.X_TAB[:-1], g=0*V.X_TAB[:-1])
    with pytest.raises(KeyError):        # a table on another grid is never reused
        V.theory_table.__wrapped__(3.0, True, root, compute=False)
    with checks() as c:
        for ka, S in ((0.4, 1e4), (1.2, 1e3)):
            c.check("gamma_theory_visc(Pm = 0) == B&L's gamma_theory",
                    abs(V.gamma_theory_visc(ka, S, 0.0)/T.gamma_theory(ka, S) - 1) < 1e-13)
            e = V.theory_exponents(ka, S, 0.0)
            c.check("theory_exponents(Pm = 0): q_S, q_k from rung 2, q_Pm = 0, q_Snu = q_S",
                    e["qS"] == V.R.theory_s_exponent(ka, S) and e["qk"] == V.R.theory_k_exponent(ka, S)
                    and e["qPm"] == 0.0 and e["qSnu"] == e["qS"])
        c.check("dprime < 0 (ka > sqrt 5): no theory growth rate", np.isnan(V.gamma_theory_visc(2.3, 1e3, 1.0)))


def test_alpha_c_bisection_on_a_synthetic_boundary():
    # gamma_max(alpha) = 0.2 (alpha0 - alpha) above the boundary, the null-cluster top -3e-6 below:
    # the bracket halves AC_NBIS times around alpha0, the linear extrapolation is exact, the 2 nx
    # re-evaluation is at the final bracket; a bracket that does not straddle 0 raises
    V = _run_script()
    a0 = 0.96858
    calls = []

    def fake(S, Pm, alpha, nx=V.NX_DENSE, root=None, compute=True, kas=None):
        calls.append((alpha, nx))
        return {"gmax": 0.2*(a0 - alpha) if alpha < a0 else -3e-6}
    with _patched(V, max_real_growth=fake), checks() as c:
        out = V.alpha_c_bisect(1e4, 100.0, 0.95, 0.99, nbis=7, log=lambda *a: None)
        c.check(f"bracket [{out['lo']:.6f}, {out['hi']:.6f}] contains alpha0, width 0.04/2^7",
                out["lo"] < a0 <= out["hi"] and abs(out["hi"] - out["lo"] - 0.04/2**7) < 1e-15)
        c.check("alpha_c_lin is the linear root", abs(out["alpha_c_lin"] - a0) < 1e-12)
        c.check("2 nx evaluations at the final bracket", calls[-2:] == [(out["lo"], 2*V.NX_DENSE),
                                                                       (out["hi"], 2*V.NX_DENSE)])
        c.check("7 bisection steps + 2 ends + 2 grid checks", len(calls) == 11)
        with pytest.raises(RuntimeError, match="bracket"):
            V.alpha_c_bisect(1e4, 100.0, 0.97, 0.99, log=lambda *a: None)
    seen = []
    with _patched(V, alpha_c_bisect=lambda S, Pm, a1, a2, root=None, log=None, compute=True:
                  seen.append((S, Pm, a1, a2, compute)) or {}):
        V.alpha_c_results()
        assert seen == [p + (False,) for p in V.ALPHA_C_POINTS], seen


@pytest.mark.fp64
def test_max_real_growth_is_the_dense_maximum():
    # max_real_growth (cached, kind "acv") against eig_dense of a Parameters built directly at the
    # returned k, and against the coarse grid; complex pairs are not growth rates
    V = _run_script()
    S, Pm, al, nx = 1e3, 10.0, 0.6, 256
    root = tempfile.mkdtemp(prefix="taranis_visc_acv_")
    r = V.max_real_growth(S, Pm, al, nx, root=root)
    Lx = V.R.kmax_box(S)[0]
    lam = stability.eig_dense(_direct(S, r["kmax"], nx, Lx, al, (Pm/S, 1/S))).values
    with checks() as c:
        c.check(f"gmax {r['gmax']:.10e} == top real eigenvalue of the direct block at kmax",
                abs(r["gmax"] - lam[0].real) < 1e-10*abs(lam[0]) and abs(lam[0].imag) < 1e-10)
        c.check("gmax > every coarse value, kmax refined off the grid (the Brent step is used)",
                r["gmax"] > max(r["coarse"]) and min(abs(r["kmax"] - k_) for k_ in V.KA_AC) > 1e-4)
        c.check("cached read-back", V.max_real_growth(S, Pm, al, nx, root=root, compute=False)["gmax"] == r["gmax"])
        c.check("_top_real skips complex pairs", V._top_real([0.1 + 0.05j, 0.1 - 0.05j, 0.02 + 1e-18j, -1.0]) == 0.02
                and V._top_real([0.1 + 0.05j]) == -np.inf)


if __name__ == "__main__":
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
