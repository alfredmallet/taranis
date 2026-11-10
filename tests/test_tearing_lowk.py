# Gate for examples/tearing_lowk_run.py, the data script of examples/tearing-lowk-branch.ipynb:
# identification of the LOW-k maximum of gamma(k) that the rung-2 campaign found for shear-flow
# tearing of Psi0 = sech^2 x, Phi0 = alpha Psi0 near alpha = 1 (dims=2, hyper = 1, S = 1/eta,
# ky block, fp64; nu = 0 except where stated). The script is imported (examples/ on sys.path).
# Small grids throughout (nx = 256, Lx = 20).
#   (i)   the energy budget against an INDEPENDENT route: for ANY block vector (not only an
#         eigenvector) the hand-derived budget R + M + C - D_nu - D_eta and the Elsasser budgets
#         T+- - D+-+- + X+- equal the energy rates 2 Re v^H W J v computed from the operator J with
#         the Fourier-space energy weights W (and E itself matches Parseval);
#   (ii)  the alpha = 1 theorem: at nu = eta the dense block has no growing mode, and for any
#         vector dE-/dt = -D-- <= 0 term by term (T- = X- = 0); at nu = 0 a real growing mode exists
#         and its E- is fed by the resistive cross term X- alone;
#   (iii) lk_solve against independent routes: lambda == eig_dense of a Parameters built directly
#         with Ly = 2 pi/k and diss = (nu, eta) (no overrides); the four sensitivities (eta, nu, k,
#         alpha) == eigenvalue central differences at O(h^2); its structure numbers == the same
#         numbers from the dense eigenvector;
#   (iv)  hand-built vectors for fields_of / structure (phase convention, psi(0) ratio, the
#         antiunitary symmetry residual, the Elsasser fraction);
#   (v)   pure helpers on synthetic inputs: point's retry ladder and cache key, scan_maxima /
#         top_real, bracketed_root (with infinite ends), branch_gap's classification and envelope
#         slope, alpha_c on an analytic two-branch family, the box rules, the Julia grid rule
#         against the reference npz, and the phase drivers' control flow (box_study, pm_results);
#   (vi)  kmax_branch end to end against a brute-force maximum of dense eigenvalues;
#   (vii) review gates (second reviewer's mutation table): each budget term against an operator
#         split (R, M from the alpha-part of J; C, D from J(0)), the structure numbers where phi
#         and psi differ, the cached campaign's seeds, continuation rules and grids, _pred,
#         julia_table, the branch classification thresholds, the Newton route's seed continuation,
#         lk_solve's gap and width, gamma_solve's residual, bracketed_root's precision and cost,
#         kmax_brent's end margin, and the alpha_c box check (boundary_box_check).
# pytest: `pytest tests/test_tearing_lowk.py`; script: `python tests/test_tearing_lowk.py`.
from _rmhd_testing import bootstrap, checks, fit_order

bootstrap()

import importlib
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import jax.numpy as jnp

import taranis as jr
from taranis import stability

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "shear_tearing_reference.npz")

S0, K0, A0, NX, LX = 1e3, 0.36, 0.995, 256, 20.0


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


def _sech2(u):
    e = np.exp(-2.0*np.abs(u))
    return 4.0*e/(1.0 + e)**2


def _x0(params, alpha):
    # the sech^2 equilibrium built HERE (not the script's), +2 images
    x = np.arange(params.nx)*params.Lx/params.nx
    p0 = sum(_sech2(x - params.Lx/2 - m*params.Lx) for m in (-1, 0, 1))
    phi, psi = jnp.asarray(alpha*p0).reshape(1, -1, 1), jnp.asarray(p0).reshape(1, -1, 1)
    return jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields


_PCACHE = {}


def _params(S, nx, Lx):
    # one Parameters per (S, nx, Lx) for the FD routes (k and diss through overrides)
    key = (S, nx, Lx)
    if key not in _PCACHE:
        p = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi, dims=2, cfl_safety=0.5, forcing=False,
                          eqpars={"diss": (0.0, 1.0/S), "hyper": 1})
        _PCACHE[key] = (p, jr.setup_kgrids(p))
    return _PCACHE[key]


def _block(S, k, alpha, nx, Lx, diss):
    p, kg = _params(S, nx, Lx)
    B, idx = stability.ky_block_matrix(_x0(p, alpha), kg, p, 1,
                                       overrides={"Ly": 2*np.pi/k, "diss": tuple(diss)})
    return p, B, idx


def _lead(S, k, alpha, nx, Lx, diss):
    _, B, _ = _block(S, k, alpha, nx, Lx, diss)
    return stability.eig_dense(B).values[0]


def _weights(params, idx, k):
    # energy weight per block entry: E = c sum (kx^2 + k^2)(|Phi|^2 + |Psi|^2), np.fft
    # conventions (f = ifft F, sum_x |f|^2 dx = (Lx/nx^2) sum |F|^2), <|a|^2>_y = |a|^2/2
    nx, Lx = params.nx, params.Lx
    kx = 2*np.pi*np.fft.fftfreq(nx, d=Lx/nx)
    return (kx[idx[:, 2]]**2 + k*k)*0.25*Lx/nx**2


# ============================================================ (i) budget vs operator energy rates

@pytest.mark.fp64
def test_budget_equals_operator_energy_rates_for_any_vector():
    L = _run_script("tearing_lowk_run")
    rng = np.random.default_rng(3)
    with checks() as c:
        for (alpha, nu) in ((A0, 3e-4), (0.6, 0.0), (1.0, 2e-3)):
            eta = 1.0/S0
            p, B, idx = _block(S0, K0, alpha, NX, LX, (nu, eta))
            r = stability.eig_dense(B)
            v = r.right[:, :4] @ (rng.normal(size=4) + 1j*rng.normal(size=4))   # NOT an eigvec
            w = _weights(p, idx, K0)
            phi, psi = idx[:, 0] == 0, idx[:, 0] == 1
            E_W = float(np.sum(w*np.abs(v)**2))
            dE_W = float(2*np.real(np.sum(w*np.conj(v)*(B @ v))))
            # Elsasser: zeta+- = phi +- psi on each kx; Ep = (1/2) c sum (kx^2+k^2)|Z+|^2
            Bv = B @ v
            n = len(idx)//2
            ordp = np.argsort(idx[phi, 2]); ords = np.argsort(idx[psi, 2])
            vp, vs = v[phi][ordp], v[psi][ords]
            Bp, Bs = Bv[phi][ordp], Bv[psi][ords]
            ws = w[phi][ordp]
            assert np.array_equal(idx[phi, 2][ordp], idx[psi, 2][ords]) and len(vp) == n
            Ep_W = float(0.5*np.sum(ws*np.abs(vp + vs)**2))
            Em_W = float(0.5*np.sum(ws*np.abs(vp - vs)**2))
            dEp_W = float(np.real(np.sum(ws*np.conj(vp + vs)*(Bp + Bs))))
            dEm_W = float(np.real(np.sum(ws*np.conj(vp - vs)*(Bp - Bs))))
            # the script's real-space budget of v. fields_of rescales and rephases v: undo by
            # comparing RATIOS (rate/energy) -- both routes are quadratic in v
            _, _, tot = L.budget(p, idx, v, K0, alpha, nu, eta)
            rate = tot["R"] + tot["M"] + tot["C"] - tot["Dnu"] - tot["Deta"]
            rp = tot["Tp"] - tot["Dpp"] + tot["Xp"]
            rm = tot["Tm"] - tot["Dmm"] + tot["Xm"]
            tag = f"alpha={alpha}, nu={nu}"
            c.check(f"{tag}: E split: Ep + Em == E (budget)",
                    abs(tot["Ep"] + tot["Em"] - tot["E"]) < 1e-12*tot["E"])
            c.check(f"{tag}: Elsasser energy fractions == Fourier route",
                    abs(tot["Em"]/tot["E"] - Em_W/E_W) < 1e-10
                    and abs(tot["Ep"]/tot["E"] - Ep_W/E_W) < 1e-10,
                    f"{tot['Em']/tot['E']} {Em_W/E_W}; {tot['Ep']/tot['E']} {Ep_W/E_W}")
            c.check(f"{tag}: dE/dt / E: budget {rate/tot['E']:+.10e} == operator "
                    f"{dE_W/E_W:+.10e}", abs(rate/tot["E"] - dE_W/E_W) < 1e-9*abs(dE_W/E_W) + 1e-12)
            c.check(f"{tag}: dEp/dt / E: budget == operator",
                    abs(rp/tot["E"] - dEp_W/E_W) < 1e-9*abs(dE_W/E_W) + 1e-12,
                    f"{rp/tot['E']} vs {dEp_W/E_W}")
            c.check(f"{tag}: dEm/dt / E: budget == operator",
                    abs(rm/tot["E"] - dEm_W/E_W) < 1e-9*abs(dE_W/E_W) + 1e-12,
                    f"{rm/tot['E']} vs {dEm_W/E_W}")
            c.check(f"{tag}: the drives are non-trivial (each term enters the check)",
                    abs(tot["C"]) > 1e-3*abs(rate) and (alpha == 0 or abs(tot["M"]) > 0))


# ============================================================ (ii) alpha = 1

@pytest.mark.fp64
def test_alpha1_no_growth_at_pm1_and_resistive_feed_at_nu0():
    L = _run_script("tearing_lowk_run")
    eta = 1.0/S0
    rng = np.random.default_rng(5)
    with checks() as c:
        for k in (0.2, K0, 1.0):
            p, B, idx = _block(S0, k, 1.0, NX, LX, (eta, eta))
            vals = stability.eig_dense(B).values
            c.check(f"alpha = 1, nu = eta, k = {k}: max Re lambda = {vals[0].real:.3e} < 0",
                    vals[0].real < 0)
            v = stability.eig_dense(B).right[:, :6] @ (rng.normal(size=6) + 1j*rng.normal(size=6))
            _, _, tot = L.budget(p, idx, v, k, 1.0, eta, eta)
            c.check(f"alpha = 1, nu = eta, k = {k}: T- == 0 and X- == 0 exactly; D-- >= 0",
                    tot["Tm"] == 0.0 and tot["Xm"] == 0.0 and tot["Dmm"] >= 0.0)
        p, B, idx = _block(S0, K0, 1.0, NX, LX, (0.0, eta))
        r = stability.eig_dense(B)
        lam = r.values[0]
        c.check(f"alpha = 1, nu = 0: a REAL growing mode ({lam:.6e})",
                lam.real > 1e-3 and abs(lam.imag) < 1e-10)
        s = L.structure(p, idx, r.right[:, 0], lam, K0, 1.0, 0.0, eta)
        c.check(f"its E- is fed by the resistive cross term alone: T-/2gE = {s['b_Tm']:.1e}, "
                f"X-/2gE = {s['b_Xm']:.2e} > 0", s["b_Tm"] == 0.0 and s["b_Xm"] > 0)
        c.check(f"budget closure {s['closure']:.1e}, Elsasser {s['closure_p']:.1e} "
                f"{s['closure_m']:.1e}", max(s["closure"], s["closure_p"], s["closure_m"]) < 1e-10)


# ============================================================ (iii) lk_solve

@pytest.mark.fp64
def test_lk_solve_matches_dense_and_fd():
    L = _run_script("tearing_lowk_run")
    S, k, al, nu = S0, K0, A0, 1e-4
    eta = 1.0/S
    # INDEPENDENT lambda: a Parameters built with Ly = 2 pi/k and diss = (nu, eta), no overrides
    pd = jr.Parameters(nx=NX, ny=8, Lx=LX, Ly=2*np.pi/k, dims=2, cfl_safety=0.5, forcing=False,
                       eqpars={"diss": (nu, eta), "hyper": 1})
    Bd, idxd = stability.ky_block_matrix(_x0(pd, al), jr.setup_kgrids(pd), pd, 1)
    rd = stability.eig_dense(Bd)
    lam = rd.values[0]
    with _patched(L, NX_DENSE=NX):
        rec = L.lk_solve(S, k, al, NX, LX, lam.real*1.03, nu=nu, lam_hint=lam.real)
    hs = (1e-2, 3e-3, 1e-3)
    fd = {"eta": [], "nu": [], "k": [], "alpha": []}
    for h in hs:
        fd["eta"].append((_lead(S, k, al, NX, LX, (nu, eta*(1 + h)))
                          - _lead(S, k, al, NX, LX, (nu, eta*(1 - h))))/(2*h*eta))
        fd["nu"].append((_lead(S, k, al, NX, LX, (nu + h*eta, eta))
                         - _lead(S, k, al, NX, LX, (nu - h*eta, eta)))/(2*h*eta))
        fd["k"].append((_lead(S, k*(1 + h), al, NX, LX, (nu, eta))
                        - _lead(S, k*(1 - h), al, NX, LX, (nu, eta)))/(2*h*k))
        fd["alpha"].append((_lead(S, k, al + h*0.1, NX, LX, (nu, eta))
                            - _lead(S, k, al - h*0.1, NX, LX, (nu, eta)))/(0.2*h))
    s_ref = L.structure(pd, idxd, rd.right[:, 0], lam, k, al, nu, eta)
    with checks() as c:
        c.check(f"lk_solve lambda == eig_dense of a directly built Parameters "
                f"({complex(rec['lam']):.12f})", abs(complex(rec["lam"]) - lam) < 1e-10*abs(lam))
        for key in fd:
            sens = complex(rec["dlam_" + key])
            errs = [abs(f - sens) for f in fd[key]]
            order = fit_order(hs, errs)
            print(f"  d gamma/d {key}: {sens.real:+.10e}; FD errs {errs[0]:.1e} -> {errs[-1]:.1e} "
                  f"(order {order:.2f})")
            c.check(f"d gamma/d {key} == eigenvalue FD at O(h^2) (order {order:.2f})",
                    1.8 < order < 2.2 and errs[-1] < 1e-4*abs(sens), f"{errs}")
        c.check("dgamma/dnu and dgamma/deta are different numbers (the tangents are distinct)",
                abs(complex(rec["dlam_nu"]) - complex(rec["dlam_eta"]))
                > 0.1*abs(complex(rec["dlam_eta"])))
        for key in ("psi0", "phi_over_psi", "x_rms_E", "em_frac", "b_C", "b_R", "b_M", "b_Deta",
                    "b_Dnu", "b_Xm"):
            c.check(f"structure {key}: lk_solve {float(rec[key]):.8f} == dense eigvec "
                    f"{s_ref[key]:.8f}", abs(float(rec[key]) - s_ref[key])
                    < 1e-6*max(1.0, abs(s_ref[key])))
        c.check("closure, residuals and isolation small",
                float(rec["closure"]) < 1e-9 and float(rec["res_r"]) < 1e-9
                and float(rec["res_l"]) < 1e-9 and float(rec["rel_err"]) < 1e-3)
        g = complex(rec["lam"]).real
        c.check("s_exponent == -(eta/gamma) dgamma/deta, pm_exponent == eta dgamma/dnu/gamma",
                abs(L.s_exponent(rec) + eta*complex(rec["dlam_eta"]).real/g) < 1e-14
                and abs(L.pm_exponent(rec) - eta*complex(rec["dlam_nu"]).real/g) < 1e-14)
        c.check("the record carries its configuration", (rec["S"], rec["k"], rec["alpha"],
                rec["nu"], rec["nx"], rec["Lx"]) == (S, k, al, nu, NX, LX))


# ============================================================ (iv) hand-built vectors

def _pack(params, idx, phi, psi):
    # block vector from real-space complex profiles (the inverse of T._eigfun's unpacking)
    F = np.stack([np.fft.fft(phi), np.fft.fft(psi)])
    return F[idx[:, 0], idx[:, 2]]


@pytest.mark.fp64
def test_fields_of_and_structure_on_hand_built_vectors():
    L = _run_script("tearing_lowk_run")
    p, _, idx = _block(S0, K0, 0.0, NX, LX, (0.0, 1.0/S0))
    x = np.arange(NX)*LX/NX - LX/2
    with checks() as c:
        # P-symmetric (f(x) = conj f(-x): even real part, odd imaginary part), psi(0) = 0.5
        psi = (0.5 + 0.5*np.tanh(x)**2)*np.exp(-x*x/4) + 0.3j*x*np.exp(-x*x/2)
        phi = psi.copy()                                      # phi = psi: Em = 0
        ph = np.exp(0.7j)                                     # an arbitrary phase on input
        v = _pack(p, idx, phi*ph, psi*ph)
        xs, f_phi, f_psi = L.fields_of(p, idx, v)
        c.check("fields_of: x centred on the sheet", abs(xs[NX//2]) < 1e-14)
        c.check("fields_of: phase removed, psi(0) real positive",
                abs(f_psi[NX//2].imag) < 1e-12 and f_psi[NX//2].real > 0)
        sc = max(np.abs(psi).max(), np.abs(phi).max())
        c.check("fields_of: scaled to max(|phi|, |psi|) = 1 and == the hand profile",
                np.abs(f_psi - psi/sc).max() < 1e-10 and np.abs(f_phi - phi/sc).max() < 1e-10,
                f"{np.abs(f_psi - psi/sc).max()}")
        s = L.structure(p, idx, v, 0.01, K0, 0.0, 0.0, 1.0/S0)
        c.check(f"psi0 == |psi(0)|/max|psi| = {abs(psi[NX//2])/np.abs(psi).max():.6f}",
                abs(s["psi0"] - abs(psi[NX//2])/np.abs(psi).max()) < 1e-10)
        c.check(f"P-symmetric profile: sym_residual {s['sym_residual']:.1e} at round-off",
                s["sym_residual"] < 1e-10)
        c.check("phi == psi: em_frac == 0 and phi_over_psi == 1",
                s["em_frac"] < 1e-20 and abs(s["phi_over_psi"] - 1) < 1e-12)
        # break the symmetry (a real odd part) and the Elsasser balance (phi = -psi)
        psi2 = psi + 0.2*x*np.exp(-x*x/2)
        s2 = L.structure(p, idx, _pack(p, idx, -psi2, psi2), 0.01, K0, 0.0, 0.0, 1.0/S0)
        c.check(f"non-symmetric profile: sym_residual {s2['sym_residual']:.3f} = O(0.1)",
                s2["sym_residual"] > 0.05)
        c.check("phi == -psi: em_frac == 1", abs(s2["em_frac"] - 1) < 1e-12)
        # |phi| > |psi|: the common scale is max(|phi|, |psi|) = max |phi|
        _, f_phi2, f_psi2 = L.fields_of(p, idx, _pack(p, idx, 2.5*psi, psi))
        c.check("fields_of: scaled by max(|phi|, |psi|) (here phi)",
                abs(np.abs(f_phi2).max() - 1) < 1e-12 and abs(np.abs(f_psi2).max() - 0.4) < 1e-12)
        # psi(0) == 0: the phase is taken at max |psi| instead
        psi3 = x*np.exp(-x*x/2)*(1 + 0.1j)
        _, _, f3 = L.fields_of(p, idx, _pack(p, idx, 0*psi3, psi3))
        i = int(np.argmax(np.abs(f3)))
        c.check("psi(0) = 0: phase fixed at max |psi| (real positive there)",
                abs(f3[i].imag) < 1e-12 and f3[i].real > 0)


# ============================================================ (v) pure helpers

def test_box_rules_overrides_and_exponents():
    L = _run_script("tearing_lowk_run")
    R = L.R
    with checks() as c:
        for S in (1e3, 1e4, 1e5):
            Lx, nx = R.kmax_box(S)
            c.check(f"box(S={S:.0e}) is the rung-2 k_max box", L.box(S) == (Lx, nx))
            for f in (0.5, 2.0, 4.0):
                L2, n2 = L.box(S, f)
                c.check(f"box x{f}: Lx scaled, dx fixed", abs(L2 - Lx*f) < 1e-5
                        and abs(L2/n2 - Lx/nx) < 1e-6*Lx/nx)
            L3, n3 = L.box(S, 1.0, 2.0)
            c.check("nx factor keeps Lx", L3 == Lx and n3 == 2*nx)
        Lg, ng = L.gs_box(1e4)
        c.check("gamma(S) box: Lx = 15/KA_GS, a quarter of the rung-1a nx rule (>= 2048)",
                Lg == 15.0/L.KA_GS and ng == max(2048, R.nx_rule(1e4, L.KA_GS, Lg)//4)
                and L.gs_box(1e6)[1] == max(2048, R.nx_rule(1e6, L.KA_GS, Lg)//4) > 2048)
        ov = L.overrides(0.25, 1e4, 3e-5)
        c.check("overrides: Ly = 2 pi/k, diss = (nu, 1/S)",
                abs(ov["Ly"] - 2*np.pi/0.25) < 1e-14 and ov["diss"] == (3e-5, 1e-4))
        c.check("tangents: eta (0, 1), nu (1, 0)", L.ETA_DIR == (0.0, 1.0) and L.NU_DIR == (1.0, 0.0))
        for u in (-1.0, -5.0, -9.0):
            c.check(f"alpha_of_u inverts u = ln(1 - alpha^2) at u = {u}",
                    abs(np.log(1 - L.alpha_of_u(u)**2) - u) < 1e-9)
        rec = dict(S=1e4, lam=0.01 + 0j, dlam_eta=50.0 + 0j, dlam_nu=-120.0 + 0j)
        c.check("s_exponent = -(eta/gamma) dgamma/deta", abs(L.s_exponent(rec) + 0.5) < 1e-14)
        c.check("pm_exponent = eta dgamma/dnu/gamma", abs(L.pm_exponent(rec) + 1.2) < 1e-14)


def test_point_retries_and_cache_key():
    L = _run_script("tearing_lowk_run")
    seq = {0: RuntimeError("refused"), 1: 1e-17 + 0j, 2: 0.04 + 0.01j, 3: 0.051 + 0j}
    calls = []

    def fake_cached(kind, root=None, compute=None, attempt=None, **cfg):
        calls.append((kind, attempt, cfg, compute()))
        out = seq.get(attempt, 0.049 + 0j)
        if isinstance(out, Exception):
            raise out
        return {"lam": out, "attempt": attempt}

    def fake_solve(S, k, alpha, nx, Lx, sigma, nu=0.0, lam_hint=None, rtol_mult=1.0):
        return dict(sigma=sigma, rtol_mult=rtol_mult, lam_hint=lam_hint, nu=nu)
    with _patched(L.U, cached=fake_cached), _patched(L, lk_solve=fake_solve), checks() as c:
        rec = L.point(1e3, 0.3, 0.6, 512, 30.0, 0.05, nu=2e-4, log=lambda *a: None)
        c.check("refusal, null member, complex pair skipped: attempt 3 returned",
                rec["attempt"] == 3 and [a for _, a, _, _ in calls] == [0, 1, 2, 3])
        c.check("kind 'lk', key = configuration incl. nu, never sigma",
                all(kd == "lk" and cfg == dict(S=1e3, k=0.3, alpha=0.6, nx=512, Lx=30.0, nu=2e-4)
                    for kd, _, cfg, _ in calls))
        c.check("the shift ladder: sigma = pred (1 + off) with the listed offsets, nu passed",
                [round(o["sigma"]/0.05 - 1, 12) for *_, o in calls] == [0.03, 0.13, 0.5, -0.03]
                and all(o["nu"] == 2e-4 and o["lam_hint"] == 0.05 and o["rtol_mult"] == 1.0
                        for *_, o in calls))
        seq.clear(); seq.update({a: -0.01 + 0j for a in range(20)}); calls.clear()
        with pytest.raises(RuntimeError, match="no growing mode"):
            L.point(1e3, 0.3, 0.6, 512, 30.0, 0.05, log=lambda *a: None)
        c.check(f"all 7 attempts before giving up ({len(calls)}), inner rtol loosened on the last 3",
                len(calls) == 7 and [o["rtol_mult"] for *_, o in calls][4:] == [100.0, 100.0, 1e4])
        # alpha >= LOOSE_ALPHA: the loose ladder, marked in the key
        calls.clear()
        with pytest.raises(RuntimeError, match="no growing mode"):
            L.point(1e3, 0.3, L.LOOSE_ALPHA, 512, 30.0, 0.05, log=lambda *a: None)
        c.check("alpha >= LOOSE_ALPHA: the loose ladder (GMRES rtol x100 first), key marked",
                [(round(o["sigma"]/0.05 - 1, 12), o["rtol_mult"]) for *_, o in calls]
                == [tuple(x) for x in L.LADDER_LOOSE] and L.LADDER_LOOSE[0][1] == 100.0
                and all(cfg.get("ladder") == "loose" for _, _, cfg, _ in calls))
        calls.clear()
        with pytest.raises(RuntimeError):
            L.point(1e3, 0.3, 0.98, 512, 30.0, 0.05, log=lambda *a: None)
        c.check("alpha below LOOSE_ALPHA: no ladder key", all("ladder" not in cfg
                                                             for _, _, cfg, _ in calls))
        calls.clear()
        with pytest.raises(RuntimeError):
            L.point(1e4, 0.179, 0.995, 512, 30.0, 0.05, log=lambda *a: None)
        c.check("LOOSE_ALPHA = 0.99: the alpha = 0.995 box points and alpha_c rows are loose "
                "(the cached campaign keys)", L.LOOSE_ALPHA == 0.99
                and all(cfg.get("ladder") == "loose" for _, _, cfg, _ in calls))


def test_scan_helpers():
    L = _run_script("tearing_lowk_run")
    kas = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    g = (1.0, 2.0, 1.5, 1.2, 1.4, 1.3, 1.1)
    rows = {ka: dict(values=np.array([gg + 0j, 0.5 + 0.1j])) for ka, gg in zip(kas, g)}
    with checks() as c:
        c.check("scan_maxima: interior local maxima in order",
                L.scan_maxima(rows) == [(0.2, 2.0), (0.5, 1.4)])
        rows2 = {ka: dict(values=np.array([gg + 0j])) for ka, gg in zip(kas, (3, 2, 1, 1.5, 1, 0.5, 0.2))}
        c.check("scan_maxima: a global maximum at an end point is listed",
                L.scan_maxima(rows2) == [(0.1, 3.0), (0.4, 1.5)])
        rows3 = {ka: dict(values=np.array([gg + 0j])) for ka, gg in zip(kas, (1, 2, 3, 4, 5, 6, 7))}
        c.check("scan_maxima: a monotone rise has only the end point", L.scan_maxima(rows3)
                == [(0.7, 7.0)])
        c.check("top_real skips complex pairs and decaying modes",
                L.top_real([0.3 + 0.2j, 0.1 + 1e-14j, -0.5]) == 0.1
                and L.top_real([-0.1, 0.2 + 0.1j]) == 0.0)
        seen = []

        def fake_dense_top(S, k, alpha, nx, Lx, nu=0.0, root=None, compute=True, ntop=8):
            seen.append((S, k, alpha, nx, Lx))
            if not compute and k > 0.5:
                raise KeyError("missing")
            return dict(values=np.array([0.01*k + 0j]))
        with _patched(L, dense_top=fake_dense_top):
            out = L.scan(1e4, 0.99, kas=(0.2, 0.4, 0.8), log=lambda *a: None)
            Lx, nx = L.box(1e4)
            c.check("scan: the rung-2 box, nx = min(NX_DENSE, box nx)",
                    all(s_[3] == min(L.NX_DENSE, nx) and s_[4] == Lx for s_ in seen))
            c.check("scan: every ka", sorted(out) == [0.2, 0.4, 0.8])
            out = L.scan(1e4, 0.99, kas=(0.2, 0.4, 0.8), compute=False)
            c.check("scan read-back skips uncached ka", sorted(out) == [0.2, 0.4])


def test_bracketed_root_with_infinite_ends():
    L = _run_script("tearing_lowk_run")
    root, slope = -4.2, -1.3
    calls = []

    def f(u):
        calls.append(u)
        if u > root + 1.0:
            return -np.inf, np.nan                # the low branch missing
        if u < root - 1.0:
            return np.inf, np.nan                 # the high branch missing
        return slope*(u - root) + 0.05*(u - root)**2, slope + 0.1*(u - root)
    with checks() as c:
        u, its = L.bracketed_root(f, -8.0, -1.0, tol=1e-3)
        c.check(f"root found ({u:.8f} vs {root})", abs(u - root) < 1e-3)
        c.check(f"Newton finishes it: <= 8 evaluations inside ({len(its)})", len(its) <= 8)
        u2, _ = L.bracketed_root(f, -1.0, -8.0, tol=1e-3)
        c.check("bracket order does not matter", abs(u2 - root) < 1e-3)
        with pytest.raises(RuntimeError, match="no sign change"):
            L.bracketed_root(f, -1.5, -0.5)
        # a linear f: one Newton step lands on the root
        calls.clear()
        u3, its3 = L.bracketed_root(lambda u: (2.0*(u - 0.3), 2.0), -1.0, 2.0, tol=1e-6)
        c.check("linear f: exact root", abs(u3 - 0.3) < 1e-12, f"{u3}")
        # loose tol: the loop stops on the bracket width, the answer is the final interpolation
        u4, its4 = L.bracketed_root(lambda u: (2.0*(u - 0.3), 2.0), -1.0, 2.0, tol=2.0)
        c.check(f"bracket-width stop: linear interpolation in the last bracket ({u4})",
                len(its4) == 1 and abs(u4 - 0.3) < 1e-12)
        # no slope (nan): secants on the last finite value, as fast as Newton for a linear f
        calls.clear()
        u5, its5 = L.bracketed_root(lambda u: (-1.3*(u - 0.3) + 0.01*(u - 0.3)**2, np.nan),
                                    -1.0, 2.0, tol=1e-8)
        c.check(f"nan slope: secant iteration converges in <= 8 evaluations ({len(its5)})",
                abs(u5 - 0.3) < 1e-8 and len(its5) <= 8)


def _fake_branch(table):
    # kmax_branch stand-in: table(k0) -> (kmax, gmax, dgamma/dalpha) or an exception
    def fake(S, alpha, k0, nu=0.0, root=None, log=print, compute=True, Lx=None, nx=None, c0=None):
        out = table(alpha, k0)
        if isinstance(out, Exception):
            raise out
        km, gm, ga = out
        return dict(kmax=km, gmax=gm, rec=dict(dlam_alpha=ga + 0j))
    return fake


def test_branch_gap_classification_and_envelope_slope():
    L = _run_script("tearing_lowk_run")
    al = 0.99
    da = -(1 - al*al)/(2*al)
    with checks() as c:
        both = _fake_branch(lambda a, k0: (0.2, 3e-3, 0.5) if k0 < 0.4 else (0.7, 2e-3, -0.1))
        with _patched(L, kmax_branch=both):
            g, d, lo, hi = L.branch_gap(1e4, al, 0.2, 0.7, log=lambda *a: None)
            c.check("both maxima: gap = ln(g_lo/g_hi)", abs(g - np.log(1.5)) < 1e-14)
            c.check("envelope slope: (g_a,lo/g_lo - g_a,hi/g_hi) dalpha/du",
                    abs(d - (0.5/3e-3 + 0.1/2e-3)*da) < 1e-10*abs(d))
        only_hi = _fake_branch(lambda a, k0: (0.69, 2e-3, 0.0))
        with _patched(L, kmax_branch=only_hi):
            g, d, lo, hi = L.branch_gap(1e4, al, 0.2, 0.7, log=lambda *a: None)
            c.check("both Newtons on the high maximum: low missing, gap = -inf",
                    g == -np.inf and lo is None and hi["kmax"] == 0.69)
        only_lo = _fake_branch(lambda a, k0: (0.21, 2e-3, 0.0))
        with _patched(L, kmax_branch=only_lo):
            g, d, lo, hi = L.branch_gap(1e4, al, 0.2, 0.7, log=lambda *a: None)
            c.check("both Newtons on the low maximum: high missing, gap = +inf",
                    g == np.inf and hi is None)
        fails = _fake_branch(lambda a, k0: RuntimeError("iterations") if k0 > 0.4
                             else (0.2, 3e-3, 0.0))
        with _patched(L, kmax_branch=fails):
            g, *_ = L.branch_gap(1e4, al, 0.2, 0.7, log=lambda *a: None)
            c.check("a non-converged Newton counts as missing", g == np.inf)
        merged = _fake_branch(lambda a, k0: (0.36, 2e-3, 0.0) if k0 < 0.4 else (0.40, 2e-3, 0.0))
        with _patched(L, kmax_branch=merged):
            with pytest.raises(RuntimeError, match="merged"):
                L.branch_gap(1e4, al, 0.35, 0.42, log=lambda *a: None)
        none = _fake_branch(lambda a, k0: RuntimeError("iterations"))
        with _patched(L, kmax_branch=none):
            with pytest.raises(RuntimeError, match="no maximum"):
                L.branch_gap(1e4, al, 0.2, 0.7, log=lambda *a: None)


def test_alpha_c_on_analytic_two_branch_family():
    # gamma*_low = 0.01 (alpha-independent), gamma*_high = 0.05 (1 - alpha^2)^0.6: equal at
    # 1 - alpha_c^2 = 0.2^(1/0.6). The low maximum exists only for 1 - alpha^2 < 3 u_c', the high
    # only for 1 - alpha^2 > u_c'/3 (a coexistence window around alpha_c, as in the data)
    L = _run_script("tearing_lowk_run")
    uc = 0.2**(1/0.6)

    def table(alpha, k0):
        q = 1 - alpha*alpha
        lo_ok, hi_ok = q < 3*uc, q > uc/3
        if k0 < 0.4:
            return (0.2, 0.01, 0.0) if lo_ok else (0.7, 0.05*q**0.6, 0.0)
        if hi_ok:
            return (0.7, 0.05*q**0.6, 0.05*0.6*q**-0.4*(-2*alpha))
        return (0.2, 0.01, 0.0)
    with _patched(L, kmax_branch=_fake_branch(table)), checks() as c:
        res = L.alpha_c(1e4, np.log(uc/10), np.log(uc*10), 0.2, 0.7, tol=1e-4,
                        log=lambda *a: None)
        q = 1 - res["alpha"]**2
        c.check(f"alpha_c: 1 - alpha_c^2 = {q:.8e} vs analytic {uc:.8e}", abs(q/uc - 1) < 1e-4)
        c.check("u and alpha consistent", abs(np.log(q) - res["u"]) < 1e-12)
        c.check("rows carry both branches near the root", res["rows"][-1]["lo"] is not None
                and res["rows"][-1]["hi"] is not None)


def test_julia_grid_rule_matches_reference():
    L = _run_script("tearing_lowk_run")
    ref = np.load(_REF)
    with checks() as c:
        n = 0
        for i in range(len(ref["S"])):
            if int(ref["whichf"][i]) != 2:
                continue
            S, ka, al = float(ref["S"][i]), float(ref["ka"][i]), float(ref["alpha"][i])
            jobs = L.julia_jobs(S, al, ka, 0.01, xlims=(10.0,), levels=(1.0, 0.5))
            n += 1
            c.check(f"grid rule at (S={S:.0e}, ka={ka}, alpha={al}): dxmin {jobs[0][4]:.6e} == "
                    f"reference {float(ref['dxmin'][i, 0]):.6e}",
                    abs(jobs[0][4] - float(ref["dxmin"][i, 0])) < 1e-12*jobs[0][4]
                    and jobs[0][6] == float(ref["eps"][i, 0]) and jobs[0][5] == 10.0)
            c.check("level 0.5 halves dxmin and eps jointly",
                    abs(jobs[1][4] - 0.5*jobs[0][4]) < 1e-15 and jobs[1][6] == 0.01)
        c.check(f"{n} reference points", n > 20)
        jobs = L.julia_jobs(1e4, 1.0, 0.18, 3e-3)
        c.check("alpha = 1: the no-flow width sets dxmin (delta_in undefined)",
                abs(jobs[0][4] - ((1e4*0.18)**(-1/3) if 0.18 < 1e4**-0.25
                                  else 1e4**-0.4*0.18**-0.6)/50) < 1e-15)
        c.check("every xlim x level", len(jobs) == len(L.JL_XLIM)*len(L.JL_LEVELS))


def test_phase_drivers_control_flow(tmp_path):
    L = _run_script("tearing_lowk_run")
    seen = []

    def fake_point(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True):
        seen.append(dict(S=S, k=k, alpha=alpha, nx=nx, Lx=Lx, nu=nu, pred=pred))
        if not compute and nx > 4096:
            raise KeyError("missing")
        return dict(lam=0.003 + 0j, dlam_eta=0.0 + 0j, S=S, seconds=1.0)

    def fake_pred(S, k, alpha, Lx, nu=0.0, compute=True, fallback=None):
        return 0.0031
    with _patched(L, point=fake_point, _pred=fake_pred), checks() as c:
        out = L.box_study(1e4, 0.995, 0.179, root=str(tmp_path), log=lambda *a: None)
        Lx, nx = L.box(1e4)
        c.check("box_study: the nx factors at the rung-2 box, then the Lx factors at fixed dx",
                [(s["Lx"], s["nx"]) for s in seen]
                == [(Lx, int(round(nx*f))) for f in L.NX_FAC]
                + [L.box(1e4, f) for f in L.box_facs(1e4, 0.995, 0.179)])
        c.check("box_study: same (S, alpha, k), nu = 0, sigma from the coarse dense top",
                all(s["S"] == 1e4 and s["alpha"] == 0.995 and s["k"] == 0.179 and s["nu"] == 0.0
                    and s["pred"] == 0.0031 for s in seen))
        c.check("box_study keys", set(out) == {("nx", f) for f in L.NX_FAC}
                | {("Lx", f) for f in L.box_facs(1e4, 0.995, 0.179)})
        c.check("Lx x4 only for the first BOX_FAC4 box points",
                4.0 in L.box_facs(*L.BOX_POINTS[0]) and all(4.0 not in L.box_facs(*pt)
                                                            for pt in L.BOX_POINTS[L.BOX_FAC4:]))
        seen.clear()
        out = L.box_study(1e4, 0.995, 0.179, root=str(tmp_path), log=lambda *a: None,
                          compute=False)
        c.check("box_study read-back skips uncached points",
                all(L.box(1e4, f)[1] <= 4096 for kind, f in out if kind == "Lx"))

    calls = []

    def fake_kmax(S, alpha, k0, nu=0.0, root=None, log=print, compute=True, Lx=None, nx=None,
                  c0=None):
        calls.append((S, alpha, k0, nu))
        return dict(kmax=0.21, gmax=0.003, rec=dict(lam=0.003 + 0j))

    def fake_top(S, k, alpha, nx, Lx, nu=0.0, root=None, compute=True, ntop=8):
        calls.append(("dense", S, k, alpha, nx, Lx, nu))
        return dict(values=np.array([(0.003 - 0.004*nu*S) + 0j]))
    seen.clear()
    with _patched(L, kmax_branch=fake_kmax, dense_top=fake_top, point=fake_point,
                  PM_POINTS=(("low", 1e3, 0.995),), PM_LIST=(0.0, 0.01, 1.0)), checks() as c:
        out = L.pm_results(root=str(tmp_path), compute=True)
        Lx, nx = L.box(1e3)
        c.check("pm_results: k fixed at the nu = 0 k* of the branch",
                all(v["k"] == 0.21 for v in out.values()) and calls[0][:4] == (1e3, 0.995,
                                                                                 calls[0][2], 0.0))
        c.check("pm_results: nu = Pm/S at each Pm, dense on min(NX_DENSE, nx)",
                [c_[6] for c_ in calls if c_[0] == "dense"] == [0.0, 0.01/1e3, 1.0/1e3]
                and all(c_[4] == min(L.NX_DENSE, nx) for c_ in calls if c_[0] == "dense"))
        c.check("pm_results: a sparse solve only where the dense top grows",
                [s["nu"] for s in seen] == [0.0, 0.01/1e3]
                and out[("low", 1e3, 0.995, 1.0)]["rec"] is None)


def _two_hump(alpha):
    # analytic gamma(k; alpha) with two maxima in ln k: a low hump at k = 0.2 of height 0.01 and a
    # high hump at k = 0.7 of height 0.05 (1 - alpha^2)^0.6, widths 0.35 in ln k
    q = 1 - alpha*alpha

    def g(k):
        lk = np.log(k)
        return 0.01*np.exp(-((lk - np.log(0.2))/0.35)**2) \
            + 0.05*q**0.6*np.exp(-((lk - np.log(0.7))/0.35)**2)
    return g


def test_kmax_brent_and_gpoint_ladder():
    L = _run_script("tearing_lowk_run")
    calls = []
    g = _two_hump(0.99)

    def fake_gpoint(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True,
                    dense_first=False):
        calls.append((k, nx, Lx, pred))
        return dict(lam=g(k) + 0j)
    with _patched(L, gpoint=fake_gpoint, _pred=lambda *a, **kw: 0.004), checks() as c:
        res, end = L.kmax_brent(1e4, 0.99, 0.25, 1024, 43.0, log=lambda *a: None)
        kk = np.exp(np.linspace(np.log(0.1), np.log(0.4), 200001))
        kb = kk[np.argmax(g(kk))]
        c.check(f"interior maximum found: k* {res['kmax']:.6f} vs grid argmax {kb:.6f}",
                res is not None and abs(np.log(res["kmax"]/kb)) < 1e-3)
        c.check("gamma* is the function value there", abs(res["gmax"] - g(res["kmax"])) < 1e-15)
        c.check("every evaluation on the requested grid; sigma from the previous gamma",
                all(n == 1024 and Lx == 43.0 for _, n, Lx, _ in calls)
                and calls[0][3] == 0.004 and calls[1][3] == g(calls[0][0]))
        calls.clear()
        res, end = L.kmax_brent(1e4, 0.99, 0.42, 1024, 43.0, log=lambda *a: None)
        c.check(f"no interior maximum in [0.28, 0.63] (gamma rises to the high hump): None, end "
                f"k = {end['kmax']:.4f}", res is None and end["kmax"] > 0.6)
    # gpoint: the loose ladder, kind "lkg", key = configuration
    seq = {0: RuntimeError("refused"), 1: 0.02 + 0.01j, 2: 0.03 + 0j}
    seen = []

    def fake_cached(kind, root=None, compute=None, attempt=None, **cfg):
        seen.append((kind, attempt, cfg, compute()))
        out = seq.get(attempt, -1.0 + 0j)
        if isinstance(out, Exception):
            raise out
        return {"lam": out}

    def fake_gs(S, k, alpha, nx, Lx, sigma, nu=0.0, rtol_mult=100.0):
        return dict(sigma=sigma, rm=rtol_mult)
    with _patched(L.U, cached=fake_cached), _patched(L, gamma_solve=fake_gs), checks() as c:
        r = L.gpoint(1e4, 0.3, 0.999, 1024, 43.0, 0.01, log=lambda *a: None)
        c.check("refusal and complex pair skipped", complex(r["lam"]) == 0.03 and len(seen) == 3)
        c.check("kind lkg, key = configuration", all(
            kd == "lkg" and cfg == dict(S=1e4, k=0.3, alpha=0.999, nx=1024, Lx=43.0, nu=0.0)
            for kd, _, cfg, _ in seen))
        c.check("the loose ladder", [(round(o["sigma"]/0.01 - 1, 12), o["rm"]) for *_, o in seen]
                == [tuple(x) for x in L.LADDER_LOOSE[:3]])
    # every shift refused: the dense fallback (nx <= GPOINT_DENSE_NX), its top REAL growing mode
    seen.clear()

    class _R:
        values = np.array([0.05 + 0.02j, 0.03 + 0j, -0.1 + 0j])
        residuals = np.array([1e-14, 2e-14, 3e-14])

    def all_fail(kind, root=None, compute=None, attempt=None, **cfg):
        seen.append(attempt)
        if attempt == "dense":
            return compute()
        raise RuntimeError("refused")
    with _patched(L.U, cached=all_fail), _patched(L, dense_block=lambda *a: (None, None, _R())), \
            checks() as c:
        r = L.gpoint(1e4, 0.3, 0.999, 1024, 43.0, 0.01, log=lambda *a: None)
        c.check("dense fallback after the whole ladder: the top real growing eigenvalue",
                seen[-1] == "dense" and len(seen) == len(L.LADDER_LOOSE) + 1
                and complex(r["lam"]) == 0.03 and r["res"] == 2e-14)
        seen.clear()
        with pytest.raises(RuntimeError, match="no growing mode"):
            L.gpoint(1e4, 0.3, 0.999, 2*L.GPOINT_DENSE_NX, 43.0, 0.01, log=lambda *a: None)
        c.check("no dense fallback above GPOINT_DENSE_NX", "dense" not in seen)
        seen.clear()
        r = L.gpoint(1e5, 0.3, 0.9996, 4096, 60.0, 0.01, log=lambda *a: None, dense_first=True)
        c.check("dense_first: no shift-invert attempt, straight to the dense spectrum (any nx)",
                seen == ["dense"] and complex(r["lam"]) == 0.03)


def test_branch_gap_brent_and_alpha_c_brent_on_analytic_family():
    L = _run_script("tearing_lowk_run")

    def fake_gpoint(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True,
                    dense_first=False):
        return dict(lam=_two_hump(alpha)(k) + 0j)
    with _patched(L, gpoint=fake_gpoint, _pred=lambda *a, **kw: 0.01), checks() as c:
        # brute-force alpha_c of the analytic family: equal local maxima
        def maxima(alpha):
            kk = np.exp(np.linspace(np.log(0.05), np.log(2.0), 400001))
            g = _two_hump(alpha)(kk)
            i = [j for j in range(1, len(kk) - 1) if g[j] > g[j - 1] and g[j] > g[j + 1]]
            return [(kk[j], g[j]) for j in i]
        from scipy.optimize import brentq

        def gap(u):
            m = maxima(L.alpha_of_u(u))
            return np.log(m[0][1]/m[-1][1]) if len(m) == 2 else np.nan
        u_ref = brentq(gap, np.log(0.03), np.log(0.2), xtol=1e-10)
        g_, _, lo, hi = L.branch_gap_brent(1e4, L.alpha_of_u(u_ref + 0.1), 0.2, 0.7,
                                           log=lambda *a: None)
        c.check("both humps: finite gap, low below high",
                np.isfinite(g_) and lo["kmax"] < 0.3 < hi["kmax"])
        res = L.alpha_c_brent(1e4, np.log(0.003), np.log(0.5), 0.2, 0.7, tol=1e-4,
                              log=lambda *a: None)
        c.check(f"alpha_c_brent: u = {res['u']:.6f} vs brute force {u_ref:.6f}",
                abs(res["u"] - u_ref) < 5e-4)
        c.check("alpha_c_brent rows: both branches present at the last evaluation",
                res["rows"][-1]["lo"] is not None and res["rows"][-1]["hi"] is not None
                and abs(res["rows"][-1]["gap"]) < 0.05)
        c.check("alpha from u", abs(res["alpha"] - L.alpha_of_u(res["u"])) < 1e-15)
        g_, _, lo, hi = L.branch_gap_brent(1e4, L.alpha_of_u(u_ref), 0.12, 0.7,
                                           log=lambda *a: None)
        c.check("a seed whose window misses the maximum is re-centred once and finds it",
                lo is not None and abs(np.log(lo["kmax"]/0.2)) < 0.05)
        # a low window that walks onto the high hump (seed between the humps) is rejected
        g_, _, lo, hi = L.branch_gap_brent(1e4, 0.5, 0.45, 0.7, log=lambda *a: None)
        c.check("a branch found on the wrong side of the seeds' mean is missing",
                lo is None and g_ == -np.inf)
    with checks() as c:
        Lb, nb, db = L.boundary_nx(2e4)
        c.check("boundary grid: rung-2 box, nx/4 (>= 1024), sparse below BOUNDARY_DENSE_S",
                Lb == L.box(2e4)[0] and nb == max(1024, L.box(2e4)[1]//4) and not db
                and L.boundary_nx(1e3)[1] == 1024)
        c.check("boundary grid from BOUNDARY_DENSE_S: dense at nx = 1024 on the rung-2 box",
                L.boundary_nx(L.BOUNDARY_DENSE_S) == (L.box(L.BOUNDARY_DENSE_S)[0], 1024, True))
        ua, ub, kl, kh = L.boundary_seeds(1e4)
        c.check("boundary seeds: bracket from BOUNDARY_BRACKETS (low-k side first)",
                np.allclose((np.exp(ua), np.exp(ub)), L.BOUNDARY_BRACKETS[1e4], rtol=1e-14)
                and kl < kh)


def test_alpha_c_brent_continues_branch_seeds():
    L = _run_script("tearing_lowk_run")
    calls = []

    def fake_gap(S, alpha, k_low, k_high, root=None, log=print, compute=True):
        q = 1 - alpha*alpha
        lo = dict(kmax=0.1 + q, gmax=0.01)
        hi = dict(kmax=0.7 + q, gmax=0.05*q**0.6)
        calls.append(((k_low, k_high), (lo["kmax"], hi["kmax"])))
        return float(np.log(lo["gmax"]/hi["gmax"])), np.nan, lo, hi
    with _patched(L, branch_gap_brent=fake_gap), checks() as c:
        L.alpha_c_brent(1e4, np.log(0.01), np.log(0.2), 0.2, 0.7, tol=1e-3, log=lambda *a: None)
        c.check("first call uses the given seeds", calls[0][0] == (0.2, 0.7))
        c.check(f"each later call is seeded with the previous call's maxima ({len(calls)} calls)",
                len(calls) > 2 and all(calls[i + 1][0] == calls[i][1]
                                       for i in range(len(calls) - 1)))


def test_kmax_branch_seeds_from_cached_dense_tops():
    L = _run_script("tearing_lowk_run")
    got = {}

    def g(k):
        return 0.01 - 0.3*(k - 0.25)**2

    def fake_top(S, k, alpha, nx, Lx, nu=0.0, root=None, compute=True, ntop=8):
        got.setdefault("tops", []).append((k, nx, Lx, nu, compute))
        return dict(values=np.array([g(k) + 0j]))

    def fake_newton(solve, k0, g0, c0, **kw):
        got.update(k0=k0, g0=g0, c0=c0)
        return [dict(rec=dict(lam=g0 + 0j))], k0, g0, None
    with _patched(L, dense_top=fake_top), _patched(L.R, newton_kmax=fake_newton), checks() as c:
        res = L.kmax_branch(1e4, 0.995, 0.2, Lx=40.0, nx=512)
        k0 = 0.2
        c0 = (g(0.9*k0) - 2*g(k0) + g(1.1*k0))/(0.1*k0)**2
        c.check(f"curvature seed = quadratic through the dense tops at k0(1 +- 0.1) ({got['c0']:.6f}"
                f" vs {c0:.6f})", abs(got["c0"] - c0) < 1e-12 and got["g0"] == g(k0))
        c.check("dense tops on NX_DENSE, the solve's Lx, nu passed",
                [t[0] for t in got["tops"]] == [0.9*k0, k0, 1.1*k0]
                and all(t[1] == L.NX_DENSE and t[2] == 40.0 and t[3] == 0.0 for t in got["tops"]))
        c.check("result carries the box", res["Lx"] == 40.0 and res["nx"] == 512)
    got.clear()

    def convex(k):
        return 0.01 + 0.3*(k - 0.25)**2
    with _patched(L, dense_top=lambda S, k, *a, **kw: dict(values=np.array([convex(k) + 0j]))), \
            _patched(L.R, newton_kmax=fake_newton), checks() as c:
        L.kmax_branch(1e4, 0.995, 0.2, Lx=40.0, nx=512)
        c.check("a non-negative dense curvature is replaced by -gamma/k0^2",
                abs(got["c0"] + convex(0.2)/0.04) < 1e-15)


def test_boundary_check_ref_and_cutoff_drivers():
    L = _run_script("tearing_lowk_run")
    calls = []

    def fake_gpoint(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True,
                    dense_first=False):
        calls.append((S, k, alpha, nx, Lx, dense_first))
        return dict(lam=pred*(1 + 1e-9) + 0j)
    res2 = dict(S=2e4, rows=[dict(alpha=0.995, lo=dict(kmax=0.13, gmax=1.7e-3),
                                  hi=dict(kmax=0.71, gmax=1.7e-3))])
    with _patched(L, gpoint=fake_gpoint), checks() as c:
        L.boundary_check(res2)
        c.check("boundary_check (sparse route): the full rung-2 nx",
                [x[3:] for x in calls] == [(L.box(2e4)[1], L.box(2e4)[0], False)]*2)
    calls.clear()
    res = dict(S=1e5, rows=[dict(alpha=0.9984, lo=dict(kmax=0.11, gmax=0.9e-3),
                                 hi=dict(kmax=0.72, gmax=1.3e-3)),
                            dict(alpha=0.999, lo=dict(kmax=0.12, gmax=1e-3), hi=None),
                            dict(alpha=0.9985, lo=dict(kmax=0.13, gmax=1.1e-3),
                                 hi=dict(kmax=0.7, gmax=1.0e-3)),
                            dict(alpha=0.998, lo=None, hi=dict(kmax=0.71, gmax=1.2e-3))])
    with _patched(L, gpoint=fake_gpoint), checks() as c:
        out = L.boundary_check(res)
        c.check("boundary_check: the LAST row with both branches; dense route: dense at 2 nx_b",
                calls == [(1e5, 0.13, 0.9985, 2048, L.box(1e5)[0], True),
                          (1e5, 0.7, 0.9985, 2048, L.box(1e5)[0], True)])
        c.check("boundary_check returns (boundary-grid, check-grid) gamma pairs",
                out["lo"][0] == 1.1e-3 and abs(out["hi"][1]/1e-3 - 1 - 1e-9) < 1e-15
                and out["nx_b"] == 1024 and out["nx"] == 2048)
    seen = []

    def fake_top(S, k, alpha, nx, Lx, nu=0.0, root=None, compute=True, ntop=8):
        seen.append((S, k, alpha, nx, Lx, nu))
        return dict(values=np.array([0.01 + 0j]))
    with _patched(L, dense_top=fake_top), checks() as c:
        L.ref_results()
        c.check("ref_results: the Julia reference's (S, ka, 0.95) points on the rung-2 box",
                [(s_[0], s_[1], s_[2]) for s_ in seen] == list(L.REF_POINTS)
                and all(s_[4] == L.box(s_[0])[0] and s_[3] == min(L.NX_DENSE, L.box(s_[0])[1])
                        for s_ in seen))
        ref = np.load(_REF)
        c.check("every REF_POINTS entry is a point of the Julia reference npz (whichf = 2)",
                all(any(int(ref["whichf"][i]) == 2 and ref["S"][i] == S_ and ref["ka"][i] == ka
                        and ref["alpha"][i] == al for i in range(len(ref["S"])))
                    for (S_, ka, al) in L.REF_POINTS))
        seen.clear()
        out = L.pmscan_results()
        c.check("pmscan: nu = Pm/S passed to every dense top, the scan grid and box",
                sorted({(s_[0], s_[2], s_[5]) for s_ in seen})
                == sorted({(S_, al, pm/S_) for (S_, al) in L.PM_SCAN for pm in L.PM_SCAN_LIST})
                and {s_[1] for s_ in seen} == set(L.KA_SCAN)
                and set(out) == {(S_, al, pm) for (S_, al) in L.PM_SCAN for pm in L.PM_SCAN_LIST})
        seen.clear()
        L.cutoff_results()
        c.check("cutoff_results: every (alpha, ka) at CUTOFF_S, nu = 0",
                sorted((s_[2], s_[1]) for s_ in seen)
                == sorted((a, k) for a in L.CUTOFF_ALPHA for k in L.CUTOFF_KA)
                and all(s_[0] == L.CUTOFF_S and s_[5] == 0.0 for s_ in seen))


@pytest.mark.fp64
def test_gamma_solve_matches_dense():
    L = _run_script("tearing_lowk_run")
    S, k, al = S0, K0, 1.0
    lam = _lead(S, k, al, NX, LX, (0.0, 1.0/S))
    r = L.gamma_solve(S, k, al, NX, LX, lam.real*1.03)
    with checks() as c:
        c.check(f"gamma_solve lambda == eig_dense ({complex(r['lam']):.12f})",
                abs(complex(r["lam"]) - lam) < 1e-10*abs(lam))
        c.check("residual gated", r["res"] < 1e-9)


# ============================================================ (vi) kmax_branch end to end

@pytest.mark.fp64
def test_kmax_branch_matches_brute_force_maximum(tmp_path):
    L = _run_script("tearing_lowk_run")
    S, al = S0, A0

    def f(lk):
        return -_lead(S, np.exp(lk), al, NX, LX, (0.0, 1.0/S)).real
    r = minimize_scalar(f, bounds=(np.log(0.2), np.log(0.6)), method="bounded",
                        options=dict(xatol=1e-9))
    kb, gb = float(np.exp(r.x)), float(-r.fun)
    with _patched(L, NX_DENSE=NX):
        res = L.kmax_branch(S, al, 0.3, root=str(tmp_path), log=print, Lx=LX, nx=NX)
    print(f"  kmax_branch k* = {res['kmax']:.10f} gamma* = {res['gmax']:.12e} "
          f"({res['iterations']} its); brute force {kb:.10f} {gb:.12e}")
    with checks() as c:
        c.check(f"k* == brute-force argmax to 2e-6 ({abs(res['kmax']/kb - 1):.1e})",
                abs(res["kmax"]/kb - 1) < 2e-6)
        c.check(f"gamma* == brute-force max to 1e-10 ({abs(res['gmax']/gb - 1):.1e})",
                abs(res["gmax"]/gb - 1) < 1e-10)
        c.check("the solve sits on the requested box", int(res["rec"]["nx"]) == NX
                and float(res["rec"]["Lx"]) == LX and res["nx"] == NX and res["Lx"] == LX)
        c.check("the curvature seed is negative (a maximum)", res["c0"] < 0)


# ============================================================ review gates (second reviewer)
# Each gate below kills a one-line mutation of tearing_lowk_run.py that the gates above let
# through (the review's mutation table): the budget terms one by one, the structure numbers on
# vectors where |phi| != |psi|, the seeds and grids of the cached campaign's drivers, and the
# alpha_c box check.

@pytest.mark.fp64
def test_budget_terms_individually_against_operator_split():
    # (i) checks the SUM R + M + C - D; this checks each term. J is affine in alpha (x0 =
    # (alpha Psi0, Psi0), N' linear in x0), so with J1 = J(alpha=1) - J(alpha=0):
    #   R = kinetic-energy rate of alpha J1, M = magnetic-energy rate of alpha J1 (the U.grad
    #   advection terms vanish on the y-average), D_eta = -magnetic rate of the eta diagonal,
    #   C = total rate of J(0) + D_eta + D_nu (the cross terms u.(B.grad b) + b.(B.grad u) live in
    #   J(0) and cancel only in the sum).
    L = _run_script("tearing_lowk_run")
    rng = np.random.default_rng(11)
    with checks() as c:
        for (alpha, nu) in ((A0, 0.0), (0.6, 4e-4), (1.0, 0.0)):
            eta = 1.0/S0
            p, B, idx = _block(S0, K0, alpha, NX, LX, (nu, eta))
            _, B0, _ = _block(S0, K0, 0.0, NX, LX, (nu, eta))
            _, B1, _ = _block(S0, K0, 1.0, NX, LX, (nu, eta))
            c.check(f"alpha={alpha}: J affine in alpha",
                    np.abs(B - B0 - alpha*(B1 - B0)).max() < 1e-12*np.abs(B).max())
            r = stability.eig_dense(B)
            v = r.right[:, :3] @ (rng.normal(size=3) + 1j*rng.normal(size=3))
            w = _weights(p, idx, K0)
            kx = 2*np.pi*np.fft.fftfreq(NX, d=LX/NX)
            k2 = kx[idx[:, 2]]**2 + K0**2
            fphi, fpsi = idx[:, 0] == 0, idx[:, 0] == 1
            E_W = float(np.sum(w*np.abs(v)**2))

            def rate(A, mask):
                return float(2*np.real(np.sum((w*mask)*np.conj(v)*(A @ v))))/E_W
            R_op = rate(alpha*(B1 - B0), fphi)
            M_op = rate(alpha*(B1 - B0), fpsi)
            Deta_op = -rate(np.diag(np.where(fpsi, -eta*k2, 0.0)), fpsi)
            Dnu_op = -rate(np.diag(np.where(fphi, -nu*k2, 0.0)), fphi)
            C_op = rate(B0, fphi | fpsi) + Deta_op + Dnu_op
            _, _, tot = L.budget(p, idx, v, K0, alpha, nu, eta)
            for key, ref in (("R", R_op), ("M", M_op), ("C", C_op), ("Deta", Deta_op),
                             ("Dnu", Dnu_op)):
                got = tot[key]/tot["E"]
                c.check(f"alpha={alpha}, nu={nu}: {key}/E budget {got:+.10e} == operator "
                        f"{ref:+.10e}", abs(got - ref) < 1e-9*max(abs(R_op), abs(C_op)) + 1e-13)


@pytest.mark.fp64
def test_structure_numbers_where_phi_and_psi_differ():
    # (iv) uses phi = +-psi, where every |phi|/|psi| choice is the same number; here phi has its
    # own profile, its own symmetry and its own size, and psi(0) is small (the alpha = 1 mode
    # has psi0 ~ 0.02) with a complex phase on input
    L = _run_script("tearing_lowk_run")
    p, _, idx = _block(S0, K0, 0.0, NX, LX, (0.0, 1.0/S0))
    x = np.arange(NX)*LX/NX - LX/2
    psi = (0.02 + x*x)*np.exp(-x*x/2) + 0.1j*x*np.exp(-x*x)        # Re even, Im odd, psi(0) small
    phi = 3.0*np.exp(-(x - 0.4)**2) + 0.5j*np.exp(-x*x)             # NOT symmetric
    ph = np.exp(-1.1j)
    v = _pack(p, idx, phi*ph, psi*ph)
    s = L.structure(p, idx, v, 0.01, K0, 0.0, 0.0, 1.0/S0)
    _, f_phi, f_psi = L.fields_of(p, idx, v)
    # energy density of the hand profile (y-averaged): (|u|^2 + |b|^2)/4 per field, independent
    kx = 2*np.pi*np.fft.fftfreq(NX, d=LX/NX)
    d = lambda f: np.fft.ifft(1j*kx*np.fft.fft(f))
    E = (K0**2*np.abs(phi)**2 + np.abs(d(phi))**2 + K0**2*np.abs(psi)**2 + np.abs(d(psi))**2)
    with checks() as c:
        c.check(f"psi0 = |psi(0)|/max|psi| = {abs(psi[NX//2])/np.abs(psi).max():.6f} (not over "
                f"max|phi|)", abs(s["psi0"] - abs(psi[NX//2])/np.abs(psi).max()) < 1e-10)
        c.check(f"phi_over_psi = max|phi|/max|psi| = {np.abs(phi).max()/np.abs(psi).max():.6f}",
                abs(s["phi_over_psi"] - np.abs(phi).max()/np.abs(psi).max()) < 1e-10)
        c.check("x_rms_E = sqrt(<x^2>_E) of the hand energy density",
                abs(s["x_rms_E"] - np.sqrt(np.sum(x*x*E)/np.sum(E))) < 1e-10)
        c.check(f"sym_residual sees the non-symmetric phi ({s['sym_residual']:.3f})",
                s["sym_residual"] > 0.1)
        c.check("small psi(0) (0.02 of max) still fixes the phase at the sheet: psi(0) real > 0",
                abs(f_psi[NX//2].imag) < 1e-12 and f_psi[NX//2].real > 0
                and np.abs(f_psi - psi/np.abs(phi).max()).max() < 1e-10)


def test_campaign_seeds_and_driver_grids():
    # the read-back replays the CACHED campaign, whose cache keys hold every k a Newton/Brent
    # iterate visited: a changed seed or continuation rule silently empties the notebook. The
    # seeds, continuation rules and grids of the drivers are pinned here against their formulas
    L = _run_script("tearing_lowk_run")
    R = L.R
    calls = []

    def fake_kmax(S, alpha, k0, nu=0.0, root=None, log=print, compute=True, Lx=None, nx=None,
                  c0=None):
        calls.append(("kmax", S, alpha, k0))
        return dict(kmax=0.9*k0, gmax=0.003, rec=dict(lam=0.003 + 0j, S=S, dlam_eta=0.0 + 0j,
                                                       psi0=0.1, x_rms_E=0.7, b_R=0.0, b_M=0.0,
                                                       b_C=0.5))

    def fake_point(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True):
        calls.append(("point", S, k, alpha, nx, Lx, pred))
        return dict(lam=0.004 + 0j, dlam_eta=0.0 + 0j, S=S)
    with _patched(L, kmax_branch=fake_kmax, point=fake_point, _pred=lambda *a, **kw: 0.0041,
                  S_A1=(1e3, 3e3, 1e4), ALPHA_GS=(1.0,), S_GS=(1e3, 3e3, 1e4)), checks() as c:
        out = L.alpha1_results(compute=True)
        ks = [x[3] for x in calls if x[0] == "kmax"]
        c.check("alpha1: k_max chain from k0 = 0.4, continued as k* 3^-0.3",
                ks[0] == 0.4 and all(abs(ks[i + 1] - 0.9*ks[i]*3.0**-0.3) < 1e-15
                                     for i in range(len(ks) - 1)))
        pts = [x for x in calls if x[0] == "point"]
        main = [x for x in pts if x[6] == 0.0041]
        chk = [x for x in pts if x[6] != 0.0041]
        c.check("alpha1 gamma(S): KA_GS on gs_box", all(x[2] == L.KA_GS and (x[5], x[4])
                                                        == L.gs_box(x[1]) for x in main))
        c.check("alpha1 grid check: the last two S at 2 nx of gs_box, sigma = the nx gamma",
                [x[1] for x in chk] == [3e3, 1e4] and all(x[4] == 2*L.gs_box(x[1])[1]
                                                          and x[6] == 0.004 for x in chk)
                and set(out["gs_check"]) == {(1.0, 3e3), (1.0, 1e4)})
    calls.clear()
    with _patched(L, kmax_branch=fake_kmax), checks() as c:
        for which, S, al in (("low", 1e4, 0.995), ("high", 2e4, 0.99), ("alpha1", 1e3, 1.0)):
            calls.clear()
            L.pm_k(which, S, al, None, None, True)
            want = (R.theory_kmax(S)[0]*(1 - al*al)**(-1/7) if which == "high"
                    else 0.2*(1e4/S)**0.3)
            c.check(f"pm_k {which}: seed {calls[0][3]:.6f} == {want:.6f}",
                    abs(calls[0][3] - want) < 1e-14*want)
    calls.clear()
    with _patched(L, kmax_branch=lambda S, al, k0, **kw: dict(
            kmax=0.2 if k0 < 0.4 else (0.3 if al == 0.995 else 0.7), gmax=0.003,
            rec=dict(lam=0.003 + 0j, psi0=0.1, x_rms_E=0.7, b_R=0.0, b_M=0.0, b_C=0.5)),
            STRUCT_POINTS=((1e4, 0.99), (1e4, 0.995))), checks() as c:
        out = L.struct_results(compute=True)
        c.check("struct_results: a 'high' Newton that lands below STRUCT_KSPLIT is dropped, both "
                "branches kept where each lands on its own side",
                set(out) == {(1e4, 0.99, "low"), (1e4, 0.99, "high"), (1e4, 0.995, "low")})
    with checks() as c:
        for S in (1e3, 1e4, 1e5):
            ua, ub, kl, kh = L.boundary_seeds(S)
            c.check(f"boundary_seeds(S={S:.0e}): k_low = 0.35 (1e3/S)^0.3, k_high = 0.85 k_th "
                    f"(geometric mean of the bracket)^(-1/7)",
                    abs(kl - 0.35*(1e3/S)**0.3) < 1e-15 and abs(
                        kh - 0.85*R.theory_kmax(S)[0]*np.exp(0.5*(ua + ub))**(-1/7)) < 1e-14)
    seen = []

    def fake_top(S, k, alpha, nx, Lx, nu=0.0, root=None, compute=True, ntop=8):
        seen.append((S, k, alpha, nx, Lx, nu))
        return dict(values=np.array([0.01 + 0j]))
    with _patched(L, dense_top=fake_top), checks() as c:
        L.cutoff_results()
        Lc, nc = L.box(L.CUTOFF_S)
        c.check("cutoff_results: the rung-2 box of CUTOFF_S, nx = min(NX_DENSE, its nx)",
                all(s_[4] == Lc and s_[3] == min(L.NX_DENSE, nc) for s_ in seen))
    seen.clear()
    with _patched(L, dense_top=fake_top, kmax_branch=fake_kmax, point=fake_point,
                  PM_POINTS=(("low", 1e4, 0.995),), PM_LIST=(0.0, 1.0)), checks() as c:
        L.pm_results(compute=True)
        c.check("pm_results: dense tops on the rung-2 box (Lx and nx)",
                all(s_[4] == L.box(1e4)[0] and s_[3] == min(L.NX_DENSE, L.box(1e4)[1])
                    for s_ in seen))


def test_pred_julia_table_and_small_helpers(tmp_path):
    L = _run_script("tearing_lowk_run")
    with checks() as c:
        tops = {0.3: 0.02 + 0j, 0.5: 0.02 + 0.01j, 0.7: -0.01 + 0j}
        with _patched(L, _dense_others=lambda S, k, a, Lx, nx, nu: (None, tops[k], 0)):
            c.check("_pred: the coarse grid's top when it is a real growing mode",
                    L._pred(1e3, 0.3, 0.99, 20.0) == 0.02)
            c.check("_pred: a complex or decaying top falls back",
                    L._pred(1e3, 0.5, 0.99, 20.0, fallback=7e-3) == 7e-3
                    and L._pred(1e3, 0.7, 0.99, 20.0, fallback=7e-3) == 7e-3)
            with pytest.raises(RuntimeError, match="no growing mode"):
                L._pred(1e3, 0.7, 0.99, 20.0)
            c.check("_pred(compute=False): the fallback, else 1",
                    L._pred(1e3, 0.3, 0.99, 20.0, compute=False, fallback=5e-3) == 5e-3
                    and L._pred(1e3, 0.3, 0.99, 20.0, compute=False) == 1.0)
        c.check("top_real: a complex pair at |Im|/|lambda| = 7e-3 is not real",
                L.top_real([0.3 + 0.002j, 0.1 + 0j]) == 0.1)
        kas = (0.1, 0.2, 0.3, 0.4, 0.5)
        rows = {ka: dict(values=np.array([g + 0j])) for ka, g in zip(kas, (1.0, 3.0, 1.0, 0.5, 0.8))}
        c.check("scan_maxima: an end point above its neighbour but below the global max is not "
                "a maximum", L.scan_maxima(rows) == [(0.2, 3.0)])
        jobs = L.julia_jobs(1e4, 0.995, 0.179, 3.4e-3, xlims=(10.0, 20.0), levels=(1.0, 0.5))
        have = {L.H._jl_key(j): (complex(j[5] + j[4]), 0) for j in jobs}
        with _patched(L.H, julia_study=lambda jobs_, path, log: have), \
                _patched(L, julia_jobs=lambda *a, **kw: jobs):
            t = L.julia_table(1e4, 0.995, 0.179, 3.4e-3, root=str(tmp_path), compute=False)
        c.check("julia_table: {xlim: [gamma per level]} in level order",
                sorted(t) == [10.0, 20.0] and all(
                    [z.real for z in t[xl]] == [xl + j[4] for j in jobs if j[5] == xl] for xl in t))


def test_branch_classification_thresholds_and_newton_seed_continuation():
    L = _run_script("tearing_lowk_run")
    q = lambda *a, **kw: None
    with checks() as c:
        # seeds 0.2/0.7: geometric mean 0.374, arithmetic 0.45. The low Newton lands at 0.42
        # (above the geometric mean), the high one at 0.5: 0.42 is within (1 - sep) of 0.5, so the
        # low branch is the high maximum seen twice (missing), not a merger
        t = _fake_branch(lambda a, k0: (0.42, 3e-3, 0.0) if k0 < 0.4 else (0.5, 2e-3, 0.0))
        with _patched(L, kmax_branch=t):
            g, *_ = L.branch_gap(1e4, 0.99, 0.2, 0.7, log=q)
            c.check("branch_gap: the side test is against the GEOMETRIC mean of the seeds",
                    g == -np.inf)
        # two distinct maxima at 0.2 and 0.38 (ratio 0.53 < sep = 0.75): a finite gap
        t = _fake_branch(lambda a, k0: (0.2, 3e-3, 0.0) if k0 < 0.4 else (0.38, 2e-3, 0.0))
        with _patched(L, kmax_branch=t):
            g, *_ = L.branch_gap(1e4, 0.99, 0.2, 0.7, log=q)
            c.check("branch_gap: maxima at k ratio 0.53 are two branches (sep = 0.75)",
                    np.isfinite(g))
        res = {0.2: dict(kmax=0.33, gmax=3e-3), 0.7: dict(kmax=0.42, gmax=2e-3)}
        with _patched(L, kmax_brent=lambda S, al, k0, nx, Lx, **kw: (res[k0], res[k0])):
            with pytest.raises(RuntimeError, match="merged"):
                L.branch_gap_brent(1e4, 0.99, 0.2, 0.7, log=q)
            c.check("branch_gap_brent: maxima at k ratio 0.79 (> 0.75) are one branch (merged)",
                    True)
    calls = []

    def fake_gap(S, alpha, k_low, k_high, root=None, log=print, compute=True):
        u = np.log(1 - alpha*alpha)
        lo = dict(kmax=0.2 + 0.01*u, gmax=0.01)
        hi = dict(kmax=0.7 + 0.01*u, gmax=0.05*(1 - alpha*alpha)**0.6)
        calls.append(((k_low, k_high), (lo["kmax"], hi["kmax"])))
        return float(np.log(lo["gmax"]/hi["gmax"])), -0.6, lo, hi
    with _patched(L, branch_gap=fake_gap), checks() as c:
        L.alpha_c(1e4, np.log(0.01), np.log(0.2), 0.2, 0.7, tol=1e-3, log=q)
        c.check(f"alpha_c (Newton route): each call seeded with the previous call's maxima "
                f"({len(calls)} calls)", len(calls) > 2 and calls[0][0] == (0.2, 0.7)
                and all(calls[i + 1][0] == calls[i][1] for i in range(len(calls) - 1)))


@pytest.mark.fp64
def test_lk_solve_gap_width_and_gamma_solve_residual():
    L = _run_script("tearing_lowk_run")
    T = L.T
    S, k, al = S0, K0, A0
    eta = 1.0/S
    pd = jr.Parameters(nx=NX, ny=8, Lx=LX, Ly=2*np.pi/k, dims=2, cfl_safety=0.5, forcing=False,
                       eqpars={"diss": (0.0, eta), "hyper": 1})
    Bd, idxd = stability.ky_block_matrix(_x0(pd, al), jr.setup_kgrids(pd), pd, 1)
    rd = stability.eig_dense(Bd)
    lam = rd.values[0]
    with _patched(L, NX_DENSE=NX):
        rec = L.lk_solve(S, k, al, NX, LX, lam.real*1.03, lam_hint=lam.real)
    gap = float(np.min(np.abs(rd.values[1:] - lam)))
    x, psi, phi, d2 = T._eigfun(pd, idxd, rd.right[:, 0], NX//2)
    prof = T._profile(x, psi, phi, d2, NX//2, LX, half=1.5)
    width = T.width_quarter(prof["xn"], prof["d2n"])
    r = L.gamma_solve(S, k, al, NX, LX, lam.real*1.03)
    with checks() as c:
        c.check(f"lk_solve gap {float(rec['gap']):.6e} == distance to the nearest other dense "
                f"eigenvalue {gap:.6e}", abs(float(rec["gap"]) - gap) < 1e-6*gap)
        c.check(f"lk_solve width {float(rec['width']):.6f} == quarter-max width of psi'' of the "
                f"dense eigenvector {width:.6f}", abs(float(rec["width"]) - width) < 1e-6*width)
        c.check(f"gamma_solve records its residual ({float(r['res']):.1e}): nonzero and gated",
                0.0 < float(r["res"]) < 1e-9)


def test_boundary_box_check():
    # the alpha_c box check (added in review): both maxima of the last both-branch row, dense, on
    # the rung-2 box at NX_DENSE and on Lx x2 at the same dx; the root moved by the envelope
    # theorem with the secant slope of the two both-branch rows closest to the root
    L = _run_script("tearing_lowk_run")
    calls = []

    def fake_gpoint(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True,
                    dense_first=False):
        calls.append((S, k, alpha, nx, Lx, dense_first))
        big = nx > L.NX_DENSE
        g = (1.0e-3*(1 - 0.01) if big else 1.0e-3) if k < 0.4 else \
            (1.1e-3*(1 + 0.005) if big else 1.1e-3)
        return dict(lam=g + 0j)
    res = dict(S=5e4, u=-5.3, rows=[
        dict(u=-5.0, alpha=L.alpha_of_u(-5.0), gap=-0.3, lo=dict(kmax=0.1, gmax=1e-3),
             hi=dict(kmax=0.7, gmax=1.3e-3)),
        dict(u=-5.6, alpha=L.alpha_of_u(-5.6), gap=np.inf, lo=dict(kmax=0.09, gmax=1e-3), hi=None),
        dict(u=-5.25, alpha=L.alpha_of_u(-5.25), gap=-0.05, lo=dict(kmax=0.095, gmax=1e-3),
             hi=dict(kmax=0.72, gmax=1.05e-3)),
        dict(u=-5.32, alpha=L.alpha_of_u(-5.32), gap=0.02, lo=dict(kmax=0.093, gmax=1e-3),
             hi=dict(kmax=0.722, gmax=0.98e-3))])
    with _patched(L, gpoint=fake_gpoint), checks() as c:
        out = L.boundary_box_check(res)
        Lx = L.box(5e4)[0]
        c.check("the last both-branch row's (k*, alpha), dense, NX_DENSE on the rung-2 box then "
                "2 NX_DENSE on Lx x2", calls == [
                    (5e4, 0.093, L.alpha_of_u(-5.32), L.NX_DENSE, Lx, True),
                    (5e4, 0.093, L.alpha_of_u(-5.32), 2*L.NX_DENSE, float(np.round(2*Lx, 6)), True),
                    (5e4, 0.722, L.alpha_of_u(-5.32), L.NX_DENSE, Lx, True),
                    (5e4, 0.722, L.alpha_of_u(-5.32), 2*L.NX_DENSE, float(np.round(2*Lx, 6)), True)])
        slope = (0.02 - (-0.05))/(-5.32 - (-5.25))
        dgap = np.log(0.99) - np.log(1.005)
        c.check(f"slope = secant of the two rows nearest the root ({out['slope']:.6f})",
                abs(out["slope"] - slope) < 1e-12)
        c.check(f"dgap = ln(0.99) - ln(1.005) for box changes -1% (lo), +0.5% (hi) "
                f"({out['dgap']:.6e})",
                abs(out["dgap"] - dgap) < 1e-14)
        c.check(f"u' = u_c - dgap/slope ({out['u']:.8f})",
                abs(out["u"] - (-5.3 - dgap/slope)) < 1e-12
                and abs(out["alpha_c"] - L.alpha_of_u(out["u"])) < 1e-15)
        c.check("a box error that lowers the low branch moves 1 - alpha_c^2 DOWN",
                out["u"] < res["u"])


def test_bracketed_root_precision_and_cost_and_brent_edge():
    L = _run_script("tearing_lowk_run")
    with checks() as c:
        # an inexact slope (2x the true one: Newton converges linearly, halving the error): the
        # answer must still be within tol (the early return is at a step below tol/4)
        for br in ((-1.0, 2.0), (-3.0, 1.0), (-1.0, 5.0)):
            u, its = L.bracketed_root(lambda u: (u - 0.3, 2.0), *br, tol=1e-3)
            c.check(f"inexact slope, bracket {br}: |u - root| = {abs(u - 0.3):.1e} <= tol",
                    abs(u - 0.3) <= 1e-3)
        # f = cbrt(u - 0.3): Newton overshoots (u -> -2u about the root) INSIDE the bracket; the
        # half-bracket trust region turns those steps into bisections. Every evaluation is a whole
        # branch_gap (two maxima) in the campaign, so the count is gated
        f = lambda u: (np.cbrt(u - 0.3), 1/(3*np.cbrt(u - 0.3)**2))
        for br in ((-1.0, 2.0), (-3.0, 1.0), (-1.0, 5.0)):
            u, its = L.bracketed_root(f, *br, tol=1e-3)
            c.check(f"cbrt root, bracket {br}: within tol in {len(its)} <= 13 evaluations",
                    abs(u - 0.3) <= 1e-3 and len(its) <= 13)
    # kmax_brent on a monotone but SATURATING gamma(k): bounded Brent ends 0.56 xatol inside the
    # window there, so the interior test needs its 2 xatol margin to report "no maximum"
    calls = []

    def fake_gpoint(S, k, alpha, nx, Lx, pred, nu=0.0, root=None, log=print, compute=True,
                    dense_first=False):
        calls.append(k)
        return dict(lam=0.01 + 0.001*np.tanh(10*np.log(k/0.25)) + 0j)
    with _patched(L, gpoint=fake_gpoint, _pred=lambda *a, **kw: 0.01), checks() as c:
        res, end = L.kmax_brent(1e4, 0.99, 0.25, 1024, 43.0, log=lambda *a: None)
        c.check(f"monotone saturating gamma(k): no interior maximum (end k = {end['kmax']:.6f})",
                res is None and end["kmax"] > 0.37)
    with checks() as c:
        c.check("gpoint's dense fallback stops at nx = 2048 (a 4096 x 4096 dense block; the "
                "cached boundary checks call gpoint up to nx = 8192)", L.GPOINT_DENSE_NX == 2048)


if __name__ == "__main__":
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
