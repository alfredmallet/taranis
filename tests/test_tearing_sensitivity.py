# plans/AUTODIFF_PLAN.md rung 1b (science) gate: eigenvalue SENSITIVITIES of resistive tearing
# through the eigen-harness (taranis/stability.py: dj_matrix, eig_dense's left vectors,
# eigenvalue_sensitivity), on small dense ky blocks. The science lives in
# examples/tearing-sensitivity.ipynb (data: examples/tearing_sensitivity_run.py); this file is its
# fast standing gate. (i)-(ii) build their own blocks; (iii)-(iv) IMPORT the data script (examples/
# is put on sys.path; it is not a package) and gate the notebook's own helpers and its one-point
# solver. Setup as rung 1b-ref: dims=2, nu = 0, hyper = 1, S = 1/eta, Psi0 = sech^2 (+ two
# periodic images), Phi0 = alpha Psi0, ky block, fp64.
#   (i)  running exponent p = d ln gamma/d ln(1 - alpha^2) at S = 1e3, ka = 1, alpha = 0.8
#        (nx = 512, Lx = 20): the sensitivity along dx0 = (Psi0, 0) equals the central difference
#        of eig_dense eigenvalues at O(h^2); p is within 2e-3 of the notebook's grid-converged
#        0.52862 (nx = 2048; observed 3e-6 at nx = 512); the eigenvalue is non-normal
#        (kappa = 127.81 on this grid, 242 grid-converged) and the right-vector-only formula
#        v^H dJ v/v^H v is wrong by >20 %. kappa is pinned two-sided: equal to ||w|| ||v||/|w^H v|
#        recomputed here from eig_dense's vectors (1e-10 relative: the same numbers, so only
#        round-off), and to 127.81 within 1 % -- a FIXED-GRID regression value (kappa is not
#        grid-converged: 242 at nx >= 2048), far above round-off drift (kappa eps kappa ~ 1e-12)
#        and far below any structural error (x2, sqrt, 1/kappa, a missing norm).
#   (ii) marginal point by Newton in the sheet width a at FIXED k = sqrt 5 and eta (family
#        Psi0 = a sech^2(x/a): B_y amplitude fixed, so Delta'a depends on ka only and vanishes
#        at ka = sqrt 5). nx = 256, Lx = 20, S = 1e3, from ka = 2. Plain Newton converges only
#        LINEARLY: on this fixed grid gamma ~ (a* - a)^2 near the root (a double root: the
#        estimate p = 1/slope(gamma/gamma') -> 2.01), so each step contracts the error by
#        ~1 - 1/p ~ 1/2 (observed 0.50-0.51 on the last steps). The multiplicity-corrected
#        estimate a - p gamma/gamma' lands on a = 1, i.e. ka = sqrt 5, to 2e-6 (observed
#        |ka* - sqrt 5| = 4e-6 after 7 iterates); pinned at 5e-5. The width tangent is FD-gated.
#        (The grid-CONVERGED problem has p = 4 inside the finite-S suppression band -- the
#        notebook; the fixed small grid is what makes this a fast gate.)
#   (iii) the notebook's pure helpers on synthetic inputs with known answers (numpy only, both
#        precisions): `exponent` on gamma = C (1 - alpha^2)^p returns p exactly; `ridge_fit` on
#        gamma = G(alpha) - c (ln k - ln k_max(alpha))^2 (exactly quadratic in ln k) returns
#        k_max, G, dG/dalpha (envelope theorem), p = d ln G/d ln(1 - alpha^2) and
#        pk = d ln k_max/d ln(1 - alpha^2); `newton_table` on plain-Newton iterates of
#        gamma = C (a* - a)^m (m = 2, 4) measures p = m and lands the multiplicity-corrected root
#        on a* exactly; `fd_gate`, driven through its own control flow with alpha_family / cached /
#        eig_only replaced by an analytic lambda(alpha), returns the central difference
#        (lambda(alpha + h) - lambda(alpha - h))/2h at the notebook's h sets (FD_H, and FD_H_NEAR1
#        once 1 - alpha < 0.05), whose error against the exact derivative is O(h^2).
#   (iv) the notebook's matrix-free one-point solver `sens_solve` (ky_block_operator +
#        band-LU-preconditioned shift_invert, stability.left_eigenvector on J^H, stability.
#        dj_operator, eigenvalue_sensitivity) at (i)'s point reproduces the DENSE path's lambda,
#        dlambda/dalpha, kappa and naive value (observed 1e-13 relative; pinned 1e-8, the pair's
#        residual gate 1e-12 times kappa), and `exponent` of its record equals (i)'s p. Directly:
#        dj_operator's action equals dj_matrix @ v, and left_eigenvector's w (matrix-free, adjoint
#        LU preconditioner) is parallel to eig_dense's left vector.
# CONJUGATION is NOT gated here: the tearing eigenvalue is real and the 1D ky block is real up to a
# diagonal phase similarity, so w^T for w^H, conj(dlambda), J^T left vectors all give the same
# numbers (the reviewer's mutations survive this file, as expected). Conjugation correctness is
# gated by tests/test_stability_sensitivity.py, on CMHD's complex (Doppler/Alfven/fast/slow) waves.
# pytest: `pytest tests/test_tearing_sensitivity.py`; script: `python tests/test_tearing_sensitivity.py`.
from _rmhd_testing import bootstrap, checks, fit_order

bootstrap()

import importlib
import os
import sys

import numpy as np
import pytest

import jax.numpy as jnp

import taranis as jr
from taranis import stability

_SQRT5 = np.sqrt(5.0)
_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
_KAPPA_512 = 127.808          # kappa at (i)'s point on its nx = 512 grid (fixed-grid regression)


def _run_script(name):
    # the notebook's data script as a module (examples/ is not a package; the script itself
    # imports tearing_eigen_run from the same directory)
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    return importlib.import_module(name)


def _sech2(u):
    e = np.exp(-2.0*np.abs(u))
    return 4.0*e/(1.0 + e)**2


def _setup(S, k, nx, Lx, alpha, a=1.0):
    # the ky = k block about psi = a sech^2((x - Lx/2)/a) (+ images), phi = alpha psi, and the
    # two tangents: d x0/d alpha = (Psi0, 0) and d x0/d a (at fixed B_y amplitude)
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/k, dims=2, cfl_safety=0.5,
                           forcing=False, eqpars={"diss": (0.0, 1.0/S), "hyper": 1})
    kgrid = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    us = [(x - Lx/2 - m*Lx)/a for m in (-1, 0, 1)]
    p0 = sum(a*_sech2(u) for u in us)
    dp = sum(_sech2(u)*(1.0 + 2.0*u*np.tanh(u)) for u in us)

    def fields(phi, psi):
        phi, psi = jnp.asarray(phi).reshape(1, -1, 1), jnp.asarray(psi).reshape(1, -1, 1)
        return jr.initialize(lambda X, Y: jnp.stack([phi + 0*Y, psi + 0*Y]), params).fields
    return params, kgrid, fields(alpha*p0, p0), fields(p0, 0*p0), fields(alpha*dp, dp)


def _lead(params, kgrid, x0):
    B, _ = stability.ky_block_matrix(x0, kgrid, params, 1)
    return B, stability.eig_dense(B)


@pytest.mark.fp64
def test_running_exponent_matches_fd():
    S, ka, alpha, nx, Lx = 1e3, 1.0, 0.8, 512, 20.0
    p_converged = 0.52862          # the notebook, nx = 2048 (grid-converged to <1e-5)
    params, kgrid, x0, dxa, _ = _setup(S, ka, nx, Lx, alpha)
    B, r = _lead(params, kgrid, x0)
    lam = r.values[0]
    dB, _ = stability.dj_matrix(x0, dxa, kgrid, params, iky=1)
    s = stability.eigenvalue_sensitivity(B, dB, lam, r.right[:, 0], r.left[:, 0],
                                         spectrum=r.values)
    p = s.dlam.real*(1 - alpha**2)/(-2*alpha*lam.real)
    hs, errs = (1e-2, 3e-3, 1e-3), []
    for h in hs:
        vals = []
        for sg in (1, -1):
            _, _, xh, _, _ = _setup(S, ka, nx, Lx, alpha + sg*h)
            vals.append(_lead(params, kgrid, xh)[1].values[0])
        errs.append(abs((vals[0] - vals[1])/(2*h) - s.dlam))
    order = fit_order(hs, errs)
    v = r.right[:, 0]
    naive = np.vdot(v, dB @ v)/np.vdot(v, v)
    print(f"  lambda = {lam:.8f}, dlambda/dalpha = {s.dlam:.8f}, p = {p:.6f} (converged "
          f"{p_converged}), kappa = {s.kappa:.1f}, gap = {s.gap:.3e}; FD errs {errs[0]:.1e} -> "
          f"{errs[-1]:.1e} (order {order:.2f}); naive {naive.real:.6f}")
    with checks() as c:
        c.check("the tearing mode is real and growing", lam.real > 1e-2 and
                abs(lam.imag) < 1e-12*r.scale)
        c.check(f"dlambda/dalpha == central FD of eig_dense at O(h^2) (order {order:.2f})",
                1.9 < order < 2.1 and errs[-1] < 1e-5*abs(s.dlam), f"{errs}")
        c.check(f"running exponent p = {p:.6f} within 2e-3 of the grid-converged {p_converged}",
                abs(p - p_converged) < 2e-3, f"{abs(p - p_converged):.2e}")
        v_, w_ = r.right[:, 0], r.left[:, 0]
        kap = np.linalg.norm(w_)*np.linalg.norm(v_)/abs(np.vdot(w_, v_))
        c.check(f"kappa = {s.kappa:.4f} is ||w|| ||v||/|w^H v| of eig_dense's pair",
                abs(s.kappa/kap - 1) < 1e-10, f"{s.kappa} vs {kap}")
        c.check(f"non-normal: kappa = {s.kappa:.3f} within 1 % of {_KAPPA_512} (this grid)",
                abs(s.kappa/_KAPPA_512 - 1) < 1e-2, f"{s.kappa:.4f}")
        c.check("the right-vector-only formula is off by > 20 %",
                abs(naive - s.dlam) > 0.2*abs(s.dlam), f"{naive} vs {s.dlam}")


@pytest.mark.fp64
def test_marginal_newton_in_sheet_width():
    S, k, nx, Lx = 1e3, _SQRT5, 256, 20.0
    a = 2.0/_SQRT5
    rows = []
    with checks() as c:
        for n in range(7):
            params, kgrid, x0, _, dxw = _setup(S, k, nx, Lx, 0.0, a)
            B, r = _lead(params, kgrid, x0)
            lam = r.values[0]
            dB, _ = stability.dj_matrix(x0, dxw, kgrid, params, iky=1)
            s = stability.eigenvalue_sensitivity(B, dB, lam, r.right[:, 0], r.left[:, 0],
                                                 spectrum=r.values)
            if n == 2:           # FD gate of the width tangent (x0 is NOT affine in a)
                hs, fe = (4e-4, 2e-4, 1e-4), []
                for h in hs:
                    lp = _lead(*_setup(S, k, nx, Lx, 0.0, a + h)[:3])[1].values[0]
                    lm = _lead(*_setup(S, k, nx, Lx, 0.0, a - h)[:3])[1].values[0]
                    fe.append(abs((lp - lm)/(2*h) - s.dlam))
                order = fit_order(hs, fe)
                c.check(f"d gamma/d a = {s.dlam.real:.8e} == FD at O(h^2) (errs {fe[0]:.1e} -> "
                        f"{fe[-1]:.1e}, order {order:.2f})",
                        1.9 < order < 2.1 and fe[-1] < 3e-5*abs(s.dlam), f"{fe}")
            rows.append((a, lam.real, s.dlam.real, s.kappa))
            a = a - lam.real/s.dlam.real
    err = [1.0 - r_[0] for r_ in rows] + [1.0 - a]
    ratios = [err[i + 1]/err[i] for i in range(len(err) - 1)]
    q = [r_[1]/r_[2] for r_ in rows]
    p = [(rows[i][0] - rows[i - 1][0])/(q[i] - q[i - 1]) for i in range(1, len(rows))]
    astar = rows[-1][0] - p[-1]*q[-1]
    for i, (ai, g, dg, kap) in enumerate(rows):
        print(f"  it {i}: ka = {k*ai:.8f}  gamma = {g:.3e}  kappa = {kap:.1f}  "
              f"ratio {ratios[i]:.3f}" + (f"  p = {p[i - 1]:.3f}" if i else ""))
    print(f"  multiplicity-corrected root: ka* = {k*astar:.8f} (sqrt 5 = {_SQRT5:.8f}, "
          f"diff {k*astar - _SQRT5:.1e})")
    with checks() as c:
        c.check("plain Newton converges from the unstable side (every iterate grows)",
                all(g > 0 for _, g, _, _ in rows) and all(e > 0 for e in err))
        c.check(f"... LINEARLY, not quadratically: last contraction ratios {ratios[-3:]} in "
                f"(0.4, 0.6)", all(0.4 < x < 0.6 for x in ratios[-3:]))
        c.check(f"double root on this grid: p -> 2 ({p[-1]:.3f})", abs(p[-1] - 2) < 0.1)
        c.check(f"multiplicity-corrected Newton lands on ka = sqrt 5 to 5e-5 "
                f"({abs(k*astar - _SQRT5):.1e})", abs(k*astar - _SQRT5) < 5e-5)


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


def test_notebook_exponent_helper():
    SR = _run_script("tearing_sensitivity_run")
    with checks() as c:
        for al, p in ((0.3, 0.5), (0.8, 4/7), (0.99, 2/3)):
            C = 0.07
            g = C*(1 - al*al)**p
            dg = C*p*(1 - al*al)**(p - 1)*(-2*al)             # d gamma/d alpha, exact
            got = SR.exponent({"alpha": al, "lam": g + 1e-13j, "dlam": dg - 1e-13j})
            c.check(f"exponent(gamma = C(1 - alpha^2)^{p:.4f}) at alpha = {al} is p",
                    abs(got - p) < 1e-12, f"{got}")


def test_notebook_ridge_fit_helper():
    SR = _run_script("tearing_sensitivity_run")
    g0, cc, k0, q, rr = 0.02, 0.004, 0.5, 4/7, 1/7

    def rows_at(al, kas):
        u = 1 - al*al
        L, dL = np.log(k0) - rr*np.log(u), 2*al*rr/u            # ln k_max and d/dalpha
        G, dG = g0*u**q, g0*q*u**(q - 1)*(-2*al)
        return {ka: {"alpha": al, "lam": G - cc*(np.log(ka) - L)**2 + 0j,
                     "dlam": dG + 2*cc*(np.log(ka) - L)*dL + 0j} for ka in kas}, \
            np.exp(L), G, dG
    with checks() as c:
        for al in (0.3, 0.9):
            u = 1 - al*al
            _, km, _, _ = rows_at(al, (1.0,))
            kas = tuple(km*1.15**np.arange(-2.3, 2.4, 1.0))      # 5 points, k_max off-centre
            rows, km, G, dG = rows_at(al, kas)
            f = SR.ridge_fit(rows)
            c.check(f"ridge_fit alpha = {al}: k_max, gamma_max from the ln-k parabola",
                    abs(f["kmax"]/km - 1) < 1e-9 and abs(f["gmax"]/G - 1) < 1e-9,
                    f"{f['kmax']} vs {km}, {f['gmax']} vs {G}")
            c.check(f"ridge_fit alpha = {al}: d gamma_max/d alpha = dgamma/dalpha at k_max "
                    f"(envelope)", abs(f["dgmax"]/dG - 1) < 1e-8, f"{f['dgmax']} vs {dG}")
            c.check(f"ridge_fit alpha = {al}: p = {q:.6f} and pk = -{rr:.6f}",
                    abs(f["p"] - q) < 1e-8 and abs(f["pk"] + rr) < 1e-8, f"{f['p']}, {f['pk']}")
            c.check(f"ridge_fit alpha = {al}: inside, cubic spread at the 20001-point grid's "
                    f"resolution", f["inside"] and f["kmax_err"] < 1e-3*km
                    and f["gmax_err"] < 1e-9*G, f"{f['kmax_err']}, {f['gmax_err']}")
            f2 = SR.ridge_fit(rows_at(al, tuple(km*1.15**np.arange(1.0, 6.0)))[0])
            c.check(f"ridge_fit alpha = {al}: a k grid entirely above k_max is not 'inside'",
                    not f2["inside"])


def test_notebook_newton_table_helper():
    SR = _run_script("tearing_sensitivity_run")
    astar, C = 1.0, 0.3
    with checks() as c:
        for m in (2, 4):
            its, a = [], 0.8
            for n in range(5):
                g, dg = C*(astar - a)**m, -m*C*(astar - a)**(m - 1)
                its.append(dict(a=a, gamma=g, dgamma=dg, step=-g/dg, nx=2048*(n + 1),
                                rec={"kappa": 10.0 + n, "rel_err": 1e-9}))
                a = a - g/dg                                        # plain Newton
            its.append(dict(a=a, rungs=[], stop="no rung"))       # a failed iterate: skipped
            rows = SR.newton_table(its)
            c.check(f"m = {m}: one row per converged iterate", len(rows) == 5)
            c.check(f"m = {m}: first row has no multiplicity estimate",
                    np.isnan(rows[0]["p"]) and np.isnan(rows[0]["astar"]))
            c.check(f"m = {m}: q = gamma/gamma' and the row fields carried through",
                    all(abs(r["q"] - it["gamma"]/it["dgamma"]) < 1e-15 and r["a"] == it["a"]
                        and r["nx"] == it["nx"] and r["kappa"] == it["rec"]["kappa"]
                        for r, it in zip(rows, its)))
            ps = [r["p"] for r in rows[1:]]
            ast = [r["astar"] for r in rows[1:]]
            c.check(f"m = {m}: measured multiplicity p = {m} (1/slope of gamma/gamma')",
                    max(abs(x - m) for x in ps) < 1e-8, f"{ps}")
            c.check(f"m = {m}: multiplicity-corrected root a - p gamma/gamma' = a*",
                    max(abs(x - astar) for x in ast) < 1e-9, f"{ast}")


def test_notebook_fd_gate_helper():
    SR = _run_script("tearing_sensitivity_run")
    C, p = 0.05, 0.55

    def lam_of(al):
        return C*(1 - al*al)**p + 0j

    def dlam_of(al):
        return C*p*(1 - al*al)**(p - 1)*(-2*al) + 0j
    calls = []

    def fake_family(S, ka, alphas, root=None, log=print, levels=(1, 2), compute=True, **kw):
        assert len(alphas) == 1 and levels == (1,)
        al = alphas[0]
        return {1: {al: {"lam": lam_of(al), "dlam": dlam_of(al), "nx": 777, "Lx": 33.0}}}, True

    def fake_cached(kind, root=None, compute=None, **cfg):
        return compute()

    def fake_eig(S, k, a, alpha, nx, Lx, sigma, lam_hint=None):
        calls.append((alpha, nx, Lx))
        return {"lam": lam_of(alpha), "res": 0.0}
    with _patched(SR, alpha_family=fake_family, cached=fake_cached, eig_only=fake_eig), \
            checks() as c:
        for al, hs_exp in ((0.8, SR.FD_H), (0.99, SR.FD_H_NEAR1)):
            calls.clear()
            rec, rows = SR.fd_gate(1e4, 1.0, al, log=lambda *a: None)
            hs = tuple(r[0] for r in rows)
            fd_ref = [(lam_of(al + h) - lam_of(al - h))/(2*h) for h in hs]
            errs = [r[2] for r in rows]
            c.check(f"alpha = {al}: the notebook's h set {hs_exp}", hs == tuple(hs_exp), f"{hs}")
            c.check(f"alpha = {al}: FD eigenvalues at the sensitivity's own (nx, Lx)",
                    all(nx == 777 and Lx == 33.0 for _, nx, Lx in calls)
                    and sorted(a_ for a_, _, _ in calls) == sorted(al + sg*h for h in hs
                                                                   for sg in (1, -1)))
            c.check(f"alpha = {al}: fd = (lambda(alpha + h) - lambda(alpha - h))/2h",
                    all(abs(r[1] - f) <= 1e-12*abs(f) for r, f in zip(rows, fd_ref)),
                    f"{[r[1] for r in rows]} vs {fd_ref}")
            order = fit_order(hs, errs)
            c.check(f"alpha = {al}: |fd - dlambda| is O(h^2) (order {order:.3f})",
                    1.9 < order < 2.1 and all(abs(r[2] - abs(r[1] - dlam_of(al))) < 1e-15
                                              for r in rows), f"{errs}")
        rec, rows = SR.fd_gate(1e4, 1.0, 0.8, log=lambda *a: None, hs=(2e-3,))
        c.check("an explicit hs overrides the default set", [r[0] for r in rows] == [2e-3])


@pytest.mark.fp64
def test_notebook_sens_solve_matches_dense_path():
    SR = _run_script("tearing_sensitivity_run")
    S, ka, alpha, nx, Lx = 1e3, 1.0, 0.8, 512, 20.0
    params, kgrid, x0, dxa, _ = _setup(S, ka, nx, Lx, alpha)
    B, r = _lead(params, kgrid, x0)
    lam = r.values[0]
    v, w = r.right[:, 0], r.left[:, 0]
    dB, _ = stability.dj_matrix(x0, dxa, kgrid, params, iky=1)
    s = stability.eigenvalue_sensitivity(B, dB, lam, v, w, spectrum=r.values)
    naive = np.vdot(v, dB @ v)/np.vdot(v, v)
    p_dense = s.dlam.real*(1 - alpha**2)/(-2*alpha*lam.real)
    sigma = lam*1.03
    rec = SR.sens_solve(S, ka, 1.0, alpha, nx, Lx, sigma, lam_hint=lam, dense_nx=nx)
    # direct: the notebook's own equilibrium, matrix-free operators and adjoint LU
    x0n, dxn = SR.equilibrium(SR.make_params(S, ka, nx, Lx), alpha)
    op, _ = stability.ky_block_operator(x0n, kgrid, params, 1)
    dop, _ = stability.dj_operator(x0n, dxn, kgrid, params, iky=1)
    A0, A1 = SR._alpha_bands(S, ka, nx, Lx)
    _, MH = SR._lu_ops(A0 + alpha*A1, sigma)
    # sens_solve's inner-solve settings (a 1e-12 GMRES rtol stalls on some ARPACK starts: the
    # attainable accuracy of the shifted solve is ~1e-11 here); a fixed start vector
    le = stability.left_eigenvector(op, lam, sigma=sigma, M=MH, tol=1e-12, gmres_rtol=1e-10,
                                    restart=30, maxiter=20, v0=np.ones(len(v), dtype=complex))
    dv_err = np.linalg.norm(dop.matvec(v) - dB @ v)/np.linalg.norm(dB @ v)
    par = abs(np.vdot(le.left, w))/(np.linalg.norm(le.left)*np.linalg.norm(w))
    rel = {k: abs(complex(rec[k])/complex(ref) - 1) for k, ref in
           (("lam", lam), ("dlam", s.dlam), ("kappa", s.kappa), ("naive", naive))}
    print(f"  sens_solve: lam {complex(rec['lam']):.10f}, dlam {complex(rec['dlam']):.10f}, "
          f"kappa {float(rec['kappa']):.4f}; rel. to dense "
          f"{', '.join(f'{k} {v_:.1e}' for k, v_ in rel.items())}; "
          f"dj_operator vs dj_matrix {dv_err:.1e}; |cos(w_left_eigenvector, w_dense)| - 1 "
          f"{par - 1:.1e}; residuals r/l {float(rec['res_r']):.1e}/{float(rec['res_l']):.1e}")
    with checks() as c:
        for k, v_ in rel.items():
            c.check(f"sens_solve's {k} == the dense path's to 1e-8", v_ < 1e-8, f"{v_:.1e}")
        c.check("exponent(sens_solve record) == the dense path's p",
                abs(SR.exponent(rec) - p_dense) < 1e-8, f"{SR.exponent(rec)} vs {p_dense}")
        c.check("dj_operator's action == dj_matrix @ v", dv_err < 1e-12, f"{dv_err:.1e}")
        c.check("left_eigenvector's w (matrix-free, adjoint LU) is parallel to eig_dense's left",
                abs(par - 1) < 1e-10 and abs(le.value - lam) < 1e-10*abs(lam), f"{par - 1:.1e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
