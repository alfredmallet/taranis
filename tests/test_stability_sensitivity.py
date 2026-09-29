# taranis.stability rung 1b machinery (plans/AUTODIFF_PLAN.md "Rung 1b"): dJ/ds along a state
# direction dx0 (forward-over-forward), left eigenvectors by shift-invert on J^H, and eigenvalue
# sensitivities dlambda/ds = w^H dJ v / w^H v with their refusal criteria.
#   1. FD gate: dJ against [J(x0 + h dx0) - J(x0 - h dx0)]/2h. RMHD and GDI (quadratic N: J is
#      affine in x0, the difference is EXACT to round-off at every h): a 1D x0 in a ky block
#      (dense and matrix-free), a 2D x0 in real coordinates, GDI on the fields array. CMHD
#      (rho and lnrho: 1/rho, h(rho), e^s are not polynomial) converges at O(h^2): a ky block
#      about an (x, z)-dependent x0, and real coordinates about a fully 3D x0.
#   2. dlambda/ds against the central difference of eig_dense eigenvalues at x0 +- h dx0,
#      O(h^2): periodic sech^2 shear tearing (Phi0 = alpha Psi0, alpha = 0.8, direction
#      d x0/d alpha = (Psi0, 0)), dense and fully matrix-free (shift_invert + left_eigenvector
#      + dj_operator); a 2D x0 in real coordinates. EXACT: CMHD about a uniform state, where
#      the background enters as k = 0 modes of the fields, so d lambda along dB0, du0, drho0
#      is the hand derivative of the closed-form -D k^2h - i k.u0 +- i omega_{f,s,A}(rho0, B0)
#      (Doppler, Alfven, fast and slow branches, both density variables).
#   3. left_eigenvector (dense LU, GMRES on the ky block, real coordinates) against eig_dense's
#      left vector up to a phase, residual-gated; a sigma nearer another eigenvalue raises.
#   4. Loud failures: defective (Jordan) and semisimple multiple eigenvalues (x0 = 0 with
#      nu = eta; the conjugate copy of a real ky-block eigenvalue in real coordinates) refuse;
#      w^H v at round-off; a w that is not the left partner; a propagator-accuracy pair; lam
#      absent from spectrum; a non-invariant dx0 in a mode block (ky, kx, kz).
# fp64 only (round-off tolerances).
# pytest: `pytest tests/test_stability_sensitivity.py`. Script:
# `python tests/test_stability_sensitivity.py`.
from _rmhd_testing import bootstrap, checks, fit_order, fresh_params

bootstrap()

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import jax.numpy as jnp

import taranis as jr
from taranis import stability

_RNG_SEED = 20260929


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


def _pair(a, b):
    # column index into b paired with each entry of a (optimal assignment)
    cost = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    r, c = linear_sum_assignment(cost)
    return c[np.argsort(r)]


def _nearest(values, lam):
    return values[np.argmin(np.abs(values - lam))]


# ------------------------------------------------------------------------ configurations

_SHEAR = dict(nx=48, Lx=16.0, k=0.5, eta=0.02, alpha=0.8)


def _shear_case():
    # periodic sech^2 sheet (localized: Lx = 16), Phi0 = alpha Psi0; dx0 = d x0/d alpha
    c = _SHEAR
    params = fresh_params(dims=2, nx=c["nx"], ny=8, Lx=c["Lx"], Ly=2*np.pi/c["k"],
                          diss=(c["eta"], c["eta"]), hyper=1)
    kgrid = jr.setup_kgrids(params)
    psi = lambda x, y: 1/jnp.cosh(x - c["Lx"]/2)**2 + 0*y
    x0 = jr.initialize(lambda x, y: jnp.stack([c["alpha"]*psi(x, y), psi(x, y)]), params).fields
    dx0 = jr.initialize(lambda x, y: jnp.stack([psi(x, y), 0*psi(x, y)]), params).fields
    return params, kgrid, np.asarray(x0), np.asarray(dx0)


def _twod_case():
    params = fresh_params(dims=2, nx=12, ny=12, Lx=2*np.pi, Ly=2*np.pi, diss=(0.02, 0.01),
                          hyper=1)
    kgrid = jr.setup_kgrids(params)
    x0 = jr.initialize(lambda x, y: jnp.stack(
        [0.5*jnp.sin(x)*jnp.cos(2*y) + 0.2*jnp.cos(x + y),
         jnp.cos(x)*jnp.cos(y) + 0.3*jnp.sin(2*x + y)]), params).fields
    dx0 = jr.initialize(lambda x, y: jnp.stack(
        [0.3*jnp.cos(x - y), 0.4*jnp.sin(x)*jnp.sin(y) + 0.2*jnp.cos(2*y)]), params).fields
    return params, kgrid, np.asarray(x0), np.asarray(dx0)


def _gdi_params():
    return fresh_params(dims=2, nx=16, ny=16, Lx=2*np.pi, Ly=2*np.pi, eqtype="GDI",
                        comm_backend="serial",
                        eqpars=dict(Ln=1.0, nu_in=0.3, v0=2.0, diss=0.01, hyper=1, gpar_fac=0.1))


_CMHD_BOX = dict(dims=3, nx=8, ny=8, nz=8, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, z_spectral=True,
                 eqtype="CMHD", comm_backend="serial")


def _cmhd_params(cs0=0.8, gamma=1.0, density_var="rho", diss=0.05, hyper=2):
    eq = dict(cs0=cs0, diss=diss, hyper=hyper, gamma=gamma)
    if density_var != "rho":
        eq["density_var"] = density_var
    return fresh_params(eqpars=eq, **_CMHD_BOX)


def _const(params, vals):
    # the fields of a spatially uniform state (k = 0 modes only)
    return np.asarray(jr.initialize(
        lambda x, y, z: jnp.stack([v + 0*x + 0*y + 0*z for v in vals]), params).fields)


# ------------------------------------------------------------------------------ gate 1

@pytest.mark.fp64
def test_fd_gate_dj():
    rng = np.random.default_rng(_RNG_SEED)
    with checks() as c:
        # RMHD, 1D x0 in the ky block: dense, matrix-free, fields-array forms
        params, kgrid, x0, dx0 = _shear_case()
        generic = np.asarray(jr.initialize(lambda x, y: jnp.stack(
            [jnp.sin(2*np.pi*x/params.Lx) + 0*y, 0.3*jnp.cos(4*np.pi*x/params.Lx) + 0*y]),
            params).fields)
        for dname, d in (("d/dalpha", dx0), ("generic 1D dx0", generic)):
            dB, idx = stability.dj_matrix(x0, d, kgrid, params, iky=1)
            for h in (1e-2, 1e-3):
                fd = (stability.ky_block_matrix(x0 + h*d, kgrid, params, 1)[0]
                      - stability.ky_block_matrix(x0 - h*d, kgrid, params, 1)[0])/(2*h)
                e = np.abs(dB - fd).max()/np.abs(dB).max()
                c.check(f"RMHD shear sheet, {dname}, ky block: dJ == central FD at h={h:g} "
                        f"(max rel {e:.1e}; exact, N quadratic)", e < 1e-10, f"{e:.3e}")
            c.check(f"RMHD {dname}: dJ is nontrivial ({np.abs(dB).max():.2f})",
                    np.abs(dB).max() > 1e-2)
            op, _ = stability.dj_operator(x0, d, kgrid, params, iky=1)
            u = rng.standard_normal(len(idx)) + 1j*rng.standard_normal(len(idx))
            e = _relerr(op.matvec(u), dB @ u)
            c.check(f"RMHD {dname}: dj_operator matvec == dj_matrix @ u (rel {e:.1e})",
                    e < 1e-13, f"{e:.3e}")
            v = np.zeros(x0.shape, dtype=complex)
            v[idx[:, 0], idx[:, 1], idx[:, 2], 1] = u
            g = np.asarray(stability.djvp_operator(x0, d, kgrid, params)(v))
            e = _relerr(g[idx[:, 0], idx[:, 1], idx[:, 2], 1], dB @ u)
            c.check(f"RMHD {dname}: djvp_operator on the fields array == the block (rel "
                    f"{e:.1e})", e < 1e-13, f"{e:.3e}")

        # RMHD, 2D x0 in real coordinates
        params, kgrid, x0, dx0 = _twod_case()
        dM, coords = stability.dj_matrix(x0, dx0, kgrid, params)
        for h in (1e-2, 1e-3):
            fd = (stability.real_matrix(x0 + h*dx0, kgrid, params, coords)[0]
                  - stability.real_matrix(x0 - h*dx0, kgrid, params, coords)[0])/(2*h)
            e = np.abs(dM - fd).max()/np.abs(dM).max()
            c.check(f"RMHD 2D x0, real coords (n={coords.n}): dJ == central FD at h={h:g} "
                    f"(max rel {e:.1e})", e < 1e-10, f"{e:.3e}")
        op, _ = stability.dj_operator(x0, dx0, kgrid, params, coords=coords)
        u = rng.standard_normal(coords.n)
        e = _relerr(op.matvec(u), dM @ u)
        c.check(f"RMHD 2D x0: real dj_operator matvec == dj_matrix @ u (rel {e:.1e})",
                e < 1e-13, f"{e:.3e}")

        # GDI, 2D x0 on the fields array
        params = _gdi_params()
        kgrid = jr.setup_kgrids(params)
        x0 = np.asarray(jr.initialize(lambda x, y: jnp.stack(
            [0.3*jnp.cos(x + y), jnp.sin(x)*jnp.cos(2*y)]), params).fields)
        dx0 = np.asarray(jr.initialize(lambda x, y: jnp.stack(
            [0.2*jnp.sin(2*x - y), 0.5*jnp.cos(x)*jnp.sin(y)]), params).fields)
        coords = stability.real_coords(kgrid, params)
        v = np.asarray(coords.unpack(jnp.asarray(rng.standard_normal(coords.n))))
        dJv = np.asarray(stability.djvp_operator(x0, dx0, kgrid, params)(v))
        for h in (1e-2, 1e-3):
            fd = (np.asarray(stability.jvp_operator(x0 + h*dx0, kgrid, params)(v))
                  - np.asarray(stability.jvp_operator(x0 - h*dx0, kgrid, params)(v)))/(2*h)
            e = _relerr(fd, dJv)
            c.check(f"GDI 2D x0, fields array: dJ v == central FD at h={h:g} (rel {e:.1e})",
                    e < 1e-10, f"{e:.3e}")

        # CMHD: not polynomial, O(h^2)
        hs = (1e-1, 3e-2, 1e-2)
        for dv, gamma in (("rho", 5.0/3.0), ("lnrho", 1.0)):
            params = _cmhd_params(gamma=gamma, density_var=dv)
            kgrid = jr.setup_kgrids(params)
            ln = dv == "lnrho"

            def f0(r):
                return jnp.log(r) if ln else r

            o = lambda x, y, z: 0*x + 0*y + 0*z
            xz = np.asarray(jr.initialize(lambda x, y, z: jnp.stack(
                [f0(1.0 + 0.2*jnp.cos(x) + 0.1*jnp.sin(z - x)) + o(x, y, z),
                 0.3*jnp.sin(x + z) + o(x, y, z), 0.2*jnp.cos(x) + o(x, y, z),
                 0.1*jnp.sin(x - z) + o(x, y, z), 0.3 + 0.2*jnp.cos(z) + o(x, y, z),
                 0.1*jnp.sin(x + z) + o(x, y, z), 1.0 + 0.1*jnp.cos(x) + o(x, y, z)]),
                params).fields)
            dxz = np.asarray(jr.initialize(lambda x, y, z: jnp.stack(
                [0.1*jnp.cos(x - z) + o(x, y, z), 0.1*jnp.cos(z) + o(x, y, z),
                 0.05*jnp.sin(x) + o(x, y, z), 0.1*jnp.cos(x + z) + o(x, y, z),
                 0.1*jnp.sin(z) + o(x, y, z), 0.05*jnp.cos(x) + o(x, y, z),
                 0.1*jnp.sin(x - z) + o(x, y, z)]), params).fields)
            full = np.asarray(jr.initialize(lambda x, y, z: jnp.stack(
                [f0(1.0 + 0.2*jnp.cos(x + y) + 0.1*jnp.sin(z - x)) + o(x, y, z),
                 0.3*jnp.sin(y + z) + o(x, y, z), 0.2*jnp.cos(x) + o(x, y, z),
                 0.1*jnp.sin(x - z) + o(x, y, z), 0.2*jnp.cos(z + y) + 0.3 + o(x, y, z),
                 0.1*jnp.sin(x + z) + o(x, y, z), 1.0 + 0.1*jnp.cos(y) + o(x, y, z)]),
                params).fields)
            dfull = np.asarray(jr.initialize(lambda x, y, z: jnp.stack(
                [0.1*jnp.cos(x - y + z) + o(x, y, z), 0.1*jnp.cos(y) + o(x, y, z),
                 0.05*jnp.sin(x + y) + o(x, y, z), 0.1*jnp.cos(z) + o(x, y, z),
                 0.1*jnp.sin(y - z) + o(x, y, z), 0.05*jnp.cos(x) + o(x, y, z),
                 0.1*jnp.sin(x + y) + o(x, y, z)]), params).fields)
            for space, x0, dx0, kw in (("ky block, x0(x,z)", xz, dxz, dict(iky=1)),
                                       ("real coords, x0(x,y,z)", full, dfull, {})):
                if kw:
                    Jop = lambda x: stability.ky_block_operator(x, kgrid, params, 1)[0]
                    n = len(stability.ky_block_index(kgrid, params, 1))
                    u = rng.standard_normal(n) + 1j*rng.standard_normal(n)
                else:
                    coords = stability.real_coords(kgrid, params)
                    kw = dict(coords=coords)
                    Jop = lambda x: stability.real_operator(x, kgrid, params, coords)[0]
                    u = rng.standard_normal(coords.n)
                dJu = stability.dj_operator(x0, dx0, kgrid, params, **kw)[0].matvec(u)
                errs = [_relerr((Jop(x0 + h*dx0).matvec(u) - Jop(x0 - h*dx0).matvec(u))/(2*h),
                                dJu) for h in hs]
                order = fit_order(hs, errs)
                c.check(f"CMHD {dv} gamma={gamma:.3f}, {space}: dJ u == central FD at O(h^2) "
                        f"(errs {errs[0]:.1e} -> {errs[-1]:.1e}, order {order:.2f})",
                        1.8 < order < 2.2 and errs[-1] < 1e-4, f"order {order:.3f}, {errs}")


# ------------------------------------------------------------------------------ gate 2

@pytest.mark.fp64
def test_sensitivity_shear_tearing():
    params, kgrid, x0, dx0 = _shear_case()
    with checks() as c:
        B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        r = stability.eig_dense(B)
        lam = r.values[0]
        c.check(f"alpha = {_SHEAR['alpha']} sech^2 shear sheet is tearing-unstable (lambda = "
                f"{lam:.5f})", lam.real > 0)
        dB, _ = stability.dj_matrix(x0, dx0, kgrid, params, iky=1)
        s = stability.eigenvalue_sensitivity(B, dB, lam, r.right[:, 0], r.left[:, 0],
                                             spectrum=r.values)
        hs, errs = (1e-2, 3e-3, 1e-3), []
        for h in hs:
            lp = _nearest(stability.eig_dense(
                stability.ky_block_matrix(x0 + h*dx0, kgrid, params, 1)[0]).values, lam)
            lm = _nearest(stability.eig_dense(
                stability.ky_block_matrix(x0 - h*dx0, kgrid, params, 1)[0]).values, lam)
            errs.append(abs((lp - lm)/(2*h) - s.dlam))
        order = fit_order(hs, errs)
        c.check(f"d lambda/d alpha = {s.dlam:.6f} (kappa {s.kappa:.2f}, gap {s.gap:.3f}) == "
                f"central FD of eig_dense at O(h^2) (errs {errs[0]:.1e} -> {errs[-1]:.1e}, "
                f"order {order:.2f})", 1.9 < order < 2.1 and errs[-1] < 1e-6 and
                abs(s.dlam) > 1e-2, f"order {order:.3f}, {errs}")
        c.check(f"sensitivity bookkeeping: kappa == 1/|w^H v| >= 1, rel_err tiny "
                f"({s.rel_err:.1e}), residuals {s.residual_right:.1e}/{s.residual_left:.1e}",
                s.kappa >= 1 and abs(s.kappa*abs(s.wv) - 1) < 1e-14 and s.rel_err < 1e-10)
        # unnormalized, rephased vectors and a rescaled J: same dlambda, kappa, rel_err (the
        # last to its round-off residuals' own jitter)
        s2 = stability.eigenvalue_sensitivity(1e4*B, 1e4*dB, 1e4*lam, 3.0*r.right[:, 0],
                                              (2 - 1j)*r.left[:, 0], spectrum=1e4*r.values)
        c.check("dlambda, kappa and rel_err are invariant under |v|, |w|, phases and J -> 1e4 J "
                f"(rel_err {s.rel_err:.2e} vs {s2.rel_err:.2e})",
                abs(s2.dlam/1e4 - s.dlam) < 1e-12*abs(s.dlam) and abs(s2.kappa - s.kappa) <
                1e-12*s.kappa and abs(s2.rel_err - s.rel_err) < 0.1*s.rel_err)
        # a non-normal block: kappa > 1 is what makes the left vector matter; dlambda from
        # v^H dJ v / v^H v (the wrong, "self-adjoint" formula) differs
        v = r.right[:, 0]
        naive = np.vdot(v, dB @ v)/np.vdot(v, v)
        c.check(f"non-normal: kappa = {s.kappa:.2f} > 1.5 and the right-vector-only formula "
                f"is off by {abs(naive - s.dlam):.2e}", s.kappa > 1.5 and
                abs(naive - s.dlam) > 1e-3*abs(s.dlam))

        # fully matrix-free: shift_invert right pair, left_eigenvector (GMRES), dj_operator
        op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
        dop, _ = stability.dj_operator(x0, dx0, kgrid, params, iky=1)
        n = len(idx)
        right = stability.shift_invert(op, lam + 1e-3 + 1e-3j, k=3, restart=n)
        lam_mf = right.values[0]
        left = stability.left_eigenvector(op, lam_mf, restart=n)
        smf = stability.eigenvalue_sensitivity(op, dop, lam_mf, right.right[:, 0], left.left,
                                               spectrum=right.values)
        e = abs(smf.dlam - s.dlam)/abs(s.dlam)
        c.check(f"matrix-free (shift_invert + left_eigenvector GMRES + dj_operator): dlambda == "
                f"dense (rel {e:.1e}, gap from shift_invert's 3 values {smf.gap:.3f})",
                e < 1e-8, f"{e:.3e}")

        # real coordinates, a 2D x0: every eigenvalue simple, FD of the real matrix
        params, kgrid, x0, dx0 = _twod_case()
        M, coords = stability.real_matrix(x0, kgrid, params)
        rr = stability.eig_dense(M)
        dM, _ = stability.dj_matrix(x0, dx0, kgrid, params, coords=coords)
        for i in (0, int(np.argmax(np.abs(rr.values.imag)))):
            lam = rr.values[i]
            s = stability.eigenvalue_sensitivity(M, dM, lam, rr.right[:, i], rr.left[:, i],
                                                 spectrum=rr.values)
            errs = []
            for h in hs:
                lp = _nearest(np.linalg.eigvals(
                    stability.real_matrix(x0 + h*dx0, kgrid, params, coords)[0]), lam)
                lm = _nearest(np.linalg.eigvals(
                    stability.real_matrix(x0 - h*dx0, kgrid, params, coords)[0]), lam)
                errs.append(abs((lp - lm)/(2*h) - s.dlam))
            order = fit_order(hs, errs)
            c.check(f"2D x0, real coords (n={coords.n}), lambda = {lam:.4f}: dlambda = "
                    f"{s.dlam:.4f} (kappa {s.kappa:.2f}) == FD at O(h^2) (errs {errs[0]:.1e} "
                    f"-> {errs[-1]:.1e}, order {order:.2f})",
                    1.9 < order < 2.1 and errs[-1] < 1e-5, f"order {order:.3f}, {errs}")
        # matrix-free in real coordinates (J^T by vjp, complexified), the complex eigenvalue
        rop, _ = stability.real_operator(x0, kgrid, params, coords)
        dop, _ = stability.dj_operator(x0, dx0, kgrid, params, coords=coords)
        right = stability.shift_invert(rop, lam + 0.01 + 0.01j, k=3, restart=coords.n)
        left = stability.left_eigenvector(rop, right.values[0], restart=coords.n)
        smf = stability.eigenvalue_sensitivity(rop, dop, right.values[0], right.right[:, 0],
                                               left.left, spectrum=right.values)
        e = abs(smf.dlam - s.dlam)/abs(s.dlam)
        c.check(f"real coords matrix-free (real_operator, J^T complexified): dlambda == dense "
                f"(rel {e:.1e}, left residual {smf.residual_left:.1e})", e < 1e-8, f"{e:.3e}")


def _cmhd_exact_with_derivative(k, cs0, gamma, D, hyper, rho, u, B, drho, du, dB):
    # the closed-form uniform-state eigenvalues (tests/test_stability_general.py:
    # -D|k|^2h - i k.u +- i omega_{f,s,A}, and the div B mode -D|k|^2h) and their hand
    # derivative along (drho, du, dB):
    #   c^2 = cs0^2 rho^(gamma-1),     dc^2 = (gamma-1) c^2 drho/rho
    #   vA = B/sqrt(rho),              dvA = dB/sqrt(rho) - vA drho/(2 rho)
    #   a = k^2 (c^2 + vA^2), b = k^2 c^2 (k.vA)^2, R = sqrt(a^2 - 4b)
    #   omega_{f,s}^2 = (a +- R)/2,    d omega = (da +- dR)/(4 omega),  dR = (a da - 2 db)/R
    #   omega_A = |k.vA|,              d omega_A = sign(k.vA) k.dvA
    #   d lambda = -i k.du +- i d omega (the damping and the div B mode do not move)
    k2 = k @ k
    c2 = cs0*cs0*rho**(gamma - 1.0)
    dc2 = (gamma - 1.0)*c2*drho/rho
    vA = B/np.sqrt(rho)
    dvA = dB/np.sqrt(rho) - vA*drho/(2*rho)
    kva, kdva = k @ vA, k @ dvA
    a, da = k2*(c2 + vA @ vA), k2*(dc2 + 2*(vA @ dvA))
    b, db = k2*c2*kva**2, k2*(dc2*kva**2 + 2*c2*kva*kdva)
    R = np.sqrt(a*a - 4*b)
    dR = (a*da - 2*db)/R
    wf, ws, wa = np.sqrt((a + R)/2), np.sqrt((a - R)/2), abs(kva)
    dwf, dws, dwa = (da + dR)/(4*wf), (da - dR)/(4*ws), np.sign(kva)*kdva
    damp, dop, ddop = -D*k2**hyper, -1j*(k @ u), -1j*(k @ du)
    lam, dlam = [], []
    for w, dw in ((wf, dwf), (ws, dws), (wa, dwa)):
        for sg in (1, -1):
            lam.append(damp + dop + sg*1j*w)
            dlam.append(ddop + sg*1j*dw)
    return np.array(lam + [damp]), np.array(dlam + [0.0])


@pytest.mark.fp64
def test_sensitivity_cmhd_uniform_exact():
    rho0, u0, B0 = 1.3, np.array([0.1, -0.05, 0.07]), np.array([0.3, 0.4, 1.0])
    cs0, D, hyper = 0.8, 0.05, 2
    mode = (1, 1, 2)                              # (ikx, iky, iz): k = (1, 1, 2)
    k = np.array([1.0, 1.0, 2.0])
    directions = (("dB0", 0.0, np.zeros(3), np.array([0.2, -0.1, 1.0])),
                  ("du0", 0.0, np.array([0.3, 0.5, -0.2]), np.zeros(3)),
                  ("drho0", 1.0, np.zeros(3), np.zeros(3)),
                  ("mixed", 0.4, np.array([0.1, 0.0, 0.3]), np.array([-0.3, 0.2, 0.1])))
    with checks() as c:
        for dv, gamma in (("rho", 5.0/3.0), ("lnrho", 1.0), ("rho", 1.0)):
            params = _cmhd_params(cs0=cs0, gamma=gamma, density_var=dv, diss=D, hyper=hyper)
            kgrid = jr.setup_kgrids(params)
            ln = dv == "lnrho"
            x0 = _const(params, [np.log(rho0) if ln else rho0, *u0, *B0])
            blk = dict(iky=mode[1], ikx=mode[0], iz=mode[2])
            B, idx = stability.ky_block_matrix(x0, kgrid, params, **blk)
            r = stability.eig_dense(B)
            for name, drho, du, dB in directions:
                dx0 = _const(params, [drho/rho0 if ln else drho, *du, *dB])
                want, dwant = _cmhd_exact_with_derivative(k, cs0, gamma, D, hyper, rho0, u0, B0,
                                                          drho, du, dB)
                dM, _ = stability.dj_matrix(x0, dx0, kgrid, params, **blk)
                cols = _pair(r.values, want)
                e = np.abs(r.values - want[cols]).max()/np.abs(want).max()
                got = np.array([stability.eigenvalue_sensitivity(
                    B, dM, r.values[i], r.right[:, i], r.left[:, i], spectrum=r.values).dlam
                    for i in range(7)])
                de = np.abs(got - dwant[cols]).max()/np.abs(dwant).max()
                c.check(f"CMHD {dv} gamma={gamma:.3f}, {name}: 7 dlambda == hand derivative of "
                        f"the closed form (rel {de:.1e}; eigenvalues rel {e:.1e}; max |dlambda| "
                        f"{np.abs(dwant).max():.3f})", de < 1e-11 and e < 1e-12, f"rel {de:.3e}")
        # matrix-free on one mode: ky_block_operator + shift_invert + left_eigenvector
        params = _cmhd_params(cs0=cs0, gamma=5.0/3.0, diss=D, hyper=hyper)
        kgrid = jr.setup_kgrids(params)
        x0 = _const(params, [rho0, *u0, *B0])
        name, drho, du, dB = directions[3]
        dx0 = _const(params, [drho, *du, *dB])
        want, dwant = _cmhd_exact_with_derivative(k, cs0, 5.0/3.0, D, hyper, rho0, u0, B0,
                                                  drho, du, dB)
        op, _ = stability.ky_block_operator(x0, kgrid, params, **blk)
        dop, _ = stability.dj_operator(x0, dx0, kgrid, params, **blk)
        j = int(np.argmax(want.imag))                     # the upper fast wave
        right = stability.shift_invert(op, want[j] + 0.02 + 0.01j, k=3, tol=1e-13, restart=7)
        left = stability.left_eigenvector(op, right.values[0], tol=1e-13, restart=7)
        s = stability.eigenvalue_sensitivity(op, dop, right.values[0], right.right[:, 0],
                                             left.left, spectrum=right.values)
        de = abs(s.dlam - dwant[j])/abs(dwant[j])
        c.check(f"CMHD matrix-free, upper fast wave, {name}: dlambda = {s.dlam:.5f} == exact "
                f"(rel {de:.1e})", de < 1e-10, f"rel {de:.3e}")


# ------------------------------------------------------------------------------ gate 3

def _phase_err(a, b):
    # 1 - |a^H b|/(|a||b|): zero iff a, b are parallel (up to a complex phase)
    return 1.0 - abs(np.vdot(a, b))/(np.linalg.norm(a)*np.linalg.norm(b))


@pytest.mark.fp64
def test_left_eigenvector():
    params, kgrid, x0, _ = _shear_case()
    with checks() as c:
        B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        r = stability.eig_dense(B)
        op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
        for i in (0, 1, 2):
            lam, w0 = r.values[i], r.left[:, i]
            # modes 1, 2 sit 2.3e-3 apart, inside the default offset 1e-3 ||J||: own sigma, dense
            # only (unpreconditioned GMRES that close to the damped cluster stalls above
            # rtol = 1e-12 -- and raises, which is the contract)
            sig = None if i == 0 else lam + 1e-4*(1 + 1j)
            paths = [("dense LU", B, {})]
            if i == 0:
                paths.append(("GMRES ky block", op, dict(restart=len(idx))))
            for path, J, kw in paths:
                le = stability.left_eigenvector(J, lam, sigma=sig, **kw)
                pe = _phase_err(le.left, w0)
                res = (np.linalg.norm(B.conj().T @ le.left - np.conj(le.value)*le.left)
                       / max(abs(le.value), r.scale))
                c.check(f"mode {i} ({lam:.4f}), {path}: w == eig_dense's left up to a phase "
                        f"(1 - |<w, w0>| = {pe:.1e}), value within {abs(le.value - lam):.1e}, "
                        f"residual {le.residual:.1e} (recomputed {res:.1e})",
                        pe < 1e-10 and abs(le.value - lam) < 1e-10 and le.residual < 1e-9
                        and res < 1e-9, f"{pe:.3e}, {abs(le.value - lam):.3e}, {res:.3e}")
            c.check(f"mode {i}: the left vector is not the right one (non-normal: "
                    f"{_phase_err(r.right[:, i], w0):.2e})", _phase_err(r.right[:, i], w0) > 1e-4)
        # sigma nearer another eigenvalue: refused, never the wrong w -- explicitly, and the
        # default offset on the close pair (modes 1, 2)
        with pytest.raises(RuntimeError, match="another eigenvalue"):
            stability.left_eigenvector(B, r.values[0], sigma=r.values[1] + 1e-6)
        with pytest.raises(RuntimeError, match="another eigenvalue"):
            stability.left_eigenvector(B, r.values[2])
        c.check("a sigma nearer another eigenvalue raises (explicit, and the default offset on "
                f"a pair {abs(r.values[1] - r.values[2]):.1e} apart)", True)

        # a genuinely complex block (CMHD uniform state with a flow: complex eigenvalues,
        # eigenvectors not real up to a phase) -- the shear block cannot tell J^H from J^T
        pc = _cmhd_params(gamma=5.0/3.0)
        kc = jr.setup_kgrids(pc)
        xc = _const(pc, [1.3, 0.1, -0.05, 0.07, 0.3, 0.4, 1.0])
        Bc, _ = stability.ky_block_matrix(xc, kc, pc, 1, ikx=1, iz=2)
        rc = stability.eig_dense(Bc)
        j = int(np.argmax(rc.values.imag))
        le = stability.left_eigenvector(Bc, rc.values[j])
        pe = _phase_err(le.left, rc.left[:, j])
        c.check(f"CMHD block, complex lambda {rc.values[j]:.4f}: dense left vector == "
                f"eig_dense's up to a phase ({pe:.1e})", pe < 1e-10 and
                abs(le.value - rc.values[j]) < 1e-10, f"{pe:.3e}")

        # real coordinates (J real, adjoint J^T), a 2D x0: complex eigenvalue
        params, kgrid, x0, _ = _twod_case()
        M, coords = stability.real_matrix(x0, kgrid, params)
        rr = stability.eig_dense(M)
        rop, _ = stability.real_operator(x0, kgrid, params, coords)
        i = int(np.argmax(np.abs(rr.values.imag)))
        le = stability.left_eigenvector(rop, rr.values[i], restart=coords.n)
        pe = _phase_err(le.left, rr.left[:, i])
        c.check(f"real coords (n={coords.n}), complex lambda {rr.values[i]:.4f}: GMRES left "
                f"vector == eig_dense's up to a phase ({pe:.1e}, residual {le.residual:.1e})",
                pe < 1e-10 and le.residual < 1e-9, f"{pe:.3e}")


# ------------------------------------------------------------------------------ gate 4

@pytest.mark.fp64
def test_loud_failures():
    rng = np.random.default_rng(_RNG_SEED)
    with checks() as c:
        # defective: a 2x2 Jordan block (computed split ~sqrt(eps)) inside a random basis
        n = 6
        T = np.diag([0.5, 0.5, -1.0, -2.0 + 1j, 3.0, -0.3j]).astype(complex)
        T[0, 1] = 1.0
        V = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
        A = V @ T @ np.linalg.inv(V)
        dA = rng.standard_normal((n, n))
        r = stability.eig_dense(A)
        jd = np.argsort(np.abs(r.values - 0.5))[:2]
        for i in jd:
            with pytest.raises(RuntimeError, match="not numerically simple"):
                stability.eigenvalue_sensitivity(A, dA, r.values[i], r.right[:, i],
                                                 r.left[:, i], spectrum=r.values)
        split = abs(r.values[jd[0]] - r.values[jd[1]])
        c.check(f"defective (Jordan) eigenvalue refused (computed split {split:.1e})", True)
        # scale-relative: the same refusal for c A at any c (delta is measured against ||J||)
        for cs in (1e-6, 1e6):
            rc = stability.eig_dense(cs*A)
            for i in np.argsort(np.abs(rc.values - 0.5*cs))[:2]:
                with pytest.raises(RuntimeError, match="not numerically simple"):
                    stability.eigenvalue_sensitivity(cs*A, cs*dA, rc.values[i], rc.right[:, i],
                                                     rc.left[:, i], spectrum=rc.values)
        c.check("the Jordan refusal is scale-invariant (J scaled by 1e-6 and 1e6)", True)
        # computed eigenvalues split below eps ||J|| are refused even with EXACT vectors (zero
        # residuals: the backward error is floored at eps ||J||, the precision J came with)
        D = np.diag([1.0, 1.0 + np.finfo(float).eps, -2.0]).astype(complex)
        e0 = np.eye(3)[:, 0]
        with pytest.raises(RuntimeError, match="not numerically simple"):
            stability.eigenvalue_sensitivity(D, np.eye(3), D[0, 0], e0, e0,
                                             spectrum=np.diag(D))
        c.check("an eigenvalue split by eps with exact (zero-residual) vectors is refused", True)
        # the simple eigenvalues of the same matrix pass, exactly: dlambda = w^H dA v/(w^H v)
        # against the similarity-transformed closed form (T's left/right vectors)
        i = int(np.argmin(np.abs(r.values - 3.0)))
        s = stability.eigenvalue_sensitivity(A, dA, r.values[i], r.right[:, i], r.left[:, i],
                                             spectrum=r.values)
        Vi = np.linalg.inv(V)
        want = (Vi @ dA @ V)[4, 4]
        c.check(f"a simple eigenvalue of the same matrix passes (dlambda rel err "
                f"{abs(s.dlam - want)/abs(want):.1e})", abs(s.dlam - want) < 1e-10*abs(want))
        # exactly nilpotent: w^H v is exactly at round-off
        N2 = np.array([[0.0, 1.0], [0.0, 0.0]])
        rn = stability.eig_dense(N2)
        with pytest.raises(RuntimeError, match="round-off|not numerically simple"):
            stability.eigenvalue_sensitivity(N2, np.eye(2), rn.values[0], rn.right[:, 0],
                                             rn.left[:, 0], spectrum=())
        c.check("exact Jordan block with spectrum=() opt-out: still refused (w^H v = "
                f"{abs(np.vdot(rn.left[:, 0], rn.right[:, 0])):.1e})", True)

        # semisimple multiple: x0 = 0 with nu = eta, L = -eta k^2 on (phi, psi) and +-kx
        params = fresh_params(dims=2, nx=16, ny=8, Lx=2*np.pi, Ly=4*np.pi, diss=(0.02, 0.02),
                              hyper=1)
        kgrid = jr.setup_kgrids(params)
        z = np.zeros((2, 1, params.nx, params.ny//2 + 1), dtype=complex)
        B, _ = stability.ky_block_matrix(z, kgrid, params, 1)
        rz = stability.eig_dense(B)
        dB, _ = stability.dj_matrix(z, z, kgrid, params, iky=1)
        with pytest.raises(RuntimeError, match="not numerically simple"):
            stability.eigenvalue_sensitivity(B, dB, rz.values[0], rz.right[:, 0],
                                             rz.left[:, 0], spectrum=rz.values)
        c.check("semisimple multiple eigenvalue (x0 = 0, nu = eta) refused", True)
        # the ky-block tearing eigenvalue is REAL, so in real coordinates it is double (the
        # ky and -ky copies)
        params, kgrid, x0, dx0 = _shear_case()
        Bs, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        p2 = fresh_params(dims=2, nx=24, ny=8, Lx=16.0, Ly=4*np.pi, diss=(0.02, 0.02), hyper=1)
        k2 = jr.setup_kgrids(p2)
        psi = lambda x, y: 1/jnp.cosh(x - 8.0)**2 + 0*y
        xs = jr.initialize(lambda x, y: jnp.stack([0.8*psi(x, y), psi(x, y)]), p2).fields
        ds = jr.initialize(lambda x, y: jnp.stack([psi(x, y), 0*psi(x, y)]), p2).fields
        M, coords = stability.real_matrix(xs, k2, p2)
        rm = stability.eig_dense(M)
        dM, _ = stability.dj_matrix(xs, ds, k2, p2, coords=coords)
        c.check(f"real coords of a 1D x0: the leading eigenvalue is double ({rm.values[:2]})",
                abs(rm.values[0] - rm.values[1]) < 1e-10 and rm.values[0].real > 0)
        with pytest.raises(RuntimeError, match="not numerically simple"):
            stability.eigenvalue_sensitivity(M, dM, rm.values[0], rm.right[:, 0], rm.left[:, 0],
                                             spectrum=rm.values)
        c.check("the +-ky double eigenvalue of a 1D x0 in real coordinates refused (use the "
                "ky block)", True)
        # ... and a spectrum of lam alone (a k=1 solve's values) is not an implicit opt-out:
        # it would accept that double eigenvalue, which along a y-dependent dx0 splits at first
        # order (no derivative) -- the review measured a returned 0.22+0.06i against a +-0.21
        # split for a cos(y) direction
        with pytest.raises(ValueError, match="lam alone"):
            stability.eigenvalue_sensitivity(M, dM, rm.values[0], rm.right[:, 0], rm.left[:, 0],
                                             spectrum=rm.values[:1])
        c.check("spectrum = [lam] alone is a ValueError, never an implicit opt-out", True)

        # wrong or inaccurate vectors
        r = stability.eig_dense(Bs)
        dB, _ = stability.dj_matrix(x0, dx0, kgrid, params, iky=1)
        with pytest.raises(RuntimeError, match="not an eigentriple"):
            stability.eigenvalue_sensitivity(Bs, dB, r.values[0], r.right[:, 0], r.right[:, 0],
                                             spectrum=r.values)
        c.check("w = v (not the left vector of a non-normal J) refused", True)
        with pytest.raises(RuntimeError, match="not an eigentriple"):
            stability.eigenvalue_sensitivity(Bs, dB, r.values[0], r.right[:, 0], r.left[:, 1],
                                             spectrum=r.values)
        c.check("another eigenvalue's left vector refused", True)
        noisy = r.right[:, 0] + 1e-6*(rng.standard_normal(len(idx))
                                      + 1j*rng.standard_normal(len(idx)))
        with pytest.raises(RuntimeError, match="not an eigentriple"):
            stability.eigenvalue_sensitivity(Bs, dB, r.values[0], noisy, r.left[:, 0],
                                             spectrum=r.values)
        c.check("a propagator-accuracy (1e-6) right vector refused at the default residual_tol",
                True)
        with pytest.raises(ValueError, match="must contain lam"):
            stability.eigenvalue_sensitivity(Bs, dB, r.values[0], r.right[:, 0], r.left[:, 0],
                                             spectrum=r.values[1:])
        c.check("a spectrum without lam is a ValueError", True)

        # non-invariant directions
        bad = np.array(dx0)
        bad[1, 0, 2, 1] = 1e-3*np.abs(dx0).max()
        for fn in (stability.dj_matrix, stability.dj_operator):
            with pytest.raises(ValueError, match="ky-independent dx0"):
                fn(x0, bad, kgrid, params, iky=1)
        stability.dj_matrix(x0, bad, kgrid, params)          # real coords: any dx0
        c.check("a ky-dependent dx0 is refused in a ky block (accepted in real coords)", True)
        with pytest.raises(ValueError, match="ky-independent x0"):
            stability.dj_matrix(bad, dx0, kgrid, params, iky=1)
        c.check("a ky-dependent x0 is refused too", True)
        params = _cmhd_params()
        kgrid = jr.setup_kgrids(params)
        x0 = _const(params, [1.3, 0.1, 0.0, 0.0, 0.3, 0.4, 1.0])
        for label, pos in (("kx", (2, 0, 1, 0)), ("kz", (2, 3, 0, 0))):
            d = np.zeros_like(x0)
            d[pos] = 1e-2*np.abs(x0).max()
            with pytest.raises(ValueError, match=f"{label}-independent dx0"):
                stability.dj_matrix(x0, d, kgrid, params, iky=1, ikx=1, iz=2)
            c.check(f"a {label}-dependent dx0 is refused in a (kx, ky, kz) mode block", True)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
