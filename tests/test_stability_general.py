# taranis.stability rung 0b (plans/AUTODIFF_PLAN.md): the harness is equation- and
# geometry-agnostic, and its matrix-free eigensolvers reproduce EXACT answers.
#   1. FD gate for GDI (2D, 3D) and CMHD (rho / lnrho, gamma = 1 / 5/3): J v against the
#      central difference of the solver's N. GDI's N is quadratic, so the difference is exact
#      to round-off; CMHD's is not polynomial (1/rho, h(rho), e^s), so it converges at O(e^2).
#   2. CMHD about a UNIFORM state (rho0 != 1, oblique B0, a mean flow u0, all k = 0 modes of
#      the evolved fields; gamma = 1 and 5/3; both density variables). The waves live in N,
#      so this gates the jvp through the CMHD nonlinear term. Exact eigenvalues per
#      wavevector: -D|k|^2h - i k.u0 +- i omega_{fast, slow, Alfven} and the div B mode -D|k|^2h
#      (not Doppler shifted: induction is a curl). Paths: shift_invert (GMRES) on the
#      per-wavevector 7x7 block operator, to round-off; propagator_eigs on a whole ky column,
#      to the stepper's O(dt^4). Unequal (D_rho, nu, eta): against a test-local 7x7
#      transcription of the linearized equations.
#   3. GDI at x0 = 0: eigenvalues = roots of the dispersion quadratic (3.7)/(5.3) (derived
#      in tests/test_gdi_linear.py, re-derived here), minus diss k^2h, 2D and 3D z_spectral,
#      by shift_invert on the ky column and by propagator_eigs (exact here: N' = 0 and the IF
#      stepper applies exp(L dt) exactly).
#   4. RMHD: propagator_eigs against eig_dense on the ky block for the cos x and sech^2
#      tearing equilibria, leading modes agreeing to the O(dt^4) time error, which is shown
#      by halving dt (fitted order ~ 4); z_spectral 3D with a z- and y-dependent x0 through
#      real coordinates against the dense real matrix.
#   5. Preconditioners: diagonal_preconditioner is (L - sigma)^-1 exactly (x0 = 0: GMRES in
#      one iteration, putzer2 real coords included); ky_averaged_preconditioner is
#      (J - sigma)^-1 exactly for a ky-independent x0, and shift_invert returns the same
#      pairs with it.
#   6. Mode blocks assert x0's invariance; the expanding box is rejected.
# fp64 only (round-off tolerances) except test_smoke_both_precisions.
# pytest: `pytest tests/test_stability_general.py`. Script: `python tests/test_stability_general.py`.
from _rmhd_testing import bootstrap, checks, fit_order, fresh_params

bootstrap()

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import jax.numpy as jnp

import taranis as jr
from taranis import _precision, stability
from taranis.physics import construct_rhs, equation_registry
from taranis.types import SimulationState

_RNG_SEED = 20260928


def _match(a, b):
    # max |a_i - b_perm(i)| over the optimal pairing of a into b (len(a) <= len(b))
    cost = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].max())


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


def _nonlinear(params, kgrid):
    # N(fields): the solver's RHS terms, built here without taranis.stability
    rhs = construct_rhs(equation_registry[params.eqtype])
    nky = params.ny//2 + 1
    state = SimulationState(
        t=np.float64(0.0), fields=None,
        forcing_state=jnp.zeros((params.n_ou, 2, params.nx, nky), dtype=_precision.ctype),
        forcing_key=None, forcing_scale=jnp.zeros((params.n_ou,), dtype=_precision.ftype))
    return lambda f: np.asarray(rhs(state._replace(fields=jnp.asarray(f, dtype=_precision.ctype)),
                                    kgrid, params)[0])


# ------------------------------------------------------------------------------ CMHD

_CMHD_BOX = dict(dims=3, nx=8, ny=8, nz=8, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, z_spectral=True,
                 eqtype="CMHD", comm_backend="serial")
_RHO0, _U0, _B0 = 1.3, np.array([0.1, -0.05, 0.07]), np.array([0.3, 0.4, 1.0])


def _cmhd_params(cs0=0.8, gamma=1.0, density_var="rho", diss=0.05, hyper=2, **kw):
    eq = dict(cs0=cs0, diss=diss, hyper=hyper, gamma=gamma)
    if density_var != "rho":
        eq["density_var"] = density_var
    return fresh_params(eqpars=eq, **{**_CMHD_BOX, **kw})


def _uniform_x0(params):
    lnrho = params.eqpars.get("density_var", "rho") == "lnrho"
    f0 = np.log(_RHO0) if lnrho else _RHO0
    vals = [f0, *_U0, *_B0]
    return jr.initialize(lambda x, y, z: jnp.stack([v + 0*x + 0*y + 0*z for v in vals]),
                         params).fields


def _kvec(params, ix, iy, iz):
    n = lambda i, m: i - m if i > m//2 else i
    return np.array([2*np.pi*n(ix, params.nx)/params.Lx, 2*np.pi*iy/params.Ly,
                     2*np.pi*n(iz, params.nz)/params.Lz])


def _cmhd_exact(k, cs0, gamma, D, hyper):
    # docs/numerics.md "Compressible MHD" linear waves about (rho0, u0, B0): c^2 = cs0^2
    # rho0^(gamma-1), v_A = B0/sqrt(rho0); omega_f^2 omega_s^2 = k^2 c^2 (k.v_A)^2
    k2 = k @ k
    c2 = cs0*cs0*_RHO0**(gamma - 1.0)
    vA = _B0/np.sqrt(_RHO0)
    s = c2 + vA @ vA
    kva = k @ vA
    wf2 = 0.5*(k2*s + np.sqrt((k2*s)**2 - 4*k2*c2*kva**2))
    ws2 = k2*c2*kva**2/wf2
    damp = -D*k2**hyper
    dop = -1j*(k @ _U0)
    waves = [damp + dop + sg*1j*w for w in (np.sqrt(wf2), np.sqrt(ws2), abs(kva))
             for sg in (1, -1)]
    return np.array(waves + [damp])


def _cmhd_transcription(k, cs0, gamma, diss3, hyper, lnrho):
    # the linearized equations about (rho0, u0, B0), written out here (field order
    # (rho or s, u, B)): with dh = (c^2/rho0) drho = c^2 ds,
    #   d_t drho = -i k.u0 drho - i rho0 k.du         (d_t ds: rho0 -> 1)
    #   d_t du   = -i k.u0 du - i k dh + ((i k x dB) x B0)/rho0
    #   d_t dB   = i k x (du x B0 + u0 x dB)
    # plus -diag(D_rho, nu, nu, nu, eta, eta, eta) |k|^2h
    c2 = cs0*cs0*_RHO0**(gamma - 1.0)
    M = np.zeros((7, 7), dtype=complex)
    ku = k @ _U0
    for j in range(7):
        e = np.zeros(7)
        e[j] = 1.0
        r, du, dB = e[0], e[1:4], e[4:7]
        out = np.zeros(7, dtype=complex)
        out[0] = -1j*ku*r - 1j*(1.0 if lnrho else _RHO0)*(k @ du)
        dh = c2*r if lnrho else c2/_RHO0*r
        out[1:4] = -1j*ku*du - 1j*k*dh + np.cross(np.cross(1j*k, dB), _B0)/_RHO0
        out[4:7] = np.cross(1j*k, np.cross(du, _B0) + np.cross(_U0, dB))
        M[:, j] = out
    d = np.array([diss3[0]] + [diss3[1]]*3 + [diss3[2]]*3)
    return M - np.diag(d*(k @ k)**hyper)


_MODES = ((1, 1, 2), (7, 2, 1))          # (ikx, iky, iz): k = (1, 1, 2) and (-1, 2, 1)


def _cmhd_mode_eigs(c, params, kgrid, x0, mode, want, label):
    # shift_invert (GMRES) on the per-wavevector block operator, at sigma near the upper and
    # the lower fast wave; the union of the two 5-nearest sets is all 7 eigenvalues
    ikx, iky, iz = mode
    op, idx = stability.ky_block_operator(x0, kgrid, params, iky, ikx=ikx, iz=iz)
    c.check(f"{label}: the (kx, ky, kz) block is the 7 fields of one mode", len(idx) == 7,
            f"n={len(idx)}")
    fast = want[np.argsort(want.imag)]
    got = []
    for sigma in (fast[-1] + 0.05 + 0.02j, fast[0] + 0.05 - 0.02j):
        res = stability.shift_invert(op, sigma, k=5, tol=1e-13, restart=7)
        near = want[np.argsort(np.abs(want - sigma))[:5]]
        e = _match(res.values, near)/np.abs(want).max()
        c.check(f"{label}, sigma={sigma:.2f}: 5 nearest eigenvalues == exact (rel {e:.1e}, "
                f"residual {res.residuals.max():.1e}, {res.stats['gmres_iters']} GMRES iters)",
                e < 1e-11 and res.residuals.max() < 1e-10, f"rel {e:.3e}")
        got.extend(res.values)
    e = _match(want, np.array(got))/np.abs(want).max()
    c.check(f"{label}: all 7 exact eigenvalues found (rel {e:.1e})", e < 1e-11, f"rel {e:.3e}")


@pytest.mark.fp64
def test_cmhd_uniform_state_exact_waves():
    with checks() as c:
        for dv in ("rho", "lnrho"):
            for gamma in (1.0, 5.0/3.0):
                cs0, D, hyper = 0.8, 0.05, 2
                params = _cmhd_params(cs0=cs0, gamma=gamma, density_var=dv, diss=D, hyper=hyper)
                kgrid = jr.setup_kgrids(params)
                x0 = _uniform_x0(params)
                for mode in _MODES:
                    k = _kvec(params, *mode)
                    want = _cmhd_exact(k, cs0, gamma, D, hyper)
                    _cmhd_mode_eigs(c, params, kgrid, x0, mode, want,
                                    f"{dv} gamma={gamma:.3f} k={k.round(2)}")
        # unequal (D_rho, nu, eta): against the independent 7x7 transcription
        for dv in ("rho", "lnrho"):
            diss3 = (0.03, 0.08, 0.02)
            params = _cmhd_params(cs0=1.1, gamma=5.0/3.0, density_var=dv, diss=diss3, hyper=1)
            kgrid = jr.setup_kgrids(params)
            x0 = _uniform_x0(params)
            for mode in _MODES:
                k = _kvec(params, *mode)
                M = _cmhd_transcription(k, 1.1, 5.0/3.0, diss3, 1, dv == "lnrho")
                want = np.linalg.eigvals(M)
                _cmhd_mode_eigs(c, params, kgrid, x0, mode, want,
                                f"{dv} (D_rho, nu, eta)={diss3} k={k.round(2)}")
            # the transcription itself against the closed form at equal diss (so it is the
            # same physics, not a second guess)
            k = _kvec(params, *_MODES[0])
            M = _cmhd_transcription(k, 1.1, 5.0/3.0, (0.04,)*3, 1, dv == "lnrho")
            e = _match(np.linalg.eigvals(M), _cmhd_exact(k, 1.1, 5.0/3.0, 0.04, 1))
            c.check(f"{dv}: 7x7 transcription at equal diss == closed form (abs {e:.1e})",
                    e < 1e-12, f"{e:.3e}")


@pytest.mark.fp64
def test_cmhd_propagator_ky_column():
    # the fastest modes of a whole ky column (every kx, kz at iky = 1) by propagator Arnoldi:
    # the smallest |k| shell, k = (0, 1, 0), leads (hyper = 2 separates the shells)
    with checks() as c:
        for dv, gamma in (("rho", 5.0/3.0), ("lnrho", 1.0)):
            cs0, D, hyper = 0.8, 0.2, 2
            params = _cmhd_params(cs0=cs0, gamma=gamma, density_var=dv, diss=D, hyper=hyper)
            kgrid = jr.setup_kgrids(params)
            x0 = _uniform_x0(params)
            idx = stability.ky_block_index(kgrid, params, 1)
            modes = sorted({(int(ix), int(iz)) for _, iz, ix in idx})
            allw = np.concatenate([_cmhd_exact(_kvec(params, ix, 1, iz), cs0, gamma, D, hyper)
                                   for ix, iz in modes])
            want = _cmhd_exact(_kvec(params, 0, 1, 0), cs0, gamma, D, hyper)
            errs = []
            for dt in (0.1, 0.05):
                r = stability.propagator_eigs(x0, kgrid, params, T=0.5, dt=dt, k=7, iky=1)
                e = _match(r.values, want)
                errs.append(e)
                c.check(f"{dv} gamma={gamma:.3f} dt={dt}: leading 7 of the ky column == the "
                        f"exact |k| = 1 waves (abs {e:.1e}; {r.applications} applications x "
                        f"{r.info.nsteps} steps, {r.seconds:.1f}s)", e < 1e-5, f"{e:.3e}")
                nxt = np.sort(allw.real)[::-1][7]
                c.check(f"{dv} dt={dt}: nothing faster was missed (next exact Re "
                        f"{nxt:.3f} < {r.values.real.min():.3f})",
                        nxt < r.values.real.min() - 1e-3)
                c.check(f"{dv} dt={dt}: no branch aliasing (max |Im lambda| T = "
                        f"{np.abs(r.values.imag).max()*r.info.T:.2f})", not r.aliased.any())
            order = fit_order([0.1, 0.05], errs)
            c.check(f"{dv}: lambda error converges at the stepper's order 4 (observed "
                    f"{order:.2f})", 3.5 < order < 4.6, f"order {order:.3f}")


# ------------------------------------------------------------------------------- GDI

def _gdi_params(dims, **eq):
    base = dict(Ln=1.0, nu_in=0.3, v0=2.0, diss=0.01, hyper=1)
    base.update(eq)
    if dims == 2:
        return fresh_params(dims=2, nx=16, ny=16, Lx=2*np.pi, Ly=2*np.pi, eqtype="GDI",
                            comm_backend="serial", eqpars=dict(gpar_fac=0.1, **base))
    return fresh_params(dims=3, nx=12, ny=12, nz=6, Lx=2*np.pi, Ly=2*np.pi, Lz=4*np.pi,
                        z_spectral=True, eqtype="GDI", comm_backend="serial",
                        eqpars=dict(D_par=0.5, gpar_fac=0.0, **base))


def _gdi_exact(kx, ky, kz, ep):
    # eq (3.7): ksq w^2 + i (nu_in ksq + g (1 + ksq)) w - g nu_in ksq + w_ped w_* +
    # i g (w_* - w_ped) = 0 with g = gpar_fac nu_in ksq + D_par kz^2, lambda = -i w - diss k^2h
    ksq = kx*kx + ky*ky
    nu, g = ep["nu_in"], ep.get("gpar_fac", 0.0)*ep["nu_in"]*ksq + ep.get("D_par", 0.0)*kz*kz
    ws, wp = ky/ep["Ln"], ky*nu*ep["v0"]
    roots = np.roots([ksq, 1j*(nu*ksq + g*(1 + ksq)), -g*nu*ksq + wp*ws + 1j*g*(ws - wp)])
    return -1j*roots - ep["diss"]*ksq**ep["hyper"]


def _gdi_column_exact(params, kgrid, iky):
    idx = stability.ky_block_index(kgrid, params, iky)
    out = []
    for iz, ix in sorted({(int(iz), int(ix)) for _, iz, ix in idx}):
        kx = np.fft.fftfreq(params.nx)[ix]*2*np.pi*params.nx/params.Lx
        ky = iky*2*np.pi/params.Ly
        kz = (np.fft.fftfreq(params.nz)[iz]*2*np.pi*params.nz/params.Lz
              if params.spatial_dimensions == 3 else 0.0)
        out.extend(_gdi_exact(kx, ky, kz, params.eqpars))
    return np.array(out), len(idx)


@pytest.mark.fp64
def test_gdi_exact_dispersion():
    with checks() as c:
        for dims in (2, 3):
            params = _gdi_params(dims)
            kgrid = jr.setup_kgrids(params)
            x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
            for iky in (1, 2):
                want, n = _gdi_column_exact(params, kgrid, iky)
                c.check(f"GDI {dims}D iky={iky}: 2 exact roots per kept mode ({n})",
                        len(want) == n)
                # one step: at N' = 0 the IF step IS exp(L T), so no time error to wait out
                r = stability.propagator_eigs(x0, kgrid, params, T=2.0, dt=2.0, k=4, iky=iky)
                top = want[np.argsort(-want.real)[:4]]
                e = _match(r.values, top)/np.abs(top).max()
                c.check(f"GDI {dims}D iky={iky}: propagator's 4 fastest == the 4 fastest exact "
                        f"roots (rel {e:.1e}, gamma_max {top.real.max():.4f})", e < 1e-11,
                        f"rel {e:.3e}")
                op, _ = stability.ky_block_operator(x0, kgrid, params, iky)
                sigma = top[0] + 0.03 + 0.02j
                res = stability.shift_invert(op, sigma, k=4, tol=1e-13, restart=n)
                near = want[np.argsort(np.abs(want - sigma))[:4]]
                e = _match(res.values, near)/np.abs(near).max()
                c.check(f"GDI {dims}D iky={iky}: shift_invert's 4 nearest == exact (rel "
                        f"{e:.1e})", e < 1e-11, f"rel {e:.3e}")
            c.check(f"GDI {dims}D: the configuration is unstable (gamma_max "
                    f"{want.real.max():.4f})", want.real.max() > 0)


# ------------------------------------------------------------------------------ FD gate

def _cmhd_wavy(lnrho):
    def ic(x, y, z):
        rho = 1.0 + 0.2*jnp.cos(x + y) + 0.1*jnp.sin(z - x)
        f0 = jnp.log(rho) if lnrho else rho
        o = 0*x + 0*y + 0*z
        return jnp.stack([f0 + o, 0.3*jnp.sin(y + z) + o, 0.2*jnp.cos(x) + o,
                          0.1*jnp.sin(x - z) + o, 0.2*jnp.cos(z + y) + 0.3 + o,
                          0.1*jnp.sin(x + z) + o, 1.0 + 0.1*jnp.cos(y) + o])
    return ic


@pytest.mark.fp64
def test_fd_gate_all_recipes():
    rng = np.random.default_rng(_RNG_SEED)
    cases = [("GDI 2D", _gdi_params(2), lambda x, y: jnp.stack([0.3*jnp.cos(x + y), jnp.sin(x)*jnp.cos(2*y)]), True),
             ("GDI 3D", _gdi_params(3), lambda x, y, z: jnp.stack([0.3*jnp.cos(x + y + z), jnp.sin(x)*jnp.cos(y - z)]), True)]
    for dv in ("rho", "lnrho"):
        for gamma in (1.0, 5.0/3.0):
            cases.append((f"CMHD {dv} gamma={gamma:.3f}",
                          _cmhd_params(gamma=gamma, density_var=dv), _cmhd_wavy(dv == "lnrho"),
                          False))
    with checks() as c:
        for name, params, ic, quadratic in cases:
            kgrid = jr.setup_kgrids(params)
            x0 = np.asarray(jr.initialize(ic, params).fields)
            J = stability.jvp_operator(x0, kgrid, params)
            N = _nonlinear(params, kgrid)
            Lv = lambda v: np.asarray(kgrid.lin.apply_L(jnp.asarray(v)))
            # a random REAL dealiased perturbation, scaled to 10% of x0
            coords = stability.real_coords(kgrid, params)
            v = np.asarray(coords.unpack(jnp.asarray(rng.standard_normal(coords.n))))
            v = 0.1*v*np.abs(x0).max()/np.abs(v).max()
            Jv = np.asarray(J(v))
            eps = (1e-1, 3e-2, 1e-2) if not quadratic else (1e-2, 1e-3)
            errs = [_relerr(Lv(v) + (N(x0 + e*v) - N(x0 - e*v))/(2*e), Jv) for e in eps]
            if quadratic:
                c.check(f"{name}: central FD is exact for a quadratic N (rel {max(errs):.1e})",
                        max(errs) < 1e-11, f"{errs}")
            else:
                order = fit_order(eps, errs)
                c.check(f"{name}: central FD converges to J v at O(eps^2) (errs "
                        f"{errs[0]:.1e} -> {errs[-1]:.1e}, order {order:.2f})",
                        1.8 < order < 2.2 and errs[-1] < 3e-5, f"order {order:.3f}, {errs}")


# ----------------------------------------------------------------------------- RMHD

def _rmhd2d(nx, Lx, k, diss):
    return fresh_params(dims=2, nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/k, diss=diss, hyper=1)


@pytest.mark.fp64
def test_rmhd_tearing_propagator_vs_dense():
    cases = (("cos x, phi0 = 0.3 psi0", _rmhd2d(32, 2*np.pi, 0.5, (0.01, 0.01)),
              lambda x, y: jnp.stack([0.3*jnp.cos(x) + 0*y, jnp.cos(x) + 0*y]), 4.0, 4),
             ("sech^2 x", _rmhd2d(48, 16.0, 0.5, (0.02, 0.02)),
              lambda x, y: jnp.stack([0*x + 0*y, 1/jnp.cosh(x - 8.0)**2 + 0*y]), 8.0, 3))
    with checks() as c:
        for name, params, ic, T, k in cases:
            kgrid = jr.setup_kgrids(params)
            x0 = jr.initialize(ic, params).fields
            B, _ = stability.ky_block_matrix(x0, kgrid, params, 1)
            dense = stability.eig_dense(B)
            mu = np.exp(dense.values*T)
            top = dense.values[np.argsort(-np.abs(mu))[:k]]
            c.check(f"{name}: tearing-unstable (gamma = {dense.values[0].real:.5f})",
                    dense.values[0].real > 0)
            dts, errs = (0.5, 0.25, 0.125), []
            for dt in dts:
                r = stability.propagator_eigs(x0, kgrid, params, T=T, dt=dt, k=k, iky=1)
                e = _match(r.values, top)
                errs.append(e)
                c.check(f"{name} dt={dt}: leading {k} == eig_dense (abs {e:.1e}, residual on J "
                        f"{r.residuals.max():.1e}; {r.applications} x {r.info.nsteps} steps, "
                        f"{r.seconds:.1f}s)", e < 1e-6 and not r.aliased.any(), f"{e:.3e}")
                rq = _match(r.rayleigh, top)
                c.check(f"{name} dt={dt}: Rayleigh quotients agree too ({rq:.1e})", rq < 1e-5)
            order = fit_order(dts, errs)
            c.check(f"{name}: error O(dt^4) (errs {errs[0]:.1e}, {errs[1]:.1e}, {errs[2]:.1e}; "
                    f"order {order:.2f})", 3.6 < order < 4.4, f"order {order:.3f}")


@pytest.mark.fp64
def test_rmhd_zspectral_3d_real_coords_propagator():
    # a y- and z-dependent x0: no block structure, real coordinates. The two k = 0 means are
    # an exactly DOUBLE zero eigenvalue, of which a single-vector Krylov space sees ONE copy:
    # k = 3 asks for the two tearing modes and one zero.
    params = fresh_params(dims=3, nx=12, ny=8, nz=6, Lx=2*np.pi, Ly=4*np.pi, Lz=2*np.pi,
                          z_spectral=True, diss=(0.02, 0.02), hyper=1, eqpars=dict(z_diss_k=0.3))
    kgrid = jr.setup_kgrids(params)
    x0 = jr.initialize(lambda x, y, z: jnp.stack(
        [0.3*jnp.cos(x)*(1 + 0.2*jnp.sin(z)) + 0.1*jnp.sin(x + 0.5*y),
         jnp.cos(x)*(1 + 0.3*jnp.sin(z)) + 0.1*jnp.cos(0.5*y + z)]), params).fields
    with checks() as c:
        M, coords = stability.real_matrix(x0, kgrid, params)
        ev = np.linalg.eigvals(M)
        top = ev[np.argsort(-ev.real)[:3]]
        c.check(f"dense: two unstable modes, then the double zero ({top})",
                top[1].real > 0.05 and abs(top[2]) < 1e-12)
        r = stability.propagator_eigs(x0, kgrid, params, T=2.0, dt=0.1, k=3)
        e = _match(r.values, top)
        c.check(f"propagator Arnoldi, real coords (n={coords.n}): leading 3 == dense (abs "
                f"{e:.1e}, residual {r.residuals.max():.1e}; {r.applications} x "
                f"{r.info.nsteps} steps, {r.seconds:.1f}s)", e < 1e-8, f"{e:.3e}")


# ---------------------------------------------------------------------- preconditioners

@pytest.mark.fp64
def test_preconditioners():
    rng = np.random.default_rng(_RNG_SEED)
    with checks() as c:
        # (L - sigma)^-1 exactly: x0 = 0, where J = L. putzer2 (nu != eta, z_spectral) has
        # a complex L, so the real-coordinate f+/f- split is exercised
        params = fresh_params(dims=3, nx=8, ny=8, nz=6, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi,
                              z_spectral=True, diss=(0.03, 0.01), hyper=1)
        kgrid = jr.setup_kgrids(params)
        x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
        sigma = -0.05 + 0.7j
        for space, kw in (("real coords", {}), ("ky column", dict(iky=1))):
            P = stability.diagonal_preconditioner(kgrid, params, **kw)(sigma)
            if kw:
                op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
            else:
                op, _ = stability.real_operator(x0, kgrid, params)
            n = op.shape[0]
            z = rng.standard_normal(n) + 1j*rng.standard_normal(n)
            mv = stability._complexify(op)
            e = _relerr(P.matvec(mv(z) - sigma*z), z)
            c.check(f"diagonal preconditioner, {space}: P (J - sigma) z == z at x0 = 0 (rel "
                    f"{e:.1e})", e < 1e-13, f"{e:.3e}")
            res = stability.shift_invert(op, sigma, k=2, restart=n,
                                         M=stability.diagonal_preconditioner(kgrid, params, **kw))
            per = res.stats["gmres_iters"]/res.stats["solves"]
            c.check(f"diagonal preconditioner, {space}: one GMRES iteration per solve "
                    f"({per:.2f})", per <= 1.0 + 1e-12, f"{per}")

        # (J(xbar) - sigma)^-1: exact for a ky-independent x0, the nu = 0 cos x sheet
        params = fresh_params(dims=2, nx=32, ny=12, Lx=2*np.pi, Ly=4*np.pi, diss=(0.0, 0.01),
                              hyper=1)
        kgrid = jr.setup_kgrids(params)
        x0 = jr.initialize(lambda x, y: jnp.stack([0*x + 0*y, jnp.cos(x) + 0*y]), params).fields
        op, coords = stability.real_operator(x0, kgrid, params)
        pre = stability.ky_averaged_preconditioner(x0, kgrid, params, coords)
        sigma = 0.02 + 0.01j
        z = rng.standard_normal(coords.n) + 1j*rng.standard_normal(coords.n)
        e = _relerr(pre(sigma).matvec(stability._complexify(op)(z) - sigma*z), z)
        c.check(f"ky-averaged preconditioner: exact inverse for a 1D x0 (rel {e:.1e})",
                e < 1e-12, f"{e:.3e}")
        M, _ = stability.real_matrix(x0, kgrid, params, coords)
        ev = np.linalg.eigvals(M)
        lead = ev[np.argmax(ev.real)]
        sig = lead + 1e-3
        res = stability.shift_invert(op, sig, k=2, restart=coords.n, M=pre)
        want = ev[np.argsort(np.abs(ev - sig))[:2]]
        e = _match(res.values, want)/abs(lead)
        per = res.stats["gmres_iters"]/res.stats["solves"]
        c.check(f"shift_invert + ky-averaged preconditioner: same 2 nearest eigenvalues (rel "
                f"{e:.1e}), {per:.2f} GMRES iterations per solve", e < 1e-9 and per <= 2.0,
                f"rel {e:.3e}, per {per}")


# ----------------------------------------------------------------------- guards / smoke

def test_block_guards_and_ebm():
    with checks() as c:
        params = _cmhd_params()
        kgrid = jr.setup_kgrids(params)
        x0 = np.asarray(_uniform_x0(params)).copy()
        stability.ky_block_index(kgrid, params, 1, ikx=1, iz=2)
        bump = 1e-2*np.abs(x0).max()     # well above the 1e3 eps round-off allowance
        x0[1, 0, 1, 0] = bump            # kx != 0 content at ky = 0, kz = 0
        with pytest.raises(ValueError, match="kx-independent"):
            stability.ky_block_matrix(x0, kgrid, params, 1, ikx=1, iz=2)
        stability.ky_block_matrix(x0, kgrid, params, 1, iz=2)    # a ky column: still valid
        c.check("a fixed-ikx block rejects an x-dependent x0; a ky column accepts it", True)
        x0[1, 0, 1, 0] = 0.0
        x0[1, 2, 0, 0] = bump            # kz != 0 content
        with pytest.raises(ValueError, match="kz-independent"):
            stability.ky_block_matrix(x0, kgrid, params, 1, ikx=1, iz=2)
        c.check("a fixed-iz block rejects a z-dependent x0", True)
        p2 = fresh_params(dims=3, nx=8, ny=8, nz=4, comm_backend="serial")
        with pytest.raises(ValueError, match="z_spectral"):
            stability.ky_block_index(jr.setup_kgrids(p2), p2, 1, iz=0)
        c.check("a fixed iz is rejected under finite-difference z", True)
        ebm = fresh_params(eqpars=dict(cs0=1.0, diss=0.01, hyper=1,
                                       expansion=dict(adot=0.1)), **_CMHD_BOX)
        with pytest.raises(ValueError, match="expanding box"):
            stability.jvp_operator(np.zeros((7, 8, 8, 5), dtype=complex), jr.setup_kgrids(ebm),
                                   ebm)
        c.check("the CMHD expanding box is rejected (time-dependent RHS)", True)


def test_smoke_both_precisions():
    # any precision: GDI at x0 = 0 is L, whose exact propagator the IF stepper applies, and a
    # CMHD uniform-state mode block runs through propagator_eigs' dense branch
    tol = 1e-10 if _precision.precision == "64" else 2e-4
    with checks() as c:
        params = _gdi_params(2)
        kgrid = jr.setup_kgrids(params)
        x0 = jnp.zeros((2, 1, params.nx, params.ny//2 + 1), dtype=_precision.ctype)
        want, _ = _gdi_column_exact(params, kgrid, 1)
        r = stability.propagator_eigs(x0, kgrid, params, T=2.0, dt=0.2, k=2, iky=1)
        top = want[np.argsort(-want.real)[:2]]
        e = _match(r.values, top)/np.abs(top).max()
        c.check(f"GDI 2D propagator: 2 fastest exact roots (rel {e:.1e})", e < tol, f"{e:.3e}")
        params = _cmhd_params()
        kgrid = jr.setup_kgrids(params)
        k = _kvec(params, 1, 1, 2)
        r = stability.propagator_eigs(_uniform_x0(params), kgrid, params, T=0.3, dt=0.02, k=7,
                                      iky=1, ikx=1, iz=2)
        want = _cmhd_exact(k, 0.8, 1.0, 0.05, 2)
        e = _match(r.values, want)/np.abs(want).max()
        c.check(f"CMHD uniform mode block via exp(J T) on 7 basis vectors (rel {e:.1e}, "
                f"{r.applications} applications)", e < (1e-6 if tol < 1e-6 else 1e-3) and
                r.applications == 7, f"{e:.3e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
