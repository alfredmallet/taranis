# FLR-MHD rung F1 (plans/FLRMHD_PLAN.md section 5), the operator-level gates:
#   F1.2  the k-local FLR factors: 1 - Gamma_0 against an independent quadrature and an
#         independently built rational series, continuity across the branch switch, the two
#         limits with their COEFFICIENTS, v_ph against the paper's closed form at three
#         Z/tau, and the rho_i = 0 branch bitwise.
#   F1.9  set_timestep against the closed-form CFL bound with an independent v_ph,max, the
#         dt <= _quiescent_dt property, and the discriminator: a kinetic-Alfven-dominated
#         state that is stable at cfl_safety blows up when v_ph,max is dropped from the CFL.
#   F1.10 the stability harness on the recipe: J(x0 = 0) == L entrywise, block eigenvalues
#         d +- i kz v_ph, the central-FD gate, the overrides independent route (J at
#         overrides == J of a Parameters built with those values) and dJ/dp against central
#         FD for every live key -- dJ/d rho_i INCLUDING the kz = 0 modes, where s^2 = 0 at
#         nu_phi == nu_psi and putzer2_precompute's sqrt has an infinite tangent, with
#         finiteness asserted explicitly.
#   F1.11 plumbing: the eqpars round trip, the RMHD -> FLRMHD cross-restart block, a bitwise
#         unforced restart, every rejection, the dead-z_hyper warning, reality over a run,
#         lsrk_scan and hoist_propagator on/off.
# The references (Gamma_0, the factors, v_ph) come from tests/_flrmhd_testing.py, which is
# numpy-only and never imports the module under test.
#
# Both precision sessions run this file; the fp32 tolerances are the measured floors (the
# factors are computed in float64 and CAST, so they carry one float32 rounding; fp64-only
# gates are marked).
# pytest: `pytest tests/test_flrmhd_operators.py`. Script: `python tests/test_flrmhd_operators.py`
# -- never under mpirun (dims=2 and z_spectral are single-process only).
from _rmhd_testing import (bootstrap, checks, fit_order, managed_manager,
                           snap_dir)

bootstrap()

import warnings
from fractions import Fraction
from math import factorial

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, overrides, stability
from taranis import snapshot_io as sn
from taranis.grids import fft, ifft
from taranis.physics import construct_rhs, equation_registry, flrmhd, rmhd
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

from _flrmhd_testing import broadband_ic, flr_ref, kgrid_ksq, one_minus_gamma0_ref

FP64 = _precision.precision == "64"
TWO_PI = 2.0*np.pi
EPS = np.finfo(float).eps

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _flr(rho_i=0.3, tau=0.5, Z=1.0, diss=(0.0, 0.0), hyper=1, eq=None, **kw):
    # a fresh FLRMHD Parameters; eq adds/overrides eqpars
    eqpars = dict(rho_i=rho_i, tau=tau, Z=Z, diss=diss, hyper=hyper)
    eqpars.update(eq or {})
    base = dict(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, dt=0.01,
                adaptive_timestep=False, dims=2, eqtype="FLRMHD", eqpars=eqpars)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(**base)


def _flr3(**kw):
    base = dict(dims=3, nz=8, Lz=TWO_PI, z_spectral=True)
    base.update(kw)
    return _flr(**base)


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


def _raises(exc, fn):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    except Exception:                      # the wrong exception type is a failure
        return ""
    return ""


# ---------------------------------------------------------------------------- gate F1.2

def _series_ref(alpha, nterms=12):
    """1 - Gamma_0 from the Cauchy product of the two defining power series,
    I_0(a) = sum_m (a/2)^(2m)/(m!)^2 and exp(-a) = sum_j (-a)^j/j!, in exact rationals --
    built here, not from the module's closed-form coefficient."""
    i0 = [Fraction(0)]*(nterms + 1)
    for m in range(nterms//2 + 1):
        i0[2*m] = Fraction(1, 4**m*factorial(m)**2)
    ex = [Fraction((-1)**j, factorial(j)) for j in range(nterms + 1)]
    g0 = [sum(i0[i]*ex[n - i] for i in range(n + 1)) for n in range(nterms + 1)]
    out = np.zeros_like(alpha)
    for n in range(nterms, 0, -1):
        out = (out - float(g0[n]))*alpha         # -(g0[1] a + g0[2] a^2 + ...), Horner
    return out


def test_one_minus_gamma0():
    """1 - Gamma_0(alpha) to 1e-13 against the quadrature on [1e-14, 1e3] and against the
    rational series on [1e-14, 1e-2]; the two branches agree to 4 eps at the switch."""
    with checks() as c:
        sw = flrmhd._ALPHA_SWITCH
        alpha = np.concatenate([np.logspace(-14, 3, 700),
                                sw*(1 + np.array([-4, -1, 0, 1, 4])*EPS),
                                [1e-3, 0.5, 2.0, 50.0, 700.0, 1e3]])
        got = np.asarray(flrmhd.one_minus_gamma0(alpha))
        c.check("one_minus_gamma0 returns float64 at every TARANIS_PRECISION",
                got.dtype == np.float64, f"{got.dtype}")
        ref = one_minus_gamma0_ref(alpha)
        e = np.max(np.abs(got/ref - 1))
        c.check(f"1 - Gamma_0 == quadrature on [1e-14, 1e3] (max rel {e:.2e})", e <= 1e-13,
                f"max rel {e:.3e} at alpha = {alpha[np.argmax(np.abs(got/ref - 1))]:.3e}")
        small = np.logspace(-14, -2, 300)
        e = np.max(np.abs(np.asarray(flrmhd.one_minus_gamma0(small))/_series_ref(small) - 1))
        c.check(f"1 - Gamma_0 == rational series on [1e-14, 1e-2] (max rel {e:.2e})",
                e <= 1e-13, f"max rel {e:.3e}")
        e = np.max(np.abs(_series_ref(small)/one_minus_gamma0_ref(small) - 1))
        c.check(f"the two references agree with each other (max rel {e:.2e})", e <= 1e-13)

        # the switch: both branch functions at the switch point, and the composed function
        # one float either side of it
        a = jnp.asarray(sw, dtype=jnp.float64)
        s_ser, s_dir = float(flrmhd._S_series(a)), float(flrmhd._S_direct(a))
        jump = abs(s_ser/s_dir - 1)
        c.check(f"series and closed-form branches agree at the switch alpha = {sw} "
                f"({jump/EPS:.2f} eps)", jump <= 4*EPS, f"{jump/EPS:.2f} eps")
        lo, hi = np.nextafter(sw, 0.0), sw
        f_lo, f_hi = (float(flrmhd.S_of_alpha(x)) for x in (lo, hi))
        step = abs(f_lo/f_hi - 1)
        c.check(f"S(alpha) is continuous across the switch ({step/EPS:.2f} eps between "
                f"neighbouring floats)", step <= 4*EPS, f"{step/EPS:.2f} eps")
        c.check("the branches are selected as documented (series strictly below the switch)",
                f_lo == float(flrmhd._S_series(jnp.asarray(lo)))
                and f_hi == float(flrmhd._S_direct(jnp.asarray(hi))))

        # the limits, with their coefficients: 1 - S -> (3/4) alpha, Gamma_0 -> (2 pi alpha)^-1/2
        al = np.logspace(-6, -2, 40)
        coef = (1 - np.asarray(flrmhd.S_of_alpha(al)))/al
        c.check(f"1 - G/ksq -> (3/4) alpha: coefficient in 0.75(1 +- 0.1) "
                f"([{coef.min():.4f}, {coef.max():.4f}])",
                coef.min() >= 0.75*0.9 and coef.max() <= 0.75*1.1)
        big = 1e3
        dev = 1 - float(flrmhd.one_minus_gamma0(big))            # = Gamma_0 = 1 - G rho_i^2/2
        bound = 1/np.sqrt(2*np.pi*big)
        c.check(f"G -> 2/rho_i^2 at alpha = 1e3: 1 - G rho_i^2/2 = {dev:.5e} within "
                f"(2 pi alpha)^-1/2 = {bound:.5e} (+-10%)", 0.9*bound <= dev <= 1.1*bound)

        # tangents exist and are finite on both sides (the overrides seam differentiates here)
        for x in (0.0, 1e-9, 0.3, np.nextafter(sw, 0.0), sw, 30.0, 1e3):
            d1 = float(jax.grad(lambda t: flrmhd.one_minus_gamma0(t))(jnp.float64(x)))
            d2 = float(jax.grad(jax.grad(lambda t: flrmhd.one_minus_gamma0(t)))(jnp.float64(x)))
            h = 1e-6*max(x, 1e-3)
            fd = (one_minus_gamma0_ref(np.array([x + h])) -
                  one_minus_gamma0_ref(np.array([max(x - h, 0.0)])))[0]/(x + h - max(x - h, 0.0))
            c.check(f"d(1 - Gamma_0)/d alpha finite and right at alpha = {x:.3g} "
                    f"({d1:.6f} vs FD {fd:.6f}), second derivative finite",
                    np.isfinite(d1) and np.isfinite(d2) and abs(d1 - fd) < 1e-5,
                    f"d1 {d1!r}, d2 {d2!r}, fd {fd!r}")


def test_factors_against_the_paper_form():
    """kgrid.eq against the factors written from the paper's definitions with the quadrature
    Gamma_0: every entry, three Z/tau, k rho_i from 0.05 to 14."""
    tol = 1e-13 if FP64 else 5e-7
    with checks() as c:
        for Z, tau in ((1.0, 0.5), (1.0, 1.0), (2.0, 4.0), (3.0, 0.7)):
            for rho_i in (0.05, 0.3, 1.25):
                params = _flr(rho_i=rho_i, tau=tau, Z=Z, Lx=TWO_PI*1.3, Ly=TWO_PI*0.7)
                kgrid = jr.setup_kgrids(params)
                ref = flr_ref(kgrid_ksq(params), rho_i, Z/tau)
                worst = {}
                for name in flrmhd.FLRFactors._fields:
                    got = np.asarray(getattr(kgrid.eq, name))
                    worst[name] = float(np.max(np.abs(got - ref[name])/np.maximum(
                        np.abs(ref[name]), 1e-300)))
                bad = {k: f"{v:.1e}" for k, v in worst.items() if not v <= tol}
                c.check(f"Z/tau = {Z/tau:.3g}, rho_i = {rho_i}: G, inv_G, pe, a, kappa, vph "
                        f"== paper form (worst rel {max(worst.values()):.1e})", not bad,
                        f"{bad}")
                dt = {np.asarray(v).dtype for v in kgrid.eq}
                c.check(f"Z/tau = {Z/tau:.3g}, rho_i = {rho_i}: factors are (nkx, nky) real "
                        f"at field precision",
                        dt == {np.dtype(_precision.ftype)}
                        and all(v.shape == (params.nx, params.ny//2 + 1) for v in kgrid.eq),
                        f"{dt}")
        # the plan's reference values at Z/tau = 2 (docs/numerics.md)
        f = flrmhd._factors_f64(np.array([0.01, 1.0, 100.0]), np.array([100.0, 1.0, 0.01]),
                                1.0, 2.0)
        vph = np.asarray(f.vph)
        c.check(f"v_ph(k rho_i = 0.1, 1, 10) at Z/tau = 2 is (1.007, 1.552, 12.37) "
                f"(got {vph[0]:.4f}, {vph[1]:.4f}, {vph[2]:.3f})",
                np.all(np.abs(vph/np.array([1.007, 1.552, 12.37]) - 1) < 5e-4))
        # the internal identities, on the module's own float64 values
        ksq = np.logspace(-6, 4, 200)
        f = flrmhd._factors_f64(ksq, 1.0/ksq, 0.7, 2.0)
        G, iG, pe, a, kap, vph = (np.asarray(v) for v in f)
        ident = max(np.max(np.abs(G*iG - 1)), np.max(np.abs(a*G/ksq - 1)),
                    np.max(np.abs(vph**2/(a*pe) - 1)), np.max(np.abs(kap**2*ksq/(G*pe) - 1)),
                    np.max(np.abs(kap*a/vph - 1)))
        c.check(f"G inv_G = 1, a = ksq/G, v_ph^2 = a pe, kappa^2 = G pe/ksq, v_ph = kappa a "
                f"(worst {ident:.1e})", ident < 1e-14)


def test_rho_i_zero_is_the_rmhd_branch():
    """rho_i = 0 returns the kgrid's own ksq / inv_ksq and exact ones -- bitwise."""
    with checks() as c:
        for label, params in (("2D", _flr(rho_i=0.0)), ("z_spectral", _flr3(rho_i=0.0))):
            kgrid = jr.setup_kgrids(params)
            eq = kgrid.eq
            c.check(f"{label}: kgrid.eq is an FLRFactors with the documented field order",
                    isinstance(eq, flrmhd.FLRFactors)
                    and eq._fields == ("G", "inv_G", "pe", "a", "kappa", "vph"))
            c.check(f"{label}: G is ksq and inv_G is inv_ksq, bitwise",
                    np.array_equal(np.asarray(eq.G), np.asarray(kgrid.ksq))
                    and np.array_equal(np.asarray(eq.inv_G), np.asarray(kgrid.inv_ksq)))
            ones = all(np.array_equal(np.asarray(v), np.ones(v.shape)) for v in
                       (eq.pe, eq.a, eq.kappa, eq.vph))
            c.check(f"{label}: pe, a, kappa, vph are exactly 1", ones)
            c.check(f"{label}: _vph_max is exactly 1.0", flrmhd._vph_max(params) == 1.0)
        for eqtype, eqp in (("RMHD", dict(diss=0.01, hyper=1)),
                            ("GDI", dict(Ln=392.0, nu_in=0.0106, v0=25.0, diss=5e-6, hyper=2))):
            p = jr.Parameters(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5,
                              eqtype=eqtype, eqpars=eqp)
            c.check(f"{eqtype}: the kgrid's eq slot stays None",
                    jr.setup_kgrids(p).eq is None)


# ---------------------------------------------------------------------------- gate F1.9

def _dealias_perp(params):
    ix = np.fft.fftfreq(params.nx)*params.nx
    iy = np.fft.rfftfreq(params.ny)*params.ny
    return (ix[:, None]/(params.nx/3.0))**2 + (iy[None, :]/(params.ny/3.0))**2 < 1.0


def _vph_max_ref(params, rho_i, z_over_tau):
    return float(np.max(flr_ref(kgrid_ksq(params), rho_i, z_over_tau)["vph"][
        _dealias_perp(params)]))


def _real(fk, params):
    # (nz, nkx, nky) coefficients -> real space, numpy (z_spectral: the z axis is kz)
    if params.z_spectral:
        return np.fft.irfftn(fk, s=(params.nz, params.nx, params.ny), axes=(0, 1, 2))
    return np.fft.irfft2(fk, s=(params.nx, params.ny), axes=(1, 2))


def _cfl_rates_ref(fields, params, pe, vmax):
    """(rate_x, rate_y) = (max(|d_y phi_e| + vmax |d_y psi|)/dx, max(|d_x phi_e| + vmax
    |d_x psi|)/dy), numpy: the two directional CFL rates, whose max sets dt."""
    phi, psi = np.asarray(fields[0]), np.asarray(fields[1])
    kx = np.fft.fftfreq(params.nx)*params.nx*TWO_PI/params.Lx
    ky = np.fft.rfftfreq(params.ny)*params.ny*TWO_PI/params.Ly
    dx_ = lambda f: _real(1j*kx[:, None]*f, params)
    dy_ = lambda f: _real(1j*ky[None, :]*f, params)
    phie = pe*phi
    max_vy = np.max(np.abs(dx_(phie)) + vmax*np.abs(dx_(psi)))
    max_vx = np.max(np.abs(dy_(phie)) + vmax*np.abs(dy_(psi)))
    return max_vx/params.dx, max_vy/params.dy


def _aniso_ic(params, direction):
    # psi varying ONLY along `direction` (so b_perp points along the other one and exactly
    # one of the two directional rates carries the v_ph,max |grad psi| term), plus a weak
    # isotropic phi so neither rate is zero
    d3 = params.spatial_dimensions == 3

    def ic(x, y, *z):
        s = x if direction == "x" else y
        L = params.Lx if direction == "x" else params.Ly
        psi = jnp.cos(3*TWO_PI*s/L) + 0.4*jnp.sin(2*TWO_PI*s/L)
        phi = 0.02*jnp.cos(TWO_PI*x/params.Lx + 0.3)*jnp.cos(2*TWO_PI*y/params.Ly)
        if d3:
            psi = psi*(1 + 0.3*jnp.cos(TWO_PI*z[0]/params.Lz))
            phi = phi*(1 + 0.3*jnp.sin(TWO_PI*z[0]/params.Lz))
        return jnp.stack([phi + 0*x + 0*y, psi + 0*x + 0*y])
    return ic


def test_set_timestep_matches_the_cfl_bound():
    """dt = cfl_safety/max((|d_y phi_e| + vmax |d_y psi|)/dx, (|d_x phi_e| + vmax |d_x psi|)/dy)
    rebuilt in numpy, with v_ph,max from the independent factors; _vph_max against the max
    of kgrid.eq.vph over the dealias mask.

    Every box is non-square with dx != dy (a dx <-> dy swap changes dt), one case has
    Z = 2, tau = 0.7 (Z/tau enters _vph_max; at Z = 1 a `1/tau` there is invisible), and
    besides the broadband state each case runs two ANISOTROPIC states, psi varying only in
    x and only in y, so that each directional rate binds in turn and carries the v_ph,max
    term on its own: on a broadband state one direction always binds and v_ph,max could be
    dropped from the other unnoticed (review of rung F1, 2026-11-11)."""
    tol = 1e-12 if FP64 else 2e-5
    with checks() as c:
        for label, params, rho_i, zt in (
                ("2D", _flr(rho_i=0.6, tau=0.5, nx=32, ny=24, Lx=TWO_PI*1.2,
                            adaptive_timestep=True), 0.6, 2.0),
                ("2D Z=2 tau=0.7", _flr(rho_i=0.6, tau=0.7, Z=2.0, nx=24, ny=32,
                                        Ly=TWO_PI*0.8, adaptive_timestep=True), 0.6, 2.0/0.7),
                ("z_spectral", _flr3(rho_i=1.5, tau=1.0, nx=16, ny=24, Lx=TWO_PI*1.4,
                                     Ly=TWO_PI*0.9, adaptive_timestep=True), 1.5, 1.0)):
            kgrid = jr.setup_kgrids(params)
            recipe = equation_registry["FLRMHD"]
            set_dt = lambda s: float(recipe.set_timestep_func(
                recipe.grad_func(s, kgrid, params), params))

            vmax = _vph_max_ref(params, rho_i, zt)
            got_vmax = flrmhd._vph_max(params)
            eq_vmax = float(np.max(np.asarray(kgrid.eq.vph)[_dealias_perp(params)]))
            c.check(f"{label}: _vph_max == independent max over the dealiased modes "
                    f"({got_vmax:.6f}, rel {abs(got_vmax/vmax - 1):.1e})",
                    abs(got_vmax/vmax - 1) < 1e-13)
            c.check(f"{label}: _vph_max == max of kgrid.eq.vph over the dealias mask and is "
                    f"above 1.2 (rel {abs(got_vmax/eq_vmax - 1):.1e})",
                    abs(got_vmax/eq_vmax - 1) < (1e-14 if FP64 else 2e-7) and got_vmax > 1.2)
            c.check(f"{label}: the grid spacings differ (dx/dy = {params.dx/params.dy:.3f})",
                    abs(params.dx/params.dy - 1) > 0.05)

            ref = flr_ref(kgrid_ksq(params), rho_i, zt)
            ceiling = rmhd._quiescent_dt(params)
            state = jr.initialize(broadband_ic(params, seed=11, amp=3.0), params)
            states = [("broadband", state, None)]
            for d in ("x", "y"):
                states.append((f"psi({d}) only", jr.initialize(_aniso_ic(params, d), params),
                               d))
            for sname, st, d in states:
                dt = set_dt(st)
                rx, ry = _cfl_rates_ref(st.fields, params, ref["pe"], vmax)
                want = params.cfl_safety/max(rx, ry)
                c.check(f"{label} [{sname}]: set_timestep == the closed-form CFL bound (dt "
                        f"{dt:.5e}, rel {abs(dt/want - 1):.1e}; rate_x {rx:.3g}, rate_y "
                        f"{ry:.3g}), below the quiescent ceiling {ceiling:.3g}",
                        abs(dt/want - 1) < tol and dt < ceiling, f"want {want:.6e}")
                # the same state with v_ph,max = 1 gives a visibly larger dt: the factor is
                # live in the direction that binds
                r1x, r1y = _cfl_rates_ref(st.fields, params, ref["pe"], 1.0)
                rmhd_like = params.cfl_safety/max(r1x, r1y)
                c.check(f"{label} [{sname}]: dropping v_ph,max would change dt by "
                        f"{rmhd_like/dt:.2f}x", rmhd_like/dt > 1.15)
                if d is not None:
                    # psi(x) -> d_x psi only -> rate_y = max_vy/dy binds; psi(y) -> rate_x
                    bind, other = (ry, rx) if d == "x" else (rx, ry)
                    c.check(f"{label} [{sname}]: the rate_{'y' if d == 'x' else 'x'} "
                            f"direction binds by {bind/other:.1f}x and v_ph,max is "
                            f"{bind/(r1y if d == 'x' else r1x):.2f}x of it",
                            bind > 3*other and bind > 1.15*(r1y if d == "x" else r1x))

            zero = state._replace(fields=jnp.zeros_like(state.fields))
            dt0 = np.asarray(recipe.set_timestep_func(recipe.grad_func(zero, kgrid, params),
                                                      params))
            c.check(f"{label}: a zero state gives exactly the _quiescent_dt ceiling",
                    dt0 == np.asarray(ceiling, dtype=dt0.dtype))
            for amp in (1e-6, 1e-2, 1.0, 1e3):
                s = state._replace(fields=state.fields*amp)
                c.check(f"{label}: dt <= _quiescent_dt at amplitude x{amp:g}",
                        set_dt(s) <= float(np.asarray(ceiling, dtype=dt0.dtype)))


def test_Z_defaults_to_one():
    """eqpars without "Z": the factors, _vph_max and the overrides seam's static value are
    those of Z = 1 (review of rung F1: a default of 2.0 passed every gate, because every
    test passed Z explicitly)."""
    tol = 1e-13 if FP64 else 5e-7
    with checks() as c:
        for label, kw in (("2D", dict(dims=2)),
                          ("z_spectral", dict(dims=3, nz=8, Lz=TWO_PI, z_spectral=True))):
            rho_i, tau = 0.8, 0.7
            params = jr.Parameters(nx=16, ny=24, Lx=TWO_PI, Ly=TWO_PI*1.2, cfl_safety=0.5,
                                   eqtype="FLRMHD",
                                   eqpars=dict(rho_i=rho_i, tau=tau, diss=0.0, hyper=1), **kw)
            c.check(f"{label}: the eqpars really carry no Z", "Z" not in params.eqpars)
            kgrid = jr.setup_kgrids(params)
            ref = flr_ref(kgrid_ksq(params), rho_i, 1.0/tau)
            worst = max(float(np.max(np.abs(np.asarray(getattr(kgrid.eq, n)) - ref[n])
                                     / np.maximum(np.abs(ref[n]), 1e-300)))
                        for n in flrmhd.FLRFactors._fields)
            c.check(f"{label}: factors with Z omitted == the paper form at Z/tau = 1/tau "
                    f"(worst rel {worst:.1e})", worst <= tol, f"{worst:.3e}")
            other = flr_ref(kgrid_ksq(params), rho_i, 2.0/tau)
            c.check(f"{label}: (Z = 2 would move pe by "
                    f"{np.max(np.abs(other['pe']/ref['pe'] - 1)):.2f})",
                    np.max(np.abs(other["pe"]/ref["pe"] - 1)) > 0.1)
            c.check(f"{label}: overrides.static_value(params, 'Z') == 1.0",
                    overrides.static_value(params, "Z") == 1.0,
                    f"{overrides.static_value(params, 'Z')!r}")
            c.check(f"{label}: flrmhd._eqpars reports Z == 1.0",
                    flrmhd._eqpars(params)[2] == 1.0)
            vm = flrmhd._vph_max(params)
            vref = _vph_max_ref(params, rho_i, 1.0/tau)
            c.check(f"{label}: _vph_max with Z omitted == independent max at Z/tau = 1/tau "
                    f"(rel {abs(vm/vref - 1):.1e})", abs(vm/vref - 1) < 1e-13)


def _kaw_ic(x, y):
    # a large-scale in-plane field (psi) carrying small-scale kinetic Alfven perturbations
    # up to the dealias edge: their frequency along the perturbed field is v_ph(k) k.b_perp
    psi = 1.0*jnp.cos(y) + 0.6*jnp.sin(x)
    small = 1e-3*(jnp.cos(9*x + 2*y) + jnp.sin(3*x - 9*y) + jnp.cos(7*x + 7*y)
                  + jnp.sin(10*x) + jnp.cos(10*y))
    return jnp.stack([small, psi + small])


def _kaw_run(nsteps=200):
    # rho_i with k_max rho_i = 10 at the 32^2 dealias edge (k_max = 32/3)
    rho_i = 10.0/(32/3.0)
    params = _flr(rho_i=rho_i, tau=0.5, nx=32, ny=32, adaptive_timestep=True, cfl_safety=0.5)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_kaw_ic, params)
    from _flrmhd_testing import invariants
    E0, _ = invariants(state.fields, params, rho_i, 2.0)
    stepper, scheme = get_scheme("lsrk54")
    end = _advance(state, kgrid, params, nsteps, scheme, stepper)
    f = np.asarray(end.fields)
    if not np.all(np.isfinite(f)):
        return np.inf, float(end.t)
    E1, _ = invariants(f, params, rho_i, 2.0)
    return abs(E1/E0 - 1), float(end.t)


@pytest.mark.fp64
def test_cfl_needs_vph_max():
    """The discriminator: at k_max rho_i = 10 (v_ph,max = 12.4) a kinetic-Alfven-dominated
    state runs 200 adaptive steps at cfl_safety = 0.5 conserving energy; the same run with
    v_ph,max forced to 1 in the CFL (RMHD's bound) blows up or loses energy conservation by
    more than 100x. fp64: the conserved-energy drift is compared at round-off scale."""
    drift, t = _kaw_run()
    real = flrmhd._vph_max
    try:
        flrmhd._vph_max = lambda params: 1.0          # the mutation: RMHD's CFL
        bad, tbad = _kaw_run()
    finally:
        flrmhd._vph_max = real
    with checks() as c:
        c.check(f"with v_ph,max in the CFL: 200 steps stable, energy drift {drift:.2e} "
                f"(t = {t:.3f})", np.isfinite(drift) and drift < 1e-3, f"{drift:.3e}")
        c.check(f"with v_ph,max = 1 the run blows up or loses energy conservation by "
                f"> 100x (drift {bad:.2e}, t = {tbad:.3f})", bad > 100*drift,
                f"{bad:.3e} vs {drift:.3e}")


# --------------------------------------------------------------------------- gate F1.10

def _ic_block3(x, y, z):
    # ky-independent, with x and z structure (so a ky block is exact) and a z-MEAN part
    return jnp.stack([0.4*jnp.sin(x)*jnp.cos(z) + 0.15*jnp.cos(2*x) + 0*y,
                      jnp.cos(x) + 0.3*jnp.sin(z)*jnp.sin(x) + 0*y])


def _ic_block2(x, y):
    return jnp.stack([0.4*jnp.sin(x) + 0.15*jnp.cos(2*x) + 0*y,
                      jnp.cos(x) + 0.3*jnp.sin(2*x) + 0*y])


_BOX3 = dict(nx=12, ny=8, nz=8, Lx=TWO_PI, Ly=2*TWO_PI, Lz=3.0)
_BOX2 = dict(nx=24, ny=8, Lx=3.0, Ly=2.0)


def _stab_cases():
    # label -> (params, x0 ic, overridable keys)
    return {
        "2D": (_flr(rho_i=0.7, tau=0.5, diss=(0.03, 0.01), **_BOX2), _ic_block2,
               ["Lx", "Ly", "Z", "diss", "rho_i", "tau"]),
        "z_spectral nu_phi == nu_psi": (
            _flr3(rho_i=0.7, tau=0.5, diss=(0.02, 0.02), eq=dict(z_diss_k=0.3, z_hyper=3),
                  **_BOX3), _ic_block3,
            ["Lx", "Ly", "Lz", "Z", "diss", "rho_i", "tau", "z_diss_k"]),
        "z_spectral nu_phi != nu_psi": (
            _flr3(rho_i=0.7, tau=0.5, Z=2.0, diss=(0.03, 0.01), eq=dict(z_diss_k=0.3),
                  **_BOX3), _ic_block3,
            ["Lx", "Ly", "Lz", "Z", "diss", "rho_i", "tau", "z_diss_k"]),
    }


def _analytic_L_block(idx, params, iky, ep, Lx=None, Ly=None, Lz=None):
    # the docs' L on the block's (field, iz, ikx) entries, factors from the quadrature
    Lx, Ly = Lx or params.Lx, Ly or params.Ly
    kx = np.fft.fftfreq(params.nx)*params.nx*TWO_PI/Lx
    ky = iky*TWO_PI/Ly
    diss = np.broadcast_to(np.asarray(ep["diss"], dtype=float), (2,))
    zsp = params.z_spectral
    if zsp:
        kz = np.fft.fftfreq(params.nz)*params.nz*TWO_PI/(Lz or params.Lz)
        kz[params.nz//2] = 0.0                               # the Nyquist plane is zeroed
    pos = {(int(f), int(iz), int(ix)): r for r, (f, iz, ix) in enumerate(idx)}
    H = np.zeros((len(idx), len(idx)), dtype=complex)
    for r, (f, iz, ix) in enumerate(idx):
        ksq = kx[ix]**2 + ky**2
        H[r, r] = -diss[f]*ksq**ep["hyper"]
        if zsp:
            ref = flr_ref(np.array([ksq]), ep["rho_i"], ep.get("Z", 1.0)/ep["tau"])
            H[r, r] += -ep.get("z_diss_k", 0.0)*kz[iz]**(2*ep.get("z_hyper", 2))
            H[r, pos[(1 - f, iz, ix)]] += 1j*kz[iz]*(ref["a"][0] if f == 0 else ref["pe"][0])
    return H


@pytest.mark.fp64
def test_zero_state_jacobian_is_L():
    """J(x0 = 0) == L entrywise: against kgrid.lin.dense() AND against the docs' matrix
    built from the quadrature factors; block eigenvalues d +- i kz v_ph."""
    with checks() as c:
        for label, (params, _, _) in _stab_cases().items():
            kgrid = jr.setup_kgrids(params)
            ep = params.eqpars
            x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
            dense = np.asarray(kgrid.lin.dense())
            for iky in (1, 2):
                B, idx = stability.ky_block_matrix(x0, kgrid, params, iky)
                D = np.zeros_like(B)
                pos = {(int(f), int(iz), int(ix)): r for r, (f, iz, ix) in enumerate(idx)}
                for r, (f, iz, ix) in enumerate(idx):
                    for g in (0, 1):
                        D[r, pos[(g, iz, ix)]] = dense[f, g, iz if dense.shape[2] > 1 else 0,
                                                       ix, iky]
                e = np.abs(B - D).max()/np.abs(D).max()
                c.check(f"{label} iky={iky}: J(0) == kgrid.lin.dense() entrywise "
                        f"(rel {e:.1e}, n={len(idx)})", e < 1e-14, f"{e:.3e}")
                H = _analytic_L_block(idx, params, iky, ep)
                e = np.abs(B - H).max()/np.abs(H).max()
                c.check(f"{label} iky={iky}: J(0) == the docs' L from the quadrature "
                        f"factors (rel {e:.1e})", e < 1e-13, f"{e:.3e}")
            if params.z_spectral and np.ndim(ep["diss"]) and ep["diss"][0] == ep["diss"][1]:
                # one (kx, kz) pair: eigenvalues d +- i kz v_ph
                kx = np.fft.fftfreq(params.nx)*params.nx*TWO_PI/params.Lx
                kz = np.fft.fftfreq(params.nz)*params.nz*TWO_PI/params.Lz
                for ikx, iz in ((2, 1), (3, 2), (1, 0)):
                    B, _ = stability.ky_block_matrix(x0, kgrid, params, 1, ikx=ikx, iz=iz)
                    ksq = kx[ikx]**2 + (TWO_PI/params.Ly)**2
                    vph = flr_ref(np.array([ksq]), ep["rho_i"], ep.get("Z", 1.0)/ep["tau"]
                                  )["vph"][0]
                    d = -ep["diss"][0]*ksq**ep["hyper"] - ep["z_diss_k"]*kz[iz]**(
                        2*ep["z_hyper"])
                    want = np.array([d - 1j*abs(kz[iz])*vph, d + 1j*abs(kz[iz])*vph])
                    ev = np.linalg.eigvals(B)
                    ev = ev[np.argsort(ev.imag)]             # the pair shares its real part
                    e = np.abs(ev - want).max()/max(np.abs(want).max(), 1e-300)
                    c.check(f"{label} (ikx, iz) = ({ikx}, {iz}): eigenvalues d +- i kz v_ph "
                            f"(v_ph {vph:.4f}, rel {e:.1e})", e < 1e-13, f"{e:.3e}")


@pytest.mark.fp64
def test_fd_gate_on_a_random_state():
    """J v against L v + [N(x0 + e v) - N(x0 - e v)]/2e on a broadband x0 with a random v:
    N is quadratic, so the central difference is exact up to round-off."""
    rng = np.random.default_rng(20261111)
    with checks() as c:
        for label, params in (("2D", _flr(rho_i=0.5, tau=0.5, diss=(0.01, 0.02))),
                              ("z_spectral", _flr3(rho_i=0.5, tau=0.5, diss=(0.01, 0.01),
                                                   eq=dict(z_diss_k=0.1)))):
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(broadband_ic(params, seed=5), params)
            x0 = np.asarray(state.fields)
            rhs = construct_rhs(equation_registry["FLRMHD"])
            N = lambda f: np.asarray(rhs(state._replace(
                fields=jnp.asarray(f, dtype=_precision.ctype)), kgrid, params)[0])
            J = stability.jvp_operator(x0, kgrid, params)
            v = rng.standard_normal(x0.shape) + 1j*rng.standard_normal(x0.shape)
            v = v*np.abs(x0).max()/np.abs(v).max()
            Jv = np.asarray(J(v))
            Lv = np.asarray(kgrid.lin.apply_L(jnp.asarray(v)))
            for eps in (1e-3, 1e-4):
                fd = Lv + (N(x0 + eps*v) - N(x0 - eps*v))/(2*eps)
                e = _relerr(fd, Jv)
                c.check(f"{label}: central FD matches J v at eps = {eps:g} (rel {e:.1e})",
                        e < 1e-9, f"{e:.3e}")
            c.check(f"{label}: the nonlinear part of J v is not negligible "
                    f"(|N'v|/|Jv| = {np.linalg.norm(Jv - Lv)/np.linalg.norm(Jv):.2f})",
                    np.linalg.norm(Jv - Lv) > 0.05*np.linalg.norm(Jv))


def _J(params, kgrid, x0, ov=None, iky=1):
    return stability.ky_block_matrix(x0, kgrid, params, iky, overrides=ov)[0]


def _rebuilt(params, values):
    # a Parameters with `values` set through the ctor / eqpars: the STATIC route
    args = dict(params._init_args)
    eq = dict(args.pop("eqpars"))
    for k, v in values.items():
        v = np.asarray(v, dtype=float)
        v = tuple(float(a) for a in v) if v.ndim else float(v)
        if k in ("Lx", "Ly", "Lz"):
            args[k] = v
        else:
            eq[k] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(eqpars=eq, **args)


def test_overrides_independent_route():
    """overridable(params) is the documented key list; J at overrides == J of a Parameters
    built with those values (the check that catches a read site bypassing getp); J at the
    static point == J at overrides=None."""
    tol = 1e-12 if FP64 else 5e-5
    with checks() as c:
        for label, (params, ic, keys) in _stab_cases().items():
            kgrid = jr.setup_kgrids(params)
            c.check(f"{label}: overridable keys are {keys}",
                    overrides.overridable(params) == keys, f"{overrides.overridable(params)}")
            x0 = np.asarray(jr.initialize(ic, params).fields)
            J0 = _J(params, kgrid, x0)
            static = {k: np.asarray(overrides.static_value(params, k), dtype=float)
                      for k in keys}
            e = _relerr(_J(params, kgrid, x0, static), J0)
            c.check(f"{label}: J at the static point == J at overrides=None (rel {e:.1e})",
                    e < (1e-14 if FP64 else 2e-6), f"{e:.3e}")
            # the plan's pair, then every key at once
            for name, point in (
                    ("{rho_i, tau}", {"rho_i": 0.45, "tau": 0.8}),
                    ("every key", {k: static[k]*(1.0 + 0.07*(i + 1))
                                   for i, k in enumerate(keys)})):
                p2 = _rebuilt(params, point)
                want = _J(p2, jr.setup_kgrids(p2), x0)
                got = _J(params, kgrid, x0, point)
                e = _relerr(got, want)
                c.check(f"{label}: J at overrides {name} == J of a Parameters built with "
                        f"them (rel {e:.1e}); it moved J by {_relerr(got, J0):.1e}",
                        e < tol and _relerr(got, J0) > 1e-3, f"{e:.3e}")
        # rho_i = 0 is the trace-time RMHD branch: the FLR keys are dead there
        p0 = _flr3(rho_i=0.0, diss=(0.02, 0.02), **_BOX3)
        c.check("rho_i = 0: overridable keys drop rho_i, tau, Z",
                overrides.overridable(p0) == ["Lx", "Ly", "Lz", "diss", "z_diss_k"],
                f"{overrides.overridable(p0)}")
        for k in ("rho_i", "tau", "Z"):
            msg = _raises(ValueError, lambda: jr.setup_kgrids(p0, overrides={k: 0.5}))
            c.check(f"rho_i = 0: overriding {k!r} is rejected as dead here",
                    "does not enter" in msg, msg)
        p = _stab_cases()["2D"][0]
        for bad, frag in (({"hyper": 2}, "static"), ({"z_hyper": 3}, "static"),
                          ({"z_diss_k": 0.1}, "does not enter"), ({"tau": -0.5}, "> 0"),
                          ({"Z": 0.0}, "> 0"), ({"rho_i": np.nan}, "finite")):
            msg = _raises(ValueError, lambda: jr.setup_kgrids(p, overrides=bad))
            c.check(f"overrides {bad} rejected ({frag!r})", frag in msg, msg)
        kg = jr.setup_kgrids(p, overrides={"rho_i": 0.4})
        stepper, scheme = get_scheme("lsrk33")
        msg = _raises(ValueError, lambda: block_of_steps(jr.initialize(_ic_block2, p), kg, p,
                                                         1, scheme, stepper))
        c.check("the solver rejects an overridden FLRMHD kgrid", "overrides" in msg, msg)


@pytest.mark.fp64
def test_dJ_dparam_against_central_fd():
    """dJ/dln p against the central difference of J for every live key, O(h^2) or
    round-off. dJ/d rho_i is taken on a ky block that CONTAINS the kz = 0 plane, where
    s^2 = 0 at nu_phi == nu_psi and the sqrt in putzer2_precompute has an infinite tangent:
    J = apply_L + N' never reads s, so dJ is finite there -- asserted, with the kz = 0 rows
    shown to be live."""
    hs = (2e-3, 1e-3)
    with checks() as c:
        for label, (params, ic, keys) in _stab_cases().items():
            kgrid = jr.setup_kgrids(params)
            x0 = np.asarray(jr.initialize(ic, params).fields)
            pt = {k: np.asarray(overrides.static_value(params, k), dtype=float) for k in keys}
            _, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
            for key in keys:
                dJ = stability.dj_matrix(x0, None, kgrid, params, iky=1, dparams={key: pt[key]},
                                         overrides=pt)[0]
                finite = bool(np.all(np.isfinite(dJ)))
                c.check(f"{label}: dJ/dln {key} is finite on the whole block "
                        f"({dJ.shape[0]} rows)", finite,
                        f"{int(np.sum(~np.isfinite(dJ)))} non-finite entries")
                if not finite:
                    continue
                errs = []
                for h in hs:
                    pp, pm = dict(pt), dict(pt)
                    pp[key], pm[key] = pt[key]*(1 + h), pt[key]*(1 - h)
                    fd = (_J(params, kgrid, x0, pp) - _J(params, kgrid, x0, pm))/(2*h)
                    errs.append(_relerr(dJ, fd))
                if errs[-1] < 1e-9:
                    ok, msg = True, f"errs {errs[0]:.1e}, {errs[1]:.1e} (affine: round-off)"
                else:
                    order = fit_order(hs, errs)
                    ok = 1.7 < order < 2.3 and errs[-1] < 1e-4
                    msg = f"errs {errs[0]:.1e}, {errs[1]:.1e}, order {order:.2f}"
                c.check(f"{label}: dJ/dln {key} == central FD ({msg}; |dJ| "
                        f"{np.linalg.norm(dJ):.2e})", ok and np.linalg.norm(dJ) > 0, msg)
                if key == "rho_i" and params.z_spectral:
                    kz0 = idx[:, 1] == 0
                    sub = dJ[np.ix_(kz0, kz0)]
                    c.check(f"{label}: the block holds {int(kz0.sum())} kz = 0 rows and "
                            f"dJ/d rho_i is live on them (|.| = {np.linalg.norm(sub):.2e})",
                            kz0.sum() > 0 and np.linalg.norm(sub) > 1e-3
                            and np.all(np.isfinite(sub)))
        # the same statement on the operator itself: s^2 = 0 on the kz = 0 plane at equal diss
        params = _stab_cases()["z_spectral nu_phi == nu_psi"][0]
        s = np.asarray(jr.setup_kgrids(params).lin.s)
        c.check("nu_phi == nu_psi: s = 0 on the whole kz = 0 plane (the case the finiteness "
                "assertion is about)", np.all(s[0] == 0) and np.any(s[1] != 0))


# --------------------------------------------------------------------------- gate F1.11

def _ic3(x, y, z):
    phi = jnp.cos(x)*jnp.cos(y)*jnp.cos(z) + 0.3*jnp.sin(2*x + y)*jnp.sin(z)
    psi = phi + 0.3*jnp.sin(x)*jnp.cos(y)*jnp.sin(z) + 0.2*jnp.cos(x - 2*y)
    return jnp.stack([phi, psi])


def _ic2(x, y):
    return _ic3(x, y, 0.4)


def test_eqpars_round_trip_and_cross_restart_block():
    with checks() as c, snap_dir() as d:
        params = _flr3(rho_i=0.3, tau=0.5, Z=1.0, diss=(1e-3, 2e-3), hyper=3,
                       eq=dict(z_diss_k=0.01, z_hyper=3))
        params.save(d)
        back = jr.Parameters.from_snapshot(d)
        c.check("eqpars round-trip through params.json",
                back.eqtype == "FLRMHD" and back.eqpars == params.eqpars, f"{back.eqpars}")
        back.save(d)                                                  # identical: a no-op
        kg, kb = jr.setup_kgrids(params), jr.setup_kgrids(back)
        c.check("the restored Parameters builds the same factors and L, bitwise",
                all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(kg.eq, kb.eq))
                and np.array_equal(np.asarray(kg.lin.L), np.asarray(kb.lin.L)))
        other = _flr3(rho_i=0.31, tau=0.5, Z=1.0, diss=(1e-3, 2e-3), hyper=3,
                      eq=dict(z_diss_k=0.01, z_hyper=3))
        msg = _raises(ValueError, lambda: other.save(d))
        c.check("a different rho_i in the same directory is a differing record",
                "different parameters" in msg, msg)
    with checks() as c, snap_dir() as d:
        # an RMHD run directory cannot be continued as FLRMHD (same state shape and dtype:
        # the record is the only guard), nor the reverse
        common = dict(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, dims=3, nz=8,
                      z_spectral=True)
        r = jr.Parameters(eqpars=dict(diss=(1e-3, 1e-3), hyper=1), **common)
        f = jr.Parameters(eqtype="FLRMHD", eqpars=dict(rho_i=0.0, tau=0.5, diss=(1e-3, 1e-3),
                                                       hyper=1), **common)
        r.save(d)
        msg = _raises(ValueError, lambda: f.save(d))
        c.check("an RMHD record blocks an FLRMHD save into the same directory (even at "
                "rho_i = 0)", "different parameters" in msg and "eqtype" in msg, msg)
        sr = jr.initialize(_ic3, r)
        sf = jr.initialize(_ic3, f)
        c.check("(the two states have the same shape and dtype: nothing else would stop it)",
                sr.fields.shape == sf.fields.shape and sr.fields.dtype == sf.fields.dtype)


def test_unforced_restart_is_bitwise():
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for label, params, ic in (
                ("2D", _flr(rho_i=0.4, diss=(1e-3, 1e-3), hyper=2), _ic2),
                ("z_spectral", _flr3(rho_i=0.4, diss=(1e-3, 2e-3)), _ic3),
                ("z_spectral adaptive cfl_every=2",
                 _flr3(rho_i=0.4, diss=(1e-3, 1e-3), adaptive_timestep=True, cfl_every=2),
                 _ic3)):
            kgrid = jr.setup_kgrids(params)
            mid = _advance(jr.initialize(ic, params), kgrid, params, 6, scheme, stepper)
            straight = _advance(mid, kgrid, params, 6, scheme, stepper)
            with snap_dir() as d:
                with managed_manager(params, d, nsnap=2) as m:
                    sn.save_snapshot(0, mid, m, params)
                    m.wait_until_finished()
                params.save(d)
                p2 = jr.Parameters.from_snapshot(d)
                back = sn.load_snapshot(0, d, p2)
                again = _advance(back, jr.setup_kgrids(p2), p2, 6, scheme, stepper)
            c.check(f"{label}: save -> from_snapshot -> load -> continue is bitwise the "
                    f"uninterrupted run (t = {float(again.t):.4f})",
                    np.array_equal(np.asarray(again.fields), np.asarray(straight.fields))
                    and float(again.t) == float(straight.t),
                    f"rel {_relerr(again.fields, straight.fields):.2e}")


def test_rejections():
    good = dict(rho_i=0.3, tau=0.5, diss=0.01, hyper=1)
    base = dict(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, eqtype="FLRMHD")

    def setup(eq, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jr.setup_kgrids(jr.Parameters(eqpars=eq, **dict(base, **kw)))

    z3 = dict(dims=3, nz=8, z_spectral=True)
    with checks() as c:
        for label, eq, kw, frag in (
                ("unknown key", dict(good, rho_s=0.1), {}, "unknown ['rho_s']"),
                ("missing rho_i", {k: v for k, v in good.items() if k != "rho_i"}, {},
                 "missing ['rho_i']"),
                ("missing tau", {k: v for k, v in good.items() if k != "tau"}, {},
                 "missing ['tau']"),
                ("missing diss and hyper", dict(rho_i=0.3, tau=0.5), {},
                 "missing ['diss', 'hyper']"),
                ("rho_i < 0", dict(good, rho_i=-0.1), {}, "rho_i"),
                ("rho_i not finite", dict(good, rho_i=float("inf")), {}, "finite"),
                ("tau = 0", dict(good, tau=0.0), {}, "tau"),
                ("tau < 0", dict(good, tau=-1.0), {}, "tau"),
                ("Z = 0", dict(good, Z=0.0), {}, "Z"),
                ("diss of length 3", dict(good, diss=(0.1, 0.1, 0.1)), {}, "diss"),
                ("diss < 0", dict(good, diss=(0.1, -0.1)), {}, ">= 0"),
                ("hyper = 0", dict(good, hyper=0), {}, "hyper"),
                ("hyper = 1.5", dict(good, hyper=1.5), {}, "hyper"),
                ("z_hyper = 0", dict(good, z_hyper=0), z3, "z_hyper"),
                ("z_hyper = 2.0 (not an int)", dict(good, z_hyper=2.0), z3, "z_hyper"),
                ("z_diss_k < 0", dict(good, z_diss_k=-1.0), z3, "z_diss_k"),
                ("z_diss_k in 2D", dict(good, z_diss_k=0.1), {}, "z_spectral")):
            msg = _raises(ValueError, lambda: setup(eq, **kw))
            c.check(f"{label}: ValueError naming it", frag in msg, msg)
        msg = _raises(NotImplementedError, lambda: setup(good, dims=3, nz=8,
                                                         comm_backend="serial"))
        c.check("finite-difference z: NotImplementedError pointing at rung F6",
                "F6" in msg and "z_spectral" in msg, msg)
        for label, kw in (("2D", {}), ("z_spectral", z3)):
            kg = setup(good, forcing=True, **kw)
            c.check(f"{label} forcing=True: accepted since rung F3 (a forcing shell mask is "
                    f"built)", kg.fmask is not None and bool(np.any(np.asarray(kg.fmask))))
        recipe = equation_registry["FLRMHD"]
        c.check("the forcing hooks are registered: forcing_scale_func and a ForcingTerm "
                "gated on forcing_active",
                recipe.forcing_scale_func is flrmhd.forcing_scale
                and [t.func for t in recipe.term_funcs] == [flrmhd.NonlinearTerm,
                                                            flrmhd.ForcingTerm]
                and recipe.term_funcs[1].active is flrmhd.forcing_active
                and flrmhd.forcing_active is rmhd.forcing_active)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = jr.Parameters(eqpars=good, forcing=True, **base)
        msg = _raises(ValueError, lambda: jr.setup_kgrids(pf, overrides={"rho_i": 0.4}))
        c.check("forcing + the overrides seam: rejected (the seam is unforced only)",
                "unforced" in msg, msg)
        msg = _raises(ValueError, lambda: jr.Parameters(
            eqpars=good, particles=dict(n=8, ensembles=[dict(qm=1.0, init=dict(
                kind="maxwellian", vth=1.0))]), **base))
        c.check("test particles: rejected by Parameters", "eqtype='RMHD'" in msg, msg)

        # CB-IMEX on the z_spectral wave L: rejected by every driver; allowed in 2D
        p3 = _flr3(diss=(1e-3, 1e-3))
        k3 = jr.setup_kgrids(p3)
        for name in ("imexcb2", "imexcb3e", "imexcb3c", "imexcb3f"):
            stepper, scheme = get_scheme(name)
            msg = _raises(ValueError, lambda: block_of_steps(jr.initialize(_ic3, p3), k3, p3,
                                                             1, scheme, stepper))
            c.check(f"z_spectral + {name}: block_of_steps raises", "IMEX" in msg, msg)
        with snap_dir() as d, managed_manager(p3, d) as m:
            for drv in ("simulate", "simulate_scan"):
                if drv == "simulate":
                    call = lambda: jr.simulate(jr.initialize(_ic3, p3), k3, p3, 1.0, 0.02, m,
                                               schemestr="imexcb3e", save=False)
                else:
                    call = lambda: jr.simulate_scan(jr.initialize(_ic3, p3), k3, p3, 1, 1.0,
                                                    0.02, m, schemestr="imexcb3e", save=False)
                msg = _raises(ValueError, call)
                c.check(f"z_spectral + imexcb3e: {drv} raises", "IMEX" in msg, msg)
        msg = _raises(ValueError, lambda: stability.propagator_operator(
            np.asarray(jr.initialize(_ic_block3, p3).fields), k3, p3, T=0.1, dt=0.01,
            scheme="imexcb3e", iky=1))
        c.check("z_spectral + imexcb3e: the stability propagator raises too",
                "IMEX" in msg, msg)
        for name in ("lsrk33", "lsrk54", "rk44"):
            stepper, scheme = get_scheme(name)
            out = _advance(jr.initialize(_ic3, p3), k3, p3, 1, scheme, stepper)
            c.check(f"z_spectral + {name}: accepted", bool(jnp.all(jnp.isfinite(out.fields))))
        p2 = _flr(diss=(1e-3, 1e-3))
        k2 = jr.setup_kgrids(p2)
        stepper, scheme = get_scheme("imexcb3e")
        a = _advance(jr.initialize(_ic2, p2), k2, p2, 4, scheme, stepper)
        stepper, scheme = get_scheme("lsrk54")
        b = _advance(jr.initialize(_ic2, p2), k2, p2, 4, scheme, stepper)
        e = _relerr(a.fields, b.fields)
        c.check(f"2D + imexcb3e: allowed (L is the dissipation diagonal), agrees with lsrk54 "
                f"to the schemes' accuracy (rel {e:.1e})", e < 1e-4, f"{e:.3e}")


def test_dead_z_hyper_warns():
    """eqpars['z_hyper'] is the exponent of the z_spectral kz filter. Set away from its
    default (2) where that filter is absent -- dims=2, or z_spectral with z_diss_k == 0 --
    it does nothing, and setup_kgrids says so (the Parameters pattern for dead
    finite-difference-z knobs). Live or default: silent."""
    good = dict(rho_i=0.3, tau=0.5, diss=0.01, hyper=1)
    base = dict(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, eqtype="FLRMHD")
    z3 = dict(dims=3, nz=8, z_spectral=True)

    def caught(eq, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = jr.Parameters(eqpars=eq, **dict(base, **kw))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            kg = jr.setup_kgrids(p)
        return [str(x.message) for x in w if "z_hyper" in str(x.message)], p, kg

    with checks() as c:
        for label, eq, kw, why in (
                ("2D, z_hyper=3", dict(good, z_hyper=3), {}, "dims=2"),
                ("z_spectral, z_hyper=3, z_diss_k absent", dict(good, z_hyper=3), z3,
                 "z_diss_k"),
                ("z_spectral, z_hyper=1, z_diss_k=0.0", dict(good, z_hyper=1, z_diss_k=0.0),
                 z3, "z_diss_k")):
            msgs, _, _ = caught(eq, **kw)
            c.check(f"{label}: one warning that z_hyper is ignored, naming why ({why!r})",
                    len(msgs) == 1 and "IGNORED" in msgs[0] and why in msgs[0], f"{msgs}")
        for label, eq, kw in (
                ("2D, z_hyper absent", good, {}),
                ("2D, z_hyper=2 (the default, explicit)", dict(good, z_hyper=2), {}),
                ("z_spectral, z_hyper=3, z_diss_k=0.1 (live)",
                 dict(good, z_hyper=3, z_diss_k=0.1), z3),
                ("z_spectral, z_hyper absent, z_diss_k=0", good, z3)):
            msgs, _, _ = caught(eq, **kw)
            c.check(f"{label}: silent", not msgs, f"{msgs}")
        # the knob really is dead where it warns, and live where it does not
        _, _, k3 = caught(dict(good, z_hyper=3), **z3)
        _, _, k2 = caught(good, **z3)
        c.check("z_spectral, z_diss_k = 0: L is bitwise the same at z_hyper = 3 and 2",
                np.array_equal(np.asarray(k3.lin.L), np.asarray(k2.lin.L)))
        _, _, k3 = caught(dict(good, z_hyper=3, z_diss_k=0.1), **z3)
        _, _, k2 = caught(dict(good, z_hyper=2, z_diss_k=0.1), **z3)
        c.check("z_spectral, z_diss_k = 0.1: z_hyper = 3 and 2 give different L",
                not np.array_equal(np.asarray(k3.lin.L), np.asarray(k2.lin.L)))
        # the stability seam supplying z_diss_k makes the exponent live: no warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = jr.Parameters(eqpars=dict(good, z_hyper=3), **dict(base, **z3))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            jr.setup_kgrids(p, overrides={"z_diss_k": 0.2})
        c.check("z_diss_k supplied through the overrides seam: silent",
                not [x for x in w if "z_hyper" in str(x.message)])
        # ... but ONLY that key does: any other override (here rho_i) leaves the kz filter
        # absent and the exponent dead, so the warning must still fire
        for label, ov in (("rho_i", {"rho_i": 0.5}), ("tau and diss", {"tau": 0.7,
                                                                      "diss": 0.02})):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                kg = jr.setup_kgrids(p, overrides=ov)
            msgs = [str(x.message) for x in w if "z_hyper" in str(x.message)]
            c.check(f"overrides of {label} only, dead z_hyper=3: still one warning naming "
                    f"z_diss_k", kg.overrides is not None and len(msgs) == 1
                    and "IGNORED" in msgs[0] and "z_diss_k" in msgs[0], f"{msgs}")


def test_reality_is_preserved_over_a_run():
    # nothing may break F(-kx,-kz,ky) = conj(F(kx,kz,ky)): the +-i kz a / i kz pe entries
    # (kz Nyquist zeroed), the factors (functions of ksq) and the brackets all keep it.
    # fft(ifft(f)) projects onto the Hermitian subspace, so it == f iff f is real.
    tol = 1e-12 if FP64 else 1e-4
    with checks() as c:
        for label, params, ic in (
                ("2D", _flr(rho_i=0.5, diss=(1e-3, 1e-3)), _ic2),
                ("z_spectral", _flr3(rho_i=0.5, diss=(1e-3, 2e-3), eq=dict(z_diss_k=1e-3)),
                 _ic3)):
            kgrid = jr.setup_kgrids(params)
            stepper, scheme = get_scheme("lsrk54")
            end = _advance(jr.initialize(ic, params), kgrid, params, 20, scheme, stepper)
            resid = float(jnp.max(jnp.abs(fft(ifft(end.fields, params), params) - end.fields)))
            scale = float(jnp.max(jnp.abs(end.fields)))
            c.check(f"{label}: the evolved state stays Hermitian (rel {resid/scale:.1e})",
                    resid/scale < tol, f"{resid/scale:.3e}")
        # the kz Nyquist plane of L's off-diagonals is zeroed (the reality constraint)
        params = _flr3(rho_i=0.5)
        L = np.asarray(jr.setup_kgrids(params).lin.L)
        c.check("L's wave entries vanish on the kz Nyquist plane and nowhere else at kz != 0",
                np.all(L[0, 1, params.nz//2] == 0) and np.all(L[1, 0, params.nz//2] == 0)
                and np.all(L[0, 1, 1] != 0) and np.all(L[1, 0, 1] != 0))


def test_lsrk_scan_and_hoist_knobs_agree():
    """hoist_propagator on/off and lsrk_scan on/off agree to round-off. The hoisted ExpOps
    are the arrays apply_exp forms, in the same op order, but unlike the RMHD putzer2 cells of
    tests/test_hoist_propagator.py the pair is NOT bitwise in every FLRMHD cell (measured at
    fp64: z_spectral fixed-dt lsrk54 differs at 4e-18 relative; XLA fuses the stage
    differently) -- the bitwise status of each cell is printed in the check name."""
    with checks() as c:
        for label, kw, ic in (("2D", dict(diss=(1e-3, 1e-3)), _ic2),
                              ("z_spectral", dict(dims=3, nz=8, Lz=TWO_PI, z_spectral=True,
                                                  diss=(1e-3, 1e-3)), _ic3)):
            for mode in (dict(adaptive_timestep=False), dict(adaptive_timestep=True,
                                                             cfl_every=2)):
                for name in ("lsrk54", "rk44"):
                    stepper, scheme = get_scheme(name)
                    out = {}
                    # rk44 has no stage loop: lsrk_scan is not read there
                    variants = ((True, True), (False, True)) + (
                        ((True, False),) if name == "lsrk54" else ())
                    for hoist, scan in variants:
                        p = _flr(rho_i=0.5, hoist_propagator=hoist, lsrk_scan=scan,
                                 **dict(kw, **mode))
                        end = _advance(jr.initialize(ic, p), jr.setup_kgrids(p), p, 6, scheme,
                                       stepper)
                        out[(hoist, scan)] = np.asarray(end.fields)
                    ref = out[(True, True)]
                    e = _relerr(out[(False, True)], ref)
                    tag = f"{label} {'cfl_every=2' if mode['adaptive_timestep'] else 'fixed dt'} {name}"
                    bit = bool(np.array_equal(out[(False, True)], ref))
                    c.check(f"{tag}: hoisted == unhoisted to round-off (rel {e:.1e}) "
                            f"[bitwise: {bit}]", e < (1e-14 if FP64 else 1e-5), f"{e:.3e}")
                    if (True, False) not in out:
                        continue
                    e = _relerr(out[(True, False)], ref)
                    c.check(f"{tag}: lsrk_scan on/off agree to round-off (rel {e:.1e})",
                            e < (1e-13 if FP64 else 1e-5), f"{e:.3e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
