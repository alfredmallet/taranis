# FLR-MHD rung F3 (plans/FLRMHD_PLAN.md sections 3.7 and 5), the forcing gates:
#   F3.1  rho_i = 0 forced equals forced RMHD: momentum and elsasser, 2D and z_spectral, 50
#         steps, same seed -- fields to 1e-12, forcing_state and forcing_key BITWISE (the RNG
#         stream is RMHD's by construction).
#   F3.2  the scale solves the FLR quadratic: with kappa and v_ph far from 1 on the forcing
#         shell, the energy a forward-Euler step of the solver's ForcingTerm injects --
#         evaluated HERE with the FLR weights built from the quadrature Gamma_0 -- is
#         target*dt, per channel. Blind to nothing in kappa: omitted or inverted, in the
#         scale or in f_phi = f_chi/kappa. And the REGISTRY: the forcing the solver steps
#         (construct_rhs of equation_registry["FLRMHD"], forced minus unforced) is
#         flrmhd.ForcingTerm, and its forcing_scale_func is flrmhd.forcing_scale.
#   F3.3  injection rates from rest at nu = 0, elsasser powers from elsasser_powers(0.8,
#         +-0.4): E/t within 30% of eps, H/t within 30% of sigma_eps*eps (sign included);
#         and the deterministic companion, diagnostics.flrmhd.injection summed over the run
#         against the measured change of E and of H, converging at first order in dt.
#   F3.4  the realised sigma_eps at k_min rho_i = 0.1: time mean of eps_H/eps over 200
#         forcing times within 2% of the request.
#   F3.5  forced restart bitwise at forcing_norm_per_step True and False, z_spectral.
#   F3.6  cfl_every = 4 from a developed forced state: finite, energy within 5% of
#         cfl_every = 1 after 100 steps.
#   plus  elsasser_powers against an independent shell mean, its shell against the
#         solver's kgrid.fmask (both precisions, Lx > Ly and Ly > Lx), the sigma_eps = 1/vbar
#         boundary, and the 2D momentum-mode psi == 0 property.
# E, H and their rates are the test-local forms of tests/_flrmhd_testing.py (independent
# Gamma_0), never diagnostics.flrmhd, except where the diagnostic itself is the thing gated.
#
# Both precision sessions run this file except the fp64-marked order measurements; fp32
# tolerances are the measured floors. pytest: `pytest tests/test_flrmhd_forcing.py`. Script:
# `python tests/test_flrmhd_forcing.py` -- never under mpirun (single-process configs).
from _rmhd_testing import bootstrap, checks, fit_order, managed_manager, snap_dir

bootstrap()

import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, snapshot_io
from taranis.diagnostics import flrmhd as dgf
from taranis.physics import construct_rhs, equation_registry, flrmhd, rmhd, shared_physics
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

from _flrmhd_testing import (broadband_ic, elsasser_modal, elsasser_rates, flr_ref,
                             invariants, kgrid_ksq, norm, shell_vbar, yfac)

FP64 = _precision.precision == "64"
TWO_PI = 2.0*np.pi

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _params(eqtype, eqpars, **kw):
    base = dict(nx=32, ny=32, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, dt=0.01,
                adaptive_timestep=False, dims=2, eqtype=eqtype, eqpars=eqpars)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(**base)


def _flr(rho_i, tau=0.5, Z=1.0, diss=(0.0, 0.0), hyper=1, eq=None, **kw):
    eqpars = dict(rho_i=rho_i, tau=tau, Z=Z, diss=diss, hyper=hyper)
    eqpars.update(eq or {})
    return _params("FLRMHD", eqpars, **kw)


_Z3 = dict(dims=3, nx=16, ny=16, nz=8, Lz=TWO_PI, z_spectral=True)


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


def _zero_ic(params):
    if params.spatial_dimensions == 3:
        return lambda x, y, z: jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))
    return lambda x, y: jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))


def _ic3(x, y, z):
    phi = jnp.cos(x)*jnp.cos(y)*jnp.cos(z) + 0.3*jnp.sin(2*x + y)*jnp.sin(z)
    psi = 0.7*phi + 0.3*jnp.sin(x)*jnp.cos(y)*jnp.sin(z) + 0.2*jnp.cos(x - 2*y)
    return jnp.stack([phi, psi])


def _ic2(x, y):
    return _ic3(x, y, 0.4)


def _logged_run(params, kgrid, state, nsteps, schemestr="lsrk54"):
    """nsteps single steps; before each one, diagnostics.flrmhd.injection of the state the
    step starts from (whose forcing_state/forcing_scale are the ones that step uses).
    Returns (end state, dict of per-step arrays: dt, eps, eps_H, self, self_H, eps_plus,
    eps_minus)."""
    stepper, scheme = get_scheme(schemestr)

    def body(s, _):
        inj = dgf.injection(s, kgrid, params)
        s2 = block_of_steps(s, kgrid, params, 1, scheme, stepper)
        out = dict(dt=s2.t - s.t, **{k: inj[k] for k in
                                     ("eps", "eps_H", "self", "self_H", "eps_plus",
                                      "eps_minus")})
        return s2, out

    end, log = jax.jit(lambda s: jax.lax.scan(body, s, None, nsteps))(state)
    return end, {k: np.asarray(v, dtype=float) for k, v in log.items()}


# ---------------------------------------------------------------------------- gate F3.1

def test_rho_i_zero_forced_equals_forced_rmhd():
    tol = 1e-12 if FP64 else 1e-4
    z3 = dict(_Z3, dt=0.004)
    grids = (("2D 32^2", dict(dt=0.004), dict(diss=(1e-4, 1e-4), hyper=2)),
             ("z_spectral 16^2 x 8, nu_phi == nu_psi", z3, dict(diss=(1e-3, 1e-3), hyper=1)),
             ("z_spectral 16^2 x 8, nu_phi != nu_psi", z3, dict(diss=(1e-3, 3e-3), hyper=1)))
    modes = (("momentum", dict(forcing_mode="momentum", forcing_power=0.7)),
             ("elsasser", dict(forcing_mode="elsasser", forcing_power_elsasser=(0.6, 0.2))),
             ("elsasser per-stage norm", dict(forcing_mode="elsasser",
                                              forcing_power_elsasser=(0.6, 0.2),
                                              forcing_norm_per_step=False)))
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for glabel, kw, eq in grids:
            for mlabel, fkw in modes:
                fk = dict(forcing=True, forcing_tau=0.3, fshell=(1, 4), forcing_seed=5, **fkw)
                pr = _params("RMHD", dict(eq), **dict(kw, **fk))
                pf = _params("FLRMHD", dict(rho_i=0.0, tau=0.5, **eq), **dict(kw, **fk))
                ic = _ic3 if pr.spatial_dimensions == 3 else _ic2
                a = _advance(jr.initialize(ic, pr), jr.setup_kgrids(pr), pr, 50, scheme,
                             stepper)
                b = _advance(jr.initialize(ic, pf), jr.setup_kgrids(pf), pf, 50, scheme,
                             stepper)
                e = _relerr(b.fields, a.fields)
                bit = bool(np.array_equal(np.asarray(a.fields), np.asarray(b.fields)))
                label = f"{glabel}, {mlabel}"
                c.check(f"{label}: fields after 50 forced steps == RMHD (rel {e:.1e}) "
                        f"[bitwise: {bit}]", e <= tol, f"{e:.3e}")
                c.check(f"{label}: forcing_state is bitwise RMHD's and nonzero",
                        np.array_equal(np.asarray(a.forcing_state), np.asarray(b.forcing_state))
                        and bool(np.any(np.asarray(b.forcing_state))))
                c.check(f"{label}: forcing_key is bitwise RMHD's",
                        np.array_equal(np.asarray(jax.random.key_data(a.forcing_key)),
                                       np.asarray(jax.random.key_data(b.forcing_key))))
                es = _relerr(b.forcing_scale, a.forcing_scale) if np.any(
                    np.asarray(a.forcing_scale)) else 0.0
                c.check(f"{label}: forcing_scale == RMHD's (rel {es:.1e}), t identical",
                        es <= (1e-11 if FP64 else 1e-3) and float(a.t) == float(b.t),
                        f"{es:.3e}")
                # the forcing did something: the same run unforced differs at O(1)
                if mlabel == "momentum":
                    pu = _params("FLRMHD", dict(rho_i=0.0, tau=0.5, **eq), **kw)
                    u = _advance(jr.initialize(ic, pu), jr.setup_kgrids(pu), pu, 50, scheme,
                                 stepper)
                    c.check(f"{label}: (the forcing moved the state by "
                            f"{_relerr(b.fields, u.fields):.2f})",
                            _relerr(b.fields, u.fields) > 0.02)


# ---------------------------------------------------------------------------- gate F3.2

def _forced_state(params, kgrid, seed, key):
    # a broadband state carrying an O-U envelope from one kick (both z-envelope components)
    state = jr.initialize(broadband_ic(params, seed=seed), params)
    fs, _ = shared_physics.ou_update(jnp.zeros_like(state.forcing_state),
                                     jax.random.key(key), 0.05, params, kgrid)
    return state._replace(forcing_state=fs)


def _channel_targets(params):
    if params.forcing_mode == "momentum":
        return None, float(params.forcing_power)
    ep, em = (float(v) for v in params.forcing_power_elsasser)
    return (ep, em), ep + em


def _registry_checks(c, tag, params, kgrid, state, kw, rho_i, tau, Z):
    """What the solver actually steps: the right-hand side built from the REGISTRY's recipe on
    the forced Parameters, minus the same on its unforced twin, is flrmhd.ForcingTerm of the
    same state -- so a recipe carrying another forcing term (rmhd.ForcingTerm: no 1/kappa on
    f_phi, and its scale solved on (phi, psi) instead of (chi, psi)) is caught whatever the
    hand-called flrmhd functions above do."""
    recipe = equation_registry["FLRMHD"]
    c.check(f"{tag}: the registry's forcing_scale_func is flrmhd.forcing_scale",
            recipe.forcing_scale_func is flrmhd.forcing_scale)
    s = flrmhd.forcing_scale(state, kgrid, params, jnp.asarray(0.02, dtype=_precision.ftype))
    st = state._replace(forcing_scale=s)
    pu = _flr(rho_i, tau=tau, Z=Z, **kw)
    ku = jr.setup_kgrids(pu)
    forced = np.asarray(construct_rhs(recipe)(st, kgrid, params)[0])
    unforced = np.asarray(construct_rhs(recipe)(st, ku, pu)[0])
    F = np.asarray(flrmhd.ForcingTerm(st, None, kgrid, params))
    e = float(np.linalg.norm(forced - unforced - F)/np.linalg.norm(F))
    # what the RMHD term would have put there instead (the mutation this check is for)
    Fr = np.asarray(rmhd.ForcingTerm(st, None, kgrid, params))
    d = float(np.linalg.norm(Fr - F)/np.linalg.norm(F))
    c.check(f"{tag}: construct_rhs(registry) forced - unforced == flrmhd.ForcingTerm (rel "
            f"{e:.1e}; |F|/|N| = {np.linalg.norm(F)/np.linalg.norm(unforced):.1e}; "
            f"rmhd.ForcingTerm differs from it by {d:.2f})",
            e <= (1e-11 if FP64 else 2e-3) and d > 0.05, f"{e:.3e}")


def test_scale_solves_the_flr_quadratic():
    """One forward-Euler step dt of the solver's ForcingTerm F = (f_phi, f_psi) changes the
    FLR energy by   dE = dt*sum_k T_E(f, F) + dt^2*E[F]   exactly (E is quadratic), and the
    per-step scale is built so that this is target*dt -- per Elsasser channel in elsasser
    mode. Both sums are formed here from the quadrature-Gamma_0 weights. The same statement
    from P and F_2 with explicit G*pe weights on the RAW envelope, and the per-stage path
    (cross term only)."""
    tol = 1e-12 if FP64 else 2e-4
    fk = dict(forcing=True, forcing_tau=0.3, forcing_seed=2)
    cases = (
        ("2D non-square, Z/tau = 2.86", dict(nx=32, ny=24, Lx=TWO_PI*1.2, fshell=(1, 4)),
         0.7, 2.0, 0.7),
        ("z_spectral 16^2 x 8, Z/tau = 2", dict(_Z3, fshell=(1, 3)), 0.9, 1.0, 0.5),
    )
    with checks() as c:
        for label, kw, rho_i, Z, tau in cases:
            zt = Z/tau
            for mode, mkw in (("momentum", dict(forcing_mode="momentum", forcing_power=0.7)),
                              ("elsasser", dict(forcing_mode="elsasser",
                                                forcing_power_elsasser=(0.6, 0.25)))):
                params = _flr(rho_i, tau=tau, Z=Z, **dict(kw, **fk, **mkw))
                kgrid = jr.setup_kgrids(params)
                vbar, shell = shell_vbar(params, rho_i, zt)
                kap = flr_ref(kgrid_ksq(params), rho_i, zt)["kappa"][shell]
                tag = f"{label}, {mode}"
                c.check(f"{tag}: kappa on the shell spans [{kap.min():.2f}, {kap.max():.2f}] "
                        f"and vbar = {vbar:.2f}: far from 1",
                        kap.max() - kap.min() > 0.15 and vbar > 1.3)
                state = _forced_state(params, kgrid, seed=21, key=13)
                f = np.asarray(state.fields)
                chan, total = _channel_targets(params)
                _registry_checks(c, tag, params, kgrid, state, dict(kw, dt=params.dt),
                                 rho_i, tau, Z)
                for dt in (0.02, 0.003):
                    s = flrmhd.forcing_scale(state, kgrid, params, jnp.asarray(
                        dt, dtype=_precision.ftype))
                    c.check(f"{tag}, dt = {dt}: forcing_scale is a ({params.n_ou},) "
                            f"{_precision.ftype.__name__} array, not capped",
                            s.shape == (params.n_ou,) and s.dtype == _precision.ftype
                            and bool(np.all(np.abs(np.asarray(s)) < params.forcing_scale_max)),
                            f"{s!r}")
                    F = np.asarray(flrmhd.ForcingTerm(state._replace(forcing_scale=s), None,
                                                      kgrid, params))
                    Tp, Tm, _ = elsasser_rates(f, F, params, rho_i, zt)
                    Sp, Sm, _ = elsasser_modal(F, params, rho_i, zt)
                    dEp = dt*Tp.sum() + dt**2*Sp.sum()
                    dEm = dt*Tm.sum() + dt**2*Sm.sum()
                    e = abs((dEp + dEm)/(total*dt) - 1)
                    c.check(f"{tag}, dt = {dt}: a forward-Euler step injects dE = target*dt "
                            f"(rel {e:.1e}; self term {dt**2*(Sp.sum() + Sm.sum())/(total*dt):.2f} "
                            f"of it)", e <= tol, f"{e:.3e}")
                    # the same from the E = (1/2) sum [G pe |phi|^2 + ksq |psi|^2] form
                    E0, _ = invariants(f, params, rho_i, zt)
                    E1, _ = invariants(f + dt*F, params, rho_i, zt)
                    e = abs((E1 - E0)/(total*dt) - 1)
                    c.check(f"{tag}, dt = {dt}: E(f + dt F) - E(f) = target*dt in the "
                            f"(G pe, ksq) form (rel {e:.1e})",
                            e <= (1e-10 if FP64 else 2e-3), f"{e:.3e}")
                    if chan is not None:
                        ep, em = abs(dEp/(chan[0]*dt) - 1), abs(dEm/(chan[1]*dt) - 1)
                        c.check(f"{tag}, dt = {dt}: per channel dE^+ = eps_+ dt (rel "
                                f"{ep:.1e}) and dE^- = eps_- dt (rel {em:.1e})",
                                ep <= tol and em <= tol, f"{ep:.3e}, {em:.3e}")
                    else:
                        c.check(f"{tag}, dt = {dt}: momentum mode forces chi only -- f_psi "
                                f"is exactly 0", not np.any(F[1]))
                        # P and F_2 on the raw envelope with explicit G*pe weights:
                        #   f_phi = s f_raw/kappa  =>  P = sum G pe Re(phi^* f_raw/kappa),
                        #   F_2 = sum (G pe/kappa^2) |f_raw|^2
                        r = flr_ref(kgrid_ksq(params), rho_i, zt)
                        fraw = np.asarray(shared_physics.reconstruct_envelope(
                            state.forcing_state, kgrid, params))[0]
                        yf, n = yfac(params)[None, :], norm(params)
                        P = np.sum(yf*r["G"]*r["pe"]*np.real(np.conj(f[0])*fraw/r["kappa"]))/n
                        F2 = np.sum(yf*r["G"]*r["pe"]/r["kappa"]**2*np.abs(fraw)**2)/n
                        s0 = float(np.asarray(s)[0])
                        e = abs((s0*P*dt + 0.5*s0**2*F2*dt**2)/(total*dt) - 1)
                        c.check(f"{tag}, dt = {dt}: s P dt + (1/2) s^2 F_2 dt^2 = target dt "
                                f"with P, F_2 from explicit G pe weights (rel {e:.1e})",
                                e <= tol, f"{e:.3e}")
                # the per-stage path (forcing_norm_per_step=False): the scale is target/P,
                # so the instantaneous cross term is the target exactly. (That path caps the
                # scale at sqrt(2 target/(F_2 dt)), rmhd._scale_epilogue; dt = 1e-5 here
                # keeps the cap from binding on this one-kick envelope.)
                pstage = _flr(rho_i, tau=tau, Z=Z, **dict(kw, **fk, **mkw, dt=1e-5,
                                                         forcing_norm_per_step=False))
                kst = jr.setup_kgrids(pstage)
                F = np.asarray(flrmhd.ForcingTerm(state, None, kst, pstage))
                Tp, Tm, _ = elsasser_rates(f, F, pstage, rho_i, zt)
                if chan is None:
                    e = abs((Tp.sum() + Tm.sum())/total - 1)
                else:
                    e = max(abs(Tp.sum()/chan[0] - 1), abs(Tm.sum()/chan[1] - 1))
                c.check(f"{tag}: per-stage normalization -- the instantaneous injection is "
                        f"the target{' per channel' if chan else ''} (rel {e:.1e})",
                        e <= tol, f"{e:.3e}")
                _registry_checks(c, f"{tag}, per-stage norm", pstage, kst, state,
                                 dict(kw, dt=1e-5), rho_i, tau, Z)


# --------------------------------------------------------------------- elsasser_powers

def _solver_shell(rho_i, **kw):
    """kgrid.fmask of the forcing-ON Parameters with these settings: the modes the solver
    really forces."""
    pon = _flr(rho_i, forcing=True, forcing_mode="elsasser", **kw)
    return np.asarray(jr.setup_kgrids(pon).fmask)


def test_elsasser_powers_shell_is_the_solvers():
    """elsasser_powers averages 1/v_ph over grids.forcing_shell(params) -- the function
    setup_kgrids fills kgrid.fmask with -- not over a shell rebuilt on the side. Gated as an
    identity against the fmask of the forced kgrid, in boxes with Lx > Ly and with Ly > Lx
    (so kunit = min(2 pi/Lx, 2 pi/Ly) is each one in turn) and non-2 pi lengths, where at
    fp32 the field-precision k/kunit of an on-edge mode can round below the integer shell
    edge: the float64 shell then differs from the solver's (counted and printed), and vbar
    must follow the solver's."""
    boxes = (("Lx > Ly", dict(nx=32, ny=24, Lx=TWO_PI*1.7, Ly=TWO_PI*0.9)),
             ("Ly > Lx", dict(nx=24, ny=32, Lx=3.0, Ly=5.1)),
             ("Ly > Lx, 3D z_spectral", dict(_Z3, Lx=2.2, Ly=TWO_PI*1.3)),
             ("square 2 pi", dict()))
    shells = ((1, 2), (1, 3), (2, 4), (3, 5), (2, 7))
    rho_i, zt = 0.6, 2.0
    ndiff, worst = 0, 0.0
    with checks() as c:
        for label, kw in boxes:
            same, vb = True, 0.0
            for fshell in shells:
                params = _flr(rho_i, fshell=fshell, **kw)              # forcing off
                fmask = _solver_shell(rho_i, fshell=fshell, **kw)
                shell, vbar = flrmhd._forcing_shell_vbar(params)
                same &= shell.dtype == bool and shell.shape == fmask.shape and bool(
                    np.array_equal(shell, fmask))
                want, _ = shell_vbar(params, rho_i, zt, shell=fmask)
                vb = max(vb, abs(vbar/want - 1))
                f64 = shell_vbar(params, rho_i, zt)[1]
                ndiff += int(not np.array_equal(f64, fmask))
                worst = max(worst, abs(shell_vbar(params, rho_i, zt)[0]/want - 1))
                ep, em = flrmhd.elsasser_powers(1.0, 0.5/want, params)
                vb = max(vb, abs((ep - em)/0.5 - 1))
            c.check(f"{label}: the helper's shell == kgrid.fmask of the forced kgrid for "
                    f"{len(shells)} shells, and vbar == the independent mean over that mask "
                    f"(worst rel {vb:.1e})", same and vb < 1e-13, f"{same}, {vb:.3e}")
        print(f"      (the float64 shell differs from kgrid.fmask in {ndiff} of "
              f"{len(boxes)*len(shells)} box/shell combinations at this precision; vbar over "
              f"it would be off by up to {worst:.1e})")


def test_elsasser_powers_at_the_realizability_boundary():
    """sigma_eps = +-1/vbar is the largest imbalance a shell can inject: one channel zero.
    In floating point the product |sigma_eps|*vbar of a sigma_eps meant to BE 1/vbar lands on
    1 - 1.1e-16, 1 or 1 + 2.2e-16 (the last when sigma_eps is 1/vbar rounded up by an ulp,
    as it is when it comes out of other arithmetic), so the bound is compared with a few ulp
    of slack and the vanishing channel is clamped at 0 -- never a spurious ValueError, never
    a negative power."""
    over, n = 0, 0
    with checks() as c:
        ok = True
        for rho_i in np.linspace(0.05, 2.0, 40):
            for kw in (dict(fshell=(1, 3)), dict(fshell=(2, 4), nx=24, ny=32)):
                params = _flr(float(rho_i), **kw)
                _, vbar = flrmhd._forcing_shell_vbar(params)
                for sig in (1.0/vbar, float(np.nextafter(1.0/vbar, np.inf))):
                    above = sig*vbar > 1.0
                    over += int(above)
                    n += 1
                    for s_ in (sig, -sig):
                        try:
                            ep, em = flrmhd.elsasser_powers(0.8, s_, params)
                        except ValueError as err:
                            ok = False
                            print(f"      rho_i = {rho_i:.3f}, |sigma|*vbar - 1 = "
                                  f"{sig*vbar - 1:.1e}: raised at the boundary: {err}")
                            continue
                        lo, hi = (em, ep) if s_ > 0 else (ep, em)
                        ok &= 0.0 <= lo <= 1e-15 and abs(hi - 0.8) <= 1e-15
                        ok &= (lo == 0.0 and hi == 0.8) if above else True
        c.check(f"sigma_eps = +-1/vbar (and one ulp above it) is accepted with the vanishing "
                f"channel in [0, 1e-15] and never negative, in all {2*n} cases", ok)
        c.check(f"... including the {over} of {n} where |sigma_eps|*vbar rounds ABOVE 1 (what "
                f"a bare `> 1` comparison rejects): there the pair is (eps, 0) exactly",
                over >= 5)
        params = _flr(0.5, fshell=(1, 3))
        _, vbar = flrmhd._forcing_shell_vbar(params)
        for bad in ((1 + 1e-12)/vbar, -(1 + 1e-12)/vbar):
            try:
                flrmhd.elsasser_powers(0.8, bad, params)
                msg = ""
            except ValueError as err:
                msg = str(err)
            c.check("|sigma_eps| vbar = 1 + 1e-12 is still a ValueError", "1/vbar" in msg, msg)


def test_elsasser_powers():
    with checks() as c:
        for label, kw, rho_i, Z, tau in (
                ("2D non-square", dict(nx=32, ny=24, Lx=TWO_PI*1.2, fshell=(2, 4)), 0.5, 2.0,
                 0.7),
                ("z_spectral", dict(_Z3, fshell=(1, 3)), 0.6, 1.0, 0.5),
                ("2D k_min rho_i = 0.1", dict(fshell=(1, 2)), 0.1, 1.0, 0.5)):
            params = _flr(rho_i, tau=tau, Z=Z, **kw)
            # the mean is over the SOLVER's shell (kgrid.fmask of the forced twin)
            vbar, shell = shell_vbar(params, rho_i, Z/tau, shell=_solver_shell(
                rho_i, tau=tau, Z=Z, **kw))
            sig = 0.6/vbar
            ep, em = flrmhd.elsasser_powers(0.8, sig, params)
            c.check(f"{label}: eps_+ + eps_- = eps exactly and both are python floats",
                    abs(ep + em - 0.8) < 1e-15 and isinstance(ep, float)
                    and isinstance(em, float))
            e = abs((ep - em)/(0.8*sig*vbar) - 1)
            c.check(f"{label}: (eps_+ - eps_-)/eps = sigma_eps*vbar with vbar = {vbar:.5f} "
                    f"from the independent shell mean over {int(shell.sum())} modes "
                    f"(rel {e:.1e})", e < 1e-13, f"{e:.3e}")
            em_, ep_ = flrmhd.elsasser_powers(0.8, -sig, params)
            c.check(f"{label}: sigma_eps -> -sigma_eps swaps the two channels",
                    (ep_, em_) == (ep, em))
            for bad in (1.0001/vbar, -1.0001/vbar):
                try:
                    flrmhd.elsasser_powers(0.8, bad, params)
                    msg = ""
                except ValueError as err:
                    msg = str(err)
                c.check(f"{label}: |sigma_eps| = {abs(bad):.4f} > 1/vbar = {1/vbar:.4f} is "
                        f"a ValueError", "1/vbar" in msg, msg)
            ok = flrmhd.elsasser_powers(0.8, 0.9999/vbar, params)
            c.check(f"{label}: just inside 1/vbar is accepted with eps_- > 0",
                    0 < ok[1] < 1e-3)
        p0 = _flr(0.0, fshell=(1, 4))
        c.check("rho_i = 0: vbar = 1, eps_+- = eps (1 +- sigma_eps)/2",
                flrmhd.elsasser_powers(1.0, 0.5, p0) == (0.75, 0.25))
        # forcing on or off, same answer (the helper reads fshell and the box only)
        pon = _flr(0.5, fshell=(2, 4), forcing=True, forcing_mode="elsasser")
        poff = _flr(0.5, fshell=(2, 4))
        c.check("the helper does not need forcing=True",
                flrmhd.elsasser_powers(0.8, 0.3, pon) == flrmhd.elsasser_powers(0.8, 0.3, poff))


# ---------------------------------------------------------------------------- gate F3.3

_EPS, _SIG = 0.8, 0.4


def _f33(kw, sigma, **extra):
    rho_i, zt = 0.5, 2.0
    base = _flr(rho_i, **dict(kw, fshell=(2, 4)))
    ep, em = flrmhd.elsasser_powers(_EPS, sigma, base)
    fk = dict(kw, fshell=(2, 4), forcing=True, forcing_mode="elsasser",
              forcing_power_elsasser=(ep, em), forcing_tau=0.5, forcing_seed=3)
    fk.update(extra)
    return _flr(rho_i, **fk), rho_i, zt


def test_injection_rates_from_rest():
    """Quiescent start, nu = 0: E(t)/t within 30% of eps and H(t)/t within 30% of
    sigma_eps*eps -- on a shell with vbar ~ 1.9, so a dropped 1/v_ph (H rate 1.9x), a factor
    2, or swapped channels (the sign) all leave the band. Both normalization paths, 2D and
    z_spectral, sigma_eps of both signs."""
    with checks() as c:
        for label, kw, sigma, norm_per_step in (
                ("2D 32^2, per-stage norm", {}, +_SIG, False),
                ("2D 32^2, per-step norm", {}, +_SIG, True),
                ("z_spectral 16^2 x 8, per-stage norm, sigma_eps < 0", _Z3, -_SIG, False),
                ("z_spectral 16^2 x 8, per-step norm", _Z3, +_SIG, True)):
            # fixed dt = 0.01 (50 steps): from rest the adaptive dt is the quiescent ceiling
            # (~1), one step of which would overshoot t_end before any forcing acts
            params, rho_i, zt = _f33(kw, sigma, forcing_norm_per_step=norm_per_step)
            kgrid = jr.setup_kgrids(params)
            vbar, _ = shell_vbar(params, rho_i, zt)
            with snap_dir() as d, managed_manager(params, d, nsnap=2) as m:
                end = jr.simulate(jr.initialize(_zero_ic(params), params), kgrid, params,
                                  t_snap=1.0, t_end=0.5, mngr=m, save=False)
            E, H = invariants(end.fields, params, rho_i, zt)
            t = float(end.t)
            rE, rH = E/t, H/t
            c.check(f"{label}: energy injection rate {rE:.3f} within 30% of eps = {_EPS} "
                    f"(shell vbar = {vbar:.3f})", 0.7*_EPS < rE < 1.3*_EPS)
            want = sigma*_EPS
            c.check(f"{label}: helicity injection rate {rH:+.3f} within 30% of sigma_eps*eps "
                    f"= {want:+.3f} (sign included; a dropped 1/v_ph would give "
                    f"{want*vbar:+.3f})", 0.7 < rH/want < 1.3, f"{rH/want:.3f}")


def _integrated(kw, dt, T, frozen):
    """A run from rest at nu = 0: (E, H, rel error of sum(eps*dt + self*dt^2) against the
    measured E, same for H, worst deviation of the per-step injection from the target)."""
    params, rho_i, zt = _f33(kw, _SIG, dt=dt, **(dict(forcing_tau=1e8) if frozen else {}))
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_zero_ic(params), params)
    if frozen:
        fs, _ = shared_physics.ou_update(state.forcing_state, jax.random.key(17), 0.05,
                                         params, kgrid)
        state = state._replace(forcing_state=fs)
    end, log = _logged_run(params, kgrid, state, int(round(T/dt)))
    E, H = invariants(end.fields, params, rho_i, zt)
    injE = np.sum(log["eps"]*log["dt"] + log["self"]*log["dt"]**2)
    injH = np.sum(log["eps_H"]*log["dt"] + log["self_H"]*log["dt"]**2)
    hit = np.max(np.abs(log["eps"][2:] + log["self"][2:]*dt - _EPS))
    return E, H, abs(E - injE)/abs(E), abs(H - injH)/abs(H), hit


def test_integrated_injection_matches_the_measured_change():
    """The deterministic companion of F3.3, at BOTH precisions: diagnostics.flrmhd.injection
    (which calls flrmhd.ForcingTerm), summed over the steps of a run from rest at nu = 0 as
    (eps*dt + self*dt^2), against the E(T) and H(T) the SOLVER produced (test-local forms).
    The difference is the Runge-Kutta step against forward Euler with the forcing frozen over
    the step: first order in dt. With the O-U forcing at dt = 0.005 it is below 5e-3 for E
    and H (fp64: 1.7e-4 / 6.8e-6 in 2D, 1.5e-3 / 1.4e-3 under z_spectral; fp32 the same to
    the digits shown -- the fp32 floor of these sums is ~1e-5, far below the gate). This is
    the check that sees a registry stepping some other forcing term than the one the
    diagnostic calls. No order is asserted here: the O-U stream draws one noise field per
    STEP, so runs at different dt see different forcing realizations (measured 2026-11-11:
    3.5e-4, 6.6e-5, 1.7e-4 for E in 2D at dt = 0.02, 0.01, 0.005)."""
    with checks() as c:
        for label, kw in (("2D 32^2", {}), ("z_spectral 16^2 x 8", _Z3)):
            E, H, eE, eH, hit = _integrated(kw, 0.005, 0.4, frozen=False)
            c.check(f"{label}, O-U forcing, dt = 0.005: the run injected E = {E:.4f} (eps*T = "
                    f"{_EPS*0.4:.3f}), H = {H:.4f}; the per-step injection is the target to "
                    f"{hit:.1e}", E > 0.2 and H > 0.05 and hit < (1e-12 if FP64 else 1e-4))
            c.check(f"{label}, O-U forcing, dt = 0.005: sum of injection == measured change, "
                    f"rel {eE:.1e} (E), {eH:.1e} (H)", eE <= 5e-3 and eH <= 5e-3)


@pytest.mark.fp64
def test_integrated_injection_converges_at_first_order():
    """With a FROZEN envelope (one O-U kick, forcing_tau = 1e8, so every dt sees the same
    forcing) the error of the sum above converges at first order or better -- order asserted
    in [0.7, 2.3]; measured 0.97 (E) and 1.61 (H, where the first-order coefficient is
    small: 1.7e-5 at the finest dt) in 2D, 1.10 and 1.07 under z_spectral -- and is below the
    plan's 1e-3 at the finest dt (z_spectral reaches it only at dt = 0.0025: 6.9e-4; the
    kinetic Alfven rotation of Theta within a step is the first-order term)."""
    with checks() as c:
        for label, kw in (("2D 32^2", {}), ("z_spectral 16^2 x 8", _Z3)):
            dts, errE, errH = (0.01, 0.005, 0.0025), [], []
            for dt in dts:
                E, H, eE, eH, _ = _integrated(kw, dt, 0.4, frozen=True)
                errE.append(eE)
                errH.append(eH)
            oE, oH = fit_order(dts, errE), fit_order(dts, errH)
            c.check(f"{label}, frozen envelope (E = {E:.3f}, H = {H:.3f}): sum of injection == "
                    f"measured dE, rel {errE[0]:.1e} -> {errE[1]:.1e} -> {errE[2]:.1e}, order "
                    f"{oE:.2f}", errE[-1] <= 1e-3 and 0.7 <= oE <= 2.3, f"{errE}")
            c.check(f"{label}, frozen envelope: sum of injection == measured dH, rel "
                    f"{errH[0]:.1e} -> {errH[1]:.1e} -> {errH[2]:.1e}, order {oH:.2f}",
                    errH[-1] <= 1e-3 and 0.7 <= oH <= 2.3, f"{errH}")


# ---------------------------------------------------------------------------- gate F3.4

def test_realised_sigma_eps():
    """k_min rho_i = 0.1 (rho_i = 0.1 in a 2 pi box, fshell = (1, 2)): the time mean of the
    realised eps_H/eps over 200 forcing times is within 2% of the requested sigma_eps. The
    injected rates are diagnosed per step by diagnostics.flrmhd.injection (cross + self)."""
    rho_i, zt, sigma, eps, tauf, dt = 0.1, 2.0, 0.6, 0.02, 0.05, 0.01
    with checks() as c:
        for label, kw in (("2D 16^2", dict(nx=16, ny=16)),
                          ("z_spectral 16^2 x 4", dict(_Z3, nz=4))):
            base = _flr(rho_i, **dict(kw, fshell=(1, 2)))
            ep, em = flrmhd.elsasser_powers(eps, sigma, base)
            params = _flr(rho_i, diss=(0.05, 0.05), **dict(
                kw, fshell=(1, 2), forcing=True, forcing_mode="elsasser",
                forcing_power_elsasser=(ep, em), forcing_tau=tauf, forcing_seed=7, dt=dt))
            kgrid = jr.setup_kgrids(params)
            n = int(round(200*tauf/dt))
            end, log = _logged_run(params, kgrid, jr.initialize(_zero_ic(params), params), n)
            dE = log["eps"]*log["dt"] + log["self"]*log["dt"]**2
            dH = log["eps_H"]*log["dt"] + log["self_H"]*log["dt"]**2
            got = dH[10:].sum()/dE[10:].sum()
            vbar, shell = shell_vbar(params, rho_i, zt)
            vph = flr_ref(kgrid_ksq(params), rho_i, zt)["vph"][shell]
            c.check(f"{label}: realised <eps_H>/<eps> = {got:.5f} over {n*dt/tauf:.0f} forcing "
                    f"times, within 2% of sigma_eps = {sigma} (rel {abs(got/sigma - 1):.1e}; "
                    f"shell v_ph spread {vph.max()/vph.min() - 1:.1e}, vbar = {vbar:.5f})",
                    abs(got/sigma - 1) <= 0.02 and bool(np.all(np.isfinite(np.asarray(
                        end.fields)))), f"{got!r}")
            # sharper than the plan's 2%, at the measured level (7e-5 2D, 7.5e-4 z_spectral):
            # 0.5% is below the 1.1% by which vbar differs from 1 here, so a dropped 1/v_ph
            # in elsasser_powers is caught on this shell too
            c.check(f"{label}: ... and within 0.5% (vbar - 1 = {vbar - 1:.4f} is resolved)",
                    abs(got/sigma - 1) <= 5e-3 and vbar - 1 > 0.01)
            c.check(f"{label}: the energy injection is the request (mean eps "
                    f"{dE[10:].sum()/log['dt'][10:].sum():.5f} vs {eps})",
                    abs(dE[10:].sum()/log["dt"][10:].sum()/eps - 1) < 1e-3)
            # NOT asserted per step: the per-step cross and self terms are separately much
            # larger than their sum whenever the envelope is anti-correlated with the field,
            # and they are distributed differently over the shell, so the INSTANTANEOUS
            # eps_H/eps wanders far outside the shell's v_ph spread; the mean is what the
            # forcing controls.
            inst = dH[10:]/dE[10:]
            print(f"      {label}: instantaneous eps_H/eps in [{inst.min():.3f}, "
                  f"{inst.max():.3f}], rms deviation {np.std(inst):.3f}")


# ---------------------------------------------------------------------------- gate F3.5

def _host_state(state):
    return {"fields": np.asarray(state.fields),
            "forcing_state": np.asarray(state.forcing_state),
            "forcing_scale": np.asarray(state.forcing_scale),
            "forcing_key": np.asarray(jax.random.key_data(state.forcing_key)),
            "t": float(state.t)}


def _run_to_end(state, kg, p, path, t_end=0.2):
    with managed_manager(p, path, nsnap=10) as mngr:
        return _host_state(jr.simulate_scan(state, kg, p, 5, 0.04, t_end, mngr, save=True))


def test_forced_restart_is_bitwise():
    """A restart from a mid-run snapshot continues the uninterrupted forced FLRMHD run
    bitwise, at forcing_norm_per_step True (the stored lagged scale is kept on entry) and
    False, z_spectral (the tests/test_forcing_norm_per_step.py pattern)."""
    with checks() as c:
        for norm_per_step in (True, False):
            base = _flr(0.5, **dict(_Z3, fshell=(1, 3)))
            ep, em = flrmhd.elsasser_powers(0.5, 0.3, base)
            p = _flr(0.5, diss=(1e-3, 2e-3), eq=dict(z_diss_k=1e-3, z_hyper=3), **dict(
                _Z3, fshell=(1, 3), forcing=True, forcing_mode="elsasser",
                forcing_power_elsasser=(ep, em), forcing_tau=0.2, forcing_seed=4,
                forcing_norm_per_step=norm_per_step))
            kg = jr.setup_kgrids(p)
            with snap_dir("flr_ref_") as d1:
                ref = _run_to_end(jr.initialize(_ic3, p), kg, p, d1)
                steps = snapshot_io.get_saved_steps(d1)
                mid = steps[len(steps)//2]
                state_mid = snapshot_io.load_snapshot(mid, d1, p)
                t_mid = float(state_mid.t)
                mid_scale = np.asarray(state_mid.forcing_scale)
                with snap_dir("flr_restart_") as d2:
                    got = _run_to_end(state_mid, kg, p, d2)
            tag = f"forcing_norm_per_step={norm_per_step}"
            c.check(f"{tag}: the mid-run snapshot really was mid-run (t = {t_mid:.3f} of "
                    f"{ref['t']:.3f})", 0.0 < t_mid < ref["t"])
            if norm_per_step:
                c.check(f"{tag}: the snapshot carries a nonzero forcing_scale",
                        bool(np.any(mid_scale != 0)), str(mid_scale))
            c.check(f"{tag}: the restart lands on the same final time", got["t"] == ref["t"])
            for key in ("fields", "forcing_state", "forcing_scale", "forcing_key"):
                c.check(f"{tag}: restarted {key} is bitwise identical",
                        np.array_equal(got[key], ref[key]))
            c.check(f"{tag}: (the run was forced: forcing_state nonzero)",
                    bool(np.any(ref["forcing_state"])))


# ---------------------------------------------------------------------------- gate F3.6

def test_cfl_every_from_a_developed_state():
    """cfl_every = 4 from a DEVELOPED forced state: finite, and within 5% of the
    cfl_every = 1 energy after 100 steps (the standing overshoot caveat is about starts
    from rest)."""
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for label, kw, rho_i in (("2D 32^2", {}, 0.3), ("z_spectral 16^2 x 8", _Z3, 0.5)):
            base = _flr(rho_i, **dict(kw, fshell=(1, 3)))
            ep, em = flrmhd.elsasser_powers(1.0, 0.3, base)
            fk = dict(fshell=(1, 3), forcing=True, forcing_mode="elsasser",
                      forcing_power_elsasser=(ep, em), forcing_tau=0.3, forcing_seed=6,
                      adaptive_timestep=True)
            mk = lambda n: _flr(rho_i, diss=(2e-3, 2e-3), **dict(kw, **fk, cfl_every=n))
            p1, p4 = mk(1), mk(4)
            k1, k4 = jr.setup_kgrids(p1), jr.setup_kgrids(p4)
            ic = _ic3 if p1.spatial_dimensions == 3 else _ic2
            dev = _advance(jr.initialize(ic, p1), k1, p1, 300, scheme, stepper)
            host = jax.tree.map(lambda v: v, dev)        # the same developed state, twice
            E0, _ = invariants(dev.fields, p1, rho_i, 2.0)
            a = _advance(dev, k1, p1, 100, scheme, stepper)
            b = _advance(host, k4, p4, 100, scheme, stepper)
            Ea, _ = invariants(a.fields, p1, rho_i, 2.0)
            Eb, _ = invariants(b.fields, p4, rho_i, 2.0)
            c.check(f"{label}: developed state at t = {float(dev.t):.3f}, E = {E0:.3f}; "
                    f"cfl_every=4 stays finite over 100 steps",
                    bool(np.all(np.isfinite(np.asarray(b.fields)))) and E0 > 0.1)
            c.check(f"{label}: energy after 100 steps, cfl_every=4 vs 1: {Eb:.4f} vs "
                    f"{Ea:.4f} (rel {abs(Eb/Ea - 1):.1e}; t = {float(b.t):.4f} vs "
                    f"{float(a.t):.4f})", abs(Eb/Ea - 1) <= 0.05, f"{abs(Eb/Ea - 1):.3e}")


# ------------------------------------------------------------------------------- extras

def test_2d_momentum_quiescent_keeps_psi_zero():
    """2D momentum forcing from rest is pure (FLR) hydro: f_psi = 0 and psi's only source
    -{phi_e, psi} vanishes, so psi stays EXACTLY zero -- as in RMHD -- while phi is forced."""
    params = _flr(0.6, forcing=True, forcing_mode="momentum", forcing_power=0.7,
                  forcing_tau=0.5, fshell=(1, 5), forcing_seed=3)
    kgrid = jr.setup_kgrids(params)
    with snap_dir() as d, managed_manager(params, d, nsnap=2) as m:
        end = jr.simulate(jr.initialize(_zero_ic(params), params), kgrid, params, t_snap=1.0,
                          t_end=0.3, mngr=m, save=False)
    f = np.asarray(end.fields)
    E, H = invariants(f, params, 0.6, 2.0)
    with checks() as c:
        c.check("psi stays EXACTLY zero", not np.any(f[1]),
                f"max|psi_k| = {np.max(np.abs(f[1])):.3e}")
        c.check(f"phi is forced: E/t = {E/float(end.t):.3f} within 2x of forcing_power = 0.7 "
                f"(FLR energy; per-step lagged norm), H = {H:.1e} exactly 0",
                0.35 < E/float(end.t) < 1.4 and H == 0.0)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
