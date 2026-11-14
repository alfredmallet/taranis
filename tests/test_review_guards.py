# Regression gates for the 2026-11-13 external-review findings that are not propagator
# arithmetic (that one lives in test_linear_propagator.py):
#
#   - the drivers raise NumericalFailure on a non-finite solution instead of saving it and
#     printing the ordinary "Ending simulation" line (a NaN time ends `t < t_end` exactly
#     like reaching t_end does; a fixed-dt run keeps a finite t over NaN fields);
#   - the drivers reject arguments under which they could never advance (t_snap <= 0, a
#     fixed dt <= 0, nblock < 1) instead of looping forever;
#   - a forcing shell reaching outside the 2/3 dealiasing region is a setup error;
#   - Parameters.save turns a rank-0 failure into the error it broadcasts, so it reaches
#     every rank instead of stranding them in the bcast;
#   - perpendicular spectra keep every retained mode in a rectangular box (binned sums
#     equal the energies), and are unchanged in a square one;
#   - parspec folds an odd nz correctly;
#   - diagnostics.cmhd refuses an lnrho state rather than reading s = ln(rho) as rho.
#
# Single-process: the 2D / z_spectral / CMHD configurations here are size==1 only.
from _rmhd_testing import (bootstrap, checks, fresh_params, make_state, managed_manager,
                           mpi_size, snap_dir)

bootstrap()

import json
import os

import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision, diagnostics
from taranis.diagnostics import cmhd as dcmhd, gdi as dgdi

TWO_PI = 2.0*np.pi


def _skip_multirank(name):
    if mpi_size() > 1:
        print(f"[SKIP] {name} -- single-process configurations")
        return True
    return False


def _raises(exc, fn):
    """(raised?, message) for fn() against the exception type exc."""
    try:
        fn()
    except exc as e:
        return True, str(e)
    except Exception as e:      # the wrong type is a failure, reported with what came
        return False, f"{type(e).__name__}: {e}"
    return False, "no exception"


def _ic2d(x, y):
    return jnp.stack([jnp.sin(x)*jnp.cos(y), jnp.cos(x + y)])


def _poison(state):
    return state._replace(fields=state.fields.at[0, 0, 1, 1].set(jnp.nan))


# ------------------------------------------------------------------ non-finite solutions

def test_drivers_raise_on_a_nonfinite_solution():
    if _skip_multirank("test_drivers_raise_on_a_nonfinite_solution"):
        return
    with checks() as c:
        for label, kw in (("adaptive dt (t goes NaN)", dict(adaptive_timestep=True)),
                          ("fixed dt (t stays finite)", dict(adaptive_timestep=False))):
            for driver in ("simulate", "simulate_scan"):
                p = fresh_params(dims=2, nx=16, ny=16, diss=0.0, dt=0.01, **kw)
                kg = jr.setup_kgrids(p)
                with snap_dir() as d, managed_manager(p, d) as mngr:
                    st = _poison(make_state(p, ic=_ic2d))
                    if driver == "simulate":
                        run = lambda: jr.simulate(st, kg, p, 0.05, 0.2, mngr)
                    else:
                        run = lambda: jr.simulate_scan(st, kg, p, 3, 0.05, 0.2, mngr)
                    ok, msg = _raises(jr.NumericalFailure, run)
                    mngr.wait_until_finished()
                    from taranis.snapshot_io import get_saved_steps
                    steps = sorted(get_saved_steps(d))
                c.check(f"{driver}, {label}: NumericalFailure", ok, msg[:120])
                # only the initial state (written before the first step) is on disk
                c.check(f"{driver}, {label}: the dead state was not saved", steps == [0],
                        f"saved steps {steps}")


def test_a_healthy_run_is_untouched_by_the_check():
    if _skip_multirank("test_a_healthy_run_is_untouched_by_the_check"):
        return
    p = fresh_params(dims=2, nx=16, ny=16, diss=0.0, dt=0.01, adaptive_timestep=False)
    kg = jr.setup_kgrids(p)
    with checks() as c, snap_dir() as d, managed_manager(p, d) as mngr:
        out = jr.simulate(make_state(p, ic=_ic2d), kg, p, 0.05, 0.1, mngr, save=False)
        c.check("simulate returns a finite state", bool(jnp.all(jnp.isfinite(out.fields)))
                and float(out.t) >= 0.1, f"t = {float(out.t)}")


# ------------------------------------------------------------------ arguments that stall

def test_drivers_reject_arguments_that_cannot_advance():
    if _skip_multirank("test_drivers_reject_arguments_that_cannot_advance"):
        return
    p = fresh_params(dims=2, nx=16, ny=16, diss=0.0, dt=0.01, adaptive_timestep=False)
    p0 = fresh_params(dims=2, nx=16, ny=16, diss=0.0, dt=0.0, adaptive_timestep=False)
    kg, kg0 = jr.setup_kgrids(p), jr.setup_kgrids(p0)
    cases = (
        ("simulate t_snap=0", lambda s, m: jr.simulate(s, kg, p, 0.0, 1.0, m, save=False)),
        ("simulate t_snap<0", lambda s, m: jr.simulate(s, kg, p, -0.1, 1.0, m, save=False)),
        ("simulate t_snap=nan", lambda s, m: jr.simulate(s, kg, p, np.nan, 1.0, m, save=False)),
        ("simulate t_end=inf", lambda s, m: jr.simulate(s, kg, p, 0.1, np.inf, m, save=False)),
        ("simulate fixed dt=0", lambda s, m: jr.simulate(s, kg0, p0, 0.1, 1.0, m, save=False)),
        ("simulate_scan nblock=0",
         lambda s, m: jr.simulate_scan(s, kg, p, 0, 0.1, 1.0, m, save=False)),
        ("simulate_scan nblock=2.5",
         lambda s, m: jr.simulate_scan(s, kg, p, 2.5, 0.1, 1.0, m, save=False)),
        ("simulate_scan t_snap=0",
         lambda s, m: jr.simulate_scan(s, kg, p, 2, 0.0, 1.0, m, save=False)),
        ("simulate_scan fixed dt=0",
         lambda s, m: jr.simulate_scan(s, kg0, p0, 2, 0.1, 1.0, m, save=False)),
    )
    with checks() as c, snap_dir() as d, managed_manager(p, d) as mngr:
        for label, run in cases:
            st = make_state(p, ic=_ic2d)
            ok, msg = _raises(ValueError, lambda: run(st, mngr))
            c.check(f"{label}: ValueError before any step", ok, msg[:100])
            # rejected before the jitted call, so the state was not donated
            c.check(f"{label}: the input state is still alive",
                    bool(jnp.all(jnp.isfinite(st.fields))))


# ------------------------------------------------------------------------- forcing shell

def test_forcing_shell_outside_the_dealias_region_is_rejected():
    if _skip_multirank("test_forcing_shell_outside_the_dealias_region_is_rejected"):
        return
    with checks() as c:
        # 32^2: the 2/3 cut is at index radius 10.67
        for label, shell, bad in (("well inside", (1, 3), False),
                                  ("last fully retained shell", (9, 10), False),
                                  ("straddling the cut", (10, 12), True),
                                  ("wholly outside", (12, 14), True)):
            p = fresh_params(dims=2, nx=32, ny=32, diss=0.0, forcing=True, fshell=shell)
            ok, msg = _raises(ValueError, lambda: jr.setup_kgrids(p))
            c.check(f"fshell={shell} ({label}): {'rejected' if bad else 'accepted'}",
                    ok == bad, msg[:110])
        p = fresh_params(nx=32, ny=32, nz=8, z_spectral=True, forcing=True, fshell=(12, 14))
        ok, msg = _raises(ValueError, lambda: jr.setup_kgrids(p))
        c.check("z_spectral (3-d dealias mask): rejected", ok, msg[:110])
        # forcing off: the shell is never built, so nothing to reject
        p = fresh_params(dims=2, nx=32, ny=32, diss=0.0, forcing=False, fshell=(12, 14))
        ok, msg = _raises(ValueError, lambda: jr.setup_kgrids(p))
        c.check("forcing off: an out-of-range fshell is ignored", not ok, msg[:110])


# ------------------------------------------------------------------- Parameters.save

def test_save_reports_a_rank0_failure_as_its_error():
    # the failure is caught and turned into the broadcast `err`, which is what lets every
    # rank raise; here (one rank) that is visible as a ValueError naming the file rather
    # than the raw exception escaping before the bcast
    p = fresh_params()
    with checks() as c, snap_dir() as d:
        if p.rank == 0:
            with open(os.path.join(d, "params.json"), "w") as f:
                f.write("{ this is not json")
        if p.size > 1:
            p.comm.Barrier()
        ok, msg = _raises(ValueError, lambda: p.save(d))
        c.check("malformed existing params.json: ValueError naming the file",
                ok and "params.json" in msg, msg[:120])
        if p.rank == 0:
            os.remove(os.path.join(d, "params.json"))
        if p.size > 1:
            p.comm.Barrier()
        p.save(d)
        p.save(d)                       # identical re-save stays a no-op
        if p.rank == 0:
            with open(os.path.join(d, "params.json")) as f:
                rec = json.load(f)
            c.check("a good save still writes the record", rec["nx"] == p.nx)


# ----------------------------------------------------------------------------- spectra

def test_perpspec_keeps_every_retained_mode_in_a_rectangular_box():
    if _skip_multirank("test_perpspec_keeps_every_retained_mode_in_a_rectangular_box"):
        return
    rtol = 1e-10 if _precision.precision == "64" else 1e-4

    def ic(x, y):   # kx = 4 lies far outside min(nx,ny)//2 * kunit = 3.2 when Ly = 20 pi
        return jnp.stack([jnp.sin(4*x) + 0.3*jnp.cos(0.3*y) + 0*x,
                          jnp.cos(7*x + 0.1*y) + jnp.sin(0.5*y) + 0*x])

    with checks() as c:
        for label, kw in (("Ly = 10 Lx", dict(Lx=TWO_PI, Ly=10*TWO_PI)),
                          ("Lx = 10 Ly", dict(Lx=10*TWO_PI, Ly=TWO_PI)),
                          ("nx = 4 ny", dict(nx=64, ny=16)),
                          ("square", dict())):
            base = dict(dims=2, nx=64, ny=64, Lx=TWO_PI, Ly=TWO_PI, diss=0.0)
            base.update(kw)
            p = fresh_params(**base)
            kg = jr.setup_kgrids(p)
            sq = lambda x, y: jnp.stack([jnp.sin(4*x)*jnp.cos(3*y), jnp.cos(2*x + y)])
            st = make_state(p, ic=sq if label in ("square", "nx = 4 ny") else ic)
            E = [float(e) for e in diagnostics.energy(st, kg, p)]
            k, s0, s1 = diagnostics.perpspec(st, kg, p)
            dk = float(k[1] - k[0])
            got = [float(jnp.sum(s0))*dk, float(jnp.sum(s1))*dk]
            err = max(abs(g - e)/e for g, e in zip(got, E))
            c.check(f"perpspec sums to the energies ({label})", err < rtol,
                    f"energies {E}, spectrum sums {got}")
        # the square-box bins are the historical ones: min(nx,ny)//2 fundamental units
        p = fresh_params(dims=2, nx=64, ny=64, diss=0.0)
        k = diagnostics.perpspec(make_state(p), jr.setup_kgrids(p), p)[0]
        c.check("square box: bin range unchanged (32 units, bin_factor 2)",
                len(k) == 16 and abs(float(k[-1]) - 31.0) < 1e-5, f"{len(k)} bins to {float(k[-1])}")


def test_gdi_spectra_keep_every_retained_mode_in_a_rectangular_box():
    if _skip_multirank("test_gdi_spectra_keep_every_retained_mode_in_a_rectangular_box"):
        return
    p = jr.Parameters(nx=64, ny=64, Lx=TWO_PI, Ly=10*TWO_PI, dims=2, eqtype="GDI",
                      cfl_safety=0.4, eqpars=dict(Ln=2.0, nu_in=0.3, v0=1.0, gpar_fac=1.0,
                                                  diss=1e-4, hyper=1))
    kg = jr.setup_kgrids(p)
    st = jr.initialize(lambda x, y: jnp.stack([jnp.sin(4*x) + 0*y, jnp.cos(6*x + 0.2*y)]), p)
    with checks() as c:
        k, sN, sphi = dgdi.perp_spectrum(st, kg, p)
        dk = float(k[1] - k[0])
        # <N^2> of sin(4x) is 1/2; (1/2)<|grad phi|^2> of cos(6x + 0.2y) is (36 + 0.04)/4
        tol = 1e-4
        c.check("GDI density spectrum sums to <N^2>",
                abs(float(jnp.sum(sN))*dk - 0.5) < tol, f"{float(jnp.sum(sN))*dk}")
        c.check("GDI potential spectrum sums to (1/2)<|grad phi|^2>",
                abs(float(jnp.sum(sphi))*dk/9.01 - 1.0) < tol, f"{float(jnp.sum(sphi))*dk}")


def test_parspec_handles_odd_and_even_nz():
    if _skip_multirank("test_parspec_handles_odd_and_even_nz"):
        return

    def ic(x, y, z):
        return jnp.stack([jnp.sin(x)*jnp.cos(y + z) + 0.5*jnp.cos(2*x + y)*jnp.sin(3*z),
                          jnp.cos(x + y)*jnp.sin(2*z) + jnp.sin(2*y) + 0*x*z])

    with checks() as c:
        for nz in (8, 9, 12, 15):
            for zs in (False, True):
                p = fresh_params(nx=16, ny=16, nz=nz, diss=0.0, z_spectral=zs,
                                 comm_backend="serial" if not zs else None)
                kg = jr.setup_kgrids(p)
                st = make_state(p, ic=ic)
                E = [float(e) for e in diagnostics.energy(st, kg, p)]
                ok, msg = _raises(Exception, lambda: diagnostics.parspec(st, kg, p))
                if ok:
                    c.check(f"parspec runs at nz={nz}, z_spectral={zs}", False, msg[:100])
                    continue
                k, s0, s1 = diagnostics.parspec(st, kg, p)
                dk = float(k[1] - k[0])
                got = [float(jnp.sum(s0))*dk, float(jnp.sum(s1))*dk]
                err = max(abs(g - e)/e for g, e in zip(got, E))
                c.check(f"parspec sums to the energies (nz={nz}, z_spectral={zs})",
                        err < 1e-4, f"energies {E}, spectrum sums {got}")


# ------------------------------------------------------------------- CMHD diagnostics

def test_cmhd_diagnostics_refuse_an_lnrho_state():
    if _skip_multirank("test_cmhd_diagnostics_refuse_an_lnrho_state"):
        return

    def params(**ep):
        return jr.Parameters(nx=8, ny=8, nz=8, Lx=TWO_PI, Ly=TWO_PI, Lz=TWO_PI, dims=3,
                             z_spectral=True, eqtype="CMHD", cfl_safety=0.4,
                             eqpars=dict(cs0=1.0, diss=0.0, hyper=1, **ep))

    def ic(field0):
        def f(x, y, z):
            shape = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
            one = jnp.ones(shape)
            return jnp.stack([field0*one, 0.1*jnp.sin(x)*one, 0*one, 0*one,
                              0*one, 0*one, one])
        return f

    with checks() as c:
        pl = params(density_var="lnrho")
        kg = jr.setup_kgrids(pl)
        st = jr.initialize(ic(0.0), pl)          # s = ln(rho) = 0, i.e. rho = 1
        for name in ("energies", "mach_numbers", "rho_min", "spectra", "energy_budget"):
            ok, msg = _raises(NotImplementedError, lambda: getattr(dcmhd, name)(st, kg, pl))
            c.check(f"diagnostics.cmhd.{name} refuses density_var='lnrho'", ok, msg[:90])
        # div B never reads field 0, so it stays available
        c.check("divB_max still works on an lnrho state",
                np.isfinite(dcmhd.divB_max(st, kg, pl)))
        for label, ep in (("density_var absent", {}), ("density_var='rho'", dict(density_var="rho"))):
            pr = params(**ep)
            kgr = jr.setup_kgrids(pr)
            sr = jr.initialize(ic(1.0), pr)
            ek, em, ei = (float(e) for e in dcmhd.energies(sr, kgr, pr))
            c.check(f"{label}: energies unchanged (E_mag = 1/2, E_kin = 0.1^2/4)",
                    abs(em - 0.5) < 1e-5 and abs(ek - 0.0025) < 1e-6, f"{ek}, {em}, {ei}")
            c.check(f"{label}: rho_min = 1", abs(dcmhd.rho_min(sr, kgr, pr) - 1.0) < 1e-5)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
