# FLR-MHD rung F1 (plans/FLRMHD_PLAN.md section 5), the linear gates:
#   F1.3  rho_i = 0 reproduces the RMHD recipe: one RHS evaluation and 20 lsrk54 steps, in
#         2D 64^2 (decaying, hyper = 2) and z_spectral 32^2 x 8 with nu_phi = nu_psi (RMHD on
#         the separable backend, FLRMHD on putzer2) and nu_phi != nu_psi (both putzer2).
#         Asserted at 1e-12 relative L2; whether each case is BITWISE is measured and
#         printed in the check name (the graphs differ: multiplies by the unit factors, and
#         a different propagator backend in the equal-diss case).
#   F1.4  small-rho_i convergence: omega^2/kz^2 - 1 of one linear mode against
#         (3/4 + Z/tau) k^2 rho_i^2/2, rho_i halved three times -- order 2.00 +- 0.05 and the
#         COEFFICIENT within 2%, at two Z/tau (a wrong O(rho_i^2) coefficient is invisible
#         to F1.3).
#   F1.5  the kinetic-Alfven dispersion relation, exact: a pure Theta^+ coefficient at
#         k rho_i in {0.01, 0.1, 0.5, 1, 2, 5, 10} and Z/tau in {2, 1} advances as
#         exp(+i kz v_ph t) with v_ph from the INDEPENDENT closed form, stays a pure Theta^+
#         (the eigenvector -- dispersion alone is blind to a <-> pe), Theta^- runs the other
#         way, a non-cubic box with kx, ky, kz all != 0 and != 1, and the decay rate
#         nu k^2h + nu_z kz^(2 h_z) at z_hyper in {2, 3}.
# References come from tests/_flrmhd_testing.py (numpy, quadrature Gamma_0), never from the
# module under test.
#
# Both precision sessions run F1.3-F1.5; the fp32 tolerances are the measured floors (see
# each test). pytest: `pytest tests/test_flrmhd_linear.py`. Script:
# `python tests/test_flrmhd_linear.py` -- never under mpirun (single-process configs only).
from _rmhd_testing import bootstrap, checks, fit_order

bootstrap()

import warnings

import numpy as np

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision
from taranis.physics import construct_rhs, equation_registry
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

from _flrmhd_testing import broadband_ic, flr_ref

FP64 = _precision.precision == "64"
TWO_PI = 2.0*np.pi

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _params(eqtype, eqpars, **kw):
    base = dict(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, dt=0.01,
                adaptive_timestep=False, dims=2, eqtype=eqtype, eqpars=eqpars)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(**base)


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


# ---------------------------------------------------------------------------- gate F1.3

def test_rho_i_zero_reproduces_rmhd():
    tol = 1e-12 if FP64 else 2e-5
    z3 = dict(dims=3, nx=32, ny=32, nz=8, Lz=TWO_PI, z_spectral=True, dt=0.004)
    cases = (
        ("2D 64^2 hyper=2", dict(dims=2, nx=64, ny=64, dt=0.002),
         dict(diss=(1e-6, 1e-6), hyper=2)),
        ("z_spectral 32^2 x 8, nu_phi == nu_psi", z3, dict(diss=(1e-3, 1e-3), hyper=1)),
        ("z_spectral 32^2 x 8, nu_phi != nu_psi", z3, dict(diss=(1e-3, 3e-3), hyper=1)),
        ("z_spectral 32^2 x 8, nu_phi == nu_psi, z_diss_k", z3,
         dict(diss=(1e-3, 1e-3), hyper=2, z_diss_k=0.02)),
        ("z_spectral 32^2 x 8, adaptive dt", dict(z3, adaptive_timestep=True),
         dict(diss=(1e-3, 2e-3), hyper=1)),
    )
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for label, kw, eq in cases:
            pr = _params("RMHD", dict(eq), **kw)
            pf = _params("FLRMHD", dict(eq, rho_i=0.0, tau=0.5), **kw)
            kr, kf = jr.setup_kgrids(pr), jr.setup_kgrids(pf)
            ic = broadband_ic(pr, seed=3, amp=1.0)
            sr, sf = jr.initialize(ic, pr), jr.initialize(ic, pf)
            c.check(f"{label}: identical initial states",
                    np.array_equal(np.asarray(sr.fields), np.asarray(sf.fields)))
            c.check(f"{label}: operators are {type(kr.lin).__name__} (RMHD) and "
                    f"{type(kf.lin).__name__} (FLRMHD); L == RMHD's L bitwise",
                    np.array_equal(np.asarray(kr.lin.dense()), np.asarray(kf.lin.dense())))
            a, ga = construct_rhs(equation_registry["RMHD"])(sr, kr, pr)
            b, gb = construct_rhs(equation_registry["FLRMHD"])(sf, kf, pf)
            e = _relerr(b, a)
            c.check(f"{label}: one RHS evaluation, rel {e:.1e} "
                    f"[bitwise: {bool(np.array_equal(np.asarray(a), np.asarray(b)))}]",
                    e <= tol, f"{e:.3e}")
            dta = float(equation_registry["RMHD"].set_timestep_func(ga, pr))
            dtb = float(equation_registry["FLRMHD"].set_timestep_func(gb, pf))
            c.check(f"{label}: set_timestep on the same state, rel {abs(dtb/dta - 1):.1e} "
                    f"[bitwise: {dta == dtb}]", abs(dtb/dta - 1) <= (1e-14 if FP64 else 1e-6))
            er = _advance(sr, kr, pr, 20, scheme, stepper)
            ef = _advance(sf, kf, pf, 20, scheme, stepper)
            e = _relerr(ef.fields, er.fields)
            bit = bool(np.array_equal(np.asarray(er.fields), np.asarray(ef.fields)))
            moved = _relerr(er.fields, sr.fields)
            c.check(f"{label}: 20 lsrk54 steps, rel {e:.1e}, t rel "
                    f"{abs(float(ef.t)/float(er.t) - 1):.1e} [bitwise: {bit}] (the run moved "
                    f"the state by {moved:.2f})",
                    e <= tol and moved > 0.05
                    and abs(float(ef.t)/float(er.t) - 1) <= (1e-13 if FP64 else 1e-6),
                    f"{e:.3e}")


# ---------------------------------------------------------------------------- gate F1.4

_MODE = (2, 1, 1)                # (ikx, iky, iz): k_perp^2 = 5, kz = 1 in the 2 pi box


def _single_mode_cos_ic(x, y, z):
    # phi only: phi_k(t) = phi_k(0) cos(kz v_ph t), with no knowledge of the eigenvector
    phi = 1e-3*jnp.cos(_MODE[0]*x + _MODE[1]*y)*jnp.cos(_MODE[2]*z)
    return jnp.stack([phi, jnp.zeros_like(phi)])


def _measured_vph2(rho_i, Z, tau, nsteps=20, dt=0.05):
    params = _params("FLRMHD", dict(rho_i=rho_i, tau=tau, Z=Z, diss=(0.0, 0.0), hyper=1),
                     dims=3, nz=8, Lz=TWO_PI, z_spectral=True, dt=dt)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_single_mode_cos_ic, params)
    stepper, scheme = get_scheme("lsrk54")
    end = _advance(state, kgrid, params, nsteps, scheme, stepper)
    ikx, iky, iz = _MODE
    ratio = complex(np.asarray(end.fields)[0, iz, ikx, iky]/np.asarray(state.fields)[0, iz, ikx, iky])
    T = float(end.t)
    omega = np.arccos(np.clip(ratio.real, -1.0, 1.0))/T          # kz v_ph T ~ 1 < pi
    return (omega/_MODE[2])**2, abs(ratio.imag)


def test_small_rho_i_convergence():
    """omega^2/kz^2 - 1 = (3/4 + Z/tau) k^2 rho_i^2/2 + O(rho_i^4), measured from the
    stepped solution of one mode."""
    ksq = _MODE[0]**2 + _MODE[1]**2
    rhos = (0.08, 0.04, 0.02, 0.01)
    with checks() as c:
        for Z, tau in ((1.0, 0.5), (1.0, 2.0)):
            dev, imag = zip(*(_measured_vph2(r, Z, tau) for r in rhos))
            dev = np.array(dev) - 1.0
            order = fit_order(rhos, dev)
            coef = dev/(0.5*ksq*np.array(rhos)**2)
            want = 0.75 + Z/tau
            c.check(f"Z/tau = {Z/tau:g}: v_ph^2 - 1 converges at order {order:.3f} in rho_i "
                    f"(2.00 +- 0.05)", abs(order - 2.0) <= 0.05, f"{order:.4f}")
            c.check(f"Z/tau = {Z/tau:g}: coefficient {coef[-1]:.4f}, {coef[-2]:.4f} vs "
                    f"3/4 + Z/tau = {want:.4f} (within 2% at the two smallest rho_i)",
                    np.all(np.abs(coef[-2:]/want - 1) <= 0.02), f"{coef}")
            c.check(f"Z/tau = {Z/tau:g}: the mode stays a standing cos(omega t) "
                    f"(imaginary part {max(imag):.1e})", max(imag) < (1e-12 if FP64 else 1e-5))


# ---------------------------------------------------------------------------- gate F1.5

def _kaw_case(kperp_rho, Z, tau, *, sign=+1, box=None, mode=(2, 1, 1), diss=0.0, hyper=1,
              zdk=0.0, z_hyper=2, nsteps=20, dt=0.01):
    """One pure Theta^(sign) plane wave cos(k.x) with psi = sign*kappa*phi, kappa from the
    INDEPENDENT closed form; returns the measured per-field complex ratio F(T)/F(0), the
    Theta^(-sign) contamination at the end, and the references."""
    Lx, Ly, Lz = box or (TWO_PI, TWO_PI, TWO_PI)
    ikx, iky, iz = mode
    kx, ky, kz = ikx*TWO_PI/Lx, iky*TWO_PI/Ly, iz*TWO_PI/Lz
    ksq = kx**2 + ky**2
    rho_i = kperp_rho/np.sqrt(ksq)
    ref = {k: float(v[0]) for k, v in flr_ref(np.array([ksq]), rho_i, Z/tau).items()}
    eq = dict(rho_i=rho_i, tau=tau, Z=Z, diss=(diss, diss), hyper=hyper)
    if zdk:
        eq.update(z_diss_k=zdk, z_hyper=z_hyper)
    params = _params("FLRMHD", eq, Lx=Lx, Ly=Ly, dims=3, nz=8, Lz=Lz, z_spectral=True, dt=dt)
    kgrid = jr.setup_kgrids(params)
    kappa = ref["kappa"]

    def ic(x, y, z):
        phi = 1e-8*jnp.cos(kx*x + ky*y + kz*z)
        return jnp.stack([phi, sign*kappa*phi])
    state = jr.initialize(ic, params)
    stepper, scheme = get_scheme("lsrk54")
    end = _advance(state, kgrid, params, nsteps, scheme, stepper)
    f0 = np.asarray(state.fields)[:, iz, ikx, iky].astype(complex)
    f1 = np.asarray(end.fields)[:, iz, ikx, iky].astype(complex)
    other = abs(kappa*f1[0] - sign*f1[1])/abs(kappa*f1[0] + sign*f1[1])
    # everything else stays empty: the single stored coefficient carries the whole state
    leak = np.asarray(end.fields).copy()
    leak[:, iz, ikx, iky] = 0.0
    return dict(ratio=f1/f0, T=float(end.t), other=other, kz=kz, ksq=ksq, ref=ref,
                leak=float(np.abs(leak).max()/np.abs(f1).max()))


def test_kaw_dispersion_exact():
    # fp32 floors (measured): phase rate 3e-6, Theta^- contamination 3e-6
    tol_w = 1e-11 if FP64 else 2e-5
    tol_v = 1e-12 if FP64 else 2e-5
    with checks() as c:
        for Z, tau in ((1.0, 0.5), (1.0, 1.0)):
            for kr in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
                r = _kaw_case(kr, Z, tau)
                want = r["kz"]*r["ref"]["vph"]
                rate = np.angle(r["ratio"])/r["T"]                # |kz v_ph T| <= 2.5 < pi
                e = np.max(np.abs(rate/want - 1))
                tag = f"Z/tau = {Z/tau:g}, k rho_i = {kr:g} (v_ph = {r['ref']['vph']:.4f})"
                c.check(f"{tag}: phi and psi advance as exp(+i kz v_ph t) (rel {e:.1e})",
                        e <= tol_w, f"rates {rate}, want {want!r}")
                amp = np.max(np.abs(np.abs(r["ratio"]) - 1))
                c.check(f"{tag}: |F(T)/F(0)| = 1 at nu = 0 ({amp:.1e}), Theta^- projection "
                        f"{r['other']:.1e}, nothing else excited ({r['leak']:.1e})",
                        amp <= tol_v and r["other"] <= tol_v and r["leak"] <= tol_v,
                        f"amp {amp:.3e}, other {r['other']:.3e}, leak {r['leak']:.3e}")
        # the sign: Theta^- runs the other way
        for kr in (0.5, 5.0):
            r = _kaw_case(kr, 1.0, 0.5, sign=-1)
            want = -r["kz"]*r["ref"]["vph"]
            rate = np.angle(r["ratio"])/r["T"]
            e = np.max(np.abs(rate/want - 1))
            c.check(f"Theta^- at k rho_i = {kr:g}: advances as exp(-i kz v_ph t) "
                    f"(rel {e:.1e}), Theta^+ projection {r['other']:.1e}",
                    e <= tol_w and r["other"] <= tol_v, f"rates {rate}, want {want!r}")
        # a non-cubic box, every wavenumber component != 0 and no 2 pi/L equal to 1
        box = (TWO_PI*1.3, TWO_PI*0.7, TWO_PI*2.5)
        for kr, mode in ((1.0, (2, 3, 1)), (3.0, (-3, 1, 2))):
            r = _kaw_case(kr, 1.0, 0.5, box=box, mode=mode, dt=0.02)
            want = r["kz"]*r["ref"]["vph"]
            rate = np.angle(r["ratio"])/r["T"]
            e = np.max(np.abs(rate/want - 1))
            c.check(f"non-cubic box, mode {mode}, k rho_i = {kr:g}: omega = kz v_ph = "
                    f"{want:.5f} (rel {e:.1e}), Theta^- projection {r['other']:.1e}",
                    e <= tol_w and r["other"] <= tol_v, f"rates {rate}, want {want!r}")
        # dissipation: decay rate nu k^2h + nu_z kz^(2 h_z), the wave unchanged
        box = (TWO_PI, TWO_PI, 3.0)                                # kz = 2 pi/3: kz^4 != kz^6
        for z_hyper in (2, 3):
            for hyper in (1, 2):
                nu, nuz = 0.02, 0.03
                r = _kaw_case(1.0, 1.0, 0.5, box=box, diss=nu, hyper=hyper, zdk=nuz,
                              z_hyper=z_hyper)
                want = nu*r["ksq"]**hyper + nuz*r["kz"]**(2*z_hyper)
                got = -np.log(np.abs(r["ratio"]))/r["T"]
                e = np.max(np.abs(got - want))/want
                wrong = nu*r["ksq"]**hyper + nuz*r["kz"]**(2*(5 - z_hyper))
                rate = np.angle(r["ratio"])/r["T"]
                ew = np.max(np.abs(rate/(r["kz"]*r["ref"]["vph"]) - 1))
                c.check(f"hyper = {hyper}, z_hyper = {z_hyper}: decay rate {want:.5f} "
                        f"(rel {e:.1e}; the other z_hyper would give {wrong:.3f}), omega "
                        f"still kz v_ph (rel {ew:.1e})",
                        e <= (1e-12 if FP64 else 2e-5) and ew <= tol_w, f"got {got}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
