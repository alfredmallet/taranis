# FLR-MHD rung F1 (plans/FLRMHD_PLAN.md section 5), the nonlinear gates:
#   F1.6  semi-discrete conservation, exact: on a broadband state spanning k rho_i from 0.2
#         to 3 the jitted nonlinear RHS changes neither E nor H -- |sum_k T|/sum_k|T| at
#         round-off -- with the quadratic-form weights (G pe, ksq, G) built HERE from the
#         quadrature Gamma_0, not taken from the module. Plus the linear operator's rates:
#         zero at nu = 0, and the docs' dissipation formulas at nu > 0.
#   F1.6b the finite-rho_i nonlinear RHS against a transcription of the paper's equations in
#         the PAPER's variables (dn_e/n_0e, A_par, varphi; Gaussian units, CGS coefficients,
#         u.grad and b.grad rather than brackets), 1e-12. Without this the finite-rho_i
#         nonlinear term is pinned only by conservation plus the two limits (plan section 14).
#   F1.7  time-discrete conservation: the drift of E and of H over a fixed time converges at
#         an order in [p - 0.4, p + 1.4] about the scheme order p -- measured ~3 for lsrk33
#         and ~5 for rk44 and lsrk54 at these dt, which is PRE-asymptotic for the two
#         fourth-order schemes (the asymptotic order is p; see the test) -- and is below
#         1e-6 at the finer dt.
#   F1.8  the invariants on known states: a pure Theta^+- single mode has H = +-E/v_ph; at
#         rho_i = 0, E and H are diagnostics.energy and the RMHD cross-helicity; the per-mode
#         inequality v_ph |H_k| <= E_k.
# E, H and their rates are the test-local formulas of tests/_flrmhd_testing.py.
#
# F1.6, F1.6b and F1.8 run in both precision sessions (fp32 tolerances are the measured
# floors); F1.7 is fp64 only -- the drifts it measures sit below the fp32 round-off floor.
# pytest: `pytest tests/test_flrmhd_conservation.py`. Script:
# `python tests/test_flrmhd_conservation.py` -- never under mpirun (single-process configs).
from _rmhd_testing import bootstrap, checks

bootstrap()

import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision
from taranis import diagnostics as dg
from taranis.physics import construct_rhs, equation_registry, shared_physics
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

from _flrmhd_testing import (broadband_ic, flr_ref, invariants, kgrid_ksq, norm,
                             one_minus_gamma0_ref, rates, weights, yfac)

FP64 = _precision.precision == "64"
TWO_PI = 2.0*np.pi

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _flr(rho_i, tau=0.5, Z=1.0, diss=(0.0, 0.0), hyper=1, eq=None, **kw):
    eqpars = dict(rho_i=rho_i, tau=tau, Z=Z, diss=diss, hyper=hyper)
    eqpars.update(eq or {})
    base = dict(nx=64, ny=64, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, dt=0.01,
                adaptive_timestep=False, dims=2, eqtype="FLRMHD", eqpars=eqpars)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(**base)


_Z3 = dict(dims=3, nx=32, ny=32, nz=8, Lz=TWO_PI, z_spectral=True)

# (label, Parameters kwargs, rho_i, Z, tau): k rho_i spans 0.15-3.2 (64^2, dealiased
# k_max = 21.3) and 0.28-3.0 (32^2, k_max = 10.7)
_CASES = (("2D 64^2, Z/tau = 2", {}, 0.15, 1.0, 0.5),
          ("2D 64^2, Z/tau = 2.86", {}, 0.15, 2.0, 0.7),
          ("z_spectral 32^2 x 8, Z/tau = 2", _Z3, 0.28, 1.0, 0.5),
          ("z_spectral 32^2 x 8, Z/tau = 0.5", _Z3, 0.28, 1.0, 2.0))


def _rhs(params, kgrid, state):
    return construct_rhs(equation_registry["FLRMHD"])(state, kgrid, params)[0]


def _correlated(ic, mix=0.6):
    # psi <- mix*phi + psi: gives the state a helicity comparable to E/v_ph
    def f(*xyz):
        a = ic(*xyz)
        return np.stack([a[0], mix*a[0] + a[1]])
    return f


# ---------------------------------------------------------------------------- gate F1.6

def test_nonlinear_term_conserves_energy_and_helicity():
    tol = 1e-13 if FP64 else 2e-5
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            params = _flr(rho_i, tau=tau, Z=Z, **kw)
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_correlated(broadband_ic(params, seed=7)), params)
            f = np.asarray(state.fields)
            krho = np.sqrt(kgrid_ksq(params)[np.abs(f[0]).max(axis=0) > 0])*rho_i
            N = np.asarray(jax.jit(lambda s, kg: _rhs(params, kg, s))(state, kgrid))
            TE, TH = rates(f, N, params, rho_i, Z/tau)
            E, H = invariants(f, params, rho_i, Z/tau)
            eE = abs(TE.sum())/np.abs(TE).sum()
            eH = abs(TH.sum())/np.abs(TH).sum()
            c.check(f"{label}: sum_k T_E / sum_k |T_E| = {eE:.1e} (k rho_i in "
                    f"[{krho[krho > 0].min():.2f}, {krho.max():.2f}])", eE <= tol, f"{eE:.3e}")
            c.check(f"{label}: sum_k T_H / sum_k |T_H| = {eH:.1e}", eH <= tol, f"{eH:.3e}")
            # the state is far from trivial: each transfer is O(1) against its invariant
            c.check(f"{label}: transfers are live (sum|T_E|/E = {np.abs(TE).sum()/E:.2f}, "
                    f"sum|T_H|/|H| = {np.abs(TH).sum()/abs(H):.2f}, H v_ph/E ~ "
                    f"{abs(H)/E:.2f})",
                    np.abs(TE).sum() > 0.1*E and np.abs(TH).sum() > 0.1*abs(H)
                    and abs(H) > 0.05*E)
            # the RMHD weights do NOT close on this state: the FLR weights are load-bearing
            TE0, TH0 = rates(f, N, params, 0.0, Z/tau)
            c.check(f"{label}: with RMHD weights (G = ksq, pe = 1) the sums do not vanish "
                    f"({abs(TE0.sum())/np.abs(TE0).sum():.1e}, "
                    f"{abs(TH0.sum())/np.abs(TH0).sum():.1e})",
                    abs(TE0.sum())/np.abs(TE0).sum() > 1e-3
                    and abs(TH0.sum())/np.abs(TH0).sum() > 2e-4)   # measured 1.2e-3 .. 7.8e-3


def test_linear_operator_rates():
    """The z_spectral L: at nu = 0 it changes neither E nor H (the waves are unitary in the
    FLR norm); at nu > 0 its rates are the docs' dissipation formulas
        eps_perp = sum k^2h [nu_phi G pe |phi|^2 + nu_psi ksq |psi|^2],  eps_z = nu_z sum kz^2hz 2 E_k
        epsH_perp = (nu_phi + nu_psi) sum k^2h H_k,                       epsH_z = 2 nu_z sum kz^2hz H_k."""
    tol = 1e-13 if FP64 else 2e-5
    with checks() as c:
        for Z, tau in ((1.0, 0.5), (1.0, 2.0)):
            rho_i = 0.28
            params = _flr(rho_i, tau=tau, Z=Z, **_Z3)
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_correlated(broadband_ic(params, seed=8)), params)
            f = np.asarray(state.fields)
            Lf = np.asarray(kgrid.lin.apply_L(state.fields))
            TE, TH = rates(f, Lf, params, rho_i, Z/tau)
            # the wave terms cancel WITHIN each mode (T_E(k) = T_H(k) = 0), so the scale is
            # the size of the cancelling pieces, not sum|T|
            w = weights(params, rho_i, Z/tau)
            sE = np.sum(w["yfac"]*(w["Gpe"]*np.abs(f[0]*Lf[0]) + w["ksq"]*np.abs(f[1]*Lf[1]))
                        )/norm(params)
            sH = np.sum(w["yfac"]*w["G"]*(np.abs(f[0]*Lf[1]) + np.abs(f[1]*Lf[0])))/norm(params)
            eE, eH = np.abs(TE).sum()/sE, np.abs(TH).sum()/sH
            c.check(f"Z/tau = {Z/tau:g}, nu = 0: L changes neither E_k ({eE:.1e}) nor H_k "
                    f"({eH:.1e}) at any mode (sum_k |T(k)| against the cancelling pieces)",
                    eE <= tol and eH <= tol, f"{eE:.3e}, {eH:.3e}")
            # with the RMHD weights the same sums are O(1): a and pe are load-bearing
            TE0, TH0 = rates(f, Lf, params, 0.0, Z/tau)
            c.check(f"Z/tau = {Z/tau:g}, nu = 0: with RMHD weights E_k is NOT conserved "
                    f"({np.abs(TE0).sum()/sE:.1e}; H_k cancels for any real weight)",
                    np.abs(TE0).sum() > 1e-2*sE)

            nu, hyper, nuz, hz = (0.02, 0.05), 2, 0.5, 3
            pd = _flr(rho_i, tau=tau, Z=Z, diss=nu, hyper=hyper,
                      eq=dict(z_diss_k=nuz, z_hyper=hz), **_Z3)
            kd = jr.setup_kgrids(pd)
            Lf = np.asarray(kd.lin.apply_L(state.fields))
            TE, TH = rates(f, Lf, pd, rho_i, Z/tau)
            w = weights(pd, rho_i, Z/tau)
            kz = (np.fft.fftfreq(pd.nz)*pd.nz*TWO_PI/pd.Lz)[:, None, None]
            kz[pd.nz//2] = 0.0
            k2h = w["ksq"]**hyper
            Ek = 0.5*w["yfac"]*(w["Gpe"]*np.abs(f[0])**2 + w["ksq"]*np.abs(f[1])**2)
            Hk = w["yfac"]*w["G"]*np.real(np.conj(f[0])*f[1])
            eps_perp = np.sum(w["yfac"]*k2h*(nu[0]*w["Gpe"]*np.abs(f[0])**2
                                             + nu[1]*w["ksq"]*np.abs(f[1])**2))/norm(pd)
            eps_z = nuz*np.sum(kz**(2*hz)*2*Ek)/norm(pd)
            epsH_perp = (nu[0] + nu[1])*np.sum(k2h*Hk)/norm(pd)
            epsH_z = 2*nuz*np.sum(kz**(2*hz)*Hk)/norm(pd)
            eE = abs(TE.sum() + eps_perp + eps_z)/(eps_perp + eps_z)
            eH = abs(TH.sum() + epsH_perp + epsH_z)/(abs(epsH_perp) + abs(epsH_z))
            c.check(f"Z/tau = {Z/tau:g}, nu > 0: dE/dt = -(eps_perp + eps_z) (rel {eE:.1e}; "
                    f"eps_z/eps_perp = {eps_z/eps_perp:.2f})",
                    eE <= (1e-12 if FP64 else 2e-5) and eps_z > 0.05*eps_perp, f"{eE:.3e}")
            c.check(f"Z/tau = {Z/tau:g}, nu > 0: dH/dt = -(epsH_perp + epsH_z) (rel {eH:.1e})",
                    eH <= (1e-12 if FP64 else 2e-5), f"{eH:.3e}")


# --------------------------------------------------------------------------- gate F1.6b

# CGS constants and an arbitrary but realistic low-beta background (Gaussian units)
_E_ESU = 4.80320471e-10          # statC
_C_CGS = 2.99792458e10           # cm/s
_M_P = 1.67262192e-24            # g
_B0 = 5.0e-4                     # G   (50 nT)
_N0I = 3.0                       # cm^-3
_L0 = 2.0e7                      # cm: the code's length unit (box length = Lx * L0)
_A_ION = 4.0                     # ion mass number (with Z = 2: an alpha-particle plasma)


def _paper_nonlinear_rhs(fields, params, rho_i_code, Z, tau):
    """d(phi, psi)/dt from the NONLINEAR terms of the paper's Eqs. (2.1)-(2.3), evaluated in
    the paper's variables and Gaussian units, then mapped back to code variables:

        (d/dt + u.grad) dn_e/n_0e = -(c/(4 pi e n_0e)) (d/dz + b.grad) lap A_par
        (d/dt + u.grad) A_par     = -c d(varphi)/dz + (c T_0e/e) (d/dz + b.grad) dn_e/n_0e
        dn_e/n_0e                 = -(Z/tau) (1 - Gamma_0) e varphi/T_0e
        u = (c/B_0) zhat x grad varphi,   b = -(1/B_0) zhat x grad A_par

    Code -> Gaussian: varphi = (B_0/c) v_A L_0 phi, A_par = -sqrt(4 pi m_i n_0i) v_A L_0 psi,
    x = L_0 x_code, t = (L_0/v_A) t_code. Pseudo-spectral in numpy with the solver's 2/3
    mask. Nothing here is shared with taranis/physics/flrmhd.py."""
    e, cl, B0, n0i, L0 = _E_ESU, _C_CGS, _B0, _N0I, _L0
    m_i = _A_ION*_M_P
    n0e = Z*n0i
    vA = B0/np.sqrt(4*np.pi*n0i*m_i)                        # cm/s
    rho_i = rho_i_code*L0                                   # cm
    T0i = (rho_i*Z*e*B0/cl)**2/(2*m_i)                      # erg: rho_i = c sqrt(2 m_i T_0i)/(Z e B_0)
    T0e = T0i/tau
    nx, ny = params.nx, params.ny
    zsp = params.z_spectral
    nz = params.nz if zsp else 1
    kx = (np.fft.fftfreq(nx)*nx*TWO_PI/params.Lx/L0)[:, None]     # 1/cm
    ky = (np.fft.rfftfreq(ny)*ny*TWO_PI/params.Ly/L0)[None, :]
    k2 = kx**2 + ky**2
    alpha = 0.5*k2*rho_i**2                                 # dimensionless
    # (the k = 0 placeholder is set on the DIMENSIONLESS alpha: k2 is in 1/cm^2 here)
    omg = np.where(k2 > 0, one_minus_gamma0_ref(np.where(k2 > 0, alpha, 1.0)), 0.0)

    def real(fk):
        if zsp:
            return np.fft.irfftn(fk, s=(nz, nx, ny), axes=(0, 1, 2))
        return np.fft.irfft2(fk, s=(nx, ny), axes=(1, 2))

    def spec(fr):
        if zsp:
            return np.fft.rfftn(fr, axes=(0, 1, 2))
        return np.fft.rfft2(fr, axes=(1, 2))

    phi_k, psi_k = np.asarray(fields[0], dtype=complex), np.asarray(fields[1], dtype=complex)
    varphi_k = (B0/cl)*vA*L0*phi_k                           # statV
    A_k = -np.sqrt(4*np.pi*m_i*n0i)*vA*L0*psi_k              # G cm
    n_k = -(Z/tau)*omg*e*varphi_k/T0e                        # dn_e/n_0e
    lapA_k = -k2*A_k

    dx = lambda fk: real(1j*kx*fk)
    dy = lambda fk: real(1j*ky*fk)
    ux, uy = -(cl/B0)*dy(varphi_k), (cl/B0)*dx(varphi_k)     # u = (c/B_0) zhat x grad varphi
    bx, by = (1.0/B0)*dy(A_k), -(1.0/B0)*dx(A_k)             # b = -(1/B_0) zhat x grad A_par
    adv = lambda vx, vy, fk: vx*dx(fk) + vy*dy(fk)
    dn_dt = -adv(ux, uy, n_k) - (cl/(4*np.pi*e*n0e))*adv(bx, by, lapA_k)
    dA_dt = -adv(ux, uy, A_k) + (cl*T0e/e)*adv(bx, by, n_k)

    ix = np.fft.fftfreq(nx)*nx
    iy = np.fft.rfftfreq(ny)*ny
    mask = ((ix[:, None]/(nx/3.0))**2 + (iy[None, :]/(ny/3.0))**2 < 1.0)[None]
    if zsp:
        mask = mask & (np.abs(np.fft.fftfreq(nz)*nz) < nz/3.0)[:, None, None]
    dn_k, dA_k = spec(dn_dt)*mask, spec(dA_dt)*mask
    # invert quasineutrality for d(varphi)/dt, then back to code units (time unit L_0/v_A)
    dvarphi_k = np.where(k2 > 0, -(tau*T0e/(Z*e))*dn_k/np.where(k2 > 0, omg, 1.0), 0.0)
    tunit = L0/vA
    dphi = (cl/B0)*dvarphi_k/(vA*L0)*tunit
    dpsi = -dA_k/(np.sqrt(4*np.pi*m_i*n0i)*vA*L0)*tunit
    return np.stack([dphi, dpsi])


def test_nonlinear_rhs_matches_the_paper_variable_transcription():
    tol = 1e-12 if FP64 else 2e-5
    with checks() as c:
        for label, kw, rho_i, Z, tau in (("2D 64^2", {}, 0.3, 2.0, 0.7),
                                         ("2D 64^2 non-square box",
                                          dict(nx=48, Lx=TWO_PI*1.4, Ly=TWO_PI*0.8), 0.2, 1.0,
                                          0.5),
                                         ("z_spectral 32^2 x 8", _Z3, 0.5, 2.0, 0.7)):
            params = _flr(rho_i, tau=tau, Z=Z, **kw)
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_correlated(broadband_ic(params, seed=9)), params)
            got = np.asarray(_rhs(params, kgrid, state))
            want = _paper_nonlinear_rhs(np.asarray(state.fields), params, rho_i, Z, tau)
            for i, name in enumerate(("d phi/dt", "d psi/dt")):
                e = np.linalg.norm(got[i] - want[i])/np.linalg.norm(want[i])
                c.check(f"{label} (rho_i = {rho_i}, Z = {Z:g}, tau = {tau:g}): {name} == the "
                        f"paper-variable transcription (rel {e:.1e})", e <= tol, f"{e:.3e}")
            # the FLR terms are a large part of this RHS: against the same transcription at
            # rho_i -> 0 (RMHD) the difference is O(1)
            rmhd_like = _paper_nonlinear_rhs(np.asarray(state.fields), params, 1e-6, Z, tau)
            d = np.linalg.norm(want - rmhd_like)/np.linalg.norm(want)
            c.check(f"{label}: the finite-rho_i terms carry {d:.2f} of the RHS", d > 0.2)


# ---------------------------------------------------------------------------- gate F1.7

def _drifts(schemestr, kw, rho_i, dt, nsteps, seed):
    params = _flr(rho_i, dt=dt, **kw)
    kgrid = jr.setup_kgrids(params)
    ic = _correlated(broadband_ic(params, seed=seed, slope=-3.0, amp=_F17_AMP))
    state = jr.initialize(ic, params)
    E0, H0 = invariants(state.fields, params, rho_i, 2.0)
    stepper, scheme = get_scheme(schemestr)
    end = _advance(state, kgrid, params, nsteps, scheme, stepper)
    E1, H1 = invariants(end.fields, params, rho_i, 2.0)
    moved = float(np.linalg.norm(np.asarray(end.fields) - np.asarray(state.fields))
                  / np.linalg.norm(np.asarray(state.fields)))
    return abs(E1/E0 - 1), abs(H1/H0 - 1), moved


_F17_AMP = 0.5
_F17_STEPS = 40


@pytest.mark.fp64
def test_time_discrete_conservation_order():
    """nu = 0, the same time T = 40 dt run at dt (40 steps) and dt/2 (80 steps): the drifts
    of E and of H converge at an order in [p - 0.4, p + 1.4] about the scheme order p and
    are below 1e-6 at the finer dt. Under z_spectral the linear waves are exact, so the
    drift is the nonlinear term's.

    What is known: a Runge-Kutta scheme of order p conserves a quadratic invariant of a
    NONLINEAR system only to its local order, so the drift over a fixed time is
    asymptotically O(dt^p) -- the extra order (|R|^2 = 1 - (w dt)^6/72 for RK4 on an
    oscillator) is a property of LINEAR skew dynamics only; classical RK4 on the rigid
    body converges to order 4 (the F1 reviewer's counter-example, 2026-11-11).
    What is measured here (2026-11-11, fp64, dt = 0.01 ... 0.00125 at fixed T, E and H,
    2D and z_spectral): lsrk33 2.89-3.10, rk44 4.9-5.05 with no trend over the range,
    lsrk54 4.83-5.15 at the dt of this gate and FALLING under refinement (4.88 -> 4.60 in
    the reviewer's sweep). So the order-5 behaviour of the two fourth-order schemes is
    pre-asymptotic. Interpretation (not established): at these dt the drift is dominated
    by the near-linear part of the error, whose leading term is conservative, and the
    genuinely nonlinear O(dt^4) term has a small constant.
    The window is therefore [p - 0.4, p + 1.4] -- the lower bound is the plan's, the
    upper one admits the pre-asymptotic extra order -- and the absolute bound does the
    rest. The plan's [p - 0.4, p + 0.6] fails for rk44 and lsrk54 at any dt this gate can
    afford above the round-off floor."""
    with checks() as c:
        # dt per case: large enough that the finer drift stays two decades above round-off
        # (1e-12 is asserted as the floor)
        for label, kw, rho_i, dt in (("2D 64^2", {}, 0.15, 0.004),
                                     ("z_spectral 32^2 x 8", _Z3, 0.28, 0.008)):
            for name, p in (("lsrk33", 3), ("rk44", 4), ("lsrk54", 4)):
                dE1, dH1, moved = _drifts(name, kw, rho_i, dt, _F17_STEPS, seed=12)
                dE2, dH2, _ = _drifts(name, kw, rho_i, dt/2, 2*_F17_STEPS, seed=12)
                oE, oH = np.log2(dE1/dE2), np.log2(dH1/dH2)
                c.check(f"{label} {name}: E drift {dE1:.2e} -> {dE2:.2e}, order {oE:.2f} "
                        f"in [p - 0.4, p + 1.4], p = {p}; state moved by {moved:.2f}",
                        p - 0.4 <= oE <= p + 1.4 and 1e-12 < dE2 <= 1e-6
                        and moved > 0.03, f"order {oE:.3f}, fine drift {dE2:.3e}")
                c.check(f"{label} {name}: H drift {dH1:.2e} -> {dH2:.2e}, order {oH:.2f} "
                        f"in [p - 0.4, p + 1.4], p = {p}",
                        p - 0.4 <= oH <= p + 1.4 and 1e-12 < dH2 <= 1e-6,
                        f"order {oH:.3f}, fine drift {dH2:.3e}")


# ---------------------------------------------------------------------------- gate F1.8

def test_invariants_on_known_states():
    tol = 1e-13 if FP64 else 2e-6
    with checks() as c:
        # a pure Theta^+- single mode: H = +-E/v_ph(k). The state is built with the MODULE's
        # kappa, E and H with the test-local weights, v_ph from the paper's closed form.
        for Z, tau in ((1.0, 0.5), (2.0, 0.7)):
            for rho_i in (0.05, 0.4, 2.0):
                params = _flr(rho_i, tau=tau, Z=Z, **dict(_Z3, nx=16, ny=16))
                kgrid = jr.setup_kgrids(params)
                ikx, iky, iz = 2, 3, 1
                kappa = float(np.asarray(kgrid.eq.kappa)[ikx, iky])
                vph = float(flr_ref(np.array([ikx**2 + iky**2]), rho_i, Z/tau)["vph"][0])
                for sign in (+1, -1):
                    def ic(x, y, z):
                        phi = jnp.cos(ikx*x + iky*y + iz*z + 0.3)
                        return jnp.stack([phi, sign*kappa*phi])
                    f = np.asarray(jr.initialize(ic, params).fields)
                    E, H = invariants(f, params, rho_i, Z/tau)
                    e = abs(H*vph/E - sign)
                    c.check(f"Z/tau = {Z/tau:.3g}, k rho_i = {np.sqrt(13)*rho_i:.2f}: pure "
                            f"Theta^{'+' if sign > 0 else '-'} has H v_ph/E = {sign:+d} "
                            f"(v_ph = {vph:.4f}, err {e:.1e})", e <= tol, f"{e:.3e}")
                    # E in closed form: phi = cos(...) has <|grad phi|^2>/2 = k^2/4, so
                    # E = (kappa^2 + kappa^2) k^2/4 = kappa^2 k^2/2
                    want = 0.5*kappa**2*13.0
                    c.check(f"   and E = kappa^2 k^2/2 = {want:.5f} (rel {abs(E/want - 1):.1e})",
                            abs(E/want - 1) <= (1e-13 if FP64 else 2e-6))
        # rho_i = 0: E and H are the RMHD energy and cross-helicity
        for label, kw in (("2D", dict(nx=32, ny=32)), ("z_spectral", _Z3)):
            params = _flr(0.0, **kw)
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_correlated(broadband_ic(params, seed=4)), params)
            E, H = invariants(state.fields, params, 0.0, 2.0)
            Ed = float(sum(dg.energy(state, kgrid, params)))
            Hd = float(shared_physics.perp_inner_product(state.fields[0], state.fields[1],
                                                         kgrid, params))
            c.check(f"{label} rho_i = 0: E == diagnostics.energy summed "
                    f"(rel {abs(E/Ed - 1):.1e})", abs(E/Ed - 1) <= (1e-14 if FP64 else 2e-6))
            c.check(f"{label} rho_i = 0: H == <grad phi . grad psi> = "
                    f"perp_inner_product(phi, psi) (rel {abs(H/Hd - 1):.1e})",
                    abs(H/Hd - 1) <= (1e-14 if FP64 else 2e-6))
        # the per-mode inequality v_ph |H_k| <= E_k on a broadband state
        rho_i, zt = 0.3, 2.0
        params = _flr(rho_i)
        f = np.asarray(jr.initialize(_correlated(broadband_ic(params, seed=6)), params).fields)
        w = weights(params, rho_i, zt)
        vph = flr_ref(w["ksq"], rho_i, zt)["vph"]
        Ek = 0.5*(w["Gpe"]*np.abs(f[0])**2 + w["ksq"]*np.abs(f[1])**2)
        Hk = w["G"]*np.real(np.conj(f[0])*f[1])
        slack = np.max(vph*np.abs(Hk) - Ek)/np.max(Ek)
        c.check(f"v_ph |H_k| <= E_k at every mode (max violation {slack:.1e} of max E_k)",
                slack <= 1e-12)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
