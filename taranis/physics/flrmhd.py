# FLR-MHD (Meyrand, Squire, Schekochihin & Dorland 2021, JPP 87, 535870301): the low-beta
# finite-Larmor-radius extension of RMHD, valid on both sides of k_perp*rho_i ~ 1.
# Derivation of record: docs/numerics.md "FLR-MHD"; plan: plans/FLRMHD_PLAN.md.
#
# Fields are (phi, psi), the SAME variables and bracket conventions as the RMHD recipe
# (u = zhat x grad phi, b_perp = zhat x grad psi, Alfven units, A_z = -psi). With the
# electron density in vorticity units nhat = Omega_i*dn_e/n_0e,
#     nhat_k = -G(k_perp) phi_k,   G = (2/rho_i^2)(1 - Gamma_0(k_perp^2 rho_i^2/2))
#     dt nhat = -{phi_e, nhat} + {psi, lap psi} + dz lap psi
#     dt psi  = -{phi_e, psi} + dz phi_e,       phi_e = phi - rho_s^2 nhat = pe*phi
# rho_s^2 = Z rho_i^2/(2 tau). That is RMHD with three k-local substitutions: vorticity
# lap phi -> nhat, advecting potential phi -> phi_e in both brackets, inv_ksq -> 1/G.
# The k-local factors live in kgrid.eq (FLRFactors), built once by `factors`, the recipe's
# kgrid_func. eqpars["rho_i"] = 0 is a trace-time branch that IS RMHD.
#
# Linear part (dt f = L f + N): dims=2 -> the dissipation diagonal, as 2D RMHD; z_spectral ->
#     L = [[d_phi, i kz a], [i kz pe, d_psi]],  d_f = -nu_f k_perp^(2 hyper) - z_diss_k kz^(2 z_hyper)
# a dense 2x2 (the putzer2 backend), omega = +-kz*v_ph(k_perp), v_ph^2 = a*pe.
#
# Forcing (rung F3): the RMHD O-U machinery on chi = kappa*phi (momentum) or on
# Theta^+- = kappa*phi +- psi (elsasser) -- in (chi, psi) the FLR energy is RMHD's, so the
# power normalization is rmhd's, unchanged; d_t phi gets f_chi/kappa.
#
# Supported: dims=2, and dims=3 with z_spectral=True (single process). FD-z is plan rung F6
# and raises NotImplementedError until then. CB-IMEX on the z_spectral wave L is rejected
# (check_scheme).
#
# This module holds only what the solver consumes; observers go in diagnostics/flrmhd.py.
import functools
import warnings
from fractions import Fraction
from math import factorial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import i0e

from .. import comms, grids
from .. import _precision
from ..overrides import getp
from . import gdi, rmhd
from . import shared_physics
from .shared_physics import bracket, grad_fields


class FLRGrads(NamedTuple):
    # one real-space (d/dx, d/dy) pair per bracket operand, each (2,nz,nx,ny)
    gphie: jnp.ndarray      # advecting potential phi_e = pe*phi
    gpsi: jnp.ndarray
    gn: jnp.ndarray         # density in vorticity units, nhat = -G*phi
    gjpar: jnp.ndarray      # parallel current, -ksq*psi


class FLRFactors(NamedTuple):
    # k-local FLR factors, each (nkx,nky) real at field precision (docs/numerics.md)
    G: jnp.ndarray          # (2/rho_i^2)(1 - Gamma_0): nhat = -G*phi.   -> ksq as rho_i -> 0
    inv_G: jnp.ndarray      # 1/G, 0 at k_perp = 0
    pe: jnp.ndarray         # 1 + rho_s^2 G: phi_e = pe*phi
    a: jnp.ndarray          # ksq/G: L's [phi,psi] factor
    kappa: jnp.ndarray      # sqrt(G*pe)/k_perp: Theta^+- = kappa*phi +- psi
    vph: jnp.ndarray        # k_perp*sqrt(pe/G) = sqrt(a*pe): omega = +-kz*vph


_EQPARS_REQUIRED = ("rho_i", "tau", "diss", "hyper")
_EQPARS_OPTIONAL = ("Z", "z_diss_k", "z_hyper")


def _check_supported(params):
    if params.spatial_dimensions == 3 and not params.z_spectral:
        raise NotImplementedError("FLRMHD dims==3 requires z_spectral=True: the "
                                  "finite-difference-z kinetic-Alfven term is not built "
                                  "(plans/FLRMHD_PLAN.md rung F6)")


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _eqpars(params):
    # pull & validate the FLRMHD eqpars: (rho_i, tau, Z, diss, hyper, z_diss_k, z_hyper).
    # Unknown keys are rejected.
    _check_supported(params)
    ep = params.eqpars
    missing = [k for k in _EQPARS_REQUIRED if k not in ep]
    unknown = [k for k in ep if k not in _EQPARS_REQUIRED + _EQPARS_OPTIONAL]
    if missing or unknown:
        raise ValueError(f"FLRMHD eqpars problem (missing {missing}, unknown {unknown}): "
                         f"required {_EQPARS_REQUIRED}, optional {_EQPARS_OPTIONAL}; "
                         f"got eqpars={ep!r}")
    rho_i, tau, Z = ep["rho_i"], ep["tau"], ep.get("Z", 1.0)
    for name, v in (("rho_i", rho_i), ("tau", tau), ("Z", Z)):
        if np.shape(v) != () or isinstance(v, (bool, np.bool_)) or not np.isfinite(v):
            raise ValueError(f"FLRMHD eqpars[{name!r}] must be a finite real scalar, got {v!r}")
    if rho_i < 0:
        raise ValueError(f"FLRMHD eqpars['rho_i'] must be >= 0 (0 is RMHD), got {rho_i!r}")
    if tau <= 0 or Z <= 0:
        raise ValueError(f"FLRMHD eqpars need tau = T_0i/T_0e > 0 and Z > 0, got "
                         f"tau={tau!r}, Z={Z!r}")
    diss = ep["diss"]
    if np.shape(diss) not in ((), (1,), (params.nfields,)):
        raise ValueError(f"FLRMHD eqpars['diss'] must be a scalar (applied to both fields) or "
                         f"a (nu_phi, nu_psi) pair, got {diss!r}")
    if not np.all(np.asarray(diss, dtype=float) >= 0):
        raise ValueError(f"FLRMHD eqpars['diss'] must be >= 0, got {diss!r}")
    hyper, z_hyper = ep["hyper"], ep.get("z_hyper", 2)
    for name, v in (("hyper", hyper), ("z_hyper", z_hyper)):
        if not _is_int(v) or v < 1:
            raise ValueError(f"FLRMHD eqpars[{name!r}] must be an int >= 1, got {v!r}")
    z_diss_k = ep.get("z_diss_k", 0.0)
    if np.shape(z_diss_k) != () or not z_diss_k >= 0:
        raise ValueError(f"FLRMHD eqpars['z_diss_k'] must be a scalar >= 0, got {z_diss_k!r}")
    if z_diss_k and not params.z_spectral:
        raise ValueError("eqpars['z_diss_k'] is a spectral-z knob (-z_diss_k*kz^(2*z_hyper)) "
                         "and needs z_spectral=True")
    return rho_i, tau, Z, diss, hyper, z_diss_k, z_hyper


def check_scheme(params, scheme):
    # the recipe's scheme_check_func, called by run.py's drivers with the scheme about to
    # run: CB-IMEX solves ALL of L implicitly, and an L-stable solve damps the kinetic Alfven
    # waves of the z_spectral L at |kz v_ph| dt >~ 1 (the standing no-IMEX-on-a-wave-L rule;
    # this L is stiffer than RMHD's by v_ph). 2D's L is the dissipation diagonal: allowed.
    from ..timestepping import IMEX_Scheme   # local: keeps physics free of the steppers
    if params.z_spectral and isinstance(scheme, IMEX_Scheme):
        raise ValueError("FLRMHD + z_spectral: the CB-IMEX schemes are rejected -- the linear "
                         "operator carries the kinetic Alfven waves (omega = kz*v_ph), which "
                         "an L-stable implicit solve damps artificially. Use an "
                         "integrating-factor scheme (lsrk54, lsrk33, rk44), which applies L "
                         "exactly")


# ------------------------------------------------------------------------- the FLR factors

# S(alpha) = (1 - Gamma_0(alpha))/alpha, Gamma_0 = I_0(alpha) exp(-alpha): everything below
# is a function of S and alpha. 1 - i0e cancels at small alpha (relative error ~ eps/alpha),
# so S has a series branch below _ALPHA_SWITCH:
#     S = sum_{n>=1} (-1)^(n+1) c_n alpha^(n-1),   c_n = (2n)!/(2^n (n!)^3)
#       = 1 - (3/4) alpha + (5/12) alpha^2 - (35/192) alpha^3 + ...
# Measured against 40-digit mpmath (jax 0.10.0): the series is within 2.2e-16 on
# [1e-14, 1] from 24 terms on, the closed form within 2.2e-16 on [1, 1e3].
_ALPHA_SWITCH = 1.0
_NSERIES = 28
_SERIES_COEFFS = tuple(float((-1)**(n + 1)*Fraction(factorial(2*n), 2**n*factorial(n)**3))
                       for n in range(1, _NSERIES + 1))


def _S_series(alpha):
    out = jnp.zeros_like(alpha)
    for c in _SERIES_COEFFS[::-1]:
        out = out*alpha + c
    return out


def _S_direct(alpha):
    return (1.0 - i0e(alpha))/alpha


def S_of_alpha(alpha):
    # (1 - Gamma_0(alpha))/alpha for alpha >= 0, float64. Each branch is evaluated on an
    # argument clamped into its own range, so the unselected one is finite (and so is its
    # tangent, which the overrides seam differentiates through).
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    small = alpha < _ALPHA_SWITCH
    return jnp.where(small, _S_series(jnp.where(small, alpha, _ALPHA_SWITCH)),
                     _S_direct(jnp.where(small, _ALPHA_SWITCH, alpha)))


def one_minus_gamma0(alpha):
    # 1 - Gamma_0(alpha) = alpha*S(alpha), float64, without the small-alpha cancellation
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    return alpha*S_of_alpha(alpha)


def _factors_f64(ksq, inv_ksq, rho_i, z_over_tau):
    # the six factors in float64 from S and alpha alone -- never as products of separately
    # rounded factors. k_perp = 0: S = 1, so pe = a = kappa = vph = 1 and G = inv_G = 0.
    ksq = jnp.asarray(ksq, dtype=jnp.float64)
    alpha = 0.5*ksq*rho_i**2
    S = S_of_alpha(alpha)
    pe = 1.0 + z_over_tau*alpha*S           # rho_s^2 G = (Z/tau)(1 - Gamma_0)
    return FLRFactors(G=ksq*S, inv_G=jnp.asarray(inv_ksq, dtype=jnp.float64)/S, pe=pe,
                      a=1.0/S, kappa=jnp.sqrt(S*pe), vph=jnp.sqrt(pe/S))


def _rmhd_limit(params):
    # static: rho_i == 0 is RMHD, a trace-time branch (and rho_i is then not overridable)
    return _eqpars(params)[0] == 0


def _warn_dead_knobs(params, kgrid):
    # z_hyper is the exponent of the z_spectral kz filter -z_diss_k*kz^(2*z_hyper): set away
    # from its default where that filter is absent (2D, or z_diss_k == 0) it does nothing.
    # Warned once per setup_kgrids (rank 0), like Parameters' dead finite-difference-z knobs.
    z_diss_k, z_hyper = _eqpars(params)[5:7]
    if kgrid.overrides is not None and "z_diss_k" in kgrid.overrides:
        return                        # the stability seam supplies the coefficient
    if z_hyper != 2 and params.rank == 0 and not (params.z_spectral and z_diss_k):
        why = ("dims=2 has no kz" if not params.z_spectral
               else "eqpars['z_diss_k'] is 0, so there is no kz filter")
        warnings.warn(f"FLRMHD eqpars['z_hyper']={z_hyper} is stored but IGNORED: it is the "
                      f"exponent of the z_spectral filter -z_diss_k*kz^(2*z_hyper), and "
                      f"{why}.", stacklevel=5)   # the caller of setup_kgrids


def factors(kgrid, params):
    # the recipe's kgrid_func: grids.setup_kgrids stores the result in kgrid.eq
    rho_i, tau, Z = _eqpars(params)[:3]
    _warn_dead_knobs(params, kgrid)
    if _rmhd_limit(params):
        ones = jnp.ones_like(kgrid.ksq)
        return FLRFactors(G=kgrid.ksq, inv_G=kgrid.inv_ksq, pe=ones, a=ones, kappa=ones,
                          vph=ones)
    if kgrid.overrides is not None:
        # the stability harness's seam (taranis/overrides.py): traced rho_i, tau, Z
        rho_i, tau, Z = (getp(params, kgrid.overrides, k) for k in ("rho_i", "tau", "Z"))
        rho_i, tau, Z = (jnp.asarray(v, dtype=jnp.float64) for v in (rho_i, tau, Z))
    f = _factors_f64(kgrid.ksq, kgrid.inv_ksq, rho_i, Z/tau)
    return FLRFactors(*(v.astype(_precision.ftype) for v in f))


@functools.lru_cache(maxsize=32)   # bounded: params is identity-hashed (gdi._max_re_lambda)
def _vph_max(params):
    # max of v_ph over the 2/3-dealiased perpendicular modes: the speed at which a
    # perturbation is carried along the perturbed field, for the CFL (setup-time python
    # float; timestep-only, NOT part of the overrides seam)
    rho_i, tau, Z = _eqpars(params)[:3]
    if rho_i == 0:
        return 1.0
    _, _, ksq, inv_ksq, _, perp_dealias = gdi._perp_grids_np(params)
    # set_timestep calls this while a jit trace is open: the inputs are concrete numpy
    # arrays, and compile-time evaluation keeps the result concrete instead of staging it
    with jax.ensure_compile_time_eval():
        vph = np.asarray(_factors_f64(ksq, inv_ksq, rho_i, Z/tau).vph)
    return float(np.max(np.where(perp_dealias, vph, 1.0)))


# ------------------------------------------------------------------------------ the recipe

def linear_matrix(kgrid, params):
    # dims=2: perp (hyper)dissipation L = -diss*k_perp^(2*hyper), as 2D RMHD.
    # z_spectral: the dense 2x2 with the kinetic Alfven waves (module header).
    _, _, _, diss_par, hyper, zdiss, z_hyper = _eqpars(params)
    if kgrid.overrides is not None:
        # the stability harness's seam (taranis/overrides.py): traced diss / z_diss_k
        diss_par = getp(params, kgrid.overrides, "diss")
        if params.z_spectral:
            zdiss = getp(params, kgrid.overrides, "z_diss_k")
    if not params.z_spectral:
        diss = jnp.array(diss_par, dtype=_precision.ftype).reshape(-1,1,1,1)
        return -diss*kgrid.ksq**hyper
    diss = jnp.broadcast_to(jnp.array(diss_par, dtype=_precision.ftype).reshape(-1),
                            (params.nfields,))
    kz = rmhd._kz_deriv(kgrid,params)
    # (nz,1,1) kz against (nkx,nky) k_perp broadcasts every entry to (nz,nkx,nky)
    dz = zdiss*kz**(2*z_hyper)
    dphi = -diss[0]*kgrid.ksq**hyper - dz
    dpsi = -diss[1]*kgrid.ksq**hyper - dz
    zero = jnp.zeros(dphi.shape, dtype=_precision.ctype)
    return jnp.stack([jnp.stack([dphi + zero, 1j*kz*kgrid.eq.a]),
                      jnp.stack([1j*kz*kgrid.eq.pe, dpsi + zero])])


def grad(state,kgrid,params):
    phik=state.fields[0]
    psik=state.fields[1]
    eq=kgrid.eq
    return FLRGrads(*grad_fields((eq.pe*phik,psik,-eq.G*phik,-kgrid.ksq*psik),kgrid,params))


def set_timestep(grads,params):
    # CFL: a perturbation is advected by u_e = zhat x grad phi_e and carried along the
    # perturbed field at v_ph, so the rate is (|u_e| + v_ph,max |b_perp|)/dx. Under
    # z_spectral the linear waves are exact (no kz limit).
    vmax = _vph_max(params)
    gphie = grads.gphie
    gpsi = grads.gpsi
    max_vy_eff = jnp.max(jnp.abs(gphie[0])+vmax*jnp.abs(gpsi[0]))
    max_vx_eff = jnp.max(jnp.abs(gphie[1])+vmax*jnp.abs(gpsi[1]))
    max_all = jnp.maximum(max_vx_eff/params.dx, max_vy_eff/params.dy)
    max_all = comms.allreduce_max(max_all,params)
    # the velocity floor is the _quiescent_dt ceiling, as in rmhd.set_timestep
    return jnp.minimum(params.cfl_safety / max_all, rmhd._quiescent_dt(params))


def NonlinearTerm(state,grads,kgrid,params,halo=None):
    NLTerm_n = bracket(grads.gpsi,grads.gjpar) - bracket(grads.gphie,grads.gn)
    NLTerm_psi = - bracket(grads.gphie,grads.gpsi)
    (NLTerm_n_k , NLTerm_psi_k) = grids.fft(jnp.stack([NLTerm_n,NLTerm_psi]),params)
    # nhat = -G*phi: dt phi = -(1/G) dt nhat
    return jnp.stack([-kgrid.eq.inv_G*NLTerm_n_k,NLTerm_psi_k])*kgrid.dealias


# --------------------------------------------------------------------------------- forcing

forcing_active = rmhd.forcing_active


def _chi_psi(fields, kgrid):
    # (chi, psi) = (kappa*phi, psi): the variables in which the FLR energy is RMHD's
    # (1/2)<|grad chi|^2 + |grad psi|^2> and Theta^+- = chi +- psi
    return jnp.stack([kgrid.eq.kappa*fields[0], fields[1]])


def forcing_scale(state,kgrid,params,dt):
    # the recipe's forcing_scale_func (run._advance_forcing / _refresh_forcing_scale): RMHD's
    # power normalization on (chi, psi), so P and F2 are the FLR-energy cross and self terms
    f_raw = shared_physics.reconstruct_envelope(state.forcing_state,kgrid,params)
    return rmhd._forcing_scale_from(_chi_psi(state.fields,kgrid),f_raw,kgrid,params,dt)


def ForcingTerm(state,grads,kgrid,params,halo=None):
    # the O-U forcing on chi (momentum) or Theta^+- (elsasser): rmhd.ForcingTerm evaluated on
    # the (chi, psi) state gives (f_chi, f_psi); d_t phi gets f_chi/kappa
    if not forcing_active(params):
        return jnp.zeros_like(state.fields)
    f = rmhd.ForcingTerm(state._replace(fields=_chi_psi(state.fields,kgrid)),grads,kgrid,
                         params,halo)
    return jnp.stack([f[0]/kgrid.eq.kappa,f[1]])


def _forcing_shell_vbar(params):
    # (shell, vbar): the forcing shell -- grids.forcing_shell, i.e. the solver's own
    # kgrid.fmask for this Parameters, so vbar is taken over the modes that are really
    # forced -- and its mean phase speed, 1/vbar = sum_shell w_k/v_ph(k) / sum_shell w_k with
    # the O-U self-energy weights w_k = yfac*ksq. Setup-time numpy (float64 weights).
    rho_i, tau, Z = _eqpars(params)[:3]
    _, _, ksq, inv_ksq, _, _ = gdi._perp_grids_np(params)
    with jax.ensure_compile_time_eval():
        shell = np.asarray(grids.forcing_shell(params))
    if not shell.any():
        raise ValueError(f"elsasser_powers: forcing shell fshell={params.fshell} contains no "
                         f"modes on this grid")
    if rho_i == 0:
        vph = np.ones_like(ksq)
    else:
        with jax.ensure_compile_time_eval():
            vph = np.asarray(_factors_f64(ksq, inv_ksq, rho_i, Z/tau).vph)
    yfac = np.full(ksq.shape[1], 2.0)
    yfac[0] = yfac[-1] = 1.0
    w = (yfac[None, :]*ksq)[shell]
    return shell, float(np.sum(w)/np.sum(w/vph[shell]))


def elsasser_powers(eps, sigma_eps, params):
    # (eps_plus, eps_minus) for forcing_power_elsasser giving total energy injection eps and
    # helicity injection eps_H = sigma_eps*eps:  eps_+- = eps*(1 +- sigma_eps*vbar)/2, with
    # 1/vbar the mean of 1/v_ph over the forcing shell weighted by the O-U self-energy
    # weights w_k = yfac*ksq (docs/numerics.md "Forcing in the FLR energy"). Exact up to the
    # shell's spread of v_ph. Setup-time python floats; reads params.fshell and the box, so
    # any Parameters with this run's grid, shell and eqpars will do (forcing on or off).
    # |sigma_eps| = 1/vbar is the largest imbalance the shell can inject (one channel exactly
    # 0): accepted to within a few ulp of the product, with that channel clamped at 0.
    _, vbar = _forcing_shell_vbar(params)
    x = sigma_eps*vbar
    if abs(x) > 1.0 + 8*np.finfo(float).eps:
        raise ValueError(f"elsasser_powers: |sigma_eps|={abs(sigma_eps)!r} exceeds the largest "
                         f"imbalance this shell can inject, 1/vbar = {1.0/vbar:.6g} (vbar = "
                         f"{vbar:.6g} is the shell-mean phase speed): one of eps_+- would be "
                         f"negative")
    x = min(1.0, max(-1.0, x))
    return (0.5*eps*(1.0 + x), 0.5*eps*(1.0 - x))
