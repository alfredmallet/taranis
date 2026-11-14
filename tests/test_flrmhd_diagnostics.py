# FLR-MHD rung F2 (plans/FLRMHD_PLAN.md sections 3.8 and 5), gates of diagnostics/flrmhd.py:
#   F2.1  energies, helicity and spectra: every entry against the test-local forms (the
#         quadrature-Gamma_0 weights of tests/_flrmhd_testing.py), E_plus + E_minus = total =
#         E_ion + E_e + E_mag, spectra summing to the totals AND matching an independent
#         numpy binning bin by bin, the rho_i = 0 reduction to diagnostics.rmhd, and the
#         non-square-box bin range.
#   F2.2  known states: pure Theta^+- (H = +-E/v_ph, sigma_c = +-1, E_H(k) v_ph = +-E(k)).
#   F2.3  transfers: per-bin against an independent first-principles T (G-form T_H), the
#         two-shell state where T_H(bin) v_ph(shell) = (T_+ - T_-)(bin), sum T = 0.
#   F2.4  the flux against the STEPPER: unforced, nu = 0, the centred difference of E(<K),
#         E^+-(<K), H(<K) over one small step equals -Pi(K) at the midpoint to O(dt^2), six K.
#         Independent of the diagnostic's own formula: catches the flux sign, a cumulative
#         sum from the wrong end, any weight error.
#   F2.5  the sub-filter transfer (kcut=): against a first-principles evaluation, its sum
#         against the full flux at kcut, zero above kcut; and the EDGE convention at the
#         default bin_factor in a square box, where modes sit exactly on kcut.
#   F2.6  dissipation against exact decay: tiny amplitude, nu > 0, nu_z > 0 -- the stepped
#         E(t), H(t) follow the closed-form exponentials, whose time derivative is
#         -(eps_perp + eps_z), -(epsH_perp + epsH_z); unequal (nu_phi, nu_psi) against the
#         docs' formulas with independent weights.
#   F2.7  injection and the budget: injection against independent rates of the solver's
#         forcing; closure at round-off on a forced dissipative state; the run-integrated
#         budget against the measured change of E and H, first order in dt; and the same
#         under per-stage normalization (forcing_norm_per_step=False), where the step
#         injects eps*dt with NO self term unless the scale is capped.
#   F2.8  kperp_kz_spectrum sums, rows and every (kz, k_perp) cell against an independent
#         2-D histogram; spectral_break on synthetic spectra, with and without kmin.
#
# Both precision sessions run this file except the fp64-marked order measurements; the fp32
# tolerances are the measured floors. pytest: `pytest tests/test_flrmhd_diagnostics.py`.
# Script: `python tests/test_flrmhd_diagnostics.py` -- never under mpirun.
from _rmhd_testing import bootstrap, checks, fit_order

bootstrap()

import warnings

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision
from taranis import diagnostics as dg
from taranis.diagnostics import flrmhd as dgf
from taranis.physics import construct_rhs, equation_registry, flrmhd, shared_physics
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

from _flrmhd_testing import (broadband_ic, elsasser_modal, elsasser_rates, flr_ref,
                             invariants, kgrid_ksq, norm, weights, yfac)

FP64 = _precision.precision == "64"
TWO_PI = 2.0*np.pi
TOL = 1e-13 if FP64 else 2e-5        # sums of ~1e3 terms against float64 references

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _flr(rho_i, tau=0.5, Z=1.0, diss=(0.0, 0.0), hyper=1, eq=None, **kw):
    eqpars = dict(rho_i=rho_i, tau=tau, Z=Z, diss=diss, hyper=hyper)
    eqpars.update(eq or {})
    base = dict(nx=32, ny=32, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5, dt=0.01,
                adaptive_timestep=False, dims=2, eqtype="FLRMHD", eqpars=eqpars)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(**base)


_Z3 = dict(dims=3, nx=16, ny=16, nz=8, Lz=TWO_PI, z_spectral=True)
_NSQ = dict(nx=32, ny=24, Lx=TWO_PI*1.3, Ly=TWO_PI*0.8)        # non-square, dx != dy
_NSQ3 = dict(_Z3, nx=16, ny=24, Lx=TWO_PI*0.9, Ly=TWO_PI*1.25, Lz=3.0)

# (label, grid kwargs, rho_i, Z, tau)
_CASES = (("2D non-square, Z/tau = 2.86", _NSQ, 0.4, 2.0, 0.7),
          ("2D 32^2, Z/tau = 2", {}, 0.25, 1.0, 0.5),
          ("z_spectral non-cubic, Z/tau = 0.5", _NSQ3, 0.6, 1.0, 2.0))


def _correlated(ic, mix=0.6):
    def f(*xyz):
        a = ic(*xyz)
        return np.stack([a[0], mix*a[0] + a[1]])
    return f


def _state(params, seed=3):
    return jr.initialize(_correlated(broadband_ic(params, seed=seed)), params)


def _rel(a, b, scale=None):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b))/(np.max(np.abs(b)) if scale is None else scale))


def _zsum(modal):
    return np.asarray(modal).sum(axis=0)                 # (nz, nkx, nky) -> (nkx, nky)


def _np_binned(kgrid, params, modal_list, bin_factor):
    """Independent binning: numpy histogram of z-summed per-mode arrays on edges
    j*bin_factor*kunit covering every mode of the grid; returns (edges, sums per bin) --
    plain sums, NOT divided by dk."""
    kmag = np.sqrt(np.asarray(kgrid.ksq, dtype=float)).ravel()
    dk = min(TWO_PI/params.Lx, TWO_PI/params.Ly)*bin_factor
    edges = dk*np.arange(0, int(np.ceil(kmag.max()/dk)) + 2)
    return edges, [np.histogram(kmag, bins=edges, weights=_zsum(m).ravel())[0]
                   for m in modal_list]


def _nonlinear(params, kgrid, state):
    return np.asarray(construct_rhs(equation_registry["FLRMHD"])(state, kgrid, params)[0])


# ---------------------------------------------------------------------------- gate F2.1

def test_energies_helicity_and_spectra():
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            zt = Z/tau
            params = _flr(rho_i, tau=tau, Z=Z, **kw)
            kgrid = jr.setup_kgrids(params)
            state = _state(params)
            f = np.asarray(state.fields)
            en = {k: float(v) for k, v in dgf.energy(state, kgrid, params).items()}
            H = float(dgf.helicity(state, kgrid, params))
            Eref, Href = invariants(f, params, rho_i, zt)
            Ep, Em, Hk = elsasser_modal(f, params, rho_i, zt)
            c.check(f"{label}: energy keys are (E_ion, E_e, E_mag, E_plus, E_minus, total)",
                    tuple(en) == ("E_ion", "E_e", "E_mag", "E_plus", "E_minus", "total"))
            c.check(f"{label}: total == test-local E (rel {abs(en['total']/Eref - 1):.1e}), "
                    f"H == test-local H (rel {abs(H/Href - 1):.1e}; H v/E ~ {H/Eref:.2f})",
                    abs(en["total"]/Eref - 1) <= TOL and abs(H/Href - 1) <= TOL
                    and abs(Href) > 0.05*Eref)
            e1 = abs((en["E_plus"] + en["E_minus"])/en["total"] - 1)
            e2 = abs((en["E_ion"] + en["E_e"] + en["E_mag"])/en["total"] - 1)
            c.check(f"{label}: E_plus + E_minus = total ({e1:.1e}) = E_ion + E_e + E_mag "
                    f"({e2:.1e})", e1 <= TOL and e2 <= TOL)
            # each piece against its definition, weights from the quadrature Gamma_0
            w = weights(params, rho_i, zt)
            r = flr_ref(w["ksq"], rho_i, zt)
            n = norm(params)
            Eion = 0.5*np.sum(w["yfac"]*r["G"]*np.abs(f[0])**2)/n
            Ee = 0.5*(Z*rho_i**2/(2*tau))*np.sum(w["yfac"]*r["G"]**2*np.abs(f[0])**2)/n
            Emag = 0.5*np.sum(w["yfac"]*w["ksq"]*np.abs(f[1])**2)/n
            errs = dict(E_ion=abs(en["E_ion"] - Eion), E_e=abs(en["E_e"] - Ee),
                        E_mag=abs(en["E_mag"] - Emag), E_plus=abs(en["E_plus"] - Ep.sum()),
                        E_minus=abs(en["E_minus"] - Em.sum()))
            c.check(f"{label}: E_ion = (1/2)<G phi^2> ({Eion/Eref:.2f} of E), E_e = "
                    f"(rho_s^2/2)<nhat^2> ({Ee/Eref:.2f}), E_mag ({Emag/Eref:.2f}), E_plus, "
                    f"E_minus == their definitions (worst {max(errs.values())/Eref:.1e} of E)",
                    max(errs.values()) <= TOL*Eref and Ee > 0.02*Eref,
                    f"{ {k: v/Eref for k, v in errs.items()} }")
            im = dgf.imbalance(state, kgrid, params)
            sc = (Ep.sum() - Em.sum())/(Ep.sum() + Em.sum())
            c.check(f"{label}: imbalance: sigma_H = H/E = {float(im['sigma_H']):+.4f}, "
                    f"sigma_c = (E+ - E-)/(E+ + E-) = {float(im['sigma_c']):+.4f}, and they "
                    f"differ at finite rho_i",
                    abs(float(im["sigma_H"]) - Href/Eref) <= 10*TOL
                    and abs(float(im["sigma_c"]) - sc) <= 10*TOL
                    and abs(float(im["sigma_H"]) - float(im["sigma_c"])) > 1e-2)
            # the RMHD energy of the same arrays is NOT this energy (docs' warning)
            Ermhd = float(sum(dg.energy(state, kgrid, params)))
            c.check(f"{label}: diagnostics.energy (RMHD) on this state is off by "
                    f"{abs(Ermhd/Eref - 1):.2f} -- it is not the FLR energy",
                    abs(Ermhd/Eref - 1) > 5e-3)

            for bf in (2.0, 1.37):
                k, sp, sm, sh = (np.asarray(v, dtype=float)
                                 for v in dgf.perp_spectra(state, kgrid, params, bin_factor=bf))
                dk = min(TWO_PI/params.Lx, TWO_PI/params.Ly)*bf
                es = max(abs(sp.sum()*dk - Ep.sum()), abs(sm.sum()*dk - Em.sum()),
                         abs(sh.sum()*dk - Href))/Eref
                c.check(f"{label}, bin_factor = {bf}: sum E_plus dk = E_plus, sum E_minus dk "
                        f"= E_minus, sum E_H dk = H (worst {es:.1e} of E; {k.size} bins to "
                        f"k = {k[-1] + dk/2:.2f})", es <= TOL, f"{es:.3e}")
                c.check(f"{label}, bin_factor = {bf}: bin centres are (j + 1/2) dk",
                        np.allclose(k, (np.arange(k.size) + 0.5)*dk, rtol=1e-6 if FP64 else 1e-5))
            # bin by bin against the independent numpy binning (bin_factor = 1.37: no mode
            # sits on a bin edge, so the two binnings cannot differ by an edge convention)
            edges, (bp, bm, bh) = _np_binned(kgrid, params, (Ep, Em, Hk), 1.37)
            nb = k.size
            eb = max(_rel(sp*dk, bp[:nb], Eref), _rel(sm*dk, bm[:nb], Eref),
                     _rel(sh*dk, bh[:nb], Eref))
            c.check(f"{label}: E_plus(k), E_minus(k), E_H(k) == an independent numpy binning "
                    f"of the per-mode definitions, bin by bin (worst {eb:.1e} of E); nothing "
                    f"beyond the last bin", eb <= TOL and not np.any(bp[nb:])
                    and not np.any(bm[nb:]), f"{eb:.3e}")
            # E_H is not (E_plus - E_minus)/v_ph(bin centre): the 1/v_ph is per mode
            vc = flr_ref(k**2, rho_i, zt)["vph"]
            naive = (sp - sm)/vc
            c.check(f"{label}: (per-bin (E_plus - E_minus)/v_ph(k_centre) differs from E_H(k) "
                    f"by {_rel(naive*dk, sh*dk, abs(Href)):.1e} of H: the per-mode 1/v_ph is "
                    f"load-bearing)", _rel(naive*dk, sh*dk, abs(Href)) > 1e-4)


def test_rho_i_zero_is_the_rmhd_diagnostics():
    with checks() as c:
        for label, kw in (("2D 32^2", {}), ("z_spectral 16^2 x 8", _Z3)):
            params = _flr(0.0, **kw)
            kgrid = jr.setup_kgrids(params)
            state = _state(params, seed=4)
            en = dgf.energy(state, kgrid, params)
            Ek, Em = (float(v) for v in dg.energy(state, kgrid, params))
            c.check(f"{label}: E_ion == RMHD E_kin, E_mag == RMHD E_mag, E_e == 0 exactly",
                    abs(float(en["E_ion"])/Ek - 1) <= TOL and abs(float(en["E_mag"])/Em - 1)
                    <= TOL and float(en["E_e"]) == 0.0)
            Hd = float(shared_physics.perp_inner_product(state.fields[0], state.fields[1],
                                                         kgrid, params))
            H = float(dgf.helicity(state, kgrid, params))
            c.check(f"{label}: H == the cross-helicity <grad phi . grad psi> "
                    f"(rel {abs(H/Hd - 1):.1e})", abs(H/Hd - 1) <= TOL)
            im = dgf.imbalance(state, kgrid, params)
            c.check(f"{label}: sigma_H == sigma_c at rho_i = 0",
                    abs(float(im["sigma_H"]) - float(im["sigma_c"])) <= 10*TOL)
            k, sp, sm, sh = (np.asarray(v, dtype=float)
                             for v in dgf.perp_spectra(state, kgrid, params))
            k0, s_phi, s_psi = (np.asarray(v, dtype=float)
                                for v in dg.perpspec(state, kgrid, params))
            c.check(f"{label}: perp_spectra uses perpspec's bins in a square box and "
                    f"E_plus(k) + E_minus(k) == perpspec's phi + psi columns "
                    f"(rel {_rel(sp + sm, s_phi + s_psi):.1e})",
                    k.shape == k0.shape and np.array_equal(k, k0)
                    and _rel(sp + sm, s_phi + s_psi) <= TOL)
        # a non-square box: both share diagnostics.core._perp_bins, which runs to the
        # largest dealiased k_perp, so both sums close (perpspec used to stop short)
        params = _flr(0.0, **_NSQ)
        kgrid = jr.setup_kgrids(params)
        state = _state(params, seed=4)
        k, sp, sm, _ = (np.asarray(v, dtype=float) for v in dgf.perp_spectra(state, kgrid, params))
        k0, s_phi, s_psi = (np.asarray(v, dtype=float) for v in dg.perpspec(state, kgrid, params))
        dk = k[1] - k[0]
        E = float(dgf.energy(state, kgrid, params)["total"])
        c.check(f"non-square box: perp_spectra and perpspec share {k.size} bins, agree per bin "
                f"({_rel(sp + sm, s_phi + s_psi):.1e}), and both sum to E "
                f"({abs((sp + sm).sum()*dk/E - 1):.1e}, "
                f"{abs((s_phi + s_psi).sum()*dk/E - 1):.1e})",
                np.array_equal(k, k0)
                and _rel(sp + sm, s_phi + s_psi) <= TOL
                and abs((sp + sm).sum()*dk/E - 1) <= TOL
                and abs((s_phi + s_psi).sum()*dk/E - 1) <= TOL)


# ---------------------------------------------------------------------------- gate F2.2

def test_known_states():
    tol = 1e-13 if FP64 else 5e-6
    with checks() as c:
        for Z, tau in ((1.0, 0.5), (2.0, 0.7)):
            for rho_i in (0.05, 0.4, 2.0):
                params = _flr(rho_i, tau=tau, Z=Z, **dict(_Z3, Lx=TWO_PI*1.2))
                kgrid = jr.setup_kgrids(params)
                ikx, iky, iz = 2, 3, 1
                kx, ky = ikx/1.2, float(iky)
                ksq = kx**2 + ky**2
                r = flr_ref(np.array([ksq]), rho_i, Z/tau)
                vph, kap = float(r["vph"][0]), float(r["kappa"][0])
                for sign in (+1, -1):
                    def ic(x, y, z):
                        phi = jnp.cos(kx*x + iky*y + iz*z + 0.3)
                        return jnp.stack([phi, sign*kap*phi])
                    state = jr.initialize(ic, params)
                    en = dgf.energy(state, kgrid, params)
                    H = float(dgf.helicity(state, kgrid, params))
                    E = float(en["total"])
                    tag = (f"Z/tau = {Z/tau:.3g}, k rho_i = {np.sqrt(ksq)*rho_i:.2f}, pure "
                           f"Theta^{'+' if sign > 0 else '-'}")
                    e = abs(H*vph/E - sign)
                    c.check(f"{tag}: H v_ph/E = {sign:+d} (v_ph = {vph:.4f} from the paper's "
                            f"closed form; err {e:.1e})", e <= tol, f"{e:.3e}")
                    want = 0.5*kap**2*ksq                  # E = kappa^2 k^2/2 for phi = cos
                    other = float(en["E_minus" if sign > 0 else "E_plus"])
                    c.check(f"{tag}: E = kappa^2 k^2/2 (rel {abs(E/want - 1):.1e}) and the "
                            f"other channel holds {other/E:.1e} of it",
                            abs(E/want - 1) <= tol and other <= tol*E)
                    im = dgf.imbalance(state, kgrid, params)
                    c.check(f"{tag}: sigma_c = {sign:+d}, sigma_H = {sign:+d}/v_ph",
                            abs(float(im["sigma_c"]) - sign) <= tol
                            and abs(float(im["sigma_H"])*vph - sign) <= tol)
                    k, sp, sm, sh = (np.asarray(v, dtype=float) for v in
                                     dgf.perp_spectra(state, kgrid, params, bin_factor=1.0))
                    dk = min(1/1.2, 1.0)
                    j = int(np.sqrt(ksq)//dk)
                    own = sp if sign > 0 else sm
                    c.check(f"{tag}: the spectrum sits in the bin holding k = {np.sqrt(ksq):.3f} "
                            f"and E_H(k) v_ph = {sign:+d} E(k) there",
                            abs(own[j]*dk/E - 1) <= tol
                            and abs(sh[j]*vph/own[j] - sign) <= tol
                            and np.count_nonzero(np.abs(own) > tol*own[j]) == 1)


# ---------------------------------------------------------------------------- gate F2.3

def test_transfers():
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            zt = Z/tau
            params = _flr(rho_i, tau=tau, Z=Z, **kw)
            kgrid = jr.setup_kgrids(params)
            state = _state(params, seed=5)
            f = np.asarray(state.fields)
            N = _nonlinear(params, kgrid, state)
            Tp, Tm, TH = elsasser_rates(f, N, params, rho_i, zt)   # first principles
            scaleE, scaleH = np.abs(Tp + Tm).sum(), np.abs(TH).sum()
            bf = 1.37
            dk = min(TWO_PI/params.Lx, TWO_PI/params.Ly)*bf
            k, tp, tm, th = (np.asarray(v, dtype=float)
                             for v in dgf.transfer(state, kgrid, params, bin_factor=bf))
            _, (bp, bm, bh) = _np_binned(kgrid, params, (Tp, Tm, TH), bf)
            nb = k.size
            e = max(_rel(tp*dk, bp[:nb], scaleE), _rel(tm*dk, bm[:nb], scaleE))
            eh = _rel(th*dk, bh[:nb], scaleH)
            c.check(f"{label}: T_plus(k), T_minus(k) == the first-principles transfer, bin by "
                    f"bin ({e:.1e} of sum|T_E|), T_H(k) == its G form ({eh:.1e} of sum|T_H|)",
                    e <= TOL and eh <= TOL, f"{e:.3e}, {eh:.3e}")
            sE, sH = abs((tp + tm).sum()*dk)/scaleE, abs(th.sum()*dk)/scaleH
            c.check(f"{label}: sum_k T_E = 0 ({sE:.1e} of sum|T_E|), sum_k T_H = 0 "
                    f"({sH:.1e}); sum_k T_plus alone is {abs(tp.sum()*dk)/scaleE:.1e} -- E^+ "
                    f"and E^- are not conserved separately", sE <= TOL and sH <= TOL)
            K, Pi, Pp, Pm, PH = (np.asarray(v, dtype=float)
                                 for v in dgf.fluxes(state, kgrid, params, bin_factor=bf))
            c.check(f"{label}: fluxes returns K = the upper bin edges, Pi = Pi_plus + "
                    f"Pi_minus, and the last entry is the conservation residual "
                    f"({abs(Pi[-1])/scaleE:.1e}, {abs(PH[-1])/scaleH:.1e})",
                    np.allclose(K, k + dk/2) and _rel(Pi, Pp + Pm, scaleE) <= TOL
                    and abs(Pi[-1]) <= TOL*scaleE and abs(PH[-1]) <= TOL*scaleH)
            c.check(f"{label}: the fluxes are live (max|Pi| = {np.abs(Pi).max()/scaleE:.2f} of "
                    f"sum|T_E|, max|Pi_H| = {np.abs(PH).max()/scaleH:.2f} of sum|T_H|)",
                    np.abs(Pi).max() > 0.02*scaleE and np.abs(PH).max() > 0.02*scaleH)

        # two shells, each with ONE k_perp (|k| = 5 and |k| = sqrt 50), coupled by triads
        # (5,0)+(0,5) = (5,5), (3,4)+(4,-3) = (7,1), (4,3)+(-3,4) = (1,7): the transfer lives
        # on those two k only, so per bin T_H * v_ph(shell) = T_plus - T_minus -- the docs'
        # per-mode identity, with v_ph from the paper's closed form
        rho_i, zt = 0.3, 2.0
        params = _flr(rho_i, **dict(nx=48, ny=48))
        kgrid = jr.setup_kgrids(params)

        def ic(x, y):
            a = (jnp.cos(5*x + 0.2) + 0.8*jnp.sin(5*y) + 0.7*jnp.cos(3*x + 4*y + 0.5)
                 + 0.6*jnp.sin(4*x - 3*y) + 0.5*jnp.cos(4*x + 3*y) + 0.9*jnp.sin(-3*x + 4*y + 1))
            b = (0.6*jnp.cos(5*x + 5*y + 0.4) + 0.5*jnp.sin(7*x + y) + 0.7*jnp.cos(x + 7*y + 2))
            return jnp.stack([a + 0.5*b, 0.4*a - 0.8*b + 0.3*jnp.sin(5*x + 1.0)])

        state = jr.initialize(ic, params)
        k, tp, tm, th = (np.asarray(v, dtype=float)
                         for v in dgf.transfer(state, kgrid, params, bin_factor=1.0))
        live = np.flatnonzero(np.abs(tp) + np.abs(tm) > (1e-9 if FP64 else 1e-4)*np.abs(tp).max())
        c.check(f"two-shell state: the transfer lives in exactly the bins of k = 5 and "
                f"k = sqrt(50) (bins {live.tolist()})", live.tolist() == [5, 7])
        for j, kk in ((5, 5.0), (7, np.sqrt(50.0))):
            vph = float(flr_ref(np.array([kk**2]), rho_i, zt)["vph"][0])
            e = abs(th[j]*vph - (tp[j] - tm[j]))/abs(tp[j] - tm[j])
            c.check(f"two-shell state, k = {kk:.3f}: T_H v_ph = T_plus - T_minus with v_ph = "
                    f"{vph:.4f} (rel {e:.1e}; (T_+ - T_-)/T_+ = {(tp[j] - tm[j])/tp[j]:+.2f})",
                    e <= (1e-12 if FP64 else 1e-4) and abs(tp[j] - tm[j]) > 1e-3*abs(tp[j]))


# ---------------------------------------------------------------------------- gate F2.4

def _band(modal, kgrid, K):
    # sum of a per-mode (nz, nkx, nky) array over k_perp < K
    kmag = np.sqrt(np.asarray(kgrid.ksq, dtype=float))
    return float(np.asarray(modal).sum(axis=0)[kmag < K].sum())


@pytest.mark.fp64
def test_flux_against_the_stepper():
    """Unforced, nu = 0. Two half steps s0 -> s1 -> s2 of size h: the centred difference
    [X(<K)(s2) - X(<K)(s0)]/(2h) of the band-limited invariants (test-local forms) equals
    -Pi_X(K) evaluated by diagnostics.flrmhd.fluxes on the midpoint state s1, to O(h^2), for
    X = E, E^+, E^-, H and six K. Under z_spectral the linear waves change no E^+-_k or H_k,
    so the statement is the same. Asserted: order 2 +- 0.3 over three h, and 1e-6 (of
    max|Pi_X|) at the finest."""
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for label, kw, rho_i, Z, tau, hs in (
                ("2D non-square, Z/tau = 2.86", _NSQ, 0.4, 2.0, 0.7, (2e-4, 1e-4, 5e-5)),
                ("z_spectral non-cubic, Z/tau = 0.5", _NSQ3, 0.6, 1.0, 2.0,
                 (2e-4, 1e-4, 5e-5))):
            zt = Z/tau
            errs = {x: [] for x in ("E", "E+", "E-", "H")}
            for h in hs:
                params = _flr(rho_i, tau=tau, Z=Z, **dict(kw, dt=h))
                kgrid = jr.setup_kgrids(params)
                s0 = jr.initialize(_correlated(broadband_ic(params, seed=5, slope=-2.0, amp=0.2)),
                                   params)
                s1 = _advance(s0, kgrid, params, 1, scheme, stepper)
                s2 = _advance(s1, kgrid, params, 1, scheme, stepper)
                K, Pi, Pp, Pm, PH = (np.asarray(v, dtype=float)
                                     for v in dgf.fluxes(s1, kgrid, params, bin_factor=1.37))
                pick = np.linspace(1, K.size - 3, 6).astype(int)      # six interior K
                m0 = elsasser_modal(s0.fields, params, rho_i, zt)
                m2 = elsasser_modal(s2.fields, params, rho_i, zt)
                d = lambda i, Kc: (_band(m2[i], kgrid, Kc) - _band(m0[i], kgrid, Kc))/(2*h)
                for name, flux, rate in (("E+", Pp, lambda Kc: d(0, Kc)),
                                         ("E-", Pm, lambda Kc: d(1, Kc)),
                                         ("E", Pi, lambda Kc: d(0, Kc) + d(1, Kc)),
                                         ("H", PH, lambda Kc: d(2, Kc))):
                    got = np.array([rate(K[j]) for j in pick])
                    errs[name].append(np.max(np.abs(got + flux[pick]))/np.max(np.abs(flux)))
                if h == hs[-1]:
                    live = {n: float(np.max(np.abs(fl[pick]))/np.max(np.abs(fl)))
                            for n, fl in (("E", Pi), ("H", PH))}
                    sgn = float(np.sign(Pi[pick][np.argmax(np.abs(Pi[pick]))]))
                    c.check(f"{label}: six K in [{K[pick[0]]:.2f}, {K[pick[-1]]:.2f}] sample "
                            f"live fluxes (max|Pi(K_j)|/max|Pi| = {live['E']:.2f}, H: "
                            f"{live['H']:.2f}; the largest Pi is "
                            f"{'forward (> 0)' if sgn > 0 else 'inverse (< 0)'})",
                            live["E"] > 0.3 and live["H"] > 0.3)
            for name in ("E", "E+", "E-", "H"):
                e = errs[name]
                order = fit_order(hs, e)
                c.check(f"{label}: d/dt X(<K) = -Pi_X(K), X = {name}: err {e[0]:.1e} -> "
                        f"{e[1]:.1e} -> {e[2]:.1e} of max|Pi|, order {order:.2f}",
                        1.7 <= order <= 2.3 and e[-1] <= 1e-6, f"{e}")


# ---------------------------------------------------------------------------- gate F2.5

def test_subfilter_transfer():
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            zt = Z/tau
            params = _flr(rho_i, tau=tau, Z=Z, **kw)
            kgrid = jr.setup_kgrids(params)
            state = _state(params, seed=6)
            f = np.asarray(state.fields)
            kmag = np.sqrt(np.asarray(kgrid.ksq, dtype=float))
            bf = 1.37
            dk = min(TWO_PI/params.Lx, TWO_PI/params.Ly)*bf
            K, Pi, Pp, Pm, PH = (np.asarray(v, dtype=float)
                                 for v in dgf.fluxes(state, kgrid, params, bin_factor=bf))
            scaleE, scaleH = np.max(np.abs(Pi)), np.max(np.abs(PH))
            jc = int(np.argmax(np.abs(Pi)))                 # a bin edge with a live flux
            for kcut, on_edge in ((float(K[jc]), True), (float(K[jc]) - 0.37*dk, False)):
                # first principles: T[f] - T[f^<] on k < kcut, with independent weights
                low = kmag < kcut
                flo = f*low
                Nfull = _nonlinear(params, kgrid, state)
                Nlo = _nonlinear(params, kgrid, state._replace(
                    fields=jnp.asarray(flo, dtype=_precision.ctype)))
                full = elsasser_rates(f, Nfull, params, rho_i, zt)
                res = elsasser_rates(flo, Nlo, params, rho_i, zt)
                sf = [(a - b)*low for a, b in zip(full, res)]
                tag = f"{label}, kcut = {kcut:.3f} ({'a bin edge' if on_edge else 'inside a bin'})"
                cons = max(abs(res[0].sum() + res[1].sum())/np.abs(res[0] + res[1]).sum(),
                           abs(res[2].sum())/np.abs(res[2]).sum())
                c.check(f"{tag}: the low-passed state conserves E and H on its own "
                        f"(sum T[f^<] = {cons:.1e} of sum|T|) and holds "
                        f"{int(low.sum())} of {int((np.abs(f[0]).max(axis=0) > 0).sum())} "
                        f"perpendicular modes", cons <= TOL and 0 < low.sum() < low.size)
                k, tp, tm, th = (np.asarray(v, dtype=float) for v in
                                 dgf.transfer(state, kgrid, params, bin_factor=bf, kcut=kcut))
                _, (bp, bm, bh) = _np_binned(kgrid, params, sf, bf)
                nb = k.size
                e = max(_rel(tp*dk, bp[:nb], scaleE), _rel(tm*dk, bm[:nb], scaleE),
                        _rel(th*dk, bh[:nb], scaleH))
                c.check(f"{tag}: transfer(kcut=) == the first-principles sub-filter transfer "
                        f"bin by bin (worst {e:.1e} of max|Pi|)", e <= 10*TOL, f"{e:.3e}")
                above = k - dk/2 >= kcut
                c.check(f"{tag}: it vanishes in every bin above kcut and is live below "
                        f"(max {np.abs(tp + tm).max()*dk/scaleE:.2f} of max|Pi|)",
                        not np.any(tp[above]) and not np.any(tm[above]) and not np.any(th[above])
                        and np.abs(tp + tm).max()*dk > 0.02*scaleE)
                # its sum is minus the flux through kcut (independent: per-mode sums)
                wantE = -(full[0] + full[1])[:, low].sum()
                wantH = -full[2][:, low].sum()
                gotE, gotH = -(tp + tm).sum()*dk, -th.sum()*dk
                e = max(abs(gotE - wantE)/scaleE, abs(gotH - wantH)/scaleH)
                c.check(f"{tag}: -sum_k T^sf = -sum_(k<kcut) T[f] = Pi(kcut) for E "
                        f"({gotE/scaleE:+.3f} of max|Pi|) and H ({gotH/scaleH:+.3f}) "
                        f"(err {e:.1e})", e <= 10*TOL, f"{e:.3e}")
                Ks, Pis, Pps, Pms, PHs = (np.asarray(v, dtype=float) for v in dgf.fluxes(
                    state, kgrid, params, bin_factor=bf, kcut=kcut))
                if on_edge:
                    e = max(_rel(Pis[jc:], np.full(Pis[jc:].shape, Pi[jc]), scaleE),
                            _rel(PHs[jc:], np.full(PHs[jc:].shape, PH[jc]), scaleH))
                    c.check(f"{tag}: fluxes(kcut=) equals the full Pi(kcut), Pi_H(kcut) at and "
                            f"beyond K = kcut (err {e:.1e}) and differs from the full flux "
                            f"below it ({_rel(Pis[:jc], Pi[:jc], scaleE):.2f})",
                            e <= 10*TOL and (jc == 0 or _rel(Pis[:jc], Pi[:jc], scaleE) > 1e-3),
                            f"{e:.3e}")
                    # NOT per channel: FLR-MHD conserves E and H, not E^+ and E^- separately
                    # (H is sum (E^+_k - E^-_k)/v_ph(k)), so sum T_+-[f^<] != 0 and the
                    # channel sub-filter fluxes differ from the full Pi_+-(kcut)
                    dch = abs(Pps[-1] - Pp[jc])/scaleE
                    c.check(f"{tag}: (per channel the two differ by {dch:.1e} of max|Pi| at "
                            f"rho_i > 0, while Pi_plus + Pi_minus agrees)",
                            abs((Pps[-1] + Pms[-1]) - Pi[jc]) <= 10*TOL*scaleE)


def test_subfilter_edge_convention_at_default_bins():
    """The retained band is k_perp < kcut, STRICTLY -- the same [lo, hi) convention as the
    bins, so that the cumulative sub-filter flux ends on the full Pi(kcut). In a square
    2 pi box at the default bin_factor = 2.0 the bin edges are K = 2, 4, ... and the modes
    (2,0), (0,2), (4,0), (0,4) sit EXACTLY on them: `<= kcut` would move those modes into the
    retained band and miss Pi(kcut) by their transfer."""
    with checks() as c:
        for label, kw, rho_i in (("2D 32^2 square", {}, 0.25),
                                 ("z_spectral 16^2 x 8 cubic", _Z3, 0.5)):
            zt = 2.0
            params = _flr(rho_i, **kw)
            kgrid = jr.setup_kgrids(params)
            state = _state(params, seed=6)
            f = np.asarray(state.fields)
            kmag = np.sqrt(kgrid_ksq(params))
            Tp, Tm, TH = elsasser_rates(f, _nonlinear(params, kgrid, state), params, rho_i, zt)
            K, Pi, _, _, PH = (np.asarray(v, dtype=float)
                               for v in dgf.fluxes(state, kgrid, params))       # bin_factor 2.0
            scaleE, scaleH = np.max(np.abs(Pi)), np.max(np.abs(PH))
            c.check(f"{label}: default bins have edges K = 2, 4, ... ({K[:3].tolist()})",
                    np.array_equal(K[:3], [2.0, 4.0, 6.0]))
            for jc in (0, 1):
                kcut = float(K[jc])
                on = kmag == kcut
                edgeE = abs((Tp + Tm)[:, on].sum())/scaleE
                edgeH = abs(TH[:, on].sum())/scaleH
                Ks, Pis, _, _, PHs = (np.asarray(v, dtype=float)
                                      for v in dgf.fluxes(state, kgrid, params, kcut=kcut))
                e = max(abs(Pis[-1] - Pi[jc])/scaleE, abs(PHs[-1] - PH[jc])/scaleH)
                wantE = -(Tp + Tm)[:, kmag < kcut].sum()
                c.check(f"{label}, kcut = {kcut:.0f} with {int(on.sum())} modes exactly on it "
                        f"(carrying {edgeE:.1e} of max|Pi|, {edgeH:.1e} of max|Pi_H|): "
                        f"Pi_sf(end) == Pi(kcut) == -sum_(k < kcut) T, E and H (err {e:.1e})",
                        e <= 10*TOL and abs(Pis[-1] - wantE) <= 10*TOL*scaleE
                        and int(on.sum()) >= 2 and edgeE > 1e-3 and edgeH > 1e-3, f"{e:.3e}")


# ---------------------------------------------------------------------------- gate F2.6

def _tiny_ic(params, seed):
    ic = _correlated(broadband_ic(params, seed=seed, slope=-1.0))
    return lambda *xyz: 1e-11*ic(*xyz)


# fp32 tolerances of the nu_phi != nu_psi budget closure (c), from the measured floors
# (2026-11-12): 1.5e-7, 4.2e-7, 4.9e-7 of the summed |per-mode rates| and 1.5e-7, 7.6e-4,
# 2.0e-6 of the dissipation rates in the three cases. The second cannot be tighter than
# ~1e-3 in the hyper = 1 z_spectral case: the cancelling wave terms are 2e3 times eps_perp
# there, so fp32 round-off alone is 2e3 * 4e-7. (Until 2026-11-12 this was a bare 2e-2.)
_CLW32 = 4e-6     # ~8x the worst floor
_CL32 = 4e-3      # ~5x the worst floor


def test_dissipation_against_exact_decay():
    """Amplitude 1e-11 (the nonlinear term is 1e-11 of the linear one; at 1e-8 its
    interplay with the dissipation is visible at 1e-9 in the stepped comparison). With nu_phi = nu_psi
    every mode's E_k and H_k decay as exp(2 d_k t), d_k = -nu k^2h - nu_z kz^(2 h_z) (the
    waves change neither), so E(t) = sum_k E_k(0) exp(2 d_k t) in closed form: (a) the
    STEPPED E(t), H(t) follow it; (b) its derivative at t = 0 is -(eps_perp + eps_z) and
    -(epsH_perp + epsH_z) of diagnostics.flrmhd.dissipation, 1e-11; (c) unequal
    (nu_phi, nu_psi): the docs' formulas with independent weights."""
    tol = 1e-11 if FP64 else 2e-4
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for label, kw, rho_i, Z, tau, hyper, hz in (
                ("2D non-square, hyper = 2", _NSQ, 0.4, 2.0, 0.7, 2, None),
                ("z_spectral non-cubic, hyper = 1, z_hyper = 3", _NSQ3, 0.6, 1.0, 2.0, 1, 3),
                ("z_spectral non-cubic, hyper = 2, z_hyper = 2", _NSQ3, 0.6, 2.0, 0.7, 2, 2)):
            zt = Z/tau
            nu, nuz = 3e-3, (0.02 if hz else 0.0)
            eq = dict(z_diss_k=nuz, z_hyper=hz) if hz else {}
            params = _flr(rho_i, tau=tau, Z=Z, diss=(nu, nu), hyper=hyper, eq=eq,
                          **dict(kw, dt=0.01))
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_tiny_ic(params, 8), params)
            f = np.asarray(state.fields)
            Ep, Em, Hk = elsasser_modal(f, params, rho_i, zt)
            ksq = kgrid_ksq(params)
            d = -nu*ksq**hyper + np.zeros((f.shape[1], 1, 1))
            if hz:
                kz = np.fft.fftfreq(params.nz)*params.nz*TWO_PI/params.Lz
                kz[params.nz//2] = 0.0
                d = d - nuz*(kz**(2*hz))[:, None, None]
            E0, H0 = (Ep + Em).sum(), Hk.sum()
            dEdt, dHdt = (2*d*(Ep + Em)).sum(), (2*d*Hk).sum()
            dis = {k: float(v) for k, v in dgf.dissipation(state, kgrid, params).items()}
            eE = abs(-(dis["eps_perp"] + dis["eps_z"])/dEdt - 1)
            eH = abs(-(dis["epsH_perp"] + dis["epsH_z"])/dHdt - 1)
            c.check(f"{label}: d/dt of the closed-form E(t) at t = 0 == -(eps_perp + eps_z) "
                    f"(rel {eE:.1e}; eps_z/eps_perp = {dis['eps_z']/dis['eps_perp']:.2f}), "
                    f"of H(t) == -(epsH_perp + epsH_z) (rel {eH:.1e})",
                    eE <= tol and eH <= tol and (not hz or dis["eps_z"] > 0.02*dis["eps_perp"]),
                    f"{eE:.3e}, {eH:.3e}")
            if hz:
                ez = -(2*(d + nu*ksq**hyper)*(Ep + Em)).sum()
                c.check(f"{label}: eps_z alone == nu_z sum kz^(2 h_z) 2 E_k "
                        f"(rel {abs(dis['eps_z']/ez - 1):.1e})", abs(dis["eps_z"]/ez - 1) <= tol)
            else:
                c.check(f"{label}: eps_z and epsH_z are exactly 0 in 2D",
                        dis["eps_z"] == 0.0 and dis["epsH_z"] == 0.0)
            # (a) the stepper really does decay the invariants at those rates
            nstep = 20
            end = _advance(state, kgrid, params, nstep, scheme, stepper)
            t = float(end.t)
            E1, H1 = invariants(end.fields, params, rho_i, zt)
            Ex, Hx = ((Ep + Em)*np.exp(2*d*t)).sum(), (Hk*np.exp(2*d*t)).sum()
            c.check(f"{label}: after {nstep} steps E(t) and H(t) are the closed-form "
                    f"exponentials (rel {abs(E1/Ex - 1):.1e}, {abs(H1/Hx - 1):.1e}; E fell by "
                    f"{1 - Ex/E0:.2f})", abs(E1/Ex - 1) <= (1e-10 if FP64 else 2e-4)
                    and abs(H1/Hx - 1) <= (1e-10 if FP64 else 2e-4) and 1 - Ex/E0 > 0.02)

            # (c) unequal coefficients: the docs' formulas, independent weights
            nus = (4e-3, 1e-3)
            pu = _flr(rho_i, tau=tau, Z=Z, diss=nus, hyper=hyper, eq=eq, **kw)
            ku = jr.setup_kgrids(pu)
            du = {k: float(v) for k, v in dgf.dissipation(state, ku, pu).items()}
            w = weights(pu, rho_i, zt)
            n = norm(pu)
            k2h = w["ksq"]**hyper
            eps_perp = np.sum(w["yfac"]*k2h*(nus[0]*w["Gpe"]*np.abs(f[0])**2
                                             + nus[1]*w["ksq"]*np.abs(f[1])**2))/n
            epsH_perp = (nus[0] + nus[1])*np.sum(k2h*Hk)
            e = max(abs(du["eps_perp"]/eps_perp - 1), abs(du["epsH_perp"]/epsH_perp - 1))
            split = nus[0]*np.sum(w["yfac"]*k2h*w["Gpe"]*np.abs(f[0])**2)/n/eps_perp
            c.check(f"{label}: nu_phi != nu_psi: eps_perp = sum k^2h [nu_phi G pe |phi|^2 + "
                    f"nu_psi ksq |psi|^2] and epsH_perp = (nu_phi + nu_psi) sum k^2h H_k "
                    f"(rel {e:.1e}; the phi term is {split:.2f} of eps_perp)",
                    e <= tol and 0.1 < split < 0.95, f"{e:.3e}")
            # with ksq in place of G pe the phi term would be visibly different
            wrong = np.sum(w["yfac"]*k2h*(nus[0]*w["ksq"]*np.abs(f[0])**2
                                          + nus[1]*w["ksq"]*np.abs(f[1])**2))/n
            c.check(f"{label}: (a ksq weight for G pe would move eps_perp by "
                    f"{abs(wrong/eps_perp - 1):.2f})", abs(wrong/eps_perp - 1) > 0.02)
            b = dgf.budget(state, ku, pu)
            cl = max(abs(float(b["closure_E"]))/eps_perp, abs(float(b["closure_H"]))/abs(
                epsH_perp))
            # the closure is a sum of per-mode rates under the full right-hand side, in which
            # the WAVE terms cancel pairwise: its round-off is set by THEIR size, which is
            # 1e3-1e4 times the dissipation rates. So it is asserted twice: against the
            # summed magnitude of the per-mode rates (a few ulp of the precision), and
            # against the dissipation rates at a small multiple of the measured floor
            full = np.asarray(construct_rhs(equation_registry["FLRMHD"])(state, ku, pu)[0]
                              + ku.lin.apply_L(state.fields))
            mTp, mTm, mTH = elsasser_rates(f, full, pu, rho_i, zt)
            magE, magH = np.abs(mTp).sum() + np.abs(mTm).sum(), np.abs(mTH).sum()
            clw = max(abs(float(b["closure_E"]))/magE, abs(float(b["closure_H"]))/magH)
            c.check(f"{label}: budget closure against the solver's own L at nu_phi != nu_psi: "
                    f"{clw:.1e} of the summed |per-mode rates| (which are {magE/eps_perp:.0e} "
                    f"times eps_perp), {cl:.1e} of the dissipation rates; dEdt = -eps_perp - "
                    f"eps_z unforced",
                    clw <= (1e-14 if FP64 else _CLW32) and cl <= (1e-10 if FP64 else _CL32)
                    and abs(float(b["dEdt"]) + du["eps_perp"] + du["eps_z"])
                    <= TOL*(du["eps_perp"] + du["eps_z"])
                    and float(b["eps"]) == 0.0, f"{clw:.3e}, {cl:.3e}")


# ---------------------------------------------------------------------------- gate F2.7

def _forced(kw, rho_i, Z, tau, mode, **extra):
    base = _flr(rho_i, tau=tau, Z=Z, **dict(kw, fshell=(1, 4)))
    fk = dict(kw, fshell=(1, 4), forcing=True, forcing_tau=0.3, forcing_seed=9,
              forcing_mode=mode)
    if mode == "elsasser":
        fk["forcing_power_elsasser"] = flrmhd.elsasser_powers(0.9, 0.25, base)
    else:
        fk["forcing_power"] = 0.9
    fk.update(extra)
    return fk


def test_injection_and_instantaneous_budget():
    stepper, scheme = get_scheme("lsrk54")
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            zt = Z/tau
            eq = dict(z_diss_k=0.01, z_hyper=3) if kw.get("z_spectral") else {}
            for mode in ("momentum", "elsasser"):
                params = _flr(rho_i, tau=tau, Z=Z, diss=(4e-3, 1e-3), hyper=2, eq=eq,
                              **_forced(kw, rho_i, Z, tau, mode, dt=0.004))
                kgrid = jr.setup_kgrids(params)
                state = _advance(_state(params, seed=7), kgrid, params, 12, scheme, stepper)
                f = np.asarray(state.fields)
                F = np.asarray(flrmhd.ForcingTerm(state, None, kgrid, params))
                inj = {k: float(v) for k, v in dgf.injection(state, kgrid, params).items()}
                Tp, Tm, TH = (v.sum() for v in elsasser_rates(f, F, params, rho_i, zt))
                Sp, Sm, SH = (v.sum() for v in elsasser_modal(F, params, rho_i, zt))
                ref = dict(eps_plus=Tp, eps_minus=Tm, eps=Tp + Tm, eps_H=TH, self_plus=Sp,
                           self_minus=Sm, self_H=SH)
                ref["self"] = Sp + Sm
                scale = abs(Tp) + abs(Tm) + Sp + Sm
                worst = max(abs(inj[k] - ref[k]) for k in ref)/scale
                tag = f"{label}, {mode}"
                c.check(f"{tag}: injection keys and values == independent rates of the "
                        f"solver's forcing (worst {worst:.1e}; eps = {inj['eps']:+.3f}, eps_H = "
                        f"{inj['eps_H']:+.3f}, self = {inj['self']:.3f})",
                        set(inj) == set(ref) and worst <= 10*TOL and abs(inj["eps"]) > 1e-3,
                        f"{worst:.3e}")
                dt = params.dt
                tot = inj["eps"] + inj["self"]*dt
                c.check(f"{tag}: eps + self*dt = the target 0.9 for the step this state "
                        f"starts (rel {abs(tot/0.9 - 1):.1e})",
                        abs(tot/0.9 - 1) <= (1e-11 if FP64 else 1e-3))
                b = {k: float(v) for k, v in dgf.budget(state, kgrid, params).items()}
                dis = {k: float(v) for k, v in dgf.dissipation(state, kgrid, params).items()}
                sc = abs(b["eps"]) + b["eps_perp"] + b["eps_z"]
                c.check(f"{tag}: budget: dEdt = eps - eps_perp - eps_z, dHdt likewise, with "
                        f"the rates of injection() and dissipation()",
                        abs(b["dEdt"] - (inj["eps"] - dis["eps_perp"] - dis["eps_z"])) <= TOL*sc
                        and abs(b["dHdt"] - (inj["eps_H"] - dis["epsH_perp"] - dis["epsH_z"]))
                        <= TOL*sc and b["eps_perp"] > 0 and (not eq or b["eps_z"] > 0))
                cl = max(abs(b["closure_E"]), abs(b["closure_H"]))/sc
                c.check(f"{tag}: closure against the solver's full RHS: {cl:.1e} of the "
                        f"rates (eps_perp/eps = {b['eps_perp']/abs(b['eps']):.2f})",
                        cl <= (1e-12 if FP64 else 2e-4), f"{cl:.3e}")
        # unforced: injection is identically zero
        params = _flr(0.3)
        kgrid = jr.setup_kgrids(params)
        inj = dgf.injection(_state(params), kgrid, params)
        c.check("unforced: every injection entry is exactly 0",
                all(float(v) == 0.0 for v in inj.values()) and len(inj) == 8)


def _budget_run(params, kgrid, state, nsteps):
    stepper, scheme = get_scheme("lsrk54")

    def body(s, _):
        b = dgf.budget(s, kgrid, params)
        inj = dgf.injection(s, kgrid, params)
        s2 = block_of_steps(s, kgrid, params, 1, scheme, stepper)
        dt = s2.t - s.t
        return s2, (b["dEdt"]*dt + inj["self"]*dt**2, b["dHdt"]*dt + inj["self_H"]*dt**2,
                    (b["eps_perp"] + b["eps_z"])*dt, b["eps"]*dt + inj["self"]*dt**2)

    end, out = jax.jit(lambda s: jax.lax.scan(body, s, None, nsteps))(state)
    return end, [float(np.sum(np.asarray(v, dtype=float))) for v in out]


@pytest.mark.fp64
def test_run_integrated_budget():
    """A forced, dissipative run from a developed-ish state with a FROZEN forcing envelope
    (so every dt sees the same forcing): sum over steps of (dEdt*dt + self*dt^2) from
    diagnostics.flrmhd.budget against the measured E(T) - E(0), and the same for H. The
    difference is the scheme against forward Euler (and the left-Riemann sum of the sinks):
    O(dt). Asserted, in units of the energy that went through the budget (injected +
    dissipated): error <= 1.0*dt at each of three dt and <= 5e-3 at the finest, for E and
    for H; and for E the fitted order in [0.7, 2.3] (measured 1.20 and 1.00)."""
    with checks() as c:
        for label, kw, rho_i, Z, tau in (_CASES[0], _CASES[2]):
            zt = Z/tau
            # mild dissipation rates (<~ 1 per unit time): the left-Riemann sum of a stiff
            # sink would dominate the first-order error (z_diss_k = 0.01 gives a kz^6 rate of
            # ~100 here and 2.4e-2 at the finest dt)
            eq = dict(z_diss_k=2e-4, z_hyper=3) if kw.get("z_spectral") else {}
            dts, errE, errH = (8e-3, 4e-3, 2e-3), [], []
            for dt in dts:
                params = _flr(rho_i, tau=tau, Z=Z, diss=(2e-2, 1e-2), hyper=1, eq=eq,
                              **_forced(kw, rho_i, Z, tau, "elsasser", dt=dt, forcing_tau=1e8))
                kgrid = jr.setup_kgrids(params)
                state = jr.initialize(_correlated(broadband_ic(params, seed=7, slope=-2.5,
                                                               amp=0.5)), params)
                fs, _ = shared_physics.ou_update(state.forcing_state, jax.random.key(23), 0.05,
                                                 params, kgrid)
                state = state._replace(forcing_state=fs)
                E0, H0 = invariants(state.fields, params, rho_i, zt)
                end, (sE, sH, sD, sI) = _budget_run(params, kgrid, state, int(round(0.32/dt)))
                E1, H1 = invariants(end.fields, params, rho_i, zt)
                errE.append(abs((E1 - E0) - sE)/(sI + sD))
                errH.append(abs((H1 - H0) - sH)/(sI + sD))
            oE, oH = fit_order(dts, errE), fit_order(dts, errH)
            c.check(f"{label}: injected {sI:.3f}, dissipated {sD:.3f}, dE = {E1 - E0:+.3f}, "
                    f"dH = {H1 - H0:+.3f} over T = 0.32", sI > 0.1 and sD > 0.02*sI)
            bound = lambda errs: all(e <= 1.0*dt for e, dt in zip(errs, dts))
            c.check(f"{label}: sum of budget dEdt == measured dE: {errE[0]:.1e} -> "
                    f"{errE[1]:.1e} -> {errE[2]:.1e} of (injected + dissipated) "
                    f"(<= 1.0*dt at every dt), order {oE:.2f}",
                    bound(errE) and errE[-1] <= 5e-3 and 0.7 <= oE <= 2.3, f"{errE}")
            # H: the bound only. Its first-order error is the sum of terms of both signs and
            # nearly cancels under z_spectral (3.8e-4, 4.4e-4, 2.8e-4: no clean order)
            c.check(f"{label}: sum of budget dHdt == measured dH: {errH[0]:.1e} -> "
                    f"{errH[1]:.1e} -> {errH[2]:.1e} of (injected + dissipated) "
                    f"(<= 1.0*dt at every dt; fitted order {oH:.2f}, not asserted)",
                    bound(errH) and errH[-1] <= 5e-3, f"{errH}")


def _per_stage(kw, rho_i, Z, tau, mode, dt, **extra):
    return _flr(rho_i, tau=tau, Z=Z, **_forced(kw, rho_i, Z, tau, mode, dt=dt,
                                                forcing_norm_per_step=False, **extra))


def test_injection_per_stage_normalization():
    """forcing_norm_per_step=False: the solver re-solves the scale at every stage as
    target/P, so the instantaneous power is the target and a step injects eps*dt -- there is
    NO self term (self* = 0), unlike the per-step path where F is frozen over the step.
    The exception is a scale sitting at its cap (P ~ 0: here a state at rest), where the
    scale is state-independent, F IS frozen, and the quadratic form is the whole injection:
    with the fixed-dt cap sqrt(2*target/(F_2 dt)) the step then injects self*dt^2 =
    target*dt. injection() against independent rates of the solver's own ForcingTerm in both
    regimes; the budget closure in this mode."""
    stepper, scheme = get_scheme("lsrk54")
    tol = 10*TOL
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            zt = Z/tau
            for mode in ("momentum", "elsasser"):
                dt = 0.004
                params = _per_stage(kw, rho_i, Z, tau, mode, dt)
                kgrid = jr.setup_kgrids(params)
                tgt = ((float(params.forcing_power),) if mode == "momentum"
                       else tuple(float(v) for v in params.forcing_power_elsasser))
                tag = f"{label}, {mode}, per-stage"
                # (a) a developed state: the scale is not capped
                state = _advance(_state(params, seed=7), kgrid, params, 12, scheme, stepper)
                f = np.asarray(state.fields)
                F = np.asarray(flrmhd.ForcingTerm(state, None, kgrid, params))
                inj = {k: float(v) for k, v in dgf.injection(state, kgrid, params).items()}
                Tp, Tm, TH = (v.sum() for v in elsasser_rates(f, F, params, rho_i, zt))
                ref = dict(eps_plus=Tp, eps_minus=Tm, eps=Tp + Tm, eps_H=TH)
                worst = max(abs(inj[k] - ref[k]) for k in ref)/sum(tgt)
                hit = (abs(inj["eps"]/tgt[0] - 1) if mode == "momentum" else
                       max(abs(inj["eps_plus"]/tgt[0] - 1), abs(inj["eps_minus"]/tgt[1] - 1)))
                c.check(f"{tag}, developed state: eps* == independent rates of the solver's "
                        f"forcing (worst {worst:.1e}) and eps is the target"
                        f"{'' if mode == 'momentum' else ' per channel'} ({hit:.1e}; eps_H = "
                        f"{inj['eps_H']:+.3f})", worst <= tol and hit <= (1e-11 if FP64 else
                                                                         1e-4),
                        f"{worst:.3e}, {hit:.3e}")
                c.check(f"{tag}, developed state: self, self_plus, self_minus, self_H are "
                        f"exactly 0 (the quadratic form of F is {elsasser_modal(F, params, rho_i, zt)[0].sum() + elsasser_modal(F, params, rho_i, zt)[1].sum():.3f}: "
                        f"NOT injected)", all(inj[k] == 0.0 for k in
                                              ("self", "self_plus", "self_minus", "self_H")))
                b = {k: float(v) for k, v in dgf.budget(state, kgrid, params).items()}
                sc = abs(b["eps"]) + b["eps_perp"] + b["eps_z"]
                cl = max(abs(b["closure_E"]), abs(b["closure_H"]))/sc
                c.check(f"{tag}, developed state: budget closure against the solver's full "
                        f"RHS {cl:.1e}", cl <= (1e-12 if FP64 else 2e-4), f"{cl:.3e}")
                # (b) at rest with one O-U kick: P = 0, the scale is the cap
                rest = jr.initialize(lambda *xyz: 0.0*_state_ic(params)(*xyz), params)
                fs, _ = shared_physics.ou_update(rest.forcing_state, jax.random.key(23), 0.05,
                                                 params, kgrid)
                rest = rest._replace(forcing_state=fs)
                F = np.asarray(flrmhd.ForcingTerm(rest, None, kgrid, params))
                inj = {k: float(v) for k, v in dgf.injection(rest, kgrid, params).items()}
                Sp, Sm, SH = (v.sum() for v in elsasser_modal(F, params, rho_i, zt))
                ref = dict(self_plus=Sp, self_minus=Sm, self_H=SH)
                ref["self"] = Sp + Sm
                worst = max(abs(inj[k] - ref[k]) for k in ref)/(Sp + Sm)
                c.check(f"{tag}, at rest (capped scale): eps = 0 exactly and self* == the "
                        f"independent quadratic forms of F (worst {worst:.1e}; self_H/self = "
                        f"{SH/(Sp + Sm):+.3f})", worst <= tol and inj["eps"] == 0.0
                        and inj["eps_H"] == 0.0 and Sp + Sm > 0, f"{worst:.3e}")
                e = abs(inj["self"]*dt/sum(tgt) - 1)
                c.check(f"{tag}, at rest: the capped step injects self*dt^2 = target*dt "
                        f"(rel {e:.1e})", e <= (1e-11 if FP64 else 1e-4), f"{e:.3e}")


def _state_ic(params, seed=7):
    return _correlated(broadband_ic(params, seed=seed))


def _injection_run(params, kgrid, state, nsteps):
    stepper, scheme = get_scheme("lsrk54")

    def body(s, _):
        inj = dgf.injection(s, kgrid, params)
        s2 = block_of_steps(s, kgrid, params, 1, scheme, stepper)
        return s2, dict(dt=s2.t - s.t, **inj)

    end, log = jax.jit(lambda s: jax.lax.scan(body, s, None, nsteps))(state)
    return end, {k: np.asarray(v, dtype=float) for k, v in log.items()}


@pytest.mark.fp64
def test_run_integrated_injection_per_stage():
    """Per-stage normalization, nu = 0, a frozen O-U envelope (every dt sees the same
    forcing), from a developed state: the measured E(T) - E(0) and H(T) - H(0) against the
    sum over steps of injection()'s (eps*dt + self*dt^2), where self is nonzero only on the
    steps whose scale is capped. Three dt. Asserted: E error <= 0.5*dt and H error <= 2.0*dt
    (relative), at every dt (measured 2026-11-12, dt = 0.004, 0.002, 0.001: E 7.5e-4, 8.8e-5,
    1.2e-4 in 2D and 3.3e-4, 4.5e-5, 3.1e-6 under z_spectral; H 3.2e-3, 1.5e-3, 4.4e-4 and
    2.3e-3, 4.0e-4, 1.7e-4). The naive forward-Euler sum that ALWAYS adds the quadratic form
    of F -- right for the per-step path, and what the docs said before 2026-11-12 --
    overshoots dE by 2.5e-2, 1.2e-2, 5.3e-3 (2D) and 1.6e-2, 8.2e-3, 4.0e-3 (z_spectral):
    asserted > 2*dt and at least 5x the error of injection()'s own sum. That is the
    first-order term the per-stage re-solve removes."""
    with checks() as c:
        for label, kw, rho_i, Z, tau in (_CASES[1], _CASES[2]):
            zt = Z/tau
            dts, eE, eH, eN, ncap = (4e-3, 2e-3, 1e-3), [], [], [], []
            for dt in dts:
                params = _per_stage(kw, rho_i, Z, tau, "elsasser", dt, forcing_tau=1e8)
                kgrid = jr.setup_kgrids(params)
                state = jr.initialize(_correlated(broadband_ic(params, seed=3, slope=-2.5,
                                                               amp=0.5)), params)
                fs, _ = shared_physics.ou_update(state.forcing_state, jax.random.key(17), 0.05,
                                                 params, kgrid)
                state = state._replace(forcing_state=fs)
                E0, H0 = invariants(state.fields, params, rho_i, zt)
                end, log = _injection_run(params, kgrid, state, int(round(0.2/dt)))
                E1, H1 = invariants(end.fields, params, rho_i, zt)
                d = log["dt"]
                # the quadratic form of the stepped F on every step, capped or not
                F2 = _quadratic_forms(params, kgrid, state, d.size, rho_i, zt)
                dE, dH = E1 - E0, H1 - H0
                eE.append(abs(np.sum(log["eps"]*d + log["self"]*d**2)/dE - 1))
                eH.append(abs(np.sum(log["eps_H"]*d + log["self_H"]*d**2)/dH - 1))
                eN.append(abs(np.sum(log["eps"]*d + F2*d**2)/dE - 1))
                ncap.append(int(np.count_nonzero(log["self"])))
            c.check(f"{label}: dE = {dE:+.4f}, dH = {dH:+.4f} over T = 0.2; capped steps "
                    f"{ncap} of {[int(round(0.2/dt)) for dt in dts]}", dE > 0.05)
            c.check(f"{label}: sum(eps*dt + self*dt^2) == measured dE: rel {eE[0]:.1e} -> "
                    f"{eE[1]:.1e} -> {eE[2]:.1e} (<= 0.5*dt)",
                    all(e <= 0.5*dt for e, dt in zip(eE, dts)), f"{eE}")
            c.check(f"{label}: ... and == measured dH: rel {eH[0]:.1e} -> {eH[1]:.1e} -> "
                    f"{eH[2]:.1e} (<= 2.0*dt)",
                    all(e <= 2.0*dt for e, dt in zip(eH, dts)), f"{eH}")
            c.check(f"{label}: (always adding the quadratic form of F overshoots dE by "
                    f"{eN[0]:.1e} -> {eN[1]:.1e} -> {eN[2]:.1e}: first order, > 2*dt and "
                    f">= 5x injection()'s error)",
                    all(n > 2*dt and n >= 5*e for n, e, dt in zip(eN, eE, dts)), f"{eN}")


def _quadratic_forms(params, kgrid, state, nsteps, rho_i, zt):
    """E[F] of the solver's ForcingTerm at the start of each of nsteps steps (test-local
    Elsasser forms), for the naive always-add-the-self-term sum."""
    stepper, scheme = get_scheme("lsrk54")

    def body(s, _):
        F = flrmhd.ForcingTerm(s, None, kgrid, params)
        return block_of_steps(s, kgrid, params, 1, scheme, stepper), F

    _, Fs = jax.jit(lambda s: jax.lax.scan(body, s, None, nsteps))(state)
    out = []
    for F in np.asarray(Fs):
        Sp, Sm, _ = elsasser_modal(F, params, rho_i, zt)
        out.append(Sp.sum() + Sm.sum())
    return np.array(out)


# ---------------------------------------------------------------------------- gate F2.8

def test_kperp_kz_spectrum():
    with checks() as c:
        for label, kw, rho_i, Z, tau in _CASES:
            zt = Z/tau
            zsp = bool(kw.get("z_spectral"))
            eq = dict(z_diss_k=0.02, z_hyper=3) if zsp else {}
            params = _flr(rho_i, tau=tau, Z=Z, diss=(4e-3, 1e-3), hyper=2, eq=eq, **kw)
            kgrid = jr.setup_kgrids(params)
            state = _state(params, seed=11)
            f = np.asarray(state.fields)
            bf = 1.37
            k, kz, E2, D2 = (np.asarray(v, dtype=float) for v in
                             dgf.kperp_kz_spectrum(state, kgrid, params, bin_factor=bf))
            en = float(dgf.energy(state, kgrid, params)["total"])
            dis = dgf.dissipation(state, kgrid, params)
            eps = float(dis["eps_perp"]) + float(dis["eps_z"])
            nzh = (params.nz//2 + 1) if zsp else 1
            c.check(f"{label}: shapes (kz: {kz.shape}, E: {E2.shape}) and kz = j 2 pi/Lz",
                    E2.shape == (nzh, k.size) and D2.shape == E2.shape and kz.shape == (nzh,)
                    and np.allclose(kz, np.arange(nzh)*(TWO_PI/params.Lz if zsp else 0.0)))
            c.check(f"{label}: sum E(kz, k) = total energy (rel {abs(E2.sum()/en - 1):.1e}), "
                    f"sum D(kz, k) = eps_perp + eps_z (rel {abs(D2.sum()/eps - 1):.1e})",
                    abs(E2.sum()/en - 1) <= TOL and abs(D2.sum()/eps - 1) <= TOL)
            _, sp, sm, _ = (np.asarray(v, dtype=float) for v in
                            dgf.perp_spectra(state, kgrid, params, bin_factor=bf))
            dk = min(TWO_PI/params.Lx, TWO_PI/params.Ly)*bf
            c.check(f"{label}: summed over kz it is (E_plus + E_minus)(k) dk of perp_spectra "
                    f"(rel {_rel(E2.sum(axis=0), (sp + sm)*dk, en):.1e} of E)",
                    _rel(E2.sum(axis=0), (sp + sm)*dk, en) <= TOL)
            # rows against first principles: fold +-kz of the per-mode energy and dissipation
            Ep, Em, _ = elsasser_modal(f, params, rho_i, zt)
            w = weights(params, rho_i, zt)
            n = norm(params)
            Dk = w["yfac"]*w["ksq"]**2*(4e-3*w["Gpe"]*np.abs(f[0])**2
                                        + 1e-3*w["ksq"]*np.abs(f[1])**2)/n
            if zsp:
                kzs = np.fft.fftfreq(params.nz)*params.nz*TWO_PI/params.Lz
                kzs[params.nz//2] = 0.0
                Dk = Dk + 0.02*(kzs**6)[:, None, None]*2*(Ep + Em)
            iz = np.abs(np.rint(np.fft.fftfreq(f.shape[1])*f.shape[1])).astype(int)
            rowsE, rowsD = np.zeros(nzh), np.zeros(nzh)
            np.add.at(rowsE, iz, (Ep + Em).sum(axis=(1, 2)))
            np.add.at(rowsD, iz, Dk.sum(axis=(1, 2)))
            e = max(_rel(E2.sum(axis=1), rowsE, en), _rel(D2.sum(axis=1), rowsD, eps))
            c.check(f"{label}: each |kz| row holds the +-kz planes' energy and dissipation "
                    f"(worst {e:.1e}; rows with energy: "
                    f"{np.flatnonzero(rowsE > 1e-12*en).tolist()})", e <= TOL, f"{e:.3e}")
            edges, (be,) = _np_binned(kgrid, params, (Ep + Em,), bf)
            c.check(f"{label}: the k_perp bins are perp_spectra's and match the independent "
                    f"binning ({_rel(E2.sum(axis=0), be[:k.size], en):.1e})",
                    _rel(E2.sum(axis=0), be[:k.size], en) <= TOL)
            # every (|kz|, k_perp) CELL against an independent numpy 2-D histogram of the
            # per-mode energy and dissipation (the marginals above cannot see a wrong joint
            # distribution, nor D's k_perp distribution at all)
            kmag3 = np.broadcast_to(np.sqrt(kgrid_ksq(params)), Ep.shape).ravel()
            izz = np.broadcast_to(iz[:, None, None], Ep.shape).ravel().astype(float)
            bins = [np.arange(nzh + 1) - 0.5, dk*np.arange(k.size + 1)]
            hE = np.histogram2d(izz, kmag3, bins=bins, weights=(Ep + Em).ravel())[0]
            hD = np.histogram2d(izz, kmag3, bins=bins, weights=Dk.ravel())[0]
            eE, eD = _rel(E2, hE, en), _rel(D2, hD, eps)
            c.check(f"{label}: E(kz, k) and D(kz, k) == an independent 2-D histogram of the "
                    f"per-mode E_k and D_k, cell by cell (worst {eE:.1e} of E, {eD:.1e} of "
                    f"eps; {np.count_nonzero(hE > 1e-9*en)} live cells; D is "
                    f"{np.abs(hD/hD.sum() - hE/hE.sum()).sum()/2:.2f} away from E's "
                    f"distribution)", eE <= TOL and eD <= TOL
                    and np.count_nonzero(hE > 1e-9*en) >= (10 if zsp else 5), f"{eE:.3e}, {eD:.3e}")
        # one mode at kz index 2 (and its mirror at -2): everything in row 2, one k bin
        params = _flr(0.5, **_Z3)
        kgrid = jr.setup_kgrids(params)
        state = jr.initialize(lambda x, y, z: jnp.stack([jnp.cos(3*x + 2*y + 2*z),
                                                         0.5*jnp.sin(3*x + 2*y + 2*z)]), params)
        k, kz, E2, _ = (np.asarray(v, dtype=float) for v in
                        dgf.kperp_kz_spectrum(state, kgrid, params, bin_factor=1.0))
        nzr = np.argwhere(E2 > 1e-12*E2.sum())
        c.check(f"single mode (3, 2, kz = 2): all the energy sits at row kz = {kz[2]:.1f} and "
                f"the bin of k = sqrt(13) (entries {nzr.tolist()})",
                nzr.tolist() == [[2, 3]] and kz[2] == 2.0)

        # z_spectral-only: axis 1 is read as kz and the normalization is nz^2. FLRMHD has no
        # finite-difference-z path yet (plan rung F6), so the refusal is exercised with a
        # Parameters stand-in that reports z_spectral=False
        class _FDz:
            z_spectral = False

            def __getattr__(self, name):
                return getattr(params, name)

        try:
            dgf.kperp_kz_spectrum(state, kgrid, _FDz())
            msg = ""
        except AssertionError as err:
            msg = str(err)
        c.check("a dims=3 Parameters without z_spectral is refused (assert naming "
                "z_spectral), not silently mis-normalized", "z_spectral-only" in msg, msg)


def test_spectral_break():
    with checks() as c:
        k = np.arange(1, 65)*0.75
        for kb, s1, s2 in ((9.0, -5/3, -3.8), (20.0, -1.5, -2.8), (5.5, -3.2, -1.0)):
            E = np.where(k < kb, (k/kb)**s1, (k/kb)**s2)*3.7
            out = dgf.spectral_break(k, E)
            c.check(f"synthetic broken power law (k* = {kb}, slopes {s1:.3f}, {s2:.2f}): "
                    f"k_break = {out['k_break']:.3f}, slopes {out['slope_lo']:.3f}, "
                    f"{out['slope_hi']:.3f}, resid {out['resid']:.1e}",
                    abs(out["k_break"]/kb - 1) < 0.02 and abs(out["slope_lo"] - s1) < 0.02
                    and abs(out["slope_hi"] - s2) < 0.02 and out["resid"] < 5e-3)
        # zeros and non-positive k are ignored; a fit range restricts the points
        E = np.where(k < 9.0, (k/9.0)**(-5/3), (k/9.0)**(-3.8))
        E2 = E.copy()
        E2[40:] = 0.0
        a = dgf.spectral_break(np.concatenate([[0.0], k]), np.concatenate([[1.0], E2]))
        b = dgf.spectral_break(k, E, kmax=k[39])
        c.check("zero-energy bins and k = 0 are dropped; kmax restricts the fit the same way",
                abs(a["k_break"] - b["k_break"]) < 1e-12 and abs(a["k_break"]/9.0 - 1) < 0.02)
        rng = np.random.default_rng(3)
        noisy = E*np.exp(0.1*rng.standard_normal(E.size))
        out = dgf.spectral_break(k, noisy)
        c.check(f"10% log-normal noise: k_break = {out['k_break']:.2f} within 20% of 9, "
                f"slopes {out['slope_lo']:.2f}, {out['slope_hi']:.2f}",
                abs(out["k_break"]/9.0 - 1) < 0.2 and abs(out["slope_lo"] + 5/3) < 0.3
                and abs(out["slope_hi"] + 3.8) < 0.3)
        # kmin: a forcing-range bump at low k is a second, sharper kink -- without kmin
        # the two-slope fit locks onto it; with kmin above it the inertial/steep break is
        # recovered
        bump = E*np.where(k <= 2.5, 30.0*(k/2.5)**3.0, 1.0)
        free = dgf.spectral_break(k, bump)
        cut = dgf.spectral_break(k, bump, kmin=3.0)
        c.check(f"low-k contamination (k <= 2.5): without kmin the fit breaks at "
                f"{free['k_break']:.2f}; with kmin = 3 it returns k_break = "
                f"{cut['k_break']:.2f} (true 9), slopes {cut['slope_lo']:.2f}, "
                f"{cut['slope_hi']:.2f}",
                abs(cut["k_break"]/9.0 - 1) < 0.02 and abs(cut["slope_lo"] + 5/3) < 0.02
                and abs(cut["slope_hi"] + 3.8) < 0.02 and abs(free["k_break"]/9.0 - 1) > 0.3)
        both = dgf.spectral_break(k, bump, kmin=3.0, kmax=k[39])
        c.check("kmin and kmax compose (same break on the doubly restricted range)",
                abs(both["k_break"]/9.0 - 1) < 0.02)
        try:
            dgf.spectral_break(k[:3], E[:3])
            msg = ""
        except ValueError as err:
            msg = str(err)
        c.check("fewer than 4 usable points is a ValueError", "at least 4" in msg, msg)
        # on a diagnostics spectrum it runs and returns finite numbers
        params = _flr(0.3)
        kgrid = jr.setup_kgrids(params)
        kk, sp, _, _ = dgf.perp_spectra(_state(params), kgrid, params)
        out = dgf.spectral_break(kk, sp)
        c.check(f"on perp_spectra's E_plus(k): finite (k_break = {out['k_break']:.2f})",
                all(np.isfinite(v) for v in out.values()))


def test_module_is_listed():
    c_all = dg.__all__
    with checks() as c:
        c.check("diagnostics.__all__ lists flrmhd and re-exports none of its names",
                "flrmhd" in c_all and dg.flrmhd is dgf
                and not any(hasattr(dg, n) for n in ("helicity", "perp_spectra", "fluxes",
                                                     "budget", "spectral_break", "injection",
                                                     "transfer", "dissipation",
                                                     "kperp_kz_spectrum"))
                and dg.energy is dg.rmhd.energy)
        for name in ("energy", "helicity", "perp_spectra", "kperp_kz_spectrum", "dissipation",
                     "injection", "transfer", "fluxes", "budget", "spectral_break"):
            c.check(f"diagnostics.flrmhd.{name} exists", callable(getattr(dgf, name, None)))
        # an RMHD kgrid, or an overridden one, is refused by name rather than misread
        pr = jr.Parameters(nx=16, ny=16, Lx=TWO_PI, Ly=TWO_PI, cfl_safety=0.5,
                           eqpars=dict(diss=0.01, hyper=1))
        kr = jr.setup_kgrids(pr)
        sr = jr.initialize(lambda x, y: jnp.stack([jnp.cos(x) + 0*y, jnp.sin(y) + 0*x]), pr)
        pf = _flr(0.3, nx=16, ny=16)
        ko = jr.setup_kgrids(pf, overrides={"rho_i": 0.5})
        sf = jr.initialize(lambda x, y: jnp.stack([jnp.cos(x) + 0*y, jnp.sin(y) + 0*x]), pf)
        for name in ("energy", "helicity", "imbalance", "perp_spectra", "kperp_kz_spectrum",
                     "dissipation", "injection", "transfer", "fluxes", "budget"):
            msgs = []
            # RMHD params + RMHD kgrid; overridden FLRMHD kgrid; and an FLRMHD Parameters
            # handed an RMHD kgrid (same shapes, no FLR factors)
            for st, kg, pp in ((sr, kr, pr), (sf, ko, pf), (sf, kr, pf)):
                try:
                    getattr(dgf, name)(st, kg, pp)
                    msgs.append("")
                except ValueError as err:
                    msgs.append(str(err))
            c.check(f"diagnostics.flrmhd.{name}: an RMHD kgrid (with RMHD or with FLRMHD "
                    f"Parameters) and an overridden kgrid are ValueErrors",
                    "FLRMHD" in msgs[0] and "overrides" in msgs[1]
                    and "without FLR factors" in msgs[2] and "'FLRMHD'" in msgs[2], f"{msgs}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
