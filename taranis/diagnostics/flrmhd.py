# FLR-MHD diagnostics (plans/FLRMHD_PLAN.md rung F2): energies and generalized helicity,
# generalized-Elsasser spectra, dissipation and injection rates, nonlinear transfers and
# fluxes (with the sub-filter transfer at a cutoff), the E and H budgets, and a spectral-break
# fit. Conventions of record: docs/numerics.md "FLR-MHD" -> "Diagnostics, flux and budget
# conventions". Read-only observers: diagnostics -> physics, never back.
#
# Everything is on the shared perp_reduce volume-average normalization (ky-doubling,
# /nz*(nx*ny)^2, nz^2 under z_spectral), so the numbers are comparable with
# diagnostics.rmhd, forcing_power and each other. The weights are kgrid.eq's (FLRFactors):
# at eqpars["rho_i"] = 0 every function here is its RMHD counterpart.
#
# NB diagnostics.energy / perpspec (the RMHD ones) on an FLRMHD state silently return the
# RMHD energy of the same arrays -- use the functions here.
#
# The transfers, the injection and the budget call the SOLVER's own flrmhd.NonlinearTerm /
# ForcingTerm / kgrid.lin.apply_L, so they cannot drift from what is stepped. The kgrid must
# be the solver's (setup_kgrids(params), no overrides): the weights are read from kgrid.eq
# and the dissipation coefficients from params.eqpars.
import jax.numpy as jnp
import numpy as np

from ..physics import construct_rhs, equation_registry, flrmhd, gdi, rmhd
from ..physics.shared_physics import perp_reduce
from .core import _binned


# ------------------------------------------------------------------------ per-mode kernels

def _check_kgrid(kgrid, params):
    # the weights come from kgrid.eq and the coefficients from params.eqpars: both must be
    # the solver's own. Only a kgrid with no FLR factors (another equation set's) or with
    # overrides can be detected: a kgrid built from a DIFFERENT FLRMHD Parameters (another
    # rho_i, tau, Z, box or grid of the same shape) is not, and gives silently inconsistent
    # numbers -- pass setup_kgrids(params) of the same params.
    if params.eqtype != "FLRMHD" or kgrid.eq is None:
        raise ValueError(f"diagnostics.flrmhd needs an FLRMHD Parameters and its kgrid "
                         f"(setup_kgrids(params)); got eqtype={params.eqtype!r} and a kgrid "
                         f"{'without' if kgrid.eq is None else 'with'} FLR factors (kgrid.eq)")
    if kgrid.overrides is not None:
        raise ValueError("diagnostics.flrmhd: this kgrid carries overrides (the stability "
                         "harness's seam); its factors are not those of params.eqpars. Build "
                         "a Parameters with the values you want and its own kgrid")


def _theta(fields, kgrid):
    # generalized Elsasser potentials Theta^+- = kappa*phi +- psi
    chi = kgrid.eq.kappa*fields[0]
    return chi + fields[1], chi - fields[1]


def _rates(fields, dfields, kgrid):
    # per-mode rates of change (E^+_k, E^-_k, H_k) when the fields change at dfields:
    #   T^+- = (1/2) ksq Re(Theta^+-* dTheta^+-),   T_H = (T^+ - T^-)/v_ph(k_perp)
    # ky-doubled, un-normalized (perp_reduce normalizes). _rates(f, f)/2 are the invariants'
    # own per-mode densities E^+-_k = (1/4) ksq |Theta^+-|^2 and H_k.
    tp, tm = _theta(fields, kgrid)
    dp, dm = _theta(dfields, kgrid)
    w = 0.5*kgrid.ksq*kgrid.yfac
    Tp = w*jnp.real(jnp.conj(tp)*dp)
    Tm = w*jnp.real(jnp.conj(tm)*dm)
    return Tp, Tm, (Tp - Tm)/kgrid.eq.vph


def _densities(fields, kgrid):
    # per-mode (E^+_k, E^-_k, H_k), ky-doubled, un-normalized
    return tuple(0.5*t for t in _rates(fields, fields, kgrid))


def _dissipation_densities(fields, kgrid, params):
    # per-mode (D_perp, D_z, DH_perp, DH_z), ky-doubled, un-normalized (docs/numerics.md
    # "Invariants"): the rates at which L's diagonal removes E and H
    #   D_perp  = k^2h [nu_phi G pe |phi|^2 + nu_psi ksq |psi|^2]     D_z  = nu_z kz^2hz 2 E_k
    #   DH_perp = (nu_phi + nu_psi) k^2h H_k                           DH_z = 2 nu_z kz^2hz H_k
    # with kz the solver's (Nyquist-zeroed, as in L).
    _, _, _, diss, hyper, zdiss, z_hyper = flrmhd._eqpars(params)
    nu = np.broadcast_to(np.asarray(diss, dtype=float).reshape(-1), (2,))
    eq = kgrid.eq
    phi2 = jnp.abs(fields[0])**2*kgrid.yfac
    psi2 = jnp.abs(fields[1])**2*kgrid.yfac
    Hk = eq.G*jnp.real(jnp.conj(fields[0])*fields[1])*kgrid.yfac
    k2h = kgrid.ksq**hyper
    wphi, wpsi = eq.G*eq.pe*phi2, kgrid.ksq*psi2            # their sum is 2 E_k
    D_perp = k2h*(float(nu[0])*wphi + float(nu[1])*wpsi)
    DH_perp = float(nu[0] + nu[1])*k2h*Hk
    if params.z_spectral:
        kz2h = rmhd._kz_deriv(kgrid, params)**(2*z_hyper)
        D_z = zdiss*kz2h*(wphi + wpsi)
        DH_z = 2.0*zdiss*kz2h*Hk
    else:
        D_z, DH_z = jnp.zeros_like(D_perp), jnp.zeros_like(DH_perp)
    return D_perp, D_z, DH_perp, DH_z


def _nonlinear(fields, state, kgrid, params):
    # the solver's own nonlinear term on `fields`
    s = state._replace(fields=fields)
    return flrmhd.NonlinearTerm(s, flrmhd.grad(s, kgrid, params), kgrid, params)


def _bins(params, bin_factor):
    # (kunit, kmax) of the perpendicular bins: perpspec's range, extended to the largest
    # dealiased k_perp where that lies beyond it (a non-square box), so that binned sums of a
    # dealiased quantity always equal its total
    kunit = min(2*np.pi/params.Lx, 2*np.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    _, _, ksq, _, _, perp_dealias = gdi._perp_grids_np(params)
    return kunit, max(kmax, float(np.sqrt(np.max(ksq[perp_dealias]))))


def _binned_perp(modal, kgrid, params, bin_factor):
    # (k centers, one spectral density per per-mode array): z-summed, then _binned
    kunit, kmax = _bins(params, bin_factor)
    specs = tuple(perp_reduce(m, params, axis=0) for m in modal)
    return _binned(jnp.sqrt(kgrid.ksq), specs, kunit, kmax, bin_factor)


# -------------------------------------------------------------------- invariants, spectra

def energy(state, kgrid, params):
    # dict(E_ion, E_e, E_mag, E_plus, E_minus, total):
    #   E_ion = (1/2)<G|phi|^2>        (ExB flow / ion response)
    #   E_e   = (rho_s^2/2)<nhat^2>    (electron pressure), rho_s^2 = Z rho_i^2/(2 tau)
    #   E_mag = (1/2)<|grad psi|^2>
    #   E_plus, E_minus = (1/4)<|grad Theta^+-|^2>
    # total = E_ion + E_e + E_mag = E_plus + E_minus.
    _check_kgrid(kgrid, params)
    rho_i, tau, Z = flrmhd._eqpars(params)[:3]
    phi, psi = state.fields[0], state.fields[1]
    eq = kgrid.eq
    phi2 = jnp.abs(phi)**2*kgrid.yfac
    E_ion = perp_reduce(0.5*eq.G*phi2, params)
    E_e = (0.25*Z*rho_i**2/tau)*perp_reduce(eq.G**2*phi2, params)
    E_mag = perp_reduce(0.5*kgrid.ksq*jnp.abs(psi)**2*kgrid.yfac, params)
    Ep, Em, _ = _densities(state.fields, kgrid)
    return dict(E_ion=E_ion, E_e=E_e, E_mag=E_mag, E_plus=perp_reduce(Ep, params),
                E_minus=perp_reduce(Em, params), total=E_ion + E_e + E_mag)


def helicity(state, kgrid, params):
    # generalized helicity H = sum_k (E^+_k - E^-_k)/v_ph(k_perp) = sum_k G Re(phi* psi)
    # = -<nhat psi>; the RMHD cross-helicity <grad phi . grad psi> at rho_i = 0
    _check_kgrid(kgrid, params)
    return perp_reduce(_densities(state.fields, kgrid)[2], params)


def imbalance(state, kgrid, params):
    # dict(sigma_H = H/E, sigma_c = (E^+ - E^-)/(E^+ + E^-)). Equal only while the energy
    # sits at k_perp rho_i << 1 (v_ph ~ 1); the paper's sigma_c is the second.
    _check_kgrid(kgrid, params)
    Ep, Em, Hk = (perp_reduce(m, params) for m in _densities(state.fields, kgrid))
    return dict(sigma_H=Hk/(Ep + Em), sigma_c=(Ep - Em)/(Ep + Em))


def perp_spectra(state, kgrid, params, bin_factor=2.0):
    # (k, E_plus(k), E_minus(k), E_H(k)): perpendicular spectral densities, z-summed;
    # sum(E_plus + E_minus)*dk = E and sum(E_H)*dk = H, dk = bin_factor*kunit. E_H(k) is
    # binned from the per-mode (E^+_k - E^-_k)/v_ph(k): the 1/v_ph is per mode, not per bin.
    _check_kgrid(kgrid, params)
    return _binned_perp(_densities(state.fields, kgrid), kgrid, params, bin_factor)


def kperp_kz_spectrum(state, kgrid, params, bin_factor=2.0):
    # (k, kz, E, D): energy and dissipation rate per (|kz|, k_perp bin), each (nz//2+1, nbins)
    # and NOT divided by the bin widths: sum(E) = energy()["total"], sum(D) = eps_perp + eps_z,
    # and E summed over kz is (E_plus + E_minus)*dk of perp_spectra. kz >= 0, the +-kz planes
    # folded. Requires the whole z domain on this rank; dims=2 gives a single kz = 0 row.
    # z_spectral (or dims=2) ONLY: axis 1 of the fields must be kz, and the normalization
    # below is the z_spectral one (nz^2); a finite-difference-z state would need a z-FFT first.
    _check_kgrid(kgrid, params)
    assert params.size == 1, "kperp_kz_spectrum requires the full z-domain on one rank"
    assert params.spatial_dimensions == 2 or params.z_spectral, (
        "kperp_kz_spectrum is z_spectral-only: it reads axis 1 as kz and its normalization "
        "is the z_spectral one")
    d3 = params.spatial_dimensions == 3
    nz = params.nz if d3 else 1
    fields = state.fields
    Ep, Em, _ = _densities(fields, kgrid)
    D_perp, D_z, _, _ = _dissipation_densities(fields, kgrid, params)
    norm = 1.0/(float(nz)*float(params.nx*params.ny))**2
    kunit, kmax = _bins(params, bin_factor)
    dk = kunit*bin_factor
    edges = jnp.arange(0, kmax + dk, dk, dtype=kgrid.ksq.dtype)
    nb = edges.shape[0] - 1
    kmag = jnp.sqrt(kgrid.ksq).reshape(-1)
    # numpy-histogram semantics: [lo, hi) bins, the last one closed, out-of-range dropped
    idx = jnp.clip(jnp.searchsorted(edges, kmag, side="right") - 1, 0, nb - 1)
    inside = kmag <= edges[-1]
    iz = np.abs(np.rint(np.fft.fftfreq(nz)*nz)).astype(int)        # |kz| index of each plane

    def fold(modal):
        w = jnp.where(inside, (modal*norm).reshape(nz, -1), 0.0)
        per_plane = jnp.zeros((nz, nb), dtype=w.dtype).at[:, idx].add(w)
        return jnp.zeros((nz//2 + 1, nb), dtype=w.dtype).at[iz].add(per_plane)

    kz = jnp.arange(nz//2 + 1, dtype=kgrid.ksq.dtype)*(2*jnp.pi/params.Lz if d3 else 0.0)
    return (edges[1:] + edges[:-1])/2, kz, fold(Ep + Em), fold(D_perp + D_z)


# ------------------------------------------------------------ dissipation, injection, budget

def dissipation(state, kgrid, params):
    # dict(eps_perp, eps_z, epsH_perp, epsH_z): the rates at which the linear operator's
    # diagonal removes E (non-negative) and H (signed, following H_k)
    _check_kgrid(kgrid, params)
    D = tuple(perp_reduce(m, params)
              for m in _dissipation_densities(state.fields, kgrid, params))
    return dict(eps_perp=D[0], eps_z=D[1], epsH_perp=D[2], epsH_z=D[3])


def injection(state, kgrid, params):
    # dict(eps_plus, eps_minus, eps, eps_H, self_plus, self_minus, self, self_H) from one call
    # of the solver's flrmhd.ForcingTerm F: eps* are the instantaneous cross terms (rates) and
    # self* the second-order coefficients, such that the step of length dt that STARTS from
    # this state injects
    #     dE = eps*dt + self*dt^2,   dH = eps_H*dt + self_H*dt^2
    # up to the scheme's O(dt) relative difference from forward Euler (the self term needs the
    # step's dt, which a state does not carry). All zero unforced.
    #  - forcing_norm_per_step=True (default): F is frozen over the step (the scale read is
    #    state.forcing_scale, the next step's), so self* are the quadratic forms E^+-[F],
    #    E[F], H[F] and dE is the per-step target exactly.
    #  - forcing_norm_per_step=False: the scale is re-solved at every stage as target/P, so
    #    the instantaneous power IS the target throughout the step and nothing is added:
    #    self* = 0 -- EXCEPT for a channel whose scale sits at its cap (rmhd._scale_epilogue:
    #    P ~ 0, e.g. a start from rest; recognised by its cross term missing the target),
    #    where the scale is state-independent, F is frozen and that channel's quadratic form
    #    is its self term. (Momentum mode has one scale: both channels together.)
    _check_kgrid(kgrid, params)
    if not params.forcing:
        z = jnp.zeros((), dtype=kgrid.ksq.dtype)
        return dict(eps_plus=z, eps_minus=z, eps=z, eps_H=z, self_plus=z, self_minus=z,
                    self_H=z, **{"self": z})
    F = flrmhd.ForcingTerm(state, None, kgrid, params)
    Tp, Tm, TH = (perp_reduce(m, params) for m in _rates(state.fields, F, kgrid))
    Ep, Em, _ = _densities(F, kgrid)
    Sp, Sm = perp_reduce(Ep, params), perp_reduce(Em, params)
    SHp, SHm = perp_reduce(Ep/kgrid.eq.vph, params), perp_reduce(Em/kgrid.eq.vph, params)
    if not params.forcing_norm_per_step:
        tol = float(np.sqrt(np.finfo(np.dtype(kgrid.ksq.dtype)).eps))
        if params.forcing_mode == "momentum":
            tgt = float(params.forcing_power)
            mp = mm = jnp.abs(Tp + Tm - tgt) > tol*abs(tgt)
        else:
            tp, tm = (float(v) for v in params.forcing_power_elsasser)
            mp, mm = jnp.abs(Tp - tp) > tol*abs(tp), jnp.abs(Tm - tm) > tol*abs(tm)
        Sp, SHp = jnp.where(mp, Sp, 0.0), jnp.where(mp, SHp, 0.0)
        Sm, SHm = jnp.where(mm, Sm, 0.0), jnp.where(mm, SHm, 0.0)
    return dict(eps_plus=Tp, eps_minus=Tm, eps=Tp + Tm, eps_H=TH, self_plus=Sp, self_minus=Sm,
                self_H=SHp - SHm, **{"self": Sp + Sm})


def budget(state, kgrid, params):
    # dict(eps, eps_perp, eps_z, dEdt, eps_H, epsH_perp, epsH_z, dHdt, closure_E, closure_H):
    #     dEdt = eps - eps_perp - eps_z,    dHdt = eps_H - epsH_perp - epsH_z
    # (instantaneous; eps, eps_H are injection()'s cross terms). closure_X = the rate of X
    # under the solver's own full right-hand side (kgrid.lin.apply_L + construct_rhs, forcing
    # included) minus dXdt: round-off by construction -- nonzero means the formulas of this
    # module and the solver's operators disagree. Costs one RHS.
    _check_kgrid(kgrid, params)
    inj = injection(state, kgrid, params)
    d = dissipation(state, kgrid, params)
    dEdt = inj["eps"] - d["eps_perp"] - d["eps_z"]
    dHdt = inj["eps_H"] - d["epsH_perp"] - d["epsH_z"]
    rhs = construct_rhs(equation_registry[params.eqtype])(state, kgrid, params)[0]
    full = rhs + kgrid.lin.apply_L(state.fields)
    Tp, Tm, TH = (perp_reduce(m, params) for m in _rates(state.fields, full, kgrid))
    return dict(eps=inj["eps"], eps_perp=d["eps_perp"], eps_z=d["eps_z"], dEdt=dEdt,
                eps_H=inj["eps_H"], epsH_perp=d["epsH_perp"], epsH_z=d["epsH_z"], dHdt=dHdt,
                closure_E=(Tp + Tm) - dEdt, closure_H=TH - dHdt)


# --------------------------------------------------------------------- transfers and fluxes

def _transfer_modal(state, kgrid, params, kcut):
    # per-mode (T_plus, T_minus, T_H) of the nonlinear term; with kcut, the sub-filter part
    fields = state.fields
    modal = _rates(fields, _nonlinear(fields, state, kgrid, params), kgrid)
    if kcut is None:
        return modal
    low = jnp.sqrt(kgrid.ksq) < kcut                       # the retained band, k_perp < kcut
    f_lo = fields*low
    resolved = _rates(f_lo, _nonlinear(f_lo, state, kgrid, params), kgrid)
    return tuple((a - b)*low for a, b in zip(modal, resolved))


def transfer(state, kgrid, params, bin_factor=2.0, kcut=None):
    # (k, T_plus(k), T_minus(k), T_H(k)): nonlinear transfer spectral densities from ONE call
    # of the solver's flrmhd.NonlinearTerm N,
    #     T^+-(k) = (1/2) ksq Re(Theta^+-* N_Theta^+-),  N_Theta^+- = kappa N_phi +- N_psi,
    #     T_H(k) = (T^+ - T^-)/v_ph(k) per mode;      T_E = T_plus + T_minus.
    # sum(T)*dk = 0 for T_E and T_H (round-off).
    # kcut: the SUB-FILTER transfer instead (one more RHS) -- T[f](k) - T[f^<](k) for
    # k_perp < kcut, 0 above, f^< the state low-passed to k_perp < kcut: the rate at which
    # interactions involving at least one mode beyond kcut change E^+-, H of the retained
    # band at wavenumber k. Its sum over k is -Pi(kcut).
    _check_kgrid(kgrid, params)
    return _binned_perp(_transfer_modal(state, kgrid, params, kcut), kgrid, params, bin_factor)


def fluxes(state, kgrid, params, bin_factor=2.0, kcut=None):
    # (K, Pi, Pi_plus, Pi_minus, Pi_H): Pi_X(K) = -sum_{k_perp < K} T_X(k), K the UPPER edges
    # of transfer()'s bins ([lo, hi) bins). Pi > 0 is transfer toward small scales. The last
    # entry is the conservation residual (round-off). With kcut: the cumulative sub-filter
    # flux, equal to the full Pi_X(kcut) at and beyond K = kcut when kcut is a bin edge.
    _check_kgrid(kgrid, params)
    k, Tp, Tm, TH = transfer(state, kgrid, params, bin_factor, kcut)
    dk = min(2*np.pi/params.Lx, 2*np.pi/params.Ly)*bin_factor
    Pp, Pm, PH = (-jnp.cumsum(T)*dk for T in (Tp, Tm, TH))
    return k + 0.5*dk, Pp + Pm, Pp, Pm, PH


# -------------------------------------------------------------------------- spectral break

def spectral_break(k, E, kmin=None, kmax=None, ncand=400):
    # Continuous two-slope (broken power law) least-squares fit of log E against log k over
    # kmin <= k <= kmax (default: every k > 0 with E > 0): for each candidate break k_b on a
    # log-spaced grid strictly inside the fit range, linear least squares for
    #     log E = c + slope_lo*min(x - x_b, 0) + slope_hi*max(x - x_b, 0),  x = log k,
    # keeping the k_b of smallest residual. Returns dict(k_break, slope_lo, slope_hi, resid)
    # (resid = rms residual in log E). Host numpy. Needs >= 4 points.
    k = np.asarray(k, dtype=float)
    E = np.asarray(E, dtype=float)
    keep = (k > 0) & (E > 0) & np.isfinite(E)
    if kmin is not None:
        keep &= k >= kmin
    if kmax is not None:
        keep &= k <= kmax
    x, y = np.log(k[keep]), np.log(E[keep])
    if x.size < 4:
        raise ValueError(f"spectral_break: needs at least 4 points with k > 0 and E > 0 in "
                         f"the fit range, got {x.size}")
    order = np.argsort(x)
    x, y = x[order], y[order]
    best = None
    # candidates between the second and the second-to-last point: each side keeps >= 2 points
    for xb in np.linspace(x[1], x[-2], int(ncand)):
        A = np.stack([np.ones_like(x), np.minimum(x - xb, 0.0), np.maximum(x - xb, 0.0)],
                     axis=1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        resid = float(np.sqrt(np.mean((A @ coef - y)**2)))
        if best is None or resid < best[0]:
            best = (resid, xb, coef)
    resid, xb, coef = best
    return dict(k_break=float(np.exp(xb)), slope_lo=float(coef[1]), slope_hi=float(coef[2]),
                resid=resid)
