# Shared helpers of the FLR-MHD gate files (plans/FLRMHD_PLAN.md rungs F1-F3). Not a test file.
#
# Everything here is INDEPENDENT of taranis/physics/flrmhd.py: numpy only, Gamma_0 by
# quadrature (never jax's i0e, never the module's series), and the factors written in the
# paper's form. A gate that compared the module against a copy of its own formulas would be
# blind to any mutation applied to both.
import numpy as np


def one_minus_gamma0_ref(alpha):
    """1 - Gamma_0(alpha) = 1 - I_0(alpha) exp(-alpha), by the cancellation-free quadrature
        1 - Gamma_0 = -(1/pi) int_0^pi expm1(-alpha (1 - cos t)) dt
    (plan section 14), as the periodic trapezoid rule over a full period: every term is
    >= 0, so nothing cancels, and the rule converges spectrally once the node count is well
    past sqrt(alpha) (the integrand's Fourier coefficients are I_n(alpha) exp(-alpha))."""
    alpha = np.asarray(alpha, dtype=float)
    amax = float(alpha.max()) if alpha.size else 0.0
    if not (np.all(alpha >= 0) and amax <= 1e7):
        # the node count grows as sqrt(alpha): refuse rather than allocate without bound
        raise ValueError(f"one_minus_gamma0_ref: alpha must lie in [0, 1e7], got max {amax!r}")
    n = 512 + int(40*np.sqrt(amax))
    t = 2*np.pi*np.arange(n)/n
    w = 1.0 - np.cos(t)
    flat = alpha.reshape(-1)
    out = np.empty_like(flat)
    chunk = max(1, min(256, 2_000_000//n))                  # <= 16 MB of work array
    for i in range(0, flat.size, chunk):
        out[i:i + chunk] = np.mean(-np.expm1(-flat[i:i + chunk, None]*w[None, :]), axis=1)
    return out.reshape(alpha.shape)


def flr_ref(ksq, rho_i, z_over_tau):
    """The k-local FLR factors from the paper's definitions (docs/numerics.md "FLR-MHD"),
    float64 numpy: dict(G, inv_G, pe, a, kappa, vph, omg) on the array ksq. k_perp = 0
    takes the continuous limits (G = inv_G = 0, everything else 1)."""
    ksq = np.asarray(ksq, dtype=float)
    k0 = ksq == 0
    ks = np.where(k0, 1.0, ksq)
    omg = one_minus_gamma0_ref(0.5*ks*rho_i**2)           # 1 - Gamma_0
    G = 2.0*omg/rho_i**2                                   # nhat = -G phi
    pe = 1.0 + z_over_tau*omg                              # rho_s^2 G = (Z/tau)(1 - Gamma_0)
    # the paper's v_ph = (k rho_i/sqrt 2) [1/(1 - Gamma_0) + Z/tau]^(1/2)
    vph = np.sqrt(ks)*rho_i/np.sqrt(2.0)*np.sqrt(1.0/omg + z_over_tau)
    out = dict(G=G, inv_G=1.0/G, pe=pe, a=ks/G, kappa=np.sqrt(G*pe/ks), vph=vph, omg=omg)
    limits = dict(G=0.0, inv_G=0.0, pe=1.0, a=1.0, kappa=1.0, vph=1.0, omg=0.0)
    return {name: np.where(k0, limits[name], v) for name, v in out.items()}


def kgrid_ksq(params):
    """(nkx, nky) float64 k_perp^2 of the Parameters' grid, rebuilt here."""
    kx = np.fft.fftfreq(params.nx)*params.nx*2*np.pi/params.Lx
    ky = np.fft.rfftfreq(params.ny)*params.ny*2*np.pi/params.Ly
    return kx[:, None]**2 + ky[None, :]**2


def yfac(params):
    """rfft2 ky-doubling factor (1 at ky = 0 and Nyquist, else 2), (nky,)."""
    f = np.full(params.ny//2 + 1, 2.0)
    f[0] = f[-1] = 1.0
    return f


def norm(params):
    """The shared volume-average normalization of a k-space quadratic sum
    (shared_physics.perp_reduce): nz (nx ny)^2, with nz^2 under z_spectral."""
    nz = params.nz if params.spatial_dimensions == 3 else 1
    return (nz**2 if params.z_spectral else nz)*float(params.nx*params.ny)**2


def invariants(fields, params, rho_i, z_over_tau):
    """(E, H) of a (2, nz, nkx, nky) fields array from the docs' formulas
        E = (1/2) sum_k [G pe |phi|^2 + ksq |psi|^2],   H = sum_k G Re(phi^* psi)
    with the weights from flr_ref (rho_i = 0: G = ksq, pe = 1)."""
    w = weights(params, rho_i, z_over_tau)
    phi, psi = np.asarray(fields[0]), np.asarray(fields[1])
    E = 0.5*np.sum(w["yfac"]*(w["Gpe"]*np.abs(phi)**2 + w["ksq"]*np.abs(psi)**2))
    H = np.sum(w["yfac"]*w["G"]*np.real(np.conj(phi)*psi))
    return E/norm(params), H/norm(params)


def weights(params, rho_i, z_over_tau):
    """dict(ksq, G, Gpe, yfac) of (nkx, nky) float64 quadratic-form weights."""
    ksq = kgrid_ksq(params)
    if rho_i == 0:
        G, pe = ksq, np.ones_like(ksq)
    else:
        r = flr_ref(ksq, rho_i, z_over_tau)
        G, pe = r["G"], r["pe"]
    return dict(ksq=ksq, G=G, Gpe=G*pe, yfac=yfac(params)[None, :])


def rates(fields, dfields, params, rho_i, z_over_tau):
    """Per-mode (T_E, T_H): the rates of change of E and H when the fields change at
    dfields, T_E = G pe Re(phi^* dphi) + ksq Re(psi^* dpsi) and
    T_H = G Re(phi^* dpsi + psi^* dphi), ky-doubled, on the shared normalization."""
    w = weights(params, rho_i, z_over_tau)
    phi, psi = np.asarray(fields[0]), np.asarray(fields[1])
    dphi, dpsi = np.asarray(dfields[0]), np.asarray(dfields[1])
    TE = w["yfac"]*(w["Gpe"]*np.real(np.conj(phi)*dphi) + w["ksq"]*np.real(np.conj(psi)*dpsi))
    TH = w["yfac"]*w["G"]*np.real(np.conj(phi)*dpsi + np.conj(psi)*dphi)
    return TE/norm(params), TH/norm(params)


def broadband_ic(params, seed, slope=-1.5, amp=1.0, kz_modes=2):
    """A real-space initial-condition function for jr.initialize: two independent random
    fields with a power-law perpendicular amplitude spectrum over the whole dealiased band
    (and, in 3D, a few kz modes), normalized to rms `amp` each. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    nx, ny = params.nx, params.ny
    d3 = params.spatial_dimensions == 3
    nz = params.nz if d3 else 1
    ksq = kgrid_ksq(params)
    shape_k = (2, nz, nx, ny//2 + 1)
    spec = np.where(ksq > 0, np.where(ksq > 0, ksq, 1.0)**(slope/2.0), 0.0)
    ck = (rng.standard_normal(shape_k) + 1j*rng.standard_normal(shape_k))*spec
    real = np.fft.irfft2(ck, s=(nx, ny), axes=(-2, -1))          # (2, nz, nx, ny), z real
    if d3:
        # keep the lowest kz modes only, so the state survives the 2/3 kz cut unchanged
        zk = np.fft.fft(real, axis=1)
        keep = np.abs(np.fft.fftfreq(nz)*nz) <= kz_modes
        real = np.fft.ifft(zk*keep[None, :, None, None], axis=1).real
    real = amp*real/np.sqrt(np.mean(real**2, axis=(1, 2, 3), keepdims=True))
    if d3:
        return lambda x, y, z: real
    return lambda x, y: real


# ------------------------------------------------- rungs F2/F3: generalized Elsasser forms

def elsasser_modal(fields, params, rho_i, z_over_tau):
    """Per-mode (E_plus, E_minus, H), ky-doubled and on the shared normalization, each
    (nz, nkx, nky), from the docs' definitions with kappa from flr_ref:
        E^+-_k = (1/4) ksq |kappa phi +- psi|^2,     H_k = G Re(phi^* psi)
    (H in its G form, so nothing here divides by v_ph: a gate comparing against the module's
    (E^+ - E^-)/v_ph form is a real cross-check)."""
    ksq = kgrid_ksq(params)
    if rho_i == 0:
        kappa, G = np.ones_like(ksq), ksq
    else:
        r = flr_ref(ksq, rho_i, z_over_tau)
        kappa, G = r["kappa"], r["G"]
    phi, psi = np.asarray(fields[0]), np.asarray(fields[1])
    yf, n = yfac(params)[None, :], norm(params)
    Ep = 0.25*yf*ksq*np.abs(kappa*phi + psi)**2/n
    Em = 0.25*yf*ksq*np.abs(kappa*phi - psi)**2/n
    Hk = yf*G*np.real(np.conj(phi)*psi)/n
    return Ep, Em, Hk


def elsasser_rates(fields, dfields, params, rho_i, z_over_tau):
    """Per-mode rates (T_plus, T_minus, T_H) of E^+, E^-, H when the fields change at
    dfields: T^+- = (1/2) ksq Re(Theta^+-* dTheta^+-) with Theta^+- = kappa phi +- psi, and
    T_H = G Re(phi^* dpsi + psi^* dphi) (the G form). Normalized like elsasser_modal."""
    ksq = kgrid_ksq(params)
    if rho_i == 0:
        kappa, G = np.ones_like(ksq), ksq
    else:
        r = flr_ref(ksq, rho_i, z_over_tau)
        kappa, G = r["kappa"], r["G"]
    phi, psi = np.asarray(fields[0]), np.asarray(fields[1])
    dphi, dpsi = np.asarray(dfields[0]), np.asarray(dfields[1])
    yf, n = yfac(params)[None, :], norm(params)
    Tp = 0.5*yf*ksq*np.real(np.conj(kappa*phi + psi)*(kappa*dphi + dpsi))/n
    Tm = 0.5*yf*ksq*np.real(np.conj(kappa*phi - psi)*(kappa*dphi - dpsi))/n
    TH = yf*G*np.real(np.conj(phi)*dpsi + np.conj(psi)*dphi)/n
    return Tp, Tm, TH


def shell_vbar(params, rho_i, z_over_tau, shell=None):
    """The forcing shell's mean phase speed, 1/vbar = sum w/v_ph / sum w over the modes
    with nmin <= k/kunit < nmax, w = yfac ksq, rebuilt here from flr_ref. `shell`: a boolean
    (nkx, nky) mask to average over instead (the solver's kgrid.fmask -- at fp32 in a
    non-2 pi box it can differ from the float64 shell built here)."""
    ksq = kgrid_ksq(params)
    if shell is None:
        kunit = min(2*np.pi/params.Lx, 2*np.pi/params.Ly)
        kk = np.sqrt(ksq)/kunit
        shell = (kk >= params.fshell[0]) & (kk < params.fshell[1])
    shell = np.asarray(shell, dtype=bool)
    vph = np.ones_like(ksq) if rho_i == 0 else flr_ref(ksq, rho_i, z_over_tau)["vph"]
    w = (yfac(params)[None, :]*ksq)[shell]
    return float(np.sum(w)/np.sum(w/vph[shell])), shell
