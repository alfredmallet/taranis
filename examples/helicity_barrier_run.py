"""The helicity barrier in FLR-MHD: the three runs of Meyrand, Squire, Schekochihin & Dorland
(2021, JPP 87, 535870301) Fig. 3, as a resumable driver (plans/FLRMHD_PLAN.md rung F7).

    python examples/helicity_barrier_run.py [imb|bal|rmhd|all] [output root]

Three forced runs at the SAME energy injection rate eps, selectable one at a time:

    imb    FLR-MHD, imbalanced forcing  sigma_eps = eps_H/eps = 0.88     (the barrier run)
    bal    FLR-MHD, balanced forcing    sigma_eps = 0                    (control)
    rmhd   rho_i = 0 (RMHD through the same recipe), sigma_eps = 0.88    (control)

Physics (docs/numerics.md "FLR-MHD"; Gaussian units there). In code units v_A = 1 and the box
is cubic, Lx = Ly = Lz = 2 pi, so k_min = 2 pi/L = 1 and the Alfven time is tau_A = Lz/v_A =
2 pi. Z = 1, tau = T_0i/T_0e = 0.5 (so rho_s = rho_i), dissipation -nu k_perp^6 - nu_z kz^6
(hyper = z_hyper = 3) with equal coefficients on phi and psi, z_spectral, IF-lsrk54, adaptive
dt. O-U forcing of the generalized Elsasser potentials Theta^+- on the shell k_perp <= 2
(2 pi/L) with |kz| = 2 pi/L, the split from flrmhd.elsasser_powers(eps, sigma_eps); the
initial condition is random and large-scale with E = 10 eps tau_A.

Choices that are NOT from the paper (it does not tabulate them) and are open to revision --
every one is an environment knob and is recorded in <run>/meta.json:
  * eps: E_0 = 10 eps tau_A fixes only the ratio. Default eps = 1/(10 tau_A), i.e. E_0 = 1:
    u_rms ~ b_rms ~ v_A at k_perp ~ k_z ~ k_min, a critically balanced outer scale.
  * nu, nu_z: nu = eps^(1/3) k_d^(-16/3) with k_d = 0.7 x the dealiased k_perp,max (a
    Kolmogorov-scale estimate for k^6), and nu_z kz,max^6 = 1 (v_A k_min) at the dealiased
    kz,max. The barrier saturates through nu_z, so nu_z is a physics parameter of the late
    state (plan section 3.6): scan it, do not tune it.
  * forcing_tau = 1 (the paper forces by negative damping; plan rung F3b is not built).

Profiles (TARANIS_HB_PROFILE):
    smoke   24^2 x 24, k_min rho_i = 0.2, 2 tau_A     seconds; proves the machinery only
    m1      64^2 x 64, k_min rho_i = 0.1, 30 tau_A    the paper's low-resolution setup; hours
    full    256^3,     k_min rho_i = 0.02, 30 tau_A   the paper's Fig. 3; NOT a laptop run

Environment (all optional):
    TARANIS_HB_PROFILE  smoke|m1|full            default smoke
    TARANIS_HB_RUN      imb|bal|rmhd|all         default all (argv[1] wins)
    TARANIS_HB_ROOT     output root              default examples/data/helicity-barrier
                                                 (argv[2] wins; examples/data is gitignored)
    TARANIS_HB_TEND     end time in tau_A        default per profile
    TARANIS_HB_WALL     wall-clock budget, s     default unlimited: stop cleanly (snapshot
                                                 written) once exceeded
    TARANIS_HB_EPS, TARANIS_HB_NU, TARANIS_HB_NUZ, TARANIS_HB_SIGMA, TARANIS_HB_KC,
    TARANIS_HB_NBLOCK, TARANIS_HB_CFL_EVERY, TARANIS_HB_KEEP   see profile()
    TARANIS_PRECISION   field precision          default 64 HERE (set before taranis imports)

Restart is rerunning: each run resumes from its newest snapshot, and reaching the end time
is a no-op. Changing a physics knob against an existing directory is a hard error from
params.save (differing record), never a silently mixed series.

Two phases per run, each with its own snapshot directory because `cfl_every` is part of the
recorded Parameters: `spinup/` (cfl_every = 1, until T_SPINUP) and `main/` (cfl_every > 1:
the CFL reduction once per block and the stage propagators hoisted over it -- only from a
developed state, the standing cfl_every trap). The time series spans both.

Output, under <root>/<profile>/<run>/:
    timeseries.txt   one row per block of NBLOCK steps (header names the columns): t, step,
                     dt, E, E_plus, E_minus, E_ion, E_e, E_mag, H, sigma_c, sigma_H, eps_perp,
                     eps_z, epsH_perp, epsH_z, inj_E, inj_H, eps_inst, epsH_inst.
                     inj_E, inj_H are the REALISED injection rates over the block: the sum over
                     its sub-blocks of diagnostics.flrmhd.injection's (eps + self*dt)*dt,
                     divided by the block's duration -- every step when cfl_every = 1, the
                     first step of each cfl block otherwise. eps_inst, epsH_inst are the
                     instantaneous cross terms at the block's end; a single value of
                     epsH_inst/eps_inst means nothing (docs: only the time mean is controlled).
    spectra/<phase>_<step>.npz   at snapshot cadence (plan section 11 "truth-data products"):
                     E_plus(k), E_minus(k), E_H(k); fluxes Pi, Pi_plus, Pi_minus, Pi_H at the
                     upper bin edges K; the (kz, k_perp) energy and dissipation spectra; and the
                     sub-filter transfers T_sf_plus/minus/H(k) and cumulative fluxes at the
                     filter scale kc < 1/rho_i.
    spinup/, main/   orbax snapshots (the newest TARANIS_HB_KEEP of each)
    meta.json        the resolved configuration

Single process (FLRMHD is z_spectral-only until plan rung F6), so the z-reducing diagnostics
are called directly; under comm_backend="jax" they would need comms.shard_call.

The companion notebook examples/helicity-barrier.ipynb reads whatever output exists.
"""
import json
import os
import sys
import time

os.environ.setdefault("TARANIS_PRECISION", "64")

import numpy as np

import jax
import jax.numpy as jnp

import taranis as jr
import taranis.snapshot_io as sn
from taranis.diagnostics import flrmhd as dgf
from taranis.physics import equation_registry, flrmhd
from taranis.run import _refresh_forcing_scale, block_of_steps
from taranis.timestepping import get_scheme

TWO_PI = 2.0*np.pi
L = TWO_PI                 # cubic box: k_min = 2 pi/L = 1
TAU_A = L/1.0              # Alfven time Lz/v_A, v_A = 1
Z, TAU = 1.0, 0.5          # the paper's; rho_s = rho_i
HYPER = Z_HYPER = 3
SCHEME = "lsrk54"
CFL_SAFETY = 0.5
FSHELL = (0.5, 2.1)        # every k_perp with 0 < k_perp <= 2 k_min; edges off the integers,
#                            so no mode sits on one (at fp32 an on-edge mode can drop out)
FORCING_TAU = 1.0
FORCING_SEED = 2021
IC_SEED = 11
SIGMA_EPS = 0.88
BIN_FACTOR = 1.0           # spectra/flux bins of width k_min

# examples/data is gitignored; anchored to this file so the script and the notebook agree
# whatever the working directory
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                            "helicity-barrier")

RUNS = ("imb", "bal", "rmhd")
RUN_LABELS = {"imb": "FLR-MHD, imbalanced", "bal": "FLR-MHD, balanced",
              "rmhd": "RMHD (rho_i = 0), imbalanced"}

_PROFILES = {
    #        n     k_min*rho_i  t_end/tau_A  t_spinup/tau_A  t_snap/tau_A  nblock cfl_every keep
    "smoke": (24,  0.2,         2.0,         0.5,            0.25,         10,    2,        4),
    "m1":    (64,  0.1,         30.0,        1.0,            0.5,          20,    4,        8),
    "full":  (256, 0.02,        30.0,        1.0,            0.5,          40,    4,        4),
}

TS_COLUMNS = ("t", "step", "dt", "E", "E_plus", "E_minus", "E_ion", "E_e", "E_mag", "H",
              "sigma_c", "sigma_H", "eps_perp", "eps_z", "epsH_perp", "epsH_z", "inj_E",
              "inj_H", "eps_inst", "epsH_inst")


def _env(name, default, cast=float):
    v = os.environ.get(name, "")
    return cast(v) if v != "" else default


def profile(name=None):
    """The resolved configuration of a profile (environment overrides applied), a plain
    dict: what meta.json records."""
    name = name or os.environ.get("TARANIS_HB_PROFILE", "smoke")
    if name not in _PROFILES:
        raise ValueError(f"TARANIS_HB_PROFILE must be one of {list(_PROFILES)}, got {name!r}")
    n, rho_i, t_end, t_spin, t_snap, nblock, cfl_every, keep = _PROFILES[name]
    eps = _env("TARANIS_HB_EPS", 1.0/(10.0*TAU_A))
    kmax = n/3.0                                    # dealiased k_perp,max = kz,max (k_min = 1)
    nu = _env("TARANIS_HB_NU", eps**(1.0/3.0)*(0.7*kmax)**(-16.0/3.0))
    nu_z = _env("TARANIS_HB_NUZ", 1.0/kmax**6)
    cfl_every = _env("TARANIS_HB_CFL_EVERY", cfl_every, int)
    nblock = _env("TARANIS_HB_NBLOCK", nblock, int)
    if nblock % cfl_every:
        raise ValueError(f"TARANIS_HB_NBLOCK={nblock} must be a multiple of cfl_every="
                         f"{cfl_every}")
    # filter scale of the sub-filter transfers: the bin edge nearest 0.5/rho_i, kept inside
    # [3, 1/rho_i) -- above the forcing shell (k_perp <= 2), below the ion scale
    dk = BIN_FACTOR
    kc = max(3.0, dk*round(0.5/rho_i/dk))
    if kc >= 1.0/rho_i:
        kc = dk*np.floor((1.0/rho_i)/dk - 1e-9)
    kc = _env("TARANIS_HB_KC", float(kc))
    return dict(profile=name, n=n, rho_i=rho_i, eps=eps, sigma_eps=_env("TARANIS_HB_SIGMA",
                                                                        SIGMA_EPS),
                nu=nu, nu_z=nu_z, E0=10.0*eps*TAU_A,
                t_end=_env("TARANIS_HB_TEND", t_end)*TAU_A, t_spinup=t_spin*TAU_A,
                t_snap=t_snap*TAU_A, nblock=nblock, cfl_every=cfl_every,
                keep=_env("TARANIS_HB_KEEP", keep, int), kc=kc, tau_A=TAU_A,
                precision=os.environ["TARANIS_PRECISION"])


def run_settings(run, cfg):
    """(rho_i, sigma_eps) of one of the three runs."""
    if run not in RUNS:
        raise ValueError(f"run must be one of {RUNS}, got {run!r}")
    return ((0.0 if run == "rmhd" else cfg["rho_i"]),
            (0.0 if run == "bal" else cfg["sigma_eps"]))


def make_params(run, cfg, phase="main", forcing=True):
    """The Parameters of `run` in phase "spinup" (cfl_every = 1) or "main"."""
    rho_i, sigma = run_settings(run, cfg)
    eqpars = dict(rho_i=rho_i, tau=TAU, Z=Z, diss=(cfg["nu"], cfg["nu"]), hyper=HYPER,
                  z_diss_k=cfg["nu_z"], z_hyper=Z_HYPER)
    kw = dict(nx=cfg["n"], ny=cfg["n"], nz=cfg["n"], Lx=L, Ly=L, Lz=L, dims=3,
              z_spectral=True, eqtype="FLRMHD", eqpars=eqpars, cfl_safety=CFL_SAFETY,
              adaptive_timestep=True, fshell=FSHELL,
              cfl_every=(1 if phase == "spinup" else cfg["cfl_every"]))
    if not forcing:
        return jr.Parameters(**kw)
    # the shell-mean phase speed enters the split: eps_+- = eps (1 +- sigma_eps vbar)/2
    ep, em = flrmhd.elsasser_powers(cfg["eps"], sigma, jr.Parameters(**kw))
    return jr.Parameters(forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(ep, em),
                         forcing_tau=FORCING_TAU, forcing_seed=FORCING_SEED, **kw)


def run_dir(run, cfg, root=None):
    root = root or os.environ.get("TARANIS_HB_ROOT", DEFAULT_ROOT)
    return os.path.join(root, cfg["profile"], run)


# ------------------------------------------------------------------------ initial condition

def _ic_modes(seed=IC_SEED):
    # (kx, ky, kz, amplitude and phase for phi, for psi): every 0 < k_perp <= 2 with
    # kz in {-1, 0, 1}, one member of each +-k pair
    rng = np.random.default_rng(seed)
    modes = []
    for kx in range(-2, 3):
        for ky in range(0, 3):
            if not 0 < kx*kx + ky*ky <= 4 or (ky == 0 and kx < 0):
                continue
            for kz in (-1, 0, 1):
                modes.append((kx, ky, kz) + tuple(rng.standard_normal(2))
                             + tuple(rng.uniform(0, TWO_PI, 2)))
    return modes


def initial_state(params, kgrid, E0):
    """Random large-scale phi and psi (independent: statistically balanced), rescaled so that
    the FLR energy of THIS Parameters is E0. The same seed gives the same (phi, psi) shapes
    in all three runs."""
    modes = _ic_modes()

    def ic(x, y, z):
        phi = sum(a*jnp.cos(kx*x + ky*y + kz*z + pa) for kx, ky, kz, a, _, pa, _ in modes)
        psi = sum(b*jnp.cos(kx*x + ky*y + kz*z + pb) for kx, ky, kz, _, b, _, pb in modes)
        return jnp.stack([phi, psi])

    state = jr.initialize(ic, params)
    E = dgf.energy(state, kgrid, params)["total"]
    scale = jnp.sqrt(E0/E).astype(state.fields.real.dtype)
    return state._replace(fields=state.fields*scale)


# ----------------------------------------------------------------------------- diagnostics

def _make_block(params, kgrid, nblock):
    """jitted state -> (state, row): nblock steps, with the realised injection accumulated
    over the sub-blocks of cfl_every steps (dt is frozen inside one) and the block-end
    diagnostics. row is a dict of float64 scalars (everything in TS_COLUMNS but `step`)."""
    stepper, scheme = get_scheme(SCHEME)
    nsub = params.cfl_every
    f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)

    def sub(s, _):
        inj = dgf.injection(s, kgrid, params)
        s2 = block_of_steps(s, kgrid, params, nsub, scheme, stepper)
        span = s2.t - s.t
        dt = span/nsub                      # the (frozen) step of this sub-block
        return s2, (f64(inj["eps"] + inj["self"]*dt)*span,
                    f64(inj["eps_H"] + inj["self_H"]*dt)*span)

    def block(state):
        t0 = state.t
        s, (dE, dH) = jax.lax.scan(sub, state, None, nblock//nsub)
        span = s.t - t0
        row = _diagnose(s, kgrid, params)
        row.update(dt=span/nblock, inj_E=jnp.sum(dE)/span, inj_H=jnp.sum(dH)/span)
        return s, row

    return jax.jit(block, donate_argnums=(0,))


def _diagnose(state, kgrid, params):
    f64 = lambda v: jnp.asarray(v, dtype=jnp.float64)
    en = dgf.energy(state, kgrid, params)
    H = dgf.helicity(state, kgrid, params)
    dis = dgf.dissipation(state, kgrid, params)
    inj = dgf.injection(state, kgrid, params)
    E = en["total"]
    out = dict(t=state.t, E=E, E_plus=en["E_plus"], E_minus=en["E_minus"], E_ion=en["E_ion"],
               E_e=en["E_e"], E_mag=en["E_mag"], H=H,
               sigma_c=(en["E_plus"] - en["E_minus"])/E, sigma_H=H/E,
               eps_perp=dis["eps_perp"], eps_z=dis["eps_z"], epsH_perp=dis["epsH_perp"],
               epsH_z=dis["epsH_z"], eps_inst=inj["eps"], epsH_inst=inj["eps_H"])
    return {k: f64(v) for k, v in out.items()}


def spectra(state, kgrid, params, kc):
    """The snapshot-cadence products as a dict of numpy arrays (plan section 11)."""
    k, Ep, Em, EH = dgf.perp_spectra(state, kgrid, params, bin_factor=BIN_FACTOR)
    K, Pi, Pp, Pm, PH = dgf.fluxes(state, kgrid, params, bin_factor=BIN_FACTOR)
    k2, kz, E2, D2 = dgf.kperp_kz_spectrum(state, kgrid, params, bin_factor=BIN_FACTOR)
    _, Tp, Tm, TH = dgf.transfer(state, kgrid, params, bin_factor=BIN_FACTOR, kcut=kc)
    _, Psf, Psfp, Psfm, PsfH = dgf.fluxes(state, kgrid, params, bin_factor=BIN_FACTOR, kcut=kc)
    en = dgf.energy(state, kgrid, params)
    out = dict(t=state.t, k=k, E_plus=Ep, E_minus=Em, E_H=EH, K=K, Pi=Pi, Pi_plus=Pp,
               Pi_minus=Pm, Pi_H=PH, kz=kz, E_kz_k=E2, D_kz_k=D2, kc=kc, T_sf_plus=Tp,
               T_sf_minus=Tm, T_sf_H=TH, Pi_sf=Psf, Pi_sf_plus=Psfp, Pi_sf_minus=Psfm,
               Pi_sf_H=PsfH, E=en["total"],
               sigma_c=(en["E_plus"] - en["E_minus"])/en["total"])
    return {name: np.asarray(v, dtype=float) for name, v in out.items()}


# ------------------------------------------------------------------------------ sidecars

def _ts_path(d):
    return os.path.join(d, "timeseries.txt")


def _append_row(d, row):
    path = _ts_path(d)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("# " + " ".join(TS_COLUMNS) + "\n")
        f.write(" ".join("%d" % row[c] if c == "step" else "%.17g" % float(row[c])
                         for c in TS_COLUMNS) + "\n")


def _truncate_timeseries(d, t):
    """Restart housekeeping: drop the rows past the restart time (the run about to start
    rewrites them). Returns the step count of the last row kept (0 if none)."""
    path = _ts_path(d)
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        lines = f.read().splitlines()
    rows = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    kept = [ln for ln in rows if float(ln.split()[0]) <= t]
    if len(kept) != len(rows):
        with open(path, "w") as f:
            f.write("".join(ln + "\n" for ln in lines[:1] + kept))
    return int(kept[-1].split()[1]) if kept else 0


def load_timeseries(d):
    """dict of column arrays from <run dir>/timeseries.txt, by header name; None if absent."""
    path = _ts_path(d)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        names = f.readline().lstrip("#").split()
    data = np.atleast_2d(np.loadtxt(path))
    if data.size == 0:
        return None
    return {n: data[:, i] for i, n in enumerate(names)}


def _spec_path(d, phase, step):
    return os.path.join(d, "spectra", f"{phase}_{step:05d}.npz")


def load_spectra(d):
    """List of the spectra dicts of a run directory, sorted by time."""
    sd = os.path.join(d, "spectra")
    if not os.path.isdir(sd):
        return []
    out = []
    for name in sorted(os.listdir(sd)):
        if name.endswith(".npz"):
            with np.load(os.path.join(sd, name)) as z:
                out.append({k: z[k] for k in z.files})
    return sorted(out, key=lambda s: float(s["t"]))


def _save_point(d, phase, state, kgrid, params, mngr, cfg):
    """One snapshot (drained) and then its spectra file: a spectra file therefore never
    exists without the snapshot it was computed from."""
    step = max(sn.get_saved_steps(mngr.directory), default=-1) + 1
    sn.save_snapshot(step, state, mngr, params)
    mngr.wait_until_finished()
    os.makedirs(os.path.join(d, "spectra"), exist_ok=True)
    tmp = _spec_path(d, phase, step) + ".tmp.npz"
    np.savez(tmp, **spectra(state, kgrid, params, cfg["kc"]))
    os.replace(tmp, _spec_path(d, phase, step))
    return step


def _drop_spectra_after(d, t):
    sd = os.path.join(d, "spectra")
    if not os.path.isdir(sd):
        return
    for name in os.listdir(sd):
        path = os.path.join(sd, name)
        if name.endswith(".tmp.npz"):
            os.remove(path)
        elif name.endswith(".npz"):
            with np.load(path) as z:
                late = float(z["t"]) > t
            if late:
                os.remove(path)


# --------------------------------------------------------------------------------- driver

def _phase_dir(d, phase):
    return os.path.join(d, phase)


def _resume_point(d, run, cfg):
    """(phase, params, state or None): where this run stands on disk. The main phase starts
    from the spin-up's last snapshot once that has reached t_spinup."""
    for phase in ("main", "spinup"):
        steps = sn.get_saved_steps(_phase_dir(d, phase)) if os.path.isdir(
            _phase_dir(d, phase)) else []
        if steps:
            params = make_params(run, cfg, phase)
            state = sn.load_snapshot(steps[-1], _phase_dir(d, phase), params)
            if phase == "spinup" and float(state.t) >= cfg["t_spinup"]:
                params = make_params(run, cfg, "main")
                state = sn.load_snapshot(steps[-1], _phase_dir(d, "spinup"), params)
                return "main", params, state
            return phase, params, state
    return "spinup", make_params(run, cfg, "spinup"), None


def drive(run, cfg=None, root=None, wall=None):
    """Advance one run to cfg["t_end"], resuming from disk. Returns True when it is there,
    False if the wall-clock budget ran out first (rerun to continue)."""
    cfg = cfg or profile()
    wall = _env("TARANIS_HB_WALL", np.inf) if wall is None else wall
    t_wall0 = time.time()
    d = run_dir(run, cfg, root)
    os.makedirs(d, exist_ok=True)
    rho_i, sigma = run_settings(run, cfg)
    meta = dict(cfg, run=run, rho_i_run=rho_i, sigma_eps_run=sigma, fshell=FSHELL,
                forcing_tau=FORCING_TAU, scheme=SCHEME, columns=TS_COLUMNS)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    phase, params, state = _resume_point(d, run, cfg)
    kgrid = jr.setup_kgrids(params)
    fresh = state is None
    if fresh:
        state = initial_state(params, kgrid, cfg["E0"])
    t = float(state.t)
    print(f"[{cfg['profile']}/{run}] {RUN_LABELS[run]}: {cfg['n']}^3, k_min rho_i = {rho_i:g}, "
          f"sigma_eps = {sigma:g}, eps = {cfg['eps']:.4g}, nu = {cfg['nu']:.3g}, nu_z = "
          f"{cfg['nu_z']:.3g}; {'fresh start' if fresh else f'resuming {phase} at'} "
          f"t = {t/TAU_A:.3f} tau_A of {cfg['t_end']/TAU_A:g}", flush=True)
    nstep = _truncate_timeseries(d, t)
    _drop_spectra_after(d, t)
    if t >= cfg["t_end"]:
        print(f"[{cfg['profile']}/{run}] nothing to do", flush=True)
        return True

    while True:
        # ---- one phase: its own Parameters record, snapshot directory and jitted block
        pdir = _phase_dir(d, phase)
        params.save(pdir)
        mngr = jr.snapshot_manager_setup(params, pdir, nsnap=cfg["keep"])
        state = _refresh_forcing_scale(state, kgrid, params)
        steps = sn.get_saved_steps(pdir)
        if not steps:
            # the phase's first state: t = 0, or the spin-up's last state carried over
            step = _save_point(d, phase, state, kgrid, params, mngr, cfg)
            if fresh:
                row = {k: float(v) for k, v in _diagnose(state, kgrid, params).items()}
                row.update(step=0, dt=np.nan, inj_E=np.nan, inj_H=np.nan)
                _append_row(d, row)
        elif not os.path.exists(_spec_path(d, phase, steps[-1])):
            # killed between a snapshot and its spectra: recompute them from that snapshot
            os.makedirs(os.path.join(d, "spectra"), exist_ok=True)
            np.savez(_spec_path(d, phase, steps[-1]),
                     **spectra(state, kgrid, params, cfg["kc"]))
        fresh = False
        advance = _make_block(params, kgrid, cfg["nblock"])
        t_stop = cfg["t_spinup"] if phase == "spinup" else cfg["t_end"]
        t_stop = min(t_stop, cfg["t_end"])
        saved, out_of_time = True, False
        while t < t_stop:
            t_prev = t
            state, row = advance(state)
            row = {k: float(v) for k, v in row.items()}
            t = row["t"]
            nstep += cfg["nblock"]
            row["step"] = nstep
            _append_row(d, row)
            saved = False
            if not np.isfinite(row["E"]):
                raise FloatingPointError(f"[{run}] energy is not finite at t = {t}")
            # absolute snapshot schedule (multiples of t_snap), so it does not depend on
            # where a previous invocation stopped
            if int(t/cfg["t_snap"]) > int(t_prev/cfg["t_snap"]) or t >= t_stop:
                step = _save_point(d, phase, state, kgrid, params, mngr, cfg)
                saved = True
                print(f"[{cfg['profile']}/{run}] {phase} snapshot {step}: t = {t/TAU_A:.3f} "
                      f"tau_A, step {nstep}, dt = {row['dt']:.3e}, E/(eps tau_A) = "
                      f"{row['E']/(cfg['eps']*TAU_A):.3f}, sigma_c = {row['sigma_c']:+.3f}, "
                      f"(eps_perp + eps_z)/eps = "
                      f"{(row['eps_perp'] + row['eps_z'])/cfg['eps']:.3f}, wall "
                      f"{time.time() - t_wall0:.0f}s", flush=True)
            if time.time() - t_wall0 > wall:
                out_of_time = True
                break
        if not saved:
            _save_point(d, phase, state, kgrid, params, mngr, cfg)
        mngr.wait_until_finished()
        mngr.close()
        if out_of_time and t < cfg["t_end"]:
            print(f"[{cfg['profile']}/{run}] wall budget reached at t = {t/TAU_A:.3f} tau_A; "
                  f"rerun to continue", flush=True)
            return False
        if t >= cfg["t_end"]:
            print(f"[{cfg['profile']}/{run}] done: t = {t/TAU_A:.3f} tau_A, {nstep} steps, "
                  f"wall {time.time() - t_wall0:.0f}s", flush=True)
            return True
        # ---- spin-up finished: continue in the main phase from the state in hand
        phase = "main"
        params = make_params(run, cfg, "main")
        kgrid = jr.setup_kgrids(params)
        fresh = False


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    which = argv[0] if argv else os.environ.get("TARANIS_HB_RUN", "all")
    root = argv[1] if len(argv) > 1 else None
    runs = RUNS if which == "all" else (which,)
    cfg = profile()
    t0 = time.time()
    wall = _env("TARANIS_HB_WALL", np.inf)
    for run in runs:
        if not drive(run, cfg, root, wall=wall - (time.time() - t0)):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
