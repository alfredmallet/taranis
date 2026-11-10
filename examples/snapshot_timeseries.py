"""Energy time series and spectra from a run's snapshot directory, on the cluster.

Usage (inside a Slurm job -- see slurms/postprocess_savio.sh):

    TARANIS_PRECISION=64 mpirun -n 1 python examples/snapshot_timeseries.py SNAP_PATH --out OUT_DIR

Reads every saved snapshot in SNAP_PATH (any layout: the per-rank tree an mpi4jax size>1
run writes, or the flat directory of a single-process / "jax"-backend run -- `load_snapshot`
unions the z-slabs onto this one process, docs/checkpointing.md "Resharding"), computes
per snapshot

    E_kin, E_mag                         (diagnostics.energy, volume averages)
    E_u(kperp), E_b(kperp)               (diagnostics.perpspec, z-averaged)
    E_u(kz),    E_b(kz)                  (diagnostics.parspec, 3D only)

and writes to OUT_DIR

    energy.txt      # t E_kin E_mag E_tot        one row per snapshot
    spectra.npz     # t, kperp, Eu_perp, Eb_perp [, kz, Eu_par, Eb_par], step
    energy.png      # E(t)
    perpspec.png    # E(kperp) at the first / middle / last snapshot, k^-5/3 guide
    parspec.png     # E(kz), 3D only

Only the small outputs need to leave the cluster -- that is the point.  Snapshots are
read one at a time, so memory is one full state (2 fields x nz x nx x (ny/2+1) complex)
plus the transforms the diagnostics need.

Requirements and traps:
- TARANIS_PRECISION must match the run's (params.json records it as "_precision"; the
  script checks and refuses on a mismatch, because a wrong precision does not fail
  cleanly at load time).
- Run single-process (`mpirun -n 1` on Savio, where mpi4py is installed; a bare `python`
  on a laptop).  Never under a multi-rank launcher: this script reads the whole z range.
- RMHD runs only (energy/perpspec/parspec are the RMHD diagnostics).  For GDI/CMHD swap in
  diagnostics.gdi / diagnostics.cmhd.
- The run must have called `params.save(snap_path)` so params.json exists; otherwise
  reconstruct the `Parameters` by hand in `load_params` below.
"""
import argparse
import json
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("snap_path", help="run's snapshot directory (holds params.json)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <snap_path>/analysis)")
    ap.add_argument("--steps", default="all",
                    help="'all', 'last', or a comma list of saved step indices")
    ap.add_argument("--bin-factor", type=float, default=2.0,
                    help="log-bin ratio for the spectra (diagnostics default 2.0)")
    ap.add_argument("--no-parspec", action="store_true",
                    help="skip the parallel spectrum (3D only; needs the full z range)")
    return ap.parse_args()


def check_precision(snap_path):
    """Refuse before `import taranis` if TARANIS_PRECISION disagrees with the run's record."""
    rec_path = os.path.join(snap_path, "params.json")
    if not os.path.exists(rec_path):
        return None
    with open(rec_path) as f:
        rec = json.load(f)
    saved = rec.get("_precision")
    want = os.environ.get("TARANIS_PRECISION", "32")
    if saved is not None and str(saved) != str(want):
        sys.exit(f"params.json records TARANIS_PRECISION={saved} but the environment has "
                 f"{want!r}: re-run with TARANIS_PRECISION={saved} (it is read at import).")
    return rec


def main():
    args = parse_args()
    snap_path = os.path.abspath(args.snap_path)
    rec = check_precision(snap_path)

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # no display on a compute node
    import matplotlib.pyplot as plt
    import taranis as jr
    import taranis.diagnostics as diag
    from taranis.snapshot_io import get_saved_steps, load_snapshot

    params = load_params(snap_path, rec, jr)
    if params.eqtype != "RMHD":
        sys.exit(f"eqtype={params.eqtype}: this script uses the RMHD diagnostics.")
    if params.size != 1:
        sys.exit(f"params.size={params.size}: run this script on ONE process (mpirun -n 1).")
    kgrid = jr.setup_kgrids(params)
    is3d = params.spatial_dimensions == 3
    do_par = is3d and not args.no_parspec

    out = args.out or os.path.join(snap_path, "analysis")
    os.makedirs(out, exist_ok=True)

    steps = select_steps(get_saved_steps(snap_path), args.steps)
    print(f"# {snap_path}: {len(steps)} snapshot(s) {steps[0]}..{steps[-1]}, "
          f"{params.nx}x{params.ny}" + (f"x{params.nz}" if is3d else "") +
          f", precision {os.environ.get('TARANIS_PRECISION', '32')}", flush=True)

    ts, ekin, emag = [], [], []
    eu_perp, eb_perp, eu_par, eb_par = [], [], [], []
    kperp = kz = None
    for i, isnap in enumerate(steps):
        state = load_snapshot(isnap, snap_path, params)
        ek, em = diag.energy(state, kgrid, params)
        bins, su, sb = diag.perpspec(state, kgrid, params, bin_factor=args.bin_factor)
        kperp = np.asarray(bins)
        ts.append(float(state.t)); ekin.append(float(ek)); emag.append(float(em))
        eu_perp.append(np.asarray(su)); eb_perp.append(np.asarray(sb))
        if do_par:
            kzb, pu, pb = diag.parspec(state, kgrid, params, bin_factor=args.bin_factor)
            kz = np.asarray(kzb)
            eu_par.append(np.asarray(pu)); eb_par.append(np.asarray(pb))
        print(f"  [{i+1}/{len(steps)}] step {isnap}: t={ts[-1]:.4f} "
              f"E_kin={ekin[-1]:.6e} E_mag={emag[-1]:.6e}", flush=True)
        del state  # one state in memory at a time

    ts, ekin, emag = map(np.asarray, (ts, ekin, emag))
    np.savetxt(os.path.join(out, "energy.txt"),
               np.column_stack([ts, ekin, emag, ekin + emag]),
               header="t E_kin E_mag E_tot", fmt="%.17g")
    arrays = dict(t=ts, step=np.asarray(steps), kperp=kperp,
                  Eu_perp=np.asarray(eu_perp), Eb_perp=np.asarray(eb_perp))
    if do_par:
        arrays.update(kz=kz, Eu_par=np.asarray(eu_par), Eb_par=np.asarray(eb_par))
    np.savez(os.path.join(out, "spectra.npz"), **arrays)

    # --- plots -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, ekin, label=r"$E_{kin}$")
    ax.plot(ts, emag, label=r"$E_{mag}$")
    ax.plot(ts, ekin + emag, "k--", label=r"$E_{tot}$")
    ax.set_xlabel("t"); ax.set_ylabel("E (volume average)"); ax.legend()
    ax.set_title(os.path.basename(snap_path))
    fig.tight_layout(); fig.savefig(os.path.join(out, "energy.png"), dpi=150); plt.close(fig)

    plot_spectra(kperp, np.asarray(eu_perp), np.asarray(eb_perp), ts, r"$k_\perp$",
                 os.path.join(out, "perpspec.png"), plt, np)
    if do_par:
        plot_spectra(kz, np.asarray(eu_par), np.asarray(eb_par), ts, r"$k_z$",
                     os.path.join(out, "parspec.png"), plt, np, guide=False)
    print(f"# wrote energy.txt, spectra.npz and PNGs to {out}", flush=True)


def load_params(snap_path, rec, jr):
    if rec is not None:
        # re-resolves the backend for THIS process (serial or size-1 mpi4jax), ignoring the
        # comm_backend the cluster run recorded -- that is what makes the resharded read work.
        return jr.Parameters.from_snapshot(snap_path)
    sys.exit(f"no params.json in {snap_path}: the run did not call params.save(snap_path). "
             f"Reconstruct its Parameters by hand here (see examples/postprocessing.ipynb).")


def select_steps(saved, spec):
    if not saved:
        sys.exit("no snapshots found")
    saved = sorted(saved)
    if spec == "all":
        return saved
    if spec == "last":
        return saved[-1:]
    want = [int(s) for s in spec.split(",")]
    missing = sorted(set(want) - set(saved))
    if missing:
        sys.exit(f"steps {missing} not saved; available: {saved}")
    return want


def plot_spectra(k, eu, eb, ts, xlabel, path, plt, np, guide=True):
    idx = sorted({0, len(ts) // 2, len(ts) - 1})
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j, i in enumerate(idx):
        c = f"C{j}"
        ax.loglog(k, eu[i], color=c, label=rf"$E_u$, t={ts[i]:.2f}")
        ax.loglog(k, eb[i], color=c, ls="--", label=rf"$E_b$, t={ts[i]:.2f}")
    if guide:
        nz = eu[idx[-1]] > 0
        if nz.any():
            k0, e0 = k[nz][0], eu[idx[-1]][nz][0]
            ax.loglog(k, e0 * (k / k0) ** (-5 / 3), "k:", label=r"$k^{-5/3}$")
    ax.set_xlabel(xlabel); ax.set_ylabel("E(k)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
