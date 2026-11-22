# Gate for the variety effects of examples/music_turbulence.py (kick types E1, stereo
# placement E2; plans-webgpu/MUSIC_EFFECTS_PLAN.md). Not a pytest module: run it as
#
#   TARANIS_PRECISION=64 python examples/music_turbulence_check.py
#
# It prints one line per check and exits non-zero if any fails. That the script with every
# switch off still produces what it produced before the effects existed is NOT checked
# here (it needs the earlier tree): it was checked once, bitwise, when they were added.
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax.numpy as jnp

import taranis as jr
import music_turbulence as mt

L = mt.LBOX
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def setup(n, diss=0.0, hyper=4):
    p = jr.Parameters(nx=n, ny=n, Lx=L, Ly=L, cfl_safety=0.4, dims=2, comm_backend="serial",
                      eqpars={"diss": (diss, diss), "hyper": hyper})
    kg = jr.setup_kgrids(p)
    kick, observe = mt.make_kick(p, kg)
    return p, kg, kick, observe


def from_rest(kick, n, bp, bm):
    zero = jnp.zeros((2, 1, n, n // 2 + 1), dtype=jnp.complex128)
    return kick(zero, mt.pad(bp), mt.pad(bm))


def blob_energy(amp, sigma):
    # volume-averaged (1/2)<|grad f|^2> of ONE gaussian f = P exp(-r^2/2 sigma^2),
    # P = amp sigma sqrt(e): pi P^2 / (2 L^2)
    return 0.5 * np.pi * (amp * sigma) ** 2 * np.e / L**2


def kick_gates():
    print("kick types (E1)")
    n, sigma, amp, nb = 128, 0.2, 0.7, 1
    p, kg, kick, observe = setup(n)
    e1 = blob_energy(amp, sigma)

    # --- which potential each type touches, and the energy of one well-separated entry
    rng = np.random.default_rng(1)
    for kind, want, touched in [("elsasser", e1, "both"), ("vortex", e1, "phi"),
                                ("island", e1, "psi"),
                                ("dipole", mt.dipole_energy_factor() * e1, "phi"),
                                ("sheet", 0.5 * (4.0 + 0.25) * e1, "both"),
                                ("sheet-island", 0.5 * (4.0 + 0.25) * e1, "psi")]:
        bp, bm = mt.kick_blobs(rng, kind, sigma, nb, amp)
        f = np.asarray(from_rest(kick, n, bp, bm))
        e = float(observe(jnp.asarray(f))[2])
        e_want = want
        if kind == "sheet-island":
            z6 = np.zeros((0, 6))
            for ch in ((bp, z6), (z6, bm)):
                g = np.asarray(from_rest(kick, n, *ch))
                check("sheet-island: each channel alone kicks both potentials (the list is 6-column)",
                      np.any(g[0]) and np.any(g[1]))
        if kind in ("elsasser", "sheet"):
            # one entry per channel, each carrying half of `want` (phi = psi = z/2). They
            # add only if they do not overlap, so measure each channel alone and sum.
            zero = np.zeros((0, bp.shape[1]))
            e = sum(float(observe(from_rest(kick, n, a, b))[2]) for a, b in ((bp, zero), (zero, bm)))
        check(f"{kind}: energy of one entry against the closed form", abs(e / e_want - 1.0) < (1e-5 if "sheet" in kind else 1e-10),   # a sheet 0.8 long meets its periodic images at ~1e-7
              f"{e:.12e} vs {e_want:.12e}")
        if touched == "phi":
            check(f"{kind}: psi exactly zero", not np.any(f[1]), f"max {np.abs(f[1]).max():.1e}")
        if touched == "psi":
            check(f"{kind}: phi exactly zero", not np.any(f[0]), f"max {np.abs(f[0]).max():.1e}")
        if touched == "both":
            check(f"{kind}: both potentials kicked", np.any(f[0]) and np.any(f[1]))

    # --- a full multi-blob kick of each one-potential type, and a vortex kick evolved
    for kind, idx in (("vortex", 1), ("dipole", 1), ("island", 0)):
        bp, bm = mt.kick_blobs(rng, kind, 0.3, 9, amp)
        f = np.asarray(from_rest(kick, n, bp, bm))
        check(f"{kind}: 9-entry kick leaves {('phi', 'psi')[idx]} exactly zero", not np.any(f[idx]))
    step = mt.Stepper(n, 1.0 / 120.0, 48.0 / (n / 3.0) ** 8, 4)
    kick_s, obs_s = mt.make_kick(step.p, step.kg)
    state = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
    bp, bm = mt.kick_blobs(rng, "vortex", 0.3, 9, 1.0)
    state = state._replace(fields=kick_s(state.fields, mt.pad(bp), mt.pad(bm)))
    state, k = step.advance(state, 1.0, float(obs_s(state.fields)[1]))
    f = np.asarray(state.fields)
    check("vortex: psi still exactly zero after stepping to t = 1", not np.any(f[1]) and np.any(f[0]),
          f"{k} steps")

    # --- the k-space list reproduces the real-space one for isotropic entries
    b4 = mt.blobs_for(rng, 0.3, 7, amp)
    c4 = mt.blobs_for(rng, 0.25, 5, amp)
    iso = lambda b: np.column_stack([b, np.ones(len(b)), rng.uniform(0, np.pi, len(b))])
    fa, fb = np.asarray(from_rest(kick, n, b4, c4)), np.asarray(from_rest(kick, n, iso(b4), iso(c4)))
    err = np.abs(fa - fb).max() / np.abs(fa).max()
    check("sheet list at ratio 1 == the blob list (any theta)", err < 1e-12, f"rel {err:.1e}")

    # --- a sheet: peak gradient ACROSS it is amp, and it is long in the direction theta
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    for theta in (0.0, 0.7, 2.1):
        s6 = np.array([[3.0, 3.3, 0.15, amp, 4.0, theta]])
        f = from_rest(kick, n, s6, np.zeros((0, 6)))
        z = 2.0 * np.asarray(mt.ifft(f, p))[0, 0]                  # z+ = phi + psi = 2 phi here
        gx = np.asarray(mt.ifft(1j * kg.kx * f, p))[0, 0] * 2.0
        gy = np.asarray(mt.ifft(1j * kg.ky * f, p))[0, 0] * 2.0
        gacross = -np.sin(theta) * gx + np.cos(theta) * gy
        # z has its k = 0 removed, which does not touch the gradient
        w = (z - z.min()) ** 2
        dx, dy = X - 3.0, Y - 3.3
        along, across = np.cos(theta) * dx + np.sin(theta) * dy, -np.sin(theta) * dx + np.cos(theta) * dy
        la, lc = np.sqrt((w * along**2).sum() / w.sum()), np.sqrt((w * across**2).sum() / w.sum())
        check(f"sheet theta = {theta}: peak gradient across = amp, length/width = 4",
              abs(np.abs(gacross).max() / amp - 1.0) < 2e-2 and abs(la / lc - 4.0) < 0.15,
              f"peak {np.abs(gacross).max() / amp:.4f} amp, ratio {la / lc:.3f}")

    # --- counts and the type table
    bp, bm = mt.kick_blobs(rng, "sheet", 0.1, 15, 1.0)
    check("sheet: nblob per channel, independent sets", len(bp) == len(bm) == 15 and not np.array_equal(bp, bm))
    b2, _ = mt.kick_blobs(rng, "sheet", 0.1, 15, 1.0, sheet_energy=2.0)
    check("sheet_energy 2: amplitude times sqrt(2)", np.allclose(np.abs(b2[:, 3]), np.sqrt(2.0), rtol=1e-15))
    bp, bm = mt.kick_blobs(rng, "sheet-island", 0.1, 15, 1.0)
    f = np.asarray(from_rest(kick, n, bp, bm))
    check("sheet-island: 15-sheet kick leaves phi exactly zero", not np.any(f[0]) and np.any(f[1]))
    bp, bm = mt.kick_blobs(rng, "dipole", 0.45, 3, 1.0)
    check("dipole: nblob pairs, z- list equal to z+", len(bp) == 6 and np.array_equal(bp, bm))
    check("kick_types_for: None, variety at 4/5/6 bands, a single name, a list",
          mt.kick_types_for(None, 4) == ("elsasser",) * 4
          and mt.kick_types_for("variety", 4) == ("dipole", "vortex", "island", "sheet")
          and mt.kick_types_for("variety", 5) == ("dipole", "vortex", "island", "vortex", "sheet")
          and mt.kick_types_for("variety", 6) == ("dipole", "vortex", "island", "vortex", "island", "sheet")
          and mt.kick_types_for("island", 3) == ("island",) * 3
          and mt.kick_types_for("sheet,vortex", 2) == ("sheet", "vortex"))
    bad = 0
    for spec, nbands in (("blob", 4), ("vortex,island", 4)):
        try:
            mt.kick_types_for(spec, nbands)
        except ValueError:
            bad += 1
    check("kick_types_for: an unknown name and a wrong count are errors", bad == 2)


def dipole_gate():
    print("dipole propagation (E1)")
    n, sigma, amp = 128, 0.3, 1.0
    step = mt.Stepper(n, 1.0 / 120.0, 48.0 * amp / (n / 3.0) ** 8, 4)
    kick, observe = mt.make_kick(step.p, step.kg)
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    for th, sign in ((0.0, 1.0), (0.9, -1.0)):
        e = np.array([np.cos(th), np.sin(th)])
        c0 = np.array([np.pi, np.pi])
        off = 0.5 * mt.DIPOLE_SEP * sigma * e
        b = np.array([[*(c0 + off), sigma, sign * amp], [*(c0 - off), sigma, -sign * amp]])
        state = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
        state = state._replace(fields=kick(state.fields, mt.pad(b), mt.pad(b)))
        e0 = float(observe(state.fields)[2])

        def centre(fields):
            # midpoint of the centroids of the positive and the negative lobe of phi
            phi = np.asarray(mt.ifft(fields, step.p))[0, 0]
            out = []
            for lobe in (np.maximum(phi, 0.0) ** 2, np.maximum(-phi, 0.0) ** 2):
                out.append([(lobe * X).sum() / lobe.sum(), (lobe * Y).sum() / lobe.sum()])
            return 0.5 * (np.array(out[0]) + np.array(out[1]))

        def centre_rate(fields):
            # d(centre)/dt at this instant from 2D Euler written out in numpy, independent
            # of the solver: d_t lap(phi) = -u.grad(lap phi), u = zhat x grad(phi)
            phi = np.asarray(mt.ifft(fields, step.p))[0, 0]
            k = np.fft.fftfreq(n, 1.0 / n)
            KX, KY = np.meshgrid(k, k, indexing="ij")
            K2 = KX**2 + KY**2
            pk = np.fft.fft2(phi)
            d = lambda q, K: np.fft.ifft2(1j * K * q).real
            wk = -K2 * pk
            dw = -(-d(pk, KY) * d(wk, KX) + d(pk, KX) * d(wk, KY))
            dphi = np.fft.ifft2(np.where(K2 > 0, -np.fft.fft2(dw) / np.maximum(K2, 1.0), 0.0)).real
            out = []
            for sg in (1.0, -1.0):
                lobe, dl = np.maximum(sg * phi, 0.0) ** 2, 2.0 * np.maximum(sg * phi, 0.0) * sg * dphi
                c = np.array([(lobe * X).sum(), (lobe * Y).sum()]) / lobe.sum()
                out.append((np.array([(dl * X).sum(), (dl * Y).sum()]) - c * dl.sum()) / lobe.sum())
            return 0.5 * (out[0] + out[1])

        v0 = centre_rate(state.fields)
        ts, cs = [0.0], [centre(state.fields)]
        state, _ = step.advance(state, 0.05 * sigma / amp, amp)
        v_first = (centre(state.fields) - cs[0]) / float(state.t)
        # three turnover times sigma/amp, in six legs
        for i in range(1, 7):
            state, _ = step.advance(state, 0.5 * i * sigma / amp, amp)
            ts.append(float(state.t)); cs.append(centre(state.fields))
        ts, cs = np.array(ts), np.array(cs)
        v = (cs[-1] - cs[0]) / (ts[-1] - ts[0])
        # each blob is carried by the other's flow u = zhat x grad(phi): at a blob's CENTRE
        # that is amp (d/sigma) exp(1/2 - d^2/2 sigma^2) = 0.803 amp at d = 1.5 sigma,
        # along sign * (zhat x e), perpendicular to the line joining the two. That is the
        # direction and an upper estimate of the speed: the partner's flow falls off
        # across the core and the lobes deform, and the pair travels at ~0.43 of it.
        perp = sign * np.array([-e[1], e[0]])
        d = mt.DIPOLE_SEP
        v_est = amp * d * np.exp(0.5 - 0.5 * d * d)
        vpar, vside = float(v @ perp), float(v @ e)
        legs = np.linalg.norm(np.diff(cs, axis=0), axis=1) / np.diff(ts)
        e1 = float(observe(state.fields)[2])
        check(f"dipole at angle {th}, sign {sign:+.0f}: travels perpendicular to its two blobs",
              abs(vside) < 0.02 * abs(vpar) and vpar > 0, f"v_perp {vpar:.3f}, v_along-the-pair {vside:+.4f}")
        check("   ... starting at the rate 2D Euler gives for this phi (numpy, not the solver)",
              np.linalg.norm(v_first - v0) < 0.03 * np.linalg.norm(v0) and v0 @ perp > 0,
              f"measured {v_first @ perp:.4f}, predicted {v0 @ perp:.4f} (side {v0 @ e:+.1e})")
        check("   ... and keeping 0.3-0.6 of the centre estimate 0.803 amp for three turnover times",
              0.3 * v_est < vpar < 0.6 * v_est and legs.std() < 0.15 * legs.mean(),
              f"{vpar:.3f} = {vpar / v_est:.2f} x {v_est:.3f}; per-leg {legs.min():.3f}..{legs.max():.3f}")
        check("   ... keeping its energy over three turnover times", e1 > 0.9 * e0, f"E/E0 = {e1 / e0:.3f}")


def stereo_gates():
    print("stereo placement (E2)")
    sr, lr = mt.click_track(duration=8.0, stereo=True)
    _, mono = mt.click_track(duration=8.0)
    check("stereo click track: its channel mean is the mono click track",
          np.abs(lr.mean(axis=1) - mono).max() < 1e-12)
    check("is_mono: equal channels yes, the stereo click no, a one-channel array yes",
          mt.is_mono(np.column_stack([mono, mono])) and not mt.is_mono(lr) and mt.is_mono(None))
    sigmas, _ = mt.band_plan(256)
    nb = len(sigmas)
    t_on, b_on, s_on, edges = mt.detect_onsets(sr, lr.mean(axis=1), nbands=nb, threshold=0.3)
    pans = mt.onset_pans(sr, lr, t_on, b_on, edges)
    beat = 0.5
    ph = (t_on / beat) % 1.0                         # 0: kick/snare on the beat, 0.5: hat
    on_beat = np.minimum(ph, 1.0 - ph) < 0.1
    ibeat = np.rint(t_on / beat - 0.5).astype(int)   # for a hat: which off-beat
    full = lambda p: 2.0 * p / (1.0 + p * p)
    hats = (b_on == nb - 1) & ~on_beat
    want = full(mt.CLICK_PANS["hat"]) * np.where(ibeat[hats] % 2 == 0, 1.0, -1.0)
    check("hats (top band): pan = +-2p/(1+p^2), alternating right, left",
          hats.sum() >= 12 and np.abs(pans[hats] - want).max() < 0.02,
          f"{hats.sum()} onsets, max error {np.abs(pans[hats] - want).max():.3f}")
    snare = (b_on == nb - 2) & on_beat
    check("snare (band below): pan = 2p/(1+p^2) to the left",
          snare.sum() >= 4 and np.abs(pans[snare] - full(mt.CLICK_PANS["snare"])).max() < 0.05,
          f"{snare.sum()} onsets, pans {pans[snare].min():+.3f}..{pans[snare].max():+.3f} "
          f"vs {full(mt.CLICK_PANS['snare']):+.3f}")
    kick = b_on == 0
    check("kick drum (bass band): centred", kick.sum() >= 8 and np.abs(pans[kick]).max() < 0.02,
          f"{kick.sum()} onsets, max |pan| {np.abs(pans[kick]).max():.3f}")
    sw = mt.onset_pans(sr, lr[:, ::-1], t_on, b_on, edges)
    check("swapping the channels flips every pan", np.abs(sw + pans).max() < 1e-12)
    silent = mt.onset_pans(sr, 0.0 * lr, t_on[:3], b_on[:3], edges)
    check("a silent frame gives pan 0, not NaN", np.array_equal(silent, np.zeros(3)))

    # --- placement: where the blobs go
    rng = np.random.default_rng(3)
    b = mt.blobs_for(rng, 0.1, 20000, 1.0, xc=0.2 * L)
    dx = (b[:, 0] - 0.2 * L + 0.5 * L) % L - 0.5 * L
    ks = np.abs(np.sort(b[:, 1]) / L - (np.arange(20000) + 0.5) / 20000).max()
    check("blobs_for(xc): x gaussian about xc with width 0.12 L, y uniform, inside the box",
          abs(dx.mean()) < 0.005 * L and abs(dx.std() / (mt.STEREO_JITTER * L) - 1.0) < 0.03
          and ks < 0.02 and b[:, :2].min() >= 0.0 and b[:, :2].max() < L,
          f"mean {dx.mean() / L:+.4f} L, std {dx.std() / L:.4f} L, y KS {ks:.4f}")
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    a1, a2 = mt.blobs_for(r1, 0.1, 9, 1.0), mt.blobs_for(r2, 0.1, 9, 1.0, xc=None)
    check("blobs_for without xc draws the same stream as before", np.array_equal(a1, a2))

    # --- the driver: bass exempt, the others land on their side; mono falls back
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        seen = {}
        orig = mt.kick_blobs

        def spy(rng, kind, sigma, nblob, amp, xc=None, *rest):
            seen.setdefault(round(float(sigma), 6), []).append(xc)
            return orig(rng, kind, sigma, nblob, amp, xc, *rest)
        mt.kick_blobs = spy
        try:
            kw = dict(n=64, nbands=4, sigma_min=3.7 * L / 256, quiet=True, duration=4.0)
            r = mt.run(sr, lr.mean(axis=1), os.path.join(tmp, "a.mp4"), stereo=lr, **kw)
            by_band = [seen.get(round(float(s), 6), []) for s in r["sigmas"]]
            hat_x = np.array(by_band[3])
            # the top band hears the hats (+-0.976) and the snare's noise (-0.882)
            spots = lambda g: 0.5 + 0.4 * g * np.array([full(0.8), -full(0.8), full(-0.6)])
            near = lambda xs, g: np.abs(xs[:, None] / L - spots(g)[None, :]).min(axis=1).max()
            check("run(stereo=): the bass band is not placed, the top band lands at 0.5 + 0.4 pan of the box",
                  all(x is None for x in by_band[0]) and len(by_band[0]) > 0 and len(hat_x) >= 6
                  and near(hat_x, 1.0) < 0.01
                  and (hat_x > 0.8 * L).any() and (hat_x < 0.2 * L).any(),
                  f"hat centres {np.round(np.unique(np.round(hat_x / L, 2)), 2)} L")
            seen.clear()
            r = mt.run(sr, lr.mean(axis=1), os.path.join(tmp, "b.mp4"), stereo=lr, stereo_gain=0.25,
                       stereo_bands=[0, 3], **kw)
            by_band = [seen.get(round(float(s), 6), []) for s in r["sigmas"]]
            hat_x = np.array(by_band[3])
            check("stereo_bands and stereo_gain: bands 0 and 3 placed, 1 and 2 not, gain scales the offset",
                  all(x is not None for x in by_band[0]) and all(x is None for x in by_band[1] + by_band[2])
                  and near(hat_x, 0.25) < 0.003)
            seen.clear()
            m2 = np.column_stack([mono, mono])
            r_m = mt.run(sr, mono, os.path.join(tmp, "c.mp4"), stereo=m2, **kw)
            r_0 = mt.run(sr, mono, os.path.join(tmp, "d.mp4"), **kw)
            check("identical channels: uniform placement, the same fields as without --stereo",
                  all(x is None for v in seen.values() for x in v) and np.array_equal(r_m["fields"], r_0["fields"]))
        finally:
            mt.kick_blobs = orig
        r_v = mt.run(sr, lr.mean(axis=1), os.path.join(tmp, "e.mp4"), stereo=lr, stereo_gain=2.0,
                     kick_types="variety", **kw)
        check("variety + stereo together run to the end, finite", np.all(np.isfinite(r_v["energy"]))
              and r_v["energy"].max() > 0, f"types {r_v['kick_types']}, E_end {r_v['energy'][-1]:.4f}")


if __name__ == "__main__":
    if jr._precision.ftype != jnp.float64:
        sys.exit("run with TARANIS_PRECISION=64")
    kick_gates()
    dipole_gate()
    stereo_gates()
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED: ' + '; '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
