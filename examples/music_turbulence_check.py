# Gate for the variety effects of examples/music_turbulence.py (kick types E1, stereo
# placement E2, the camera pulse E3, the mean field and its bass-hit walk E4, the zoom-out
# re-grid E5, the section trigger; plans-webgpu/MUSIC_EFFECTS_PLAN.md) and of the stage options (the
# rectangular box S2 and the maintained current sheets S4; plans-webgpu/MUSIC_STAGE_PLAN.md).
# Not a pytest module: run it as
#
#   TARANIS_PRECISION=64 python examples/music_turbulence_check.py
#
# It prints one line per check and exits non-zero if any fails. That the script with every
# switch off still produces what it produced before the effects existed is NOT checked
# here (it needs the earlier tree): it was checked once, bitwise, when they were added --
# and again for the stage options (seven configurations against the tree before them, at
# both precisions: fields, energy and the movie file byte for byte).
import os
import sys
import tempfile

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
    bs, _ = mt.kick_blobs(rng, "sheet", 0.1, 400, 1.0)
    bd, _ = mt.kick_blobs(rng, "dipole", 0.1, 400, 1.0)
    d = (bd[:400, 0:2] - bd[400:, 0:2] + 0.5 * L) % L - 0.5 * L
    dang = np.arctan2(d[:, 1], d[:, 0])
    # uniform on [0, pi) has mean pi/2 and std pi/sqrt(12) = 0.907; on [-pi, pi), std 1.814
    check("the orientations are random: sheets uniform over [0, pi), dipole axes over the full circle",
          bs[:, 5].min() >= 0.0 and bs[:, 5].max() < np.pi and abs(bs[:, 5].mean() - 0.5 * np.pi) < 0.15
          and abs(bs[:, 5].std() - 0.907) < 0.08 and abs(dang.std() - 1.814) < 0.15
          and np.allclose(np.hypot(d[:, 0], d[:, 1]), 1.5 * 0.1),
          f"sheet theta std {bs[:, 5].std():.3f}, dipole axis std {dang.std():.3f}")
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


def pulse_gates():
    print("camera pulse (E3)")
    fps = 30
    tf = np.arange(90) / fps
    z = mt.pulse_zoom(tf, [1.0], [0.5], a=0.08, tau=0.25)
    want = np.where(tf >= 1.0, 1.0 + 0.04 * np.exp(-(tf - 1.0) / 0.25), 1.0)
    check("one onset: 1 before it, 1 + a*strength at its frame, exp(-t/tau) after",
          np.abs(z - want).max() < 1e-15 and z[30] == 1.04 and np.all(z[:30] == 1.0))
    z = mt.pulse_zoom(tf, [1.0, 1.1, 1.2], [1.0, 0.1, 1.0], a=0.08, tau=0.25)
    i = lambda t: int(round(t * fps))
    check("a weaker onset during the decay does not lower the zoom; a stronger one lifts it TO 1 + a",
          abs(z[i(1.1)] - (1.0 + 0.08 * np.exp(-0.1 / 0.25))) < 1e-15 and abs(z[i(1.2)] - 1.08) < 1e-15
          and z.max() <= 1.08 + 1e-15)
    z2 = mt.pulse_zoom(tf, [1.2, 1.0, 1.1], [1.0, 1.0, 0.1], a=0.08, tau=0.25)
    check("onset order does not matter; no onsets is 1 everywhere",
          np.array_equal(z, z2) and np.array_equal(mt.pulse_zoom(tf, [], []), np.ones(90)))

    n = 64
    rng = np.random.default_rng(0)
    img = rng.standard_normal((n, n))
    check("zoom_image at 1 is the image itself", mt.zoom_image(img, 1.0) is img)
    x = np.arange(n)
    c = 0.5 * (n - 1)
    f = lambda a0, a1: np.cos(2 * np.pi * 2 * a0 / n + 0.3) * np.sin(2 * np.pi * a1 / n)
    X0, X1 = np.meshgrid(x, x, indexing="ij")
    # bilinear error bound for this f: (k0^2 + k1^2)/8 = 6.0e-3. 0.5 looks past the edge of
    # the box on every side, so it passes only if the sampling wraps.
    for s_ in (1.08, 1.5, 0.5):
        zi = mt.zoom_image(f(X0, X1), s_, blur=False)     # the sampling rule alone
        ex = f(c + (X0 - c) / s_, c + (X1 - c) / s_)
        err = np.abs(zi - ex).max()
        check(f"zoom {s_}: pixel i shows the periodic field at c + (i - c)/s (bilinear error only)",
              err < 6.5e-3, f"max error {err:.1e}")
    sm = f(X0, X1)
    a_, b_ = mt.zoom_image(sm, 1.2), mt.zoom_image(sm[::-1, ::-1], 1.2)[::-1, ::-1]
    check("the zoom is about the centre: it commutes with flipping the image", np.abs(a_ - b_).max() < 1e-13)

    import tempfile
    sr, mono = mt.click_track(duration=4.0)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(n=64, nbands=4, sigma_min=3.7 * L / 256, quiet=True)
        r0 = mt.run(sr, mono, os.path.join(tmp, "a.mp4"), **kw)
        r1 = mt.run(sr, mono, os.path.join(tmp, "b.mp4"), pulse=0.08, **kw)
        same = open(os.path.join(tmp, "a.mp4"), "rb").read() == open(os.path.join(tmp, "b.mp4"), "rb").read()
        t_on, b_on, s_on = r1["onsets"]
        bass = b_on == 0
        check("run(pulse=): the flow is untouched (fields, energy bitwise), the movie differs",
              np.array_equal(r0["fields"], r1["fields"]) and np.array_equal(r0["energy"], r1["energy"])
              and not same and r0["zooms"] is None)
        zz = r1["zooms"]
        check("   ... the zoom trace IS pulse_zoom of the bass band's onsets, and of no other band's",
              np.array_equal(zz, mt.pulse_zoom(np.arange(len(zz)) / 30.0, t_on[bass], s_on[bass], 0.08, 0.25))
              and not np.array_equal(zz, mt.pulse_zoom(np.arange(len(zz)) / 30.0, t_on[~bass], s_on[~bass], 0.08, 0.25)))
        k = np.ceil(t_on[bass] * 30 - 1e-9).astype(int)
        check("   ... and the zoom peaks on the frames after the bass onsets, within 1..1 + a",
              zz.min() >= 1.0 and zz.max() <= 1.08 + 1e-12 and zz[0] == 1.0 and bass.sum() >= 5
              and np.all(zz[k[k < len(zz)]] > zz[np.maximum(k[k < len(zz)] - 1, 0)]),
              f"{bass.sum()} bass onsets, max zoom {zz.max():.4f}")


def mean_field_gates():
    print("in-plane mean field (E4)")
    n = 128
    b0_of_t, bmax = mt.mean_field(0.5, 0.6, [1.0, 5.0], [0.5 * np.pi, -1.0], turn_s=2.0)
    ang = lambda t: float(np.arctan2(*[float(v) for v in b0_of_t(t)][::-1]))
    mag = lambda t: float(np.hypot(*[float(v) for v in b0_of_t(t)]))
    check("mean_field: fixed magnitude; angle theta0, half-way through a turn, after it, after two",
          bmax == 0.5 and all(abs(mag(t) - 0.5) < 1e-15 for t in (0.0, 1.7, 9.0))
          and abs(ang(0.5) - 0.6) < 1e-14 and abs(ang(2.0) - (0.6 + 0.25 * np.pi)) < 1e-14
          and abs(ang(4.0) - (0.6 + 0.5 * np.pi)) < 1e-14 and abs(ang(8.0) - (0.6 + 0.5 * np.pi - 1.0)) < 1e-14)
    check("   ... the turn is a smoothstep: a quarter of the way in time is 5/32 of the angle",
          abs(ang(1.5) - (0.6 + 0.15625 * 0.5 * np.pi)) < 1e-14)

    th = 0.6
    B = 0.5 * np.array([np.cos(th), np.sin(th)])
    step = mt.Stepper(n, 1.0 / 120.0, 0.0, 4, b0=mt.mean_field(0.5, th, [], []))
    kick, observe = mt.make_kick(step.p, step.kg)
    kx, ky = np.asarray(step.kg.kx), np.asarray(step.kg.ky)
    blob, none = np.array([[2.0, 3.0, 0.3, 1.0]]), np.zeros((0, 4))
    for sign, name, way in ((1, "z+", "against"), (-1, "z-", "along")):
        s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
        s = s._replace(fields=kick(s.fields, mt.pad(blob if sign > 0 else none), mt.pad(none if sign > 0 else blob)))
        f0, e0 = np.asarray(s.fields), float(observe(s.fields)[2])
        s, k = step.advance(s, 2.0, 1.0)
        t, f = float(s.t), np.asarray(s.fields)
        shifted = lambda d: f0 * np.exp(-1j * d * t * (kx * B[0] + ky * B[1]))     # f0(x - d B t)
        err = np.abs(f - shifted(-sign)).max() / np.abs(f0).max()
        bad = np.abs(f - shifted(sign)).max() / np.abs(f0).max()
        e1 = float(observe(s.fields)[2])
        check(f"one {name} blob from rest translates rigidly {way} B0 at |B0| (a distance {0.5 * t:.2f} = 3.3 sigma)",
              err < 1e-4 and bad > 1.0, f"rel error {err:.1e} ({k} steps); the other direction {bad:.2f}")
        check("   ... energy unchanged by the term (no dissipation in this run)", abs(e1 / e0 - 1.0) < 1e-4,
              f"E/E0 - 1 = {e1 / e0 - 1.0:+.1e}")

    # a turning field: the blob has moved by the time integral of B0
    b0 = mt.mean_field(0.5, 0.2, [0.5], [2.0], turn_s=1.0)
    step = mt.Stepper(n, 1.0 / 120.0, 0.0, 4, b0=b0)
    kick, observe = mt.make_kick(step.p, step.kg)
    s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
    s = s._replace(fields=kick(s.fields, mt.pad(none), mt.pad(blob)))
    f0 = np.asarray(s.fields)
    s, k = step.advance(s, 2.5, 1.0)
    t = float(s.t)
    tq = np.linspace(0.0, t, 20001)
    bq = np.array([[float(v) for v in b0[0](x)] for x in tq[::20]])
    tc = tq[::20]
    disp = np.sum(0.5 * (bq[1:] + bq[:-1]) * np.diff(tc)[:, None], axis=0)
    f = np.asarray(s.fields)
    err = np.abs(f - f0 * np.exp(-1j * (kx * disp[0] + ky * disp[1]))).max() / np.abs(f0).max()
    check("a z- blob under a field that turns by 2 rad: displaced by the integral of B0 dt",
          err < 2e-3 and abs(np.hypot(*disp) - 0.5 * t) > 0.05,
          f"rel error {err:.1e}; path {0.5 * t:.2f}, net displacement {np.hypot(*disp):.2f}")

    # both channels overlapping: the nonlinear run conserves energy with the term on
    step = mt.Stepper(n, 1.0 / 120.0, 0.0, 4, b0=mt.mean_field(0.5, th, [], []))
    kick, observe = mt.make_kick(step.p, step.kg)
    rng = np.random.default_rng(2)
    s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
    s = s._replace(fields=kick(s.fields, mt.pad(mt.blobs_for(rng, 0.45, 3, 0.5)), mt.pad(mt.blobs_for(rng, 0.45, 3, 0.5))))
    e0 = float(observe(s.fields)[2])
    s, k = step.advance(s, 1.0, 1.0)
    e1 = float(observe(s.fields)[2])
    check("z+ and z- together, nonlinear, no dissipation: energy conserved with B0 on", abs(e1 / e0 - 1.0) < 1e-3,
          f"E/E0 - 1 = {e1 / e0 - 1.0:+.1e} over {k} steps")
    st0 = mt.Stepper(n, 1.0 / 120.0, 0.0, 4)
    kick0 = mt.make_kick(st0.p, st0.kg)[0]
    ks = []
    for stp, kk in ((st0, kick0), (step, kick)):
        s0 = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), stp.p)
        # the adaptive path directly: advance() would pick the fixed dt_cap for one of them
        ks.append(int(stp._adaptive(s0._replace(fields=kk(s0.fields, mt.pad(blob), mt.pad(none))), 2.0)[1]))
    check("the field's rate joins the CFL sum: the same blob takes more steps with B0 = 0.5",
          ks[1] > 1.2 * ks[0], f"{ks[1]} vs {ks[0]} steps to t = 2")


def driven_field_gates():
    print("driven mean field (the bass-hit random walk's path)")
    n, th = 128, 0.6
    B = 0.5 * np.array([np.cos(th), np.sin(th)])
    blob, none = np.array([[2.0, 3.0, 0.3, 1.0]]), np.zeros((0, 4))
    out = []
    for b0 in (mt.mean_field(0.5, th, [], []), "driven-explicit", "driven"):
        step = mt.Stepper(n, 1.0 / 20.0, 0.0, 4, b0=b0)       # dt_cap out of the way: adaptive dt
        kick, observe = mt.make_kick(step.p, step.kg)
        s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
        s = s._replace(fields=kick(s.fields, mt.pad(blob), mt.pad(0.5 * blob)))
        s, k = step.advance(s, 1.0, 1.0, *((B,) if isinstance(b0, str) else ()))
        out.append((np.asarray(s.fields), float(s.t), k))
    (fa, ta, ka), (fb, tb, kb), (fc, tc, kc) = out
    err = np.abs(fa - fb).max() / np.abs(fa).max()
    check("driven-explicit, constant field, reproduces mean_field's run (nonlinear, both channels)",
          err < 1e-10 and ka == kb and abs(ta - tb) < 1e-12, f"rel {err:.1e}, {ka} and {kb} steps")
    # the exact-linear run ends at its own t (fewer, longer steps), so the two differ by
    # the evolution over tc - ta as well as by the time-stepping error
    err = np.abs(fa - fc).max() / np.abs(fa).max()
    check("driven (field in the exact linear step): the same nonlinear run in fewer steps",
          err < 3e-2 and abs(tc - ta) < 2e-2 and kc < ka,
          f"rel {err:.1e} with t apart by {tc - ta:+.1e}; {kc} against {ka} steps")

    # one z- blob alone has no nonlinearity: the exact step translates it to round-off, and
    # held piecewise constant over two advances it is displaced by the sum of B0 dt
    kx, ky = np.asarray(step.kg.kx), np.asarray(step.kg.ky)
    s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
    s = s._replace(fields=kick(s.fields, mt.pad(none), mt.pad(blob)))
    f0 = np.asarray(s.fields)
    B2 = np.array([-0.3, 0.9])
    s, _ = step.advance(s, 1.0, 1.0, B)
    t1 = float(s.t)
    s, _ = step.advance(s, 1.5, 1.0, B2)
    disp = B * t1 + B2 * (float(s.t) - t1)
    err = np.abs(np.asarray(s.fields) - f0 * np.exp(-1j * (kx * disp[0] + ky * disp[1]))).max() / np.abs(f0).max()
    check("two advances with different fields: the blob is displaced by the sum of B0 dt, to round-off",
          err < 1e-11, f"rel {err:.1e}")
    # with dissipation: the exact operator keeps the solver's own diagonal beside the field
    nu = 2e-9
    stepd = mt.Stepper(n, 1.0 / 20.0, nu, 4, b0="driven")
    kd, _ = mt.make_kick(stepd.p, stepd.kg)
    s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), stepd.p)
    s = s._replace(fields=kd(s.fields, mt.pad(none), mt.pad(blob)))
    f0 = np.asarray(s.fields)
    s, _ = stepd.advance(s, 1.0, 1.0, B)
    t = float(s.t)
    want = f0 * np.exp(-1j * t * (kx * B[0] + ky * B[1])) * np.exp(-nu * (kx**2 + ky**2) ** 4 * t)
    err = np.abs(np.asarray(s.fields) - want).max() / np.abs(f0).max()
    lost = 1.0 - np.abs(want).max() / np.abs(f0).max()
    check("driven with hyperviscosity: the blob is translated AND each mode decays by exp(-nu k^8 t)",
          err < 1e-11 and np.abs(np.asarray(s.fields) - f0 * np.exp(-1j * t * (kx * B[0] + ky * B[1]))).max() / np.abs(f0).max() > 1e-4,
          f"rel {err:.1e}; without the decay it would be off by more than 1e-4")

    # the walk itself, through run(): 8 s of the click track, whose last quarter has no kick drum
    sr, x = mt.click_track(duration=8.0)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(n=64, nbands=4, sigma_min=3.7 * L / 256, quiet=True, field_walk=30.0, mean_field_b0=1.5)
        res = mt.run(sr, x, os.path.join(tmp, "w.mp4"), ratio_walk=2.0, **kw)
        fix = mt.run(sr, x, os.path.join(tmp, "f.mp4"), **kw)
    tr, en = res["walk"], res["energy"]
    t_on, b_on, _ = res["onsets"]
    fps = res["fps"]
    # frames with no onset since the frame before: the field was set from that frame's energy
    quiet = np.array([i for i in range(1, len(tr)) if not np.any((t_on > (i - 1) / fps) & (t_on <= i / fps))])
    rel = np.abs(tr[quiet, 2] - tr[quiet, 1] * np.sqrt(2.0 * en[quiet - 1])) / np.maximum(tr[quiet, 2], 1e-30)
    check("run(field_walk=): |B0| = r sqrt(2 E) of the energy last observed, frame by frame",
          len(quiet) > 100 and rel.max() < 1e-12 and tr[:, 2].max() > 0.1, f"{len(quiet)} frames, rel {rel.max():.1e}")
    check("   ... the ratio walks, reflected, inside [0, 1.5] even at a step of 2 per hit; without ratio_walk it stays at 1.5",
          tr[:, 1].min() >= 0.0 and tr[:, 1].max() <= 1.5 and np.ptp(tr[:, 1]) > 0.5 and np.all(fix["walk"][:, 1] == 1.5),
          f"range {tr[:, 1].min():.3f} .. {tr[:, 1].max():.3f}")
    last_bass = t_on[b_on == 0].max()
    i0 = int(np.ceil((last_bass + 1.5) * fps))
    later = np.sum((t_on > last_bass + 1.5) & (b_on != 0))
    check("   ... the direction turns at the BASS onsets only: once they stop it settles, while the other bands play on",
          last_bass < 6.5 and later >= 3 and np.ptp(tr[i0:, 0]) < 1e-6 and np.ptp(tr[:i0, 0]) > 0.2,
          f"last bass onset {last_bass:.2f} s, {later} later onsets in other bands, turn before {np.ptp(tr[:i0, 0]):.2f} rad, after {np.ptp(tr[i0:, 0]):.1e}")
    try:
        mt.run(sr, x, "x.mp4", n=64, quiet=True, ratio_walk=0.3, mean_field_b0=1.0)
        refused = False
    except ValueError:
        refused = True
    check("ratio_walk without field_walk is refused", refused)

    bad = 0
    for st, args in ((step, ()), (mt.Stepper(32, 1.0 / 120.0, 0.0, 4), (B,))):
        try:
            st.advance(None, 1.0, 1.0, *args)
        except ValueError:
            bad += 1
    check("bvec missing in driven mode, or given without it, is an error", bad == 2)


def regrid_gates():
    print("zoom-out re-grid (E5)")
    n = 128
    step = mt.Stepper(n, 1.0 / 120.0, 0.0, 4)
    kick, observe = mt.make_kick(step.p, step.kg)
    rng = np.random.default_rng(3)
    s = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)
    # wide blobs only: every mode with any amplitude survives the doubling (sigma 0.4 puts
    # e^-35 of the peak at k = n/6 = 21, and the minimum-image cut at the box edge is e^-31)
    s = s._replace(fields=kick(s.fields, mt.pad(mt.blobs_for(rng, 0.4, 3, 1.0)), mt.pad(mt.blobs_for(rng, 0.4, 3, 1.0))))
    f = np.asarray(s.fields)
    g = mt.regrid(f, step.kg.dealias)
    kx, ky = np.asarray(step.kg.kx), np.asarray(step.kg.ky)
    xc = 0.5 * (n - 1) * mt.LBOX / n
    real = lambda a: np.fft.irfft2(a[:, 0], s=(n, n))
    shifted = real(f * np.exp(-1j * xc * (kx + ky)))                  # f(x - x_c) on the grid
    idx = (2 * np.arange(n)) % n
    err = np.abs(real(g) - 0.5 * shifted[:, idx][:, :, idx]).max() / np.abs(real(g)).max()
    check("regrid: g(x) = f(2x - x_c)/2 at every grid point, phi and psi (a band-limited state)",
          err < 1e-9, f"rel {err:.1e}")
    e0, e1 = float(observe(jnp.asarray(f))[2]), float(observe(jnp.asarray(g))[2])
    u0, u1 = float(observe(jnp.asarray(f))[1]), float(observe(jnp.asarray(g))[1])
    check("   ... the energy and the peak speed are unchanged (same velocities, half the size)",
          abs(e1 / e0 - 1.0) < 1e-9 and abs(u1 / u0 - 1.0) < 2e-2,   # the peak is read off the grid
          f"E ratio - 1 = {e1 / e0 - 1.0:+.1e}, umax ratio {u1 / u0:.6f}")
    # the picture: zoom 1/2 of the old speed image is the new speed image at zoom 1
    # (on phi, which is smooth: |u| has cusps at its zeros, where bilinear sampling at the
    # half pixels is off by several per cent of the peak whatever the state)
    img0 = np.asarray(observe(jnp.asarray(f))[0]["phi"]).T[::-1]
    img1 = np.asarray(observe(jnp.asarray(g))[0]["phi"]).T[::-1]
    d = np.abs(0.5 * mt.zoom_image(img0, 0.5, blur=False) - img1).max() / np.abs(img1).max()
    check("   ... and the old picture at zoom 1/2 (unblurred) is the new picture: no jump at a re-grid",
          d < 1e-2, f"max difference {d:.1e} of the peak (bilinear sampling at half pixels)")
    img0 = np.abs(img0)
    # what is dropped: a mode past the half-range goes, one inside it stays
    h = np.zeros_like(f)
    h[0, 0, 5, 3] = 1.0; h[0, 0, n - 7, 2] = 2.0; h[0, 0, n // 4 + 4, 3] = 1.0; h[0, 0, 3, n // 4 + 1] = 1.0
    # ... and the edge itself: 2 k = n/2 is the Nyquist line, in x (both signs) and in y
    h[0, 0, n // 4, 1] = 1.0; h[0, 0, n - n // 4, 1] = 1.0; h[0, 0, 1, n // 4] = 1.0
    gh = mt.regrid(h)
    check("a mode (kx, ky) lands on (2 kx, 2 ky), negative kx too; one that would pass n/2 is dropped",
          abs(abs(gh[0, 0, 10, 6]) - 0.5) < 1e-15 and abs(abs(gh[0, 0, n - 14, 4]) - 1.0) < 1e-15
          and np.count_nonzero(gh) == 2)
    gm = mt.regrid(h, step.kg.dealias)
    check("   ... with the solver's mask, nothing is left outside the dealiasing cut",
          not np.any(gm * (1.0 - np.asarray(step.kg.dealias))))
    # the run: two re-grids in 5 s of the click track
    sr, x = mt.click_track(duration=5.0)
    res = mt.run(sr, x, os.path.join(tempfile.mkdtemp(), "z.mp4"), n=64, zoom_out=2.0, quiet=True)
    rg = res["regrids"]
    check("run(zoom_out=2): re-grids at 2 and 4 s, each keeping most of the energy, the run finite",
          [t for t, _ in rg] == [2.0, 4.0] and all(0.3 < r <= 1.0 + 1e-6 for _, r in rg) and np.all(np.isfinite(res["energy"])),
          "energy kept " + " ".join(f"{r:.3f}" for _, r in rg))
    zs = mt.zoom_out_scale
    check("zoom_out_scale: 1 at a re-grid, falling to 1/2 at the next (the picture pulls BACK), counted from the re-grids done",
          zs(0.0, 10.0, 0) == 1.0 and abs(zs(5.0, 10.0, 0) - 2.0 ** -0.5) < 1e-15 and abs(zs(9.999, 10.0, 0) - 0.5) < 1e-3
          and zs(10.0, 10.0, 1) == 1.0 and abs(zs(15.0, 10.0, 1) - 2.0 ** -0.5) < 1e-15
          and zs(5.5, 1.1, 5) == 1.0 and zs(9.9, 1.1, 9) == 1.0)       # where 5.5 % 1.1 is 1.1 - 4e-16
    check("zoom_image at s < 1 shows the tiling: a constant image stays constant, the mean is kept",
          np.allclose(mt.zoom_image(np.full((16, 16), 3.0), 0.6), 3.0)
          and abs(mt.zoom_image(img0, 0.5).mean() / img0.mean() - 1.0) < 2e-2)


def section_gates():
    print("section trigger")
    sr = 22050
    rng = np.random.default_rng(0)
    from scipy.signal import lfilter
    lo = lfilter(*mt._bandpass(80.0, 400.0, sr), rng.standard_normal(60 * sr))
    hi = lfilter(*mt._bandpass(2000.0, 8000.0, sr), rng.standard_normal(60 * sr))
    t = np.arange(60 * sr) / sr
    steady = lo + 0.3 * hi
    ts, ss, tr = mt.detect_sections(sr, steady)
    check("steady noise for 60 s: no trigger", len(ts) == 0, f"novelty max {tr[:, 1].max():.3f} decades")
    # bass-heavy until 20 s, treble-heavy until 40 s, bass-heavy again
    w = ((t >= 20.0) & (t < 40.0)).astype(float)
    x = (1.0 - 0.9 * w) * lo + (0.1 + 0.9 * w) * hi
    ts, ss, tr = mt.detect_sections(sr, x)
    check("the balance of the bands changes at 20 s and 40 s: one trigger 1-8 s after each, no other",
          len(ts) == 2 and 1.0 < ts[0] - 20.0 < 8.0 and 1.0 < ts[1] - 40.0 < 8.0,
          "triggers at " + ", ".join(f"{a:.1f}" for a in ts))
    # causal: the triggers of the first 30 s do not depend on what comes after
    ts30 = mt.detect_sections(sr, x[:30 * sr])[0]
    check("causal: the first 30 s alone give the same first trigger", len(ts30) == 1 and ts30[0] == ts[0])
    # changes at 20, 24 and 40 s with short windows: the one at 24 s falls in the 8 s gap
    w = (((t >= 20.0) & (t < 24.0)) | (t >= 40.0)).astype(float)
    y = (1.0 - 0.9 * w) * lo + (0.1 + 0.9 * w) * hi
    ts = mt.detect_sections(sr, y, win_s=1.0, smooth_s=0.3)[0]
    ts2 = mt.detect_sections(sr, y, win_s=1.0, smooth_s=0.3, min_gap_s=2.0)[0]
    near = lambda a, b: 0.0 < a - b < 3.0
    check("changes at 20, 24, 40 s (1 s windows): the second is inside min_gap_s = 8 s and is skipped; "
          "with min_gap_s = 2 it fires",
          len(ts) == 2 and near(ts[0], 20.0) and near(ts[1], 40.0)
          and len(ts2) == 3 and near(ts2[1], 24.0),
          "8 s: " + ", ".join(f"{a:.1f}" for a in ts) + "; 2 s: " + ", ".join(f"{a:.1f}" for a in ts2))
    ts = mt.detect_sections(sr, 0.0 * x)[0]
    check("silence: no trigger, no NaN", len(ts) == 0)

    import tempfile
    sr, mono = mt.click_track(duration=6.0)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(n=64, nbands=4, sigma_min=3.7 * L / 256, quiet=True)
        r0 = mt.run(sr, mono, os.path.join(tmp, "a.mp4"), **kw)
        r1 = mt.run(sr, mono, os.path.join(tmp, "b.mp4"), mean_field_b0=0.5, section_every=4, **kw)
        nbass = int(np.sum(r1["onsets"][1] == 0))
        check("run(mean_field_b0=, section_every=4): finite, the flow differs, a redraw every 4th bass onset",
              np.all(np.isfinite(r1["energy"])) and not np.array_equal(r0["fields"], r1["fields"])
              and len(r1["sections"]) == nbass // 4 and r0["sections"] is None,
              f"{nbass} bass onsets, redraws at " + ", ".join(f"{a:.2f}" for a in r1["sections"]))
        r2 = mt.run(sr, mono, os.path.join(tmp, "c.mp4"), mean_field_b0=0.5, **kw)
        check("run(mean_field_b0=) with the detector on a 6 s track: no section yet, still runs",
              len(r2["sections"]) == 0 and np.all(np.isfinite(r2["energy"])))


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def stage_box_gates():
    print("the box takes the picture's shape (S2)")
    A = 16.0 / 9.0
    # --- the box
    check("box_for: 1 is exactly the square 2 pi box; 16:9 is long in x, 9:16 long in y, short side 2 pi",
          mt.box_for(1.0) == (L, L) and mt.box_for() == (L, L) and mt.box_for(A) == (L * A, L)
          and mt.box_for(1.0 / A)[0] == L and abs(mt.box_for(1.0 / A)[1] / (L * A) - 1.0) < 1e-15
          and raises(lambda: mt.box_for(0.0)) and raises(lambda: mt.box_for(-2.0)))

    # --- band plan
    s0, n0 = mt.band_plan(256)
    s1, n1 = mt.band_plan((256, 256), None, None, L, L)
    check("band_plan, square default unchanged: 4 bands at 256^2, 0.45 down to 3.7 dx, 3/5/9/15 blobs",
          np.array_equal(s0, s1) and np.array_equal(n0, n1) and len(s0) == 4 and s0[0] == 0.45
          and s0[-1] == 3.7 * L / 256 and list(n0) == [3, 5, 9, 15]
          # to the last bit, as the square formula 3.7 * L / n rounds it (not 3.7 * (L / n))
          and all(mt.band_plan(n)[0][-1] == 0.45 * ((3.7 * L / n) / 0.45) ** 1.0 for n in range(100, 400))
          and all(len(mt.band_plan(n)[0]) == k for n, k in ((256, 4), (512, 5), (1024, 6))),
          f"sigmas {np.round(s0, 4)}, blobs {list(n0)}")
    rows = []
    for (nx, ny), asp, want in (((256, 256), A, L * A / 256),       # stretched cells: dx is larger
                                ((512, 256), A, L / 256),           # dx = 0.89 dy: dy is larger
                                ((256, 512), 1.0 / A, L / 256),     # portrait: dx is larger
                                ((256, 128), A, L / 128),           # the light rung
                                ((512, 256), 2.0, L / 256)):        # square cells
        Lx, Ly = mt.box_for(asp)
        sg, nb = mt.band_plan((nx, ny), None, None, Lx, Ly)
        rows.append(abs(sg[-1] / (3.7 * want) - 1.0) < 1e-14 and sg[0] == 0.45
                    and abs(want / max(Lx / nx, Ly / ny) - 1.0) < 1e-14
                    and np.array_equal(nb, np.rint(3 * 0.45 / sg * Lx * Ly / L**2).astype(int)))
    check("band_plan, rectangular: narrowest = 3.7 x the LARGER spacing, widest 0.45, on 5 box/grid pairs",
          all(rows), f"{rows}")
    s2, n2 = mt.band_plan((512, 256), None, None, 2.0 * L, L)
    check("   ... a 2:1 box at the square's spacing has the square's widths and twice its blobs",
          np.allclose(s2, s0, rtol=1e-14) and list(n2) == [6, 10, 17, 30] and n2.sum() > 1.9 * n0.sum(),
          f"blobs {list(n2)} against {list(n0)}")
    s3, _ = mt.band_plan((256, 256), None, None, L * A, L)
    check("   ... 16:9 on 256^2 (stretched cells) has 3 bands: the narrowest blob is 3.7 x the long cell side",
          len(s3) == 3 and abs(s3[-1] - 3.7 * L * A / 256) < 1e-14, f"sigmas {np.round(s3, 4)}")

    # --- placement
    Lx, Ly = 3.0 * L, L
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    a1, a2 = mt.blobs_for(r1, 0.1, 9, 1.0), mt.blobs_for(r2, 0.1, 9, 1.0, None, (L, L))
    check("blobs_for(box = the square box) draws the stream it always drew", np.array_equal(a1, a2))
    r1, r2 = np.random.default_rng(6), np.random.default_rng(6)
    a1, a2 = mt.blobs_for(r1, 0.1, 9, 1.0), mt.blobs_for(r2, 0.1, 9, 1.0, None, (4.0 * L, L))
    check("   ... and in a 4:1 box the SAME uniforms, x over Lx and y over Ly",
          np.array_equal(4.0 * a1[:, 0], a2[:, 0]) and np.array_equal(a1[:, 1:], a2[:, 1:]))
    rng = np.random.default_rng(3)
    b = mt.blobs_for(rng, 0.1, 20000, 1.0, None, (Lx, Ly))
    ks = lambda v, ell: np.abs(np.sort(v) / ell - (np.arange(len(v)) + 0.5) / len(v)).max()
    check("blobs_for in a 3:1 box: x uniform over Lx, y uniform over Ly",
          ks(b[:, 0], Lx) < 0.02 and ks(b[:, 1], Ly) < 0.02 and b[:, 0].max() > 0.99 * Lx and b[:, 1].max() < Ly,
          f"KS x {ks(b[:, 0], Lx):.4f}, y {ks(b[:, 1], Ly):.4f}")
    b = mt.blobs_for(rng, 0.1, 20000, 1.0, 0.9 * Lx, (Lx, Ly))
    dx = (b[:, 0] - 0.9 * Lx + 0.5 * Lx) % Lx - 0.5 * Lx
    check("   ... with xc: gaussian about xc of width 0.12 Lx, wrapped into [0, Lx)",
          abs(dx.mean()) < 0.005 * Lx and abs(dx.std() / (mt.STEREO_JITTER * Lx) - 1.0) < 0.03
          and b[:, 0].min() >= 0.0 and b[:, 0].max() < Lx and (b[:, 0] < 0.1 * Lx).any() and ks(b[:, 1], Ly) < 0.02,
          f"std {dx.std() / Lx:.4f} Lx")
    bd, bm = mt.kick_blobs(rng, "dipole", 0.45, 400, 1.0, None, 4.0, 1.0, (Lx, Ly))
    d = bd[:400, 0:2] - bd[400:, 0:2]
    d = (d + 0.5 * np.array([Lx, Ly])) % np.array([Lx, Ly]) - 0.5 * np.array([Lx, Ly])
    check("dipole in a 3:1 box: both blobs inside the box, 1.5 sigma apart across either edge",
          np.allclose(np.hypot(d[:, 0], d[:, 1]), 1.5 * 0.45) and bd[:, 0].min() >= 0 and bd[:, 0].max() < Lx
          and bd[:, 0].max() > 2.5 * L and bd[:, 1].min() >= 0 and bd[:, 1].max() < Ly and np.array_equal(bd, bm))
    for kind in ("elsasser", "vortex", "island", "sheet", "sheet-island"):
        bp, _ = mt.kick_blobs(np.random.default_rng(8), kind, 0.1, 300, 1.0, None, 4.0, 1.0, (Lx, Ly))
        if not (bp[:, 0].max() > 2.5 * L and bp[:, 1].max() < Ly):
            check(f"{kind} in a 3:1 box is placed over the whole box", False)
            break
    else:
        check("every kick type in a 3:1 box is placed over the whole box", True)

    # --- the kick itself on a rectangular box, square cells (128 x 64) and stretched (128 x 128)
    Lx, Ly = 2.0 * L, L
    sigma, amp = 0.3, 0.7
    for nx, ny in ((128, 64), (128, 128), (256, 64)):
        p = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=0.4, dims=2, comm_backend="serial",
                          eqpars={"diss": (0.0, 0.0), "hyper": 4})
        kg = jr.setup_kgrids(p)
        kick, observe = mt.make_kick(p, kg)
        zero = jnp.zeros((2, 1, nx, ny // 2 + 1), dtype=jnp.complex128)
        x0, y0 = 1.6 * L, 0.3 * L                                    # x0 is outside a 2 pi box
        b4 = np.array([[x0, y0, sigma, amp]])
        f = kick(zero, mt.pad(b4), mt.pad(np.zeros((0, 4))))
        e = float(observe(f)[2])
        want = 0.5 * np.pi * (amp * sigma) ** 2 * np.e / (Lx * Ly) / 2.0      # one channel: phi = psi = z/2
        z = np.asarray(mt.ifft(f, p))[0, 0]
        i, j = np.unravel_index(np.argmax(z), z.shape)
        gx = 2.0 * np.asarray(mt.ifft(1j * kg.kx * f, p))[0, 0]
        check(f"blob at x = 1.6 * 2 pi in a 2:1 box on {nx} x {ny}: energy per unit AREA pi P^2/(4 Lx Ly), peak |grad| = amp, centred there",
              abs(e / want - 1.0) < 1e-9 and abs(np.abs(gx).max() / amp - 1.0) < 5e-2
              and abs(i * Lx / nx - x0) <= 0.51 * Lx / nx and abs(j * Ly / ny - y0) <= 0.51 * Ly / ny,
              f"E ratio - 1 = {e / want - 1.0:+.1e}, peak {np.abs(gx).max() / amp:.4f} amp")
        b6 = np.array([[x0, y0, sigma, amp, 1.0, 0.4]])
        g = kick(zero, mt.pad(b6), mt.pad(np.zeros((0, 6))))
        err = float(jnp.abs(f - g).max() / jnp.abs(f).max())
        check("   ... and the k-space (sheet) kick at ratio 1 is the same kick", err < 1e-12, f"rel {err:.1e}")
    # a sheet's length is along theta in BOX coordinates, cells stretched or not
    nx, ny = 128, 128
    p = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=0.4, dims=2, comm_backend="serial",
                      eqpars={"diss": (0.0, 0.0), "hyper": 4})
    kg = jr.setup_kgrids(p)
    kick, observe = mt.make_kick(p, kg)
    X, Y = np.meshgrid(np.arange(nx) * Lx / nx, np.arange(ny) * Ly / ny, indexing="ij")
    theta = 0.7
    f = kick(jnp.zeros((2, 1, nx, ny // 2 + 1), dtype=jnp.complex128), mt.pad(np.array([[6.0, 3.0, 0.2, amp, 3.0, theta]])),
             mt.pad(np.zeros((0, 6))))
    z = 2.0 * np.asarray(mt.ifft(f, p))[0, 0]
    w = (z - z.min()) ** 2
    ddx, ddy = X - 6.0, Y - 3.0
    along, across = np.cos(theta) * ddx + np.sin(theta) * ddy, -np.sin(theta) * ddx + np.cos(theta) * ddy
    la, lc = np.sqrt((w * along**2).sum() / w.sum()), np.sqrt((w * across**2).sum() / w.sum())
    check("a sheet on stretched cells (2:1 box, 128^2): length/width = 3 along theta in box coordinates",
          abs(la / lc - 3.0) < 0.15, f"ratio {la / lc:.3f}")

    # --- per unit area: the square state and its own two tiles in a 2:1 box
    n = 64
    p1, kg1, kick1, obs1 = setup(n)
    p2 = jr.Parameters(nx=2 * n, ny=n, Lx=2.0 * L, Ly=L, cfl_safety=0.4, dims=2, comm_backend="serial",
                       eqpars={"diss": (0.0, 0.0), "hyper": 4})
    kick2, obs2 = mt.make_kick(p2, jr.setup_kgrids(p2))
    rng = np.random.default_rng(11)
    bp, bm = mt.blobs_for(rng, 0.4, 3, 1.0), mt.blobs_for(rng, 0.3, 4, 0.6)
    two = lambda b: np.concatenate([b, b + np.array([L, 0.0, 0.0, 0.0])])
    fa = from_rest(kick1, n, bp, bm)
    fb = kick2(jnp.zeros((2, 1, 2 * n, n // 2 + 1), dtype=jnp.complex128), mt.pad(two(bp)), mt.pad(two(bm)))
    (ia, ua, ea), (ib, ub, eb) = obs1(fa), obs2(fb)
    pa, pb = np.percentile(np.asarray(ia["speed"]), 99.5), np.percentile(np.asarray(ib["speed"]), 99.5)
    check("per unit area: a state and its two tiles in a 2:1 box have the same energy, peak speed and "
          "colour-scale percentile (so the walk's sqrt(2E) is an rms)",
          abs(float(eb) / float(ea) - 1.0) < 1e-9 and abs(float(ub) / float(ua) - 1.0) < 1e-9 and abs(pb / pa - 1.0) < 1e-2
          and np.allclose(np.asarray(ib["speed"])[:n], np.asarray(ia["speed"]), atol=1e-9),
          f"E ratio - 1 = {float(eb) / float(ea) - 1.0:+.1e}, umax ratio - 1 = {float(ub) / float(ua) - 1.0:+.1e}, "
          f"percentile ratio {pb / pa:.5f} (4096 against 8192 samples)")

    # --- dissipation
    ck = mt.cutoff_k
    check("cutoff_k: n/3 exactly on the square box; the smaller of the two cutoffs otherwise",
          all(ck(n, n) == n / 3.0 for n in (64, 128, 256, 512))
          and abs(ck(256, 256, L * A, L) - 48.0) < 1e-12              # x: (256/3)/(16/9)
          and abs(ck(512, 256, L * A, L) - 256 / 3.0) < 1e-12         # y: x would be 96
          and abs(ck(256, 512, *mt.box_for(1.0 / A)) - 256 / 3.0) < 1e-12
          and abs(ck(256, 128, L * A, L) - 128 / 3.0) < 1e-12,
          f"16:9: 256^2 -> {ck(256, 256, L * A, L):.2f}, 512x256 -> {ck(512, 256, L * A, L):.2f}")
    rows = []
    for (nx, ny), asp in (((96, 96), A), ((192, 96), A), ((96, 192), 1.0 / A), ((96, 96), 1.0)):
        Lx, Ly = mt.box_for(asp)
        p = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=0.4, dims=2, comm_backend="serial",
                          eqpars={"diss": (0.0, 0.0), "hyper": 4})
        kg = jr.setup_kgrids(p)
        mask = np.asarray(kg.dealias).reshape(nx, ny // 2 + 1) > 0
        kxm = np.abs(np.asarray(kg.kx).reshape(nx, -1)[:, 0])[mask[:, 0]].max()
        kym = np.asarray(kg.ky).reshape(-1, ny // 2 + 1)[0][mask[0]].max()
        # the solver's own mask: the last retained mode of the coarser axis is within one mode of it
        dk = 2 * np.pi / (Lx if kxm < kym else Ly)
        rows.append((round(float(min(kxm, kym)), 3), round(float(ck(nx, ny, Lx, Ly)), 3),
                     -1e-9 <= ck(nx, ny, Lx, Ly) - min(kxm, kym) <= dk))
    check("   ... it is the solver's own dealiasing edge (the last retained mode of the coarser axis, within one mode)",
          all(r[2] for r in rows), " ".join(f"{a}/{b}" for a, b, _ in rows))

    # --- the frame
    fs = mt.frame_shape
    check("frame_shape: the grid itself for square cells; else the box's shape at the finer spacing, even sides",
          fs(256, 256) == (256, 256) and fs(512, 256, 2.0 * L, L) == (512, 256)
          and fs(256, 256, L * A, L) == (456, 256) and fs(512, 256, L * A, L) == (512, 288)
          and fs(256, 256, *mt.box_for(1.0 / A)) == (256, 456) and fs(128, 128, L * A, L) == (228, 128)
          and fs(128, 64, L * A, L) == (128, 72),
          f"16:9: 256^2 -> {fs(256, 256, L * A, L)}, 512x256 -> {fs(512, 256, L * A, L)}")
    u = np.random.default_rng(0).standard_normal((64, 64))
    fx = lambda r, c: np.cos(2 * np.pi * r + 0.3) * np.sin(4 * np.pi * c)      # r, c in [0, 1): the box
    rr, cc = np.meshgrid((np.arange(64) + 0.5) / 64, (np.arange(96) + 0.5) / 96, indexing="ij")
    r2, c2 = np.meshgrid((np.arange(64) + 0.5) / 64, (np.arange(170) + 0.5) / 170, indexing="ij")
    out = mt.resample_frame(fx(rr, cc), (64, 170))
    err = np.abs(out - fx(r2, c2)).max()
    check("resample_frame: the same shape is the image itself; otherwise the whole box on the new pixels (bilinear, periodic)",
          mt.resample_frame(u, (64, 64)) is u and out.shape == (64, 170) and err < (4 * np.pi / 96) ** 2 / 8 * 1.05,
          f"max error {err:.1e}")

    # --- the re-grid on nx != ny
    # (wide blobs on 256 x 128, so that every mode with any amplitude survives the doubling,
    # as in regrid_gates)
    nx, ny = 256, 128
    p = jr.Parameters(nx=nx, ny=ny, Lx=2.0 * L, Ly=L, cfl_safety=0.4, dims=2, comm_backend="serial",
                      eqpars={"diss": (0.0, 0.0), "hyper": 4})
    kg = jr.setup_kgrids(p)
    kick, observe = mt.make_kick(p, kg)
    rng = np.random.default_rng(3)
    f = np.asarray(kick(jnp.zeros((2, 1, nx, ny // 2 + 1), dtype=jnp.complex128),
                        mt.pad(mt.blobs_for(rng, 0.4, 3, 1.0, None, (2.0 * L, L))),
                        mt.pad(mt.blobs_for(rng, 0.4, 3, 1.0, None, (2.0 * L, L)))))
    g = mt.regrid(f, kg.dealias)
    real = lambda a: np.fft.irfft2(a[:, 0], s=(nx, ny))
    ix, jy = np.fft.fftfreq(nx, 1.0 / nx)[:, None], np.arange(ny // 2 + 1)[None, :]
    shifted = real(f * np.exp(-1j * (np.pi * (nx - 1) / nx * ix + np.pi * (ny - 1) / ny * jy)))
    err = np.abs(real(g) - 0.5 * shifted[:, (2 * np.arange(nx)) % nx][:, :, (2 * np.arange(ny)) % ny]).max() / np.abs(real(g)).max()
    e0, e1 = float(observe(jnp.asarray(f))[2]), float(observe(jnp.asarray(g))[2])
    check("regrid on 256 x 128 (2:1 box): g(x) = f(2x - x_c)/2 at every grid point, energy unchanged",
          err < 1e-9 and abs(e1 / e0 - 1.0) < 1e-9, f"rel {err:.1e}, E ratio - 1 = {e1 / e0 - 1.0:+.1e}")

    # --- the driver on a rectangular box
    import imageio
    sr, lr = mt.click_track(duration=4.0, stereo=True)
    mono = lr.mean(axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        r_sq = mt.run(sr, mono, os.path.join(tmp, "sq.mp4"), n=64, quiet=True)
        r_1 = mt.run(sr, mono, os.path.join(tmp, "sq1.mp4"), n=32, grid=(64, 64), aspect=1.0, quiet=True)
        same = open(os.path.join(tmp, "sq.mp4"), "rb").read() == open(os.path.join(tmp, "sq1.mp4"), "rb").read()
        check("run(aspect=1, grid=(n, n)) IS run(n=n): fields, energy and the movie file byte for byte; diss = 48/(n/3)^8",
              np.array_equal(r_sq["fields"], r_1["fields"]) and np.array_equal(r_sq["energy"], r_1["energy"]) and same
              and r_sq["diss"] == 48.0 / (64 / 3.0) ** 8 and r_sq["frame"] == (64, 64) and r_sq["box"] == (L, L)
              and r_sq["sheets"] is None)
        seen = []
        orig = mt.kick_blobs

        def spy(rng, kind, sigma, nblob, amp, xc=None, *rest):
            seen.append((round(float(sigma), 6), nblob, xc, rest))
            return orig(rng, kind, sigma, nblob, amp, xc, *rest)
        mt.kick_blobs = spy
        try:
            kw = dict(grid=(128, 64), aspect=2.0, nbands=4, quiet=True, kick_types="variety", stereo=lr, stereo_gain=1.0)
            r = mt.run(sr, mono, os.path.join(tmp, "r.mp4"), **kw)
        finally:
            mt.kick_blobs = orig
        Lx = 2.0 * L
        full = lambda q: 2.0 * q / (1.0 + q * q)
        spots = 0.5 + 0.4 * np.array([full(0.8), -full(0.8), full(-0.6)])
        top = np.array([s[2] for s in seen if s[0] == round(float(r["sigmas"][3]), 6)])
        counts = {s[0]: s[1] for s in seen}
        size = imageio.get_reader(os.path.join(tmp, "r.mp4")).get_meta_data()["size"]
        check("run on a 2:1 box, 128 x 64: stereo pan maps onto x over Lx (hats at Lx (0.5 + 0.4 pan)), the box goes to every kick",
              len(top) >= 6 and np.abs(top[:, None] / Lx - spots[None, :]).min(axis=1).max() < 0.01
              and top.max() > 1.5 * L and all(s[3][-1] == (Lx, L) for s in seen),
              f"hat centres {np.round(np.unique(np.round(top / Lx, 2)), 2)} Lx")
        check("   ... blob counts are the square's times 2, diss is pinned at the smaller cutoff, the frame is 128 x 64",
              [counts[round(float(s), 6)] for s in r["sigmas"]] == list(r["nblobs"])
              and np.array_equal(r["nblobs"], np.rint(2 * 3 * 0.45 / r["sigmas"]).astype(int))
              and abs(r["diss"] / (48.0 / (64 / 3.0) ** 8) - 1.0) < 1e-14 and r["frame"] == (128, 64) and tuple(size) == (128, 64)
              and r["fields"].shape == (2, 1, 128, 33) and np.all(np.isfinite(r["energy"])) and r["energy"].max() > 0,
              f"blobs {list(r['nblobs'])}, diss {r['diss']:.3e}, movie {tuple(size)}")
        # stretched cells, the pulse and the zoom-out together
        kw = dict(n=128, aspect=A, quiet=True, duration=3.0)
        ra = mt.run(sr, mono, os.path.join(tmp, "a.mp4"), **kw)
        rb = mt.run(sr, mono, os.path.join(tmp, "b.mp4"), pulse=0.08, **kw)
        rc = mt.run(sr, mono, os.path.join(tmp, "c.mp4"), pulse=0.08, zoom_out=1.0, grid=(128, 64), **kw)
        sa = imageio.get_reader(os.path.join(tmp, "a.mp4")).get_meta_data()["size"]
        sc = imageio.get_reader(os.path.join(tmp, "c.mp4")).get_meta_data()["size"]
        differ = open(os.path.join(tmp, "a.mp4"), "rb").read() != open(os.path.join(tmp, "b.mp4"), "rb").read()
        check("16:9 on 128^2 (stretched cells): the movie is 228 x 128, diss pinned at k = 24; the pulse leaves the flow alone",
              tuple(sa) == (228, 128) and ra["frame"] == (228, 128) and abs(ra["diss"] / (48.0 / 24.0 ** 8) - 1.0) < 1e-13
              and np.array_equal(ra["fields"], rb["fields"]) and differ and np.all(np.isfinite(ra["energy"])),
              f"movie {tuple(sa)}, diss {ra['diss']:.3e}")
        check("   ... and the zoom-out re-grids a 128 x 64 grid in a 16:9 box: twice, finite, each keeping most of the energy",
              [t for t, _ in rc["regrids"]] == [1.0, 2.0] and all(0.3 < q <= 1.0 + 1e-6 for _, q in rc["regrids"])
              and np.all(np.isfinite(rc["energy"])) and tuple(sc) == (128, 72),
              "energy kept " + " ".join(f"{q:.3f}" for _, q in rc["regrids"]) + f", movie {tuple(sc)}")
        # the walk in a rectangular box: |B0| = r sqrt(2 E), E per unit area
        rw = mt.run(sr, mono, os.path.join(tmp, "w.mp4"), grid=(128, 64), aspect=2.0, nbands=4, quiet=True,
                    field_walk=30.0, mean_field_b0=1.5, ratio_walk=2.0)
        tr, en = rw["walk"], rw["energy"]
        t_on = rw["onsets"][0]
        fps = rw["fps"]
        q = np.array([i for i in range(1, len(tr)) if not np.any((t_on > (i - 1) / fps) & (t_on <= i / fps))])
        rel = np.abs(tr[q, 2] - tr[q, 1] * np.sqrt(2.0 * en[q - 1])) / np.maximum(tr[q, 2], 1e-30)
        check("the walking field on a 2:1 box: runs, and |B0| = r sqrt(2 E) of the volume-averaged energy",
              len(q) > 50 and rel.max() < 1e-12 and tr[:, 2].max() > 0.05 and np.all(np.isfinite(en)),
              f"{len(q)} frames, rel {rel.max():.1e}")


def stage_sheet_gates():
    print("current sheets as a background (S4)")
    # --- the equilibrium
    Lx, a, amp, strength = 2.0 * L, 0.4, 0.7, 1.5
    psi0 = mt.sheet_psi0(strength, amp, a)
    x = np.linspace(0.0, Lx, 400001)
    psi = np.asarray(mt.sheet_equilibrium(x, Lx, psi0, a))
    b = np.gradient(psi, x)
    trap = lambda v, q: float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(q)))
    xp = x[np.argmax(np.abs(b))]
    check("sheet_psi0: the peak field of psi0 sech^2((x - Lx/2)/a) is strength * amp, at tanh^2 = 1/3",
          abs(np.abs(b).max() / (strength * amp) - 1.0) < 1e-6 and abs(np.tanh((xp - 0.5 * Lx) / a) ** 2 - 1.0 / 3.0) < 1e-3
          and abs(psi0 / (0.75 * np.sqrt(3.0) * a * strength * amp) - 1.0) < 1e-15,
          f"peak {np.abs(b).max():.8f} vs {strength * amp}")
    e_num = 0.5 * trap(b * b, x) / Lx
    check("   ... its energy per unit area is 8 psi0^2/(15 a Lx); psi peaks at Lx/2; no net field; periodic",
          abs(e_num / mt.sheet_energy_eq(psi0, a, Lx) - 1.0) < 1e-6 and abs(x[np.argmax(psi)] - 0.5 * Lx) < 1e-4
          and abs(trap(b, x)) < 1e-12 and abs(psi[0] - psi[-1]) < 1e-15 and abs(b[1] + b[-2]) < 1e-12
          and b[np.argmin(np.abs(x - 0.4 * Lx))] > 0 > b[np.argmin(np.abs(x - 0.6 * Lx))],
          f"E ratio - 1 = {e_num / mt.sheet_energy_eq(psi0, a, Lx) - 1.0:+.1e}")

    # --- the two halves of the source
    n = 64
    nu = 3e-9
    st = mt.Stepper(n, 1.0 / 120.0, nu, 4, source=True)
    col = np.asarray(jr.initialize(lambda x, y: jnp.stack([0.0 * x + 0.0 * y, mt.sheet_equilibrium(x, L, 1.0, 0.5) + 0.0 * y]),
                                   st.p).fields)[1, 0, :, 0]
    refused = raises(lambda: st.advance(None, 1.0, 1.0))
    st.set_equilibrium(col)
    kx = np.fft.fftfreq(n, 1.0 / n)
    src = np.asarray(st.src)
    want = nu * kx ** 8 * col
    rest = src.copy()
    rest[1, 0, :, 0] = 0.0
    check("set_equilibrium: the source is +nu k^8 psi_eq on psi's k_y = 0 column (-L psi_eq), zero elsewhere",
          np.abs(src[1, 0, :, 0] - want).max() <= 1e-14 * np.abs(want).max() and np.abs(want).max() > 0 and not np.any(rest)
          and src.shape == (2, 1, n, n // 2 + 1))
    f = np.random.default_rng(0).standard_normal((2, 1, n, n // 2 + 1)) + 0j
    g = np.asarray(mt.drag(jnp.asarray(f), 0.9, jnp.asarray(col)))
    g0 = np.asarray(mt.drag(jnp.asarray(f), 0.9))
    other = g.copy(); other[1, 0, :, 0] = g0[1, 0, :, 0]
    check("drag: without an equilibrium fields * decay; with one, psi's column relaxes TO it and nothing else changes",
          np.array_equal(g0, f * 0.9) and np.array_equal(other, g0)
          and np.abs(g[1, 0, :, 0] - (col + (f[1, 0, :, 0] - col) * 0.9)).max() < 1e-13
          and np.abs(np.asarray(mt.drag(jnp.asarray(g0 * 0 + 0j).at[1, 0, :, 0].set(col), 0.5, jnp.asarray(col)))[1, 0, :, 0] - col).max() < 1e-15)
    check("Stepper(source=True): advance before set_equilibrium, and a mean field with it, are errors; so is set_equilibrium without it",
          refused and raises(lambda: mt.Stepper(32, 1.0 / 120.0, 0.0, 4, b0="driven", source=True))
          and raises(lambda: mt.Stepper(32, 1.0 / 120.0, 0.0, 4, b0=mt.mean_field(0.5, 0.0, [], []), source=True))
          and raises(lambda: mt.Stepper(32, 1.0 / 120.0, 0.0, 4).set_equilibrium(col[:32])))

    # --- stationary with the drag on and no kicks; the control decays at the drag rate
    sr = 22050
    silent = np.zeros(6 * sr)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(n=128, quiet=True, sheets=1.5)
        r = mt.run(sr, silent, os.path.join(tmp, "s.mp4"), **kw)
        c = mt.run(sr, silent, os.path.join(tmp, "c.mp4"), sheets_source=False, **kw)
        e, ec, info = r["energy"], c["energy"], r["sheets"]
        f = r["fields"]
        colerr = np.abs(f[1, 0, :, 0] - info["psi_eq"]).max() / np.abs(info["psi_eq"]).max()
        a_want = 2.0 * 3.7 * L / 128
        check("run(sheets=1.5), 6 s of silence, drag on: no kick, the energy and the state do not move",
              len(r["onsets"][0]) == 0 and len(e) == 180 and np.abs(e / e[0] - 1.0).max() < 1e-10 and colerr < 1e-10
              and not np.any(f[0]) and not np.any(f[1][:, :, 1:]) and r["nsteps"] > 300,
              f"max |E/E0 - 1| = {np.abs(e / e[0] - 1.0).max():.1e}, state rel {colerr:.1e}, {r['nsteps']} steps")
        check("   ... and it IS the requested equilibrium: a = 2 x the narrowest blob, peak field 1.5 amp, energy 8 psi0^2/(15 a Lx)",
              abs(info["a"] / a_want - 1.0) < 1e-14 and abs(info["psi0"] / mt.sheet_psi0(1.5, 1.0, a_want) - 1.0) < 1e-14
              and abs(e[0] / info["energy"] - 1.0) < 1e-8 and abs(r["umax"][0] / 1.5 - 1.0) < 2e-3,
              f"E0/E_eq - 1 = {e[0] / info['energy'] - 1.0:+.1e}, peak field {r['umax'][0]:.4f}")
        # a frame is observed up to one step after its time: `late` is by how much (the
        # events of a silent run are its frames)
        tf = np.arange(len(ec)) / 30.0 + c["late"]
        dev = np.abs(ec / (ec[0] * np.exp(-tf)) - 1.0)
        check("the control, the same run WITHOUT the source: the sheets decay at the drag rate, E = E0 exp(-t/memory)",
              dev.max() < 1e-3 and ec[-1] / ec[0] < 3e-3 and np.array_equal(c["sheets"]["psi_eq"], info["psi_eq"]),
              f"E(3 s)/E0 = {ec[90] / ec[0]:.5f} vs {np.exp(-tf[90]):.5f}; E(6 s)/E0 = {ec[-1] / ec[0]:.2e}; "
              f"max deviation from the exponential {dev.max():.1e}")
        m = mt.run(sr, silent[:3 * sr], os.path.join(tmp, "m.mp4"), memory=2.0, sheets_source=False, **kw)
        devm = np.abs(m["energy"] / (m["energy"][0] * np.exp(-(np.arange(len(m["energy"])) / 30.0 + m["late"]) / 2.0)) - 1.0)
        check("   ... at memory = 2 the control decays half as fast (it is the drag, not the hyperviscosity)",
              devm.max() < 1e-3 and 0.2 < m["energy"][-1] / m["energy"][0] < 0.25, f"max deviation {devm.max():.1e}")
        nd = mt.run(sr, silent[:3 * sr], os.path.join(tmp, "n.mp4"), memory=None, **kw)
        nc = mt.run(sr, silent[:3 * sr], os.path.join(tmp, "nc.mp4"), memory=None, sheets_source=False, **kw)
        lost = 1.0 - nc["energy"][-1] / nc["energy"][0]
        check("   ... and with no drag at all: maintained, stationary; not maintained, the hyperviscosity alone erodes them "
              "(the other half of the source)",
              np.abs(nd["energy"] / nd["energy"][0] - 1.0).max() < 1e-10 and 1e-5 < lost < 1e-3
              and np.all(np.diff(nc["energy"]) < 0), f"unmaintained: 1 - E(3 s)/E0 = {lost:.1e}")

        # --- kicked, then left alone
        x = np.zeros(12 * sr)
        x[sr:2 * sr] = mt.click_track(duration=1.0, sr=sr)[1]
        k = mt.run(sr, x, os.path.join(tmp, "k.mp4"), **kw)
        kc = mt.run(sr, x, os.path.join(tmp, "kc.mp4"), sheets_source=False, **kw)
        ek, ekc = k["energy"] / info["energy"], kc["energy"] / info["energy"]
        t_last = k["onsets"][0].max()
        colk = np.abs(k["fields"][1, 0, :, 0] - info["psi_eq"]).max() / np.abs(info["psi_eq"]).max()
        check("kicked for 1 s, then 10 s alone: the energy jumps, then relaxes back to the equilibrium's (within 5%)",
              len(k["onsets"][0]) >= 4 and t_last < 2.0 and ek.max() > 1.5 and abs(ek[-1] - 1.0) < 0.05
              and abs(ek[-1] - 1.0) < 0.25 * np.abs(ek[90:181] - 1.0).max() and colk < 0.1,
              f"E/E_eq: peak {ek.max():.2f}, at 3 s {ek[90]:.3f}, 6 s {ek[180]:.3f}, 12 s {ek[-1]:.4f}; psi_eq column off by {colk:.1e}")
        check("   ... the control without the source goes to nothing instead", ekc.max() > 1.5 and ekc[-1] < 1e-2,
              f"E/E_eq at 12 s {ekc[-1]:.1e}")
        check("   ... and the kicks never entered psi_eq: it is the silent run's, bit for bit, though psi's column was kicked",
              np.array_equal(k["sheets"]["psi_eq"], info["psi_eq"])
              and np.abs(np.fft.ifft(k["fields"][1, 0, :, 0] - info["psi_eq"])).max() > 1e-6 * np.abs(np.fft.ifft(info["psi_eq"])).max())

        # --- width, strength, a rectangular box, the phase-2 look
        w4 = mt.run(sr, silent[:sr], os.path.join(tmp, "w.mp4"), n=128, quiet=True, sheets=0.5, sheets_width=4.0, amp=2.0)
        check("sheets=0.5, sheets_width=4, amp=2: a = 4 x the narrowest blob, peak field 0.5 * 2",
              abs(w4["sheets"]["a"] / (4.0 * 3.7 * L / 128) - 1.0) < 1e-14 and abs(w4["umax"][0] / 1.0 - 1.0) < 2e-3
              and abs(w4["energy"][-1] / w4["energy"][0] - 1.0) < 1e-10, f"peak field {w4['umax'][0]:.4f}")
        rr = mt.run(sr, silent[:sr], os.path.join(tmp, "r.mp4"), grid=(256, 128), aspect=2.0, quiet=True, sheets=1.5)
        prof = np.fft.ifft(rr["sheets"]["psi_eq"]).real
        check("sheets in a 2:1 box (256 x 128): centred on Lx/2, vary along x only, stationary, energy 8 psi0^2/(15 a Lx)",
              np.argmax(prof) == 128 and abs(rr["energy"][0] / mt.sheet_energy_eq(rr["sheets"]["psi0"], rr["sheets"]["a"], 2.0 * L) - 1.0) < 1e-8
              and abs(rr["energy"][-1] / rr["energy"][0] - 1.0) < 1e-10 and not np.any(rr["fields"][1][:, :, 1:]),
              f"E0 {rr['energy'][0]:.5f}")
        _, lr = mt.click_track(duration=3.0, sr=sr, stereo=True)
        v = mt.run(sr, lr.mean(axis=1), os.path.join(tmp, "v.mp4"), n=128, quiet=True,
                   sheets=1.5, kick_types="variety", stereo=lr, stereo_gain=2.0, pulse=0.08, sheet_energy=2.0)
        check("sheets with variety kicks, stereo placement and the pulse: runs to the end, finite, above the equilibrium's energy",
              np.all(np.isfinite(v["energy"])) and v["energy"].max() > 1.05 * v["energy"][0]
              and abs(v["energy"][0] / v["sheets"]["energy"] - 1.0) < 1e-8 and len(v["onsets"][0]) > 10,
              f"E/E_eq up to {v['energy'].max() / v['sheets']['energy']:.2f}")

        # --- two backgrounds are refused
        base = dict(n=64, quiet=True, sheets=1.5)
        go = lambda **kw2: mt.run(sr, silent[:sr], os.path.join(tmp, "x.mp4"), **{**base, **kw2})
        check("sheets with mean_field_b0, with field_walk, with the walking field + ratio_walk, or with ratio_walk: ValueError",
              raises(lambda: go(mean_field_b0=0.5)) and raises(lambda: go(field_walk=30.0))
              and raises(lambda: go(mean_field_b0=1.5, field_walk=30.0, ratio_walk=2.0)) and raises(lambda: go(ratio_walk=0.3)))
        ok = go(sheets=0.0, mean_field_b0=1.5, field_walk=30.0, ratio_walk=2.0)
        check("   ... each of them runs without sheets, and sheets alone run; negative sheets or a zero width are errors",
              ok["sheets"] is None and ok["walk"] is not None and go()["sheets"] is not None
              and go(sheets=0.0, mean_field_b0=0.5)["sections"] is not None
              and raises(lambda: go(sheets=-1.0)) and raises(lambda: go(sheets_width=0.0)))


def stage_harris_gates():
    print("current sheets, the harris profile (S4)")
    # --- the profile: B_y = +d psi/dx (B = zhat x grad psi, the script's convention)
    Lx, a, b0 = 2.0 * L, 0.3, 2.1
    x = np.linspace(0.0, Lx, 400001)
    psi = np.asarray(mt.sheet_harris(x, Lx, b0, a))
    by = np.gradient(psi, x)
    c = mt.harris_slope(a, Lx)
    want = b0 * (np.tanh((x - 0.25 * Lx) / a) - np.tanh((x - 0.75 * Lx) / a) - c)
    trap = lambda v, q: float(np.sum(0.5 * (v[1:] + v[:-1]) * np.diff(q)))
    at = lambda q: by[np.argmin(np.abs(x - q * Lx))]
    check("sheet_harris: d psi/dx = b0 [tanh((x - Lx/4)/a) - tanh((x - 3Lx/4)/a) - c], c = 1 to e^(-Lx/2a)",
          np.abs(by - want)[1:-1].max() < 1e-7 * b0 and 0.0 < 1.0 - c < 4.0 * np.exp(-0.5 * Lx / a) and 1.0 - c > 1e-12,
          f"max error {np.abs(by - want)[1:-1].max():.1e}, 1 - c = {1.0 - c:.1e}")
    check("   ... the field is -b0 outside the sheets, +b0 between them, zero AT them, and b0 is its peak",
          abs(at(0.0) + b0) < 1e-6 and abs(at(0.5) - b0) < 1e-6 and abs(at(0.999) + b0) < 1e-6
          and abs(at(0.25)) < 1e-3 and abs(at(0.75)) < 1e-3 and abs(np.abs(by).max() / b0 - 1.0) < 1e-6,
          f"B_y at 0, Lx/2: {at(0.0):+.6f}, {at(0.5):+.6f}; max |B| / b0 = {np.abs(by).max() / b0:.8f}")
    e_num = 0.5 * trap(by * by, x) / Lx
    check("   ... energy (b0^2/2)(1 - 4a/Lx); exactly periodic psi; no net flux; currents of opposite sign at the two sheets",
          abs(e_num / mt.harris_energy_eq(b0, a, Lx) - 1.0) < 1e-6 and abs(psi[0] - psi[-1]) < 1e-13 * np.abs(psi).max()
          and abs(trap(by, x)) < 1e-9 and np.gradient(by, x)[100000] > 0 > np.gradient(by, x)[300000]
          and 0.9 * 0.5 * b0**2 < e_num < 0.5 * b0**2,
          f"E ratio - 1 = {e_num / mt.harris_energy_eq(b0, a, Lx) - 1.0:+.1e}, psi(Lx) - psi(0) = {psi[-1] - psi[0]:.1e}")
    u = np.array([-800.0, -3.0, 0.0, 0.5, 900.0])
    check("log_cosh: ln cosh where cosh does not overflow, |u| - ln 2 where it would",
          np.allclose(np.asarray(mt.log_cosh(u))[1:4], np.log(np.cosh(u[1:4])), rtol=0, atol=1e-15)
          and np.allclose(np.asarray(mt.log_cosh(u))[[0, 4]], np.abs(u[[0, 4]]) - np.log(2.0), rtol=1e-15))

    sr = 22050
    silent = np.zeros(6 * sr)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(n=128, quiet=True, sheets=3.0, sheets_profile="harris")
        r = mt.run(sr, silent, os.path.join(tmp, "s.mp4"), **kw)
        c0 = mt.run(sr, silent, os.path.join(tmp, "c.mp4"), sheets_source=False, **kw)
        e, ec, info = r["energy"], c0["energy"], r["sheets"]
        f = r["fields"]
        colerr = np.abs(f[1, 0, :, 0] - info["psi_eq"]).max() / np.abs(info["psi_eq"]).max()
        a_want = 2.0 * 3.7 * L / 128
        prof = np.fft.ifft(info["psi_eq"]).real / 128                 # psi_eq(x): the rfft2 column is ny times it
        byg = np.fft.ifft(1j * np.fft.fftfreq(128, 1.0 / 128) * info["psi_eq"]).real / 128
        check("run(sheets=3, harris), 6 s of silence, drag on: no kick, the energy and the state do not move",
              len(r["onsets"][0]) == 0 and len(e) == 180 and np.abs(e / e[0] - 1.0).max() < 1e-10 and colerr < 1e-10
              and not np.any(f[0]) and not np.any(f[1][:, :, 1:]) and r["nsteps"] > 300,
              f"max |E/E0 - 1| = {np.abs(e / e[0] - 1.0).max():.1e}, state rel {colerr:.1e}, {r['nsteps']} steps")
        check("   ... and it IS the requested equilibrium: a = 2 x the narrowest blob, field +-3 amp reversing at Lx/4 and 3Lx/4, "
              "energy (b0^2/2)(1 - 4a/Lx)",
              info["profile"] == "harris" and info["b0"] == 3.0 and info["psi0"] is None and abs(info["a"] / a_want - 1.0) < 1e-14
              and abs(e[0] / info["energy"] - 1.0) < 1e-3 and abs(e[0] / mt.harris_energy_eq(3.0, a_want, L) - 1.0) < 1e-3
              and abs(byg[0] + 3.0) < 1e-2 and abs(byg[64] - 3.0) < 1e-2 and abs(byg[32]) < 1e-3 and abs(byg[96]) < 1e-3
              and abs(r["umax"][0] / 3.0 - 1.0) < 1e-2 and e[0] > 3.0,
              f"E0 = {e[0]:.4f} vs {info['energy']:.4f}; B_y at 0, Lx/2: {byg[0]:+.4f}, {byg[64]:+.4f}; peak field {r['umax'][0]:.4f}")
        tf = np.arange(len(ec)) / 30.0 + c0["late"]
        dev = np.abs(ec / (ec[0] * np.exp(-tf)) - 1.0)
        check("the control, harris WITHOUT the source: the sheets decay at the drag rate, E = E0 exp(-t/memory)",
              dev.max() < 1e-3 and ec[-1] / ec[0] < 3e-3 and np.array_equal(c0["sheets"]["psi_eq"], info["psi_eq"]),
              f"E(6 s)/E0 = {ec[-1] / ec[0]:.2e}; max deviation from the exponential {dev.max():.1e}")
        nd = mt.run(sr, silent[:3 * sr], os.path.join(tmp, "n.mp4"), memory=None, **kw)
        nc = mt.run(sr, silent[:3 * sr], os.path.join(tmp, "nc.mp4"), memory=None, sheets_source=False, **kw)
        lost = 1.0 - nc["energy"][-1] / nc["energy"][0]
        check("   ... with no drag: maintained, stationary; not maintained, the hyperviscosity erodes the tanh layers",
              np.abs(nd["energy"] / nd["energy"][0] - 1.0).max() < 1e-10 and lost > 1e-7 and np.all(np.diff(nc["energy"]) < 0),
              f"unmaintained: 1 - E(3 s)/E0 = {lost:.1e}")
        x = np.zeros(4 * sr)
        x[sr:2 * sr] = mt.click_track(duration=1.0, sr=sr)[1]
        k = mt.run(sr, x, os.path.join(tmp, "k.mp4"), **kw)
        moved = np.abs(np.fft.ifft(k["fields"][1, 0, :, 0] - info["psi_eq"])).max() / np.abs(np.fft.ifft(info["psi_eq"])).max()
        check("kicked: psi_eq is the silent run's bit for bit, though psi's column was kicked; the run stays finite",
              len(k["onsets"][0]) >= 4 and np.array_equal(k["sheets"]["psi_eq"], info["psi_eq"]) and moved > 1e-6
              and np.all(np.isfinite(k["energy"])) and k["energy"].max() > k["energy"][0],
              f"column moved by {moved:.1e} of psi_eq, E/E_eq up to {k['energy'].max() / k['energy'][0]:.3f}")
        rr = mt.run(sr, silent[:sr], os.path.join(tmp, "r.mp4"), grid=(256, 128), aspect=2.0, quiet=True, sheets=2.0,
                    sheets_profile="harris")
        byr = np.fft.ifft(1j * np.fft.fftfreq(256, 1.0 / 256) * (2 * np.pi / (2.0 * L)) * rr["sheets"]["psi_eq"]).real / 128
        check("harris in a 2:1 box (256 x 128): reverses at Lx/4 and 3Lx/4 of the LONG side, +-2 away from them, stationary, "
              "energy (b0^2/2)(1 - 4a/Lx)",
              abs(byr[64]) < 1e-6 and abs(byr[192]) < 1e-6 and abs(byr[128] - 2.0) < 1e-2 and abs(byr[0] + 2.0) < 1e-2
              and abs(rr["energy"][0] / mt.harris_energy_eq(2.0, rr["sheets"]["a"], 2.0 * L) - 1.0) < 1e-3
              and abs(rr["energy"][-1] / rr["energy"][0] - 1.0) < 1e-10 and not np.any(rr["fields"][1][:, :, 1:]),
              f"E0 {rr['energy'][0]:.4f} vs {mt.harris_energy_eq(2.0, rr['sheets']['a'], 2.0 * L):.4f}")
        b1 = mt.run(sr, silent[:sr], os.path.join(tmp, "b.mp4"), n=128, quiet=True, sheets=1.5)
        go = lambda **kw2: mt.run(sr, silent[:sr], os.path.join(tmp, "x.mp4"), n=128, quiet=True, sheets=1.5, **kw2)
        check("run()'s default profile is still the bump; an unknown profile, harris with a mean field, or a > Lx/8 are errors",
              b1["sheets"]["profile"] == "bump" and b1["sheets"]["psi0"] is not None
              and raises(lambda: go(sheets_profile="tanh")) and raises(lambda: go(sheets_profile="harris", mean_field_b0=0.5))
              and raises(lambda: go(sheets_profile="harris", field_walk=30.0))
              and raises(lambda: go(sheets_profile="harris", sheets_width=5.0)) and go(sheets_profile="harris")["sheets"]["b0"] == 1.5)
        stop = raises(lambda: mt.run(sr, x, os.path.join(tmp, "m.mp4"), max_steps=50, **kw), RuntimeError)
        check("max_steps: the run aborts once it has taken more", stop and mt.run(sr, silent[:sr], os.path.join(tmp, "m2.mp4"), max_steps=10**6, **kw)["nsteps"] > 50)


if __name__ == "__main__":
    if jr._precision.ftype != jnp.float64:
        sys.exit("run with TARANIS_PRECISION=64")
    kick_gates()
    dipole_gate()
    stereo_gates()
    pulse_gates()
    mean_field_gates()
    driven_field_gates()
    regrid_gates()
    section_gates()
    stage_box_gates()
    stage_sheet_gates()
    stage_harris_gates()
    print(f"\n{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED: ' + '; '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
