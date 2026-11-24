"""Generate the reference for devtools/checkmusic.js (MUSIC_PLAN M2).

    python webgpu/gen_musicfixture.py       # writes webgpu/devtools/fixtures/music_ref.json

`webgpu/music.js` ports two pieces of the offline script, examples/music_turbulence.py, to
the browser: `band_plan` (-> musicBandPlan) and the causal onset detector
`CausalOnsetDetector` (-> MusicDetector). This file runs the PYTHON ones and records what
they return, so the gate holds the port to the reference implementation's own numbers.

1. `band_plan` at n = 256, 512, 1024 and with explicit nbands / sigma_min.

2. The detector on a test signal, under several option sets (each option the port takes is
   varied in at least one). The signal is stored with the fixture, and is kept SMALL on
   purpose -- the fixture is committed:
     * it is 8 s at 11025 Hz, as int16 (176 kB, 235 kB in base64). The detector is run at
       nfft = 512, which at this rate is the same 46 ms window the page uses at
       44100 / 2048, so the code path is the production one at a quarter of the bins;
     * both sides read the SAME quantised samples (int16 / 32768), so the comparison does
       not depend on reproducing a synthesiser: the gate does its own Blackman window and
       FFT and feeds its magnitudes to the port, as this file feeds numpy's to the original.
   The signal is formula-defined (sines, exponentials and an integer LCG for the noise --
   no numpy RNG, no scipy filter), so regenerating it gives the same bytes.

3. (MUSIC_STAGE_PLAN S2 / S4, key `stage`) the rectangular box and the current sheets:
   `box_for`, `cutoff_k`, `band_plan` on the page's three grid rungs in 16:9, 2:1, 2.4:1 and
   portrait boxes, the dissipation and stereo-placement expressions of `run()`, `blobs_for` /
   `kick_blobs` in a rectangular box driven by a SCRIPTED generator (so the page, which has a
   different RNG, can be fed the same uniforms and normals), and the `bump` sheet profile
   (`sheet_psi0`, `sheet_equilibrium`, `sheet_energy_eq`). Everything recorded before this
   key is unchanged by it: the key is appended.

Deterministic: running this twice writes the same file.
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

import numpy as np

import music_turbulence as mt

SR, DUR = 11025, 8.0


def lcg_noise(n, seed):
    # integer LCG (the classic 1103515245 / 12345 mod 2^31), mapped to [-1, 1)
    out, x = np.empty(n), seed
    for i in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out[i] = x / 1073741824.0 - 1.0
    return out


def test_signal():
    n = int(SR * DUR)
    x = np.zeros(n)

    def add(t0, sig, gain=1.0):
        i0 = int(round(t0 * SR))
        m = min(len(sig), n - i0)
        if m > 0:
            x[i0:i0 + m] += gain * sig[:m]

    tt = np.arange(int(0.4 * SR)) / SR
    kick = np.sin(2 * np.pi * (45.0 * tt + 3.0 * (1 - np.exp(-tt / 0.05)))) * np.exp(-tt / 0.12)
    snare = lcg_noise(len(tt), 12345) * np.exp(-tt / 0.06) + 0.4 * np.sin(2 * np.pi * 220.0 * tt) * np.exp(-tt / 0.05)
    hn = lcg_noise(len(tt) + 2, 987)
    hat = (hn[2:] - 2 * hn[1:-1] + hn[:-2]) * 0.25 * np.exp(-tt / 0.02)      # second difference: high-passed
    attack = np.minimum(tt / 0.005, 1.0)
    note = lambda f, tau: attack * np.sin(2 * np.pi * f * tt) * np.exp(-tt / tau)
    beat = 0.5
    for i in range(int(DUR / beat)):
        add(i * beat, kick, 1.0 if i % 3 else 0.6)
        if i % 4 in (1, 3):
            add(i * beat, snare, 0.6 if i % 8 == 1 else 0.35)
        add((i + 0.5) * beat, hat, 0.5 if i % 2 else 0.25)
    for t0, f, tau, g in [(0.37, 330.0, 0.25, 0.5), (1.11, 440.0, 0.2, 0.4), (1.86, 660.0, 0.15, 0.5),
                          (2.63, 990.0, 0.2, 0.3), (3.30, 1320.0, 0.1, 0.45), (4.12, 330.0, 0.3, 0.35),
                          (4.71, 2200.0, 0.08, 0.4), (5.43, 440.0, 0.2, 0.5), (6.19, 1760.0, 0.1, 0.3),
                          (6.90, 660.0, 0.25, 0.45), (7.37, 3300.0, 0.06, 0.35)]:
        add(t0, note(f, tau), g)
    pcm = np.round(0.8 * 32767 * x / np.max(np.abs(x))).astype("<i2")
    return pcm


# option sets, in the PORT's names (music.js MusicDetector) -> the python keyword
KEYS = dict(nbands="nbands", fmin="fmin", fmax="fmax", fps="fps", nfft="nfft", threshold="threshold",
            minGap="min_gap_s", meanS="mean_s", refDecayS="ref_decay_s", normS="norm_s", warmupS="warmup_s",
            fluxFloor="flux_floor", peakDecayS="peak_decay_s")
CASES = [
    ("the page's settings: 4 bands, threshold 0.3", dict(nbands=4, threshold=0.3)),
    ("the detector's defaults: 3 bands, threshold 0.15", dict()),
    ("5 bands, threshold 0.5", dict(nbands=5, threshold=0.5)),
    ("a long refractory gap (0.3 s)", dict(nbands=4, threshold=0.3, minGap=0.3)),
    ("a short trailing mean (0.05 s)", dict(nbands=4, threshold=0.3, meanS=0.05)),
    ("a short normalisation window (1 s)", dict(nbands=4, threshold=0.3, normS=1.0)),
    ("a fast reference decay (0.2 s)", dict(nbands=4, threshold=0.3, refDecayS=0.2)),
    ("a late warm-up (3 s)", dict(nbands=4, threshold=0.3, warmupS=3.0)),
    ("a narrower frequency range (100-2000 Hz)", dict(nbands=4, threshold=0.3, fmin=100.0, fmax=2000.0)),
    ("30 frames per second", dict(nbands=4, threshold=0.3, fps=30.0)),
    ("nfft 256", dict(nbands=3, threshold=0.3, nfft=256)),
    # the noise floor (off in every case above). This signal has no quiet stretch for a
    # realistic floor to act on, so the two cases set it ABOVE the band's own peak-hold
    # (1.5x): every strength is then scaled by it and the weaker onsets drop out, which is
    # what makes the option, and the peak-hold's decay time, show in the answer.
    ("a noise floor of 1.5 of the band's 30 s peak-hold", dict(nbands=4, threshold=0.3, fluxFloor=1.5)),
    ("a noise floor of 1.5 of a 0.5 s peak-hold", dict(nbands=4, threshold=0.3, fluxFloor=1.5, peakDecayS=0.5)),
]


class ScriptRng:
    """The three numpy Generator methods blobs_for / kick_blobs call, fed from recorded
    streams: `u` the uniforms in [0, 1) (uniform returns lo + (hi - lo) * u, which is what
    numpy computes), `g` the standard normals, `s` the signs. Each is consumed in call
    order, so the fixture can say exactly which number became which coordinate."""

    def __init__(self, u, g, s):
        self.u, self.g, self.s = list(u), list(g), list(s)

    def _take(self, src, n):
        out, src[:] = src[:n], src[n:]
        assert len(out) == n, "the scripted stream ran out"
        return np.array(out, dtype=float)

    def uniform(self, lo, hi, size=None):
        shape = (size,) if np.ndim(size) == 0 else tuple(size)
        v = self._take(self.u, int(np.prod(shape))).reshape(shape)
        return np.asarray(lo, float) + (np.asarray(hi, float) - np.asarray(lo, float)) * v

    def standard_normal(self, n):
        return self._take(self.g, n)

    def choice(self, opts, size):
        return self._take(self.s, size)


def lcg_unit(n, seed):
    # n numbers in [0, 1) from the same integer LCG as lcg_noise
    return [float(v) for v in 0.5 * (lcg_noise(n, seed) + 1.0)]


RUNGS = (("light", 256, 128), ("standard", 256, 256), ("sharp", 512, 256))    # long x short


def stage_rows():
    st = dict(source="examples/music_turbulence.py box_for, cutoff_k, band_plan, blobs_for, kick_blobs, "
                     "sheet_psi0, sheet_equilibrium, sheet_energy_eq; the diss and stereo expressions of run()",
              lbox=float(mt.LBOX), hyper=4)
    st["boxes"] = [dict(aspect=a, box=[float(v) for v in mt.box_for(a)])
                   for a in (1.0, 16.0 / 9.0, 2.0, 2.4, 1.6, 9.0 / 16.0, 0.5, 1.0 / 2.4, 390.0 / 844.0)]

    def plan(tag, aspect, rung, nx, ny, nbands=None, sigma_min=None):
        Lx, Ly = mt.box_for(aspect)
        row = dict(tag=tag, aspect=aspect, rung=rung, grid=[nx, ny], box=[float(Lx), float(Ly)],
                   nbands=nbands, sigmaMin=sigma_min, cutoff_k=float(mt.cutoff_k(nx, ny, Lx, Ly)))
        # run()'s default dissipation, at hyper 4: 48 amp / cutoff_k^(2 hyper)
        row["diss"] = [[amp, float(48.0 * amp / mt.cutoff_k(nx, ny, Lx, Ly) ** (2 * 4))] for amp in (1.0, 0.3, 2.5)]
        try:
            sig, nbl = mt.band_plan((nx, ny), nbands, sigma_min, Lx, Ly)
            row.update(sigmas=[float(v) for v in sig], nblobs=[int(v) for v in nbl])
        except ValueError as e:
            row["error"] = str(e)
        return row

    plans = []
    for tag, aspect in (("16:9", 16.0 / 9.0), ("2:1", 2.0), ("2.4:1", 2.4), ("9:16", 9.0 / 16.0), ("1:2.4", 1.0 / 2.4)):
        for rung, nl, ns in RUNGS:
            nx, ny = (nl, ns) if aspect >= 1.0 else (ns, nl)
            plans.append(plan(tag, aspect, rung, nx, ny))
    # the square box's rungs (sharp is standard there), and the rule with explicit arguments
    plans.append(plan("square", 1.0, "light", 128, 128))
    plans.append(plan("square", 1.0, "standard", 256, 256))
    plans.append(plan("16:9, 3 bands", 16.0 / 9.0, "light", 256, 128, 3))
    plans.append(plan("2:1, sigma_min 0.2", 2.0, "sharp", 512, 256, None, 0.2))
    plans.append(plan("9:16, 2 bands, sigma_min 0.3", 9.0 / 16.0, "standard", 256, 256, 2, 0.3))
    st["plans"] = plans

    # stereo placement: run()'s xcs = Lx (0.5 + STEREO_SPAN clip(gain pan))
    st["panx"] = [dict(aspect=a, pan=p, gain=g,
                       xc=float(mt.box_for(a)[0] * (0.5 + mt.STEREO_SPAN * np.clip(g * p, -1.0, 1.0))))
                  for a in (16.0 / 9.0, 9.0 / 16.0, 1.0) for p, g in ((-1.0, 1.0), (-0.3, 2.0), (0.0, 2.0), (0.45, 1.0), (0.8, 2.0))]

    # placement: blobs_for / kick_blobs in a rectangular box, from scripted streams. `u`,
    # `g`, `s` are what the generator handed out, in order; `zp` / `zm` the two channel
    # lists, rows (x, y, sigma, amp[, ratio, theta]).
    place, seed = [], 4242
    for tag, aspect in (("16:9", 16.0 / 9.0), ("9:16", 9.0 / 16.0)):
        box = mt.box_for(aspect)
        for kind in mt.KICK_TYPES:
            for xc in (None, 0.31 * box[0]):
                n = 4
                seed += 1
                u, g = lcg_unit(8 * n, seed), [float(v) for v in 1.7 * lcg_noise(4 * n, seed + 1000)]
                sg = [1.0 if v >= 0.5 else -1.0 for v in lcg_unit(4 * n, seed + 2000)]
                rng = ScriptRng(u, g, sg)
                zp, zm = mt.kick_blobs(rng, kind, 0.3, n, 0.7, xc, 3.0, 2.0, box)
                place.append(dict(tag=tag, aspect=aspect, box=[float(v) for v in box], kind=kind, xc=xc, sigma=0.3,
                                  nblob=n, amp=0.7, ratio=3.0, sheetEnergy=2.0,
                                  used=[len(u) - len(rng.u), len(g) - len(rng.g), len(sg) - len(rng.s)],
                                  u=u, g=g, s=sg, zp=[[float(v) for v in r] for r in zp], zm=[[float(v) for v in r] for r in zm]))
    st["placement"] = place

    # the maintained current sheets (the bump profile): a = width * the narrowest blob,
    # psi0 from the peak field, psi_eq on the grid (every `stride`-th point, x = i Lx/nx),
    # the closed-form energy, and the energy and peak field of the sampled profile itself
    # (spectral derivative on the full grid) for a check that does not use the closed forms
    sheets = []
    for tag, aspect, nx, ny, strength, amp, width in (
            ("16:9 standard, the page's defaults", 16.0 / 9.0, 256, 256, 5.0, 1.0, 2.0),
            ("16:9 sharp, strength 8, width 1", 16.0 / 9.0, 512, 256, 8.0, 1.0, 1.0),
            ("9:16 light, strength 1.5, amp 0.7, width 4", 9.0 / 16.0, 128, 256, 1.5, 0.7, 4.0),
            ("square standard, strength 5", 1.0, 256, 256, 5.0, 2.0, 2.0),
            ("2.4:1 standard, strength 3, width 3", 2.4, 256, 256, 3.0, 1.0, 3.0)):
        Lx, Ly = mt.box_for(aspect)
        sig, _ = mt.band_plan((nx, ny), None, None, Lx, Ly)
        a = width * float(sig[-1])
        psi0 = float(mt.sheet_psi0(strength, amp, a))
        x = np.arange(nx) * (Lx / nx)
        psi = np.asarray(mt.sheet_equilibrium(x, Lx, psi0, a))
        assert psi.dtype == np.float64, psi.dtype
        dpsi = np.fft.irfft(1j * (2.0 * np.pi / Lx) * np.arange(nx // 2 + 1) * np.fft.rfft(psi), nx)
        stride = nx // 64
        sheets.append(dict(tag=tag, aspect=aspect, grid=[nx, ny], box=[float(Lx), float(Ly)], strength=strength, amp=amp,
                           width=width, a=a, psi0=psi0, stride=stride, psi=[float(v) for v in psi[::stride]],
                           energy=float(mt.sheet_energy_eq(psi0, a, Lx)),
                           energy_grid=float(0.5 * np.mean(dpsi ** 2)), peak_grid=float(np.max(np.abs(dpsi)))))
    st["sheets"] = sheets
    return st


def main():
    out = dict(source="examples/music_turbulence.py band_plan and CausalOnsetDetector")
    plans = []
    for n, nb, smin in [(256, None, None), (512, None, None), (1024, None, None), (2048, None, None),
                        (256, 3, None), (256, 6, None), (256, None, 0.2), (512, 4, 0.1), (256, 1, None),
                        (128, None, None), (256, 2, 0.45)]:
        try:
            sig, nbl = mt.band_plan(n, nb, smin)
            plans.append(dict(n=n, nbands=nb, sigmaMin=smin, sigmas=[float(v) for v in sig],
                              nblobs=[int(v) for v in nbl]))
        except ValueError as e:      # the offline NBLOB_MAX refusal: recorded as a refusal
            plans.append(dict(n=n, nbands=nb, sigmaMin=smin, error=str(e)))
    out["plans"] = plans
    out["nblob_max"] = mt.NBLOB_MAX

    pcm = test_signal()
    x = pcm.astype(np.float64) / 32768.0
    out["signal"] = dict(sr=SR, n=len(pcm), dtype="int16 little-endian, x = v / 32768",
                         b64=base64.b64encode(pcm.tobytes()).decode("ascii"))
    cases = []
    for tag, opts in CASES:
        full = dict(nfft=512)
        full.update(opts)
        kw = {KEYS[k]: v for k, v in full.items()}
        # (detect_onsets_causal itself has no floor argument: it stays what it was. The floor
        # cases drive the class directly, cut into frames exactly as that function cuts them.)
        if "flux_floor" in kw:
            det = mt.CausalOnsetDetector(SR, **kw)
            win, rows = np.blackman(kw["nfft"]), []
            for i in range(int(len(x) / SR * det.fps)):
                end = int(round((i + 1) / det.fps * SR))
                if end >= kw["nfft"]:
                    rows += det.push(np.abs(np.fft.rfft(win * x[end - kw["nfft"]:end])) / kw["nfft"], end / SR)
            rows.sort(key=lambda r: r[0])
            t, b, s, tf, edges = ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                                  [r[3] for r in rows], det.edges)
        else:
            t, b, s, edges, tf = mt.detect_onsets_causal(SR, x, **kw)
        cases.append(dict(tag=tag, opts=full, edges=[float(v) for v in edges],
                          onsets=[[float(tf[i]), int(b[i]), float(s[i]), float(t[i])] for i in range(len(t))]))
    out["cases"] = cases

    # MUSIC_EFFECTS_PLAN: the rules music.js ports that have a deterministic python side.
    # (The blob placements and the field walk draw from a different RNG on the page: those
    # are held to the rule's properties in the gate, not to recorded draws.)
    fx = dict(source="examples/music_turbulence.py kick_types_for, pulse_zoom, onset_pans, constants",
              kick_types=list(mt.KICK_TYPES), dipole_sep=mt.DIPOLE_SEP, sheet_ratio=mt.SHEET_RATIO,
              stereo_span=mt.STEREO_SPAN, stereo_jitter=mt.STEREO_JITTER,
              dipole_energy=[[sep, float(mt.dipole_energy_factor(sep))] for sep in (1.5, 1.0, 2.5)],
              click_pans=mt.CLICK_PANS)
    fx["types"] = [dict(spec=spec, nbands=nb, types=list(mt.kick_types_for(spec, nb)))
                   for spec in ("variety", "sheet", None) for nb in range(1, 8)]
    fx["types"].append(dict(spec="dipole,island,sheet-island", nbands=3,
                            types=list(mt.kick_types_for("dipole,island,sheet-island", 3))))
    # the camera pulse: onsets (time, strength) in arrival order, sampled at 60 frames a second
    t_on = [0.25, 0.5, 0.55, 1.0, 1.02, 1.6, 1.6]
    s_on = [1.0, 0.4, 0.9, 0.2, 1.0, 0.7, 0.3]
    tf = np.arange(150) / 60.0
    fx["pulse"] = [dict(a=a, tau=tau, t_on=t_on, s_on=s_on, fps=60, nframes=len(tf),
                        zoom=[float(v) for v in mt.pulse_zoom(tf, t_on, s_on, a, tau)])
                   for a, tau in ((0.08, 0.25), (0.15, 0.1))]
    # the pan: a band-limited tone at amplitude gains (1 - p, 1 + p), through onset_pans
    sr, pans = 11025, []
    tt = np.arange(int(0.2 * sr)) / sr
    tone = np.sin(2 * np.pi * 1000.0 * tt) * np.exp(-((tt - 0.1) / 0.02) ** 2)
    for p_amp in (-1.0, -0.6, 0.0, 0.3, 0.8, 1.0):
        lr = np.stack([(1 - p_amp) * tone, (1 + p_amp) * tone], axis=1)
        pans.append([p_amp, float(mt.onset_pans(sr, lr, np.array([0.1]), np.array([0]), [500.0, 2000.0])[0])])
    fx["pans"] = pans
    out["effects"] = fx
    out["stage"] = stage_rows()
    dst = os.path.join(HERE, "devtools", "fixtures", "music_ref.json")
    with open(dst, "w") as fh:
        json.dump(out, fh)
    print("wrote", dst, os.path.getsize(dst), "bytes;", len(plans), "plans;",
          ", ".join(str(len(c["onsets"])) for c in cases), "onsets")


if __name__ == "__main__":
    main()
