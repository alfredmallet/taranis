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
    dst = os.path.join(HERE, "devtools", "fixtures", "music_ref.json")
    with open(dst, "w") as fh:
        json.dump(out, fh)
    print("wrote", dst, os.path.getsize(dst), "bytes;", len(plans), "plans;",
          ", ".join(str(len(c["onsets"])) for c in cases), "onsets")


if __name__ == "__main__":
    main()
