# Music -> turbulence, offline: a piece of music drives 2D RMHD through IMPULSIVE kicks, and
# the movie is muxed with the audio so the two are exactly in sync.
#
#   python examples/music_turbulence.py                    # built-in click track, 20 s
#   python examples/music_turbulence.py track.wav          # any PCM WAV
#   python examples/music_turbulence.py track.wav --n 512 --field vorticity --out out.mp4
#
# The map (three pieces, nothing else):
#   1. an ONSET in a frequency band is one kick, its size set by the onset strength;
#   2. each kick places NEW gaussian blobs at random positions, so the pattern is re-made
#      on the beat;
#   3. the BAND sets the blob width sigma: bass -> wide blobs, treble -> narrow ones. The
#      widths run from the box scale down to a few grid spacings, and the number of bands
#      is how many distinguishable widths fit in between (band_plan).
# The z+ = phi + psi and z- = phi - psi channels get independent blobs.
#
# Variety (plans-webgpu/MUSIC_EFFECTS_PLAN.md), every switch OFF by default -- with none of
# them given the script does exactly what the three lines above say, bit for bit:
#   --kick-types variety        E1: each band gets a kick TYPE (KICK_TYPES; `variety` is
#                               bass dipole, low-mid vortex, high-mid island, treble sheet;
#                               or a list, one type per band, low to high)
#   --stereo [--stereo-gain 2]  E2: a band's blobs land where the band sits in the stereo
#                               image (x = left..right), the bass band excepted
#   --pulse [0.08]              E3: the picture zooms in by that fraction on each bass
#                               onset and relaxes back (display only; the flow is untouched)
#   --mean-field [0.5]          E4: a uniform in-plane magnetic field of that strength
#                               (x amp), along which z+ and z- stream in opposite
#                               directions; its direction is redrawn at each section
#                               change (SectionDetector, or --section-every N bass onsets)
#   --field-walk [--ratio-walk] instead: the direction (and |B0| / rms fluctuation) take a
#                               random-walk step at each bass onset
#   --zoom-out [20]             E5: the camera pulls back by a factor 2 every that many
#                               seconds; at each factor 2 the state is replaced by its own
#                               2x2 tiling (same velocities, half the size), so old
#                               structure shrinks toward the dissipation scale
#
# The stage (plans-webgpu/MUSIC_STAGE_PLAN.md), also OFF by default:
#   --aspect A [--grid NXxNY]   S2: the box takes the picture's shape. The SHORT side stays
#                               2 pi and the long side is 2 pi * max(A, 1/A): A > 1 is
#                               landscape (long in x, e.g. 1.7778 for 16:9), A < 1 portrait
#                               (long in y, e.g. 0.5625). --grid sets nx and ny apart (cells
#                               need not be square; default --n in both). See box_for and
#                               band_plan for what follows the box: blob widths, blob
#                               counts, placement, the hyperviscosity. The movie frame has
#                               the box's shape, with square pixels (frame_shape).
#   --sheets [1.5]              S4: a maintained current-sheet pair as the background, a
#     [--sheet-width 2]         function of x alone, its field that multiple of --amp and
#     [--sheet-profile bump]    a = that many narrowest-blob widths; restored against the
#                               hyperviscosity and the drag, while the kicks decay as
#                               before. Excludes --mean-field / --field-walk. Two profiles:
#                               `bump` (the default), psi0 sech^2((x - Lx/2)/a), whose
#                               field lives only within ~a of x = Lx/2 (sheet_equilibrium);
#                               `harris`, a periodic double Harris sheet -- the field is
#                               +-B0 everywhere and reverses at x = Lx/4 and 3Lx/4, so its
#                               energy is ~B0^2/2 whatever the width (sheet_harris).
#   --max-steps N               abort (RuntimeError) once the run has taken N steps
#   python examples/music_turbulence_check.py   is the gate for all of them (fp64).
#
# A kick is a state edit between steps, not a forcing term: `forcing` stays off and the
# solver is untouched. The kick amplitude is a velocity increment (cm/s in CGS; here Alfven
# units, the peak |grad z| of the blob), so it does not depend on dt.
#
# Timing: simulation time = RATE * audio time, with the solver's ordinary adaptive dt. Kicks
# and frames are events in t: the driver steps until t passes the next event, so each lands
# at most one step late (~ms of audio). When the flow is nearly at rest the adaptive dt
# would grow past a frame, so there the driver uses a second, fixed-dt Parameters with
# dt = RATE/(4 fps).
#
# Memory: a uniform linear drag, applied between events as fields *= exp(-dt/(2 MEMORY)),
# makes the energy forget on MEMORY seconds of audio. Without it 2D MHD decays as a power
# law, far slower than an eddy turnover (pure z+ or pure z- structures do not decay
# nonlinearly at all), the energy integrates over ~10 beats and the tempo is washed out.
#
# What decides whether the beat is visible: the kick has to be comparable to the ambient
# flow at the blob scale, i.e. the beat period has to be >~ the eddy turnover time there.
# MEMORY, RATE and AMP are the knobs; the printed "beat contrast" is the check.
#
# Dependencies: numpy, scipy, matplotlib (colormap only), imageio + imageio-ffmpeg.
import argparse
import os
import subprocess
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import propagators
from taranis.grids import fft, ifft
from taranis.physics import equation_registry, construct_rhs
from taranis.timestepping import get_scheme

LBOX = 2.0 * np.pi
FIELDS = ("vorticity", "current", "phi", "psi", "speed", "bperp")   # last two are magnitudes
NBLOB_MAX = 128         # blobs per kick per channel that the jitted kick is padded to


# ---------------------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------------------
def click_track(duration=20.0, bpm=120.0, sr=44100, seed=0, stereo=False):
    # kick on every beat, snare on beats 2 and 4, hi-hat on the off-beat eighths; the last
    # quarter of the track drops the kick so "what happens when the beat stops" is in it.
    # stereo=True returns (n, 2) samples [left, right] instead: the kick in the centre, the
    # snare left (CLICK_PANS), the hats alternating right and left. An instrument at
    # amplitude pan p has gains (1 - p, 1 + p), so the channel mean is the mono track.
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    out = np.zeros_like(t)
    lr = np.zeros((len(t), 2))
    beat = 60.0 / bpm

    def add(t0, sig, pan=0.0):
        i0 = int(t0 * sr)
        n = min(len(sig), len(out) - i0)
        if n > 0:
            out[i0:i0 + n] += sig[:n]
            lr[i0:i0 + n, 0] += (1.0 - pan) * sig[:n]
            lr[i0:i0 + n, 1] += (1.0 + pan) * sig[:n]

    tt = np.arange(int(0.4 * sr)) / sr
    kick = np.sin(2 * np.pi * (45.0 * tt + 60.0 * 0.05 * (1 - np.exp(-tt / 0.05)))) * np.exp(-tt / 0.12)
    snare = (0.6 * rng.standard_normal(len(tt)) * np.exp(-tt / 0.06)
             + 0.4 * np.sin(2 * np.pi * 220.0 * tt) * np.exp(-tt / 0.05))
    b_hp, a_hp = _highpass(6000.0, sr)
    from scipy.signal import lfilter
    hat = 0.35 * lfilter(b_hp, a_hp, rng.standard_normal(len(tt))) * np.exp(-tt / 0.02)
    snare = lfilter(*_bandpass(400.0, 3000.0, sr), snare)

    nbeats = int(duration / beat)
    for i in range(nbeats):
        if i < 0.75 * nbeats:
            add(i * beat, kick)
        if i % 4 in (1, 3):
            add(i * beat, snare, CLICK_PANS["snare"])
        add((i + 0.5) * beat, hat, CLICK_PANS["hat"] * (1 if i % 2 == 0 else -1))
    if stereo:
        return sr, 0.8 * lr / np.max(np.abs(out))
    return sr, 0.8 * out / np.max(np.abs(out))


CLICK_PANS = dict(kick=0.0, snare=-0.6, hat=0.8)     # amplitude pans of the stereo click track


def _highpass(fc, sr):
    from scipy.signal import butter
    return butter(4, fc / (0.5 * sr), "high")


def _bandpass(f0, f1, sr):
    from scipy.signal import butter
    return butter(2, [f0 / (0.5 * sr), f1 / (0.5 * sr)], "band")


def load_wav(path, stereo=False):
    # (sr, mono mix); with stereo=True (sr, mono mix, lr) where lr is the (n, 2) [left,
    # right] samples on the mono mix's normalisation, or None for a one-channel file.
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    lr = x[:, :2] if x.ndim == 2 and x.shape[1] >= 2 else None
    if x.ndim == 2:
        x = x.mean(axis=1)
    norm = max(np.max(np.abs(x)), 1e-30)
    if stereo:
        return sr, x / norm, None if lr is None else lr / norm
    return sr, x / norm


def is_mono(lr, tol=1e-4):
    # two channels that carry the same signal: every pan would be 0 and every blob would
    # land in the centre column, so stereo placement is not used on such a file
    return lr is None or lr.ndim != 2 or np.abs(lr[:, 0] - lr[:, 1]).max() <= tol * np.abs(lr).max()


def onset_pans(sr, lr, t_on, b_on, edges, win_s=0.046):
    """Stereo position of each onset: pan = (E_R - E_L)/(E_R + E_L) in [-1, 1], E the
    energy (sum of |X|^2 over the onset's band) of each channel in ONE Hann-windowed
    analysis frame of `win_s` centred on the onset time. -1 is hard left, +1 hard right.

    The detectors are not involved: they run on the mono mix and hand over (time, band).
    An instrument alone in its band at amplitude gains (1 - p, 1 + p) reads 2p/(1 + p^2).
    """
    nper = int(win_s * sr)
    win = np.hanning(nper)
    freqs = np.fft.rfftfreq(nper, 1.0 / sr)
    pans = np.zeros(len(t_on))
    for i, (t, b) in enumerate(zip(t_on, b_on)):
        i0 = min(max(int(round(t * sr)) - nper // 2, 0), max(len(lr) - nper, 0))
        seg = lr[i0:i0 + nper]
        X = np.abs(np.fft.rfft(win[:len(seg), None] * seg, n=nper, axis=0)) ** 2
        el, er = X[(freqs >= edges[b]) & (freqs < edges[b + 1])].sum(axis=0)
        pans[i] = (er - el) / (er + el) if er + el > 0.0 else 0.0
    return pans


def detect_onsets(sr, x, nbands=3, fmin=40.0, fmax=10000.0, hop_s=0.010, win_s=0.046,
                  threshold=0.15, min_gap_s=0.07):
    """Per-band spectral-flux onsets: returns (times [s], band index, strength in (0,1]).

    Band b covers the b-th of `nbands` log-spaced intervals of [fmin, fmax]. The flux is the
    positive time difference of the log-compressed magnitude, summed over the band and
    normalised by its 99th percentile; an onset is a local maximum that clears a moving
    mean by `threshold`, at most one per `min_gap_s` per band.
    """
    nper = int(win_s * sr)
    hop = int(hop_s * sr)
    f, t, X = stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    t = t  # window centres
    mag = np.log1p(100.0 * np.abs(X) / (np.abs(X).max() + 1e-30))
    flux = np.maximum(np.diff(mag, axis=1, prepend=mag[:, :1]), 0.0)
    edges = np.geomspace(fmin, min(fmax, 0.5 * sr), nbands + 1)
    times, bands, strengths = [], [], []
    w = max(1, int(0.10 / hop_s))          # moving-mean half width
    g = max(1, int(min_gap_s / hop_s))     # local-max half width
    for b in range(nbands):
        sel = (f >= edges[b]) & (f < edges[b + 1])
        s = flux[sel].sum(axis=0)
        s = s / (np.percentile(s, 99.0) + 1e-30)
        mean = np.convolve(s, np.ones(2 * w + 1) / (2 * w + 1), mode="same")
        for i in range(len(s)):
            lo, hi = max(0, i - g), min(len(s), i + g + 1)
            if s[i] == s[lo:hi].max() and s[i] > mean[i] + threshold:
                times.append(t[i]); bands.append(b); strengths.append(min(s[i], 1.0))
    order = np.argsort(times)
    return (np.array(times)[order], np.array(bands, dtype=int)[order],
            np.array(strengths)[order], edges)


class CausalOnsetDetector:
    """The per-frame core of `detect_onsets_causal`, fed MAGNITUDE SPECTRA one at a time.

    This is the object a live page runs (plans-webgpu/MUSIC_PLAN.md, M2): `webgpu/music.js`'s
    `MusicDetector` is a line-for-line port, and `webgpu/gen_musicfixture.py` records this
    class's output as the fixture `webgpu/devtools/checkmusic.js` holds the port to. Keep the
    two in step -- change one, change the other, regenerate the fixture.

    `push(mag, t_now)` takes the magnitude spectrum of the last `nfft` samples under a
    Blackman window, scaled by 1/nfft (what a Web Audio `AnalyserNode` with
    smoothingTimeConstant 0 holds: 10^(dB/20) of `getFloatFrequencyData`), and the time the
    window ENDS. It returns the onsets it can now declare, as (t_centre, band, strength,
    t_fire): the peak was one frame ago, `t_centre` is that frame's analysis-window centre
    and `t_fire` is `t_now`, when the caller knows.

    All windows are counted in FRAMES at the nominal `fps`, so the caller must push at about
    that rate. A gap longer than `gap_s` (a hidden tab, a seek) only re-primes the previous
    spectrum: the flux across a gap is the music having moved on, not an onset.

    `flux_floor` (default 0 = off, and then nothing here differs from before it existed) is
    the noise floor the live page turns on: a band's normaliser is held at no less than
    `flux_floor` times a peak-hold of that band's own flux decaying over `peak_decay_s`, so
    flux far below what the band recently carried is not an onset. Its failure mode is the
    other side of that: after a loud passage, onsets weaker than `flux_floor` of it are
    missed until the peak-hold has decayed; and a source that is noise from its first
    sample (a microphone in a quiet room) has no louder past to be measured against.
    """

    def __init__(self, sr, nbands=3, fmin=40.0, fmax=10000.0, fps=60.0, nfft=2048,
                 threshold=0.15, min_gap_s=0.07, mean_s=0.20, ref_decay_s=30.0,
                 norm_s=20.0, warmup_s=0.5, gap_s=0.1, flux_floor=0.0, peak_decay_s=30.0):
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        self.sr, self.nfft, self.nbands, self.fps = sr, nfft, nbands, fps
        self.edges = np.geomspace(fmin, min(fmax, 0.5 * sr), nbands + 1)
        self.sels = [(freqs >= self.edges[b]) & (freqs < self.edges[b + 1]) for b in range(nbands)]
        self.threshold, self.min_gap_s, self.warmup_s, self.gap_s = threshold, min_gap_s, warmup_s, gap_s
        self.nmean = max(1, int(round(mean_s * fps)))
        self.nnorm = max(1, int(round(norm_s * fps)))
        self.decay = np.exp(-1.0 / (ref_decay_s * fps))
        self.ref, self.prev, self.t_last = 0.0, None, None
        self.hist = [[] for _ in range(nbands)]          # raw band flux, all frames so far
        self.last_fire = [-np.inf] * nbands
        # the noise floor (OFF at flux_floor = 0, which leaves every number below as it was):
        # each band keeps a slow peak-hold of its own raw flux, and its normaliser is never
        # allowed below flux_floor times that peak. Without it the running 99th percentile
        # promotes ANY non-zero floor to "full strength" once the music stops.
        self.flux_floor = flux_floor
        self.pdecay = np.exp(-1.0 / (peak_decay_s * fps))
        self.peak = [0.0] * nbands

    def push(self, mag, t_now):
        fps = self.fps
        self.ref = max(self.ref * self.decay, float(mag.max()), 1e-30)
        comp = np.log1p(100.0 * mag / self.ref)
        gap = self.t_last is not None and t_now - self.t_last > self.gap_s
        self.t_last = t_now
        if self.prev is None or gap:
            self.prev = comp
            return []
        flux = np.maximum(comp - self.prev, 0.0)
        self.prev = comp
        out = []
        for b in range(self.nbands):
            h = self.hist[b]
            h.append(float(flux[self.sels[b]].sum()))
            if self.flux_floor > 0.0:
                self.peak[b] = max(self.peak[b] * self.pdecay, h[-1])
            if len(h) < 3 or t_now < self.warmup_s:
                continue
            q = np.percentile(h[-self.nnorm:], 99.0) + 1e-30
            if self.flux_floor > 0.0:
                q = max(q, self.flux_floor * self.peak[b])
            s0, s1, s2 = h[-3] / q, h[-2] / q, h[-1] / q      # s1 is the candidate peak
            mean = np.mean(h[-1 - self.nmean:-1]) / q         # trailing, ends at the candidate
            t_peak = t_now - 1.0 / fps
            if (s1 > s0 and s1 >= s2 and s1 > mean + self.threshold
                    and t_peak - self.last_fire[b] >= self.min_gap_s):
                self.last_fire[b] = t_peak
                out.append((t_peak - 0.5 * self.nfft / self.sr, b, min(s1, 1.0), t_now))
        return out


def detect_onsets_causal(sr, x, nbands=3, fmin=40.0, fmax=10000.0, fps=60.0, nfft=2048,
                         threshold=0.15, min_gap_s=0.07, mean_s=0.20, ref_decay_s=30.0,
                         norm_s=20.0, warmup_s=0.5):
    """The onset detector a LIVE page can run (plans-webgpu/MUSIC_PLAN.md, M0.b): same map
    as `detect_onsets`, nothing that looks at the future or at the whole track.

    Modelled on what a browser hands over: once per display frame (`fps`) the spectrum of
    the last `nfft` samples under a Blackman window -- a Web Audio `AnalyserNode` with
    smoothingTimeConstant 0. Against the offline detector:
      * the compression reference is a peak-hold of the spectral maximum decaying over
        `ref_decay_s`, where offline uses the whole-track maximum;
      * each band's flux is normalised by its 99th percentile over the trailing `norm_s`
        seconds, where offline uses the whole track;
      * the moving mean is trailing (`mean_s`), where offline centres it;
      * a peak is declared ONE frame after it happens (s[i-1] > s[i-2], s[i-1] >= s[i]),
        where offline looks `min_gap_s` both ways; `min_gap_s` is a refractory time here.
    The per-frame logic is `CausalOnsetDetector`; this function only cuts the signal into
    the frames a browser would see and feeds them in.

    Returns (times, bands, strengths, edges, fire_times): `times` are the analysis-window
    centres of the peak frames (comparable with `detect_onsets`), `fire_times` when the
    page would KNOW -- half a window plus one frame later.
    """
    det = CausalOnsetDetector(sr, nbands=nbands, fmin=fmin, fmax=fmax, fps=fps, nfft=nfft,
                              threshold=threshold, min_gap_s=min_gap_s, mean_s=mean_s,
                              ref_decay_s=ref_decay_s, norm_s=norm_s, warmup_s=warmup_s)
    win = np.blackman(nfft)
    times, bands, strengths, fires = [], [], [], []
    for i in range(int(len(x) / sr * fps)):
        end = int(round((i + 1) / fps * sr))    # the window ENDS now
        if end < nfft:
            continue
        mag = np.abs(np.fft.rfft(win * x[end - nfft:end])) / nfft
        for t, b, s, tf in det.push(mag, end / sr):
            times.append(t); bands.append(b); strengths.append(s); fires.append(tf)
    order = np.argsort(times, kind="stable")
    return (np.array(times)[order], np.array(bands, dtype=int)[order],
            np.array(strengths)[order], det.edges, np.array(fires)[order])


class SectionDetector:
    """The SLOW trigger (plans-webgpu/MUSIC_EFFECTS_PLAN.md): fires when the music changes
    character -- a section change, a few times a minute at most -- causally, from the same
    per-frame magnitude spectra `CausalOnsetDetector` is fed (see its docstring for what
    `mag` and `t_now` are, and for the gap rule, which is the same here).

      * per band, log10 of the band energy relative to a peak-hold of the total energy
        (decaying over `ref_decay_s`), floored at `floor_db`, then smoothed with a
        `smooth_s` exponential;
      * novelty = rms over the bands of (its mean over the last `win_s` seconds minus its
        mean over the `win_s` before that), in decades;
      * normalised by the `pct`-th percentile of the novelty over the trailing `norm_s`
        seconds, but never by less than `min_novelty` -- the absolute floor that keeps a
        steady passage from firing on its own fluctuations;
      * a trigger is a local maximum of the normalised novelty above `threshold`, declared
        one frame late, at most one per `min_gap_s`.

    `push` returns [] or [(t_now, strength)], strength being the normalised novelty. A
    change is seen about `win_s`/2 + `smooth_s` after it happens; nothing fires in the
    first 2 `win_s`.
    """

    def __init__(self, sr, nbands=4, fmin=40.0, fmax=10000.0, fps=60.0, nfft=2048,
                 smooth_s=2.0, win_s=4.0, norm_s=60.0, pct=75.0, threshold=1.5,
                 min_novelty=0.1, min_gap_s=8.0, ref_decay_s=30.0, floor_db=-60.0, gap_s=0.1):
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        edges = np.geomspace(fmin, min(fmax, 0.5 * sr), nbands + 1)
        self.sels = [(freqs >= edges[b]) & (freqs < edges[b + 1]) for b in range(nbands)]
        self.fps, self.gap_s, self.threshold = fps, gap_s, threshold
        self.min_novelty, self.min_gap_s, self.pct = min_novelty, min_gap_s, pct
        self.alpha = 1.0 - np.exp(-1.0 / (smooth_s * fps))
        self.nw = max(1, int(round(win_s * fps)))
        self.nnorm = max(1, int(round(norm_s * fps)))
        self.decay = np.exp(-1.0 / (ref_decay_s * fps))
        self.floor = 10.0 ** (floor_db / 10.0)
        self.ref, self.smooth, self.t_last = 0.0, None, None
        self.hist, self.nov, self.last_fire = [], [], -np.inf

    def push(self, mag, t_now):
        gap = self.t_last is not None and t_now - self.t_last > self.gap_s
        self.t_last = t_now
        if gap:                       # the music moved on while nobody was looking
            self.hist, self.nov, self.smooth = [], [], None
        e = np.array([float(np.sum(mag[sel] ** 2)) for sel in self.sels])
        self.ref = max(self.ref * self.decay, float(e.sum()), 1e-300)
        lev = np.log10(np.maximum(e / self.ref, self.floor))
        self.smooth = lev if self.smooth is None else self.smooth + self.alpha * (lev - self.smooth)
        self.hist.append(self.smooth)
        del self.hist[:-2 * self.nw]
        if len(self.hist) < 2 * self.nw:
            return []
        h = np.array(self.hist)
        d = float(np.sqrt(np.mean((h[self.nw:].mean(axis=0) - h[:self.nw].mean(axis=0)) ** 2)))
        self.nov.append(d)
        del self.nov[:-self.nnorm]
        if len(self.nov) < 3:
            return []
        q = max(float(np.percentile(self.nov, self.pct)), self.min_novelty)
        s0, s1, s2 = self.nov[-3] / q, self.nov[-2] / q, self.nov[-1] / q
        t_peak = t_now - 1.0 / self.fps
        if s1 > s0 and s1 >= s2 and s1 > self.threshold and t_peak - self.last_fire >= self.min_gap_s:
            self.last_fire = t_peak
            return [(t_now, s1)]
        return []


def detect_sections(sr, x, nbands=4, fps=60.0, nfft=2048, **kw):
    """Run `SectionDetector` over a whole signal the way a page would see it (one Blackman
    frame per display frame, as `detect_onsets_causal`). Returns (fire times [s], strengths,
    the novelty trace per frame as (t, novelty in decades))."""
    det = SectionDetector(sr, nbands=nbands, fps=fps, nfft=nfft, **kw)
    win = np.blackman(nfft)
    times, strengths, trace = [], [], []
    for i in range(int(len(x) / sr * fps)):
        end = int(round((i + 1) / fps * sr))
        if end < nfft:
            continue
        mag = np.abs(np.fft.rfft(win * x[end - nfft:end])) / nfft
        for t, s in det.push(mag, end / sr):
            times.append(t); strengths.append(s)
        if det.nov:
            trace.append((end / sr, det.nov[-1]))
    return np.array(times), np.array(strengths), np.array(trace).reshape(-1, 2)


# ---------------------------------------------------------------------------------------
# solver pieces
# ---------------------------------------------------------------------------------------
class Stepper:
    """Step until t >= target (one jitted while_loop, the same construction as
    `taranis.run.simulate`'s). Adaptive dt normally; a fixed dt_cap when the flow is so slow
    that the adaptive dt would exceed it (each mode has its own Parameters and kgrid -- a
    kgrid is never reused across Parameters)."""

    def __init__(self, n, dt_cap, diss, hyper, cfl=0.4, b0=None, Lx=LBOX, Ly=LBOX, source=False):
        # n: the grid, one number (n x n) or (nx, ny); Lx, Ly: the box.
        # source=True: the psi equation gets a constant source, set by set_equilibrium()
        # before the first advance (the maintained current sheets; not with a mean field)
        # b0: None, (b0_of_t, b0_max) -- the in-plane mean field, see mean_field() -- or
        # "driven": (B0x, B0y) is then an argument of each advance() call, held over it,
        # and the field is in the EXACT linear step, so it costs no timestep
        # ("driven-explicit": the same in the RHS and the CFL sum, for comparison)
        self.driven = isinstance(b0, str)
        if self.driven and b0 not in ("driven", "driven-explicit"):
            raise ValueError(f"b0 must be None, (b0_of_t, b0_max), 'driven' or 'driven-explicit', got {b0!r}")
        self.exact = b0 == "driven"
        self.b0_max = 0.0 if b0 is None or self.driven else float(b0[1])
        if source and b0 is not None:
            raise ValueError("the maintained equilibrium (source=True) does not go with a mean field")
        nx, ny = (n, n) if np.ndim(n) == 0 else n
        kw = dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=cfl, dims=2, comm_backend="serial",
                  eqpars={"diss": (diss, diss), "hyper": hyper})
        self.p = jr.Parameters(adaptive_timestep=True, **kw)
        self.kg = jr.setup_kgrids(self.p)
        p_fix = jr.Parameters(adaptive_timestep=False, dt=dt_cap, **kw)
        self.source, self.src, self.psi_eq = bool(source), None, None
        self._adaptive = self._build(self.p, self.kg, b0, self.source)
        self._fixed = self._build(p_fix, jr.setup_kgrids(p_fix), b0, self.source)
        # the smaller spacing is the one the fixed-dt switch in advance() has to respect
        self.dt_cap, self.cfl, self.dx = dt_cap, cfl, min(Lx / nx, Ly / ny)

    def set_equilibrium(self, psi_eq):
        """The maintained equilibrium: `psi_eq` (nx,) complex, the k_y = 0 column of psi's
        rfft2 (so psi_eq is a function of x alone), taken ONCE -- from the dealiased initial
        state. The psi equation then carries the constant source -L psi_eq, L the solver's
        own linear diagonal for psi (-nu k^(2 hyper)), which cancels the hyperviscous decay
        of the equilibrium exactly and leaves everything else to decay as before. The drag
        is not in the solver's L (it is the split step in run()), so its half of the source
        is applied there, in the same split step: see drag()."""
        if not self.source:
            raise ValueError("set_equilibrium needs Stepper(source=True)")
        psi_eq = jnp.asarray(psi_eq)
        lpsi = self.kg.lin.dense()[1, 1]                    # (1, nx, nky)
        src = jnp.zeros((2,) + lpsi.shape, dtype=psi_eq.dtype)
        self.src = src.at[1, 0, :, 0].set(-lpsi[0, :, 0] * psi_eq)
        self.psi_eq = psi_eq

    @staticmethod
    def _build(params, kgrid, b0=None, source=False):
        recipe = equation_registry[params.eqtype]
        rhs, set_timestep = construct_rhs(recipe), recipe.set_timestep_func
        stepper, scheme = get_scheme("lsrk33")
        if source:
            # the maintained equilibrium: a constant added to the nonlinear term, `src`
            # (the fields' shape, zero outside psi's k_y = 0 column) an argument of the call
            rhs0 = rhs

            def to_target(state, target, src):
                def rhs(state, kgrid, params):
                    f, grads = rhs0(state, kgrid, params)
                    return f + src.astype(f.dtype), grads

                def body(c):
                    return stepper(c[0], kgrid, params, rhs, set_timestep, scheme), c[1] + 1
                return jax.lax.while_loop(lambda c: c[0].t < target, body, (state, 0))
            return jax.jit(to_target, donate_argnums=(0,))
        if b0 == "driven":
            # the linear pair below, d_t (phi, psi) = i k.B0 (psi, phi), as the off-diagonal
            # of the solver's own L: the integrating-factor step then carries the field's
            # waves exactly at any dt (the dense 2x2 "putzer2" operator, built the way
            # setup_kgrids builds it, from the traced B0 of this call)
            ftype = kgrid.kx.dtype

            def to_target(state, target, bvec):
                kb = 1j * (kgrid.kx * bvec[0].astype(ftype) + kgrid.ky * bvec[1].astype(ftype))[None]
                d = kgrid.lin.dense()
                L = jnp.stack([jnp.stack([d[0, 0], kb + 0 * d[0, 0]]), jnp.stack([kb + 0 * d[0, 0], d[1, 1]])])
                kg = kgrid._replace(lin=propagators.build(L, params))

                def body(c):
                    return stepper(c[0], kg, params, rhs, set_timestep, scheme), c[1] + 1
                return jax.lax.while_loop(lambda c: c[0].t < target, body, (state, 0))
            return jax.jit(to_target, donate_argnums=(0,))
        if isinstance(b0, str):
            # the same linear pair as below, with (B0x, B0y) a traced argument of the call
            rhs0, set_timestep0 = rhs, set_timestep
            ftype = kgrid.kx.dtype

            def to_target(state, target, bvec):
                bx, by = bvec[0].astype(ftype), bvec[1].astype(ftype)
                kb, bmag = kgrid.kx * bx + kgrid.ky * by, jnp.sqrt(bx * bx + by * by)

                def rhs(state, kgrid, params):
                    f, grads = rhs0(state, kgrid, params)
                    return f + 1j * kb * state.fields[::-1], grads

                def set_timestep(grads, params):
                    dt = set_timestep0(grads, params)
                    return 1.0 / (1.0 / dt + bmag * (1.0 / params.dx + 1.0 / params.dy) / params.cfl_safety)

                def body(c):
                    return stepper(c[0], kgrid, params, rhs, set_timestep, scheme), c[1] + 1
                return jax.lax.while_loop(lambda c: c[0].t < target, body, (state, 0))
            return jax.jit(to_target, donate_argnums=(0,))
        if b0 is not None:
            # A uniform in-plane field B0 = zhat x grad(psi0) is psi -> psi + psi0 in the
            # brackets (d_t lap phi = {psi, j} - {phi, lap phi}, d_t psi = -{phi, psi},
            # {a, b} = a_x b_y - a_y b_x), which adds the k-local linear pair
            #     d_t phi = (B0.grad) psi,   d_t psi = (B0.grad) phi
            # i.e. d_t z+- = +-(B0.grad) z+-: z+ = phi + psi streams along -B0 and z-
            # along +B0, at |B0|. It is added to the solver's RHS here (explicit), B0 a
            # function of the traced state.t, and its rate joins the CFL sum.
            b0_of_t, b0_max = b0
            rhs0, set_timestep0 = rhs, set_timestep
            ftype = kgrid.kx.dtype

            def rhs(state, kgrid, params):
                f, grads = rhs0(state, kgrid, params)
                bx, by = b0_of_t(state.t)
                kb = kgrid.kx * bx.astype(ftype) + kgrid.ky * by.astype(ftype)
                return f + 1j * kb * state.fields[::-1], grads

            def set_timestep(grads, params):
                dt = set_timestep0(grads, params)
                return 1.0 / (1.0 / dt + b0_max * (1.0 / params.dx + 1.0 / params.dy) / params.cfl_safety)

        def to_target(state, target):
            def body(c):
                return stepper(c[0], kgrid, params, rhs, set_timestep, scheme), c[1] + 1
            return jax.lax.while_loop(lambda c: c[0].t < target, body, (state, 0))
        return jax.jit(to_target, donate_argnums=(0,))

    def advance(self, state, target, umax, bvec=None):
        # umax is the speed at entry; returns (state, steps taken). Consumes `state`.
        # (the mean field counts as speed: the fixed dt_cap must also resolve its waves)
        # bvec = (B0x, B0y): required with b0 = "driven", and only then
        if self.driven != (bvec is not None):
            raise ValueError("bvec goes with b0 = 'driven', and only with it")
        b0 = float(np.hypot(*bvec)) if self.driven and not self.exact else self.b0_max
        slow = 0.5 * self.cfl * self.dx / max(umax + 2.0 * b0, 1e-30) > self.dt_cap
        args = (jnp.asarray(bvec, float),) if self.driven else ()
        if self.source:
            if self.src is None:
                raise ValueError("Stepper(source=True): call set_equilibrium() before advance()")
            args = (self.src,)
        state, nstep = (self._fixed if slow else self._adaptive)(state, target, *args)
        return state, int(nstep)


def mean_field(b0, theta0, t_turn, dtheta, turn_s=2.0):
    """(b0_of_t, b0) for `Stepper`: a field of fixed magnitude `b0` whose direction starts
    at angle `theta0` from x and, at each time t_turn[i], turns by dtheta[i] over `turn_s`
    (a smoothstep). b0_of_t(t) -> (B0x, B0y) is jax-traceable in t. Times in SIMULATION
    units."""
    tt = jnp.asarray(np.concatenate([[0.0], np.asarray(t_turn, float)]))
    dd = jnp.asarray(np.concatenate([[0.0], np.asarray(dtheta, float)]))

    def b0_of_t(t):
        u = jnp.clip((t - tt) / turn_s, 0.0, 1.0)
        th = theta0 + jnp.sum(dd * u * u * (3.0 - 2.0 * u))
        return b0 * jnp.cos(th), b0 * jnp.sin(th)
    return b0_of_t, float(b0)


def regrid(fields, dealias=None):
    """The zoom-out re-grid (E5): `fields` (2, 1, n, n//2 + 1), the rfft2 of (phi, psi) on
    an n x n grid of a periodic box L, replaced by the transform of

        g(x) = f(2 x - x_c) / 2,      x_c = (n - 1)/2 grid spacings in x and in y

    -- the box's own 2x2 tiling seen from twice as far, about the display's centre pixel
    (zoom_image's centre, so a picture at zoom 1/2 before is the picture at zoom 1 after).
    Mode k goes to 2k with the phase exp(-i k.x_c). The 1/2 keeps the VELOCITIES grad f
    what they were: the same flow at half the size, so the energy of every mode that
    survives is unchanged. Modes that land outside the retained range are dropped: those
    with |2 k_x| or 2 k_y at or past n/2, and then `dealias` (the solver's mask) if given.
    Returns a numpy array of the same shape and dtype."""
    f = np.asarray(fields)
    if f.shape[3] != f.shape[2] // 2 + 1:
        return _regrid_rect(f, dealias)
    n = f.shape[2]
    out = np.zeros_like(f)
    ix = np.fft.fftfreq(n, 1.0 / n).astype(int)          # the integer k_x of each row
    keep_i = np.nonzero(2 * np.abs(ix) < n // 2)[0]
    j = np.arange((n // 2 - 1) // 2 + 1)                    # 2 j <= n/2 - 1
    xc = 0.5 * (n - 1) * (2.0 * np.pi / n)                # in units where k is the integer
    ph = np.exp(-1j * xc * (ix[keep_i][:, None] + j[None, :]))
    out[:, :, ((2 * ix[keep_i]) % n)[:, None], (2 * j)[None, :]] = 0.5 * ph * f[:, :, keep_i][..., j]
    if dealias is not None:
        out = out * np.asarray(dealias)
    return out.astype(f.dtype)


def _regrid_rect(f, dealias=None):
    # regrid on an nx x ny grid with nx != ny: the same map, each axis with its own n (the
    # box lengths do not enter: k -> 2k and x_c = (n - 1)/2 spacings are in index units)
    nx, ny = f.shape[2], 2 * (f.shape[3] - 1)
    out = np.zeros_like(f)
    ix = np.fft.fftfreq(nx, 1.0 / nx).astype(int)
    keep_i = np.nonzero(2 * np.abs(ix) < nx // 2)[0]
    j = np.arange((ny // 2 - 1) // 2 + 1)
    ph = np.exp(-1j * (0.5 * (nx - 1) * (2.0 * np.pi / nx) * ix[keep_i][:, None]
                       + 0.5 * (ny - 1) * (2.0 * np.pi / ny) * j[None, :]))
    out[:, :, ((2 * ix[keep_i]) % nx)[:, None], (2 * j)[None, :]] = 0.5 * ph * f[:, :, keep_i][..., j]
    if dealias is not None:
        out = out * np.asarray(dealias)
    return out.astype(f.dtype)


def box_for(aspect=1.0):
    """(Lx, Ly) of the box for a picture of that aspect (width over height): the SHORT side
    is 2 pi and the long side 2 pi * max(aspect, 1/aspect). aspect >= 1 is landscape (long
    in x), aspect < 1 portrait (long in y). aspect 1 is exactly the square 2 pi box."""
    if not aspect > 0.0:
        raise ValueError(f"aspect must be positive, got {aspect}")
    return (LBOX * aspect, LBOX) if aspect >= 1.0 else (LBOX, LBOX / aspect)


def cutoff_k(nx, ny, Lx=LBOX, Ly=LBOX):
    """The smaller of the two dealiasing cutoffs, in k: the solver's 2/3 mask is in mode
    INDEX (n/3 along each axis), so in a rectangular box or grid the two axes stop at
    different wavenumbers, and the coarser one is the one that has to be resolved."""
    return min((nx / 3.0) * (2.0 * np.pi / Lx), (ny / 3.0) * (2.0 * np.pi / Ly))


def frame_shape(nx, ny, Lx=LBOX, Ly=LBOX):
    """(width, height) in pixels of the movie frame: the box's shape with SQUARE pixels, at
    the pixel size of the finer of the two grid spacings, each side rounded to an even
    number (the codec needs that). (nx, ny) itself whenever the cells are square."""
    if Lx * ny == Ly * nx:
        return nx, ny
    px = min(Lx / nx, Ly / ny)
    return 2 * int(round(0.5 * Lx / px)), 2 * int(round(0.5 * Ly / px))


def resample_frame(u, shape):
    """`u` (rows, columns), periodic, resampled bilinearly to shape = (rows, columns): the
    whole box on the new pixels, pixel centres at (i + 1/2)/n of the box before and after.
    Returns u itself when the shape is already that."""
    if u.shape == tuple(shape):
        return u
    from scipy.ndimage import map_coordinates
    i0 = (np.arange(shape[0]) + 0.5) * u.shape[0] / shape[0] - 0.5
    i1 = (np.arange(shape[1]) + 0.5) * u.shape[1] / shape[1] - 0.5
    return map_coordinates(u, np.meshgrid(i0, i1, indexing="ij"), order=1, mode="grid-wrap")


# the maintained current sheets (S4)
def sheet_psi0(strength, amp, a):
    """psi0 of psi_eq = psi0 sech^2((x - Lx/2)/a) whose peak field is strength * amp.
    B = zhat x grad(psi), so |B| = |d psi_eq/dx| = (2 psi0/a) sech^2(u) |tanh(u)|, which is
    largest where tanh^2(u) = 1/3: (2 psi0/a) (2/3) / sqrt(3) = 4 psi0 / (3 sqrt(3) a).
    Hence psi0 = (3 sqrt(3)/4) a strength amp."""
    return 0.75 * np.sqrt(3.0) * a * strength * amp


def sheet_equilibrium(x, Lx, psi0, a):
    """psi_eq(x) = psi0 sech^2((x - Lx/2)/a), summed over the nearest periodic images so it
    is smooth across the box edge (the images are below 1e-6 psi0 for a < Lx/16): one
    current sheet at x = Lx/2 (the field reverses across it) with weaker return currents of
    the opposite sign either side, no net flux. numpy or jax."""
    return psi0 * sum(1.0 / jnp.cosh((x - 0.5 * Lx + m * Lx) / a) ** 2 for m in (-1, 0, 1))


def log_cosh(u):
    """ln cosh(u) without the overflow: cosh u = e^|u| (1 + e^-2|u|)/2. numpy or jax."""
    z = jnp.abs(u)
    return z + jnp.log1p(jnp.exp(-2.0 * z)) - np.log(2.0)


def harris_slope(a, Lx):
    """The mean of tanh((x - Lx/4)/a) - tanh((x - 3Lx/4)/a) over the box,
    (2a/Lx) [ln cosh(3Lx/4a) - ln cosh(Lx/4a)]: 1 - O(e^(-Lx/2a)). It is what sheet_harris
    subtracts in place of the 1 of the ideal profile, so that psi_eq is exactly periodic."""
    lc = lambda u: u + np.log1p(np.exp(-2.0 * u))           # ln cosh + ln 2, u > 0 (plain numpy:
    return float(2.0 * a / Lx * (lc(0.75 * Lx / a) - lc(0.25 * Lx / a)))   # this is called under a trace)


def sheet_harris(x, Lx, b0, a):
    """The periodic double Harris sheet. With B = zhat x grad(psi), i.e. B_y = +d psi/dx,

        B_y(x)    = b0 [tanh((x - Lx/4)/a) - tanh((x - 3Lx/4)/a) - c]
        psi_eq(x) = b0 a [ln cosh((x - Lx/4)/a) - ln cosh((x - 3Lx/4)/a)] - b0 c (x - Lx/2)

    c = harris_slope(a, Lx) = 1 up to e^(-Lx/2a): the field is -b0 outside the two sheets
    and +b0 between them, reversing over ~a at x = Lx/4 and 3Lx/4 (currents of opposite
    sign), with no net flux. c instead of 1 makes psi_eq(0) = psi_eq(Lx) exactly; B_y is
    periodic by symmetry. b0 is the far field AND the peak field. numpy or jax."""
    c = harris_slope(a, Lx)
    return b0 * a * (log_cosh((x - 0.25 * Lx) / a) - log_cosh((x - 0.75 * Lx) / a)) - b0 * c * (x - 0.5 * Lx)


def harris_energy_eq(b0, a, Lx):
    """Volume-averaged energy (1/2)<B_y^2> of sheet_harris: B_y^2 = b0^2 except within the
    two sheets, each of which removes b0^2 int (1 - tanh^2) dx = 2 a b0^2, so
    (b0^2/2)(1 - 4a/Lx) -- up to e^(-Lx/2a), the sheets' overlap."""
    return 0.5 * b0 ** 2 * (1.0 - 4.0 * a / Lx)


SHEET_PROFILES = ("bump", "harris")


def sheet_energy_eq(psi0, a, Lx):
    """Volume-averaged energy (1/2)<|grad psi_eq|^2> of the equilibrium:
    (1/2)(2 psi0/a)^2 (a/Lx) int sech^4 tanh^2 du, the integral being 4/15."""
    return 8.0 * psi0 ** 2 / (15.0 * a * Lx)


def drag(fields, decay, psi_eq=None):
    """The drag over one interval: fields * decay, decay = exp(-dt/(2 memory)). With a
    maintained equilibrium `psi_eq` (see Stepper.set_equilibrium) the drag's half of the
    source -L psi_eq is integrated exactly with it: psi's k_y = 0 column relaxes TO psi_eq
    instead of to zero, psi -> psi_eq + (psi - psi_eq) decay. Everything else, kicks
    included, decays as without it."""
    fields = fields * decay
    if psi_eq is not None:
        fields = fields.at[1, 0, :, 0].add(((1.0 - decay) * psi_eq).astype(fields.dtype))
    return fields


def make_kick(params, kgrid):
    Lx, Ly = float(params.Lx), float(params.Ly)
    x = jnp.linspace(0.0, Lx, params.nx, endpoint=False).reshape(-1, 1)
    y = jnp.linspace(0.0, Ly, params.ny, endpoint=False).reshape(1, -1)

    def potential(blobs):
        # blobs (NBLOB_MAX, 4): x0, y0, sigma, peak |grad| (signed; 0 = unused slot).
        # gaussian A exp(-r^2/2 sigma^2) has peak |grad| = A e^{-1/2}/sigma at r = sigma.
        def one(b):
            ddx = (x - b[0] + 0.5 * Lx) % Lx - 0.5 * Lx   # minimum image
            ddy = (y - b[1] + 0.5 * Ly) % Ly - 0.5 * Ly
            return b[3] * b[2] * np.exp(0.5) * jnp.exp(-(ddx**2 + ddy**2) / (2.0 * b[2]**2))
        return jnp.sum(jax.vmap(one)(blobs), axis=0)

    def potential_k(blobs):
        # blobs (NBLOB_MAX, 6): x0, y0, sigma, peak |grad| (signed), ratio, theta -- an
        # ANISOTROPIC gaussian A exp(-a^2/2 sigma^2 - b^2/2 (ratio sigma)^2), b the
        # coordinate along the direction at angle theta from x and a the one across it.
        # A = amp sigma sqrt(e) as for a blob of the narrow width, so `amp` is the peak
        # gradient ACROSS the sheet. Evaluated analytically in k space (the unnormalised
        # transform of the periodised gaussian), which is what the page's kick kernel
        # does: exactly periodic at any width, where a rotated minimum-image gaussian
        # would be cut at the box edge.
        kx, ky = kgrid.kx, kgrid.ky
        def one(b):
            c, s = jnp.cos(b[5]), jnp.sin(b[5])
            ka, kb = -s * kx + c * ky, c * kx + s * ky
            sa, sb = b[2], b[2] * b[4]
            w = b[3] * sa * np.exp(0.5) * 2.0 * np.pi * sa * sb * params.nx * params.ny / (Lx * Ly)
            return w * jnp.exp(-0.5 * ((sa * ka)**2 + (sb * kb)**2) - 1j * (kx * b[0] + ky * b[1]))
        return jnp.sum(jax.vmap(one)(blobs), axis=0)

    @jax.jit
    def kick(fields, blobs_p, blobs_m):
        # 4 columns per blob: today's isotropic kick, built in real space. 6 columns: the
        # anisotropic one, built in k space. The shape is static, so this is a trace-time
        # branch and the 4-column graph is the one that existed before sheets did.
        if blobs_p.shape[1] == 6:
            zp, zm = potential_k(blobs_p), potential_k(blobs_m)
            dk = jnp.stack([0.5 * (zp + zm), 0.5 * (zp - zm)])[:, None] * kgrid.dealias
        else:
            zp, zm = potential(blobs_p), potential(blobs_m)
            d = jnp.stack([0.5 * (zp + zm), 0.5 * (zp - zm)])[:, None]      # (phi, psi)
            dk = fft(d, params) * kgrid.dealias
        dk = dk.at[:, :, 0, 0].set(0.0)
        return fields + dk.astype(fields.dtype)

    @jax.jit
    def observe(fields):
        # the displayable fields, the max advecting speed (max of |grad z+|, |grad z-|) and
        # the energy
        phi, psi = fields[0], fields[1]
        def grad2(f):
            gx = ifft(1j * kgrid.kx * f, params); gy = ifft(1j * kgrid.ky * f, params)
            return (gx**2 + gy**2)[0]
        img = dict(vorticity=ifft(-kgrid.ksq * phi, params)[0],
                   current=ifft(-kgrid.ksq * psi, params)[0],
                   phi=ifft(phi, params)[0], psi=ifft(psi, params)[0],
                   speed=jnp.sqrt(grad2(phi)), bperp=jnp.sqrt(grad2(psi)))
        umax = jnp.sqrt(jnp.maximum(jnp.max(grad2(phi + psi)), jnp.max(grad2(phi - psi))))
        norm = (params.nx * params.ny) ** 2
        e = 0.5 * jnp.sum(kgrid.yfac * kgrid.ksq * (jnp.abs(phi)**2 + jnp.abs(psi)**2)) / norm
        return img, umax, e

    return kick, observe


SIGMA_MAX = 0.45         # widest blob: box-limited (energy peak at k = sqrt(3/2)/sigma ~ 2.7)
SIGMA_MIN_DX = 3.7       # narrowest blob in grid spacings: its energy peak then sits a factor
                         # ~3 in k below where the default hyperviscosity takes over (k ~ n/6),
                         # which leaves an unforced cascade under the smallest blob
SIGMA_STEP = 1.8         # neighbouring blob widths closer than this look the same: a blob's
                         # spectrum k^3 exp(-k^2 sigma^2) is as wide as its peak wavenumber
NBLOB_BASS = 3           # blobs per kick per channel in the bass band; others scale as 1/sigma


def band_plan(n, nbands=None, sigma_min=None, Lx=LBOX, Ly=LBOX):
    """(sigmas, nblobs) per band, bass first. n: the grid, one number or (nx, ny).

    Widths are log-spaced from SIGMA_MAX down to sigma_min (default SIGMA_MIN_DX grid
    spacings, so it shrinks with n). The number of bands defaults to as many widths as fit
    with a factor SIGMA_STEP between neighbours: 4 at 256^2, 5 at 512^2, 6 at 1024^2. The
    blob count goes as 1/sigma, which makes the energy per kick fall only as sigma (it is
    nblob sigma^2) instead of sigma^2.

    In a rectangular box (Lx, Ly) or on a grid with nx != ny: the narrowest blob is
    SIGMA_MIN_DX of the LARGER of the two grid spacings (it has to be resolved along both
    axes); the widest stays SIGMA_MAX, a fraction of the short side, which is always 2 pi;
    and the blob counts are multiplied by the box's area over (2 pi)^2 -- exactly 1 for the
    square box -- so a wider box is not sparser.
    """
    nx, ny = (n, n) if np.ndim(n) == 0 else n
    if sigma_min is None:
        sigma_min = max(SIGMA_MIN_DX * Lx / nx, SIGMA_MIN_DX * Ly / ny)
    if not 0.0 < sigma_min <= SIGMA_MAX:
        raise ValueError(f"sigma_min must be in (0, {SIGMA_MAX}], got {sigma_min}")
    if nbands is None:
        nbands = max(1, int(round(1.0 + np.log(SIGMA_MAX / sigma_min) / np.log(SIGMA_STEP))))
    sigmas = SIGMA_MAX * (sigma_min / SIGMA_MAX) ** (np.arange(nbands) / max(nbands - 1, 1))
    nblobs = np.rint(NBLOB_BASS * SIGMA_MAX / sigmas * ((Lx / LBOX) * (Ly / LBOX))).astype(int)
    if nblobs.max() > NBLOB_MAX:
        raise ValueError(f"{nblobs.max()} blobs per kick in the top band exceeds NBLOB_MAX = "
                         f"{NBLOB_MAX}: raise it or sigma_min")
    return sigmas, nblobs


def blobs_for(rng, sigma, nblob, amp, xc=None, box=None):
    # one kick's blobs for ONE channel: positions uniform, signs random. Called once per
    # channel, which is what makes the z+ and z- sets independent. With `xc` (stereo
    # placement) x is instead gaussian about xc, width STEREO_JITTER of the box; y stays
    # uniform. xc=None draws exactly what it always drew.
    # box = (Lx, Ly), default the square 2 pi box: x is drawn over Lx and y over Ly, from
    # the same stream (one uniform per coordinate), and the stereo scatter is in units of Lx.
    Lx, Ly = (LBOX, LBOX) if box is None else box
    out = np.zeros((nblob, 4))
    out[:, 0:2] = rng.uniform(0.0, (Lx, Ly), size=(nblob, 2))
    if xc is not None:
        out[:, 0] = (xc + STEREO_JITTER * Lx * rng.standard_normal(nblob)) % Lx
    out[:, 2] = sigma
    out[:, 3] = amp * rng.choice([-1.0, 1.0], size=nblob)
    return out


KICK_TYPES = ("elsasser", "vortex", "island", "dipole", "sheet", "sheet-island")
DIPOLE_SEP = 1.5         # separation of a dipole's two blobs, in sigma
SHEET_RATIO = 4.0        # a sheet's length over its width
STEREO_SPAN = 0.4        # hard left / right put a band's centre at (0.5 -+ this) of the box
STEREO_JITTER = 0.12     # gaussian scatter of the blobs about that centre, in box lengths


def kick_types_for(spec, nbands):
    """One kick type per band, bass first, from --kick-types: None (all `elsasser`, the
    phase-1 kick), `variety` (bass dipole, treble sheet, vortex and island alternating in
    between), one type for every band, or a comma list with one entry per band."""
    if spec is None:
        return ("elsasser",) * nbands
    names = [s.strip() for s in spec.split(",")] if isinstance(spec, str) else list(spec)
    if names == ["variety"]:
        names = ["dipole"] + [("vortex", "island")[i % 2] for i in range(nbands - 2)] + ["sheet"]
        names = names[:1] if nbands == 1 else names
    elif len(names) == 1:
        names = names * nbands
    bad = [s for s in names if s not in KICK_TYPES]
    if bad or len(names) != nbands:
        raise ValueError(f"--kick-types: need `variety`, one type, or {nbands} of {KICK_TYPES} "
                         f"(one per band, low to high); got {names}")
    return tuple(names)


def kick_blobs(rng, kind, sigma, nblob, amp, xc=None, ratio=SHEET_RATIO, sheet_energy=1.0, box=None):
    """One kick's (z+ list, z- list). Every entry has peak |grad z| = `amp` in its channel,
    and phi += (z+ + z-)/2, psi += (z+ - z-)/2, so:

      elsasser  independent sets in the two channels                   (the phase-1 kick)
      vortex    the same set in both: phi only, a velocity kick, psi untouched
      island    the same set, sign flipped in z-: psi only, a magnetic kick, phi untouched
      dipole    a vortex kick made of pairs: two opposite-signed blobs DIPOLE_SEP sigma
                apart at a random orientation. Each advects the other, so the pair travels
                perpendicular to the line joining them. `nblob` pairs; a pair carries
                dipole_energy_factor() = 1.5 times one blob's energy.
      sheet     independent sets of ANISOTROPIC gaussians, width sigma and length
                ratio*sigma at a random orientation; 6 columns (see make_kick). `nblob`
                per channel; a sheet carries (ratio + 1/ratio)/2 times a blob's energy,
                times `sheet_energy` (the amplitude is scaled by its square root).
      sheet-island  one set of sheets, sign flipped in z-: psi only, magnetic sheets. The
                same energy per kick as `sheet`.

    box = (Lx, Ly) as in blobs_for (default: the square 2 pi box).
    """
    if kind == "elsasser":
        return blobs_for(rng, sigma, nblob, amp, xc, box), blobs_for(rng, sigma, nblob, amp, xc, box)
    if kind in ("sheet", "sheet-island"):
        def one():
            b = blobs_for(rng, sigma, nblob, amp * np.sqrt(sheet_energy), xc, box)
            return np.column_stack([b, np.full(nblob, ratio), rng.uniform(0.0, np.pi, nblob)])
        if kind == "sheet":
            return one(), one()
        b = one()
        m = b.copy()
        m[:, 3] *= -1.0
        return b, m
    b = blobs_for(rng, sigma, nblob, amp, xc, box)
    if kind == "dipole":
        th = rng.uniform(0.0, 2.0 * np.pi, nblob)
        off = 0.5 * DIPOLE_SEP * sigma * np.column_stack([np.cos(th), np.sin(th)])
        b = np.concatenate([b, b])
        b[:nblob, 0:2] += off
        b[nblob:, 0:2] -= off
        b[nblob:, 3] *= -1.0
        b[:, 0:2] %= LBOX if box is None else np.asarray(box)
    m = b.copy()
    if kind == "island":
        m[:, 3] *= -1.0
    elif kind not in ("vortex", "dipole"):
        raise ValueError(f"unknown kick type {kind!r}")
    return b, m


def dipole_energy_factor(sep=DIPOLE_SEP):
    # energy of two opposite-signed gaussian blobs `sep` sigma apart over that of ONE blob:
    # the cross term of their gradients is -(1 - q) e^{-q} of a blob's, q = sep^2/4
    q = 0.25 * sep * sep
    return 2.0 * (1.0 - (1.0 - q) * np.exp(-q))


def pad(blobs):
    if len(blobs) > NBLOB_MAX:
        raise ValueError(f"{len(blobs)} blobs in one kick exceeds NBLOB_MAX = {NBLOB_MAX}")
    out = np.zeros((NBLOB_MAX, blobs.shape[1]))
    out[:, 2] = 1.0                      # unused slots: sigma 1, amplitude 0
    if blobs.shape[1] == 6:
        out[:, 4] = 1.0                  # ... and ratio 1
    out[:len(blobs)] = blobs
    return out


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def run(sr, audio, out, *, n=256, fps=30, rate=1.0, amp=1.0, memory=1.0, nbands=None, diss=None,
        hyper=4, cfl=0.4, threshold=0.3, causal=False, field="speed", cmap=None, sigma_min=None, seed=0,
        duration=None, audio_path=None, quiet=False, kick_types=None, sheet_ratio=SHEET_RATIO,
        stereo=None, stereo_gain=1.0, stereo_bands=None, sheet_energy=1.0,
        pulse=0.0, pulse_tau=0.25, pulse_band=0, mean_field_b0=0.0, section_every=None,
        section_kw=None, turn_s=2.0, field_walk=None, ratio_walk=0.0, walk_tau=0.1, zoom_out=0.0,
        aspect=1.0, grid=None, sheets=0.0, sheets_width=2.0, sheets_source=True, sheets_profile="bump",
        max_steps=None):
    # max_steps: abort with a RuntimeError once the run has taken that many steps.
    # sheets_profile: "bump" (the default) is the
    # sech^2 profile described below, `sheets` its peak field in units of amp; "harris" is
    # the periodic double Harris sheet (sheet_harris), `sheets` its far field B0 in units
    # of amp. Width, maintenance and the exclusions are the same for both.
    # aspect, grid: the box (box_for(aspect): short side 2 pi) and the grid (nx, ny), default
    # (n, n). The defaults are the square 2 pi box on n x n, and then nothing below differs
    # from what it was before either existed.
    # sheets: the maintained current-sheet equilibrium psi_eq = psi0 sech^2((x - Lx/2)/a)
    # as the initial condition and background, its peak field `sheets` * amp (0 = off) and
    # a = sheets_width * the narrowest blob width. It is restored against the hyperviscosity
    # (a source in the psi equation, Stepper.set_equilibrium) and the drag (drag()); the
    # kicks are not part of it and decay as before. Not with a mean field of either kind.
    # sheets_source=False keeps the initial condition and drops the restoring source: the
    # control the gate uses (the sheets then decay at the drag rate).
    # zoom_out: seconds of audio per factor 2 of the zoom-out (0 = off). The picture shows
    # a window of the tiling that grows as 2^(t/zoom_out); every zoom_out seconds the state
    # is re-gridded (regrid) and the window is the box again.
    # field_walk: None, or the rms turn in DEGREES of the mean field's direction at each
    # full-strength onset of pulse_band (a random walk; the step scales with the onset's
    # strength). mean_field_b0 is then |B0| in units of the rms fluctuation sqrt(2 E) at the
    # time, not of amp, so the field keeps its hold as the energy builds up; with
    # ratio_walk > 0 that ratio also random-walks, by rms ratio_walk per full-strength
    # onset, reflected into [0, mean_field_b0]. Both relax to their new values over walk_tau
    # seconds. This replaces the section trigger.
    # mean_field_b0: |B0| in units of amp (0 = off). Its direction is redrawn at each
    # section change: SectionDetector's triggers (section_kw are its options), or with
    # section_every = N at every N-th onset of the bass band.
    # kick_types: see kick_types_for. stereo: the (n, 2) [left, right] samples behind the
    # mono `audio` (same length), which switches stereo placement on; stereo_bands: the
    # bands that take part (default: all but the lowest -- the bass is centred in nearly
    # every mix and would land in one central column).
    import imageio

    if duration is not None:
        audio = audio[:int(duration * sr)]
        stereo = None if stereo is None else stereo[:int(duration * sr)]
    T = len(audio) / sr
    if sheets and (mean_field_b0 or field_walk is not None or ratio_walk):
        raise ValueError("sheets and a mean field (mean_field_b0, field_walk, ratio_walk) are "
                         "two backgrounds: choose one")
    if sheets_profile not in SHEET_PROFILES:
        raise ValueError(f"sheets_profile must be one of {SHEET_PROFILES}, got {sheets_profile!r}")
    if sheets < 0.0 or (sheets and not sheets_width > 0.0):
        raise ValueError(f"sheets must be >= 0 and sheets_width > 0, got {sheets}, {sheets_width}")
    Lx, Ly = box_for(aspect)
    nx, ny = (n, n) if grid is None else grid
    box = None if (Lx, Ly) == (LBOX, LBOX) else (Lx, Ly)
    sigmas, nblobs = band_plan((nx, ny), nbands, sigma_min, Lx, Ly)
    nbands = len(sigmas)
    kinds = kick_types_for(kick_types, nbands)
    if causal:
        t_on, b_on, s_on, edges, _ = detect_onsets_causal(sr, audio, nbands=nbands, threshold=threshold)
    else:
        t_on, b_on, s_on, edges = detect_onsets(sr, audio, nbands=nbands, threshold=threshold)
    pans, xcs = None, None
    if stereo is not None and is_mono(stereo):
        if not quiet:
            print("  stereo placement: the two channels are identical -- uniform placement", flush=True)
    elif stereo is not None:
        if len(stereo) != len(audio):
            raise ValueError("stereo must have the same number of samples as audio")
        if stereo_bands is None:
            stereo_bands = range(1, nbands)
        if any(not 0 <= b < nbands for b in stereo_bands):
            raise ValueError(f"stereo_bands must be band indices 0..{nbands - 1}, got {list(stereo_bands)}")
        pans = onset_pans(sr, stereo, t_on, b_on, edges)
        # x runs left to right in the movie (frame = img.T[::-1])
        # (and x is the long side in landscape; in portrait the same map covers the short one)
        xcs = Lx * (0.5 + STEREO_SPAN * np.clip(stereo_gain * pans, -1.0, 1.0))
        xcs = [float(x) if b in stereo_bands else None for x, b in zip(xcs, b_on)]
    if not quiet:
        for b in range(nbands):
            line = (f"  band {b}: {edges[b]:6.0f}-{edges[b + 1]:6.0f} Hz -> sigma {sigmas[b]:.4f} "
                    f"(k_peak {np.sqrt(1.5) / sigmas[b]:5.1f}), {nblobs[b]:3d} blobs/kick, "
                    f"{np.sum(b_on == b) / T:4.1f} kicks/s")
            if kick_types is not None:
                line += f", {kinds[b]}"
            if pans is not None and np.any(b_on == b):
                pb = pans[b_on == b]
                line += (f", pan {pb.mean():+.2f} +- {pb.std():.2f}"
                         + ("" if b in stereo_bands else " (not placed)"))
            print(line, flush=True)
    nframes = int(T * fps)
    if diss is None:
        # hyperviscous rate nu k^(2 hyper) = 48 amp at the dealiasing cutoff n/3, whatever
        # the order (at hyper 2 this is the earlier 3 amp/(n/6)^4). A cutoff rate of ~5-12
        # leaves a flat, upturning tail -- grid-scale noise that looks like a CFL problem but
        # does not respond to the timestep. Scales with amp because the strain rate the
        # dissipation has to beat does. hyper 4 (k^8) is the default: the rate falls to
        # 3 amp at k = n/4.2 where k^4 reached it at n/6, ~1.4x more undamped range, with
        # the same bass energy jump per kick (256^2 click track: E(40)/E(10) 6.8e-2 against
        # 2.3e-2, E(60)/E(10) 1.1e-2 against 1.0e-3, tail still falling to 2e-4 at k = 84).
        # In a rectangular box or grid the cutoff is the SMALLER of the two axes' (cutoff_k,
        # which is n/3 for the square 2 pi box).
        diss = 48.0 * amp / cutoff_k(nx, ny, Lx, Ly) ** (2 * hyper)

    b0, sections = None, None
    walk = None
    if ratio_walk and (field_walk is None or not mean_field_b0):
        raise ValueError("ratio_walk needs field_walk and a mean field: it walks |B0| / rms")
    if mean_field_b0 and field_walk is not None:
        rng_b = np.random.default_rng([seed, 5])
        th = rng_b.uniform(0.0, 2.0 * np.pi)
        r = rng_b.uniform(0.0, mean_field_b0) if ratio_walk else float(mean_field_b0)
        walk = dict(th=th, r=r, th_to=th, r_to=r, bvec=(0.0, 0.0), trace=[])
        b0 = "driven"
        if not quiet:
            print(f"  mean field: direction walks {field_walk:.0f} deg rms per band-{pulse_band} onset, "
                  + (f"|B0|/rms walks {ratio_walk:.2f} per onset in 0-{mean_field_b0:.2f}"
                     if ratio_walk else f"|B0| = {mean_field_b0:.2f} rms"), flush=True)
    elif mean_field_b0:
        if section_every:
            sections = t_on[b_on == 0][section_every - 1::section_every]
        else:
            sections = detect_sections(sr, audio, nbands=nbands, **(section_kw or {}))[0]
        sections = sections[sections < T]
        rng_b = np.random.default_rng([seed, 4])            # its own stream: the kicks keep theirs
        theta0 = rng_b.uniform(0.0, 2.0 * np.pi)
        # each turn is at least 45 degrees either way, so a section change always shows
        dtheta = rng_b.uniform(0.25 * np.pi, np.pi, len(sections)) * rng_b.choice([-1.0, 1.0], len(sections))
        b0 = mean_field(mean_field_b0 * amp, theta0, rate * sections, dtheta, rate * turn_s)
        if not quiet:
            print(f"  mean field {mean_field_b0 * amp:.2f}, direction redrawn at "
                  + (" ".join(f"{t:.0f}" for t in sections) or "no time") + " s", flush=True)
    step = Stepper((nx, ny), rate / (4.0 * fps), diss, hyper, cfl, b0, Lx, Ly, source=bool(sheets) and sheets_source)
    kick, observe = make_kick(step.p, step.kg)
    psi_eq, sheet_info = None, None
    if sheets:
        a_eq = sheets_width * float(sigmas[-1])
        harris = sheets_profile == "harris"
        psi0 = None if harris else sheet_psi0(sheets, amp, a_eq)
        if harris and not a_eq < Lx / 8.0:
            raise ValueError(f"harris sheets of width a = {a_eq:.3f} do not fit: need a < Lx/8 = {Lx / 8.0:.3f}")

        def sheet_ic(x, y):
            psi = (sheet_harris(x, Lx, sheets * amp, a_eq) if harris
                   else sheet_equilibrium(x, Lx, psi0, a_eq)) + 0.0 * y
            return jnp.stack([jnp.zeros_like(psi), psi])
        state = jr.initialize(sheet_ic, step.p)
        # the equilibrium the source maintains IS the dealiased initial state's k_y = 0
        # column, taken once: nothing a kick adds to that column later is part of it
        col = state.fields[1, 0, :, 0] + 0.0
        if sheets_source:
            step.set_equilibrium(col)
            psi_eq = step.psi_eq
        sheet_info = dict(psi0=psi0, a=a_eq, psi_eq=np.asarray(col), profile=sheets_profile, b0=sheets * amp,
                          energy=harris_energy_eq(sheets * amp, a_eq, Lx) if harris else sheet_energy_eq(psi0, a_eq, Lx))
        if not quiet:
            print(f"  current sheets ({sheets_profile}): "
                  + ("" if harris else f"psi0 = {psi0:.4f}, ")
                  + f"a = {a_eq:.4f} ({sheets_width:g} x the narrowest blob), "
                  f"{'field +-' if harris else 'peak field '}{sheets * amp:.2f}, energy {sheet_info['energy']:.4f}"
                  + ("" if sheets_source else " (NOT maintained)"), flush=True)
    else:
        state = jr.initialize(lambda x, y: jnp.zeros((2, 1, nx, ny)), step.p)
    fw, fh = frame_shape(nx, ny, Lx, Ly)
    if not quiet and (box is not None or nx != ny):
        print(f"  box {Lx:.4f} x {Ly:.4f} on {nx} x {ny} (cells {Lx / nx:.4f} x {Ly / ny:.4f}), "
              f"frame {fw} x {fh}, cutoff k = {cutoff_k(nx, ny, Lx, Ly):.1f}", flush=True)

    # events in audio seconds: (time, kind, payload); kind 0 = kick, 1 = frame, so a kick
    # that coincides with a frame is in that frame
    events = [(float(t), 0, (int(b), float(s), None if xcs is None else xcs[i]))
              for i, (t, b, s) in enumerate(zip(t_on, b_on, s_on)) if t < T]
    events += [(i / fps, 1, i) for i in range(nframes)]
    if zoom_out:
        # kind -1: at a shared time the re-grid comes first, so that frame shows the new box
        events += [(k * zoom_out, -1, k) for k in range(1, int(T / zoom_out) + 1) if k * zoom_out < T]
    events.sort(key=lambda ev: (ev[0], ev[1]))

    # the camera pulse (display only): zoom per frame from the onsets of one band
    zooms = None
    if pulse:
        sel = (b_on == pulse_band) & (t_on < T)
        zooms = pulse_zoom(np.arange(nframes) / fps, t_on[sel], s_on[sel], pulse, pulse_tau)
    rng = np.random.default_rng(seed)
    signed = field not in ("speed", "bperp")
    cm = get_cmap(cmap or ("RdBu_r" if signed else "magma"))
    video = out if audio_path is None else out + ".silent.mp4"
    writer = imageio.get_writer(video, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None, pixelformat="yuv420p")
    vmax, energy, umaxs, late, jumps = None, [], [], [], [[] for _ in range(nbands)]
    umax, t_prev, nsteps, e, t_walk, regrids = 0.0, 0.0, 0, 0.0, 0.0, []
    if sheets:
        # the run does not start from rest: the first advance needs the equilibrium's speed
        _, umax, e = observe(state.fields)
        umax, e = float(umax), float(e)
    t0 = time.perf_counter()
    for t_ev, kind, payload in events:
        if walk is None:
            state, k = step.advance(state, rate * t_ev, umax)
        else:
            # relax towards the walk's current target, then hold B0 over this advance
            # (events are at most a frame apart)
            w = 1.0 - np.exp(-(t_ev - t_walk) / walk_tau)
            walk["th"] += w * (walk["th_to"] - walk["th"])
            walk["r"] += w * (walk["r_to"] - walk["r"])
            t_walk = t_ev
            bmag = walk["r"] * np.sqrt(2.0 * e)
            walk["bvec"] = (bmag * np.cos(walk["th"]), bmag * np.sin(walk["th"]))
            state, k = step.advance(state, rate * t_ev, umax, walk["bvec"])
        nsteps += k
        if max_steps is not None and nsteps > max_steps:
            writer.close()
            raise RuntimeError(f"more than max_steps = {max_steps} steps by t = {t_ev:.2f} s ({nsteps})")
        t_now = float(state.t)
        late.append((t_now - rate * t_ev) / rate)
        if memory is not None and t_now > t_prev:
            state = state._replace(fields=drag(state.fields, float(np.exp(-(t_now - t_prev) / (2.0 * memory * rate))), psi_eq))
        t_prev = t_now
        if kind == -1:
            e_before = float(observe(state.fields)[2])
            state = state._replace(fields=jnp.asarray(regrid(state.fields, step.kg.dealias)))
            _, umax, e = observe(state.fields)
            umax, e = float(umax), float(e)
            regrids.append((t_ev, e / e_before if e_before > 0 else 1.0))
            continue
        if kind == 0:
            band, strength, xc = payload
            if walk is not None and band == pulse_band:
                walk["th_to"] += np.deg2rad(field_walk) * strength * rng_b.standard_normal()
                if ratio_walk:
                    # reflect into [0, mean_field_b0]
                    x = (walk["r_to"] + ratio_walk * strength * rng_b.standard_normal()) % (2.0 * mean_field_b0)
                    walk["r_to"] = x if x <= mean_field_b0 else 2.0 * mean_field_b0 - x
            e_before = float(observe(state.fields)[2])
            bp, bm = kick_blobs(rng, kinds[band], sigmas[band], nblobs[band], amp * strength,
                                xc, sheet_ratio, sheet_energy, *(() if box is None else (box,)))
            state = state._replace(fields=kick(state.fields, pad(bp), pad(bm)))
            _, umax, e = observe(state.fields)
            umax, e = float(umax), float(e)
            if e_before > 0:
                jumps[band].append((t_ev, (e - e_before) / e_before))
            continue
        imgs, umax, e = observe(state.fields)
        umax, e = float(umax), float(e)
        if not np.isfinite(e):
            raise FloatingPointError(f"non-finite energy at t = {t_ev:.2f} s")
        energy.append(e); umaxs.append(umax)
        if walk is not None:
            walk["trace"].append((walk["th"], walk["r"], float(np.hypot(*walk["bvec"]))))
        img = np.asarray(imgs[field])
        # colour scale: slow-moving 99.5th percentile (time constant 4 s), so a beat shows
        # as a brightening rather than being normalised away frame by frame
        cur = max(np.percentile(np.abs(img), 99.5), 1e-12)
        vmax = cur if vmax is None else max(vmax + (cur - vmax) / (4.0 * fps), 0.5 * cur)
        # x runs left to right and y bottom to top; resampled to the box's shape with square
        # pixels when the cells are not square (a no-op otherwise)
        u = resample_frame(img.T[::-1] / vmax, (fh, fw))
        zz = 1.0 if zooms is None else zooms[payload]
        if zoom_out:
            zz *= zoom_out_scale(t_ev, zoom_out, len(regrids))
        if zz != 1.0:
            u = zoom_image(u, zz)
        rgb = cm(0.5 + 0.5 * np.clip(u, -1.0, 1.0) if signed else np.clip(u, 0.0, 1.0))[..., :3]
        writer.append_data((255 * rgb).astype(np.uint8))
        if not quiet and payload % (5 * fps) == 0:
            print(f"  t = {t_ev:6.1f} s   E = {e:8.4f}   umax = {umax:6.2f}   "
                  f"steps = {nsteps:6d}   wall = {time.perf_counter() - t0:6.1f} s", flush=True)
    writer.close()

    if audio_path is not None:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", video, "-i", audio_path, "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-shortest", out], check=True)
        os.remove(video)

    return dict(energy=np.array(energy), umax=np.array(umaxs), late=np.array(late), jumps=jumps,
                onsets=(t_on, b_on, s_on), kick_types=kinds, pans=pans, zooms=zooms, sections=sections, regrids=regrids, sheets=sheet_info, box=(Lx, Ly), grid=(nx, ny), frame=(fw, fh), walk=None if walk is None else np.array(walk["trace"]), edges=edges, sigmas=sigmas, nblobs=nblobs, fps=fps, diss=diss, nsteps=nsteps, fields=np.asarray(state.fields),
                duration=T, wall=time.perf_counter() - t0)


def zoom_out_scale(t, period, ndone):
    """The zoom-out's display scale at audio time t: 2^-(t - ndone*period)/period, in
    [1/2, 1], `ndone` the number of re-grids already applied. Counted from the re-grids
    rather than from t % period, which for a period that is not a whole number of frames
    can sit a rounding error below `period` on the very frame a re-grid has just run."""
    return 2.0 ** (-min(max((t - ndone * period) / period, 0.0), 1.0))


def pulse_zoom(t_frames, t_on, s_on, a=0.08, tau=0.25):
    """The camera pulse: zoom factor s >= 1 at each frame time. An onset at t0 of strength
    q lifts the zoom TO 1 + a q (never lowers it), and between onsets s - 1 decays as
    exp(-(t - t0)/tau). An onset at exactly a frame time is in that frame. Causal."""
    out = np.ones(len(t_frames))
    order = np.argsort(t_on, kind="stable")
    t_on, s_on = np.asarray(t_on, float)[order], np.asarray(s_on, float)[order]
    excess, t_last, j = 0.0, 0.0, 0
    for i, t in enumerate(t_frames):
        while j < len(t_on) and t_on[j] <= t:
            excess = max(excess * np.exp(-(t_on[j] - t_last) / tau), a * s_on[j])
            t_last = t_on[j]
            j += 1
        out[i] = 1.0 + excess * np.exp(-(t - t_last) / tau)
    return out


def zoom_image(img, s, blur=True):
    """`img` (PERIODIC, any shape, square pixels) magnified by s about its centre, same shape: output pixel i
    shows the input at c + (i - c)/s, c = (n - 1)/2 along each axis, bilinear, wrapping. The box is
    periodic, so s < 1 shows the tiling (after a blur matched to the coarser sampling, unless
    blur=False); s == 1 returns img itself."""
    if s == 1.0:
        return img
    from scipy.ndimage import map_coordinates
    if s < 1.0 and blur:
        # zooming OUT samples more than one input pixel per output pixel: blur first, by
        # the gaussian whose width is that of the box an output pixel now covers
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, 0.5 * np.sqrt(max(1.0 / (s * s) - 1.0, 0.0)), mode="wrap")
    n0, n1 = img.shape
    i0 = 0.5 * (n0 - 1) + (np.arange(n0) - 0.5 * (n0 - 1)) / s
    i1 = 0.5 * (n1 - 1) + (np.arange(n1) - 0.5 * (n1 - 1)) / s
    return map_coordinates(img, np.meshgrid(i0, i1, indexing="ij"), order=1, mode="grid-wrap")


BRAT_GREEN = (0x8A / 255.0, 0xCE / 255.0, 0.0)


def get_cmap(name):
    """A matplotlib colormap by name, plus the music page's two: `brat` (#8ACE00 at zero,
    falling linearly to black) and `brat_r` (black at zero, rising to #8ACE00)."""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap
    if name in ("brat", "brat_r"):
        ends = [BRAT_GREEN, (0.0, 0.0, 0.0)]
        return LinearSegmentedColormap.from_list(name, ends if name == "brat" else ends[::-1])
    return colormaps[name]


def beat_contrast(res, band=0):
    """Mean fractional energy jump (E_after - E_before)/E_before across the kicks of `band`,
    after the first quarter of the track (spin-up).

    This is T_beat/t_memory in disguise: ~0 means the flow integrates over the beat and the
    tempo is invisible, >~0.3 means each beat visibly re-makes the flow.
    """
    j = [x for t, x in res["jumps"][band] if t > 0.25 * res["duration"]]
    return (float(np.mean(j)), len(j)) if j else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser(description="music -> 2D RMHD turbulence movie")
    ap.add_argument("wav", nargs="?", help="PCM WAV file; omit for the built-in click track")
    ap.add_argument("--out", default="music_turbulence.mp4")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--aspect", type=float, default=1.0,
                    help="S2: the picture's width over height. The box's short side stays 2 pi and "
                         "its long side is 2 pi * max(A, 1/A): A > 1 landscape (1.7778 = 16:9), "
                         "A < 1 portrait (0.5625 = 9:16)")
    ap.add_argument("--grid", default=None, metavar="NXxNY",
                    help="S2: the grid, x by y, e.g. 512x256 (default: --n in both; the cells need "
                         "not be square)")
    ap.add_argument("--sheets", type=float, nargs="?", const=1.5, default=0.0,
                    help="S4: a maintained current-sheet pair psi0 sech^2((x - Lx/2)/a) as the "
                         "background, peak field in units of --amp (1.5 if no value); not with "
                         "--mean-field / --field-walk / --ratio-walk")
    ap.add_argument("--sheet-profile", choices=SHEET_PROFILES, default="bump",
                    help="with --sheets: bump (default) = psi0 sech^2((x - Lx/2)/a), --sheets = its "
                         "peak field/amp; harris = periodic double Harris sheet, field +-B0 "
                         "everywhere, reversing at x = Lx/4 and 3Lx/4, --sheets = B0/amp")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="abort once the run has taken this many steps")
    ap.add_argument("--sheet-width", type=float, default=2.0,
                    help="with --sheets: a in units of the narrowest blob width")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rate", type=float, default=1.0, help="simulation time units per second of audio")
    ap.add_argument("--amp", type=float, default=1.0, help="kick: peak |grad z| of a full-strength blob")
    ap.add_argument("--memory", type=float, default=1.0,
                    help="energy e-folding time of the drag, in seconds of audio (0 = no drag)")
    ap.add_argument("--bands", type=int, default=None,
                    help="number of log-spaced bands, 40 Hz-10 kHz (default: from the grid, see band_plan)")
    ap.add_argument("--sigma-min", type=float, default=None,
                    help="narrowest blob width (default: 3.7 grid spacings)")
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="onset threshold above the local mean flux (band 99th percentile = 1); "
                         "raise to keep only the stronger hits")
    ap.add_argument("--causal", action="store_true",
                    help="use the live-capable onset detector (detect_onsets_causal)")
    ap.add_argument("--kick-types", default=None,
                    help="E1: `variety` (bass dipole, vortex/island in between, treble sheet), one "
                         f"of {', '.join(KICK_TYPES)} for every band, or a comma list with one per "
                         "band, low to high (default: elsasser everywhere)")
    ap.add_argument("--sheet-ratio", type=float, default=SHEET_RATIO, help="length/width of a sheet")
    ap.add_argument("--sheet-energy", type=float, default=1.0,
                    help="multiplies the energy of every sheet (amplitude by its square root)")
    ap.add_argument("--stereo", action="store_true",
                    help="E2: place each band's blobs in x by the band's stereo pan at the onset")
    ap.add_argument("--stereo-gain", type=float, default=1.0,
                    help="multiplies the pan before placement (most mixes are narrow: try 2-3)")
    ap.add_argument("--stereo-bands", default=None,
                    help="comma list of the band indices placed by pan (default: all but band 0)")
    ap.add_argument("--pulse", type=float, nargs="?", const=0.08, default=0.0,
                    help="E3: zoom in by this fraction (0.08 if no value) on a full-strength bass "
                         "onset, relaxing over --pulse-tau; display only")
    ap.add_argument("--pulse-tau", type=float, default=0.25, help="relaxation time of the pulse, seconds")
    ap.add_argument("--pulse-band", type=int, default=0, help="the band whose onsets pulse the camera")
    ap.add_argument("--mean-field", type=float, nargs="?", const=0.5, default=0.0,
                    help="E4: uniform in-plane magnetic field, in units of --amp (0.5 if no value); "
                         "z+ and z- stream along it in opposite directions")
    ap.add_argument("--section-every", type=int, default=None,
                    help="redraw the mean field's direction every N-th bass onset instead of at "
                         "the detected section changes")
    ap.add_argument("--section-threshold", type=float, default=None,
                    help="SectionDetector threshold (default 1.5; lower = more section changes)")
    ap.add_argument("--field-walk", type=float, nargs="?", const=30.0, default=None,
                    help="random-walk the mean field's direction instead: rms turn in degrees "
                         "(30 if no value) at each full-strength onset of --pulse-band. "
                         "--mean-field is then |B0| in units of the rms fluctuation at the time")
    ap.add_argument("--ratio-walk", type=float, default=0.0,
                    help="with --field-walk: also random-walk |B0|/rms by this rms step per "
                         "full-strength onset, within 0 to --mean-field")
    ap.add_argument("--zoom-out", type=float, nargs="?", const=20.0, default=0.0,
                    help="E5: pull the camera back by a factor 2 every this many seconds (20 if "
                         "no value); at each factor 2 the state becomes its own 2x2 tiling")
    ap.add_argument("--click-stereo", action="store_true",
                    help="the click track in stereo: snare left, hats alternating right/left")
    ap.add_argument("--bpm", type=float, default=120.0, help="click track tempo")
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: whole file; 20 for the click track)")
    ap.add_argument("--field", choices=FIELDS, default="speed")
    ap.add_argument("--cmap", default=None,
                    help="colormap: brat, brat_r or any matplotlib name (default: magma for "
                         "speed/bperp, RdBu_r for the signed fields)")
    ap.add_argument("--diss", type=float, default=None,
                    help="hyperviscosity nu = eta (default: rate 48 amp at the cutoff n/3, see run())")
    ap.add_argument("--hyper", type=int, default=4,
                    help="dissipation order: nu k^(2 hyper); 4 = k^8 (default), 2 = k^4")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lr = None
    if a.wav is None:
        sr, audio = click_track(duration=a.duration or 20.0, bpm=a.bpm, stereo=a.click_stereo)
        audio_path = os.path.splitext(a.out)[0] + "_click.wav"
        wavfile.write(audio_path, sr, (32767 * np.clip(audio, -1.0, 1.0)).astype(np.int16))
        if a.click_stereo:
            lr, audio = audio, audio.mean(axis=1)
        duration = None
    else:
        sr, audio, lr = load_wav(a.wav, stereo=True)
        audio_path, duration = a.wav, a.duration
    if a.stereo and lr is None:
        print("  --stereo: one-channel input -- uniform placement")
    stereo_bands = None if a.stereo_bands is None else [int(b) for b in a.stereo_bands.split(",")]
    grid = None
    if a.grid is not None:
        try:
            grid = tuple(int(v) for v in a.grid.lower().split("x"))
            assert len(grid) == 2 and min(grid) > 0
        except (ValueError, AssertionError):
            ap.error(f"--grid: need NXxNY, e.g. 512x256, got {a.grid!r}")
    res = run(sr, audio, a.out, n=a.n, fps=a.fps, rate=a.rate, amp=a.amp, memory=a.memory or None, nbands=a.bands, sigma_min=a.sigma_min, diss=a.diss, hyper=a.hyper, cfl=a.cfl, threshold=a.threshold, causal=a.causal,
              field=a.field, cmap=a.cmap, seed=a.seed, duration=duration, audio_path=audio_path,
              kick_types=a.kick_types, sheet_ratio=a.sheet_ratio, stereo=lr if a.stereo else None,
              stereo_gain=a.stereo_gain, stereo_bands=stereo_bands, sheet_energy=a.sheet_energy,
              pulse=a.pulse, pulse_tau=a.pulse_tau, pulse_band=a.pulse_band,
              mean_field_b0=a.mean_field, section_every=a.section_every,
              section_kw=None if a.section_threshold is None else dict(threshold=a.section_threshold),
              field_walk=a.field_walk, ratio_walk=a.ratio_walk, zoom_out=a.zoom_out,
              aspect=a.aspect, grid=grid, sheets=a.sheets, sheets_width=a.sheet_width,
              sheets_profile=a.sheet_profile, max_steps=a.max_steps)
    t_on, b_on, _ = res["onsets"]
    print(f"wrote {a.out}  ({res['wall']:.0f} s wall, {res['nsteps']} steps = "
          f"{res['nsteps'] / res['duration']:.1f} per second of audio, diss = {res['diss']:.2e}, "
          f"events late by {1e3 * res['late'].mean():.1f} ms mean / {1e3 * res['late'].max():.1f} ms max)")
    if res["regrids"]:
        print("  re-grids at " + " ".join(f"{t:g}" for t, _ in res["regrids"]) + " s; energy kept: "
              + " ".join(f"{r:.3f}" for _, r in res["regrids"]))
    for b in range(len(res["sigmas"])):
        c, m = beat_contrast(res, b)
        print(f"  band {b} ({res['edges'][b]:.0f}-{res['edges'][b + 1]:.0f} Hz): "
              f"{int(np.sum(b_on == b))} onsets, beat contrast {c:+.2f} over {m}")


if __name__ == "__main__":
    main()
