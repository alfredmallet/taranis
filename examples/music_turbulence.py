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
#   python examples/music_turbulence_check.py   is the gate for both (fp64).
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


# ---------------------------------------------------------------------------------------
# solver pieces
# ---------------------------------------------------------------------------------------
class Stepper:
    """Step until t >= target (one jitted while_loop, the same construction as
    `taranis.run.simulate`'s). Adaptive dt normally; a fixed dt_cap when the flow is so slow
    that the adaptive dt would exceed it (each mode has its own Parameters and kgrid -- a
    kgrid is never reused across Parameters)."""

    def __init__(self, n, dt_cap, diss, hyper, cfl=0.4):
        kw = dict(nx=n, ny=n, Lx=LBOX, Ly=LBOX, cfl_safety=cfl, dims=2, comm_backend="serial",
                  eqpars={"diss": (diss, diss), "hyper": hyper})
        self.p = jr.Parameters(adaptive_timestep=True, **kw)
        self.kg = jr.setup_kgrids(self.p)
        p_fix = jr.Parameters(adaptive_timestep=False, dt=dt_cap, **kw)
        self._adaptive = self._build(self.p, self.kg)
        self._fixed = self._build(p_fix, jr.setup_kgrids(p_fix))
        self.dt_cap, self.cfl, self.dx = dt_cap, cfl, LBOX / n

    @staticmethod
    def _build(params, kgrid):
        recipe = equation_registry[params.eqtype]
        rhs, set_timestep = construct_rhs(recipe), recipe.set_timestep_func
        stepper, scheme = get_scheme("lsrk33")

        def to_target(state, target):
            def body(c):
                return stepper(c[0], kgrid, params, rhs, set_timestep, scheme), c[1] + 1
            return jax.lax.while_loop(lambda c: c[0].t < target, body, (state, 0))
        return jax.jit(to_target, donate_argnums=(0,))

    def advance(self, state, target, umax):
        # umax is the speed at entry; returns (state, steps taken). Consumes `state`.
        slow = 0.5 * self.cfl * self.dx / max(umax, 1e-30) > self.dt_cap
        state, nstep = (self._fixed if slow else self._adaptive)(state, target)
        return state, int(nstep)


def make_kick(params, kgrid):
    x = jnp.linspace(0.0, LBOX, params.nx, endpoint=False).reshape(-1, 1)
    y = jnp.linspace(0.0, LBOX, params.ny, endpoint=False).reshape(1, -1)

    def potential(blobs):
        # blobs (NBLOB_MAX, 4): x0, y0, sigma, peak |grad| (signed; 0 = unused slot).
        # gaussian A exp(-r^2/2 sigma^2) has peak |grad| = A e^{-1/2}/sigma at r = sigma.
        def one(b):
            ddx = (x - b[0] + 0.5 * LBOX) % LBOX - 0.5 * LBOX   # minimum image
            ddy = (y - b[1] + 0.5 * LBOX) % LBOX - 0.5 * LBOX
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
            w = b[3] * sa * np.exp(0.5) * 2.0 * np.pi * sa * sb * params.nx * params.ny / LBOX**2
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


def band_plan(n, nbands=None, sigma_min=None):
    """(sigmas, nblobs) per band, bass first.

    Widths are log-spaced from SIGMA_MAX down to sigma_min (default SIGMA_MIN_DX grid
    spacings, so it shrinks with n). The number of bands defaults to as many widths as fit
    with a factor SIGMA_STEP between neighbours: 4 at 256^2, 5 at 512^2, 6 at 1024^2. The
    blob count goes as 1/sigma, which makes the energy per kick fall only as sigma (it is
    nblob sigma^2) instead of sigma^2.
    """
    if sigma_min is None:
        sigma_min = SIGMA_MIN_DX * LBOX / n
    if not 0.0 < sigma_min <= SIGMA_MAX:
        raise ValueError(f"sigma_min must be in (0, {SIGMA_MAX}], got {sigma_min}")
    if nbands is None:
        nbands = max(1, int(round(1.0 + np.log(SIGMA_MAX / sigma_min) / np.log(SIGMA_STEP))))
    sigmas = SIGMA_MAX * (sigma_min / SIGMA_MAX) ** (np.arange(nbands) / max(nbands - 1, 1))
    nblobs = np.rint(NBLOB_BASS * SIGMA_MAX / sigmas).astype(int)
    if nblobs.max() > NBLOB_MAX:
        raise ValueError(f"{nblobs.max()} blobs per kick in the top band exceeds NBLOB_MAX = "
                         f"{NBLOB_MAX}: raise it or sigma_min")
    return sigmas, nblobs


def blobs_for(rng, sigma, nblob, amp, xc=None):
    # one kick's blobs for ONE channel: positions uniform, signs random. Called once per
    # channel, which is what makes the z+ and z- sets independent. With `xc` (stereo
    # placement) x is instead gaussian about xc, width STEREO_JITTER of the box; y stays
    # uniform. xc=None draws exactly what it always drew.
    out = np.zeros((nblob, 4))
    out[:, 0:2] = rng.uniform(0.0, LBOX, size=(nblob, 2))
    if xc is not None:
        out[:, 0] = (xc + STEREO_JITTER * LBOX * rng.standard_normal(nblob)) % LBOX
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


def kick_blobs(rng, kind, sigma, nblob, amp, xc=None, ratio=SHEET_RATIO, sheet_energy=1.0):
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
    """
    if kind == "elsasser":
        return blobs_for(rng, sigma, nblob, amp, xc), blobs_for(rng, sigma, nblob, amp, xc)
    if kind in ("sheet", "sheet-island"):
        def one():
            b = blobs_for(rng, sigma, nblob, amp * np.sqrt(sheet_energy), xc)
            return np.column_stack([b, np.full(nblob, ratio), rng.uniform(0.0, np.pi, nblob)])
        if kind == "sheet":
            return one(), one()
        b = one()
        m = b.copy()
        m[:, 3] *= -1.0
        return b, m
    b = blobs_for(rng, sigma, nblob, amp, xc)
    if kind == "dipole":
        th = rng.uniform(0.0, 2.0 * np.pi, nblob)
        off = 0.5 * DIPOLE_SEP * sigma * np.column_stack([np.cos(th), np.sin(th)])
        b = np.concatenate([b, b])
        b[:nblob, 0:2] += off
        b[nblob:, 0:2] -= off
        b[nblob:, 3] *= -1.0
        b[:, 0:2] %= LBOX
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
        stereo=None, stereo_gain=1.0, stereo_bands=None, sheet_energy=1.0):
    # kick_types: see kick_types_for. stereo: the (n, 2) [left, right] samples behind the
    # mono `audio` (same length), which switches stereo placement on; stereo_bands: the
    # bands that take part (default: all but the lowest -- the bass is centred in nearly
    # every mix and would land in one central column).
    import imageio

    if duration is not None:
        audio = audio[:int(duration * sr)]
        stereo = None if stereo is None else stereo[:int(duration * sr)]
    T = len(audio) / sr
    sigmas, nblobs = band_plan(n, nbands, sigma_min)
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
        xcs = LBOX * (0.5 + STEREO_SPAN * np.clip(stereo_gain * pans, -1.0, 1.0))
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
        diss = 48.0 * amp / (n / 3.0) ** (2 * hyper)

    step = Stepper(n, rate / (4.0 * fps), diss, hyper, cfl)
    kick, observe = make_kick(step.p, step.kg)
    state = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)

    # events in audio seconds: (time, kind, payload); kind 0 = kick, 1 = frame, so a kick
    # that coincides with a frame is in that frame
    events = [(float(t), 0, (int(b), float(s), None if xcs is None else xcs[i]))
              for i, (t, b, s) in enumerate(zip(t_on, b_on, s_on)) if t < T]
    events += [(i / fps, 1, i) for i in range(nframes)]
    events.sort(key=lambda ev: (ev[0], ev[1]))

    rng = np.random.default_rng(seed)
    signed = field not in ("speed", "bperp")
    cm = get_cmap(cmap or ("RdBu_r" if signed else "magma"))
    video = out if audio_path is None else out + ".silent.mp4"
    writer = imageio.get_writer(video, fps=fps, codec="libx264", quality=8,
                                macro_block_size=None, pixelformat="yuv420p")
    vmax, energy, umaxs, late, jumps = None, [], [], [], [[] for _ in range(nbands)]
    umax, t_prev, nsteps = 0.0, 0.0, 0
    t0 = time.perf_counter()
    for t_ev, kind, payload in events:
        state, k = step.advance(state, rate * t_ev, umax)
        nsteps += k
        t_now = float(state.t)
        late.append((t_now - rate * t_ev) / rate)
        if memory is not None and t_now > t_prev:
            state = state._replace(fields=state.fields * float(np.exp(-(t_now - t_prev) / (2.0 * memory * rate))))
        t_prev = t_now
        if kind == 0:
            band, strength, xc = payload
            e_before = float(observe(state.fields)[2])
            bp, bm = kick_blobs(rng, kinds[band], sigmas[band], nblobs[band], amp * strength,
                                xc, sheet_ratio, sheet_energy)
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
        img = np.asarray(imgs[field])
        # colour scale: slow-moving 99.5th percentile (time constant 4 s), so a beat shows
        # as a brightening rather than being normalised away frame by frame
        cur = max(np.percentile(np.abs(img), 99.5), 1e-12)
        vmax = cur if vmax is None else max(vmax + (cur - vmax) / (4.0 * fps), 0.5 * cur)
        u = img.T[::-1] / vmax
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
                onsets=(t_on, b_on, s_on), kick_types=kinds, pans=pans, edges=edges, sigmas=sigmas, nblobs=nblobs, fps=fps, diss=diss, nsteps=nsteps, fields=np.asarray(state.fields),
                duration=T, wall=time.perf_counter() - t0)


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
    res = run(sr, audio, a.out, n=a.n, fps=a.fps, rate=a.rate, amp=a.amp, memory=a.memory or None, nbands=a.bands, sigma_min=a.sigma_min, diss=a.diss, hyper=a.hyper, cfl=a.cfl, threshold=a.threshold, causal=a.causal,
              field=a.field, cmap=a.cmap, seed=a.seed, duration=duration, audio_path=audio_path,
              kick_types=a.kick_types, sheet_ratio=a.sheet_ratio, stereo=lr if a.stereo else None,
              stereo_gain=a.stereo_gain, stereo_bands=stereo_bands, sheet_energy=a.sheet_energy)
    t_on, b_on, _ = res["onsets"]
    print(f"wrote {a.out}  ({res['wall']:.0f} s wall, {res['nsteps']} steps, diss = {res['diss']:.2e}, "
          f"events late by {1e3 * res['late'].mean():.1f} ms mean / {1e3 * res['late'].max():.1f} ms max)")
    for b in range(len(res["sigmas"])):
        c, m = beat_contrast(res, b)
        print(f"  band {b} ({res['edges'][b]:.0f}-{res['edges'][b + 1]:.0f} Hz): "
              f"{int(np.sum(b_on == b))} onsets, beat contrast {c:+.2f} over {m}")


if __name__ == "__main__":
    main()
