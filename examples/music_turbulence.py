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
def click_track(duration=20.0, bpm=120.0, sr=44100, seed=0):
    # kick on every beat, snare on beats 2 and 4, hi-hat on the off-beat eighths; the last
    # quarter of the track drops the kick so "what happens when the beat stops" is in it.
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    out = np.zeros_like(t)
    beat = 60.0 / bpm

    def add(t0, sig):
        i0 = int(t0 * sr)
        n = min(len(sig), len(out) - i0)
        if n > 0:
            out[i0:i0 + n] += sig[:n]

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
            add(i * beat, snare)
        add((i + 0.5) * beat, hat)
    return sr, 0.8 * out / np.max(np.abs(out))


def _highpass(fc, sr):
    from scipy.signal import butter
    return butter(4, fc / (0.5 * sr), "high")


def _bandpass(f0, f1, sr):
    from scipy.signal import butter
    return butter(2, [f0 / (0.5 * sr), f1 / (0.5 * sr)], "band")


def load_wav(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return sr, x / max(np.max(np.abs(x)), 1e-30)


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
    Written as a per-frame loop over explicit state so that it ports to JS line for line.

    Returns (times, bands, strengths, edges, fire_times): `times` are the analysis-window
    centres of the peak frames (comparable with `detect_onsets`), `fire_times` when the
    page would KNOW -- half a window plus one frame later.
    """
    win = np.blackman(nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    edges = np.geomspace(fmin, min(fmax, 0.5 * sr), nbands + 1)
    sels = [(freqs >= edges[b]) & (freqs < edges[b + 1]) for b in range(nbands)]
    nframes = int(len(x) / sr * fps)
    nmean = max(1, int(round(mean_s * fps)))
    nnorm = max(1, int(round(norm_s * fps)))
    decay = np.exp(-1.0 / (ref_decay_s * fps))

    ref, prev = 0.0, None
    hist = [[] for _ in range(nbands)]          # raw band flux, all frames so far
    last_fire = [-np.inf] * nbands
    times, bands, strengths, fires = [], [], [], []
    for i in range(nframes):
        end = int(round((i + 1) / fps * sr))    # the window ENDS now
        if end < nfft:
            continue
        mag = np.abs(np.fft.rfft(win * x[end - nfft:end])) / nfft
        ref = max(ref * decay, float(mag.max()), 1e-30)
        comp = np.log1p(100.0 * mag / ref)
        if prev is None:
            prev = comp
            continue
        flux = np.maximum(comp - prev, 0.0)
        prev = comp
        t_now = end / sr
        for b in range(nbands):
            h = hist[b]
            h.append(float(flux[sels[b]].sum()))
            if len(h) < 3 or t_now < warmup_s:
                continue
            q = np.percentile(h[-nnorm:], 99.0) + 1e-30
            s0, s1, s2 = h[-3] / q, h[-2] / q, h[-1] / q      # s1 is the candidate peak
            mean = np.mean(h[-1 - nmean:-1]) / q              # trailing, ends at the candidate
            t_peak = t_now - 1.0 / fps
            if (s1 > s0 and s1 >= s2 and s1 > mean + threshold
                    and t_peak - last_fire[b] >= min_gap_s):
                last_fire[b] = t_peak
                times.append(t_peak - 0.5 * nfft / sr); bands.append(b)
                strengths.append(min(s1, 1.0)); fires.append(t_now)
    order = np.argsort(times, kind="stable")
    return (np.array(times)[order], np.array(bands, dtype=int)[order],
            np.array(strengths)[order], edges, np.array(fires)[order])


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

    @jax.jit
    def kick(fields, blobs_p, blobs_m):
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


def blobs_for(rng, sigma, nblob, amp):
    # one kick's blobs for ONE channel: positions uniform, signs random. Called once per
    # channel, which is what makes the z+ and z- sets independent.
    out = np.zeros((nblob, 4))
    out[:, 0:2] = rng.uniform(0.0, LBOX, size=(nblob, 2))
    out[:, 2] = sigma
    out[:, 3] = amp * rng.choice([-1.0, 1.0], size=nblob)
    return out


def pad(blobs):
    out = np.zeros((NBLOB_MAX, 4))
    out[:, 2] = 1.0                      # unused slots: sigma 1, amplitude 0
    n = min(len(blobs), NBLOB_MAX)
    out[:n] = blobs[:n]
    return out


# ---------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------
def run(sr, audio, out, *, n=256, fps=30, rate=1.0, amp=1.0, memory=1.0, nbands=None, diss=None,
        hyper=2, cfl=0.4, threshold=0.15, causal=False, field="speed", sigma_min=None, seed=0,
        duration=None, audio_path=None, quiet=False):
    import imageio
    from matplotlib import colormaps

    if duration is not None:
        audio = audio[:int(duration * sr)]
    T = len(audio) / sr
    sigmas, nblobs = band_plan(n, nbands, sigma_min)
    nbands = len(sigmas)
    if causal:
        t_on, b_on, s_on, edges, _ = detect_onsets_causal(sr, audio, nbands=nbands, threshold=threshold)
    else:
        t_on, b_on, s_on, edges = detect_onsets(sr, audio, nbands=nbands, threshold=threshold)
    if not quiet:
        for b in range(nbands):
            print(f"  band {b}: {edges[b]:6.0f}-{edges[b + 1]:6.0f} Hz -> sigma {sigmas[b]:.4f} "
                  f"(k_peak {np.sqrt(1.5) / sigmas[b]:5.1f}), {nblobs[b]:3d} blobs/kick, "
                  f"{np.sum(b_on == b) / T:4.1f} kicks/s", flush=True)
    nframes = int(T * fps)
    if diss is None:
        # hyperviscous rate nu k^(2 hyper) = 3 amp at k = n/6, i.e. ~50 amp at the dealiasing
        # cutoff n/3 (hyper 2): enough that the spectrum falls ~3 decades over the last
        # octave. A cutoff rate of ~5 left a flat, upturning tail -- grid-scale noise that
        # looks like a CFL problem but does not respond to the timestep. Scales with amp
        # because the strain rate the dissipation has to beat does.
        diss = 3.0 * amp / (n / 6.0) ** (2 * hyper)

    step = Stepper(n, rate / (4.0 * fps), diss, hyper, cfl)
    kick, observe = make_kick(step.p, step.kg)
    state = jr.initialize(lambda x, y: jnp.zeros((2, 1, n, n)), step.p)

    # events in audio seconds: (time, kind, payload); kind 0 = kick, 1 = frame, so a kick
    # that coincides with a frame is in that frame
    events = [(float(t), 0, (int(b), float(s))) for t, b, s in zip(t_on, b_on, s_on) if t < T]
    events += [(i / fps, 1, i) for i in range(nframes)]
    events.sort(key=lambda ev: (ev[0], ev[1]))

    rng = np.random.default_rng(seed)
    cmap, cmap_mag = colormaps["RdBu_r"], colormaps["magma"]
    signed = field not in ("speed", "bperp")
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
            band, strength = payload
            e_before = float(observe(state.fields)[2])
            bp = blobs_for(rng, sigmas[band], nblobs[band], amp * strength)
            bm = blobs_for(rng, sigmas[band], nblobs[band], amp * strength)
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
        rgb = (cmap_mag(np.clip(u, 0.0, 1.0)) if signed is False else
               cmap(0.5 + 0.5 * np.clip(u, -1.0, 1.0)))[..., :3]
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
                onsets=(t_on, b_on, s_on), edges=edges, sigmas=sigmas, nblobs=nblobs, fps=fps, diss=diss, nsteps=nsteps, fields=np.asarray(state.fields),
                duration=T, wall=time.perf_counter() - t0)


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
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="onset threshold above the local mean flux (band 99th percentile = 1); "
                         "raise to keep only the stronger hits")
    ap.add_argument("--causal", action="store_true",
                    help="use the live-capable onset detector (detect_onsets_causal)")
    ap.add_argument("--bpm", type=float, default=120.0, help="click track tempo")
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: whole file; 20 for the click track)")
    ap.add_argument("--field", choices=FIELDS, default="speed")
    ap.add_argument("--diss", type=float, default=None,
                    help="hyperviscosity nu = eta (default: 3 amp/(n/6)^4, see run())")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.wav is None:
        sr, audio = click_track(duration=a.duration or 20.0, bpm=a.bpm)
        audio_path = os.path.splitext(a.out)[0] + "_click.wav"
        wavfile.write(audio_path, sr, (32767 * audio).astype(np.int16))
        duration = None
    else:
        sr, audio = load_wav(a.wav)
        audio_path, duration = a.wav, a.duration
    res = run(sr, audio, a.out, n=a.n, fps=a.fps, rate=a.rate, amp=a.amp, memory=a.memory or None, nbands=a.bands, sigma_min=a.sigma_min, diss=a.diss, cfl=a.cfl, threshold=a.threshold, causal=a.causal,
              field=a.field, seed=a.seed, duration=duration, audio_path=audio_path)
    t_on, b_on, _ = res["onsets"]
    print(f"wrote {a.out}  ({res['wall']:.0f} s wall, {res['nsteps']} steps, diss = {res['diss']:.2e}, "
          f"events late by {1e3 * res['late'].mean():.1f} ms mean / {1e3 * res['late'].max():.1f} ms max)")
    for b in range(len(res["sigmas"])):
        c, m = beat_contrast(res, b)
        print(f"  band {b} ({res['edges'][b]:.0f}-{res['edges'][b + 1]:.0f} Hz): "
              f"{int(np.sum(b_on == b))} onsets, beat contrast {c:+.2f} over {m}")


if __name__ == "__main__":
    main()
