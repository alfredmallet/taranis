"use strict";
// ===========================================================================
// music.js -- the PURE logic of the music page (music.html): everything that
// turns audio into kicks, and the audio clock into a step count, with no DOM,
// no Web Audio and no WebGPU in it. plans-webgpu/MUSIC_PLAN.md, phase M2.
//
//   musicBandPlan      blob width and count per frequency band, from the grid
//   MusicDetector      the causal per-band onset detector, one spectrum per frame
//   musicShouldPush    the rule that keeps the detector near its nominal frame rate
//   musicRng           a seeded uniform generator (mulberry32)
//   musicBlobsFor      one kick's blobs for ONE Elsasser channel
//   musicKickList      ... for both: the list Solver.kick takes for ONE onset
//   musicClickTrack    the built-in test signal: kick / snare / hi-hat, seamless loop
//   musicClock         the audio-clock controller: how many steps this frame
//
// It is the browser twin of examples/music_turbulence.py, the offline reference
// (which made the movies this page is judged against):
//   musicBandPlan  <->  band_plan
//   MusicDetector  <->  CausalOnsetDetector      (a line-for-line port)
//   musicBlobsFor  <->  blobs_for                (same rule, a different RNG)
// devtools/checkmusic.js holds the first two to numbers recorded FROM the python
// (devtools/fixtures/music_ref.json, written by gen_musicfixture.py), so a change on
// either side has to be made on both and the fixture regenerated.
//
// Loaded as a classic <script src="music.js"> after solver2d.js and before the
// page's inline script, which holds the audio and DOM glue. No module syntax, no
// build step, works from file://. It uses NOTHING from common.js / physics.js /
// solver2d.js, so node can evaluate it on its own -- that is how the gate runs it.
// Every top-level name starts `music` / `Music` / `MUSIC_`.
// ===========================================================================

// the periodic box the map is defined on (the offline script's LBOX)
const MUSIC_LBOX = 2 * Math.PI;
// widest blob: box-limited (energy peak at k = sqrt(3/2)/sigma ~ 2.7)
const MUSIC_SIGMA_MAX = 0.45;
// narrowest blob in grid spacings: its energy peak then sits a factor ~3 in k below where
// the offline dissipation rule takes over (k ~ n/6), leaving an unforced cascade under it
const MUSIC_SIGMA_MIN_DX = 3.7;
// neighbouring widths closer than this look the same: a blob's spectrum k^3 exp(-k^2 s^2)
// is as wide as its peak wavenumber
const MUSIC_SIGMA_STEP = 1.8;
// blobs per kick per channel in the bass band; the others scale as 1/sigma
const MUSIC_NBLOB_BASS = 3;
// the offline script's NBLOB_MAX: band_plan refuses above it. The page passes the
// solver's KICK_MAX (32) instead, which is what one kick() can actually hold.
const MUSIC_NBLOB_MAX = 128;

// (sigmas, nblobs) per band, bass first: band_plan in the offline script. Widths are
// log-spaced from MUSIC_SIGMA_MAX down to sigmaMin (default MUSIC_SIGMA_MIN_DX grid
// spacings, so it shrinks with n); the number of bands defaults to as many widths as fit
// with a factor MUSIC_SIGMA_STEP between neighbours (4 at 256^2, 5 at 512^2, 6 at 1024^2);
// the blob count goes as 1/sigma. `nblobMax` is the most one band may ask for.
function musicBandPlan(n, nbands, sigmaMin, nblobMax) {
  if (!(Number.isInteger(n) && n > 0)) throw new Error("musicBandPlan: n must be a positive integer, got " + n);
  const smin = (sigmaMin === undefined || sigmaMin === null) ? MUSIC_SIGMA_MIN_DX * MUSIC_LBOX / n : sigmaMin;
  if (!(typeof smin === "number" && smin > 0 && smin <= MUSIC_SIGMA_MAX))
    throw new Error("musicBandPlan: sigma_min must be in (0, " + MUSIC_SIGMA_MAX + "], got " + smin);
  let nb = nbands;
  if (nb === undefined || nb === null)
    nb = Math.max(1, Math.round(1 + Math.log(MUSIC_SIGMA_MAX / smin) / Math.log(MUSIC_SIGMA_STEP)));
  if (!(Number.isInteger(nb) && nb >= 1)) throw new Error("musicBandPlan: nbands must be an integer >= 1, got " + nbands);
  const cap = (nblobMax === undefined || nblobMax === null) ? MUSIC_NBLOB_MAX : nblobMax;
  const sigmas = [], nblobs = [];
  for (let i = 0; i < nb; i++) {
    const s = MUSIC_SIGMA_MAX * Math.pow(smin / MUSIC_SIGMA_MAX, i / Math.max(nb - 1, 1));
    sigmas.push(s);
    nblobs.push(Math.round(MUSIC_NBLOB_BASS * MUSIC_SIGMA_MAX / s));
  }
  const most = Math.max.apply(null, nblobs);
  if (most > cap)
    throw new Error("musicBandPlan: " + most + " blobs per kick in the top band exceeds the limit " + cap
                    + ": raise sigma_min or use fewer bands");
  return { sigmas: sigmas, nblobs: nblobs };
}

// ---------------------------------------------------------------------------
// numpy's summation and percentile, transcribed: the detector's decisions are
// comparisons of sums, so the port sums in numpy's ORDER (pairwise, eight
// accumulators) rather than left to right, and the gate can then ask for the
// same onsets and not merely similar ones.
// ---------------------------------------------------------------------------
function musicNpSum(a, lo, hi) {
  const n = hi - lo;
  if (n < 8) {
    let r = 0;
    for (let i = lo; i < hi; i++) r += a[i];
    return r;
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return musicNpSum(a, lo, lo + n2) + musicNpSum(a, lo + n2, hi);
}
// numpy.percentile(a[lo:hi], q), the default linear method
function musicPercentile(a, lo, hi, q) {
  const v = Float64Array.from(a.slice(lo, hi)).sort();
  const pos = (v.length - 1) * (q / 100), i = Math.floor(pos), t = pos - i;
  if (i + 1 >= v.length) return v[v.length - 1];
  const x = v[i], y = v[i + 1], d = y - x;
  return t >= 0.5 ? y - d * (1 - t) : x + d * t;
}

// ---------------------------------------------------------------------------
// the causal onset detector (CausalOnsetDetector in the offline script)
// ---------------------------------------------------------------------------
// push(mag, tNow) takes the magnitude spectrum of the last `nfft` samples under a
// Blackman window, scaled by 1/nfft -- what a Web Audio AnalyserNode with
// smoothingTimeConstant 0 holds, 10^(dB/20) of getFloatFrequencyData -- and the time
// the window ENDS. Bin k is frequency k*sr/nfft; the array may stop at nfft/2 - 1 (the
// analyser's frequencyBinCount) or carry the Nyquist bin as numpy's rfft does: every bin
// handed over counts toward the compression reference and no band reaches Nyquist.
//
// It returns the onsets it can now declare, [{ t, band, strength, tFire }]: the peak was
// one frame ago, `t` is that frame's analysis-window centre and `tFire` is tNow, when the
// caller knows. Per band, per frame:
//   flux     = sum over the band of max(0, c - c_prev),  c = log1p(100 mag / ref)
//   ref      = a peak-hold of the spectral maximum, decaying over refDecayS
//   s        = flux / (its 99th percentile over the trailing normS seconds)
//   an onset = s[i-1] > s[i-2], s[i-1] >= s[i], s[i-1] > trailing mean + threshold,
//              at least minGap after the band's last one, and not before warmupS
// All windows are counted in FRAMES at the nominal `fps`, so the caller must push at
// about that rate (musicShouldPush). A gap longer than gapS -- a hidden tab, a seek --
// only re-primes the previous spectrum: the flux across a gap is the music having moved
// on, not an onset.
class MusicDetector {
  constructor(o) {
    o = o || {};
    const def = (v, d) => (v === undefined || v === null ? d : v);
    this.sr = o.sr;
    this.nfft = def(o.nfft, 2048);
    this.nbands = def(o.nbands, 3);
    this.fps = def(o.fps, 60);
    this.threshold = def(o.threshold, 0.15);
    this.minGap = def(o.minGap, 0.07);
    this.warmupS = def(o.warmupS, 0.5);
    this.gapS = def(o.gapS, 0.1);
    const fmin = def(o.fmin, 40), fmax = Math.min(def(o.fmax, 10000), 0.5 * this.sr);
    if (!(this.sr > 0) || !Number.isInteger(this.nfft) || this.nfft < 4 || !Number.isInteger(this.nbands)
        || this.nbands < 1 || !(this.fps > 0) || !(fmin > 0) || !(fmax > fmin))
      throw new Error("MusicDetector: bad configuration (sr " + o.sr + ", nfft " + o.nfft + ", nbands "
                      + o.nbands + ", fps " + o.fps + ", fmin " + fmin + ", fmax " + fmax + ")");
    // band edges: nbands log-spaced intervals of [fmin, fmax]; band b holds the bins with
    // edges[b] <= k*sr/nfft < edges[b+1], a contiguous run [k0, k1)
    this.edges = [];
    for (let b = 0; b <= this.nbands; b++)
      this.edges.push(b === 0 ? fmin : b === this.nbands ? fmax : fmin * Math.pow(fmax / fmin, b / this.nbands));
    const nbin = this.nfft / 2 + 1, df = this.sr / this.nfft;
    this.bins = [];
    for (let b = 0; b < this.nbands; b++) {
      let k0 = nbin, k1 = 0;
      for (let k = 0; k < nbin; k++) {
        const f = k * df;
        if (f >= this.edges[b] && f < this.edges[b + 1]) { if (k < k0) k0 = k; k1 = k + 1; }
      }
      this.bins.push(k0 < k1 ? [k0, k1] : [0, 0]);
    }
    this.nmean = Math.max(1, Math.round(def(o.meanS, 0.20) * this.fps));
    this.nnorm = Math.max(1, Math.round(def(o.normS, 20) * this.fps));
    this.decay = Math.exp(-1 / (def(o.refDecayS, 30) * this.fps));
    this.reset();
  }
  reset() {
    this.ref = 0;
    this.prev = null;
    this.tLast = null;
    this.hist = [];
    for (let b = 0; b < this.nbands; b++) this.hist.push([]);
    this.lastFire = new Array(this.nbands).fill(-Infinity);
    this.flux = null;
  }
  push(mag, tNow) {
    const n = mag.length;
    if (n < this.bins[this.nbands - 1][1])
      throw new Error("MusicDetector.push: " + n + " bins, the top band needs " + this.bins[this.nbands - 1][1]);
    let mx = 0;
    for (let k = 0; k < n; k++) if (mag[k] > mx) mx = mag[k];
    this.ref = Math.max(this.ref * this.decay, mx, 1e-30);
    const comp = new Float64Array(n);
    for (let k = 0; k < n; k++) comp[k] = Math.log1p(100 * mag[k] / this.ref);
    const gap = this.tLast !== null && tNow - this.tLast > this.gapS;
    this.tLast = tNow;
    if (this.prev === null || gap) { this.prev = comp; return []; }
    if (!this.flux || this.flux.length !== n) this.flux = new Float64Array(n);
    const flux = this.flux, prev = this.prev;
    for (let k = 0; k < n; k++) { const d = comp[k] - prev[k]; flux[k] = d > 0 ? d : 0; }
    this.prev = comp;
    const out = [];
    for (let b = 0; b < this.nbands; b++) {
      const h = this.hist[b];
      h.push(musicNpSum(flux, this.bins[b][0], this.bins[b][1]));
      // only the last nnorm values are ever read (and nmean + 1 < nnorm of them twice)
      if (h.length > 2 * this.nnorm + 8) h.splice(0, h.length - this.nnorm - 4);
      const L = h.length;
      if (L < 3 || tNow < this.warmupS) continue;
      const q = musicPercentile(h, Math.max(0, L - this.nnorm), L, 99) + 1e-30;
      const s0 = h[L - 3] / q, s1 = h[L - 2] / q, s2 = h[L - 1] / q;   // s1 is the candidate
      const m0 = Math.max(0, L - 1 - this.nmean);
      const mean = musicNpSum(h, m0, L - 1) / (L - 1 - m0) / q;         // trailing, ends at it
      const tPeak = tNow - 1 / this.fps;
      if (s1 > s0 && s1 >= s2 && s1 > mean + this.threshold && tPeak - this.lastFire[b] >= this.minGap) {
        this.lastFire[b] = tPeak;
        out.push({ t: tPeak - 0.5 * this.nfft / this.sr, band: b, strength: Math.min(s1, 1), tFire: tNow });
      }
    }
    return out;
  }
}

// Feed the detector this frame? Its windows are counted in frames at `fps`, so a 120 Hz
// display must not push twice as often: a frame is taken once 3/4 of a nominal interval
// has passed since the last one taken. 60 Hz -> every frame; 120 Hz -> every other;
// 144 Hz -> every other (72 Hz, 20% fast). Below `fps` every frame is taken and the
// detector simply sees fewer of them, which costs onsets (MUSIC_PLAN M0.b: recall 0.69 at
// 60 Hz, 0.44 at 30).
function musicShouldPush(tNow, tLast, fps) {
  if (tLast === null || tLast === undefined) return true;
  const d = tNow - tLast;
  return !(d >= 0) || d >= 0.75 / fps;      // a clock that went backwards: take it
}

// ---------------------------------------------------------------------------
// blobs
// ---------------------------------------------------------------------------
// a seeded uniform generator on [0, 1): mulberry32. NOT the offline script's numpy
// stream -- positions differ from the movies', the distribution does not.
function musicRng(seed) {
  let a = (seed | 0) >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// one kick's blobs for ONE channel (blobs_for): `nblob` gaussians of width `sigma` at
// uniform positions in the box, each of amplitude +-amp with a fair random sign. Three
// draws per blob, in the order x, y, sign.
function musicBlobsFor(rng, sigma, nblob, amp, pol) {
  const out = [];
  for (let i = 0; i < nblob; i++) {
    const x = rng() * MUSIC_LBOX, y = rng() * MUSIC_LBOX, sgn = rng() < 0.5 ? -1 : 1;
    out.push({ x: x, y: y, sigma: sigma, amp: sgn * amp, pol: pol });
  }
  return out;
}
// the list Solver.kick takes for ONE onset of band `band`: the z+ set, then an
// INDEPENDENT z- set (a second run of draws, never a copy), amplitude amp * strength.
function musicKickList(rng, plan, band, amp, strength) {
  const s = plan.sigmas[band], n = plan.nblobs[band], a = amp * strength;
  return musicBlobsFor(rng, s, n, a, 1).concat(musicBlobsFor(rng, s, n, a, -1));
}

// ---------------------------------------------------------------------------
// the built-in click track
// ---------------------------------------------------------------------------
// A kick on every beat, a snare on beats 2 and 4 of each bar, a hi-hat on every
// off-beat eighth -- the offline script's click track without its closing kick-less
// quarter, because this one LOOPS: `beats` whole beats (a multiple of 4), and every
// sound's tail wraps round the end, so the loop has no seam and an onset that straddles
// it is still an onset. Deterministic: the noise is musicRng(seed), the filters are two
// biquads, there is no other state.
//   kick   a sine sweeping 105 -> 45 Hz, 120 ms decay     (the bass band)
//   snare  band-passed noise (400-3000 Hz) + a 220 Hz tone, ~55 ms
//   hat    high-passed noise (6 kHz), 20 ms
// Returns { samples (Float32Array, peak 0.8), sr, bpm, beat, duration, events }, with
// events = [{ t, kind }] sorted by time, t in seconds from the start of the loop.
function musicBiquad(x, b0, b1, b2, a1, a2) {
  const y = new Float64Array(x.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}
// RBJ cookbook biquads, Q = 1/sqrt(2)
function musicHighpass(x, fc, sr) {
  const w = 2 * Math.PI * fc / sr, c = Math.cos(w), al = Math.sin(w) / Math.SQRT2, a0 = 1 + al;
  return musicBiquad(x, (1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0, -2 * c / a0, (1 - al) / a0);
}
function musicLowpass(x, fc, sr) {
  const w = 2 * Math.PI * fc / sr, c = Math.cos(w), al = Math.sin(w) / Math.SQRT2, a0 = 1 + al;
  return musicBiquad(x, (1 - c) / 2 / a0, (1 - c) / a0, (1 - c) / 2 / a0, -2 * c / a0, (1 - al) / a0);
}
function musicClickTrack(o) {
  o = o || {};
  const sr = o.sr === undefined ? 44100 : o.sr, bpm = o.bpm === undefined ? 120 : o.bpm;
  const beats = o.beats === undefined ? 16 : o.beats;
  if (!(sr > 0) || !(bpm > 0) || !isFinite(bpm) || !(Number.isInteger(beats) && beats > 0 && beats % 4 === 0))
    throw new Error("musicClickTrack: need sr > 0, bpm > 0 and beats a positive multiple of 4 (sr " + sr
                    + ", bpm " + bpm + ", beats " + beats + ")");
  const beat = 60 / bpm, n = Math.round(beats * beat * sr), rng = musicRng(o.seed === undefined ? 1 : o.seed);
  const m = Math.round(0.4 * sr);                       // each sound is 0.4 s long at most
  const kick = new Float64Array(m), snare = new Float64Array(m), hat = new Float64Array(m);
  const noiseS = new Float64Array(m), noiseH = new Float64Array(m);
  for (let i = 0; i < m; i++) { noiseS[i] = 2 * rng() - 1; }
  for (let i = 0; i < m; i++) { noiseH[i] = 2 * rng() - 1; }
  const bp = musicLowpass(musicHighpass(noiseS, 400, sr), 3000, sr), hp = musicHighpass(noiseH, 6000, sr);
  for (let i = 0; i < m; i++) {
    const t = i / sr;
    kick[i] = Math.sin(2 * Math.PI * (45 * t + 60 * 0.05 * (1 - Math.exp(-t / 0.05)))) * Math.exp(-t / 0.12);
    snare[i] = 0.9 * bp[i] * Math.exp(-t / 0.06) + 0.4 * Math.sin(2 * Math.PI * 220 * t) * Math.exp(-t / 0.05);
    hat[i] = 0.5 * hp[i] * Math.exp(-t / 0.02);
  }
  const out = new Float64Array(n), events = [];
  const add = (t0, sig, kind) => {
    const i0 = Math.round(t0 * sr);
    for (let i = 0; i < sig.length; i++) out[(i0 + i) % n] += sig[i];   // the tail wraps: no seam
    events.push({ t: i0 / sr, kind: kind });
  };
  for (let i = 0; i < beats; i++) {
    add(i * beat, kick, "kick");
    if (i % 4 === 1 || i % 4 === 3) add(i * beat, snare, "snare");
    add((i + 0.5) * beat, hat, "hat");
  }
  let pk = 0;
  for (let i = 0; i < n; i++) pk = Math.max(pk, Math.abs(out[i]));
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = 0.8 * out[i] / pk;
  events.sort((a, b) => a.t - b.t);
  return { samples: samples, sr: sr, bpm: bpm, beat: beat, duration: n / sr, events: events };
}

// ---------------------------------------------------------------------------
// the audio clock
// ---------------------------------------------------------------------------
// Simulation time follows the audio:  target = t0 + rate * (audioT - a0).  Each frame
// the page hands over what it has -- the audio clock now, and the NEWEST simulation time
// and dt a stats readback returned -- and gets back how many steps to take. `st` is the
// controller's state, mutated in place: { init, t0, a0, lastA, rate, behindUntil }.
//
//   n       steps this frame, 0 .. maxSteps
//   target  the simulation time the audio clock asks for
//   lag     (target - simT) / rate: seconds of AUDIO the simulation is behind (+) or ahead (-)
//   rebased the clock was re-anchored this frame (t0 <- simT at this audio time)
//   behind  the "cannot keep up" indicator: lit for behindHold seconds of audio after a
//           rebase forced by falling behind
//
// The step count is ceil((target - simT) / dt): the frame ends at or just past the audio
// clock, by less than one step. ONE exception. In a quiet passage the CFL dt can be many
// frames long (its floor is cfl*dx/0.1, six frames at 256^2), and rounding up would then
// put the simulation that far AHEAD, where it must sit still until the audio catches up --
// and a kick landing in that wait would hang frozen on screen for up to six frames. So a
// step that would overshoot by more than `maxAhead` (one frame of simulation time) is not
// taken yet: n rounds DOWN instead, the simulation trails by less than one such step, and
// the first step after a kick -- which the solver takes at a freshly computed dt -- moves
// it at once.
//
// dt is whatever the last readback said, and after a kick that is a PRE-kick dt -- too
// LARGE, since a kick only speeds the flow up. Nothing here loops on it: the count it gives
// is taken (it can only undershoot), and the next frame's readback corrects it. The one
// thing a kick does change: on a frame that kicked (`inp.kicked`), if the simulation is
// behind the clock at all, at least ONE step is taken even where the stale dt says none
// fits -- the solver computes that step's dt afresh, so it moves the kicked flow at once
// and gives the next frame a dt worth dividing by. Rebase (never catch up) when the simulation is more than lagMax seconds of audio
// away from the clock in either direction, when the clock runs backwards, when `rate`
// changes, on the first frame after a stop, and after more than lagMax with no frame at
// all (a hidden tab). Only the first of those is "behind".
function musicClock(st, inp) {
  const res = { n: 0, target: inp.simT, lag: 0, rebased: false, behind: false };
  if (!inp.running || !isFinite(inp.audioT) || !isFinite(inp.simT) || !(inp.rate > 0)) {
    st.init = false;
    return res;
  }
  const lagMax = inp.lagMax === undefined ? 0.25 : inp.lagMax;
  const hold = inp.behindHold === undefined ? 1 : inp.behindHold;
  const rebase = () => { st.t0 = inp.simT; st.a0 = inp.audioT; st.rate = inp.rate; st.init = true; res.rebased = true; };
  // (the last clause: no frame at all for longer than lagMax -- a hidden tab -- is an absence,
  // not a failure to keep up, so it re-anchors here, before the lag test could call it one)
  if (!st.init || !(inp.audioT >= st.lastA) || st.rate !== inp.rate || inp.audioT - st.lastA > lagMax) rebase();
  st.lastA = inp.audioT;
  let target = st.t0 + inp.rate * (inp.audioT - st.a0), lag = (target - inp.simT) / inp.rate;
  if (Math.abs(lag) > lagMax) {
    if (lag > 0) st.behindUntil = inp.audioT + hold;      // fell behind, as opposed to ran ahead
    rebase();
    target = inp.simT; lag = 0;
  }
  res.target = target; res.lag = lag;
  res.behind = st.behindUntil !== undefined && inp.audioT < st.behindUntil;
  const gap = target - inp.simT;
  if (!(gap > 0)) return res;
  const maxSteps = inp.maxSteps === undefined ? 64 : inp.maxSteps;
  if (!(inp.dt > 0) || !isFinite(inp.dt)) { res.n = Math.min(1, maxSteps); return res; }   // dt not known yet
  const maxAhead = inp.maxAhead === undefined ? Infinity : inp.maxAhead;   // default: always round up
  let n = Math.ceil(gap / inp.dt);
  if (n * inp.dt - gap > maxAhead) n = Math.floor(gap / inp.dt);
  if (inp.kicked && n < 1) n = 1;
  res.n = Math.max(0, Math.min(maxSteps, n));
  return res;
}
