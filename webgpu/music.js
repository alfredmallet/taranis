"use strict";
// ===========================================================================
// music.js -- the PURE logic of the music page (music.html): everything that
// turns audio into kicks, and the audio clock into a step count, with no DOM,
// no Web Audio and no WebGPU in it. plans-webgpu/MUSIC_PLAN.md, phase M2.
//
//   musicBandPlan      blob width and count per frequency band, from the grid
//   musicWindowAspect / musicBoxFor / musicCutoffK / musicGridFor / musicRungFor /
//   musicAutoRung      the window-shaped box, its grid and the auto rung (MUSIC_STAGE_PLAN S2)
//   musicSheet*        the maintained current-sheet equilibrium (MUSIC_STAGE_PLAN S4)
//   MusicDetector      the causal per-band onset detector, one spectrum per frame
//   musicShouldPush    the rule that keeps the detector near its nominal frame rate
//   musicRng           a seeded uniform generator (mulberry32)
//   musicBlobsFor      one kick's blobs for ONE Elsasser channel
//   musicKickList      ... for both: the list Solver.kick takes for ONE onset
//   musicKickTypesFor / musicKickBlobs   the kick TYPE per band (MUSIC_EFFECTS_PLAN E1)
//   musicPan / musicPanX   an onset's stereo position, and where that puts its blobs (E2)
//   musicPulse*        the camera pulse: the zoom a bass onset lifts and time relaxes (E3)
//   musicWalk*         the in-plane mean field's random walk on bass onsets (E4)
//   musicClickTrack    the built-in test signal: kick / snare / hi-hat, seamless loop
//   musicClock         the audio-clock controller: how many steps this frame
//   musicTabConstraints / musicInputConstraints   what a live capture is asked for
//   musicRouting       which sources the page may play, and so can delay-compensate
//   musicStartError    what to tell the visitor when a source will not start
//   musicTabAudioLikely  whether this browser captures tab audio at all (wording only)
//   musicSync          how long to delay the sound and hold the kicks, from the output latency
//   MUSIC_CMAPS        the page's extra colormaps: magma, brat and brat_r
//
// It is the browser twin of examples/music_turbulence.py, the offline reference
// (which made the movies this page is judged against):
//   musicBandPlan  <->  band_plan
//   MusicDetector  <->  CausalOnsetDetector      (a line-for-line port)
//   musicBlobsFor  <->  blobs_for                (same rule, a different RNG)
//   musicKickTypesFor, musicKickBlobs  <->  kick_types_for, kick_blobs   (likewise)
//   musicPan  <->  onset_pans' formula    musicPulse*  <->  pulse_zoom
//   musicWalk*  <->  the --field-walk / --ratio-walk rules in run()
//   musicBoxFor, musicCutoffK  <->  box_for, cutoff_k
//   musicSheetPsi0, musicSheetEq, musicSheetEnergy  <->  sheet_psi0, sheet_equilibrium,
//                                                        sheet_energy_eq (the bump profile)
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
// `n` is the grid: one number (n x n) or [nx, ny]. In a rectangular box (Lx, Ly; default the
// square 2 pi box) or on a grid with nx != ny (MUSIC_STAGE_PLAN S2): the narrowest blob is
// MUSIC_SIGMA_MIN_DX of the LARGER of the two grid spacings (it has to be resolved along
// both axes); the widest stays MUSIC_SIGMA_MAX, a fraction of the short side, which is
// always 2 pi; and the counts are multiplied by the box's area over (2 pi)^2 -- exactly 1
// for the square box -- so a wider box is not sparser. Counts round half to even, as
// numpy's rint does.
function musicRint(v) {
  const f = Math.floor(v), d = v - f;
  return d < 0.5 ? f : d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1);
}
function musicBandPlan(n, nbands, sigmaMin, nblobMax, Lx, Ly) {
  const pos = v => Number.isInteger(v) && v > 0, pair = Array.isArray(n);
  if (!(pair ? n.length === 2 && pos(n[0]) && pos(n[1]) : pos(n)))
    throw new Error("musicBandPlan: n must be a positive integer or [nx, ny], got " + n);
  const nx = pair ? n[0] : n, ny = pair ? n[1] : n;
  const lx = (Lx === undefined || Lx === null) ? MUSIC_LBOX : Lx, ly = (Ly === undefined || Ly === null) ? MUSIC_LBOX : Ly;
  if (!(typeof lx === "number" && typeof ly === "number" && lx > 0 && ly > 0 && isFinite(lx) && isFinite(ly)))
    throw new Error("musicBandPlan: the box must be two finite positive lengths, got " + Lx + ", " + Ly);
  const smin = (sigmaMin === undefined || sigmaMin === null)
    ? Math.max(MUSIC_SIGMA_MIN_DX * lx / nx, MUSIC_SIGMA_MIN_DX * ly / ny) : sigmaMin;
  if (!(typeof smin === "number" && smin > 0 && smin <= MUSIC_SIGMA_MAX))
    throw new Error("musicBandPlan: sigma_min must be in (0, " + MUSIC_SIGMA_MAX + "], got " + smin);
  let nb = nbands;
  if (nb === undefined || nb === null)
    nb = Math.max(1, Math.round(1 + Math.log(MUSIC_SIGMA_MAX / smin) / Math.log(MUSIC_SIGMA_STEP)));
  if (!(Number.isInteger(nb) && nb >= 1)) throw new Error("musicBandPlan: nbands must be an integer >= 1, got " + nbands);
  const cap = (nblobMax === undefined || nblobMax === null) ? MUSIC_NBLOB_MAX : nblobMax;
  const area = (lx / MUSIC_LBOX) * (ly / MUSIC_LBOX);
  const sigmas = [], nblobs = [];
  for (let i = 0; i < nb; i++) {
    const s = MUSIC_SIGMA_MAX * Math.pow(smin / MUSIC_SIGMA_MAX, i / Math.max(nb - 1, 1));
    sigmas.push(s);
    nblobs.push(musicRint(MUSIC_NBLOB_BASS * MUSIC_SIGMA_MAX / s * area));
  }
  const most = Math.max.apply(null, nblobs);
  if (most > cap)
    throw new Error("musicBandPlan: " + most + " blobs per kick in the top band exceeds the limit " + cap
                    + ": raise sigma_min or use fewer bands");
  return { sigmas: sigmas, nblobs: nblobs };
}

// ---------------------------------------------------------------------------
// the box takes the window's shape (MUSIC_STAGE_PLAN S2)
// ---------------------------------------------------------------------------
// the most elongated box: a window longer than this is not followed (the stage shows the box whole, with sides left over)
const MUSIC_ASPECT_MAX = 2.4;
// The picture's aspect (width over height) for a window of w x h CSS pixels: w/h clamped
// to [1/MUSIC_ASPECT_MAX, MUSIC_ASPECT_MAX]. `mode` "square" -- and a window that is not
// two finite positive numbers -- is 1, the shipped square box.
function musicWindowAspect(w, h, mode) {
  if (mode === "square" || !(w > 0) || !(h > 0) || !isFinite(w) || !isFinite(h)) return 1;
  return Math.min(MUSIC_ASPECT_MAX, Math.max(1 / MUSIC_ASPECT_MAX, w / h));
}
// [Lx, Ly] of the box for a picture of that aspect (box_for in the offline script): the
// SHORT side is 2 pi and the long side 2 pi * max(aspect, 1/aspect). aspect >= 1 is
// landscape (long in x), aspect < 1 portrait (long in y); 1 is exactly the square 2 pi box.
function musicBoxFor(aspect) {
  if (!(typeof aspect === "number" && aspect > 0 && isFinite(aspect)))
    throw new Error("musicBoxFor: aspect must be a finite positive number, got " + aspect);
  return aspect >= 1 ? [MUSIC_LBOX * aspect, MUSIC_LBOX] : [MUSIC_LBOX, MUSIC_LBOX / aspect];
}
// The smaller of the two dealiasing cutoffs, in k (cutoff_k): the solver's 2/3 mask is in
// mode INDEX (n/3 along each axis), so in a rectangular box or grid the two axes stop at
// different wavenumbers, and the coarser one is the one the dissipation has to resolve.
// n/3 exactly for the square 2 pi box.
function musicCutoffK(nx, ny, Lx, Ly) {
  return Math.min((nx / 3) * (2 * Math.PI / Lx), (ny / 3) * (2 * Math.PI / Ly));
}
// The grid ladder, long side x short side: light 256 x 128, standard 256 x 256, sharp
// 512 x 256. Returns [nx, ny] for a box (Lx, Ly): the long side of the grid lies along the
// long side of the box (a tall box gets ny > nx). A SQUARE box takes square grids: light
// 128 x 128, standard 256 x 256, and sharp is standard (512 x 512 is not offered: it is
// four times the work of the shipped page).
const MUSIC_RUNGS = ["light", "standard", "sharp"];
function musicGridFor(rung, Lx, Ly) {
  if (MUSIC_RUNGS.indexOf(rung) < 0) throw new Error("musicGridFor: unknown rung " + rung);
  if (Lx === Ly) return rung === "light" ? [128, 128] : [256, 256];
  const ls = rung === "light" ? [256, 128] : rung === "standard" ? [256, 256] : [512, 256];
  return Lx > Ly ? ls : [ls[1], ls[0]];
}
// The rung the grid select asks for. `sel` is "auto", or a rung by name; `autoRung` is
// where auto stands (musicAutoRung). Auto is never sharp, whatever it is handed.
function musicRungFor(sel, autoRung) {
  if (sel === "auto" || MUSIC_RUNGS.indexOf(sel) < 0) return autoRung === "light" ? "light" : "standard";
  return sel;
}
// Where auto stands for the NEXT build. It starts on standard; if the run it has just
// watched was rebased for falling behind MORE than `limit` times in its first seconds
// (`nBehind`: the page counts them, over MUSIC_AUTO_WINDOW seconds of audio from Start) it
// drops ONE rung. It never climbs, and it is never sharp: light stays light whatever the
// count, and anything that is not light is treated as standard.
const MUSIC_AUTO_WINDOW = 10, MUSIC_AUTO_REBASES = 3;
function musicAutoRung(rung, nBehind, limit) {
  if (rung === "light") return "light";
  return nBehind > (limit === undefined ? MUSIC_AUTO_REBASES : limit) ? "light" : "standard";
}

// ---------------------------------------------------------------------------
// current sheets as a maintained background (MUSIC_STAGE_PLAN S4; the offline script's
// sheet_psi0 / sheet_equilibrium / sheet_energy_eq -- the `bump` profile only)
// ---------------------------------------------------------------------------
// psi0 of psi_eq = psi0 sech^2((x - Lx/2)/a) whose peak field is strength * amp:
// |B| = |d psi_eq/dx| = (2 psi0/a) sech^2(u) |tanh(u)| is largest where tanh^2(u) = 1/3,
// 4 psi0 / (3 sqrt(3) a) there. Hence psi0 = (3 sqrt(3)/4) a strength amp.
function musicSheetPsi0(strength, amp, a) {
  return 0.75 * Math.sqrt(3) * a * strength * amp;
}
// psi_eq(x), summed over the nearest periodic images so it is smooth across the box edge:
// ONE current sheet at x = Lx/2 (j = psi'' peaks there at -2 psi0 / a^2; the field reverses
// across it) with two return lobes of the opposite sign, a third as strong, either side; no
// net flux
function musicSheetEq(x, Lx, psi0, a) {
  let s = 0;
  for (let m = -1; m <= 1; m++) { const c = Math.cosh((x - 0.5 * Lx + m * Lx) / a); s += 1 / (c * c); }
  return psi0 * s;
}
// ... on the solver's real-space layout (row-major ix*ny + iy, x = ix Lx/nx), independent
// of y: what Solver.setICFromReal(null, .) takes
function musicSheetField(nx, ny, Lx, psi0, a) {
  const r = new Float32Array(nx * ny);
  for (let i = 0; i < nx; i++) r.fill(musicSheetEq(i * Lx / nx, Lx, psi0, a), i * ny, (i + 1) * ny);
  return r;
}
// volume-averaged energy (1/2)<|grad psi_eq|^2> of the equilibrium:
// (1/2)(2 psi0/a)^2 (a/Lx) int sech^4 tanh^2 du, the integral being 4/15
function musicSheetEnergy(psi0, a, Lx) {
  return 8 * psi0 * psi0 / (15 * a * Lx);
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
//
// `fluxFloor` (0 = off, the offline default; the page uses MUSIC_FLUX_FLOOR) is the noise
// floor: a band's normaliser is held at no less than fluxFloor times a peak-hold of that
// band's own flux, decaying over peakDecayS. Measured on a rock track followed by noise
// (4 bands, threshold 0.3): without it, noise 40-90 dB below the music gives ~9 onsets a
// second at strength 0.9, whatever its level; with 0.3 over 30 s, none at -60 dB and below
// and 5 a second at -40 dB, at the cost of 0.1% of the music's own onsets. What it cannot
// do: a source that is noise from its first sample has no louder past to be measured
// against (a microphone in a quiet room still kicks), and after a loud passage onsets
// weaker than fluxFloor of it are missed until the peak-hold has decayed.
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
    // the noise floor (flux_floor in the python; 0 = off, and then nothing differs): each
    // band keeps a slow peak-hold of its own raw flux, decaying over peakDecayS, and its
    // normaliser is never allowed below fluxFloor times that. Without it the running 99th
    // percentile promotes ANY non-zero floor to full strength once the music stops.
    this.fluxFloor = def(o.fluxFloor, 0);
    this.pdecay = Math.exp(-1 / (def(o.peakDecayS, 30) * this.fps));
    if (!(this.fluxFloor >= 0) || !(this.pdecay > 0))
      throw new Error("MusicDetector: bad configuration (fluxFloor " + o.fluxFloor + ", peakDecayS " + o.peakDecayS + ")");
    this.reset();
  }
  reset() {
    this.ref = 0;
    this.prev = null;
    this.tLast = null;
    this.hist = [];
    for (let b = 0; b < this.nbands; b++) this.hist.push([]);
    this.lastFire = new Array(this.nbands).fill(-Infinity);
    this.peak = new Array(this.nbands).fill(0);
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
      if (this.fluxFloor > 0) this.peak[b] = Math.max(this.peak[b] * this.pdecay, h[h.length - 1]);
      // only the last nnorm values are ever read (and nmean + 1 < nnorm of them twice)
      if (h.length > 2 * this.nnorm + 8) h.splice(0, h.length - this.nnorm - 4);
      const L = h.length;
      if (L < 3 || tNow < this.warmupS) continue;
      let q = musicPercentile(h, Math.max(0, L - this.nnorm), L, 99) + 1e-30;
      if (this.fluxFloor > 0) q = Math.max(q, this.fluxFloor * this.peak[b]);
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
// draws per blob, in the order x, y, sign. `box` = [Lx, Ly] (default the square 2 pi box):
// x is drawn over Lx and y over Ly.
function musicBlobsFor(rng, sigma, nblob, amp, pol, box) {
  const out = [], Lx = box ? box[0] : MUSIC_LBOX, Ly = box ? box[1] : MUSIC_LBOX;
  for (let i = 0; i < nblob; i++) {
    const x = rng() * Lx, y = rng() * Ly, sgn = rng() < 0.5 ? -1 : 1;
    out.push({ x: x, y: y, sigma: sigma, amp: sgn * amp, pol: pol });
  }
  return out;
}
// the list Solver.kick takes for ONE onset of band `band`: the z+ set, then an
// INDEPENDENT z- set (a second run of draws, never a copy), amplitude amp * strength.
// `o` (MUSIC_EFFECTS_PLAN; absent = the phase-1 kick above, draw for draw):
//   types        one kick type per band (musicKickTypesFor)
//   xc           the x the blobs gather about (stereo placement; null/undefined = uniform)
//   ratio        a sheet's length over its width (default MUSIC_SHEET_RATIO)
//   sheetEnergy  multiplies a sheet's energy (its amplitude by the square root)
//   box          [Lx, Ly] of the box (absent = the square 2 pi box)
function musicKickList(rng, plan, band, amp, strength, o) {
  const s = plan.sigmas[band], n = plan.nblobs[band], a = amp * strength;
  if (o === undefined || o === null) return musicBlobsFor(rng, s, n, a, 1).concat(musicBlobsFor(rng, s, n, a, -1));
  return musicKickBlobs(rng, o.types ? o.types[band] : "elsasser", s, n, a, o);
}

// ---------------------------------------------------------------------------
// kick types (MUSIC_EFFECTS_PLAN E1; kick_types_for / kick_blobs in the offline script)
// ---------------------------------------------------------------------------
// Every entry has peak |grad z| = amp in its channel, and phi += (z+ + z-)/2,
// psi += (z+ - z-)/2, so:
//   elsasser      independent sets in the two channels                 (the phase-1 kick)
//   vortex        the same set in both: phi only, a velocity kick
//   island        the same set, sign flipped in z-: psi only, a magnetic kick
//   dipole        a vortex kick made of PAIRS: two opposite-signed blobs MUSIC_DIPOLE_SEP
//                 sigma apart at a random orientation; the pair travels perpendicular to
//                 the line joining them. nblob pairs, so 2*nblob slots per channel
//   sheet         independent sets of anisotropic gaussians, width sigma and length
//                 ratio*sigma at a random orientation (Solver.kick's `ratio` / `theta`)
//   sheet-island  one set of sheets, sign flipped in z-: magnetic sheets
const MUSIC_KICK_TYPES = ["elsasser", "vortex", "island", "dipole", "sheet", "sheet-island"];
const MUSIC_DIPOLE_SEP = 1.5;      // separation of a dipole's two blobs, in sigma
const MUSIC_SHEET_RATIO = 4;       // a sheet's length over its width
const MUSIC_STEREO_SPAN = 0.4;     // hard left / right put a band's centre at (0.5 -+ this) of the box
const MUSIC_STEREO_JITTER = 0.12;  // gaussian scatter of the blobs about that centre, in box lengths
// One kick type per band, bass first: null/undefined (all `elsasser`), "variety" (bass
// dipole, treble sheet, vortex and island alternating in between), one type for every
// band, or a comma list / array with one entry per band.
function musicKickTypesFor(spec, nbands) {
  if (spec === undefined || spec === null) return new Array(nbands).fill("elsasser");
  let names = typeof spec === "string" ? spec.split(",").map(x => x.trim()) : Array.from(spec);
  if (names.length === 1 && names[0] === "variety") {
    names = ["dipole"];
    for (let i = 0; i < nbands - 2; i++) names.push(i % 2 === 0 ? "vortex" : "island");
    names.push("sheet");
    if (nbands === 1) names = names.slice(0, 1);
  } else if (names.length === 1) names = new Array(nbands).fill(names[0]);
  if (names.length !== nbands || names.some(x => MUSIC_KICK_TYPES.indexOf(x) < 0))
    throw new Error("musicKickTypesFor: need `variety`, one type, or " + nbands + " of " + MUSIC_KICK_TYPES.join(", ")
                    + " (one per band, low to high); got " + names.join(","));
  return names;
}
// a standard normal from two uniforms (Box-Muller; the cosine branch only)
function musicNormal(rng) {
  const u = 1 - rng(), v = rng();            // u in (0, 1]: the log is finite
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const musicWrap = (x, L) => { const l = L === undefined ? MUSIC_LBOX : L; return x - l * Math.floor(x / l); };
// one channel's set under stereo placement: musicBlobsFor's draws, then x REPLACED by a
// gaussian about xc (width MUSIC_STEREO_JITTER of the box's x side); y stays uniform
function musicPlaced(rng, sigma, nblob, amp, pol, xc, box) {
  const out = musicBlobsFor(rng, sigma, nblob, amp, pol, box), Lx = box ? box[0] : MUSIC_LBOX;
  if (xc !== undefined && xc !== null)
    for (const b of out) b.x = musicWrap(xc + MUSIC_STEREO_JITTER * Lx * musicNormal(rng), Lx);
  return out;
}
// the energy of two opposite-signed gaussian blobs `sep` sigma apart over that of ONE:
// the cross term of their gradients is -(1 - q) e^(-q) of a blob's, q = sep^2/4
function musicDipoleEnergy(sep) {
  const q = 0.25 * (sep === undefined ? MUSIC_DIPOLE_SEP : sep) * (sep === undefined ? MUSIC_DIPOLE_SEP : sep);
  return 2 * (1 - (1 - q) * Math.exp(-q));
}
function musicKickBlobs(rng, kind, sigma, nblob, amp, o) {
  o = o || {};
  const xc = o.xc, box = o.box || null, Lx = box ? box[0] : MUSIC_LBOX, Ly = box ? box[1] : MUSIC_LBOX;
  const flip = l => l.map(b => Object.assign({}, b, { amp: -b.amp, pol: -1 }));
  const same = l => l.map(b => Object.assign({}, b, { pol: -1 }));
  if (kind === "elsasser") return musicPlaced(rng, sigma, nblob, amp, 1, xc, box).concat(musicPlaced(rng, sigma, nblob, amp, -1, xc, box));
  if (kind === "sheet" || kind === "sheet-island") {
    const ratio = o.ratio === undefined ? MUSIC_SHEET_RATIO : o.ratio;
    const a = amp * Math.sqrt(o.sheetEnergy === undefined ? 1 : o.sheetEnergy);
    const one = pol => musicPlaced(rng, sigma, nblob, a, pol, xc, box)
      .map(b => Object.assign(b, { ratio: ratio, theta: Math.PI * rng() }));
    if (kind === "sheet") return one(1).concat(one(-1));
    const p = one(1);
    return p.concat(flip(p));
  }
  let b = musicPlaced(rng, sigma, nblob, amp, 1, xc, box);
  if (kind === "dipole") {
    const pair = [[], []];
    for (const q of b) {
      const th = 2 * Math.PI * rng(), ox = 0.5 * MUSIC_DIPOLE_SEP * sigma * Math.cos(th),
            oy = 0.5 * MUSIC_DIPOLE_SEP * sigma * Math.sin(th);
      pair[0].push(Object.assign({}, q, { x: musicWrap(q.x + ox, Lx), y: musicWrap(q.y + oy, Ly) }));
      pair[1].push(Object.assign({}, q, { x: musicWrap(q.x - ox, Lx), y: musicWrap(q.y - oy, Ly), amp: -q.amp }));
    }
    b = pair[0].concat(pair[1]);
  }
  if (kind === "island") return b.concat(flip(b));
  if (kind === "vortex" || kind === "dipole") return b.concat(same(b));
  throw new Error("musicKickBlobs: unknown kick type " + kind);
}

// ---------------------------------------------------------------------------
// stereo placement (MUSIC_EFFECTS_PLAN E2)
// ---------------------------------------------------------------------------
// An onset's stereo position: pan = (E_R - E_L)/(E_R + E_L) in [-1, 1], E the energy (sum
// of |X|^2 over the band's bins [k0, k1)) of each channel's spectrum at the onset. -1 is
// hard left. An instrument alone in its band at amplitude gains (1 - p, 1 + p) reads
// 2p/(1 + p^2); silence in both reads 0. (The offline script takes a Hann frame centred
// on the onset; the page takes the two analysers' current window, which holds it.)
function musicPan(magL, magR, k0, k1) {
  let el = 0, er = 0;
  for (let k = k0; k < k1; k++) { el += magL[k] * magL[k]; er += magR[k] * magR[k]; }
  return er + el > 0 ? (er - el) / (er + el) : 0;
}
// Do the two channels carry different signals at all? A mono source up-mixed to two equal
// channels reads pan 0 in every band, which musicPanX would turn into "everything in the
// centre column"; the page places nothing until this has been true once in a session (the
// offline script's is_mono, causally). True when some bin differs by more than `tol` of
// the larger spectrum's peak.
function musicChannelsDiffer(magL, magR, tol) {
  let pk = 0, d = 0;
  for (let k = 0; k < magL.length; k++) {
    pk = Math.max(pk, magL[k], magR[k]);
    d = Math.max(d, Math.abs(magL[k] - magR[k]));
  }
  return d > (tol === undefined ? 1e-3 : tol) * pk;
}
// where that puts the band's blobs: x = Lx (0.5 + SPAN * clip(gain * pan)); x runs left to
// right on the card. `Lx` is the box's x side (default 2 pi): the long side in landscape; in
// portrait the same map covers the short one.
function musicPanX(pan, gain, Lx) {
  return (Lx === undefined ? MUSIC_LBOX : Lx) * (0.5 + MUSIC_STEREO_SPAN * Math.min(1, Math.max(-1, (gain === undefined ? 1 : gain) * pan)));
}

// ---------------------------------------------------------------------------
// the camera pulse (MUSIC_EFFECTS_PLAN E3; pulse_zoom in the offline script)
// ---------------------------------------------------------------------------
// Display only. An onset at time t of strength q lifts the zoom TO 1 + a*q (never lowers
// it), and between onsets zoom - 1 decays as exp(-(t - t0)/tau). `st` = { excess, tLast }.
function musicPulseHit(st, t, strength, a, tau) {
  st.excess = Math.max((st.excess || 0) * Math.exp(-(t - (st.tLast || 0)) / tau), a * strength);
  st.tLast = t;
}
function musicPulseZoom(st, t, tau) {
  return 1 + (st.excess || 0) * Math.exp(-(t - (st.tLast || 0)) / tau);
}

// ---------------------------------------------------------------------------
// the mean field's random walk (MUSIC_EFFECTS_PLAN E4, Alfred 2026-11-21)
// ---------------------------------------------------------------------------
// A uniform in-plane field B0 whose direction and strength take a random-walk step at
// each bass onset. |B0| is `r` times the rms fluctuation sqrt(2E) at the time, so the
// field keeps its hold as the energy builds; r walks in [0, rmax], reflected at both ends.
//   st = { th, r, thTo, rTo }: the angle and ratio in force, and where they are heading
//   musicWalkInit(rng, rmax, walkRatio)   a uniform angle; r uniform in [0, rmax] if it
//                                         walks, rmax if it does not
//   musicWalkHit(st, rng, strength, deg, ratioStep, rmax)
//        the onset: thTo += deg (degrees, rms) * strength * N(0,1), and, if ratioStep > 0,
//        rTo += ratioStep * strength * N(0,1), reflected into [0, rmax]
//   musicWalkRelax(st, dt, tau)           th and r move toward their targets, e-folding tau
//   musicWalkField(st, energy)            (B0x, B0y) = r sqrt(2E) (cos th, sin th)
function musicWalkInit(rng, rmax, walkRatio) {
  const th = 2 * Math.PI * rng(), r = walkRatio ? rmax * rng() : rmax;
  return { th: th, r: r, thTo: th, rTo: r };
}
function musicReflect(x, rmax) {
  if (!(rmax > 0)) return 0;
  const y = x - 2 * rmax * Math.floor(x / (2 * rmax));
  return y <= rmax ? y : 2 * rmax - y;
}
function musicWalkHit(st, rng, strength, deg, ratioStep, rmax) {
  st.thTo += deg * Math.PI / 180 * strength * musicNormal(rng);
  if (ratioStep > 0) st.rTo = musicReflect(st.rTo + ratioStep * strength * musicNormal(rng), rmax);
}
function musicWalkRelax(st, dt, tau) {
  const w = dt > 0 ? 1 - Math.exp(-dt / tau) : 0;
  st.th += w * (st.thTo - st.th);
  st.r += w * (st.rTo - st.r);
}
function musicWalkField(st, energy) {
  const b = st.r * Math.sqrt(2 * Math.max(0, energy));
  return [b * Math.cos(st.th), b * Math.sin(st.th)];
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
// `stereo: true` (MUSIC_EFFECTS_PLAN E2) adds `left` and `right`: the kick in the centre,
// the snare left, the hats alternating right and left (MUSIC_CLICK_PANS). An instrument at
// amplitude pan p has gains (1 - p, 1 + p), so the channel mean is `samples` itself.
const MUSIC_CLICK_PANS = { kick: 0, snare: -0.6, hat: 0.8 };
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
  const lr = o.stereo ? [new Float64Array(n), new Float64Array(n)] : null;
  const add = (t0, sig, kind, pan) => {
    const i0 = Math.round(t0 * sr);
    for (let i = 0; i < sig.length; i++) out[(i0 + i) % n] += sig[i];   // the tail wraps: no seam
    if (lr) for (let i = 0; i < sig.length; i++) {
      lr[0][(i0 + i) % n] += (1 - pan) * sig[i];
      lr[1][(i0 + i) % n] += (1 + pan) * sig[i];
    }
    events.push(lr ? { t: i0 / sr, kind: kind, pan: pan } : { t: i0 / sr, kind: kind });
  };
  for (let i = 0; i < beats; i++) {
    add(i * beat, kick, "kick", MUSIC_CLICK_PANS.kick);
    if (i % 4 === 1 || i % 4 === 3) add(i * beat, snare, "snare", MUSIC_CLICK_PANS.snare);
    add((i + 0.5) * beat, hat, "hat", MUSIC_CLICK_PANS.hat * (i % 2 === 0 ? 1 : -1));
  }
  let pk = 0;
  for (let i = 0; i < n; i++) pk = Math.max(pk, Math.abs(out[i]));
  const samples = new Float32Array(n);
  for (let i = 0; i < n; i++) samples[i] = 0.8 * out[i] / pk;
  events.sort((a, b) => a.t - b.t);
  const res = { samples: samples, sr: sr, bpm: bpm, beat: beat, duration: n / sr, events: events };
  if (lr) {
    res.left = new Float32Array(n); res.right = new Float32Array(n);
    for (let i = 0; i < n; i++) { res.left[i] = 0.8 * lr[0][i] / pk; res.right[i] = 0.8 * lr[1][i] / pk; }
  }
  return res;
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
// thing a kick does change: a frame that kicked (`inp.kicked`) takes at least ONE step,
// even where the stale dt says none fits and even when the simulation is ahead -- the
// solver computes that step's dt afresh, so it is a small one, it moves the kicked flow at
// once and it gives the next frame a dt worth dividing by. Rebase (never catch up) when
// the simulation is more than lagMax seconds of audio behind the clock (or lagMax plus one
// step ahead of it), when the clock runs backwards, when `rate` changes, on the first
// frame after a stop, and when the caller says the page was hidden (`inp.resumed`). Only
// the first of those is "behind". The hold-back above is itself bounded: see the code.
function musicClock(st, inp) {
  const res = { n: 0, target: inp.simT, lag: 0, rebased: false, behind: false, resumed: false };
  if (!inp.running || !isFinite(inp.audioT) || !isFinite(inp.simT) || !(inp.rate > 0)) {
    st.init = false;
    return res;
  }
  const lagMax = inp.lagMax === undefined ? 0.25 : inp.lagMax;
  const hold = inp.behindHold === undefined ? 1 : inp.behindHold;
  const dtOk = inp.dt > 0 && isFinite(inp.dt);
  const rebase = () => { st.t0 = inp.simT; st.a0 = inp.audioT; st.rate = inp.rate; st.init = true; res.rebased = true; };
  // `inp.resumed`: the page was HIDDEN since the last frame (the caller knows from the
  // document's visibility, which is the only honest source -- a long gap between frames on
  // a visible page is a device that cannot keep up, and must not be excused as an absence)
  if (inp.resumed) res.resumed = true;
  if (!st.init || !(inp.audioT >= st.lastA) || st.rate !== inp.rate || inp.resumed) rebase();
  st.lastA = inp.audioT;
  let target = st.t0 + inp.rate * (inp.audioT - st.a0), lag = (target - inp.simT) / inp.rate;
  // behind by more than lagMax: rebase, and say so. AHEAD is allowed one whole step more
  // than lagMax, because a single step at the quiescent dt (0.098 at 256^2) is by itself
  // more than lagMax of audio once rate < 0.39 -- and being one step ahead in silence is
  // the clock working, not failing.
  const aheadMax = lagMax + (dtOk ? inp.dt / inp.rate : 0);
  if (lag > lagMax || lag < -aheadMax) {
    if (lag > 0) st.behindUntil = inp.audioT + hold;      // fell behind, as opposed to ran ahead
    rebase();
    target = inp.simT; lag = 0;
  }
  res.target = target; res.lag = lag;
  res.behind = st.behindUntil !== undefined && inp.audioT < st.behindUntil;
  const gap = target - inp.simT;
  const maxSteps = inp.maxSteps === undefined ? 64 : inp.maxSteps;
  // a frame that kicked always takes a step, behind the clock or not: its dt is recomputed
  // by the solver from the kicked state, so it is a small one, and the flow moves at once
  if (!(gap > 0)) { res.n = inp.kicked ? Math.min(1, maxSteps) : 0; return res; }
  if (!dtOk) { res.n = Math.min(1, maxSteps); return res; }   // dt not known yet
  const maxAhead = inp.maxAhead === undefined ? Infinity : inp.maxAhead;   // default: always round up
  let n = Math.ceil(gap / inp.dt);
  // the hold-back, and its BOUND: a step that would overshoot by more than maxAhead waits --
  // but only while the simulation is less than half of lagMax behind. Past that it is
  // taken: otherwise, at a low rate, the wait for a step to "fit" outlasts lagMax and the
  // clock would rebase for ever without stepping (rate 0.35: one step in ten seconds).
  if (n * inp.dt - gap > maxAhead && gap < 0.5 * lagMax * inp.rate) n = Math.floor(gap / inp.dt);
  if (inp.kicked && n < 1) n = 1;
  res.n = Math.max(0, Math.min(maxSteps, n));
  return res;
}

// ---------------------------------------------------------------------------
// sync: where the sound and the picture are each held back (fix round, 2026-11-21)
// ---------------------------------------------------------------------------
// Two delays the page did not choose stand between a hit and the visitor:
//   Ldet  detection: a kick is on screen this long after the hit entered the analyser --
//         half the analysis window + one frame to see the peak + one frame to draw it.
//         Measured in Chrome on the M1: 0.039 s to kick() returning, plus a frame.
//   Lout  the audio output: the sound leaves the speaker this long after the page handed
//         it over. AudioContext.outputLatency: ~0 on a laptop's speakers, 0.168 s measured
//         on Bluetooth earphones.
// and two it can add:
//   D     a delay on the SOUND (a DelayNode), only where the page is the one playing it
//   H     a hold-back on the KICKS (they wait in a queue), always possible
// A hit entering the analyser at time 0 is HEARD at D + Lout and SEEN at Ldet + H. With
//   v = Lout - Ldet + trim        (trim > 0: the picture later, trim < 0: the sound later)
// the picture is `trim` after the sound when  H - D = v,  so: v >= 0 -> hold the kicks
// (H = v, D = 0); v < 0 -> delay the sound (D = -v, H = 0) where that is possible.
//   page plays it (click, file, a silenced tab)   both sides available, as above
//   a tab the page does not play                  the tab is heard through the SAME output,
//                                                 undelayed: H = max(0, v), D unavailable
//   an input device                               the sound is already in the room: Lout
//                                                 does not apply, v = -Ldet + trim
// `residual` is what is left over: (seen - heard) - trim, i.e. 0 when the sync is exact
// and +x when the picture is still x seconds late because the sound could not be delayed.
// D is capped at maxD (the DelayNode's length) and H at maxH.
const MUSIC_LDET = 0.039 + 1 / 60;
function musicSync(o) {
  const num = v => (typeof v === "number" && isFinite(v) ? v : 0);
  const ldet = o.ldet === undefined ? MUSIC_LDET : o.ldet, trim = num(o.trimMs) / 1000;
  const maxD = o.maxD === undefined ? 1 : o.maxD, maxH = o.maxH === undefined ? 1 : o.maxH;
  // an input device's sound never passes through this machine's output
  const lout = o.kind === "input" ? 0 : Math.max(0, num(o.outputLatency));
  const v = lout - ldet + trim;
  let D = 0, H = 0;
  if (v >= 0) H = Math.min(maxH, v);
  else if (o.play) D = Math.min(maxD, -v);
  return { D: D, H: H, lout: lout, ldet: ldet, trim: trim, canDelay: !!o.play, residual: (ldet + H) - (D + lout) - trim };
}
// the mode line, from a sync result and the routing's sentence
function musicSyncText(sy, why) {
  const ms = x => (1000 * x).toFixed(0) + " ms";
  let t = (sy.canDelay ? "sound delayed " + ms(sy.D) + ", " : "") + "kicks held " + ms(sy.H)
        + " (output latency " + ms(sy.lout) + ", detection " + ms(sy.ldet)
        + (sy.trim ? ", trim " + (sy.trim > 0 ? "+" : "") + ms(sy.trim) : "") + ")";
  if (sy.residual > 0.0005) t += " -- the picture still runs " + ms(sy.residual) + " after the sound";
  return t + ": " + why;
}

// ---------------------------------------------------------------------------
// live sources (MUSIC_PLAN M3 / M4): the rules, as data
// ---------------------------------------------------------------------------
// The four sources, in the page's select. `click` and `file` are buffers the page plays
// itself; `tab` is another tab's audio (getDisplayMedia); `input` is an audio input device
// -- a loopback driver, or a microphone (getUserMedia).
const MUSIC_SOURCES = ["click", "file", "tab", "input"];
// the sync TRIM the visitor may add to the automatic value, in milliseconds either way
const MUSIC_TRIM_MAX = 150;
// the page's noise floor (MusicDetector fluxFloor / peakDecayS)
const MUSIC_FLUX_FLOOR = 0.3, MUSIC_PEAK_DECAY = 120;

// What getDisplayMedia is asked for. A video track is MANDATORY in the request (Chrome
// rejects audio-only display capture); it is asked for as small and slow as the API lets
// one ask. The audio is asked for RAW -- every processing stage is tuned for speech and
// would flatten the transients the detector looks for -- and with
// suppressLocalAudioPlayback, so that the captured tab falls silent and the page's own
// delayed copy is the only one heard. Whether that took effect is read back off the
// track (musicRouting), never assumed.
function musicTabConstraints() {
  return { video: { frameRate: 1, width: 64, height: 64 },
           audio: { suppressLocalAudioPlayback: true, echoCancellation: false, noiseSuppression: false,
                    autoGainControl: false },
           // a TAB's audio only: the whole system's would include this page's own output
           systemAudio: "exclude" };
}
// What getUserMedia is asked for: raw audio from one device (or the default one).
function musicInputConstraints(deviceId) {
  const a = { echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  if (deviceId) a.deviceId = { exact: deviceId };
  return { audio: a };
}
// May the page send this source to the speakers -- which is what makes delaying the SOUND
// possible at all? `suppressed` is what the capture's audio track REPORTS for
// suppressLocalAudioPlayback, `surface` what its video track reports for displaySurface.
//   click, file  the page is the only player: yes
//   tab          only if the capture really is a browser TAB (surface === "browser") and
//                the tab itself was silenced (suppressed === true); otherwise the visitor
//                already hears it, undelayed, and a second copy from the page is an echo
//   input        NEVER: a microphone routed to the speakers feeds back, and a loopback
//                device is already being heard
// Returns { play, compensated, why }: `why` is the sentence the page shows.
function musicRouting(kind, suppressed, surface) {
  if (kind === "click" || kind === "file")
    return { play: true, compensated: true, why: "the page plays this source" };
  if (kind === "tab") {
    if (surface !== "browser")
      return { play: false, compensated: false,
               why: "that capture is not a browser tab, so the page analyses it and does not play it" };
    if (suppressed === true)
      return { play: true, compensated: true, why: "the tab is silenced and the page plays its audio" };
    return { play: false, compensated: false,
             why: "this browser did not silence the captured tab, so the page does not play it too" };
  }
  if (kind === "input")
    return { play: false, compensated: false, why: "an input device is never played back" };
  throw new Error("musicRouting: unknown source " + kind);
}
// What to tell the visitor when a source will not start. `err` is the rejection (a
// DOMException: only its name is used), or the string "noaudio" for a capture that came
// back with no audio track.
// Does this browser capture a TAB's audio at all? There is no direct test, so this reads a
// proxy: the engines that deliver tab audio through getDisplayMedia (Chrome, Edge) are the
// ones that list the `suppressLocalAudioPlayback` constraint; Safari and Firefox hand back
// a video-only stream and do not list it. Used ONLY to word a failure -- never to refuse a
// capture up front, so a wrong answer here cannot break a browser that does work.
function musicTabAudioLikely(md) {
  try {
    return !!(md && md.getDisplayMedia && md.getSupportedConstraints
              && md.getSupportedConstraints().suppressLocalAudioPlayback === true);
  } catch (e) { return false; }
}

// `tabAudioLikely` (musicTabAudioLikely) only changes the wording of a tab capture that
// came back without audio; omitted, it is taken as true (the Chrome wording).
function musicStartError(kind, err, tabAudioLikely) {
  const name = typeof err === "string" ? err : (err && err.name) || "";
  if (name === "noaudio")
    return kind === "tab"
      ? (tabAudioLikely === false
         ? "that capture has no audio, and this browser does not seem to capture tab audio at all "
           + "(Safari and Firefox do not): use Chrome or Edge on a computer, or choose an audio file here"
         : "that capture has no audio: pick a Chrome TAB (not a window or a screen) and switch on "
           + "\"Also share tab audio\" in the picker")
      : "that device delivered no audio track";
  if (name === "NotAllowedError" || name === "SecurityError")
    return kind === "tab" ? "the tab capture was cancelled or not allowed"
                          : "access to the audio input was not allowed (check the site's microphone permission)";
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return kind === "tab" ? "nothing to capture was found" : "that audio input is not available (unplugged?)";
  if (name === "NotReadableError" || name === "AbortError")
    return kind === "tab" ? "the tab could not be captured" : "the audio input could not be opened (in use by another program?)";
  if (name === "unsupported")
    return kind === "tab" ? "this browser cannot capture a tab's audio (getDisplayMedia is missing)"
                          : "this browser cannot open an audio input (getUserMedia is missing)";
  return "could not start the " + (kind === "tab" ? "tab capture" : "audio input") + (name ? " (" + name + ")" : "");
}

// ---------------------------------------------------------------------------
// the music page's two extra colormaps (fix round)
// ---------------------------------------------------------------------------
// In the form every colormap of the demo takes -- a degree-6 polynomial per channel, low
// order first, Horner then clamp (physics.js cpoly6, common.js cmapRGB) -- and registered
// by the page through common.js's cmapExtend, so they exist on that page only: the shared
// table, the other pages' menus and their emitted WGSL are untouched.
//   magma  matplotlib's, the map of the offline |u| movies: a least-squares fit to its
//          256-point table, max channel error 0.0226 (gen_musiccmap.py prints these rows)
//   brat   #8ACE00 -> black, linear, EXACT in this form: rgb = (1 - t) * (138, 206, 0) / 255
//          (the cover: still flow is the green, the fastest is black)
//   brat_r brat reversed, black -> #8ACE00: rgb = t * (138, 206, 0) / 255
const MUSIC_CMAPS = [
  { name: "magma", coef: [
      [-0.002067, -0.000688, -0.009548],
      [0.250486, 0.694455, 2.495287],
      [8.345901, -3.596031, 0.329057],
      [-27.666969, 14.253853, -13.646583],
      [52.170684, -27.944584, 12.881091],
      [-50.758572, 29.053880, 4.269936],
      [18.664253, -11.490027, -5.570769]
    ] },
  { name: "brat", coef: [
      [0.541176, 0.807843, 0.000000],
      [-0.541176, -0.807843, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000]
    ] },
  { name: "brat_r", coef: [
      [0.000000, 0.000000, 0.000000],
      [0.541176, 0.807843, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000],
      [0.000000, 0.000000, 0.000000]
    ] }
];
