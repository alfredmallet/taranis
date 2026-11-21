// The MUSIC PAGE gate (plans-webgpu/MUSIC_PLAN.md, phase M2): music.js's pure logic against
// numbers recorded from the offline reference implementation, and music.html booted and
// DRIVEN under the stub -- real click track, real detector, real solver wiring, a fake
// AudioContext standing in for the one thing node does not have.
//
//   node checkmusic.js [webgpu-dir]        exit code 1 on any failure
//
// CI reports, never gates (the sibling scripts' contract).
//
// Mutation-tested once by the implementer (2026-11-21): 131 one-line mutations of music.js,
// the page glue and the loop hook, 123 killed. The 8 survivors each became a row -- the tie
// and reference-bin rows in leg 3, the 240 bpm seam, the default lagMax, Reset not lighting
// BEHIND, the 48 kHz context, and the replaced-solver row counting on the prototype -- but
// those rows were NOT re-run against their mutants (the round was cut short by agreement).
//
// What is checked, and against what
// ---------------------------------
//   1  music.js is PURE: it evaluates in a bare context with no DOM, no Web Audio and none
//      of the shared scripts, and every top-level name is in its own namespace.
//   2  musicBandPlan against the offline `band_plan` (fixtures/music_ref.json, written by
//      ../gen_musicfixture.py from examples/music_turbulence.py), and its refusals.
//   3  MusicDetector against the offline `CausalOnsetDetector`: the SAME onsets -- same
//      frame, same band, strength to 1e-12 -- on the fixture's signal, under eleven option
//      sets that between them vary every option the port takes. Both sides read the same
//      quantised samples; the gate does its own Blackman window and FFT. Then what the
//      python has no case for: the gap re-prime, reset, a live threshold, bin counts.
//   4  musicShouldPush, musicRng, musicBlobsFor, musicKickList.
//   5  musicClickTrack: deterministic, shaped as documented, and the detector -- in the
//      PAGE's configuration, 44100 Hz / 2048 / 4 bands / threshold 0.3 -- finds its kicks,
//      snares and hats at their scheduled times, across the loop's seam.
//   6  musicClock: the count, the cap, the look-ahead rule, every rebase.
//   7  music.html under the stub: what solver it builds, what the controls do to it.
//   8  ... and RUNNING: a fake AudioContext plays the page's own click track, the fake
//      analyser computes the true spectrum of what is "playing", and frames are driven one
//      at a time. One kick() per onset, with the band's sigma and count per channel; kicks
//      before that frame's steps; the steps taken are the clock's; simulation time tracks
//      the audio; falling behind rebases; Stop stops.
//   9  the loop hook is null on the other two pages, whose loop then steps `stepsPerFrame`
//      as it always did, and the state mark there is the string it always was.
//
// What a stub cannot say (and this file therefore does not claim): that a real
// AnalyserNode returns these spectra, that audio is audible, how late a real frame is, or
// anything about a GPU. The stub's stats readback is all zeros, so legs 8 replaces
// `readStats` with a model (t advances by a fixed dt per step, the energy by a fixed
// amount per kick): what is verified there is the page's ARITHMETIC AND ORDER given a
// readback, not the solver's.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const e2 = v => (isFinite(v) ? v.toExponential(2) : String(v));
const throws = fn => { try { fn(); return ""; } catch (e) { return String(e && e.message || e) || "threw"; } };
const boot = (d, page, opts) => new Promise(res => {
  const env = require("./stubenv")(d, page, null, opts);
  setTimeout(() => res(env));
});
const turn = async n => { for (let i = 0; i < (n || 40); i++) await new Promise(r => setImmediate(r)); };
const FIX = path.join(__dirname, "fixtures", "music_ref.json");

// ---------------------------------------------------------------------------
// the gate's own signal processing (nothing below is the app's)
// ---------------------------------------------------------------------------
// |FFT(window * x[end-nfft : end])| / nfft over bins 0 .. nfft/2. `periodic` picks the
// window's denominator: numpy.blackman is symmetric (nfft - 1), the Web Audio one is
// periodic (nfft). `at(i)` returns sample i (so a caller can loop or zero-pad).
function spectrum(at, end, nfft, periodic) {
  const re = new Float64Array(nfft), im = new Float64Array(nfft), den = periodic ? nfft : nfft - 1;
  for (let i = 0; i < nfft; i++) {
    const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / den) + 0.08 * Math.cos(4 * Math.PI * i / den);
    re[i] = w * at(end - nfft + i);
  }
  for (let i = 1, j = 0; i < nfft; i++) {
    let b = nfft >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) { const t = re[i]; re[i] = re[j]; re[j] = t; }
  }
  for (let len = 2; len <= nfft; len <<= 1) {
    const ang = -2 * Math.PI / len;
    for (let i = 0; i < nfft; i += len) for (let k = 0; k < len / 2; k++) {
      const c = Math.cos(ang * k), s = Math.sin(ang * k), a = i + k, b = a + len / 2;
      const tr = re[b] * c - im[b] * s, ti = re[b] * s + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
  const mag = new Float64Array(nfft / 2 + 1);
  for (let k = 0; k <= nfft / 2; k++) mag[k] = Math.hypot(re[k], im[k]) / nfft;
  return mag;
}
// python's round(): half to even (the frame boundaries of detect_onsets_causal)
const pyRound = v => { const f = Math.floor(v); return v - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(v); };
// run a detector over a signal exactly as detect_onsets_causal cuts it into frames
function runDetector(det, x, sr, fps, nfft, opts) {
  const o = opts || {}, out = [];
  const at = o.loop ? (o.zeroBefore ? i => (i < 0 ? 0 : x[i % x.length]) : i => x[((i % x.length) + x.length) % x.length])
                    : i => (i >= 0 && i < x.length ? x[i] : 0);
  const nfr = Math.floor((o.seconds === undefined ? x.length / sr : o.seconds) * fps);
  for (let i = 0; i < nfr; i++) {
    // `timeOf` lets a caller reproduce another clock exactly (leg 8: the fake AudioContext's)
    const tNow = o.timeOf ? o.timeOf(i) : pyRound((i + 1) / fps * sr) / sr;
    const end = o.timeOf ? Math.round(tNow * sr) : pyRound((i + 1) / fps * sr);
    if (end < nfft && !o.timeOf) continue;
    let mag = spectrum(at, end, nfft, !!o.periodic);
    if (o.dropNyquist) mag = mag.subarray(0, nfft / 2);
    if (o.toDb) for (let k = 0; k < mag.length; k++) mag[k] = Math.pow(10, Math.fround(20 * Math.log10(mag[k])) / 20);
    if (o.before) o.before(det, i, tNow);
    for (const e of det.push(mag, tNow)) out.push(e);
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

// ===========================================================================
// 1. music.js is pure
// ===========================================================================
function legPure(state) {
  const src = fs.readFileSync(path.join(dir, "music.js"), "utf8");
  // a context with NOTHING in it: no document, no window, no AudioContext, no common.js
  const ctx = vm.createContext({});
  const msg = throws(() => vm.runInContext(src, ctx, { filename: "music.js" }));
  ok("music.js evaluates in a bare context: no DOM, no Web Audio, none of the shared scripts", msg === "", msg);
  const M = state.M = vm.runInContext(`({ musicBandPlan, MusicDetector, musicShouldPush, musicRng, musicBlobsFor,
    musicKickList, musicClickTrack, musicClock, musicNpSum, musicPercentile, MUSIC_LBOX, MUSIC_SIGMA_MAX,
    MUSIC_SIGMA_MIN_DX, MUSIC_SIGMA_STEP, MUSIC_NBLOB_BASS, MUSIC_NBLOB_MAX })`, ctx);
  const names = [];
  for (const l of src.split("\n")) {
    const m = /^(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/.exec(l);
    if (m) names.push(m[1]);
  }
  const stray = names.filter(n => !/^(music|Music|MUSIC_)/.test(n));
  ok("every top-level name is in the music namespace (" + names.length + " names)", stray.length === 0 && names.length >= 12,
     stray.join(", ") || "music* / Music* / MUSIC_*");
  // ... so loading it beside the three shared files redeclares nothing
  const shared = ["common.js", "physics.js", "solver2d.js"].map(f => fs.readFileSync(path.join(dir, f), "utf8")).join("\n");
  const clash = names.filter(n => new RegExp("^(?:async\\s+)?(?:function\\*?|const|let|var|class)\\s+" + n.replace(/\$/g, "\\$") + "\\b", "m").test(shared));
  ok("  ... and none is declared at the top level of common.js / physics.js / solver2d.js", clash.length === 0, clash.join(", "));
  const code = src.split("\n").filter(l => !/^\s*\/\//.test(l)).join("\n");
  const dom = ["document", "window", "navigator", "AudioContext", "requestAnimationFrame", "performance", "location",
               "device", "solver", "GPU"].filter(w => new RegExp("\\b" + w + "\\b").test(code));
  ok("  ... its code names no DOM, Web Audio, WebGPU or page global", dom.length === 0, dom.join(", "));
  ok("  ... and has no module syntax (it must load from file://)",
     !/(^|[^\w.$])import\s*[({"'*]/m.test(code) && !/^\s*export[\s{*]/m.test(code) && /^"use strict";/.test(src));
  ok("the map's constants are the offline script's: box 2pi, sigma_max 0.45, 3.7 dx, step 1.8, 3 bass blobs, cap 128",
     M.MUSIC_LBOX === 2 * Math.PI && M.MUSIC_SIGMA_MAX === 0.45 && M.MUSIC_SIGMA_MIN_DX === 3.7 &&
     M.MUSIC_SIGMA_STEP === 1.8 && M.MUSIC_NBLOB_BASS === 3 && M.MUSIC_NBLOB_MAX === 128);
}

// ===========================================================================
// 2. the band plan
// ===========================================================================
function legPlan(state) {
  const M = state.M;
  if (!fs.existsSync(FIX)) { ok("fixtures/music_ref.json is present", false, "run ../gen_musicfixture.py"); return; }
  const F = state.F = JSON.parse(fs.readFileSync(FIX, "utf8"));
  ok("the fixture is the offline script's, and small enough to commit (< 500 kB)",
     /band_plan and CausalOnsetDetector/.test(F.source) && fs.statSync(FIX).size < 500000 && F.nblob_max === M.MUSIC_NBLOB_MAX,
     fs.statSync(FIX).size + " bytes");
  let n = 0;
  for (const p of F.plans) {
    const tag = "band_plan(n = " + p.n + (p.nbands === null ? "" : ", nbands = " + p.nbands)
              + (p.sigmaMin === null ? "" : ", sigma_min = " + p.sigmaMin) + ")";
    let r = null, err = "";
    try { r = M.musicBandPlan(p.n, p.nbands, p.sigmaMin); } catch (e) { err = e.message; }
    if (p.error) { ok(tag + ": refused by both", !!err, err.slice(0, 50)); continue; }
    let ds = 0;
    if (r && r.sigmas.length === p.sigmas.length)
      for (let i = 0; i < p.sigmas.length; i++) ds = Math.max(ds, Math.abs(r.sigmas[i] / p.sigmas[i] - 1));
    ok(tag + ": " + p.nblobs.length + " bands, blobs " + p.nblobs.join(" "),
       !!r && r.sigmas.length === p.sigmas.length && ds < 1e-14 &&
       JSON.stringify(Array.from(r.nblobs)) === JSON.stringify(p.nblobs), err || "max relative sigma difference " + e2(ds));
    n++;
  }
  ok("  ... " + n + " plans compared, including n = 256, 512, 1024 by default and explicit nbands / sigma_min",
     n >= 9 && [256, 512, 1024].every(v => F.plans.some(p => p.n === v && p.nbands === null && p.sigmaMin === null)));
  // the rule itself, at the page's grid, from the definition rather than from the fixture
  const p256 = M.musicBandPlan(256);
  const smin = 3.7 * 2 * Math.PI / 256;
  ok("at n = 256 the rule gives 4 bands from sigma 0.45 down to 3.7 dx = " + smin.toFixed(4) + ", a constant ratio between them",
     p256.sigmas.length === 4 && p256.sigmas[0] === 0.45 && Math.abs(p256.sigmas[3] - smin) < 1e-15 &&
     Math.abs(p256.sigmas[0] / p256.sigmas[1] - p256.sigmas[2] / p256.sigmas[3]) < 1e-12 &&
     p256.sigmas[0] / p256.sigmas[1] > 1.6 && p256.sigmas[0] / p256.sigmas[1] < 1.8,
     "sigmas " + Array.from(p256.sigmas).map(v => v.toFixed(4)).join(" "));
  ok("  ... the blob count is round(3 * 0.45 / sigma): 3 in the bass band, 15 in the top one",
     Array.from(p256.nblobs).every((v, i) => v === Math.round(3 * 0.45 / p256.sigmas[i])) && p256.nblobs[0] === 3 && p256.nblobs[3] === 15);
  // nblobMax: what one kick() can hold
  ok("with nblobMax = 32 (the solver's KICK_MAX) n = 256 fits and n = 512 fits (30)",
     throws(() => M.musicBandPlan(256, null, null, 32)) === "" && throws(() => M.musicBandPlan(512, null, null, 32)) === "");
  const m1024 = throws(() => M.musicBandPlan(1024, null, null, 32));
  ok("  ... and n = 1024 (59 in the top band) is REFUSED, naming the count and the limit",
     /59/.test(m1024) && /32/.test(m1024), m1024.slice(0, 70) || "no throw");
  ok("  ... exactly at the limit it is accepted (15 <= 15) and one below refused (15 > 14)",
     throws(() => M.musicBandPlan(256, null, null, 15)) === "" && /15/.test(throws(() => M.musicBandPlan(256, null, null, 14))));
  ok("  ... and without nblobMax the limit is the offline 128: n = 2048 (119) fits, n = 4096 does not",
     throws(() => M.musicBandPlan(2048)) === "" && /128/.test(throws(() => M.musicBandPlan(4096))));
  for (const [tag, args, re] of [
    ["n = 0", [0], /positive integer/], ["n = 256.5", [256.5], /positive integer/], ["n = -256", [-256], /positive integer/],
    ["n = \"256\"", ["256"], /positive integer/], ["n = NaN", [NaN], /positive integer/],
    ["nbands = 0", [256, 0], /nbands/], ["nbands = 2.5", [256, 2.5], /nbands/], ["nbands = -1", [256, -1], /nbands/],
    ["sigma_min = 0", [256, null, 0], /sigma_min/], ["sigma_min < 0", [256, null, -0.1], /sigma_min/],
    ["sigma_min above sigma_max", [256, null, 0.46], /sigma_min/], ["sigma_min = NaN", [256, null, NaN], /sigma_min/],
    ["sigma_min = \"0.2\"", [256, null, "0.2"], /sigma_min/]]) {
    const m = throws(() => M.musicBandPlan.apply(null, args));
    ok("musicBandPlan refuses " + tag, re.test(m), m.slice(0, 60) || "no throw");
  }
  ok("  ... while sigma_min = sigma_max exactly is a legal (degenerate) plan",
     throws(() => M.musicBandPlan(256, 2, 0.45)) === "" && M.musicBandPlan(256, 2, 0.45).sigmas[1] === 0.45);
}

// ===========================================================================
// 3. the detector
// ===========================================================================
function legDetector(state) {
  const M = state.M, F = state.F;
  if (!F) { ok("fixture loaded", false); return; }
  const raw = Buffer.from(F.signal.b64, "base64"), sr = F.signal.sr, x = new Float64Array(F.signal.n);
  for (let i = 0; i < x.length; i++) x[i] = raw.readInt16LE(2 * i) / 32768;
  state.x = x; state.sr = sr;
  ok("the fixture's signal: " + (x.length / sr) + " s at " + sr + " Hz, int16", raw.length === 2 * F.signal.n && x.length > 4 * sr,
     raw.length + " bytes");
  const seen = new Set();
  let total = 0;
  for (const c of F.cases) {
    Object.keys(c.opts).forEach(k => seen.add(k));
    const o = Object.assign({ sr: sr }, c.opts), det = new M.MusicDetector(o);
    const got = runDetector(det, x, sr, o.fps || 60, o.nfft);
    let same = got.length === c.onsets.length, ds = 0, dt = 0, firstBad = -1;
    if (same) for (let i = 0; i < got.length; i++) {
      // [t_fire, band, strength, t_centre]: the frame and the band EXACTLY, the rest to round-off
      if (got[i].tFire !== c.onsets[i][0] || got[i].band !== c.onsets[i][1]) { same = false; firstBad = i; break; }
      ds = Math.max(ds, Math.abs(got[i].strength - c.onsets[i][2]));
      dt = Math.max(dt, Math.abs(got[i].t - c.onsets[i][3]));
    }
    let de = 0;
    for (let i = 0; i < c.edges.length; i++) de = Math.max(de, Math.abs(det.edges[i] / c.edges[i] - 1));
    total += c.onsets.length;
    ok(c.tag + ": the same " + c.onsets.length + " onsets", same && ds < 1e-12 && dt < 1e-12 && de < 1e-12 &&
       c.onsets.length > 20 && det.edges.length === c.edges.length,
       same ? "frame and band exact; max |d strength| " + e2(ds) + ", |d t| " + e2(dt) + ", edges " + e2(de)
            : "got " + got.length + (firstBad >= 0 ? ", first mismatch at onset " + firstBad : ""));
  }
  const need = ["nbands", "threshold", "minGap", "meanS", "normS", "refDecayS", "warmupS", "fmin", "fmax", "fps", "nfft"];
  ok("  ... " + total + " onsets in all, and between them the cases vary every option the port takes",
     need.every(k => seen.has(k)), "varied: " + Array.from(seen).sort().join(" "));
  // the cases really do differ from the base one (an option silently ignored would
  // otherwise reproduce it): by count, by band layout or by strength
  const base = F.cases[0], sig = c => JSON.stringify(c.onsets.map(o => [o[0], o[1], Math.round(o[2] * 1e9)]));
  const dup = F.cases.slice(1).filter(c => sig(c) === sig(base)).map(c => c.tag);
  ok("  ... and no case reproduces the first one's onsets (each option changes the answer)", dup.length === 0, dup.join("; "));

  // ---- what the fixture has no case for ---------------------------------------
  const P = Object.assign({ sr: sr }, base.opts);                     // the page's settings
  const ref = runDetector(new M.MusicDetector(P), x, sr, 60, P.nfft);
  // warm-up: nothing fires before it
  ok("no onset is declared before the warm-up (0.5 s by default)", ref.length > 0 && ref.every(e => e.tFire >= 0.5) &&
     ref.some(e => e.tFire < 1.0), "first at " + (ref[0] ? ref[0].tFire.toFixed(3) : "-") + " s");
  // the times it reports: the peak is ONE frame before it is known, the centre half a window before that
  ok("an onset is reported one frame after its peak: tFire - t = 1/fps + nfft/(2 sr)",
     ref.every(e => Math.abs((e.tFire - e.t) - (1 / 60 + 0.5 * P.nfft / sr)) < 1e-12),
     "latency " + (1e3 * (1 / 60 + 0.5 * P.nfft / sr)).toFixed(1) + " ms at these settings");
  ok("  ... strengths lie in (0, 1], and some are clipped at 1", ref.every(e => e.strength > 0 && e.strength <= 1) &&
     ref.some(e => e.strength === 1) && ref.some(e => e.strength < 0.9));
  ok("  ... and a band never fires twice within minGap", (() => {
    const last = {}; for (const e of ref) { if (last[e.band] !== undefined && e.t - last[e.band] < 0.07 - 1e-9) return false; last[e.band] = e.t; }
    return true; })());
  // the analyser hands over nfft/2 bins (no Nyquist): same answer
  const noNy = runDetector(new M.MusicDetector(P), x, sr, 60, P.nfft, { dropNyquist: true });
  ok("fed nfft/2 bins (an AnalyserNode's frequencyBinCount, no Nyquist bin) it gives the same onsets",
     JSON.stringify(noNy) === JSON.stringify(ref), noNy.length + " vs " + ref.length);
  const few = throws(() => new M.MusicDetector(P).push(new Float64Array(8), 1));
  ok("  ... and too few bins to reach the top band is refused", /bins/.test(few), few.slice(0, 60) || "no throw");
  // reset() is a new detector
  const d2 = new M.MusicDetector(P);
  runDetector(d2, x, sr, 60, P.nfft); d2.reset();
  ok("reset() returns it to its initial state: a second pass gives the first pass's onsets",
     JSON.stringify(runDetector(d2, x, sr, 60, P.nfft)) === JSON.stringify(ref));
  // the threshold is read live
  const hi = runDetector(new M.MusicDetector(Object.assign({}, P, { threshold: 0.8 })), x, sr, 60, P.nfft);
  const live = runDetector(new M.MusicDetector(P), x, sr, 60, P.nfft, { before: (d, i, t) => { if (t >= 4) d.threshold = 0.8; } });
  const key = e => e.tFire.toFixed(6) + "/" + e.band;
  ok("`threshold` is read on every push: raised to 0.8 at t = 4 s, the onsets are the 0.3 ones before and the 0.8 ones after",
     hi.length < ref.length && JSON.stringify(live.map(key)) ===
       JSON.stringify(ref.filter(e => e.tFire < 4).concat(hi.filter(e => e.tFire >= 4)).sort((a, b) => a.t - b.t).map(key)),
     ref.length + " at 0.3, " + hi.length + " at 0.8, " + live.length + " switched");
  // the gap: a jump in time re-primes instead of comparing across it
  const dg = new M.MusicDetector(P), dn = new M.MusicDetector(P);
  const at = i => (i >= 0 && i < x.length ? x[i] : 0), fr = k => pyRound((k + 1) / 60 * sr);
  let gapOn = 0, noGapOn = 0;
  for (let k = 0; k < 150; k++) if (fr(k) >= P.nfft) { dg.push(spectrum(at, fr(k), P.nfft), fr(k) / sr); dn.push(spectrum(at, fr(k), P.nfft), fr(k) / sr); }
  // ... then the spectrum from 2 s LATER, handed over as if 2 s had passed (dg) or one frame (dn)
  const far = spectrum(at, fr(270), P.nfft);
  gapOn += dg.push(far, fr(270) / sr).length;
  const hLenGap = dg.hist[0].length;
  noGapOn += dn.push(far, fr(150) / sr).length;
  const hLenNo = dn.hist[0].length;
  ok("a gap longer than gapS (0.1 s) only re-primes: no flux is recorded across it, and nothing fires",
     gapOn === 0 && hLenGap === hLenNo - 1, "history " + hLenGap + " with the gap, " + hLenNo + " without");
  const nextG = dg.push(spectrum(at, fr(271), P.nfft), fr(271) / sr);
  ok("  ... and the frame after the gap is an ordinary one again (flux recorded)", dg.hist[0].length === hLenGap + 1 && Array.isArray(nextG));
  ok("  ... while a gap of exactly one long frame under gapS is NOT a gap", (() => {
    const d = new M.MusicDetector(P); d.push(spectrum(at, fr(100), P.nfft), 1.0); d.push(spectrum(at, fr(101), P.nfft), 1.05);
    const n0 = d.hist[0].length; d.push(spectrum(at, fr(102), P.nfft), 1.14); return n0 === 1 && d.hist[0].length === 2; })());
  // two things no recorded signal can force, from the definition (survivors of the M2
  // mutation round): a TIE between the peak frame and the next still fires (s1 >= s2), and
  // the compression reference is the maximum over EVERY bin handed over, bin 0 included.
  // Spectra are built by hand: one loud bin (the reference) outside every band, and equal
  // increments in two different bins of band 1 on consecutive frames -- the two fluxes are
  // then the same number bit for bit.
  const tie = new M.MusicDetector({ sr: 44100, nbands: 3, threshold: 0.15, warmupS: 0 });
  const k1 = tie.bins[1][0] + 2, k2 = tie.bins[1][0] + 5, frameOf = (a, b) => { const m = new Float64Array(1025).fill(1e-6); m[1024] = 1; m[k1] = a; m[k2] = b; return m; };
  const seq = [[1e-6, 1e-6], [1e-6, 1e-6], [1e-6, 1e-6], [0.01, 1e-6], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01]];
  let tieOut = [];
  seq.forEach((v, i) => { tieOut = tieOut.concat(tie.push(frameOf(v[0], v[1]), (i + 1) / 60)); });
  const hT = tie.hist[1];
  ok("a TIE fires: two consecutive frames of bit-identical flux are one onset, at the first (s1 >= s2, not >)",
     hT[2] === hT[3] && hT[2] > 0 && hT[1] === 0 && tieOut.length === 1 && tieOut[0].band === 1 &&
     Math.abs(tieOut[0].tFire - 5 / 60) < 1e-12, tieOut.length + " onset(s); flux " + hT.slice(0, 5).map(e2).join(" "));
  const refD = new M.MusicDetector({ sr: 44100 }), m0 = new Float64Array(1025).fill(1e-3);
  m0[0] = 0.5; refD.push(m0, 0.1);
  const refN = new M.MusicDetector({ sr: 44100 }), mN = new Float64Array(1025).fill(1e-3);
  mN[1024] = 0.25; refN.push(mN, 0.1);
  ok("the compression reference is the maximum over every bin handed over: bin 0 (DC) and the last one count",
     refD.ref === 0.5 && refN.ref === 0.25, "ref " + refD.ref + " and " + refN.ref);
  // configuration refusals
  for (const [tag, o] of [["no sample rate", {}], ["sr = 0", { sr: 0 }], ["nfft = 3", { sr: sr, nfft: 3 }],
                          ["nbands = 0", { sr: sr, nbands: 0 }], ["nbands = 2.5", { sr: sr, nbands: 2.5 }],
                          ["fps = 0", { sr: sr, fps: 0 }], ["fmin = 0", { sr: sr, fmin: 0 }],
                          ["fmax <= fmin", { sr: sr, fmin: 500, fmax: 400 }]]) {
    const m = throws(() => new M.MusicDetector(o));
    ok("MusicDetector refuses " + tag, /bad configuration/.test(m), m.slice(0, 40) || "no throw");
  }
  const d0 = new M.MusicDetector({ sr: 44100 });
  ok("its defaults are the offline ones: nfft 2048, 3 bands 40-10000 Hz, fps 60, threshold 0.15, minGap 0.07, warm-up 0.5, "
     + "mean 12 frames, norm 1200 frames", d0.nfft === 2048 && d0.nbands === 3 && d0.fps === 60 && d0.threshold === 0.15 &&
     d0.minGap === 0.07 && d0.warmupS === 0.5 && d0.gapS === 0.1 && d0.nmean === 12 && d0.nnorm === 1200 &&
     d0.edges[0] === 40 && d0.edges[3] === 10000 && Math.abs(d0.decay - Math.exp(-1 / 1800)) < 1e-15);
  ok("  ... fmax is capped at Nyquist, and the bands tile the bins between the edges with no gap or overlap", (() => {
    const d = new M.MusicDetector({ sr: 8000, nbands: 4 });
    if (d.edges[4] !== 4000) return false;
    for (let b = 0; b < 3; b++) if (d.bins[b][1] !== d.bins[b + 1][0]) return false;
    const df = 8000 / 2048;
    return d.bins[0][0] === Math.ceil(40 / df) && d.bins[3][1] === 1024 && d.bins.every(v => v[1] > v[0]); })());
  // numpy's summation order, which is what makes "the same onsets" achievable
  const a = []; for (let i = 0; i < 300; i++) a.push(Math.sin(i * 1.7) * Math.pow(10, (i % 9) - 4));
  const naive = (lo, hi) => { let s = 0; for (let i = lo; i < hi; i++) s += a[i]; return s; };
  ok("musicNpSum agrees with a plain sum to round-off at every length class (< 8, <= 128, > 128)",
     [[0, 5], [0, 8], [3, 100], [0, 128], [0, 129], [7, 300]].every(([lo, hi]) =>
       Math.abs(M.musicNpSum(a, lo, hi) - naive(lo, hi)) <= 1e-9 * Math.abs(naive(lo, hi)) + 1e-12) && M.musicNpSum(a, 4, 4) === 0);
  ok("musicPercentile is numpy's linear method: p99 of 0..100 is 99, p50 of [1, 2, 4] is 2, p99 of [5] is 5, p99 of [0, 10] is 9.9",
     M.musicPercentile(Array.from({ length: 101 }, (_, i) => 100 - i), 0, 101, 99) === 99 &&
     M.musicPercentile([4, 1, 2], 0, 3, 50) === 2 && M.musicPercentile([5], 0, 1, 99) === 5 &&
     Math.abs(M.musicPercentile([10, 0], 0, 2, 99) - 9.9) < 1e-12 && M.musicPercentile([9, 1, 2, 3], 1, 3, 100) === 2);
}

// ===========================================================================
// 4. the push rule, the generator, the blobs
// ===========================================================================
function legBlobs(state) {
  const M = state.M;
  ok("musicShouldPush: the first frame is taken; then once 3/4 of a nominal interval has passed",
     M.musicShouldPush(0, null, 60) === true && M.musicShouldPush(0, undefined, 60) === true &&
     M.musicShouldPush(1 / 60, 0, 60) === true && M.musicShouldPush(0.0125, 0, 60) === true &&
     M.musicShouldPush(0.0124, 0, 60) === false && M.musicShouldPush(1 / 120, 0, 60) === false);
  const rate = hz => { let t = 0, last = null, n = 0; for (let i = 0; i < 1200; i++) { t = i / hz; if (M.musicShouldPush(t, last, 60)) { last = t; n++; } } return n / (1200 / hz); };
  ok("  ... so a 60 Hz display pushes at 60, a 120 Hz one at 60, a 144 Hz one at 72, a 30 Hz one at 30",
     Math.abs(rate(60) - 60) < 0.5 && Math.abs(rate(120) - 60) < 0.5 && Math.abs(rate(144) - 72) < 0.5 && Math.abs(rate(30) - 30) < 0.5,
     [60, 120, 144, 30].map(h => rate(h).toFixed(1)).join(" "));
  ok("  ... and a clock that ran backwards is taken (never a stuck detector)", M.musicShouldPush(1, 2, 60) === true &&
     M.musicShouldPush(NaN, 2, 60) === true);
  // the generator
  const r1 = M.musicRng(1), r1b = M.musicRng(1), r2 = M.musicRng(2);
  const a = [], b = [], c = [];
  for (let i = 0; i < 4000; i++) { a.push(r1()); b.push(r1b()); c.push(r2()); }
  const mean = a.reduce((s, v) => s + v, 0) / a.length;
  ok("musicRng: deterministic under its seed, different under another, uniform on [0, 1)",
     JSON.stringify(a) === JSON.stringify(b) && JSON.stringify(a) !== JSON.stringify(c) &&
     a.every(v => v >= 0 && v < 1) && Math.abs(mean - 0.5) < 0.02 && new Set(a).size > 3990,
     "mean " + mean.toFixed(4) + ", first " + a[0].toFixed(6));
  // one channel's blobs
  const L = 2 * Math.PI, bl = M.musicBlobsFor(M.musicRng(7), 0.2, 2400, 0.75, -1);
  ok("musicBlobsFor: the count asked for, every blob at the given sigma and polarity",
     bl.length === 2400 && bl.every(q => q.sigma === 0.2 && q.pol === -1));
  ok("  ... every amplitude is +-amp exactly, with a fair sign", bl.every(q => Math.abs(q.amp) === 0.75) &&
     Math.abs(bl.filter(q => q.amp > 0).length / 2400 - 0.5) < 0.04, bl.filter(q => q.amp > 0).length + " of 2400 positive");
  const quad = [0, 0, 0, 0];
  for (const q of bl) quad[(q.x < L / 2 ? 0 : 1) + (q.y < L / 2 ? 0 : 2)]++;
  ok("  ... every position inside the box [0, 2pi)^2, spread over it (each quadrant holds about a quarter)",
     bl.every(q => q.x >= 0 && q.x < L && q.y >= 0 && q.y < L) && quad.every(v => Math.abs(v / 2400 - 0.25) < 0.04), quad.join(" "));
  ok("  ... x and y are independent draws (no blob on the diagonal, correlation ~ 0)", (() => {
    let sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
    for (const q of bl) { sx += q.x; sy += q.y; sxy += q.x * q.y; sxx += q.x * q.x; syy += q.y * q.y; }
    const n = bl.length, r = (sxy / n - sx * sy / n / n) / Math.sqrt((sxx / n - sx * sx / n / n) * (syy / n - sy * sy / n / n));
    return Math.abs(r) < 0.06 && bl.every(q => q.x !== q.y); })());
  ok("  ... deterministic under the seed, different under another",
     JSON.stringify(M.musicBlobsFor(M.musicRng(7), 0.2, 50, 0.75, 1)) === JSON.stringify(M.musicBlobsFor(M.musicRng(7), 0.2, 50, 0.75, 1)) &&
     JSON.stringify(M.musicBlobsFor(M.musicRng(7), 0.2, 50, 0.75, 1)) !== JSON.stringify(M.musicBlobsFor(M.musicRng(8), 0.2, 50, 0.75, 1)));
  ok("  ... and zero blobs is an empty list", M.musicBlobsFor(M.musicRng(1), 0.2, 0, 1, 1).length === 0);
  // one onset's list
  const plan = M.musicBandPlan(256);
  for (let band = 0; band < 4; band++) {
    const kl = M.musicKickList(M.musicRng(3), plan, band, 1.5, 0.4), n = plan.nblobs[band];
    const zp = kl.slice(0, n), zm = kl.slice(n);
    ok("musicKickList, band " + band + ": " + n + " z+ blobs then " + n + " z- blobs, all at sigma " + plan.sigmas[band].toFixed(4)
       + " and |amp| = amp * strength", kl.length === 2 * n && zp.every(q => q.pol === 1) && zm.every(q => q.pol === -1) &&
       kl.every(q => q.sigma === plan.sigmas[band] && Math.abs(q.amp) === 1.5 * 0.4));
  }
  const kl = M.musicKickList(M.musicRng(3), plan, 3, 1, 1), n3 = plan.nblobs[3];
  const xs = new Set(kl.map(q => q.x.toFixed(12)));
  const g = M.musicRng(3), seqP = M.musicBlobsFor(g, plan.sigmas[3], n3, 1, 1), seqM = M.musicBlobsFor(g, plan.sigmas[3], n3, 1, -1);
  ok("  ... the z- set is an INDEPENDENT second run of draws, never a copy of the z+ one",
     xs.size === 2 * n3 && JSON.stringify(kl) === JSON.stringify(seqP.concat(seqM)) &&
     JSON.stringify(kl.slice(0, n3).map(q => [q.x, q.y])) !== JSON.stringify(kl.slice(n3).map(q => [q.x, q.y])));
  ok("  ... and successive onsets draw fresh positions (the generator is consumed, not reseeded)", (() => {
    const r = M.musicRng(3), k1 = M.musicKickList(r, plan, 0, 1, 1), k2 = M.musicKickList(r, plan, 0, 1, 1);
    return JSON.stringify(k1) !== JSON.stringify(k2); })());
  ok("  ... every band's list fits one Solver.kick: at most 15 <= KICK_MAX = 32 per channel at 256^2",
     Math.max.apply(null, Array.from(plan.nblobs)) === 15);
}

// ===========================================================================
// 5. the click track
// ===========================================================================
function legClick(state) {
  const M = state.M;
  const c = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 });
  const same = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 });
  const other = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 2 });
  const eq = (p, q) => { if (p.length !== q.length) return false; for (let i = 0; i < p.length; i++) if (p[i] !== q[i]) return false; return true; };
  let pk = 0, bad0 = 0;
  for (let i = 0; i < c.samples.length; i++) { pk = Math.max(pk, Math.abs(c.samples[i])); if (!isFinite(c.samples[i])) bad0++; }
  ok("musicClickTrack: 16 beats at 120 bpm = 8 s of fp32 samples at 44100 Hz, peak 0.8, all finite",
     c.samples.length === 8 * 44100 && c.duration === 8 && c.beat === 0.5 && c.sr === 44100 && c.bpm === 120 &&
     Math.abs(pk - 0.8) < 1e-6 && bad0 === 0, c.samples.length + " samples, peak " + pk.toFixed(6));
  ok("  ... deterministic: the same arguments give the same samples, bit for bit; another seed differs",
     eq(c.samples, same.samples) && !eq(c.samples, other.samples));
  const by = k => c.events.filter(e => e.kind === k).map(e => e.t);
  const near = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-4);
  const beats = Array.from({ length: 16 }, (_, i) => i * 0.5);
  ok("  ... its schedule: a kick on every beat, a snare on beats 2 and 4 of each bar, a hat on every off-beat",
     near(by("kick"), beats) && near(by("snare"), beats.filter((_, i) => i % 4 === 1 || i % 4 === 3)) &&
     near(by("hat"), beats.map(t => t + 0.25)) && c.events.length === 40 &&
     c.events.every((e, i) => i === 0 || e.t >= c.events[i - 1].t), by("kick").length + " kicks, " + by("snare").length + " snares, " + by("hat").length + " hats");
  const c90 = M.musicClickTrack({ sr: 48000, bpm: 90, beats: 8 });
  ok("  ... bpm and the sample rate are honoured: 8 beats at 90 bpm, 48 kHz = 5.333 s, kicks 0.667 s apart",
     c90.samples.length === Math.round(8 * (60 / 90) * 48000) && Math.abs(c90.beat - 2 / 3) < 1e-15 &&
     Math.abs(c90.events.filter(e => e.kind === "kick")[1].t - 2 / 3) < 1e-4);
  for (const [tag, o] of [["bpm = 0", { bpm: 0 }], ["bpm = NaN", { bpm: NaN }], ["beats = 6 (not whole bars)", { beats: 6 }],
                          ["beats = 0", { beats: 0 }], ["sr = 0", { sr: 0 }]]) {
    const m = throws(() => M.musicClickTrack(o));
    ok("musicClickTrack refuses " + tag, /need sr/.test(m), m.slice(0, 40) || "no throw");
  }
  // the seam: every sound's tail WRAPS round the end of the buffer, so the loop is the
  // periodic signal itself. Twice as many beats is then exactly two copies -- without the
  // wrap, the second half would start with the first half's last hat still ringing and
  // the first half would not.
  const c32 = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 32, seed: 1 });
  let seam = 0;
  for (let i = 0; i < c32.samples.length; i++) seam = Math.max(seam, Math.abs(c32.samples[i] - c.samples[i % c.samples.length]));
  ok("  ... the loop is SEAMLESS: tails wrap round the end, so 32 beats are exactly two copies of 16",
     c32.samples.length === 2 * c.samples.length && seam < 1e-6, "max |difference| " + e2(seam));
  // ... at 120 bpm nothing audible is still ringing at the seam, so the row above cannot tell
  // a wrap from a cut. At 240 bpm the last kick (0.4 s long, a beat of 0.25 s) is.
  const f16 = M.musicClickTrack({ sr: 44100, bpm: 240, beats: 16, seed: 1 }), f32 = M.musicClickTrack({ sr: 44100, bpm: 240, beats: 32, seed: 1 });
  let seamF = 0, headE = 0;
  for (let i = 0; i < f32.samples.length; i++) seamF = Math.max(seamF, Math.abs(f32.samples[i] - f16.samples[i % f16.samples.length]));
  for (let i = 0; i < 0.14 * 44100; i++) headE = Math.max(headE, Math.abs(f16.samples[i]));
  ok("  ... also at 240 bpm, where the last kick's tail really does cross the seam and land on the first beat",
     seamF < 1e-6 && f16.duration === 4 && headE > 0.3, "max |difference| " + e2(seamF));
  // ---- what the detector makes of it, in the PAGE's configuration ---------------
  // 44100 Hz, the analyser's 2048-sample PERIODIC Blackman window, 4 bands, threshold
  // 0.3, 60 frames a second, the track LOOPED three times (24 s) as the page plays it.
  for (const [sr, bpm] of [[44100, 120], [48000, 100]]) {
    const ct = sr === 44100 ? c : M.musicClickTrack({ sr: sr, bpm: bpm, beats: 16, seed: 1 });
    const det = new M.MusicDetector({ sr: sr, nfft: 2048, nbands: 4, fps: 60, threshold: 0.3 });
    const ons = runDetector(det, ct.samples, sr, 60, 2048, { loop: true, periodic: true, seconds: 3 * ct.duration, dropNyquist: true });
    const T = ct.duration, sched = k => [0, 1, 2].flatMap(l => ct.events.filter(e => e.kind === k).map(e => e.t + l * T)).filter(t => t >= T);
    // an onset "at" a scheduled time: its analysis-window centre within 30 ms of it
    const hit = (t, bands) => ons.some(e => bands.indexOf(e.band) >= 0 && Math.abs(e.t - t) < 0.03);
    const kicks = sched("kick"), snares = sched("snare"), hats = sched("hat");
    const tag = sr + " Hz, " + bpm + " bpm";
    ok(tag + ": every scheduled kick of loops 2-3 is an onset in the BASS band within 30 ms",
       kicks.every(t => hit(t, [0])), kicks.filter(t => hit(t, [0])).length + " of " + kicks.length);
    ok("  ... every hat an onset in the TOP band, every snare one in a middle band",
       hats.every(t => hit(t, [3])) && snares.every(t => hit(t, [1, 2])),
       hats.filter(t => hit(t, [3])).length + " of " + hats.length + " hats, "
       + snares.filter(t => hit(t, [1, 2])).length + " of " + snares.length + " snares");
    const late = ons.filter(e => e.t >= T);
    const stray0 = late.filter(e => e.band === 0 && !kicks.some(t => Math.abs(e.t - t) < 0.03));
    const stray3 = late.filter(e => e.band === 3 && !hats.concat(snares).some(t => Math.abs(e.t - t) < 0.03));
    ok("  ... and nothing else is: no bass onset away from a kick, no top-band onset away from a hat or snare",
       stray0.length === 0 && stray3.length === 0, stray0.length + " stray bass, " + stray3.length + " stray top; "
       + late.length + " onsets in 2 loops");
    if (sr === 44100) {
      // the bass kick's strength is what the kick amplitude is multiplied by
      const ks = late.filter(e => e.band === 0).map(e => e.strength);
      ok("  ... the kicks are full-strength onsets (strength >= 0.9), so the bass kick is at the amp slider's value",
         ks.length >= 30 && ks.every(v => v >= 0.9), "min " + Math.min.apply(null, ks).toFixed(3));
      state.clickOnsets = ons;
    }
  }
}

// ===========================================================================
// 6. the clock
// ===========================================================================
function legClock(state) {
  const M = state.M;
  const run = (st, o) => M.musicClock(st, Object.assign({ running: true, rate: 1, maxSteps: 64, lagMax: 0.25 }, o));
  // an established clock: anchored at audio 10 s <-> simulation 4
  const est = (rate) => { const st = {}; run(st, { audioT: 10, simT: 4, dt: 0.01, rate: rate || 1 }); return st; };
  let st = {}, r = run(st, { audioT: 10, simT: 4, dt: 0.01 });
  ok("the first frame ANCHORS the clock (t0 = simT, a0 = audioT) and takes no step",
     r.rebased === true && r.n === 0 && r.target === 4 && r.lag === 0 && st.t0 === 4 && st.a0 === 10 && st.init === true && r.behind === false);
  r = run(st, { audioT: 10.05, simT: 4, dt: 0.0101 });
  ok("then target = t0 + rate * (audioT - a0): 50 ms of audio later it is 4.05, lag 0.05 s, and n = 5 steps of dt 0.0101",
     Math.abs(r.target - 4.05) < 1e-12 && Math.abs(r.lag - 0.05) < 1e-12 && r.n === 5 && r.rebased === false);
  ok("  ... n is the CEILING: a gap of 0.0501 at dt 0.01 is 6 steps, of 0.0499 is 5",
     run(est(), { audioT: 10.0501, simT: 4, dt: 0.01 }).n === 6 && run(est(), { audioT: 10.0499, simT: 4, dt: 0.01 }).n === 5);
  ok("  ... `rate` scales the target: at rate 2, 50 ms of audio is 0.1 of simulation time, 10 steps; lag stays in AUDIO seconds",
     (() => { const q = run(est(2), { audioT: 10.05, simT: 4, dt: 0.0101, rate: 2 }); return q.n === 10 && Math.abs(q.target - 4.1) < 1e-12 && Math.abs(q.lag - 0.05) < 1e-12; })());
  ok("  ... the simulation at or ahead of the target takes no step",
     run(est(), { audioT: 10.05, simT: 4.0501, dt: 0.01 }).n === 0 && run(est(), { audioT: 10.05, simT: 4.06, dt: 0.01 }).n === 0 &&
     run(est(), { audioT: 10.05, simT: 4.06, dt: 0.01 }).lag < 0);
  ok("  ... the count is capped at maxSteps (64 by default, and the cap is the argument's)",
     run(est(), { audioT: 10.2, simT: 4, dt: 0.001 }).n === 64 && run(est(), { audioT: 10.2, simT: 4, dt: 0.001, maxSteps: 7 }).n === 7 &&
     M.musicClock(est(), { running: true, rate: 1, audioT: 10.2, simT: 4, dt: 0.001 }).n === 64);
  ok("  ... with dt not yet known (0, NaN: before the first step) it takes ONE step, to find out",
     run(est(), { audioT: 10.05, simT: 4, dt: 0 }).n === 1 && run(est(), { audioT: 10.05, simT: 4, dt: NaN }).n === 1 &&
     run(est(), { audioT: 10.05, simT: 4, dt: -1 }).n === 1);
  // the look-ahead rule
  ok("maxAhead: a step that would overshoot the target by more than it is NOT taken (n rounds down)",
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60 }).n === 0 &&
     run(est(), { audioT: 10.1, simT: 4, dt: 0.098, maxAhead: 1 / 60 }).n === 1 &&
     run(est(), { audioT: 10.2, simT: 4, dt: 0.098, maxAhead: 1 / 60 }).n === 2);
  ok("  ... but on a frame that KICKED, one step is taken anyway (its dt is recomputed by the solver; the stale one says none fits)",
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: true }).n === 1 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: false }).n === 0 &&
     run(est(), { audioT: 10.2, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: true }).n === 2);
  ok("  ... never when the simulation is not behind, and never more than the stale count where that is >= 1",
     run(est(), { audioT: 10.05, simT: 4.06, dt: 0.098, maxAhead: 1 / 60, kicked: true }).n === 0 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.0101, maxAhead: 1 / 60, kicked: true }).n === 5 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: true, maxSteps: 0 }).n === 0);
  ok("  ... while an overshoot within it rounds up as usual, and without maxAhead it always rounds up",
     run(est(), { audioT: 10.05, simT: 4, dt: 0.06, maxAhead: 1 / 60 }).n === 1 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.07, maxAhead: 1 / 60 }).n === 0 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098 }).n === 1);
  // rebase: too far behind
  // (reached over two frames: ONE jump of more than lagMax is an absence, tested below)
  st = est(); run(st, { audioT: 10.2, simT: 4, dt: 0.01 }); r = run(st, { audioT: 10.3, simT: 4, dt: 0.01 });
  ok("more than lagMax (0.25 s of audio) behind: REBASE -- no catching up, the anchor moves to now, the indicator lights",
     r.rebased === true && r.n === 0 && r.lag === 0 && r.target === 4 && st.t0 === 4 && st.a0 === 10.3 && r.behind === true);
  ok("  ... exactly at lagMax it does not (the test is >), and the lag is measured in audio seconds whatever the rate",
     run(est(), { audioT: 10.25, simT: 4, dt: 0.01 }).rebased === false &&
     run(est(2), { audioT: 10.2, simT: 4, dt: 0.01, rate: 2 }).rebased === false &&
     run(est(2), { audioT: 10.26, simT: 4, dt: 0.01, rate: 2 }).rebased === true);
  ok("  ... the indicator stays lit for behindHold (1 s of audio by default) and then goes out",
     run(st, { audioT: 10.8, simT: 4.5, dt: 0.01 }).behind === true && run(st, { audioT: 11.29, simT: 4.99, dt: 0.01 }).behind === true &&
     run(st, { audioT: 11.31, simT: 5.01, dt: 0.01 }).behind === false);
  st = est(); r = run(st, { audioT: 10.3, simT: 4, dt: 0.0101, lagMax: 0.5 });
  ok("  ... lagMax is the argument's: at 0.5 a lag of 0.3 s is simply stepped, not rebased", r.rebased === false && r.n === 30);
  ok("  ... and lagMax DEFAULTS to 0.25 s when the caller gives none", (() => {
    const call = (a) => { const z = {}; M.musicClock(z, { running: true, rate: 1, audioT: 10, simT: 4, dt: 0.0101 });
                          return M.musicClock(z, { running: true, rate: 1, audioT: a, simT: 4, dt: 0.0101 }); };
    return call(10.2).rebased === false && call(10.3).rebased === true; })());
  // rebase: the simulation ran ahead
  st = est(); r = run(st, { audioT: 10.05, simT: 4.5, dt: 0.01 });
  ok("more than lagMax AHEAD also rebases, but that is not `behind`", r.rebased === true && r.n === 0 && r.behind === false && st.t0 === 4.5);
  // rebase: backwards, rate change, stop
  st = est(); run(st, { audioT: 10.1, simT: 4.1, dt: 0.01 }); r = run(st, { audioT: 9.9, simT: 4.1, dt: 0.01 });
  ok("an audio clock that runs BACKWARDS rebases (and is not `behind`)", r.rebased === true && r.n === 0 && st.a0 === 9.9 && st.t0 === 4.1 && r.behind === false);
  st = est(); r = run(st, { audioT: 10.05, simT: 4.05, dt: 0.01, rate: 2 });
  ok("a change of `rate` rebases: the old anchor no longer means anything", r.rebased === true && st.rate === 2 && st.a0 === 10.05);
  st = est(); r = M.musicClock(st, { running: false, audioT: 10.05, simT: 4, dt: 0.01, rate: 1 });
  ok("stopped: no steps, and the clock is un-anchored", r.n === 0 && r.rebased === false && st.init === false);
  r = run(st, { audioT: 30, simT: 4, dt: 0.01 });
  ok("  ... so the first frame after a stop rebases instead of trying to make up the time away",
     r.rebased === true && r.n === 0 && st.a0 === 30 && r.behind === false);
  // a paused audio clock: time stands still, nothing to do, nothing to rebase
  st = est(); run(st, { audioT: 10.05, simT: 4.0501, dt: 0.01 });
  r = run(st, { audioT: 10.05, simT: 4.0501, dt: 0.01 });
  ok("a paused audio clock (the same time twice) takes no step and does not rebase", r.n === 0 && r.rebased === false);
  // a hidden tab: frames stopped, audio went on
  st = est(); r = run(st, { audioT: 25, simT: 4, dt: 0.01 });
  ok("a long absence (hidden tab: audio 15 s on, simulation where it was) rebases on return -- and is not `behind`",
     r.rebased === true && r.n === 0 && st.a0 === 25 && r.behind === false);
  st = est(); run(st, { audioT: 10.2, simT: 4.2, dt: 0.01 }); r = run(st, { audioT: 10.44, simT: 4.2, dt: 0.01 });
  ok("  ... while frames that DID come, each under lagMax apart, with the simulation not following, are `behind`",
     (() => { const z = est(); let reb = 0, q; for (const a of [10.1, 10.2, 10.3]) { q = run(z, { audioT: a, simT: 4, dt: 0.01 }); if (q.rebased) reb++; }
              return reb === 1 && q.rebased === true && q.behind === true; })() && r.rebased === false);
  for (const [tag, o] of [["a non-finite audio time", { audioT: NaN, simT: 4, dt: 0.01 }], ["a non-finite simulation time", { audioT: 10, simT: NaN, dt: 0.01 }],
                          ["rate 0", { audioT: 10, simT: 4, dt: 0.01, rate: 0 }]]) {
    st = est(); r = run(st, o);
    ok("  ... " + tag + " takes no step and un-anchors", r.n === 0 && st.init === false);
  }
  // a steady run: frames at 60 Hz, a fixed dt -- the simulation stays within one step of the clock
  st = {}; let simT = 0, worst = 0, steps = 0, reb = 0;
  for (let f = 0; f < 600; f++) {
    const q = run(st, { audioT: f / 60, simT: simT, dt: 0.0037 });
    if (q.rebased) reb++;
    if (f > 0) worst = Math.max(worst, Math.abs(q.lag));
    simT += q.n * 0.0037; steps += q.n;
  }
  ok("a steady 10 s at 60 Hz with dt = 0.0037: one rebase (the anchor), and the lag never exceeds a frame",
     reb === 1 && worst <= 1 / 60 + 1e-9 && Math.abs(simT - 599 / 60) < 0.0037, "worst lag " + (1e3 * worst).toFixed(2) + " ms, " + steps + " steps");
}

// ---------------------------------------------------------------------------
// a fake AudioContext: the one thing node does not have. It PLAYS nothing; its
// analyser computes the true spectrum of what would be playing at `currentTime`.
// ---------------------------------------------------------------------------
function fakeAudio(sr) {
  const A = { t: 0, sr: sr || 44100, ctxs: [], decoded: null, analyserCalls: 0 };
  A.Ctx = class {
    constructor() { this.sampleRate = A.sr; this.state = "suspended"; this.destination = { dest: 1 };
                    this.baseLatency = 0.005; this.outputLatency = 0.02; this.sources = []; this.analysers = [];
                    this.resumed = 0; A.ctxs.push(this); }
    get currentTime() { return A.t; }
    createBuffer(ch, len, rate) {
      const d = new Float32Array(len);
      return { numberOfChannels: ch, length: len, sampleRate: rate, duration: len / rate, getChannelData: () => d };
    }
    createBufferSource() {
      const s = { buffer: null, loop: false, started: null, stopped: 0, onended: null, to: null, disc: 0,
                  connect(n) { this.to = n; n.from = this; }, disconnect() { this.disc++; },
                  start() { this.started = A.t; }, stop() { this.stopped++; } };
      this.sources.push(s);
      return s;
    }
    createAnalyser() {
      const an = { fftSize: 1024, smoothingTimeConstant: 0.8, from: null, to: null,
        get frequencyBinCount() { return this.fftSize / 2; },
        connect(n) { this.to = n; }, disconnect() {},
        getFloatFrequencyData(arr) {
          A.analyserCalls++;
          const s = this.from, d = s.buffer.getChannelData(0), n = d.length, N = this.fftSize;
          const end = Math.round((A.t - s.started) * A.sr);
          const at = s.loop ? i => (i < 0 ? 0 : d[i % n]) : i => (i >= 0 && i < n ? d[i] : 0);
          const mag = spectrum(at, end, N, true);
          for (let k = 0; k < arr.length; k++) arr[k] = 20 * Math.log10(mag[k]);      // -Infinity for silence
        } };
      this.analysers.push(an);
      return an;
    }
    async decodeAudioData(ab) { A.decoded = ab; return A.fileBuffer(this); }
    async resume() { this.resumed++; this.state = "running"; }
  };
  return A;
}
// a booted music page with a model stats readback: t advances by `dt` per step, the energy
// by 1 per kick, and everything the solver is asked to do is logged in order
const MODEL = `function(dt){
  const s = solver;
  s.__dt = dt; s.__t = 0; s.__e = 0.5; s.__log = []; s.__reads = 0;
  const step = s.step.bind(s), kick = s.kick.bind(s), setIC = s.setIC.bind(s);
  s.step = ce => { s.__t += s.__dt; s.__log.push({ op: "step", ce: ce }); return step(ce); };
  s.kick = l => { const r = kick(l); s.__e += 1; s.__log.push({ op: "kick", list: JSON.parse(JSON.stringify(l)), r: r }); return r; };
  s.setIC = z => { s.__t = 0; s.__e = 0.5; return setIC(z); };
  s.readStats = async () => { s.__reads++; const a = new Float32Array(12); a[0] = s.__nsteps0 === s.nsteps && s.nsteps === 0 ? 0 : s.__dt;
                              a[1] = s.__t; a[2] = 0.6 * s.__e; a[3] = 0.4 * s.__e; return a; };
  s.__nsteps0 = 0;
}`;
async function musicPage(search, sr) {
  const env = await boot(dir, "music.html", search ? { search: search } : undefined);
  await turn();
  const A = fakeAudio(sr);
  env.run("function(C){ window.AudioContext = C; }", A.Ctx);
  const frame = async t => { if (t !== undefined) A.t = t; env.frame(); await turn(); };
  const takeLog = () => JSON.parse(env.run("function(){ const l = solver.__log; solver.__log = []; return JSON.stringify(l); }"));
  return { env: env, A: A, frame: frame, takeLog: takeLog, click: async id => { const h = env.getEl(id).onclick; const r = h && h({}); if (r && r.then) await r; await turn(); },
           input: async (id, v) => { env.getEl(id).value = String(v); const h = env.getEl(id).oninput || env.getEl(id).onchange; if (h) h({}); await turn(); } };
}
// the lookups shared code makes for controls this page does not have: null-tolerant in a
// browser (`_uiOn` / `_uiVal`), a logged miss under the stub's stricter getElementById
const TOLERATED = ["cbAutoDiss", "cbForce", "selIC", "selPreset", "cbEpsLock"];
const realFails = env => env.fails.filter(f => !TOLERATED.some(id => String(f).indexOf('"' + id + '"') >= 0));

// ===========================================================================
// 7. music.html, booted
// ===========================================================================
async function legBoot(state) {
  const P = state.page = await musicPage("?musicdebug=energy"), env = P.env, M = state.M;
  ok("music.html boots under the stub with no failure of its own", realFails(env).length === 0,
     realFails(env).slice(0, 3).join(" | ") || env.fails.length + " tolerated control lookups");
  const html = fs.readFileSync(path.join(dir, "music.html"), "utf8");
  const srcs = Array.from(html.matchAll(/<script src="([^"]+)"><\/script>/g)).map(m => m[1]);
  ok("it loads common.js, physics.js, solver2d.js, music.js -- in that order, and nothing else",
     srcs.join(",") === "common.js,physics.js,solver2d.js,music.js", srcs.join(","));
  ok("  ... carries no analytics beacon, and no other page links to it",
     !/goatcounter|gc\.zgo\.at/.test(html.replace(/<!--[\s\S]*?-->/g, "")) &&
     ["rmhd2d.html", "rmhd3d.html", "index.html", "docs.html", "reading.html", "common.js", "physics.js", "solver2d.js"]
       .every(f => {
         const t = fs.readFileSync(path.join(dir, f), "utf8");
         // a link is an href, or the name in a string literal a page could build one from;
         // a COMMENT naming the page (common.js explains its hook) is not a link
         const code = t.split("\n").filter(l => !/^\s*(\/\/|\*|<!--)/.test(l)).join("\n");
         return !/href\s*=\s*["']?music\.html/.test(t) && !/["'`][^"'`\n]*music\.html/.test(code);
       }));
  const yml = fs.readFileSync(path.join(dir, "..", ".github", "workflows", "pages.yml"), "utf8");
  ok("  ... and pages.yml prunes music.html and music.js from the deployed site (a development build until it ships)",
     /rm -f\s+_site\/webgpu\/music\.html\s+_site\/webgpu\/music\.js/.test(yml) &&
     !/for f in[^\n]*\n?[^\n]*music\.(html|js)/.test(yml));
  const q = JSON.parse(env.run(`function(){ const s = solver; return JSON.stringify({ p: s.p, gk: s.g.kick, pl: !!s.pl.blobKick,
    buf: !!s.buf.kicks, lin0: s.g.gridB[1], lin1: s.g.gridB[5], k1: s.g.gridA[6], blob: s.blobMode, nk: s.nkicks,
    hook: stepHook === musicStep, run: running, plan: music.plan, kmax: KICK_MAX, mark: stateMark(),
    sel: cards.disp.map(c => c.selField.value), charts: cards.chart.map(c => c.type()), dbg: MUSIC_DEBUG,
    hasDbg: typeof window.musicDebug }); }`));
  const diss = 3 / Math.pow(256 / 6, 4);
  ok("the solver is 256^2 in a 2pi box, hyper 2, Pm 1, no equilibrium source",
     q.p.nx === 256 && q.p.ny === 256 && q.p.Lx === 2 * Math.PI && q.p.Ly === 2 * Math.PI && q.p.hyper === 2 && q.p.pm === 1 && q.p.eqsrc === false);
  ok("  ... built with kick: true (kernel, pipeline and buffer present) and NOT in blob mode",
     q.p.kick === true && q.gk === true && q.pl && q.buf && q.blob === false && q.nk === 0);
  ok("  ... the OU forcing is OFF: epsP = epsM = 0 (the Solver's own defaults are 0.15)", q.p.epsP === 0 && q.p.epsM === 0);
  ok("  ... the drag is mu = 1 / (memory * rate) = 1 at the defaults, and it is in the column: gridB.y(k = 0) = -1/2",
     q.p.mu === 1 && q.lin0 === -0.5, "mu " + q.p.mu + ", lin(k=0) " + q.lin0);
  ok("  ... the dissipation is the offline rule 3 * amp / (n/6)^4 = " + diss.toExponential(3) + " at amp 1",
     Math.abs(q.p.diss / diss - 1) < 1e-14 && Math.abs(q.lin1 - Math.fround(Math.fround(-diss * q.k1 * q.k1) - 0.5)) < 1e-9,
     "diss " + q.p.diss.toExponential(4));
  ok("the band plan is musicBandPlan(256) under the solver's KICK_MAX: 4 bands, 3 5 9 15 blobs",
     JSON.stringify(q.plan) === JSON.stringify(M.musicBandPlan(256, null, null, q.kmax)) && q.kmax === 32 && q.plan.nblobs.join() === "3,5,9,15");
  ok("the display card shows |u| (field 4) by default, beside an energy and a spectrum chart",
     q.sel.join() === "4" && q.charts.join() === "energy,spectrum");
  ok("the step hook is installed, nothing is running, and no AudioContext exists before Start",
     q.hook === true && q.run === false && P.A.ctxs.length === 0 && q.hasDbg === "undefined" && q.dbg === true);
  ok("the controls' defaults: threshold 0.30, memory 1.0 s, amp 1.0, rate 1.00, 120 bpm, click track, cfl 0.40 / every 4",
     env.getEl("rThr").value === "0.3" && env.getEl("rMem").value === "1" && env.getEl("rMAmp").value === "1" &&
     env.getEl("rRate").value === "1" && env.getEl("nBpm").value === "120" && env.getEl("selSrc").value === "click" &&
     env.getEl("rCfl").value === "0.4" && env.getEl("rCflEvery").value === "4" && env.getEl("vThr").textContent === "0.30");
  ok("the strip lists the four bands with their sigma and blob count, and no warning",
     /σ 0\.450 ×3/.test(env.getEl("musicStrip").textContent) && /σ 0\.091 ×15/.test(env.getEl("musicStrip").textContent) &&
     !/BEHIND|dropped/.test(env.getEl("musicStrip").textContent), env.getEl("musicStrip").textContent.slice(0, 60));
  ok("the file row is hidden while the source is the click track, and shown for a file",
     env.getEl("rowFile").style.display === "none" &&
     (env.getEl("selSrc").value = "file", env.getEl("selSrc").onchange({}), env.getEl("rowFile").style.display === "") &&
     (env.getEl("selSrc").value = "click", env.getEl("selSrc").onchange({}), env.getEl("rowFile").style.display === "none"));

  // ---- the controls reach the solver -----------------------------------------
  // (every slider is driven through its own oninput, on the live solver)
  const sv = () => JSON.parse(env.run("function(){ return JSON.stringify({ mu: solver.p.mu, diss: solver.p.diss, cfl: solver.p.cfl, lin0: solver.g.gridB[1], lin1: solver.g.gridB[5], k1: solver.g.gridA[6], hyper: solver.p.hyper }); }"));
  await P.input("rMem", 2);
  let v = sv();
  ok("memory 2 s -> mu = 1/2 on the live solver, and in the column (-1/4 at k = 0)", v.mu === 0.5 && v.lin0 === -0.25 && env.getEl("vMem").textContent === "2.0");
  await P.input("rRate", 0.5);
  v = sv();
  ok("rate 0.5 -> mu = 1 / (2 * 0.5) = 1: the memory is in seconds of AUDIO", v.mu === 1 && v.lin0 === -0.5 && env.getEl("vRate").textContent === "0.50");
  await P.input("rMAmp", 2);
  v = sv();
  ok("amp 2 -> the dissipation doubles (the offline rule scales with amp), the drag untouched, hyper still 2",
     Math.abs(v.diss / (2 * diss) - 1) < 1e-14 && v.mu === 1 && v.lin0 === -0.5 && v.hyper === 2 &&
     Math.abs(v.lin1 - Math.fround(-2 * diss * v.k1 * v.k1 - 0.5)) < 1e-9, "diss " + v.diss.toExponential(4));
  await P.input("rCfl", 0.25);
  ok("cfl_safety 0.25 -> solver.p.cfl", sv().cfl === 0.25 && env.getEl("vCfl").textContent === "0.25");
  for (const [id, val] of [["rMem", 1], ["rRate", 1], ["rMAmp", 1], ["rCfl", 0.4]]) await P.input(id, val);
  v = sv();
  ok("  ... and back at the defaults the solver is back where it booted", v.mu === 1 && Math.abs(v.diss / diss - 1) < 1e-14 && v.cfl === 0.4);
  const man = JSON.parse(env.run("function(){ return JSON.stringify(runManifest().forcing); }"));
  ok("a saved params.json says what drove the run: forcing.kind = \"kick\", not reproducible, with the page's settings",
     man.on === true && man.kind === "kick" && man.ou === false && man.reproducible === false && man.threshold === 0.3 &&
     man.memory === 1 && man.amp === 1 && man.rate === 1 && man.bands === 4 && man.source === "click" && man.mu === 1, JSON.stringify(man).slice(0, 80));
}

// ===========================================================================
// 8. music.html, running
// ===========================================================================
async function legRun(state) {
  const P = state.page, env = P.env, A = P.A, M = state.M;
  env.run(MODEL, 0.004);
  // ---- Start ---------------------------------------------------------------------
  A.t = 100;                                   // the context has been alive a while
  await P.click("btnRun");
  const ctx = A.ctxs[0], an = ctx && ctx.analysers[0], src = ctx && ctx.sources[0];
  ok("Start creates ONE AudioContext (on the click, not before), resumes it, and starts a looping buffer source",
     A.ctxs.length === 1 && ctx.resumed === 1 && ctx.sources.length === 1 && src.loop === true && src.started === 100);
  const ref = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 });
  const buf = src.buffer.getChannelData(0);
  let same = buf.length === ref.samples.length;
  for (let i = 0; same && i < buf.length; i++) if (buf[i] !== ref.samples[i]) same = false;
  ok("  ... whose buffer IS musicClickTrack at the context's sample rate and the bpm box's tempo", same, buf.length + " samples");
  ok("  ... wired source -> analyser -> destination, the analyser at fftSize 2048 with smoothing 0",
     src.to === an && an.to === ctx.destination && an.fftSize === 2048 && an.smoothingTimeConstant === 0);
  const d = JSON.parse(env.run(`function(){ const d = music.det; return JSON.stringify({ sr: d.sr, nfft: d.nfft, nb: d.nbands, fps: d.fps,
    thr: d.threshold, run: running, t0: music.audio.t0, db: music.db.length, mag: music.mag.length, dbg: typeof window.musicDebug }); }`));
  ok("  ... the detector is built at the context's rate: 44100 Hz, nfft 2048, 4 bands, 60 fps, threshold 0.3",
     d.sr === 44100 && d.nfft === 2048 && d.nb === 4 && d.fps === 60 && d.thr === 0.3 && d.db === 1024 && d.mag === 1024);
  ok("  ... and the page is running, with audio time 0 at the source's first sample", d.run === true && d.t0 === 100 &&
     env.getEl("btnRun").textContent === "Pause" && d.dbg === "object");
  // ---- 12 s of frames at 60 Hz --------------------------------------------------
  const frames = [];
  for (let f = 1; f <= 720; f++) {
    await P.frame(100 + f / 60);
    frames.push({ f: f, a: f / 60, log: P.takeLog() });
  }
  const dbg = JSON.parse(env.run("function(){ return JSON.stringify({ kicks: window.musicDebug.kicks, frames: window.musicDebug.frames, rebases: window.musicDebug.rebases }); }"));
  const kicksLog = frames.flatMap(fr => fr.log.filter(e => e.op === "kick").map(e => Object.assign({ f: fr.f, a: fr.a }, e)));
  // the reference: the page's own detector configuration run by the GATE over the same audio
  const det = new M.MusicDetector({ sr: 44100, nfft: 2048, nbands: 4, fps: 60, threshold: 0.3 });
  // ... at the very times the page saw (audio time = currentTime - 100, in fp64 as the page
  // computes it), through the analyser's fp32 dB, with silence before the first sample
  const want = runDetector(det, ref.samples, 44100, 60, 2048, { loop: true, zeroBefore: true, periodic: true, seconds: 12,
    dropNyquist: true, toDb: true, timeOf: i => (100 + (i + 1) / 60) - 100 });
  ok("one analyser read per frame, and the spectrum handed to the detector is 10^(dB/20) of it",
     A.analyserCalls === 720, A.analyserCalls + " reads in 720 frames");
  // fp32 dB -> magnitude costs ~1e-7 relative: the same onsets but not bit-identical strengths
  let sameOn = kicksLog.length === want.length, ds = 0;
  for (let i = 0; sameOn && i < want.length; i++) {
    if (dbg.kicks[i].band !== want[i].band || Math.abs(dbg.kicks[i].tFire - want[i].tFire) > 1e-9) sameOn = false;
    else ds = Math.max(ds, Math.abs(dbg.kicks[i].strength - want[i].strength));
  }
  ok("ONE solver.kick() per onset the detector declares: " + want.length + " onsets in 12 s, " + kicksLog.length + " kicks",
     sameOn && want.length > 60 && ds < 1e-9, "same frame and band each; max |d strength| " + e2(ds));
  ok("  ... none before the warm-up (0.5 s of audio)", dbg.kicks.every(k => k.tFire >= 0.5) && dbg.kicks[0].tFire < 1.1,
     "first kick at " + dbg.kicks[0].tFire.toFixed(3) + " s");
  const plan = M.musicBandPlan(256);
  let shape = 0, amps = 0, inBox = 0;
  kicksLog.forEach((k, i) => {
    const b = dbg.kicks[i].band, zp = k.list.filter(q => q.pol > 0), zm = k.list.filter(q => q.pol < 0);
    if (zp.length === plan.nblobs[b] && zm.length === plan.nblobs[b] && k.list.every(q => q.sigma === plan.sigmas[b]) &&
        k.r.used[0] === plan.nblobs[b] && k.r.used[1] === plan.nblobs[b] && k.r.dropped === 0) shape++;
    if (k.list.every(q => Math.abs(Math.abs(q.amp) - 1 * dbg.kicks[i].strength) < 1e-12)) amps++;
    if (k.list.every(q => q.x >= 0 && q.x < 2 * Math.PI && q.y >= 0 && q.y < 2 * Math.PI)) inBox++;
  });
  ok("  ... each with ITS band's sigma and blob count in EACH channel, nothing dropped", shape === kicksLog.length,
     shape + " of " + kicksLog.length + "; bands hit: " + [0, 1, 2, 3].map(b => dbg.kicks.filter(k => k.band === b).length).join(" "));
  ok("  ... each blob at amplitude amp * strength, inside the box", amps === kicksLog.length && inBox === kicksLog.length);
  ok("  ... all four bands fire on the click track (bass on the kicks, top on the hats)",
     [0, 1, 2, 3].every(b => dbg.kicks.some(k => k.band === b)));
  // never merged: a frame with two onsets has two kick() calls
  const multi = frames.filter(fr => fr.log.filter(e => e.op === "kick").length > 1);
  ok("  ... and a frame with several onsets makes several kicks, never one merged list (" + multi.length + " such frames)",
     multi.length > 0 && kicksLog.every(k => k.list.length <= 2 * 15));
  // the blobs are the seeded stream, consumed in order
  const rng = M.musicRng(1), expect = dbg.kicks.map(k => M.musicKickList(rng, plan, k.band, 1, k.strength));
  ok("  ... the blob positions are musicKickList on the page's seeded generator, consumed onset by onset",
     kicksLog.every((k, i) => JSON.stringify(k.list.map(q => [q.x, q.y, q.pol])) === JSON.stringify(expect[i].map(q => [q.x, q.y, q.pol]))));
  // order within a frame
  const orderBad = frames.filter(fr => { let stepped = false; for (const e of fr.log) { if (e.op === "step") stepped = true; else if (stepped) return true; } return false; });
  ok("in every frame the kicks come BEFORE that frame's steps", orderBad.length === 0 && frames.some(fr => fr.log.some(e => e.op === "kick") && fr.log.some(e => e.op === "step")),
     orderBad.length + " frames out of order");
  // the steps taken are the clock's
  const nStepped = frames.map(fr => fr.log.filter(e => e.op === "step").length);
  const nAsked = dbg.frames.map(f => f.n);
  ok("the steps the loop takes each frame are the count the clock returned (" + nStepped.reduce((a, b) => a + b, 0) + " steps in 720 frames)",
     dbg.frames.length === 720 && nStepped.every((n, i) => n === nAsked[i]) && nStepped.reduce((a, b) => a + b, 0) > 2500);
  // ... and that count is musicClock's, recomputed here from each frame's recorded inputs
  const st = {};
  let clockBad = 0;
  dbg.frames.forEach(f => {
    if (!f.fresh) { if (f.n !== 0) clockBad++; return; }
    const c = M.musicClock(st, { running: true, audioT: f.a, simT: f.simT, dt: f.dt, rate: 1, maxSteps: 64, lagMax: 0.25,
                                 maxAhead: 1 / 60, kicked: f.kicks > 0 });
    if (c.n !== f.n || Math.abs(c.target - f.target) > 1e-9) clockBad++;
  });
  ok("  ... which is musicClock(newest readback, audio clock, rate, cap 64, lagMax 0.25, maxAhead one frame, kicked), frame for frame",
     clockBad === 0, clockBad + " frames disagree");
  ok("  ... each frame's record names the readback it used: simT, dt and the energy e, and how many kicks it made",
     dbg.frames.filter(f => f.fresh).every(f => isFinite(f.simT) && isFinite(f.dt) && isFinite(f.e)) &&
     dbg.frames.reduce((a, f) => a + f.kicks, 0) === kicksLog.length &&
     frames.every((fr, i) => fr.log.filter(e => e.op === "kick").length === dbg.frames[i].kicks));
  ok("  ... every step is taken with cflEvery 1 for the first 200 steps and the slider's 4 after (the shared loop's rule)",
     (() => { let n = 0, both = 0; for (const fr of frames) { const ce = n < 200 ? 1 : 4;      // decided once per frame
                for (const e of fr.log) if (e.op === "step") { if (e.ce !== ce) return false; n++; } if (ce === 4) both++; } return n > 400 && both > 0; })());
  // simulation time tracks the audio
  const lag = dbg.frames.filter(f => f.a > 1 && f.fresh).map(f => f.lag);
  const simEnd = env.run("function(){ return solver.__t; }");
  // the anchor: the first frame with a readback ties simulation time 0 to that audio time
  const anchor = dbg.frames.find(f => f.rebased);
  ok("simulation time tracks rate * audio time: 12 s of audio after an anchor at " + anchor.a.toFixed(3) + " s it is at "
     + simEnd.toFixed(3) + ", and the lag is never beyond a frame",
     Math.abs(simEnd - (12 - anchor.a)) < 0.004 + 1e-9 && Math.max.apply(null, lag.map(Math.abs)) <= 1 / 60 + 0.004 + 1e-9 && dbg.rebases === 1,
     "worst |lag| " + (1e3 * Math.max.apply(null, lag.map(Math.abs))).toFixed(1) + " ms; rebases " + dbg.rebases + " (the anchor)");
  // the state mark and the stats cache see a kick
  ok("a kick moves the state mark (\"/k<n>\" appended) and so re-opens the stats cache",
     /^\d+\/\d+\/k\d+$/.test(env.run("function(){ return stateMark(); }")) &&
     env.run("function(){ return stateMark().split('/k')[1] === String(solver.nkicks); }") === true, env.run("function(){ return stateMark(); }"));
  // the debug record
  const k0 = dbg.kicks.find(k => k.band === 0 && k.tOnset > 2);
  ok("?musicdebug: each kick records tOnset <= tFire <= tApplied, its band and strength, the energy either side, and the scheduled click",
     dbg.kicks.every(k => k.tOnset < k.tFire && k.tFire <= k.tApplied + 1e-12 && isFinite(k.strength) && k.band >= 0 &&
                          k.used[0] === plan.nblobs[k.band] && k.dropped === 0) &&
     !!k0 && !!k0.sched && k0.sched.kind === "kick" && Math.abs(k0.sched.dt) < 0.03 &&
     Math.abs(k0.sched.t - 0.5 * Math.round(k0.sched.t / 0.5)) < 1e-4 && Math.abs(k0.tOnset - k0.sched.t - k0.sched.dt) < 1e-9,
     k0 ? "bass kick: onset " + k0.tOnset.toFixed(3) + ", scheduled " + k0.sched.t.toFixed(3) + ", applied " + k0.tApplied.toFixed(3) : "none");
  ok("  ... eAfter - eBefore is the model's energy per kick (1), differenced from two readbacks, kick by kick",
     dbg.kicks.filter(k => isFinite(k.eBefore)).length > 50 &&
     dbg.kicks.filter(k => isFinite(k.eBefore)).every(k => Math.abs((k.eAfter - k.eBefore) - 1) < 1e-4));
  const sum = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.summary()); }"));
  ok("  ... summary(): frames, rebases, the lag, and per band the kick count, energy jump and latency",
     sum.frames > 300 && sum.rebases === 0 && sum.lagSeconds.max <= 1 / 60 + 0.005 && sum.bands.length === 4 &&
     sum.bands[0].kicks >= 13 && sum.bands[0].energyJump.n > 0 && sum.bands[0].latencySeconds.mean > 0.03 &&
     sum.bands[0].latencySeconds.mean < 0.06 && sum.bands[0].latencyFromScheduled.n > 0,
     "bass: " + sum.bands[0].kicks + " kicks, latency " + (1e3 * sum.bands[0].latencySeconds.mean).toFixed(1) + " ms from onset, "
     + (1e3 * sum.bands[0].latencyFromScheduled.mean).toFixed(1) + " ms from the scheduled click");
  // ---- the threshold slider reaches the running detector -------------------------
  await P.input("rThr", 0.9);
  ok("the threshold slider is read by the RUNNING detector", env.run("function(){ return music.det.threshold; }") === 0.9);
  await P.input("rThr", 0.3);
  // ---- a rate change rebases, and the drag follows ---------------------------------
  P.takeLog();
  await P.input("rRate", 2);
  await P.frame(100 + 721 / 60);
  let fr = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }"));
  ok("moving the rate slider rebases the clock on the next frame (no jump to the new target) and halves the memory in simulation time",
     fr.rebased === true && fr.n === 0 && env.run("function(){ return solver.p.mu; }") === 0.5);
  await P.frame(100 + 722 / 60);
  fr = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }"));
  ok("  ... after which a frame advances rate / 60 = 1/30 of simulation time: " + fr.n + " steps of 0.004",
     fr.n === Math.ceil((2 / 60) / 0.004) || fr.n === Math.ceil((2 / 60) / 0.004) - 1, "n = " + fr.n);
  await P.input("rRate", 1);
  await P.frame(100 + 723 / 60);
  // ---- falling behind: the cap, then the rebase and the indicator --------------------
  env.run("function(){ solver.__dt = 0.0001; }");          // 64 steps a frame now cover 0.0064 of the 0.0167 asked for
  let capped = 0, reb = 0, lit = 0, maxLag = 0;
  for (let f = 724; f <= 780; f++) {
    await P.frame(100 + f / 60);
    fr = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }"));
    if (fr.n === 64) capped++;
    if (fr.rebased) reb++;
    if (isFinite(fr.lag)) maxLag = Math.max(maxLag, fr.lag);
    if (/BEHIND/.test(env.getEl("musicStrip").textContent)) lit++;
  }
  ok("a solver that cannot keep up (dt too small for 64 steps a frame): the count sits at the cap of 64",
     capped > 40, capped + " of 57 frames at the cap");
  ok("  ... the lag is never chased past 0.25 s: the clock REBASES instead, and the BEHIND indicator lights",
     reb >= 1 && maxLag <= 0.25 + 1e-9 && lit > 0, reb + " rebase(s), worst lag " + maxLag.toFixed(3) + " s, indicator lit in " + lit + " frames");
  env.run("function(){ solver.__dt = 0.004; }");
  // ---- a hidden tab: the audio goes on, no frame runs --------------------------------
  for (let f = 781; f <= 900; f++) await P.frame(100 + f / 60);       // settle, indicator out
  ok("  ... and once it can keep up again the indicator goes out", !/BEHIND/.test(env.getEl("musicStrip").textContent));
  const simBefore = env.run("function(){ return solver.__t; }");
  P.takeLog();
  await P.frame(100 + 900 / 60 + 30);                                  // 30 s later
  fr = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }"));
  const hidLog = P.takeLog();
  ok("after 30 s with no frame (a hidden tab) the clock rebases on return: no steps, no kick out of the gap, no BEHIND",
     fr.rebased === true && fr.n === 0 && hidLog.length === 0 && env.run("function(){ return solver.__t; }") === simBefore &&
     !/BEHIND/.test(env.getEl("musicStrip").textContent));
  await P.frame(100 + 901 / 60 + 30);
  ok("  ... and the next frame is an ordinary one", JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }")).n >= 4);
  // ---- Reset: the simulation time jumps back, the clock re-anchors -------------------
  await P.click("btnReset");
  P.takeLog();
  await P.frame(100 + 902 / 60 + 30);
  fr = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }"));
  ok("Reset (a rest state at t = 0): the frame right after takes no step on the stale readback",
     fr.fresh === false && fr.n === 0 && P.takeLog().filter(e => e.op === "step").length === 0);
  const rebBefore = env.run("function(){ return window.musicDebug.rebases; }");
  await P.frame(100 + 903 / 60 + 30); await P.frame(100 + 904 / 60 + 30);
  fr = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.frames[window.musicDebug.frames.length - 1]); }"));
  ok("  ... then it re-anchors at the new time and runs on (still playing, no rebase storm)",
     fr.fresh === true && fr.n >= 1 && env.run("function(){ return running; }") === true && env.run("function(){ return solver.__t; }") < 0.1);
  ok("  ... as a plain re-anchor, not as \"fell behind\": one rebase, and the BEHIND indicator stays dark",
     env.run("function(){ return window.musicDebug.rebases; }") === rebBefore + 1 && !/BEHIND/.test(env.getEl("musicStrip").textContent) &&
     env.run("function(){ return music.behind; }") === false);
  // ---- Stop ---------------------------------------------------------------------------
  await P.click("btnRun");
  P.takeLog();
  const callsBefore = A.analyserCalls;
  for (let f = 905; f <= 915; f++) await P.frame(100 + f / 60 + 30);
  ok("Stop: the source is stopped and disconnected, the page is not running, and no step, kick or analyser read follows",
     src.stopped === 1 && src.disc >= 1 && env.run("function(){ return running; }") === false && P.takeLog().length === 0 &&
     A.analyserCalls === callsBefore && env.run("function(){ return music.audio === null; }") === true && env.getEl("btnRun").textContent === "Run");
  // ---- Start again: the same context, a fresh detector and source ----------------------
  await P.click("btnRun");
  ok("Start again reuses the AudioContext and builds a fresh source and detector (warm-up counted from the new start)",
     A.ctxs.length === 1 && ctx.sources.length === 2 && ctx.sources[1].started === A.t &&
     env.run("function(){ return music.det.hist[0].length; }") === 0 && env.run("function(){ return running; }") === true);
  // ---- the bpm box and the source select stop a playing click track ------------------
  env.getEl("nBpm").value = "90"; env.getEl("nBpm").onchange({}); await turn();
  ok("changing the bpm stops the click track (the next Start builds it at the new tempo)", env.run("function(){ return running; }") === false);
  await P.click("btnRun");
  ok("  ... and it does: 16 beats at 90 bpm", ctx.sources[2].buffer.length === Math.round(16 * (60 / 90) * 44100) &&
     JSON.parse(env.run("function(){ return JSON.stringify(runManifest().forcing); }")).bpm === 90);
  // ---- a file ---------------------------------------------------------------------------
  env.getEl("selSrc").value = "file"; env.getEl("selSrc").onchange({}); await turn();
  ok("choosing another source stops what is playing", env.run("function(){ return running; }") === false);
  await P.click("btnRun");
  ok("Start with no file chosen says so and starts nothing", env.run("function(){ return running; }") === false &&
     /choose an audio file/.test(env.getEl("status").textContent) && ctx.sources.length === 3);
  const tone = new Float32Array(44100 * 2);
  for (let i = 0; i < tone.length; i++) tone[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / 44100) * (i > 30000 ? 1 : 0);
  A.fileBuffer = c => { const b = c.createBuffer(1, tone.length, 44100); b.getChannelData(0).set(tone); return b; };
  const bytes = new ArrayBuffer(16);
  env.getEl("fMusic").files = [{ name: "tone.wav", arrayBuffer: async () => bytes }];
  await P.click("btnRun");
  const fsrc = ctx.sources[3];
  ok("with a file: its bytes go to decodeAudioData, and the decoded buffer plays ONCE (no loop) through a fresh analyser",
     A.decoded === bytes && !!fsrc && fsrc.loop === false && fsrc.buffer.length === tone.length && fsrc.started === A.t &&
     env.run("function(){ return running && music.audio.kind === 'file' && music.audio.click === null; }") === true &&
     /tone\.wav, 2\.0 s/.test(env.getEl("vFile").textContent));
  ok("  ... the manifest then names the source as a file, with no bpm",
     (() => { const m = JSON.parse(env.run("function(){ return JSON.stringify(runManifest().forcing); }")); return m.source === "file" && m.bpm === null; })());
  fsrc.onended && fsrc.onended({}); await turn();
  ok("  ... and when the file ends the page stops", env.run("function(){ return running; }") === false && fsrc.onended === null);
  A.fileBuffer = () => { throw new Error("not audio"); };
  await P.click("btnRun");
  ok("a file that does not decode is reported and starts nothing", env.run("function(){ return running; }") === false &&
     /could not decode tone\.wav/.test(env.getEl("status").textContent));
  ok("through all of it the page logged no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 2).join(" | "));
}

// ===========================================================================
// 8b. without ?musicdebug: no record, no extra readbacks
// ===========================================================================
async function legPlain(state) {
  // ?musicdebug WITHOUT =energy: the record, and no readback per kick
  const D = await musicPage("?musicdebug");
  D.env.run(MODEL, 0.004);
  D.A.t = 5;
  await D.click("btnRun");
  for (let f = 1; f <= 180; f++) await D.frame(5 + f / 60);
  const dd = JSON.parse(D.env.run("function(){ return JSON.stringify({ k: window.musicDebug.kicks, f: window.musicDebug.frames.length, r: solver.__reads, e: MUSIC_DEBUG_E }); }"));
  ok("?musicdebug alone keeps the kick and frame records but makes NO readback per kick (one per frame); the per-kick energies are NaN",
     dd.e === false && dd.k.length > 10 && dd.f === 180 && dd.r <= 181 && dd.k.every(k => k.eBefore === null && k.eAfter === null && k.tApplied >= k.tFire),
     dd.k.length + " kicks, " + dd.r + " readbacks in 180 frames");
  const P = await musicPage(""), env = P.env, A = P.A;
  env.run(MODEL, 0.004);
  A.t = 5;
  await P.click("btnRun");
  let kicks = 0, steps = 0;
  for (let f = 1; f <= 180; f++) { await P.frame(5 + f / 60); const l = P.takeLog(); kicks += l.filter(e => e.op === "kick").length; steps += l.filter(e => e.op === "step").length; }
  const reads = env.run("function(){ return solver.__reads; }");
  ok("without ?musicdebug the page runs the same (" + kicks + " kicks, " + steps + " steps in 3 s) and keeps no record",
     kicks > 10 && steps > 600 && env.run("function(){ return typeof window.musicDebug; }") === "undefined" &&
     env.run("function(){ return MUSIC_DEBUG; }") === false);
  ok("  ... and makes ONE stats readback per frame, none per kick", reads <= 181 && reads >= 178, reads + " readbacks in 180 frames");
  // ---- a quiet start: the CFL dt is SIX frames long until the first kick -----------------
  // The model the solver actually presents after silence: dt at the quiescent floor
  // (cfl*dx/0.1 = 0.098 at 256^2) until a kick, then -- from the kicked step on, which the
  // solver takes at a freshly computed dt -- a small one.
  const Q = await musicPage("?musicdebug");
  Q.env.run(MODEL, 0.098);
  Q.env.run("function(){ const k = solver.kick; solver.kick = l => { solver.__dt = 0.003; return k(l); }; }");
  Q.A.t = 5;
  await Q.click("btnRun");
  let firstKick = -1;
  const perFrame = [];
  for (let f = 1; f <= 90 && (firstKick < 0 || f < firstKick + 6); f++) {
    await Q.frame(5 + f / 60);
    const l = Q.takeLog();
    perFrame.push({ f: f, steps: l.filter(e => e.op === "step").length, kicks: l.filter(e => e.op === "kick").length });
    if (firstKick < 0 && l.some(e => e.op === "kick")) firstKick = f;
  }
  const qf = JSON.parse(Q.env.run("function(){ return JSON.stringify(window.musicDebug.frames); }"));
  // (from 0.25 s on: the very first step is taken blind -- dt is not known until a step has
  // been made -- and lands 0.098 ahead; nothing can kick before the 0.5 s warm-up)
  const pre = qf.filter((f, i) => i + 1 < firstKick && f.fresh && f.a > 0.25), ahead = Math.max.apply(null, pre.map(f => -f.lag));
  ok("in the silence before the first kick the simulation is never more than a frame AHEAD of the audio (steps of 0.098 are held back)",
     firstKick > 20 && pre.length > 10 && ahead <= 1 / 60 + 1e-9 && perFrame.slice(0, firstKick - 1).every(p => p.steps <= 1),
     "first kick in frame " + firstKick + "; most ahead " + (1e3 * ahead).toFixed(1) + " ms; steps before it "
     + perFrame.slice(0, firstKick - 1).reduce((a, p) => a + p.steps, 0));
  const kf = perFrame[firstKick - 1], gapAtKick = qf[firstKick - 1].lag;
  ok("  ... and the frame that kicks takes a step AT ONCE, though the stale dt (0.098) says none fits in the " + (1e3 * gapAtKick).toFixed(0) + " ms gap",
     kf.kicks >= 1 && kf.steps === 1 && gapAtKick > 0 && gapAtKick < 0.098 - 1 / 60, kf.steps + " step(s) in the kick frame");
  const after = qf.slice(firstKick, firstKick + 5);
  ok("  ... after which the fresh dt is read back and the simulation is within a frame of the clock again, in two frames",
     after.length === 5 && after[2].dt === Math.fround(0.003) && Math.abs(after[3].lag) <= 1 / 60 + 0.003 && after.every(f => !f.rebased),
     "lag over the next frames: " + after.map(f => (1e3 * f.lag).toFixed(0)).join(" ") + " ms");
  // ---- a 120 Hz display: the detector is still fed at its nominal 60 ---------------------
  const H = await musicPage("?musicdebug");
  H.env.run(MODEL, 0.004);
  H.A.t = 5;
  await H.click("btnRun");
  let hk = 0;
  for (let f = 1; f <= 360; f++) { await H.frame(5 + f / 120); hk += H.takeLog().filter(e => e.op === "kick").length; }
  const hf = JSON.parse(H.env.run("function(){ return JSON.stringify(window.musicDebug.frames.map(f => f.n)); }"));
  ok("at 120 frames a second the analyser is read every OTHER frame (the detector's windows are counted in frames at 60)",
     H.A.analyserCalls === 180 && hk > 10, H.A.analyserCalls + " reads in 360 frames, " + hk + " kicks in 3 s");
  ok("  ... while the clock still runs every frame: half the steps per frame, the same simulation time",
     hf.length === 360 && Math.abs(H.env.run("function(){ return solver.__t; }") - 3) < 0.03 &&
     hf.slice(60).every(n => n >= 1 && n <= 4), "simulation time " + H.env.run("function(){ return solver.__t; }").toFixed(3) + " after 3 s");
  // ---- a context that is not at 44100 Hz --------------------------------------------------
  const W = await musicPage("", 48000);
  W.A.t = 2;
  await W.click("btnRun");
  ok("on a 48 kHz AudioContext the click track is synthesised at 48 kHz and the detector is built for 48 kHz",
     W.A.ctxs[0].sources[0].buffer.length === 8 * 48000 && W.A.ctxs[0].sources[0].buffer.sampleRate === 48000 &&
     W.env.run("function(){ return music.det.sr; }") === 48000 &&
     W.env.run("function(){ return music.det.bins[0][0]; }") === Math.ceil(40 / (48000 / 2048)));
  ok("with no AudioContext in the browser, Start says so and starts nothing", await (async () => {
    const Q = await musicPage("");
    Q.env.run("function(){ window.AudioContext = undefined; window.webkitAudioContext = undefined; }");
    await Q.click("btnRun");
    return Q.env.run("function(){ return running; }") === false && /no Web Audio/.test(Q.env.getEl("status").textContent); })());
  ok("  ... and the prefixed webkitAudioContext is used where that is all there is", await (async () => {
    const Q = await musicPage("");
    Q.env.run("function(C){ window.AudioContext = undefined; window.webkitAudioContext = C; }", Q.A.Ctx);
    await Q.click("btnRun");
    return Q.env.run("function(){ return running; }") === true && Q.A.ctxs.length === 1; })());
}

// ===========================================================================
// 9. the other two pages are untouched
// ===========================================================================
async function legOthers(state) {
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const env = await boot(dir, page);
    await turn();
    const q = JSON.parse(env.run(`function(){ return JSON.stringify({ hook: stepHook, mark: stateMark(), nk: solver.nkicks,
      music: typeof musicClock, spf: stepsPerFrame, run: running }); }`));
    ok(page + ": stepHook is null, music.js is not loaded, and the state mark is \"<nsteps>/<seq>\" as it always was",
       q.hook === null && q.music === "undefined" && /^\d+\/\d+$/.test(q.mark), "mark " + q.mark + ", nkicks " + q.nk);
    // one running frame steps exactly stepsPerFrame times, as before the hook existed
    env.run(`function(){ stepsPerFrame = 3; globalThis.__n = 0; const s = solver.step.bind(solver); solver.step = ce => { __n++; return s(ce); }; setRunning(true); }`);
    env.frame(); await turn();
    ok("  ... and a running frame there takes exactly `stepsPerFrame` steps (3)", env.run("function(){ return __n; }") === 3,
       env.run("function(){ return __n; }") + " steps");
    env.run("function(){ setRunning(false); }");
  }
  // the hook's contract, on the 2D page: a number, a promise, and a solver retired meanwhile
  const env = await boot(dir, "rmhd2d.html");
  await turn();
  env.run(`function(){ globalThis.__n = 0; globalThis.__seen = []; const s = solver.step.bind(solver); solver.step = ce => { __n++; return s(ce); };
    stepsPerFrame = 9; stepHook = sv => { __seen.push(sv === solver); return 5; }; setRunning(true); }`);
  env.frame(); await turn();
  ok("with a hook installed the loop steps what the hook RETURNS (5), not stepsPerFrame (9), and hands it the live solver",
     env.run("function(){ return __n; }") === 5 && env.run("function(){ return __seen.join(); }") === "true");
  env.run("function(){ __n = 0; stepHook = async sv => { await null; return 2; }; }");
  env.frame(); await turn();
  ok("  ... a hook may return a promise", env.run("function(){ return __n; }") === 2);
  // (counted on the PROTOTYPE: the replacement solver is a new object, which a wrapper on
  // the old instance would never see)
  env.run(`function(){ globalThis.__np = 0; const ps = Solver.prototype.step; Solver.prototype.step = function (ce) { __np++; return ps.call(this, ce); };
    stepHook = async sv => { await null; rebuild(); return 7; }; }`);
  env.frame(); await turn();
  ok("  ... and if the solver was replaced while the hook awaited, NO step is taken on the new one",
     env.run("function(){ return __np; }") === 0 && env.run("function(){ return solver.nsteps; }") === 0);
  env.run("function(){ __n = 0; const s = solver.step.bind(solver); solver.step = ce => { __n++; return s(ce); }; }");
  env.run("function(){ __n = 0; globalThis.__calls = 0; stepHook = sv => { __calls++; return 4; }; setRunning(false); }");
  env.frame(); await turn();
  ok("  ... the hook is not called at all while the page is paused", env.run("function(){ return __calls; }") === 0 && env.run("function(){ return __n; }") === 0);
}

// ---------------------------------------------------------------------------
const LEGS = [
  ["1. music.js is pure and self-contained", legPure],
  ["2. musicBandPlan against the offline band_plan", legPlan],
  ["3. MusicDetector against the offline CausalOnsetDetector", legDetector],
  ["4. the push rule, the generator, the blobs", legBlobs],
  ["5. the click track, and what the detector makes of it", legClick],
  ["6. the clock", legClock],
  ["7. music.html booted: the solver it builds, what the controls do", legBoot],
  ["8. music.html running: kicks, steps, the clock, Stop, a file", legRun],
  ["8b. without ?musicdebug", legPlain],
  ["9. the other two pages: hook null, loop and state mark unchanged", legOthers]
];
(async () => {
  const state = {};
  for (const [title, fn] of LEGS) {
    console.log(title);
    try { await fn(state); } catch (e) { ok("the leg ran to its end", false, String(e && e.stack || e).split("\n").slice(0, 3).join(" | ")); }
  }
  console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
  process.exit(bad ? 1 : 0);
})();
