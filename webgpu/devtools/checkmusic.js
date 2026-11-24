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
//  16  (MUSIC_STAGE_PLAN S2 / S4, step 3) the window-shaped box and the current sheets: the
//      rectangular rules (box, band plan, cutoff and dissipation, stereo and blob placement,
//      the sheet profile) against the fixture's `stage` key, written from the offline script;
//      window -> box -> grid and the auto rung as pure rules; then the page -- at its
//      defaults the shipped parameter set bit for bit, no rebuild on a resize, what the
//      box / grid / background selects build (never b0 with eqsrc), the sheets' initial
//      condition and its Reset, the memory slider with the source on, auto dropping one rung.
//      Mutation-tested by the implementer (2026-11-23, .scratch-port/mutate.py: one-line
//      mutants of music.js and the page glue against legs 1, 2 and 16 on a scratch copy).
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
    musicTabConstraints, musicInputConstraints, musicRouting, musicStartError, musicTabAudioLikely, MUSIC_SOURCES, MUSIC_TRIM_MAX, MUSIC_LDET, MUSIC_FLUX_FLOOR, MUSIC_PEAK_DECAY, musicSync, musicSyncText, MUSIC_CMAPS,
    MUSIC_SIGMA_MIN_DX, MUSIC_SIGMA_STEP, MUSIC_NBLOB_BASS, MUSIC_NBLOB_MAX,
    musicKickTypesFor, musicKickBlobs, musicNormal, musicDipoleEnergy, musicPan, musicPanX, musicPulseHit,
    musicPulseZoom, musicChannelsDiffer, musicWalkInit, musicReflect, musicWalkHit, musicWalkRelax, musicWalkField, MUSIC_KICK_TYPES,
    MUSIC_DIPOLE_SEP, MUSIC_SHEET_RATIO, MUSIC_STEREO_SPAN, MUSIC_STEREO_JITTER, MUSIC_CLICK_PANS })`, ctx);
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
  // comments and string literals out first: a MESSAGE may say "window" or "device"
  const code = src.split("\n").filter(l => !/^\s*\/\//.test(l)).join("\n")
                  .replace(/"(?:[^"\\\n]|\\.)*"/g, '""').replace(/\/\/.*$/gm, "");
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
     run(est(), { audioT: 10.12, simT: 4, dt: 0.07, maxAhead: 1 / 60 }).n === 1);
  // the hold-back is BOUNDED (review F2): past half of lagMax behind, the step is taken
  ok("  ... but only while less than half of lagMax behind: at a gap of 0.125 * rate and more the step IS taken, overshoot or not",
     run(est(), { audioT: 10.124, simT: 4, dt: 0.5, maxAhead: 1 / 60 }).n === 0 &&
     run(est(), { audioT: 10.126, simT: 4, dt: 0.5, maxAhead: 1 / 60 }).n === 1 &&
     run(est(0.5), { audioT: 10.124, simT: 4, dt: 0.5, rate: 0.5, maxAhead: 1 / 120 }).n === 0 &&
     run(est(0.5), { audioT: 10.126, simT: 4, dt: 0.5, rate: 0.5, maxAhead: 1 / 120 }).n === 1 &&
     run(est(), { audioT: 10.2, simT: 4, dt: 0.098, maxAhead: 1 / 60 }).n === 3);
  ok("  ... but on a frame that KICKED, one step is taken anyway (its dt is recomputed by the solver; the stale one says none fits)",
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: true }).n === 1 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: false }).n === 0);
  ok("  ... even when the simulation is AHEAD of the clock (the step is a small, freshly computed one), and exactly one",
     run(est(), { audioT: 10.05, simT: 4.06, dt: 0.098, maxAhead: 1 / 60, kicked: true }).n === 1 &&
     run(est(), { audioT: 10.05, simT: 4.06, dt: 0.098, maxAhead: 1 / 60, kicked: false }).n === 0);
  ok("  ... never more than the stale count where that is >= 1, and never past the cap",
     run(est(), { audioT: 10.05, simT: 4, dt: 0.0101, maxAhead: 1 / 60, kicked: true }).n === 5 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098, maxAhead: 1 / 60, kicked: true, maxSteps: 0 }).n === 0 &&
     run(est(), { audioT: 10.05, simT: 4.06, dt: 0.098, kicked: true, maxSteps: 0 }).n === 0);
  ok("  ... while an overshoot within it rounds up as usual, and without maxAhead it always rounds up",
     run(est(), { audioT: 10.05, simT: 4, dt: 0.06, maxAhead: 1 / 60 }).n === 1 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.07, maxAhead: 1 / 60 }).n === 0 &&
     run(est(), { audioT: 10.05, simT: 4, dt: 0.098 }).n === 1);
  // rebase: too far behind
  st = est(); r = run(st, { audioT: 10.3, simT: 4, dt: 0.01 });
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
  // ... with one whole step of grace: one quiescent step is by itself more than lagMax of
  // audio at a low rate, and being one step ahead in silence is not a failure (review F2)
  ok("  ... AHEAD is allowed lagMax plus one step: at rate 0.25, dt 0.098 (0.39 s of audio), 0.3 s ahead is fine and 0.7 s is not",
     run(est(0.25), { audioT: 10.05, simT: 4 + 0.0125 + 0.075, dt: 0.098, rate: 0.25 }).rebased === false &&
     run(est(0.25), { audioT: 10.05, simT: 4 + 0.0125 + 0.175, dt: 0.098, rate: 0.25 }).rebased === true &&
     run(est(), { audioT: 10.05, simT: 4.31, dt: 0, }).rebased === true && run(est(), { audioT: 10.05, simT: 4.29, dt: 0 }).rebased === false);
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
  // a hidden tab is whatever the CALLER says it is (`resumed`, from the document's
  // visibility): re-anchor quietly. A long gap with no such word is a device that cannot
  // keep up, and is `behind` (review F4)
  st = est(); r = run(st, { audioT: 25, simT: 4, dt: 0.01, resumed: true });
  ok("back from a hidden tab (`resumed`): the clock re-anchors with no step, and it is NOT `behind`",
     r.rebased === true && r.n === 0 && st.a0 === 25 && r.behind === false && r.resumed === true);
  st = est(); r = run(st, { audioT: 25, simT: 4, dt: 0.01 });
  ok("  ... the same gap on a page that was VISIBLE is a device that cannot keep up: rebase, and `behind`",
     r.rebased === true && r.n === 0 && r.behind === true && r.resumed === false);
  st = est(); r = run(st, { audioT: 10.6, simT: 4, dt: 0.01 });
  ok("  ... down to a single slow frame: 0.6 s since the last one with the simulation where it was is `behind`", r.rebased === true && r.behind === true);
  // silence at a low rate (review F2): dt sits at the quiescent floor, 0.098 at 256^2. The
  // simulation must still advance at `rate`, with no rebase after the anchor and BEHIND dark
  for (const rate of [0.25, 0.35, 1, 2]) {
    const z = {}; let sim = 0, steps = 0, reb = 0, beh = 0, dtk = 0;
    for (let f = 0; f <= 1200; f++) {
      const q = run(z, { audioT: f / 60, simT: sim, dt: dtk, rate: rate, maxAhead: rate / 60 });
      if (q.rebased) reb++;
      if (q.behind) beh++;
      sim += q.n * 0.098; steps += q.n; dtk = 0.098;
    }
    ok("20 s of silence at rate " + rate + " (dt 0.098 throughout): the simulation advances at the rate asked for, one rebase (the anchor), BEHIND never lit",
       reb === 1 && beh === 0 && Math.abs(sim - rate * 20) <= 0.098 + 0.125 * rate + 1e-9, steps + " steps, simulation time " + sim.toFixed(3) + " for " + (rate * 20).toFixed(2));
  }
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

// ===========================================================================
// 6b. the live-source rules (M3 / M4), as data
// ===========================================================================
function legRules(state) {
  const M = state.M;
  ok("the four sources, and the delay's default and range: click / file / tab / input, 50 ms in 0..150",
     JSON.stringify(Array.from(M.MUSIC_SOURCES)) === '["click","file","tab","input"]' && M.MUSIC_TRIM_MAX === 150);
  ok("a tab capture excludes system audio (a TAB's audio only -- the system's would include this page's own output)", M.musicTabConstraints().systemAudio === "exclude");
  const tc = M.musicTabConstraints();
  ok("a tab capture asks for a video track (the API requires one), as small and slow as it can be asked for",
     !!tc.video && tc.video.frameRate === 1 && tc.video.width <= 64 && tc.video.height <= 64);
  ok("  ... and for RAW audio with the captured tab silenced: no echo cancellation, noise suppression or gain control, suppressLocalAudioPlayback",
     tc.audio.echoCancellation === false && tc.audio.noiseSuppression === false && tc.audio.autoGainControl === false &&
     tc.audio.suppressLocalAudioPlayback === true);
  const ic = M.musicInputConstraints("bh2"), id = M.musicInputConstraints(""), iu = M.musicInputConstraints();
  ok("an input device is asked for RAW audio too, from exactly the device chosen, and no video",
     ic.audio.echoCancellation === false && ic.audio.noiseSuppression === false && ic.audio.autoGainControl === false &&
     ic.audio.deviceId.exact === "bh2" && ic.video === undefined);
  ok("  ... and from the default device (no deviceId at all) when none is chosen",
     !("deviceId" in id.audio) && !("deviceId" in iu.audio) && id.audio.echoCancellation === false);
  // the routing rule: who may reach the speakers
  const R = (k, sup, surf) => M.musicRouting(k, sup, surf === undefined ? "browser" : surf);
  ok("routing: the click track and a file are played by the page and so compensated, whatever `suppressed` says",
     [undefined, null, true, false].every(v => R("click", v).play && R("click", v).compensated && R("file", v).play && R("file", v).compensated));
  ok("  ... a captured tab is played ONLY if its track reports the tab silenced (=== true)",
     R("tab", true).play === true && R("tab", true).compensated === true);
  ok("  ... and with anything else -- false, undefined, null, the string \"true\", 1 -- it is NOT played (it would be heard twice)",
     [false, undefined, null, "true", 1].every(v => R("tab", v).play === false && R("tab", v).compensated === false));
  ok("  ... an input device is NEVER played, whatever is passed",
     [undefined, null, true, false, 1].every(v => R("input", v).play === false && R("input", v).compensated === false));
  ok("  ... every answer carries the sentence the page shows, and the two uncompensated ones say why",
     ["click", "file", "tab", "input"].every(k => typeof R(k, false).why === "string" && R(k, false).why.length > 20) &&
     /did not silence/.test(R("tab", false).why) && /never played/.test(R("input", true).why) && /silenced/.test(R("tab", true).why));
  ok("  ... and an unknown source is refused", /unknown source/.test(throws(() => R("speaker", true))));
  // review F6: a capture that is not a browser TAB is never played, silenced or not
  ok("  ... a capture whose video track is not displaySurface \"browser\" (a window, a monitor, unknown) is NOT played even if silenced",
     ["window", "monitor", null, "", "Browser"].every(v => M.musicRouting("tab", true, v).play === false && /not a browser tab/.test(M.musicRouting("tab", true, v).why)) &&
     M.musicRouting("tab", true, "browser").play === true && M.musicRouting("tab", true).play === false &&
     M.musicRouting("click", null, null).play === true && M.musicRouting("input", true, "browser").play === false);
  // the messages
  const E = (k, n) => M.musicStartError(k, typeof n === "string" && n !== "noaudio" && n !== "unsupported" ? { name: n } : n);
  ok("a tab capture with no audio track says what to do: pick a TAB and switch on \"Also share tab audio\"",
     /Chrome TAB/.test(E("tab", "noaudio")) && /Also share tab audio/.test(E("tab", "noaudio")) && !/tab/i.test(E("input", "noaudio").replace("audio track", "")));
  ok("  ... a cancelled or refused capture, an unplugged device, a busy device and a browser without the API each get their own sentence",
     /cancelled or not allowed/.test(E("tab", "NotAllowedError")) && /not allowed/.test(E("input", "NotAllowedError")) &&
     /not available/.test(E("input", "NotFoundError")) && /not available/.test(E("input", "OverconstrainedError")) &&
     /could not be opened/.test(E("input", "NotReadableError")) && /getDisplayMedia is missing/.test(E("tab", "unsupported")) &&
     /getUserMedia is missing/.test(E("input", "unsupported")));
  // Safari / Firefox: a tab capture comes back video-only. The wording follows the proxy,
  // and only the wording: omitted or true it is the Chrome hint, false it names the browser
  const TL = M.musicTabAudioLikely;
  ok("musicTabAudioLikely: true only with getDisplayMedia AND the suppressLocalAudioPlayback constraint listed; a throwing or missing API is false",
     TL({ getDisplayMedia() {}, getSupportedConstraints: () => ({ suppressLocalAudioPlayback: true }) }) === true &&
     TL({ getDisplayMedia() {}, getSupportedConstraints: () => ({}) }) === false &&
     TL({ getDisplayMedia() {}, getSupportedConstraints: () => ({ suppressLocalAudioPlayback: "yes" }) }) === false &&
     TL({ getSupportedConstraints: () => ({ suppressLocalAudioPlayback: true }) }) === false &&
     TL({ getDisplayMedia() {} }) === false && TL(null) === false &&
     TL({ getDisplayMedia() {}, getSupportedConstraints() { throw new Error("x"); } }) === false);
  ok("a tab capture with no audio: the Chrome hint by default and when tab audio is likely, the browser named when it is not; an input device's message does not change",
     /Also share tab audio/.test(M.musicStartError("tab", "noaudio")) && /Also share tab audio/.test(M.musicStartError("tab", "noaudio", true)) &&
     /Safari and Firefox do not/.test(M.musicStartError("tab", "noaudio", false)) && /Chrome or Edge/.test(M.musicStartError("tab", "noaudio", false)) &&
     !/Also share tab audio/.test(M.musicStartError("tab", "noaudio", false)) &&
     M.musicStartError("input", "noaudio", false) === M.musicStartError("input", "noaudio") &&
     M.musicStartError("tab", { name: "NotAllowedError" }, false) === M.musicStartError("tab", { name: "NotAllowedError" }));
  ok("  ... and anything else is reported by name, never swallowed", /\(TypeError\)/.test(E("tab", "TypeError")) &&
     /could not start/.test(M.musicStartError("input", null)) && /could not start/.test(M.musicStartError("tab", {})));
  // the detector on a stream joined mid-way: no burst while its normalisation has little history
  const F = state.F, x = state.x, sr = state.sr;
  if (x) {
    const det = new M.MusicDetector(Object.assign({ sr: sr }, F.cases[0].opts));
    const on = runDetector(det, x, sr, 60, F.cases[0].opts.nfft);
    const n = (a, b) => on.filter(e => e.tFire >= a && e.tFire < b).length;
    ok("the detector's first seconds are not a burst: the first 2 s hold no more onsets than a later 2 s (" + n(0, 2) + " vs " + n(4, 6) + ", " + n(6, 8) + ")",
       n(0, 2) <= Math.max(n(4, 6), n(6, 8)) && n(0, 0.5) === 0 && n(0, 2) > 0);
  }
}


// ===========================================================================
// 6c. sync (sound delay / kick hold-back), the noise floor, the colormaps
// ===========================================================================
function legSync(state) {
  const M = state.M, L = M.MUSIC_LDET, S = o => M.musicSync(o), near = (a, b) => Math.abs(a - b) < 1e-12;
  ok("the detection latency is a named constant: 39 ms (measured, onset to kick()) + one frame = " + (1e3 * L).toFixed(1) + " ms",
     near(L, 0.039 + 1 / 60));
  // page-played sources: both sides available
  let r = S({ kind: "click", play: true, outputLatency: 0 });
  ok("the page plays it, output latency 0 (laptop speakers): the SOUND is delayed by the detection latency, no kick is held",
     near(r.D, L) && r.H === 0 && near(r.residual, 0) && r.canDelay === true);
  r = S({ kind: "file", play: true, outputLatency: 0.02 });
  ok("  ... 20 ms: the sound is delayed by Ldet - 20 ms", near(r.D, L - 0.02) && r.H === 0 && near(r.residual, 0));
  r = S({ kind: "tab", play: true, outputLatency: 0.168 });
  ok("  ... 168 ms (Bluetooth earphones): the sound cannot be made earlier, so the KICKS are held " + (1e3 * (0.168 - L)).toFixed(0) + " ms and the sound is not delayed",
     r.D === 0 && near(r.H, 0.168 - L) && near(r.residual, 0) &&
     M.musicSyncText(r, "why") === "sound delayed 0 ms, kicks held 112 ms (output latency 168 ms, detection 56 ms): why", M.musicSyncText(r, "why"));
  r = S({ kind: "click", play: true, outputLatency: L });
  ok("  ... exactly the detection latency: neither", r.D === 0 && r.H === 0 && near(r.residual, 0));
  ok("  ... an output latency that is undefined, null, NaN, negative, infinite or a string counts as 0",
     [undefined, null, NaN, -0.1, Infinity, "0.168"].every(v => { const q = S({ kind: "click", play: true, outputLatency: v }); return near(q.D, L) && q.H === 0 && q.lout === 0; }));
  // a tab the page does not play: heard through the same output, undelayed
  r = S({ kind: "tab", play: false, outputLatency: 0 });
  ok("a tab the page does NOT play, output 0: nothing can be done -- no delay, no hold, the picture " + (1e3 * L).toFixed(0) + " ms late",
     r.D === 0 && r.H === 0 && near(r.residual, L) && r.canDelay === false && /picture still runs 56 ms after the sound/.test(M.musicSyncText(r, "w")) &&
     !/sound delayed/.test(M.musicSyncText(r, "w")));
  r = S({ kind: "tab", play: false, outputLatency: 0.168 });
  ok("  ... on a 168 ms output it is heard that late, so the kicks are held the same " + (1e3 * (0.168 - L)).toFixed(0) + " ms and it is exact",
     r.D === 0 && near(r.H, 0.168 - L) && near(r.residual, 0));
  // an input device: the sound is in the room already
  r = S({ kind: "input", play: false, outputLatency: 0.168 });
  ok("an input device: the output latency does not apply (the sound is already in the room) -- no hold, the picture Ldet late",
     r.D === 0 && r.H === 0 && r.lout === 0 && near(r.residual, L));
  ok("  ... and it is never given a sound delay, even if asked with play: true and a negative trim... unless `play` -- which the page never passes for it",
     S({ kind: "input", play: false, outputLatency: 0, trimMs: -100 }).D === 0);
  // the trim: + makes the picture later, - the sound later; it moves to whichever side is free
  ok("trim +30 ms on a 0 ms output comes off the sound's delay; +100 ms empties it and holds the kicks for the rest",
     near(S({ kind: "click", play: true, outputLatency: 0, trimMs: 30 }).D, L - 0.03) &&
     S({ kind: "click", play: true, outputLatency: 0, trimMs: 100 }).D === 0 && near(S({ kind: "click", play: true, outputLatency: 0, trimMs: 100 }).H, 0.1 - L));
  ok("  ... -50 ms adds to the sound's delay; on Bluetooth, -50 ms comes off the kick hold and -200 ms turns it into a sound delay",
     near(S({ kind: "click", play: true, outputLatency: 0, trimMs: -50 }).D, L + 0.05) &&
     near(S({ kind: "tab", play: true, outputLatency: 0.168, trimMs: -50 }).H, 0.168 - L - 0.05) &&
     near(S({ kind: "tab", play: true, outputLatency: 0.168, trimMs: -200 }).D, 0.2 - 0.168 + L) && S({ kind: "tab", play: true, outputLatency: 0.168, trimMs: -200 }).H === 0);
  ok("  ... where the sound cannot be delayed a negative trim is simply not available: both stay at 0",
     S({ kind: "tab", play: false, outputLatency: 0, trimMs: -100 }).D === 0 && S({ kind: "tab", play: false, outputLatency: 0, trimMs: -100 }).H === 0 &&
     near(S({ kind: "input", play: false, trimMs: 80 }).H, 0.08 - L));
  let inv = 0, n = 0;
  for (const play of [true, false]) for (const kind of ["click", "tab", "input"]) for (const lo of [0, 0.01, 0.0557, 0.1, 0.168, 0.4])
    for (const tr of [-150, -60, 0, 60, 150]) {
      const q = S({ kind: kind, play: play, outputLatency: lo, trimMs: tr }); n++;
      if (!(q.D >= 0 && q.H >= 0) || (q.D > 0 && q.H > 0) || (!play && q.D !== 0) || (play && Math.abs(q.residual) > 1e-12) || q.residual < -1e-12) inv++;
    }
  ok("  ... over " + n + " combinations: D and H are never negative and never both in force, D is 0 unless the page plays the source, and where it does the sync is exact",
     inv === 0, inv + " violations");
  ok("  ... both are capped (the delay node is 1 s long): an absurd trim cannot ask for more",
     S({ kind: "click", play: true, outputLatency: 0, trimMs: -5000 }).D === 1 && S({ kind: "click", play: true, outputLatency: 3 }).H === 1 &&
     S({ kind: "click", play: true, outputLatency: 3, maxH: 0.5 }).H === 0.5 && S({ kind: "click", play: true, trimMs: -5000, maxD: 0.2 }).D === 0.2);

  // ---- the noise floor (review F7), on the fixture's signal followed by noise ------------
  const x = state.x, sr = state.sr;
  if (x) {
    let rms = 0; for (let i = 0; i < x.length; i++) rms += x[i] * x[i]; rms = Math.sqrt(rms / x.length);
    const run = (db, floor) => {
      // 8 s of music, then 32 s of noise: the normaliser's 20 s window has to EMPTY of music
      // before the percentile is the noise's own, which is when the promotion starts
      const y = new Float64Array(5 * x.length); y.set(x);
      let v = 4321;
      for (let i = x.length; i < y.length; i++) { v = (1103515245 * v + 12345) % 2147483648; y[i] = rms * Math.pow(10, db / 20) * 1.7 * (v / 1073741824 - 1); }
      const det = new M.MusicDetector(Object.assign({ sr: sr, nfft: 512, nbands: 4, threshold: 0.3 }, floor ? { fluxFloor: M.MUSIC_FLUX_FLOOR, peakDecayS: M.MUSIC_PEAK_DECAY } : {}));
      const on = runDetector(det, y, sr, 60, 512), T = x.length / sr;
      return { music: on.filter(e => e.tFire <= T).length, noise: on.filter(e => e.tFire > T + 0.3).length,
               late: on.filter(e => e.tFire > T + 21).length };
    };
    const a60 = run(-60, false), b60 = run(-60, true), b80 = run(-80, true), a80 = run(-80, false);
    ok("WITHOUT the floor, noise 60 and 80 dB below the music that preceded it becomes a stream of onsets once the 20 s window holds nothing else ("
       + a60.late + " and " + a80.late + " in the last 11 s) -- the running percentile promotes any floor",
       a60.late > 20 && a80.late > 20);
    ok("WITH the page's floor (0.3 of a 120 s peak-hold of the band's own flux) that noise gives NONE, and the music's own onsets are untouched (" + b60.music + " vs " + a60.music + ")",
       b60.noise === 0 && b80.noise === 0 && b60.late === 0 && b60.music === a60.music && a60.music > 50);
  }
  ok("fluxFloor defaults to OFF (0) with a 30 s peak-hold, as the offline class does; a negative one is refused",
     new M.MusicDetector({ sr: 44100 }).fluxFloor === 0 && Math.abs(new M.MusicDetector({ sr: 44100 }).pdecay - Math.exp(-1 / 1800)) < 1e-15 &&
     /bad configuration/.test(throws(() => new M.MusicDetector({ sr: 44100, fluxFloor: -1 }))));
  // history trimming (a reviewer survivor): a run long enough to trim, against the same
  // detector with trimming made impossible -- the answers must be identical
  {
    const mk = () => new M.MusicDetector({ sr: 44100, nbands: 3, threshold: 0.1, normS: 0.5, warmupS: 0, fluxFloor: 0.2, peakDecayS: 2 });
    const d1 = mk(), d2 = mk();
    for (const h of d2.hist) h.splice = () => [];                    // never trims
    let seed = 99, out1 = [], out2 = [], maxLen = 0;
    const rnd = () => { seed = (seed * 48271) % 2147483647; return seed / 2147483647; };
    for (let f = 0; f < 600; f++) {
      const m = new Float64Array(1025);
      for (let k = 0; k < 1025; k++) m[k] = 1e-3 * rnd() + ((f % 23 === 0 && k > 20 && k < 60) ? 0.3 * rnd() : 0) + ((f % 37 === 5 && k > 300 && k < 500) ? 0.2 * rnd() : 0);
      out1 = out1.concat(d1.push(m, (f + 1) / 60)); out2 = out2.concat(d2.push(m, (f + 1) / 60));
      maxLen = Math.max(maxLen, d1.hist[0].length);
    }
    ok("the detector's history is TRIMMED on a long run (never past 2 nnorm + 8 = 68 values of " + d2.hist[0].length + ") and the onsets are those of an untrimmed one",
       maxLen <= 2 * d1.nnorm + 9 && d2.hist[0].length > 500 && out1.length > 10 && JSON.stringify(out1) === JSON.stringify(out2), out1.length + " onsets");
  }

  // ---- the two colormaps ----------------------------------------------------------------
  const cf = path.join(__dirname, "fixtures", "music_cmap.json");
  if (!fs.existsSync(cf)) { ok("fixtures/music_cmap.json is present", false, "run ../gen_musiccmap.py"); return; }
  const C = JSON.parse(fs.readFileSync(cf, "utf8")), maps = M.MUSIC_CMAPS;
  const poly = (co, t) => [0, 1, 2].map(c => { let v = co[6][c]; for (let k = 5; k >= 0; k--) v = v * t + co[k][c]; return Math.max(0, Math.min(1, v)); });
  ok("MUSIC_CMAPS is magma, brat, brat_r, each 7 rows of 3 coefficients (the demo's degree-6 form)",
     maps.length === 3 && maps[0].name === "magma" && maps[1].name === "brat" && maps[2].name === "brat_r" && maps.every(m => m.coef.length === 7 && Array.from(m.coef).every(r => r.length === 3)));
  ok("magma's coefficients are the ones gen_musiccmap.py fitted to matplotlib's table", JSON.stringify(maps[0].coef) === JSON.stringify(C.magma_coef), C.source);
  let err = 0;
  C.magma_table.forEach((row, i) => { const v = poly(maps[0].coef, i / 255); for (let c = 0; c < 3; c++) err = Math.max(err, Math.abs(v[c] - row[c])); });
  ok("  ... and the fit is within 0.023 of matplotlib's magma in every channel at all 256 points (the recorded maximum)",
     err < 0.023 && Math.abs(err - C.magma_fit_max_error) < 1e-5 && C.magma_table.length === 256, "max |fit - table| = " + err.toFixed(4));
  ok("  ... it starts black and ends pale: (0, 0, 0) at 0 within the fit, near (0.99, 0.99, 0.75) at 1",
     poly(maps[0].coef, 0).every(v => v < 0.02) && poly(maps[0].coef, 1)[0] > 0.95 && poly(maps[0].coef, 1)[2] > 0.7);
  // labels swapped 2026-11-21 (Alfred): `brat` is green -> black, `brat_r` black -> green
  const dn = maps[1], up = maps[2];
  const b1 = poly(up.coef, 1), b0 = poly(up.coef, 0);
  ok("brat_r is black at 0 and #8ACE00 at 1: (138, 206, 0) / 255",
     b0.every(v => v === 0) && b1.map(v => Math.round(255 * v)).join() === "138,206,0" && C.brat_hex === "#8ACE00" &&
     JSON.stringify(up.coef) === JSON.stringify(C.brat_r_coef));
  // brat is the cover: its green at 0 (still flow), black at 1 -- brat_r run backwards
  let rev = 0, rmono = true;
  for (let i = 0; i <= 200; i++) {
    const a = poly(dn.coef, i / 200), b = poly(up.coef, 1 - i / 200);
    rev = Math.max(rev, Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));
    if (i > 0 && !(a[1] < poly(dn.coef, (i - 1) / 200)[1])) rmono = false;
  }
  ok("brat is #8ACE00 at 0 and black at 1, brat(t) = brat_r(1 - t) to 1e-6, green strictly falling",
     poly(dn.coef, 0).map(v => Math.round(255 * v)).join() === "138,206,0" && poly(dn.coef, 1).every(v => Math.abs(v) < 1e-12) &&
     rev < 1e-6 && rmono && JSON.stringify(dn.coef) === JSON.stringify(C.brat_coef), "max |brat(t) - brat_r(1-t)| = " + rev.toExponential(1));
  let mono = true, lin = 0;
  const luma = v => 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
  for (let i = 1; i <= 200; i++) {
    const a = poly(up.coef, (i - 1) / 200), b = poly(up.coef, i / 200);
    if (!(luma(b) > luma(a)) || b[0] < a[0] || b[1] < a[1]) mono = false;
    lin = Math.max(lin, Math.abs(b[0] - (i / 200) * b1[0]), Math.abs(b[1] - (i / 200) * b1[1]), b[2]);
  }
  ok("  ... exactly linear in between, so its lightness rises monotonically and the fastest flow is the green", mono && lin < 1e-12);
}

// ---------------------------------------------------------------------------
// a fake AudioContext: the one thing node does not have. It PLAYS nothing; its
// analyser computes the true spectrum of what would be playing at `currentTime`.
// ---------------------------------------------------------------------------
// Every node records what it was connected TO (`outs`) and FROM (`ins`), so a leg can ask
// the one question that matters about a graph: what reaches the speakers, and through
// what. Streams (tab capture, an input device) carry the samples they would deliver.
// `stereo` (leg 13): the context can also split channels -- createGain, createChannelSplitter,
// buffers with a Float32Array per channel -- and an analyser fed by a splitter output reads
// THAT channel, one fed by the source reads the channel mean (Web Audio's down-mix).
function fakeAudio(sr, stereo) {
  const A = { t: 0, sr: sr || 44100, ctxs: [], decoded: null, analyserCalls: 0, outLat: 0.02 };
  const node = (kind, extra) => Object.assign({ kind: kind, outs: [], ins: [], disc: 0, chan: [],
    connect(n, out) { this.outs.push(n); n.ins.push(this); this.chan.push([n, out]); },
    disconnect() { this.disc++; for (const n of this.outs) n.ins.splice(n.ins.indexOf(this), 1); this.outs.length = 0; } }, extra);
  A.Ctx = class {
    constructor() { this.sampleRate = A.sr; this.state = "suspended"; this.destination = node("destination", {});
                    this.baseLatency = 0.005; this.sources = []; this.analysers = [];
                    this.streamSources = []; this.delays = []; this.resumed = 0; A.ctxs.push(this); }
    get currentTime() { return A.t; }
    get outputLatency() { return A.outLat; }       // the leg's to change, mid-session too
    createBuffer(ch, len, rate) {
      const d = new Float32Array(len), all = [d];
      for (let c = 1; c < ch && stereo; c++) all.push(new Float32Array(len));
      return { numberOfChannels: ch, length: len, sampleRate: rate, duration: len / rate, getChannelData: c => all[stereo ? c : 0] };
    }
    createBufferSource() {
      const s = node("buffer", { buffer: null, loop: false, started: null, stopped: 0, onended: null,
                                 start() { this.started = A.t; }, stop() { this.stopped++; } });
      this.sources.push(s);
      return s;
    }
    createMediaStreamSource(stream) {
      if (A.throwOn === "streamSource") throw new Error("createMediaStreamSource failed");
      const s = node("stream", { stream: stream });
      this.streamSources.push(s);
      return s;
    }
    createDelay(max) {
      // setTargetAtTime lands at once here, and is logged: a leg can tell a smooth move from a jump
      const d = node("delay", { maxDelay: max, delayTime: { value: 0, ramps: [],
        setTargetAtTime(v, t, tc) { this.ramps.push({ v: v, t: t, tc: tc }); this.value = v; } } });
      this.delays.push(d);
      return d;
    }
    createAnalyser() {
      if (A.throwOn === "analyser") throw new Error("createAnalyser failed");
      const an = node("analyser", { fftSize: 1024, smoothingTimeConstant: 0.8,
        getFloatFrequencyData(arr) {
          A.analyserCalls++;
          let s = this.ins[0], N = this.fftSize, ch = -1;
          if (s.kind === "splitter") { ch = s.chan.find(c => c[0] === this)[1]; s = s.ins[0].ins[0]; }   // splitter <- gain <- source
          const au = s.kind === "stream" ? s.stream.__audio : { d: s.buffer.getChannelData(0), started: s.started, loop: s.loop };
          let d = au.d;
          const n = d.length, end = Math.round((A.t - au.started) * A.sr);
          if (s.kind !== "stream" && s.buffer.numberOfChannels === 2 && stereo) {
            const l = s.buffer.getChannelData(0), r = s.buffer.getChannelData(1);
            d = ch === 0 ? l : ch === 1 ? r : Float32Array.from(l, (v, i) => 0.5 * (v + r[i]));
          }
          const at = au.loop ? i => (i < 0 ? 0 : d[i % n]) : i => (i >= 0 && i < n ? d[i] : 0);
          const mag = spectrum(at, end, N, true);
          for (let k = 0; k < arr.length; k++) arr[k] = 20 * Math.log10(mag[k]);      // -Infinity for silence
        } });
      // (a getter would be flattened to a value by Object.assign in `node`)
      Object.defineProperty(an, "frequencyBinCount", { get() { return this.fftSize / 2; } });
      this.analysers.push(an);
      return an;
    }
    async decodeAudioData(ab) { A.decoded = ab; return A.fileBuffer(this); }
    async resume() { this.resumed++; if (A.resumeHold) await A.resumeHold; this.state = "running"; }
    __node(kind) { return node(kind, {}); }
  };
  return A;
}
// does `n` reach the speakers, and through which kinds of node? null = it does not
function toSpeakers(n, seen) {
  if (n.kind === "destination") return [];
  for (const o of n.outs) { const p = toSpeakers(o, seen); if (p) return [o.kind].concat(p); }
  return null;
}
// navigator.mediaDevices, faked: every call is logged, and what each returns is the leg's
function fakeMedia(A, samples) {
  const M = { display: [], user: [], enumerations: 0, labels: false, displayError: null, userError: null,
              devices: [{ kind: "audioinput", deviceId: "default", label: "Default" },
                        { kind: "audioinput", deviceId: "bh2", label: "BlackHole 2ch" },
                        { kind: "videoinput", deviceId: "cam", label: "FaceTime Camera" },
                        { kind: "audioinput", deviceId: "mic", label: "MacBook Microphone" },
                        { kind: "audiooutput", deviceId: "spk", label: "Speakers" }] };
  const track = (kind, settings, label) => ({ kind: kind, label: label || kind, readyState: "live", stops: 0, onended: null,
    stop() { this.stops++; this.readyState = "ended"; }, getSettings() { return settings || {}; } });
  // o: { audio: settings | false, video: bool }
  M.stream = o => {
    const tr = [];
    if (o.audio !== false) tr.push(track("audio", o.audio, o.label));
    if (o.video) tr.push(track("video", { displaySurface: o.surface === undefined ? "browser" : o.surface }));
    return { __audio: { d: samples, started: A.t, loop: true }, tracks: tr,
             getTracks() { return tr.slice(); }, getAudioTracks() { return tr.filter(t => t.kind === "audio"); },
             getVideoTracks() { return tr.filter(t => t.kind === "video"); } };
  };
  M.nextDisplay = () => M.stream({ audio: { suppressLocalAudioPlayback: true }, video: true, label: "tab: a web player" });
  M.nextUser = c => M.stream({ audio: { deviceId: c.audio.deviceId ? c.audio.deviceId.exact : "default" }, label: "BlackHole 2ch" });
  M.supported = { suppressLocalAudioPlayback: true };   // a Chromium: it lists the constraint
  M.api = {
    ondevicechange: null,
    getSupportedConstraints: () => M.supported,
    getDisplayMedia: async c => { M.display.push(c); if (M.displayHold) await M.displayHold; if (M.displayError) throw M.displayError; return M.last = M.nextDisplay(c); },
    getUserMedia: async c => { M.user.push(c); if (M.userError) throw M.userError; M.labels = true; return M.last = M.nextUser(c); },
    enumerateDevices: async () => { M.enumerations++; return M.devices.map(d => ({ kind: d.kind, deviceId: d.deviceId, label: M.labels ? d.label : "" })); }
  };
  return M;
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
// `fx` true keeps the page's DEFAULT effects (MUSIC_EFFECTS_PLAN: variety kicks, stereo
// placement, the pulse, the mean field); otherwise every one is switched OFF through its
// own control first, which makes legs 7-12 the standing proof that the page with the
// effects off is the phase-1 page they were written against. Leg 13 is the effects' own.
const FX_OFF = `function(){
  el("selKick").value = "elsasser"; el("cbStereo").checked = false;
  el("rPulse").value = "0"; el("rField").value = "0";
  syncLabels();
}`;
async function musicPage(search, sr, fx) {
  const env = await boot(dir, "music.html", search ? { search: search } : undefined);
  await turn();
  if (!fx) env.run(FX_OFF);
  const A = fakeAudio(sr, fx === "stereo");
  if (fx === "stereo") {
    A.Ctx.prototype.createGain = function () { return Object.assign(this.__node("gain"), { channelCount: 1 }); };
    A.Ctx.prototype.createChannelSplitter = function (n) { return Object.assign(this.__node("splitter"), { numberOfOutputs: n }); };
  }
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
  const P = state.page = await musicPage("?plain&musicdebug=energy"), env = P.env, M = state.M;
  ok("music.html boots under the stub with no failure of its own", realFails(env).length === 0,
     realFails(env).slice(0, 3).join(" | ") || env.fails.length + " tolerated control lookups");
  const html = fs.readFileSync(path.join(dir, "music.html"), "utf8");
  const srcs = Array.from(html.matchAll(/<script src="([^"]+)"><\/script>/g)).map(m => m[1]);
  // MUSIC_STAGE_PLAN (2026-11-23): the phase-3 files follow the four, and only these may
  const PHASE3 = ["stage.js", "traces.js"];
  ok("it loads common.js, physics.js, solver2d.js, music.js -- in that order -- then only the phase-3 files (" + PHASE3.join(", ") + ")",
     srcs.slice(0, 4).join(",") === "common.js,physics.js,solver2d.js,music.js" &&
     srcs.slice(4).every((f, i, a) => PHASE3.indexOf(f) >= 0 && a.indexOf(f) === i), srcs.join(","));
  // (linked since 2026-11-24: the 2D and the 3D page each carry ONE href to it in their header
  // line, beside "further reading"; the shared scripts and the other pages still do not
  // name it, so nothing else can build a link to it)
  ok("  ... carries no analytics beacon; the 2D and 3D pages link to it once each, in their header line, and no shared script or other page does",
     !/goatcounter|gc\.zgo\.at/.test(html.replace(/<!--[\s\S]*?-->/g, "")) &&
     ["rmhd2d.html", "rmhd3d.html"].every(f => {
       const t = fs.readFileSync(path.join(dir, f), "utf8");
       return (t.match(/music\.html/g) || []).length === 1 &&
              /<a href="reading\.html">further reading<\/a>\n  &middot; <a href="music\.html">driven by music<\/a>/.test(t);
     }) &&
     ["index.html", "docs.html", "reading.html", "common.js", "physics.js", "solver2d.js"]
       .every(f => {
         const t = fs.readFileSync(path.join(dir, f), "utf8");
         // a link is an href, or the name in a string literal a page could build one from;
         // a COMMENT naming the page (common.js explains its hook) is not a link
         const code = t.split("\n").filter(l => !/^\s*(\/\/|\*|<!--)/.test(l)).join("\n");
         return !/href\s*=\s*["']?music\.html/.test(t) && !/["'`][^"'`\n]*music\.html/.test(code);
       }) &&
     /mkdir -p _site\/music\n[\s\S]*url=\.\.\/webgpu\/music\.html/.test(fs.readFileSync(path.join(dir, "..", ".github", "workflows", "pages.yml"), "utf8")));
  const yml = fs.readFileSync(path.join(dir, "..", ".github", "workflows", "pages.yml"), "utf8");
  // published 2026-11-21 (Alfred): the page deploys -- not pruned, required in the staged
  // site, and cache-busted with the other pages (music.js stamped like common.js)
  ok("  ... and pages.yml DEPLOYS music.html and music.js: not pruned, required in the staged site, and cache-busted",
     !/rm -f[^\n]*music\.(html|js)/.test(yml) &&
     /for f in[^\n]*music\.html[^\n]*\n[^\n]*music\.js/.test(yml) &&
     /src=\\"music\.js\\"\|src=\\"music\.js\?v=\$v\\"/.test(yml) &&
     /sed -i[\s\S]*?_site\/webgpu\/music\.html\n/.test(yml));
  const q = JSON.parse(env.run(`function(){ const s = solver; return JSON.stringify({ p: s.p, gk: s.g.kick, pl: !!s.pl.blobKick,
    buf: !!s.buf.kicks, lin0: s.g.gridB[1], lin1: s.g.gridB[5], k1: s.g.gridA[6], blob: s.blobMode, nk: s.nkicks,
    hook: stepHook === musicStep, run: running, plan: music.plan, kmax: KICK_MAX, mark: stateMark(),
    sel: cards.disp.map(c => c.selField.value), charts: cards.chart.map(c => c.type()), dbg: MUSIC_DEBUG,
    hasDbg: typeof window.musicDebug }); }`));
  const diss = 48 / Math.pow(256 / 3, 8);     // rate 48*amp at the cutoff n/3, hyper 4 (k^8)
  ok("the solver is 256^2 in a 2pi box, hyper 4, Pm 1, no equilibrium source",
     q.p.nx === 256 && q.p.ny === 256 && q.p.Lx === 2 * Math.PI && q.p.Ly === 2 * Math.PI && q.p.hyper === 4 && q.p.pm === 1 && q.p.eqsrc === false);
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
  ok("amp 2 -> the dissipation doubles (the offline rule scales with amp), the drag untouched, hyper still 4",
     Math.abs(v.diss / (2 * diss) - 1) < 1e-14 && v.mu === 1 && v.lin0 === -0.5 && v.hyper === 4 &&
     Math.abs(v.lin1 - Math.fround(-2 * diss * Math.pow(v.k1, 4) - 0.5)) < 1e-9, "diss " + v.diss.toExponential(4));
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
  // the graph: the analyser reads the source UNDELAYED and feeds nothing; the speakers get
  // the source through a delay node and through nothing else
  const dly = ctx.delays[0];
  ok("  ... wired source -> analyser (a dead end, at fftSize 2048 with smoothing 0) and source -> delay -> speakers",
     src.outs.length === 2 && src.outs[0] === an && src.outs[1] === dly && an.outs.length === 0 && an.ins[0] === src &&
     dly.outs.length === 1 && dly.outs[0] === ctx.destination && ctx.destination.ins.length === 1 &&
     an.fftSize === 2048 && an.smoothingTimeConstant === 0 && toSpeakers(src).join() === "delay,destination");
  // the sync in force: this fake context reports an output latency of 20 ms, less than the
  // detection latency, so the SOUND is delayed by the difference and no kick is held
  const sy0 = M.musicSync({ kind: "click", play: true, outputLatency: 0.02, trimMs: 0 });
  ok("  ... the delay is musicSync's for a 20 ms output: " + (1e3 * sy0.D).toFixed(1) + " ms on the sound, kicks not held, and the mode line says so",
     ctx.delays.length === 1 && dly.delayTime.value === sy0.D && sy0.H === 0 && Math.abs(sy0.D - (M.MUSIC_LDET - 0.02)) < 1e-15 && dly.maxDelay >= 1 &&
     /^sound delayed 36 ms, kicks held 0 ms \(output latency 20 ms, detection 56 ms\): the page plays this source$/.test(env.getEl("musicMode").textContent),
     env.getEl("musicMode").textContent);
  ok("  ... the detector is the page's: the noise floor on (0.3 of a 120 s peak-hold)",
     env.run("function(){ return music.det.fluxFloor; }") === 0.3 && M.MUSIC_FLUX_FLOOR === 0.3 && M.MUSIC_PEAK_DECAY === 120 &&
     Math.abs(env.run("function(){ return music.det.pdecay; }") - Math.exp(-1 / (120 * 60))) < 1e-15);
  const d = JSON.parse(env.run(`function(){ const d = music.det; return JSON.stringify({ sr: d.sr, nfft: d.nfft, nb: d.nbands, fps: d.fps,
    thr: d.threshold, run: running, t0: music.audio.t0, db: music.db.length, mag: music.mag.length, dbg: typeof window.musicDebug }); }`));
  ok("  ... the detector is built at the context's rate: 44100 Hz, nfft 2048, 4 bands, 60 fps, threshold 0.3",
     d.sr === 44100 && d.nfft === 2048 && d.nb === 4 && d.fps === 60 && d.thr === 0.3 && d.db === 1024 && d.mag === 1024);
  ok("  ... and the page is running, with audio time 0 at the source's first sample", d.run === true && d.t0 === 100 &&
     env.getEl("btnRun").textContent === "Stop" && d.dbg === "object");
  // ---- 12 s of frames at 60 Hz --------------------------------------------------
  const frames = [];
  for (let f = 1; f <= 720; f++) {
    await P.frame(100 + f / 60);
    frames.push({ f: f, a: f / 60, log: P.takeLog() });
  }
  const dbg = JSON.parse(env.run("function(){ return JSON.stringify({ kicks: window.musicDebug.kicks, frames: window.musicDebug.frames, rebases: window.musicDebug.rebases }); }"));
  const kicksLog = frames.flatMap(fr => fr.log.filter(e => e.op === "kick").map(e => Object.assign({ f: fr.f, a: fr.a }, e)));
  // the reference: the page's own detector configuration run by the GATE over the same audio
  const det = new M.MusicDetector({ sr: 44100, nfft: 2048, nbands: 4, fps: 60, threshold: 0.3,
                                    fluxFloor: M.MUSIC_FLUX_FLOOR, peakDecayS: M.MUSIC_PEAK_DECAY });
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
  // `lag` is taken before the frame's steps (about a frame's advance, by construction);
  // `lagAfter` is where the frame ends, and in this model -- a constant, known dt -- it is
  // exactly lag - n * dt, never positive by more than nothing and never negative by a step
  const la = dbg.frames.filter(f => f.fresh && f.a > 1);
  ok("  ... `lagAfter` = lag - n * dt / rate: where the frame ENDS. Steady, it is within one step of the clock while `lag` sits a frame behind",
     la.every(f => Math.abs(f.lagAfter - (f.lag - f.n * f.dt)) < 1e-12) &&
     la.every(f => f.lagAfter <= 1e-9 && f.lagAfter > -0.004 - 1e-9) && la.filter(f => f.lag > 0.012).length > la.length / 2,
     "lagAfter in [" + (1e3 * Math.min.apply(null, la.map(f => f.lagAfter))).toFixed(2) + ", "
     + (1e3 * Math.max.apply(null, la.map(f => f.lagAfter))).toFixed(2) + "] ms");
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
  env.run("function(){ document.visibilityState = 'hidden'; document.onvisibilitychange({}); document.visibilityState = 'visible'; document.onvisibilitychange({}); }");
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
     src.stopped === 1 && src.disc >= 1 && dly.disc >= 1 && ctx.destination.ins.length === 0 &&
     env.run("function(){ return running; }") === false && P.takeLog().length === 0 &&
     A.analyserCalls === callsBefore && env.run("function(){ return music.audio === null; }") === true && env.getEl("btnRun").textContent === "Start");
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
  const D = await musicPage("?plain&musicdebug");
  D.env.run(MODEL, 0.004);
  D.A.t = 5;
  await D.click("btnRun");
  for (let f = 1; f <= 180; f++) await D.frame(5 + f / 60);
  const dd = JSON.parse(D.env.run("function(){ return JSON.stringify({ k: window.musicDebug.kicks, f: window.musicDebug.frames.length, r: solver.__reads, e: MUSIC_DEBUG_E }); }"));
  ok("?musicdebug alone keeps the kick and frame records but makes NO readback per kick (one per frame); the per-kick energies are NaN",
     dd.e === false && dd.k.length > 10 && dd.f === 180 && dd.r <= 181 && dd.k.every(k => k.eBefore === null && k.eAfter === null && k.tApplied >= k.tFire),
     dd.k.length + " kicks, " + dd.r + " readbacks in 180 frames");
  const P = await musicPage("?plain"), env = P.env, A = P.A;
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
  const Q = await musicPage("?plain&musicdebug");
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
  const H = await musicPage("?plain&musicdebug");
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
  const W = await musicPage("?plain", 48000);
  W.A.t = 2;
  await W.click("btnRun");
  ok("on a 48 kHz AudioContext the click track is synthesised at 48 kHz and the detector is built for 48 kHz",
     W.A.ctxs[0].sources[0].buffer.length === 8 * 48000 && W.A.ctxs[0].sources[0].buffer.sampleRate === 48000 &&
     W.env.run("function(){ return music.det.sr; }") === 48000 &&
     W.env.run("function(){ return music.det.bins[0][0]; }") === Math.ceil(40 / (48000 / 2048)));
  ok("with no AudioContext in the browser, Start says so and starts nothing", await (async () => {
    const Q = await musicPage("?plain");
    Q.env.run("function(){ window.AudioContext = undefined; window.webkitAudioContext = undefined; }");
    await Q.click("btnRun");
    return Q.env.run("function(){ return running; }") === false && /no Web Audio/.test(Q.env.getEl("status").textContent); })());
  ok("  ... and the prefixed webkitAudioContext is used where that is all there is", await (async () => {
    const Q = await musicPage("?plain");
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

// ===========================================================================
// 10. live sources and the delayed speaker branch (M3 / M4), on the stub
// ===========================================================================
async function legLive(state) {
  const M = state.M;
  const click = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 }).samples;
  const P = await musicPage("?plain&musicdebug"), env = P.env, A = P.A, FM = fakeMedia(A, click);
  env.run("function(m){ navigator.mediaDevices = m; }", FM.api);
  env.run(MODEL, 0.004);
  const st = () => JSON.parse(env.run(`function(){ const a = music.audio; return JSON.stringify({ run: running, starting: !!music.starting,
    live: !!a, kind: a && a.kind, comp: a && a.route.compensated, play: a && a.route.play, sup: a && a.suppressed,
    hist: music.det ? music.det.hist[0].length : -1, ref: music.det ? music.det.ref : -1, sr: music.det ? music.det.sr : -1,
    status: el("status").textContent, mode: el("musicMode").textContent, dbg: window.musicDebug ? window.musicDebug.audio : null }); }`));
  const pick = async v => { env.getEl("selSrc").value = v; env.getEl("selSrc").onchange({}); await turn(); };
  const ctx0 = () => A.ctxs[0];
  let T = 10;
  const run = async (sec) => { let k = 0; const n = Math.round(sec * 60); for (let f = 1; f <= n; f++) { await P.frame(T + f / 60); k += P.takeLog().filter(e => e.op === "kick").length; } T += sec; return k; };

  // ---- the select, before anything is started -------------------------------------
  ok("the source select offers the four sources", Array.from(env.getEl("selSrc").options).map(o => o.value).join() === "tab,file,input,click");
  ok("before Start the mode line says what each source will get: the click track compensated",
     /^sound delayed 56 ms, kicks held 0 ms \(output latency 0 ms, detection 56 ms\): the page plays this source$/.test(env.getEl("musicMode").textContent) &&
     env.getEl("rTrim").value === "0" && env.getEl("vTrim").textContent === "0", env.getEl("musicMode").textContent);
  await pick("tab");
  ok("  ... a tab: decided at Start (it depends on whether this browser silences the tab)", /decided at Start/.test(env.getEl("musicMode").textContent));
  await pick("input");
  ok("  ... an input device: NOT compensated, because it is never played back",
     /^kicks held 0 ms \(output latency 0 ms, detection 56 ms\) -- the picture still runs 56 ms after the sound: an input device is never played back$/.test(env.getEl("musicMode").textContent),
     env.getEl("musicMode").textContent);
  // the device list: audio INPUTS only; no names before access has been allowed
  const opts = () => Array.from(env.getEl("selDev").options).map(o => o.value + "=" + o.textContent);
  ok("choosing it shows the device row and lists the audio inputs -- nameless until access is allowed, and never an output or a camera",
     env.getEl("rowDev").style.display === "" && FM.enumerations >= 1 &&
     opts().join("|") === "=default input|bh2=audio input 2|mic=audio input 3", opts().join("|"));
  await pick("click");
  ok("  ... and the row is hidden again for the other sources", env.getEl("rowDev").style.display === "none");

  // ---- a TAB, silenced by the browser: played by the page, delayed -------------------
  await pick("tab");
  A.t = T;
  await P.click("btnRun");
  let q = st(), ctx = ctx0();
  const s1 = FM.last, at1 = s1.getAudioTracks()[0], vt1 = s1.getVideoTracks()[0];
  ok("Start on \"another tab's audio\" calls getDisplayMedia ONCE, with musicTabConstraints",
     FM.display.length === 1 && JSON.stringify(FM.display[0]) === JSON.stringify(M.musicTabConstraints()) && FM.user.length === 0);
  const ss = ctx.streamSources[0], an = ctx.analysers[0], dl = ctx.delays[0];
  ok("  ... the stream becomes a MediaStreamSource feeding the analyser (a dead end)",
     q.live && q.run && q.kind === "tab" && ctx.streamSources.length === 1 && ss.stream === s1 && ss.outs[0] === an && an.outs.length === 0);
  ok("  ... the track reports the tab silenced, so the page plays it: source -> delay (50 ms) -> speakers, and nothing else reaches them",
     q.sup === true && q.play === true && q.comp === true && toSpeakers(ss).join() === "delay,destination" &&
     dl.delayTime.value === M.musicSync({ kind: "tab", play: true, outputLatency: 0.02 }).D &&
     ctx.destination.ins.length === 1 && ctx.destination.ins[0] === dl &&
     /^sound delayed 36 ms, kicks held 0 ms \(output latency 20 ms, detection 56 ms\): the tab is silenced and the page plays its audio$/.test(q.mode), q.mode);
  ok("  ... both tracks are left LIVE (the unused video track is kept: stopping it is unverified browser behaviour)",
     at1.readyState === "live" && vt1.readyState === "live" && at1.stops === 0 && vt1.stops === 0 && typeof at1.onended === "function");
  const kTab = await run(3);
  ok("  ... and the captured audio drives the flow like any other source: " + kTab + " kicks in 3 s, none in the first half second",
     kTab > 10 && JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.kicks.map(k => k.tFire)); }")).every(t => t >= 0.5));
  ok("?musicdebug records the source: kind, label, sample rate, latencies, the delay in force, suppression, compensation, the video track",
     q.dbg.kind === "tab" && q.dbg.label === "tab: a web player" && q.dbg.sampleRate === 44100 && q.dbg.baseLatency === 0.005 &&
     q.dbg.outputLatency === 0.02 && Math.abs(q.dbg.delayMs - 1e3 * (M.MUSIC_LDET - 0.02)) < 1e-9 && q.dbg.holdMs === 0 && q.dbg.trimMs === 0 &&
     Math.abs(q.dbg.ldetMs - 1e3 * M.MUSIC_LDET) < 1e-9 && q.dbg.outputLatencyUsedMs === 20 && Math.abs(q.dbg.residualMs) < 1e-9 &&
     q.dbg.surface === "browser" && q.dbg.mode === q.mode && q.dbg.suppressed === true && q.dbg.compensated === true &&
     q.dbg.played === true && q.dbg.videoTrackAlive === true && q.dbg.trackSettings.suppressLocalAudioPlayback === true &&
     q.dbg.nfft === 2048 && q.dbg.bins === 1024, JSON.stringify(q.dbg).slice(0, 90));
  // the trim slider, live: +30 ms makes the PICTURE later. Here the sound is being delayed
  // 36 ms, so the trim comes off that first; -30 ms adds to it
  const ramps0 = dl.delayTime.ramps.length;
  await P.input("rTrim", 30);
  q = st();
  ok("the trim slider moves the live delay node SMOOTHLY (setTargetAtTime, never a jump): +30 ms takes 30 ms off the sound's delay",
     Math.abs(dl.delayTime.value - (M.MUSIC_LDET - 0.02 - 0.03)) < 1e-12 && dl.delayTime.ramps.length === ramps0 + 1 &&
     dl.delayTime.ramps[ramps0].tc > 0 && /^sound delayed 6 ms, kicks held 0 ms \(output latency 20 ms, detection 56 ms, trim \+30 ms\)/.test(q.mode) &&
     Math.abs(q.dbg.trimMs - 30) < 1e-9 && env.getEl("vTrim").textContent === "+30", q.mode);
  await P.input("rTrim", 100);
  q = st();
  ok("  ... past what the sound's delay can give, the rest goes to holding the KICKS: +100 ms = sound 0 ms, kicks held 64 ms",
     dl.delayTime.value === 0 && Math.abs(q.dbg.holdMs - 1e3 * (0.02 - M.MUSIC_LDET + 0.1)) < 1e-9 && q.dbg.delayMs === 0 &&
     /^sound delayed 0 ms, kicks held 64 ms/.test(q.mode) && ctx.destination.ins.length === 1 && toSpeakers(ss).join() === "delay,destination", q.mode);
  await P.input("rTrim", -50);
  q = st();
  ok("  ... and a negative trim makes the sound later: -50 ms = sound delayed 86 ms, kicks not held",
     Math.abs(dl.delayTime.value - (M.MUSIC_LDET - 0.02 + 0.05)) < 1e-12 && q.dbg.holdMs === 0 && /trim -50 ms/.test(q.mode));
  await P.input("rTrim", 0);
  ok("  ... and `summary()` carries the audio record", Math.abs(JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.summary().audio); }")).delayMs - 1e3 * (M.MUSIC_LDET - 0.02)) < 1e-9);
  // the capture ends from outside (Chrome's "Stop sharing")
  at1.readyState = "ended"; at1.onended({}); await turn();
  q = st();
  ok("the capture ending from outside (the audio track's `ended`) stops the page, stops every track, and says so",
     q.run === false && q.live === false && vt1.stops === 1 && /tab capture ended/.test(q.status) && ctx.destination.ins.length === 0 && ss.disc >= 1);
  const kAfter = await run(0.5);
  ok("  ... and nothing is kicked or stepped afterwards", kAfter === 0 && A.analyserCalls > 0);

  // ---- a TAB the browser did NOT silence: analysed, never played ----------------------
  for (const [tag, settings] of [["reports suppressLocalAudioPlayback false", { suppressLocalAudioPlayback: false }],
                                 ["does not report it at all", {}]]) {
    FM.nextDisplay = () => FM.stream({ audio: settings, video: true, label: "tab: a web player" });
    const nd = ctx.delays.length;
    A.t = T;
    await P.click("btnRun");
    q = st();
    const s2 = ctx.streamSources[ctx.streamSources.length - 1];
    ok("a tab whose track " + tag + ": the page does NOT play it (it is already heard) -- no delay node, nothing reaches the speakers",
       q.live && q.kind === "tab" && q.sup === false && q.play === false && q.comp === false && ctx.delays.length === nd &&
       toSpeakers(s2) === null && ctx.destination.ins.length === 0 && s2.outs.length === 1 && s2.outs[0].kind === "analyser");
    ok("  ... and says so: no \"sound delayed\", the picture 36 ms late (the tab is heard through the same 20 ms output, undelayed), with the reason",
       /^kicks held 0 ms \(output latency 20 ms, detection 56 ms\) -- the picture still runs 36 ms after the sound: this browser did not silence the captured tab/.test(q.mode) &&
       q.dbg.compensated === false && q.dbg.played === false && q.dbg.delayMs === 0 && q.dbg.holdMs === 0 &&
       Math.abs(q.dbg.residualMs - 1e3 * (M.MUSIC_LDET - 0.02)) < 1e-9 && q.dbg.suppressed === false, q.mode);
    const k2 = await run(1.5);
    ok("  ... it still drives the flow (" + k2 + " kicks in 1.5 s)", k2 > 3);
    await P.click("btnRun");
    ok("  ... Stop stops BOTH tracks of the capture", FM.last.getTracks().every(t => t.stops === 1 && t.readyState === "ended") && st().run === false);
  }
  FM.nextDisplay = () => FM.stream({ audio: { suppressLocalAudioPlayback: true }, video: true, label: "tab: a web player" });

  // ---- a browser that does not capture tab audio at all (Safari, Firefox) --------------
  // the capture comes back video-only AND the constraint is not listed: the message names
  // the browser rather than sending the visitor to a toggle that does not exist there
  FM.supported = {};
  FM.nextDisplay = () => FM.stream({ audio: false, video: true });
  await P.click("btnRun");
  ok("a video-only capture in a browser that does not list suppressLocalAudioPlayback: nothing starts, and the message says to use Chrome or Edge or a file (not the picker toggle)",
     st().run === false && /Safari and Firefox do not/.test(st().status) && /Chrome or Edge/.test(st().status) &&
     !/Also share tab audio/.test(st().status) && FM.last.tracks.every(t => t.readyState === "ended"), st().status);
  FM.supported = { suppressLocalAudioPlayback: true };

  // ---- a capture with NO audio track ---------------------------------------------------
  FM.nextDisplay = () => FM.stream({ audio: false, video: true });
  const nAn = ctx.analysers.length;
  await P.click("btnRun");
  q = st();
  ok("a capture with no audio track (a window or screen, or the toggle off): nothing starts, the video track is stopped, and the message says what to do",
     q.run === false && q.live === false && q.starting === false && FM.last.getVideoTracks()[0].stops === 1 &&
     /Also share tab audio/.test(q.status) && /Chrome TAB/.test(q.status) && ctx.analysers.length === nAn, q.status.slice(0, 60));
  // ---- cancelled / refused ----------------------------------------------------------------
  FM.displayError = { name: "NotAllowedError", message: "Permission denied" };
  await P.click("btnRun");
  q = st();
  ok("a cancelled or refused capture: nothing starts, and the page says so", q.run === false && q.live === false && q.starting === false &&
     /cancelled or not allowed/.test(q.status) && ctx.analysers.length === nAn);
  FM.displayError = null;
  FM.nextDisplay = () => FM.stream({ audio: { suppressLocalAudioPlayback: true }, video: true, label: "tab: a web player" });
  await P.click("btnRun");
  ok("  ... and the next Start works (the page is not left stuck in \"starting\")", st().run === true && st().kind === "tab");

  // ---- switching source while one is live --------------------------------------------------
  await run(1);
  const sLive = FM.last, histBefore = st().hist, detBefore = env.run("function(){ return music.det; }");
  await pick("click");
  q = st();
  ok("choosing another source while a tab is live stops it: every track stopped, the page not running",
     histBefore > 30 && q.run === false && q.live === false && sLive.getTracks().every(t => t.stops === 1));
  A.t = T;
  await P.click("btnRun");
  q = st();
  ok("  ... and the next source gets a NEW detector with no history (its normalisation and reference start again)",
     q.run && q.kind === "click" && q.hist === 0 && q.ref === 0 && env.run("function(){ return music.det; }") !== detBefore &&
     env.run("function(){ return music.tPush; }") === null);
  const kSw = await run(2);
  const firstK = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.kicks.map(k => k.tFire)); }"));
  ok("  ... with no kick in its first half second and no burst after (" + kSw + " kicks in 2 s; the debug record restarts with the source)",
     firstK.length === kSw && firstK.every(t => t >= 0.5) && kSw > 5 && kSw < 30 && st().dbg.kind === "click" && st().dbg.suppressed === null);
  await P.click("btnRun");

  // ---- a switch while the picker is still open ----------------------------------------------
  await pick("tab");
  let release;
  FM.displayHold = new Promise(r => { release = r; });
  const nDisp = FM.display.length;
  const pending = env.getEl("btnRun").onclick({});
  await turn();
  ok("while the picker is open the page is \"starting\", not running", st().starting === true && st().run === false && FM.display.length === nDisp + 1);
  env.getEl("btnRun").onclick({}); await turn();
  ok("  ... a second click on Start does not open a second picker", FM.display.length === nDisp + 1);
  await pick("click");
  FM.displayHold = null; release(); await turn(); await pending; await turn();
  q = st();
  ok("  ... and if the source is changed before the picker answers, the capture it returns is given straight back: tracks stopped, nothing started",
     q.run === false && q.live === false && q.starting === false && FM.last.getTracks().every(t => t.stops === 1) && ctx.destination.ins.length === 0);

  // ---- an INPUT DEVICE: analysed, never played -------------------------------------------------
  await pick("input");
  const nd2 = ctx.delays.length;
  env.getEl("selDev").value = "bh2";
  A.t = T;
  await P.click("btnRun");
  q = st();
  const si = ctx.streamSources[ctx.streamSources.length - 1];
  ok("Start on an input device calls getUserMedia with musicInputConstraints(the chosen device): raw audio, deviceId exact",
     FM.user.length === 1 && JSON.stringify(FM.user[0]) === JSON.stringify(M.musicInputConstraints("bh2")) && q.run && q.kind === "input");
  ok("  ... it is NEVER connected to the speakers: source -> analyser only, no delay node created, the destination has no input",
     toSpeakers(si) === null && si.outs.length === 1 && si.outs[0].kind === "analyser" && ctx.delays.length === nd2 &&
     ctx.destination.ins.length === 0 && q.play === false && q.comp === false);
  ok("  ... the page says the picture runs 56 ms after the sound (the output latency does not apply: the sound is already in the room)",
     /^kicks held 0 ms \(output latency 0 ms, detection 56 ms\) -- the picture still runs 56 ms after the sound: an input device is never played back$/.test(q.mode) &&
     q.dbg.kind === "input" && q.dbg.played === false && q.dbg.delayMs === 0 && q.dbg.holdMs === 0 && q.dbg.outputLatencyUsedMs === 0 &&
     q.dbg.outputLatency === 0.02 && q.dbg.compensated === false && q.dbg.suppressed === null && q.dbg.surface === null && q.dbg.videoTrackAlive === false, q.mode);
  await turn();
  ok("  ... once access is allowed the device list has its names, and the choice is kept",
     opts().join("|") === "=default input|bh2=BlackHole 2ch|mic=MacBook Microphone" && env.getEl("selDev").value === "bh2", opts().join("|"));
  await P.input("rTrim", -100);
  ok("  ... the trim slider cannot give it a path to the speakers (a negative trim has no sound to delay), though a positive one holds its kicks",
     ctx.delays.length === nd2 && ctx.destination.ins.length === 0 && st().dbg.delayMs === 0 && st().dbg.holdMs === 0);
  await P.input("rTrim", 80);
  ok("  ... +80 ms: kicks held " + (80 - 1e3 * M.MUSIC_LDET).toFixed(0) + " ms", Math.abs(st().dbg.holdMs - (80 - 1e3 * M.MUSIC_LDET)) < 1e-9 && ctx.destination.ins.length === 0);
  await P.input("rTrim", 0);
  const kIn = await run(1.5);
  ok("  ... it drives the flow (" + kIn + " kicks in 1.5 s)", kIn > 3);
  const sIn = FM.last;
  env.getEl("selDev").value = "mic"; env.getEl("selDev").onchange({}); await turn();
  ok("choosing another device while one is live stops it (its track is released)", st().run === false && sIn.getTracks()[0].stops === 1);
  // even a track that claims the tab-silencing flag is not played: the rule is the source KIND
  FM.nextUser = () => FM.stream({ audio: { suppressLocalAudioPlayback: true }, label: "odd device" });
  env.getEl("selDev").value = "";
  A.t = T;
  await P.click("btnRun");
  ok("the default device is asked for with NO deviceId; and an input is not played even if its track claimed to be \"suppressed\"",
     !("deviceId" in FM.user[FM.user.length - 1].audio) && toSpeakers(ctx.streamSources[ctx.streamSources.length - 1]) === null &&
     ctx.destination.ins.length === 0 && st().play === false);
  // the device unplugged
  const tIn = FM.last.getAudioTracks()[0];
  tIn.readyState = "ended"; tIn.onended({}); await turn();
  ok("an input that ends (unplugged) stops the page and says so", st().run === false && /audio input ended/.test(st().status));
  // refused / missing
  FM.userError = { name: "NotAllowedError" };
  await P.click("btnRun");
  ok("a refused input: nothing starts, with the permission hint", st().run === false && /not allowed/.test(st().status) && /microphone permission/.test(st().status));
  FM.userError = { name: "NotFoundError" };
  await P.click("btnRun");
  ok("an unplugged input: nothing starts, and it is called unavailable", st().run === false && /not available/.test(st().status));
  FM.userError = null;
  // a device list that changes under the page
  FM.devices.push({ kind: "audioinput", deviceId: "usb", label: "USB Interface" });
  const en = FM.enumerations;
  FM.api.ondevicechange && FM.api.ondevicechange({}); await turn();
  ok("a `devicechange` re-reads the list", FM.enumerations === en + 1 && /usb=USB Interface/.test(opts().join("|")));

  // ---- a browser with no capture API ---------------------------------------------------------
  env.run("function(){ navigator.mediaDevices = { enumerateDevices: async () => [] }; }");
  await pick("tab"); await P.click("btnRun");
  const noTab = st().status;
  await pick("input"); await P.click("btnRun");
  ok("without getDisplayMedia / getUserMedia the two live sources say the browser cannot do it, and the click track still works",
     /getDisplayMedia is missing/.test(noTab) && /getUserMedia is missing/.test(st().status) && st().run === false);
  env.run("function(){ navigator.mediaDevices = undefined; }");
  await pick("click"); A.t = T; await P.click("btnRun");
  ok("  ... and with no mediaDevices at all the page still starts its own sources", st().run === true && st().kind === "click");
  await P.click("btnRun");

  // ---- the file source goes through the same delayed branch -------------------------------------
  const tone = new Float32Array(44100);
  A.fileBuffer = c => { const b = c.createBuffer(1, tone.length, 44100); b.getChannelData(0).set(tone); return b; };
  env.getEl("fMusic").files = [{ name: "t.wav", arrayBuffer: async () => new ArrayBuffer(8) }];
  await pick("file"); A.t = T; await P.click("btnRun");
  const fs2 = ctx.sources[ctx.sources.length - 1];
  ok("a file is played through the delay too: source -> delay -> speakers, source -> analyser, compensated",
     st().run && toSpeakers(fs2).join() === "delay,destination" && fs2.outs[0].kind === "analyser" && fs2.loop === false &&
     st().comp === true && st().dbg.kind === "file" && st().dbg.label === "t.wav");
  const man = JSON.parse(env.run("function(){ return JSON.stringify(runManifest().forcing); }"));
  ok("  ... and a saved params.json records the source, that it was compensated, and by how much",
     man.source === "file" && man.compensated === true && Math.abs(man.delayMs - 1e3 * (M.MUSIC_LDET - 0.02)) < 1e-9 && man.holdMs === 0 && man.trimMs === 0);
  await P.click("btnRun");
  ok("through all of it the page logged no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 2).join(" | "));

  // ---- a 48 kHz context with a live source -----------------------------------------------------
  const W = await musicPage("?plain", 48000), FW = fakeMedia(W.A, M.musicClickTrack({ sr: 48000, bpm: 120, beats: 16, seed: 1 }).samples);
  W.env.run("function(m){ navigator.mediaDevices = m; }", FW.api);
  W.env.run(MODEL, 0.004);
  W.env.getEl("selSrc").value = "tab"; W.env.getEl("selSrc").onchange({}); await turn();
  W.A.t = 3; await W.click("btnRun");
  let k48 = 0;
  for (let f = 1; f <= 150; f++) { await W.frame(3 + f / 60); k48 += W.takeLog().filter(e => e.op === "kick").length; }
  ok("a captured tab on a 48 kHz context: the detector's bands follow the context's rate, and it kicks (" + k48 + " in 2.5 s)",
     W.env.run("function(){ return music.det.sr; }") === 48000 &&
     W.env.run("function(){ return music.det.bins[3][1]; }") === Math.ceil(10000 / (48000 / 2048)) && k48 > 8);
}

// ===========================================================================
// 11. the review's findings and the output-latency sync, on the page
// ===========================================================================
async function legFix(state) {
  const M = state.M, L = M.MUSIC_LDET;
  const click = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 }).samples;
  const mk = async (search) => {
    const P = await musicPage(search === undefined ? "?plain&musicdebug" : search), FM = fakeMedia(P.A, click);
    P.env.run("function(m){ navigator.mediaDevices = m; }", FM.api);
    P.env.run(MODEL, 0.004);
    P.FM = FM;
    P.st = () => JSON.parse(P.env.run(`function(){ const a = music.audio; return JSON.stringify({ run: running, starting: !!music.starting,
      live: !!a, kind: a && a.kind, play: a && a.route.play, q: music.queue.length, behind: music.behind, clock: music.clock,
      status: el("status").textContent, mode: el("musicMode").textContent, strip: el("musicStrip").textContent,
      btn: el("btnRun").textContent, dropped: music.dropped, dbg: window.musicDebug ? window.musicDebug.audio : null,
      kicks: window.musicDebug ? window.musicDebug.kicks : [] }); }`));
    P.pick = async v => { P.env.getEl("selSrc").value = v; P.env.getEl("selSrc").onchange({}); await turn(); };
    P.T = 10;
    P.run = async (sec, fps) => { const logs = [], n = Math.round(sec * (fps || 60)); for (let f = 1; f <= n; f++) { await P.frame(P.T + f / (fps || 60)); logs.push(P.takeLog()); } P.T += sec; return logs; };
    return P;
  };
  const kicksIn = logs => logs.reduce((a, l) => a + l.filter(e => e.op === "kick").length, 0);

  // ---- colormaps: on this page only ---------------------------------------------------------
  let P = await mk(), env = P.env, A = P.A;
  const cm = JSON.parse(env.run(`function(){ const c = cards.disp[0]; return JSON.stringify({ names: cmapNames(), shared: CMAP_NAMES,
    opts: Array.from(c.selCmap.options).map(o => cmapNames()[+o.value]), sel: c.selCmap.value, cmap: c.cmap(), field: c.selField.value,
    rgb4: cmapRGB(4, 0.37), rgb5: cmapRGB(5, 0), rgb0: cmapRGB(0, 0.37), code: solver.pl.colorize.__code, dflt: MUSIC_CMAP_DEFAULT, arrows: c.cbArrow.checked }); }`));
  ok("the music page's display starts with the arrow overlay OFF (the shared default is on)", cm.arrows === false);
  ok("the music page's colormap menu is the shared four plus magma, brat and brat_r, and the speed display starts on magma",
     cm.names.join() === "afmhot,viridis,RdBu,grayscale,magma,brat,brat_r" && cm.shared.length === 4 && cm.opts.join() === cm.names.join() &&
     cm.sel === "4" && cm.cmap === 4 && cm.dflt === 4 && cm.field === "4", cm.opts.join(" "));
  const polyOf = (co, t) => [0, 1, 2].map(c => { let v = co[6][c]; for (let k = 5; k >= 0; k--) v = v * t + co[k][c]; return Math.max(0, Math.min(1, v)); });
  ok("the CPU twin (cmapRGB: the colour bar, the saved PNG's bar) evaluates the SAME table: magma at 0.37, brat's green at 0, afmhot unmoved",
     JSON.stringify(cm.rgb4) === JSON.stringify(polyOf(M.MUSIC_CMAPS[0].coef, 0.37)) && cm.rgb5.map(v => Math.round(255 * v)).join() === "138,206,0" &&
     JSON.stringify(cm.rgb0) === JSON.stringify([0.74, 0.24, 0]));
  // the emitted display kernel carries exactly those coefficients
  const lits = tag => { const m = new RegExp("if \\(which == " + tag + "u\\) \\{[^\\n]*\\n\\s*return cpoly6\\(t, ([\\s\\S]*?)\\);\\n").exec(cm.code);
                        return m ? Array.from(m[1].matchAll(/vec3<f32>\(([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\)/g)).map(v => [+v[1], +v[2], +v[3]]) : null; };
  ok("the EMITTED colorize kernel has branches 4 (magma) and 5 (brat) whose vec3 literals are MUSIC_CMAPS' coefficients, number for number",
     JSON.stringify(lits(4)) === JSON.stringify(M.MUSIC_CMAPS[0].coef) && JSON.stringify(lits(5)) === JSON.stringify(M.MUSIC_CMAPS[1].coef) &&
     JSON.stringify(lits(6)) === JSON.stringify(M.MUSIC_CMAPS[2].coef) &&
     lits(1).length === 7 && lits(2).length === 7 && /if \(which == 3u\) \{ return vec3<f32>\(t, t, t\); \}/.test(cm.code));
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const o = await boot(dir, page); await turn();
    const q = JSON.parse(o.run(`function(){ return JSON.stringify({ n: cmapNames(), extra: CMAP_EXTRA.length, same: cmapWGSL() === CMAP_WGSL,
      opts: Array.from(cards.disp[0].selCmap.options).map(x => cmapNames()[+x.value]),
      code: (solver.pl.colorize && solver.pl.colorize.__code) || "" }); }`));
    ok(page + ": no extra colormap is registered -- the menu is the four it always was, and the colormap text is CMAP_WGSL itself",
       q.n.join() === "afmhot,viridis,RdBu,grayscale" && q.extra === 0 && q.same === true && q.opts.join() === q.n.join() && !/which == 4u/.test(q.code));
  }
  ok("cmapExtend refuses a duplicate name and a malformed table",
     /already exists/.test(throws(() => env.run("function(){ cmapExtend([{ name: 'magma', coef: MUSIC_CMAPS[0].coef }]); }"))) &&
     /already exists/.test(throws(() => env.run("function(){ cmapExtend([{ name: 'viridis', coef: MUSIC_CMAPS[0].coef }]); }"))) &&
     /7 rows of 3/.test(throws(() => env.run("function(){ cmapExtend([{ name: 'x', coef: [[0, 0, 0]] }]); }"))) &&
     /7 rows of 3/.test(throws(() => env.run("function(){ cmapExtend([{ name: 'y', coef: MUSIC_CMAPS[0].coef.map(r => [r[0], NaN, r[2]]) }]); }"))) &&
     env.run("function(){ return cmapNames().length; }") === 7);

  // ---- the run button says what it does ----------------------------------------------------
  ok("the run button reads Start, not Run (and Stop, not Pause: it is a full stop)", P.st().btn === "Start");

  // ---- B: Bluetooth -- the output is later than the detection, so the KICKS wait -----------
  A.outLat = 0.168; A.t = P.T;
  await P.click("btnRun");
  let q = P.st();
  const ctx = A.ctxs[0], dl = ctx.delays[0], H = 0.168 - L;
  ok("on a 168 ms output (Bluetooth) the page delays the sound by NOTHING and holds every kick " + (1e3 * H).toFixed(0) + " ms; the mode line says so",
     dl.delayTime.value === 0 && q.mode === "sound delayed 0 ms, kicks held 112 ms (output latency 168 ms, detection 56 ms): the page plays this source" &&
     q.btn === "Stop", q.mode);
  ok("  ... the debug record: outputLatency 0.168, delayMs 0, holdMs " + (1e3 * H).toFixed(1) + ", trimMs 0, ldetMs " + (1e3 * L).toFixed(1) + ", residualMs 0",
     q.dbg.outputLatency === 0.168 && q.dbg.delayMs === 0 && Math.abs(q.dbg.holdMs - 1e3 * H) < 1e-9 && q.dbg.trimMs === 0 &&
     Math.abs(q.dbg.ldetMs - 1e3 * L) < 1e-9 && Math.abs(q.dbg.outputLatencyUsedMs - 168) < 1e-9 && Math.abs(q.dbg.residualMs) < 1e-9 && q.dbg.played === true);
  ok("  ... and there is still exactly ONE path to the speakers, through the (zero) delay", ctx.destination.ins.length === 1 && toSpeakers(ctx.sources[0]).join() === "delay,destination");
  env.run("function(){ globalThis.__declared = []; const d = music.det, p = d.push.bind(d); d.push = (m, t) => { const o = p(m, t); for (const e of o) __declared.push({ band: e.band, strength: e.strength, tFire: e.tFire }); return o; }; }");
  const logsB = await P.run(6);
  q = P.st();
  const decl = JSON.parse(env.run("function(){ return JSON.stringify(__declared); }")), kk = q.kicks;
  ok("every kick records the time it was DUE (declared + the hold) and is applied on the first frame at or after it, never before",
     kk.length > 20 && kk.every(k => Math.abs(k.tDue - (k.tFire + H)) < 1e-9 && k.tApplied >= k.tDue - 1e-12 && k.tApplied - k.tDue < 1 / 60 + 1e-9),
     "applied " + (1e3 * Math.min.apply(null, kk.map(k => k.tApplied - k.tDue))).toFixed(1) + " to "
     + (1e3 * Math.max.apply(null, kk.map(k => k.tApplied - k.tDue))).toFixed(1) + " ms after due");
  ok("  ... so sound -> kick is now " + (1e3 * (kk[0].tApplied - kk[0].tFire)).toFixed(0) + " ms later than it was declared, about the 112 ms held",
     kk.every(k => k.tApplied - k.tFire >= H - 1e-9 && k.tApplied - k.tFire < H + 1 / 60 + 1e-9));
  ok("  ... one kick() per onset, in the order they were declared, none lost and none twice: applied + still queued = declared",
     kk.length + q.q === decl.length && kk.every((k, i) => k.band === decl[i].band && k.strength === decl[i].strength && k.tFire === decl[i].tFire) &&
     kicksIn(logsB) === kk.length, kk.length + " applied + " + q.q + " queued of " + decl.length + " declared");
  ok("  ... and still before that frame's steps", logsB.every(l => { let st = false; for (const e of l) { if (e.op === "step") st = true; else if (st) return false; } return true; }));
  // a rebase for falling behind does not touch the queue: nothing lost, nothing twice
  env.run("function(){ solver.__dt = 0.0001; }");
  const nReb0 = env.run("function(){ return window.musicDebug.rebases; }");
  await P.run(1.5);
  env.run("function(){ solver.__dt = 0.004; }");
  await P.run(1.5);
  q = P.st();
  const decl2 = JSON.parse(env.run("function(){ return JSON.stringify(__declared); }"));
  ok("across a rebase for falling behind (" + (env.run("function(){ return window.musicDebug.rebases; }") - nReb0) + " of them) the queue is untouched: still applied + queued = declared, in order",
     env.run("function(){ return window.musicDebug.rebases; }") > nReb0 && q.kicks.length + q.q === decl2.length &&
     q.kicks.every((k, i) => k.tFire === decl2[i].tFire && k.band === decl2[i].band));
  // the output device changes under the session: read about once a second
  await P.run(1.5);
  A.outLat = 0.02;
  const r0 = dl.delayTime.ramps.length;
  await P.run(1.2);
  q = P.st();
  ok("the output latency is re-read about once a second: switched to a 20 ms output mid-session, the sound's delay RAMPS to 36 ms and the hold drops to 0",
     Math.abs(dl.delayTime.value - (L - 0.02)) < 1e-12 && dl.delayTime.ramps.length >= r0 + 1 && dl.delayTime.ramps.length <= r0 + 2 &&
     q.dbg.holdMs === 0 && q.dbg.outputLatency === 0.02 && /^sound delayed 36 ms, kicks held 0 ms/.test(q.mode), q.mode);
  await P.run(1);
  const tail = P.st().kicks.slice(-5);
  ok("  ... and kicks declared after that are due at once (their hold is the one in force when they were declared)", tail.every(k => Math.abs(k.tDue - k.tFire) < 1e-12));
  A.outLat = NaN; await P.run(1.2);
  ok("  ... an output latency the browser does not report (NaN / undefined) is taken as 0: sound delayed by the whole detection latency",
     Math.abs(dl.delayTime.value - L) < 1e-12 && P.st().dbg.outputLatencyUsedMs === 0);
  // Stop empties the queue; a new session cannot inherit a kick
  A.outLat = 0.168; await P.run(1.3);
  await P.input("rTrim", 150);                       // hold ~262 ms: something is always waiting
  await P.run(1.2);
  q = P.st();
  ok("with a long hold something is always waiting in the queue", q.q > 0, q.q + " queued");
  await P.click("btnRun");
  q = P.st();
  ok("Stop empties the queue, clears the clock and the BEHIND flag, and the button reads Start again",
     q.q === 0 && q.run === false && JSON.stringify(q.clock) === "{}" && q.behind === false && q.btn === "Start");
  await P.input("rTrim", 0);
  A.t = P.T; await P.click("btnRun");
  const l0 = await P.run(0.45);
  ok("  ... so the next session's first half second kicks nothing (nothing inherited)", kicksIn(l0) === 0 && P.st().kicks.length === 0);
  // a hidden tab clears what was waiting
  await P.input("rTrim", 150); await P.run(2);
  const qBefore = P.st().q, kBefore = P.st().kicks.length;
  env.run("function(){ document.visibilityState = 'hidden'; document.onvisibilitychange({}); document.visibilityState = 'visible'; document.onvisibilitychange({}); }");
  P.T += 20;
  const lh = await P.run(1 / 60);
  ok("back from a hidden tab the queue is dropped (" + qBefore + " stale onsets), no kick comes out of the gap, and BEHIND stays dark",
     qBefore > 0 && kicksIn(lh) === 0 && P.st().kicks.length === kBefore && !/BEHIND/.test(P.st().strip) && P.st().q === 0);
  await P.input("rTrim", 0);

  // ---- F4: a long gap while VISIBLE is "behind" ----------------------------------------------
  await P.run(1);
  P.T += 0.6;                                         // one frame 0.6 s long, the page visible
  await P.run(2 / 60);
  ok("a 0.6 s frame on a VISIBLE page is a device that cannot keep up: the clock rebases and BEHIND lights", /BEHIND/.test(P.st().strip) && P.st().behind === true);
  // ---- F3: BEHIND does not survive the session -------------------------------------------------
  await P.click("btnRun");
  ok("Stop while BEHIND is lit puts it out at once", P.st().behind === false && !/BEHIND/.test(P.st().strip));
  A.t = P.T; await P.click("btnRun");
  await P.run(0.3);
  ok("  ... and the next session does not start with it lit (its audio clock starts again from 0)", !/BEHIND/.test(P.st().strip) && P.st().behind === false);

  // ---- F9: a non-finite state is reported, not sat on ---------------------------------------------
  env.run("function(){ solver.__t = NaN; }");
  await P.run(3 / 60);
  q = P.st();
  ok("a simulation that has gone non-finite stops the page and says so (Reset, then Start)", q.run === false && /non-finite/.test(q.status) && /Reset/.test(q.status) && q.live === false);
  env.run("function(){ solver.__t = 0; }");
  await P.click("btnReset");

  // ---- the "blobs dropped" counter accumulates ---------------------------------------------------
  A.t = P.T; await P.click("btnRun");
  env.run("function(){ const k = solver.kick; solver.kick = l => { const r = k(l); return { used: r.used, dropped: 2 }; }; }");
  await P.run(2);
  q = P.st();
  ok("blobs a kick() reports dropped are counted across kicks and shown (2 each over " + q.kicks.length + " kicks)",
     q.kicks.length > 3 && q.dropped === 2 * q.kicks.length && new RegExp(q.dropped + " blobs dropped").test(q.strip));
  await P.click("btnRun");

  // ---- survivors: a file re-chosen; any track ending; a stale session's track ---------------------
  const tone = new Float32Array(4 * 44100);
  A.fileBuffer = c => { const b = c.createBuffer(1, tone.length, 44100); b.getChannelData(0).set(tone); return b; };
  env.getEl("fMusic").files = [{ name: "a.wav", arrayBuffer: async () => new ArrayBuffer(8) }];
  await P.pick("file"); A.t = P.T; await P.click("btnRun");
  ok("a file is playing", P.st().run === true && P.st().kind === "file");
  env.getEl("fMusic").files = [{ name: "b.wav", arrayBuffer: async () => new ArrayBuffer(8) }];
  env.getEl("fMusic").onchange({}); await turn();
  ok("choosing ANOTHER file while one plays stops the session (the next Start plays the new one)", P.st().run === false && P.st().live === false);
  await P.pick("tab"); A.t = P.T; await P.click("btnRun");
  const sA = P.FM.last, vA = sA.getVideoTracks()[0], aA = sA.getAudioTracks()[0], oldAudioEnded = aA.onended;
  ok("a tab capture is live, both tracks carrying an `ended` handler", P.st().run && typeof vA.onended === "function" && typeof aA.onended === "function");
  vA.readyState = "ended"; vA.onended({}); await turn();
  ok("the VIDEO track ending ends the session too (any track, not only the audio one), and stops the audio track",
     P.st().run === false && aA.stops === 1 && /tab capture ended/.test(P.st().status));
  await P.pick("click"); A.t = P.T; await P.click("btnRun");
  oldAudioEnded({}); await turn();
  ok("a track of a session that is OVER ending later does not stop the session that replaced it", P.st().run === true && P.st().kind === "click");
  await P.click("btnRun");

  // ---- F6: a capture that is not a browser tab is analysed, not played -----------------------------
  await P.pick("tab");
  P.FM.nextDisplay = () => P.FM.stream({ audio: { suppressLocalAudioPlayback: true }, video: true, surface: "window", label: "a window" });
  const nd = ctx.delays.length;
  A.t = P.T; await P.click("btnRun");
  q = P.st();
  ok("a capture whose video track says displaySurface \"window\" is NOT played, though it reports itself silenced: no delay node, nothing at the speakers",
     q.run && q.play === false && ctx.delays.length === nd && ctx.destination.ins.length === 0 && q.dbg.surface === "window" &&
     /not a browser tab/.test(q.mode) && /pick a Chrome TAB/.test(q.status));
  ok("  ... and the request itself excluded system audio", P.FM.display[P.FM.display.length - 1].systemAudio === "exclude");
  await P.click("btnRun");
  P.FM.nextDisplay = () => P.FM.stream({ audio: { suppressLocalAudioPlayback: true }, video: true, label: "tab" });

  // ---- F5: a capture that was granted is never left running by a later failure ---------------------
  for (const [kind, where] of [["tab", "streamSource"], ["tab", "analyser"], ["input", "streamSource"], ["input", "analyser"]]) {
    await P.pick(kind);
    A.throwOn = where;
    await P.click("btnRun");
    A.throwOn = null;
    q = P.st();
    ok("a granted " + (kind === "tab" ? "tab capture" : "input device") + " followed by a throw in " + (where === "streamSource" ? "createMediaStreamSource" : "createAnalyser")
       + ": every track stopped, nothing running, not stuck in \"starting\", and a message",
       P.FM.last.getTracks().length > 0 && P.FM.last.getTracks().every(t => t.stops === 1) && q.run === false && q.live === false &&
       q.starting === false && /could not start/.test(q.status) && ctx.destination.ins.length === 0, q.status.slice(0, 70));
  }
  // ... and an input "capture" with no audio track at all
  await P.pick("input");
  P.FM.nextUser = () => P.FM.stream({ audio: false, video: true });
  await P.click("btnRun");
  ok("an input device that delivers no audio track: its stream is stopped, nothing starts, and the page says so",
     P.FM.last.getTracks().every(t => t.stops === 1) && P.st().run === false && /no audio track/.test(P.st().status));
  ok("through all of it the page logged no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 2).join(" | "));

  // ---- F1: a second Start, or a change of source, while resume() is still pending ---------------------
  for (const kind of ["click", "tab"]) {
    const Q = await mk();
    await Q.pick(kind);
    let release;
    Q.A.resumeHold = new Promise(r => { release = r; });
    const p1 = Q.env.getEl("btnRun").onclick({});
    await turn();
    const c2 = Q.A.ctxs[0];
    ok(kind + ": while resume() is pending the page is still \"starting\" (the context is created, suspended, resume called once)",
       Q.st().starting === true && Q.st().run === false && c2.resumed === 1 && c2.state === "suspended");
    Q.env.getEl("btnRun").onclick({}); await turn();
    ok("  ... a second Start click in that window acquires NOTHING more (" + (kind === "tab" ? "one getDisplayMedia" : "one buffer source") + ")",
       kind === "tab" ? Q.FM.display.length === 1 : c2.sources.length === 1);
    await Q.pick(kind === "click" ? "file" : "click");            // the visitor changes their mind
    release(); await turn(); await p1; await turn();
    const z = Q.st();
    ok("  ... and a change of source in that window is honoured: when resume() returns nothing starts, " + (kind === "tab" ? "the capture's tracks are stopped" : "the buffer source is never started")
       + ", nothing reaches the speakers", z.run === false && z.live === false && z.starting === false && c2.destination.ins.length === 0 &&
       (kind === "tab" ? Q.FM.last.getTracks().every(t => t.stops === 1) : c2.sources[0].started === null));
    // and with no change of mind, the same hold simply starts when it is released
    const R = await mk();
    await R.pick(kind);
    let rel2; R.A.resumeHold = new Promise(r => { rel2 = r; });
    const p2 = R.env.getEl("btnRun").onclick({}); await turn();
    R.env.getEl("btnRun").onclick({}); await turn();
    rel2(); await turn(); await p2; await turn();
    ok("  ... left alone, it starts exactly ONE session when resume() returns", R.st().run === true && R.st().kind === kind &&
       R.A.ctxs[0].destination.ins.length === 1 && (kind === "tab" ? R.FM.display.length === 1 : R.A.ctxs[0].sources.length === 1));
  }

  // ---- the look-ahead bound scales with the rate (a reviewer survivor) ----------------------------------
  const Z = await mk();
  Z.env.run("function(){ solver.__dt = 0.098; }");
  await Z.input("rRate", 2);
  const sil = new Float32Array(20 * 44100);
  Z.A.fileBuffer = c => { const b = c.createBuffer(1, sil.length, 44100); return b; };
  Z.env.getEl("fMusic").files = [{ name: "silence.wav", arrayBuffer: async () => new ArrayBuffer(8) }];
  await Z.pick("file"); Z.A.t = Z.T; await Z.click("btnRun");
  await Z.run(4);
  const zf = JSON.parse(Z.env.run("function(){ return JSON.stringify(window.musicDebug.frames); }"));
  const recompute = ahead => { const st = {}; let bad = 0; zf.forEach(f => { if (!f.fresh) return;
    const c = M.musicClock(st, { running: true, audioT: f.a, simT: f.simT, dt: f.dt, rate: 2, maxSteps: 64, lagMax: 0.25, maxAhead: ahead, kicked: f.kicks > 0 });
    if (c.n !== f.n) bad++; }); return bad; };
  ok("at rate 2 in silence the page's step counts are musicClock's with maxAhead = rate / 60 (2/60), and NOT with a fixed 1/60",
     recompute(2 / 60) === 0 && recompute(1 / 60) > 0 && zf.filter(f => f.n > 0).length > 20, recompute(1 / 60) + " frames differ under 1/60");
  ok("  ... and silence at rate 2 advances at rate 2 with BEHIND dark: simulation time " + Z.env.run("function(){ return solver.__t; }").toFixed(2) + " after 4 s",
     Math.abs(Z.env.run("function(){ return solver.__t; }") - 8) < 0.4 && !/BEHIND/.test(Z.st().strip) && Z.env.run("function(){ return window.musicDebug.rebases; }") === 1);
  // ... and the reviewer's deadlock itself: silence at rate 0.25 and 0.35 on the page
  for (const rate of [0.25, 0.35]) {
    const Y = await mk();
    Y.env.run("function(){ solver.__dt = 0.098; }");
    await Y.input("rRate", rate);
    Y.A.fileBuffer = c => c.createBuffer(1, sil.length, 44100);
    Y.env.getEl("fMusic").files = [{ name: "silence.wav", arrayBuffer: async () => new ArrayBuffer(8) }];
    await Y.pick("file"); Y.A.t = Y.T; await Y.click("btnRun");
    const ly = await Y.run(10);
    const steps = ly.reduce((a, l) => a + l.filter(e => e.op === "step").length, 0), simT = Y.env.run("function(){ return solver.__t; }");
    ok("silence at rate " + rate + " for 10 s: the page steps at the requested rate (" + steps + " steps of 0.098 = " + simT.toFixed(2) + " for " + (10 * rate).toFixed(2)
       + "), one rebase, BEHIND never lit", Math.abs(simT - 10 * rate) < 0.2 && steps >= Math.floor(10 * rate / 0.098) - 1 &&
       Y.env.run("function(){ return window.musicDebug.rebases; }") === 1 && !/BEHIND/.test(Y.st().strip) &&
       JSON.parse(Y.env.run("function(){ return JSON.stringify(window.musicDebug.frames.map(f => f.rebased)); }")).filter(Boolean).length === 1);
  }
}

// ===========================================================================
// 12. the fixed colour scale (2026-11-21)
// ===========================================================================
// What the display divides by is the chain's 4-byte `maxVal` buffer. Fixed: the page writes
// m * amp there before each render and the autoscale's maxFinal lands in a scratch buffer.
// Off: nothing is written and the render is the class's own, dispatch for dispatch.
async function legScale() {
  const P = await musicPage("?plain&musicdebug"), env = P.env;
  env.run(`function(){
    window.__w = [];
    const q = device.queue, wb = q.writeBuffer.bind(q);
    q.writeBuffer = (b, off, data) => { if (solver && solver.disp[0] && b === solver.disp[0].buf.maxVal) window.__w.push(Array.from(new Float32Array(data.buffer || data))[0]); return wb(b, off, data); };
  }`);
  // one render of card 0 through whatever `solver.render` is (or the class's own method),
  // returned as [pipeline, which buffer maxFinal/colorize use, dispatch counts]
  const RENDER = `function(own){
    const D = solver.chain(0), c = cards.disp[0];
    gpu0 = null; window.__w.length = 0; __env_reset();
    if (own) Solver.prototype.render.call(solver, c.ctx, 0); else solver.render(c.ctx, 0);
    return 0;
  }`;
  const render = own => {
    env.gpuReset();
    env.run(RENDER.replace("gpu0 = null; ", "").replace("__env_reset();", ""), !!own);
    const names = env.run("function(){ const D = solver.disp[0]; return [D.buf.maxVal, D.buf.musicAuto, D.buf.maxPart]; }");
    const tag = b => b === names[0] ? "maxVal" : b === names[1] ? "musicAuto" : b === names[2] ? "maxPart" : "-";
    const log = env.gpu.dispatches.map(x => x.pipe.__name + "(" + x.d.join(",") + ")"
      + (x.pipe.__name === "maxFinal" ? "->" + tag(x.bg.__buffers[1]) : "")
      + (x.pipe.__name === "colorize" ? "<-" + (x.bg.__buffers.some(b => b === names[0]) ? "maxVal" : "?") : ""));
    return { log: log, w: JSON.parse(env.run("function(){ return JSON.stringify(window.__w); }")) };
  };
  const ui = () => JSON.parse(env.run("function(){ const u = musicUi(); return JSON.stringify({ fix: u.fix, m: u.scaleM, amp: u.amp, max: musicScaleMax(u) }); }"));
  const bar = () => JSON.parse(env.run("function(){ const c = cards.disp[0]; c.barMode = solver.modeOf(0); return JSON.stringify({ max: c.barMax, ticks: c.barTicks() }); }"));
  const f32 = x => Math.fround(x);

  ok("the page has the box (ticked) and the slider, at 1.5 x amp", ui().fix === true && Math.abs(ui().m - 1.5) < 2e-3 && Math.abs(ui().max - 1.5) < 2e-3,
     JSON.stringify(ui()));
  ok("  ... the slider spans 0.25 to 4, logarithmically", (() => { const r = env.getEl("rScale"); return Math.pow(2, +r.min) === 0.25 && Math.pow(2, +r.max) === 4; })());
  ok("  ... the solver's render is wrapped on the INSTANCE; the class method is untouched",
     env.run("function(){ return Object.prototype.hasOwnProperty.call(solver, 'render') && !/musicScale/.test(String(Solver.prototype.render)); }"));

  const on = render(false);
  ok("fixed: one write of m * amp to maxVal ahead of the render", on.w.length === 1 && on.w[0] === f32(ui().max), JSON.stringify(on.w));
  ok("  ... the autoscale is still computed, into the scratch buffer, and colorize reads maxVal",
     on.log.filter(l => /^maxFinal/.test(l)).length === 1 && on.log.some(l => /^maxFinal\(1,1,1\)->musicAuto$/.test(l)) &&
     on.log.some(l => /^colorize.*<-maxVal$/.test(l)), on.log.join(" "));

  let all = true, note = [];
  for (const [sl, amp] of [[-2, 0.5], [-1, 2], [0, 0.3], [1, 1.7], [2, 3], [0.585, 1]]) {
    await P.input("rScale", sl); await P.input("rMAmp", amp);
    const u = ui(), want = Math.pow(2, sl) * amp, b0 = bar(), r = render(false);
    const good = Math.abs(u.max - want) < 1e-12 && r.w.length === 1 && r.w[0] === f32(want) && b0.max === u.max;
    if (!good) { all = false; note.push(sl + "/" + amp + ": " + JSON.stringify([u.max, r.w, b0.max])); }
  }
  ok("the range handed to the display is m * amp at six (m, amp), and the colour bar holds the same number at once", all, note.join(" | "));

  // live: moving either slider marks the card dirty, and the frame loop's own render writes it
  await P.input("rScale", 1); await P.input("rMAmp", 2);
  env.run("function(){ window.__w.length = 0; }");
  ok("  ... either slider marks the card for a redraw (no step needed)", env.run("function(){ return cards.disp[0].dirty === true; }"));
  // (read before the frames: the stub's readback of maxVal is zeros, a real one returns the 4 just written)
  const tk = bar();
  ok("  ... the colour bar's ticks are that range (0 .. m * amp for |u|)", tk.max === 4 && /4/.test(tk.ticks[2]) && /2/.test(tk.ticks[1]), JSON.stringify(tk));
  const built = env.run("function(){ window.__sv = solver; return solver.nsteps; }");
  await P.frame(); await P.frame();
  const lw = JSON.parse(env.run("function(){ return JSON.stringify(window.__w); }"));
  ok("  ... and the frame loop's next render writes the new value, with no rebuild and no step",
     lw.length >= 1 && lw[lw.length - 1] === f32(4) && env.run("function(){ return window.__sv === solver; }") &&
     env.run("function(){ return solver.nsteps; }") === built, JSON.stringify(lw));
  ok("  ... the readout and musicDebug name it, and on this background (the mean field) the scale's reference IS amp: \"fixed at 4.00 (2.00 × amp)\"",
     /^colour scale fixed at 4\.00 \(2\.00 × amp\)/.test(env.run("function(){ return musicScaleReadout(); }")) &&
     env.run("function(){ const u = musicUi(); return musicScaleRef(u) === u.amp && u.amp === 2 && el('vScale').textContent === '2.00 → 4.00'; }") === true &&
     env.run("function(){ const r = musicScaleRecord(); return r.fixed === true && r.max === 4 && r.m === 2 && r.amp === 2; }"),
     env.run("function(){ return musicScaleReadout(); }"));

  // a signed field: +-max on the bar
  env.run("function(){ solver.disp[0].mode = 0; musicScaleChanged(); }");
  const sg = bar();
  ok("  ... a signed field's bar reads -max .. +max", /4/.test(sg.ticks[0]) && sg.ticks[1] === "0" && /^\+4/.test(sg.ticks[2]), JSON.stringify(sg.ticks));
  const sgr = render(false);
  ok("  ... and is rendered against the same fixed value", sgr.w.length === 1 && sgr.w[0] === 4 && sgr.log.some(l => /->musicAuto/.test(l)));

  // a sigma mode keeps its own pinned +-1 path
  const sig = env.run("function(){ for (let m = 0; m < 40; m++) if (dispIsSigma(m)) return m; return -1; }");
  env.run("function(m){ solver.disp[0].mode = m; }", sig);
  const s1 = render(false), s2 = render(true);
  ok("a sigma mode is left alone: nothing written, maxFinal back on maxVal, the class's own log",
     sig >= 0 && s1.w.length === 0 && s1.log.join() === s2.log.join() && s1.log.some(l => /^maxFinal.*->maxVal$/.test(l)), s1.log.join(" "));
  env.run("function(){ solver.disp[0].mode = 4; }");

  // box off: today's autoscale exactly
  const fixedLog = render(false).log;
  env.getEl("cbFixScale").checked = false; env.getEl("cbFixScale").onchange({}); await turn();
  const off = render(false), cls = render(true);
  ok("box off: nothing is written to maxVal and maxFinal writes it, as before", off.w.length === 0 && off.log.some(l => /^maxFinal\(1,1,1\)->maxVal$/.test(l)) &&
     !off.log.some(l => /musicAuto/.test(l)), off.log.join(" "));
  ok("  ... the dispatch log is the class's own render, entry for entry", off.log.join() === cls.log.join() && off.log.length > 3);
  ok("  ... and the fixed render differs from it ONLY in where maxFinal writes",
     fixedLog.join().replace("->musicAuto", "->maxVal") === off.log.join());
  ok("  ... the bar drops the fixed range and waits for the autoscale readback; the slider is disabled",
     bar().max === null && env.getEl("rScale").disabled === true && /autoscale/.test(env.run("function(){ return musicScaleReadout(); }")),
     JSON.stringify([bar().max, env.getEl("rScale").disabled, env.run("function(){ return musicScaleReadout(); }")]));
  env.getEl("cbFixScale").checked = true; env.getEl("cbFixScale").onchange({}); await turn();
  ok("  ... ticking it again restores the fixed range at once", bar().max === 4 && render(false).w[0] === 4);

  // a rebuild (Restart builds a new Solver) keeps the option
  env.run("function(){ rebuild(); }"); await turn();
  env.run("function(){ solver.chain(0).mode = 4; }");
  const rb = render(false);
  ok("a rebuilt solver is wrapped again, and its scratch buffer is freed with the chain",
     rb.w.length === 1 && rb.w[0] === 4 && env.run("function(){ return 'musicAuto' in solver.disp[0].buf; }"), JSON.stringify(rb.w));
  ok("no failure logged on the page", realFails(env).length === 0, realFails(env).slice(0, 3).join(" | "));

  // the other pages: no control, no wrapper
  for (const pg of ["rmhd2d.html", "rmhd3d.html"]) {
    const E = await boot(dir, pg); await turn();
    ok(pg + ": render is the class's own method, and the page has neither control",
       E.run("function(){ return !Object.prototype.hasOwnProperty.call(solver, 'render') && typeof musicScaleApply === 'undefined'; }") === true &&
       !/cbFixScale|rScale/.test(fs.readFileSync(path.join(dir, pg), "utf8")));
  }
}

// ---------------------------------------------------------------------------
// ===========================================================================
// 13. the effects (MUSIC_EFFECTS_PLAN): the rules, and the page with its defaults
// ===========================================================================
async function legEffects(state) {
  const M = state.M, F = state.F, X = F.effects, L = M.MUSIC_LBOX;
  ok("the fixture carries the offline effects rules", !!X && /kick_types_for/.test(X.source));
  if (!X) return;
  // ---- the rules against the python ------------------------------------------------
  ok("the constants are the offline script's: the six kick types, dipole 1.5 sigma, sheet ratio 4, span 0.4, jitter 0.12, the click pans",
     JSON.stringify(M.MUSIC_KICK_TYPES) === JSON.stringify(X.kick_types) && M.MUSIC_DIPOLE_SEP === X.dipole_sep
     && M.MUSIC_SHEET_RATIO === X.sheet_ratio && M.MUSIC_STEREO_SPAN === X.stereo_span && M.MUSIC_STEREO_JITTER === X.stereo_jitter
     && JSON.stringify(M.MUSIC_CLICK_PANS) === JSON.stringify(X.click_pans));
  const tbad = X.types.filter(c => JSON.stringify(M.musicKickTypesFor(c.spec, c.nbands)) !== JSON.stringify(c.types));
  ok("musicKickTypesFor is kick_types_for: variety, one type and none at 1-7 bands, and a list", tbad.length === 0 && X.types.length === 22,
     tbad.length + " of " + X.types.length + " differ; variety at 4: " + M.musicKickTypesFor("variety", 4).join(" "));
  ok("  ... an unknown name and a wrong count are refused", throws(() => M.musicKickTypesFor("blob", 4)) !== "" && throws(() => M.musicKickTypesFor("dipole,sheet", 4)) !== "");
  ok("musicDipoleEnergy is dipole_energy_factor (1.5 at the default separation)",
     X.dipole_energy.every(([sep, v]) => Math.abs(M.musicDipoleEnergy(sep) - v) < 1e-14) && Math.abs(M.musicDipoleEnergy() - X.dipole_energy[0][1]) < 1e-14);
  let pz = 0;
  for (const c of X.pulse) {
    const st = {};
    let j = 0;
    for (let i = 0; i < c.nframes; i++) {
      const t = i / c.fps;
      while (j < c.t_on.length && c.t_on[j] <= t) { M.musicPulseHit(st, c.t_on[j], c.s_on[j], c.a, c.tau); j++; }
      pz = Math.max(pz, Math.abs(M.musicPulseZoom(st, t, c.tau) - c.zoom[i]));
    }
  }
  ok("musicPulseHit / musicPulseZoom reproduce pulse_zoom frame for frame (two settings, 150 frames each)", pz < 1e-13, "max difference " + pz.toExponential(1));
  const mg = Float64Array.from({ length: 64 }, (_, k) => Math.exp(-(((k - 30) / 6) ** 2)));
  let pd = 0;
  for (const [p, want] of X.pans) {
    const got = M.musicPan(mg.map(v => (1 - p) * v), mg.map(v => (1 + p) * v), 10, 50);
    pd = Math.max(pd, Math.abs(got - want), Math.abs(got - 2 * p / (1 + p * p)));
  }
  ok("musicPan: gains (1 - p, 1 + p) read 2p/(1 + p^2), as onset_pans measures on a tone", pd < 1e-9, "max difference " + pd.toExponential(1));
  ok("  ... silence reads 0, and only the band's bins count",
     M.musicPan(new Float64Array(64), new Float64Array(64), 0, 64) === 0
     && M.musicPan(mg, mg.map((v, k) => (k < 10 ? 5 : 1) * v), 10, 50) === 0);
  ok("musicChannelsDiffer: equal spectra no (silence included), a difference above 1e-3 of the peak yes, one below it no",
     !M.musicChannelsDiffer(mg, mg) && !M.musicChannelsDiffer(new Float64Array(64), new Float64Array(64))
     && M.musicChannelsDiffer(mg, mg.map((v, k) => (k === 30 ? 1.01 : 1) * v)) && !M.musicChannelsDiffer(mg, mg.map((v, k) => (k === 30 ? 1.0001 : 1) * v)));
  ok("musicPanX: L (0.5 + 0.4 clip(gain pan)) -- centre, hard left, a narrow mix widened, clipped",
     M.musicPanX(0) === 0.5 * L && Math.abs(M.musicPanX(-1) - 0.1 * L) < 1e-15 && Math.abs(M.musicPanX(0.25, 2) - 0.7 * L) < 1e-15
     && Math.abs(M.musicPanX(0.8, 3) - 0.9 * L) < 1e-15);

  // ---- the kick types ------------------------------------------------------------------
  const plan = M.musicBandPlan(256, null, null, 32), sg = 0.2, key = b => b.x + "," + b.y + "," + b.amp;
  const split = l => [l.filter(b => b.pol > 0), l.filter(b => b.pol < 0)];
  ok("musicKickList with all-elsasser options and no xc IS the phase-1 list, draw for draw",
     JSON.stringify(M.musicKickList(M.musicRng(3), plan, 2, 0.7, 0.5, { types: M.musicKickTypesFor(null, 4), xc: null, sheetEnergy: 2 }))
     === JSON.stringify(M.musicKickList(M.musicRng(3), plan, 2, 0.7, 0.5)));
  const [vp, vm_] = split(M.musicKickBlobs(M.musicRng(5), "vortex", sg, 7, 0.5)), [ip, im] = split(M.musicKickBlobs(M.musicRng(5), "island", sg, 7, 0.5));
  ok("vortex: the z- set IS the z+ set (phi only); island: the same set with the sign flipped (psi only)",
     vp.length === 7 && vm_.length === 7 && vp.every((b, i) => key(b) === key(vm_[i]))
     && ip.every((b, i) => b.x === im[i].x && b.y === im[i].y && b.amp === -im[i].amp) && ip.every(b => Math.abs(b.amp) === 0.5));
  const [dp, dm] = split(M.musicKickBlobs(M.musicRng(5), "dipole", sg, 7, 0.5));
  const mi = d => d - L * Math.round(d / L);
  let sepBad = 0;
  for (let i = 0; i < 7; i++) {
    const a = dp[i], b = dp[7 + i];
    if (Math.abs(Math.hypot(mi(a.x - b.x), mi(a.y - b.y)) - 1.5 * sg) > 1e-12 || a.amp !== -b.amp) sepBad++;
  }
  ok("dipole: nblob PAIRS, 1.5 sigma apart (minimum image), opposite signs, inside the box, the same in both channels",
     dp.length === 14 && dm.length === 14 && sepBad === 0 && dp.every((b, i) => key(b) === key(dm[i]))
     && dp.every(b => b.x >= 0 && b.x < L && b.y >= 0 && b.y < L), sepBad + " bad pairs");
  const sh = M.musicKickBlobs(M.musicRng(5), "sheet", sg, 9, 0.5, { sheetEnergy: 2 }), [sp, sm] = split(sh);
  ok("sheet: nblob per channel, independent sets, ratio 4, theta in [0, pi), amplitude times sqrt(sheet energy)",
     sp.length === 9 && sm.length === 9 && sp.every((b, i) => b.x !== sm[i].x && b.theta !== sm[i].theta)
     && sh.every(b => b.ratio === 4 && b.theta >= 0 && b.theta < Math.PI && Math.abs(Math.abs(b.amp) - 0.5 * Math.SQRT2) < 1e-15));
  const [mp, mm] = split(M.musicKickBlobs(M.musicRng(5), "sheet-island", sg, 9, 0.5, { ratio: 3 }));
  ok("sheet-island: one set, sign flipped in z-, the ratio asked for",
     mp.every((b, i) => b.x === mm[i].x && b.theta === mm[i].theta && b.amp === -mm[i].amp && b.ratio === 3 && mm[i].ratio === 3));
  ok("  ... an unknown type throws", throws(() => M.musicKickBlobs(M.musicRng(5), "blob", sg, 3, 1)) !== "");
  const pl = M.musicKickBlobs(M.musicRng(8), "elsasser", sg, 4000, 1, { xc: 0.3 * L });
  const xs = pl.map(b => mi(b.x - 0.3 * L)), mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length), ym = pl.reduce((a, b) => a + b.y, 0) / pl.length / L;
  ok("stereo placement (xc): x is gaussian about xc with width 0.12 L, y stays uniform, everything inside the box",
     Math.abs(mean) < 0.01 * L && Math.abs(sd / L - 0.12) < 0.005 && Math.abs(ym - 0.5) < 0.02 && pl.every(b => b.x >= 0 && b.x < L),
     "mean offset " + (mean / L).toFixed(4) + " L, width " + (sd / L).toFixed(4) + " L, mean y " + ym.toFixed(3) + " L");

  // ---- the field walk --------------------------------------------------------------------
  const rw = M.musicRng(11), w0 = M.musicWalkInit(rw, 2, true), w1 = M.musicWalkInit(rw, 2, false);
  ok("musicWalkInit: a uniform angle; r uniform in [0, rmax] if it walks, rmax if not; the targets start there",
     w0.th >= 0 && w0.th < 2 * Math.PI && w0.r >= 0 && w0.r <= 2 && w0.thTo === w0.th && w0.rTo === w0.r && w1.r === 2);
  let s2 = 0, s2h = 0, rOut = 0, rMoved = 0;
  const NW = 6000, wa = M.musicWalkInit(rw, 2, true), wb = M.musicWalkInit(rw, 2, true), wc = M.musicWalkInit(rw, 2, false);
  let rSeen = [Infinity, -Infinity];
  for (let i = 0; i < NW; i++) {
    let t = wa.thTo; M.musicWalkHit(wa, rw, 1, 30, 0.4, 2); s2 += (wa.thTo - t) ** 2;
    t = wb.thTo; M.musicWalkHit(wb, rw, 0.5, 30, 0.4, 2); s2h += (wb.thTo - t) ** 2;
    if (!(wa.rTo >= 0 && wa.rTo <= 2)) rOut++;
    rSeen = [Math.min(rSeen[0], wa.rTo), Math.max(rSeen[1], wa.rTo)];
    M.musicWalkHit(wc, rw, 1, 30, 0, 2); if (wc.rTo !== 2) rMoved++;
  }
  const rms = Math.sqrt(s2 / NW) * 180 / Math.PI, rmsH = Math.sqrt(s2h / NW) * 180 / Math.PI;
  ok("musicWalkHit: the turn is gaussian, 30 degrees rms at full strength and half that at strength 0.5",
     Math.abs(rms / 30 - 1) < 0.04 && Math.abs(rmsH / 15 - 1) < 0.04, rms.toFixed(2) + " and " + rmsH.toFixed(2) + " degrees over " + NW + " hits");
  ok("  ... the ratio walks reflected inside [0, rmax] and reaches both ends; with step 0 it never moves",
     rOut === 0 && rSeen[0] < 0.05 && rSeen[1] > 1.95 && rMoved === 0, "range " + rSeen[0].toFixed(3) + " .. " + rSeen[1].toFixed(3));
  ok("musicReflect: -0.2 -> 0.2, rmax + 0.3 -> rmax - 0.3, 2 rmax + 0.1 -> 0.1, rmax = 0 -> 0",
     Math.abs(M.musicReflect(-0.2, 2) - 0.2) < 1e-15 && Math.abs(M.musicReflect(2.3, 2) - 1.7) < 1e-15
     && Math.abs(M.musicReflect(4.1, 2) - 0.1) < 1e-14 && M.musicReflect(1, 0) === 0);
  const wr = { th: 1, r: 0.5, thTo: 2, rTo: 1.5 };
  M.musicWalkRelax(wr, 0.1, 0.1);
  const e1 = 1 - Math.exp(-1);
  M.musicWalkRelax(wr, 0, 0.1); M.musicWalkRelax(wr, -1, 0.1);
  ok("musicWalkRelax: angle and ratio close a fraction 1 - exp(-dt/tau) of the way; dt <= 0 moves nothing",
     Math.abs(wr.th - (1 + e1)) < 1e-15 && Math.abs(wr.r - (0.5 + e1)) < 1e-15);
  const bf = M.musicWalkField({ th: 0.7, r: 1.5 }, 0.32);
  ok("musicWalkField: |B0| = r sqrt(2E) at the walk's angle; zero energy gives zero",
     Math.abs(Math.hypot(bf[0], bf[1]) - 1.5 * 0.8) < 1e-15 && Math.abs(Math.atan2(bf[1], bf[0]) - 0.7) < 1e-15
     && M.musicWalkField({ th: 0.7, r: 1.5 }, 0).join() === "0,0");

  // ---- the stereo click track --------------------------------------------------------------
  const ck = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1, stereo: true }), c0 = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 });
  let cm = 0, cs = 0;
  for (let i = 0; i < ck.samples.length; i++) { cm = Math.max(cm, Math.abs(0.5 * (ck.left[i] + ck.right[i]) - ck.samples[i])); if (ck.samples[i] !== c0.samples[i]) cs++; }
  const hats = ck.events.filter(e => e.kind === "hat");
  ok("the stereo click track: its channel mean is the mono track (which is unchanged), snare left, hats alternating right and left",
     cm < 1e-7 && cs === 0 && c0.left === undefined && ck.events.filter(e => e.kind === "snare").every(e => e.pan === -0.6)
     && hats.every((e, i) => e.pan === (i % 2 === 0 ? 0.8 : -0.8)) && ck.events.filter(e => e.kind === "kick").every(e => e.pan === 0),
     "max |mean - mono| " + cm.toExponential(1));

  // ---- the page, with its DEFAULTS, on a stereo context --------------------------------------
  const P = await musicPage("?plain&musicdebug", 44100, "stereo"), env = P.env, A = P.A;
  const ui = JSON.parse(env.run("function(){ return JSON.stringify(musicUi()); }"));
  ok("the page's defaults: variety kicks, sheet energy 2, stereo on at gain 2, pulse 0.08, mean field up to 1.5 rms turning 30 degrees, strength walking",
     ui.kick === "variety" && ui.sheetE === 2 && ui.stereo === true && ui.panGain === 2 && ui.pulse === 0.08 && ui.field === 1.5
     && ui.fieldDeg === 30 && ui.fieldWalk === true, JSON.stringify([ui.kick, ui.sheetE, ui.stereo, ui.panGain, ui.pulse, ui.field, ui.fieldDeg, ui.fieldWalk]));
  ok("  ... its solver is built with the mean-field stage kernel (b0: true), field zero at boot",
     env.run("function(){ return solver.p.b0 === true && !!solver.buf.b0 && solver._b0[0] === 0 && solver._b0[1] === 0; }"));
  env.run(MODEL, 0.004);
  A.t = 100;
  await P.click("btnRun");
  const ctx = A.ctxs[0], src = ctx.sources[0];
  ok("Start on a context that can split channels: a 2-channel click buffer (left, right), and three analysers",
     src.buffer.numberOfChannels === 2 && ctx.analysers.length === 3 && src.buffer.getChannelData(0)[2000] === ck.left[2000]
     && src.buffer.getChannelData(1)[2000] === ck.right[2000]);
  ok("  ... the two channel analysers hang off a splitter and reach no speaker; the speakers still get the source through the delay only",
     toSpeakers(ctx.analysers[1]) === null && toSpeakers(ctx.analysers[2]) === null && toSpeakers(ctx.analysers[0]) === null
     && ctx.analysers[1].ins[0].kind === "splitter" && JSON.stringify(toSpeakers(src)) === JSON.stringify(["delay", "destination"]),
     JSON.stringify(toSpeakers(src)));
  const zooms = [], b0s = [], es = [], wk = [];
  for (let i = 1; i <= 8 * 60; i++) {
    await P.frame(100 + i / 60);
    zooms.push(env.run("function(){ return cards.disp[0].cv.style.transform || ''; }"));
    b0s.push(env.run("function(){ return [solver._b0[0], solver._b0[1], solver.__e]; }"));
    wk.push(env.run("function(){ const w = music.walk; return w ? [w.th, w.thTo, w.r, w.rTo, cards.disp[0].cvVec.style.transform || ''] : null; }"));
  }
  const dbg = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.kicks); }")), log = P.takeLog().filter(e => e.op === "kick");
  const want = M.musicKickTypesFor("variety", 4);
  ok("eight seconds of the click track: every kick is its band's variety type (dipole, vortex, island, sheet)",
     dbg.length > 40 && dbg.length === log.length && dbg.every(k => k.type === want[k.band]) && [0, 1, 2, 3].every(b => dbg.some(k => k.band === b)),
     dbg.length + " kicks; bands " + [0, 1, 2, 3].map(b => dbg.filter(k => k.band === b).length).join(" "));
  let shape = 0;
  log.forEach((k, i) => {
    const b = dbg[i].band, n = plan.nblobs[b], [zp, zm] = split(k.list), a = dbg[i].strength;
    if (b === 0) { if (zp.length === 2 * n && zm.length === 2 * n && zp.every((q, j) => key(q) === key(zm[j]) && Math.abs(Math.abs(q.amp) - a) < 1e-12)) shape++; }
    else if (b === 3) { if (zp.length === n && k.list.every(q => q.ratio === 4 && Math.abs(Math.abs(q.amp) - a * Math.SQRT2) < 1e-12)) shape++; }
    else if (zp.length === n && zp.every((q, j) => q.x === zm[j].x && q.amp === (b === 1 ? 1 : -1) * zm[j].amp && q.ratio === undefined)) shape++;
  });
  ok("  ... with that type's list: bass 2 x 3 dipole blobs in both channels, vortex / island sets, treble sheets at sqrt(2) the amplitude; nothing dropped",
     shape === log.length && log.every(k => k.r.dropped === 0), shape + " of " + log.length);
  const bass = dbg.filter(k => k.band === 0), hat = dbg.filter(k => k.band === 3), snr = dbg.filter(k => k.band === 2 || k.band === 1);
  const hatL = hat.filter(k => k.pan < -0.5), hatR = hat.filter(k => k.pan > 0.5);
  ok("stereo: the bass is not placed (xc null); the hats read hard right and hard left in turn and land at 0.9 L and 0.1 L",
     bass.length > 10 && bass.every(k => k.xc === null && Math.abs(k.pan) < 0.05) && hatL.length >= 5 && hatR.length >= 5
     && hatL.every(k => Math.abs(k.xc - 0.1 * L) < 1e-9) && hatR.every(k => Math.abs(k.xc - 0.9 * L) < 1e-9),
     hatR.length + " right, " + hatL.length + " left of " + hat.length + "; pans " + hat.slice(0, 4).map(k => k.pan.toFixed(2)).join(" "));
  // the blobs of a placed kick really sit about xc
  const placed = log.filter((k, i) => dbg[i].band === 3 && dbg[i].pan > 0.5).flatMap(k => k.list.map(q => mi(q.x - 0.9 * L)));
  const pm = placed.reduce((a, b) => a + b, 0) / placed.length, psd = Math.sqrt(placed.reduce((a, b) => a + (b - pm) ** 2, 0) / placed.length);
  ok("  ... and those kicks' sheets are scattered about it with width 0.12 L", Math.abs(pm) < 0.02 * L && Math.abs(psd / L - 0.12) < 0.015,
     placed.length + " sheets: mean offset " + (pm / L).toFixed(4) + " L, width " + (psd / L).toFixed(4) + " L");
  const zv = zooms.map(z => (z ? parseFloat(/scale\(([0-9.]+)\)/.exec(z)[1]) : 1));
  const zmax = Math.max.apply(null, zv);
  ok("the pulse: the card's canvas is scaled up to 1 + 0.08 x strength on a bass kick and relaxes between them, never below 1",
     zmax > 1.03 && zmax <= 1.0801 && zv.every(z => z >= 1) && zv.filter((z, i) => i > 0 && z < zv[i - 1]).length > zv.length / 2
     && env.run("function(){ return cards.disp[0].wrap.style.overflow; }") === "hidden", "max " + zmax.toFixed(4));
  const bm = b0s.map(b => Math.hypot(b[0], b[1])), ang = b0s.map(b => Math.atan2(b[1], b[0]));
  const over = b0s.filter((b, i) => bm[i] > 1.5 * Math.sqrt(2 * b[2]) * (1 + 1e-6)).length;
  const turns = ang.filter((a, i) => i > 0 && bm[i] > 0 && Math.abs(a - ang[i - 1]) > 1e-4).length;
  ok("the mean field: set every frame from the energy readback, never above 1.5 sqrt(2E), its direction turning with the bass",
     bm[bm.length - 1] > 0 && over === 0 && turns > 30 && new Set(bm.map(v => v.toFixed(3))).size > 30,
     "final |B0| " + bm[bm.length - 1].toFixed(3) + " (limit " + (1.5 * Math.sqrt(2 * b0s[b0s.length - 1][2])).toFixed(3) + "), " + turns + " frames with a turn");
  // ---- the same run, held to the rules rather than to bounds (review 2026-11-22) -----------
  // left is left: a hat the click track SCHEDULED on the right reads right
  const hs = hat.filter(k => k.sched && k.sched.kind === "hat" && Math.abs(k.sched.dt) < 0.06);
  ok("stereo orientation: every hat reads the side its click was scheduled on (+0.8 right, -0.8 left)",
     hs.length >= 10 && hs.every(k => Math.sign(k.pan) === Math.sign(k.sched.pan)) && hs.some(k => k.sched.pan > 0) && hs.some(k => k.sched.pan < 0),
     hs.length + " hats; " + hs.filter(k => k.sched.pan > 0).length + " scheduled right");
  // the pulse, frame for frame, from the bass kicks alone
  const fT = i => (i + 1) / 60, bassAt = bass.map(k => ({ t: k.tApplied, s: k.strength }));
  let zd = 0, vecBad = 0;
  { const st = {}; let j = 0;
    for (let i = 0; i < zv.length; i++) {
      while (j < bassAt.length && bassAt[j].t <= fT(i) + 1e-9) { M.musicPulseHit(st, bassAt[j].t, bassAt[j].s, 0.08, 0.25); j++; }
      zd = Math.max(zd, Math.abs(zv[i] - M.musicPulseZoom(st, fT(i), 0.25)));
      if (wk[i] && wk[i][4] !== zooms[i]) vecBad++;
    } }
  ok("the pulse is the BASS kicks' and nothing else's: every frame's scale is musicPulseZoom of those kicks at their strengths",
     zd < 6e-5 && bass.some(k => k.strength < 0.999),     // some hit below full strength: 0.08 x the shortfall would show
     "max difference " + zd.toExponential(1) + " (the scale is written to 4 decimals); weakest bass kick " + Math.min.apply(null, bass.map(k => k.strength)).toFixed(4));
  ok("  ... and the arrow overlay carries the same transform as the canvas on every frame", vecBad === 0, vecBad + " frames differ");
  // the walk: its targets move on frames with a bass kick and on no other
  const bassFrames = new Set(bass.map(k => Math.round(k.tApplied * 60) - 1));
  let moved = 0, strayTh = 0, strayR = 0, rsq = 0, nr = 0;
  for (let i = 1; i < wk.length; i++) {
    if (!wk[i] || !wk[i - 1]) continue;
    const dth = wk[i][1] !== wk[i - 1][1], dr = wk[i][3] !== wk[i - 1][3];
    if (dth) { moved++; if (!bassFrames.has(i)) strayTh++; }
    if (dr && !bassFrames.has(i)) strayR++;
    if (bassFrames.has(i)) { const k = bass.find(q => Math.round(q.tApplied * 60) - 1 === i); rsq += ((wk[i][3] - wk[i - 1][3]) / k.strength) ** 2; nr++; }
  }
  const rstep = Math.sqrt(rsq / Math.max(nr, 1));
  ok("the field's direction and strength targets step on the frames of a bass kick and on no other frame",
     moved >= 10 && strayTh === 0 && strayR === 0 && wk.every(w => !w || (w[3] >= 0 && w[3] <= 1.5)), moved + " steps, " + strayTh + " + " + strayR + " stray");
  ok("  ... the strength step is rms (1.5 / 6) x the kick's strength (reflection included), not zero and not several times that",
     nr >= 10 && rstep > 0.1 && rstep < 0.45, "rms step / strength " + rstep.toFixed(3) + " over " + nr + " kicks (rule: 0.25)");
  // |B0| is r sqrt(2 E) of the TOTAL energy the last readback reported, at the walk's angle
  let bd = 0, ad = 0, nb = 0;
  for (let i = 5; i < wk.length; i++) {
    if (!wk[i]) continue;
    const want = wk[i][2] * Math.sqrt(2 * b0s[i - 1][2]);
    bd = Math.max(bd, Math.abs(bm[i] - want) / want);
    ad = Math.max(ad, Math.abs(Math.sin(ang[i] - wk[i][0])));
    nb++;
  }
  ok("|B0| = r sqrt(2 (E_u + E_b)) from the previous frame's readback, pointing along the walk's angle, on every frame",
     nb > 400 && bd < 1e-5 && ad < 1e-5, "max relative difference " + bd.toExponential(1) + ", angle " + ad.toExponential(1) + " over " + nb + " frames");
  // Reset mid-session forgets the pulse and the walk
  await P.click("btnReset");
  ok("Reset clears the effects' state: no pulse, no walk, the canvas unzoomed",
     env.run("function(){ return music.walk === null && music.pulse.excess === undefined && music.zoom === 1 && (cards.disp[0].cv.style.transform || '') === ''; }"));
  for (let i = 1; i <= 30; i++) await P.frame(108 + i / 60);
  ok("  ... and the manifest records the effects", (() => { const m = JSON.parse(env.run("function(){ return JSON.stringify(manifestForcing()); }")).effects;
     return m && m.kicks === "variety" && m.meanField === 1.5 && m.stereo === true && m.pulse === 0.08; })());
  // field off: zero at once; pulse off: no transform
  await P.input("rField", 0); await P.input("rPulse", 0);
  await P.frame(100 + 8.5 + 1 / 60);
  ok("mean field 0 switches it off (B0 = 0 on the next frame) and pulse 0 removes the zoom",
     env.run("function(){ return solver._b0[0] === 0 && solver._b0[1] === 0 && music.walk === null && (cards.disp[0].cv.style.transform || '') === ''; }"));
  await P.input("rPulse", 0.08);
  for (let i = 2; i <= 40; i++) await P.frame(100 + 8.5 + i / 60);
  await P.click("btnRun");
  ok("Stop leaves the picture unzoomed", env.run("function(){ return (cards.disp[0].cv.style.transform || '') === '' && music.zoom === 1; }"));
  // ---- the controls, moved off their defaults ------------------------------------------------
  env.run(`function(){ el("selKick").value = "sheet"; el("rSheetE").value = "3"; el("rPanGain").value = "0.5";
                        el("rField").value = "2"; el("rFieldDeg").value = "0"; el("cbFieldWalk").checked = false; syncLabels(); }`);
  P.takeLog();
  await P.click("btnRun");
  ok("Start again begins with fresh effects state (nothing of the last session's pulse or walk)",
     env.run("function(){ return music.walk === null && music.pulse.excess === undefined && music.stereoSeen === false; }"));
  const t1 = P.A.t, w2 = [], b2 = [];
  for (let i = 1; i <= 5 * 60; i++) {
    await P.frame(t1 + i / 60);
    w2.push(env.run("function(){ const w = music.walk; return w ? [w.th, w.thTo, w.r, w.rTo] : null; }"));
    b2.push(env.run("function(){ return [solver._b0[0], solver._b0[1], solver.__e]; }"));
  }
  const d2 = JSON.parse(env.run("function(){ return JSON.stringify(window.musicDebug.kicks); }")), l2 = P.takeLog().filter(e => e.op === "kick");
  ok("kicks = sheets at sheet energy 3: every band's list is sheets at sqrt(3) x amp x strength",
     l2.length > 25 && l2.every((k, i) => d2[i].type === "sheet" && k.list.every(q => q.ratio === 4 && Math.abs(Math.abs(q.amp) - Math.sqrt(3) * d2[i].strength) < 1e-12)),
     l2.length + " kicks");
  const h2 = d2.filter(k => k.band === 3 && k.xc !== null);
  ok("pan gain 0.5: a placed hat sits at L (0.5 + 0.4 x 0.5 x its pan), not at the clipped edge",
     h2.length >= 5 && h2.every(k => Math.abs(k.xc - L * (0.5 + 0.2 * k.pan)) < 1e-9 && Math.abs(k.xc / L - 0.5) < 0.2), h2.length + " placed hats");
  const thTo2 = new Set(w2.filter(Boolean).map(w => w[1])), last2 = w2.length - 1;
  ok("turn 0 with the strength walk off: the direction never moves and |B0| settles at exactly 2 sqrt(2E)",
     thTo2.size === 1 && w2.filter(Boolean).every(w => w[3] === 2) && Math.abs(w2[last2][2] - 2) < 1e-9
     && Math.abs(Math.hypot(b2[last2][0], b2[last2][1]) / (2 * Math.sqrt(2 * b2[last2 - 1][2])) - 1) < 1e-5,
     "|B0| / sqrt(2E) = " + (Math.hypot(b2[last2][0], b2[last2][1]) / Math.sqrt(2 * b2[last2 - 1][2])).toFixed(5));
  await P.click("btnRun");
  ok("  ... and the page raised no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 3).join(" | "));

  // a context that cannot split channels (and a mono source): placement falls back to uniform
  const Q = await musicPage("?plain&musicdebug", 44100, true);
  Q.env.run(MODEL, 0.004);
  Q.A.t = 50;
  await Q.click("btnRun");
  for (let i = 1; i <= 3 * 60; i++) await Q.frame(50 + i / 60);
  const dq = JSON.parse(Q.env.run("function(){ return JSON.stringify(window.musicDebug.kicks); }"));
  // a MONO source on a context that CAN split: the up-mixed channels are equal, so nothing is
  // ever placed (the review's finding: pan 0 used to send every band to the centre column) --
  // and it is an input device, so this is also the splitter branch's "nothing reaches the
  // speakers" row
  const R = await musicPage("?plain&musicdebug", 44100, "stereo"), FR = fakeMedia(R.A, c0.samples);
  R.env.run("function(m){ navigator.mediaDevices = m; }", FR.api);
  R.env.run(MODEL, 0.004);
  R.env.getEl("selSrc").value = "input"; R.env.getEl("selSrc").onchange({}); await turn();
  R.A.t = 20;
  await R.click("btnRun");
  for (let i = 1; i <= 4 * 60; i++) await R.frame(20 + i / 60);
  const dr = JSON.parse(R.env.run("function(){ return JSON.stringify(window.musicDebug.kicks); }")), rc = R.A.ctxs[0];
  ok("a mono source on a splitting context: three analysers, equal channels, and EVERY kick unplaced (pan null, xc null)",
     rc.analysers.length === 3 && dr.length > 20 && dr.every(k => k.pan === null && k.xc === null)
     && R.env.run("function(){ return music.stereoSeen; }") === false, dr.length + " kicks");
  ok("  ... and with the splitter chain present an input device still reaches no speaker, through any node",
     rc.streamSources.length === 1 && toSpeakers(rc.streamSources[0]) === null && rc.delays.length === 0
     && rc.destination.ins.length === 0);
  // the click track is mono again when stereo placement is unticked at Start
  const U = await musicPage("?plain", 44100, "stereo");
  U.env.run("function(){ el('cbStereo').checked = false; syncLabels(); }");
  await U.click("btnRun");
  const ub = U.A.ctxs[0].sources[0].buffer;
  let us = ub.numberOfChannels === 1;
  for (let i = 0; us && i < c0.samples.length; i += 97) if (ub.getChannelData(0)[i] !== c0.samples[i]) us = false;
  ok("with stereo placement unticked the click track is the phase-1 mono buffer, sample for sample", us);
  ok("without a channel splitter the click is mono, one analyser, and every kick is unplaced (pan null, xc null) -- the other effects still run",
     Q.A.ctxs[0].analysers.length === 1 && Q.A.ctxs[0].sources[0].buffer.numberOfChannels === 1 && dq.length > 15
     && dq.every(k => k.pan === null && k.xc === null && k.type === want[k.band]), dq.length + " kicks");
}

// ===========================================================================
// 14. the fading traces (MUSIC_STAGE_PLAN S3): traces.js
// ===========================================================================
// What a stub cannot say: how any of it LOOKS (the halo's legibility, the colours against
// the picture, whether the fixed vertical range frames a real spectrum), and how long a
// real readback takes. Checked here: the pure rules, and the structure and schedule.
async function legTraces() {
  const src = fs.readFileSync(path.join(dir, "traces.js"), "utf8");
  const close = (a, b, tol) => Math.abs(a - b) <= (tol || 1e-12) * Math.max(1, Math.abs(b));
  // ---- pure, in a bare context ----------------------------------------------------------
  const ctx = vm.createContext({});
  const msg = throws(() => vm.runInContext(src, ctx, { filename: "traces.js" }));
  ok("traces.js evaluates in a bare context: no DOM, no GPU, none of the shared scripts", msg === "", msg);
  const T = vm.runInContext(`({ tracesWeight, tracesDue, tracesPush, tracesExpire, tracesBinRange, tracesSpecRange,
    tracesLogMap, tracesGuide, tracesKLabels, tracesBrightEnd, tracesColourT, tracesCss, tracesFit, tracesLayout,
    tracesSpecRuns, tracesCutMap, tracesDecades, tracesPow10, tracesLenText, tracesXLabels, tracesMidRow, tracesRowCopies,
    tracesCutRange, tracesCutLabels, tracesPlot, tracesLabelCol, TRACES_EVERY, TRACES_TAU, TRACES_FLOOR, TRACES_TOP, TRACES_DECADES,
    TRACES_SPEC_FRAC, TRACES_CUT_FRAC, TRACES_CUT_TAU_FRAC, musicTraces })`, ctx);
  ok("  ... with nothing left of the first layout: no Elsasser families, no component pairs, no dash anywhere in the file (and ONE call of the solver's readCutLine: the side panels' line cuts, off by default)",
     vm.runInContext("typeof TRACES_FAMILIES + typeof TRACES_CUT_PAIRS + typeof TRACES_FRAC", ctx) === "undefinedundefinedundefined" &&
     !/cbTrEls|selTrCut/.test(src.replace(/\/\/[^\n]*/g, "")) && (src.replace(/\/\/[^\n]*/g, "").match(/readCutLine\(/g) || []).length === 1 &&
     /sv\.readCutLine\(0\)/.test(src) && (src.match(/setLineDash\(/g) || []).length === 1 && /setLineDash\(\[\]\)/.test(src));
  ok("  ... and there it is OFF: no group, attach() refuses, frame() does nothing",
     T.musicTraces.on === false && T.musicTraces.groups().length === 0 && T.musicTraces.attach({}) === null &&
     throws(() => T.musicTraces.frame(null, 0)) === "" && T.musicTraces.frames === 0);
  const flagOf = q => { const c = vm.createContext({ location: { search: q } }); vm.runInContext(src, c); return vm.runInContext("TRACES_ON", c); };
  const tracesFlag = vm.runInContext("tracesFlag", ctx);
  const TON = ["", undefined, "?musicdebug", "?stage", "?stage=1", "?traces", "?plainly", "?x=plain", "?explain", "?plain_2", "?plain&traces", "?traces&plain", "?plain=1&traces=1", "?a=1&plain&b&traces"];
  const TOFF = ["?plain", "?a=1&plain", "?plain=1", "?plain&musicdebug", "?plain&tracesfoo", "?plain&traces-old", "?plain&x=traces", "?plain&notraces", "?plain&stage"];
  ok("tracesFlag: the traces are ON unless the URL says ?plain (a whole word: ?plain=1 is it, ?plainly / ?explain / ?x=plain are not) -- and even then with ?traces (a whole word too: ?plain&tracesfoo is off); ?stage means nothing",
     TON.every(q => tracesFlag(q) === true) && TOFF.every(q => tracesFlag(q) === false), TON.filter(q => tracesFlag(q) !== true).concat(TOFF.filter(q => tracesFlag(q) !== false)).join(" "));
  ok("  ... and TRACES_ON is that function of location.search (false where there is no location at all, as in the bare context above)",
     TON.filter(q => q !== undefined).every(q => flagOf(q) === true) && TOFF.every(q => flagOf(q) === false) && /const TRACES_ON = typeof location !== "undefined" && tracesFlag\(location\.search\);/.test(src));
  ok("the defaults are the plan's: every 6th frame, tau 2 s, dropped below 2%",
     T.TRACES_EVERY === 6 && T.TRACES_TAU === 2 && T.TRACES_FLOOR === 0.02);
  ok("fade weight = exp(-age / tau): 1 at age 0, 1/e at tau, 1/e^2 at 2 tau; a negative age is new",
     T.tracesWeight(0, 2) === 1 && close(T.tracesWeight(2, 2), Math.exp(-1)) && close(T.tracesWeight(4, 2), Math.exp(-2)) &&
     close(T.tracesWeight(2, 1), Math.exp(-2)) && T.tracesWeight(-3, 2) === 1);
  // 2% at tau = 2 s is an age of 2 ln 50 = 7.824 s
  const h = [{ t: 0, id: "a" }, { t: 1000 - 7850, id: "old" }, { t: 1000 - 7800, id: "kept" }, { t: 1000, id: "new" }];
  const nd = T.tracesExpire(h, 1000, 2);
  ok("expiry at 2%: with tau 2 s a line 7.80 s old (2.02%) stays and one 7.85 s old (1.97%) goes; order kept",
     nd === 1 && h.map(e => e.id).join() === "a,kept,new", "dropped " + nd + ", left " + h.map(e => e.id).join());
  const h2 = [{ t: 0 }, { t: 3000 }];
  ok("  ... and the fade time is an argument: the same line survives tau 2 s and not tau 0.5 s; so is the floor",
     T.tracesExpire(h2.slice(), 4000, 2) === 0 && T.tracesExpire(h2.slice(), 4000, 0.5) === 1 && T.tracesExpire(h2.slice(), 4000, 2, 0.5) === 1);
  const h3 = [];
  for (let i = 0; i < 5; i++) T.tracesPush(h3, { t: i }, 3);
  ok("push appends and keeps the newest `max`", h3.map(e => e.t).join() === "2,3,4" && T.tracesPush([], { t: 1 }).length === 1);
  const due = [];
  for (let f = 0; f <= 24; f++) if (T.tracesDue(f, 6)) due.push(f);
  ok("a sample is due on frames 6, 12, 18, 24 and on no other", due.join() === "6,12,18,24", due.join());
  const R = { k0: 1, k1: 100, e0: 1e-6, e1: 1 }, B = { x: 10, y: 20, w: 200, h: 300 };
  const m1 = T.tracesLogMap(1, 1, R, B), m2 = T.tracesLogMap(100, 1e-6, R, B), m3 = T.tracesLogMap(10, 1e-3, R, B);
  ok("log-log mapping: (k0, e1) is the box's top-left, (k1, e0) its bottom-right, the geometric middle its centre",
     close(m1[0], 10) && close(m1[1], 20) && close(m2[0], 210) && close(m2[1], 320) && close(m3[0], 110) && close(m3[1], 170),
     [m1, m2, m3].map(p => p.map(v => v.toFixed(2)).join(",")).join(" | "));
  const g = T.tracesGuide(R, B), g2 = T.tracesGuide(R, B, -1.5);
  // one decade of k is w/2 = 100 px, one of E is h/6 = 50 px: -5/3 falls 250/3 px per 100 px
  ok("the guide is -5/3 in the data and (5/3)(h / 6 decades)/(w / 2 decades) = 0.8333 px down per px across; inside the box",
     close(Math.log(g.eb / g.ea) / Math.log(g.kb / g.ka), -5 / 3) && close((g.y1 - g.y0) / (g.x1 - g.x0), 5 / 6, 1e-12) &&
     close((g2.y1 - g2.y0) / (g2.x1 - g2.x0), 0.75, 1e-12) &&
     g.x0 >= B.x && g.x1 <= B.x + B.w && g.y0 >= B.y && g.y1 <= B.y + B.h && g.x1 > g.x0,
     ((g.y1 - g.y0) / (g.x1 - g.x0)).toFixed(4));
  const sq = T.tracesBinRange(256, 256, 2 * Math.PI, 2 * Math.PI), r21 = T.tracesBinRange(512, 256, 4 * Math.PI, 2 * Math.PI),
        r12 = T.tracesBinRange(256, 512, 2 * Math.PI, 4 * Math.PI);
  ok("first complete bin: 1 in a square box (bins 1..84 at 256^2, kunit 1)",
     sq.first === 1 && sq.last === 84 && sq.nb === 85 && close(sq.kunit, 1), JSON.stringify(sq));
  ok("  ... 2 in a 2:1 box, landscape or portrait (kunit 1/2: bin 1 holds long-direction modes only), up to bin 169 at 512 x 256",
     r21.first === 2 && r21.last === 169 && close(r21.kunit, 0.5) && r12.first === 2 && r12.last === 169 && close(r12.kunit, 0.5),
     JSON.stringify(r21));
  const fb = a => T.tracesBinRange(256, 256, 2 * Math.PI * a, 2 * Math.PI).first;
  ok("  ... and round(Lmax / Lmin) in between: 1 at 1.4, 2 at 1.5, 16:9 and 2.4, 3 at 2.5; 256^2 at 16:9 keeps bins up to 84",
     fb(1.4) === 1 && fb(1.5) === 2 && fb(16 / 9) === 2 && fb(2.4) === 2 && fb(2.5) === 3 &&
     T.tracesBinRange(256, 256, 2 * Math.PI * 16 / 9, 2 * Math.PI).last === 84);
  const sr = T.tracesSpecRange(2, sq);
  ok("the vertical range is fixed against amp: top " + T.TRACES_TOP + " amp^2, " + T.TRACES_DECADES + " decades; the horizontal one is the drawn bins",
     close(sr.e1, T.TRACES_TOP * 4) && close(sr.e0, T.TRACES_TOP * 4 * Math.pow(10, -T.TRACES_DECADES)) && sr.k0 === 1 && sr.k1 === 84 &&
     close(T.tracesSpecRange(1, r21).k0, 1) && close(T.tracesSpecRange(1, r21).k1, 84.5));
  ok("  ... FOUR decades under 0.3 amp^2 (six was too flat in a band a fifth of the screen high), and the cut's fade is a quarter of the slider's",
     T.TRACES_DECADES === 4 && T.TRACES_TOP === 0.3 && close(T.tracesSpecRange(1, sq).e0, 3e-5) && T.TRACES_CUT_TAU_FRAC === 0.25);
  const dg = n => { const b = T.tracesBinRange(n, n, 1, 1), r = T.tracesSpecRange(1, b); return Math.round(Math.log10(r.e1 / r.e0)); };
  ok("  ... and the span GROWS with the grid, (5/3) log10 of the extra k range rounded up: 4 decades at 128 and 256, 5 at 512, 6 at 1024 (the top never moves)",
     dg(128) === 4 && dg(256) === 4 && dg(512) === 5 && dg(1024) === 6 &&
     close(T.tracesSpecRange(1, T.tracesBinRange(1024, 1024, 1, 1)).e1, T.TRACES_TOP), [128, 256, 512, 1024].map(dg).join());
  const kl = T.tracesKLabels(1, 84), kl2 = T.tracesKLabels(1, 4);
  ok("two or three wavenumber labels, from 1-2-5, inside the range", kl.join() === "1,10,50" && kl2.join() === "1,2", kl.join() + " | " + kl2.join());
  ok("colour by age: the bright end when new, the middle when old, 1/e of the way left at tau -- from either end",
     T.tracesColourT(0, 2, 1) === 1 && close(T.tracesColourT(1e6, 2, 1), 0.5) && close(T.tracesColourT(2, 2, 1), 0.5 + 0.5 / Math.E) &&
     T.tracesColourT(0, 2, 0) === 0 && close(T.tracesColourT(1e6, 2, 0), 0.5) && close(T.tracesColourT(2, 2, 0), 0.5 - 0.5 / Math.E));
  ok("  ... the bright end is the lighter one, and a colour is written as rgba()",
     T.tracesBrightEnd([0, 0, 0], [1, 1, 1]) === 1 && T.tracesBrightEnd([1, 1, 0.9], [0.1, 0, 0.2]) === 0 &&
     T.tracesCss([1, 0.5, 0], 0.5) === "rgba(255,128,0,0.500)" && T.tracesCss([2, -1, 0.2], 3) === "rgba(255,0,51,1.000)");
  const f1 = T.tracesFit(512, 512, 2), f2 = T.tracesFit(300.4, 200, 1.5), f3 = T.tracesFit(0, 0, 1);
  ok("the backing size is the host's client size times devicePixelRatio, never 0",
     f1.w === 1024 && f1.h === 1024 && f2.w === 451 && f2.h === 300 && f3.w === 1 && f3.h === 1);
  const L1 = T.tracesLayout(1920, 1080), L2 = T.tracesLayout(1080, 1920), rs0 = () => T.tracesCutRange("signed", 2);
  ok("placement: the spectrum is a band across the FULL width along the bottom, a fifth of the SHORT side high",
     T.TRACES_SPEC_FRAC === 0.2 && L1.spec.x === 0 && L1.spec.w === 1920 && close(L1.spec.h, 216) && close(L1.spec.y + L1.spec.h, 1080) &&
     L2.spec.x === 0 && L2.spec.w === 1080 && close(L2.spec.h, 216) && close(L2.spec.y + L2.spec.h, 1920), JSON.stringify(L1.spec));
  ok("  ... and the cut a band across the FULL width along the top, edge to edge (its x is the picture's), a fifth of the short side high",
     T.TRACES_CUT_FRAC === 0.2 && L1.cut.x === 0 && L1.cut.y === 0 && L1.cut.w === 1920 && close(L1.cut.h, 216) &&
     L2.cut.x === 0 && L2.cut.y === 0 && L2.cut.w === 1080 && close(L2.cut.h, 216) && L1.cut.y + L1.cut.h < L1.spec.y, JSON.stringify(L1.cut));
  // the plot areas inside the bands: edge to edge without tick labels; with them, clear of a
  // label column on the left (and, the spectrum, of the k row along its bottom)
  const P0 = T.tracesPlot(L1, false, 40, 14), P1 = T.tracesPlot(L1, true, 40, 14);
  ok("with the tick labels OFF both bands plot edge to edge: the plot boxes ARE the bands, whatever the column would be",
     JSON.stringify(P0.spec) === JSON.stringify(L1.spec) && JSON.stringify(P0.cut) === JSON.stringify(L1.cut) && JSON.stringify(P0.cutClip) === JSON.stringify(L1.cut));
  ok("  ... ON, the spectrum is squeezed into what is left of the label column and above the k row",
     P1.spec.x === 40 && P1.spec.w === 1880 && P1.spec.y === L1.spec.y && close(P1.spec.h, 216 - 14) && close(P1.spec.x + P1.spec.w, 1920));
  ok("  ... while the cut's MAPPING box is untouched (sample ix stays at (ix + 0.5) / nx of the host's width) and only its clip starts after the column",
     JSON.stringify(P1.cut) === JSON.stringify(L1.cut) && P1.cutClip.x === 40 && P1.cutClip.w === 1880 && P1.cutClip.y === 0 && close(P1.cutClip.h, 216) &&
     close(T.tracesCutMap(0, 256, 0, rs0(), P1.cut)[0], 1920 * 0.5 / 256) && close(T.tracesCutMap(255, 256, 0, rs0(), P1.cut)[0], 1920 * 255.5 / 256));
  ok("  ... the column is the widest label plus a pad either side, and nothing when there is no label",
     T.tracesLabelCol([10, 24.8, 12.4], 4) === 32.8 && T.tracesLabelCol([], 4) === 0 && T.tracesLabelCol([37.2], 4) === 45.2);
  // [E_u | E_b | H_c], nb = 4
  const perp = [9, 1, 0, 3, 9, 1, 0, 1, 0, 0.5, 0, -1], bn = { first: 1, last: 3, kunit: 2 };
  const rt = T.tracesSpecRuns(perp, 4, bn);
  ok("ONE spectrum line, the total energy E_u + E_b off [E_u | E_b | H_c]: k = bin * kunit, bin 0 never drawn, an empty bin breaks the line",
     JSON.stringify(rt) === "[[2,2],[6,4]]" && T.tracesSpecRuns.length === 3, JSON.stringify(rt));
  ok("  ... and it starts at the first complete bin", JSON.stringify(T.tracesSpecRuns(perp, 4, { first: 3, last: 3, kunit: 2 })) === "[[6,4]]");
  // the cut: one row of a row-major (ix * ny + iy) buffer
  const rc = T.tracesRowCopies(4, 8, T.tracesMidRow(8)), rc2 = T.tracesRowCopies(512, 256, T.tracesMidRow(256));
  ok("the cut's row: the middle row iy = ny/2, gathered at byte offsets 4 (ix ny + ny/2) -- a stride of 4 ny -- into nx floats plus one for the scale",
     T.tracesMidRow(8) === 4 && T.tracesMidRow(256) === 128 && rc.src.join() === "16,48,80,112" && rc.scaleAt === 16 && rc.size === 20 &&
     rc2.src.length === 512 && rc2.src[0] === 4 * 128 && rc2.src[1] - rc2.src[0] === 4 * 256 && rc2.src[511] === 4 * (511 * 256 + 128) &&
     rc2.scaleAt === 2048 && rc2.size === 2052, rc.src.join() + " | " + rc.size);
  const rs = T.tracesCutRange("signed", 2), rm = T.tracesCutRange("mag", 2), rg = T.tracesCutRange("sigma", 7), r0 = T.tracesCutRange("signed", 0);
  ok("the cut's range IS the colour scale: a signed field on -max..+max, a magnitude on 0..max, the sigma modes on -1..+1 whatever the scale says",
     rs.lo === -2 && rs.hi === 2 && rs.signed === true && rm.lo === 0 && rm.hi === 2 && rm.signed === false &&
     rg.lo === -1 && rg.hi === 1 && rg.signed === true && r0.hi > 0 && r0.lo === -r0.hi && isFinite(T.tracesCutMap(0, 4, 0, r0, B)[1]) &&
     T.tracesCutRange("mag", NaN).hi > 0);
  const c1 = T.tracesCutMap(0, 4, 2, rs, B), c2 = T.tracesCutMap(3, 4, -2, rs, B), c3 = T.tracesCutMap(1, 4, 0, rs, B),
        c4 = T.tracesCutMap(1, 4, 0, rm, B), c5 = T.tracesCutMap(1, 4, 2, rm, B), c6 = T.tracesCutMap(2, 4, 1, rm, B);
  // box x 10, w 200: texel centres at 35, 85, 135, 185
  ok("  ... sample ix sits over the CENTRE of the picture's texel ix; a signed field has its zero mid-band, a magnitude at the band's bottom",
     close(c1[0], 35) && close(c1[1], 20) && close(c2[0], 185) && close(c2[1], 320) && close(c3[0], 85) && close(c3[1], 170) &&
     close(c4[1], 320) && close(c5[1], 20) && close(c6[0], 135) && close(c6[1], 170),
     [c1, c2, c3, c4, c5, c6].map(q => q.map(v => v.toFixed(1)).join(",")).join(" | "));
  const lab = a => a.map(e => e.f + ":" + e.t).join(" ");
  ok("tick labels, the cut's scale: +max / 0 / -max top, middle, bottom for a signed field; max / 0 for a magnitude; +-1 for sigma",
     lab(T.tracesCutLabels(rs)) === "1:+2 0.5:0 0:−2" && lab(T.tracesCutLabels(rm)) === "1:2 0:0" && lab(T.tracesCutLabels(rg)) === "1:+1 0.5:0 0:−1" &&
     lab(T.tracesCutLabels(T.tracesCutRange("mag", 0.14567))) === "1:0.15 0:0", lab(T.tracesCutLabels(rs)) + " | " + lab(T.tracesCutLabels(rm)));
  ok("  ... its x: 0, Lx/2, Lx -- in units of pi where that is clean (2pi, 4pi, pi boxes), plain numbers otherwise (a 16:9 box)",
     lab(T.tracesXLabels(2 * Math.PI)) === "0:0 0.5:π 1:2π" && lab(T.tracesXLabels(4 * Math.PI)) === "0:0 0.5:2π 1:4π" &&
     lab(T.tracesXLabels(Math.PI)) === "0:0 0.5:π/2 1:π" && lab(T.tracesXLabels(3 * Math.PI)) === "0:0 0.5:3π/2 1:3π" &&
     lab(T.tracesXLabels(2 * Math.PI * 16 / 9)) === "0:0 0.5:5.59 1:11.2" && T.tracesLenText(10) === "10",
     lab(T.tracesXLabels(2 * Math.PI)) + " | " + lab(T.tracesXLabels(2 * Math.PI * 16 / 9)));
  ok("  ... and the spectrum's E: whole decades strictly inside the range, thinned to three at most, written 10 to a superscript",
     T.tracesDecades(3e-7, 0.3).join() === "-1,-3,-5" && T.tracesDecades(1e-6, 1).join() === "-1,-3,-5" && T.tracesDecades(3e-5, 0.3).join() === "-1,-3" && T.tracesDecades(1e-3, 1).join() === "-1,-2" &&
     T.tracesDecades(3e-7, 0.3, 6).join() === "-1,-2,-3,-4,-5,-6" && T.tracesDecades(1.2e-6, 1.2).join() === "0,-2,-4" &&
     T.tracesPow10(-3) === "10⁻³" && T.tracesPow10(0) === "10⁰" && T.tracesPow10(12) === "10¹²",
     T.tracesDecades(3e-7, 0.3).join() + " | " + T.tracesPow10(-3));

  // ---- without the flag: the shipped page -------------------------------------------------
  const Z = await musicPage("?plain&musicdebug");
  const grp = env => env.descendants(env.getEl("controls"), "details").map(d => d.id);
  for (let f = 0; f < 8; f++) { Z.env.run("function(){ stateBumped(); }"); await Z.frame(); }
  ok("with ?plain (and no ?traces) nothing exists: no canvas.traces anywhere, no grpTraces, no frame counted, no readback started",
     Z.env.allEls.every(e => e.className !== "traces") && grp(Z.env).indexOf("grpTraces") < 0 && grp(Z.env).length === 4 &&
     Z.env.run("function(){ return musicTraces.on === false && musicTraces.canvas === null && musicTraces.frames === 0 && musicTraces.reads.spec === 0 && musicTraces.reads.cut === 0; }") &&
     realFails(Z.env).length === 0, grp(Z.env).join());
  ok("  ... and the frame hook is the page's own, unwrapped",
     Z.env.run("function(){ return String(frameHook).indexOf('musicTraces') < 0 && String(frameHook).indexOf('musicScaleProbe') >= 0; }") === true);
  ok("  ... and common.js's nbins() agrees with the bin range used here, square and 2:1 (same formula, two files)",
     Z.env.run("function(){ return nbins(256, 256, 2 * Math.PI, 2 * Math.PI) === tracesBinRange(256, 256, 2 * Math.PI, 2 * Math.PI).nb && nbins(512, 256, 4 * Math.PI, 2 * Math.PI) === tracesBinRange(512, 256, 4 * Math.PI, 2 * Math.PI).nb && nbins(256, 256, 2 * Math.PI * 16 / 9, 2 * Math.PI) === tracesBinRange(256, 256, 2 * Math.PI * 16 / 9, 2 * Math.PI).nb && solver.g.kunit === tracesBinRange(256, 256, solver.p.Lx, solver.p.Ly).kunit; }") === true);

  // ---- ?traces ------------------------------------------------------------------------------
  const P = await musicPage("?plain&traces"), env = P.env;
  ok("with ?traces the page boots with no failure of its own, and the frame hook is the wrapper",
     realFails(env).length === 0 && env.run("function(){ return String(frameHook).indexOf('musicTraces.frame') >= 0; }") === true, realFails(env).slice(0, 3).join(" | "));
  const q = JSON.parse(env.run(`function(){ const c = musicTraces.canvas, w = cards.disp[0].wrap;
    return JSON.stringify({ on: musicTraces.on, child: w.children.indexOf(c), last: w.children[w.children.length - 1] === c,
      n: w.children.filter(e => e.className === "traces").length, kind: c.kind, cls: c.className, st: c.style, w: c.width, h: c.height,
      spec: el("cbTrSpec").checked, cut: el("cbTrCut").checked, ticks: el("cbTrTicks").checked, sides: el("cbTrSides").checked, sideA: el("selTrA").value, sideB: el("selTrB").value,
      sideO: [el("selTrA"), el("selTrB")].map(e => e.options.map(o => o.value).join()),
      tau: el("rTrTau").value, vTau: el("vTrTau").textContent, o: musicTraces.opts(),
      ids: (function ids(e){ return (e.id ? [e.id] : []).concat(...(e.children || []).map(ids)); })(el("grpTraces")) }); }`));
  ok("  ... ONE <canvas class=\"traces\"> is attached to the display card's wrap, after (so over) the picture",
     q.on === true && q.child >= 0 && q.last === true && q.n === 1 && q.kind === "canvas" && q.cls === "traces" &&
     env.allEls.filter(e => e.className === "traces").length === 1);
  ok("  ... absolutely positioned to fill it, taking no pointer events, and sized from it (the stub's 512 x 512 at ratio 1)",
     q.st.position === "absolute" && q.st.left === "0" && q.st.top === "0" && q.st.width === "100%" && q.st.height === "100%" &&
     q.st.pointerEvents === "none" && q.w === 512 && q.h === 512, q.w + " x " + q.h);
  ok("  ... the group grpTraces is there, with exactly these controls: spectrum OFF, cut OFF, tick labels ON, fade 2 s, and -- on a second row -- side panels ON with their first strip on \"spec\" and their second on \"time\" (each select offers spec / cuts / time; the one-select selTrSide is gone)",
     grp(env).indexOf("grpTraces") === grp(env).length - 1 && grp(env).length === 5 && q.spec === false && q.cut === false && q.ticks === true &&
     q.tau === "2" && q.vTau === "2.0" && q.o.spec === false && q.o.cut === false && q.o.ticks === true && q.o.tau === 2 &&
     q.sides === true && q.sideA === "spec" && q.sideB === "time" && q.sideO.join("|") === "spec,cuts,time|spec,cuts,time" && !env.allEls.some(e => e.id === "selTrSide") &&
     q.o.sides === true && q.o.sideA === "spec" && q.o.sideB === "time" && !("side2" in q.o) &&
     q.ids.join() === "grpTraces,cbTrSpec,cbTrCut,cbTrTicks,rTrTau,vTrTau,cbTrSides,selTrA,selTrB", q.ids.join());
  // On THIS page (?plain&traces: the card layout) nobody tells the traces where the picture
  // is, so the side panels have nowhere to go, and the spectrum box is unticked: at its
  // defaults the overlay draws nothing at all.
  const dflt = env.run(`function(){ const c = musicTraces.ctx, s0 = c.stroke; let n = 0; c.stroke = () => { n++; return s0(); };
    const nb = solver.nb, p = new Float32Array(3 * nb); for (let b = 1; b < nb; b++) p[b] = p[nb + b] = 0.005 * Math.pow(b, -5 / 3);
    musicTraces.spec = [{ t: 50000, perp: p, nb: nb, bins: tracesBinRange(256, 256, 2 * Math.PI, 2 * Math.PI) }]; musicTraces.ener = [{ t: 49000, v: [1, 1, 1, 1] }, { t: 50000, v: [2, 2, 2, 2] }];
    c.__texts.length = 0; musicTraces.draw(50000); c.stroke = s0; const out = [n, c.__texts.length, musicTraces.picture === null].join(); musicTraces.spec = []; musicTraces.ener = []; return out; }`);
  ok("  ... and at those defaults, on this card page (no picture hook: musicTraces.picture is null), the overlay strokes nothing and writes nothing even with a spectrum and energies to show -- until a box is ticked", dflt === "0,0,true", dflt);
  // (every row from here to the side panels' own section was written for the bottom spectrum
  // with the side panels off, the defaults until 2026-11-24: both boxes are put that way here,
  // through the controls, and those rows go on testing what they tested)
  env.getEl("cbTrSpec").checked = true;
  env.getEl("cbTrSides").checked = false;
  // a resize: the stub has no ResizeObserver, so the canvas looks at its host every frame
  env.run("function(){ const w = cards.disp[0].wrap; w.clientWidth = 300; w.clientHeight = 200; window.devicePixelRatio = 2; }");
  await P.frame();
  const sz = JSON.parse(env.run("function(){ return JSON.stringify([musicTraces.canvas.width, musicTraces.canvas.height]); }"));
  ok("  ... and re-fitted when the host changes: 300 x 200 at devicePixelRatio 2 is a 600 x 400 backing store", sz.join() === "600,400", sz.join("x"));
  env.run("function(){ const w = cards.disp[0].wrap; w.clientWidth = 512; w.clientHeight = 512; window.devicePixelRatio = 1; }");

  // the schedule: every push is recorded with the frame it was taken on, every spectrum
  // readback of the solver is counted (whoever asked)
  // ... and so is every copyBufferToBuffer out of the display chain's dispR / maxVal (the
  // cut's row gather: `cut` counts the encoders that took one), and every readCutLine
  env.run(`function(){ const push = tracesPush; window.__pushes = []; window.__sp = 0; window.__cut = 0; window.__cutline = 0; window.__copies = [];
    tracesPush = function(h, e, m){ window.__pushes.push({ f: musicTraces.frames, t: e.t, kind: e.kind, mode: e.mode, max: e.max, n: e.row ? e.row.length : -1,
      which: h === musicTraces.spec ? "spec" : h === musicTraces.cut ? "cut" : h === musicTraces.ener ? "ener" : h === musicTraces.rows4 ? "rows4" : "?" }); return push(h, e, m); };
    window.__wrap = sv => { const rs = sv.readSpectrum.bind(sv), rc = sv.readCutLine.bind(sv);
      sv.readSpectrum = () => { window.__sp++; return rs(); }; sv.readCutLine = iz => { window.__cutline++; return rc(iz); }; };
    window.__wrap(solver);
    const dev = solver.device, mk = dev.createCommandEncoder;
    dev.createCommandEncoder = () => { const e = mk(), cp = e.copyBufferToBuffer; let seen = false;
      e.copyBufferToBuffer = (a, ao, b, bo, n) => { const D = solver && solver.disp[0];
        if (D && a === D.buf.dispR && !seen) { seen = true; window.__cut++; }
        if (D && seen && (a === D.buf.dispR || a === D.buf.maxVal)) {
          window.__copies.push({ src: a === D.buf.dispR ? "dispR" : "maxVal", ao: ao, bo: bo, n: n, size: b.size, usage: b.usage, dst: b }); }
        return cp(a, ao, b, bo, n); };
      return e; }; }`);
  const st = () => JSON.parse(env.run("function(){ return JSON.stringify({ f: musicTraces.frames, p: window.__pushes, sp: window.__sp, cut: window.__cut, cutline: window.__cutline, r: musicTraces.reads, card: musicTraces.hasSpecCard() }); }"));
  const reset = () => env.run("function(){ musicTraces.frames = 0; window.__pushes = []; window.__sp = 0; window.__cut = 0; window.__cutline = 0; window.__copies = []; musicTraces.reads = { spec: 0, cut: 0 }; }");
  const run = async (n, bump) => { for (let f = 0; f < n; f++) { if (bump) env.run("function(){ stateBumped(); }"); await P.frame(); } };
  // (a) the spectrum chart card is on the page: its readback is the one used
  reset();
  await run(24, true);
  const a = st();
  ok("with the spectrum CARD on the page a line is pushed on frames 6, 12, 18, 24 only, from the card's own readback: traces.js starts none",
     a.card === true && a.f === 24 && a.p.map(e => e.f).join() === "6,12,18,24" && a.p.every(e => e.which === "spec") &&
     a.r.spec === 0 && a.r.cut === 0 && a.sp >= 4 && a.cut === 0, "pushes on " + a.p.map(e => e.f).join() + "; " + a.sp + " card readbacks, " + a.r.spec + " own");
  // frame 24's own card readback landed after that frame's sample: it is the state the page
  // stopped on, and it is taken on the next due frame -- once
  reset();
  await run(24, false);
  ok("  ... and once the state stands still, the card's last readback is pushed ONCE and then nothing (there is nothing new to push)",
     st().p.map(e => e.f).join() === "6" && st().f === 24, "pushes on " + st().p.map(e => e.f).join());
  // (b) no spectrum card (the stage): traces.js reads for itself
  env.run("function(){ cardsLayout({ disp: [{ sel: 4, cmap: MUSIC_CMAP_DEFAULT, arrows: false }], charts: ['energy'] }); musicTraces.attach(cards.disp[0].wrap); }");
  await turn();
  ok("attach() again (the cards were rebuilt) leaves ONE canvas, on the new host",
     env.allEls.filter(e => e.className === "traces" && e.parentNode).length === 1 &&
     env.run("function(){ return musicTraces.canvas.parentNode === cards.disp[0].wrap && musicTraces.hasSpecCard() === false; }"));
  reset();
  await run(24, true);
  const b = st();
  ok("with NO spectrum card it reads for itself: one readSpectrum on each of frames 6, 12, 18, 24 and on no other, each pushed on its frame",
     b.card === false && b.sp === 4 && b.r.spec === 4 && b.p.map(e => e.f).join() === "6,12,18,24" && b.cut === 0 && b.r.cut === 0,
     b.sp + " readbacks, pushes on " + b.p.map(e => e.f).join());
  reset();
  await run(18, false);
  ok("  ... none on a state that has not moved", st().sp === 0 && st().p.length === 0 && st().f === 18, st().sp + " readbacks in 18 still frames");
  // a readback that does not come back: the frames go on, and no second one is started
  env.run(`function(){ window.__hold = []; solver.readSpectrum = () => { window.__sp++; return new Promise(r => window.__hold.push(r)); }; }`);
  reset();
  await run(30, true);
  const c0 = st();
  ok("a slow readback never stalls the frame: 30 frames pass with it in flight, and ONE is in flight, not five",
     c0.f === 30 && c0.sp === 1 && c0.p.length === 0 && env.run("function(){ return window.__hold.length; }") === 1, c0.sp + " started, " + c0.f + " frames");
  env.run("function(){ window.__hold.shift()({ perp: new Float32Array(3 * solver.nb) }); }");
  await turn();
  const c1p = st();
  await run(6, true);
  const c1s = st();
  ok("  ... when it lands it is pushed, and the next due frame starts the next",
     c1p.p.length === 1 && c1s.p.length === 1 && c1s.sp === 2 && env.run("function(){ return window.__hold.length; }") === 1, c1s.p.length + " pushed, " + c1s.sp + " started");
  // ... and one that lands after its solver was replaced is not pushed
  env.run("function(){ window.__old = window.__hold.shift(); window.__nb = solver.nb; rebuild(); window.__wrap(solver); }");
  await turn();
  env.run("function(){ window.__old({ perp: new Float32Array(3 * window.__nb) }); }");
  await turn();
  ok("  ... one that lands after its solver was replaced is dropped", st().p.length === 1, st().p.length + " pushes");
  // (c) the cut
  reset();
  await run(12, true);
  const d0 = st();
  env.getEl("cbTrCut").checked = true;
  reset();
  await run(12, true);
  const d1 = st();
  ok("the line cut is OFF by default (nothing is copied out of the display chain) and, switched on, gathers on the same due frames",
     d0.cut === 0 && d0.p.length === 2 && d0.p.every(e => e.which === "spec") && d1.cut === 2 && d1.r.cut === 2 &&
     d1.p.filter(e => e.which === "cut").map(e => e.f).join() === "6,12", d0.cut + " then " + d1.cut + " row gathers");
  // what one gather is: the page shows |u| at 256^2
  const cps = JSON.parse(env.run(`function(){ const c = window.__copies, half = c.length / 2;
    return JSON.stringify({ n: c.length, one: c.slice(0, half).map(e => [e.src, e.ao, e.bo, e.n]), sizes: c.map(e => e.size), usage: c[0].usage,
      want: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, oneDst: c.every(e => e.dst === c[0].dst), nx: solver.p.nx, ny: solver.p.ny,
      mode: solver.modeOf(cards.disp[0].ci), same: JSON.stringify(c.slice(0, half).map(e => [e.src, e.ao, e.bo])) === JSON.stringify(c.slice(half).map(e => [e.src, e.ao, e.bo])) }); }`));
  ok("  ... one gather is nx = 256 four-byte copies out of dispR at 4 (ix ny + ny/2), the middle row, to 4 ix -- and the 4 bytes of maxVal after them",
     cps.n === 2 * 257 && cps.same && cps.one.length === 257 &&
     cps.one.slice(0, 256).every((e, ix) => e[0] === "dispR" && e[1] === 4 * (ix * 256 + 128) && e[2] === 4 * ix && e[3] === 4) &&
     cps.one[256].join() === "maxVal,0,1024,4", cps.n + " copies; the first " + cps.one[0].join() + ", the last " + cps.one[256].join());
  ok("  ... into ONE staging buffer of (nx + 1) floats, map-readable, used again by the next gather; readCutLine (the other cut) is never called",
     cps.sizes.every(z => z === 1028) && cps.usage === cps.want && cps.oneDst === true && d1.cutline === 0 && d0.cutline === 0, cps.sizes[0] + " bytes");
  ok("  ... each line carries the row, the scale and the kind of field the picture shows: |u| here, a magnitude",
     cps.mode === 4 && d1.p.filter(e => e.which === "cut").every(e => e.kind === "mag" && e.mode === 4 && e.n === 256 && e.max === 0));
  reset();
  await run(18, false);
  ok("  ... and nothing is gathered on a state that has not moved", st().cut === 0 && st().r.cut === 0 && st().f === 18, st().cut + " gathers in 18 still frames");
  // the kind follows the display's field, and a change of field starts the history again
  // (sample() driven by hand, with its own times: no draw, so nothing expires in between)
  const kinds = JSON.parse(env.run(`function(){ const out = [], go = t => { stateBumped(); musicTraces.sample(solver, t); };
    musicTraces.cut.length = 0; musicTraces.mark.cut = null; window.__go = go; return "[]"; }`));
  void kinds;
  const sampleAt = async t => { env.run("function(t){ window.__go(t); }", t); await turn(); return JSON.parse(env.run("function(){ return JSON.stringify(musicTraces.cut.map(e => [e.t, e.kind, e.mode])); }")); };
  const k1 = await sampleAt(1000), k2 = await sampleAt(1100);
  env.run("function(){ window.__modeOf = solver.modeOf; solver.modeOf = () => 0; }");
  const k3 = await sampleAt(1200);
  env.run("function(){ solver.modeOf = () => 8; }");
  const k4 = await sampleAt(1300);
  env.run("function(){ solver.modeOf = () => 9; }");
  const k5 = await sampleAt(1400);
  env.run("function(){ solver.modeOf = () => 7; }");
  const k6 = await sampleAt(1500);
  env.run("function(){ solver.modeOf = () => 11; }");
  const k7 = await sampleAt(1600);
  env.run("function(){ solver.modeOf = window.__modeOf; musicTraces.cut.length = 0; }");
  ok("  ... lines of one field accumulate; another field starts the history again, with its own range rule (vorticity signed, sigma_c / sigma_r fixed, |z-| a magnitude, omega- signed)",
     JSON.stringify(k1) === "[[1000,\"mag\",4]]" && JSON.stringify(k2) === "[[1000,\"mag\",4],[1100,\"mag\",4]]" &&
     JSON.stringify(k3) === "[[1200,\"signed\",0]]" && JSON.stringify(k4) === "[[1300,\"sigma\",8]]" && JSON.stringify(k5) === "[[1400,\"sigma\",9]]" &&
     JSON.stringify(k6) === "[[1500,\"mag\",7]]" && JSON.stringify(k7) === "[[1600,\"signed\",11]]", JSON.stringify([k2, k3, k4, k5, k6, k7]));
  // a chain that was never rendered has no dispR: nothing is read, and nothing throws
  ok("  ... and where the display chain does not exist yet nothing is gathered", env.run(`function(){ const r0 = musicTraces.reads.cut, d = solver.disp;
    solver.disp = []; stateBumped(); let threw = ""; try { musicTraces.sample(solver, 5); } catch (e) { threw = String(e); }
    solver.disp = d; return threw === "" && musicTraces.reads.cut === r0 && musicTraces.busy.cut === false && solver.disp.length >= 1; }`) === true);
  await turn();                                // (that sample's spectrum readback lands)
  env.getEl("cbTrCut").checked = false;
  env.getEl("cbTrSpec").checked = false;
  reset();
  await run(12, true);
  const d2 = st();
  ok("  ... and with both off nothing is read and both histories are emptied",
     d2.sp === 0 && d2.cut === 0 && d2.p.length === 0 && env.run("function(){ return musicTraces.spec.length === 0 && musicTraces.cut.length === 0; }"));
  env.getEl("cbTrSpec").checked = true;

  // ---- the drawing: what is stroked, in what order, in what colour ---------------------
  // (a k^-5/3 spectrum, H_c = E_tot / 2)
  const cx = env.run("function(){ return musicTraces.ctx; }");
  const log = [];
  const stroke0 = cx.stroke, dash0 = cx.setLineDash;
  let dash = [];
  const cur = [], move0 = cx.moveTo, line0 = cx.lineTo;
  cx.moveTo = (x, y) => { cur.length = 0; cur.push([x, y]); return move0(x, y); };
  cx.lineTo = (x, y) => { cur.push([x, y]); return line0(x, y); };
  cx.setLineDash = d => { dash = Array.from(d); return dash0(d); };
  // ... and the clip in force: the last rect() that was clip()ped, until the restore()
  let rectIn = null, clipIn = null;
  const rect0 = cx.rect, clip0 = cx.clip, rest0 = cx.restore;
  cx.rect = (x, y, w, h) => { rectIn = [x, y, w, h]; return rect0(x, y, w, h); };
  cx.clip = () => { clipIn = rectIn; return clip0(); };
  cx.restore = () => { clipIn = null; return rest0(); };
  cx.stroke = () => { log.push({ s: cx.strokeStyle, w: cx.lineWidth, dash: dash.length, shadow: cx.shadowBlur || 0, pts: cur.map(q => q.slice()), clip: clipIn && clipIn.slice() }); return stroke0(); };
  const inject = () => env.run(`function(){ const nb = solver.nb, p = new Float32Array(3 * nb);
    for (let b = 1; b < nb; b++) { const e = 0.01 * Math.pow(b, -5 / 3); p[b] = 0.5 * e; p[nb + b] = 0.5 * e; p[2 * nb + b] = 0.5 * e; }
    const bins = tracesBinRange(256, 256, 2 * Math.PI, 2 * Math.PI);
    musicTraces.spec = [{ t: 100000 - 2000, perp: p, nb: nb, bins: bins }, { t: 100000, perp: p, nb: nb, bins: bins }];
    musicTraces.cut = []; }`);
  inject();
  const css = (rgb, al) => "rgba(" + rgb.map(v => Math.round(255 * Math.max(0, Math.min(1, v)))).join(",") + "," + al.toFixed(3) + ")";
  const cm = t => JSON.parse(env.run("function(t){ return JSON.stringify(cmapRGB(cards.disp[0].cmap(), t)); }", t));
  const draw = now => { log.length = 0; cx.__texts.length = 0; env.run("function(t){ musicTraces.draw(t); }", now); return log.slice(); };
  const s1 = draw(100000);
  ok("two lines and the guide are six strokes: each a dark halo first (wider, low opacity), then the line itself, oldest line first",
     s1.length === 6 && [0, 2, 4].every(i => /^rgba\(0,0,0,/.test(s1[i].s) && s1[i].w > 2 * s1[i + 1].w && !/^rgba\(0,0,0,/.test(s1[i + 1].s)) &&
     s1[2].s === "rgba(0,0,0," + (0.35 * Math.exp(-1)).toFixed(3) + ")" && s1[4].s === "rgba(0,0,0,0.350)" && s1.every(e => e.dash === 0),
     s1.map(e => e.s + "@" + e.w).join("  "));
  ok("  ... the newest in the colormap's BRIGHT end at full opacity, the one a fade time old 1/e of the way from the middle at opacity 1/e",
     s1.length === 6 && s1[5].s === css(cm(1), 1) && s1[3].s === css(cm(0.5 + 0.5 / Math.E), Math.exp(-1)) && s1[5].s !== s1[3].s, s1[3] && s1[5] && s1[3].s + " then " + s1[5].s);
  // the band: full width, the bottom fifth of the 512 x 512 host; E runs 3e-5 .. 0.3 (amp 1).
  // With the tick labels on (the default) the plot keeps clear of the label column -- the
  // widest label, 10^-1 at the stub's 6.2 px a character, plus 4 px either side = 32.8 --
  // and of the 14 px k row along the bottom.
  const SBAND = { x: 0, y: 512 - 102.4, w: 512, h: 102.4 }, SB = { x: 32.8, y: 512 - 102.4, w: 512 - 32.8, h: 102.4 - 14 }, SR = { k0: 1, k1: 84, e0: 3e-5, e1: 0.3 };
  const newest = s1[5].pts, pFirst = T.tracesLogMap(1, 0.01, SR, SB), pLast = T.tracesLogMap(84, 0.01 * Math.pow(84, -5 / 3), SR, SB);
  const sameBox = (a, b) => !!a && close(a[0], b.x, 1e-9) && close(a[1], b.y, 1e-9) && close(a[2], b.w, 1e-9) && close(a[3], b.h, 1e-9);
  ok("  ... the spectrum runs along the bottom from the label column to the right edge: bin 1 at the column, bin 84 at the edge, 84 points, the guide at -5/3",
     newest.length === 84 && close(newest[0][0], 32.8, 1e-9) && close(newest[83][0], 512, 1e-9) && close(newest[0][1], pFirst[1], 1e-6) && close(newest[83][1], pLast[1], 1e-5) &&
     close((s1[1].pts[1][1] - s1[1].pts[0][1]) / (s1[1].pts[1][0] - s1[1].pts[0][0]), (5 / 3) * (88.4 / 4) / (479.2 / Math.log10(84)), 1e-9),
     newest.length + " points, x " + newest[0][0].toFixed(1) + " .. " + newest[83][0].toFixed(1) + ", y " + newest[0][1].toFixed(1) + " .. " + newest[83][1].toFixed(1));
  ok("  ... over FOUR decades, so this spectrum's tail falls below the plot -- and is CLIPPED at its edge: every stroke is inside a clip that is exactly the plot box, above the k row",
     newest[83][1] > SB.y + SB.h && newest[0][1] < SB.y + SB.h && newest[0][1] > SB.y && s1.every(e => sameBox(e.clip, SB)) && SB.y + SB.h < 512 - 13,
     "last point y " + newest[83][1].toFixed(1) + ", plot bottom " + (SB.y + SB.h).toFixed(1) + ", clip " + JSON.stringify(s1[5].clip));
  const txt = () => cx.__texts.map(t => t.t + "@" + t.align);
  const yDec = d => T.tracesLogMap(1, Math.pow(10, d), SR, SB)[1];
  ok("  ... tick labels, as text only: k in the row under the plot (the ends aligned inward), the decades of E in the column to its left, at their own heights",
     txt().join(" ") === "k = 1@left 10@center 50@right 10⁻¹@left 10⁻³@left" &&
     cx.__texts.slice(0, 3).every(t => close(t.y, 512 - 4)) && close(cx.__texts[0].x, 32.8 + 4, 1e-9) && close(cx.__texts[1].x, T.tracesLogMap(10, 1, SR, SB)[0], 1e-9) &&
     close(cx.__texts[2].x, T.tracesLogMap(50, 1, SR, SB)[0] - 4, 1e-9) &&
     [-1, -3].every((d, i) => close(cx.__texts[3 + i].y, yDec(d), 1e-9) && cx.__texts[3 + i].x === 4 && cx.__texts[3 + i].base === "middle" &&
                              cx.__texts[3 + i].x + 0.62 * 10 * 4 <= 32.8 - 4 + 1e-9),
     txt().join(" "));
  ok("  ... each over a dark shadow of its own, which is off again before anything is stroked; and nothing is ever dashed",
     cx.shadowBlur === 0 && s1.every(e => e.shadow === 0 && e.dash === 0) && /rgba\(0,0,0,0\.9/.test(String(env.run("function(){ let seen = ''; const c = musicTraces.ctx, f = c.fillText; c.fillText = (...a) => { seen = c.shadowColor + '|' + c.shadowBlur; return f(...a); }; musicTraces.draw(100000); c.fillText = f; return seen; }"))));
  env.getEl("cbTrTicks").checked = false;
  const s1b = draw(100000), nb1 = s1b.length === 6 ? s1b[5].pts : [[NaN, NaN]];
  ok("  ... and all of them go with the one toggle (cbTrTicks), which takes nothing else away",
     cx.__texts.length === 0 && s1b.length === 6 && JSON.stringify(s1b.map(e => e.s)) === JSON.stringify(s1.map(e => e.s)));
  ok("  ... without them there is no margin: the spectrum runs edge to edge over the whole band, bin 1 at x = 0, clipped to the band",
     nb1.length === 84 && close(nb1[0][0], 0) && close(nb1[83][0], 512, 1e-9) && close(nb1[0][1], T.tracesLogMap(1, 0.01, SR, SBAND)[1], 1e-6) &&
     s1b.every(e => sameBox(e.clip, SBAND)), "x " + nb1[0][0].toFixed(1) + " .. " + nb1[nb1.length - 1][0].toFixed(1) + ", clip " + JSON.stringify(s1b[5] && s1b[5].clip));
  env.getEl("cbTrTicks").checked = true;
  const s2 = draw(100000 + 5900);
  ok("5.9 s later the older line (7.9 s, under 2%) is gone from the history and the picture; the other is fainter and nearer the middle",
     s2.length === 4 && env.run("function(){ return musicTraces.spec.length; }") === 1 &&
     s2[3].s === css(cm(0.5 + 0.5 * Math.exp(-2.95)), Math.exp(-2.95)), s2.length + " strokes, " + (s2[3] && s2[3].s));
  env.run("function(){ cards.disp[0].selCmap.value = '1'; }");
  const s3 = draw(100000 + 5900);
  ok("the colour follows the display's colormap when it changes (viridis here)", s3.length === 4 && s2.length === 4 && s3[3].s === css(cm(0.5 + 0.5 * Math.exp(-2.95)), Math.exp(-2.95)) && s3[3].s !== s2[3].s, s3[3] && s3[3].s);
  inject();
  await P.input("rTrTau", 0.5);
  draw(100000);
  const s5 = draw(100000 + 1900);
  ok("the fade slider is live: at 0.5 s a line 1.9 s old (2.2%) is still drawn, 2 s old is not; its label follows",
     s5.length === 4 && draw(100000 + 2000).length === 2 && env.getEl("vTrTau").textContent === "0.5", s5.length + " strokes");
  await P.input("rTrTau", 2);
  // the cut, drawn: ONE line per sample over its halo, in the band along the top, each on
  // the scale its own picture was coloured with
  env.getEl("cbTrCut").checked = true;
  const cutIn = (list) => env.run(`function(L){ musicTraces.spec = []; musicTraces.cut = L.map(e => { const r = new Float32Array(256); r.fill(e.fill); (e.head || []).forEach((v, i) => { r[i] = v; });
    return { t: e.t, row: r, max: e.max, kind: e.kind, mode: e.mode }; }); }`, list);
  cutIn([{ t: 200000, fill: 0, head: [2, 0, -2], max: 2, kind: "signed", mode: 0 }]);
  const s6 = draw(200000), cp6 = s6.length === 4 ? s6[3].pts : [];
  ok("the cut is ONE line over its halo (beside the spectrum's guide), across the FULL width along the top: sample ix over the centre of texel ix",
     s6.length === 4 && /^rgba\(0,0,0,/.test(s6[2].s) && s6[2].w > 2 * s6[3].w && s6[3].s === css(cm(1), 1) && s6.every(e => e.dash === 0) &&
     cp6.length === 256 && close(cp6[0][0], 1) && close(cp6[1][0], 3) && close(cp6[255][0], 511), cp6.length + " points, x " + (cp6[0] && cp6[0][0]) + " .. " + (cp6[255] && cp6[255][0]));
  const CB = { x: 0, y: 0, w: 512, h: 102.4 }, CCLIP = { x: 32.8, y: 0, w: 512 - 32.8, h: 102.4 };
  env.getEl("cbTrTicks").checked = false;
  const s6b = draw(200000), cp6b = s6b.length === 4 ? s6b[3].pts : [];
  env.getEl("cbTrTicks").checked = true;
  draw(200000);
  ok("  ... with the tick labels on it is CLIPPED out of the label column, not squeezed: the same 256 points at the same x and y as with them off, where the clip is the whole band",
     s6.length === 4 && sameBox(s6[3].clip, CCLIP) && sameBox(s6[2].clip, CCLIP) && sameBox(s6b[3].clip, CB) && cp6b.length === 256 &&
     JSON.stringify(cp6) === JSON.stringify(cp6b) && cp6.every((q, ix) => close(q[0], 512 * (ix + 0.5) / 256, 1e-12)),
     "clip " + JSON.stringify(s6[3] && s6[3].clip) + " against " + JSON.stringify(s6b[3] && s6b[3].clip));
  ok("  ... a signed field on -max..+max of the picture's scale: +max at the band's top, 0 mid-band, -max at its bottom (a fifth of the host down)",
     cp6.length === 256 && close(cp6[0][1], 0) && close(cp6[1][1], 51.2) && close(cp6[2][1], 102.4) && close(cp6[100][1], 51.2), cp6.slice(0, 3).map(q => q[1].toFixed(1)).join(" "));
  ok("  ... labelled with that scale up the left edge and x along the bottom of the band (2pi box)",
     txt().slice(-5).join(" ") === "+2@left 0@left −2@left x = π@center 2π@right" && close(cx.__texts[cx.__texts.length - 5].y, 0) && close(cx.__texts[cx.__texts.length - 4].y, 51.2) &&
     close(cx.__texts[cx.__texts.length - 3].y, 102.4) && close(cx.__texts[cx.__texts.length - 2].x, 256) && close(cx.__texts[cx.__texts.length - 1].x, 512 - 4) &&
     close(cx.__texts[cx.__texts.length - 1].y, 102.4 - 4), txt().slice(-5).join(" "));
  // ... where the page says where the picture is (musicTraces.picture: the stage, whose
  // picture need not span its host), the cut is mapped over THAT extent and not the host's.
  // (the stub records every stroke's points; this host is 512 CSS px wide at one backing
  // pixel each, so the numbers below are both)
  ok("musicTraces.picture is null unless a page sets it (this one, ?traces alone, does not): the cut spans its host", env.run("function(){ return musicTraces.picture; }") === null);
  const setPic = v => env.run("function(v){ musicTraces.picture = v === null ? null : () => v; }", v);
  setPic({ x: 100, w: 256, tile: false });
  const sp1 = draw(200000), pp1 = sp1.length === 4 ? sp1[3].pts : [];
  ok("with picture() = { x: 100, w: 256 } the cut is mapped over those 256 px: sample ix at 100 + (ix + 0.5), the same y as before, still one line over its halo and still clipped out of the label column only",
     sp1.length === 4 && pp1.length === 256 && pp1.every((q, ix) => close(q[0], 100 + 256 * (ix + 0.5) / 256, 1e-12) && q[1] === cp6[ix][1]) &&
     sameBox(sp1[3].clip, CCLIP) && sameBox(sp1[2].clip, CCLIP), pp1.length + " points, x " + (pp1[0] && pp1[0][0]) + " .. " + (pp1[255] && pp1[255][0]));
  ok("  ... its x labels follow the picture (pi at its middle, 228; 2 pi at its right edge, 356 less the pad) while the scale's labels stay at the HOST's left edge (x = 4)",
     txt().slice(-5).join(" ") === "+2@left 0@left −2@left x = π@center 2π@right" && cx.__texts.slice(-5, -2).every(t => t.x === 4) &&
     close(cx.__texts[cx.__texts.length - 2].x, 228) && close(cx.__texts[cx.__texts.length - 1].x, 356 - 4) && close(cx.__texts[cx.__texts.length - 5].y, 0) &&
     close(cx.__texts[cx.__texts.length - 3].y, 102.4), cx.__texts.slice(-5).map(t => t.t + "@" + t.x).join(" "));
  setPic({ x: 100, w: 256, tile: true });
  const sp2 = draw(200000), xs2 = sp2.slice(2).filter((e, i) => i % 2 === 1).map(e => e.pts[0][0]).sort((a, b) => a - b);
  ok("  ... and with tile: true the line is repeated every picture-width across the host, as the picture is: three copies here, starting at x = -156, 100 and 356 (the first from left of the host, the last running past its right edge at 512), each its own halo and line with identical y",
     sp2.length === 8 && xs2.length === 3 && close(xs2[0], -156 + 0.5) && close(xs2[1], 100.5) && close(xs2[2], 356.5) &&
     sp2.slice(2).every(e => e.pts.length === 256 && e.pts.every((q, ix) => q[1] === cp6[ix][1] && close(q[0] - e.pts[0][0], ix, 1e-9)) && sameBox(e.clip, CCLIP)) &&
     [2, 4, 6].every(i => /^rgba\(0,0,0,/.test(sp2[i].s) && sp2[i + 1].s === css(cm(1), 1)), sp2.length + " strokes, first x " + xs2.join(" "));
  setPic({ x: 0, w: 512, tile: true });
  const sp3 = draw(200000);
  ok("  ... a picture that spans the host repeats nothing (one line, at the full-width x), and a picture() that answers null or a zero width falls back to the host's whole width",
     sp3.length === 4 && JSON.stringify(sp3[3].pts) === JSON.stringify(cp6) &&
     (setPic(null), env.run("function(){ musicTraces.picture = () => null; }"), JSON.stringify((draw(200000)[3] || {}).pts) === JSON.stringify(cp6)) &&
     (setPic({ x: 50, w: 0, tile: true }), (d => d.length === 4 && JSON.stringify(d[3].pts) === JSON.stringify(cp6))(draw(200000))));
  setPic(null);
  cutIn([{ t: 200000, fill: 0, head: [2, 1], max: 2, kind: "mag", mode: 4 }]);
  const s7 = draw(200000), cp7 = s7.length === 4 ? s7[3].pts : [];
  ok("  ... a magnitude on 0..max: 0 at the band's bottom, max at its top; labelled max and 0",
     cp7.length === 256 && close(cp7[0][1], 0) && close(cp7[1][1], 51.2) && close(cp7[2][1], 102.4) && txt().slice(-4).join(" ") === "2@left 0@left x = π@center 2π@right", txt().slice(-4).join(" "));
  cutIn([{ t: 200000, fill: 0, head: [1, -1, 0.5], max: 123, kind: "sigma", mode: 8 }]);
  const cp8 = draw(200000)[3].pts;
  ok("  ... sigma on its fixed -1..+1, whatever the scale buffer holds", close(cp8[0][1], 0) && close(cp8[1][1], 102.4) && close(cp8[2][1], 25.6) && txt().slice(-5, -2).join(" ") === "+1@left 0@left −1@left");
  // the column is ONE column for both bands, as wide as the longest label in either: with
  // the spectrum off and a small scale on the cut it is the cut's "0.0012" (6 characters)
  env.getEl("cbTrSpec").checked = false;
  cutIn([{ t: 200000, fill: 0, max: 0.0012345, kind: "mag", mode: 4 }]);
  const s8 = draw(200000);
  ok("  ... the label column follows the longest label shown (here the cut's own 0.0012: 6 characters and two pads, 45.2 px)",
     s8.length === 2 && sameBox(s8[1].clip, { x: 45.2, y: 0, w: 512 - 45.2, h: 102.4 }) && txt()[0] === "0.0012@left", JSON.stringify(s8[1] && s8[1].clip) + " " + txt()[0]);
  env.getEl("cbTrSpec").checked = true;
  // two lines taken under different scales (the autoscale moved): each drawn on its own.
  // The cut's fade is a QUARTER of the slider's 2 s: a line 0.5 s old is one fade time old.
  cutIn([{ t: 200000 - 500, fill: 1, max: 1, kind: "signed", mode: 0 }, { t: 200000, fill: 1, max: 4, kind: "signed", mode: 0 }]);
  const s9 = draw(200000);
  ok("  ... each line on the scale ITS picture was coloured with (the same value is full scale under max 1 and 5/8 of the way up under max 4); the labels are the newest's",
     s9.length === 6 && s9[3].pts.every(q => close(q[1], 0)) && s9[5].pts.every(q => close(q[1], 102.4 * (1 - 5 / 8))) && txt().slice(-5, -4).join() === "+4@left", s9.length + " strokes");
  ok("the cut fades FOUR TIMES faster than the spectrum: at the slider's 2 s a cut line 0.5 s old is at opacity 1/e, 1/e of the way from the middle (halo 0.35/e)",
     s9.length === 6 && s9[3].s === css(cm(0.5 + 0.5 / Math.E), Math.exp(-1)) && s9[2].s === "rgba(0,0,0," + (0.35 * Math.exp(-1)).toFixed(3) + ")" && s9[5].s === css(cm(1), 1),
     s9[3] && s9[3].s);
  // ... and the two histories expire on their own clocks: 2% is 7.82 s for the spectrum and 1.96 s for the cut
  const both = age => { inject(); env.run("function(a){ const r = new Float32Array(256); r.fill(1); musicTraces.cut = [{ t: 100000 - a, row: r, max: 2, kind: 'signed', mode: 0 }]; }", age);
    const n = draw(100000).length; return [n, env.run("function(){ return musicTraces.spec.length; }"), env.run("function(){ return musicTraces.cut.length; }")]; };
  const b19 = both(1900), b20 = both(2000);
  ok("  ... a cut line 1.9 s old (2.2%) is still drawn and one 2.0 s old (1.8%) is dropped, while the spectrum's line 2 s old (37%) stays; the slider scales both",
     b19.join() === "8,2,1" && b20.join() === "6,2,0" && (() => { env.getEl("rTrTau").value = "4"; const b = both(3900).join(), c = both(4000).join(); env.getEl("rTrTau").value = "2"; return b === "8,2,1" && c === "6,2,0"; })(),
     b19.join() + " | " + b20.join());
  env.run("function(){ musicTraces.spec = []; }");
  cutIn([{ t: 200000 - 500, fill: 1, max: 1, kind: "signed", mode: 0 }, { t: 200000, fill: 1, max: 4, kind: "signed", mode: 0 }]);
  env.getEl("cbTrTicks").checked = false;
  ok("  ... and the cut's labels go with the same toggle", draw(200000).length === 6 && cx.__texts.length === 0);
  env.getEl("cbTrTicks").checked = true;
  env.getEl("cbTrCut").checked = false;
  env.getEl("cbTrSpec").checked = false;
  ok("with both overlays off nothing is stroked or written", draw(200000).length === 0 && cx.__texts.length === 0);
  ok("nothing non-finite reached the canvas, and the page logged no failure of its own through all of it", realFails(env).length === 0, realFails(env).slice(0, 3).join(" | "));
  { // ---- the side panels (cbTrSides: on by default, switched off above for the rows before this) ----
  // Where the page says where the picture is (musicTraces.picture, the stage's) and it leaves
  // two strips of the host free, eight panels take the place of the bottom spectrum: four
  // spectra (E_u, E_b, E+, E-) in the first strip and, in the second, the same four
  // energies against time or four SIGNED line cuts (u_x, b_x, z+_x, z-_x along the solver's
  // own cut line, x = Lx/2). What the stub cannot say: what any of it looks like, or that the numbers a
  // GPU hands back are these (every stub readback is zeros; the rows that need values put
  // them in by hand, and say so).
  const TS = vm.runInContext("({ tracesLane, tracesLaneRuns, tracesSidePanels, tracesLineFields, tracesSideKind, tracesGuide, TRACES_LANES, TRACES_CUT_NAMES, TRACES_SIDE_MIN, TRACES_SIDE_KINDS })", ctx);
  ok("tracesLane: 0 is E_u, 1 is E_b, 2 is E+ = E_u + E_b + H_c, 3 is E- = E_u + E_b - H_c -- (3, 2, 1) -> 3, 2, 6, 4 -- which is common.js's SPEC_SETS convention, named E_u, E_b, E+, E−",
     [0, 1, 2, 3].map(i => TS.tracesLane(i, 3, 2, 1)).join() === "3,2,6,4" && [0, 1, 2, 3].map(i => TS.tracesLane(i, 3, 2, -1)).join() === "3,2,4,6" &&
     Array.from(TS.TRACES_LANES).join() === "E_u,E_b,E+,E\u2212" &&
     /pm: \[\["E\+", COL\.zp, \(u, b, h\) => u \+ b \+ h\], \["E-", COL\.zm, \(u, b, h\) => u \+ b - h\]\]/.test(fs.readFileSync(path.join(dir, "common.js"), "utf8")));
  // eight bins: E_u = b, E_b = 2b, H_c = b, but H_c = 10 at bin 3 (E- < 0 there) and E_u = NaN at bin 5
  const lp = new Float32Array(24);
  for (let b = 0; b < 8; b++) { lp[b] = b; lp[8 + b] = 2 * b; lp[16 + b] = b; }
  lp[16 + 3] = 10; lp[5] = NaN;
  const lb = { first: 1, last: 6, kunit: 0.5 }, lr = i => JSON.stringify(Array.from(TS.tracesLaneRuns(lp, 8, lb, i)).map(r => Array.from(r)));
  ok("tracesLaneRuns: one lane's (k, E) over bins first..last, BROKEN where the lane is not positive and finite -- E- breaks alone at the bin where H_c exceeds E_u + E_b, and a NaN in E_u breaks every lane that reads it (E_b, which does not, runs through); it stops at the last bin the readback has",
     lr(0) === "[[0.5,1,1,2,1.5,3,2,4],[3,6]]" && lr(1) === "[[0.5,2,1,4,1.5,6,2,8,2.5,10,3,12]]" && lr(2) === "[[0.5,4,1,8,1.5,19,2,16],[3,24]]" &&
     lr(3) === "[[0.5,2,1,4],[2,8],[3,12]]" && JSON.stringify(Array.from(TS.tracesLaneRuns(lp, 4, lb, 0)).map(r => Array.from(r))) === "[[0.5,1,1,2,1.5,3]]", lr(3));
  // the panels: by this file's own arithmetic, a strip cut into four and each box inset by the gap
  const boxEq = (b, x, y, w, h) => close(b.x, x, 1e-9) && close(b.y, y, 1e-9) && close(b.w, w, 1e-9) && close(b.h, h, 1e-9);
  const lrWant = (W, H, x, w, g, i, second) => [(second ? x + w : 0) + g, H * i / 4 + g, (second ? W - x - w : x) - 2 * g, H / 4 - 2 * g];
  const tbWant = (W, H, y, h, g, i, second) => [W * i / 4 + g, (second ? y + h : 0) + g, W / 4 - 2 * g, (second ? H - y - h : y) - 2 * g];
  const sp1 = TS.tracesSidePanels(1280, 800, 240, 0, 800, 800, 90, 10), sp2p = TS.tracesSidePanels(390, 844, 0, 227, 390, 390, 90, 10);
  ok("tracesSidePanels, the square picture in a 1280 x 800 window: the strips are LEFT and RIGHT of it, each cut into four ROWS -- a: x 10..230, b: x 1050..1270, rows 200 high less the 10 px gap all round (y 10..190, 210..390, ...)",
     !!sp1 && sp1.a.length === 4 && sp1.b.length === 4 && sp1.a.every((b, i) => boxEq(b, 10, 200 * i + 10, 220, 180)) && sp1.b.every((b, i) => boxEq(b, 1050, 200 * i + 10, 220, 180)),
     JSON.stringify(sp1 && sp1.a[1]));
  ok("  ... in the portrait phone (390 x 844, the picture at y 227..617) the strips are ABOVE and BELOW, each cut into four COLUMNS: 97.5 wide less the gap, a: y 10..217, b: y 627..834",
     !!sp2p && sp2p.a.every((b, i) => boxEq(b, 97.5 * i + 10, 10, 77.5, 207)) && sp2p.b.every((b, i) => boxEq(b, 97.5 * i + 10, 627, 77.5, 207)), JSON.stringify(sp2p && sp2p.b[3]));
  ok("  ... no room, no panels: null where neither pair of strips is at least `min` across (TRACES_SIDE_MIN = 90 CSS px) -- 89 either side is null and 90 is not; a picture that fills the host; a picture with room on ONE side only",
     TS.TRACES_SIDE_MIN === 90 && TS.tracesSidePanels(1000, 800, 89, 0, 822, 800, 90, 10) === null && TS.tracesSidePanels(1000, 800, 90, 0, 820, 800, 90, 10) !== null &&
     TS.tracesSidePanels(1280, 800, 0, 0, 1280, 800, 90, 10) === null && TS.tracesSidePanels(1280, 800, 400, 0, 800, 800, 90, 10) === null &&
     TS.tracesSidePanels(800, 1000, 0, 89, 800, 822, 90, 10) === null && TS.tracesSidePanels(800, 1000, 0, 90, 800, 820, 90, 10) !== null);
  let spn = 0, spBad = 0, spNull = 0;
  for (const W of [390, 800, 1280, 1920, 3000]) for (const H of [480, 800, 844, 1080]) for (const asp of [0.5, 1, 16 / 9, 2.4]) for (const g of [0, 4, 10]) for (const min of [60, 90]) {
    // (the picture as the stage lays it out: whole, centred)
    const wide = W / H < asp, w = wide ? W : H * asp, h = wide ? W / asp : H, x = (W - w) / 2, y = (H - h) / 2;
    const s = TS.tracesSidePanels(W, H, x, y, w, h, min, g);
    spn++;
    const wantLR = x >= min, wantTB = !wantLR && y >= min;
    if (!wantLR && !wantTB) { spNull++; if (s !== null) spBad++; continue; }
    const want = wantLR ? (i, sec) => lrWant(W, H, x, w, g, i, sec) : (i, sec) => tbWant(W, H, y, h, g, i, sec);
    const all = s ? Array.from(s.a).concat(Array.from(s.b)) : [];
    const hit = (p, q) => p.x < q.x + q.w - 1e-9 && q.x < p.x + p.w - 1e-9 && p.y < q.y + q.h - 1e-9 && q.y < p.y + p.h - 1e-9;
    const good = !!s && s.a.length === 4 && s.b.length === 4 && s.a.every((b, i) => boxEq(b, ...want(i, false))) && s.b.every((b, i) => boxEq(b, ...want(i, true))) &&
      all.every(b => b.w > 0 && b.h > 0 && b.x >= g - 1e-9 && b.y >= g - 1e-9 && b.x + b.w <= W - g + 1e-9 && b.y + b.h <= H - g + 1e-9 && !hit(b, { x: x, y: y, w: w, h: h })) &&
      all.every((b, i) => all.every((c, j) => i === j || !hit(b, c)));
    if (!good) spBad++;
  }
  ok("  ... over " + spn + " hosts, box shapes, gaps and minimum widths (" + spNull + " of them with no room: null): four boxes per strip, `a` in the first strip (left, or top) and `b` in the second, each its quarter of the strip inset by the gap, inside the host, clear of the picture and of each other",
     spn === 480 && spNull > 50 && spNull < spn - 50 && spBad === 0, spBad + " bad");
  ok("  ... and left / right wins where a picture leaves room both ways (200 x 300 at (300, 400) in 1000 x 1000: rows 250 high in strips 300 and 500 wide)",
     (s => !!s && s.a.every((b, i) => boxEq(b, 0, 250 * i, 300, 250)) && s.b.every((b, i) => boxEq(b, 500, 250 * i, 500, 250)))(TS.tracesSidePanels(1000, 1000, 300, 400, 200, 300, 90)));
  // the line cuts: 4 n floats (u_x | u_y | b_x | b_y) -> u_x, b_x, z+_x = u_x + b_x, z-_x = u_x - b_x, SIGNED
  const la = [3, 0, 1, -1,  4, 1, 0, 0,  0, 1, 1, 1,  0, -1, 0, 0], lm = TS.tracesLineFields(new Float32Array(la), 4);
  const lmY = TS.tracesLineFields(new Float32Array(la.map((v, k) => ((k >> 2) & 1 ? 99 - v : v))), 4);
  ok("tracesLineFields, by hand: u_x = (3, 0, 1, -1), b_x = (0, 1, 1, 1) give z+_x = u_x + b_x = (3, 1, 2, 0) and z-_x = u_x - b_x = (3, -1, 0, -2), signs kept; the y components are not read; four Float32Arrays of n, named u_x, b_x, z+_x, z−_x (and the withdrawn tracesLineMags is gone)",
     lm.length === 4 && Array.from(lm).every(r => r.length === 4 && r.constructor.name === "Float32Array") &&
     Array.from(lm[0]).join() === "3,0,1,-1" && Array.from(lm[1]).join() === "0,1,1,1" && Array.from(lm[2]).join() === "3,1,2,0" && Array.from(lm[3]).join() === "3,-1,0,-2" &&
     JSON.stringify(Array.from(lmY).map(r => Array.from(r))) === JSON.stringify(Array.from(lm).map(r => Array.from(r))) &&
     Array.from(TS.TRACES_CUT_NAMES).join() === "u_x,b_x,z+_x,z−_x" && vm.runInContext("typeof tracesLineMags", ctx) === "undefined", Array.from(lm).map(r => Array.from(r).join("/")).join("  "));

  ok("tracesSideKind: what a strip shows is \"spec\", \"cuts\" or \"time\" (TRACES_SIDE_KINDS, the two selects' options), and anything else -- a missing control -- is the default handed in",
     ["spec", "cuts", "time"].every(k => TS.tracesSideKind(k, "x") === k) && [undefined, null, "", "Spec", "energy", 0].every(v => TS.tracesSideKind(v, "time") === "time" && TS.tracesSideKind(v, "spec") === "spec") &&
     Array.from(TS.TRACES_SIDE_KINDS).map(k => k[0]).join() === "spec,cuts,time");
  ok("  ... and with no controls at all (traces.js in the bare context) opts() is: spectrum off, cut off, side panels OFF, first strip spec, second strip time, ticks on, fade 2 s",
     JSON.stringify(T.musicTraces.opts()) === JSON.stringify({ spec: false, cut: false, sides: false, sideA: "spec", sideB: "time", ticks: true, tau: 2 }), JSON.stringify(T.musicTraces.opts()));

  // ---- on the page: OFF, the overlay is what it was ------------------------------------------------
  const sq = () => JSON.parse(env.run(`function(){ return JSON.stringify({ ener: musicTraces.ener.map(e => e.v), rows4: musicTraces.rows4.map(e => e.rows.map(r => r.length)),
    spec: musicTraces.spec.length, line: musicTraces.reads.line || 0, o: musicTraces.opts(), old: typeof musicTraces.readPlane + typeof musicTraces.stage4 + typeof tracesSideRow }); }`));
  const sq0 = sq();
  ok("side panels OFF (switched off above), through every frame and readback since: no energy history, no line cuts, the solver's readCutLine never called (reads.line never counted), and nothing of the withdrawn plane read exists (readPlane, stage4, tracesSideRow)",
     sq0.o.sides === false && sq0.o.sideA === "spec" && sq0.o.sideB === "time" && sq0.ener.length === 0 && sq0.rows4.length === 0 && sq0.line === 0 && st().cutline === 0 && sq0.old === "undefinedundefinedundefined", JSON.stringify(sq0));
  const LRPIC = { x: 156, y: 0, w: 200, h: 512, tile: false }, TBPIC = { x: 0, y: 150, w: 512, h: 212, tile: false };
  const panelsOf = (pic, i, second) => pic === LRPIC ? { x: (second ? 356 : 0) + 10, y: 128 * i + 10, w: 136, h: 108 } : { x: 128 * i + 10, y: (second ? 362 : 0) + 10, w: 108, h: 130 };
  const enerIn = () => env.run(`function(){ musicTraces.ener = [4, 3, 2, 1, 0].map((age, j) => ({ t: 100000 - 1000 * age, v: [1, 2, 3, 4].map(m => m * (j + 1)) })); }`);
  const tx = () => cx.__texts.map(t => [t.t, t.x, t.y, t.align, t.base]);
  env.getEl("cbTrSpec").checked = true;
  inject();
  const base = draw(100000), baseT = JSON.stringify(tx());
  setPic(LRPIC);
  enerIn();
  const offPic = draw(100000);
  ok("  ... and with them off a picture hook changes nothing of the spectrum: the same six strokes (guide and two lines, in the band along the bottom) and the same labels, with or without one",
     base.length === 6 && base.every(e => sameBox(e.clip, SB)) && JSON.stringify(offPic) === JSON.stringify(base) && JSON.stringify(tx()) === baseT);

  // ---- ON, with a picture that leaves two strips ---------------------------------------------------
  // (the host is the card's 512 x 512 wrap at one backing pixel per CSS pixel; the picture is
  // put at x 156..356, so the strips are 156 wide: rows 128 high, boxes 136 x 108 after the gap)
  env.getEl("cbTrSides").checked = true;
  inject();
  enerIn();
  const on1 = draw(100000), on1T = tx();
  const clipIs = (e, b) => sameBox(e.clip, b);
  // (per row of panels: the first strip's box, then the second's. A "spec" box is six
  // strokes -- the k^(-5/3) guide, halo and line, then the two spectrum lines, halo and
  // line each; a "time" or a "cuts" box is four)
  const NST = { spec: 6, time: 4, cuts: 4 };
  const blocks = (d, pic, kA, kB, boxOf) => { const out = []; let at = 0;
    for (let i = 0; i < 4; i++) for (const [k, sec] of [[kA, false], [kB, true]]) { const B = (boxOf || panelsOf)(pic, i, sec); out.push({ k: k, i: i, B: B, s: d.slice(at, at + NST[k]) }); at += NST[k]; }
    return out; };
  const lane = (i, e) => [0.5 * e, 0.5 * e, 1.5 * e, 0.5 * e][i];         // inject(): E_u = E_b = H_c = e / 2
  const guideOK = (b) => { const g = TS.tracesGuide(SR, b.B);
    return b.s[0].s === "rgba(0,0,0,0.3)" && b.s[0].w === 3 * (b.B.w > 200 ? 2 : 1) && b.s[1].pts.length === 2 && close(b.s[1].pts[0][0], g.x0, 1e-9) && close(b.s[1].pts[0][1], g.y0, 1e-9) &&
           close(b.s[1].pts[1][0], g.x1, 1e-9) && close(b.s[1].pts[1][1], g.y1, 1e-9) && JSON.stringify(b.s[0].pts) === JSON.stringify(b.s[1].pts); };
  const kindOK = b => {
    if (b.s.length !== NST[b.k] || !b.s.every(e => clipIs(e, b.B) && e.dash === 0)) return false;
    if (b.k === "spec") { const ln = b.s[5], p0 = T.tracesLogMap(1, lane(b.i, 0.01), SR, b.B), p9 = T.tracesLogMap(10, lane(b.i, 0.01 * Math.pow(10, -5 / 3)), SR, b.B);
      return guideOK(b) && ln.pts.length === 84 && close(ln.pts[0][0], b.B.x, 1e-9) && close(ln.pts[0][1], p0[1], 1e-6) && close(ln.pts[9][0], p9[0], 1e-9) && close(ln.pts[9][1], p9[1], 1e-6) &&
             ln.s === css(cm(1), 1) && b.s[3].s === css(cm(0.5 + 0.5 / Math.E), Math.exp(-1)) && b.s[4].s === "rgba(0,0,0,0.350)"; }
    if (b.k === "time") { const a = b.s[1], z = b.s[3], B = b.B;
      return a.pts.length === 4 && z.pts.length === 2 && [4, 3, 2, 1].every((age, j) => close(a.pts[j][0], B.x + B.w * (1 - age / 8), 1e-9) && close(a.pts[j][1], B.y + B.h * (1 - 0.92 * (j + 1) / 5), 1e-9)) &&
             close(z.pts[0][0], a.pts[3][0], 1e-12) && close(z.pts[1][0], B.x + B.w, 1e-9) && close(z.pts[1][1], B.y + B.h * 0.08, 1e-9) &&
             b.s[0].s === "rgba(0,0,0," + (0.35 * Math.exp(-0.5)).toFixed(3) + ")" && b.s[2].s === "rgba(0,0,0,0.350)" && z.s === css(cm(1), 1); }
    const old = b.s[0], ln = b.s[3], B = b.B;
    return ln.pts.length === 256 && close(ln.pts[0][0], B.x + B.w * 0.5 / 256, 1e-9) && close(ln.pts[255][0], B.x + B.w * 255.5 / 256, 1e-9) &&
           close(ln.pts[0][1], B.y + B.h * 0.04, 1e-9) && close(ln.pts[1][1], B.y + B.h * 0.73, 1e-9) && close(ln.pts[2][1], B.y + B.h * 0.5, 1e-9) && close(ln.pts[255][1], B.y + B.h * 0.5, 1e-9) &&
           old.s === "rgba(0,0,0," + (0.35 * Math.exp(-1)).toFixed(3) + ")" && ln.s === css(cm(1), 1); };
  const titleOf = (k, i) => k === "spec" ? ["E_u", "E_b", "E+", "E−"][i] + "(k)" : k === "time" ? ["E_u", "E_b", "E+", "E−"][i] + "(t)  max " + 5 * (i + 1)
                                         : ["u_x", "b_x", "z+_x", "z−_x"][i] + "(y)  ±" + (i + 1);
  const titlesOK = (t, bl, pad) => t.length === 8 && bl.every((b, k) => JSON.stringify(t[k]) === JSON.stringify([titleOf(b.k, b.i), b.B.x + (pad || 4), b.B.y + (pad || 4), "left", "top"]));
  const rowsIn = () => env.run(`function(){ musicTraces.rows4 = [500, 0].map(age => ({ t: 100000 - age, rows: [1, 2, 3, 4].map(m => { const r = new Float32Array(256); r[0] = m; r[1] = -m / 2; return r; }) })); }`);
  const bl1 = blocks(on1, LRPIC, "spec", "time");
  ok("side panels ON (first strip spec, second time, the defaults): the spectrum along the bottom is GONE (nothing clipped to its band, no k or decade labels) and every stroke is clipped to one of eight boxes -- per row, six strokes in the left box (the guide, then two spectrum lines, halo and line each) then four in the right (the energy's history in two pieces)",
     on1.length === 40 && !on1.some(e => sameBox(e.clip, SB) || sameBox(e.clip, SBAND)) && bl1.every(b => b.s.length === NST[b.k] && b.s.every(e => clipIs(e, b.B))) && on1.every(e => e.dash === 0) &&
     !on1T.some(t => /^k = |^10/.test(t[0])), on1.length + " strokes; clips " + Array.from(new Set(on1.map(e => e.clip && e.clip.join()))).length);
  ok("  ... the left boxes are the four SPECTRA, log-log on the bottom spectrum's own axes (k 1..84, four decades under 0.3 amp^2): E_u, E_b, E+ = 3 x and E- = 1 x the kinetic one for the injected E_u = E_b = H_c, the older line at 1/e",
     bl1.filter(b => b.k === "spec").length === 4 && bl1.filter(b => b.k === "spec").every(kindOK), on1[5] && on1[5].pts.length + " points; E_u at k = 1: y " + on1[5].pts[0][1]);
  ok("  ... each under its own k^(-5/3) GUIDE, drawn first: tracesGuide for that box (a decade under the top at 20% of the log-k axis, to 80% of it), a dark halo then the faint line -- four guides, one per spectrum box, and none anywhere else",
     bl1.filter(b => b.k === "spec").every(guideOK) && on1.filter(e => e.s === "rgba(0,0,0,0.3)").length === 4 && bl1.filter(b => b.k === "time").every(b => !b.s.some(e => e.s === "rgba(0,0,0,0.3)")),
     JSON.stringify(on1[1] && on1[1].pts));
  ok("  ... the right boxes are the same four energies against TIME: 4 x the fade time across (8 s), now at the right edge, zero at the bottom and the largest value on show 0.92 of the way up, in pieces of up to four samples each faded by its own age (samples 4, 3, 2, 1, 0 s old: a piece of four, 1 s old, and a piece of two, new)",
     bl1.filter(b => b.k === "time").length === 4 && bl1.filter(b => b.k === "time").every(kindOK), JSON.stringify(on1[9] && on1[9].pts));
  ok("  ... with the tick labels on each box has a title and nothing else, at its top LEFT (4 px in): \"E_u(k)\" ... on the spectra, \"E_u(t)  max 5\" ... (the largest value on show: 5, 10, 15, 20) on the histories",
     titlesOK(on1T, bl1), on1T.map(t => t[0] + "@" + t[1] + "," + t[2] + " " + t[3]).join(" | "));
  env.getEl("cbTrSpec").checked = false;
  const on2 = draw(100000);
  env.getEl("cbTrTicks").checked = false;
  const on3 = draw(100000), on3T = tx().length;
  env.getEl("cbTrTicks").checked = true;
  ok("  ... the panels do not need the `spectrum` box (unticked, its default: the same 40 strokes), and without tick labels they are the same strokes with no text at all",
     JSON.stringify(on2) === JSON.stringify(on1) && JSON.stringify(on3) === JSON.stringify(on1) && on3T === 0);
  setPic(TBPIC);
  const on4 = draw(100000), bl4 = blocks(on4, TBPIC, "spec", "time");
  ok("  ... a picture that spans the host's WIDTH (y 150..362) puts the panels above and below it, four columns each: the same 40 strokes, clipped to boxes 108 x 130 at x = 10, 138, 266, 394, the first strip (the spectra) on top",
     on4.length === 40 && bl4.every(kindOK) && close(on4[5].pts[0][0], 10, 1e-9) && close(on4[9].pts[1][0], 10 + 108, 1e-9) && on4[0].clip[1] === 10 && on4[6].clip[1] === 372,
     Array.from(new Set(on4.map(e => e.clip && e.clip.join()))).join(" | "));
  // second strip = line cuts (values put in by hand: lane i's row is i + 1 at sample 0, MINUS half that at sample 1, zero after)
  env.getEl("selTrB").value = "cuts";
  setPic(LRPIC);
  rowsIn();
  const cu = draw(100000), cuT = tx(), blc = blocks(cu, LRPIC, "spec", "cuts");
  ok("second strip = line cuts: the right boxes show u_x, b_x, z+_x, z−_x along the line instead of the histories, SIGNED -- sample i at (i + 0.5) / n of the box, zero at its mid-height, the largest |value| on show 0.92 of the way from there to its top (and minus half of it 0.23 of the box below the middle), fading as the line cut does (0.5 s old: 1/e)",
     cu.length === 40 && blc.every(kindOK) && JSON.stringify(cu.slice(0, 6)) === JSON.stringify(on1.slice(0, 6)), cu.length + " strokes; " + JSON.stringify(cu[9] && cu[9].pts.slice(0, 3)));
  ok("  ... titled \"u_x(y)  ±1\", \"b_x(y)  ±2\", \"z+_x(y)  ±3\", \"z−_x(y)  ±4\", at the top left like the rest (the left boxes keep their spectra and titles)",
     titlesOK(cuT, blc) && cuT[1][0] === "u_x(y)  ±1" && cuT[0][0] === "E_u(k)", cuT.map(t => t[0]).join(" | "));
  // every pairing of the two selects, the same kind twice included
  const combos = [];
  for (const kA of ["spec", "cuts", "time"]) for (const kB of ["spec", "cuts", "time"]) {
    env.getEl("selTrA").value = kA; env.getEl("selTrB").value = kB;
    inject(); enerIn(); rowsIn();
    const d = draw(100000), t = tx(), bl = blocks(d, LRPIC, kA, kB), guides = d.filter(e => e.s === "rgba(0,0,0,0.3)").length;
    const good = d.length === 4 * (NST[kA] + NST[kB]) && bl.every(kindOK) && titlesOK(t, bl) && guides === 4 * ((kA === "spec") + (kB === "spec")) &&
                 bl.every(b => (b.s[0].s === "rgba(0,0,0,0.3)") === (b.k === "spec")) && !d.some(e => sameBox(e.clip, SB) || sameBox(e.clip, SBAND));
    if (!good) combos.push(kA + "+" + kB + " (" + d.length + " strokes, " + guides + " guides)");
  }
  ok("all NINE pairings of the two selects (spec / cuts / time on the first strip x the same on the second, a kind twice included) draw the first select's kind in the LEFT boxes and the second's in the RIGHT: the right strokes at the right places in each box, a guide in exactly the spectrum boxes, the matching title at each top left, and never the bottom band",
     combos.length === 0, combos.join("; "));
  env.getEl("selTrA").value = "time"; env.getEl("selTrB").value = "spec";
  setPic(TBPIC); inject(); enerIn(); rowsIn();
  const sw = draw(100000), blsw = blocks(sw, TBPIC, "time", "spec");
  ok("  ... and above / below likewise: with time first and spec second on the picture that spans the width, the histories are on TOP and the spectra (with their guides) underneath",
     sw.length === 40 && blsw.every(kindOK) && sw[0].clip[1] === 10 && sw[4].clip[1] === 372 && sw[4].s === "rgba(0,0,0,0.3)", sw.length + " strokes");
  env.getEl("selTrA").value = "spec";
  setPic(LRPIC);
  env.getEl("selTrB").value = "time";
  env.run("function(){ musicTraces.rows4 = []; }");

  // ---- ON, but nowhere to put them: no panels, and the bottom band is back ---------------------------
  env.getEl("cbTrSpec").checked = true;
  const noPanel = [];
  for (const [tag, pic] of [["tiled", Object.assign({}, LRPIC, { tile: true })], ["narrow", { x: 56, y: 0, w: 400, h: 512, tile: false }],
                            ["fills", { x: 0, y: 0, w: 512, h: 512, tile: false }], ["no height", { x: 156, y: 0, w: 200, h: 0, tile: false }]]) {
    setPic(pic); inject(); enerIn();
    const d = draw(100000);
    if (!(JSON.stringify(d) === JSON.stringify(base) && JSON.stringify(tx()) === baseT)) noPanel.push(tag);
  }
  env.run("function(){ musicTraces.picture = () => null; }"); inject(); enerIn();
  if (JSON.stringify(draw(100000)) !== JSON.stringify(base)) noPanel.push("hook answers null");
  setPic(null); inject(); enerIn();
  const noHook = draw(100000), noHookT = JSON.stringify(tx());
  ok("side panels ON but no room for them -- the sides repeat the picture (tile), the strips are under 90 px (56 here), the picture fills the host, has no height, or the hook answers null: NO panel is drawn and the bottom spectrum is back, stroke for stroke and label for label",
     noPanel.length === 0, noPanel.join(", "));
  ok("  ... and on a page with no picture hook at all -- this one, ?plain&traces: the card page -- the option draws no panels either: the bottom spectrum as before",
     env.run("function(){ return musicTraces.picture; }") === null && JSON.stringify(noHook) === JSON.stringify(base) && noHookT === baseT);
  env.getEl("cbTrSpec").checked = false;
  setPic(Object.assign({}, LRPIC, { tile: true }));
  ok("  ... with the `spectrum` box unticked as well there is then nothing to draw", draw(100000).length === 0 && cx.__texts.length === 0);

  // ---- ON: what is sampled, frame by frame -----------------------------------------------------------
  // (the loop's stats readback is zeros under the stub: it is replaced by E_u = 3, E_b = 2,
  // H_c = 1 -- floats 2, 3 and 8 -- and counted; the spectrum box stays unticked. The stub's
  // clock jumps seconds per frame, so the histories EXPIRE between frames here: what was
  // sampled is read off the push log, not off how much of it is still held)
  setPic(LRPIC);
  env.run(`function(){ window.__stats = 0; solver.readStats = async () => { window.__stats++; const s = new Float32Array(12); s[2] = 3; s[3] = 2; s[8] = 1; return s; };
    musicTraces.spec = []; musicTraces.ener = []; musicTraces.mark.spec = null; }`);
  env.getEl("cbTrSides").checked = false;
  reset();
  await run(24, true);
  const e0 = st(), e0s = env.run("function(){ const n = window.__stats; window.__stats = 0; return n; }"), e0q = sq();
  env.getEl("cbTrSides").checked = true;
  reset();
  await run(24, true);
  const e1 = st(), e1s = env.run("function(){ const n = window.__stats; window.__stats = 0; return n; }"), e1q = sq();
  ok("side panels ON, 24 frames of a moving state: the energies' history is sampled by TIME, at most every 33 ms -- under the stub, whose clock steps 250 ms a call, that is every one of the 24 frames -- each (E_u, E_b, E+, E-) = (3, 2, 6, 4) off the loop's own stats readback, which is read no more often than with the panels off (" + e0s + " times either way)",
     e1.p.filter(e => e.which === "ener").map(e => e.f).join() === Array.from({ length: 24 }, (_, i) => i + 1).join() && e1q.ener.length >= 1 && e1q.ener.every(v => v.join() === "3,2,6,4") &&
     e1s === e0s && e0s > 0 && e0q.ener.length === 0 && e0.p.length === 0, e1.p.filter(e => e.which === "ener").map(e => e.f).join() + "; stats " + e0s + " / " + e1s);
  ok("  ... the SPECTRUM is sampled for the panels although its own box is unticked (one readSpectrum on frames 6, 12, 18, 24; with both off, above, none), and nothing else is read: no row gather, no readCutLine",
     e0.sp === 0 && e1.sp === 4 && e1.r.spec === 4 && e1.p.filter(e => e.which === "spec").map(e => e.f).join() === "6,12,18,24" && e1.cut === 0 && e1.r.cut === 0 && e1.cutline === 0 && e1q.line === 0 &&
     e1q.rows4.length === 0 && e1.p.every(e => e.which === "spec" || e.which === "ener"), e1.sp + " spectrum readbacks, pushes " + e1.p.map(e => e.which[0] + e.f).join(" "));
  reset();
  await run(12, false);
  const e2 = st();
  ok("  ... a state that stands still adds nothing: no energy sample, no spectrum", e2.p.length === 0 && e2.sp === 0 && e2.f === 12, e2.p.length + " pushes");
  // second side = cuts: the solver's own readCutLine, here answering u = (3, 4), b = (i, 9) at sample i
  env.run(`function(){ window.__iz = []; solver.readCutLine = iz => { window.__cutline++; window.__iz.push(iz); const ny = solver.p.ny, a = new Float32Array(4 * ny);
    for (let i = 0; i < ny; i++) { a[i] = 3; a[ny + i] = 4; a[2 * ny + i] = i; a[3 * ny + i] = 9; } return Promise.resolve(a); }; }`);
  env.getEl("selTrB").value = "cuts";
  reset();
  await run(12, true);
  const c1 = st(), c1q = sq();
  const c1v = JSON.parse(env.run("function(){ const e = musicTraces.rows4[musicTraces.rows4.length - 1]; return JSON.stringify({ v: e.rows.map(r => [r[0], r[7], r[255]]), iz: window.__iz, keys: Object.keys(e).sort() }); }"));
  ok("second side = cuts, 12 frames: ONE readCutLine(0) on each due frame (6 and 12), counted in reads.line, and no other readback of its own beyond the spectrum's two -- no row gather out of the display chain",
     c1.cutline === 2 && c1q.line === 2 && c1v.iz.join() === "0,0" && c1.p.filter(e => e.which === "rows4").map(e => e.f).join() === "6,12" && c1.cut === 0 && c1.r.cut === 0 && c1.sp === 2,
     c1.cutline + " line reads, pushes " + c1.p.map(e => e.which[0] + e.f).join(" "));
  ok("  ... each pushed as four rows of ny = 256 -- u_x = 3, b_x = i, z+_x = 3 + i, z-_x = 3 - i of what the solver answered -- with a time and nothing else (no mode, no kind)",
     c1q.rows4.length >= 1 && c1q.rows4.every(r => r.join() === "256,256,256,256") && c1v.keys.join() === "rows,t" &&
     JSON.stringify(c1v.v) === JSON.stringify([[3, 3, 3], [0, 7, 255], [3, 10, 258], [3, -4, -252]]), JSON.stringify(c1v.v));
  // (sample() by hand, with no draw in between, so nothing expires: one line on |u|'s mode, one on another)
  env.run("function(){ musicTraces.rows4 = []; stateBumped(); musicTraces.sample(solver, 1000); }");
  await turn();
  env.run("function(){ window.__modeOf2 = solver.modeOf; solver.modeOf = () => 0; stateBumped(); musicTraces.sample(solver, 1100); }");
  await turn();
  const c2q = sq(), c2t = env.run("function(){ return musicTraces.rows4.map(e => e.t).join(); }");
  env.run("function(){ solver.modeOf = window.__modeOf2; }");
  reset();
  await run(12, false);
  const c3 = st();
  ok("  ... another displayed field does NOT start that history again (these cuts are not the picture's field: both lines are held), and a state that stands still reads no line",
     c2q.rows4.length === 2 && c2t === "1000,1100" && c2q.line === 4 && c3.cutline === 0 && c3.p.every(e => e.which !== "rows4") && c3.sp === 0, c2q.rows4.length + " lines; " + c3.cutline + " reads in 12 still frames");
  env.getEl("selTrB").value = "time";
  reset();
  await run(6, true);
  const c4 = st(), c4q = sq();
  ok("  ... back on \"time\" the next sample empties the line cuts and reads no more of them", c4q.rows4.length === 0 && c4.cutline === 0 && c4.p.every(e => e.which !== "rows4") && c4q.ener.length > 0);
  env.getEl("selTrB").value = "cuts";
  env.run("function(){ window.__rcl = solver.readCutLine; solver.readCutLine = undefined; }");
  reset();
  await run(12, true);
  const c5 = st(), c5q = sq();
  env.run("function(){ solver.readCutLine = window.__rcl; }");
  ok("  ... and a solver with no readCutLine is tolerated: frames pass, nothing is counted or pushed for the cuts, nothing is logged as a failure",
     c5.f === 12 && c5q.rows4.length === 0 && c5q.line === 0 && c5.p.every(e => e.which !== "rows4") && c5.sp === 2 && realFails(env).length === 0, realFails(env).slice(0, 2).join(" | "));
  // what is read follows what the two strips show (the spectrum box unticked; six frames of a moving state, one due sample)
  const sampBad = [], sampLog = [];
  for (const kA of ["spec", "cuts", "time"]) for (const kB of ["spec", "cuts", "time"]) {
    env.getEl("selTrA").value = kA; env.getEl("selTrB").value = kB;
    reset();
    await run(6, true);
    const r6 = st(), has = k => kA === k || kB === k;
    const got = [r6.sp > 0, r6.cutline > 0, r6.p.some(e => e.which === "ener"), r6.p.some(e => e.which === "rows4"), r6.cut > 0].join();
    sampLog.push(kA[0] + kB[0] + ":" + r6.sp + "/" + r6.cutline + "/" + r6.p.filter(e => e.which === "ener").length);
    if (got !== [has("spec"), has("cuts"), has("time"), has("cuts"), false].join() || r6.sp > 1 || r6.cutline > 1) sampBad.push(kA + "+" + kB + " " + got);
  }
  env.getEl("selTrA").value = "cuts"; env.getEl("selTrB").value = "cuts"; env.getEl("cbTrSpec").checked = true;
  reset();
  await run(6, true);
  const ccs = st();
  env.getEl("cbTrSpec").checked = false; env.getEl("selTrA").value = "spec"; env.getEl("selTrB").value = "time";
  ok("what is sampled follows the two selects, in all nine pairings: the spectrum is read only where a strip shows spec (spec + spec: no line read, no energy pushed), the solver's cut line only where one shows cuts (cuts + cuts: no spectrum), the energies pushed only where one shows time (time + time: neither read) -- one read at most per due frame, and never a row gather",
     sampBad.length === 0 && sampLog.length === 9, sampBad.join("; ") || sampLog.join(" "));
  ok("  ... and cuts + cuts with the `spectrum` box TICKED does read the spectrum (for the bottom band, should the panels have no room)", ccs.sp === 1 && ccs.cutline === 1 && !ccs.p.some(e => e.which === "ener"), ccs.sp + "/" + ccs.cutline);
  env.getEl("selTrB").value = "time";
  env.getEl("cbTrSides").checked = false;
  reset();
  await run(6, true);
  ok("side panels OFF again: the next frames empty the energies' history (and, the spectrum box being unticked too, read nothing at all)",
     sq().ener.length === 0 && sq().rows4.length === 0 && st().sp === 0 && st().p.length === 0 && st().cutline === 0, JSON.stringify(st().p));
  // the energies' clock, with times of this file's own (frame() by hand, the state moved before each)
  const TE = vm.runInContext("({ tracesEnerDue, TRACES_ENER_MS, TRACES_ENER_MAX })", ctx);
  ok("tracesEnerDue: due when there is no sample yet, and then once 32 ms have passed since the last (TRACES_ENER_MS = 33, less one for a 60 Hz frame's jitter): 31 ms is not, 32 is; a clock that stepped back is not",
     TE.TRACES_ENER_MS === 33 && TE.tracesEnerDue(0, 5) === true && TE.tracesEnerDue(undefined, 5) === true && TE.tracesEnerDue(1000, 1016) === false && TE.tracesEnerDue(1000, 1031) === false &&
     TE.tracesEnerDue(1000, 1032) === true && TE.tracesEnerDue(1000, 1033) === true && TE.tracesEnerDue(1000, 990) === false);
  ok("  ... and the history holds 768 of them (TRACES_ENER_MAX, handed to tracesPush): more than 4 x the fade slider's longest 6 s at 30 a second (720), where the other histories' cap of 256 would cut the panel short",
     TE.TRACES_ENER_MAX === 768 && TE.TRACES_ENER_MAX >= 4 * 6 * 1000 / TE.TRACES_ENER_MS && 4 * 6 * 1000 / TE.TRACES_ENER_MS > 256 && +env.getEl("rTrTau").max === 6 &&
     /tracesPush\(this\.ener, \{[^\n]*\}, TRACES_ENER_MAX\);/.test(src) && T.tracesPush(Array.from({ length: 768 }, (_, i) => ({ t: i })), { t: 768 }, 768).length === 768);
  env.getEl("cbTrSides").checked = true;
  env.run("function(){ musicTraces.ener = []; musicTraces.enerAt = 0; musicTraces.mark.ener = null; }");
  const enerAt = ts => JSON.parse(env.run(`function(ts){ const out = []; for (const [t, bump] of ts) { if (bump) stateBumped(); const n = musicTraces.ener.length; statsCache.s = new Float32Array([0, 0, 3, 2, 0, 0, 0, 0, 1, 0, 0, 0]);
    musicTraces.frame(solver, t); if (musicTraces.ener.length > n) out.push(t); } return JSON.stringify(out); }`, ts));
  const et = enerAt([[5000000, true], [5000016, true], [5000031, true], [5000032, true], [5000048, true], [5000063, true], [5000064, true], [5000100, false], [5000200, false], [5000216, true]]);
  ok("  ... on the page, frames at +0, 16, 31, 32, 48, 63, 64 ms of a moving state sample at +0, +32 and +64 only; a state that has not moved is not sampled however long it waits (+100, +200), and is once it has (+216)",
     et.join() === "5000000,5000032,5000064,5000216" && env.run("function(){ return musicTraces.enerAt; }") === 5000216, et.map(t => "+" + (t - 5000000)).join(" "));
  env.getEl("cbTrSides").checked = false;
  env.run("function(){ musicTraces.frame(solver, 5000300); }");

  // a handler that THROWS must not leave its readback marked busy for ever (each chain ends
  // .catch(console.error).then(busy = false)): tracesPush is made to throw for the next three pushes
  await turn();                                    // (whatever the frames above started has landed)
  env.getEl("cbTrSpec").checked = true; env.getEl("cbTrCut").checked = true; env.getEl("cbTrSides").checked = true; env.getEl("selTrB").value = "cuts";
  env.run(`function(){ musicTraces.spec = []; musicTraces.cut = []; musicTraces.rows4 = []; musicTraces.reads = { spec: 0, cut: 0, line: 0 };
    window.__errs = []; window.__cerr = console.error; console.error = e => { window.__errs.push(String(e && e.message || e)); };
    const tp = tracesPush; let k = 3; window.__tp = tp; tracesPush = function(h, e, m){ if (k > 0) { k--; throw new Error("boom in the handler"); } return tp(h, e, m); };
    stateBumped(); musicTraces.sample(solver, 6000000); }`);
  await turn();
  const th1 = JSON.parse(env.run("function(){ return JSON.stringify({ busy: [musicTraces.busy.spec, musicTraces.busy.cut, musicTraces.busy4], r: musicTraces.reads, n: [musicTraces.spec.length, musicTraces.cut.length, musicTraces.rows4.length], errs: window.__errs }); }"));
  env.run("function(){ stateBumped(); musicTraces.sample(solver, 6000100); }");
  await turn();
  const th2 = JSON.parse(env.run(`function(){ const o = { busy: [musicTraces.busy.spec, musicTraces.busy.cut, musicTraces.busy4], r: musicTraces.reads, n: [musicTraces.spec.length, musicTraces.cut.length, musicTraces.rows4.length] };
    console.error = window.__cerr; tracesPush = window.__tp; return JSON.stringify(o); }`));
  ok("a throw INSIDE a readback's own handler -- the spectrum's, the line cut's and the side cuts', one each -- is reported (console.error) and leaves none of the three marked busy, with nothing pushed",
     th1.busy.join() === "false,false,false" && th1.r.spec === 1 && th1.r.cut === 1 && th1.r.line === 1 && th1.n.join() === "0,0,0" && th1.errs.length === 3 && th1.errs.every(e => /boom in the handler/.test(e)),
     JSON.stringify(th1));
  ok("  ... so the next sample of a moved state starts each of the three again, and those land", th2.busy.join() === "false,false,false" && th2.r.spec === 2 && th2.r.cut === 2 && th2.r.line === 2 && th2.n.join() === "1,1,1", JSON.stringify(th2));
  env.getEl("cbTrCut").checked = false; env.getEl("cbTrSpec").checked = false;

  // a side cut older than the CUT's fade (a quarter of the slider's) is gone, though a spectrum line that old stays
  setPic(LRPIC);
  const oldCut = age => { env.run(`function(age){ musicTraces.rows4 = [{ t: 7000000 - age, rows: [1, 2, 3, 4].map(m => { const r = new Float32Array(256); r[0] = m; return r; }) }];
      const nb = solver.nb, p = new Float32Array(3 * nb); for (let b = 1; b < nb; b++) { p[b] = p[nb + b] = 0.005 * Math.pow(b, -5 / 3); }
      musicTraces.spec = [{ t: 7000000 - age, perp: p, nb: nb, bins: tracesBinRange(256, 256, 2 * Math.PI, 2 * Math.PI) }]; musicTraces.ener = []; }`, age);
    const d = draw(7000000); return [d.length, env.run("function(){ return musicTraces.rows4.length; }"), env.run("function(){ return musicTraces.spec.length; }")].join(); };
  const oc19 = oldCut(1900), oc20 = oldCut(2000);
  ok("the side cuts fade on the CUT's clock, a quarter of the slider's 2 s: a line 1.9 s old (2.2%) is drawn in all four boxes beside the spectra of the same age (and their guides), one 2.0 s old (1.8%) is dropped while those spectra (37%) stay",
     oc19 === "24,1,1" && oc20 === "16,0,1", oc19 + " | " + oc20);
  env.run("function(){ musicTraces.rows4 = []; musicTraces.spec = []; }");

  // devicePixelRatio 2: the host is still 512 CSS px, the canvas 1024 backing px, and the
  // picture hook answers in CSS px -- every length of the panels and of the cut doubles
  env.getEl("selTrB").value = "time";
  env.run("function(){ window.devicePixelRatio = 2; }");
  const box2 = (i, second) => ({ x: (second ? 712 : 0) + 20, y: 256 * i + 20, w: 272, h: 216 });
  setPic(LRPIC); inject(); enerIn();
  const hd = draw(100000), hdW = env.run("function(){ return musicTraces.canvas.width + 'x' + musicTraces.canvas.height; }"), hdT = tx();
  ok("at devicePixelRatio 2 (a 1024 x 1024 canvas on the same 512 px host) the picture's CSS box is scaled to backing pixels for the panels: the strips are 312 wide, the boxes 272 x 216 after a 20 px gap, at x = 20 and x = 732, and the titles sit 8 px inside them",
     hdW === "1024x1024" && hd.length === 40 && blocks(hd, LRPIC, "spec", "time", (pc, i, sec) => box2(i, sec)).every(kindOK) && close(hd[9].pts[1][0], 732 + 272, 1e-9) && close(hd[5].pts[0][0], 20, 1e-9) &&
     hd[0].w === 6 && hd[1].w === 2 && titlesOK(hdT, blocks(hd, LRPIC, "spec", "time", (pc, i, sec) => box2(i, sec)), 8) && hdT[0][1] === 28 && hdT[0][2] === 28 && hdT[1][1] === 732 + 8, hdW + "; clips " + Array.from(new Set(hd.map(e => e.clip && e.clip.join()))).slice(0, 2).join(" | "));
  env.getEl("cbTrSpec").checked = true;
  setPic({ x: 80, y: 0, w: 352, h: 512, tile: false }); inject(); enerIn();
  const hdNarrow = draw(100000);
  setPic({ x: 100, y: 0, w: 312, h: 512, tile: false }); inject(); enerIn();
  const hdRoom = draw(100000);
  ok("  ... and the 90 px minimum is CSS pixels there too: strips of 80 CSS px (160 backing) take no panels -- the bottom spectrum is drawn, in its band of the 1024 px canvas -- and strips of 100 do",
     hdNarrow.length === 6 && hdNarrow.every(e => e.clip && close(e.clip[1], 1024 - 204.8, 1e-9) && e.clip[0] + e.clip[2] === 1024) &&
     hdRoom.length === 40 && blocks(hdRoom, LRPIC, "spec", "time", (pc, i, sec) => ({ x: (sec ? 824 : 0) + 20, y: 256 * i + 20, w: 160, h: 216 })).every(b => b.s.length === NST[b.k] && b.s.every(e => clipIs(e, b.B))), hdNarrow.length + " and " + hdRoom.length + " strokes");
  env.getEl("cbTrSpec").checked = false; env.getEl("cbTrSides").checked = false; env.getEl("cbTrCut").checked = true;
  cutIn([{ t: 200000, fill: 0, head: [2, 0, -2], max: 2, kind: "signed", mode: 0 }]);
  setPic({ x: 100, w: 256, y: 0, h: 512, tile: false });
  const hc1 = draw(200000);
  setPic({ x: 100, w: 256, y: 0, h: 512, tile: true });
  const hc2 = draw(200000), hx = hc2.filter((e, i) => i % 2 === 1).map(e => e.pts[0][0]).sort((a, b) => a - b);
  ok("  ... and for the line cut: with picture() = { x: 100, w: 256 } (CSS) sample ix is at 200 + 2 (ix + 0.5) backing px, in a band 204.8 high; tiled, the three copies start at -312, 200 and 712 (+1)",
     hc1.length === 2 && hc1[1].pts.length === 256 && hc1[1].pts.every((q, ix) => close(q[0], 200 + 2 * (ix + 0.5), 1e-12)) && close(hc1[1].pts[0][1], 0) && close(hc1[1].pts[2][1], 204.8) &&
     hc2.length === 6 && hx.length === 3 && close(hx[0], -311) && close(hx[1], 201) && close(hx[2], 713), hc1.length + " strokes, x " + (hc1[1] && hc1[1].pts[0][0]) + " .. ; tiled " + hx.join(" "));
  env.getEl("cbTrCut").checked = false;
  env.run("function(){ window.devicePixelRatio = 1; musicTraces.cut = []; musicTraces.fit(); }");

  // the page's own frame hook still runs under the traces' wrapper
  const probe = async E => { E.env.run("function(){ music.autoMax = NaN; music.autoAt = -1e9; cards.disp[0].dirty = true; }"); await E.frame(); await E.frame();
    return JSON.parse(E.env.run("function(){ return JSON.stringify({ a: music.autoMax, read: musicScaleReadout(), fixed: !!solver.disp[0].musicFixed, f: musicTraces.frames }); }")); };
  const pb = await probe(P);
  ok("under the traces' frame-hook wrapper the page's OWN hook still runs every frame: musicScaleProbe reads what the autoscale would be (0 under the stub, where it was NaN) and the readout says so",
     pb.fixed === true && pb.a === 0 && /; the autoscale would be 0\.00/.test(pb.read) && /frameHook = async sv => \{ if \(sv === solver\) musicTraces\.frame\(sv, performance\.now\(\)\); await fh\(sv\); \};/.test(fs.readFileSync(path.join(dir, "music.html"), "utf8")),
     JSON.stringify(pb));
  setPic(null);
  ok("through all of the side panels nothing non-finite reached the canvas and the page logged no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 3).join(" | "));
  }
  // any host: the stage's element is not the card's wrap
  const moved = env.run(`function(){ const h = document.createElement("div"), old = musicTraces.canvas; h.id = "stage";
    h.clientWidth = 1280; h.clientHeight = 720; const c = musicTraces.attach(h);
    return c !== old && old.parentNode === null && c.parentNode === h && h.children.length === 1 && c.width === 1280 && c.height === 720; }`);
  ok("attach(host) takes ANY host: moved to a 1280 x 720 element, the old canvas is removed and the new one is sized from the new host", moved === true);
  // The stage (the bare URL) has the traces too, and there the shell (stage.js, leg 15) has made #stage. The
  // wiring line hands the traces `document.querySelector("#stage") || cards.disp[0].wrap`: in
  // a browser that is #stage itself (the canvas beside the picture, so the zoom pulse does
  // not scale it). The stub's querySelector knows no "#stage", so HERE the canvas lands on
  // the card's wrap -- which the shell has moved into #stage. Both halves are stated.
  const S = await musicPage("");
  const stq = JSON.parse(S.env.run(`function(){ const c = musicTraces.canvas, w = cards.disp[0].wrap;
    return JSON.stringify({ on: musicTraces.on, cv: !!c, inWrap: !!c && c.parentNode === w, wrapInStage: !!stage.el && w.parentNode === stage.el,
      qs: document.querySelector("#stage") === null, n: w.children.filter(x => x.className === "traces").length }); }`));
  ok("the bare URL has the traces on: ONE canvas, inside #stage -- under the stub on the picture's wrap (which the shell moved into #stage), because the stub cannot resolve \"#stage\"; the wiring line asks for #stage first",
     stq.on === true && stq.cv && stq.inWrap && stq.wrapInStage && stq.qs === true && stq.n === 1 &&
     /musicTraces\.attach\(document\.querySelector\("#stage"\) \|\| cards\.disp\[0\]\.wrap\);/.test(fs.readFileSync(path.join(dir, "music.html"), "utf8")) &&
     grp(S.env).indexOf("grpTraces") >= 0 && realFails(S.env).length === 0, JSON.stringify(stq));
  S.env.run("function(){ music.autoMax = NaN; music.autoAt = -1e9; cards.disp[0].dirty = true; }");
  await S.frame(); await S.frame();
  ok("  ... and there too the page's own frame hook runs under the traces' wrapper (musicScaleProbe: the autoscale read, 0 under the stub where it was NaN)",
     S.env.run("function(){ return music.autoMax === 0 && /the autoscale would be/.test(musicScaleReadout()) && musicTraces.frames >= 2; }") === true);
  // On the stage the picture hook IS set, so at the defaults the side panels are what is drawn.
  // (In a browser the traces' canvas is on #stage; the stub cannot resolve "#stage", so it is
  // attached there by hand, on the stub's 1280 x 800 window, as the wiring line would.)
  const sd = JSON.parse(S.env.run(`function(){ stage.el.clientWidth = 1280; stage.el.clientHeight = 800; musicTraces.attach(stage.el);
    const c = musicTraces.ctx, R = c.rect, C = c.clip, X = c.restore, K = c.stroke, out = []; let r = null, cl = null;
    c.rect = (...a) => { r = a; return R(...a); }; c.clip = () => { cl = r; return C(); }; c.restore = () => { cl = null; return X(); }; c.stroke = () => { out.push(cl ? cl.join() : "none"); return K(); };
    const nb = solver.nb, p = new Float32Array(3 * nb); for (let b = 1; b < nb; b++) p[b] = p[nb + b] = p[2 * nb + b] = 0.005 * Math.pow(b, -5 / 3);
    musicTraces.spec = [{ t: 9000000, perp: p, nb: nb, bins: tracesBinRange(256, 256, 2 * Math.PI, 2 * Math.PI) }];
    musicTraces.ener = [{ t: 8999000, v: [1, 1, 1, 1] }, { t: 9000000, v: [2, 2, 2, 2] }];
    c.__texts.length = 0; musicTraces.draw(9000000);
    return JSON.stringify({ clips: out, texts: c.__texts.map(t => t.t), o: musicTraces.opts(), pic: musicTraces.picture(), wh: [musicTraces.canvas.width, musicTraces.canvas.height],
      host: musicTraces.canvas.parentNode === stage.el, hook: musicTraces.picture === stagePicture }); }`));
  const sdWant = [];
  for (let i = 0; i < 4; i++) { for (let k = 0; k < 4; k++) sdWant.push([10, 200 * i + 10, 220, 180].join()); for (let k = 0; k < 2; k++) sdWant.push([1050, 200 * i + 10, 220, 180].join()); }
  ok("on the stage AT ITS DEFAULTS (spectrum box off, side panels on: spec | time), with the traces on #stage in the 1280 x 800 window: the square picture leaves strips 240 wide and the eight panels are drawn in them -- per row a spectrum with its guide at x 10..230 and a history at x 1050..1270 -- with their titles, and no bottom band",
     sd.host && sd.hook && sd.wh.join() === "1280,800" && sd.o.spec === false && sd.o.sides === true && sd.o.sideA === "spec" && sd.o.sideB === "time" &&
     sd.pic.x === 240 && sd.pic.w === 800 && sd.pic.tile === false && JSON.stringify(sd.clips) === JSON.stringify(sdWant) &&
     sd.texts.join("|") === "E_u(k)|E_u(t)  max 2|E_b(k)|E_b(t)  max 2|E+(k)|E+(t)  max 2|E−(k)|E−(t)  max 2" && realFails(S.env).length === 0, sd.clips.length + " strokes; " + sd.texts.join("|"));
  // the flag, by booting: what a URL builds
  const builds = async q => { const E = await musicPage(q); const g = E.env.descendants(E.env.getEl("controls"), "details").map(d => d.id);
    return [E.env.run("function(){ return [STAGE_ON, musicTraces.on, !!stage.el, !!musicTraces.canvas, typeof musicTraces.picture === 'function'].join(); }"), g.indexOf("grpTraces") >= 0, E.env.allEls.some(e => e.id === "selFill"), E.env.getEl("selGrid").value].join(); };
  const bPlain = await builds("?plain"), bPT = await builds("?plain&traces"), bStage = await builds("?stage"), bPTf = await builds("?plain&tracesfoo");
  ok("what a URL builds: ?plain is the card page with no stage and no traces; ?plain&traces the card page with the traces (their group, their canvas, no picture hook); ?plain&tracesfoo is ?plain; and ?stage, which means nothing any more, is the bare URL -- the stage, its `sides` select, the traces with the picture hook, grid auto",
     bPlain === "false,false,false,false,false,false,false,standard" && bPT === "false,true,false,true,false,true,false,standard" && bPTf === bPlain &&
     bStage === "true,true,true,true,true,true,true,auto", [bPlain, bPT, bPTf, bStage].join(" | "));
}

// ===========================================================================
// 15. the stage (plans-webgpu/MUSIC_STAGE_PLAN.md S1): stage.js / stage.css, live unless ?plain
// ===========================================================================
// What the stub can say: the pure rules (window -> canvas, the idle fade and the press that
// only wakes, where a status goes, the fullscreen feature test), the STRUCTURE the page
// builds with the flag, and that without the flag none of it exists. What it cannot: that
// stage.css lays any of it out as meant, that a real pointer or touch event reaches the
// listeners in the order the rules assume, fullscreen, or anything about iOS. The stub's
// elements take no events at all, so the idle rows drive stageEvent / stageTick by hand.
async function legStage(state) {
  const src = fs.readFileSync(path.join(dir, "stage.js"), "utf8");
  const ctx = vm.createContext({});
  const msg = throws(() => vm.runInContext(src, ctx, { filename: "stage.js" }));
  ok("stage.js evaluates in a bare context (no DOM, no location, none of the shared scripts): the flag is ON there -- the stage is the default -- and nothing is built by loading it",
     msg === "" && vm.runInContext("STAGE_ON === true && stage.el === null && stage.idle === null && stage.timer === 0", ctx) === true, msg);
  const ctxP = vm.createContext({ location: { search: "?plain" } });
  vm.runInContext(src, ctxP, { filename: "stage.js" });
  const SP = vm.runInContext("({ stageBuild, stageAdopt, stageAspect, stageResize, stageFrame, stagePicture, stageFill })", ctxP);
  const S = vm.runInContext(`({ stageFlag, stageFit, stageTiles, stageFill, stageFrame, stagePicture, stageIdleNew, stageIdleEvent, stageStatusPlace, stageFullscreen,
    stageBuild, stageAdopt, stageAspect, stageResize, STAGE_CAP, STAGE_IDLE_MS, STAGE_GRACE_MS })`, ctx);
  ok("  ... and with ?plain in the URL it is OFF, its entry points no-ops (no DOM here, and none is asked for): stageBuild / stageAdopt / stageResize / stageFrame do nothing, stageAspect() and stagePicture() are null, and the sides are black (stageFill() is \"bars\")",
     vm.runInContext("STAGE_ON", ctxP) === false && throws(() => { SP.stageBuild(); SP.stageAdopt(); SP.stageResize(true); SP.stageFrame(); SP.stageFrame(true); }) === "" && SP.stageAspect() === null &&
     SP.stagePicture() === null && SP.stageFill() === "bars" && vm.runInContext("stage.el === null && stage.tile === null && stage.fit === null && stage.timer === 0", ctxP) === true &&
     /function stageBuild\(\) \{\n  if \(!STAGE_ON \|\| stage\.el\) return;/.test(src) && (src.match(/\bstageListen\(\)/g) || []).length === 2);
  const names = [];
  for (const l of src.split("\n")) {
    const m = /^(?:async\s+)?(?:function\*?|const|let|var|class)\s+([A-Za-z_$][\w$]*)/.exec(l);
    if (m) names.push(m[1]);
  }
  const shared = ["common.js", "physics.js", "solver2d.js", "music.js"].map(f => fs.readFileSync(path.join(dir, f), "utf8")).join("\n");
  const html = fs.readFileSync(path.join(dir, "music.html"), "utf8");
  const clash = names.filter(n => new RegExp("^(?:async\\s+)?(?:function\\*?|const|let|var|class)\\s+" + n + "\\b", "m").test(shared + "\n" + html));
  ok("every top-level name is stage* / STAGE_* (" + names.length + " names), and none is declared by the shared files, music.js or the page",
     names.length >= 20 && names.every(n => /^(stage|STAGE_)/.test(n)) && clash.length === 0,
     names.filter(n => !/^(stage|STAGE_)/.test(n)).concat(clash).join(", "));

  // ---- the flag -------------------------------------------------------------------------
  const SON = ["", undefined, "?musicdebug", "?stage", "?stage=1", "?musicdebug&stage", "?traces", "?plainly", "?x=plain", "?explain", "?plain_2", "?plains&a"];
  const SOFF = ["?plain", "?musicdebug&plain", "?plain=1", "?plain&musicdebug=energy", "?plain&traces", "?traces&plain", "?plain&tracesfoo", "?plain&stage", "?stage&plain"];
  ok("stageFlag: the stage is ON unless the URL carries ?plain / &plain as a whole word -- the bare URL, ?stage (which means nothing any more), ?plainly, ?explain and ?x=plain are on; ?plain, &plain, ?plain=1 are off, whatever else is there (?plain&traces, ?plain&stage)",
     SON.every(q => S.stageFlag(q) === true) && SOFF.every(q => S.stageFlag(q) === false), SON.filter(q => S.stageFlag(q) !== true).concat(SOFF.filter(q => S.stageFlag(q) !== false)).join(" "));
  ok("  ... and the two flags together: ?plain&traces is the card page WITH the traces (stage off, traces on), ?plain&tracesfoo has neither, the bare URL both",
     (tf => [["?plain&traces", false, true], ["?plain&tracesfoo", false, false], ["", true, true], ["?plain", false, false], ["?stage", true, true], ["?traces", true, true]]
       .every(([q, a, b]) => S.stageFlag(q) === a && tf(q) === b))((c => (vm.runInContext(fs.readFileSync(path.join(dir, "traces.js"), "utf8"), c), vm.runInContext("tracesFlag", c)))(vm.createContext({}))));

  // ---- window -> canvas -------------------------------------------------------------------
  // (two rules in one function: COVER, the five-argument form, which the page no longer
  // uses, and CONTAIN, the sixth argument, which is what stageMeasure always passes)
  const fit = (w, h, d, a, c) => S.stageFit(w, h, d, a, c);
  const fitIn = (w, h, d, a, c) => S.stageFit(w, h, d, a, c, true);
  const eq = (f, o) => Object.keys(o).every(k => Math.abs(f[k] - o[k]) < 1e-9);
  ok("stageFit (cover), a landscape window 1280 x 800 at dpr 1: the square picture is 1280 CSS px, cropped 240 top and bottom; backing 1280^2",
     eq(fit(1280, 800, 1), { cssW: 1280, cssH: 1280, left: 0, top: -240, w: 1280, h: 1280 }), JSON.stringify(fit(1280, 800, 1)));
  ok("  ... a portrait phone 390 x 844 at dpr 3: 844 CSS px, cropped 227 left and right; 2532 backing px CAPPED at 1920",
     eq(fit(390, 844, 3), { cssW: 844, cssH: 844, left: -227, top: 0, w: 1920, h: 1920 }), JSON.stringify(fit(390, 844, 3)));
  ok("  ... the backing size follows window x devicePixelRatio below the cap: 800 x 600 at dpr 2 -> 1600, at dpr 1 -> 800, at 1.5 -> 1200",
     fit(800, 600, 2).w === 1600 && fit(800, 600, 1).w === 800 && fit(800, 600, 1.5).w === 1200 && fit(800, 600, 2).cssW === 800);
  ok("  ... the cap is 1920 on the long side: 1920 wide is untouched, 1921 and a 4K window at dpr 2 are 1920",
     S.STAGE_CAP === 1920 && fit(1920, 1080, 1).w === 1920 && fit(1921, 1080, 1).w === 1920 && fit(3840, 2160, 2).w === 1920 &&
     fit(3840, 2160, 2).h === 1920 && fit(3840, 2160, 2).cssW === 3840);
  ok("  ... and it is an argument: cap 1000 -> 1000; an odd size is rounded to an even one (1365 -> 1366)",
     fit(1280, 800, 2, 1, 1000).w === 1000 && fit(1365, 700, 1).w === 1366 && fit(1365, 700, 1).h === 1366);
  // a box that is not square (the later window-shaped box): cover, either way round
  ok("  ... a 2:1 box in a square 1000 window covers by height (2000 x 1000 CSS, 500 off each side; backing 1920 x 960)",
     eq(fit(1000, 1000, 1, 2), { cssW: 2000, cssH: 1000, left: -500, top: 0, w: 1920, h: 960 }), JSON.stringify(fit(1000, 1000, 1, 2)));
  ok("  ... and in a 4:1 window by width (1600 x 800 CSS, 200 off top and bottom); a 1:2 box in a landscape window likewise",
     eq(fit(1600, 400, 1, 2), { cssW: 1600, cssH: 800, left: 0, top: -200, w: 1600, h: 800 }) &&
     eq(fit(1000, 500, 1, 0.5), { cssW: 1000, cssH: 2000, left: 0, top: -750, w: 960, h: 1920 }));
  // CONTAIN (the sixth argument): the whole box inside the window, what the page shows
  ok("stageFit (contain), the same 1280 x 800 window at dpr 1: the square picture is 800 CSS px, 240 in from the left and the right, nothing cropped; backing 800^2",
     eq(fitIn(1280, 800, 1), { cssW: 800, cssH: 800, left: 240, top: 0, w: 800, h: 800 }), JSON.stringify(fitIn(1280, 800, 1)));
  ok("  ... the portrait phone 390 x 844 at dpr 3: 390 CSS px, 227 above and below; backing 1170^2 (390 x 3, under the cap)",
     eq(fitIn(390, 844, 3), { cssW: 390, cssH: 390, left: 0, top: 227, w: 1170, h: 1170 }), JSON.stringify(fitIn(390, 844, 3)));
  ok("  ... a 2:1 box in a square 1000 window fits by width (1000 x 500 CSS, 250 above and below), in a 4:1 window by height (800 x 400, 400 either side); a 1:2 box in a landscape window by height (250 x 500, 375 either side)",
     eq(fitIn(1000, 1000, 1, 2), { cssW: 1000, cssH: 500, left: 0, top: 250, w: 1000, h: 500 }) &&
     eq(fitIn(1600, 400, 1, 2), { cssW: 800, cssH: 400, left: 400, top: 0, w: 800, h: 400 }) &&
     eq(fitIn(1000, 500, 1, 0.5), { cssW: 250, cssH: 500, left: 375, top: 0, w: 250, h: 500 }),
     JSON.stringify([fitIn(1000, 1000, 1, 2), fitIn(1600, 400, 1, 2), fitIn(1000, 500, 1, 0.5)]));
  ok("  ... a box of the window's own shape is the same under both rules: it fills the window exactly, at (0, 0)",
     [[1280, 800, 1, 1.6], [1920, 1080, 2, 16 / 9], [390, 844, 3, 390 / 844], [600, 600, 1, 1]].every(q =>
       JSON.stringify(fitIn(...q)) === JSON.stringify(fit(...q)) && Math.abs(fitIn(...q).cssW - q[0]) < 1e-9 && Math.abs(fitIn(...q).cssH - q[1]) < 1e-9 &&
       Math.abs(fitIn(...q).left) < 1e-9 && Math.abs(fitIn(...q).top) < 1e-9));
  ok("  ... the backing size is the PICTURE's CSS size x devicePixelRatio (800 x 600 at dpr 2 -> 1200, at dpr 1 -> 600), under the same cap on its long side: a 4K window at dpr 2 is 2160 CSS px and 1920 backing, a 2.4 box in 3000 x 1000 is 2400 x 1000 CSS and 1920 x 800",
     fitIn(800, 600, 2).w === 1200 && fitIn(800, 600, 2).h === 1200 && fitIn(800, 600, 1).w === 600 && fitIn(3840, 2160, 2).cssW === 2160 && fitIn(3840, 2160, 2).w === 1920 &&
     fitIn(3840, 2160, 2).h === 1920 && eq(fitIn(3000, 1000, 1, 2.4), { cssW: 2400, cssH: 1000, left: 300, top: 0, w: 1920, h: 800 }) && fitIn(1280, 800, 2, 1, 1000).w === 1000,
     JSON.stringify(fitIn(3000, 1000, 1, 2.4)));
  let sweep = 0, sweepBad = 0, sweepIn = 0, sweepInBad = 0;
  for (const w of [320, 390, 768, 1024, 1366, 1920, 2560, 5120]) for (const h of [480, 844, 1080, 1440, 2880])
    for (const d of [1, 1.25, 2, 3]) for (const a of [0.5, 1, 16 / 9, 2.4]) for (const contain of [false, true]) {
      const f = contain ? fitIn(w, h, d, a) : fit(w, h, d, a);
      const oneExact = Math.abs(f.cssW - w) < 1e-9 || Math.abs(f.cssH - h) < 1e-9;
      // cover: no smaller than the window on either side, offsets <= 0; contain: no larger, offsets >= 0
      const extent = contain ? f.cssW <= w + 1e-9 && f.cssH <= h + 1e-9 && f.left >= -1e-9 && f.top >= -1e-9
                             : f.cssW >= w - 1e-9 && f.cssH >= h - 1e-9 && f.left <= 1e-9 && f.top <= 1e-9;
      const centred = Math.abs(f.left - (w - f.cssW) / 2) < 1e-9 && Math.abs(f.top - (h - f.cssH) / 2) < 1e-9;
      const shape = Math.abs(f.cssW / f.cssH - a) < 1e-9 && Math.abs(f.w / f.h - a) < 2.5 / Math.min(f.w, f.h) * Math.max(a, 1 / a) + 1e-9;
      const size = Math.max(f.w, f.h) <= 1920 && f.w % 2 === 0 && f.h % 2 === 0 && f.w >= 2 && f.h >= 2 &&
                   (Math.max(f.cssW, f.cssH) * d > 1920 ? Math.max(f.w, f.h) === 1920 : Math.abs(f.w - f.cssW * d) <= 1 && Math.abs(f.h - f.cssH * d) <= 1);
      const good = oneExact && extent && centred && shape && size;
      if (contain) { sweepIn++; if (!good) sweepInBad++; } else { sweep++; if (!good) sweepBad++; }
    }
  ok("  ... cover, over " + sweep + " windows, pixel ratios and box shapes: it covers (one side exact, the other no smaller), is centred, keeps the box's shape, and the backing is window x dpr or the cap",
     sweep === 640 && sweepBad === 0, sweepBad + " bad");
  ok("  ... contain, over the same " + sweepIn + ": it fits INSIDE (one side exact, the other no larger, both offsets >= 0), is centred, keeps the box's shape, and the backing is the picture x dpr or the cap",
     sweepIn === 640 && sweepInBad === 0, sweepInBad + " bad");
  ok("  ... bad input never throws or returns nonsense: a 0 / NaN window is 1 x 1 CSS and a 2 x 2 canvas, dpr and aspect fall back to 1 (either rule)",
     [fit(0, 0, 0), fit(NaN, -5, NaN, NaN, NaN), fit(undefined, undefined), fitIn(0, 0, 0), fitIn(NaN, -5, NaN, NaN, NaN), fitIn(undefined, undefined)]
       .every(f => f.cssW === 1 && f.cssH === 1 && f.w === 2 && f.h === 2) &&
     fit(1280, 800, 0).w === 1280 && fit(1280, 800, 1, -1).cssH === 1280 && fitIn(1280, 800, 0).w === 800 && fitIn(1280, 800, 1, -1).cssH === 800);

  // ---- the repeats: which copies of the picture cover the window ------------------------------
  // (pure: [x, y] of each copy of a w x h picture at (left, top) in a W x H window)
  const tiles = (...a) => S.stageTiles(...a);
  const t1 = tiles(1280, 800, 240, 0, 800, 800), t2 = tiles(390, 844, 0, 227, 390, 390);
  ok("stageTiles, the 800 px square centred in 1280 x 800: three copies side by side, at x = -560, 240 (its own place) and 1040; in the portrait phone three stacked, at y = -163, 227, 617",
     JSON.stringify(t1) === "[[-560,0],[240,0],[1040,0]]" && JSON.stringify(t2) === "[[0,-163],[0,227],[0,617]]", JSON.stringify(t1) + " " + JSON.stringify(t2));
  ok("  ... a picture that already fills the window is exactly ONE copy, at its own place; so is one larger than the window",
     JSON.stringify(tiles(1280, 800, 0, 0, 1280, 800)) === "[[0,0]]" && JSON.stringify(tiles(390, 844, 0, 0, 390, 844)) === "[[0,0]]" &&
     JSON.stringify(tiles(1280, 800, -100, -50, 1480, 900)) === "[[-100,-50]]");
  ok("  ... a picture smaller than the window both ways repeats both ways: 200 x 300 at (300, 400) in 1000 x 1000 is 6 columns (x = -100 .. 900) by 4 rows (y = -200 .. 700)",
     (() => { const t = tiles(1000, 1000, 300, 400, 200, 300), xs = Array.from(new Set(t.map(p => p[0]))).sort((a, b) => a - b), ys = Array.from(new Set(t.map(p => p[1]))).sort((a, b) => a - b);
       return t.length === 24 && xs.join() === "-100,100,300,500,700,900" && ys.join() === "-200,100,400,700"; })());
  let tl = 0, tlBad = 0;
  for (const W of [320, 1000, 1280, 1920]) for (const H of [480, 800, 1080]) for (const w of [1, 97, 320, 800, 2000]) for (const h of [1, 131, 480, 3000])
    for (const left of [-w + 1, -37, 0, 17, 240, W - 1]) for (const top of [-h + 1, 0, 113, H - 1]) {
      if (w * h < 2000) continue;                  // (a 1 x 1 picture in a 1920 x 1080 window is two million copies)
      const t = tiles(W, H, left, top, w, h), xs = Array.from(new Set(t.map(p => p[0]))).sort((a, b) => a - b), ys = Array.from(new Set(t.map(p => p[1]))).sort((a, b) => a - b);
      tl++;
      const lattice = t.length === xs.length * ys.length && new Set(t.map(p => p.join())).size === t.length &&
                      xs.every(x => (x - left) % w === 0) && ys.every(y => (y - top) % h === 0);
      const abut = xs.every((x, i) => i === 0 || x - xs[i - 1] === w) && ys.every((y, i) => i === 0 || y - ys[i - 1] === h);     // no gap, no overlap
      const covers = xs.length > 0 && ys.length > 0 && xs[0] <= 0 && xs[xs.length - 1] + w >= W && ys[0] <= 0 && ys[ys.length - 1] + h >= H;
      const needed = t.every(p => p[0] < W && p[0] + w > 0 && p[1] < H && p[1] + h > 0);    // every copy shows in the window: none wasted
      // the picture's own place is one of them exactly when any of it is in the window
      const inWin = left < W && left + w > 0 && top < H && top + h > 0;
      const own = t.some(p => p[0] === left && p[1] === top) === inWin;
      if (!(lattice && abut && covers && needed && own)) tlBad++;
    }
  ok("  ... over " + tl + " windows, picture sizes and offsets: the copies lie on the picture's own lattice and include its own place (whenever any of it is in the window), abut exactly (step = its size: no gap, no overlap), cover the window edge to edge, and none lies wholly outside it",
     tl > 1000 && tlBad === 0, tlBad + " bad");
  ok("  ... degenerate input gives NO copies and never loops: a picture under one pixel, a zero or negative window, NaN",
     [tiles(1280, 800, 0, 0, 0, 800), tiles(1280, 800, 0, 0, 800, 0.5), tiles(0, 800, 0, 0, 800, 800), tiles(1280, -1, 0, 0, 800, 800),
      tiles(1280, 800, 0, 0, NaN, 800), tiles(NaN, 800, 0, 0, 800, 800), tiles()].every(t => Array.isArray(t) && t.length === 0));

  // ---- the idle fade, and the press that only wakes ----------------------------------------
  const run = { running: true }, ev = (st, k, t, o) => S.stageIdleEvent(st, k, t, o === undefined ? run : o);
  let st = S.stageIdleNew(0);
  ok("stageIdle, running: shown at first, still at 2999 ms without activity, hidden at 3000 (STAGE_IDLE_MS)",
     S.STAGE_IDLE_MS === 3000 && st.shown === true && ev(st, "tick", 2999).shown === true && ev(st, "tick", 3000).shown === false);
  ok("  ... any activity brings them back and restarts the 3 s: a move at 5000 shows, 7999 shown, 8000 hidden",
     ev(st, "move", 5000).shown === true && ev(st, "tick", 7999).shown === true && ev(st, "tick", 8000).shown === false);
  st = S.stageIdleNew(0);
  ok("  ... NOT running (before Start, after Stop) they are always shown, however long nothing happens; so are they while held (the mouse rests on them)",
     ev(st, "tick", 1e7, { running: false }).shown === true && ev(st, "tick", 2e7, { running: true, hold: true }).shown === true &&
     ev(st, "tick", 2e7, { running: true }).shown === false && ev(st, "tick", 2e7, { running: false }).shown === true);
  ok("  ... only a tick hides: a state nobody has ticked is still shown, so a press on it is an ordinary press",
     (st = S.stageIdleNew(0), ev(st, "down", 9000).waking === false && ev(st, "click", 9050).swallow === false));
  // the tap that wakes
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  const d1 = ev(st, "down", 5000), u1 = ev(st, "up", 5080), c1 = ev(st, "click", 5085), c1b = ev(st, "click", 5090);
  ok("a press while they are HIDDEN shows them and is swallowed: waking from the press to its click, that click swallowed, the next one not",
     d1.shown === true && d1.waking === true && d1.swallow === false && u1.waking === true && c1.swallow === true && c1.waking === false &&
     c1b.swallow === false, JSON.stringify([d1, u1, c1, c1b]));
  ok("  ... and from then on they are ordinary controls: a press and its click go through, and the 3 s run from the last of them",
     ev(st, "down", 6000).waking === false && ev(st, "click", 6100).swallow === false && ev(st, "tick", 9099).shown === true && ev(st, "tick", 9100).shown === false);
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ok("  ... a mouse that MOVED first has already woken them: its press is not waking and its click is not swallowed",
     ev(st, "move", 5000).shown === true && ev(st, "down", 5200).waking === false && ev(st, "up", 5300).waking === false && ev(st, "click", 5301).swallow === false);
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ev(st, "down", 5000);
  ok("  ... a LONG press that woke them is still swallowed when it is let go 5 s later (no timeout while the press is down)",
     ev(st, "tick", 9000).waking === true && ev(st, "up", 10000).waking === true && ev(st, "click", 10001).swallow === true);
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ev(st, "down", 5000);
  ev(st, "up", 5100);
  ok("  ... a waking press that produces no click (a drag, a key) disarms " + S.STAGE_GRACE_MS + " ms after it is released: waking at +399, not at +400, and a later click goes through",
     S.STAGE_GRACE_MS === 400 && ev(st, "tick", 5499).waking === true && ev(st, "tick", 5500).waking === false && ev(st, "click", 5600).swallow === false);
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ev(st, "down", 5000);
  ev(st, "up", 5050);
  const d2 = ev(st, "down", 5200);
  ok("  ... a second press INSIDE that grace is the same gesture (a double tap): still waking, its click swallowed; one after the grace is not",
     d2.waking === true && ev(st, "up", 5250).waking === true && ev(st, "click", 5251).swallow === true &&
     (st = S.stageIdleNew(0), ev(st, "tick", 4000), ev(st, "down", 5000), ev(st, "up", 5050), ev(st, "down", 5450).waking === false) &&
     ev(st, "click", 5500).swallow === false);
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ev(st, "down", 5000);
  ev(st, "up", 5050);
  ev(st, "down", 5200);
  ok("  ... and that second press, HELD past the grace, is still waking (the grace runs from a release, never under a press) and its click swallowed",
     ev(st, "tick", 5500).waking === true && ev(st, "tick", 9000).waking === true && ev(st, "up", 9100).waking === true && ev(st, "click", 9101).swallow === true);
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ev(st, "down", 5000);
  ok("  ... a waking press whose release never comes (a key held while the window loses focus) stays armed for as long as nothing ends it -- a minute here -- and the \"up\" a blur then delivers ends it: waking 399 ms later, not at 400, and the next click goes through",
     ev(st, "tick", 65000).waking === true && ev(st, "up", 65000).waking === true && ev(st, "tick", 65399).waking === true && ev(st, "tick", 65400).waking === false &&
     ev(st, "click", 65500).swallow === false && /for \(const k of \["pointerup", "pointercancel", "keyup", "blur"\]\)\s*\n\s*w\.addEventListener\(k, \(\) => \{ stageEvent\("up", stageNow\(\)\);/.test(src));
  st = S.stageIdleNew(0);
  ev(st, "tick", 4000);
  ok("  ... while hidden and NOT running a tick shows them again, and a press then is not a wake (Stop from another route, e.g. the capture ending)",
     ev(st, "tick", 4100, { running: false }).shown === true && ev(st, "down", 4200, { running: false }).waking === false);

  // ---- where a status goes; the fullscreen feature test --------------------------------------
  ok("stageStatusPlace: nothing for an empty or unstyled status; the drawer for info and for an error with a live solver; the CENTRE for an error with no solver, a lost device or a WebGPU error",
     S.stageStatusPlace("info", "", true) === "none" && S.stageStatusPlace("", "x", true) === "none" && S.stageStatusPlace("err", "", false) === "none" &&
     S.stageStatusPlace("info", "choose an audio file first", true) === "drawer" && S.stageStatusPlace("info", "x", false) === "drawer" &&
     S.stageStatusPlace("err", "could not start: x", true) === "drawer" && S.stageStatusPlace("err", "Error: x", true) === "drawer" &&
     S.stageStatusPlace("err", "WebGPU is not available in this browser. ", false) === "centre" &&
     S.stageStatusPlace("err", "WebGPU device lost: x", true) === "centre" && S.stageStatusPlace("err", "WebGPU error: x", true) === "centre");
  const calls = [];
  const fsDoc = o => Object.assign({ fullscreenElement: null }, o), fsEl = o => Object.assign({}, o);
  const std = S.stageFullscreen(fsDoc({ fullscreenEnabled: true, exitFullscreen() { calls.push("exit"); this.fullscreenElement = null; } }),
                                fsEl({ requestFullscreen() { calls.push("req"); } }));
  const dW = fsDoc({ webkitFullscreenEnabled: true, webkitFullscreenElement: null, webkitExitFullscreen() { calls.push("wexit"); } });
  const wk = S.stageFullscreen(dW, fsEl({ webkitRequestFullscreen() { calls.push("wreq"); dW.webkitFullscreenElement = {}; } }));
  ok("stageFullscreen: null with no API at all (iPhone Safari), with it disabled, or with no way out; the standard API, or the webkit-prefixed one, otherwise",
     S.stageFullscreen({}, {}) === null && S.stageFullscreen(null, null) === null &&
     S.stageFullscreen(fsDoc({ fullscreenEnabled: false, exitFullscreen() {} }), fsEl({ requestFullscreen() {} })) === null &&
     S.stageFullscreen(fsDoc({ fullscreenEnabled: true }), fsEl({ requestFullscreen() {} })) === null &&
     S.stageFullscreen(fsDoc({ webkitFullscreenEnabled: false, webkitExitFullscreen() {} }), fsEl({ webkitRequestFullscreen() {} })) === null &&
     !!std && std.event === "fullscreenchange" && !!wk && wk.event === "webkitfullscreenchange");
  ok("  ... toggle() enters when not fullscreen and leaves when it is (prefixed: wreq then wexit)",
     (std.toggle(), wk.toggle(), wk.on() === true) && (wk.toggle(), calls.join() === "req,wreq,wexit") && std.on() === false, calls.join());

  // ---- stage.css and the page's markup, as text -------------------------------------------------
  const css = fs.readFileSync(path.join(dir, "stage.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
  const sels = [];
  for (const m of css.replace(/@media[^{]*\{/g, "").matchAll(/([^{}]+)\{[^{}]*\}/g))
    for (const s of m[1].split(",")) if (s.trim()) sels.push(s.trim());
  const loose = sels.filter(s => !/^(html\.stage\b|#stage)/.test(s));
  ok("stage.css: every one of its " + sels.length + " selectors starts at html.stage or at an element stage.js creates (#stage...), so it matches nothing with ?plain (where neither exists)",
     sels.length >= 25 && loose.length === 0, loose.join(" | "));
  ok("  ... #stage is position: fixed; inset: 0, 100dvh with a 100vh fallback BEFORE it; the picture, the repeated picture (#stageTile, the whole window), canvas.traces, the controls, the centred error stack 0 < 1 < 2 < 3 < 4",
     /#stage \{[^}]*position: fixed; inset: 0; height: 100vh; height: 100dvh;/.test(css) &&
     /#stage > \.cvwrap \{[^}]*z-index: 0;/.test(css) && /#stageTile \{ position: absolute; z-index: 1; left: 0; top: 0; width: 100%; height: 100%; \}/.test(css) &&
     /#stage > canvas\.traces \{ z-index: 2; \}/.test(css) && /#stageUi \{[^}]*z-index: 3;/.test(css) && /#stageFatal \{[^}]*z-index: 4;/.test(css) &&
     (css.match(/z-index:/g) || []).length === 5, (css.match(/z-index: \d+/g) || []).join(", "));
  ok("  ... faded or waking, the bar and the drawer take no pointer events",
     /#stage\.idle #stageBar, #stage\.idle #stageDrawer,\s*#stage\.waking #stageBar, #stage\.waking #stageDrawer \{ pointer-events: none; \}/.test(css) &&
     /#stage\.idle #stageUi \{ opacity: 0; \}/.test(css));
  // (a line of the stage: it loads one of its files, names one of its functions -- called or
  // handed over -- or carries its mark)
  const marked = html.split("\n").filter(l => /stage\.(js|css)|\bstage[A-Z]\w*\b|STAGE:/.test(l));
  const mk = re => marked.findIndex(l => re.test(l));
  const order = [/<link rel="stylesheet" href="stage\.css">/, /<script src="stage\.js"[^>]*><\/script>/,
                 /^\s*\/\/ STAGE: what the window shows beside a box of another shape/,                   // the `sides` select (selFill)
                 /^stageBuild\(\);/, /^function cardAspect\(\) \{ return stageAspect\(\) \|\| musicCardSize\(\); \}/,
                 /^  if \(typeof stage !== "undefined" && stage\.tile\) stage\.tile\.style\.transform = t;/,   // the zoom pulse reaches the repeats
                 /^  stageAdopt\(\);/,
                 /^\s*\/\/ STAGE: the picture need not span the window/,                                    // the block that follows
                 /^    musicTraces\.picture = stagePicture;/, /^    renderCards = function \(a\) \{ const r = rc\(a\); stageFrame\(r > 0\); return r; \};/].map(mk);
  ok("music.html: every line that names the stage is marked STAGE: -- the stylesheet, the script, and eight wiring lines (the sides select, stageBuild, cardAspect, the zoom on the repeats, stageAdopt, and the block that hands the traces stagePicture and wraps renderCards with stageFrame(r > 0): whether a card rendered in that task)",
     marked.length === 10 && marked.every(l => /STAGE:/.test(l)) && order.every((i, k) => i === k),
     marked.length + " lines; unmarked: " + (marked.filter(l => !/STAGE:/.test(l)).map(l => l.trim().slice(0, 60)).join(" | ") || "none") + "; order " + order.join());
  // (every tag, attributes or not: leg 7 pins the same list from the bare tags)
  const all = Array.from(html.matchAll(/<script[^>]*\ssrc="([^"]+)"/g)).map(m => m[1]);
  ok("  ... its scripts, every tag counted: common.js, physics.js, solver2d.js, music.js, stage.js, traces.js -- stage.js after music.js, before the page's own",
     all.join() === "common.js,physics.js,solver2d.js,music.js,stage.js,traces.js" && html.indexOf('src="stage.js"') < html.indexOf("<script>\n"), all.join());
  const others = fs.readdirSync(dir).filter(f => /\.(html|js|css)$/.test(f) && f !== "music.html" && f !== "stage.js" && f !== "stage.css");
  const users = others.filter(f => /stage\.(js|css)\b|\bstage(Build|Adopt|Aspect|Resize|Frame|Picture|Fill|Tiles|Measure)\b/.test(fs.readFileSync(path.join(dir, f), "utf8")));
  ok("no other page or shared file loads stage.js / stage.css or calls into it (" + others.length + " files)", others.length >= 9 && users.length === 0, users.join(", "));

  // ---- WITH ?plain: nothing of it exists - ---------------------------------------------------
  const N = await musicPage("?plain&musicdebug"), nenv = N.env;
  const nq = JSON.parse(nenv.run(`function(){ const c = cards.disp[0]; return JSON.stringify({ on: STAGE_ON, el: stage.el, idle: stage.idle, timer: stage.timer, back: stage.back,
    asp: cardAspect(), cv: [c.cv.width, c.cv.height, c.cvVec.width], wrapIn: c.wrap.parentNode === c.root, charts: cards.chart.map(x => x.type()),
    run: el("btnRun").parentNode.id, ctl: el("grpMusic").parentNode.id, arrow: c.cbArrow.parentNode.style.display, off: !!c.rOffX && c.rOffX.style.display,
    add: el("btnAddDisp").style.display, grpOpen: [el("grpMusic").open, el("grpSim").open] }); }`));
  ok("with ?plain: STAGE_ON is false, no element with a stage id was created, no timer and no idle state",
     nq.on === false && nq.el === null && nq.idle === null && nq.timer === 0 && nq.back === null &&
     nenv.allEls.filter(e => /^stage/i.test(e.id || "")).length === 0, nenv.allEls.filter(e => /^stage/i.test(e.id || "")).map(e => e.id).join());
  ok("  ... no `sides` select (selFill) and no #stageTile either, the traces are told nothing about where the picture is (musicTraces.picture is null) and renderCards is the shared one, unwrapped",
     nenv.allEls.filter(e => e.id === "selFill" || e.id === "stageTile").length === 0 &&
     nenv.run("function(){ return musicTraces.picture === null && stage.tile === null && stage.fit === null && String(renderCards).indexOf('stageFrame') < 0 && stagePicture() === null; }") === true);
  ok("  ... the card is the shipped one: 512 x 512 (cardAspect and both canvases), its canvas inside the card, arrows / offsets / + display in place, both charts",
     nq.asp.w === 512 && nq.asp.h === 512 && nq.cv.join() === "512,512,512" && nq.wrapIn === true && nq.charts.join() === "energy,spectrum" &&
     nq.arrow === undefined && nq.off === undefined && nq.add === undefined, JSON.stringify(nq.cv));
  ok("  ... and the controls are where they were: the Start button in #topbar, the groups in #controls, open as groupsInit left them",
     nq.run === "topbar" && nq.ctl === "controls" && nq.grpOpen.join() === "true,true" && nenv.getEl("controls").parentNode === null);
  ok("  ... stageResize / stageAdopt / stageBuild called there change nothing",
     nenv.run(`function(){ stageBuild(); stageAdopt(); stageResize(true); const c = cards.disp[0];
       return stage.el === null && c.cv.width === 512 && cards.chart.length === 2 && c.wrap.parentNode === c.root; }`) === true && realFails(nenv).length === 0);

  // ---- THE BARE URL: the stage----------------------------------------------------------------------
  const P = await musicPage("?musicdebug"), env = P.env;
  ok("the bare URL (the stage, since 2026-11-24 the page's default layout) boots under the stub with no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 3).join(" | "));
  // (2026-11-23, later the same day: on the stage the box is SQUARE by default again, as it is
  // without the flag, and the stage shows it WHOLE -- contain, never a crop -- with the two
  // sides left over black or carrying the picture again. Every row below is about the shell
  // and about a square picture in a window; the window-shaped box is leg 16's subject.)
  // (one frame, and the shell's 250 ms tick -- fired by hand, the stub has no timers running)
  await P.frame();
  env.run("function(){ stageTick(stageNow()); }");
  const up = (e, id) => { for (let p = e; p; p = p.parentNode) if (p.id === id) return true; return false; };
  const sEl = env.getEl("stage"), kid = id => sEl.children.filter(c => c.id === id).length;
  const q = JSON.parse(env.run(`function(){ const c = cards.disp[0], h = c.selField.parentNode; return JSON.stringify({ on: STAGE_ON,
    kids: stage.el.children.map(x => x.id || x.className), ui: stage.ui.children.map(x => x.id),
    wrapIn: c.wrap.parentNode === stage.el, cvIn: c.cv.parentNode === c.wrap, charts: cards.chart.length, chartEls: cards.hostC.children.length,
    cv: [c.cv.width, c.cv.height], vec: [c.cvVec.width, c.cvVec.height], css: c.wrap.style, asp: cardAspect(),
    headIn: h.parentNode === stage.slotPic, footIn: c.foot.parentNode === stage.slotPic, head: h.className,
    arrow: [c.cbArrow.checked, c.cbArrow.parentNode.style.display], off: [c.rOffX.value, c.rOffX.style.display, c.rOffX.lab.style.display, c.rOffY.style.display, c.offset()],
    close: c.btnClose.style.display, filt: el("cbFilter").parentNode.style.display,
    sel: c.selField.value, cmap: c.selCmap.value === String(MUSIC_CMAP_DEFAULT), fs: [stage.fs, stage.btnFs.style.display], menu: stage.btnMenu.textContent,
    open: stageDrawerIsOpen(), p: [solver.p.nx, solver.p.ny, solver.p.Lx === solver.p.Ly], hook: stepHook === musicStep, timer: stage.timer > 0,
    box: el("selBox").value, tile: [stage.tile.kind, stage.tile.parentNode === stage.el, stage.tile.style.display],
    wired: musicTraces.picture === stagePicture && String(renderCards).indexOf("stageFrame") >= 0 }); }`));
  // (the ORDER of stacking is stage.css's z-indexes, pinned above as text; this is the DOM)
  ok("#stage exists and holds four children: #stageTile (a canvas for the repeated picture, made first, hidden), the controls column #stageUi (#stageBar, #stageDrawer), #stageFatal, and the adopted picture (the card's .cvwrap, its WebGPU canvas inside)",
     q.on === true && kid("stageTile") === 1 && kid("stageUi") === 1 && kid("stageFatal") === 1 && q.kids.join() === "stageTile,stageUi,stageFatal,cvwrap" && q.ui.join() === "stageBar,stageDrawer" &&
     q.wrapIn === true && q.cvIn === true && sEl.children.length === 4 && q.tile.join() === "canvas,true,none", q.kids.join() + " | tile " + q.tile.join());
  ok("  ... and the page's two hooks into it are in place: the traces ask stagePicture where the picture is, and renderCards is wrapped to call stageFrame", q.wired === true);
  const bar = ["btnRun", "btnReset", "selSrc", "stageBtnMenu", "stageBtnFs"];
  ok("the bar holds Start, Reset, the source select, the drawer button and the fullscreen button -- the page's #topbar, moved whole, plus the stage's two",
     bar.every(id => up(env.getEl(id), "stageBar") && !up(env.getEl(id), "stageDrawer")) && env.getEl("topbar").parentNode === env.getEl("stageBar") &&
     env.getEl("btnRun").textContent === "Start" && q.menu === "controls");
  ok("  ... the fullscreen button is HIDDEN here: the stub's document has no Fullscreen API, which is the iPhone Safari case",
     q.fs[0] === null && q.fs[1] === "none");
  const groups = env.descendants(env.getEl("controls"), "details");
  // (the file row is #controls markup, so it rides along in a browser; the stub does not
  // model markup nesting and cannot show it)
  ok("ALL the control groups are in the drawer (" + groups.map(g => g.id).join(", ") + "), inside #controls, which moved whole",
     groups.map(g => g.id).join() === "grpMusic,grpFx,grpSim,grpDisp,grpTraces" && groups.every(g => up(g, "stageDrawer") && up(g, "controls")) &&
     up(env.getEl("controls"), "stageDrawer") && ["rThr", "selKick", "rCfl", "btnSaveAll", "cbFixScale", "cbTrSpec"].every(id => up(env.getEl(id), "stageDrawer")));
  env.run(`function(){ controlsBuild({ topbar: [{ k: "btn", id: "btnLate", t: "late" }], groups: [{ id: "grpLate", summary: "added later", rows: [[{ k: "lab", t: "x" }, { k: "rng", id: "rLate", min: 0, max: 1, step: 0.1, v: 0 }]] }] }); }`);
  ok("  ... and a group someone adds LATER through controlsBuild lands in the drawer (its top-bar item in the bar) with no wiring",
     up(env.getEl("grpLate"), "stageDrawer") && up(env.getEl("rLate"), "stageDrawer") && up(env.getEl("btnLate"), "stageBar"));
  ok("the status, sync and readout lines are in the drawer: #status, the sync line (#musicMode), the band strip, #steps, #readout, the build line",
     ["status", "musicMode", "musicStrip", "steps", "readout", "buildline"].every(id => up(env.getEl(id), "stageDrawer")) &&
     env.getEl("status").parentNode.id === "stageStatus");
  const links = env.allEls.filter(e => e.kind === "a" && up(e, "stageDrawer")).map(e => e.href);
  ok("a \"what is this\" section in the drawer links to docs.html and rmhd2d.html", links.join() === "docs.html,rmhd2d.html" &&
     env.allEls.filter(e => e.kind === "a" && up(e, "stageAbout")).length === 2, links.join());
  ok("the chart cards are ABSENT: no energy or spectrum card object, none in the chart column",
     q.charts === 0 && q.chartEls === 0 && env.allEls.filter(e => /card chart/.test(e.className) && up(e, "charts")).length === 0, q.charts + " cards, " + q.chartEls + " elements");
  ok("the picture's options are in the drawer: the card's header (field, colormap) and footer (caption, colour bar, save / rec), |u| in magma as on the card page",
     q.headIn && q.footIn && q.head === "cardhead" && q.sel === "4" && q.cmap === true);
  ok("  ... arrows and the per-card offset sliders are NOT shown (arrows off, offsets zero and hidden with their labels), nor close, + display, + chart, rec all, the k filter",
     q.arrow[0] === false && q.arrow[1] === "none" && q.off[0] === "0" && q.off[1] === "none" && q.off[2] === "none" && q.off[3] === "none" &&
     q.off[4].join() === "0,0" && q.close === "none" && q.filt === "none" &&
     /html\.stage #btnAddDisp, html\.stage #btnAddChart, html\.stage #btnRecAll \{ display: none !important; \}/.test(css), JSON.stringify([q.arrow, q.off]));
  const sv0 = env.run("function(){ globalThis.__sv0 = solver; return solver.modeOf(0); }");
  env.run(`function(){ const c = cards.disp[0]; c.selField.value = "0"; c.selField.onchange({}); }`);
  ok("  ... and they still drive the picture from there: the field select changes the solver's display mode",
     env.run("function(){ return solver.modeOf(0); }") !== sv0 && (env.run(`function(){ const c = cards.disp[0]; c.selField.value = "4"; c.selField.onchange({}); return solver.modeOf(0); }`) === sv0));
  ok("in the drawer the groups start folded, all but the first", env.run(`function(){ return [el("grpMusic").open, el("grpFx").open, el("grpSim").open, el("grpDisp").open].join(); }`) === "true,false,false,false");
  ok("the drawer starts closed; its button opens and closes it and says so (aria-expanded)",
     q.open === false && (env.getEl("stageBtnMenu").onclick({}), env.run("function(){ return stageDrawerIsOpen(); }") === true) &&
     env.getEl("stageBtnMenu").attrs["aria-expanded"] === "true" && env.getEl("stageDrawer").classList.contains("open") &&
     (env.getEl("stageBtnMenu").onclick({}), env.run("function(){ return stageDrawerIsOpen(); }") === false) && env.getEl("stageBtnMenu").attrs["aria-expanded"] === "false");
  ok("the solver is the shipped one -- 256^2 in a square box, the box select's own default here too -- with the music step hook installed, and the stage runs a timer",
     q.p.join() === "256,256,true" && q.box === "square" && q.hook === true && q.timer === true);

  // ---- the canvas follows the window ------------------------------------------------------------
  // (the stub's window is 1280 x 800 at dpr 1, and its elements report no client size, so the
  // stage reads window.innerWidth / innerHeight -- the fallback a browser never needs)
  ok("the square picture is shown WHOLE: the canvas backing store is the window's SHORT side squared, 800 x 800 at the stub's 1280 x 800 dpr 1, in an 800 px CSS box centred in the window -- 240 px in from its left edge, at its top",
     q.cv.join() === "800,800" && q.asp.w === 800 && q.asp.h === 800 && q.css.left === "240.00px" && q.css.top === "0.00px" &&
     q.css.width === "800.00px" && q.css.height === "800.00px", q.cv.join("x") + " " + JSON.stringify(q.css));
  ok("  ... and the unused arrow overlay is kept at 1 x 1 (the shared code would make it canvas x dpr)", q.vec.join() === "1,1", q.vec.join("x"));
  const win = (w, h, d, now) => JSON.parse(env.run(`function(w, h, d, now){ window.innerWidth = w; window.innerHeight = h; window.devicePixelRatio = d;
    stageResize(now); const c = cards.disp[0];
    return JSON.stringify({ cv: [c.cv.width, c.cv.height], css: [c.wrap.style.left, c.wrap.style.top, c.wrap.style.width, c.wrap.style.height],
      vec: c.cvVec.width, same: solver === __sv0, grave: graveyard.length, n: [solver.p.nx, solver.p.ny], pending: stage.backTimer !== 0, dirty: c.dirty }); }`, w, h, d, now));
  let r = win(390, 844, 3, true);
  ok("a portrait phone, 390 x 844 at dpr 3: the picture is the window's WIDTH, a 390 px CSS box 227 px down from the top (bars above and below), on a 1170 x 1170 canvas (390 x 3, under the cap)",
     r.cv.join() === "1170,1170" && r.css.join() === "0.00px,227.00px,390.00px,390.00px" && r.vec === 1, r.cv.join("x") + " " + r.css.join(" "));
  r = win(800, 600, 2, true);
  ok("800 x 600 at dpr 2: the canvas is 1200 x 1200 (the picture's 600 CSS px x devicePixelRatio), the CSS box 600 px, 100 in from the left", r.cv.join() === "1200,1200" && r.css.join() === "100.00px,0.00px,600.00px,600.00px", r.cv.join("x") + " " + r.css.join(" "));
  r = win(800, 600, 1, true);
  ok("  ... the same window at dpr 1 (dragged to another screen): 600 x 600, the CSS box unchanged, the card marked for a redraw", r.cv.join() === "600,600" && r.css.join() === "100.00px,0.00px,600.00px,600.00px" && r.dirty === true, r.cv.join("x"));
  r = win(3840, 2160, 2, true);
  ok("a 4K window at dpr 2: the picture is 2160 CSS px (840 in from the left), 4320 px wanted, the canvas CAPPED at 1920 x 1920", r.cv.join() === "1920,1920" && r.css.join() === "840.00px,0.00px,2160.00px,2160.00px", r.cv.join("x") + " " + r.css.join(" "));
  ok("through every one of those the SOLVER is the same object, still 256 x 256, and nothing was retired: a resize re-fits the canvas and rebuilds nothing",
     r.same === true && r.grave === 0 && r.n.join() === "256,256");
  r = win(1024, 768, 1, false);
  ok("a resize EVENT (not forced) moves the CSS box at once and leaves the backing store for " + env.run("function(){ return STAGE_RESIZE_MS; }") + " ms, so a window being dragged does not flash",
     r.cv.join() === "1920,1920" && r.css.join() === "128.00px,0.00px,768.00px,768.00px" && r.pending === true, r.cv.join("x") + " " + r.css.join(" "));
  await new Promise(res => setTimeout(res, 200));
  r = JSON.parse(env.run("function(){ const c = cards.disp[0]; return JSON.stringify({ cv: [c.cv.width, c.cv.height], pending: stage.backTimer !== 0, same: solver === __sv0, vec: c.cvVec.width }); }"));
  ok("  ... after which the canvas follows by itself: 768 x 768 (the 1024 x 768 window's short side), same solver", r.cv.join() === "768,768" && r.pending === false && r.same === true && r.vec === 1, r.cv.join("x"));
  r = win(1024, 768, 1, false);
  ok("  ... and a resize event that changes nothing arms nothing", r.pending === false && r.cv.join() === "768,768", r.cv.join("x") + " pending " + r.pending);
  // any re-apply of the cards asks for the window as it is NOW (a rebuild with another box)
  env.run("function(){ window.innerWidth = 640; window.innerHeight = 480; cardsSync(); }");
  ok("any cardsSync (a control moved, a rebuild) sizes the canvas for the window as it is then: 480 x 480 in 640 x 480, 80 in from the left",
     env.run("function(){ const c = cards.disp[0]; return [c.cv.width, c.cv.height, c.wrap.style.width, c.wrap.style.left].join(); }") === "480,480,480.00px,80.00px",
     env.run("function(){ const c = cards.disp[0]; return [c.cv.width, c.cv.height, c.wrap.style.width, c.wrap.style.left].join(); }"));
  env.run("function(){ window.innerWidth = 1280; window.innerHeight = 800; stageResize(true); }");

  // ---- the sides: black, or the picture again --------------------------------------------------
  // (`sides`, selFill, exists only on the stage.) On "tile" stageFrame -- called by the wrapped
  // renderCards, once a frame, and told whether a card rendered -- copies the picture's canvas over the whole window onto
  // #stageTile with drawImage; the stub's 2D context logs each drawImage's source and
  // destination. What the stub cannot say: that a WebGPU canvas is readable at that moment,
  // or what any of it looks like.)
  const tile = env.getEl("stageTile"), selFill = env.getEl("selFill");
  const drawn = () => (tile.__cx2d ? tile.__cx2d.__draws.map(d => d.at.join()) : null);
  const fq = () => JSON.parse(env.run(`function(){ return JSON.stringify({ fill: stageFill(), pic: stagePicture(), disp: stage.tile.style.display,
    wh: [stage.tile.width, stage.tile.height], src: !stage.tileCtx || stage.tileCtx.__draws.every(d => d.src === cards.disp[0].cv) }); }`));
  // the copies wanted, by this file's own arithmetic: the window x dpr scaled down to the cap,
  // the picture (the window's short side, squared, centred) scaled with it, repeated to cover
  const wantTiles = (W, H, dpr) => {
    const long = Math.max(W, H) * dpr, sc = dpr * (long > 1920 ? 1920 / long : 1), side = Math.min(W, H);
    const tw = Math.round(W * sc), th = Math.round(H * sc), w = Math.round(side * sc), x0 = Math.round((W - side) / 2 * sc), y0 = Math.round((H - side) / 2 * sc), out = [];
    for (let y = y0 - Math.ceil(y0 / w) * w; y < th; y += w) for (let x = x0 - Math.ceil(x0 / w) * w; x < tw; x += w) out.push([x, y, w, w].join());
    return { wh: [tw, th].join(), at: out };
  };
  const cardQ = () => JSON.parse(env.run("function(){ const c = cards.disp[0]; return JSON.stringify({ dirty: c.dirty, need: c.needsRender(), key: stage.tileKey, running: running }); }"));
  await P.frame();
  await P.frame();
  let f0 = fq();
  ok("the `sides` select exists on the stage, in the simulation group in the drawer: black (\"bars\", the default) or repeat the picture (\"tile\")",
     selFill.options.map(o => o.value).join() === "bars,tile" && selFill.value === "bars" && up(selFill, "grpSim") && up(selFill, "stageDrawer") && f0.fill === "bars",
     selFill.options.map(o => o.value).join() + " = " + selFill.value);
  ok("  ... on black, frames come and go and #stageTile stays hidden and untouched: no 2D context is ever asked of it",
     f0.disp === "none" && drawn() === null && env.run("function(){ return stage.tileCtx; }") === null && cardQ().key === "", f0.disp + " " + JSON.stringify(drawn()));
  ok("  ... stagePicture() is the picture's CSS box as laid out -- x = 240, w = 800, y = 0, h = 800 in the 1280 x 800 window -- and tile: false",
     JSON.stringify(f0.pic) === JSON.stringify({ x: 240, w: 800, y: 0, h: 800, tile: false }) &&
     env.run("function(){ const p = stagePicture(), f = stage.fit; return p.x === f.left && p.w === f.cssW && p.y === f.top && p.h === f.cssH && cards.disp[0].wrap.style.left === f.left.toFixed(2) + 'px'; }") === true,
     JSON.stringify(f0.pic));
  // The page is PAUSED here, and a WebGPU canvas can only be read in the task that rendered
  // it (anywhere else it reads back blank -- seen in Chrome). So stageFrame(drawn) copies
  // only on a frame whose card rendered, and where the copies on show are not the current
  // ones it marks the card dirty, which makes the NEXT frame render and copy.
  const tl_pre = cardQ();
  await P.input("selFill", "tile");
  await P.frame();
  const tl_a1 = fq(), tl_q1 = cardQ(), tl_d1 = drawn().slice();
  const tl_clears = [], tl_clear0 = tile.__cx2d.clearRect;
  tile.__cx2d.clearRect = (...a) => { tl_clears.push(a.join()); return tl_clear0(...a); };
  await P.frame();
  f0 = fq();
  const tl_q2 = cardQ(), tl_d2 = drawn().slice();
  const w0 = wantTiles(1280, 800, 1);
  ok("on \"repeat the picture\", on a PAUSED page: the first frame has no render to copy (the card was not dirty), so it shows #stageTile, sizes it for the window (1280 x 800 at dpr 1), draws NOTHING on it -- and marks the card dirty",
     tl_pre.running === false && tl_pre.dirty === false && tl_pre.need === false && tl_a1.fill === "tile" && tl_a1.disp === "block" && tl_a1.wh.join() === "1280,800" && tl_d1.length === 0 && tl_q1.dirty === true && tl_q1.need === true && tl_q1.key === "",
     tl_a1.disp + " " + tl_a1.wh.join("x") + ", " + tl_d1.length + " copies, dirty " + tl_q1.dirty);
  ok("  ... so the following frame renders, and THAT one copies: the canvas cleared once, then the picture's canvas drawn three times, at x = -560, 240 (the picture's own place) and 1040, each 800 x 800",
     w0.wh === "1280,800" && w0.at.join(" ") === "-560,0,800,800 240,0,800,800 1040,0,800,800" && JSON.stringify(tl_d2) === JSON.stringify(w0.at) && f0.src === true &&
     tl_clears.join(" ") === "0,0,1280,800" && tl_q2.dirty === false && tl_q2.key === "1280,800,800,800,240,0", JSON.stringify(tl_d2) + " cleared " + tl_clears.join(" "));
  ok("  ... and stagePicture() says so (tile: true, the same box), which is what repeats the line cut with it and keeps the side panels off", f0.pic.x === 240 && f0.pic.w === 800 && f0.pic.y === 0 && f0.pic.h === 800 && f0.pic.tile === true, JSON.stringify(f0.pic));
  tile.__cx2d.__draws.length = 0;
  await P.frame();
  await P.frame();
  await P.frame();
  const tl_q3 = cardQ();
  ok("  ... after which the paused page is STEADY: the copies on show are the current ones, so further frames draw nothing, clear nothing and dirty nothing",
     drawn().length === 0 && tl_clears.length === 1 && tl_q3.dirty === false && tl_q3.need === false && tl_q3.key === tl_q2.key, drawn().length + " copies, dirty " + tl_q3.dirty);
  // a resize that leaves the picture's own canvas alone (the window gets wider: still 800 x 800)
  env.run("function(){ window.innerWidth = 1400; stageResize(true); }");
  const tl_r0 = cardQ(), tl_cvSame = env.run("function(){ return cards.disp[0].cv.width + 'x' + cards.disp[0].cv.height; }");
  await P.frame();
  const tl_r1 = cardQ(), tl_rd1 = drawn().slice();
  await P.frame();
  const tl_r2 = cardQ(), tl_rd2 = drawn().slice(), tl_w14 = wantTiles(1400, 800, 1);
  ok("a resize while paused re-dirties: the window widened to 1400 (the picture's canvas untouched, " + tl_cvSame + ", and the card not dirty from the resize itself), the next frame copies nothing and marks the card, and the one after tl_clears and copies for the new window -- x = -500, 300, 1100 on 1400 x 800",
     tl_cvSame === "800x800" && tl_r0.dirty === false && tl_rd1.length === 0 && tl_r1.dirty === true && tl_w14.at.join(" ") === "-500,0,800,800 300,0,800,800 1100,0,800,800" &&
     JSON.stringify(tl_rd2) === JSON.stringify(tl_w14.at) && tl_r2.dirty === false && tl_r2.key === "1400,800,800,800,300,0" && tl_clears.join(" ") === "0,0,1280,800 0,0,1400,800" && fq().wh.join() === "1400,800",
     tl_rd1.length + " then " + tl_rd2.join(" ") + "; cleared " + tl_clears.join(" "));
  tile.__cx2d.__draws.length = 0;
  env.run("function(){ stageFrame(true); stageFrame(true); }");
  ok("  ... while the picture keeps rendering (stageFrame(true), twice by hand: a running page) every frame copies afresh over the last, with no clear and no dirtying of its own",
     drawn().length === 6 && JSON.stringify(drawn().slice(3)) === JSON.stringify(tl_w14.at) && tl_clears.length === 2 && cardQ().dirty === false, drawn().length + " copies, " + tl_clears.length + " tl_clears");
  // (a window whose change also resizes the picture's canvas dirties the card by itself --
  // cardsSync -- so the copies arrive one frame sooner; two frames cover both, and must
  // leave exactly one set)
  const tileAt = async (W, H, dpr) => { win(W, H, dpr, true); tile.__cx2d.__draws.length = 0; await P.frame(); await P.frame(); const o = fq(); o.at = drawn().slice(); o.want = wantTiles(W, H, dpr); o.q = cardQ(); return o; };
  let ta = await tileAt(800, 600, 2), tb = await tileAt(390, 844, 3), tc = await tileAt(3840, 2160, 2);
  ok("  ... its backing store follows the window x devicePixelRatio under the picture's own cap: 800 x 600 at dpr 2 is 1600 x 1200 with three 1200 px copies (x = -1000, 200, 1400); 4K at dpr 2 is 1920 x 1080 with three 1080 px copies (x = -660, 420, 1500) -- one set each, within two frames",
     ta.wh.join() === "1600,1200" && ta.at.join(" ") === "-1000,0,1200,1200 200,0,1200,1200 1400,0,1200,1200" && JSON.stringify(ta.at) === JSON.stringify(ta.want.at) &&
     tc.wh.join() === "1920,1080" && tc.at.join(" ") === "-660,0,1080,1080 420,0,1080,1080 1500,0,1080,1080" && JSON.stringify(tc.at) === JSON.stringify(tc.want.at) && ta.src && tc.src &&
     ta.q.dirty === false && tc.q.dirty === false && tc.q.key === "1920,1080,1080,1080,420,0",
     ta.wh.join("x") + " " + ta.at.join(" ") + " | " + tc.wh.join("x") + " " + tc.at.join(" "));
  ok("  ... and in the portrait phone (390 x 844 at dpr 3, capped: 887 x 1920) the copies are STACKED, three of them 887 px square, the middle one at the picture's own place",
     tb.wh.join() === tb.want.wh && tb.wh.join() === "887,1920" && tb.at.length === 3 && JSON.stringify(tb.at) === JSON.stringify(tb.want.at) &&
     tb.at.join(" ") === "0,-371,887,887 0,516,887,887 0,1403,887,887" && tb.pic.tile === true && tb.pic.x === 0 && tb.pic.w === 390 && tb.pic.y === 227 && tb.pic.h === 390, tb.wh.join("x") + " " + tb.at.join(" "));
  win(1280, 800, 1, true);
  await P.frame();
  await P.frame();
  await P.input("selFill", "bars");
  tile.__cx2d.__draws.length = 0;
  await P.frame();
  f0 = fq();
  const tl_qb = cardQ();
  await P.input("selFill", "tile");
  await P.frame();
  const tl_qt = cardQ(), tl_dt1 = drawn().length;
  await P.frame();
  const tl_dt2 = drawn().slice();
  await P.input("selFill", "bars");
  await P.frame();
  ok("  ... back on black the next frame hides it again, draws nothing more and FORGETS the copies (the key is reset), so choosing the repeats again starts over: a frame that marks the card, then one that copies",
     f0.fill === "bars" && f0.disp === "none" && f0.pic.tile === false && tl_qb.key === "" && tl_qb.dirty === false && tl_qt.dirty === true && tl_dt1 === 0 &&
     JSON.stringify(tl_dt2) === JSON.stringify(w0.at) && fq().disp === "none" && cardQ().key === "", f0.disp + ", then " + tl_dt1 + " and " + tl_dt2.length + " copies");

  // ---- the zoom pulse -----------------------------------------------------------------------------
  const z = JSON.parse(env.run(`function(){ musicZoomApply(1.08); const c = cards.disp[0];
    const o = { t: c.cv.style.transform, clip: c.wrap.style.overflow, host: c.wrap.parentNode.id, ui: stage.ui.style.transform, st: stage.el.style.transform,
                tile: stage.tile.style.transform };
    musicZoomApply(1); o.back = c.cv.style.transform; o.tileBack = stage.tile.style.transform; return JSON.stringify(o); }`));
  ok("the zoom pulse still works: a CSS scale on the picture's canvas, clipped by its wrap inside #stage; the controls and #stage are not scaled",
     z.t === "scale(1.0800)" && z.clip === "hidden" && z.host === "stage" && z.ui === undefined && z.st === undefined && z.back === "", JSON.stringify(z));
  ok("  ... and it reaches the repeated picture: #stageTile takes the SAME scale (so the copies pulse as one with the picture under them) and loses it with it",
     z.tile === "scale(1.0800)" && z.tile === z.t && z.tileBack === "" && /stage\.tile\) stage\.tile\.style\.transform = t;/.test(html), JSON.stringify([z.tile, z.tileBack]));

  // ---- idle fade and the waking press, on the page (events driven by hand) -------------------------
  const cls = () => ["idle", "waking", "fatal"].filter(c => sEl.classList.contains(c)).join();
  const sev = (k, t) => JSON.parse(env.run("function(k, t){ return JSON.stringify(stageEvent(k, t)); }", k, t));
  const T = 1e9;                                   // far beyond the stub's own clock
  env.run("function(t){ stageEvent('move', t); stageTick(t + 60000); }", T);
  ok("before Start the bar is ALWAYS shown: a minute without activity and #stage is not idle", cls() === "" && env.run("function(){ return running; }") === false, cls());
  env.run(MODEL, 0.004);
  P.A.t = 5;
  await P.click("btnRun");
  env.run("function(t){ stageEvent('move', t); stageTick(t + 2999); }", 2 * T);
  const before = cls();
  env.run("function(t){ stageTick(t + 3000); }", 2 * T);
  ok("after Start: shown 2999 ms after the last activity, faded (#stage.idle) at 3000",
     env.run("function(){ return running; }") === true && before === "" && cls() === "idle", "\"" + before + "\" then \"" + cls() + "\"");
  const w1 = sev("down", 2 * T + 5000), c1cls = cls(), w2 = sev("up", 2 * T + 5060), w3 = sev("click", 2 * T + 5065);
  ok("a tap on the faded page brings them back NON-interactive (#stage.waking) and its click is the one the page's capture listener cancels; Stop is not pressed",
     w1.shown && w1.waking && c1cls === "waking" && w2.waking && w3.swallow === true && cls() === "" &&
     env.run("function(){ return running; }") === true && env.getEl("btnRun").textContent === "Stop", c1cls + " -> \"" + cls() + "\"");
  const w4 = sev("down", 2 * T + 6000), w5 = sev("click", 2 * T + 6100);
  await P.click("btnRun");
  ok("  ... the next tap is an ordinary one: not swallowed, and Stop stops", w4.waking === false && w5.swallow === false && env.run("function(){ return running; }") === false);
  env.run("function(t){ stageTick(t); }", 2 * T + 60000);
  ok("  ... and stopped, they stay", cls() === "");
  await P.click("btnRun");
  env.run("function(t){ stageEvent('move', t); stage.hover = true; stageTick(t + 60000); }", 3 * T);
  const held = cls();
  env.run("function(t){ stage.hover = false; stageTick(t + 60000); }", 3 * T);
  ok("while the mouse rests on the bar or the drawer they do not fade; once it leaves, they do", held === "" && cls() === "idle", "\"" + held + "\" then \"" + cls() + "\"");
  await P.click("btnRun");
  env.tick();
  ok("the stage's own timer (the stub's interval, fired once) runs the same tick: stopped, so shown again", cls() === "");
  // touch has no hover: there an OPEN drawer holds the controls up (it is being read)
  await P.click("btnRun");
  env.run("function(t){ stage.hover = false; stage.touch = true; stageDrawerOpen(true); stageEvent('move', t); stageTick(t + 60000); }", 5 * T);
  const th0 = cls();
  env.run("function(t){ stageDrawerOpen(false); stageTick(t + 60000); }", 5 * T);
  const th1 = cls();
  env.run("function(t){ stageDrawerOpen(true); stageTick(t + 60001); }", 5 * T);
  const th2 = cls();
  env.run("function(t){ stage.touch = false; stageTick(t + 60002); }", 5 * T);
  const th3 = cls();
  env.run("function(){ stageDrawerOpen(false); }");
  await P.click("btnRun");
  ok("on TOUCH (stage.touch: the last pointer was a finger) an open drawer holds them up while running, a minute past the 3 s; closing it lets them fade, opening it again holds them again -- and with a mouse (touch false) the open drawer alone does not",
     th0 === "" && th1 === "idle" && th2 === "" && th3 === "idle" && env.run("function(){ return stage.touch === false && running === false; }") === true &&
     /touch: false,/.test(src) && /hold: stage\.hover \|\| stage\.isFatal \|\| \(stage\.touch && stageDrawerIsOpen\(\)\) \}\);/.test(src) &&
     (src.match(/stage\.touch = e\.pointerType === "touch";/g) || []).length === 2, [th0, th1, th2, th3].map(c => '"' + c + '"').join(" "));

  // ---- status ---------------------------------------------------------------------------------------
  env.run("function(){ stageDrawerOpen(false); showStatus('choose an audio file first', 'info'); stageTick(4e9); }");
  ok("a status message stays in the drawer and OPENS it (a closed drawer would hide it)",
     env.getEl("status").parentNode.id === "stageStatus" && env.run("function(){ return stageDrawerIsOpen(); }") === true && cls() === "");
  env.run("function(){ stageDrawerOpen(false); stageTick(4e9 + 250); }");
  ok("  ... once: closing the drawer again is respected while the message stands", env.run("function(){ return stageDrawerIsOpen(); }") === false);
  env.run("function(){ showStatus('WebGPU device lost: gone', 'err'); stageTick(4e9 + 500); }");
  ok("an error that stops the page (the device lost) shows CENTRED: #status moves into #stageFatal and #stage is .fatal",
     env.getEl("status").parentNode.id === "stageFatal" && cls() === "fatal" && env.getEl("status").textContent === "WebGPU device lost: gone");
  env.getEl("stageFatal").onclick({});
  ok("  ... a click dismisses it back into the drawer, and it stays dismissed", env.getEl("status").parentNode.id === "stageStatus" && cls() === "" &&
     (env.run("function(){ stageTick(4e9 + 750); }"), env.getEl("status").parentNode.id === "stageStatus"));
  ok("through all of it the stage page logged no failure of its own", realFails(env).length === 0, realFails(env).slice(0, 2).join(" | "));

  // ---- no WebGPU at all, on the stage---------------------------------------------------------------
  // (gpuFallback looks up controls this page has never had -- btnParams, btnText, intro:
  // the same misses are logged on a no-WebGPU boot WITHOUT the flag, which is the reference)
  const noGpuFails = async search => { const e = await boot(dir, "music.html", { search: search, noGpu: true }); await turn(); return e; };
  const G0 = await noGpuFails("?plain"), G = await noGpuFails("");
  const own = e => realFails(e).filter(f => realFails(G0).indexOf(f) < 0);
  G.tick();
  const gs = G.getEl("stage");
  ok("on the stage with NO WebGPU: the stage and its bar are built, there is no picture, and the browser advice is centred (taken back from the poster card common.js moved it to)",
     !!gs && gs.children.filter(c => c.className === "cvwrap").length === 0 && gs.classList.contains("fatal") &&
     G.getEl("status").parentNode.id === "stageFatal" && /WebGPU is not available/.test(G.getEl("status").textContent) &&
     up(G.getEl("btnRun"), "stageBar") && G.getEl("btnRun").disabled === true && up(G.getEl("grpMusic"), "stageDrawer") &&
     own(G).length === 0 && realFails(G).join() === realFails(G0).join() && G0.allEls.filter(e => /^stage/i.test(e.id || "")).length === 0,
     own(G).slice(0, 2).join(" | "));
}

// ===========================================================================
// 16. the window-shaped box and the current sheets (MUSIC_STAGE_PLAN S2 / S4, step 3)
// ===========================================================================
// The rectangular rules against the offline script (fixtures/music_ref.json, key `stage`,
// written by ../gen_musicfixture.py), then the page: window -> box -> grid, the three
// selects, the auto rung, the sheets' initial condition and its Reset. What the stub cannot
// say: anything a GPU computes (that the sheets are HELD, what they look like, what the
// timestep does), how a real window reports its size, or how long a rebuild takes.
async function legShape(state) {
  const F = state.F && state.F.stage;
  ok("the fixture carries the offline rectangular-box and sheet rows", !!F && /box_for, cutoff_k, band_plan/.test(F.source) && F.hyper === 4);
  if (!F) return;
  const ctx = vm.createContext({});
  vm.runInContext(fs.readFileSync(path.join(dir, "music.js"), "utf8"), ctx, { filename: "music.js" });
  const M = vm.runInContext(`({ musicBandPlan, musicRint, musicWindowAspect, musicBoxFor, musicCutoffK, musicGridFor, musicRungFor,
    musicAutoRung, musicSheetPsi0, musicSheetEq, musicSheetField, musicSheetEnergy, musicKickBlobs, musicBlobsFor, musicPanX, musicWrap,
    musicRng, MUSIC_LBOX, MUSIC_ASPECT_MAX, MUSIC_RUNGS, MUSIC_AUTO_WINDOW, MUSIC_AUTO_REBASES })`, ctx);
  const L = 2 * Math.PI, rel = (a, b) => Math.abs(a / b - 1);

  // ---- the box ------------------------------------------------------------------------
  let db = 0;
  for (const b of F.boxes) { const g = M.musicBoxFor(b.aspect); db = Math.max(db, rel(g[0], b.box[0]), rel(g[1], b.box[1])); }
  ok("musicBoxFor is box_for at " + F.boxes.length + " aspects (16:9, 2:1, 2.4:1, the stub's 1.6, portrait ones): the short side 2 pi, the long side 2 pi * the aspect",
     F.boxes.length >= 9 && db < 1e-15 && F.lbox === M.MUSIC_LBOX, "max relative difference " + e2(db));
  ok("  ... aspect 1 is EXACTLY the square 2 pi box; landscape is long in x, portrait long in y, and the short side is 2 pi bit for bit",
     M.musicBoxFor(1).join() === [L, L].join() && M.musicBoxFor(2)[0] === 2 * L && M.musicBoxFor(2)[1] === L &&
     M.musicBoxFor(0.5)[0] === L && M.musicBoxFor(0.5)[1] === 2 * L && M.musicBoxFor(16 / 9)[1] === L && M.musicBoxFor(9 / 16)[0] === L);
  ok("  ... and a non-positive or non-finite aspect is refused",
     [0, -1, NaN, Infinity, "2", undefined].every(a => /aspect/.test(throws(() => M.musicBoxFor(a)))));

  // ---- the band plan, the cutoff, the dissipation ---------------------------------------
  let np = 0, worst = 0;
  const seen = {};
  for (const p of F.plans) {
    const tag = "band_plan((" + p.grid.join(", ") + "), box " + p.tag + (p.nbands === null ? "" : ", nbands = " + p.nbands)
              + (p.sigmaMin === null ? "" : ", sigma_min = " + p.sigmaMin) + ")";
    const box = M.musicBoxFor(p.aspect);
    let r = null, err = "";
    try { r = M.musicBandPlan(p.grid, p.nbands, p.sigmaMin, null, box[0], box[1]); } catch (e) { err = e.message; }
    if (p.error) { ok(tag + ": refused by both", !!err, err.slice(0, 50)); continue; }
    let ds = 0;
    if (r && r.sigmas.length === p.sigmas.length)
      for (let i = 0; i < p.sigmas.length; i++) ds = Math.max(ds, rel(r.sigmas[i], p.sigmas[i]));
    const kc = M.musicCutoffK(p.grid[0], p.grid[1], box[0], box[1]);
    const dd = Math.max.apply(null, p.diss.map(([amp, v]) => rel(48 * amp / Math.pow(kc, 2 * F.hyper), v)));
    worst = Math.max(worst, ds, rel(kc, p.cutoff_k), dd);
    ok(tag + ": " + p.nblobs.length + " bands, blobs " + p.nblobs.join(" ") + ", cutoff k " + p.cutoff_k.toFixed(2),
       !!r && r.sigmas.length === p.sigmas.length && ds < 1e-14 && JSON.stringify(Array.from(r.nblobs)) === JSON.stringify(p.nblobs) &&
       rel(kc, p.cutoff_k) < 1e-15 && dd < 1e-14, err || "sigma " + e2(ds) + ", cutoff " + e2(rel(kc, p.cutoff_k)) + ", diss " + e2(dd));
    if (p.nbands === null && p.sigmaMin === null) seen[p.tag + "/" + p.rung] = 1;
    np++;
  }
  ok("  ... " + np + " plans: the three rungs at 16:9, 2:1, 2.4:1 and in portrait (9:16, 1:2.4), the square box's two, and explicit nbands / sigma_min",
     np >= 20 && ["16:9", "2:1", "2.4:1", "9:16", "1:2.4"].every(t => ["light", "standard", "sharp"].every(r => seen[t + "/" + r])) &&
     seen["square/light"] && seen["square/standard"], "worst relative difference " + e2(worst));
  // the rules from their definitions, at 16:9 on the standard rung (stretched cells)
  const b169 = M.musicBoxFor(16 / 9), p169 = M.musicBandPlan([256, 256], null, null, null, b169[0], b169[1]);
  const last = a => a[a.length - 1];
  ok("the narrowest blob is 3.7 times the LARGER grid spacing (16:9 on 256 x 256: dx = " + (b169[0] / 256).toFixed(4) + " > dy = " + (L / 256).toFixed(4) + "), the widest stays 0.45",
     last(p169.sigmas) === 3.7 * b169[0] / 256 && last(p169.sigmas) > 3.7 * L / 256 * 1.7 && p169.sigmas[0] === 0.45 &&
     last(M.musicBandPlan([256, 256], null, null, null, L, b169[0]).sigmas) === 3.7 * b169[0] / 256, "sigma_min " + last(p169.sigmas).toFixed(4));
  ok("  ... and on a grid that is finer along the long side the SHORT side's spacing can be the larger one (16:9 on 512 x 256: 3.7 * 2 pi / 256)",
     last(M.musicBandPlan([512, 256], null, null, null, b169[0], b169[1]).sigmas) === 3.7 * L / 256 &&
     last(M.musicBandPlan([256, 512], null, null, null, b169[1], b169[0]).sigmas) === 3.7 * L / 256);
  const area = (b169[0] / L) * (b169[1] / L);
  ok("the blob counts are rint(3 * 0.45 / sigma * area / (2 pi)^2): " + p169.nblobs.join(" ") + " at 16:9 against " +
     p169.sigmas.map(s => Math.round(1.35 / s)).join(" ") + " for the same widths in the square box",
     p169.nblobs.every((n, i) => n === M.musicRint(3 * 0.45 / p169.sigmas[i] * area)) && p169.nblobs[0] === 5 &&
     p169.nblobs.some((n, i) => n !== Math.round(1.35 / p169.sigmas[i])));
  ok("  ... the factor is the AREA, both sides: a 4 pi x 4 pi box has four times the square box's counts at the same widths, 4 pi x pi the same count as the square one",
     JSON.stringify(M.musicBandPlan(256, 3, 0.2, null, 2 * L, 2 * L).nblobs) === JSON.stringify(M.musicBandPlan(256, 3, 0.2).nblobs.map((n, i) => M.musicRint(1.35 / M.musicBandPlan(256, 3, 0.2).sigmas[i] * 4))) &&
     M.musicBandPlan(256, 3, 0.2, null, 2 * L, 2 * L).nblobs[0] === 12 && M.musicBandPlan(256, 3, 0.2, null, 2 * L, L / 2).nblobs.join() === M.musicBandPlan(256, 3, 0.2).nblobs.join() &&
     M.musicBandPlan(256, 3, 0.2, null, 2 * L, L / 2).nblobs[0] === 3);
  ok("  ... rounding half to EVEN, as numpy's rint: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, 3.5 -> 4, 2.4999 -> 2, 2.5001 -> 3",
     [0.5, 1.5, 2.5, 3.5, 2.4999, 2.5001, 7].map(M.musicRint).join() === "0,2,2,4,2,3,7");
  ok("with the square box the factor is exactly 1: musicBandPlan([256, 256], .., 2 pi, 2 pi) is musicBandPlan(256), bit for bit; so is a box left out",
     JSON.stringify(M.musicBandPlan([256, 256], null, null, 32, L, L)) === JSON.stringify(M.musicBandPlan(256, null, null, 32)) &&
     JSON.stringify(M.musicBandPlan([512, 512])) === JSON.stringify(M.musicBandPlan(512)) &&
     JSON.stringify(M.musicBandPlan(256, 3, 0.2, null, L, L)) === JSON.stringify(M.musicBandPlan(256, 3, 0.2)));
  ok("musicBandPlan refuses a grid that is not a positive integer or a pair of them, and a box that is not two positive lengths",
     [[[256]], [[256, 0]], [[256, 128.5]], [[256, 128, 64]], [["256", 128]]].every(a => /positive integer/.test(throws(() => M.musicBandPlan.apply(null, a)))) &&
     [[256, null, null, null, 0, L], [256, null, null, null, L, -1], [256, null, null, null, NaN, L], [256, null, null, null, "6", L]]
       .every(a => /box/.test(throws(() => M.musicBandPlan.apply(null, a)))));
  ok("musicCutoffK is the SMALLER of the two dealias cutoffs: n/3 exactly in the square 2 pi box, the long side's at 16:9 on 256 x 256, the short side's on 512 x 256",
     M.musicCutoffK(256, 256, L, L) === 256 / 3 && M.musicCutoffK(256, 256, b169[0], L) === (256 / 3) * (L / b169[0]) &&
     M.musicCutoffK(512, 256, b169[0], L) === 256 / 3 && M.musicCutoffK(256, 512, L, b169[0]) === 256 / 3 &&
     M.musicCutoffK(256, 256, b169[0], L) < 256 / 3 * 0.6);
  // KICK_MAX: one kick() holds 32 slots per channel; a dipole band takes two per blob
  let most = 0, mostD = 0, nsw = 0, thrown = 0;
  for (let i = 0; i <= 140; i++) {
    const asp = 1 + i * 0.01;
    for (const a of [asp, 1 / asp]) for (const rung of M.MUSIC_RUNGS) {
      const bx = M.musicBoxFor(a), g = M.musicGridFor(rung, bx[0], bx[1]);
      try {
        const p = M.musicBandPlan(g, null, null, 32, bx[0], bx[1]);
        most = Math.max(most, Math.max.apply(null, p.nblobs)); mostD = Math.max(mostD, 2 * p.nblobs[0]);
      } catch (e) { thrown++; }
      nsw++;
    }
  }
  ok("the area-scaled counts never exceed the solver's KICK_MAX = 32 per channel: over " + nsw + " boxes (aspect 1 to 2.4 both ways, every rung) the most is " + most + ", and a bass DIPOLE band (two slots per blob) at most " + mostD,
     thrown === 0 && most <= 32 && most === 30 && mostD <= 32 && mostD === 14, thrown + " refused");

  // ---- stereo placement, and blob placement in a rectangular box ------------------------
  let dpx = 0;
  for (const c of F.panx) dpx = Math.max(dpx, Math.abs(M.musicPanX(c.pan, c.gain, M.musicBoxFor(c.aspect)[0]) - c.xc));
  ok("musicPanX(pan, gain, Lx) is run()'s Lx (0.5 + 0.4 clip(gain pan)): over the long side in landscape, the short one in portrait; without Lx it is the 2 pi it was",
     F.panx.length >= 15 && dpx < 1e-13 && M.musicPanX(0.5, 1) === L * 0.7 && Math.abs(M.musicPanX(1, 1, 10) - 9) < 1e-14 && Math.abs(M.musicPanX(-1, 3, 10) - 1) < 1e-14 && M.musicPanX(0, 5, 10) === 5, "max difference " + e2(dpx));
  // the offline kick_blobs was run on a SCRIPTED generator; the page's draw order differs
  // (per blob x, y, sign -- then, placed, two uniforms per normal -- then the angles), so the
  // same numbers are dealt to it in ITS order: uniform -> the same uniform; a normal g ->
  // (1 - exp(-g^2/2), 0 or 1/2), which Box-Muller's cosine branch turns back into g
  const script = (c) => {
    const pu = c.u.slice(), pg = c.g.slice(), ps = c.s.slice(), seq = [], n = c.nblob, take = (a, k) => a.splice(0, k);
    const placed = () => {
      const xy = take(pu, 2 * n), gs = c.xc === null ? [] : take(pg, n), sg = take(ps, n);
      for (let i = 0; i < n; i++) seq.push(xy[2 * i], xy[2 * i + 1], sg[i] > 0 ? 0.75 : 0.25);
      for (const g of gs) seq.push(1 - Math.exp(-0.5 * g * g), g >= 0 ? 0 : 0.5);
    };
    const angles = () => { for (const v of take(pu, n)) seq.push(v); };
    if (c.kind === "elsasser") { placed(); placed(); }
    else if (c.kind === "sheet") { placed(); angles(); placed(); angles(); }
    else if (c.kind === "sheet-island") { placed(); angles(); }
    else if (c.kind === "dipole") { placed(); angles(); }
    else placed();
    return { seq: seq, used: [c.u.length - pu.length, c.g.length - pg.length, c.s.length - ps.length] };
  };
  let pbad = [], pmax = 0, long = 0;
  for (const c of F.placement) {
    const sc = script(c);
    let at = 0;
    const rng = () => { if (at >= sc.seq.length) throw new Error("the page drew more numbers than the offline script"); return sc.seq[at++]; };
    let list = null, err = "";
    try { list = M.musicKickBlobs(rng, c.kind, c.sigma, c.nblob, c.amp, { xc: c.xc, ratio: c.ratio, sheetEnergy: c.sheetEnergy, box: c.box }); }
    catch (e) { err = e.message; }
    const rows = pol => (list || []).filter(b => (b.pol < 0) === (pol < 0)).map(b => b.ratio === undefined ? [b.x, b.y, b.sigma, b.amp] : [b.x, b.y, b.sigma, b.amp, b.ratio, b.theta]);
    let d = 0, shape = !!list && at === sc.seq.length && sc.used.join() === c.used.join();
    for (const [got, want] of [[rows(1), c.zp], [rows(-1), c.zm]]) {
      if (got.length !== want.length) { shape = false; continue; }
      for (let i = 0; i < want.length; i++) {
        if (got[i].length !== want[i].length) { shape = false; continue; }
        for (let j = 0; j < want[i].length; j++) d = Math.max(d, Math.abs(got[i][j] - want[i][j]));
        if (want[i][0] > L || want[i][1] > L) long++;
        if (!(got[i][0] >= 0 && got[i][0] < c.box[0] && got[i][1] >= 0 && got[i][1] < c.box[1])) shape = false;
      }
    }
    pmax = Math.max(pmax, d);
    if (!shape || !(d < 1e-12)) pbad.push(c.tag + " " + c.kind + (c.xc === null ? "" : " placed") + (err ? ": " + err : " (" + e2(d) + ")"));
  }
  ok("musicKickBlobs in a rectangular box is kick_blobs(.., box): " + F.placement.length + " kicks (six types, uniform and placed, 16:9 and 9:16) dealt the offline script's own uniforms, normals and signs give its blobs -- x over Lx, y over Ly, the stereo scatter 0.12 Lx about xc and wrapped by Lx, a dipole's halves wrapped per axis",
     F.placement.length === 24 && pbad.length === 0 && long > 40, pbad.slice(0, 3).join(" | ") || "max difference " + e2(pmax) + "; " + long + " blobs beyond 2 pi on the long side");
  const rg = M.musicRng(11), far = M.musicBlobsFor(rg, 0.3, 400, 1, 1, [3 * L, L]), tall = M.musicBlobsFor(M.musicRng(11), 0.3, 400, 1, 1, [L, 3 * L]);
  ok("musicBlobsFor(.., box): 400 blobs fill a 6 pi x 2 pi box in x and stay inside 2 pi in y (and the other way round in a tall box); without a box it draws what it always drew",
     far.every(b => b.x >= 0 && b.x < 3 * L && b.y >= 0 && b.y < L) && far.filter(b => b.x > 2 * L).length > 100 &&
     tall.every((b, i) => Math.abs(b.x - far[i].x / 3) < 1e-14 && Math.abs(b.y - far[i].y * 3) < 1e-13 && b.amp === far[i].amp) && tall.filter(b => b.y > 2 * L).length > 100 &&
     JSON.stringify(M.musicBlobsFor(M.musicRng(5), 0.2, 6, 1, -1)) === JSON.stringify(M.musicBlobsFor(M.musicRng(5), 0.2, 6, 1, -1, [L, L])));

  // ---- the sheets ------------------------------------------------------------------------
  for (const c of F.sheets) {
    const box = M.musicBoxFor(c.aspect), plan = M.musicBandPlan(c.grid, null, null, null, box[0], box[1]);
    const a = c.width * last(plan.sigmas), psi0 = M.musicSheetPsi0(c.strength, c.amp, a), nx = c.grid[0], ny = c.grid[1];
    const fld = M.musicSheetField(nx, ny, box[0], psi0, a);
    let d = 0, flat = fld.length === nx * ny && fld.BYTES_PER_ELEMENT === 4;
    for (let k = 0; k < c.psi.length; k++) {
      const i = k * c.stride;
      d = Math.max(d, Math.abs(M.musicSheetEq(i * box[0] / nx, box[0], psi0, a) - c.psi[k]) / c.psi0);
      for (const j of [0, 1, ny >> 1, ny - 1]) if (fld[i * ny + j] !== Math.fround(M.musicSheetEq(i * box[0] / nx, box[0], psi0, a))) flat = false;
    }
    // the gate's own derivative of the page's profile: a naive DFT along x, in fp64
    const f = Float64Array.from({ length: nx }, (_, i) => M.musicSheetEq(i * box[0] / nx, box[0], psi0, a));
    let e = 0;
    const dre = new Float64Array(nx), kmax = nx / 2;
    for (let k = 1; k < kmax; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < nx; i++) { const th = 2 * Math.PI * k * i / nx; re += f[i] * Math.cos(th); im -= f[i] * Math.sin(th); }
      const kk = 2 * Math.PI * k / box[0];
      e += kk * kk * (re * re + im * im) / (nx * nx);                 // (1/2) * 2 (the +-k pair) * k^2 |f_k|^2
      for (let i = 0; i < nx; i++) { const th = 2 * Math.PI * k * i / nx; dre[i] += 2 * kk * (-re * Math.sin(th) - im * Math.cos(th)) / nx; }
    }
    let peak = 0;
    for (let i = 0; i < nx; i++) peak = Math.max(peak, Math.abs(dre[i]));
    ok("sheets, " + c.tag + ": a = " + c.a.toFixed(4) + " (" + c.width + " x the narrowest blob), psi0 = " + c.psi0.toFixed(4) + ", psi_eq at " + c.psi.length + " points, energy " + c.energy.toPrecision(4),
       rel(a, c.a) < 1e-14 && rel(psi0, c.psi0) < 1e-14 && d < 1e-13 && flat && rel(M.musicSheetEnergy(psi0, a, box[0]), c.energy) < 1e-14 &&
       rel(e, c.energy_grid) < 1e-9 && rel(e, c.energy) < 2e-3 && rel(peak, c.peak_grid) < 1e-9 && rel(peak, c.strength * c.amp) < 0.02,
       "psi " + e2(d) + " of psi0; grid energy vs closed form " + e2(rel(e, c.energy)) + "; peak field " + peak.toFixed(4) + " for " + (c.strength * c.amp));
  }
  ok("  ... " + F.sheets.length + " sheet rows: landscape, portrait and square boxes, the page's defaults (strength 5, width 2) among them",
     F.sheets.length >= 5 && F.sheets.some(c => c.strength === 5 && c.width === 2 && c.amp === 1 && c.aspect > 1));
  const sy = (x, Lx) => M.musicSheetEq(x, Lx, 1, 0.4);
  ok("musicSheetEq: psi0 at the centre x = Lx/2, even about it, the same at x = 0 and x = Lx (the periodic images), and its peak field 4 psi0 / (3 sqrt(3) a) is what musicSheetPsi0 inverts",
     Math.abs(sy(5, 10) - 1) < 1e-10 && Math.abs(sy(3.7, 10) - sy(6.3, 10)) < 1e-15 && sy(0, 10) === sy(10, 10) && sy(0, 10) > 0 && sy(0, 10) < 1e-9 &&
     Math.abs(sy(0.3, 2) - (1 / Math.cosh(-0.7 / 0.4) ** 2 + 1 / Math.cosh(-2.7 / 0.4) ** 2 + 1 / Math.cosh(1.3 / 0.4) ** 2)) < 1e-15 &&
     M.musicSheetPsi0(5, 1, 0.4) === 0.75 * Math.sqrt(3) * 0.4 * 5 && M.musicSheetPsi0(2, 3, 0.1) === 3 * M.musicSheetPsi0(2, 1, 0.1) &&
     M.musicSheetPsi0(0, 1, 0.4) === 0 && M.musicSheetEnergy(2, 0.5, 10) === 8 * 4 / (15 * 0.5 * 10));

  // ---- window -> box -> grid, and the auto rung: the pure rules ---------------------------
  const wa = M.musicWindowAspect;
  ok("musicWindowAspect: the window's width over height -- 1920 x 1080 is 16/9, 390 x 844 a tall 0.462, 1280 x 800 is 1.6 -- clamped to [1/2.4, 2.4]",
     M.MUSIC_ASPECT_MAX === 2.4 && wa(1920, 1080, "window") === 16 / 9 && wa(390, 844, "window") === 390 / 844 && wa(1280, 800, "window") === 1.6 &&
     wa(3440, 1000, "window") === 2.4 && wa(2400, 1000, "window") === 2.4 && wa(2399, 1000, "window") === 2.399 &&
     wa(300, 2000, "window") === 1 / 2.4 && wa(1000, 1000, "window") === 1);
  ok("  ... `square`, and a window that is not two finite positive numbers, is 1",
     wa(1920, 1080, "square") === 1 && wa(390, 844, "square") === 1 && [[0, 800], [1280, 0], [NaN, 800], [1280, Infinity], [-5, 5], [undefined, undefined]].every(a => wa(a[0], a[1], "window") === 1));
  const gf = (rung, a) => { const b = M.musicBoxFor(a); return M.musicGridFor(rung, b[0], b[1]).join("x"); };
  ok("musicGridFor, long side x short side: light 256 x 128, standard 256 x 256, sharp 512 x 256 in landscape; in portrait the long side is y (128 x 256, 256 x 256, 256 x 512)",
     gf("light", 16 / 9) === "256x128" && gf("standard", 16 / 9) === "256x256" && gf("sharp", 16 / 9) === "512x256" &&
     gf("light", 9 / 16) === "128x256" && gf("standard", 9 / 16) === "256x256" && gf("sharp", 9 / 16) === "256x512" &&
     gf("light", 2.4) === "256x128" && gf("sharp", 1.01) === "512x256" && gf("sharp", 1 / 1.01) === "256x512" && M.MUSIC_RUNGS.join() === "light,standard,sharp");
  ok("  ... a SQUARE box takes square grids: light 128 x 128, standard 256 x 256, and sharp is standard (512 x 512 is not offered); an unknown rung is refused",
     gf("light", 1) === "128x128" && gf("standard", 1) === "256x256" && gf("sharp", 1) === "256x256" &&
     /rung/.test(throws(() => M.musicGridFor("auto", L, L))) && /rung/.test(throws(() => M.musicGridFor(undefined, 2 * L, L))));
  ok("musicRungFor: a named rung is itself; auto is where auto stands -- standard, or light once it has dropped -- and NEVER sharp, whatever it is handed",
     ["light", "standard", "sharp"].every(r => ["light", "standard", "sharp"].every(a => M.musicRungFor(r, a) === r)) &&
     M.musicRungFor("auto", "standard") === "standard" && M.musicRungFor("auto", "light") === "light" && M.musicRungFor("auto", "sharp") === "standard" &&
     M.musicRungFor("auto", undefined) === "standard" && M.musicRungFor("nonsense", "light") === "light");
  const ar = M.musicAutoRung;
  ok("musicAutoRung: from standard, MORE than " + M.MUSIC_AUTO_REBASES + " fall-behind rebases in the first " + M.MUSIC_AUTO_WINDOW + " s drop ONE rung, to light (3 does not, 4 does); the limit is an argument",
     M.MUSIC_AUTO_REBASES === 3 && M.MUSIC_AUTO_WINDOW === 10 && ar("standard", 0) === "standard" && ar("standard", 3) === "standard" && ar("standard", 4) === "light" &&
     ar("standard", 1000) === "light" && ar("standard", 4, 4) === "standard" && ar("standard", 5, 4) === "light" && ar("standard", 1, 0) === "light");
  ok("  ... it never climbs (light stays light at any count, 0 included) and is never sharp (handed sharp it answers standard, or light)",
     [0, 1, 3, 4, 1e6].every(n => ar("light", n) === "light") && ar("sharp", 0) === "standard" && ar("sharp", 9) === "light" && ar(undefined, 0) === "standard" &&
     (() => { let r = "standard"; const out = []; for (const n of [0, 9, 0, 0, 50, 0]) { r = ar(r, n); out.push(r); } return out.join(); })() === "standard,light,light,light,light,light");

  // ---- the page, WITH ?plain: at its defaults it is the shipped one ----------------------
  const SNAP = `function(){ const s = solver, c = cards.disp[0]; return JSON.stringify({ p: s.p, ui: uiParams(), sh: music.shape, plan: music.plan,
    cv: [c.cv.width, c.cv.height], sel: c.selField.value, lit: music.lit.length, grave: graveyard.length,
    b0: Array.from(s._b0), hasB0: !!s.buf.b0, hasEqk: !!s.buf.eqk, lin0: s.g.gridB[1], box: el("musicBox").textContent,
    dis: ["rSheetB", "rSheetW", "rField", "rFieldDeg", "cbFieldWalk"].map(id => !!el(id).disabled) }); }`;
  const snap = env => JSON.parse(env.run(SNAP));
  // the model readback (MODEL), ONCE per solver: it wraps step / kick / setIC, and a second
  // install on the same solver would advance its clock twice per step
  const model = env => env.run("function(dt){ if (!solver.__log) (" + MODEL + ")(dt); }", 0.004);
  const setWin = (env, w, h) => env.run("function(w, h){ window.innerWidth = w; window.innerHeight = h; }", w, h);
  const mark = env => env.run("function(){ globalThis.__svm = solver; }");
  const same = env => env.run("function(){ return solver === __svm; }");
  const P = await musicPage("?plain&musicdebug"), env = P.env;
  let q = snap(env);
  const diss0 = 48 * Math.max(1, 0.1) / Math.pow(256 / 3, 2 * 4);      // the shipped expression, MUSIC_N / 3
  const shipped = JSON.stringify({ nx: 256, ny: 256, Lx: L, Ly: L, hyper: 4, diss: diss0, pm: 1, eqsrc: false, epsP: 0, epsM: 0, tau: 1,
                                   fshell: [1, 3], smax: 1e4, cfl: 0.4, seed: 7, kick: true, mu: 1, b0: true });
  ok("with ?plain the three selects are at box = square, grid = STANDARD (not auto: the shipped page never changes its grid), background = walking mean field; sheet field 5, sheet width 2",
     env.getEl("selBox").value === "square" && env.getEl("selGrid").value === "standard" && env.getEl("selGrid").options.map(o => o.value).join() === "auto,light,standard,sharp" && env.getEl("selBg").value === "field" &&
     env.getEl("rSheetB").value === "5" && env.getEl("rSheetW").value === "2" && env.getEl("vSheetB").textContent === "5.0" && env.getEl("vSheetW").textContent === "2.00" &&
     ["rSheetB", "rSheetW"].every(id => env.getEl(id).min !== undefined) && env.getEl("rSheetB").max === "8" && env.getEl("rSheetW").min === "1" && env.getEl("rSheetW").max === "4");
  ok("  ... and what the page hands the Solver is EXACTLY the shipped parameter set, key for key and bit for bit (no eqsrcDrag key; diss is 48 / (256/3)^8, the old MUSIC_N / 3 expression's own double)",
     JSON.stringify(q.ui) === shipped && !("eqsrcDrag" in q.p) && q.p.diss === diss0 && q.p.b0 === true && q.p.eqsrc === false && q.hasB0 && !q.hasEqk, JSON.stringify(q.ui).slice(0, 90));
  ok("  ... the band plan is musicBandPlan(256) under KICK_MAX, the card 512 x 512 showing |u|, and the line under the selects says so",
     JSON.stringify(q.plan) === JSON.stringify(M.musicBandPlan(256, null, null, 32)) && q.cv.join() === "512,512" && q.sel === "4" && q.lit === 4 &&
     /^box 6\.28 × 6\.28 on 256 × 256 \(standard\), 4 bands$/.test(q.box), q.box);
  ok("  ... the sheet sliders are disabled while the mean field is the background, the mean-field controls live", q.dis.join() === "true,true,false,false,false");
  // a resize, then Reset and Start: on the square box nothing is ever rebuilt
  mark(env);
  model(env);
  setWin(env, 390, 844);
  env.run("function(){ cardsSync(); }");
  await P.frame();
  await P.click("btnReset");
  P.A.t = 50;
  await P.click("btnRun");
  await P.frame(50.02);
  q = snap(env);
  ok("on the square box a window resized to portrait, then Reset, then Start, rebuild NOTHING: the same Solver object, the shipped parameters, nothing retired",
     same(env) === true && JSON.stringify(q.ui) === shipped && q.grave === 0 && env.run("function(){ return running; }") === true);
  await P.click("btnRun");
  const htmlS = fs.readFileSync(path.join(dir, "music.html"), "utf8");
  const inline = htmlS.slice(htmlS.indexOf("<script>\n"));
  ok("music.html installs no resize listener at all (the stage's own, in stage.js, re-fits the canvas and never touches the solver)",
     !/["'`]resize["'`]|onresize|orientationchange|ResizeObserver/.test(inline.split("\n").filter(l => !/^\s*\/\//.test(l)).join("\n")));

  // ---- box = the window's shape ------------------------------------------------------------
  setWin(env, 1920, 1080);
  await P.input("selBox", "window");
  await P.frame();
  q = snap(env);
  const f169 = F.plans.find(p => p.tag === "16:9" && p.rung === "standard");
  ok("box = window in a 1920 x 1080 window: a REBUILD (a new Solver, the old one retired and freed) on a 16:9 box, 2 pi (16/9) x 2 pi, standard rung 256 x 256",
     same(env) === false && q.grave === 0 && q.p.nx === 256 && q.p.ny === 256 && q.p.Lx === f169.box[0] && q.p.Ly === L && q.sh.aspect === 16 / 9 && q.sh.rung === "standard",
     q.p.nx + " x " + q.p.ny + " in " + q.p.Lx.toFixed(4) + " x " + q.p.Ly.toFixed(4));
  ok("  ... everything else is the shipped set (b0, no eqsrc, the drag), the dissipation is pinned at the SMALLER cutoff (the offline rule: 48 amp / cutoff_k^8 = " + f169.diss[0][1].toExponential(3) + "), and it is in the column",
     JSON.stringify(Object.assign({}, q.ui, { Lx: L, diss: diss0 })) === shipped && rel(q.p.diss, f169.diss[0][1]) < 1e-14 && q.p.diss > 50 * diss0 && q.lin0 === -0.5,
     "diss " + q.p.diss.toExponential(4) + " against the square box's " + diss0.toExponential(3));
  ok("  ... the band plan is the offline one for that box and grid (" + f169.nblobs.join(" ") + " blobs, sigma_min " + last(f169.sigmas).toFixed(4) + "), the strip has its " + f169.sigmas.length + " bands",
     q.plan.sigmas.length === f169.sigmas.length && q.plan.sigmas.every((s, i) => rel(s, f169.sigmas[i]) < 1e-14) && q.plan.nblobs.join() === f169.nblobs.join() &&
     q.lit === f169.sigmas.length && env.getEl("musicStrip").textContent.split("|").length === f169.sigmas.length &&
     new RegExp("^box 11\\.17 × 6\\.28 on 256 × 256 \\(standard\\), " + f169.sigmas.length + " bands$").test(q.box), q.box);
  ok("  ... and the display card takes the box's shape: 512 x 288 (was 512 x 512)", q.cv.join() === "512,288", q.cv.join("x"));
  // per unit area: the energy the walk's sqrt(2E) reads, and the fixed colour scale
  const pua = JSON.parse(env.run(`function(){ const m = /const INVN2: f32 = ([-+.\\de]+);/.exec(solver.pl.energyPartial.__code);
    return JSON.stringify({ inv: m ? parseFloat(m[1]) : NaN, box: /\\bL[xy]\\b|INVLX|INVLY/.test(solver.pl.energyPartial.__code), scale: musicScaleMax(musicUi()),
      fld: musicWalkField({ r: 1.5, th: 0 }, 0.08)[0] }); }`));
  ok("  ... the energy the walking field's sqrt(2E) reads is a VOLUME AVERAGE on the rectangular box too (the kernel's weight is 1 / (nx ny)^2, no box length in it), and the fixed colour scale is a multiple of amp alone: neither needs an area factor",
     rel(pua.inv, 1 / Math.pow(256 * 256, 2)) < 1e-11 && pua.box === false && Math.abs(pua.scale - Math.pow(2, 0.585)) < 1e-12 && Math.abs(pua.fld - 1.5 * 0.4) < 1e-15, JSON.stringify(pua));
  await P.input("rMAmp", 2.5);
  ok("  ... the amp slider still moves the dissipation by the same rule on the live solver (amp 2.5: " + f169.diss[2][1].toExponential(3) + ")",
     f169.diss[2][0] === 2.5 && rel(snap(env).p.diss, f169.diss[2][1]) < 1e-14 && same(env) === false);
  await P.input("rMAmp", 1);
  // a kick on the rectangular box: positions over the long side, the pan mapped onto Lx
  model(env);
  env.getEl("cbStereo").checked = true;
  P.A.t = 10;
  await P.click("btnRun");
  await P.frame(10.02);
  await P.frame(10.04);
  P.takeLog();
  env.run(`function(){ music.stereoSeen = true; const t = music.audio.ctx.currentTime - music.audio.t0;
    for (let i = 0; i < 12; i++) music.queue.push({ due: t, o: { band: 2, strength: 1, t: t, tFire: t }, pan: null });
    music.queue.push({ due: t, o: { band: 1, strength: 0.5, t: t, tFire: t }, pan: 0.35 }); }`);
  await P.frame(10.06);
  const kl = P.takeLog().filter(l => l.op === "kick");
  const free = [].concat.apply([], kl.slice(0, 12).map(k => k.list)), placedK = kl[12] ? kl[12].list : [];
  const xc = M.musicPanX(0.35, 2, f169.box[0]), mx = placedK.reduce((a, b) => a + b.x, 0) / Math.max(1, placedK.length);
  ok("a kick on the 16:9 box: " + free.length + " unplaced blobs spread over the whole long side (x up to Lx = 11.17, " + free.filter(b => b.x > L).length + " of them beyond 2 pi) with y inside 2 pi",
     kl.length === 13 && free.length === 12 * 2 * f169.nblobs[2] && free.every(b => b.x >= 0 && b.x < f169.box[0] && b.y >= 0 && b.y < L) &&
     free.filter(b => b.x > L).length > 0.3 * free.length && free.filter(b => b.x > 0.9 * f169.box[0]).length > 0, kl.length + " kicks");
  ok("  ... and a placed one (pan 0.35, gain 2) gathers about x = Lx (0.5 + 0.4 * 0.7) = " + xc.toFixed(2) + ", which is beyond the square box's whole width: mean x " + mx.toFixed(2) + ", half the amplitude at strength 0.5",
     placedK.length === 2 * f169.nblobs[1] && xc > L && Math.abs(mx - xc) < 0.12 * f169.box[0] && placedK.every(b => Math.abs(b.amp) === 0.5 && b.sigma === q.plan.sigmas[1]), "xc " + xc.toFixed(3));
  // a resize mid-run rebuilds nothing; the new shape is taken at the next Reset
  mark(env);
  setWin(env, 390, 844);
  env.run("function(){ cardsSync(); }");
  await P.frame(10.08);
  await P.frame(10.10);
  q = snap(env);
  ok("a window turned to portrait MID-RUN rebuilds nothing: the same Solver, still 16:9, the card still 512 x 288 -- and the line says the new shape waits for Start or Reset",
     same(env) === true && q.p.Lx === f169.box[0] && q.p.Ly === L && q.cv.join() === "512,288" && q.grave === 0 && /taken at the next Start or Reset/.test(q.box), q.box.slice(0, 110));
  await P.click("btnReset");
  await P.frame(10.12);
  q = snap(env);
  const bP = M.musicBoxFor(390 / 844);
  ok("  ... Reset takes it: a tall box 2 pi x 2 pi (844/390), still running, the card 236 x 512, standard is 256 x 256 either way round",
     same(env) === false && q.p.Lx === L && q.p.Ly === bP[1] && q.p.Ly > 2 * L && q.p.nx === 256 && q.p.ny === 256 && q.cv.join() === "236,512" &&
     env.run("function(){ return running; }") === true && !/taken at the next/.test(q.box), q.p.Lx.toFixed(3) + " x " + q.p.Ly.toFixed(3) + ", card " + q.cv.join("x"));
  const detN = JSON.parse(env.run("function(){ return JSON.stringify([music.det.nbands, music.plan.sigmas.length, music.lit.length, music.det.sr, music.det.threshold, music.det.fluxFloor, music.queue.length]); }"));
  ok("  ... the tall box has 2 bands where 16:9 had 3, and the RUNNING session's detector was replaced by one with that many (same rate, threshold and noise floor; nothing queued from the old one), so no onset can name a band the plan has not got",
     detN[0] === 2 && detN[1] === 2 && detN[2] === 2 && f169.sigmas.length === 3 && detN[3] === 44100 && detN[4] === 0.3 && detN[5] === 0.3 && detN[6] === 0, detN.join());
  await P.click("btnRun");
  await P.input("selGrid", "light");
  q = snap(env);
  ok("grid = light in portrait: 128 x 256 (ny > nx: the long side of the grid lies along the long side of the box)", q.p.nx === 128 && q.p.ny === 256 && q.sh.rung === "light" && /on 128 × 256 \(light\)/.test(q.box), q.box);
  ok("  ... its band plan is the rule's for THAT grid and box (both spacings: here the long side's is the larger), and its dissipation sits at the SMALLER cutoff, which here is the long side's (256/3 * 2 pi / Ly < 128/3)",
     JSON.stringify(q.plan) === JSON.stringify(M.musicBandPlan([128, 256], null, null, 32, L, bP[1])) && last(q.plan.sigmas) === 3.7 * bP[1] / 256 &&
     JSON.stringify(q.plan) !== JSON.stringify(M.musicBandPlan([128, 128], null, null, 32, L, bP[1])) && q.p.diss === 48 / Math.pow((256 / 3) * (L / bP[1]), 8) && (256 / 3) * (L / bP[1]) < 128 / 3, "sigma_min " + last(q.plan.sigmas).toFixed(4));
  await P.input("selGrid", "sharp");
  q = snap(env);
  const kcP = M.musicCutoffK(256, 512, L, bP[1]);
  ok("grid = sharp in portrait: 256 x 512, the dissipation at the LONG side's cutoff (512/3 * 2 pi / Ly = " + kcP.toFixed(1) + " < 256/3), each axis with its own length",
     q.p.nx === 256 && q.p.ny === 512 && /on 256 × 512 \(sharp\)/.test(q.box) && q.p.diss === 48 / Math.pow(kcP, 8) && kcP === (512 / 3) * (L / bP[1]) && kcP < 256 / 3 &&
     JSON.stringify(q.plan) === JSON.stringify(M.musicBandPlan([256, 512], null, null, 32, L, bP[1])), q.box + "; diss " + q.p.diss.toExponential(3));
  setWin(env, 3440, 1000);
  await P.click("btnReset");
  q = snap(env);
  ok("a 3440 x 1000 window: the aspect is CLAMPED at 2.4 (Lx = 2.4 * 2 pi), sharp is 512 x 256, and Start after a Reset on the same window does not rebuild again",
     q.p.Lx === 2.4 * L && q.p.Ly === L && q.p.nx === 512 && q.p.ny === 256 &&
     (mark(env), P.A.t = 20, true) && (await P.click("btnRun"), same(env) === true) && (await P.click("btnRun"), true));
  setWin(env, 1000, 3000);
  P.A.t = 30;
  await P.click("btnRun");
  q = snap(env);
  ok("  ... a window changed while STOPPED is taken by Start: 1000 x 3000 -> the tall clamp, 2 pi x 2.4 * 2 pi on 256 x 512, and the session starts on it",
     same(env) === false && q.p.Lx === L && rel(q.p.Ly, 2.4 * L) < 1e-15 && q.p.nx === 256 && q.p.ny === 512 && env.run("function(){ return running && !!music.audio; }") === true);
  await P.click("btnRun");
  await P.input("selGrid", "standard");
  await P.input("selBox", "square");
  q = snap(env);
  ok("box = square again, grid = standard: back to the shipped parameter set exactly, the card 512 x 512",
     JSON.stringify(q.ui) === shipped && q.cv.join() === "512,512" && JSON.stringify(q.plan) === JSON.stringify(M.musicBandPlan(256, null, null, 32)));
  await P.input("selGrid", "light");
  const sqL = snap(env);
  await P.input("selGrid", "sharp");
  q = snap(env);
  ok("on the SQUARE box light is 128 x 128 and sharp is 256 x 256, the same as standard (the tooltip says so)",
     sqL.p.nx === 128 && sqL.p.ny === 128 && q.p.nx === 256 && q.p.ny === 256 && JSON.stringify(q.ui) === shipped &&
     /SQUARE box[^.]*128 × 128[^.]*sharp is the same 256 × 256 as standard/.test(env.getEl("selGrid").title), sqL.p.nx + "^2 and " + q.p.nx + "^2");
  await P.input("selGrid", "auto");
  await P.frame();

  // one axis of the grid at a time, on one box: each is a rebuild
  setWin(env, 1920, 1080);
  for (const [id, v] of [["selBox", "window"], ["selGrid", "standard"]]) await P.input(id, v);
  const steps = [];
  for (const g of ["light", "standard", "sharp", "standard"]) {
    mark(env);
    await P.input("selGrid", g);
    const s = snap(env);
    steps.push(g + " " + s.p.nx + "x" + s.p.ny + " " + (same(env) ? "same" : "rebuilt") + " " + (s.p.Lx === f169.box[0] && s.p.Ly === L ? "16:9" : "?"));
  }
  ok("on one 16:9 box the grid select alone rebuilds, one axis at a time: standard -> light changes ny only (256 x 128), -> standard, -> sharp changes nx only (512 x 256), and back",
     steps.join() === "light 256x128 rebuilt 16:9,standard 256x256 rebuilt 16:9,sharp 512x256 rebuilt 16:9,standard 256x256 rebuilt 16:9", steps.join(" | "));
  await P.input("selGrid", "auto");
  await P.frame();

  // ---- the background select ------------------------------------------------------------------
  const flags = [];
  for (const box of ["square", "window"]) for (const grid of ["auto", "light", "standard", "sharp"]) for (const bg of ["field", "sheets", "none"]) {
    setWin(env, 1600, 900);
    env.getEl("selBox").value = box; env.getEl("selGrid").value = grid; env.getEl("selBg").value = bg;
    env.getEl("selBg").onchange({});
    await turn(4);
    const s = snap(env);
    flags.push({ box: box, grid: grid, bg: bg, b0: s.p.b0, eq: s.p.eqsrc, drag: s.p.eqsrcDrag, mu: s.p.mu, hasB0: s.hasB0, hasEqk: s.hasEqk, n: s.p.nx + "x" + s.p.ny });
  }
  await P.frame();
  const wantN = { square: { auto: "256x256", light: "128x128", standard: "256x256", sharp: "256x256" }, window: { auto: "256x256", light: "256x128", standard: "256x256", sharp: "512x256" } };
  ok("  (every one of those selections is taken at once, each on its own grid: a change of ny alone -- 256 x 256 to 256 x 128 on the same box -- is a rebuild too)",
     flags.every(f => f.n === wantN[f.box][f.grid]), flags.filter(f => f.n !== wantN[f.box][f.grid]).map(f => f.box + "/" + f.grid + " " + f.n).join(", "));
  ok("the background builds its own kernel flag and never both: over every box x grid x background (" + flags.length + " solvers) the mean field and `none` are b0: true, eqsrc: false with no eqsrcDrag key; the sheets are b0: false, eqsrc: true, eqsrcDrag: true",
     flags.length === 24 && flags.every(f => !(f.b0 && f.eq) && f.mu === 1 &&
       (f.bg === "sheets" ? f.b0 === false && f.eq === true && f.drag === true && !f.hasB0 && f.hasEqk
                          : f.b0 === true && f.eq === false && f.drag === undefined && f.hasB0 && !f.hasEqk)),
     flags.filter(f => f.b0 && f.eq).length + " with both; grids " + Array.from(new Set(flags.map(f => f.n))).join(" "));
  // none: the shipped graph with the field held at zero, whatever its sliders say
  setWin(env, 1920, 1080);
  for (const [id, v] of [["selBox", "square"], ["selGrid", "auto"], ["selBg", "field"]]) await P.input(id, v);
  await P.frame();
  mark(env);
  await P.input("selBg", "none");
  ok("mean field -> none is a rebuild too (as a Reset), although the two build the same kernels: a new Solver, and the manifest says `none`",
     same(env) === false && snap(env).sh.bg === "none" && env.run("function(){ return runManifest().forcing.background; }") === "none" && JSON.stringify(snap(env).ui) === shipped);
  env.getEl("rField").value = "1.5";
  model(env);
  P.A.t = 40;
  await P.click("btnRun");
  for (let i = 1; i <= 6; i++) await P.frame(40 + 0.02 * i);
  q = snap(env);
  const walkNone = env.run("function(){ return music.walk; }");
  ok("background = none: the shipped parameter set (b0: true, so the default graph), the mean field HELD at zero while running although its slider says 1.5, no walk, its controls and the sheets' disabled",
     JSON.stringify(q.ui) === shipped && q.b0.join() === "0,0,0,0" && walkNone === null && q.dis.join() === "true,true,true,true,true" &&
     env.run("function(){ return musicUi().field; }") === 0 && env.run("function(){ return runManifest().forcing.effects.meanField; }") === 0);
  ok("  ... and on `none`, as on the mean field, the fixed colour scale's reference is amp", env.run("function(){ const u = musicUi(); return musicScaleRef(u) === u.amp && musicScaleMax(u) === u.scaleM * u.amp && /× amp\\)/.test(musicScaleReadout()); }") === true,
     env.run("function(){ return musicScaleReadout(); }"));
  await P.input("selBg", "field");
  model(env);                            // (a new solver: the model readback goes on it again)
  for (let i = 7; i <= 12; i++) await P.frame(40 + 0.02 * i);
  q = snap(env);
  ok("  ... and back on the walking mean field (a rebuild, mid-run) the field is on again: B0 = r sqrt(2E) != 0, a walk exists",
     Math.hypot(q.b0[0], q.b0[1]) > 0 && env.run("function(){ return !!music.walk && running; }") === true && q.dis.join() === "true,true,false,false,false", "B0 " + Math.hypot(q.b0[0], q.b0[1]).toFixed(3));
  await P.click("btnRun");
  env.getEl("rField").value = "0";

  // ---- the current sheets -----------------------------------------------------------------------
  // every initial condition any Solver is given from here on, in order
  env.run(`function(){ const f = Solver.prototype.setICFromReal, g = Solver.prototype.setIC; globalThis.__ic = [];
    Solver.prototype.setICFromReal = function(a, b){ const nx = this.p.nx, ny = this.p.ny; let flat = !!b && b.length === nx * ny;
      for (let i = 0; flat && i < nx; i++) for (let j = 1; j < ny; j++) if (b[i * ny + j] !== b[i * ny]) flat = false;
      __ic.push({ k: "real", phi: a === null, flat: flat, col: b ? Array.from({ length: nx }, (_, i) => b[i * ny]) : null, live: this === solver, n: [nx, ny] });
      return f.call(this, a, b); };
    Solver.prototype.setIC = function(z){ __ic.push({ k: "builtin", zero: z === true }); return g.call(this, z); }; }`);
  const ics = () => JSON.parse(env.run("function(){ const l = __ic; globalThis.__ic = []; return JSON.stringify(l); }"));
  const fs169 = F.sheets.find(c => c.strength === 5 && c.width === 2 && c.amp === 1 && c.aspect > 1);
  setWin(env, 1920, 1080);
  await P.input("selBox", "window");
  ics();
  env.gpuReset();
  await P.input("selBg", "sheets");
  await P.frame();
  let ic = ics();
  q = snap(env);
  const colErr = (col, c, scale) => { let d = 0; for (let k = 0; k < c.psi.length; k++) d = Math.max(d, Math.abs(col[k * c.stride] - Math.fround(scale * c.psi[k]))); return d / (scale * c.psi0); };
  const sheet = () => JSON.parse(env.run("function(){ return JSON.stringify(music.sheet); }"));
  const realIC = ic.filter(c => c.k === "real");
  ok("background = current sheets (16:9, standard, the defaults): a rebuild whose initial condition is setICFromReal(null, psi_eq) on the NEW solver -- phi null, psi independent of y -- and it is the offline bump: a = " + fs169.a.toFixed(4) + ", psi0 = " + fs169.psi0.toFixed(4),
     realIC.length === 1 && realIC[0].phi === true && realIC[0].flat === true && realIC[0].live === true && realIC[0].n.join() === "256,256" &&
     colErr(realIC[0].col, fs169, 1) < 1e-7 && rel(sheet().a, fs169.a) < 1e-14 && rel(sheet().psi0, fs169.psi0) < 1e-14 && rel(sheet().energy, fs169.energy) < 1e-14 &&
     ic.filter(c => c.k === "builtin" && c.zero).length === 0, ic.map(c => c.k).join(",") + "; psi " + e2(colErr(realIC[0] ? realIC[0].col : [], fs169, 1)) + " of psi0");
  const si = env.gpu.dispatches.map((d, i) => d.pipe.__name === "srcInit" ? i : -1).filter(i => i >= 0);
  ok("  ... that upload is when the solver takes its maintained equilibrium: `srcInit` is dispatched with it, over ceil(NM / 64) (twice in a rebuild: the constructor's own built-in IC first, then the sheets, whose column is the one that stands)",
     si.length === 2 && si.every(i => env.gpu.dispatches[i].d[0] === Math.ceil(256 * 129 / 64)) && q.hasEqk &&
     ic.map(c => c.k).join() === "builtin,real", ic.map(c => c.k + (c.zero ? "(zero)" : "")).join(","));
  ok("  ... the card switches to a magnetic field, |b| (field 5), the sheet sliders come alive and the mean-field ones go dead, and the line states the cost: the peak field, a, the energy, and that sheets this strong shorten the timestep",
     q.sel === "5" && q.dis.join() === "false,false,true,true,true" && /current sheets: peak field 5\.00, width a = 0\.323, energy /.test(q.box) &&
     /shorten the timestep \(about 1\.3-1\.5 times the steps at 5, twice at 8\)/.test(q.box) && env.run("function(){ return solver.modeOf(0); }") === 5, q.box.slice(40));
  // the fixed colour scale: on the sheets its reference is the sheet's own peak IN THE FIELD SHOWN,
  // over 1.5 -- and still amp for a field the sheet is not part of (2026-11-24)
  const sc = () => JSON.parse(env.run(`function(){ const u = musicUi(), c = cards.disp[0]; return JSON.stringify({ m: u.scaleM, amp: u.amp, ref: musicScaleRef(u), max: musicScaleMax(u),
    bar: c.barMax, dirty: c.dirty, v: el("vScale").textContent, read: musicScaleReadout(), rec: musicScaleRecord().max, b: music.sheet ? music.sheet.b : null, sh: music.sheet, mode: solver.modeOf(c.ci),
    all: Array.from({ length: 12 }, (_, m) => musicScaleRef({ amp: 0.7, scaleM: 2 }, m)), all2: Array.from({ length: 12 }, (_, m) => musicScaleMax({ amp: 0.7, scaleM: 2 }, m)) }); }`));
  const sc5 = sc();
  ok("  ... and the fixed colour scale changes its reference: on the sheets, for the field the card shows there (|b|), it is the sheet's PEAK FIELD / 1.5 (5 / 1.5 here), not amp, so the slider's default 1.5 puts the top of the scale on the peak itself -- 5.00, to the slider's own rounding of 1.5 (2^0.585)",
     sc5.b === 5 && sc5.amp === 1 && sc5.ref === 5 / 1.5 && sc5.max === sc5.m * (5 / 1.5) && Math.abs(sc5.m - Math.pow(2, 0.585)) < 1e-12 && Math.abs(sc5.max / 5 - 1) < 1e-4 && sc5.rec === sc5.max,
     JSON.stringify([sc5.ref, sc5.max]));
  ok("  ... the slider's label and the readout say so: \"1.50 → 5.00\", \"colour scale fixed at 5.00 (1.50 × the sheets' peak in this field / 1.5)\"",
     sc5.v === "1.50 → 5.00" && /^colour scale fixed at 5\.00 \(1\.50 × the sheets' peak in this field \/ 1\.5\)/.test(sc5.read) && !/× amp/.test(sc5.read), sc5.v + " | " + sc5.read);
  // the reference per displayed field, from the sheet as laid (psi = psi0 sech^2((x - Lx/2) / a), peak field b):
  // this file's own arithmetic on the offline fixture's psi0 and a, with an amp (0.7) no peak could be mistaken for
  const refWant = (m, sh, amp) => (m === 5 || m === 6 || m === 7) ? sh.b / 1.5 : m === 3 ? Math.abs(sh.psi0) / 1.5 : (m === 1 || m === 10 || m === 11) ? 2 * Math.abs(sh.psi0) / (sh.a * sh.a) / 1.5 : amp;
  const fxSheet = { b: 5, psi0: fs169.psi0, a: fs169.a };
  const NAMES = ["vorticity", "j", "phi", "psi", "|u|", "|b|", "|z+|", "|z-|", "sigma_c", "sigma_r", "omega+", "omega-"];
  const refBad = sc5.all.map((v, m) => (rel(v, refWant(m, fxSheet, 0.7)) < 1e-13 && v === refWant(m, sc5.sh, 0.7) && sc5.all2[m] === 2 * v ? "" : NAMES[m] + " " + v)).filter(x => x);
  ok("on the sheets the reference depends on the FIELD SHOWN, all twelve display modes: |b|, |z+|, |z-| take the peak field / 1.5 (" + (5 / 1.5).toFixed(3) + "); psi takes psi0 / 1.5 (" + (fs169.psi0 / 1.5).toFixed(3) +
     "); j, omega+, omega- take the current at the centre, 2 psi0 / a^2, / 1.5 (" + (2 * fs169.psi0 / (fs169.a * fs169.a) / 1.5).toFixed(2) + "); the vorticity, phi, |u| and the sigma modes -- fields the sheet is not in -- keep amp",
     refBad.length === 0 && sc5.all.length === 12 && sc5.all[4] === 0.7 && sc5.all[0] === 0.7 && sc5.all[2] === 0.7 && sc5.all[8] === 0.7 && sc5.all[9] === 0.7 &&
     new Set([sc5.all[5], sc5.all[3], sc5.all[1], 0.7]).size === 4 && sc5.all[1] > 5 * sc5.all[5], refBad.join("; ") || sc5.all.map(v => v.toPrecision(4)).join(" "));
  ok("  ... with no mode handed in it is the field the first card is showing (musicScaleMode: the live chain's): |b| here, so musicScaleRef(u) is the |b| entry of that table",
     sc5.mode === 5 && sc5.ref === refWant(5, sc5.sh, 1) && env.run("function(){ return musicScaleMode() === solver.modeOf(cards.disp[0].ci) && musicScaleRef(musicUi()) === musicScaleRef(musicUi(), 5); }") === true);
  // switching the card's FIELD, and nothing else: the next render writes that field's range to maxVal,
  // the colour bar takes it and the slider's label follows
  env.run(`function(){ window.__mw = []; window.__bars = []; const q = device.queue, wb = q.writeBuffer.bind(q); window.__wb0 = q.writeBuffer;
    q.writeBuffer = (b, off, data) => { if (solver && solver.disp[0] && b === solver.disp[0].buf.maxVal) window.__mw.push(Array.from(new Float32Array(data.buffer || data))[0]); return wb(b, off, data); };
    const c = cards.disp[0], sb = c.setBarRange; window.__sb0 = sb; c.setBarRange = function(v){ window.__bars.push(v); return sb.call(this, v); }; }`);
  const pickF = async v => { env.run(`function(v){ window.__mw.length = 0; window.__bars.length = 0; const c = cards.disp[0]; c.selField.value = v; c.selField.onchange({}); }`, String(v));
    await P.frame();
    return JSON.parse(env.run(`function(){ const u = musicUi(); return JSON.stringify({ mode: solver.modeOf(0), w: window.__mw, bars: window.__bars, v: el("vScale").textContent, m: u.scaleM, read: musicScaleReadout(),
      fixed: solver.disp[0].musicFixed, auto: solver.disp[0].bg.maxFinal === solver.disp[0].musicBg.auto }); }`)); };
  const swBad = [], swLog = [];
  for (const m of [4, 1, 3, 6, 2, 10, 5]) {
    const r = await pickF(m), want = r.m * refWant(m, sc5.sh, 1);
    swLog.push(NAMES[m] + " " + r.v);
    if (!(r.mode === m && r.w.length >= 1 && r.w[r.w.length - 1] === Math.fround(want) && r.bars.indexOf(want) >= 0 && r.v === "1.50 → " + want.toPrecision(3) && r.fixed === true &&
          (refWant(m, sc5.sh, 1) === 1 ? /× amp\)/ : /× the sheets' peak in this field \/ 1\.5\)/).test(r.read) && r.read.indexOf("fixed at " + want.toPrecision(3)) >= 0)) swBad.push(NAMES[m] + ": " + JSON.stringify(r));
  }
  ok("switching the card's field on the sheets, and touching nothing else (|b| -> |u| -> j -> psi -> |z+| -> phi -> omega+ -> back to |b|): the next render writes THAT field's range to maxVal, the colour bar is set to it, and the slider's label and the readout follow -- |u| and phi on amp (\"1.50 → 1.50\", \"× amp\"), the rest on the sheet's peak in that field",
     swBad.length === 0 && swLog[0] === "|u| 1.50 → 1.50" && swLog[6] === "|b| 1.50 → 5.00" && swLog[1] === "j 1.50 → " + (sc5.m * 2 * fs169.psi0 / (fs169.a * fs169.a) / 1.5).toPrecision(3), swBad.join(" ; ") || swLog.join(" | "));
  const sgm = JSON.parse(env.run(`function(){ const D = solver.disp[0], c = cards.disp[0], u = musicUi(), out = [];
    for (const m of [8, 9]) { window.__mw.length = 0; window.__bars.length = 0; D.mode = m; solver.render(c.ctx, 0);
      out.push([dispIsSigma(m), window.__mw.length, window.__bars.length, D.musicFixed, D.bg.maxFinal === D.musicBg.auto, musicScaleRef(u, m) === u.amp]); }
    window.__mw.length = 0; D.mode = 5; solver.render(c.ctx, 0); out.push(window.__mw.slice());
    device.queue.writeBuffer = window.__wb0; c.setBarRange = window.__sb0; return JSON.stringify(out); }`));
  ok("  ... and the sigma modes are still left alone on the sheets: sigma_c and sigma_r write nothing to maxVal, set no bar range and render through the autoscale's own bind group (their fixed -1..+1); back on |b| the sheet's range is written again",
     sgm.slice(0, 2).every(r => r.join() === "true,0,0,false,true,true") && sgm[2].length === 1 && sgm[2][0] === Math.fround(sc5.max) && realFails(env).length === 0, JSON.stringify(sgm));
  const tip = env.getEl("selBg").title;
  ok("  ... the tooltip is honest about it: ONE current sheet through the centre with weaker return currents (not a pair), hyper 4 has no resistive layer, this is NOT the tearing instability but a restored sheet broken up by kicks, and rmhd2d.html has the real one; the sliders' say a change waits for the next Reset",
     /no resistive layer/.test(tip) && /NOT the tearing instability/.test(tip) && /it is a restored sheet broken up by kicks/.test(tip) && /with weaker return currents/.test(tip) && !/\bpair\b/.test(tip) && /rmhd2d\.html/.test(tip) &&
     /next Reset/.test(env.getEl("rSheetB").title) && /next Reset/.test(env.getEl("rSheetW").title) && /shorten the timestep/.test(env.getEl("rSheetB").title));
  // the sliders do nothing until Reset; Reset restores the sheets
  mark(env);
  await P.input("rSheetB", 8);
  await P.input("rSheetW", 1);
  await P.input("rMAmp", 2);
  const quiet = ics();
  const scQ = sc();
  env.gpuReset();
  await P.click("btnReset");
  const sc16 = sc();                               // (read before any frame: the stub's readbacks return zeros)
  ic = ics();
  ok("the scale's reference is the sheet AS LAID DOWN: with the sliders at 8 and amp 2 but no Reset yet it is still 5 / 1.5 for |b| (the amp slider does not move it here), and Reset, which lays a peak field of 16, makes it 16 / 1.5 -- the label, the readout and the card's colour bar at once (\"1.50 → 16.0\")",
     scQ.b === 5 && scQ.amp === 2 && scQ.ref === 5 / 1.5 && scQ.max === sc5.max && scQ.v === "1.50 → 5.00" &&
     sc16.b === 16 && sc16.ref === 16 / 1.5 && sc16.max === sc16.m * (16 / 1.5) && sc16.v === "1.50 → 16.0" && sc16.bar === sc16.max && sc16.dirty === true &&
     /^colour scale fixed at 16\.0 \(1\.50 × the sheets' peak in this field \/ 1\.5\)/.test(sc16.read) &&
     scQ.all[4] === 0.7 && scQ.all[5] === 5 / 1.5 && sc16.all[5] === 16 / 1.5 && sc16.all[3] === Math.abs(sc16.sh.psi0) / 1.5 && sc16.all[1] === 2 * Math.abs(sc16.sh.psi0) / (sc16.sh.a * sc16.sh.a) / 1.5 && sc16.all[4] === 0.7, JSON.stringify([scQ.max, sc16.max, sc16.bar, sc16.v]));
  const s8 = sheet(), a8 = last(q.plan.sigmas), want8 = M.musicSheetPsi0(8, 2, a8);
  ok("sheet field 8, width 1, amp 2: moving the sliders uploads NOTHING (" + quiet.length + " initial conditions); they take effect at Reset",
     quiet.length === 0 && rel(sheet().psi0, want8) < 1e-14 && rel(s8.a, a8) < 1e-14 && s8.b === 16);
  ok("Reset RESTORES the sheets -- one setICFromReal(null, psi_eq) on the same solver, never the blank setIC(true) -- with the equilibrium taken again (srcInit), psi0 = (3 sqrt(3)/4) a * 8 * 2",
     same(env) === true && ic.length === 1 && ic[0].k === "real" && ic[0].phi && ic[0].flat && ic[0].live &&
     Math.abs(ic[0].col[128] - Math.fround(M.musicSheetEq(128 * f169.box[0] / 256, f169.box[0], want8, a8))) < 1e-6 * want8 && Math.abs(ic[0].col[128] / want8 - 1) < 1e-6 &&
     env.gpu.dispatches.filter(d => d.pipe.__name === "srcInit").length === 1, ic.map(c => c.k).join(",") + "; centre " + (ic[0] && ic[0].col[128]) + " for psi0 " + want8.toFixed(4));
  for (const [id, v] of [["rSheetB", 5], ["rSheetW", 2], ["rMAmp", 1]]) await P.input(id, v);
  await P.click("btnReset");
  ic = ics();
  ok("  ... and back at the defaults the next Reset lays the first profile again, bit for bit", ic.length === 1 && JSON.stringify(ic[0].col) === JSON.stringify(realIC[0].col));
  // strength 0 is still the sheets' solver, with nothing in it
  await P.input("rSheetB", 0);
  await P.click("btnReset");
  ic = ics();
  ok("sheet field 0: still the sheets' solver (eqsrc), an all-zero psi, and the line drops the cost note",
     ic.length === 1 && ic[0].k === "real" && ic[0].col.every(v => v === 0) && snap(env).p.eqsrc === true && !/shorten the timestep/.test(snap(env).box) && sheet().energy === 0);
  const sc0 = sc();
  ok("  ... and with no sheet to refer to (peak field 0) the colour scale falls back to amp, for EVERY displayed field: 1.50 × amp, and the readout says amp again",
     sc0.b === 0 && sc0.amp === 1 && sc0.ref === 1 && sc0.max === sc0.m && sc0.v === "1.50 → 1.50" && /^colour scale fixed at 1\.50 \(1\.50 × amp\)/.test(sc0.read) && sc0.all.every(v => v === 0.7) && sc0.all2.every(v => v === 1.4), sc0.v + " | " + sc0.read);
  await P.input("rSheetB", 5);
  await P.click("btnReset");
  ics();
  // the memory slider on a live solver with the source on
  const SRC = `function(){ const s = solver, b = s.bg.nlAssemble.__buffers; return JSON.stringify({ mu: s.p.mu, diss: s.p.diss, drag: s.p.eqsrcDrag, eq: s.p.eqsrc,
    lin0: s.g.gridB[1], lin1: s.g.gridB[5], k1: s.g.gridA[6], gridB: b.indexOf(s.buf.gridB), eqk: b.indexOf(s.buf.eqk), n: b.length }); }`;
  const dev = env.run("function(){ return device; }"), wq = dev.queue, worig = wq.writeBuffer, wrote = [];
  wq.writeBuffer = function (b, off, data) { wrote.push(b); return worig.apply(wq, arguments); };
  let memErr = "", v2 = null, gb = null;
  try {
    await P.input("rMem", 2);
    v2 = JSON.parse(env.run(SRC));
    gb = env.run("function(){ return solver.buf.gridB; }");
  } catch (e) { memErr = String(e && e.message || e); } finally { wq.writeBuffer = worig; }
  const f169d = f169.diss[0][1];
  ok("the memory slider on the LIVE sheets solver (setDrag with the source on) does not throw: mu = 1/2, the drag is in the rebuilt column (-1/4 at k = 0, -diss k^8 - 1/4 beside it) and that column is uploaded",
     memErr === "" && !!v2 && v2.mu === 0.5 && v2.lin0 === -0.25 && Math.abs(v2.lin1 - Math.fround(Math.fround(-f169d * Math.pow(v2.k1, 4)) - 0.25)) < 1e-9 &&
     wrote.indexOf(gb) >= 0 && same(env) === true && realFails(env).length === 0, memErr || "mu " + (v2 && v2.mu) + ", lin(k=0) " + (v2 && v2.lin0));
  ok("  ... and the sheets stay HELD by construction: the source kernel (nlAssemble) is bound to that same gridB buffer and to eqk, so it reads the new column -- -gridB.y * psi_eq cancels the new decay rate exactly -- and no initial condition was re-uploaded (eqk is still the Reset's)",
     !!v2 && v2.gridB >= 0 && v2.eqk >= 0 && v2.eqk === v2.n - 1 && v2.drag === true && v2.eq === true && ics().length === 0, JSON.stringify(v2));
  env.getEl("selBg").value = "field";              // the select moved, its handler not yet run
  const inForce = JSON.parse(env.run("function(){ return JSON.stringify(uiParams()); }"));
  env.getEl("selBg").value = "sheets";
  ok("  ... uiParams() describes the solver IN FORCE (music.shape), not a select that has moved since: still eqsrc + eqsrcDrag, no b0",
     inForce.eqsrc === true && inForce.eqsrcDrag === true && inForce.b0 === false);
  const noOpt = throws(() => env.run("function(){ const p = Object.assign({}, uiParams()); delete p.eqsrcDrag; new Solver(device, p); }"));
  ok("  ... which rests on the opt-in: the page's own parameters WITHOUT eqsrcDrag are refused by the solver", /eqsrcDrag/.test(noOpt), noOpt.slice(0, 80));
  await P.input("rMem", 1);
  ics();
  // running on the sheets: kicks land, steps are taken, the missing mean-field uniform is never touched
  model(env);
  P.A.t = 60;
  await P.click("btnRun");
  for (let i = 1; i <= 40; i++) await P.frame(60 + i / 60);
  env.run(`function(){ const t = music.audio.ctx.currentTime - music.audio.t0; music.queue.push({ due: t, o: { band: 0, strength: 1, t: t, tFire: t }, pan: null }); }`);
  for (let i = 41; i <= 44; i++) await P.frame(60 + i / 60);
  const lg = P.takeLog();
  ok("a session on the sheets runs: steps are taken, a bass onset kicks (and pulses), no mean field is asked of a solver that has none, and nothing is logged as a failure",
     lg.filter(l => l.op === "step").length > 20 && lg.filter(l => l.op === "kick").length >= 1 && env.run("function(){ return running && music.walk === null && !solver.buf.b0; }") === true &&
     realFails(env).length === 0 && !/non-finite|could not/.test(env.getEl("status").textContent), realFails(env).slice(0, 2).join(" | ") || lg.filter(l => l.op === "step").length + " steps");
  await P.click("btnReset");
  ic = ics();
  ok("  ... and Reset mid-run restores the sheets there too (the kicks are gone, the equilibrium is back)", ic.length === 1 && ic[0].k === "real" && JSON.stringify(ic[0].col) === JSON.stringify(realIC[0].col));
  await P.click("btnRun");
  setWin(env, 2000, 1000);
  await P.click("btnReset");
  ic = ics();
  setWin(env, 1920, 1080);
  await P.click("btnReset");
  ok("  ... a Reset that REBUILDS (the window became 2:1) lays the sheets once, on the new box -- the constructor's built-in IC, then one setICFromReal on a 4 pi-long box, not two",
     ic.map(c => c.k).join() === "builtin,real" && ic[1].live && snap(env).p.Lx === f169.box[0] && Math.abs(ic[1].col[128] / M.musicSheetPsi0(5, 1, 2 * last(M.musicBandPlan([256, 256], null, null, 32, 2 * L, L).sigmas)) - 1) < 1e-6 &&
     ics().map(c => c.k).join() === "builtin,real", ic.map(c => c.k).join());
  // the card's field: set by a change of background, and by nothing else
  const pick = v => env.run(`function(v){ const c = cards.disp[0]; c.selField.value = v; c.selField.onchange({}); }`, v);
  const fld = () => snap(env).sel + "/" + env.run("function(){ return solver.modeOf(0); }");
  const seq = [];
  await P.input("selBg", "field");
  const scF = sc();                                // (before any frame, as above)
  seq.push(fld());                                 // sheets -> mean field: the page's default, |u|
  await P.input("selBg", "sheets");
  const scS = sc();
  seq.push(fld());                                 // -> sheets: |b|
  ok("the colour scale's reference follows the BACKGROUND select at once, the card's colour bar with it: leaving the sheets it is amp again (1.50 × amp = 1.50) -- for all twelve display modes, as it is off the sheets -- and choosing them it is the new sheet's peak in the field shown / 1.5 (|b|: top of the scale 5.00)",
     scF.ref === scF.amp && scF.amp === 1 && scF.max === scF.m && scF.bar === scF.max && /× amp\)/.test(scF.read) && scF.all.every(v => v === 0.7) && scF.all2.every(v => v === 1.4) &&
     scS.b === 5 && scS.mode === 5 && scS.ref === 5 / 1.5 && scS.max === scS.m * (5 / 1.5) && scS.bar === scS.max && /× the sheets' peak in this field \/ 1\.5\)/.test(scS.read) && scS.all[4] === 0.7,
     JSON.stringify([scF.max, scF.bar, scS.max, scS.bar]));
  pick("1");                                       // the visitor picks the current
  await P.click("btnReset");
  const scJ = sc();                                // (before any frame)
  ok("a field the visitor picks on the sheets keeps ITS reference through Reset: the current, then Reset -- the scale is 1.50 × (2 psi0 / a^2) / 1.5 of the sheet just laid, on the colour bar and the label at once, and the readout says the sheets' peak in this field",
     scJ.mode === 1 && scJ.ref === 2 * Math.abs(scJ.sh.psi0) / (scJ.sh.a * scJ.sh.a) / 1.5 && rel(scJ.ref, 2 * fs169.psi0 / (fs169.a * fs169.a) / 1.5) < 1e-13 && scJ.max === scJ.m * scJ.ref &&
     scJ.bar === scJ.max && scJ.v === "1.50 → " + scJ.max.toPrecision(3) && /× the sheets' peak in this field \/ 1\.5\)/.test(scJ.read) && scJ.max > 5 * scS.max, JSON.stringify([scJ.ref, scJ.bar, scJ.v]));
  seq.push(fld());                                 // ... which survives Reset
  await P.input("selGrid", "light");
  await P.input("selGrid", "auto");
  seq.push(fld());                                 // ... and two rebuilds that are not a change of background
  model(env);
  P.A.t = 80;
  await P.click("btnRun");
  for (let i = 1; i <= 5; i++) await P.frame(80 + 0.02 * i);
  await P.click("btnRun");
  seq.push(fld());                                 // ... and a Start, frames, a Stop
  await P.input("selBg", "none");
  seq.push(fld());                                 // leaving the sheets: the default again
  pick("3");
  await P.input("selBg", "field");
  seq.push(fld());                                 // none -> mean field: no sheets either side, the pick (psi) stays
  await P.click("btnReset");
  seq.push(fld());
  await P.input("selBg", "sheets");
  seq.push(fld());                                 // -> sheets again: |b| again
  ics();
  ok("on the current sheets the picture shows the MAGNETIC FIELD by default: choosing them puts |b| (field 5) on the card and on the solver, going back to the mean field puts the page's default |u| (4) back",
     seq[0] === "4/4" && seq[1] === "5/5" && seq[8] === "5/5" && env.run("function(){ return [musicBgField('sheets'), musicBgField('field'), musicBgField('none')].join(); }") === "5,4,4" &&
     /cardsLayout\(\{ disp: \[\{ sel: musicBgField\(music\.shape\.bg\), cmap: MUSIC_CMAP_DEFAULT/.test(htmlS), seq.join(" "));
  ok("  ... only the CHANGE of background sets it: a field the visitor picks afterwards (the current, 1) survives Reset, two rebuilds from the grid select, and a Start / frames / Stop",
     seq[2] === "1/1" && seq[3] === "1/1" && seq[4] === "1/1", seq.slice(2, 5).join(" "));
  ok("  ... leaving the sheets for `none` restores the default too; between `none` and the mean field (no sheets on either side) nothing is touched, so psi picked there stays, through a Reset as well",
     seq[5] === "4/4" && seq[6] === "3/3" && seq[7] === "3/3", seq.slice(5, 8).join(" "));
  const man = JSON.parse(env.run("function(){ return JSON.stringify(runManifest().forcing); }"));
  ok("a saved params.json records the box, the grid, the rung, the background and the sheets as laid down",
     man.box[0] === f169.box[0] && man.box[1] === L && man.grid.join() === "256,256" && man.rung === "standard" && man.background === "sheets" &&
     rel(man.sheets.psi0, fs169.psi0) < 1e-14 && rel(man.sheets.a, fs169.a) < 1e-14 && man.sheets.peakField === 5 && man.effects.meanField === 0, JSON.stringify(man.sheets));
  await P.input("selBg", "field");
  await P.input("selBox", "square");
  await P.frame();
  ok("  ... and with all three selects back at their defaults the page is the shipped one again", JSON.stringify(snap(env).ui) === shipped && snap(env).sel === "4" && realFails(env).length === 0);

  // ---- the auto rung, on the page -----------------------------------------------------------------
  const A = await musicPage("?plain&musicdebug"), aenv = A.env;
  const au = () => JSON.parse(aenv.run("function(){ return JSON.stringify({ behind: music.behind, a: music.auto, n: [solver.p.nx, solver.p.ny], sel: el('selGrid').value, st: el('status').textContent, box: el('musicBox').textContent, run: running }); }"));
  // one run: Start at audio time t0, `nb` frames that each find the clock a second behind, at
  // audio times t0 + from + i; then Stop
  const session = async (t0, from, nb) => {
    model(aenv);
    A.A.t = t0;
    await A.click("btnRun");
    for (let i = 1; i <= 3; i++) await A.frame(t0 + 0.02 * i);
    // (each followed by two ordinary frames INSIDE the clock's 1 s BEHIND hold: lit, not counted)
    for (let i = 0; i < nb; i++) { await A.frame(t0 + from + i); await A.frame(t0 + from + i + 0.02); await A.frame(t0 + from + i + 0.04); }
    const r = au();
    await A.click("btnRun");
    return r;
  };
  // (2026-11-24: with ?plain the grid select says standard, so auto is CHOSEN here; on the
  // stage it is the default, which the stage rows below state)
  await A.input("selGrid", "auto");
  aenv.run("function(){ globalThis.__svm = solver; }");
  let r = await session(100, 1, 3);
  ok("auto, a run that is rebased for falling behind 3 times in its first 10 s: counted (3 -- the REBASES, not the nine frames on which BEHIND was lit), nothing changes mid-run, and the next Start stays on standard (3 is not MORE than 3)",
     r.a.n === 3 && r.behind === true && r.a.rung === "standard" && r.n.join() === "256,256" && aenv.run("function(){ return solver === __svm; }") === true, JSON.stringify(r.a));
  r = await session(200, 1, 6);
  const mid = aenv.run("function(){ return solver === __svm; }");
  ok("  ... the count starts again with each run (3 was forgotten: this run's own 6 are counted), and a run that falls behind 6 times still finishes on the grid it started on",
     r.a.n === 6 && r.a.rung === "standard" && r.n.join() === "256,256" && mid === true && r.box.indexOf("dropped") < 0, JSON.stringify(r.a));
  A.A.t = 300;
  await A.click("btnRun");
  r = au();
  ok("the NEXT Start drops ONE rung: auto is on light (128 x 128 on the square box), the select still says auto, and the status line and the line under the selects both say why",
     r.run === true && r.a.rung === "light" && r.n.join() === "128,128" && r.sel === "auto" && r.a.n === 0 &&
     /auto dropped to light -- the simulation fell behind the audio 6 times in the first 10 s/.test(r.st) && /auto dropped to light/.test(r.box) && /on 128 × 128 \(light, auto\)/.test(r.box),
     r.st);
  await A.click("btnRun");
  r = await session(400, 1, 9);
  A.A.t = 500;
  await A.click("btnRun");
  const low = au();
  await A.click("btnRun");
  ok("  ... on light, a run that falls behind 9 times changes nothing: there is no rung below, and auto says nothing new",
     r.a.n === 9 && low.a.rung === "light" && low.n.join() === "128,128" && low.st === "", JSON.stringify(low.a));
  r = await session(600, 20, 0);
  A.A.t = 700;
  await A.click("btnRun");
  const calm = au();
  await A.click("btnRun");
  ok("  ... and it NEVER climbs: a run with no rebase at all, then Start, is still light; nor is it ever sharp",
     r.a.n === 0 && calm.a.rung === "light" && calm.n.join() === "128,128" && aenv.run("function(){ return musicShapeWanted().rung; }") === "light", JSON.stringify([r.a, calm.a, calm.n]));
  // only the first MUSIC_AUTO_WINDOW seconds count, and only while the select says auto
  const B = await musicPage("?plain&musicdebug"), benv = B.env;
  await B.input("selGrid", "auto");
  const bu = () => JSON.parse(benv.run("function(){ return JSON.stringify({ a: music.auto, n: [solver.p.nx, solver.p.ny], behind: music.behind, box: el('musicBox').textContent }); }"));
  model(benv);
  B.A.t = 100;
  await B.click("btnRun");
  for (let i = 1; i <= 3; i++) await B.frame(100 + 0.02 * i);
  for (let i = 0; i < 8; i++) await B.frame(100 + 10.5 + i);
  const late = bu();
  await B.click("btnRun");
  B.A.t = 200;
  await B.click("btnRun");
  const afterLate = bu();
  await B.click("btnRun");
  ok("falling behind AFTER the first 10 s of a run is not counted (8 rebases from 10.5 s on, BEHIND lit, count 0): the next Start stays on standard",
     late.a.n === 0 && late.behind === true && afterLate.a.rung === "standard" && afterLate.n.join() === "256,256", JSON.stringify(late));
  await B.input("selGrid", "standard");
  model(benv);
  B.A.t = 300;
  await B.click("btnRun");
  for (let i = 1; i <= 3; i++) await B.frame(300 + 0.02 * i);
  for (let i = 0; i < 8; i++) await B.frame(300 + 1 + i);
  const named = bu();
  await B.click("btnRun");
  await B.input("selGrid", "auto");
  ok("  ... nor is it counted while a rung is chosen BY NAME (standard, 8 rebases, count 0): going back to auto afterwards is still standard",
     named.a.n === 0 && named.behind === true && bu().a.rung === "standard" && bu().n.join() === "256,256");
  // a pending drop is also taken by Reset
  model(benv);
  B.A.t = 400;
  await B.click("btnRun");
  for (let i = 1; i <= 3; i++) await B.frame(400 + 0.02 * i);
  for (let i = 0; i < 5; i++) await B.frame(400 + 1 + i);
  await B.click("btnReset");
  const viaReset = bu();
  ok("a drop that is due is also taken by Reset, mid-run: 5 rebases, Reset -> light, still running, and the status line says so",
     viaReset.a.rung === "light" && viaReset.n.join() === "128,128" && benv.run("function(){ return running; }") === true &&
     /auto dropped to light/.test(benv.getEl("status").textContent) && realFails(benv).length === 0, benv.getEl("status").textContent.slice(0, 60));
  await B.click("btnRun");
  await B.input("selGrid", "standard");
  ok("  ... and choosing standard by name overrides it (256 x 256 again), as the message says -- and the line under the selects then stops repeating auto's note (it was there while auto was the choice)",
     /dropped to light/.test(viaReset.box) && bu().n.join() === "256,256" && /choose standard to override/.test(bu().a.note) && !/dropped/.test(bu().box) && /on 256 × 256 \(standard\), 4 bands$/.test(bu().box), bu().box);

  // ---- the shipped page (no ?stage, grid = standard by default) never drops ---------------------------
  const C = await musicPage("?plain&musicdebug"), cenv = C.env;
  const cu0 = () => JSON.parse(cenv.run("function(){ return JSON.stringify({ a: music.auto, n: [solver.p.nx, solver.p.ny], behind: music.behind, sel: el('selGrid').value, st: el('status').textContent, box: el('musicBox').textContent, ui: uiParams(), lit: music.lit.length, run: running }); }"));
  cenv.run("function(){ globalThis.__svm = solver; }");
  model(cenv);
  C.A.t = 100;
  await C.click("btnRun");
  for (let i = 1; i <= 3; i++) await C.frame(100 + 0.02 * i);
  for (let i = 0; i < 9; i++) await C.frame(100 + 1 + i);
  const cRun = cu0();
  await C.click("btnRun");
  C.A.t = 200;
  await C.click("btnRun");
  const cStart = cu0();
  await C.click("btnRun");
  await C.click("btnReset");
  const cReset = cu0();
  await C.input("selBg", "none");
  await C.input("selBg", "field");
  await C.input("selGrid", "standard");
  const cSel = cu0();
  ok("with ?plain, at the defaults, a run that is rebased for falling behind 9 times in its first 10 s (BEHIND lit) counts NOTHING, and the next Start, a Reset and changes of the selects all leave the grid at 256 x 256 with 4 bands: no drop, no note, nothing in the status line",
     cRun.sel === "standard" && cRun.behind === true && cRun.a.n === 0 && [cRun, cStart, cReset, cSel].every(c => c.a.rung === "standard" && c.a.note === "" && c.a.n === 0 && c.n.join() === "256,256" && c.lit === 4 &&
       !/dropped/.test(c.st) && /^box 6\.28 × 6\.28 on 256 × 256 \(standard\), 4 bands$/.test(c.box)) && realFails(cenv).length === 0, JSON.stringify([cRun.a, cStart.n, cSel.box]));
  ok("  ... and what the Solver holds is still the shipped parameter set, bit for bit (the two rebuilds were the background select's, none of them a change of grid)",
     [cRun, cStart, cReset, cSel].every(c => JSON.stringify(c.ui) === shipped) && cStart.run === true && cenv.run("function(){ return solver.p.nx === 256 && solver.p.ny === 256; }") === true, JSON.stringify(cSel.ui).slice(0, 80));
  // auto earns a drop, and the visitor then chooses a rung by name: the drop is not taken behind their back
  await C.input("selGrid", "auto");
  model(cenv);
  C.A.t = 300;
  await C.click("btnRun");
  for (let i = 1; i <= 3; i++) await C.frame(300 + 0.02 * i);
  for (let i = 0; i < 6; i++) await C.frame(300 + 1 + i);
  const dEarn = cu0();
  await C.click("btnRun");
  cenv.run("function(){ showStatus('', 'info'); }");
  await C.input("selGrid", "standard");
  const dName = cu0();
  await C.input("selGrid", "auto");
  const dBack = cu0();
  C.A.t = 400;
  await C.click("btnRun");
  const dStart = cu0();
  await C.click("btnRun");
  ok("on auto, a run that falls behind 6 times has EARNED a drop (count 6) -- but choosing standard by name before the next Start takes none: no \"auto dropped\" note anywhere, 256 x 256, and auto's own rung is still standard",
     dEarn.sel === "auto" && dEarn.a.n === 6 && dEarn.a.rung === "standard" && dName.sel === "standard" && dName.a.rung === "standard" && dName.a.note === "" && dName.a.n === 0 &&
     dName.n.join() === "256,256" && !/dropped/.test(dName.st) && !/dropped/.test(dName.box), JSON.stringify([dEarn.a, dName.a, dName.st]));
  ok("  ... and going back to auto afterwards does not jump to light from that old run (its count was zeroed when the rung was chosen): standard at the select, and at the next Start",
     dBack.sel === "auto" && dBack.a.rung === "standard" && dBack.a.n === 0 && dBack.n.join() === "256,256" && /on 256 × 256 \(standard, auto\), 4 bands$/.test(dBack.box) &&
     dStart.run === true && dStart.a.rung === "standard" && dStart.n.join() === "256,256" && !/dropped/.test(dStart.st) && realFails(cenv).length === 0, JSON.stringify([dBack.a, dStart.n, dBack.box]));

  // ---- ON THE STAGE: square by default; the window's shape on request, and then the picture fills #stage --
  // (2026-11-23: the default box is square with the flag too, and the stage shows whatever box
  // it has WHOLE -- contain -- so a window of another shape than the box has bars, or repeats,
  // and never a crop. The rows about the window-shaped box choose it through the select.)
  const S = await musicPage("?musicdebug"), senv = S.env;
  const SS = `function(){ const s = solver, c = cards.disp[0], st = c.wrap.style; return JSON.stringify({ p: [s.p.nx, s.p.ny, s.p.Lx, s.p.Ly], b0: s.p.b0, eq: s.p.eqsrc,
    cv: [c.cv.width, c.cv.height], css: [st.left, st.top, st.width, st.height].map(parseFloat), box: el("selBox").value, grid: el("selGrid").value, bg: el("selBg").value,
    opts: el("selBox").options.map(o => o.value), line: el("musicBox").textContent, lit: music.lit.length, plan: music.plan,
    inStage: c.wrap.parentNode === stage.el, drawer: ["selBox", "selGrid", "selBg", "rSheetB", "musicBox"].every(id => { for (let p = el(id); p; p = p.parentNode) if (p.id === "stageDrawer") return true; return false; }) }); }`;
  const ss = () => JSON.parse(senv.run(SS));
  const cssIs = (t, l, tp, w, h) => Math.abs(t.css[0] - l) < 0.006 && Math.abs(t.css[1] - tp) < 0.006 && Math.abs(t.css[2] - w) < 0.006 && Math.abs(t.css[3] - h) < 0.006;
  let t = ss();
  ok("on the stage the box select defaults to SQUARE, as with ?plain (grid AUTO -- the default only on the stage -- and background the mean field): the shipped 2 pi x 2 pi on 256 x 256 with 4 bands, and the window's shape is still on offer in the select",
     t.box === "square" && t.grid === "auto" && t.bg === "field" && t.opts.join() === "window,square" && t.p.join() === [256, 256, L, L].join() && t.b0 === true && t.eq === false &&
     t.lit === 4 && JSON.stringify(t.plan) === JSON.stringify(M.musicBandPlan(256, null, null, 32)) && /^box 6\.28 × 6\.28 on 256 × 256 \(standard, auto\), 4 bands$/.test(t.line) &&
     realFails(senv).length === 0, t.box + " " + t.p.join(" ") + " | " + t.line);
  ok("  ... shown whole in the stub's 1280 x 800 window: an 800 x 800 canvas in an 800 x 800 CSS box at (240, 0), bars either side",
     t.inStage && t.cv.join() === "800,800" && cssIs(t, 240, 0, 800, 800), t.cv.join("x") + " at " + t.css.join(" "));
  await S.input("selBox", "window");
  await S.frame();
  t = ss();
  ok("box = the window's shape, chosen there: the box is the stub window's 1280 x 800, 1.6 * 2 pi x 2 pi on 256 x 256",
     t.box === "window" && t.grid === "auto" && t.bg === "field" && t.p[0] === 256 && t.p[1] === 256 && t.p[2] === 1.6 * L && t.p[3] === L && t.b0 === true && t.eq === false &&
     realFails(senv).length === 0, t.p.join(" "));
  ok("  ... and the picture then FILLS #stage exactly, no bars: a 1280 x 800 canvas in a 1280 x 800 CSS box at (0, 0) -- stage.js reads the box's aspect off the solver",
     t.inStage && t.cv.join() === "1280,800" && cssIs(t, 0, 0, 1280, 800), t.cv.join("x") + " at " + t.css.join(" "));
  senv.getEl("selFill").value = "tile";
  await S.frame();                                 // (paused: this frame marks the card, the next renders and copies)
  await S.frame();
  const one = senv.getEl("stageTile").__cx2d.__draws.map(d => d.at.join());
  senv.getEl("selFill").value = "bars";
  await S.frame();
  ok("  ... so with the sides on \"repeat the picture\" there is nothing to repeat: ONE copy, the window's own 1280 x 800 at (0, 0)", one.join(" ") === "0,0,1280,800", one.join(" "));
  ok("  ... the new controls are in the drawer with the rest", t.drawer === true);
  senv.run("function(){ globalThis.__svm = solver; window.innerWidth = 390; window.innerHeight = 844; stageResize(true); }");
  t = ss();
  ok("a rotation to 390 x 844 rebuilds nothing (the same solver, the same 1.6 box): the stage shows the WHOLE landscape picture across the tall window's width -- 390 x 243.75 CSS (390 / 1.6), 300.125 px of bar above and below -- on a 390 x 244 canvas",
     senv.run("function(){ return solver === __svm; }") === true && t.p[2] === 1.6 * L && cssIs(t, 0, (844 - 390 / 1.6) / 2, 390, 390 / 1.6) && Math.abs(t.css[1] - 300.125) < 0.006 && t.cv.join() === "390,244",
     t.cv.join("x") + " at " + t.css.join(" "));
  const npic = JSON.parse(senv.run("function(){ return JSON.stringify(stagePicture()); }"));
  ok("  ... and stagePicture() answers that NON-square box, each field its own: x = 0, w = 390 (the width), y = 300.125, h = 243.75 (the height), not tiled",
     npic.x === 0 && npic.w === 390 && npic.y === 300.125 && npic.h === 243.75 && npic.tile === false, JSON.stringify(npic));
  await S.click("btnReset");
  t = ss();
  ok("  ... and Reset takes the tall window: 2 pi x 2 pi (844/390), and the picture fills it exactly again (390 x 844 CSS, no bars)",
     senv.run("function(){ return solver === __svm; }") === false && t.p[2] === L && t.p[3] === M.musicBoxFor(390 / 844)[1] &&
     cssIs(t, 0, 0, 390, 844) && t.cv.join() === "390,844", t.cv.join("x") + " at " + t.css.join(" "));
  senv.run("function(){ window.innerWidth = 3000; window.innerHeight = 1000; }");
  await S.click("btnReset");
  t = ss();
  ok("a window beyond the clamp (3000 x 1000): the box stops at 2.4, and the picture is shown whole, NOT cropped -- 2400 x 1000 CSS, a 300 px bar either side -- on a 1920 x 800 canvas (the cap on its long side)",
     t.p[2] === 2.4 * L && t.p[3] === L && cssIs(t, 300, 0, 2400, 1000) && t.cv.join() === "1920,800", t.cv.join("x") + " at " + t.css.join(" "));
  // the stage's own box wins over the browser window (100dvh: the visible viewport on a phone)
  senv.run("function(){ stage.el.clientWidth = 1000; stage.el.clientHeight = 500; window.innerWidth = 1280; window.innerHeight = 800; }");
  await S.click("btnReset");
  t = ss();
  ok("the shape is read off #stage's own box where it reports one (1000 x 500 -> 2:1), not off the browser window (1280 x 800)",
     t.p[2] === 2 * L && t.p[3] === L && senv.run("function(){ const w = musicWindow(); return w.w + 'x' + w.h; }") === "1000x500", t.p.join(" "));
  await S.input("selBg", "sheets");
  t = ss();
  ok("on the stage the sheets build the same flags (eqsrc, no b0) on the window's box, and the page logs no failure", t.eq === true && t.b0 === false && t.p[2] === 2 * L && realFails(senv).length === 0,
     realFails(senv).slice(0, 2).join(" | "));
  await S.input("selBox", "square");
  await S.input("selBg", "field");
  t = ss();
  ok("  ... and box = square there is the shipped square solver", t.p.join() === [256, 256, L, L].join() && t.b0 === true && t.eq === false);

  // ---- the traces overlay follows a rebuild (it keys its bins off the live solver) ------------------
  const T = await musicPage("?plain&traces"), tenv = T.env;
  tenv.run("function(){ window.innerWidth = 1920; window.innerHeight = 1080; }");
  await T.input("selBox", "window");
  await T.input("selGrid", "sharp");
  for (let i = 0; i < 14; i++) await T.frame();
  ok("with ?traces a rebuild onto a 16:9 box at 512 x 256 leaves the overlay running (frames pass, nothing logged as a failure)",
     tenv.run("function(){ return solver.p.nx + 'x' + solver.p.ny + ' ' + (musicTraces.on && !!musicTraces.canvas); }") === "512x256 true" && realFails(tenv).length === 0,
     realFails(tenv).slice(0, 2).join(" | "));
  // the side panels' line cuts run along y: ny samples, on a grid where nx is not ny
  await T.input("selGrid", "light");
  tenv.getEl("cbTrSides").checked = true;
  tenv.getEl("selTrB").value = "cuts";
  tenv.run(`function(){ window.__hold = []; const wrap = sv => { sv.readCutLine = iz => { const ny = sv.p.ny, a = new Float32Array(4 * ny);
      for (let i = 0; i < ny; i++) { a[i] = i; a[ny + i] = -7; a[2 * ny + i] = 1000 + i; a[3 * ny + i] = -9; }
      return window.__held ? new Promise(r => window.__hold.push(() => r(a))) : Promise.resolve(a); }; };
    window.__wrapLine = wrap; wrap(solver); musicTraces.rows4 = []; stateBumped(); musicTraces.sample(solver, 1000); }`);
  await turn();
  const nr = JSON.parse(tenv.run("function(){ const e = musicTraces.rows4[0]; return JSON.stringify({ g: [solver.p.nx, solver.p.ny], n: musicTraces.rows4.length, len: e ? e.rows.map(r => r.length) : [], v: e ? e.rows.map(r => [r[0], r[5], r[127]]) : [] }); }"));
  ok("on a 256 x 128 grid (the 16:9 box on light) the side panels' line cuts are rows of ny = 128, not nx: u_x = i, b_x = 1000 + i, their sum and their difference, read off the right quarters of the solver's 4 ny floats",
     nr.g.join() === "256,128" && nr.n === 1 && nr.len.join() === "128,128,128,128" && JSON.stringify(nr.v) === JSON.stringify([[0, 5, 127], [1000, 1005, 1127], [1000, 1010, 1254], [-1000, -1000, -1000]]), JSON.stringify(nr));
  // a line read in flight when its solver is retired pushes nothing
  tenv.run("function(){ window.__held = true; musicTraces.rows4 = []; stateBumped(); musicTraces.sample(solver, 2000); }");
  await turn();
  const fl0 = tenv.run("function(){ return [window.__hold.length, musicTraces.busy4, musicTraces.rows4.length].join(); }");
  await T.input("selGrid", "standard");
  tenv.run("function(){ window.__held = false; window.__wrapLine(solver); window.__hold.shift()(); }");
  await turn();
  const fl1 = tenv.run("function(){ return [solver.p.nx + 'x' + solver.p.ny, musicTraces.busy4, musicTraces.rows4.length].join(); }");
  tenv.run("function(){ stateBumped(); musicTraces.sample(solver, 3000); }");
  await turn();
  const fl2 = tenv.run("function(){ return musicTraces.rows4.map(e => e.t + ':' + e.rows[0].length).join(); }");
  ok("  ... and a line read still in flight when its solver is retired (a rebuild onto 256 x 256) lands on nothing: not pushed, the read no longer marked busy, and the next sample reads the NEW solver's line",
     fl0 === "1,true,0" && fl1 === "256x256,false,0" && fl2 === "3000:256" && realFails(tenv).length === 0, fl0 + " | " + fl1 + " | " + fl2);
}

const LEGS = [
  ["1. music.js is pure and self-contained", legPure],
  ["2. musicBandPlan against the offline band_plan", legPlan],
  ["3. MusicDetector against the offline CausalOnsetDetector", legDetector],
  ["4. the push rule, the generator, the blobs", legBlobs],
  ["5. the click track, and what the detector makes of it", legClick],
  ["6. the clock", legClock],
  ["6b. the live-source rules: constraints, routing, messages", legRules],
  ["6c. sync from the output latency, the noise floor, the colormaps", legSync],
  ["7. music.html booted: the solver it builds, what the controls do", legBoot],
  ["8. music.html running: kicks, steps, the clock, Stop, a file", legRun],
  ["8b. without ?musicdebug", legPlain],
  ["9. the other two pages: hook null, loop and state mark unchanged", legOthers],
  ["10. live sources and the delayed speaker branch (M3 / M4)", legLive],
  ["11. the review's findings and the output-latency sync, on the page", legFix],
  ["12. the fixed colour scale", legScale],
  ["13. the effects: the rules against the offline script, and the page with its defaults", legEffects],
  ["14. the fading traces (traces.js): the pure rules, the canvas, the schedule", legTraces],
  ["15. the stage (the default layout): the rules, the structure, and nothing of it with ?plain", legStage],
  ["16. the window-shaped box and the current sheets: the rectangular rules against the offline script, and the page", legShape]
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
