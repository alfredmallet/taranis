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

// ===========================================================================
// 10. live sources and the delayed speaker branch (M3 / M4), on the stub
// ===========================================================================
async function legLive(state) {
  const M = state.M;
  const click = M.musicClickTrack({ sr: 44100, bpm: 120, beats: 16, seed: 1 }).samples;
  const P = await musicPage("?musicdebug"), env = P.env, A = P.A, FM = fakeMedia(A, click);
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
  ok("the source select offers the four sources", Array.from(env.getEl("selSrc").options).map(o => o.value).join() === "click,file,tab,input");
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
  const W = await musicPage("", 48000), FW = fakeMedia(W.A, M.musicClickTrack({ sr: 48000, bpm: 120, beats: 16, seed: 1 }).samples);
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
    const P = await musicPage(search === undefined ? "?musicdebug" : search), FM = fakeMedia(P.A, click);
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
  const P = await musicPage("?musicdebug"), env = P.env;
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
  ok("  ... the readout and musicDebug name it", /fixed at 4\.00/.test(env.run("function(){ return musicScaleReadout(); }")) &&
     env.run("function(){ const r = musicScaleRecord(); return r.fixed === true && r.max === 4 && r.m === 2 && r.amp === 2; }"));

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
  const P = await musicPage("?musicdebug", 44100, "stereo"), env = P.env, A = P.A;
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
  const Q = await musicPage("?musicdebug", 44100, true);
  Q.env.run(MODEL, 0.004);
  Q.A.t = 50;
  await Q.click("btnRun");
  for (let i = 1; i <= 3 * 60; i++) await Q.frame(50 + i / 60);
  const dq = JSON.parse(Q.env.run("function(){ return JSON.stringify(window.musicDebug.kicks); }"));
  // a MONO source on a context that CAN split: the up-mixed channels are equal, so nothing is
  // ever placed (the review's finding: pan 0 used to send every band to the centre column) --
  // and it is an input device, so this is also the splitter branch's "nothing reaches the
  // speakers" row
  const R = await musicPage("?musicdebug", 44100, "stereo"), FR = fakeMedia(R.A, c0.samples);
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
  const U = await musicPage("", 44100, "stereo");
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
  ["13. the effects: the rules against the offline script, and the page with its defaults", legEffects]
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
