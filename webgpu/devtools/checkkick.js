// The IMPULSIVE BLOB KICK and the UNIFORM DRAG gate (plans-webgpu/MUSIC_PLAN.md, phase M1):
// the numbers, not the names.
//
//   node checkkick.js [webgpu-dir]        exit code 1 on any failure
//
// CI reports, never gates (the sibling scripts' contract): every row prints PASS / FAIL and
// the exit code is for a human reading the log.
//
// What M1 added, and what stands behind each piece here
// -----------------------------------------------------
//   physics.js   KICK_MAX, blobKickWGSL        the kernel that ADDS gaussian blobs to the fields
//   solver2d.js  buildShaders: `if (g.kick)`   its gated emission
//                Solver: _packKicks, kick      the pack and the one-shot dispatch
//                Solver: _cflStale, step       a kick marks the CFL stale; the next step
//                                              recomputes dt whatever its cflEvery says
//                Solver: _dragLin, setDrag     the drag, folded into gridB's linear column
// `blobKick` is a kernel no base commit ever emitted, and it is emitted ONLY for a solver
// built with `kick: true`, so the byte-identity gates (`checksolver2d`, `checkiso`,
// `checkeigf`, `check2dspec`) have nothing to say about what it computes. This file is that
// statement, in checkblob.js's idiom: the production path -- the real `_packKicks` and the
// real emitted kernel, executed byte for byte on wgsl_reflect's interpreter -- against TWO
// references that share nothing with it:
//   (a) written here: the analytic gaussian laid down in REAL space, periodised, and
//       transformed by a naive O(N^2) DFT, with the potential peak found by maximising
//       |grad f| numerically (so common.js's icBlobPeak is not borrowed) and the Elsasser ->
//       (phi, psi) map written out from its definition;
//   (b) a FIXTURE, fixtures/kick_offline.json, recorded by ../gen_kickfixture.py from the
//       offline reference implementation itself -- `make_kick` in
//       examples/music_turbulence.py, run unmodified at fp64: real-space gaussians, taranis'
//       own rfft2 and taranis' own dealias mask. That is the code the offline movies were
//       made with, so (b) is "the live kick is the offline kick" and not a re-derivation.
// (a) is here as well as (b) because the offline script only knows a square 2*pi box: the
// rectangular-box leg, which is the only thing that separates nx/Lx from ny/Ly in the
// weight, has to be built here.
//
// EXECUTED on the interpreter: `blobKick` (every factor, the phase, the two early returns,
// the two-half write) and `stage` (the LSRK33 integrating-factor update, which is where the
// drag acts: with a zero RHS the three stages multiply the fields by exp(lin * dt)).
// CALLED for real: `_packKicks` on a bare `{ p, _kicks }` receiver; `kick`, `step`, `setDrag`,
// `refreshDissipation` and the constructor on the stub device. The stub's dispatch log is
// filled at ENCODE time and sees nothing else, so leg 5 wraps the queue AND the command
// encoder (`spy`): uploads, every clearBuffer / copy, each pass begun and whether it was
// ended, finish, submit -- and snapshots the solver around a kick. (Round 2, after the
// adversarial review: fourteen mutants had survived exactly where the gate was not looking
// -- inside the encoder, at blob 0 of a list, at zero amplitude, and in every drag path of
// a configuration it never built: kick: true, hyper >= 3, a non-square grid.)
// NOT verified here: anything on a real GPU. The stub records what was encoded and the
// interpreter executes the kernel text; neither is a device. The dissipation-rate readout
// with the drag on (energyPartial's sc[6]) is asserted only as "it reads the same gridB
// column" -- the reduction kernels do not run on the interpreter (workgroup barriers).
//
// The interpreter's quirk, measured in leg 1 and decisive here
// ---------------------------------------------------------
// wgsl_reflect 1.5.0 reads a one-argument `@workgroup_size(64)` DIGIT BY DIGIT, as a 6 x 4
// workgroup: global_invocation_id.x = 6*wgid + lane over 6 lanes, and each of those runs
// FOUR times (the "y" lanes). A kernel that writes its output (blobBuild, everything
// checkblob runs) cannot tell. Both kernels this file executes can:
//   * `blobKick` accumulates into the fields. It is therefore emitted with the full
//     three-argument `@workgroup_size(64, 1, 1)` (physics.js says why), which the
//     interpreter reads correctly -- 64 lanes, each id once -- so it runs here BYTE FOR
//     BYTE, at ceil(NM / 64) workgroups. That this is ALSO what `kick` dispatches is leg 5's
//     row on the 32^2 and 32x16 grids, where NM / 64 is 8.5 and 4.5 and a floor would leave
//     the last modes unkicked (at 64^2 it is exactly 33 and ceil and floor agree).
//   * `stage` multiplies the fields by exp(lin*dt*gamma) and is an existing kernel whose
//     text is pinned against a base commit, so it keeps its `(64)`. runSteps executes it
//     with that ONE attribute respelled `(64, 1, 1)` -- the identical workgroup in WGSL --
//     and asserts the substitution happened exactly once and is the whole difference.
"use strict";
const fs = require("fs"), path = require("path");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const e2 = v => (isFinite(v) ? v.toExponential(2) : String(v));
const boot = (d, page, demo) => new Promise(res => {
  const env = require("./stubenv")(d, page, demo);
  setTimeout(() => res(env));
});
async function wgslMod() {
  const p = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  if (!fs.existsSync(p)) return null;
  return await import(pathToFileURL(p).href);
}
const kconst = (src, n) => {
  const m = new RegExp("const " + n + ": (?:i32|u32|f32) = ([-+\\d.e]+)u?;").exec(src);
  return m ? parseFloat(m[1]) : NaN;
};
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);
const sameBits = (a, b) => {
  if (a.length !== b.length) return -2;
  const x = bits(a), y = bits(b);
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return i;
  return -1;
};
const throws = fn => { try { fn(); return ""; } catch (e) { return String(e && e.message || e) || "threw"; } };

// ---------------------------------------------------------------------------
// grids (checkblob's two: a square 2*pi box with exact integer k, and the same box halved
// in y, which is what separates the two factors of w = ... (nx*ny)/(Lx*Ly))
// ---------------------------------------------------------------------------
const SQ = { nx: 32, ny: 32, Lx: 2 * Math.PI, Ly: 2 * Math.PI, diss: 1e-13, hyper: 4, fshell: [1, 3] };
const RECT = { nx: 32, ny: 16, Lx: 2 * Math.PI, Ly: Math.PI, diss: 1e-13, hyper: 4, fshell: [1, 3] };

// the app's own emission + grid arrays for one parameter set, with the kick asked for or not
const GRIDSNIP = `function(P, kick){
  const gr = makeGrid(P);
  const g = Object.assign({ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly, kick: kick }, gr);
  const S = buildShaders(g);
  return { S: S, names: Object.keys(S), nm: gr.nm, nkx: gr.nkx, nky: gr.nky,
           gridA: Array.from(gr.gridA), gridB: Array.from(gr.gridB), kmax: KICK_MAX,
           bmax: BLOB_FORCE_MAX, lsrk: [LSRK33.alpha.slice(), LSRK33.beta.slice(), LSRK33.gamma.slice()] };
}`;
function gridOf(env, P, kick) {
  const G = env.run(GRIDSNIP, P, kick === undefined ? true : kick);
  G.gridA = Float32Array.from(G.gridA);
  G.gridB = Float32Array.from(G.gridB);
  G.src = G.S.blobKick;
  return G;
}
// the REAL _packKicks on a bare receiver; the result comes back as plain JSON
const PACK = `function(P, arr, list){
  const r = Solver.prototype._packKicks.call({ p: P, _kicks: arr }, list);
  return JSON.stringify(r);
}`;
const SENT_PACK = 7.5;
function pack(env, P, G, list, into) {
  const a = into || new Float32Array(2 * G.kmax * 4).fill(SENT_PACK);
  const r = JSON.parse(env.run(PACK, P, a, list));
  return { a: a, used: r.used, dropped: r.dropped };
}
// the production kernel, executed byte for byte over ceil(NM / 64) workgroups -- the count
// leg 5 shows `kick` asks for, on grids where it is not an integer. `fields` is copied.
function runKick(M, G, packed, fields, ngroups) {
  const f = fields ? Float32Array.from(fields) : new Float32Array(4 * G.nm);
  new M.WgslExec(new M.WgslParser().parse(G.src)).dispatchWorkgroups(
    "main", [ngroups || Math.ceil(G.nm / 64), 1, 1], { 0: { 0: f, 1: G.gridA, 2: G.gridB, 3: packed } });
  return f;
}
// a deterministic state that is nowhere zero (no RNG anywhere near this repo's checks)
function walk(n, s0) {
  const f = new Float32Array(n);
  let s = s0;
  for (let i = 0; i < n; i++) { s = (s * 48271) % 2147483647; f[i] = s / 2147483647 - 0.5; }
  return f;
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE (nothing below is the app's)
// ---------------------------------------------------------------------------
// A blob is a REAL-SPACE gaussian on an Elsasser potential whose peak |grad f| is `amp`.
// The potential peak is found by maximising |grad f| over r numerically.
function potentialPeak(amp, sg) {
  const g = r => (r / (sg * sg)) * Math.exp(-0.5 * r * r / (sg * sg));
  let lo = 0, hi = 8 * sg;
  for (let i = 0; i < 200; i++) {
    const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3;
    if (g(a) < g(b)) lo = a; else hi = b;
  }
  return amp / g(0.5 * (lo + hi));
}
function dealiasRef(P, nkx, nky) {
  const cutx = P.nx / 3.0, cuty = P.ny / 3.0, de = new Float64Array(nkx * nky);
  for (let i = 0; i < nkx; i++) {
    const ix = (i < P.nx / 2) ? i : i - P.nx;
    for (let j = 0; j < nky; j++)
      de[i * nky + j] = ((ix / cutx) * (ix / cutx) + (j / cuty) * (j / cuty)) < 1.0 ? 1 : 0;
  }
  return de;
}
// what one kick adds to (phi, psi), from the definition: each channel's gaussians in real
// space (periodised over IMG images each way), a naive DFT with the solver's unnormalized
// e^{-ikx} convention over the retained modes, nothing at k = 0, then
//   z+ = phi + psi, z- = phi - psi   =>   phi = (z+ + z-)/2, psi = (z+ - z-)/2.
// `cap` blobs per channel at most, in list order; amp == 0 takes no slot.
const IMG = 3;
function refKick(P, nkx, nky, cap, list) {
  const nm = nkx * nky, de = dealiasRef(P, nkx, nky);
  const dx = P.Lx / P.nx, dy = P.Ly / P.ny, z = [new Float64Array(2 * nm), new Float64Array(2 * nm)];
  for (let half = 0; half < 2; half++) {
    const mine = (list || []).filter(b => +b.amp !== 0 && (b.pol < 0 ? 1 : 0) === half).slice(0, cap);
    const f = new Float64Array(P.nx * P.ny);
    for (const b of mine) {
      const Pk = potentialPeak(+b.amp, +b.sigma), inv = 0.5 / (b.sigma * b.sigma);
      // a sheet (leg 7): the gaussian is `ratio` times longer along the direction at `theta`
      const aniso = b.ratio !== undefined && b.ratio !== 1, ct = Math.cos(b.theta || 0), st = Math.sin(b.theta || 0);
      for (let i = 0; i < P.nx; i++) for (let j = 0; j < P.ny; j++) {
        let s = 0;
        for (let p = -IMG; p <= IMG; p++) for (let q = -IMG; q <= IMG; q++) {
          const ax = i * dx - b.x + p * P.Lx, ay = j * dy - b.y + q * P.Ly;
          if (aniso) {
            const al = ct * ax + st * ay, ac = -st * ax + ct * ay;      // along, across
            s += Math.exp(-inv * (ac * ac + al * al / (b.ratio * b.ratio)));
          } else s += Math.exp(-inv * (ax * ax + ay * ay));
        }
        f[i * P.ny + j] += Pk * s;
      }
    }
    for (let a = 0; a < nkx; a++) {
      const ix = (a < P.nx / 2) ? a : a - P.nx, kx = ix * 2 * Math.PI / P.Lx;
      for (let q = 0; q < nky; q++) {
        const ky = q * 2 * Math.PI / P.Ly, m = a * nky + q;
        if (!de[m] || (a === 0 && q === 0)) continue;
        let re = 0, im = 0;
        for (let i = 0; i < P.nx; i++) {
          const px = kx * (i * dx);
          for (let j = 0; j < P.ny; j++) {
            const th = px + ky * (j * dy), v = f[i * P.ny + j];
            re += v * Math.cos(th); im -= v * Math.sin(th);
          }
        }
        z[half][2 * m] = re; z[half][2 * m + 1] = im;
      }
    }
  }
  const out = new Float64Array(4 * nm);
  for (let i = 0; i < 2 * nm; i++) {
    out[i] = 0.5 * (z[0][i] + z[1][i]);              // phi half
    out[2 * nm + i] = 0.5 * (z[0][i] - z[1][i]);     // psi half
  }
  return out;
}
// production against reference, per HALF (phi, psi): the peak-relative worst absolute miss
// and the worst per-mode relative miss over the modes carrying real amplitude. Per half,
// because a wrong psi must not hide under a phi that is ten times larger.
const REL_FLOOR = 1e-4;
function compareHalf(prod, ref, off, n) {
  let peak = 0;
  for (let m = 0; m < n; m++) peak = Math.max(peak, Math.hypot(ref[off + 2 * m], ref[off + 2 * m + 1]));
  let abs = 0, rel = 0, cnt = 0;
  for (let m = 0; m < n; m++) {
    const i = off + 2 * m, a = Math.hypot(ref[i], ref[i + 1]);
    const d = Math.hypot(prod[i] - ref[i], prod[i + 1] - ref[i + 1]);
    abs = Math.max(abs, d);
    if (a >= REL_FLOOR * peak) { cnt++; rel = Math.max(rel, d / a); }
  }
  return { peak: peak, abs: peak > 0 ? abs / peak : abs, rel: rel, n: cnt };
}
function compare(prod, ref, nm) {
  const a = compareHalf(prod, ref, 0, nm), b = compareHalf(prod, ref, 2 * nm, nm);
  return { phi: a, psi: b, abs: Math.max(a.abs, b.abs), rel: Math.max(a.rel, b.rel),
           note: "phi: peak " + e2(a.peak) + " max|d|/peak " + e2(a.abs) + " worst rel " + e2(a.rel)
                 + " (" + a.n + " modes) | psi: peak " + e2(b.peak) + " max|d|/peak " + e2(b.abs)
                 + " worst rel " + e2(b.rel) + " (" + b.n + " modes)" };
}

// ===========================================================================
// 1. the emission: gated, parsed, sized, and covering every mode
// ===========================================================================
async function legEmission(state) {
  const M = state.M;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const env = state.env = await boot(dir, "rmhd2d.html");
  const G = state.G = gridOf(env, SQ), R = state.R = gridOf(env, RECT);
  const G0 = gridOf(env, SQ, false);
  ok("buildShaders emits `blobKick` for a grid built with kick: true", typeof G.src === "string",
     G.names.length + " kernels");
  ok("  ... and NOT without it (the gate is the flag, as srcInit's is eqsrc)",
     G0.names.indexOf("blobKick") < 0 && G0.names.indexOf("sheetKick") < 0 && G.names.indexOf("sheetKick") >= 0
     && G0.names.length === G.names.length - 2,     // blobKick and its anisotropic twin (leg 7)
     G0.names.length + " kernels without, " + G.names.length + " with");
  // every OTHER kernel is the same text with the flag on or off
  const moved = G0.names.filter(n => G0.S[n] !== G.S[n]);
  ok("  ... and asking for the kick moves no other kernel's text",
     moved.length === 0, moved.join(", ") || G0.names.length + " kernels byte-identical");
  // the page that never asks: nothing compiled, nothing allocated
  const live = JSON.parse(env.run(`function(){ return JSON.stringify({
    pl: !!(solver && solver.pl.blobKick), buf: !!(solver && solver.buf.kicks),
    bg: !!(solver && solver.bg.blobKick), gk: solver ? solver.g.kick : null,
    mu: solver ? solver.p.mu : null, kicks: solver && solver._kicks ? solver._kicks.length : -1,
    nk: solver ? solver.nkicks : -1 }); }`));
  const liveShader = env.gpu.shaders.some(s => s.label === "blobKick" || /NKICK/.test(s.code || ""));
  ok("rmhd2d.html's own solver has no blobKick pipeline, bind group, buffer or shader module",
     !live.pl && !live.buf && !live.bg && live.gk === false && !liveShader,
     "g.kick " + live.gk + ", " + env.gpu.shaders.length + " shader modules at boot");
  ok("  ... and it runs with the drag off (p.mu === 0)", live.mu === 0, "p.mu = " + live.mu);
  ok("the kernel parses", (() => { try { new M.WgslParser().parse(G.src); return true; }
                                    catch (e) { return false; } })());
  const NK = kconst(G.src, "NKICK");
  ok("NKICK in the kernel == KICK_MAX == 32 == the solver's packed stride",
     NK === G.kmax && NK === 32 && live.kicks === 2 * G.kmax * 4 && NK === kconst(R.src, "NKICK"),
     "NKICK " + NK + ", KICK_MAX " + G.kmax + ", _kicks " + live.kicks + " floats");
  ok("  ... and the spacebar forcing's BLOB_FORCE_MAX is still 8, its kernel still NBLOB = 8",
     G.bmax === 8 && kconst(G.S.blobBuild, "NBLOB") === 8, "BLOB_FORCE_MAX " + G.bmax);
  const bodyOf = s => s.slice(s.indexOf("const NKICK"));
  const bodies = new Set([bodyOf(G.src), bodyOf(R.src)]);
  for (const pr of [64, 128, 256, 512, 1024])
    bodies.add(bodyOf(gridOf(env, Object.assign({}, SQ, { nx: pr, ny: pr })).src));
  ok("  ... and the kernel body is byte-identical at every grid", bodies.size === 1,
     bodies.size + " distinct bodies over 7 grids");
  // ---- the interpreter, measured rather than assumed (header) ---------------
  const visits = attr => {
    const src = `@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(${attr})
fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x >= 128u) { return; } o[g.x] = o[g.x] + 1.0; }`;
    const o = new Float32Array(128);
    new M.WgslExec(new M.WgslParser().parse(src)).dispatchWorkgroups("main", [2, 1, 1], { 0: { 0: o } });
    return o;
  };
  const vFull = visits("64, 1, 1"), vShort = visits("64");
  state.shortVisits = vShort[0];
  ok("the interpreter runs `@workgroup_size(64, 1, 1)` faithfully: 2 workgroups = ids 0..127, each ONCE",
     vFull.every(v => v === 1), "visit counts " + Array.from(new Set(vFull)).join(","));
  ok("  ... which is why blobKick spells it out: the kernel says (64, 1, 1), exactly once",
     G.src.split("@workgroup_size(64, 1, 1)").length === 2 && !/@workgroup_size\(64\)/.test(G.src));
  console.log("        (for the record: the short `@workgroup_size(64)` runs each id " + vShort[0]
              + " time(s) here, over " + vShort.filter(v => v > 0).length + " ids for 2 workgroups)");
  // coverage: seed with a sentinel no kick could produce by accident and count what moved.
  // Every retained mode but k = 0 must move in BOTH halves (the blob is in both channels'
  // sum for phi, and with a single z+ blob psi = phi).
  for (const [tag, g, P] of [["32x32", G, SQ], ["32x16", R, RECT]]) {
    const SENT = 1.5;            // small, so a rim-sized delta cannot round away against it
    const p1 = pack(env, P, g, [{ x: 0.75, y: 1.5, sigma: 0.25, amp: 0.5, pol: 1 }]);
    const f = runKick(M, g, p1.a, new Float32Array(4 * g.nm).fill(SENT));
    let want = 0, got = 0, stray = 0;
    for (let h = 0; h < 2; h++) for (let m = 0; m < g.nm; m++) {
      const inside = g.gridB[4 * m] > 0.5 && m !== 0, i = 2 * (h * g.nm + m);
      // "moved" is either word: Im stays put exactly where k.x0 = 0 (sin(0) is exact)
      const moved = f[i] !== SENT || f[i + 1] !== SENT;
      if (inside) { want++; if (moved) got++; }
      else if (moved) stray++;
    }
    ok("the EMITTED kernel, run byte for byte, moves every retained non-zero mode of both halves ("
       + tag + ")", got === want && stray === 0 && want > 0,
       got + " of " + want + " moved, " + stray + " outside the retained set touched");
  }
}

// ===========================================================================
// 2. _packKicks: the half, the weight, the cap, what is refused, the zero-fill
// ===========================================================================
function legPack(state) {
  const env = state.env, G = state.G, NK = G.kmax;
  const slot = (a, h, b) => Array.from(a.slice(4 * (h * NK + b), 4 * (h * NK + b) + 4));
  const two = pack(env, SQ, G, [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
                                { x: 2.5, y: 0.25, sigma: 0.375, amp: -0.5, pol: -1 }]);
  const p0 = slot(two.a, 0, 0), p1 = slot(two.a, 1, 0);
  ok("pol > 0 packs into the z+ half's slot 0, pol < 0 into the z- half's (at NKICK, not 8)",
     p0[0] === 0.75 && p0[1] === 1.5 && p0[2] === 0.5 && p1[0] === 2.5 && p1[1] === 0.25 && p1[2] === 0.375,
     "z+ " + p0.slice(0, 3) + " / z- " + p1.slice(0, 3));
  ok("  ... the weight carries the amplitude's sign", p0[3] > 0 && p1[3] < 0,
     "w = " + e2(p0[3]) + " / " + e2(p1[3]));
  ok("  ... and it returns the per-channel counts", two.used[0] === 1 && two.used[1] === 1 && two.dropped === 0,
     JSON.stringify(two.used) + ", dropped " + two.dropped);
  let rest = 0;
  for (let h = 0; h < 2; h++) for (let b = 1; b < NK; b++) if (slot(two.a, h, b).some(v => v !== 0)) rest++;
  ok("  ... every other slot is exactly zero, sentinel erased",
     rest === 0 && Array.from(two.a).every(v => v !== SENT_PACK), rest + " unused slots non-zero");
  // the weight is the SAME number setBlobs packs for the same blob (one definition of a
  // blob's k-space weight in the solver; its magnitude is leg 3's business)
  const blobArr = new Float32Array(2 * G.bmax * 4);
  env.run(`function(P, arr, list){ Solver.prototype.setBlobs.call({ p: P, _blobs: arr }, list); }`,
          SQ, blobArr, [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 }]);
  ok("  ... and the weight is bit for bit setBlobs' for the same blob",
     bits(Float32Array.of(p0[3]))[0] === bits(blobArr)[3], e2(p0[3]) + " vs " + e2(blobArr[3]));
  const pols = pack(env, SQ, G, [{ x: 1, y: 0, sigma: 0.5, amp: 1, pol: 0 },
                                 { x: 2, y: 0, sigma: 0.5, amp: 1 },
                                 { x: 3, y: 0, sigma: 0.5, amp: -1, pol: -0.5 }]);
  ok("  ... pol == 0 and a missing pol are z+; any negative pol is z-",
     slot(pols.a, 0, 0)[0] === 1 && slot(pols.a, 0, 1)[0] === 2 && slot(pols.a, 1, 0)[0] === 3 &&
     pols.used[0] === 2 && pols.used[1] === 1);

  // ---- the cap: first KICK_MAX in list order, the rest dropped AND counted ----
  const many = [];
  for (let i = 0; i < NK + 3; i++) many.push({ x: 0.125 * (i + 1), y: 0, sigma: 0.5, amp: 1, pol: 1 });
  const cap = pack(env, SQ, G, many);
  let order = true, zminus = 0;
  for (let b = 0; b < NK; b++) if (slot(cap.a, 0, b)[0] !== 0.125 * (b + 1)) order = false;
  for (let b = 0; b < NK; b++) if (slot(cap.a, 1, b).some(v => v !== 0)) zminus++;
  ok(NK + 3 + " z+ blobs keep the FIRST " + NK + " in list order", order,
     "x0 = " + [0, 1, NK - 1].map(b => slot(cap.a, 0, b)[0]).join(", ... ,"));
  ok("  ... the overflow is dropped, never wrapped into the z- half", zminus === 0,
     zminus + " z- slots written by a z+ overflow");
  ok("  ... and it is REPORTED: used [" + NK + ", 0], dropped 3",
     cap.used[0] === NK && cap.used[1] === 0 && cap.dropped === 3,
     JSON.stringify(cap.used) + ", dropped " + cap.dropped);
  const mixed = pack(env, SQ, G, many.concat([{ x: 5.5, y: 0, sigma: 0.5, amp: 1, pol: -1 }]));
  ok("  ... and a full z+ channel does not consume the z- channel's slots",
     slot(mixed.a, 1, 0)[0] === 5.5 && mixed.used[1] === 1 && mixed.dropped === 3);
  const exact = pack(env, SQ, G, many.slice(0, NK));
  ok("  ... exactly KICK_MAX per channel fits with nothing dropped",
     exact.used[0] === NK && exact.dropped === 0 && slot(exact.a, 0, NK - 1)[0] === 0.125 * NK);

  // ---- amp == 0 takes no slot ----------------------------------------------
  const z0 = pack(env, SQ, G, [{ x: 1, y: 0, sigma: 0.5, amp: 0, pol: 1 },
                               { x: 2, y: 0, sigma: 0.5, amp: 0.5, pol: 1 }]);
  ok("amp == 0 is skipped and takes no slot", slot(z0.a, 0, 0)[0] === 2 && z0.used[0] === 1 && z0.dropped === 0,
     "slot 0 x0 = " + slot(z0.a, 0, 0)[0]);

  // ---- the zero-fill --------------------------------------------------------
  ok("an empty list zeroes every slot and reports [0, 0]", (() => {
    const e = pack(env, SQ, G, []); return Array.from(e.a).every(v => v === 0) && e.used[0] + e.used[1] === 0; })());
  ok("  ... and so does a null list", Array.from(pack(env, SQ, G, null).a).every(v => v === 0));
  const a3 = new Float32Array(2 * NK * 4).fill(SENT_PACK);
  pack(env, SQ, G, many.slice(0, 3), a3); pack(env, SQ, G, many.slice(0, 1), a3);
  let stale = 0;
  for (let b = 1; b < NK; b++) if (slot(a3, 0, b).some(v => v !== 0)) stale++;
  ok("  ... and a shorter list on the SAME buffer erases the slots it no longer fills",
     stale === 0 && slot(a3, 0, 0)[0] === 0.125, stale + " stale slots after 3 blobs then 1");

  // ---- positions are packed modulo the box ---------------------------------
  const L = SQ.Lx;
  const wr = pack(env, SQ, G, [{ x: L + 0.75, y: -0.5, sigma: 0.5, amp: 1, pol: 1 },
                               { x: 0.75, y: L - 0.5, sigma: 0.5, amp: 1, pol: 1 },
                               { x: 1e30, y: -1e30, sigma: 0.5, amp: 1, pol: 1 }]);
  const w0s = slot(wr.a, 0, 0), w1s = slot(wr.a, 0, 1), w2s = slot(wr.a, 0, 2);
  ok("x and y are packed MODULO the box: (L + 0.75, -0.5) lands on (0.75, L - 0.5)",
     Math.abs(w0s[0] - 0.75) < 2e-6 && Math.abs(w0s[1] - w1s[1]) < 2e-6 && w1s[0] === 0.75,
     "packed (" + w0s[0] + ", " + w0s[1] + ") vs (" + w1s[0] + ", " + w1s[1] + ")");
  ok("  ... a position already in [0, L) is packed BIT FOR BIT (fl32 of what was given)",
     w1s[0] === Math.fround(0.75) && w1s[1] === Math.fround(L - 0.5));
  ok("  ... and x0 = 1e30, y0 = -1e30 (finite, and k.x0 would not be) pack to a bounded position",
     isFinite(w2s[0]) && isFinite(w2s[1]) && Math.abs(w2s[0]) <= 2 * L && Math.abs(w2s[1]) <= 2 * L,
     "packed (" + w2s[0] + ", " + w2s[1] + ")");
  state.farBlob = wr.a;
  // ... each axis modulo ITS OWN length: on the 2pi x pi box, x = 1.5 pi stays (it is inside
  // Lx) while y = 1.5 pi wraps to pi/2, and x = 2.5 pi wraps to pi/2 while y = 0.5 stays
  const R = state.R, rw = pack(env, RECT, R, [{ x: 1.5 * Math.PI, y: 1.5 * Math.PI, sigma: 0.5, amp: 1, pol: 1 },
                                              { x: 2.5 * Math.PI, y: 0.5, sigma: 0.5, amp: 1, pol: 1 }]);
  const r0s = slot(rw.a, 0, 0), r1s = slot(rw.a, 0, 1);
  ok("  ... and each axis wraps on its OWN length (2pi x pi box): (1.5pi, 1.5pi) -> (1.5pi, 0.5pi), (2.5pi, 0.5) -> (0.5pi, 0.5)",
     r0s[0] === Math.fround(1.5 * Math.PI) && Math.abs(r0s[1] - 0.5 * Math.PI) < 2e-6 &&
     Math.abs(r1s[0] - 0.5 * Math.PI) < 2e-6 && r1s[1] === 0.5,
     "packed (" + r0s[0].toFixed(5) + ", " + r0s[1].toFixed(5) + ") and (" + r1s[0].toFixed(5) + ", " + r1s[1] + ")");

  // ---- what is refused, and that refusing leaves the buffer alone -----------
  // The policy: EVERY blob is validated before anything is packed -- wherever it sits in the
  // list, and whether or not its amplitude is zero (a zero-amplitude blob is skipped only
  // AFTER it has been checked) -- and the check is on what will actually be uploaded, so a
  // finite input whose fp32 WEIGHT is not finite is refused too.
  const good = { x: 1, y: 1, sigma: 0.5, amp: 1, pol: 1 };
  const badOnes = [["NaN x", { x: NaN }, /not finite/], ["Infinity y", { y: Infinity }, /not finite/],
                   ["NaN amp", { amp: NaN }, /not finite/], ["Infinity amp", { amp: Infinity }, /not finite/],
                   ["sigma 0", { sigma: 0 }, /not finite/], ["sigma < 0", { sigma: -0.5 }, /not finite/],
                   ["NaN sigma", { sigma: NaN }, /not finite/], ["Infinity sigma", { sigma: Infinity }, /not finite/],
                   ["a missing sigma", { sigma: undefined }, /not finite/],
                   ["sigma 0 AND amp 0 (checked before it is skipped)", { sigma: 0, amp: 0 }, /not finite/],
                   ["NaN sigma AND amp 0", { sigma: NaN, amp: 0 }, /not finite/],
                   ["NaN x AND amp 0", { x: NaN, amp: 0 }, /not finite/],
                   ["sigma 1e13 (finite; its fp32 weight is Infinity)", { sigma: 1e13 }, /overflows fp32/],
                   ["amp 1e39 (finite; its fp32 weight is Infinity)", { amp: 1e39 }, /overflows fp32/],
                   ["amp -1e39", { amp: -1e39 }, /overflows fp32/],
                   ["sigma 1e39 (finite in fp64, Infinity in fp32)", { sigma: 1e39, amp: 1e-300 }, /overflows fp32/]];
  for (const [tag, over, re] of badOnes) {
    const bad1 = Object.assign({}, good, over);
    // the bad blob FIRST, in the MIDDLE and LAST: the check is on every entry, not on one
    for (const [where, lst, idx] of [["first", [bad1, good, good], 0], ["second", [good, bad1, good], 1],
                                     ["last", [good, good, bad1], 2]]) {
      const keep = new Float32Array(2 * NK * 4).fill(SENT_PACK);
      const msg = throws(() => pack(env, SQ, G, lst, keep));
      ok("a blob with " + tag + ", " + where + " in the list, THROWS naming blob " + idx + ", table untouched",
         re.test(msg) && new RegExp("blob " + idx + " ").test(msg) && Array.from(keep).every(v => v === SENT_PACK),
         msg.slice(0, 48) || "no throw");
    }
  }
  // ... and the largest weight that DOES fit is accepted (the refusal is the overflow, not size)
  const big = pack(env, SQ, G, [{ x: 1, y: 1, sigma: 0.5, amp: 1e36, pol: 1 }]);
  ok("  ... while a large but representable weight is packed (amp 1e36: w ~ 3e37 < fp32 max)",
     big.used[0] === 1 && isFinite(slot(big.a, 0, 0)[3]) && slot(big.a, 0, 0)[3] > 1e37, "w = " + e2(slot(big.a, 0, 0)[3]));
}

// ===========================================================================
// 3. the cross-check: production k space against the two references
// ===========================================================================
const CASES = [
  { tag: "one z+ blob, offset (0.75, 1.5)", grid: "SQ",
    list: [{ x: 0.75, y: 1.5, sigma: 0.375, amp: 0.25, pol: 1 }] },
  { tag: "one z- blob (psi must come out MINUS phi)", grid: "SQ",
    list: [{ x: 2.5, y: 0.25, sigma: 0.375, amp: -0.5, pol: -1 }] },
  { tag: "a DIFFERENT blob in each channel", grid: "SQ",
    list: [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
           { x: 2.5, y: 0.25, sigma: 0.375, amp: -0.5, pol: -1 }] },
  { tag: "three per channel, both signs (superposition)", grid: "SQ",
    list: [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
           { x: 4.25, y: 5.0, sigma: 0.375, amp: 0.75, pol: 1 },
           { x: 1.25, y: 0.75, sigma: 0.4375, amp: -0.625, pol: 1 },
           { x: 5.5, y: 3.25, sigma: 0.5, amp: 0.375, pol: -1 },
           { x: 3.0, y: 4.5, sigma: 0.375, amp: -0.5, pol: -1 },
           { x: 0.25, y: 6.0, sigma: 0.625, amp: 1.0, pol: -1 }] },   // wide: the images matter
  { tag: "a rectangular box (32x16, Ly = Lx/2)", grid: "RECT",
    list: [{ x: 0.75, y: 1.5, sigma: 0.375, amp: 0.5, pol: 1 },
           { x: 3.5, y: 2.25, sigma: 0.5, amp: -0.25, pol: -1 }] }
];
// REL_TOL is per mode over the modes above 1e-4 of the peak, and holds for one or two blobs.
// A SUM of many blobs cancels: the fp32 absolute error (~1e-7 of the peak) then sits on modes
// that are themselves 1e-4 of it, so the many-blob cases get REL_SUM (measured: 2e-6 with
// six blobs, 1.9e-5 with sixty-four) while keeping the same absolute bound.
const ABS_TOL = 5e-7, REL_TOL = 2e-6, REL_SUM = 1e-5;
function legCross(state) {
  const M = state.M, env = state.env;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  for (const c of CASES) {
    const P = c.grid === "SQ" ? SQ : RECT, G = c.grid === "SQ" ? state.G : state.R;
    const prod = runKick(M, G, pack(env, P, G, c.list).a);
    const r = compare(prod, refKick(P, G.nkx, G.nky, G.kmax, c.list), G.nm);
    ok(c.tag, r.abs < ABS_TOL && r.rel < REL_TOL && r.phi.peak > 0 && r.psi.peak > 0, r.note);
  }
  // the slots past BLOB_FORCE_MAX are really read: a full table of KICK_MAX per channel,
  // every blob different, against the reference summing all of them
  const G = state.G, full = [];
  for (let i = 0; i < G.kmax; i++) {
    full.push({ x: 0.19 * i + 0.3, y: 0.11 * i + 0.7, sigma: 0.375 + 0.0078125 * (i % 9), amp: 0.25 - 0.015625 * (i % 5), pol: 1 });
    full.push({ x: 6.1 - 0.17 * i, y: 0.13 * i + 0.2, sigma: 0.4375 + 0.0078125 * (i % 7), amp: -0.125 + 0.03125 * (i % 4), pol: -1 });
  }
  const pf = pack(env, SQ, G, full);
  const rf = compare(runKick(M, G, pf.a), refKick(SQ, G.nkx, G.nky, G.kmax, full), G.nm);
  ok("a FULL table, " + G.kmax + " different blobs per channel (every slot is read)",
     pf.used[0] === G.kmax && pf.used[1] === G.kmax && rf.abs < ABS_TOL && rf.rel < 4 * REL_SUM, rf.note);
  // ... and one more per channel is not: the reference capped at KICK_MAX still matches
  const over = full.concat([{ x: 1, y: 1, sigma: 0.5, amp: 5, pol: 1 }, { x: 2, y: 2, sigma: 0.5, amp: 5, pol: -1 }]);
  const po = pack(env, SQ, G, over);
  ok("  ... and the " + (G.kmax + 1) + "th of a channel adds nothing (bitwise the full table's kick)",
     po.dropped === 2 && sameBits(runKick(M, G, po.a), runKick(M, G, pf.a)) === -1, "dropped " + po.dropped);

  // ---- (b) the offline reference implementation's own numbers ---------------
  const fx = path.join(__dirname, "fixtures", "kick_offline.json");
  if (!fs.existsSync(fx)) { ok("fixtures/kick_offline.json is present", false, "run ../gen_kickfixture.py"); return; }
  const F = JSON.parse(fs.readFileSync(fx, "utf8"));
  ok("the fixture is the offline make_kick's, on this file's square grid",
     F.nx === SQ.nx && F.ny === SQ.ny && Math.abs(F.Lx - SQ.Lx) < 1e-15 && Math.abs(F.Ly - SQ.Ly) < 1e-15
     && /make_kick/.test(F.source) && F.cases.length >= 3, F.source);
  for (const c of F.cases) {
    const ref = Float64Array.from(c.fields);
    const prod = runKick(M, G, pack(env, SQ, G, c.list).a);
    const r = compare(prod, ref, G.nm);
    // The offline gaussian is MINIMUM-IMAGE (one image), where the analytic k-space form is
    // the fully periodic one: they differ by the neighbours' truncated tail,
    // exp(-(L/2)^2/2 sigma^2) of the peak -- 3e-9 at sigma = 0.5, which is why the fixture
    // stops there (at 0.625 it is 3e-6 and shows as 9e-4 on the weakest retained modes;
    // the music map's widest blob is 0.45). So the fp32 tolerances apply unchanged.
    ok("offline make_kick: " + c.tag, ref.length === 4 * G.nm && r.abs < ABS_TOL
       && r.rel < (c.list.length > 2 ? REL_SUM : REL_TOL)
       && r.phi.peak > 0 && r.psi.peak > 0, r.note);
    // the offline mask and k = 0 rule are the kernel's: where it wrote nothing, so did we
    let dis = 0, touched = 0;
    for (let i = 0; i < ref.length; i += 2) {
      const zr = ref[i] === 0 && ref[i + 1] === 0, zp = prod[i] === 0 && prod[i + 1] === 0;
      if (zr !== zp) dis++;
      if (!zp) touched++;
    }
    ok("  ... and the two agree on WHICH modes a kick touches (mask and k = 0)", dis === 0 && touched > 0,
       dis + " modes zero in one and not the other; " + touched + " touched");
  }
}

// ===========================================================================
// 4. what a tolerance cannot see
// ===========================================================================
function legExact(state) {
  const M = state.M, env = state.env, G = state.G, nm = G.nm;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const de = dealiasRef(SQ, G.nkx, G.nky);
  let mism = 0, nOut = 0;
  for (let m = 0; m < nm; m++) { if (de[m] !== G.gridB[4 * m]) mism++; if (!de[m]) nOut++; }
  ok("the 2/3 mask rebuilt here from the rule IS gridB's column 0", mism === 0,
     mism + " modes differ; " + nOut + " of " + nm + " outside the cut");

  // a deliberately narrow pair, so the gaussian is O(0.1) of its peak at the cut
  const narrow = [{ x: 0.75, y: 1.5, sigma: 0.25, amp: 0.5, pol: 1 },
                  { x: 3.0, y: 4.5, sigma: 0.25, amp: -0.5, pol: -1 }];
  const pk = pack(env, SQ, G, narrow).a;
  const seed = walk(4 * nm, 20261);
  const d0 = runKick(M, G, pk);                 // what the kick adds to nothing
  const f1 = runKick(M, G, pk, seed);           // ... and to a state that is nowhere zero
  const sb = bits(seed), fb = bits(f1);

  // ---- k = 0 and the out-of-band modes: untouched BIT FOR BIT ---------------
  let k0 = 0;
  for (const i of [0, 1, 2 * nm, 2 * nm + 1]) if (sb[i] !== fb[i]) k0++;
  ok("k = 0 is left untouched, bit for bit, in both halves (on a state that is non-zero there)",
     k0 === 0 && seed[0] !== 0 && d0[0] === 0 && d0[1] === 0 && d0[2 * nm] === 0 && d0[2 * nm + 1] === 0,
     k0 + " of 4 words moved");
  let out = 0, peak = 0, inMin = Infinity;
  for (let h = 0; h < 2; h++) for (let m = 0; m < nm; m++) {
    const i = 2 * (h * nm + m);
    peak = Math.max(peak, Math.hypot(d0[i], d0[i + 1]));
    if (!de[m] && (sb[i] !== fb[i] || sb[i + 1] !== fb[i + 1])) out++;
  }
  ok("every mode outside the 2/3 cut is left untouched, bit for bit (the kernel dealiases)",
     out === 0 && nOut > 0, out + " of " + (2 * nOut) + " out-of-band modes moved");
  // the rim just inside the cut, measured on |delta phi| + |delta psi| per mode: the two
  // blobs have equal and opposite amplitudes, so phi ALONE vanishes exactly wherever
  // k.(x1 - x2) is a multiple of 2*pi (two retained modes here) and psi carries it there
  let rimN = 0;
  for (let m = 1; m < nm; m++) {
    if (!de[m]) continue;
    const j = m % G.nky, ii = (m - j) / G.nky;
    const nb = [[ii + 1, j], [ii - 1, j], [ii, j + 1], [ii, j - 1]];
    if (nb.some(([u, v]) => u >= 0 && u < G.nkx && v >= 0 && v < G.nky && !de[u * G.nky + v])) {
      rimN++;
      inMin = Math.min(inMin, Math.hypot(d0[2 * m], d0[2 * m + 1])
                              + Math.hypot(d0[2 * (nm + m)], d0[2 * (nm + m) + 1]));
    }
  }
  ok("  ... and that is a cliff, not a rounding: the rim just inside the cut is > 1e-3 of the peak",
     inMin / peak > 1e-3 && rimN > 0, "smallest of " + rimN + " retained rim modes = " + e2(inMin / peak) + " of the peak");
  // ... including a -0 sitting outside the cut or at k = 0: adding +0 would flip its sign bit
  const neg = new Float32Array(4 * nm).fill(-0);
  const fn = runKick(M, G, pk, neg);
  let flipped = 0;
  for (let h = 0; h < 2; h++) for (let m = 0; m < nm; m++) {
    if (de[m] && m !== 0) continue;
    const i = 2 * (h * nm + m);
    if (bits(fn)[i] !== 0x80000000 || bits(fn)[i + 1] !== 0x80000000) flipped++;
  }
  ok("  ... down to the sign of a zero: a -0 there is still -0 (an early return, not `+ 0`)",
     flipped === 0, flipped + " untouched-class modes lost their sign bit");

  // ---- additive: fields + (what the kick adds to nothing), in fp32 ----------
  let addBad = 0, addMoved = 0;
  for (let i = 0; i < 4 * nm; i++) {
    if (Math.fround(seed[i] + d0[i]) !== f1[i]) addBad++;
    if (f1[i] !== seed[i]) addMoved++;
  }
  ok("the kick is ADDITIVE: on a non-zero state every word is fl32(state + delta), exactly",
     addBad === 0 && addMoved > nm, addBad + " words differ; " + addMoved + " of " + (4 * nm) + " moved");
  // ... so it is not idempotent: a second identical kick adds the same again
  const f2 = runKick(M, G, pk, f1);
  let twice = 0;
  for (let i = 0; i < 4 * nm; i++) if (Math.fround(f1[i] + d0[i]) !== f2[i]) twice++;
  ok("  ... and a second identical kick adds the same delta again", twice === 0, twice + " words differ");

  // ---- the Elsasser map, exactly --------------------------------------------
  const zp = runKick(M, G, pack(env, SQ, G, [narrow[0]]).a);
  const zm = runKick(M, G, pack(env, SQ, G, [narrow[1]]).a);
  let pEq = 0, mEq = 0, pNz = 0, mNz = 0;
  for (let i = 0; i < 2 * nm; i++) {
    if (bits(zp)[i] !== bits(zp)[2 * nm + i]) pEq++;
    if (zm[i] !== -zm[2 * nm + i]) mEq++;
    if (zp[i] !== 0) pNz++;
    if (zm[i] !== 0) mNz++;
  }
  ok("a z+-only kick gives delta psi == delta phi, bit for bit", pEq === 0 && pNz > nm / 2,
     pEq + " words differ over " + pNz + " non-zero");
  ok("a z--only kick gives delta psi == -delta phi, exactly", mEq === 0 && mNz > nm / 2,
     mEq + " words differ over " + mNz + " non-zero");
  // phi is HALF the channel sum: against blobBuild's own z+ modes for the same blob
  const bb = new Float32Array(2 * G.bmax * 4);
  env.run(`function(P, arr, list){ Solver.prototype.setBlobs.call({ p: P, _blobs: arr }, list); }`, SQ, bb, [narrow[0]]);
  const frc = new Float32Array(4 * nm);
  new M.WgslExec(new M.WgslParser().parse(G.S.blobBuild)).dispatchWorkgroups(
    "main", [2 * nm, 1, 1], { 0: { 0: frc, 1: G.gridA, 2: G.gridB, 3: bb } });
  let half = 0, halfN = 0;
  for (let m = 1; m < nm; m++) for (let c = 0; c < 2; c++) {
    if (frc[2 * m + c] === 0) continue;
    halfN++;
    if (Math.fround(0.5 * frc[2 * m + c]) !== zp[2 * m + c]) half++;
  }
  ok("  ... and delta phi is exactly HALF the z+ modes blobBuild writes for the same blob",
     half === 0 && halfN > nm / 2, half + " of " + halfN + " words differ");
  // the same blob in BOTH channels is a pure velocity kick: psi exactly untouched
  const both = runKick(M, G, pack(env, SQ, G, [narrow[0], Object.assign({}, narrow[0], { pol: -1 })]).a);
  let psiNz = 0, phiBad = 0;
  for (let i = 0; i < 2 * nm; i++) {
    if (both[2 * nm + i] !== 0) psiNz++;
    if (both[i] !== Math.fround(2 * zp[i])) phiBad++;
  }
  ok("  ... and the SAME blob in both channels moves phi alone (delta psi == 0, delta phi doubled)",
     psiNz === 0 && phiBad === 0, psiNz + " psi words non-zero, " + phiBad + " phi words off");

  // ---- reality --------------------------------------------------------------
  const wide = runKick(M, G, pack(env, SQ, G, CASES[2].list).a);
  let pairs = 0, broke = 0, nyq = 0;
  for (let h = 0; h < 2; h++) for (let i = 0; i < G.nkx; i++) {
    const m = i * G.nky, mm = ((G.nkx - i) % G.nkx) * G.nky;
    if (!de[m] || !de[mm] || i === 0) continue;
    const a = 2 * (h * nm + m), b = 2 * (h * nm + mm);
    pairs++;
    if (wide[a] !== wide[b] || wide[a + 1] !== -wide[b + 1]) broke++;
  }
  for (let i = 0; i < G.nkx; i++) {
    if (de[i * G.nky + G.nky - 1]) nyq++;
    if (i === SQ.nx / 2 && de[i * G.nky]) nyq++;
  }
  ok("the ky = 0 row of what a kick adds is EXACTLY hermitian, in both halves",
     broke === 0 && pairs > 0, broke + " of " + pairs + " (kx, -kx) pairs broken, bit for bit");
  ok("  ... and the ky / kx Nyquist lines, where the analytic form is not exact, are outside the cut",
     nyq === 0, nyq + " retained modes on a Nyquist line");

  // ---- a far-away position is the same blob, and cannot poison the fields ----
  // packed modulo the box (leg 2). Executed: nothing non-finite comes out, and the blob
  // given at (L + 0.75, -0.5) is the blob at (0.75, L - 0.5) to fp32 phase accuracy.
  const far = runKick(M, G, state.farBlob);
  let nonFinite = 0;
  for (let i = 0; i < far.length; i++) if (!isFinite(far[i])) nonFinite++;
  ok("a kick holding x0 = 1e30, y0 = -1e30 leaves every word of both fields finite",
     nonFinite === 0, nonFinite + " non-finite words of " + far.length);
  const kA = runKick(M, G, pack(env, SQ, G, [{ x: SQ.Lx + 0.75, y: -0.5, sigma: 0.5, amp: 1, pol: 1 }]).a);
  const kB = runKick(M, G, pack(env, SQ, G, [{ x: 0.75, y: SQ.Ly - 0.5, sigma: 0.5, amp: 1, pol: 1 }]).a);
  let wrapMiss = 0, wrapPeak = 0;
  for (let i = 0; i < kA.length; i++) { wrapMiss = Math.max(wrapMiss, Math.abs(kA[i] - kB[i])); wrapPeak = Math.max(wrapPeak, Math.abs(kB[i])); }
  ok("  ... and a position outside the box is the SAME kick as its image inside it",
     wrapMiss / wrapPeak < 2e-5 && wrapPeak > 0, "max |difference| / peak = " + e2(wrapMiss / wrapPeak));

  // ---- a zero table adds nothing, to any state ------------------------------
  ok("an all-zero table leaves every retained mode bit for bit as it was",
     sameBits(runKick(M, G, pack(env, SQ, G, []).a, seed), seed) === -1);
}

// ===========================================================================
// 5. kick(): the dispatch, the upload, the refusals -- on the stub device
// ===========================================================================
// THE MUSIC PAGE'S CONFIGURATION, constructed: kick: true together with the drag, at hyper 2
// with the offline script's dissipation rule and the OU forcing zeroed (MUSIC_PLAN). On a
// 64^2 grid for the main rows: there NM = 2112, so the energy reduction's nPartM = 3 -- at
// 32^2 nPartM is 1 and a kick that dispatched energyPartial ONCE would be indistinguishable
// from one that dispatched it nPartM times. The dispatch-COUNT row is repeated on 32^2 and
// 32x16, where NM / 64 is not an integer (8.5 and 4.5) and ceil differs from floor.
const musicP = (nx, ny, Ly) => ({ nx: nx, ny: ny, Lx: 2 * Math.PI, Ly: Ly || 2 * Math.PI, hyper: 2,
  diss: 3 / Math.pow(nx / 6, 4), fshell: [1, 3], epsP: 0, epsM: 0, kick: true, mu: 1 });
const W64 = musicP(64, 64);
// what the device was asked to do during fn(): queue.writeBuffer and queue.submit, and --
// one level down -- every command encoder made, with its clearBuffer / copyBufferToBuffer /
// copyTextureToBuffer calls, the passes it began and whether each was ENDED, and finish().
// Each event carries the dispatch count at that moment, which is what orders an upload or
// a submit against the dispatches. The stub's own dispatch log is filled at ENCODE time and
// says nothing about any of this.
function spy(env, dev) {
  const ev = [], q = dev.queue, o = { wb: q.writeBuffer, sub: q.submit, enc: dev.createCommandEncoder };
  const at = () => env.gpu.dispatches.length;
  q.writeBuffer = function (b, off, data) {
    const bytes = data.buffer ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(0);
    ev.push({ op: "write", b: b, off: off, bytes: Uint8Array.from(bytes), at: at() });
    return o.wb.call(q, b, off, data);
  };
  q.submit = function (list) { ev.push({ op: "submit", n: list ? list.length : -1, at: at() }); return o.sub.apply(q, arguments); };
  dev.createCommandEncoder = function () {
    const e = o.enc.apply(dev, arguments);
    ev.push({ op: "encoder", at: at() });
    for (const m of ["clearBuffer", "copyBufferToBuffer", "copyTextureToBuffer"]) {
      const f = e[m];
      e[m] = function () { ev.push({ op: m, args: Array.from(arguments), at: at() }); return f.apply(e, arguments); };
    }
    for (const m of ["beginComputePass", "beginRenderPass"]) {
      const f = e[m];
      e[m] = function () {
        const p = f.apply(e, arguments), end = p.end;
        ev.push({ op: m, at: at() });
        p.end = function () { ev.push({ op: "end", at: at() }); return end.apply(p, arguments); };
        return p;
      };
    }
    const fin = e.finish;
    e.finish = function () { ev.push({ op: "finish", at: at() }); return fin.apply(e, arguments); };
    return e;
  };
  return { ev: ev, reset() { env.gpuReset(); ev.length = 0; },
           ops: () => ev.map(x => x.op).join(","),
           restore() { q.writeBuffer = o.wb; q.submit = o.sub; dev.createCommandEncoder = o.enc; } };
}
// everything about a solver a kick has no business changing, as comparable values: the
// identity of every buffer, pipeline and bind group, the flags and counters, the CPU-side
// arrays. `nkicks` and `_cflStale` are returned beside it, since those two SHOULD move.
const SNAP = `function(name){
  const s = globalThis[name], flat = o => Object.keys(o).sort().map(k => [k, o[k]]);
  return { ids: [].concat(flat(s.buf), flat(s.pl), flat(s.bg)).map(e => e[1]),
           keys: [].concat(flat(s.buf), flat(s.pl), flat(s.bg)).map(e => e[0]).join(","),
           plain: JSON.stringify({ blobMode: s.blobMode, nsteps: s.nsteps, p: s.p, ns: s.ns, nb: s.nb,
             blobs: Array.from(s._blobs), disp: s.disp.length, gk: s.g.kick, nm: s.g.nm,
             gridB: Array.from(s.g.gridB), gridA: Array.from(s.g.gridA) }),
           nkicks: s.nkicks, stale: s._cflStale };
}`;
const sameSnap = (a, b) => a.keys === b.keys && a.plain === b.plain && a.ids.length === b.ids.length &&
  a.ids.every((x, i) => x === b.ids[i]);
function legWiring(state) {
  const env = state.env, G = gridOf(env, W64), SQ = W64;
  const dev = env.run("function(){ return device; }");
  const sp = spy(env, dev), wbOf = () => sp.ev.filter(e => e.op === "write"), subOf = () => sp.ev.filter(e => e.op === "submit");
  const names = () => env.gpu.dispatches.map(d => d.pipe.__name);
  const shape = () => env.gpu.dispatches.map(d => d.pipe.__name + ":" + d.d.join("x")).join(" ");
  const count = n => names().filter(x => x === n).length;
  try {
    env.run(`function(P){ globalThis.__ks = new Solver(device, P); }`, SQ);
    const info = () => JSON.parse(env.run(`function(){ const s = __ks; return JSON.stringify({
      nk: s.nkicks, ns: s.nsteps, nm: s.g.nm, npm: s.nPartM, ksize: s.buf.kicks.size, gk: s.g.kick,
      code: s.pl.blobKick.__code, name: s.pl.blobKick.__name, mu: s.p.mu, lin0: s.g.gridB[1] }); }`));
    const i0 = info();
    ok("the music configuration builds: kick: true WITH the drag (mu 1, hyper 2), pipeline = the emitted kernel",
       i0.name === "blobKick" && i0.code === G.src && i0.gk === true && i0.mu === 1 && i0.lin0 === -0.5, i0.name);
    ok("  ... and a kicks buffer of 2 * KICK_MAX vec4", i0.ksize === 2 * G.kmax * 16, i0.ksize + " bytes");
    const bufs = env.run("function(){ const s = __ks; return [s.buf.fields, s.buf.gridA, s.buf.gridB, s.buf.kicks]; }");
    const bgb = env.run("function(){ return __ks.bg.blobKick.__buffers; }");
    ok("  ... bound as [fields, gridA, gridB, kicks], in the kernel's binding order",
       bgb.length === 4 && bufs.every((b, i) => b === bgb[i]));

    // ---- one kick ------------------------------------------------------------
    const list = [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
                  { x: 2.5, y: 0.25, sigma: 0.375, amp: -0.5, pol: -1 }];
    const before = env.run(SNAP, "__ks");
    sp.reset();
    const r = JSON.parse(env.run("function(l){ return JSON.stringify(__ks.kick(l)); }", list));
    const after = env.run(SNAP, "__ks");
    ok("kick() dispatches blobKick, then the energy refresh, and nothing else",
       names().join(",") === "blobKick,energyPartial,energyFinal", names().join(","));
    const d = env.gpu.dispatches;
    ok("  ... blobKick over ceil(NM / 64) workgroups, ONE invocation per mode",
       d.length === 3 && d[0].d[0] === Math.ceil(i0.nm / 64) && d[0].d[1] === 1 && d[0].d[2] === 1,
       d.length ? d[0].d.join("x") + " for NM = " + i0.nm : "no dispatch");
    ok("  ... through its own bind group", d.length === 3 && d[0].bg === env.run("function(){ return __ks.bg.blobKick; }"));
    ok("  ... energyPartial over nPartM = " + i0.npm + " (not 1), energyFinal once",
       d.length === 3 && d[1].d[0] === i0.npm && i0.npm > 1 && d[2].d[0] === 1 &&
       d[1].bg === env.run("function(){ return __ks.bg.energyPartial; }") &&
       d[2].bg === env.run("function(){ return __ks.bg.energyFinal; }"),
       d.length === 3 ? d[1].d[0] + " and " + d[2].d[0] : "no dispatch");
    const want = pack(env, SQ, G, list).a;
    const up = wbOf().filter(w => w.b === bufs[3]);
    ok("  ... after ONE upload of the packed table to the kicks buffer, at offset 0, before the dispatch",
       wbOf().length === 1 && up.length === 1 && up[0].off === 0 && up[0].at === 0 &&
       sameBits(new Float32Array(up[0].bytes.buffer), want) === -1,
       wbOf().length + " writeBuffer call(s), " + (up[0] ? up[0].bytes.length : 0) + " bytes");
    // the ENCODER, op for op: one encoder, one compute pass begun and ENDED around the three
    // dispatches, finish, submit -- and NOTHING else: no clearBuffer (a cleared `fields`
    // would be a kick onto zero), no copyBufferToBuffer, no render pass
    ok("  ... one encoder: write, encoder, beginComputePass, end, finish, submit -- in that order, nothing else",
       sp.ops() === "write,encoder,beginComputePass,end,finish,submit", sp.ops());
    const evEnd = sp.ev.filter(e => e.op === "end")[0], evBeg = sp.ev.filter(e => e.op === "beginComputePass")[0];
    ok("  ... the pass is begun before the first dispatch and ENDED after the third",
       !!evBeg && !!evEnd && evBeg.at === 0 && evEnd.at === 3);
    ok("  ... and SUBMITTED once, one command buffer, after it was ended and finished",
       subOf().length === 1 && subOf()[0].n === 1 && subOf()[0].at === 3, subOf().length + " submit(s)");
    ok("  ... it changes NOTHING else about the solver: every buffer / pipeline / bind group identity, "
       + "blobMode, nsteps, the parameters, the blob table, gridA / gridB",
       sameSnap(before, after) && JSON.parse(after.plain).blobMode === false,
       sameSnap(before, after) ? "identical snapshot" : "snapshot moved");
    ok("  ... except the two it should: nkicks " + before.nkicks + " -> " + after.nkicks + ", and the CFL marked stale",
       after.nkicks === before.nkicks + 1 && before.stale === false && after.stale === true);
    const i1 = info();
    ok("  ... and returns _packKicks' report", r.used[0] === 1 && r.used[1] === 1 && r.dropped === 0, JSON.stringify(r));

    // ---- kick, then step: the step is an ordinary OU step with the CFL forced ----
    // (review F1) `step(cflEvery)` recomputes dt only when nsteps % cflEvery == 0, so the dt
    // in sc[0] after a kick is a PRE-kick dt for up to cflEvery - 1 steps -- after silence
    // the quiescent floor. A kick therefore marks the CFL stale and the NEXT step recomputes
    // it regardless. Two solvers from the same parameters, K (kicked) and R (never kicked),
    // stepped side by side with cflEvery = 4.
    env.run(`function(P){ globalThis.__kK = new Solver(device, P); globalThis.__kR = new Solver(device, P); }`, SQ);
    const stepLog = (name, ce) => { sp.reset(); env.run("function(n, ce){ globalThis[n].step(ce); }", name, ce);
                                    return { shape: shape(), cfl: count("cflFinal"), part: count("cflPartial"),
                                             blob: count("blobBuild"), ou: count("ou"), ops: sp.ops() }; };
    const K = [], R = [];
    for (let n = 0; n < 2; n++) { K.push(stepLog("__kK", 4)); R.push(stepLog("__kR", 4)); }   // nsteps 0, 1
    ok("before any kick the two solvers encode identical steps (CFL at nsteps 0, none at 1)",
       K[0].shape === R[0].shape && K[1].shape === R[1].shape && R[0].cfl === 1 && R[1].cfl === 0,
       "cflFinal per step: " + R.map(x => x.cfl).join(","));
    sp.reset();
    env.run("function(l){ __kK.kick(l); }", list);                                             // nsteps 2: a skip step
    K.push(stepLog("__kK", 4)); R.push(stepLog("__kR", 4));                                    // nsteps 2
    ok("the step AFTER a kick dispatches cflPartial and cflFinal exactly once, at an nsteps that skips it",
       K[2].cfl === 1 && K[2].part === 1 && R[2].cfl === 0 && R[2].part === 0,
       "kicked " + K[2].cfl + ", never-kicked " + R[2].cfl + " (nsteps 2, cflEvery 4)");
    K.push(stepLog("__kK", 4)); R.push(stepLog("__kR", 4));                                    // nsteps 3
    ok("  ... and the step after THAT does not (the mark is cleared by the step that used it)",
       K[3].cfl === 0 && K[3].shape === R[3].shape && env.run("function(){ return __kK._cflStale; }") === false);
    K.push(stepLog("__kK", 4)); R.push(stepLog("__kR", 4));                                    // nsteps 4: natural
    ok("  ... the kicked step is, dispatch for dispatch, a natural CFL step: same pipelines, same counts, same order",
       K[2].shape === R[4].shape && K[4].shape === R[4].shape && R[4].cfl === 1,
       K[2].shape.split(" ").length + " dispatches");
    ok("  ... and still the OU step: ou + scale, no blobBuild (a kick does not switch blob mode on)",
       K[2].ou === 1 && K[2].blob === 0 && /scale:1x1x1$/.test(K[2].shape) && K[2].ops === R[4].ops, K[2].ops);
    // the never-kicked solver is the pre-kick code path: its CFL schedule is nsteps % cflEvery
    // and nothing else, over two full cycles and at cflEvery 1
    for (let n = 5; n < 12; n++) R.push(stepLog("__kR", 4));
    ok("a solver that is NEVER kicked recomputes dt at nsteps % cflEvery == 0 and only then (12 steps, cflEvery 4)",
       R.map(x => x.cfl).join("") === "100010001000" && env.run("function(){ return __kR._cflStale; }") === false,
       "cflFinal per step: " + R.map(x => x.cfl).join(""));
    const noCfl = R[1].shape, withCfl = R[0].shape;
    ok("  ... with exactly two step shapes, differing by the two CFL dispatches",
       R.every((x, n) => x.shape === (n % 4 === 0 ? withCfl : noCfl)) &&
       withCfl.split(" ").length === noCfl.split(" ").length + 2);
    // the mark is one bit, not a counter, and only a kick that kicked sets it
    env.run("function(l){ __kK.kick(l); __kK.kick(l); }", list);                               // nsteps 5
    const k5 = stepLog("__kK", 4), k6 = stepLog("__kK", 4);
    ok("two kicks before a step force ONE recompute, on that step only", k5.cfl === 1 && k6.cfl === 0);
    env.run("function(){ __kK.kick([]); __kK.kick(null); __kK.kick([{ x: 1, y: 1, sigma: 0.5, amp: 0, pol: 1 }]); }");
    const k7 = stepLog("__kK", 4);                                                             // nsteps 7
    ok("an empty / null / all-zero-amplitude kick marks nothing: the next step keeps its schedule",
       k7.cfl === 0 && k7.shape === noCfl);
    throws(() => env.run("function(l){ __kK.kick(l); }", [{ x: 1, y: 1, sigma: 0.5, amp: NaN, pol: 1 }]));
    ok("  ... and neither does a REFUSED kick", env.run("function(){ return __kK._cflStale; }") === false);
    const k8 = stepLog("__kK", 4);                                                             // nsteps 8: natural
    env.run("function(l){ __kK.kick(l); }", list);                                             // nsteps 9
    const k9 = stepLog("__kK", 1), k10 = stepLog("__kK", 1);
    ok("at cflEvery 1 (and on a natural CFL step) a kick changes nothing: one CFL per step either way",
       k8.cfl === 1 && k9.cfl === 1 && k10.cfl === 1 && k9.shape === withCfl);

    // ---- a kick in ONE channel is still a kick --------------------------------
    for (const [tag, l, u] of [["z+ only", [list[0]], [1, 0]], ["z- only", [list[1]], [0, 1]]]) {
      sp.reset();
      const ru = JSON.parse(env.run("function(l){ return JSON.stringify(__ks.kick(l)); }", l));
      ok("a " + tag + " list dispatches, uploads and submits like any other",
         env.gpu.dispatches.length === 3 && wbOf().length === 1 && subOf().length === 1 &&
         ru.used[0] === u[0] && ru.used[1] === u[1],
         env.gpu.dispatches.length + " dispatches, used " + JSON.stringify(ru.used));
    }
    const i1b = info();
    ok("  ... and each was counted", i1b.nk === i1.nk + 2, "nkicks " + i1.nk + " -> " + i1b.nk);
    i1.nk = i1b.nk;
    // ---- nothing to do ---------------------------------------------------------
    for (const [tag, l] of [["an empty list", []], ["null", null],
                            ["a list of zero-amplitude blobs", [{ x: 1, y: 1, sigma: 0.5, amp: 0, pol: 1 }]]]) {
      sp.reset();
      env.run("function(l){ __ks.kick(l); }", l);
      ok(tag + ": no dispatch, no upload, no encoder, no submit, no count",
         env.gpu.dispatches.length === 0 && sp.ev.length === 0 && info().nk === i1.nk);
    }
    // ---- a refused list reaches nothing ---------------------------------------
    for (const [tag, bad] of [["a NaN amplitude", { x: 1, y: 1, sigma: 0.5, amp: NaN, pol: 1 }],
                              ["an fp32-infinite weight (sigma 1e13)", { x: 1, y: 1, sigma: 1e13, amp: 1, pol: 1 }]]) {
      sp.reset();
      const msg = throws(() => env.run("function(l){ __ks.kick(l); }", [list[0], bad]));
      ok("a list with " + tag + " throws out of kick() with no dispatch, upload, encoder, submit or count",
         /blob 1/.test(msg) && env.gpu.dispatches.length === 0 && sp.ev.length === 0 && info().nk === i1.nk,
         msg.slice(0, 50) || "no throw");
    }
    // ---- overflow is reported through kick() too ------------------------------
    const many = [];
    for (let i = 0; i < G.kmax + 2; i++) many.push({ x: 0.1 * i, y: 0, sigma: 0.5, amp: 1, pol: -1 });
    sp.reset();
    const ro = JSON.parse(env.run("function(l){ return JSON.stringify(__ks.kick(l)); }", many));
    ok("an over-full channel through kick(): used [0, " + G.kmax + "], dropped 2, and it IS applied",
       ro.used[0] === 0 && ro.used[1] === G.kmax && ro.dropped === 2 &&
       env.gpu.dispatches.length === 3 && wbOf().length === 1, JSON.stringify(ro));

    // ---- ceil, not floor: grids where NM / 64 is not an integer ----------------
    for (const [tag, P] of [["32x32", musicP(32, 32)], ["32x16 (Ly = pi)", musicP(32, 16, Math.PI)]]) {
      env.run(`function(P){ globalThis.__kc = new Solver(device, P); }`, P);
      sp.reset();
      env.run("function(l){ __kc.kick(l); }", list);
      const nm = env.run("function(){ return __kc.g.nm; }"), got = env.gpu.dispatches[0] ? env.gpu.dispatches[0].d[0] : -1;
      ok("on " + tag + ", NM / 64 = " + (nm / 64) + ": blobKick is dispatched over the CEILING, " + Math.ceil(nm / 64),
         nm % 64 !== 0 && got === Math.ceil(nm / 64) && got * 64 >= nm, got + " workgroups for NM = " + nm);
      env.run("function(){ __kc.destroy(); }");
    }

    // ---- a solver that did not ask --------------------------------------------
    env.run(`function(P){ globalThis.__k0 = new Solver(device, Object.assign({}, P, { kick: false })); }`, SQ);
    sp.reset();
    const m0 = throws(() => env.run("function(l){ __k0.kick(l); }", list));
    const has = JSON.parse(env.run(`function(){ return JSON.stringify({ pl: !!__k0.pl.blobKick,
      bg: !!__k0.bg.blobKick, buf: !!__k0.buf.kicks, gk: __k0.g.kick, stale: __k0._cflStale }); }`));
    ok("a solver built WITHOUT kick: true has no kernel, bind group or buffer, and kick() throws",
       /kick: true/.test(m0) && !has.pl && !has.bg && !has.buf && has.gk === false && has.stale === false &&
       env.gpu.dispatches.length === 0 && sp.ev.length === 0, m0.slice(0, 60) || "no throw");
    // ... and it steps on the unchanged schedule too (the default path of every existing page)
    const z = [];
    for (let n = 0; n < 5; n++) { sp.reset(); env.run("function(){ __k0.step(4); }"); z.push(count("cflFinal")); }
    ok("  ... and steps on the same schedule: CFL at nsteps % cflEvery == 0 only", z.join("") === "10001", z.join(""));
    env.run("function(){ __ks.destroy(); __k0.destroy(); __kK.destroy(); __kR.destroy(); }");
  } finally { sp.restore(); }
}

// ===========================================================================
// 6. the drag: gridB's linear column, the refusals, and the decay it produces
// ===========================================================================
// the production stage kernel, executed: one LSRK33 step with a ZERO right-hand side is
// three dispatches, each fields <- exp(lin * dt * gamma_s) * fields (delta stays zero)
function runSteps(M, G, src, gridB, fields, dt, nsteps) {
  const nm = G.nm, f = Float32Array.from(fields), delta = new Float32Array(4 * nm), rhs = new Float32Array(4 * nm);
  const sc = new Float32Array(12); sc[0] = dt;
  // the one respelling (header): `(64)` -> `(64, 1, 1)`, once, and nothing else
  const parts = src.split("@workgroup_size(64)");
  if (parts.length !== 2) throw new Error("stage: expected exactly one @workgroup_size(64)");
  const ex = new M.WgslExec(new M.WgslParser().parse(parts.join("@workgroup_size(64, 1, 1)")));
  for (let n = 0; n < nsteps; n++) {
    delta.fill(0);                                   // `step` clears delta first
    for (let s = 0; s < 3; s++)
      ex.dispatchWorkgroups("main", [Math.ceil(2 * nm / 64), 1, 1], { 0: { 0: f, 1: delta, 2: rhs, 3: gridB, 4: sc,
        5: Float32Array.of(G.lsrk[0][s], G.lsrk[1][s], G.lsrk[2][s], 0) } });
  }
  return f;
}
function energyOf(G, f) {             // sum yfac * ksq * |f|^2 over both halves (any normalization)
  let e = 0;
  for (let h = 0; h < 2; h++) for (let m = 0; m < G.nm; m++) {
    const i = 2 * (h * G.nm + m);
    e += G.gridB[4 * m + 3] * G.gridA[4 * m + 2] * (f[i] * f[i] + f[i + 1] * f[i + 1]);
  }
  return e;
}
function legDrag(state) {
  const M = state.M, env = state.env, G = state.G, nm = G.nm;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const dev = env.run("function(){ return device; }");
  const wb = [], q = dev.queue, orig = q.writeBuffer;
  q.writeBuffer = function (b, off, data) {
    const bytes = data.buffer ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(0);
    wb.push({ b: b, off: off, bytes: Uint8Array.from(bytes) });
    return orig.call(q, b, off, data);
  };
  // build a solver, return its gridB (as the CPU holds it) and what was uploaded to the
  // gridB buffer; `after` is run on it first (a method call, by source)
  const BUILD = `function(P, after){
    const s = globalThis.__ds = new Solver(device, P);
    if (after) (new Function("s", after))(s);
    return { gridB: Array.from(s.g.gridB), mu: s.p.mu, hasMuInG: "mu" in s.g };
  }`;
  const build = (P, after) => {
    wb.length = 0;
    const r = env.run(BUILD, P, after || "");
    r.gridB = Float32Array.from(r.gridB);
    const gb = env.run("function(){ return __ds.buf.gridB; }");
    const ups = wb.filter(w => w.b === gb);
    r.uploaded = ups.length ? new Float32Array(ups[ups.length - 1].bytes.buffer) : null;
    r.nUp = ups.length;
    return r;
  };
  try {
    const DP = Object.assign({}, SQ, { diss: 1e-3, hyper: 1 });      // a column that is not all ~0
    const base = Float32Array.from(env.run("function(P){ return Array.from(makeGrid(P).gridB); }", DP));

    // ---- mu = 0 is not a code path at all ------------------------------------
    for (const [tag, P] of [["absent", DP], ["0", Object.assign({}, DP, { mu: 0 })]]) {
      const s = build(P);
      ok("mu " + tag + ": the solver's gridB is makeGrid's, bit for bit, and so is the upload",
         sameBits(s.gridB, base) === -1 && s.uploaded && sameBits(s.uploaded, base) === -1 && s.mu === 0,
         "first differing word " + sameBits(s.gridB, base));
    }
    // refreshDissipation at mu = 0 is the pre-drag formula, bit for bit (it rebuilds from
    // gridA's fl32 ksq, which is its own long-standing arithmetic -- transcribed here)
    const A = G.gridA, oldRefresh = (diss, hyper) => {
      const g = Float32Array.from(base);
      for (let m = 0; m < nm; m++) g[4 * m + 1] = -diss * Math.pow(A[4 * m + 2], hyper);
      return g;
    };
    const r0 = build(DP, "s.refreshDissipation(2e-3, 2);");
    ok("refreshDissipation at mu = 0 writes exactly what it always wrote",
       sameBits(r0.gridB, oldRefresh(2e-3, 2)) === -1 && sameBits(r0.uploaded, r0.gridB) === -1 && r0.nUp === 2);
    // no WGSL moves with mu: the drag is the column and nothing else
    env.gpuReset(); build(DP);
    const sh0 = env.gpu.shaders.map(s => s.label + "\u0000" + s.code).join("\u0001");
    env.gpuReset(); build(Object.assign({}, DP, { mu: 0.75 }));
    const sh1 = env.gpu.shaders.map(s => s.label + "\u0000" + s.code).join("\u0001");
    ok("a solver with mu != 0 compiles the SAME shader modules, text for text (no WGSL carries the drag)",
       sh0 === sh1 && sh0.length > 1000, env.gpu.shaders.length + " modules");

    // ---- mu != 0: -mu/2 on the linear column, nothing else -------------------
    const MU = 0.75, sm = build(Object.assign({}, DP, { mu: MU }));
    let linBad = 0, otherBad = 0, linMin = 0;
    for (let m = 0; m < nm; m++) {
      if (sm.gridB[4 * m + 1] !== Math.fround(base[4 * m + 1] - 0.5 * MU)) linBad++;
      for (const c of [0, 2, 3]) if (bits(sm.gridB)[4 * m + c] !== bits(base)[4 * m + c]) otherBad++;
      linMin = Math.min(linMin, base[4 * m + 1]);
    }
    ok("mu = 0.75: gridB.y is makeGrid's minus mu/2 = 0.375 at EVERY mode (k = 0 and past the cut included)",
       linBad === 0 && sm.gridB[1] === -0.375, linBad + " modes off; lin(k=0) = " + sm.gridB[1]
       + ", most negative base lin " + e2(linMin));
    ok("  ... the dealias, fmask and yfac columns are untouched", otherBad === 0, otherBad + " words moved");
    ok("  ... and that IS what is uploaded", sm.uploaded && sameBits(sm.uploaded, sm.gridB) === -1);
    ok("  ... `mu` stays a solver parameter: the shader constants object never sees it", sm.hasMuInG === false);
    // the dissipation-rate readout reads this same column (energyPartial: -B.y * (ek + em)),
    // so with the drag on it is the total sink. Asserted as text: the reduction kernels do
    // not run on the interpreter.
    ok("  ... energyPartial's rate lane reads gridB.y, so sc[6] becomes hyperviscous + mu*E",
       /acc = acc \+ vec4<f32>\(ek, em, -B\.y \* \(ek \+ em\)/.test(G.S.energyPartial) &&
       /let B: vec4<f32> = gridB\[m\];/.test(G.S.energyPartial));

    // ---- refreshDissipation and setDrag carry it -----------------------------
    const rr = build(Object.assign({}, DP, { mu: MU }), "s.refreshDissipation(2e-3, 2);");
    const wantR = oldRefresh(2e-3, 2);
    for (let m = 0; m < nm; m++) wantR[4 * m + 1] = wantR[4 * m + 1] - 0.5 * MU;
    ok("refreshDissipation rebuilds the column WITH the drag (new diss/hyper, same mu)",
       sameBits(rr.gridB, wantR) === -1 && sameBits(rr.uploaded, rr.gridB) === -1,
       "first differing word " + sameBits(rr.gridB, wantR));
    const sd = build(DP, "s.setDrag(0.5);");
    const wantS = oldRefresh(DP.diss, DP.hyper);
    for (let m = 0; m < nm; m++) wantS[4 * m + 1] = wantS[4 * m + 1] - 0.25;
    ok("setDrag(0.5) on a live solver: p.mu = 0.5, the column rebuilt with -0.25, and uploaded",
       sd.mu === 0.5 && sameBits(sd.gridB, wantS) === -1 && sameBits(sd.uploaded, sd.gridB) === -1 && sd.nUp === 2);
    const sb = build(Object.assign({}, DP, { mu: MU }), "s.setDrag(0);");
    ok("setDrag(0) takes it off again: the column is refreshDissipation's mu = 0 one, bit for bit",
       sb.mu === 0 && sameBits(sb.gridB, oldRefresh(DP.diss, DP.hyper)) === -1 && sameBits(sb.uploaded, sb.gridB) === -1);

    // ---- what is refused ------------------------------------------------------
    const ctor = P => throws(() => env.run("function(P){ new Solver(device, P); }", P));
    for (const [tag, over, re] of [
      ["mu != 0 with Pm = 2", { mu: MU, pm: 2 }, /Pm/], ["mu != 0 with Pm = 0", { mu: MU, pm: 0 }, /Pm/],
      ["mu != 0 with the maintained-equilibrium source", { mu: MU, eqsrc: true }, /eqsrc/],
      ["mu < 0", { mu: -0.5 }, /mu must be/], ["mu = NaN", { mu: NaN }, /mu must be/],
      ["mu = Infinity", { mu: Infinity }, /mu must be/], ["mu = \"0.5\" (a string)", { mu: "0.5" }, /mu must be/]]) {
      const msg = ctor(Object.assign({}, DP, over));
      ok("the constructor THROWS on " + tag, re.test(msg), msg.slice(0, 70) || "no throw");
    }
    ok("  ... while Pm = 2 and eqsrc with mu = 0 still build (the refusal is the combination)",
       ctor(Object.assign({}, DP, { pm: 2 })) === "" && ctor(Object.assign({}, DP, { eqsrc: true })) === "" &&
       ctor(Object.assign({}, DP, { pm: 2, mu: 0 })) === "");
    // a refused setDrag leaves the solver as it was: p.mu, the CPU column and the GPU copy
    const keep = build(Object.assign({}, DP, { mu: MU }));
    const colBefore = Float32Array.from(keep.gridB);
    wb.length = 0;
    const ms = throws(() => env.run("function(){ __ds.setDrag(-1); }"));
    const aft = env.run("function(){ return { mu: __ds.p.mu, gridB: Array.from(__ds.g.gridB) }; }");
    const gbBuf = env.run("function(){ return __ds.buf.gridB; }");
    const lastUp = wb.filter(w => w.b === gbBuf).pop();
    // (the restore goes through refreshDissipation, so the column is its fl32-ksq rebuild)
    const wantK = oldRefresh(DP.diss, DP.hyper);
    for (let m = 0; m < nm; m++) wantK[4 * m + 1] = wantK[4 * m + 1] - 0.5 * MU;
    ok("a refused setDrag(-1) throws and leaves mu, the column and the uploaded copy at the old drag",
       /mu must be/.test(ms) && aft.mu === MU && sameBits(Float32Array.from(aft.gridB), wantK) === -1 &&
       lastUp && sameBits(new Float32Array(lastUp.bytes.buffer), wantK) === -1 &&
       Math.abs(colBefore[4 * 5 + 1] - wantK[4 * 5 + 1]) < 1e-6, ms.slice(0, 50) || "no throw");

    for (const [tag, arg] of [["\"0.5\" (a string)", "\"0.5\""], ["NaN", "NaN"], ["Infinity", "Infinity"]]) {
      const m2 = throws(() => env.run("function(){ __ds.setDrag(" + arg + "); }"));
      ok("  ... and so is setDrag(" + tag + "): refused, mu still " + MU,
         /mu must be/.test(m2) && env.run("function(){ return __ds.p.mu; }") === MU, m2.slice(0, 50) || "no throw");
    }

    // ---- the same statements in the configurations that matter (review F3) -----
    // Everything above runs at hyper 1 on the square 32^2 grid without the kick. The music
    // page is none of those: kick: true, hyper 2, the offline dissipation rule. So each of
    // the three ways the column is built -- the constructor, refreshDissipation, setDrag --
    // and setDrag's rollback is re-asserted, at EVERY mode, on:
    //   * the music configuration itself (kick + drag, hyper 2);
    //   * hyper 4 (the 2D app's default) and hyper 3;
    //   * a rectangular grid in a rectangular box, with the kick.
    // The expectation is never the code's own: the constructor's is makeGrid's column for
    // the SAME parameters minus mu/2, the two rebuilds' is refreshDissipation's long-standing
    // formula (fl32 ksq from gridA) transcribed here, minus mu/2.
    const CFGS = [
      ["the music configuration (kick: true, hyper 2, mu 1)", musicP(32, 32), 1],
      ["hyper 4 (the 2D app's default), mu 0.75", Object.assign({}, SQ, { diss: 1e-9, hyper: 4 }), 0.75],
      ["hyper 3, mu 0.25", Object.assign({}, SQ, { diss: 1e-6, hyper: 3 }), 0.25],
      ["a 32x16 grid in a 2pi x pi box, with the kick, hyper 2, mu 0.5",
       Object.assign({}, RECT, { diss: 2e-4, hyper: 2, kick: true }), 0.5]];
    for (const [tag, P0, mu] of CFGS) {
      const P = Object.assign({}, P0, { mu: 0 }), Pm = Object.assign({}, P0, { mu: mu });
      const g = gridOf(env, P, !!P.kick), n = g.nm;
      const mk = Float32Array.from(env.run("function(P){ return Array.from(makeGrid(P).gridB); }", P));
      const rebuilt = (diss, hyper, m) => {
        const c = Float32Array.from(mk);
        for (let i = 0; i < n; i++) c[4 * i + 1] = -diss * Math.pow(g.gridA[4 * i + 2], hyper);
        if (m) for (let i = 0; i < n; i++) c[4 * i + 1] = c[4 * i + 1] - 0.5 * m;
        return c;
      };
      const ctorWant = Float32Array.from(mk);
      let visc = 0;
      for (let i = 0; i < n; i++) { ctorWant[4 * i + 1] = mk[4 * i + 1] - 0.5 * mu; visc = Math.min(visc, mk[4 * i + 1]); }
      const c1 = build(Pm);
      ok(tag + ": the CONSTRUCTOR's column is makeGrid's minus mu/2 at all " + n + " modes, and is uploaded",
         sameBits(c1.gridB, ctorWant) === -1 && sameBits(c1.uploaded, ctorWant) === -1 && c1.mu === mu && visc < -1e-3,
         "first differing word " + sameBits(c1.gridB, ctorWant) + "; most negative viscous lin " + e2(visc));
      const c2 = build(Pm, "s.refreshDissipation(" + (2 * P.diss) + ", " + P.hyper + ");");
      ok("  ... refreshDissipation (2 x diss, same hyper) rebuilds it WITH the drag",
         sameBits(c2.gridB, rebuilt(2 * P.diss, P.hyper, mu)) === -1 && sameBits(c2.uploaded, c2.gridB) === -1,
         "first differing word " + sameBits(c2.gridB, rebuilt(2 * P.diss, P.hyper, mu)));
      const c3 = build(P, "s.setDrag(" + mu + ");");
      ok("  ... setDrag(" + mu + ") from mu = 0 rebuilds at THIS solver's diss and hyper = " + P.hyper,
         c3.mu === mu && sameBits(c3.gridB, rebuilt(P.diss, P.hyper, mu)) === -1 && sameBits(c3.uploaded, c3.gridB) === -1,
         "first differing word " + sameBits(c3.gridB, rebuilt(P.diss, P.hyper, mu)));
      const c4 = build(Pm, "s.setDrag(0);");
      ok("  ... setDrag(0) takes it off: refreshDissipation's mu = 0 column",
         c4.mu === 0 && sameBits(c4.gridB, rebuilt(P.diss, P.hyper, 0)) === -1 && sameBits(c4.uploaded, c4.gridB) === -1);
      const c5 = build(Pm, "try { s.setDrag(-1); } catch (e) { s.__err = String(e.message); }");
      ok("  ... and a refused setDrag(-1) rolls back to mu = " + mu + " at this diss and hyper (column and upload)",
         c5.mu === mu && /mu must be/.test(env.run("function(){ return __ds.__err || ''; }")) &&
         sameBits(c5.gridB, rebuilt(P.diss, P.hyper, mu)) === -1 && sameBits(c5.uploaded, c5.gridB) === -1,
         "first differing word " + sameBits(c5.gridB, rebuilt(P.diss, P.hyper, mu)));
      // ... and the decay that column produces, mode by mode, through the emitted stage kernel
      const f0c = walk(4 * n, 4242), dtc = 0.125, nsc = 3, Tc = dtc * nsc;
      const fc = runSteps(M, g, g.S.stage, c1.gridB, f0c, dtc, nsc);
      // (relative where the factor is representable; past 1e-20 -- the hyperviscous tail at
      // hyper 2 with the offline rule reaches exp(-360) -- fp32 has underflowed and the
      // statement is that the mode is gone)
      let miss = 0, gone = 0, goneBad = 0, cmpN = 0;
      for (let i = 0; i < n; i++) {
        const want = Math.exp((mk[4 * i + 1] - 0.5 * mu) * Tc);
        for (const w of [2 * i, 2 * i + 1, 2 * (n + i), 2 * (n + i) + 1]) {
          if (want < 1e-20) { gone++; if (Math.abs(fc[w] / f0c[w]) > 1e-19) goneBad++; }
          else { cmpN++; miss = Math.max(miss, Math.abs(fc[w] / f0c[w] / want - 1)); }
        }
      }
      ok("  ... and 3 LSRK33 steps of `stage` on it damp every mode by exp((lin - mu/2) T), phi and psi alike",
         miss < 5e-6 && goneBad === 0 && cmpN > n, "worst relative miss " + e2(miss) + " over " + cmpN
         + " words; " + gone + " words past fp32 underflow, " + goneBad + " of them not gone");
    }

    // ---- the decay, EXECUTED through the production stage kernel --------------
    // diss = 0, so the drag is the only thing in the column: with a zero RHS every mode's
    // amplitude must fall by exp(-mu T/2) and the energy by exp(-mu T).
    const ZP = Object.assign({}, SQ, { diss: 0, hyper: 1 });
    const f0 = walk(4 * nm, 777), dt = 0.125, NS = 5, T = dt * NS;
    const gz = build(Object.assign({}, ZP, { mu: MU })).gridB;
    const fT = runSteps(M, G, G.S.stage, gz, f0, dt, NS);
    const eRatio = energyOf(G, fT) / energyOf(G, f0), eWant = Math.exp(-MU * T);
    ok("5 LSRK33 steps of the emitted `stage` kernel on the dragged column: E(T)/E(0) = exp(-mu T)",
       Math.abs(eRatio / eWant - 1) < 2e-6, "got " + eRatio.toFixed(9) + ", exp(-" + MU + "*" + T + ") = "
       + eWant.toFixed(9) + ", rel " + e2(eRatio / eWant - 1));
    let amp = 0;
    const aWant = Math.exp(-0.5 * MU * T);
    for (let i = 0; i < 4 * nm; i++) amp = Math.max(amp, Math.abs(fT[i] / f0[i] / aWant - 1));
    ok("  ... UNIFORMLY: every word of both halves (phi and psi, every k) fell by exp(-mu T/2)",
       amp < 2e-6, "worst relative miss " + e2(amp) + " over " + (4 * nm) + " words");
    // ... and with mu = 0 the same five steps are the identity, bit for bit
    const gz0 = build(ZP).gridB;
    ok("  ... and at mu = 0, diss = 0 the same five steps leave the fields bit for bit",
       sameBits(runSteps(M, G, G.S.stage, gz0, f0, dt, NS), f0) === -1);
    // ... and the drag ADDS to the hyperviscosity, mode by mode, rather than replacing it
    const gd = build(Object.assign({}, DP, { mu: MU })).gridB;
    const fD = runSteps(M, G, G.S.stage, gd, f0, dt, NS);
    let comb = 0, visc = 0;
    for (let m = 0; m < nm; m++) {
      const want = Math.exp((-DP.diss * Math.pow(G.gridA[4 * m + 2], DP.hyper) - 0.5 * MU) * T);
      visc = Math.max(visc, DP.diss * Math.pow(G.gridA[4 * m + 2], DP.hyper) * T);
      for (const i of [2 * m, 2 * (nm + m)]) comb = Math.max(comb, Math.abs(fD[i] / f0[i] / want - 1));
    }
    ok("  ... and with diss != 0 each mode falls by exp(-(diss k^2h + mu/2) T): the two ADD",
       comb < 5e-6 && visc > 0.1, "worst relative miss " + e2(comb) + "; largest viscous exponent " + e2(visc));
    env.run("function(){ if (globalThis.__ds) __ds.destroy(); }");
  } finally { q.writeBuffer = orig; }
}

// ===========================================================================
// 7. MUSIC_EFFECTS_PLAN: the sheet kick, and the mean field in the stage kernel
// ===========================================================================
const PACKS = `function(P, arr, list){
  const r = Solver.prototype._packKicks.call({ p: P, _sheets: arr }, list, true);
  return JSON.stringify(r);
}`;
function packSheets(env, P, G, list) {
  const a = new Float32Array(2 * G.kmax * 8).fill(SENT_PACK);
  const r = JSON.parse(env.run(PACKS, P, a, list));
  return { a: a, used: r.used, dropped: r.dropped };
}
const runSheet = (M, G, packed) => runKick(M, Object.assign({}, G, { src: G.S.sheetKick }), packed);
// the b0 stage kernel, executed: its own emission (already `(64, 1, 1)`), NM invocations
function runStepsB(M, G, src, gridB, fields, rhs, dt, nsteps, b0) {
  const nm = G.nm, f = Float32Array.from(fields), delta = new Float32Array(4 * nm);
  const sc = new Float32Array(12); sc[0] = dt;
  const ex = new M.WgslExec(new M.WgslParser().parse(src));
  for (let n = 0; n < nsteps; n++) {
    delta.fill(0);
    for (let s = 0; s < 3; s++)
      ex.dispatchWorkgroups("main", [Math.ceil(nm / 64), 1, 1], { 0: { 0: f, 1: delta, 2: rhs, 3: gridB, 4: sc,
        5: Float32Array.of(G.lsrk[0][s], G.lsrk[1][s], G.lsrk[2][s], 0), 6: G.gridA, 7: Float32Array.of(b0[0], b0[1], 0, 0) } });
  }
  return f;
}
// the same recurrence in fp64, written out as the 2x2 matrix it is (per mode)
function refStepsB(G, gridB, fields, rhs, dt, nsteps, b0) {
  const nm = G.nm, f = Float64Array.from(fields), d = new Float64Array(4 * nm);
  for (let n = 0; n < nsteps; n++) {
    d.fill(0);
    for (let s = 0; s < 3; s++) for (let m = 0; m < nm; m++) {
      const al = G.lsrk[0][s], be = G.lsrk[1][s], ga = G.lsrk[2][s];
      const E = Math.exp(gridB[4 * m + 1] * dt * ga), th = (G.gridA[4 * m] * b0[0] + G.gridA[4 * m + 1] * b0[1]) * dt * ga;
      const c = Math.cos(th), sn = Math.sin(th), i0 = 2 * m, i1 = 2 * (nm + m);
      // M = E [[c, i sn], [i sn, c]] applied to a pair of complex numbers (a0, a1)
      const app = (a0r, a0i, a1r, a1i) => [E * (c * a0r - sn * a1i), E * (c * a0i + sn * a1r),
                                           E * (c * a1r - sn * a0i), E * (c * a1i + sn * a0r)];
      const dn = app(al * d[i0] + dt * rhs[i0], al * d[i0 + 1] + dt * rhs[i0 + 1],
                     al * d[i1] + dt * rhs[i1], al * d[i1 + 1] + dt * rhs[i1 + 1]);
      const fn = app(f[i0], f[i0 + 1], f[i1], f[i1 + 1]);
      d[i0] = dn[0]; d[i0 + 1] = dn[1]; d[i1] = dn[2]; d[i1 + 1] = dn[3];
      f[i0] = fn[0] + be * dn[0]; f[i0 + 1] = fn[1] + be * dn[1];
      f[i1] = fn[2] + be * dn[2]; f[i1 + 1] = fn[3] + be * dn[3];
    }
  }
  return f;
}
function legEffects(state) {
  const M = state.M, env = state.env, G = state.G, nm = G.nm;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  // ---- (a) the sheet kick --------------------------------------------------
  ok("buildShaders emits `sheetKick` under the same flag, and it parses",
     typeof G.S.sheetKick === "string" && G.S.sheetKick !== G.src && !!new M.WgslParser().parse(G.S.sheetKick));
  const sl = [{ x: 0.75, y: 1.5, sigma: 0.25, amp: 0.25, pol: 1, ratio: 4, theta: 0.6 },
              { x: 2.5, y: 0.25, sigma: 0.3125, amp: -0.5, pol: -1 }];
  const ps = packSheets(env, SQ, G, sl), a = ps.a, o1 = 8 * G.kmax;
  const w0 = 2 * Math.PI * (SQ.nx * SQ.ny) / (SQ.Lx * SQ.Ly), e = Math.sqrt(15);
  ok("_packKicks(list, true): 8 floats per slot -- (x0, y0, sigma, w * ratio), (ex, ey, 0, 0), e = sqrt(ratio^2 - 1) (cos, sin)",
     ps.used[0] === 1 && ps.used[1] === 1 && a[0] === 0.75 && a[1] === 1.5 && a[2] === 0.25
     && a[3] === Math.fround(potentialPeak(0.25, 0.25) * 0.25 * 0.25 * w0 * 4)
     && a[4] === Math.fround(e * Math.cos(0.6)) && a[5] === Math.fround(e * Math.sin(0.6)) && a[6] === 0 && a[7] === 0,
     "w " + a[3] + ", e (" + a[4] + ", " + a[5] + ")");
  ok("  ... a blob with no ratio is a sheet of ratio 1: e = 0, the blob's own weight; z- at slot NKICK",
     a[o1] === 2.5 && a[o1 + 3] === Math.fround(potentialPeak(-0.5, 0.3125) * 0.3125 * 0.3125 * w0) && a[o1 + 4] === 0 && a[o1 + 5] === 0
     && a[8] === 0 && a[11] === 0, "slot 1 of z+ zeroed: " + (a[8] === 0));
  let thrown = 0;
  for (const bad of [{ ratio: 0.5 }, { ratio: NaN }, { ratio: Infinity }, { theta: NaN }, { ratio: -2 }]) {
    try { packSheets(env, SQ, G, [Object.assign({ x: 1, y: 1, sigma: 0.3, amp: 1, pol: 1, ratio: 2, theta: 0 }, bad)]); }
    catch (err) { if (/ratio >= 1/.test(err.message)) thrown++; }
  }
  ok("  ... ratio < 1, a non-finite ratio and a non-finite theta THROW", thrown === 5, thrown + " of 5");
  // ratio 1 through the sheet kernel IS the blob kernel's kick, bit for bit
  const bl = CASES[3].list;
  ok("a list with no ratio through `sheetKick` is `blobKick`'s kick, bit for bit",
     sameBits(runSheet(M, G, packSheets(env, SQ, G, bl).a), runKick(M, G, pack(env, SQ, G, bl).a)) === -1);
  // against the real-space DFT of the periodised anisotropic gaussian
  const sheets = [
    ["one z+ sheet, ratio 4 at 0.6 rad", [sl[0]]],
    ["a z- sheet along y (theta = pi/2), ratio 3", [{ x: 3.1, y: 0.4, sigma: 0.3125, amp: 0.5, pol: -1, ratio: 3, theta: Math.PI / 2 }]],
    ["sheets and a blob, both channels", [sl[0], sl[1], { x: 5.2, y: 4.1, sigma: 0.25, amp: -0.75, pol: -1, ratio: 2.5, theta: 2.2 },
                                           { x: 1.1, y: 5.5, sigma: 0.375, amp: 0.5, pol: 1, ratio: 2, theta: 1.4 }]]
  ];
  for (const [tag, list] of sheets) {
    const r = compare(runSheet(M, G, packSheets(env, SQ, G, list).a), refKick(SQ, G.nkx, G.nky, G.kmax, list), nm);
    // 2 * ABS_TOL: the DFT of a 32-point sample ALIASES the narrow direction's tail (at
    // sigma 0.25, exp(-sigma^2 k^2/2) is 3e-7 of the peak at the first alias, k = 22), which
    // the analytic kernel does not have. The offline fixture below is analytic too, and
    // there the plain tolerance holds.
    ok("sheetKick vs the real-space DFT: " + tag, r.abs < 2 * ABS_TOL && r.rel < REL_SUM && r.phi.peak > 0 && r.psi.peak > 0, r.note);
  }
  // theta and theta + pi are the same sheet; theta + pi/2 is not
  const tw = l => runSheet(M, G, packSheets(env, SQ, G, l).a);
  const t0 = tw([sl[0]]), t1 = tw([Object.assign({}, sl[0], { theta: 0.6 + Math.PI })]), t2 = tw([Object.assign({}, sl[0], { theta: 0.6 + Math.PI / 2 })]);
  ok("  ... theta + pi is the same sheet, theta + pi/2 a different one",
     compare(t1, Float64Array.from(t0), nm).abs < ABS_TOL && compare(t2, Float64Array.from(t0), nm).abs > 0.1);
  const F = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "kick_offline.json"), "utf8"));
  ok("the fixture carries the offline sheet cases", Array.isArray(F.sheet_cases) && F.sheet_cases.length >= 3);
  for (const c of (F.sheet_cases || [])) {
    const r = compare(runSheet(M, G, packSheets(env, SQ, G, c.list).a), Float64Array.from(c.fields), nm);
    ok("offline make_kick (potential_k): " + c.tag, r.abs < ABS_TOL && r.rel < REL_SUM && r.phi.peak > 0 && r.psi.peak > 0, r.note);
  }
  // kick() picks the kernel by the list
  const names = () => env.gpu.dispatches.map(d => d.pipe.__name).join(",");
  env.run(`function(P){ globalThis.__es = new Solver(device, P); }`, Object.assign({}, W64, { b0: true }));
  env.gpuReset();
  env.run("function(l){ __es.kick(l); }", [sl[1]]);
  const nBlob = names();
  env.gpuReset();
  env.run("function(l){ __es.kick(l); }", sl);
  ok("kick(): a list with a ratio != 1 dispatches sheetKick (+ the energy refresh), one without dispatches blobKick",
     nBlob === "blobKick,energyPartial,energyFinal" && names() === "sheetKick,energyPartial,energyFinal", nBlob + " | " + names());
  const sb = env.run("function(){ const s = __es; return { n: s.buf.sheets.size, ok: s.bg.sheetKick.__buffers[3] === s.buf.sheets, st: s._cflStale, nk: s.nkicks }; }");
  ok("  ... through a table of 2 * KICK_MAX * 2 vec4, bound fourth; counted and marked like any kick",
     sb.n === 2 * G.kmax * 32 && sb.ok && sb.st === true && sb.nk === 2);

  // ---- (b) the mean field --------------------------------------------------
  const BS = env.run(`function(P){ const gr = makeGrid(P);
    const S = buildShaders(Object.assign({ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly, kick: true, b0: true }, gr));
    return S; }`, SQ);
  const moved = Object.keys(G.S).filter(n => G.S[n] !== BS[n]);
  ok("b0: true replaces `stage` and moves NO other kernel's text", moved.join(",") === "stage" && Object.keys(BS).length === G.names.length, moved.join(","));
  const ZP = Object.assign({}, SQ, { diss: 0, hyper: 1 });
  const gz = Float32Array.from(env.run("function(P){ return Array.from(makeGrid(P).gridB); }", ZP));
  const zero = new Float32Array(4 * nm), dt = 0.0625, NS = 4, T = dt * NS, B0 = [0.75, -0.5];
  // a state that is pure z- (psi = -phi), then pure z+: free streaming is an exact phase
  for (const [sign, name, way] of [[-1, "z-", "along +B0"], [1, "z+", "along -B0"]]) {
    const f0 = walk(4 * nm, 311);
    for (let i = 0; i < 2 * nm; i++) f0[2 * nm + i] = sign * f0[i];
    const fT = runStepsB(M, G, BS.stage, gz, f0, zero, dt, NS, B0);
    let worst = 0, wrong = 0;
    for (let m = 0; m < nm; m++) {
      // f(x - sign' B0 t): z- moves along +B0, i.e. picks up exp(-i k.B0 T); z+ the conjugate
      const th = sign * (G.gridA[4 * m] * B0[0] + G.gridA[4 * m + 1] * B0[1]) * T, c = Math.cos(th), sn = Math.sin(th);
      const wr = c * f0[2 * m] - sn * f0[2 * m + 1], wi = c * f0[2 * m + 1] + sn * f0[2 * m];
      const xr = c * f0[2 * m] + sn * f0[2 * m + 1], xi = c * f0[2 * m + 1] - sn * f0[2 * m];
      const a0 = Math.hypot(f0[2 * m], f0[2 * m + 1]);
      worst = Math.max(worst, Math.hypot(fT[2 * m] - wr, fT[2 * m + 1] - wi) / a0);
      wrong = Math.max(wrong, Math.hypot(fT[2 * m] - xr, fT[2 * m + 1] - xi) / a0);
    }
    ok("the b0 `stage` kernel, 4 steps at dt 1/16 with no RHS: a pure " + name + " state streams " + way + " EXACTLY (a phase per mode)",
       worst < 2e-5 && wrong > 1, "worst relative miss " + e2(worst) + "; the other direction " + e2(wrong));
    ok("  ... energy unchanged", Math.abs(energyOf(G, fT) / energyOf(G, f0) - 1) < 2e-6, e2(energyOf(G, fT) / energyOf(G, f0) - 1));
  }
  // a general state WITH a right-hand side and dissipation, against the fp64 recurrence
  const DPp = Object.assign({}, SQ, { diss: 2e-5, hyper: 2, mu: 0.75, kick: true });
  const gd = Float32Array.from(env.run("function(P){ const s = new Solver(device, P); const g = Array.from(s.g.gridB); s.destroy(); return g; }", DPp));
  const f0 = walk(4 * nm, 99), rhs = walk(4 * nm, 5);
  const fT = runStepsB(M, G, BS.stage, gd, f0, rhs, dt, NS, B0), fR = refStepsB(G, gd, f0, rhs, dt, NS, B0);
  let rel = 0, sc = 0;
  for (let i = 0; i < 4 * nm; i++) { sc = Math.max(sc, Math.abs(fR[i])); }
  for (let i = 0; i < 4 * nm; i++) rel = Math.max(rel, Math.abs(fT[i] - fR[i]) / sc);
  ok("  ... with a RHS, the drag and hyperviscosity: every word is the fp64 2x2 recurrence's", rel < 5e-6, "max |d| / max " + e2(rel));
  // B0 = 0: the values of the plain stage kernel
  const plain = runSteps(M, G, G.S.stage, gd, f0, dt, NS), viaB = runStepsB(M, G, BS.stage, gd, f0, zero, dt, NS, [0, 0]);
  let dz = 0;
  for (let i = 0; i < 4 * nm; i++) if (plain[i] !== viaB[i]) dz++;
  ok("  ... and at B0 = 0 it gives the plain `stage` kernel's values, word for word", dz === 0, dz + " words differ");
  // the solver side, on the stub device
  const st = JSON.parse(env.run(`function(){ const s = __es; return JSON.stringify({ code: s.pl.stage.__code, n: s.bg.stage[0].__buffers.length,
    last: s.bg.stage[0].__buffers[7] === s.buf.b0 && s.bg.stage[0].__buffers[6] === s.buf.gridA, size: s.buf.b0.size, nm: s.g.nm }); }`));
  ok("a solver built with b0: true compiles that kernel and binds [.., gridA, b0] after the six",
     st.code === env.run(`function(P){ const gr = makeGrid(P); return buildShaders(Object.assign({ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly, kick: true, b0: true, pm: 1 }, gr)).stage; }`, W64)
     && st.n === 8 && st.last && st.size === 16);
  env.gpuReset();
  env.run("function(){ __es.step(1); }");
  const sd = env.gpu.dispatches.filter(d => d.pipe.__name === "stage").map(d => d.d[0]);
  env.run(`function(P){ globalThis.__ps = new Solver(device, P); }`, W64);
  env.gpuReset();
  env.run("function(){ __ps.step(1); }");
  const sd0 = env.gpu.dispatches.filter(d => d.pipe.__name === "stage").map(d => d.d[0]);
  ok("  ... its three stages dispatch over ceil(NM / 64); a solver without b0 over ceil(2 NM / 64), as ever",
     sd.join() === [0, 0, 0].map(() => Math.ceil(st.nm / 64)).join() && sd0.join() === [0, 0, 0].map(() => Math.ceil(2 * st.nm / 64)).join(),
     sd.join() + " | " + sd0.join());
  const dev = env.run("function(){ return device; }"), q = dev.queue, orig = q.writeBuffer, wb = [];
  q.writeBuffer = function (b, off, data) { wb.push({ b: b, data: Array.from(data) }); return orig.call(q, b, off, data); };
  try {
    env.run("function(){ __es.setMeanField(0.5, -0.25); __es.setMeanField(0.5, -0.25); }");
    const b0buf = env.run("function(){ return __es.buf.b0; }");
    ok("setMeanField(0.5, -0.25) uploads (0.5, -0.25, 0, 0) to the uniform, once (the same value again writes nothing)",
       wb.length === 1 && wb[0].b === b0buf && wb[0].data.join() === "0.5,-0.25,0,0", wb.length + " writes");
    let refused = 0;
    for (const args of ["NaN, 0", "0, Infinity", "'1', 0", ""]) {
      try { env.run("function(){ __es.setMeanField(" + args + "); }"); } catch (err) { if (/two finite numbers/.test(err.message)) refused++; }
    }
    ok("  ... a non-finite or non-numeric component THROWS and writes nothing", refused === 4 && wb.length === 1, refused + " of 4");
    let noB0 = "", noPm = "";
    try { env.run("function(){ __ps.setMeanField(1, 0); }"); } catch (err) { noB0 = err.message; }
    try { env.run("function(P){ new Solver(device, P); }", Object.assign({}, W64, { mu: 0, b0: true, pm: 2 })); } catch (err) { noPm = err.message; }
    ok("  ... a solver built without b0 has no uniform and setMeanField throws; b0 with Pm != 1 is refused",
       /without `b0: true`/.test(noB0) && /needs Pm = 1/.test(noPm) && env.run("function(){ return __ps.buf.b0 === undefined && __ps.pl.stage.__code !== __es.pl.stage.__code; }"),
       noB0 + " | " + noPm);
  } finally { q.writeBuffer = orig; }
  env.run("function(){ __es.destroy(); __ps.destroy(); }");
}

// ---------------------------------------------------------------------------
const LEGS = [
  ["1. the emission: gated on `kick`, parsed, sized, covering every retained mode", legEmission],
  ["2. _packKicks: the half, the weight, the cap, the refusals, the zero-fill", legPack],
  ["3. the cross-check: production k space vs a real-space DFT and vs the offline make_kick", legCross],
  ["4. k = 0, the dealias cliff, additivity, the Elsasser map, reality", legExact],
  ["5. kick(): the dispatch, the upload, the refusals (stub device)", legWiring],
  ["6. the drag: the linear column, the refusals, and the decay through `stage`", legDrag],
  ["7. effects: the sheet kick, and the mean field in the stage kernel's exact linear step", legEffects]
];
(async () => {
  const state = {};
  state.M = await wgslMod();
  for (const [title, fn] of LEGS) { console.log(title); await fn(state); }
  console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
  process.exit(bad ? 1 : 0);
})();
