"use strict";
// ===========================================================================
// traces.js -- the music page's FADING TRACES (plans-webgpu/MUSIC_STAGE_PLAN.md, S3):
// the shell spectrum and a line cut drawn over the picture as lines that fade with age,
// in the display's own colormap. A 2D canvas laid over a host element; never part of the
// solver, and not in a recording or a saved PNG (those are the WebGPU canvas alone).
//
// Live with the stage (the page's layout unless ?plain) and, on the card layout, with
// ?traces. Otherwise this file defines names and
// does nothing: groups() is empty, attach() returns null, frame() returns at once.
//
// What is drawn (2026-11-23, after Alfred saw the first version in Chrome):
//   * the SPECTRUM, total energy only, in a band across the full width along the BOTTOM;
//   * the CUT, in a band across the full width along the TOP: the scalar the picture is
//     showing, along x on the picture's middle row, so that the line sits over the image
//     it was cut from -- same x, and a vertical range that IS the picture's colour scale;
//   * one line per sample, never dashed; small tick labels and no axis, frame or grid.
// And after the second look: the cut fades four times faster than the spectrum (twenty
// overlapping lines of different shapes is clutter), the spectrum spans four decades, and
// while the tick labels are shown the lines keep clear of them (tracesPlot).
// And the SIDE PANELS (an option, off by default): where the page says the picture leaves
// two free strips (musicTraces.picture, the stage's black sides), four spectra -- E_u, E_b,
// E+, E- -- in one and, in the other, those four energies against time or four signed cuts
// (u_x, b_x, z+_x, z-_x along x = Lx/2). They take the bottom spectrum's place.
//
// Two halves:
//   * PURE functions (traces*, no DOM, no GPU, nothing from the other scripts): the
//     history and its expiry, the fade weight, the log-log mapping, the guide line, the
//     first complete bin, the colour of a line of a given age, the canvas sizing, the
//     placement, the cut's row extraction plan and its range rule, the tick labels.
//     devtools/checkmusic.js evaluates this file in a bare context for them.
//   * `musicTraces`, the DOM side: one <canvas class="traces"> filling a host element,
//     the control group `grpTraces`, and one call per frame.
//
// Ages are wall-clock milliseconds (performance.now()) and are always an ARGUMENT here,
// never read inside, so the fade is testable without a clock.
//
// Readbacks. Nothing here awaits inside the frame: a readback is started, the frame goes
// on, and the result is pushed when it arrives (stamped with the time it was asked for).
//   * The spectrum is taken from the spectrum chart card's own readback when that card is
//     on the page (common.js keeps the last one in `autoDissCache`; it is refreshed about
//     every 300 ms while the state moves) and read here -- at most every TRACES_EVERY-th
//     frame, one in flight at a time -- only when no such card exists.
//   * The cut is one row of the display chain's `dispR`, the real-space buffer the
//     colorize pass has just coloured (so it carries the card's field, band and offset),
//     plus the 4 bytes of `maxVal`, the full scale that pass divided by. Real-space
//     buffers are row-major ix * ny + iy, so a line along x is strided: nx 4-byte
//     copyBufferToBuffer calls and one more for the scale, into ONE (nx + 1)-float staging
//     buffer, in one encoder, then one mapAsync. No kernel, no dispatch. Same schedule,
//     and only while the cut is switched on.
// ===========================================================================

const TRACES_EVERY = 6;          // frames between pushes (about 10 a second at 60 Hz)
const TRACES_TAU = 2;            // s: the fade time's default (the slider's)
const TRACES_FLOOR = 0.02;       // a line fainter than this is dropped
const TRACES_MAX = 256;          // history cap, whatever the fade time
const TRACES_SPEC_FRAC = 0.2;    // the spectrum band's height, as a fraction of the host's short side
const TRACES_CUT_FRAC = 0.2;     // the cut band's
const TRACES_DECADES = 4;        // the spectrum's vertical span (a tail below it is clipped at the band's edge)
const TRACES_TOP = 0.3;          // ... whose top is this multiple of amp^2 (see tracesSpecRange)
const TRACES_CUT_TAU_FRAC = 0.25;  // the cut fades this much FASTER: its time constant, as a fraction of the slider's
const TRACES_SLOPE = -5 / 3;     // the guide line

// on with the stage (the page's layout unless `?plain`), and on the card layout with `?traces`
function tracesFlag(search) { const q = String(search || ""); return !/[?&]plain(?![\w-])/.test(q) || /[?&]traces(?![\w-])/.test(q); }
const TRACES_ON = typeof location !== "undefined" && tracesFlag(location.search);

// ---------------------------------------------------------------------------
// pure
// ---------------------------------------------------------------------------
// the opacity of a line `ageS` seconds old: exp(-age / tau); a line from the future (a
// clock that stepped back) is simply new
function tracesWeight(ageS, tau) {
  if (!(ageS > 0)) return 1;
  return Math.exp(-ageS / Math.max(tau, 1e-6));
}
// is this frame (counted from 1) one that takes a sample?
function tracesDue(frame, every) { return frame > 0 && frame % every === 0; }
// append an entry (anything carrying `t`, ms) and keep at most `max`, dropping the OLDEST
function tracesPush(hist, entry, max) {
  hist.push(entry);
  const m = max === undefined ? TRACES_MAX : max;
  if (hist.length > m) hist.splice(0, hist.length - m);
  return hist;
}
// drop every entry whose weight at `nowMs` is below `floor` (default 2%). Entries are in
// arrival order but are NOT assumed sorted by t. Returns how many were dropped.
function tracesExpire(hist, nowMs, tau, floor) {
  const f = floor === undefined ? TRACES_FLOOR : floor;
  let w = 0;
  for (let i = 0; i < hist.length; i++)
    if (tracesWeight((nowMs - hist[i].t) / 1000, tau) >= f) hist[w++] = hist[i];
  const n = hist.length - w;
  hist.length = w;
  return n;
}
// The bins worth drawing. Bin b holds the modes with round(|k| / kunit) = b, where
// kunit = 2pi / max(Lx, Ly) is the LONGER side's fundamental (solver2d.js makeGrid). In a
// box that is not square the shorter side's fundamental is kunit * Lmax / Lmin, so the
// bins below round(Lmax / Lmin) hold modes along the long direction only: `first` is the
// first bin whose shell reaches both axes. `last` is the top bin common.js's nbins() keeps
// (the smaller of the two axis dealias cutoffs, in kunit), by the same formula.
function tracesBinRange(nx, ny, Lx, Ly) {
  const a = Lx || 1, b = Ly || 1, Lmax = Math.max(a, b), Lmin = Math.min(a, b);
  const nb = Math.floor(Math.min((nx / 3) * (Lmax / a), (ny / 3) * (Lmax / b)));
  const first = Math.max(1, Math.floor(Lmax / Lmin + 0.5));
  return { first: first, last: Math.max(first, nb - 1), nb: nb, kunit: 2 * Math.PI / Lmax };
}
// The spectrum's axes. Horizontal: the drawn bins, in physical k. Vertical: FIXED against
// the amp slider -- top = TRACES_TOP * amp^2, TRACES_DECADES decades down -- and never
// against the data, so no line moves unless the flow does and a quiet passage shows as a
// spectrum that sinks (the same choice as the page's fixed colour scale).
function tracesSpecRange(amp, bins) {
  const a = Math.max(amp || 0, 1e-3), e1 = TRACES_TOP * a * a;
  return { k0: bins.first * bins.kunit, k1: Math.max(bins.last, bins.first + 1) * bins.kunit,
           e0: e1 * Math.pow(10, -TRACES_DECADES), e1: e1 };
}
// (k, E) -> canvas pixels in `box` {x, y, w, h} (y down), log-log over `r` {k0, k1, e0, e1}.
// Not clamped: the caller clips to the box.
function tracesLogMap(k, e, r, box) {
  const fx = Math.log(k / r.k0) / Math.log(r.k1 / r.k0);
  const fy = Math.log(e / r.e0) / Math.log(r.e1 / r.e0);
  return [box.x + box.w * fx, box.y + box.h * (1 - fy)];
}
// the guide: a straight line of log-log slope `slope` (default -5/3) across the middle of
// the plot, one decade under its top at its left end. Returns { k, e } pairs and pixels.
function tracesGuide(r, box, slope) {
  const p = slope === undefined ? TRACES_SLOPE : slope;
  const ka = r.k0 * Math.pow(r.k1 / r.k0, 0.2), kb = r.k0 * Math.pow(r.k1 / r.k0, 0.8);
  const ea = r.e1 / 10, eb = ea * Math.pow(kb / ka, p);
  const A = tracesLogMap(ka, ea, r, box), B = tracesLogMap(kb, eb, r, box);
  return { ka: ka, ea: ea, kb: kb, eb: eb, x0: A[0], y0: A[1], x1: B[0], y1: B[1] };
}
// two or three wavenumbers to label, from 1-2-5 per decade inside [k0, k1]: the first,
// the last, and the middle one when there are more than two
function tracesKLabels(k0, k1) {
  const c = [];
  for (let d = Math.floor(Math.log10(k0)) - 1; d <= Math.ceil(Math.log10(k1)); d++)
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, d);
      if (v >= k0 * (1 - 1e-9) && v <= k1 * (1 + 1e-9)) c.push(v);
    }
  if (c.length <= 3) return c;
  return [c[0], c[c.length >> 1], c[c.length - 1]];
}
// the decades of E to label: the whole powers of ten STRICTLY inside (e0, e1), thinned to
// every 2nd, 3rd, ... (counted down from the top one) until at most `maxN` are left.
// Returns exponents, largest first.
function tracesDecades(e0, e1, maxN) {
  const hi = Math.ceil(Math.log10(e1) - 1e-9) - 1, lo = Math.floor(Math.log10(e0) + 1e-9) + 1;
  const m = maxN === undefined ? 3 : maxN;
  let step = 1;
  while (Math.floor((hi - lo) / step) + 1 > m) step++;
  const out = [];
  for (let d = hi; d >= lo; d -= step) out.push(d);
  return out;
}
// "10" with its exponent in superscript digits: 10⁻³
function tracesPow10(d) {
  const S = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return "10" + String(d).split("").map(ch => S[ch]).join("");
}
// a length as text: in units of pi where that is a whole or a half number of them (0, pi/2,
// pi, 3pi/2, 2pi, ...), a plain number to three figures otherwise
function tracesLenText(x) {
  if (x === 0) return "0";
  const h = 2 * x / Math.PI, n = Math.round(h);
  if (Math.abs(h - n) > 1e-6 * Math.max(1, Math.abs(h))) return String(parseFloat(x.toPrecision(3)));
  if (n % 2 === 0) return (n === 2 ? "" : String(n / 2)) + "π";
  return (n === 1 ? "" : String(n)) + "π/2";
}
// the cut's x labels: the left edge, the middle and the right edge of the box
function tracesXLabels(Lx) {
  return [{ f: 0, t: "0" }, { f: 0.5, t: tracesLenText(Lx / 2) }, { f: 1, t: tracesLenText(Lx) }];
}
// which end of a colormap is the bright one: 1 unless t = 0 is lighter (rgb in [0, 1])
function tracesBrightEnd(rgb0, rgb1) {
  const L = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  return L(rgb0) > L(rgb1) ? 0 : 1;
}
// where in the colormap a line of this age takes its colour: the bright end when new,
// sliding to the middle (0.5) with the same exponential as its opacity
function tracesColourT(ageS, tau, bright) {
  const b = bright === 0 ? 0 : 1;
  return 0.5 + (b - 0.5) * tracesWeight(ageS, tau);
}
function tracesCss(rgb, alpha) {
  const c = v => Math.round(255 * Math.max(0, Math.min(1, v)));
  return "rgba(" + c(rgb[0]) + "," + c(rgb[1]) + "," + c(rgb[2]) + "," + Math.max(0, Math.min(1, alpha)).toFixed(3) + ")";
}
// the canvas's backing size: the host's client size times devicePixelRatio, at least 1
function tracesFit(cw, ch, dpr) {
  const d = dpr > 0 ? dpr : 1;
  return { w: Math.max(1, Math.round(cw * d)), h: Math.max(1, Math.round(ch * d)) };
}
// placement, in backing pixels. Both are bands across the FULL width, each a fifth of the
// host's short side high: the spectrum along the bottom edge, the cut along the top. The
// cut's box is exactly the host's width, edge to edge, because its x IS the picture's x.
function tracesLayout(W, H) {
  const s = Math.min(W, H), hs = s * TRACES_SPEC_FRAC, hc = s * TRACES_CUT_FRAC;
  return { spec: { x: 0, y: H - hs, w: W, h: hs }, cut: { x: 0, y: 0, w: W, h: hc } };
}
// Where each band actually plots, given the bands of tracesLayout. With the tick labels
// OFF both run edge to edge. With them ON a column `col` wide on the left is kept for the
// y labels and the spectrum also gives up a row `row` high along its bottom for the k
// labels, so no line is ever drawn under a label:
//   spec     the spectrum's plot box: its log-k axis is SQUEEZED into what is left;
//   cut      the cut's MAPPING box, which never changes -- sample ix stays at
//            (ix + 0.5) / nx of the host's width, over its texel of the picture;
//   cutClip  what of the cut is drawn: the line is simply not drawn inside the column.
function tracesPlot(L, ticks, col, row) {
  if (!ticks) return { spec: L.spec, cut: L.cut, cutClip: L.cut };
  const c = Math.max(0, Math.min(col, L.spec.w - 1)), r = Math.max(0, Math.min(row, L.spec.h - 1));
  return { spec: { x: L.spec.x + c, y: L.spec.y, w: L.spec.w - c, h: L.spec.h - r },
           cut: L.cut,
           cutClip: { x: L.cut.x + c, y: L.cut.y, w: L.cut.w - c, h: L.cut.h } };
}
// the label column's width: the widest label in it, with `pad` either side
function tracesLabelCol(widths, pad) {
  let w = 0;
  for (const v of widths) if (v > w) w = v;
  return w > 0 ? w + 2 * pad : 0;
}
// the total energy's points off a spectrum readback [E_u | E_b | H_c] (3 * nb floats):
// E_u + E_b over bins first..last, at k = bin * kunit. A bin that is not positive and
// finite BREAKS the line there (a log axis has no place for it). Returns runs of
// [k, v, k, v, ...].
function tracesSpecRuns(perp, nb, bins) {
  const runs = [];
  let cur = null;
  for (let b = bins.first; b <= Math.min(bins.last, nb - 1); b++) {
    const v = perp[b] + perp[nb + b];
    if (v > 0 && isFinite(v)) { if (!cur) runs.push(cur = []); cur.push(b * bins.kunit, v); }
    else cur = null;
  }
  return runs;
}
// ---- the side panels ----------------------------------------------------------
// One of the four energies off a spectrum readback's bin b, or off (E_u, E_b, H_c) scalars:
// 0 kinetic E_u, 1 magnetic E_b, 2 E+ = E_u + E_b + H_c, 3 E- = E_u + E_b - H_c (common.js
// SPEC_SETS' convention).
// the energies in time: one point at most every TRACES_ENER_MS, and a history long enough
// for the widest panel (4 x the fade slider's 6 s = 24 s, 720 points)
const TRACES_ENER_MS = 33, TRACES_ENER_MAX = 768;
function tracesEnerDue(lastMs, nowMs) { return !(lastMs > 0) || nowMs - lastMs >= TRACES_ENER_MS - 1; }
const TRACES_LANES = ["E_u", "E_b", "E+", "E\u2212"];
function tracesLane(which, eu, eb, hc) {
  return which === 0 ? eu : which === 1 ? eb : which === 2 ? eu + eb + hc : eu + eb - hc;
}
// as tracesSpecRuns, for one lane
function tracesLaneRuns(perp, nb, bins, which) {
  const runs = [];
  let cur = null;
  for (let b = bins.first; b <= Math.min(bins.last, nb - 1); b++) {
    const v = tracesLane(which, perp[b], perp[nb + b], perp[2 * nb + b]);
    if (v > 0 && isFinite(v)) { if (!cur) runs.push(cur = []); cur.push(b * bins.kunit, v); }
    else cur = null;
  }
  return runs;
}
// the four SIGNED fields along a cut line, off its [u_x | u_y | b_x | b_y] stack (n each):
// u_x, b_x, z+_x = u_x + b_x, z-_x = u_x - b_x, in TRACES_LANES' order. The line runs along
// y, so x is the TRANSVERSE component -- the one a divergence-free field varies in freely
// along it (the y components are tied to the x-derivatives the cut does not see).
function tracesLineFields(a, n) {
  const out = [0, 1, 2, 3].map(() => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    const ux = a[i], bx = a[2 * n + i];
    out[0][i] = ux; out[1][i] = bx; out[2][i] = ux + bx; out[3][i] = ux - bx;
  }
  return out;
}
const TRACES_CUT_NAMES = ["u_x", "b_x", "z+_x", "z\u2212_x"];
// Where the eight panels go: the two strips of a W x H host that a picture at (x, y), size
// w x h, leaves free -- left and right of it (each cut into four rows), or, where the
// picture spans the width, above and below (four columns each). `a` is the first strip's
// four boxes (the spectra), `b` the second's (the energies in time), each box inset by
// `gap`. null where neither pair of strips is at least `min` across: no room, no panels.
const TRACES_SIDE_MIN = 90;
// what a strip of side panels can show (the two selects' options), and a wanted kind made safe
const TRACES_SIDE_KINDS = [["spec", "spectra"], ["cuts", "line cuts"], ["time", "energies in time"]];
const TRACES_SIDE_TIP = "what this strip shows (first: left, or top on a tall window). spectra: kinetic, "
  + "magnetic, E+ and E-, each with a k^(-5/3) guide. line cuts: the x components of u, b, z+ and z- along "
  + "the vertical line through the middle of the box (x = Lx/2), against y. energies in time: those four "
  + "energies' histories.";
function tracesSideKind(v, dflt) { return v === "spec" || v === "cuts" || v === "time" ? v : dflt; }      // CSS px: the least width (or height) of a strip that takes panels
function tracesSidePanels(W, H, x, y, w, h, min, gap) {
  const g = gap || 0, cut = (X, Y, Wd, Hd, rows) => {
    const out = [];
    for (let i = 0; i < 4; i++) {
      const bx = rows ? X : X + Wd * i / 4, by = rows ? Y + Hd * i / 4 : Y, bw = rows ? Wd : Wd / 4, bh = rows ? Hd / 4 : Hd;
      out.push({ x: bx + g, y: by + g, w: Math.max(1, bw - 2 * g), h: Math.max(1, bh - 2 * g) });
    }
    return out;
  };
  if (x >= min && W - (x + w) >= min) return { a: cut(0, 0, x, H, true), b: cut(x + w, 0, W - (x + w), H, true) };
  if (y >= min && H - (y + h) >= min) return { a: cut(0, 0, W, y, false), b: cut(0, y + h, W, H - (y + h), false) };
  return null;
}
// ---- the cut ----------------------------------------------------------------
// The picture's middle row. The display texture is written at (ix, iy) and blitted with
// ix running left to right and iy = 0 at the TOP (physics.js RENDER_WGSL samples at
// 1 - uv.y), so row ny/2 is the one half-way down the screen.
function tracesMidRow(ny) { return ny >> 1; }
// How one row iy of a row-major (ix * ny + iy) float buffer is gathered: `src[ix]` is the
// BYTE offset of sample ix in the source and 4 * ix its byte offset in the staging buffer;
// one more float (the full scale) goes at `scaleAt`. `size` is the staging buffer's.
function tracesRowCopies(nx, ny, iy) {
  const src = [];
  for (let ix = 0; ix < nx; ix++) src.push(4 * (ix * ny + iy));
  return { src: src, scaleAt: 4 * nx, size: 4 * (nx + 1) };
}
// The cut's vertical range IS the colour scale the picture was drawn with (physics.js
// dispX): `kind` "signed" (a signed scalar: -max..+max, zero mid-band), "mag" (a vector
// magnitude: 0..max, zero at the band's bottom), "sigma" (sigma_c / sigma_r: fixed -1..+1,
// whatever `max` says). `max` is the display chain's `maxVal` -- the fixed colour scale or
// the autoscale, whichever the picture used.
function tracesCutRange(kind, max) {
  if (kind === "sigma") return { lo: -1, hi: 1, signed: true };
  const m = max > 0 && isFinite(max) ? max : 1e-30;
  return kind === "mag" ? { lo: 0, hi: m, signed: false } : { lo: -m, hi: m, signed: true };
}
// sample ix of nx, value v -> pixels: x at the CENTRE of the picture's texel ix (so the
// line sits over the image), y up the box over [lo, hi]. Not clamped: the caller clips,
// as the colormap saturates.
function tracesCutMap(ix, nx, v, range, box) {
  return [box.x + box.w * (ix + 0.5) / nx, box.y + box.h * (1 - (v - range.lo) / (range.hi - range.lo))];
}
// the cut's scale labels: { f: fraction up the band, t: text }
function tracesCutLabels(range) {
  const n = v => String(parseFloat(v.toPrecision(2)));
  return range.signed ? [{ f: 1, t: "+" + n(range.hi) }, { f: 0.5, t: "0" }, { f: 0, t: "−" + n(range.hi) }]
                      : [{ f: 1, t: n(range.hi) }, { f: 0, t: "0" }];
}

// ---------------------------------------------------------------------------
// the DOM side
// ---------------------------------------------------------------------------
const musicTraces = {
  on: TRACES_ON,
  picture: null,                 // () -> { x, w, tile } in CSS px of the host, or null: the cut's x extent
  host: null, canvas: null, ctx: null, ro: null,
  frames: 0,                    // frames seen since attach
  spec: [], cut: [],            // the two histories: { t, perp, nb, bins } / { t, row, max, kind, mode }
  reads: { spec: 0, cut: 0 },   // readbacks STARTED here (the card's are not counted)
  busy: { spec: false, cut: false }, mark: { spec: null, cut: null }, cacheAt: null,
  ener: [], enerAt: 0,          // the side panels' energies in time: { t, v: [4] }
  rows4: [], busy4: false, mark4: null,   // ... and their four cut lines: { t, rows: [4] }
  stage: null,                  // the cut's staging buffer: { buf, size, device }
  ui: null,                     // the controls, looked up once at attach
  // overridable lookups (the page's, when they exist)
  card: () => (typeof cards !== "undefined" && cards.disp && cards.disp[0] ? cards.disp[0] : null),
  cmapOf() { const c = this.card(); return c ? c.cmap() : 0; },
  ampOf: () => (typeof musicUi === "function" ? musicUi().amp : 1),

  // the control group, for controlsBuild's `groups` list: empty without the flag
  groups() {
    if (!TRACES_ON) return [];
    return [{ id: "grpTraces", summary: "traces", rows: [
      [{ k: "cbl", id: "cbTrSpec", t: "spectrum", v: false,
         ti: "the total energy spectrum along the bottom of the picture, log-log, as lines that "
           + "fade with age; the faint straight line is k^(-5/3). The vertical range is fixed "
           + "against the amp slider." },
       { k: "cbl", id: "cbTrCut", t: "line cut", v: false,
         ti: "the field the picture shows, along the horizontal line through its middle, drawn "
           + "along the top: same x as the picture, and a vertical range that is its colour scale" },
       { k: "cbl", id: "cbTrTicks", t: "tick labels", v: true,
         ti: "small numbers on both traces: k and E on the spectrum, x and the scale on the cut" },
       { k: "lab", t: "fade (s)" },
       { k: "rng", id: "rTrTau", min: 0.5, max: 6, step: 0.5, v: TRACES_TAU,
         ti: "a spectrum line's opacity is exp(-age / this); it is dropped below 2%. The line "
           + "cut fades four times faster." },
       { k: "val", id: "vTrTau" }],
      [{ k: "cbl", id: "cbTrSides", t: "side panels", v: true,
         ti: "where the picture leaves black sides: four panels in each, of the kind chosen "
           + "for that side, fading as the other traces do. They take the place of the "
           + "spectrum along the bottom. Nothing is drawn while the sides repeat the picture "
           + "or are too narrow." },
       { k: "lab", t: "first", for: "selTrA" },
       { k: "sel", id: "selTrA", o: TRACES_SIDE_KINDS, v: "spec", ti: TRACES_SIDE_TIP },
       { k: "lab", t: "second", for: "selTrB" },
       { k: "sel", id: "selTrB", o: TRACES_SIDE_KINDS, v: "time", ti: TRACES_SIDE_TIP }]
    ] }];
  },

  // ONE <canvas class="traces"> filling `host`, on top of whatever is in it, taking no
  // pointer events. Returns the canvas (null without the flag or without a host).
  attach(host) {
    if (!TRACES_ON || !host) return null;
    this.detach();
    const cv = document.createElement("canvas");
    cv.className = "traces";
    const st = cv.style;
    st.position = "absolute"; st.left = "0"; st.top = "0"; st.width = "100%"; st.height = "100%";
    st.display = "block"; st.pointerEvents = "none";
    // an absolutely positioned child needs a positioned host (the card's wrap and #stage are)
    if (typeof getComputedStyle === "function") {
      try { if (getComputedStyle(host).position === "static") host.style.position = "relative"; } catch (e) { /* not an element */ }
    }
    host.appendChild(cv);
    this.host = host; this.canvas = cv; this.ctx = cv.getContext("2d");
    this.frames = 0; this.spec = []; this.cut = []; this.ener = []; this.rows4 = []; this.busy4 = false; this.mark4 = null; this.enerAt = 0;
    this.busy = { spec: false, cut: false }; this.mark = { spec: null, cut: null }; this.cacheAt = null;
    const g = id => (typeof document !== "undefined" && document.querySelector && document.querySelector("#" + id))
                    || (typeof el === "function" ? el(id) : null);
    this.ui = { spec: g("cbTrSpec"), cut: g("cbTrCut"), ticks: g("cbTrTicks"), tau: g("rTrTau"), vTau: g("vTrTau"), sides: g("cbTrSides"), sideA: g("selTrA"), sideB: g("selTrB") };
    if (this.ui.tau) this.ui.tau.oninput = () => this.label();
    this.label();
    this.fit();
    if (typeof ResizeObserver !== "undefined") { this.ro = new ResizeObserver(() => this.fit()); this.ro.observe(host); }
    return cv;
  },
  detach() {
    if (this.ro) { this.ro.disconnect(); this.ro = null; }
    if (this.canvas && this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
    this.host = this.canvas = this.ctx = null;
  },
  label() { if (this.ui && this.ui.vTau) this.ui.vTau.textContent = this.opts().tau.toFixed(1); },
  // what the controls say (their defaults where a control is missing)
  opts() {
    const u = this.ui || {}, t = u.tau ? parseFloat(u.tau.value) : NaN;
    return { spec: u.spec ? !!u.spec.checked : false, cut: u.cut ? !!u.cut.checked : false,
             sides: u.sides ? !!u.sides.checked : false,
             sideA: tracesSideKind(u.sideA && u.sideA.value, "spec"), sideB: tracesSideKind(u.sideB && u.sideB.value, "time"),
             ticks: u.ticks ? !!u.ticks.checked : true, tau: isFinite(t) && t > 0 ? t : TRACES_TAU };
  },
  dpr() { return (typeof window !== "undefined" && window.devicePixelRatio) || 1; },
  // (re)size the backing store from the host; true if it changed
  fit() {
    const h = this.host, cv = this.canvas;
    if (!h || !cv) return false;
    let cw = h.clientWidth, ch = h.clientHeight;
    if (!(cw > 0 && ch > 0) && h.getBoundingClientRect) { const r = h.getBoundingClientRect(); cw = r.width; ch = r.height; }
    const s = tracesFit(cw || 0, ch || 0, this.dpr());
    if (cv.width === s.w && cv.height === s.h) return false;
    cv.width = s.w; cv.height = s.h;
    return true;
  },

  // once per frame, after the frame's render: maybe start a sample, then redraw. `sv` is
  // the live solver, `nowMs` is performance.now(). Synchronous: nothing is awaited.
  frame(sv, nowMs) {
    if (!this.canvas) return;
    this.frames++;
    if (sv && tracesDue(this.frames, TRACES_EVERY)) this.sample(sv, nowMs);
    // the side panels' energies in time: the loop's own stats readback (no read of ours),
    // at most every TRACES_ENER_MS of a state that has moved -- 30 a second whatever the
    // display's rate, so a kick shows as a step and the cap holds the longest panel
    const oo = this.opts();
    if (!(oo.sides && (oo.sideA === "time" || oo.sideB === "time"))) this.ener.length = 0;
    else if (sv && tracesEnerDue(this.enerAt, nowMs) && typeof statsCache !== "undefined" && statsCache.s) {
      const s = statsCache.s, mark = typeof stateMark === "function" ? stateMark() : this.frames;
      if (mark !== this.mark.ener && isFinite(s[2]) && isFinite(s[3])) {
        this.mark.ener = mark; this.enerAt = nowMs;
        tracesPush(this.ener, { t: nowMs, v: [0, 1, 2, 3].map(i => tracesLane(i, s[2], s[3], isFinite(s[8]) ? s[8] : 0)) }, TRACES_ENER_MAX);
      }
    }
    this.draw(nowMs);
  },
  // start whatever samples are wanted and not already in flight. Only for a state that
  // has moved since the last one (a paused page reads nothing, and its lines fade out).
  sample(sv, nowMs) {
    const o = this.opts(), mark = typeof stateMark === "function" ? stateMark() : this.frames;
    const want = k => o.sides && (o.sideA === k || o.sideB === k);   // a strip of side panels shows kind k
    if (!o.spec && !want("spec")) { this.spec.length = 0; this.mark.spec = null; }
    else if (this.hasSpecCard()) {
      // the chart card's readback (common.js loop): take each new one once
      const c = autoDissCache;
      if (c.sv === sv && c.perp && c.at !== this.cacheAt) { this.cacheAt = c.at; this.pushSpec(c.perp, sv, nowMs); }
    } else if (!this.busy.spec && mark !== this.mark.spec) {
      this.busy.spec = true; this.mark.spec = mark; this.reads.spec++;
      sv.readSpectrum().then(sp => { if (sv === solver && this.canvas) this.pushSpec(sp.perp, sv, nowMs); },
                             () => { /* the solver was retired under the map */ })
        .catch(e => console.error(e)).then(() => { this.busy.spec = false; });
    }
    // the side panels' four cuts: u_x, b_x, z+_x, z-_x (signed) along the line x = Lx/2, off the
    // solver's own cut-line readback (4 * ny floats: u_x, u_y, b_x, b_y -- what rmhd2d's
    // cut chart reads), one small map
    if (!want("cuts")) { this.rows4.length = 0; this.mark4 = null; }
    else if (!this.busy4 && mark !== this.mark4 && sv.readCutLine) {
      this.busy4 = true; this.mark4 = mark; this.reads.line = (this.reads.line || 0) + 1;
      const ny = sv.p.ny;
      sv.readCutLine(0).then(a => {
        if (sv === solver && this.canvas) tracesPush(this.rows4, { t: nowMs, rows: tracesLineFields(a, ny) });
      }, () => { /* retired */ })
        .catch(e => console.error(e)).then(() => { this.busy4 = false; });
    }
    if (!o.cut) { this.cut.length = 0; this.mark.cut = null; }
    else if (!this.busy.cut && mark !== this.mark.cut) {
      const c = this.card(), ci = c ? c.ci | 0 : 0, mode = sv.modeOf ? sv.modeOf(ci) : 0;
      const p = this.readRow(sv, ci);
      if (p) {
        this.busy.cut = true; this.mark.cut = mark; this.reads.cut++;
        const kind = typeof dispIsSigma === "function" && dispIsSigma(mode) ? "sigma"
                   : typeof dispIsVector === "function" && dispIsVector(mode) ? "mag" : "signed";
        p.then(r => {
          if (sv !== solver || !this.canvas) return;
          // another field: the lines of the old one mean nothing on the new one's scale
          if (this.cut.length && this.cut[this.cut.length - 1].mode !== mode) this.cut.length = 0;
          tracesPush(this.cut, { t: nowMs, row: r.row, max: r.max, kind: kind, mode: mode });
        }, () => { /* retired */ })
          .catch(e => console.error(e)).then(() => { this.busy.cut = false; });
      }
    }
  },
  hasSpecCard() {
    return typeof cards !== "undefined" && typeof autoDissCache !== "undefined" && !!cards.chart &&
           cards.chart.some(c => c.type() === "spectrum");
  },
  pushSpec(perp, sv, nowMs) {
    const p = sv.p;
    tracesPush(this.spec, { t: nowMs, perp: perp, nb: sv.nb, bins: tracesBinRange(p.nx, p.ny, p.Lx, p.Ly) });
  },
  // The picture's middle row of display chain `ci`, and the full scale it was coloured
  // with: a promise of { row: Float32Array(nx), max }, or null when that chain has not
  // been built (nothing has been rendered through it yet). nx + 1 four-byte copies in one
  // encoder into one staging buffer, one map. The caller holds `busy.cut`, so the staging
  // buffer is never mapped twice.
  readRow(sv, ci) {
    const D = sv.disp && sv.disp[ci], dev = sv.device;
    if (!D || !D.buf || !D.buf.dispR || !D.buf.maxVal || !dev) return null;
    const nx = sv.p.nx, plan = tracesRowCopies(nx, sv.p.ny, tracesMidRow(sv.p.ny));
    let S = this.stage;
    if (!S || S.size !== plan.size || S.device !== dev) {
      if (S) { try { S.buf.destroy(); } catch (e) { /* its device is gone */ } }
      S = this.stage = { size: plan.size, device: dev,
                         buf: dev.createBuffer({ size: plan.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) };
    }
    const st = S.buf, enc = dev.createCommandEncoder();
    for (let ix = 0; ix < nx; ix++) enc.copyBufferToBuffer(D.buf.dispR, plan.src[ix], st, 4 * ix, 4);
    enc.copyBufferToBuffer(D.buf.maxVal, 0, st, plan.scaleAt, 4);
    dev.queue.submit([enc.finish()]);
    return st.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(st.getMappedRange().slice(0));
      st.unmap();
      return { row: a.subarray(0, nx), max: a[nx] };
    }, e => { if (this.stage === S) this.stage = null; try { st.destroy(); } catch (e2) { /* gone */ } throw e; });
  },

  // redraw both overlays for the time `nowMs`
  draw(nowMs) {
    const c = this.ctx, cv = this.canvas;
    if (!c || !cv) return;
    if (!this.ro) this.fit();                    // no ResizeObserver: look every frame
    const o = this.opts(), W = cv.width, H = cv.height, d = W / Math.max(1, (this.host.clientWidth || W / this.dpr()));
    const px = d > 0 && isFinite(d) ? d : 1;     // backing pixels per CSS pixel
    const tauCut = o.tau * TRACES_CUT_TAU_FRAC;  // the cut's own, faster, fade
    tracesExpire(this.spec, nowMs, o.tau);
    tracesExpire(this.cut, nowMs, tauCut);
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, W, H);
    if (!o.spec && !o.cut && !o.sides) return;
    const which = this.cmapOf(), rgb = t => cmapRGB(which, t);
    const bright = tracesBrightEnd(rgb(0), rgb(1)), L = tracesLayout(W, H);
    c.lineJoin = "round"; c.lineCap = "round";
    c.setLineDash([]);                           // nothing here is ever dashed
    // one line: the dark halo first (wider, low opacity), then the line in its colour
    const stroke = (pts, age, tau) => {
      if (pts.length < 4) return;
      const w = tracesWeight(age, tau);
      c.beginPath();
      c.moveTo(pts[0], pts[1]);
      for (let i = 2; i < pts.length; i += 2) c.lineTo(pts[i], pts[i + 1]);
      c.strokeStyle = "rgba(0,0,0," + (0.35 * w).toFixed(3) + ")"; c.lineWidth = 3.5 * px;
      c.stroke();
      c.strokeStyle = tracesCss(rgb(tracesColourT(age, tau, bright)), w); c.lineWidth = 1.25 * px;
      c.stroke();
    };
    const clipTo = b => { c.save(); c.beginPath(); c.rect(b.x, b.y, b.w, b.h); c.clip(); };
    // a tick label: small text in the bright end's colour over a soft dark shadow (the
    // text's halo), and nothing else -- no axis, no tick mark
    const pad = 4 * px, font = Math.round(10 * px) + "px sans-serif";
    const text = (t, x, y, align, base) => {
      c.font = font; c.textAlign = align; c.textBaseline = base;
      c.shadowColor = "rgba(0,0,0,0.95)"; c.shadowBlur = 3 * px;
      c.fillStyle = tracesCss(rgb(bright), 0.85);
      c.fillText(t, x, y);
      c.shadowBlur = 0; c.shadowColor = "rgba(0,0,0,0)";
    };
    // what each band will be labelled with, first: the label column is as wide as the
    // longest of them (one column for both bands), and exists only while ticks are shown
    const sv = typeof solver !== "undefined" ? solver : null;
    const bins = !(o.spec || (o.sides && (o.sideA === "spec" || o.sideB === "spec"))) ? null : this.spec.length ? this.spec[this.spec.length - 1].bins
               : sv ? tracesBinRange(sv.p.nx, sv.p.ny, sv.p.Lx, sv.p.Ly) : null;
    const r = bins ? tracesSpecRange(this.ampOf(), bins) : null;
    const decs = r ? tracesDecades(r.e0, r.e1) : [];
    const lastCut = o.cut && this.cut.length ? this.cut[this.cut.length - 1] : null;
    const cutLabs = lastCut ? tracesCutLabels(tracesCutRange(lastCut.kind, lastCut.max)) : [];
    let col = 0;
    if (o.ticks) {
      c.font = font;
      col = tracesLabelCol(decs.map(dcd => c.measureText(tracesPow10(dcd)).width)
                               .concat(cutLabs.map(lb => c.measureText(lb.t).width)), pad);
    }
    const P = tracesPlot(L, o.ticks, col, 14 * px);
    // the side panels, where the page says where the picture is and it leaves room
    const pc = o.sides && this.picture ? this.picture() : null;
    const side = pc && !pc.tile && pc.w > 0 && pc.h > 0
      ? tracesSidePanels(W, H, pc.x * px, pc.y * px, pc.w * px, pc.h * px, TRACES_SIDE_MIN * px, 10 * px) : null;
    if (side) {
      tracesExpire(this.ener, nowMs, o.tau);
      tracesExpire(this.rows4, nowMs, tauCut);
      const span = 4 * o.tau;                      // s of history across an energy panel (2% opacity at its old end)
      // one panel of one strip: `kind` is what the strip shows, `i` the lane, `B` the box
      const panel = (kind, i, B) => {
        let title = "";
        if (kind === "spec") {
          // the lane's spectrum on the bottom band's own axes, with its k^(-5/3) guide
          title = TRACES_LANES[i] + "(k)";
          if (bins) {
            clipTo(B);
            const g = tracesGuide(r, B);
            c.beginPath(); c.moveTo(g.x0, g.y0); c.lineTo(g.x1, g.y1);
            c.strokeStyle = "rgba(0,0,0,0.3)"; c.lineWidth = 3 * px; c.stroke();
            c.strokeStyle = tracesCss(rgb(0.5 + 0.5 * (bright - 0.5)), 0.55); c.lineWidth = px; c.stroke();
            for (const e of this.spec) {
              const age = (nowMs - e.t) / 1000;
              for (const run of tracesLaneRuns(e.perp, e.nb, e.bins, i)) {
                const pts = [];
                for (let j = 0; j < run.length; j += 2) { const p = tracesLogMap(run[j], run[j + 1], r, B); pts.push(p[0], p[1]); }
                stroke(pts, age, o.tau);
              }
            }
            c.restore();
          }
        } else if (kind === "cuts") {
          // against position along the line, signed, zero at mid-height, out to the largest |value| on show
          let hi = 0;
          for (const e of this.rows4) for (const v of e.rows[i]) if (Math.abs(v) > hi) hi = Math.abs(v);
          clipTo(B);
          if (hi > 0) for (const e of this.rows4) {
            const row = e.rows[i], n = row.length, rg = { lo: -hi / 0.92, hi: hi / 0.92 }, pts = [];
            for (let ix = 0; ix < n; ix++) {
              if (!isFinite(row[ix])) { pts.length = 0; break; }
              const p = tracesCutMap(ix, n, row[ix], rg, B);
              pts.push(p[0], p[1]);
            }
            stroke(pts, (nowMs - e.t) / 1000, tauCut);
          }
          c.restore();
          title = TRACES_CUT_NAMES[i] + "(y)" + (hi > 0 ? "  \u00b1" + parseFloat(hi.toPrecision(2)) : "");
        } else {
          // the energy against time: now at the right edge, `span` seconds back at the left,
          // from zero up to the largest value on show; each short piece fades with ITS age
          let top = 0;
          for (const e of this.ener) if (e.v[i] > top) top = e.v[i];
          if (top > 0) {
            clipTo(B);
            const at = e => [B.x + B.w * (1 - (nowMs - e.t) / 1000 / span), B.y + B.h * (1 - 0.92 * e.v[i] / top)];
            for (let j = 0; j + 1 < this.ener.length; j += 3) {
              const pts = [];
              for (let q = j; q <= Math.min(j + 3, this.ener.length - 1); q++) { const p = at(this.ener[q]); pts.push(p[0], p[1]); }
              stroke(pts, (nowMs - this.ener[Math.min(j + 3, this.ener.length - 1)].t) / 1000, o.tau);
            }
            c.restore();
          }
          title = TRACES_LANES[i] + "(t)" + (top > 0 ? "  max " + parseFloat(top.toPrecision(2)) : "");
        }
        if (o.ticks) text(title, B.x + pad, B.y + pad, "left", "top");
      };
      for (let i = 0; i < 4; i++) { panel(o.sideA, i, side.a[i]); panel(o.sideB, i, side.b[i]); }
    }
    if (o.spec && !side) {
      const B = P.spec;
      if (bins) {
        clipTo(B);
        const g = tracesGuide(r, B);
        c.beginPath(); c.moveTo(g.x0, g.y0); c.lineTo(g.x1, g.y1);
        c.strokeStyle = "rgba(0,0,0,0.3)"; c.lineWidth = 3 * px; c.stroke();
        c.strokeStyle = tracesCss(rgb(0.5 + 0.5 * (bright - 0.5)), 0.55); c.lineWidth = px; c.stroke();
        for (const e of this.spec) {                 // oldest first: the newest is on top
          const age = (nowMs - e.t) / 1000;
          for (const run of tracesSpecRuns(e.perp, e.nb, e.bins)) {
            const pts = [];
            for (let i = 0; i < run.length; i += 2) { const p = tracesLogMap(run[i], run[i + 1], r, B); pts.push(p[0], p[1]); }
            stroke(pts, age, o.tau);
          }
        }
        c.restore();
        if (o.ticks) {
          // k in the row under the plot, decades of E in the column to its left
          const ks = tracesKLabels(r.k0, r.k1);
          ks.forEach((k, i) => {
            const x = tracesLogMap(k, r.e0, r, B)[0];
            const t = (i === 0 ? "k = " : "") + (k >= 1 ? String(Math.round(k)) : String(parseFloat(k.toPrecision(2))));
            const al = i === 0 ? "left" : i === ks.length - 1 ? "right" : "center";
            text(t, x + (al === "left" ? pad : al === "right" ? -pad : 0), L.spec.y + L.spec.h - pad, al, "alphabetic");
          });
          for (const dcd of decs)
            text(tracesPow10(dcd), L.spec.x + pad, tracesLogMap(r.k0, Math.pow(10, dcd), r, B)[1], "left", "middle");
        }
      }
    }
    if (o.cut && this.cut.length) {
      // mapped over the WHOLE band (its x is the picture's), drawn only right of the labels
      // ... or, where the page says where the picture is (musicTraces.picture: the stage,
      // whose picture need not span the host), over the picture and each of its repeats
      const pic = this.picture ? this.picture() : null, own = !!(pic && pic.w > 0);
      const B = own ? { x: pic.x * px, y: P.cut.y, w: pic.w * px, h: P.cut.h } : P.cut;
      const offs = [0];
      if (own && pic.tile) {
        offs.length = 0;
        for (let x = B.x - Math.ceil(B.x / B.w) * B.w; x < W; x += B.w) offs.push(x - B.x);
      }
      clipTo(P.cutClip);
      for (const e of this.cut) {
        // each line on the scale ITS picture was coloured with
        const age = (nowMs - e.t) / 1000, n = e.row.length, rg = tracesCutRange(e.kind, e.max), pts = [];
        for (let ix = 0; ix < n; ix++) {
          if (!isFinite(e.row[ix])) { pts.length = 0; break; }
          const p = tracesCutMap(ix, n, e.row[ix], rg, B);
          pts.push(p[0], p[1]);
        }
        for (const dx of offs) stroke(dx ? pts.map((v, i) => (i & 1 ? v : v + dx)) : pts, age, tauCut);
      }
      c.restore();
      if (o.ticks) {
        // the newest line's scale in the column on the left, x along the band's bottom
        for (const lb of cutLabs)
          text(lb.t, P.cut.x + pad, B.y + B.h * (1 - lb.f), "left", lb.f === 1 ? "top" : lb.f === 0 ? "bottom" : "middle");
        if (sv) tracesXLabels(sv.p.Lx).forEach((lb, i) => {
          // the left one would sit on the scale's lowest label: it starts after it
          if (i === 0) return;
          text((i === 1 ? "x = " : "") + lb.t, B.x + B.w * lb.f - (lb.f === 1 ? pad : 0), B.y + B.h - pad,
               lb.f === 1 ? "right" : "center", "alphabetic");
        });
      }
    }
  }
};
