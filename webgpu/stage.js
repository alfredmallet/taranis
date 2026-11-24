"use strict";
// ===========================================================================
// stage.js -- the music page as a STAGE (plans-webgpu/MUSIC_STAGE_PLAN.md, S1).
// Loaded by music.html only. It is the page's layout unless the URL carries `?plain`: with
// that flag every entry point below returns at its first line, no element is created and no
// listener is installed, and the page is the card layout of phases 1 and 2.
//
// Without `?plain`:
//   * `#stage` (position: fixed, the whole window) holds the picture -- the display card's
//     own `.cvwrap`, moved here -- shown WHOLE, centred in the window: the sides left over are black or repeat
//     the picture (the box is periodic) -- the page's `sides` control. The canvas backing size
//     follows the window times devicePixelRatio, capped at STAGE_CAP on its long side. A
//     resize re-fits the canvas and nothing else: the solver is never rebuilt.
//   * a slim bar (the page's own #topbar, plus a drawer button and a fullscreen button) and
//     a drawer holding the status line, the picture's options, the page's whole #controls
//     panel, the readout and the "what is this" links. The drawer takes #controls ITSELF,
//     so a group anyone adds through controlsBuild lands in it with no wiring here.
//   * bar and drawer fade after STAGE_IDLE_MS without activity and return on any; they are
//     always shown while nothing is running. A press that only wakes them is swallowed.
//
// Stacking inside #stage, low to high (stage.css): the picture, any `canvas.traces` child
// (traces.js attaches it; this file never touches it), then the bar and the drawer.
//
// The pure parts -- stageFlag, stageFit, stageIdleNew / stageIdleEvent, stageStatusPlace,
// stageFullscreen -- take everything they need as arguments (the clock included) and touch
// no DOM, so devtools/checkmusic.js checks them directly.
// ===========================================================================

const STAGE_CAP = 1920;            // px, the canvas backing store's long side at most
const STAGE_IDLE_MS = 3000;        // bar and drawer fade after this long without activity
const STAGE_GRACE_MS = 400;        // after a waking press is released, how long a click still belongs to it
const STAGE_RESIZE_MS = 120;       // a window being dragged: the backing store waits for it to settle

// ---- pure: the flag ---------------------------------------------------------------
// The stage IS the page (since 2026-11-24): on unless the URL carries `?plain` / `&plain` as
// a whole word (`?plain=1` is it, `?plainly` is not), which keeps the card layout the page
// was born with -- for the gates and for a look at the charts. `?stage` is accepted and
// means nothing any more.
function stageFlag(search) { return !/[?&]plain(?![\w-])/.test(String(search || "")); }

// ---- pure: window -> canvas -------------------------------------------------------
// The picture is a box of aspect `aspect` (width / height; 1 while the box is square)
// scaled -- without `contain`, which the page never does any more -- to COVER a window of winW x winH CSS pixels, centred, the overhang cropped.
//   cssW, cssH   the picture's CSS size: one of them is the window's, the other is larger
//   left, top    its CSS offset inside the window (each <= 0)
//   w, h         the canvas backing size: the CSS size times `dpr`, scaled down uniformly
//                so that its LONG side is at most `cap`, rounded to even integers >= 2
//                (video encoders refuse odd frame sizes)
// Bad input never throws: a non-finite or non-positive window is 1 x 1, dpr and aspect 1.
// `contain` true: scaled to FIT INSIDE the window instead (nothing cropped; left / top >= 0,
// the window shows through on two sides) -- what the page uses, see stageMeasure.
function stageFit(winW, winH, dpr, aspect, cap, contain) {
  const pos = (v, d) => (isFinite(v) && v > 0 ? v : d);
  const W = pos(winW, 1), H = pos(winH, 1), a = pos(aspect, 1), r = pos(dpr, 1);
  const c = pos(cap === undefined ? STAGE_CAP : cap, STAGE_CAP);
  const wide = (W / H >= a) !== !!contain;       // cover: a window wider than the box fits the width
  const cssW = wide ? W : H * a, cssH = wide ? W / a : H;
  const long = Math.max(cssW, cssH) * r, s = long > c ? c / long : 1;
  const even = v => Math.max(2, 2 * Math.round(v / 2));
  return { cssW: cssW, cssH: cssH, left: (W - cssW) / 2, top: (H - cssH) / 2,
           w: even(cssW * r * s), h: even(cssH * r * s) };
}

// ---- pure: the idle fade, and the press that only wakes -------------------------------
// One state object, advanced by events that carry their own time (ms, any monotonic clock):
//   stageIdleEvent(st, kind, now, o) -> { shown, swallow, waking }
// kind:
//   "move"   activity that is not a press: pointer movement, wheel, a control's input
//   "down"   a press BEGINS: pointerdown, or a keydown that is not a repeat
//   "up"     that press ends: pointerup / pointercancel / keyup
//   "click"  a click event is about to be delivered
//   "tick"   time passed (the page calls it on a timer)
// o.running  false: nothing is playing, and the controls are ALWAYS shown
// o.hold     true: something is holding them up (the mouse rests on them, a fatal error)
// The rules:
//   * shown while !running or hold, and otherwise for STAGE_IDLE_MS after the last
//     activity; "tick" is the only event that hides.
//   * a "down" that arrives while they are HIDDEN shows them and arms the swallow: that
//     press woke the controls and must not also press one. While armed, `waking` is true
//     (the page keeps the controls non-interactive) and "click" returns swallow = true,
//     which disarms. A press that never produces a click disarms STAGE_GRACE_MS after its
//     "up"; a "down" inside that grace belongs to the same gesture and stays swallowed.
//   * "move" never arms anything: a mouse that moved has woken the controls before it
//     clicks, and that click goes through.
function stageIdleNew(now) {
  return { last: now, shown: true, armed: false, upAt: null };
}
function stageIdleEvent(st, kind, now, o) {
  const running = !!(o && o.running), hold = !!(o && o.hold);
  // an armed swallow whose press ended long enough ago has expired, whatever comes next
  if (st.armed && st.upAt !== null && now - st.upAt >= STAGE_GRACE_MS) { st.armed = false; st.upAt = null; }
  let swallow = false;
  if (kind === "down") {
    if (!st.shown) { st.armed = true; st.upAt = null; }
    else if (st.armed) st.upAt = null;           // a second press inside the grace: still the wake
    st.last = now; st.shown = true;
  } else if (kind === "up") {
    if (st.armed && st.upAt === null) st.upAt = now;
    st.last = now;
  } else if (kind === "click") {
    swallow = st.armed;
    st.armed = false; st.upAt = null;
    st.last = now; st.shown = true;
  } else if (kind === "move") {
    st.last = now; st.shown = true;
  } else {                                       // "tick"
    st.shown = !running || hold || now - st.last < STAGE_IDLE_MS;
  }
  return { shown: st.shown, swallow: swallow, waking: st.armed };
}

// ---- pure: where a status message goes --------------------------------------------------
// "none" (nothing to show), "drawer", or "centre" for an error that stops the page: there
// is no solver (WebGPU missing or refused), or the device was lost or raised an error.
function stageStatusPlace(kind, text, hasSolver) {
  if (!text || (kind !== "err" && kind !== "info")) return "none";
  if (kind === "err" && (!hasSolver || /^WebGPU (device lost|error)/.test(text))) return "centre";
  return "drawer";
}

// ---- pure: the Fullscreen API, or null where there is none -------------------------------
// `doc` is the document, `host` the element to put on the whole screen. Standard names
// first, the webkit-prefixed ones (older Safari, iPad) second. iPhone Safari has neither
// for a plain element: null, and the page then shows no button.
function stageFullscreen(doc, host) {
  if (!doc || !host) return null;
  const req = host.requestFullscreen || host.webkitRequestFullscreen;
  const exit = doc.exitFullscreen || doc.webkitExitFullscreen;
  const enabled = doc.fullscreenEnabled !== undefined ? doc.fullscreenEnabled : doc.webkitFullscreenEnabled;
  if (!req || !exit || !enabled) return null;
  const std = !!host.requestFullscreen;
  const on = () => !!(doc.fullscreenElement || doc.webkitFullscreenElement);
  return { on: on, event: std ? "fullscreenchange" : "webkitfullscreenchange",
           // returns whatever the browser does (a promise, or nothing on the prefixed API)
           toggle: () => (on() ? exit.call(doc) : req.call(host)) };
}

// ===========================================================================
// the page side
// ===========================================================================
const STAGE_ON = stageFlag(typeof location !== "undefined" ? location.search : "");
const stage = {
  el: null, ui: null, bar: null, drawer: null, fatal: null,   // #stage and its chrome
  slotStatus: null, slotPic: null, slotRead: null,            // the drawer's hosts
  btnMenu: null, btnFs: null, fs: null,
  back: null,                    // the backing size in force: what cardAspect() answers
  backTimer: 0,
  idle: null,                    // stageIdleEvent's state
  hover: false,                  // the mouse rests on the bar or the drawer
  touch: false,                  // the last pointer was a finger
  isFatal: false, seen: "", dismissed: "",
  adopted: [],                   // { card, nodes } of the display cards taken over
  tile: null, tileCtx: null, tileKey: "",     // the canvas the repeated picture is drawn on (stageFrame)
  fit: null,                     // the picture's CSS box in force (stageCss)
  timer: 0
};
const stageNow = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
function stageMk(tag, id, cls, parent) {
  const e = document.createElement(tag);
  if (id) e.id = id;
  if (cls) e.className = cls;
  if (parent) parent.appendChild(e);
  return e;
}
// move a node under `host` (out of wherever it is)
function stageMove(node, host) {
  if (!node || !host || node.parentNode === host) return;
  if (node.parentNode && node.parentNode.removeChild) node.parentNode.removeChild(node);
  host.appendChild(node);
}
function stageSection(title) {
  const s = stageMk("div", "", "stagesec", stage.drawer);
  if (title) stageMk("div", "", "stagehd", s).textContent = title;
  return s;
}

// ---- build: called once by music.html, right after controlsBuild -----------------------
function stageBuild() {
  if (!STAGE_ON || stage.el) return;
  const root = document.documentElement;
  if (root && root.classList) root.classList.add("stage");
  // the picture reaches under a phone's notch and rounded corners only if the viewport says so
  const vp = document.querySelector && document.querySelector('meta[name="viewport"]');
  if (vp && vp.setAttribute && !/viewport-fit/.test(vp.getAttribute("content") || ""))
    vp.setAttribute("content", (vp.getAttribute("content") || "") + ", viewport-fit=cover");

  // (devtools' stub document has no <body>: there #stage hangs off #display)
  const S = stage.el = stageMk("div", "stage", "", document.body || document.getElementById("display"));
  stage.tile = stageMk("canvas", "stageTile", "", S);
  stage.tile.style.display = "none";
  stage.ui = stageMk("div", "stageUi", "", S);
  stage.bar = stageMk("div", "stageBar", "", stage.ui);
  stage.drawer = stageMk("div", "stageDrawer", "", stage.ui);
  stage.fatal = stageMk("div", "stageFatal", "", S);

  // the bar: the page's own top bar, whole (so whatever controlsBuild puts in it is here),
  // then the two buttons that are the stage's own
  stageMove(document.getElementById("topbar"), stage.bar);
  const bm = stage.btnMenu = stageMk("button", "stageBtnMenu", "", stage.bar);
  bm.textContent = "controls";
  bm.title = "show or hide the controls";
  bm.setAttribute("aria-expanded", "false");
  bm.onclick = () => stageDrawerOpen(!stageDrawerIsOpen());
  const bf = stage.btnFs = stageMk("button", "stageBtnFs", "", stage.bar);
  bf.textContent = "fullscreen";
  bf.title = "fill the whole screen";
  stage.fs = stageFullscreen(document, root);
  if (!stage.fs) bf.style.display = "none";       // no Fullscreen API (iPhone Safari): no button
  else {
    bf.onclick = () => {
      try { const p = stage.fs.toggle(); if (p && p.catch) p.catch(() => {}); } catch (e) { /* refused */ }
    };
    if (document.addEventListener) document.addEventListener(stage.fs.event, stageFsSync);
  }

  // the drawer, top to bottom: status, the picture's options, the controls, the readout, about
  stage.slotStatus = stageSection("");
  stage.slotStatus.id = "stageStatus";
  // (#status itself is placed by stageStatusSync, at the tick that ends this function)
  stage.slotPic = stageSection("picture");
  stage.slotPic.id = "stagePic";
  stageMove(document.getElementById("controls"), stageSection(""));
  stage.slotRead = stageSection("");
  stage.slotRead.id = "stageRead";
  stageMove(document.getElementById("steps"), stage.slotRead);
  stageMove(document.getElementById("readout"), stage.slotRead);
  const about = stageSection("what is this");
  about.id = "stageAbout";
  const p = stageMk("div", "", "hint", about);
  p.appendChild(document.createTextNode("2D reduced-MHD turbulence, computed in this page as you "
    + "watch: every onset in the music kicks the flow. "));
  const link = (href, text) => { const a = stageMk("a", "", "", p); a.href = href; a.textContent = text; return a; };
  link("docs.html", "how it works");
  p.appendChild(document.createTextNode(" · "));
  link("rmhd2d.html", "the simulation, with its charts");
  stageMove(document.getElementById("buildline"), about);

  stage.idle = stageIdleNew(stageNow());
  stage.back = stageMeasure();
  stageListen();
  stage.timer = setInterval(() => stageTick(stageNow()), 250);
  stageTick(stageNow());
}

// ---- the picture --------------------------------------------------------------------
// the window, as the stage sees it: #stage's own box where the browser reports one (it is
// 100dvh, the VISIBLE viewport on a phone), else the window's
function stageWindow() {
  const S = stage.el, w = typeof window !== "undefined" ? window : {};
  return { w: (S && S.clientWidth) || w.innerWidth || 1, h: (S && S.clientHeight) || w.innerHeight || 1,
           dpr: w.devicePixelRatio || 1 };
}
// the box's aspect: the live solver's (square today; the window-shaped box of S2 follows
// by itself, and a resize after it was built then crops to cover)
function stageBoxAspect() {
  const p = typeof solver !== "undefined" && solver ? solver.p : null;
  return p && p.Lx > 0 && p.Ly > 0 ? p.Lx / p.Ly : 1;
}
// The whole box is always shown (contain), centred: a window of another shape than the box
// has two sides left over, which are black or carry the picture again (stageFill).
function stageMeasure() {
  const W = stageWindow();
  return stageFit(W.w, W.h, W.dpr, stageBoxAspect(), STAGE_CAP, true);
}
// what the sides show: "tile" (the picture repeated -- the box is periodic, so it joins
// itself) or "bars" (black). The page's selFill; black where there is no such control.
function stageFill() {
  const s = typeof document !== "undefined" && document.getElementById ? document.getElementById("selFill") : null;
  return s && s.value === "tile" ? "tile" : "bars";
}
// pure: the copies of a picture at (left, top), size w x h, that cover a W x H window --
// [x, y] of each, the picture's own place included. Sizes in whole pixels, so copies abut.
function stageTiles(W, H, left, top, w, h) {
  const out = [];
  if (!(w >= 1 && h >= 1 && W > 0 && H > 0)) return out;
  const i0 = -Math.ceil(left / w), i1 = Math.ceil((W - left) / w) - 1;
  const j0 = -Math.ceil(top / h), j1 = Math.ceil((H - top) / h) - 1;
  for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) out.push([left + i * w, top + j * h]);
  return out;
}
// Once per rendered frame, in the SAME task as the render (a WebGPU canvas can only be read
// then): with the sides on "tile", copy the picture over the whole window. The tile canvas
// sits over the picture and shows it too, so every copy is the same pixels and the camera
// pulse (music.html scales this canvas) moves them as one.
// `drawn`: the card rendered in this task. A WebGPU canvas reads back blank in any other,
// so on a frame without a render (a paused page) nothing is copied -- and if the copies on
// show are not the current ones (just switched on, the window or the picture moved), the
// card is marked dirty, which makes the next frame render and this copy it.
function stageFrame(drawn) {
  const t = stage.tile;
  if (!t) return;
  const a = stage.adopted[0], f = stage.fit, on = stageFill() === "tile" && !!a && !!f;
  const disp = on ? "block" : "none";
  if (t.style.display !== disp) t.style.display = disp;
  if (!on) { stage.tileKey = ""; return; }
  const W = stageWindow(), long = Math.max(W.w, W.h) * W.dpr, s = W.dpr * (long > STAGE_CAP ? STAGE_CAP / long : 1);
  const tw = Math.max(1, Math.round(W.w * s)), th = Math.max(1, Math.round(W.h * s));
  if (t.width !== tw || t.height !== th) { t.width = tw; t.height = th; }
  const c = stage.tileCtx || (stage.tileCtx = t.getContext("2d"));
  if (!c) return;
  const w = Math.max(1, Math.round(f.cssW * s)), h = Math.max(1, Math.round(f.cssH * s));
  const key = [tw, th, w, h, Math.round(f.left * s), Math.round(f.top * s)].join();
  if (!drawn) { if (key !== stage.tileKey) a.card.dirty = true; return; }
  if (key !== stage.tileKey) c.clearRect(0, 0, tw, th);
  stage.tileKey = key;
  for (const p of stageTiles(tw, th, Math.round(f.left * s), Math.round(f.top * s), w, h))
    c.drawImage(a.card.cv, p[0], p[1], w, h);
}
// where the picture is, for the trace overlay: its left edge and width in CSS pixels of
// #stage, and whether it repeats. null with ?plain.
function stagePicture() {
  if (!STAGE_ON || !stage.fit) return null;
  return { x: stage.fit.left, w: stage.fit.cssW, y: stage.fit.top, h: stage.fit.cssH, tile: stageFill() === "tile" };
}
// put the picture's CSS box where the fit says
function stageCss(f) {
  stage.fit = f;
  for (const a of stage.adopted) {
    const st = a.card.wrap.style;
    st.left = f.left.toFixed(2) + "px"; st.top = f.top.toFixed(2) + "px";
    st.width = f.cssW.toFixed(2) + "px"; st.height = f.cssH.toFixed(2) + "px";
  }
}
// What music.html's cardAspect() answers on the stage (null with ?plain): the backing size
// for the window AS IT IS NOW. The shared card code asks every time it re-applies a card
// (cardsSync: a control moved, a rebuild), so a solver rebuilt with another box shape is
// picked up with no call from here.
function stageAspect() {
  if (!STAGE_ON) return null;
  const f = stage.back = stageMeasure();
  stageCss(f);
  return { w: f.w, h: f.h };
}
// Re-fit to the window. The CSS box moves at once; the backing store follows when the
// window has settled (`now` true: at once), because setting a canvas's size clears it and
// a window being dragged would flash. The solver is not involved at any point.
function stageResize(now) {
  if (!STAGE_ON || !stage.el) return;
  const f = stageMeasure();
  stageCss(f);
  clearTimeout(stage.backTimer);
  stage.backTimer = 0;
  if (stage.back && f.w === stage.back.w && f.h === stage.back.h) return;
  if (now) stageBackApply();
  else stage.backTimer = setTimeout(stageBackApply, STAGE_RESIZE_MS);
}
function stageBackApply() {
  stage.backTimer = 0;
  // the cards size their own canvases, through cardAspect() -> stageAspect(): re-apply them
  if (typeof cards !== "undefined" && cards.cfg && cards.disp.length && typeof solver !== "undefined" && solver) cardsSync();
  else stage.back = stageMeasure();
  stageOverlayTrim();
}
// The card's arrow overlay is a second canvas, sized canvas x devicePixelRatio by the
// shared code: 3840 x 3840 at the cap on a retina screen, for an overlay this page never
// draws. Keep it at one pixel (it is hidden by stage.css; a saved PNG composites it, blank).
function stageOverlayTrim() {
  for (const a of stage.adopted) {
    const v = a.card.cvVec;
    if (v && (v.width !== 1 || v.height !== 1)) { v.width = 1; v.height = 1; }
  }
}

// ---- adopt: called by music.html after cardsLayout ---------------------------------------
// Take over what the card layout made: the charts go, the display card's canvas becomes the
// picture, and its header (field, colormap, contours) and footer (caption, colour bar,
// save / rec and their result strip) move into the drawer. Safe to call again after any
// later re-layout.
function stageAdopt() {
  if (!STAGE_ON || !stage.el) return;
  // the science charts are not on the stage (and cost a readback each)
  for (const c of cards.chart.slice()) cardClose(c);
  // cards that are gone: their moved nodes go too
  stage.adopted = stage.adopted.filter(a => {
    if (cards.disp.indexOf(a.card) >= 0) return true;
    for (const n of a.nodes) if (n.parentNode) n.parentNode.removeChild(n);
    return false;
  });
  for (const c of cards.disp) {
    if (stage.adopted.some(a => a.card === c)) continue;
    const head = c.selField.parentNode;
    stageMove(c.wrap, stage.el);
    stageMove(head, stage.slotPic);
    stageMove(c.foot, stage.slotPic);
    // not on this page: arrows, the per-card offsets, closing the one picture
    let moved = false;
    if (c.cbArrow.checked) { c.cbArrow.checked = false; moved = true; }
    if (c.cbArrow.parentNode) c.cbArrow.parentNode.style.display = "none";
    for (const r of [c.rOffX, c.rOffY]) if (r) { if (r.value !== "0") { r.value = "0"; moved = true; } _rngShow(r, false); }
    c.btnClose.style.display = "none";
    stage.adopted.push({ card: c, nodes: [c.wrap, head, c.foot] });
    if (moved && solver) c.apply();
  }
  // ... nor the per-card k filter (its slider is a card's). "+ display", "+ chart" and
  // "rec all" are hidden by stage.css: the shared code rewrites the last one's inline display.
  const cf = document.getElementById("cbFilter");
  if (cf && cf.parentNode) cf.parentNode.style.display = "none";
  // in a drawer the groups start folded, all but the first
  const host = document.getElementById("controls");
  let i = 0;
  for (const d of (host && host.children) || [])
    if ((d.tagName || d.kind || "").toLowerCase() === "details") d.open = i++ === 0;
  stageResize(true);
  stageOverlayTrim();
}

// ---- the drawer ------------------------------------------------------------------------
function stageDrawerIsOpen() { return !!stage.drawer && stage.drawer.classList.contains("open"); }
function stageDrawerOpen(b) {
  if (!stage.drawer) return;
  stage.drawer.classList.toggle("open", !!b);
  stage.btnMenu.setAttribute("aria-expanded", b ? "true" : "false");
  stage.btnMenu.classList.toggle("on", !!b);
}
function stageFsSync() {
  if (!stage.fs || !stage.btnFs) return;
  const on = stage.fs.on();
  stage.btnFs.textContent = on ? "exit fullscreen" : "fullscreen";
  stage.btnFs.title = on ? "leave fullscreen" : "fill the whole screen";
  stageResize(false);
}

// ---- status: the drawer, or the centre for an error that stops the page -------------------
// showStatus (common.js) writes #status; this looks at what it wrote, on the timer. A new
// message opens the drawer (a closed drawer would hide "choose an audio file first") and
// wakes the controls. The centred one is dismissed by a click and returns with a new text.
function stageStatusSync() {
  const st = document.getElementById("status");
  if (!st || !stage.el) return;
  const kind = st.className, text = st.textContent || "", key = kind + "|" + text;
  const place = stageStatusPlace(kind, text, typeof solver !== "undefined" && !!solver);
  const fatal = place === "centre" && stage.dismissed !== key;
  // gpuFallback (common.js) moves #status under its poster card, which the stage hides:
  // wherever it was put, it belongs in one of these two
  stageMove(st, fatal ? stage.fatal : stage.slotStatus);
  stage.isFatal = fatal;
  stage.el.classList.toggle("fatal", fatal);
  stage.fatal.onclick = () => { stage.dismissed = key; stageStatusSync(); };
  if (key !== stage.seen) {
    stage.seen = key;
    if (place !== "none") {
      if (place === "drawer") stageDrawerOpen(true);
      stageEvent("move", stageNow());
    }
  }
}

// ---- idle fade: events in, classes out ----------------------------------------------------
function stageRunning() { return typeof running !== "undefined" && !!running; }
function stageEvent(kind, now) {
  if (!stage.idle) return { shown: true, swallow: false, waking: false };
  // held up: the mouse rests on them, a fatal error, or -- on touch, which has no hover --
  // the drawer is open (it is being read; closing it lets them fade)
  const r = stageIdleEvent(stage.idle, kind, now, { running: stageRunning(),
    hold: stage.hover || stage.isFatal || (stage.touch && stageDrawerIsOpen()) });
  stage.el.classList.toggle("idle", !r.shown);
  stage.el.classList.toggle("waking", r.waking);
  return r;
}
function stageTick(now) {
  if (!stage.el) return;
  stageStatusSync();
  stageOverlayTrim();
  stageEvent("tick", now);
}
function stageListen() {
  const w = typeof window !== "undefined" ? window : null;
  if (!w || !w.addEventListener) return;
  const cap = { capture: true, passive: true }, capActive = { capture: true, passive: false };
  const inUi = t => !!(t && t.closest && t.closest("#stageBar, #stageDrawer"));
  w.addEventListener("pointermove", e => {
    stage.hover = e.pointerType === "mouse" && inUi(e.target);
    stage.touch = e.pointerType === "touch";
    stageEvent("move", stageNow());
  }, cap);
  w.addEventListener("pointerdown", e => {
    stage.touch = e.pointerType === "touch";
    const r = stageEvent("down", stageNow());
    // the press that woke the controls: no compatibility mouse events from it either (a
    // <select> opens on mousedown), and stageTick lifts `waking` after the grace
    if (r.waking && e.cancelable) e.preventDefault();
  }, capActive);
  // ("blur": a key held while the window loses focus never sends its keyup, and an armed
  // swallow with no "up" would never expire)
  for (const k of ["pointerup", "pointercancel", "keyup", "blur"])
    w.addEventListener(k, () => { stageEvent("up", stageNow()); setTimeout(() => stageTick(stageNow()), STAGE_GRACE_MS + 20); }, cap);
  w.addEventListener("keydown", e => {
    if (e.key === "Escape" && stage.idle.shown && stageDrawerIsOpen()) stageDrawerOpen(false);
    stageEvent(e.repeat ? "move" : "down", stageNow());
  }, cap);
  w.addEventListener("click", e => {
    if (!e.isTrusted) return;                     // a click the page makes itself (a download link)
    if (!stageEvent("click", stageNow()).swallow) return;
    e.preventDefault();
    e.stopImmediatePropagation();
  }, capActive);
  for (const k of ["wheel", "input", "change", "focusin"]) w.addEventListener(k, () => stageEvent("move", stageNow()), cap);
  if (document.documentElement && document.documentElement.addEventListener)
    document.documentElement.addEventListener("pointerleave", () => { stage.hover = false; });
  w.addEventListener("resize", () => stageResize(false));
  if (w.visualViewport && w.visualViewport.addEventListener)
    w.visualViewport.addEventListener("resize", () => stageResize(false));
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => stageResize(false)).observe(stage.el);
}
