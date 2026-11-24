# MUSIC_STAGE_PLAN — phase 3 of the music project: the page as a stage

> **STATUS (2026-11-23): accepted by Alfred the day it was drafted, with the four decisions
> at the end settled as proposed. S1–S4 are BUILT behind `?stage` the same day, with the
> changes Alfred made while watching it recorded in "Build record" at the end — where that
> section and the text above it disagree, the record is what was built. Not yet done: the
> stage as the page's only layout (the card layout and its charts still exist without the
> flag), and a look on a phone.**
> Phases 1 and 2 are published (MUSIC_PLAN.md, MUSIC_EFFECTS_PLAN.md); nothing in them is
> reopened here except where this file says so by name.

## Why

`music.html` borrows the science pages' layout: a title, a control panel, a 512-pixel
display card, an energy chart and a spectrum chart. That is the right frame for
`rmhd2d.html` and the wrong one for this page, which is watched, not read. Alfred's brief
(2026-11-23): the picture should fill the window with no borders; the spectrum and the line
cut become optional overlays, drawn as fading traces in the colormap's colours; the box
should take the window's shape; and a maintained current-sheet equilibrium should be
offered as a background, as an alternative to the walking mean field. The page stays a
demonstration — the spectrum overlay is there because the flow still shows a −5/3 range.

Four parts:

| part | what | touches |
|---|---|---|
| S1 | the stage: full-window picture, controls in a drawer | `music.html`, CSS |
| S2 | the box takes the window's shape | `music.html`, `music.js`, the offline script, fixtures |
| S3 | fading trace overlays (spectrum, line cut) | `music.html`, new draw code |
| S4 | current sheets as a maintained background | `solver2d.js` (JS only), `music.html`, the offline script |

## What the code says today (read 2026-11-23)

- **The grid is 256², fixed** (`MUSIC_N = 256`, box 2π × 2π, hyper 4). So "512 × 256 at
  most" is not a reduction: it is twice the cells of the shipped page.
- **The solver is already rectangular.** `solver2d.js` takes `nx, ny, Lx, Ly`
  independently (`makeGrid`, the dealias cuts, `nbins(nx, ny, Lx, Ly)`), and
  `rmhd2d.html`'s equilibrium demos run 512 × 128 in a 4π × 2π box. Each axis must be a
  power of two in [32, 1024] (the line FFT), so the grid's shape is 1:1, 2:1 or 4:1 only.
- **The music code assumes a square** in `music.js`: `MUSIC_LBOX` is used for both axes in
  blob positions, wrapping, the stereo map (`musicPanX`) and `musicBandPlan(n, …)`;
  `musicDiss` in `music.html` uses `MUSIC_N / 3`.
- **The offline script is square** (`nx=n, ny=n, Lx=LBOX, Ly=LBOX`), and so are both
  fixtures the page is pinned to.
- **The maintained equilibrium exists** (`eqsrc`, REFINE_PLAN J2.3): `srcInit` copies the
  k_y = 0 column of ψ at Reset into `eqk`, and `nlAssemble` adds `−gridB.y · eqk`. It is a
  compile-time flag, so turning it on is a rebuild.
- **`eqsrc` and the drag are refused together** (`Solver._dragLin` throws). The reason
  given is that `gridB.y` then includes the drag, so the source would hold the equilibrium
  up against the drag as well as the resistivity. On the science pages that would falsify a
  quoted growth rate. On this page it is the wanted behaviour: without it the drag
  (energy e-folding time 1 s of audio by default) would remove the sheets in a second.
- **The mean field is a second compile-time flag** (`b0: true`, the variant `stage`
  kernel), on always on this page.
- **Timings, laptop (M1), from FFTPERF_PLAN.md (2026-08-21):** whole step 6.7 ms at 512²
  and 30 ms at 1024², ratio 4.4–4.5 per doubling of both axes. That puts 256² near 1.5 ms
  and 512 × 256 near 3.2 ms. The real-Chrome test of the shipped page (2026-11-22, click
  track, defaults) needed 6.3 steps per frame. **512 × 256 has not been timed.**
- **The zoom pulse is a CSS transform** on the card canvas; it is not in recordings or
  saved PNGs. Recordings capture the WebGPU canvas only, video only.

## Rules

- **The solver's kernels do not change.** S4 needs one JS-side opt-in and nothing in WGSL;
  `devtools/checksolver2d.js`'s byte pins must pass with no new allowance block. If a part
  turns out to need a kernel change, stop and bring it back to Alfred.
- **The other pages are untouched**, in behaviour and in speed. Everything new in
  `common.js` is called only from `music.html`.
- **Built behind `?stage` first.** The current page keeps working unchanged until Alfred has
  judged the new one on screen; only then does the stage become the page and the card
  layout come off it (step 6). No long-lived switch between the two layouts.
- **Offline first where a movie can settle it** (S4, and the rectangular box in S2). S1 and
  S3 are page chrome with no offline counterpart, so they are judged on the page.
- **With the new options at their "off" values the simulation is the shipped one**: square
  2π box at 256², walking mean field, same kicks draw for draw. The existing fixtures are
  the proof and are not regenerated to make anything pass.
- Carried over: kicks stay state edits; one implementer, gates run by the implementer, one
  adversarial review at the end; `beths.wav` stays local; `TARANIS_PRECISION=32` for the
  script; commit and push only at Alfred's word.

## S1 — the stage

- The picture is one canvas filling the window (`100vw × 100dvh`, no margins, no title
  block). The WebGPU canvas's backing size follows the window times `devicePixelRatio`,
  capped (proposed: 1920 on the long side) so a 4K window does not quadruple the display
  pass.
- **Controls** live in a drawer over the picture: Start/Stop and the source select stay
  reachable in a slim bar; the three control groups open from it. Bar and drawer fade out
  after a few seconds without pointer or touch activity and return on any. Before Start the
  bar is always shown.
- **Status and errors** (no WebGPU, capture refused, the sync line) move into the drawer,
  except errors that stop the page, which show centred.
- **A "what is this" link** in the drawer goes to the docs and to `rmhd2d.html`. The
  science charts (energy, the framed spectrum) leave this page at step 6.
- **Fullscreen** button (the Fullscreen API; absent on iPhone Safari, where the button is
  hidden).
- The pulse stays a CSS transform; the window edge clips it.
- The existing display options (field, colormap, fixed colour scale) stay, in the drawer.
  Arrows and the per-card offset sliders are dropped from this page.

## S2 — the box takes the window's shape

**Shape.** The box's aspect is the window's, read when Start (or Reset) is pressed:
`Lshort = 2π` on the window's short side and `Llong = 2π · aspect` on the long side, with
the aspect clamped to [1, 2.4]. Portrait windows get a tall box (`ny > nx` where the grid
is not square). A window resized or rotated mid-run does not rebuild anything: the picture
is cropped to cover the new window until the next Start. Rebuilding on every resize event
would restart the flow while a window is being dragged.

**Grid.** The cells need not be square, which is what lets a power-of-two grid fill a 16:9
window exactly. A three-rung ladder, long side × short side:

| rung | grid | cells vs today | cell shape at 16:9 | expected step cost vs today |
|---|---|---|---|---|
| light | 256 × 128 | 0.5 | 0.89 : 1 | about 0.45 |
| standard | 256 × 256 | 1 | 1.78 : 1 | 1 |
| sharp | 512 × 256 | 2 | 0.89 : 1 | about 2.2 |

- *Standard* costs what the page costs now and fills the window, at the price of cells
  1.78 times wider than tall on a 16:9 screen (the short axis carries resolution the long
  axis cannot match).
- *Sharp* is Alfred's proposed ceiling. By the timings above it is about 3.2 ms per step on
  the M1 against 1.5 ms now, while the step count per frame stays the same (the CFL step is
  set by the smaller spacing, which does not change). Whether that still keeps real time is
  what step 0 measures; the estimate says it is marginal on the M1 and out of reach on
  phones.
- *Light* is for phones and old machines.

**Choosing the rung.** A select in the drawer (`auto`, light, standard, sharp), default
`auto`. Auto starts on standard and uses the clock the page already has: if the run is
rebased for falling behind more than a set number of times in the first seconds, it drops
one rung at the next natural rebuild and says so in the status line; it never moves up by
itself. (A timing probe before Start was considered and is the fallback if the rebase
count proves too noisy.)

**What has to stop assuming a square** (each is a small change, each gets a gate row):

- `musicBandPlan`: the narrowest blob is 3.7 times the *larger* grid spacing; the widest
  stays 0.45 (a fraction of the short side).
- Blob positions and wrapping use `Lx` and `Ly` separately; stereo pan maps onto x in
  landscape. In portrait x is the short side — placement still works, over a narrower
  span.
- `musicDiss`: the hyperviscous rate is pinned at the *smaller* of the two dealias cutoffs
  in k, so the coarser direction is the one that is resolved.
- Blob counts per kick scale with box area over (2π)², so a wider box is not sparser.
  With a square box this factor is exactly 1.
- The walk's `sqrt(2E)` and the fixed colour scale are per unit area already; check, do not
  assume.

**Offline.** `examples/music_turbulence.py` gains `--aspect` and `--grid NXxNY` (taranis
already takes `nx, ny, Lx, Ly`), with the same four rules. The fixtures gain rectangular
rows written from it; the square rows are untouched.

## S3 — fading traces

A 2D canvas over the picture, drawn by the page, never part of the solver.

- **Spectrum.** `Solver.readSpectrum()` already exists and is what the chart card reads.
  Every few frames (proposed: every 6th, about 10 per second) the newest spectrum is
  pushed onto a short history; each frame the history is redrawn with opacity
  `exp(−age/τ)`, τ about 2 s, entries below 2% dropped. Log–log, no frame, no ticks: a
  faint −5/3 guide line and two or three small wavenumber labels. Which spectra: total
  energy by default; an option for the two Elsasser energies as two families of lines.
- **Line cut.** `readCutLine` gives (u_x, u_y, b_x, b_y) along a line; the overlay draws
  one or two components the same way. Off by default.
- **Colour.** Lines take their colour from the display's colormap: the newest from the
  bright end, sliding toward the middle as they age. Because a line drawn in the image's
  own colours can vanish against it, every line gets a thin dark halo (drawn first, wider,
  low opacity); a soft dark scrim under the plot area is the fallback if the halo is not
  enough.
- **Placement.** Spectrum bottom-left, cut along the bottom edge, each about a third of the
  window's short side; both toggled from the drawer. They do not fade with the controls.
- **Cost.** One extra readback every 6th frame at most (the spectrum card does one per
  frame today, so this is a reduction) and a few hundred line segments on a 2D canvas.
- **Bins.** In a non-square box the lowest few bins are sparsely filled; the overlay starts
  at the first bin whose shell is complete in both directions.

Recordings: the overlays and the pulse are outside the WebGPU canvas, so a recording made
from this page shows the picture alone. Composing them into recorded frames is deferred
(see "Not in this plan").

## S4 — current sheets as a background

A **background** select replaces the mean-field controls' on/off: `walking mean field`
(today's default), `current sheets`, `none`. The two real backgrounds are exclusive, as
Alfred asked; switching is a rebuild, because each is a compile-time kernel flag (`b0` or
`eqsrc`, never both).

- **The equilibrium.** `ψ_eq = ψ0 sech²((x − Lx/2)/a)`, the form `rmhd2d.html`'s tearing
  presets use (`common.js`, "Tearing [19]"): no net flux, periodic, ONE current sheet at the box
  centre with weaker return currents of the opposite sign either side (the review's
  correction, 2026-11-24: this text first said "two current sheets"). It varies along x only, which is what the
  maintained source requires (it reads the k_y = 0 column). On a landscape window the
  sheets are vertical, so stereo placement puts left-channel sounds on the left sheet.
  In portrait they stay vertical, closer together.
- **Controls.** *sheet field (× amp)*: the equilibrium's peak field as a multiple of the
  amp slider, proposed range 0–3, default 1.5 (the same number as the mean field's default,
  as a starting point for the offline judging). *sheet width*: `a` as a multiple of the
  narrowest blob width, proposed default 2.
- **Maintained against the drag.** A new solver parameter `eqsrcDrag: true` lifts
  `_dragLin`'s refusal, for this page only. Nothing else changes: the source is already
  `−gridB.y · eqk`, and with the drag folded into `gridB.y` it cancels the drag's decay of
  the equilibrium exactly, as it cancels the resistive decay now. Without the flag the
  refusal stands, so `rmhd2d.html` cannot reach the combination by accident.
- **What the kicks do to it.** `eqk` is taken once, at Reset, from the initial condition.
  Kicks add to the k_y = 0 column afterwards and the source does not see them, so kicks
  decay under the drag as before while the sheets are restored. Each kick tears at the
  sheets; the source rebuilds them over the drag time.
- **Honest label.** With hyper 4 there is no resistive layer, so this is not the tearing
  demo: it is a restored current-sheet pair being broken up by impulsive kicks. The
  drawer's tooltip says so and points at `rmhd2d.html` for the real instability.
- **Amplitude slider.** `ψ0` is fixed at Reset; moving *sheet field* or *amp* afterwards
  takes effect at the next Reset (stated in the tooltip). Re-deriving `eqk` live is
  possible later and not needed to judge the effect.
- **Offline.** The script gets `--sheets [STRENGTH]` and `--sheet-width`. No change to
  `taranis/`: the source `−L·ψ_eq` is added to the nonlinear term the script already hands
  the stepper, with `L` the script's own diagonal (hyperviscosity plus drag). It raises
  with `--mean-field` or `--field-walk`, as `--ratio-walk` without `--field-walk` does now.

## Steps

0. **Measure** (no decisions needed, an hour): time 256 × 128, 256², 512 × 256 and 512² in
   real Chrome on the M1 through the existing `?bench` harness, and the music page's steps
   per frame at each on the click track. If a phone is to hand, the same there. Record the
   table here; it settles decision 1.
1. **S4 and the rectangular box, offline.** `--sheets`, `--aspect`, `--grid` in the script,
   gate rows in `music_turbulence_check.py`. Movies for Alfred: sheets at two strengths and
   two widths on the Beths clip and the click track; one 16:9 movie at standard and one at
   sharp, same clip, to judge the stretched cells.
2. **Alfred judges**; defaults and the rung question are fixed from the movies.
3. **S2 and S4 on the page**, under the current layout: the grid from the window and the
   rung select, the background select, `eqsrcDrag`. Fixtures regenerated with the new rows;
   `checkmusic.js`, `checkkick.js`, `checksolver2d.js` green, the last with no new
   allowance.
4. **S1 behind `?stage`**: the full-window canvas, the drawer, idle fade, fullscreen.
5. **S3 behind `?stage`**: the two overlays.
6. **Alfred judges the stage** in Chrome on the laptop and on a phone. On acceptance:
   `?stage` becomes the page, the card layout and the two charts come off it, the README
   sections are rewritten.
7. **One adversarial review** of the whole phase (mutation-tested, as phase 2), fixes, then
   merge and publish at Alfred's word.

## Gates

- `examples/music_turbulence_check.py`: the rectangular band plan, placement and
  dissipation rules; the sheet equilibrium is stationary with no kicks and drag on (the
  discriminating check: without the source it decays at the drag rate); energy returns to
  the equilibrium's after a kick; `--sheets` with a mean field raises.
- `devtools/checkmusic.js`: rectangular fixture rows; the window-to-grid rule (landscape,
  portrait, clamp, each rung); no rebuild on resize; the background select builds the right
  flags and never both; the auto rung drops and never climbs; with everything at "off" the
  legacy legs pass unchanged.
- `devtools/checkkick.js`: blob and sheet kicks on a rectangular grid against the offline
  fixture.
- `devtools/checksolver2d.js`: byte pins unchanged; a leg that `mu ≠ 0` with `eqsrc` still
  throws without `eqsrcDrag` and holds the equilibrium with it.
- S1 and S3 are layout and drawing: gated under the stub for structure (canvas sized from
  the window, drawer present, history length and fade weights of the trace) and judged by
  eye in real Chrome for everything else. The full page-gate suite is run before step 7.

## Risks

- **Sharp may not hold real time** even on the M1. Step 0 answers it before anything is
  built on the assumption.
- **Stretched cells may look soft** along the long axis on standard. The step-1 movies are
  the test; if they do, the choices are light/sharp only (square-ish cells) with a crop.
- **Sheets plus strong kicks may just look like noise**, or the restored sheets may
  dominate the picture and hide the music. Offline judging, two strengths.
- **iPhone Safari**: no fullscreen API, `dvh` and the address bar, and the mic path is
  untested there. The stage must degrade to "fills the visible viewport".
- **Idle-fading controls on touch**: a tap that only wakes the controls must not also
  press Stop.

## Not in this plan

- Recording the overlays and the pulse, and audio in recordings.
- The offline render mode in the browser (fixed 1/30 s of audio per frame at high
  resolution). It would reuse S2 and wants the compositing above; a separate plan.
- Re-gridding on the page (E5 stays offline only) and sustained-sound kicks (E6, declined).
- Both backgrounds at once.

## Decisions for Alfred

1. **The grid ceiling.** The page is 256² today, so 512 × 256 doubles the work. Proposed:
   the three-rung ladder with `auto` starting on standard (256², stretched cells) and sharp
   (512 × 256) as an explicit choice. Alternative: sharp as the default on machines that
   hold it. Step 0's table decides.
2. **Stretched cells or a crop.** Proposed: stretch (the whole box is on screen). The
   alternative keeps cells within 11% of square and crops up to 11% of the long side.
3. **Do the charts leave the page** at step 6, or stay available behind a link/flag?
   Proposed: leave; `rmhd2d.html` is the science view.
4. **Sheet orientation.** Proposed: always varying along x (vertical sheets), which the
   existing source supports as is. Horizontal sheets on a portrait window would need the
   source to read the k_x = 0 row as well — a kernel change, so out by the first rule
   unless Alfred wants it.

**Settled 2026-11-23 (Alfred):** 1 — the ladder as proposed, `auto` starting on standard,
sharp an explicit choice; step 0's table can still move the default. 2 — stretched cells.
3 — the charts leave the page at step 6. 4 — vertical sheets only.

## Parallel build (agreed 2026-11-23) and the layout contract

Alfred agreed to a first wave of three implementers working at once, replacing phase 2's
"one implementer" rule for this phase; the single adversarial review at the end stays.
Streams: **offline** (step 1), **stage shell** (step 4), **traces** (step 5). Step 0 and the
`eqsrcDrag` opt-in are done by the coordinator; step 3 (the page port of S2 and S4) follows
the offline stream and is serial, as are the judging, the integration and the review.

So that the shell and the traces can be built apart and merged:

- **Files.** The shell owns `webgpu/stage.js` and `webgpu/stage.css`; the traces own
  `webgpu/traces.js`. Neither edits `common.js`, `solver2d.js`, `physics.js` or `music.js`.
  Each may add to `music.html` only: its own `<script>`/`<link>` tag, its own control
  group or rows, and a few clearly marked wiring lines (`// STAGE:` / `// TRACES:`).
  Neither changes existing lines' behaviour when its flag is absent.
- **Flags.** The shell is live only with `?stage` in the URL. The traces are live only with
  `?traces` or `?stage`. With neither, the page is the shipped one and `checkmusic.js`'s
  existing legs pass untouched.
- **The traces attach to a host element.** `traces.js` exposes
  `musicTraces.attach(host)`: it creates ONE `<canvas class="traces">`, absolutely
  positioned to fill `host`, `pointer-events: none`, backing size = host's client size
  times `devicePixelRatio`, and re-fits itself on resize (ResizeObserver). Until the shell
  exists the wiring attaches it to the display card's wrap (`cards.disp[0].wrap`); under
  `?stage` the host is `#stage`. The traces never assume which.
- **`#stage`** is the shell's full-window element (`position: fixed; inset: 0`), holding
  the picture. Stacking, low to high: the picture, `canvas.traces`, the control bar and
  drawer. The pulse's CSS transform applies to the picture only; the traces do not scale.
- **Controls.** Both streams add controls through `controlsBuild` groups like every other
  control on the page; the shell moves ALL groups into the drawer generically (by walking
  what `controlsBuild` made), so a group added by anyone else later lands in the drawer
  with no further wiring. The traces' group is `grpTraces`.
- **Trace timing and colour.** Ages are wall-clock (`performance.now()`), passed in as an
  argument wherever they are used so the fade can be tested without a clock. Colour comes
  from the display card's current colormap through whatever lookup the page already has
  for it, re-read when the colormap changes.
- **Pure parts are separate functions** (history push/expiry, fade weights, the log-log
  mapping, the window-to-canvas sizing) so the stub gates can check them without a GPU or
  a DOM.

## Step 0 record (2026-11-23): the grid timings

Real Chrome on the M1, `music.html`'s own solver parameters (kicks, drag, the mean-field
`stage` kernel with a non-zero field), a fresh `Solver` per grid, 300 back-to-back
`step(1)` calls drained once, 6 repeats with the first dropped. Three implementer agents
were running node and python on the CPU at the time; the GPU was otherwise idle.

| grid | ms/step (median, min–max) | steps/s it can do | steps/s the music asks for | headroom |
|---|---|---|---|---|
| 256 × 128 | 0.95 (0.90–0.96) | 1050 | about 165 (half: the short side's cells are twice as big) | 6.4 |
| 256 × 256 | 1.69 (1.63–1.75) | 590 | 330 | 1.8 |
| 512 × 256 | 3.30 (3.08–3.58) | 300 | 330 | **0.9** |
| 512 × 512 | 7.58 (6.75–7.69) | 130 | 660 | 0.2 |

The demand is measured, on the shipped page: click track at 120 bpm, every default, 20 s
after a 5 s warm-up — 330 steps/s at 60 frames/s (5.5 steps per frame). Offline, the Beths
clip at 256² with the walking field asked for about 370 per second of audio, and loud
passages more. The light rung's demand is inferred (the CFL step follows the smaller of the
two spacings, which on that rung is twice the standard one), not measured.

**What it settles.** *Sharp (512 × 256) does not hold real time on the M1*: it falls about
10% short on the click track and further on dense music, before the display pass is
counted. It stays in the ladder as an explicit choice for machines with a faster GPU, and
`auto` never selects it — which is decision 1 as Alfred settled it. Standard has less than
a factor of two in hand on the M1, so phones are expected to need light.

## The `eqsrcDrag` opt-in — record (2026-11-23)

Done by the coordinator. `Solver._dragLin` refuses `mu ≠ 0` with `eqsrc` unless
`eqsrcDrag: true`; the message changed, the line count did not, no WGSL changed.
`checksolver2d.js`'s pinned text for `_dragLin` was edited in place to match (a JS pin, not
a kernel pin). `checkkick.js` leg 6 gained three rows: the opt-in builds; it does not lift
the Pm refusal, is refused when false and is inert alone; the column it builds is the
`mu ≠ 0` one bit for bit. `checkkick`, `checksolver2d` and `checkmusic` pass.

Real GPU (Chrome, M1), the page's own parameters with `b0: false`, drag rate 1,
ψ = 0.3 sech²((x − π)/0.3), no kicks, 1200 steps:

| | t | energy |
|---|---|---|
| start | 0 | 0.0254648 |
| `eqsrc` + `eqsrcDrag` | 15.3 (15 drag times) | 0.0254638 |
| no source | 29.9 | 2.8e-15 |

So the source holds the sheets against the drag to 4 parts in 10⁵, and without it they are
gone. Not yet tried: kicks on top of it, and `setDrag` on a live solver with the source on.

## Build record (2026-11-23): what Alfred changed while watching it

Built by parallel agents (traces, stage shell, offline, then the page port) and judged in
real Chrome on the M1. Changes from the text above, all Alfred's decisions the same day:

- **The stage's box is the SQUARE one by default** (`selBox = square`, 256 × 256, 4 bands).
  The window-shaped box of S2 is built and kept as an option (`selBox = window`), not the
  default: on a 2.3 : 1 window it left 2 bands and 2.3 : 1 cells on the standard rung.
  Decision 2 (stretched cells) stands for that option.
- **The whole box is always shown** (contain, never cover). Where the window is another
  shape, the `sides` control (`selFill`) chooses black (default) or the picture repeated —
  the box is periodic, so the copies join. The copies are drawn on a 2D canvas
  (`#stageTile`) in the same task as the render, by wrapping `renderCards` on this page
  only; a paused page re-renders once when the copies on show are stale. A saved PNG or a
  recording is the single square picture.
- **Traces**: the line cut runs across the top and the total-energy spectrum across the
  bottom, full width, no dashed lines; tick labels (no drawn axes) behind a toggle; the cut
  shows the field the picture shows, fades four times faster, and follows the picture's x
  extent (and its repeats). The spectrum spans four decades. A left margin exists only
  while tick labels are shown.
- **Side panels** (`cbTrSides`, off by default): in the two black strips, four spectra
  (E_u, E_b, E+, E−) on one side and on the other either the same four energies against
  time or four signed line cuts (u_x, b_x, z+_x, z−_x along x = Lx/2, the solver's own
  `readCutLine`) — `selTrSide`. They replace the bottom spectrum while shown; nothing is
  drawn where the sides repeat the picture or a strip is under 90 px.
- **Sheets**: the flux bump (not Harris), strength 5 × amp (slider 0–8), width 2 (1–4);
  choosing them puts `magnetic |b|` on the picture, and the fixed colour scale's reference
  becomes the sheet's peak field / 1.5 (`musicScaleRef`), so the slider's default tops out
  at the peak. The current j was tried and dropped: saturated on the fixed scale, a flat
  midtone on autoscale. Offline cost (256², steps per second of audio against the plain
  page's 371): bump × 3 376, × 5 470, × 8 720; Harris × 3 632. On the page, × 5 on the
  square standard grid held 500–600 steps a second on the M1, in real time.
- **Seen and left alone**: with the default width the square box's sheet is thin
  (a = 0.18) and the kicks tear it into islands within seconds; it regrows on the memory
  time, so at memory 3 s it is mostly gone under a 120 bpm click track. The width slider is
  the remedy.
- **Not looked at**: any phone (WebGPU needs a secure page, so only after a deploy),
  fullscreen, touch.

## The adversarial review (2026-11-24)

One reviewer, reading the whole diff, 156 mutants of its own (143 killed, 10 real gate
holes, 3 equivalent) and a side-by-side run of `main`'s page and this one through the same
harness. No blocker. Fixed:

- **MAJOR — the no-flag page could change its grid.** `selGrid` defaulted to `auto`
  everywhere, so a slow machine would have dropped the shipped page to 128 × 128 at the next
  Start. Without `?stage` the default is now `standard`; `auto` is the stage's default. With
  that, the no-flag page matched `main` control for control, parameter for parameter, kick
  for kick (53) and step for step (1742).
- The auto drop is taken and announced only while the select says auto; the sheet is ONE
  current sheet with return currents, not two (tooltip, comments, offline docstring, S4
  above); the energies in time are sampled by time with their own history cap; readback
  chains clear their busy flag whatever the handler did; on touch an open drawer holds the
  controls up; a window blur ends a waking press; the two flag regexes agree; stale
  comments and README text.
- The ten gate holes are closed (a non-square `stagePicture`, side cuts on a non-square
  grid, a solver retired mid-read, devicePixelRatio 2 for the panels and the cut, the side
  cuts' own fade, the scale probe under the traces' frame hook).

Left: with `?traces` alone, closing the first of two display cards detaches the overlay;
in "repeat" mode on a window above the resolution cap the centre picture is the copy's
resolution; side panels sample even where they cannot be drawn. Unverified by anyone:
fullscreen, touch and the idle fade on a real phone.

## The stage becomes the page (2026-11-24)

Alfred, after the deploy: the bottom spectrum and the top cut are off by default and the
side panels on; each strip has its own select (spectra / line cuts / energies in time, any
pairing); every side spectrum carries the k^(-5/3) guide; and the `?stage` flag is dropped.
The stage is now what `music.html` serves. The card layout was NOT deleted: it is
`music.html?plain`, because seven gate legs test the solver wiring through it (and it is
still the way to see the charts). Decision 3 (the chart cards leave the page) is met for
the page a visitor gets; removing the card layout's code is not done.
