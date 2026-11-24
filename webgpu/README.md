# WebGPU RMHD (2D + 3D)

**This file is repo-facing**: how the thing is built, what the contracts are, why each
piece is the way it is, and how to verify a change. The **user-facing** documentation —
every button and every chart, in plain language — is `docs.html`, and the reading list
that takes a visitor from the demo to the literature is `reading.html`. Both are pages of
the app set itself and both are linked from the `.sub` line of `rmhd2d.html` /
`rmhd3d.html`. Nothing here is linked from those pages. When a control changes,
`docs.html` changes with it; when the *reason* for a control changes, this file does.

Two browser apps over one shared core: `rmhd2d.html` (2D) and `rmhd3d.html` (3D
spectral-z, the `z_spectral=True` path — Alfvén coupling applied exactly via the
closed-form 2×2 wave propagator, no wave CFL; z-slice or three-face cube display,
resolutions 64²×32 to 256²×64 plus the long-box 64²×128 / 64²×256, free Lz,
`z_diss_k` kz⁴ dissipation with auto). `rmhd2d.html` is the front door: `index.html` is an
immediate redirect to it (ONEPAGE_PLAN B), the two apps carry the same 2D/3D tab strip,
and the background essay that used to be the landing page now lives in two places: its
lead paragraphs are the "what is all this?" pane right under the subtitle (open until
dismissed once — remembered like the params toggle), and the five background panes are
the rail beside the canvas — intro, tabs, rail and no-WebGPU poster all built by
`chromeBuild` in `common.js`, so every piece of that text exists once. The preconfigured runs are presets *inside* each
app (dropdown, or `?demo=` as a deep link), not separate pages. The 3D contract is `SPEC3D.md`; its reference vectors are
`refvectors3d.json` from `gen_refvectors3d.py` (16²×8, fp64, including a dedicated
exp(L·τ) propagator vector). The rest of this README describes the 2D app; everything
carries over to 3D except where SPEC3D.md says otherwise.

## 2D app

A browser-based port of the repo's 2D RMHD solver: pseudospectral (numpy-rfft2
layout), LSRK33 integrating-factor stepper, optional elsasser Ornstein–Uhlenbeck forcing
with per-step power normalization, adaptive CFL dt with `cfl_every`-style blocks. fp32
(WebGPU has no f64). All physics runs in WGSL compute shaders, including a
workgroup-shared-memory Stockham FFT; the only CPU work per step is drawing the
shell-restricted OU noise (~12 modes), and under the spacebar-blob forcing below there is
none at all — its blob list is uploaded once a frame, from `setBlobs`.

## Run

Open `rmhd2d.html` (or `rmhd3d.html`, or `index.html`, which redirects to the 2D app) in a
WebGPU-capable browser (Chrome, Firefox, Safari — 2026 versions all ship it). Works from
`file://`, no server or build step: `<script src>` and `<link rel=stylesheet>` are allowed
there, which is why the reference vectors stay *inlined* in each app (`fetch` is not). If
no adapter is found the page says so and shows `poster.png` — a still of a real run — in
place of the live one, with the what-is rail opened. Every visit boots **paused** on its
preset (plain visits get the default forced-turbulence one at 256²): the big green **Run**
in the top bar is the call to action, and turns into a red **Pause** while running.

## Self-test

The **Self-test** button runs the 32² reference vectors (inlined from
`refvectors.json`, generated at fp64 by `gen_refvectors.py` against `taranis` itself)
through the same GPU pipelines as production: FFT layout, static grids, nonlinear term,
forcing term + normalization scales, energy, CFL dt, and one full deterministic LSRK33
step (recorded state → recorded dt and scales → compare), plus a statistical
injection-rate check of the OU path. Thresholds are fp32-appropriate (1e-5…5e-4
relative L2; 20% on the statistical check). The 3D page adds the exp(L·τ) propagator,
the forcing envelope on the kz = ±2π/Lz planes, and two energy-budget rows for the
spectra.

Two further rows run at the resolution the page is **currently set to**, not at the
reference grid: **forward transform at `<nx>×<ny>[×<nz>]`, analytic** and **roundtrip at
`<res>`**. The recorded vectors exercise a 32-point (2D) and an 8/16-point (3D) transform
line only; these check the shipped FFT at the line lengths production actually uses, which
is what any rewrite of `fftKernel` has to stay green against. There is no reference file:
`fftAnalyticCase` (common.js) fills real space with three cosines at fixed integer
wavenumbers (|k| ≤ 3, every k_y > 0, none at a Nyquist — so each lands in exactly one stored
rfft bin on every grid the pages offer) and computes in float64 the unnormalized spectrum
they must produce: `amp·nr/2·exp(i·phase)` in three bins, exact zeros elsewhere. The rows go
through the production pipelines (the nonlinear term's transform chain) on the LIVE solver —
`realNL`/`nlk` are RHS scratch the next step rewrites, so the paused run keeps its state —
at the same 1e-5 tolerance; the reference's own f32 quantization puts a floor of ~2e-8
under the comparison before the transform's rounding adds to it.

To regenerate the vectors after changing the physics:

```
PYTHONPATH=<repo root> TARANIS_PRECISION=64 python3 webgpu/gen_refvectors.py
```

then re-inline the JSON into `rmhd2d.html` (it sits on a single marked line — splice it
programmatically; do not try to hand-edit a 180 kB line).

## Benchmark (`?bench`)

Adding `?bench` to either app's URL grows a **bench** panel under the readout — a developer
entry point like `?recdebug`, not a control: without the flag the panel does not exist and
`window.bench` is undefined. It answers the one question the app's own `steps/s` cannot:
*what* a step's time is spent on (FFTPERF_PLAN, `plans-webgpu/`).

Each button runs one campaign and appends one JSON record to the textarea (phones have no
console — copy-paste is the transport); the same functions are on `window.bench` (`whole`,
`kernels`, `ladder`, `chains` on 3D, `gradsHash`, `all`, with `cfg` for K / R / reps). Every cell is
timed the same way: submit the batch, `await onSubmittedWorkDone()`, `performance.now()`
around it, R+1 reps with the first discarded, median and min reported. The frame loop is held
off while a campaign runs, so nothing renders or reads back inside a timed window; there is no
controller and no `timestamp-query`.

- **whole step** — K back-to-back `solver.step` calls at the panel's `cfl_every`, drained
  once at the end. K submits, not one: the per-step encode and the OU noise draw are inside
  the timed region, exactly as in the frame loop — read it as a step *as the app runs it*,
  not as pure GPU time. The record carries ms/step and, from the page's own dispatch list,
  the buffer bytes one step binds and reads/writes (the table in FFTPERF_PLAN Appendix A) as
  GB/s, and butterflies as G butterflies/s.
- **per kernel** — each kernel alone: `reps` dispatches of one pipeline in one compute
  pass, on the solver's own buffers at the dispatch shape the step gives it (FFT kernels at
  their own chain's lane count: the nonlinear chain's 2; the gradient chain's 8 on 2D, and
  on 3D one two-lane chunk — four chunks per stage, so ×12 per step), µs per dispatch.
- **FFT ladder** — each FFT kernel three ways: the shipped kernel; one with the `cos`/`sin`
  twiddle lines replaced by constants (`consttw`); one that keeps only the load and the
  store (`copy`). Read them as `T_mem` = copy (memory traffic, strided reads included),
  `T_bf` = consttw − copy (butterflies, barriers, workgroup-memory round trips) and
  `T_tw` = full − consttw (the transcendentals); `1.0·x` folds under any compiler, so `T_bf`
  is a lower bound and `T_tw` an upper one. A share is a share of that cell on that device —
  never compare against another device's.
- **grad chain** (3D) — the whole gradient chain as the RHS encodes it: four per-pair
  `prepGrads` dispatches, each followed by its own three inverse passes at two lanes (the
  cell FFTPERF_PLAN item C is timed on).
- **grads hash** — times nothing. It re-applies the page's selected initial condition,
  encodes one gradient chain (the dispatches the RHS makes) and reads `realGrads` back,
  reporting a 32-bit FNV-1a digest of each of the eight real-space lanes and one over the
  whole buffer, plus the byte length. Same page, IC, resolution and device ⇒ same digests,
  so a change to the chain that is meant to be bitwise is shown to be bitwise by comparing
  two records. The digest is over the buffer's `Uint32Array` view, low byte first by shift,
  so it does not depend on the device's endianness. It re-applies the IC first, not last,
  and writes only the gradient scratch.

A campaign pauses the run; the per-kernel, ladder and chain campaigns leave the buffers as
garbage (kernels run out of order on the live state), so each ends by re-applying the
page's selected initial condition — fields, forcing state and scalars. One variant per page load, nothing
else running, plugged in.

## Files

- `index.html` — an immediate redirect to `rmhd2d.html` (meta refresh + `location.replace`
  + a plain link, `rel=canonical`). It exists because it is the URL GitHub Pages serves
  for `webgpu/` and the one old links point at; there is no landing page any more. Note
  that `pages.yml` seds this file too, so its `style.css` reference and its `buildid` span
  have to stay.
- `poster.png` — a 512² still of a real forced-turbulence run (|u|, afmhot, arrows,
  colorbar), shown *only* where `initGPU` fails. The `<img>` is created on that path and
  nowhere else, so a working visit never fetches it.
- `docs.html` — the user-facing manual: the topbar, every control group, the display and
  chart cards and their options, the colorbar, save/record, and one plain line per preset.
  Same stylesheet, no script, no build step. It is the page the apps link at the top; the
  accuracy rule when editing it is that every control named there must exist by id in the
  built panel (`controlsBuild`'s spec fragments and each app's own spec).
- `reading.html` — the annotated reading list: four "basics" sections (RMHD's origins,
  plasma physics, turbulence, heliophysics) then Alfred's own tour of the arguments the
  field has had, one `<details>` per thread. Same skeleton and stylesheet as `docs.html`,
  no script, no build step; linked from the `.sub` line of both apps and of `docs.html`.
  It is the ONLY page with outbound links, so two rules apply when editing it: every
  external `<a>` carries `target="_blank" rel="noopener"`, and every citation is a
  publisher DOI where one exists (arXiv only as a fallback). Its handful of page-specific
  rules (`h2`, `ul.refs`, `.todo`) live in a `<style>` block in the file rather than in
  `style.css`, which stays owned by the apps.
- `rmhd2d.html`, `rmhd3d.html` — the two apps. Each holds ONLY its inlined reference
  vectors and its UI layout and defaults, on top of `<script src="common.js">` and
  `<script src="physics.js">` (in that order). 3D also holds its own dimension-specific
  WGSL and `Solver`; 2D's are `solver2d.js`, which it loads third.
- `common.js` — the shared pieces that carry no equation: RNG, reference-vector
  flatteners, the FFT kernel template and the **rfft row pair** (`fftRowPair`: the y
  forward/inverse kernels, which both pages emitted as byte-identical twins) with the
  `?bench` harness's `probe` emissions and the harness itself (timing, the ladder, the
  panel; each page keeps its own bench spec — which kernels, over which buffers, at which
  dispatch shape — next to its solver), the analytic transform case of the self-test, the generic
  reductions (CFL, energy tail,
  max-reduce), device bring-up and the **pooled readback** (`readBuf`, whose staging
  buffers are keyed by byte length and reused rather than allocated per call, plus
  `readBufOnce` for the megabyte-sized reads that must not be pooled), the
  **colormap table** (`CMAP_COEF` / `cmapRGB`, the one
  source both the WGSL and the editor preview read), the **display quantity table**
  (`DISP_FIELDS`, physics.js's modes with their definitions — one table, both pages), the
  chart + overlay drawing (energy
  trace, spectra, cut trace, arrows), the **card system** (`DisplayCard`, `Recorder`,
  `ChartCard`,
  `cardsInit`/`cardsLayout`/`cardsSync`), the **control-panel builder**
  (`controlsBuild(spec)` and the row/group fragments the two pages share — the sticky
  top bar, the cfl row, the hyper/diss row, the whole forcing group, the IC group and
  the displays group; each app's page markup is now an empty `#topbar` / `#controls`
  plus its own spec), the **page chrome** (`chromeBuild`: the 2D/3D tab strip, whose
  inactive side is a plain link, the `RAIL_LEAD` / `RAIL_PANES` text of the what-is rail,
  and `gpuFallback`, the poster + opened rail an engine with no WebGPU gets), the preset
  machinery (`presetBoot`,
  `presetWrite`) and the boot wiring both apps share (`bootApply`,
  `wireCommonControls`, the locked slider pairs,
  `syncCommonLabels`), the CPU-side IC
  construction (glyph raster, periodic gaussian blur, the ζ± → (φ,ψ) normalization
  `icZetaFields`, gaussian z-envelope, packet placement + χ, and the whole `custom` blob
  editor — deposit math, its own **view**, preview, pointer→grid mapping, driven by one
  per-app hook object through `icDrawWire`), the z-plane trackers (`trackCentroid`,
  `trackArgmax`), the self-test table and the frame loop (with two per-app hooks,
  `frameHook` and `readoutExtra`).

  **The render gate** (2026-08-12). A display chain runs only when its picture can have
  changed: `renderCards(paused)` draws a card when `DisplayCard.needsRender()` says so.
  That asks three things — has the STATE moved (`seenMark !== stateMark()`, so a step, an
  IC upload, a preset and a rebuild all invalidate by construction and no caller has to
  know this gate exists), has this CARD moved (`dirty`, set by `apply()` and `_resize()`,
  i.e. by the controls that change no state), and is a take reading frames off it. Before this, a PAUSED page re-ran every chain at rAF
  rate forever — in 3D a full volume inverse transform per card per frame, two more per
  active contour set, and in the volume view the whole 512²×256 raymarch — over a state
  that could not move. The readbacks that read what a render left behind (arrows, the
  colorbar autoscale) are gated on the card's own frame counter; the ones that read the
  SOLVER (stats, the cut line, the spectra, and 3D's plane tracker and field-line march)
  are gated on `stateMark()`, the pair `(nsteps, stateSeq)`. `stateSeq` is what catches a
  state that jumped without a step — an IC upload resets `nsteps` to 0 — and is bumped by
  `chartsReset`, which both apps already call on exactly those paths. Each gate still lets
  ONE readback through after the last step, so a paused page shows the state you paused
  on, not the one 300 ms before it, and `cardsThrottleReset` clears the markers so a chart
  card added to a still page is filled rather than told "nothing new".

  The rule the 3D hooks got wrong first time, and the one to keep in mind when adding a
  fourth consumer of anything gated here: **a marker may only ever suppress a REPEAT.**
  `fieldLineHook` shipped gating on "the state has not moved AND something is cached",
  which is true as soon as ANY of its three consumers has been served — so a second one
  appearing over a still state (and both pages boot paused, so that is the ordinary case)
  was locked out for good, leaving the k∥ curve empty and a second lines card blank. It
  now asks `flStale()`: is anybody asking who has nothing? `planeTrackHook` asks its
  drift test the same way, before the early-out rather than after it.

  Three more things the gate has to get right, all pinned by `devtools/checkidle.js`: a
  live recording forces a draw (the encoder reads the texture that render produced, and a
  skipped frame would hand it an expired one — RECRAF), and a paused frame *settles* each
  active contour set's adapting range instead of relaxing it (`contSettle` writes the zero
  range that `contLevel` reads as "no history", so the last frame drawn takes the measured
  max outright). The relaxation exists to damp flicker on a moving field; on a still one it
  is a spacing that would never arrive, frozen wherever the gate stopped the frames. And
  leaving the IC editor marks every card, because save and cancel change no state at all
  and the cards have been detached for the length of the edit.
- `physics.js` — the shared RMHD kernels and the whole slice display chain, as
  templates over one constants object per app (`{pre, hasZ, wgReal, nDisp, arrow, ns,
  envFn}`), plus the shared `struct Mode` and `CMAP_WGSL` (four colormaps, one
  implementation, expanded from common.js's coefficient table at emit time):
  `prepGrads`, `bracket`, `nlAssemble`, `energyPartial`, `ou`,
  `scale`, `blobBuild` (templated here beside the forcing it replaces, though only the 2D
  solver instantiates it), `icFinish`, `prepDisp`, `vecMag`/`vecMagSq`, `maxSumPartial`,
  `sigmaCombine`, the arrow gather, the cut card's `cutPrep`, the colorize (whose shading
  — value→colour plus the contour overlay — is one fragment the 3D cube faces share),
  the contour level table and the blit (the 3D-only
  `zpPrep`/`planeEnergy` tracking kernels stay in the 3D app). Every 2D/3D
  difference is derived inside the templates from the single flag `hasZ` (the 3D index
  split m = iz·NMP + mp, the kz dealias factor, the kz⁴ term in the linear diagonal),
  so the apps hand over sizes, not code. Deliberately NOT shared, and still per-app:
  `stage` (diagonal exponential vs the 2×2 Alfvén propagator), `forcingAdd`/`envExpand`,
  the spectra, and the 3D `sliceExtract`/`faceExtract`/cube path — plus `makeGrid` and
  the `Solver` classes, which are per-DIMENSION (2D's live in `solver2d.js`, 3D's in its
  own page). The refactor that moved this out of the apps was gated on the
  generated WGSL staying byte-identical at every resolution preset in both apps.
- `solver2d.js` — the 2D solver, loaded as a third classic script after `physics.js`:
  `makeGrid` (the CPU-side k grids), `buildShaders` (the 2D-specific WGSL, on top of
  physics.js's shared templates) and the 2D `Solver` class (buffers, pipelines, the
  LSRK33 step, forcing, the diagnostic readbacks and the display chains). It is loadable by
  a second page, so it may use common.js and physics.js and nothing
  from a page's inline script — `devtools/names.mjs` checks that.
- `style.css` — the shared dark theme.
- `SPEC.md`, `SPEC3D.md` — the implementation contracts, extracted from the JAX source;
  read these before touching the math.
- `music.html`, `music.js` — the music page (MUSIC_PLAN M2; see "The music page"): a
  third page on the shared framework, loading `common.js`, `physics.js`, `solver2d.js`
  and then `music.js`. `music.js` is PURE — the band plan, the onset detector, the blobs,
  the click track and the audio clock, with no DOM, Web Audio or WebGPU in it — and the
  page's inline script is the audio and DOM glue. **Deployed since 2026-11-21** (`pages.yml` requires it in the staged
  site and cache-busts `music.js` with the rest), but linked from nowhere yet and carrying
  no analytics beacon.
- `stage.js`, `stage.css`, `traces.js` — the music page's STAGE layout and its fading-trace
  overlays (plans-webgpu/MUSIC_STAGE_PLAN.md; see "The music page: the stage"). Loaded by
  `music.html` only. The stage is the page's layout since 2026-11-24; with `?plain` (the
  card layout of phases 1 and 2, which the gates and the charts use) every stage entry point
  returns at its first line and no CSS rule matches, and the traces are then on only with
  `?traces`. Deployed and
  cache-busted by `pages.yml` with the rest.
- `gen_musicfixture.py` — records `devtools/fixtures/music_ref.json`, the offline
  `band_plan`'s and `CausalOnsetDetector`'s own output for `devtools/checkmusic.js`.
  Pruned from the deployed site with the other `*.py`.
- `gen_musiccmap.py` — fits the music page's `magma` to matplotlib's table and writes
  `devtools/fixtures/music_cmap.json` (the table, the coefficients, the fit error) for
  `devtools/checkmusic.js`; `brat` is exact. Pruned from the deployed site like the others.
- `gen_kickfixture.py` — records `devtools/fixtures/kick_offline.json`, the offline
  `make_kick`'s own output for `devtools/checkkick.js` (see "The impulsive kick and the
  drag"). Pruned from the deployed site with the other `*.py`.
- `gen_refvectors.py`, `refvectors.json`, `gen_refvectors3d.py`, `refvectors3d.json` —
  reference-vector generators and their output.
- `devtools/` — the sandbox verification tooling (no GPU needed): the shared DOM+WebGPU
  stub every tool boots a real page on, the WGSL byte-diff harness, the fp64 physics and
  arithmetic checks, the clone detector and the layout audit. `devtools/README.md` says
  what each one covers; the standing rule is that a phase captures a FRESH WGSL baseline
  from clean git state before editing and diffs against it at the end.

## Cards: displays and charts

The UI is built from **cards**. A *display card* is one WebGPU canvas plus its own
quantity selector, colormap, arrows checkbox, contour-overlay selectors, in 2D its two
periodic x/y offset sliders (below) and — in 3D — its
own z slice / view select (manual slider, `track z⁺ / z⁻`, the same three with the
cube-faces view, or the field-lines view). Card index *is* the solver's display-chain
index, so N cards cost exactly N chains (scratch, bind groups, gather targets, textures),
each built lazily on first use.
*Any* card can be closed down to the last one (whose × goes disabled) — the IC editor has
its own view and anchors nothing. A *chart card* is one 2D canvas with a type selector — energy trace / spectra / cut — plus that
type's own options; several cards of the same type are
allowed and share one throttled readback per frame. Nothing here is a special case of
anything else: there is no dual-view flag and no fixed chart stack. (What each control
*does* is `docs.html`; below is why it is built this way.)

**Chart options** (each is a small select in the card's header; the first value of each
is the default and is what the chart drew before they existed):

| chart | option | values |
| --- | --- | --- |
| energy trace | which energies | `E_u, E_b` (with E_tot) · `E⁺, E⁻` (with E_tot) |
| spectra | which spectra | `E_u, E_b` · `E⁺, E⁻` · both |
| spectra (3D) | direction | ⊥ + ∥ · ⊥ only · ∥ only · ⊥ + k∥ (field line) |
| spectra | fit line | `pin to field` · `set A` · `off`, plus a numeric index `p` and amplitude `A` |
| spectra | pin / unpin | buttons, not selects: freeze the drawn curves as ghosts, or clear them |
| cut trace | component pair | `u_x, u_y` · `b_x, b_y` · `|z⁺|, |z⁻|` |
| cut trace (3D) | z source | manual slider · track z⁺ · track z⁻ |
| anisotropy (3D) | y axis | `k∥/k⊥` only while `ANISO_CHI_UI` is off; the χ ordinate is flag-gated (see below). The rest of its options — `aq`, `ad`, the fit trio — are ANISO_PLAN's |
| island width (2D) | — | log W(t) with a fitted γ = 2·d(ln W)/dt; needs the tearing IC (see below) |
| k_y = 2π/L_y mode (2D) | — | log A(t) of u_x / b_x at that k_y; needs the KH IC (see below) |
| eigenfunction (2D) | k_y bin | `1`…`9`, the box fundamental first; picks which column is read back |
| eigenfunction (2D) | fields | `ψ + φ` · `ψ only` · `φ only` |

**The spectrum fit line** (FEEDBACK item 8) is a straight E = A k<sup>p</sup> from just
above the forcing shell to the last bin, per CHART CARD, so two cards can carry two
slopes over the same spectrum. `p` defaults to −5/3 (the index box holds decimals, and a
value within `FIT_SNAP` of −5/3 or −3/2 snaps to the exact fraction and is *labelled* as
it, so the legend reads `k^-5/3` / `k^-3/2` and anything else reads as the number it is).
`pin to field` takes A off the spectrum itself, exactly as the old fixed guide did;
`set A` uses the amplitude box (A at k = 1) and falls back to the pinned anchor while the
box is empty; `off` hides the line and both boxes. NaN in either box is ignored.

**Pinned ghost spectra.** `pin` freezes the curves the spectrum card is *drawing* as faint
ghosts under the live ones, so the next thing you do can be compared against the last
thing you did (the workflows it exists for are in `docs.html`). It costs nothing on the
GPU — the snapshot is the CPU-side `[points, colour, dash, label]` list `specCurves` just
built, deep-copied.

Because it is the *drawn* curves that are frozen, a pin taken as `E⁺ / E⁻` stays E⁺/E⁻
however the card's own `sq` / `sd` selectors are switched afterwards: the ghost is a record
of a moment, not a live view. Details:

- **Per card**, like the fit line, and at most **4**; a fifth press is refused. `unpin`
  appears only when there is something to clear and clears all of them in one press
  (pins are cheap to retake).
- **Physical k.** A pin remembers the `kunit` it was taken under, and every ghost x is
  redrawn at `kunit_pin / kunit_live`, so changing the box (standard ↔ wide 4π × 2π) leaves
  the ghost at the same physical k instead of the same bin. An unchanged box is a factor of
  exactly 1. Resolution changes need nothing — the axis is log k/kunit out to the live
  `nb` and the clip rect crops any ghost tail beyond it.
- **Range.** Pinned ⊥ curves join the y-range pool and the `specFloor` knee walk exactly as
  live ⊥ curves do (otherwise the comparison can sit off the axis); pinned ∥ curves never
  stretch it, the same rule live ∥ obeys.
- **Drawing.** Under the live curves and under the fit line, at `lineWidth` 1 and
  `globalAlpha` 0.45 / 0.34 / 0.26 / 0.20 by age, each keeping its own hue and dash (so a
  pinned parallel spectrum stays dashed). Forcing markers and the fit line are not pinned.
  The legend gets **one** collapsed entry, `2 pinned @t=8.1, 12.3` — a 3D card on both × both
  draws eight curves, so one entry per pinned curve would cost eight legend items per pin.
- **Lifetime.** Pins survive parameter changes, IC resets, `chartsReset`, pause and
  rebuilds, and a preset switch transplants them onto the incoming spectrum cards
  positionally (`cardsLayout`) — which is what makes "pin the decayed spectrum, pick the
  forced preset" work. They are cleared by `unpin`, by retyping the card, and by closing
  it. Nothing is persisted across a reload. A card with ghosts but no live data yet draws
  axes and ghosts instead of `spectra — waiting…`.

The Elsasser energies are E<sup>±</sup> = E_kin + E_mag ± H_c with the cross helicity
H_c = ⟨u·b⟩, i.e. E<sup>±</sup> = ½⟨|z<sup>±</sup>|²⟩ and E_tot = (E⁺+E⁻)/2 — the
convention `taranis/physics/rmhd.py` uses for the forcing powers, which is what lets all
three curves share one axis. H_c costs nothing extra: it rides in the fourth (previously
zero) lane of `energyPartial`'s vec4 accumulator, and the spectra kernels bin it as a
third lane, so E<sup>±</sup>(k) = E_u(k) + E_b(k) ± H_c(k) needs no second kernel.

The **preset** dropdown in the sticky top bar (Run / Reset / t·step·dt·steps-per-second)
picks a whole configuration — controls *and* card layout. `?demo=NAME` is the same thing
as a deep link. Controls live in collapsible `<details>` groups (simulation /
dissipation / forcing / initial condition / displays & charts), open by default on a wide
viewport and collapsed on a narrow one; the state is deliberately not persisted.

## Display modes

Both apps: vorticity / current / phi / psi as signed fields (symmetric ±max), and the
perpendicular vector magnitudes "velocity |u|", "magnetic |b|" and the two Elsasser
fields "|z⁺|", "|z⁻|" (z± = φ±ψ, displayed as |ẑ×∇z±|), each with an optional ≤32×32
arrow overlay. Grid row y=0 is drawn at the top of the canvas.

### The display offset (2D): sliding the picture, not the box

Each 2D display card carries two sliders, `offset x` / `offset y`, in **box fractions** over
±0.5 (which is every distinct offset there is: ±L/2 are the same picture). They exist so that
a structure sitting across a box edge — a current sheet, an island, an arriving packet — can
be brought to the middle of the card and looked at. `common.js: DisplayCard.offset()` reads
them; the app multiplies by its own box and `modeWords` writes `[Sx, Sy]` in **length** units.

It is one multiply in `prepDisp`, on the display chain's way out: the translation phase
e<sup>−i k·S</sup>, so the card draws f(x − S). The ⊥ box is periodic and the display's
inverse transform is the same one the state uses, so this is an **exact roll** of the same
field at any offset — nothing is cropped, interpolated, or wrapped in from a neighbouring
copy. The words ride in the two the k⊥ band left as padding (`MODE_SHIFT_STRUCT` renames
`bpad` to `sh`; a `vec2<f32>` aligns to 8, so the offsets are unmoved and every Mode uniform
is still `MODE_BYTES`), and they ride *every* uniform the chain preps a field through — the
card's own, the sigma mate, both contour potentials — so the contours and the second σ half
are rolled *with* the field and never against it. The arrow overlay needs nothing: it gathers
the real-space buffer the shifted chain just wrote, at the same anchors. The colorbar range is
unchanged by construction, a roll being a permutation of the same values.

The phase is evaluated in fp32 at up to |k·S| ~ 1600 rad (512², both sliders at their stop),
where the argument itself carries ~1e-4 rad of rounding. Measured against an exact roll that
is 5e-5 on a field of rms 0.59 — a hundredth of one 8-bit colour step, so it is a rounding
statement and not a visible one.

**Off is bitwise off**, the band's rule: at `[0, 0]` the whole `if` is skipped, so the
unshifted picture is unshifted bit for bit. And **only the picture moves**: the state, the
time step, the energies, the spectra, the island / mode / eigenfunction cards and the cut
trace all read the unshifted field — the cut still cuts the real x = L<sub>x</sub>/2 (which is
what makes `e^{i k_x L_x/2} = (−1)^{ix}` in `cutPrep` true, and is where the resonant surface
of the tearing and KH equilibria actually is). Saving or recording an offset card saves the
offset picture, as with the filter. The caption says `offset: (+0.25, −0.10)L` while it is
active, because a picture that no longer shares the box's origin must say so.

Two deliberate absences. It is **2D only** — `cardsInit`'s `cfg.offset` builds the sliders and
`buildShaders`'s `shift` emits the phase, and the 3D app sets neither: a 3D card's picture may
be the whole box (cube faces, field lines, the volume march), and "slide the picture" is a
statement about a plane. The 3D `prepDisp` is byte-identical to the pre-offset text, the `sigR`
/ `band` gating pattern. And there is **no preset key** (`addDisplayCard`'s state bag has none):
a preset author who wants the structure centred shifts the IC, which is free on a periodic box
and moves everything — charts included — instead of only the picture.

One reality caveat, for anyone who takes the phase somewhere else: at an offset that is not a
whole number of cells, e<sup>−i k·S</sup> breaks the rfft2 reality constraint at the single
self-conjugate points (k_x-Nyquist and k_y-Nyquist), which is why an offset field is only a
real field to begin with because the 2/3 dealias holds those modes at *exactly* zero — the IC
is masked (`run.py::initialize`) and so is every nonlinear term, so there is nothing there to
misrepresent. A path that fed an unmasked field through this phase would need the mirrors
handled explicitly.

The **cut trace** chart card is self-contained: it plots its own selected pair of
components along y at x = Lx/2 (of its own z plane in 3D — manual or tracked), and does
not depend on which display cards exist. It prepares its own line rather than gathering
a row out of a display chain, and does it without a 2D transform: e<sup>i k_x L_x/2</sup>
is just (−1)<sup>ix</sup>, so the k_x sum (and in 3D the k_z sum, with the plane's phase)
is analytic in one small kernel (`cutPrep`), and only the inverse along y is a transform
— four rows through the existing `rowsC2R`. The kernel always emits all four of
(u_x, u_y, b_x, b_y), so the pair selector, |z<sup>±</sup>| = |u ± b| included, is pure
CPU arithmetic on 4·ny numbers. Signed pairs are autoscaled symmetrically, the
magnitude pair to [0,max].

The **eigenfunction** chart card (2D only, EIGF_PLAN) is that card's transpose, and
the only chart with a readback of its own. It plots |ψ̂(x, k_y)| and |φ̂(x, k_y)| against x
at one selected k_y, on a **linear, autoscaled** y axis — the mode grows exponentially, so
the shape is the content, and a linear axis also makes |φ̂| = 0 on the resonant surface a
zero rather than a −∞ spike needing a floor clamp. The path is deliberately the cheapest
one that exists:

- `eigfGather` (`physics.js`) is a **pure gather**: the strided column m = ix·NKY + j₀ of
  φ and ψ into a compact 2·NX complex buffer (4 kB at nx = 256), one dispatch, no
  arithmetic, `fields` bound read-only, j₀ from a 16-byte uniform so the k_y selector costs
  a `writeBuffer` rather than a pipeline. It exists only on the 2D page.
- the inverse along **kx** is CPU work on those few kB (`common.js: eigfProfile`), through
  the same `fftPow2` the field-line spectrum uses — nx is a power of two in every box. The
  1/(nx·ny) is the full inverse-rfft2 normalization, so what is plotted is the coefficient
  c_j(x) of ψ = Σ_j c_j(x) e^{i k_y y}.
- `src: "eigf"` is a third throttled readback in the frame loop, on the cut line's idiom
  (100 ms, keyed on `stateMark()`), with the card's **k_y bin** where the cut card has its
  z plane: one gather per distinct bin on screen, so two cards on the same k_y cost one
  round trip and two on different ones cost two.

Two things it does *not* do. It never subtracts an equilibrium: every equilibrium seed here
is y-independent, so it lives entirely in the k_y = 0 column (the same fact `srcInit` uses
to read ψ_eq,k out of that column), and any other column has it removed exactly and for
free — and the column is the LIVE one, so once the run is nonlinear the profile is measured
against the state's own mean profile, flattened current and Reynolds-driven mean flow
included. And it quotes no Δ′: a legend readout fitting ψ̂′/ψ̂ either side of x₀ was drafted
and dropped in review (window-dependent, and not interesting enough at this resolution to
earn a legend slot) — the card is the plot alone. What is drawn is the **outer** solution;
the resistive layer is ~1 cell wide at the shipped presets (δ_lin/dx = 1.03 on `tearing`),
which is why neither the hint nor `docs.html` promises a layer.

**Colormaps** are per display card: `afmhot` (default), `viridis`, `RdBu`, `grayscale`.
One WGSL `cmap(x, which)` serves the slice colorize and the 3D cube colorize; afmhot is
matplotlib's exact closed form, viridis and RdBu are degree-6 least-squares fits of each
channel to matplotlib 3.10's tables sampled at 256 points (max abs channel error 0.017
and 0.040; both clamped, since a fit overshoots slightly at the ends). The coefficients
live once, in `common.js: CMAP_COEF` — `physics.js` expands them into the WGSL at emit
time and the IC editor's CPU preview reads the same table through `cmapRGB`, so the two
cannot drift.

**The colorbar** (FEEDBACK_2026-08-10 item 12) is a third consumer of that same table, and
deliberately not a fourth code path: the STRIP is always the full colormap swept over
t ∈ [0,1], because that is exactly what `dispX` produces for every mode (signed −s…+s →
0…1, magnitude 0…s → 0…1, σ_c/σ_r −1…+1 → 0…1), so one 132×9 2D canvas painted through
`cmapRGB` serves all ten modes and **no WGSL changes** — the gate on the whole feature was
both dumps staying byte-identical. Only the three LABELS branch, on the same three
classes. It lives at the right of the card's caption line (`.viewfoot`, which wraps), so
it never covers the field canvas, and it is hidden in the 3D lines view, which renders no
colour field at all.

The number it quotes is the one the kernel used, not a CPU estimate of it: the autoscale
lives only in the per-chain `maxVal` buffer that `maxFinal` writes every render (the same
buffer in both apps, and the three-face maximum in the cube view — which is what those
pixels were divided by too), and no existing readback carries it, `readStats` being
energies rather than extrema. So the frame loop takes its own, on the arrow readback's
snapshot-and-guard idiom: 4 bytes per card at `CBAR_PERIOD` = 350 ms, an order of
magnitude cheaper than the 8 kB arrow gather beside it, skipped entirely for the σ modes
(fixed ±1, no autoscale exists) and while the editor view owns the screen. A field change
drops the stale range rather than relabelling the new mode with it.

**Save / record** (item 13) are per display card, in the same footer. `save` composites
the WebGPU canvas, the overlay canvas and the colorbar into an offscreen 2D canvas and
hands it to `toBlob`; the card **re-renders first**, because WebGPU has no
`preserveDrawingBuffer` — `getCurrentTexture` is transient and the canvas holds only its
last *presented* image, so the capture has to be taken in the same task as a fresh
present.

**Capture and delivery are separate steps, and the result strip is shared** (IO_PLAN item
2). Three contracts, in the order a later export feature meets them:

- `captureShot()` on **both** card classes returns a `Promise<Blob>` and delivers nothing.
  A display card's initiates its render and its composite **synchronously** (no `await`
  before `toBlob` — the texture is transient), so N cards can be captured in one task and
  only then awaited; that is what "save all as one archive" needs and why the split exists
  before there is a second caller. `saveShot()` is `captureShot()` + one `recResult`, and is
  the only thing the per-card button calls.
- **Nothing hands a finished file to `dlBlob`.** Every file — both recording legs, the
  display card's picture and now the chart card's — goes through `cardResult(card, kind,
  blob, name, seconds)`, which parks it on the card's footer behind size/length text and a
  download (plus `share`, where the engine can share files) button, because on a phone a
  silent download lands in Files and is hard to send on. `dlBlob` is reached from exactly two
  places: that strip's own buttons, and `cardResult`'s dead-card branch — a card closed
  between the press and the deferred `toBlob` downloads its file rather than losing it, which
  is why `destroy()` sets `dead` on both classes.
- The strip's host is `card.foot` and it is **always present on both classes**, with the two
  slots in `card.resEl` (`png` / `video`: a save replaces only the last save). On a chart
  card the footer is the constructor's and `_barBuild` rebuilds only the colour scale
  *inside* it — it used to own the whole footer, which would have taken a waiting file with
  it on every retype. `cardResult`/`cardResClear` are free functions for that reason: two
  card classes, one implementation.

A chart card's `save` is on its header row (it has no caption line) and needs no re-render —
a chart is a persistent 2D canvas, sized by `chartCtx` at `dpr`, so `toBlob` already yields
the physical-pixel image. The exception is a type declaring `bar` (today `gen2d`): its colour
scale is a *second* canvas, so the capture composites plot + scale onto one canvas through
`cbarDraw`, the same shared geometry `DisplayCard.barStamp` stamps with. Names come from
`chartName(kind, ext)` beside `shotName(mode, ext)`, both over one `capName` —
`taranis-<app>-<what>-t<simT>.<ext>`. Gate: `devtools/checkchartsave.js`.

**One archive, not N downloads** (IO_PLAN item 3). `save all`, beside `+ display` /
`+ chart` in the displays & charts group, writes every display card's composite PNG, every
chart card's PNG and a `params.json` into one ZIP. Firing several `<a download>` clicks in
a row raises Chrome's multi-download prompt and is silent on iOS — the exact failure the
result strip exists to avoid, once per card. Three contracts:

- `zipStore(members) -> Blob`, `members: [{ name, data: Uint8Array }]`, **member order = the
  order given**. Local headers, central directory, EOCD, method 0 (stored) throughout, CRC-32
  on the reflected polynomial `0xEDB88320`; the central directory's `offset` is a **byte**
  offset into the finished buffer. No deflate: a PNG is already compressed and a float field
  gains a few percent, so this stays ~80 lines with no dependency and no worker — and a
  stored archive is what `numpy.load` reads, which makes it the `.npz` writer too (item 4).
  Names are UTF-8 with general-purpose bit 11 set. No ZIP64, so 4 GB is the cap.
- `runManifest(extra) -> object` / `manifestMember(extra) -> {name, data}` — the run's
  description (app, sim time and step, grid, box, dissipation + hyper exponent, forcing
  state, IC preset, seed, CFL), built from `liveParams()` at the press. **One builder for
  both exports**, so the archive of pictures and the field export cannot describe the same
  run differently; `extra` is merged over the result.
- **Capture ordering is the constraint.** `_cardShots()` calls `captureShot()` on every
  display card and then every chart card in ONE synchronous pass and returns the promises;
  `saveAllZip()` awaits nothing until that pass is over. A display card's texture is
  transient, so an `await` between two cards would capture the second from an expired one.

The archive's result strip is the **page's**, not a card's: `saveAll` is a plain object with
a `foot`, a `resEl` slot (`zip`) and a `dead` flag, which is all `cardResult` ever asks of a
card — so the file describing every card is not parked behind one card's × and cannot be lost
when that card is closed. Its host is a `viewfoot` div in the control group
(`{ k: "hintdiv", cls: "viewfoot" }`). Gate: `devtools/checkzip.js`.

**The fields themselves, as `.npz`** (IO_PLAN item 4). `save fields`, next to `save all`,
writes ONE time slice of the real-space state: `phi.npy`, `psi.npy`, the coordinate
vectors `x.npy` / `y.npy` (and `z.npy` in 3D) and the same `params.json`, in a stored ZIP
— which is all an `.npz` is, so `zipStore` is the writer and `npyBytes` is the only new
format (`.npy` v1.0: magic, version, a 16-bit header length, an ASCII dict padded so the
DATA starts **64-byte aligned**, then raw little-endian values). One writer for all five
members; the shape tuple is the only thing that differs. Five contracts:

- **The array axis order IS the buffer layout read literally**: `(nx, ny)` in 2D from
  `ix*ny + iy`, `(nz, nx, ny)` in 3D from `(iz*nx + ix)*ny + iy`, C order, `<f4`. That is
  byte for byte what `setICFromReal` consumes, which is what would make "load fields" a
  small follow-up rather than a rewrite. It is stated in `params.json` (`export.axes`) and
  in `docs.html`, and it is the leg the gate exists for — a silently transposed field is
  the classic way to waste someone's afternoon.
- **The export owns a THIRD and FOURTH Mode uniform per display chain** (`modeX`,
  `modeX2`, with `prepDispX` / `prepDispX2`), **pinned** to plain `phi` and plain `psi`
  with no band, no offset and no colormap: written once in `_makeChain` and **never
  again**. `B.mode` and `B.modeM` are a card's LIVE display uniforms — written only by
  `setDisplayMode`, carrying that card's band filter and display offset — and the contour
  path's `modeC` feeds `dispK2`, never `dispR`. Borrowing any of them would corrupt that
  card until its next `apply()`. No new pipeline, kernel or WGSL: the export is the
  display chain's own stages, differently bound, which is why the emitted WGSL of both
  pages is byte-identical across this feature. It is **not** allocation-free.
- **`Solver.encodeExport(pass, chain)` is the one per-app piece** — the 2D and 3D display
  chains differ by exactly one stage (`prepDisp → colsInv → rowsC2R` against
  `prepDisp → zInv → colsInv → rowsC2R`), so each app spends six lines on it and the
  uniform pin, the readback, the packing, the delivery and the whole UI live once in
  `common.js` (`readFieldPair`, `npzFields`, `saveFieldsNpz`). Both fields go through ONE
  compute pass and both readbacks are submitted before the first `await`, so `phi` and
  `psi` are the same instant of the run even while it steps; the `psi` leg runs second
  because its prep zeroes the k-scratch the `phi` leg used. It runs at a **frame boundary**
  (a button press is its own task), never inside a render.
- **A field readback must NOT be pooled.** `_stagePool` has no eviction, deliberately,
  because every other pooled size is kilobytes; a field is 1 MB at 2D 512², 4 MB per field
  at 3D 128²×64 and **16 MB** at the largest offered grid, 256²×64, so one export through
  `readBuf` would strand tens of MB for the life of the page. `readBufOnce` allocates its
  staging buffer and `destroy()`s it when the read resolves — the rule is written at the
  pool as well as here. Measured archives: 2.10 MB at 2D 512², 8.40 MB at 2D 1024²,
  33.56 MB at 3D 256²×64.
- **What the export actually costs in memory is FOUR times the archive, not one.** At the
  largest 3D grid the press holds, all at once: the two fields `readBufOnce` copies out of
  the mapped range (2 × 16.8 MB), the two `.npy` members `npyBytes` copies them into again
  (2 × 16.8 MB), the finished ZIP buffer `zipStore` assembles (33.6 MB) and the `Blob`
  copy of it (33.6 MB) — **≈134 MB of host memory** (measured 128 MiB in the node
  harness), on top of 33.6 MB of GPU staging while the two reads are in flight. Everything
  after the await is one uninterrupted main-thread block: two full-array copies, a CRC-32
  pass per member over ~34 MB, and one whole-archive copy (≈180 ms in node, more on a
  phone). That is the number the on-device memory check is sized against — not the 33.6 MB
  of the file.

The export has its **own** slot on the page strip (`saveAll.resEl.npz`, `RES_WHAT.npz`)
and its own busy flag: an archive of pictures and a field export are different files, so
pressing one must not throw the other away. Name: `capName("fields", "npz")`. Gate:
`devtools/checkzip.js` sections D and E, which verify the file **externally** — `zipfile`
for the archive, `numpy.load` for dtype, shape, axis order and values, with every
expectation built from the solver's grid rather than from the file's own header. Section E
reruns those legs on a grid whose axes are **not** equal (the 2D `wide` box, 256×64 over
4π × 2π, and a 3D 64²×128), because on the square boot grids a transposed axis order is
invisible however carefully it is asserted. `rec` records the field canvas (that canvas alone: the arrows, the field lines
and the colorbar are in the PNG only) with a 30 s hard stop, on whichever of **two legs**
the engine supports.

**Leg 1, WebCodecs (preferred).** A `VideoEncoder` configured as `avc1.4200<level>`
Constrained-Baseline-class H.264, `avc: {format:"avc"}` (so the chunks are length-prefixed
samples and the first chunk's `metadata.decoderConfig.description` is the avcC payload the
`stsd` box needs), 5 Mbit/s constant, level picked from the frame's macroblock count.
The frames come off **the frame loop itself**: `loop()` calls each display card's
`recCapture()` in the same synchronous task as that card's `render()`, before any `await`
— WebGPU has no `preserveDrawingBuffer`, so a capture deferred even one microtask would
wrap an expired `getCurrentTexture` (the same reason `save` re-renders). A **slot cadence**
keeps it at 30 fps: `W.due` is the next slot's wall clock, a callback before it captures
nothing (a 120 Hz phone captures on about every 4th and drops nothing), and a loop so late
that whole further slots went by counts those into `W.drop` and re-bases `due` on *now* —
never backfilling, since a backfilled frame would sit at a timestamp it never had.
**What a due slot does** is the RECASYNC part (2026-08-12): building a `VideoFrame`
*from the canvas* is a synchronous readback on the main thread — 15–17 ms per capture on
a real iPhone (`?recdebug`), which pushed every slot-due pass past its vsync window and
was the stutter itself. So on an engine whose WebCodecs can build a `VideoFrame` from
**bytes** (probed once with a 2×2 frame, `recBufOff` latch, degrade silently), a due slot
only *submits* a GPU-side `copyTextureToBuffer` of the canvas texture into one of three
staging buffers (the one always-on cost: every card's context is configured with
`COPY_SRC`) and returns — microseconds. The bytes arrive when `mapAsync` resolves, a beat
later, and are encoded inside one **ordered promise chain**, where frame index, timestamp
`round(n·10⁶/30)` µs and `keyFrame: n % 30 === 0` are all assigned at *encode* time — map
resolution order across buffers is nothing to rely on, `VideoEncoder` wants monotonic
timestamps, and `mp4Mux` writes a uniform `stts`, so a dropped or failed capture leaves
fewer frames and never a hole. No free buffer (three maps in flight), a mid-take canvas
resize, a failed copy, or `encodeQueueSize` past `REC_QMAX` all *drop* the slot without
advancing `n`: the timestamps stay an exact 1/30 s apart and a slow machine records fewer
seconds of wall clock rather than a sample table that lies about its timing. Stop drains
the in-flight captures first (500 ms cap, so a hung map cannot hold the file hostage,
and the stragglers count as drops), then flushes, then closes the encoder, frees the
pool and muxes — on every route out. Engines without the
bytes constructor keep the old direct `VideoFrame(canvas)` path unchanged.
The `setInterval` at 1000/30 ms is still there but **demoted to a watchdog**
(RECRAF_PLAN, 2026-08-12): it re-renders and encodes only where the rAF feeder is
**known-absent** — `document.hidden` or the editor view. Round 1 parked it on a timing
heuristic (no capture for ~117 ms) and an on-device test showed why that fails: a loaded
phone's loop gaps beat any threshold, waking the watchdog *alongside* a perfectly live
rAF feeder — both feeders at once, more main-thread work than the old recorder. So the
watchdog now keys on the condition it exists for, i.e. exactly where there is no rAF to
ride — a
background tab, whose rAF is throttled to a stop, the editor view, which renders no cards,
and the headless stub. It was feeding leg 1 from that timer that made a live recording
stutter visibly on an iPhone: an *extra* full `render()` per tick plus the frame copy and
the encode, all on the main thread and at an arbitrary phase against the rAF loop. Riding
the render costs zero extra renders, and the watchdog's second render now recurs only where
there is no visible display to stutter. A watchdog-fed frame re-bases `W.due` too, so the
slots it put in the file are never double-booked as drops when the rAF loop resumes. The
watchdog keeps the *synchronous* canvas capture even when the buffer path is on: it renders
off-screen anyway, so a stall costs nothing visible, and the async pool stays out of the
background-throttled world where a hidden page starves maps of callbacks. On stop (button,
timer, `destroy()`, encoder error) it drains, flushes and `mp4Mux` writes the file.
`?recdebug` in the URL adds one readout line per live recording — frames fed by rAF vs
the watchdog (wd must stay 0 on a visible page), drops, the longest gap between loop
passes, and the capture cost split: `vf` is the max ms a capture cost the MAIN THREAD
(copy+submit on the buffer path — expected ≲1 — or the full `VideoFrame(canvas)` readback
on the sync path), `enc` the max ms in the encode step (on the buffer path that includes
building the frame from bytes, off the hot path in the chain), and `lag` the max
capture-submit→encode delay — tens of ms are fine, frames are stamped by index, not
arrival; it stays 0 on the sync path. This is how a phone, with no devtools console,
reports what a stutter is made of. Deliberately absent from `docs.html`.

**Why we mux it ourselves.** Chrome's `MediaRecorder` `video/mp4` is a *fragmented* MP4:
`moov` + `mvex`, one `moof` per fragment, `trun default_sample_flags 0x10000` — "not a
sync sample, dependency unknown" — and a keyframe only every ~1.4 s. Desktop players
ignore the flags; iOS AVFoundation believes them, drops every delta sample, and plays a
30 s recording as about three stills. The codec was never the problem. `mp4Mux` writes the
plain progressive shape instead: `ftyp` + `mdat` + `moov`, real `stts`/`stss`/`stsc`/
`stsz`/`stco` tables (media timescale 1000·fps, so one frame is exactly 1000 ticks),
`avcC` verbatim in `stsd`, `stco` 32-bit with an assert instead of a `co64` branch that
30 s could never reach, and no `moof`, `mvex`, `trun`, `ctts`, `edts` or `sdtp` anywhere.
`moov` comes last, which is why `mdat`'s offset is known when the tables are built.

**Leg 2, `MediaRecorder`** over `captureStream(30)`, unchanged, for engines without
WebCodecs: MP4 (H.264) where `isTypeSupported` allows it, WebM otherwise, `onstop` the
single write path. It is also where leg 1 bails to, mid-press and for the rest of the
session, if a first chunk ever arrives without a `decoderConfig.description` — without an
avcC there is no playable progressive file to write. The button is shown when **either**
leg can run and is simply absent otherwise. Filenames are
`taranis-<page>-<field>-t<time>.{png,mp4,webm}`.

**Every open display, one video** (IO_PLAN item 5). `rec all`, beside `save all` in the
displays & charts group, records *every* open display card into ONE file. There is no
selection UI and none is needed — `CARD_MAX_DISP` is 3, so "all of them" is at most three
tiles and the visitor composes the recording by opening the displays they want. The
motivating case is z⁺ and z⁻ side by side in an imbalanced run, where the reading depends
on the two panels being the **same frames**; two files an editor has to line up lose
exactly that. Five contracts:

- **One recorder, N sources, one slot clock.** `Recorder` takes a HOST (the button, the
  result strip, `dead`) and a SOURCE list, and a card's own `rec` is the N = 1 case of the
  same code — the layout `recTiles` returns for one card *is* the frame `rec` has always
  encoded, down to `recCompose` handing that slot's mapped range to the `VideoFrame`
  constructor with no copy. Two recorders would drift: one `W.n`, one timestamp ladder,
  one `VideoEncoder`, one mp4. The feeder is `renderCards`' existing loop with one capture
  **after** it — same synchronous task, after every card has rendered and before any
  await, for the reason the per-card capture rides the render. A source of a live take
  reports `needsRender()` true, so the gate cannot skip the texture the slot is about to
  read.
- **All-or-nothing slots.** Every source is checked before anything is committed, and the
  staging pool hands out a SLOT (one buffer per source) rather than a buffer, so a slot is
  captured whole or dropped whole and `W.n` does not advance either way; the slot's N maps
  are joined (`allSettled`) before it takes its place in the ordered encode chain. This is
  the RECRAF/RECASYNC honesty rule with one clause added: fewer frames, never a lying
  sample table **and** never a composite whose panels are from different moments. A
  composite take costs N × `REC_POOL` staging buffers.
- **Composite at encode time.** `recCompose` copies each source's mapped bytes row-wise
  into one frame buffer at its tile offset and blits that tile's label patch, clipped to
  **its own tile** — N memcpy loops, no 2D canvas on this leg. The frame buffer is ONE
  per take (`W.frame`, made with the pool and freed with it): 3 × 1024² is 12 MB, so a
  fresh one per slot would be ~360 MB/s of garbage at 30 fps, and `VideoFrame(BufferSource)`
  copies at construction, so the buffer is free again the moment the frame is built. The
  one-tile, aligned, unlabelled slot — a card's own `rec` — still hands the mapped range
  straight to the constructor with no copy and no scratch at all. Layout is one **row or
  one column, whichever comes out closer to square** (a tie goes to the row: side by side
  is the case this exists for); all cards show the same run, so equal `cv` sizes are
  **asserted**, not scaled — mismatched sources, and sources with no pixels, return no
  layout at all rather than a card list quietly one short.
- **Probe the composite size, halve the TILES.** The press asks `isConfigSupported` for the
  composite config and, if refused, for the same tiles at half size (point-sampled in the
  same row loop; a per-frame resample is the cost this leg exists to avoid) — and the
  result strip's line then says `tiles at half size`. The card list is never truncated to
  fit, and an encoder that refuses both sizes starts nothing and says so rather than
  failing silently.
- **Leg 1 only, and one canvas or none.** Leg 2 records a canvas *stream*, which cannot
  cover several WebGPU canvases without compositing through a real 2D canvas, so the
  action is offered only where the WebCodecs leg runs (`recAll.sync`, which also disables
  it below two cards) and single-card recording is untouched everywhere else. `W.cv` is
  the one canvas the sync paths may read — null for a composite — which is also why the
  watchdog does not feed a composite take on a hidden page: it would have to build a
  VideoFrame from a canvas that does not exist, so the take simply records fewer frames.
  It is also why a composite that has latched OFF the buffer path (`recPoolMake`'s
  out-of-memory catch, a thrown copy, a failed compose) **drops every remaining slot**:
  there is no canvas to fall back to, so the slot is counted lost rather than built from
  nothing. Labels are rendered ONCE at start into small opaque RGBA patches
  (`recLabelPatch`, in the frame's own byte order) and blitted per frame — the one
  exception being a field **retyped under a live take**, which re-renders that tile's
  patch on the next slot (`recRelabel`): the tile's pixels have already moved, so the
  caption moves with them instead of lying about them. A card's own take is unlabelled, as
  it always was. **A source card closed mid-take ends the take and writes the file** — the
  encoder's frame size cannot move, so a vanished source could only be papered over with a
  stale tile, and the strip is the page's, so the file survives the card that went.

`devtools/stubenv.js` stubs `toBlob`, `captureStream`, `MediaRecorder`, `VideoEncoder` /
`VideoFrame` / `EncodedVideoChunk`, `Blob` (keeping the *bytes*), `URL.createObjectURL`,
and `setInterval` as a hand-driven pump (`env.tick(n)`) with `env.fireTimeout(ms)` for the
30 s cap — so `bootstub.js` runs the whole WebCodecs path headlessly (pump, forced
keyframes, backpressure, flush, an `ftyp+mdat+moov` download with no `moof`, the 30 s cap,
destroy-mid-record, the no-avcC bail) and then the same MediaRecorder legs as before. Its
stub `requestAnimationFrame` is a no-op and its `performance.now` jumps 250 ms per *call*,
so `env.tick(n)` always drives the **watchdog** path; the rAF-side feeder is driven by
calling `recCapture()` directly, which is how the slot cadence, the skipped-slot counting
and the handoff in both directions are covered.
`devtools/checkmp4.js` drives `mp4Mux` with **real** H.264: ffmpeg encodes a test pattern,
the script cuts it into samples and builds the avcC (the only Annex-B code in the project,
and it is in the test), and ffprobe/ffmpeg check the result — top-level boxes, sync samples
exactly on the forced indices, equal pts deltas, `30/1`, and a decode with zero errors, for
a square canvas, the 1024×256 wide box, a one-frame file and the 1024×512 two-tile
composite. `devtools/checkrecall.js` is item 5's own gate: the tiler and `recCompose`
alone (synthetic patterns in, one composite out, every row read back at its expected
offset, both layouts, padded and tight rows, the halved tiles, a source with no pixels,
the label blit clipped to its tile in both layouts, and the composite scratch reused
byte-for-byte), then the action on both booted pages — the offer rule, a two-card take
from first copy to muxed file, an all-or-nothing slot driven three times (a resized
source, a rejected map, and the buffer path latched off) with the file's `stts` read back
as a single uniform run, a field retyped mid-take being re-captioned, the
`isConfigSupported` refusal halving the tiles without truncating the card list, a card
closed mid-take, and the single-card path asserted unchanged — its own canvas, its own
name, no label, and its **sync** fallback still recording when the buffer path is off.

**Contour overlays** are per display card too: ψ contours (= the perpendicular magnetic
field lines), φ contours (= the streamlines), or **both at once** (ψ + φ — the alignment
view), at 8 / 16 / 32 levels, over whatever field the card displays or over a **plain
background** (the second per-card select: contour ink on a blank plate, when the colours
underneath are in the way). They are drawn inside the shared colorize kernel, not as a
second pass: each active set's potential reaches the display scratch through ONE
extra inverse transform per card frame (reusing scratch that is dead by then — the second
component's, and for the second set the σ_c half-1 buffer), and a texel is inked when
`floor(pot/Δ)` differs from that of its +x or +y
neighbour — a crossing test, so no derivatives and nothing a compute shader cannot do.
Δ is **uniform**, so the line density is proportional to |B⊥| (or |u⊥|), which is the
honest picture rather than a prettier equal-area one. Δ = 2·range/nlev with the range
adapted on the GPU (up at once, down by 5% of the gap per frame) so the lines do not
flicker from frame to frame, and each set has its own range (ψ and φ are unrelated in
size). The first set's ink is black over a light background and white over a dark one, so
it survives every colormap and the plain plate; the second set is a fixed magenta accent
that none of the four colormaps produces, so the two sets are always told apart. On the
cube view the contours are drawn on the **top face only** — that is the one face whose
in-plane potential the chain already has.

**Cross helicity σ_c** = (|z⁺|²−|z⁻|²)/(|z⁺|²+|z⁻|²), pointwise, is the one mode with a
**fixed** colour range (±1, symmetric — no autoscale, so colours are comparable
between frames and between runs). Where the local Elsasser energy density |z⁺|²+|z⁻|²
falls below 1e-4 × its maximum over the displayed field, σ_c is rendered as exactly 0:
in quiet regions the ratio is pure noise. It costs four inverse transforms per frame
(both components of both z±) instead of the vector modes' two — display cost only, no
physics buffer is touched.

**Residual energy σ_r** = (|u|²−|b|²)/(|u|²+|b|²) is the same two-half machinery with
the u and b vectors as the pair, and shares σ_c's fixed ±1 range and relative
quiet-region floor. Both apps offer it, and in 3D it goes wherever σ_c goes — a z slice
or the three cube faces — since the two modes differ only in which pair of vectors the
chain's two halves are pinned to (the floor is then the maximum over that same target:
the plane in a slice view, the three faces together in a cube view). Note the floor's
rendering convention: quiet pixels show the neutral mid-colour, which for σ_r is also
the equipartition colour — a grey quiet region means "too little energy to measure",
not "measured equipartition".

3D only: **cube faces** are a *view*, not a field — three of the last four entries of a
display card's z-source select ("cube faces", "cube + track z⁺", "cube + track z⁻"; the
fourth is the field-lines view below), orthogonal
to the field selector, so **any** quantity (σ_c, σ_r and the vector magnitudes included)
can be drawn as a cube. It draws the three visible boundary faces of the box in the oblique
view `examples/forced-turbulence-3D.ipynb` uses (matplotlib `view_init(elev=30,
azim=45)`), colorized through the same shared shading as a slice (so each mode keeps its
own colour range) with one common autoscale across the three faces, depth-cued by a
per-face darkening of 1.0 / 0.85 / 0.7. The **top face is the card's own plane** — its
slider in manual mode, its tracker otherwise, so a cube card riding a packet puts the
collision front on top; the two side faces stay x = Lx and y = Ly boundary slices. On the
vector modes the arrow overlay is drawn on the top face, the same anchors and directions
put through that face's affine projection (CPU-side, on the overlay canvas; side faces
get none — u⊥/b⊥ is not tangent to them). Any display card may take the view (each chain
owns its own three face buffers and three face textures — a rounding error next to its
nz·nx·ny scratch volume); the cut chart does not offer it. The 3D spectrum
card also carries the parallel spectra as dashed curves (|kz| bins 1…nz/2, ±kz paired,
kz = 0 omitted from the log axis); **the y limits are set by the perpendicular spectra
alone**, so the dashed curves are plotted inside that range and never stretch it (unless
∥ only is selected, when there is nothing else to scale to).

### 3D field lines, and k∥ measured along them

**Field lines** are the last *view* in a display card's z-source select, next to the cube
entries (and, like them, not offered by the cut chart). It traces a fixed 8×8 grid of
magnetic field lines seeded on the bottom face and draws them — with the twelve box edges
— through the same oblique box projection on the card's overlay canvas. The march is a compute
kernel solving dx⊥/dz = b⊥/B₀ with B₀ = v_A = 1: RK2 midpoint (the midpoint field is the
mean of the two bracketing planes, so it is second order in dz with no extra storage),
bilinear in plane, uniform dz, periodic ⊥ wrap. Its volume is the RHS's *own* gradient
prep — ∇⊥φ and ∇⊥ψ in real space — so b⊥ = ẑ×∇ψ needs no new gradient or FFT kernel, and
the volume never leaves the GPU: what comes back is the **polylines only** (N_lines × nz
positions plus the same number of (u, b) samples, tens of kB). It runs at ~2 Hz, not per
step, in its own submit. There is no depth sorting and no occlusion: a 2D canvas, one
pass, every line visible. A line that leaves the box is drawn wrapped, with the pen
lifted at the seam.

Behind the lines the GPU canvas carries no field: it is cleared to the contour plate and
draws the box's **top boundary plane as contour ink only**, through the cube projection
and the same shared colorize kernel — every texel of that face which is not ink is exactly
the background colour, so the face is transparent with no plate, no blend state and no
second kernel (the whole mode chain is skipped in this view, since nothing reads its
colour). The card's own contour selectors stay live and drive that face — ψ (the default
on entering the view), φ, both, or off — so the line endpoints are seen puncturing the
exit plane on its own ⊥ structure. The field selector and the arrow overlay are inert
here, and so are the tracker and the z slider: the view is the whole box, and its face is
the top boundary, not a tracked plane.

The same march samples (u, b) **along** each line, which is what the spectra card's
`⊥ + k∥ (field line)` option bins: the true parallel spectrum, measured along B rather
than along the z coordinate. The samples are uniform in z (arc length = z to leading
order in RMHD) but not periodic — a line exits perpendicularly displaced — so each line
is Hann-windowed before the transform, the window's mean square divides back out (so
Parseval still holds and the numbers mean the same thing as the coordinate E(k∥) plotted
beside them), the ±kz bins are folded, and the ensemble of lines is averaged. The three
binned lanes are the usual [E_u | E_b | H_c], so E<sup>±</sup>(k∥) needs no separate
path, and the y-limit rule is unchanged: the ⊥ spectra set the range. The two consumers
are independent: with such a chart open and no card in the lines view the lines are still
traced and simply not drawn, and the along-line samples — four times the polylines' size —
come back only while that chart is open.

### χ = k⊥δb/(k∥v_A): the anisotropy card's second ordinate — OFF by flag

**Not currently offered in the UI**: `ANISO_CHI_UI = false` in `common.js` (Alfred,
on-device 2026-08-14 — χ came out ~2 at the defaults and strongly scale-dependent, so the
flat-line claim is not attainable in a low-resolution real-time run). The code below is
all still live and checked — `checkaniso` drives the χ branch directly, bypassing the
header — and one `true` re-offers the select, the tooltips' χ clauses and the docs' #chi
section (hidden beside the same note) with it.

`ay` on the aniso card (CHI_PLAN). Not a second card: χ differs from the shipped curve by
one multiplication and a label, and everything else — the tails, the level window, the lane
select, the legend, the reference line — is the same code. Default is `ratio`, which is
bit-for-bit what shipped (`devtools/checkaniso.js` §8.1 asserts that against the base commit
through `git show`, on every (data × `aq` × `ad`) combination, for `ay` absent *and* spelled
out).

Why it is worth an ordinate: under the RMHD rescaling z → z/ε, Z → εZ, t → t/ε both k∥ and
δb pick up a factor ε, so `k∥/k⊥` moves with L_z (the card's standing gauge caveat) while
χ does not. Critical balance stops being "a slope of −1/3 in a quantity whose level is a
convention" and becomes "this is O(1) across the inertial range".

- **δb² = Q, the matched level itself** — the tail energy content above k⊥, which is the
  currency the matching is already built on, so it needs no second interpolation and is
  defined at every point of the curve. v_A = 1 in these units, and k⊥/k∥ is a pure ratio
  (`parKfac` puts the parallel bins in the perpendicular `kunit`), so `chi = kp*sqrt(Q)/kz`
  carries no unit of its own. It is an O(1) convention and the hint says so; the *flatness*
  is the claim, the level is an order of magnitude.
- **Elsasser pairing.** χ± = k⊥Z∓/(k∥±v_A): only **δb** crosses lanes. The (kp, kz) matching
  stays inside the selected lane — that pair *is* the measurement of k∥±(k⊥) from Z±'s own
  geometry — and δb is a lookup of the *opposite* lane's perpendicular tail at kp
  (`_anisoQAt`, the inverse direction of `_anisoAt`). Building the perpendicular tail from
  the opposite lane wholesale is the adjacent wrong implementation: it moves which k⊥ every
  level maps to. checkaniso §8.3 pins kp's **provenance** (bit-identical to the ratio curve's
  k⊥) and not merely the sign of the ± split, because a pure amplitude asymmetry cannot tell
  the two apart. On `tot` there is no opposite lane and δb² is the matched Q itself. A lane
  whose opposite carries nothing (E⁻ ≡ 0) has no shearing field, so the card draws nothing
  rather than quietly substituting the own-lane level.
- **The reference line** switches with the ordinate: a −1/3 power law is a statement about
  the ratio, so on χ it becomes the horizontal χ = 1 (`ANISO_CHI_REF`), the index box hides,
  and the amplitude box renames the level. The line is included in the y-range, because "is
  the level 1?" is unanswerable with the 1 off the top of the frame.
- **Estimator bias, measured.** `flSpectrum` must Hann-window each line before transforming,
  and the periodic Hann kernel is exactly (1/6, 2/3, 1/6) in bins: that fattens the parallel
  tail, the matching returns k∥ high, and χ comes back low. checkaniso §8.4 measures the
  factor end to end — a synthetic field with a prescribed ridge k∥(k⊥) = round(k⊥^2/3) and a
  prescribed δb, through the real `flSpectrum` and the real `anisoCurves` — and **reports**
  α = χ_measured/χ_true = **0.988** (0.95 at the low-k⊥ end, where the ridge sits a few bins
  up). Reported, never gated. The obvious calibration is null by construction: sweeping L_z
  cannot expose the bias, because the forcing is scattered onto the kz = ±2π/L_z planes and
  the ridge sits at the same *bin indices* whatever L_z is.

Cost: nothing. No WGSL, no buffer, no readback, no kernel — the same CPU arithmetic on the
same spectrum readback the card already rides.

## Initial conditions

The solvers know exactly two ICs: the built-in large-scale mode pattern (`setIC()`, what
the reference vectors use) and an exactly quiescent state (`setIC(true)`). Everything
else is built on the CPU and uploaded through **`setICFromReal(phi, psi)`** — real-space
`Float32Array`s in the buffers' own layout (`ix*ny + iy` in 2D, `(iz*nx + ix)*ny + iy` in
3D) — which forward-transforms them and applies the 2/3 dealias exactly like the
constructor does (unmasked beyond-cutoff IC energy would persist and alias). The **IC**
selector then offers `large-scale modes`, `quiescent`, `letters`, `custom` and
`expression` (plus the two equilibrium presets in 2D, and the sinusoidal packet pair in
3D); **Reset** re-applies the current one.

### Expression ICs: the contract

`expression` gives φ and ψ two text boxes and uploads what they say. Three rules bind
anything that touches it (IO_PLAN item 1; the parser and everything around it live in
`common.js`, since both pages load it and there is nowhere else shared):

- **No `eval`, no `new Function`.** The parser is hand-written — tokenizer → shunting-yard
  to RPN → a switch-on-opcode evaluator — and not mainly for security (the page has no
  secrets): an engine's `SyntaxError` names nothing a student can act on, and the accepted
  language would otherwise be all of JavaScript, which we would then have to document.
  Every token carries its source index, so every failure is a message with a character
  position and never a throw or a silent zero. It also means the cost is an interpreter's:
  ~4.5 ns per stack op with a native `Math`, so a 30-op expression is ~140 ms at 1024² and
  ~0.6 s at 256²×64 — one button press, not a frame. Codegen would close that and is the
  one thing the no-`new Function` rule forbids.
- **Code units.** `x`, `y` (and `z` in 3D) are the coordinates `setIC` itself uses:
  `x = ix*Lx/nx`, so the box ends at `Lx`/`Ly`/`Lz` and nothing is normalized to 2π or to
  1. `Lx`, `Ly` (and `Lz`) are constants for that reason. `z` and `Lz` are **unknown
  names** on the 2D page rather than the zero `icDrawGrid` would hand out.
- **Warn, never block.** A non-periodic expression is reported per axis, by the seam it
  fails on and with the number, and then runs. A **non-finite** one is the one refusal: a
  NaN through the forward FFT reaches every mode and the run is silently dead from step 0.
  The guard tests the **float32 that is stored**, not the double that was computed —
  `exp(100)` is an unremarkable double and an `Infinity` the instant it lands in the
  field, so a guard reading the double would wave it straight through.

The preset registers through `icRegister` like an equilibrium does and skips
`icZetaFields` — what is typed has to mean what it says, which is also why the amplitude
sliders' rows stay hidden for it. Either box may be empty: that is `null`, which
`setICFromReal` already reads as exactly zero. Gate: `devtools/checkexpr.js`.

### ζ±, amplitudes, and where the normalization happens

What the letter and blob ICs build is the Elsasser **potential** ζ±; the Elsasser
*fields* are z± = ẑ×∇ζ±, and the evolved variables are φ = (ζ⁺+ζ⁻)/2, ψ = (ζ⁺−ζ⁻)/2.
The UI says ζ± wherever a potential is meant (paint target, amplitude sliders) and z±
wherever a field is (display modes, tracking) — except in the (φ, ψ) paint basis below.
Identifiers stay `zp`/`zm`.

Stored ICs are kept at their natural scale and normalized **only at apply time**, by
`common.js: icZetaFields`: each potential is zero-meaned plane by plane (a pure gauge for
a perpendicular gradient) and scaled so that max |ẑ×∇ζ±| over the volume equals **its
own amplitude slider** — ζ⁺ and ζ⁻ have separate sliders, linked by a lock checkbox that
is on by default. Because the stored arrays are never mutated, pause → move a slider →
**Reset** genuinely rescales the same drawing, any number of times (before Phase G the
amplitude was baked in per blob at deposit, and Reset re-uploaded the old one).
Consequences worth knowing: the amplitude is a property of the whole drawing, not of one
blob, so overlapping strokes redistribute instead of stacking past it; and a ζ⁺ drawing
and a ζ⁻ drawing are normalized independently, so their *relative* size is set by the two
sliders, not by how hard you scribbled.

**The (φ, ψ) basis.** *Which* pair those two sliders normalize follows the **paint**
target (`icAmpBasis`, and only while `custom` — the drawing — is the selected IC, since
nothing else has a paint target). Painting ζ⁺/ζ⁻ keeps the historical basis above.
Painting φ or ψ switches `icZetaFields` to normalize the *combinations* instead: the
labels become **φ amp** and **ψ amp**, and they mean max |∇φ| = max |u| and
max |∇ψ| = max |b| — the peak flow speed and the peak perpendicular field strength,
independently. Both are exact linear maps of the same stored ζ± pair, and a drawing with
only φ strokes (or only ψ ones) comes out *bitwise identical* under either basis with the
amp lock on. The basis exists for the mixed case: a drawing carrying both φ and ψ strokes
stores ζ± = P ± Q, so normalizing ζ± can only rescale an inseparable mixture — "a strong
vortex plus a weak island" is not expressible in that basis at all, and it is what the
`rmhd variables` demo closes with. χ (3D) is written in Elsasser amplitudes, so its
readout goes through `icAmpZeta`, which maps a (φ, ψ) pair back to
(a⁺, a⁻) = (a_φ+a_ψ, |a_φ−a_ψ|) — exact for co-located strokes of the same shape,
an estimate otherwise, like the k̄⊥ beside it.

The **letters** preset builds one potential per field from a rasterized glyph
(A → ζ⁺, B → ζ⁻; fixed, `common.js: IC_LETTERS` — the free-text input was dropped in the
mobile pass): the glyph is drawn on an offscreen canvas at 60% of the box and
gaussian-smoothed at a **physical length σ_letter = Lx/32** (`IC_SIGMA_PERP_FRAC`, the
single constant that is also the blob-width slider's default), via
`ctx.filter="blur(Npx)"` with N = σ_letter/dx where the engine supports it and an exact
separable periodic gaussian otherwise. Specifying the blur as a length, not a pixel
count, is what makes the letters resolution-independent: identical k⊥ content, hence
identical χ, at 128² and 512² (node check: normalized E(k) agrees to 3% rel L2, k̄ to
0.2%). In 3D each field is additionally multiplied by a gaussian z-envelope of peak
exactly 1 to make a *wave packet* (σ_z from its own slider, see below). The smoothing is
not cosmetic: the spectral gradient the GPU takes of an unsmoothed glyph edge overshoots
the finite-difference one used for the normalization by ~19%, and at σ ≥ 2 px they agree
to ~0.1%, which is what makes the amplitude a statement about the displayed field. (The
dealias at upload still trims it slightly, so the knob is approximate.)

## Drawn ICs: the blob editor (`custom`)

Selecting **custom** shows one extra row of controls; **edit IC** pauses the run and
switches the main area to a **dedicated editor view** (`#editview` replaces `#display`).
The view is one empty host div per app, filled by `common.js: icEditBuild` exactly as the
card system builds its cards — so its canvas is an ordinary card canvas, nothing is ever
painted over a live display, and no display card is special because of it (the 3D z-plane
slider is a flag on the hook object, not a second copy of the header). Click or drag and
each sample
deposits a **periodic gaussian blob** into the chosen target potential — `ζ⁺`, `ζ⁻`, `φ`
or `ψ`. The editor keeps only the two potentials on the CPU: a φ blob is one blob in
*both* ζ±, a ψ blob one of each sign. Leaving the view is one of

- **save & run** — keep the drawing, apply it (`icZetaFields` → `setICFromReal`), resume;
- **save** — keep it as the **Reset** target and stay paused;
- **cancel** — restore the snapshot taken when the editor opened (everything drawn since
  is discarded);

and **clear** empties the drawing without leaving (there is no undo stack). Nothing
touches the GPU while editing — the canvas is redrawn from the CPU arrays (the signed
mapping of the `colorize` kernel, in the first display card's colormap, autoscaled) at
most once per animation frame. The drawing is dropped when the grid changes (resolution,
or Lz in 3D) because its layout is per-grid.

- **Amplitude** is NOT a deposit knob: blobs go in at unit amplitude (peak P = 1·σ·√e for
  f = P·exp(−r²/2σ²), whose |∇f| peaks at P/(σ√e)) and the ζ± sliders scale the finished
  drawing — see the normalization section above.
- **σ⊥** (and **σ_z** in 3D) are fractions of the box, floored at 2 cells — anything
  narrower is eaten by the 2/3 dealias at upload. Their sliders' range, step and default
  come from the shared constants (`icSigmaSliderInit`), so σ⊥ defaults to exactly
  σ_letter and σ_z cannot exceed the packet cap Lz/12.
- **Sign**: the "negative" checkbox XOR the right mouse button, so a right-drag is a
  negative stroke without touching the controls.
- Blobs are truncated at 5σ (keeping 1−4·10⁻⁶ of the gaussian) and wrapped with the
  minimum-image convention `icGaussZ` already used, so a blob painted on the box edge is
  exactly a translate of one painted in the middle.
- **3D**: each blob is a perpendicular gaussian times a z-envelope of peak 1 centred on
  the plane the editor's **own z-plane slider** selects (opened on whatever plane the
  first display card was showing). The cube view is irrelevant now — the editor no
  longer borrows a card's slice, so editing is never refused.

Mouse → grid: `getBoundingClientRect` is in CSS pixels, so the responsive canvas width and
the device pixel ratio both divide out. The display puts grid point i at screen fraction
(i+0.5)/n — texel centres as the linear sampler sees them, the same convention
`drawArrows` uses to place its cells — and row iy = 0 is at the **top** (the render pass
samples v = 1−uv.y). So the continuous grid coordinate under the cursor is u·n−0.5 and the
nearest grid point is round(u·n−0.5); the deposit uses the continuous position, wrapped
into [0,L).

## Presets (the dropdown, and `?demo=`)

A preset is a **configuration**, never new physics. A registry entry carries `set`
(control id → value), `prep` (the "auto" seeds), `layout` (the display + chart cards it
wants) and a one-line `hint`. `presetBoot` (common.js) fills the dropdown, preselects
`?demo=NAME` if the URL carries one, and writes the controls *before* the first solver is
built; picking another entry from the dropdown runs the same path plus a rebuild.
Everything stays adjustable afterwards. There are no separate demo pages.

- **2D `decaying A / B packets`** (`rmhd2d.html?demo=decay`) — letters A/B at 512²,
  forcing off, hyper=4 seeded at ν_marg and then driven by auto-diss (see
  *Dissipation* below), two displays showing |z⁺| and |z⁻|. Watch the
  spectrum settle onto the quasi-universal decaying slope, and switch a display to σ_c
  for dynamic alignment. Its hint states the one thing the amplitude slider does *not*
  do here: ideal 2D RMHD is self-similar in amplitude ((a,t) → (a/λ, λt)), and dt is
  adaptive, so the per-frame dynamics are amplitude-independent — amplitude only matters
  against the fixed dissipation. (In 3D it matters through χ.)
- **2D `rmhd variables (φ, ψ)`** (`rmhd2d.html?demo=rmhdvars`) — the only preset that
  opens on an **empty** drawing: 256², forcing off, fixed ν (no auto-diss — a laminar
  vortex has nothing at the dissipation scale to measure), IC `custom` with **paint** on
  φ and a big blob (σ⊥ = L_x/8), one display of φ with its own contours and no chart
  cards. The user's single click *is* the initial condition, and the hint walks the
  variables from there: φ blob → a stationary vortex whose φ contours are the streamlines
  of u; the same field as vorticity ω = ∇²φ; then the same click in ψ → a magnetic island
  whose ψ contours are the B⊥ field lines; then a strong φ blob plus a weak ψ one, which
  is what the (φ, ψ) amplitude basis above exists for.
- **2D `Kelvin–Helmholtz shear layers`** (`rmhd2d.html?demo=kh`) and
  **2D `tearing mode`** (`rmhd2d.html?demo=tearing`) — the two equilibrium demos; see
  the next section.
- **3D `Alfven-wave collision`** (`rmhd3d.html?demo=collision`) — 64²×256, Lz = 8π,
  forcing off, two counter-propagating **sinusoidal** packets, on one volume view of j∥.
  (It opened on the letter packets until the on-device pass: the sinusoids fill the box,
  so their shells read as surfaces rather than as two blobs in a long column. The letters
  are one entry away in the IC dropdown, unchanged.)

## Equilibrium demos: Kelvin–Helmholtz and tearing (2D)

Unlike every other IC these two are **equilibria plus a seed**, so their knobs are
physical (U₀, b₀, ψ₀, the layer width a, the seed amplitude) and they skip the ζ±
normalization entirely — rescaling an equilibrium to a fixed max |∇ζ| would not be the
equilibrium anyone asked for. They are registered exactly like the letters and the
drawing (`icRegister` in common.js: `rows` = the control rows the preset shows, `hyper` =
the exponent it locks, `src` = whether it offers the maintained-flux source, `fields(g)` =
the (φ, ψ) pair), so adding a third costs one record.

The two SHIPPED presets run in a **rectangular box**: 512 × 128 on 4π × 2π, chosen by the
`box` select next to the resolution (which now means n_x — n_y and both box lengths follow
the box). A long x holds the layer's far field inside the periodic box, a short y carries
exactly one unstable wavelength k_y = 2π/L_y. (The two TEARNL presets below take the other
two boxes: `collapse` the square one, `chain` the large 8π × 8π.) Rectangular boxes are 2D-only; everything downstream
that used to assume a square perpendicular plane is now min-based or aspect-aware
(the shell-bin count `nbins`, the arrow subsample, the display and editor canvases,
which get equal pixels per unit *length* so that contours and arrows are not sheared).

- **KH**: u_y(x) = U₀[tanh((x−L_x/4)/a) − tanh((x−3L_x/4)/a) − 1] — two layers of
  opposite sign, which is what periodicity forces, and independent of each other while
  a ≪ |x₂−x₁| = L_x/2 (the preset's default is L_x/2 = 10a). ψ_eq is the same profile
  scaled by b₀, i.e. an in-plane field along ŷ. The potentials are the analytic
  antiderivative, a·ln cosh(·) written overflow-safe, so u_y and b_y are exact to
  round-off. Ideal 2D MHD stabilizes the layer at **b₀ ≥ U₀** (the shear is then slower
  than the Alfvén speed tying the field lines together); dissipation softens that
  threshold rather than removing it.
- **Tearing**: ψ_eq = ψ₀ sech²((x−L_x/2)/a) (Numata/Loureiro-style — net-flux free, and
  periodic to O(e^{−L_x/2a})), φ_eq = 0. b_y = ψ_eq′ **vanishes on x = L_x/2**, so that
  line is the resonant surface of every k_y mode; the seed perturbs ψ there with the same
  even-in-x envelope, so ψ̃(x_s) is exactly the seed slider. L_y/a sets Δ′a (8.40 at the
  defaults, from a shooting solve of the outer equation — `devtools/eqlinear.py`).

**hyper is LOCKED to 1 by the tearing preset** (the select goes disabled and shows why):
hyper-dissipation has no resistive layer and no Rutherford stage, so it would falsify
exactly the physics that demo exists to show. KH does **not** lock it — it is an ideal
instability, and hyper is a legitimate way to sharpen its secondary structure.

**Maintaining the equilibrium flux.** Tearing carries a `maintain equilibrium flux`
checkbox (on by default) that adds the static source **S = −η∇²ψ_eq** to the ψ equation,
cancelling the equilibrium's own resistive decay so that the demo shows the instability
and not the layer spreading. ψ_eq,k is extracted once per Reset from the k_y = 0 column of
the uploaded IC — which *is* the equilibrium, every seed here being y-independent and so
living entirely in that column —
by the tiny `srcInit` kernel, and `nlAssemble` then adds −lin_L·ψ_eq,k: the SAME diagonal
the stage applies, so the source follows the η slider with no bookkeeping of its own and
uses η (ψ's coefficient), never ν. Like Pm it is emitted at WGSL-generation time and only
when the preset asks, so every other path keeps its byte-identical kernel text. With the
source on the measured growth rate is the frozen-equilibrium eigenvalue (0.0284 vs
0.028716 at the benchmark, `devtools/checkj.js` §4b) and the maintained ψ_eq is stationary
to round-off; with it off, see the caveat under the table below.

**The nonlinear pair: `tearing: X-point collapse` and `tearing: island chain`** (keys
`collapse` / `chain`; TEARNL, 2026-08-13). Two more presets on
the SAME tearing equilibrium and with no new physics — only the controls move — sitting
either side of the instability's nonlinear fork. All three tearing presets run at
`selRes` 256 and open on **island width + cut trace b_x/b_y**, except `chain`, which has no
island chart (see the broadband seed below) and opens on **cut b + cut u** instead. Both run the maintained-flux source
**off** (neither quotes a rate against a held equilibrium, and a source feeding flux in
would make the collapse externally driven) and auto-diss off (the controller places the
*cascade termination* at the dealias scale and has no term that knows a reconnection layer
exists; on a quiescent equilibrium it drives η to ~1e-9).

- **`collapse`** — the square box at a = 0.1L_x, so Δ′a = **37.8** against the shipped
  preset's 8.4. The island passes the critical width W_c ~ 1/Δ′ immediately and the
  X-point collapses into a sheet, which itself tears and sheds a secondary island
  (Loureiro et al., *PRL* **95**, 235003 (2005)). Its `a / L_x` slider walks Δ′a from 37.8
  down to 8.40 at a = 0.2, which is the shipped preset's own value — 0.2×2π is its
  0.1×4π, and Δ′a depends only on k_y a in this profile, so the two are *exactly* equal
  and the preset contains `tearing` as a slider endpoint.
- **`chain`** — the LARGE box (8π × 8π, i.e. the square one four times bigger, not a long
  thin one), seeded broadband (below), so the layer selects its own k_y out of 24 offered:
  Δ′ falls monotonically with k_y while the resistive layer gets faster, and the two peak
  together at k_y = 1.5, i.e. six islands, which then coalesce in pairs. Expect *roughly*
  six rather than exactly six, and expect it to move with `#nSeed`: γ is flat near its peak
  (modes 4–8 within 9%), so over the linear stage the winner leads its neighbours by only
  ~7% and the phases settle the rest. What the preset actually demonstrates is that the box
  FUNDAMENTAL loses, its γ being 3× down. Square cells are mandatory and not a nicety: the
  Sweet–Parker sheets between merging islands lie normal to y. L_x/a = 53 because an
  island's extent across the sheet grows with its wavelength along it and each merger
  doubles that wavelength — size the box on the LAST merger, not the first.

**Broadband seed.** `rowTear` carries a `broadband seed` checkbox (default off, so the
shipped preset is untouched). On, the seed's y factor is Σ_{n=1..N} cos(nk₁y + φ_n) at
equal amplitude and random phase instead of a single cos(k₁y), normalized so its maximum
over the grid is 1 — which is what keeps the `seed` slider meaning the same physical thing
(peak perturbed flux at the resonant surface) in both branches, the x envelope being sech²
and hence 1 there. N is derived, not a constant: Δ′ > 0 below k_y a = √5 (an analytic
property of the sech² outer solution, independent of a and of the box), and N covers that
band with a quarter of headroom, which is 24 at the `chain` parameters. The phases come
from the page's own `#nSeed` stream, so the IC stays reproducible. The island-width chart
turns **off** for a broadband seed (`icEq.on` stays false): W = 4√(ψ̃/|ψ″|) is a
single-mode formula read off max − min of ψ on the resonant line, and with 24 modes that
extremum belongs to whichever island is largest at that instant — after the first merger
not even the count is fixed.

**Per-field dissipation (Pm).** The 2D linear operator is diagonal per field, so ν (on φ)
and η (on ψ) need not be equal: the **diss slider is η**, and the `Pm` box next to it is
the magnetic Prandtl number ν/η, default 1 (Pm = 0 is allowed — a completely inviscid φ,
which in a long run piles kinetic energy up at the grid scale). It is a compile-time
constant of the stage kernel — at Pm = 1 the emitted WGSL is character-for-character the
scalar-dissipation text every other path uses, and changing it rebuilds the solver like a
resolution change. Exactly two kernels carry it: `stage` (the φ half of the integrating
factor) and `energyPartial` (the kinetic half of the dissipation-rate lane). 3D keeps
ν = η — its 2×2 Alfvén propagator needs an equal diagonal.

**Island width.** The `island width` chart card (2D only) plots W(t) on a log axis, so
the linear tearing stage is a straight line, the Rutherford stage bends over, and
saturation flattens. Near the resonant surface
ψ ≈ ψ_s + ½ψ″(x−x_s)² + ψ̃ cos(k_y y), so the separatrix half-width obeys
½|ψ″|w² = 2ψ̃ and

> **W = 2w = 4·√(ψ̃/|ψ″|) = 4·√((ψ_X−ψ_O)/2|ψ″|)**

with ψ_X−ψ_O the peak-to-peak of ψ along x = L_x/2 and ψ″ **measured** on the
equilibrium profile (a 4th-order second difference at x_s, on the grid the run actually
uses — not the analytic 2ψ₀/a²). The card costs no kernel and no extra round trip: the
cut chart already reads b_x = −∂_yψ on that line at ~10 Hz, and ψ along it is one
spectral integration of that line (`icLineIntegrate`, exact for a band-limited periodic
line where a quadrature rule would be second order; the k = 0 gauge drops out because
only max − min is used).

Its legend carries the same fitted growth rate the k_y mode card does, through the same
helper (`fitLogSlope`): the trailing `MODE_FIT_DT` = 10 t-units of ln W, least squares,
R² ≥ `MODE_FIT_R2`, quoted as **γ = 2 × d(ln W)/dt** because W ∝ ψ̃^½ ∝ e^{γt/2} in the
linear stage. Only the *rise* gate is its own number (`ISLAND_FIT_RISE` = 0.05 against
`MODE_FIT_RISE` = 1.0): the tearing demo runs an order of magnitude slower than KH *and*
plots the square root, so one window of its linear stage rises γ·Δt/2 = 0.14 ln-units,
19× less than KH's 2.67 — the KH gate would blank the legend for the whole run. 0.05 is
`MODE_FIT_RISE` scaled to the tearing rate, keeping the same ~2.7× margin over the gate,
which is still ~1.9× with **maintain equilibrium flux** off (where the measured slope
drops 30–40%). It is a *local* rate by construction, so it falls away from 0.0287 and
eventually blanks as the Rutherford stage (W ∼ t) bends the curve over. The fp64 mirror
(`devtools/checkj.js` §4c) drives the app's own `islandWidth` + `islandFitGamma` off the
pseudospectral solver and gets 0.0278 against the 1D eigenvalue's 0.0287 (3.2%).

**The k_y = 2π/L_y mode.** KH's counterpart of the island chart, and on the same readback:
the `k_y = 2π/L_y mode` card (2D only, so it is off the 3D page exactly as `island width`
is) plots the m = 1 Fourier amplitude of u_x — and of b_x — along x = L_x/2 on a log
axis, and fits γ over the trailing `MODE_FIT_DT` = 10 t-units of *sim* time into its
legend (`γ_fit`, blank unless that window is finite, positive, rose by `MODE_FIT_RISE`
and fits with R² ≥ `MODE_FIT_R2`; a sample count would span wildly different stretches of
t on different devices, since the cut readback is wall-clock throttled). It exists because the **energy trace cannot show
the linear stage at all**: the equilibrium shear carries ~10⁶ times the seed's energy, so
E(t) is flat while the mode grows through six decades. u_x = −∂_yφ and b_x = −∂_yψ are the
two rows of the cut stack with *exactly zero* equilibrium content (the equilibrium is
y-independent with u and B along ŷ), so what is plotted is pure perturbation. b_x is
identically 0 at b₀ = 0, which a log axis cannot carry, so that series is simply absent
until there is something positive to draw. One caveat on the amplitude, not the rate: the
cut line sits midway between the two layers, where the mode is evanescent — A_u there is
0.17 of its peak on-layer value at the preset's defaults (a vortex-sheet estimate,
e^{−k_y L_x/4} = e^{−π} per layer, would say 0.09: the finite-width layer's eigenfunction
decays slower than e^{−k_y|x|}) — the log-y line is offset downward, its slope is not affected
(`devtools/checkj.js` §5 fits 0.26738 off this very line, 0.42 % from the eigenvalue).

**Linear-theory references** (`devtools/eqlinear.py`: a 1D generalized eigenvalue solve
of the linearized system on Fourier differentiation matrices, at k_y = 2π/L_y, converged
in the mode count; `devtools/checkj.js` reproduces each with an independent fp64
pseudospectral 2D run of the app's own ICs):

| case | parameters | γ (eigenvalue) | γ (2D run) |
| --- | --- | --- | --- |
| tearing | η = ν = 10⁻³ (Pm = 1), ψ₀ = 1.65, a = 0.1L_x | 0.028716 | 0.028609 |
| tearing | η = ν = 10^−2.5 | 0.051395 | 0.051300 |
| tearing | η = 10⁻², ν = 10⁻³ (Pm = 0.1) | 0.114636 | 0.11401 |
| tearing | η = 10⁻³, ν = 0 (Pm = 0) | 0.043646 | 0.043373 |
| KH | b₀ = 0, U₀ = 1, a = 0.05L_x, ν = 10^−3.5 | 0.266260 | 0.26657 |
| KH | b₀ = 0.5 U₀ | 0.206229 | 0.20377 |
| KH | b₀ = 1.2 U₀ | 0.003646 (resistive residue) | decaying |

Those rates are for a **frozen equilibrium**, which is what an eigenvalue problem assumes
— and what `maintain equilibrium flux` reproduces in the demo (0.0283 free-running with
the source on, 1.3 % off the eigenvalue). **With the source off** ψ_eq also diffuses at
~η/a² and slowly lowers Δ′: over t = 30…80 at η = 10⁻³ the measured tearing rate is
≈ 0.018, some 37 % below the frozen-equilibrium value. KH has no such source and does not
need one (ν/a² ≪ γ there: 3.6 %).

## Dissipation: the slider's range, and auto-diss

The perpendicular operator is −ν·k⊥<sup>2·hyper</sup>, so the **diss** slider carries
log₁₀ ν. Two grid-only numbers set everything here (`common.js`, beside `ctrlDissRow`):

- **k_d** = `DISS_KD_FRAC`·(largest retained k⊥). In this app that cutoff is the 2/3
  dealias, which is exactly what the ⊥ spectrum is binned out to, so k_d = 0.6·nb·kunit.
- **ν_marg** = k₁<sup>1/3</sup>·k_d<sup>2/3−2·hyper</sup> — the marginal coefficient for
  an O(1) box-scale amplitude: walk a Kolmogorov u(k) = u₁(k/k₁)<sup>−1/3</sup> from
  u₁ = 1 at k₁ = kunit down to k_d and set the dissipation rate there equal to the
  nonlinear rate. It is the old "auto" button's formula with ε<sup>1/3</sup> → u₁k₁<sup>1/3</sup>,
  i.e. *without* its dependence on the forcing sliders.

**The slider is a demo instrument** (FEEDBACK item 7). Its range is recomputed from the
live hyper / resolution / box by `dissRangeSync` (called from `syncCommonLabels`, so every
path that can change them ends in it): the top is Re ~ 1 at the box scale,
ν_top = k₁<sup>1−2·hyper</sup>; the bottom is `DISS_DECADES_BELOW` = 3 decades under
ν_marg, where the cascade no longer terminates inside the retained band. One sweep
therefore goes viscous → turbulent with an ε-independent dissipation rate → numerically
unresolved. A range <input> sanitizes assigned values against min/max/step, so
`presetWrite` **opens** the range to the hard bounds before writing and the following
`syncLabels` narrows it again — widened outward around whatever is stored, so a re-range
never moves the physical value, and every value written from code goes through
`dissWriteLog` (quantized to the slider's own 0.05-decade step, clamped, live path only).

**Auto-diss is a tickbox, ON by default** (FEEDBACK item 6), not the old one-shot button.
It sets ν continuously from the *measured* amplitude near the dissipation scale, so it
needs no assumption about where the energy came from — which is what makes it work for
KH and tearing as well as for a forced run. The rule is Jono Squire's rmhd-gpu
(`rmhdgpu/auto_dissipation.py`), ported with its structure and defaults: E_d = kinetic +
magnetic energy in the logarithmic shell k_d·e<sup>∓0.5</sup>, u_d = √(2E_d),
ν_target = u_d·k_d<sup>1−2·hyper</sup> (i.e. ν k_d<sup>2n</sup> = u_d k_d: the cascade
terminates at k_d), relaxed in log space by 0.2 per update with the change capped at a
factor 2. Three app-forced deviations: the cadence is wall-clock (`AUTODISS_PERIOD`,
2 Hz — the loop's step count per frame is not ours to choose, and every other readback
hook here is wall-clock too); [ν_min, ν_max] *is* the slider's dynamic range above; and a
shell with no measurable energy **holds** ν rather than driving it to ν_min, because a
quiescent or freshly-seeded IC has nothing at k_d yet. E_d comes off the app's own ⊥
spectrum bins — the same kernels the spectra card uses. With a spectrum card open the
controller rides that card's cached readback (the cards refresh faster than the
controller updates, so a fresh one almost always exists); only with no card open does it
take a readback of its own — so there is no new kernel and no new buffer, and at most
one spectrum pass is ever added.

While ticked the slider is disabled and *follows* the controller, so the level is always
visible and unticking simply leaves the manual slider where the controller left it.
Changing **hyper** makes the controller jump straight to the new exponent's target (one
fresh measurement, no relaxation cap — the cap is for measurement noise, not for a
re-parameterized operator); with the box unticked a hyper change never touches your
slider. The
pure core (`autoDissShellE` / `autoDissTarget` / `autoDissRelax`) takes state in and
returns ν, which is what `devtools/checks.js` §6 drives — including a closed-loop cascade
model whose fixed point is ν_marg to within a decade. Before the first measurement a
preset that wants the app to choose gets `autoDissSeed()` = ν_marg; the one preset that
turns the controller **off** is 2D `tearing`, whose whole point is a quoted η against a
linear reference growth rate (the same argument that makes it lock hyper).

## Forcing controls

ε⁺ and ε⁻ are separate log sliders with a lock checkbox (on by default); they are in the
same units — each is a contribution to dE/dt, so the total injection rate is their sum
(`rmhd._forcing_scale_from`'s convention). Unlocking them drives an imbalanced cascade.
The scale that realizes those rates solves `½·F₂·dt·s² + P·s = ε` for its positive root
(`shared_physics.selfnorm_scale`, mirrored in `physics.js` `scaleWGSL`), so a quiescent
start injects `ε·dt` on the first forced step (the second step overall — the O-U
envelope is still zero on step one) instead of an ε-independent kick; `smax` is only a
last-resort clip. Derivation and the dated behaviour change: `docs/numerics.md`.
The **band n** pair of handles sets the forcing shell [n_min, n_max) in units of the box
wavenumber; they cannot cross. The band is baked into the grid (`fmask`) *and* into the
OU kernel's `NS`, so changing it triggers the ordinary rebuild path on handle release —
which is why it is an `onchange`, not an `oninput`. Nothing about the generated WGSL
changes except the `NS` constant, exactly as a resolution change does.

### Spacebar blob forcing (2D, prototype)

`cbBlobForce` replaces the OU shell with forcing the visitor drives: keydown spawns one
gaussian blob at a random position in a randomly chosen Elsasser channel, the hold grows
its σ, keyup tapers it off and drops it (a 60 ms cosine attack so there is no step in the
RHS, a 1200 ms σ ramp, a 120 ms cos² release — all wall clock, because what those three
shape is the interaction and not the physics). At most `BLOB_FORCE_MAX` = 8 live blobs per
channel; an overflowing channel loses its **oldest**, never the one being held. `step`
dispatches `blobBuild` *instead of* `ou` + `scale`, so while blob mode is on the ε
sliders, `tau` and the band drive nothing at all (the band handles still trigger the
ordinary rebuild on release — they just no longer force anything).

`solver2d.js: setBlobs` has a cap of its own, and **its overflow policy is the opposite
one**: it fills in list order and drops what does not fit, i.e. keeps the *oldest*, where
the page evicts the oldest so the blob being *held* survives. The two can only disagree on
a list the page cannot produce — `frcBlobSpawn` evicts at spawn time, so every channel it
hands over already holds ≤ `BLOB_FORCE_MAX` and `setBlobs` never truncates in production.
They are left different rather than reconciled because they answer different questions:
the page's is "which blob does the visitor lose", `setBlobs`' is "what does a caller that
overruns a fixed buffer get", and first-N-in-list-order is the honest answer to the second
(and the one `devtools/checkblob.js` leg 2 pins).

**It is built analytically in k space, which is why there is no FFT, no RNG and no
hermitian symmetrization.** A real-space gaussian on the potential,
f = P·exp(−|x−x₀|²/2σ²), transforms to P·2πσ²·exp(−σ²k²/2)·exp(−i k·x₀).
`physics.js: blobBuildWGSL` evaluates the two k-dependent factors per mode and sums the
slots; `solver2d.js: setBlobs` folds everything else into each slot's weight
w = P·2πσ²·(nx·ny)/(Lx·Ly) — the continuous transform at k = 0, times the unnormalized
DFT's nx·ny, divided by the cell area. exp(−i k·x₀) *is* by construction the transform of
a real field, so the rfft2 reality constraint on the ky = 0 / ky = Nyquist columns holds
exactly and none of `drawNoise`'s mirror machinery (the one that divides by √2, not 2) is
needed: that machinery exists because the OU path *draws* its noise, and a drawn field has
to be made real by hand. **The dealias mask is applied inside the kernel** because a
gaussian is not compactly supported in k and `forcingAdd` — unlike `nlAssemble` — does not
dealias, so nothing downstream would trim the tail.

**The power normalization is deliberately bypassed** (`sc[4] = sc[5] = 1`, written by
`setBlobMode` and again at the end of `_uploadIC`, which runs `scale` on the way past).
`scaleWGSL` solves for the scale that makes the step inject a *target* rate, so
normalizing a blob would divide the user's amplitude straight back out — the slider would
stop meaning anything and every blob, loud or quiet, would inject the same ε. The
consequence is real and is the honest price of the knob: what the step injects is that
same left-hand side evaluated at s = 1, `P + ½·F₂·dt` with P = ⟨∇z·∇f⟩, so a blob landing
against the flow already there has P < 0 and **takes energy out**, and the total energy is
pinned to nothing. Blob mode is an instrument for looking at 2D RMHD by hand, not a
controlled-ε turbulence driver; the OU path remains the one with a quotable injection
rate. The forcing buffer is cleared on
either mode transition, because the modes it holds mean nothing in the other mode and a
leftover OU envelope would land on the fields whole once the scales are at 1.

**The `sc[4] = sc[5] = 1` line at the end of `_uploadIC` is load-bearing, and deleting it
blows the run up rather than quietening it.** Work out what `scale` — which `_uploadIC`
dispatches on the way past — writes on *that* state: the scalars were just zeroed, so
dt = sc[0] = 0, and the forcing buffer was just cleared, so P = F₂ = 0. `selfnormScale`
then takes its `F2*dt == 0` branch, `s = tgt/P` = 2ε/0, and the closing
`clamp(s, −smax, smax)` delivers **smax = 1e4** at the shipped default. Blob mode never
runs `scale` again, so that 1e4 would stand for the whole run: with `forcing` ticked,
pressing **Reset** would multiply every blob by 1e4 from the second step on (the first
still sees the cleared forcing buffer). `s = 0` is only the `tgt == 0` short-circuit,
i.e. only with `forcing` *unticked* — so the failure hides in exactly the configuration
the page's hint tells a visitor to use, and appears the moment they tick the box.

**The amplitude convention** follows the blob editor's: the slider is the peak forcing
rate max |∇f|, and the potential peak that realizes it comes from `common.js: icBlobPeak`
(P = amp·σ·√e, since |∇f| of a gaussian peaks at P/(σ√e)). So growing σ inflates the blob
— more area, more energy per unit time — without changing what it forces the flow at,
which is what makes the hold-to-grow gesture read as "bigger" rather than "louder".

**Polarity selects a buffer half, not a sign.** The blob array is z⁺ slots then z⁻ slots
and the kernel reads `blobs[half*NBLOB + b]`, so the two channels are independent lists
and a z⁻ blob is not a negated z⁺ one. What z⁺ forcing *is* follows `forcingAdd`:
`rhs_φ += ½(f₊+f₋)`, `rhs_ψ += ½(f₊−f₋)`, so a z⁺ blob alone forces φ = ψ, i.e. an
Alfvénic perturbation with u aligned with b, and equal z⁺ and z⁻ blobs cancel in ψ and
leave a pure velocity vortex. The random polarity per press is what gives a visitor both
channels without a control for it.

**`blobMode` is per instance and defaults FALSE**, and what that default protects is
**every `Solver` nothing calls `setBlobMode` on**. The page's live solver is not one of
them: `rebuild()` calls `solver.setBlobMode(frcBlobOn())` on every build, so it takes its
mode from the checkbox and never keeps the constructor's. The ones that do keep it are
`runSelfTest`'s own `new Solver(…)` and the extra-grid solvers `?bench`'s FFT ladder
constructs. The self-test is where it bites: it steps 2600 times through
`ouInjectionRow`, measuring dE/dt + ⟨D⟩ against ε, and in blob mode that solver would
dispatch `blobBuild` over an all-zero blob list instead of `ou` + `scale` — i.e. run with
**no forcing at all**, and report ≈ 0 against ε = 1.

*Measured, 2026-08-30* (this is a correction: the paragraph here previously claimed
`devtools/checkbench.js` would catch a flipped default, and it does not). Setting
`this.blobMode = true` in the constructor and rerunning the gates: `checkbench` is
**byte-identical to baseline, every row green** — its wiring leg drives the *page's*
solver, which `rebuild()` has already put back in OU mode, and it never ticks
`cbBlobForce`. `checkblob`, `checkidle`, `checkonepage` and `checkgc` are byte-identical
too, and `checkzip` differs only in the temp directory name in its own log. Of the gates
rerun under that mutation, the one that fails is **`checksolver2d`**, and it fails as a
text pin (`BLOB_SOLVER` records the line `this.blobMode = false;`), not as a statement
about behaviour. So: the default is protected by a text pin and by the self-test on a
real device, and by nothing that runs in node against the shipped page.

**The σ window is `[max(3·max(dx,dy), min(Lx,Ly)/64), (slider)·min(Lx,Ly)]`, both ends off
the SHORT box dimension and the COARSE cell** — a blob is round and the box need not be
(`wide` is 4π × 2π at 256 × 64, so dy = 2·dx there).

*The floor.* Three cells of the **coarser** axis is the physical bound: a narrower
gaussian puts its spectral tail past the 2/3 dealias cut, where the forcing would be
aliased rather than resolved. At σ = 3·dy the factor exp(−σ²k²/2) at the cut is
exp(−2π²) = 3e−9; at the 1.5·dy that `3·dx` used to give on `wide` it is 7e−3, which is
not a tail. The min(Lx,Ly)/64 term takes over at 256² and above, where three cells is a
needle and the blob would be invisible.

*The ceiling.* The amplitude convention (below) is exact for an **isolated** gaussian,
and what breaks it is the periodic images adding. Measured max|∇f|/amp with the slider
at its top: **0.979** on a square box (2% low — the images subtract at the peak-gradient
radius), and **1.207** on `wide` while the ceiling was Lx/4 = Ly/2, i.e. 21% *above*
what the slider claims. Off the short side it is **1.001** there. So min(Lx,Ly) makes
the convention hold to ~2% on every box at every setting; it does not make it exact, and
no choice of ceiling would. `max σ` is clamped up to the floor, so a slider below it
simply gives a blob that does not grow.

**What stays OU-flavoured while blob mode is on** — accepted, not bugs, but do not be
misled by them:

- the **spectrum card still draws its dashed forcing-shell markers** from
  `solver.p.fshell` (the `fshell` it is handed at `common.js`'s card refresh), and still
  draws its fit line from `kA = fitKA(nb, fshell)` — the upper band handle — to the last
  bin, using that same kA as the amplitude anchor's fallback. The real injection scale
  is 1/σ and moves with the `max σ` slider; neither the markers nor the fit's left end
  follows it.
- the **readout still prints `s+` and `s-`**, which in blob mode are the pinned 1 rather
  than a normalization that means anything.
- a **`?bench` run reports the OU dispatch set**: `benchSpec2D`'s `stepIO` byte table is
  a static list, `blobBuild` is not in it, and `ou`/`scale` sit in `checkbench`'s
  `UNCOUNTED`. Bench the OU path and read the numbers as the OU path's.

**The page owns the envelope, the key handling and the readout line**; the solver's only
new surface is `setBlobMode` / `setBlobs`, and the list reaches it through the existing
`frameHook`. It is therefore **one frame stale** by the time a step sees it — accepted,
16 ms against a 1200 ms growth ramp, and the alternative is a per-step CPU hook on a path
that has none. `setBlobs` both packs and *uploads*, because a list constant over a whole
frame has no business being re-sent at every step. (The listeners go on `window`, never
`document`: `devtools/stubenv.js` gives only `window` an `addEventListener`, so a
`document` listener would throw at boot and take every page-booting gate with it.)

`common.js` carries exactly one thing for this feature, and it is the **run manifest**:
`_forcingRecord` branches on `solver.blobMode` — the same flag `step` branches on, so the
manifest cannot disagree with the kernels that ran — and records the blob parameters plus
an explicit *"interactive, NOT reproducible from these parameters"*. Keyed on `cbForce`
alone it described the wrong forcing twice over: ticked it quoted an ε/shell/τ that drove
nothing, and **unticked — the configuration the page's own hint recommends — it wrote
`{"on": false}` for a run that was being forced the whole time**. That record ships inside
both `save all` and `save fields`. `rmhd3d.html` has no blob mode, so `solver.blobMode` is
`undefined` there and its manifest is byte-for-byte what it was. The values come from the
page through the ordinary hook idiom (`manifestForcing`, alongside `frameHook` /
`readoutExtra` / `specExtra`); with no hook the record still says blob, still says the OU
numbers were not running and still says the run is not reproducible.

**The spacebar is shared with the browser.** Every Space press the page *takes* is
`preventDefault`ed and the first blurs whatever has focus — the visitor gets here by
clicking the green **Run** button, which then keeps focus, and Space must force the flow
rather than pause it. Two classes of control are exempt from that: a text or number field
being typed into (and the IC editor, which owns the keyboard outright), and **a focused
checkbox or radio**, where Space is the activation key *and* the only keyboard way to work
the control. That second exemption exists because without it `cbBlobForce` was a one-way
door: Tab to it, Space (the box was still unticked, so the handler let the press through
and the browser ticked it), Space again — and now blob mode was on, the press was
swallowed, focus was blurred and a blob appeared. **The box could never be unticked from
the keyboard.** Buttons are deliberately *not* exempt, so the click-Run-then-hold-space
gesture still works; nothing is lost, since Enter activates a focused `<button>` too and
this handler only ever looks at Space.

**Verifying it.** The coefficient was checked against an exact forward DFT of the analytic
gaussian, evaluated over every dealiased mode: 3.2e-7 relative, which is the fp32 floor —
that is the check to redo if the weight, the DFT normalization or the σ convention moves.
The standing *physical* check needs no tooling and is the one to run after any change to
`blobBuild` or `forcingAdd`: with a quiescent IC, one blob and one blob only is an exact
stationary solution of 2D RMHD (ζ⁻ = 0 leaves nothing to advect ζ⁺), so it must sit still
and decay; add one of the opposite polarity and the pair must tangle immediately. A lone
blob that stirs means polarity is leaking across the (φ, ψ) map — with the one exception
that at Pm ≠ 1 the two potentials damp at different rates, so a lone ζ⁺ blob grows a
little ζ⁻ of its own and eventually moves. Run that check at the default Pm = 1.

### The impulsive kick and the drag (2D solver; MUSIC_PLAN M1, no page uses them yet)

Two additions to `solver2d.js` for the music page `plans-webgpu/MUSIC_PLAN.md` plans. Both
are **off by default** — `mu: 0`, `kick: false` in the `Solver` parameters — and
`rmhd2d.html` asks for neither, so its kernel list, every kernel's text and every number
it computes are what they were (`devtools/checksolver2d.js` records the new JavaScript as
its fourth allowance and still pins the WGSL byte for byte; a dump of both pages' kernels
is identical before and after, and so are the dispatch log and every uploaded byte of a
solver that is never kicked).

**`kick(list)` is an impulse, not a forcing.** `list` is `setBlobs`' — `[{x, y, sigma,
amp, pol}]`, `pol < 0` the ζ⁻ channel — but `amp` is the peak |∇ζ| the blob **adds** to
its channel, an increment, where `setBlobs`' is a rate. One call packs the list
(`_packKicks`), uploads it and dispatches `blobKick` once over `ceil(NM/64)` workgroups:

    φ += (ζ⁺ + ζ⁻)/2,   ψ += (ζ⁺ − ζ⁻)/2

with ζ± the sum of that channel's blobs, evaluated analytically in k space exactly as
`blobBuild` does (same weight, `icBlobPeak(amp, σ)·2πσ²·nx·ny/(Lx·Ly)`). It then
refreshes the energy scalars (so a stats readback sees the post-kick energy), bumps
`solver.nkicks` and marks the CFL stale. Nothing else is written — not `t`, not the
forcing state, not `blobMode`, not any other buffer. Call it between steps.

- **The amplitude is an ELSASSER increment.** u = (ζ⁺ + ζ⁻)/2 and b = (ζ⁺ − ζ⁻)/2, so a
  lone ζ⁺ blob of amplitude `amp` has peak |u| = peak |b| = **amp/2**. Only the same blob
  in both channels is a pure velocity kick of peak `amp`.
- **A kick does not move `nsteps`.** Anything that keys a redraw or a cached readback on
  `nsteps` alone (`common.js`' `stateMark`) will miss it: key on `nkicks` as well.
- **A kick does not refresh dt, so the next step does.** `step(cflEvery)` recomputes dt
  only when `nsteps % cflEvery == 0`; after a kick `sc[0]` still holds the PRE-kick dt,
  and after silence that is the quiescent floor `cfl·dx/0.1` — about 10·amp times what
  the kicked flow allows. `kick` therefore sets `_cflStale`, and the next `step` runs the
  CFL reduction whatever its `cflEvery` says, then clears the mark. One bit: two kicks
  before a step force one recompute; an empty or refused kick marks nothing; a solver
  that is never kicked steps on the unchanged schedule. **What this does not fix:** the
  dt a caller *read back* before the kick is still the old one, so a step count derived
  as `ceil((target − t)/dt_last)` right after a kick over-estimates how far each step
  goes (MUSIC_PLAN's clock formula — re-read dt after the first post-kick step).
- **Kick energy enters no budget accumulator.** `sc[7]` integrates the dissipation rate
  and nothing integrates what a kick adds; the energy jump of a kick is the difference of
  two energy readbacks, before and after.
- **Only for a solver built with `kick: true`.** `buildShaders` emits `blobKick` behind
  `g.kick`, the way `srcInit` sits behind `eqsrc`; without the flag there is no kernel,
  pipeline, bind group or buffer, and `kick()` throws.
- **Its own table, sized for ONE onset.** `KICK_MAX` = 32 slots per channel
  (`physics.js`) in its own `kicks` buffer; `BLOB_FORCE_MAX` = 8, `setBlobs` and the
  spacebar mode are not involved. The music map's densest band needs 15 blobs per channel
  at 256² and 30 at 512², so **call `kick` once per onset**: two onsets of one frame merged
  into one list can need 60 and would lose the overflow.
- **What it does with a bad list.** EVERY blob is validated before anything is packed,
  wherever it sits in the list and whether or not its amplitude is zero: a non-finite
  `x`/`y`/`amp`, a `sigma` that is not a finite positive number, or a blob whose packed
  **fp32 weight** is not finite (`sigma: 1e13` is a finite number and an infinite weight)
  throws, naming the blob, with the table untouched and nothing dispatched. A NaN or
  infinite weight would poison every mode of both fields permanently — a kick has no next
  step to overwrite it. `x` and `y` are packed **modulo the box** (same blob on a periodic
  domain; it keeps the kernel's phase k·x₀ bounded and accurate in fp32 — a position
  already in [0, L) is packed bit for bit). `amp == 0` is then skipped and takes no slot.
  More than `KICK_MAX` in a channel keeps the first `KICK_MAX` in list order and **drops
  the rest, reporting the count**: `kick` returns `{used: [n⁺, n⁻], dropped}`. An empty
  list dispatches nothing.
- **Two classes of mode are left untouched bit for bit**, by an early return rather than
  by adding zero: k = 0 — which is *inert*, not zeroed: whatever the fields hold there
  stays, the kick just adds nothing to the mean of a potential — and everything outside
  the 2/3 cut (a gaussian is not compactly supported in k, and the fields are dealiased
  everywhere else).
- **`@workgroup_size(64, 1, 1)` is spelled out on purpose.** It is the same workgroup as
  every other kernel's `(64)`, but devtools' WGSL interpreter reads the one-argument form
  digit by digit — as a 6 × 4 workgroup, each invocation id run four times. A kernel that
  *writes* its output cannot tell; this one *accumulates*, so the short form would make
  `checkkick.js` see four kicks. Do not "tidy" it.
- A lone ζ⁺ kick is still a stationary solution (previous section): a fresh blob only
  starts to distort where it overlaps the other channel.
- **The OU forcing is still there unless the page turns it off.** A kick solver steps the
  ordinary OU path (`ou` + `scale`), and the `Solver` defaults are `epsP = epsM = 0.15`.
  The music page must construct with `epsP: 0, epsM: 0`, or the shell forcing and its
  lagged scales run underneath the kicks.

**The drag is one column of `gridB`.** `mu` is the **energy** e-folding rate of a uniform
linear drag: every mode's amplitude decays as e^(−μt/2) on top of the hyperviscosity.
`_dragLin` subtracts μ/2 from the linear diagonal the kernels already read
(`gridB.y = −diss·k^(2·hyper) − μ/2`), so `stage`'s exact exponential carries it and **no
WGSL changes** — a solver with `mu ≠ 0` compiles the same shader modules text for text.
It is applied to `makeGrid`'s column in the constructor and to the rebuilt one in
`refreshDissipation`; `setDrag(mu)` changes it on a live solver through that same
rebuild (`makeGrid` itself is unchanged).

- **What "off is bit-identical" covers, exactly.** At `mu = 0` `_dragLin` returns before
  touching anything, so *construction* gives `makeGrid`'s column bit for bit and
  *`refreshDissipation`* writes exactly what it always wrote. **`setDrag(0)` on a fresh
  solver is not a no-op on the bits**: it rebuilds through `refreshDissipation`, whose
  long-standing formula takes k² from `gridA`'s fp32 where `makeGrid` uses fp64, so the
  column can move by one ulp — the same thing any dissipation-slider move has always done.
  Nothing calls `setDrag` unless a page does.
- **The dissipation readout becomes the total sink.** `energyPartial`'s rate lane is
  `−gridB.y·(e_k + e_m)`, so with the drag on `sc[6]` is hyperviscous dissipation **plus
  μE**, and `sc[7]` its time integral. That is the honest number for an energy budget;
  label it as such wherever it is shown.
- **Refused, loudly, instead of being silently wrong:** `mu ≠ 0` with `pm ≠ 1` (`_dissLin`
  multiplies the φ half's exponent by Pm, which would scale the drag on φ too) and
  `mu ≠ 0` with `eqsrc` (the maintained-flux source is `−gridB.y·ψ_eq`, which would then
  hold the equilibrium up against the drag as well -- unless the page asks for exactly
  that with `eqsrcDrag: true`, as the music page's current sheets do). Also a negative, non-finite or
  non-number `mu`. A refused `setDrag` leaves `p.mu` and both copies of the column as
  they were.
- It damps **every** mode, k = 0 and the modes past the cut included (the latter are zero
  anyway; the former is inert).

**The gate is `devtools/checkkick.js`** (201 rows): the emitted `blobKick` executed byte
for byte against a real-space DFT written in the gate *and* against
`devtools/fixtures/kick_offline.json`, which `gen_kickfixture.py` records from the
offline reference implementation itself (`make_kick` in `examples/music_turbulence.py`,
fp64); the two untouched classes, additivity and the Elsasser map at the bit level;
`kick()` on the stub device with the queue *and the command encoder* wrapped (one upload,
one pass begun and ended around three dispatches, one submit, no `clearBuffer` or copy,
and a snapshot showing nothing else about the solver moved); kick-then-step dispatch
logs for the stale-CFL mark; and the drag as the column, the refusals, and the decay it
produces through the real `stage` kernel — at hyper 1, 2, 3 and 4, on a rectangular grid,
and in the music configuration itself (`kick: true` with `mu`, hyper 2). Regenerate the
fixture with

    python webgpu/gen_kickfixture.py

Its σ are all ≤ 0.5 on purpose: `make_kick`'s gaussian is minimum-image and the kernel's
analytic form is the fully periodic one, which differ by e^(−(L/2)²/2σ²) of the peak —
3e-9 at σ = 0.5 but 3e-6 at 0.625 (9e-4 on the weakest retained modes). The music map's
widest blob is 0.45. **Nothing here has run on a real GPU**: the stub records what was
encoded and the interpreter executes the kernel text.

## The music page (`music.html`; MUSIC_PLAN — deployed, not yet linked from the other pages)

Whatever is playing drives a 256² 2D RMHD run: an onset in a frequency band is one
impulsive kick of gaussian blobs (previous section), and the audio clock sets the
simulation time. Four sources: the built-in click track, a local audio file, another
tab's audio, and an audio input device. It is the live twin of the offline
`examples/music_turbulence.py`, which stays the reference and the way to make a movie.

**Serve and open** (it loads from `file://` too, but a local server is the tested way):

    cd webgpu && python -m http.server 8766
    # then http://localhost:8766/music.html            (add ?musicdebug to keep a record)

**What the page fixes** (`uiParams`): N = 256, a 2π box, hyper 4 (k⁸), Pm 1, no equilibrium
source, **`epsP = epsM = 0`** (no OU forcing — the `Solver` defaults are 0.15), a solver
built with `kick: true` and `mu = 1/(memory·rate)`, and `diss = 48·amp/(n/3)⁸` (rate 48·amp at the dealiasing cutoff), the offline
rule. The initial condition is rest. Its controls are few: source, Start/Reset, threshold
(0.3), memory (1 s), amp (1), rate (1), the click track's bpm, the sync trim (0), the
input device, and the two CFL knobs the
shared loop reads. The `Solver`'s drag is set from `memory` in seconds of **audio**, so
changing `rate` changes μ too.

**`music.js` is the logic, and it is pure.** It names nothing from the DOM, Web Audio,
WebGPU or the three shared scripts, so node evaluates it on its own — that is how
`devtools/checkmusic.js` runs it. Every top-level name starts `music` / `Music` /
`MUSIC_`.

- `musicBandPlan(n, nbands?, sigmaMin?, nblobMax?, Lx?, Ly?)` (`n` a number or `[nx, ny]`;
  the box defaults to the square 2π one) is the offline `band_plan`: widths
  log-spaced from 0.45 down to 3.7 grid spacings, as many as fit with a factor 1.8 between
  neighbours (4 at 256²), blob count ∝ 1/σ from 3 in the bass band — 3, 5, 9, 15 at 256².
  The page passes `KICK_MAX` as `nblobMax`, so a plan one `kick()` cannot hold is refused
  at boot (n = 1024 would need 59).
- `MusicDetector` is a line-for-line port of the offline `CausalOnsetDetector` (the
  per-frame core `detect_onsets_causal` was refactored around for this). `push(mag, tNow)`
  takes one magnitude spectrum per frame and returns the onsets it can now declare — one
  frame after their peak, so sound → declaration is half the analysis window plus a frame,
  **40 ms** at 44.1 kHz / 2048 / 60 Hz. Its sums are taken in numpy's pairwise order and
  its percentile is numpy's linear one, which is what lets the gate demand the *same*
  onsets as the python and not merely similar ones.
- All its windows are counted in **frames at a nominal 60 Hz**, so `musicShouldPush` feeds
  it at about that rate whatever the display does: every frame at 60 Hz, every other at
  120. **Below 60 Hz it loses onsets** (M0.b: recall 0.69 at 60 Hz, 0.44 at 30); moving it
  to an `AudioWorklet` with a fixed hop is the fix if a device needs it.
- A gap longer than 0.1 s between pushes (a hidden tab) only re-primes the detector: the
  flux across a gap is the music having moved on, not an onset.
- `musicKickList` is the list ONE `Solver.kick` takes for one onset: the band's blobs for
  ζ⁺, then an independent second set for ζ⁻, each of amplitude `amp × strength` with a
  random sign, at positions from a seeded generator (`MUSIC_SEED`, mulberry32 — not the
  offline script's numpy stream, so positions differ from the movies').
- `musicClickTrack` is the built-in source: a kick on every beat, a snare on beats 2 and
  4, a hat on every off-beat, 16 beats, **looped** — every sound's tail wraps round the
  end of the buffer, so the loop has no seam. Unlike the offline click track it has no
  kick-less closing quarter.

**One frame** (`musicStep`, called by the shared loop through `stepHook`):

1. read the analyser (`getFloatFrequencyData`, magnitude = 10^(dB/20)) and push it;
2. **one `solver.kick()` per onset** — never a merged list: `KICK_MAX` holds one band's
   blobs, and two onsets in one list could need twice that;
3. ask `musicClock` how many steps the audio clock wants, and return that.

The loop then takes those steps, renders, and reads the stats back; the next frame's
count is computed from that readback. Kicks therefore always precede their frame's steps.

**The clock** (`musicClock`, pure). Target simulation time is
`t0 + rate·(audioT − a0)`; the step count is `ceil((target − simT)/dt)` capped at 64 per
frame, from the newest `simT` and `dt` a readback has returned. Three things it does that
the bare formula does not:

- **It never chases.** More than 0.25 s of audio behind (or ahead), a clock that ran
  backwards, a change of `rate`, a Reset, the first frame after a stop or a long absence:
  it *rebases* — the anchor moves to now and no step is taken to make up the difference.
  Falling behind lights the BEHIND indicator for a second.
- **It does not run ahead in silence — within a bound.** At rest the CFL dt is its floor,
  `cfl·dx/0.1` — 0.098 at 256², six frames. Rounding up would put the simulation that far
  ahead of the music, where it must wait; so a step that would overshoot by more than one
  frame (`maxAhead`) is not taken yet and `n` rounds down instead — but only while the
  simulation is less than half of `lagMax` behind. Past that the step is taken (and being
  one step ahead is allowed: the ahead-rebase threshold is `lagMax` plus one step).
  Without the bound, at `rate` < 0.37 the wait outlasted `lagMax` and the clock rebased
  for ever without stepping. (The first step after a
  start is the one exception: dt is unknown until a step has been made.)
- **A frame that kicked takes at least one step**, behind the clock or not, even
  where the dt in hand says none fits. That dt is a PRE-kick one (too large — a kick only
  speeds the flow up), and the solver recomputes dt on that very step (`_cflStale`), so the
  kicked flow moves at once and the next frame has a dt worth dividing by. Nothing loops
  on a stale dt: its count can only undershoot, and the next readback corrects it.

**The hook in the shared loop.** `common.js` gained `stepHook(solver)`: null on every other
page, where the loop is the loop it always was (`n = stepsPerFrame`). When set it is
awaited once per running frame, before the steps, and returns the count; a solver
replaced while it awaited gets no step. The adaptive `stepsPerFrame` controller is left
exactly as it is and simply goes unread on this page — the step count here follows an
outside reference, which is the thing LOOPLAT_PLAN's post-mortem says every controller
that failed was missing. `stateMark()` appends `/k<nkicks>` when a solver has been
kicked, so a kick re-opens the stats cache and the cards' redraw; the string is unchanged
on a solver that never is. `_forcingRecord` knows a kicked run (`g.kick`), so a saved
`params.json` says `forcing.kind = "kick"` with the page's settings instead of
`{"on": false}`.

**Four sources** (the `source` select; `musicAcquire` obtains each, `musicStart` builds the
same graph round whatever it returns):

| source | how | may the page play it? |
|---|---|---|
| click track | `musicClickTrack` into a looping buffer | yes |
| audio file | `<input type=file>` → `decodeAudioData` | yes |
| another tab's audio | `getDisplayMedia(musicTabConstraints())` | only if the track reports the tab silenced |
| audio input device | `getUserMedia(musicInputConstraints(deviceId))` | **never** |

- **The graph.** `source → analyser` always, and the analyser is a dead end: the detector
  reads the signal UNDELAYED. Where the page may play the source there is a second
  branch, `source → DelayNode → speakers` — the only place in the page anything is
  connected to `ctx.destination`. `musicRouting(kind, suppressed, surface)` is the rule
  for who may be played, as data.
- **Sync is automatic, from the audio output's latency** (`musicSync`, fix round). Two
  delays stand between a hit and the visitor that the page did not choose: the detection
  latency `MUSIC_LDET` (39 ms measured from onset to `kick()`, plus a frame: 56 ms) and
  the output latency `AudioContext.outputLatency` (≈ 0 on a laptop's speakers; **0.168 s
  measured on Bluetooth earphones**). And two it can add: a delay D on the *sound* (the
  DelayNode, only where the page plays the source) and a hold-back H on the *kicks* (they
  wait in `music.queue`). A hit is heard at D + Lout and seen at Ldet + H, so with
  `v = Lout − Ldet + trim`: v ≥ 0 → hold the kicks (H = v, D = 0); v < 0 → delay the
  sound (D = −v, H = 0). On speakers that is "sound delayed 56 ms, kicks held 0 ms"; on
  those earphones "sound delayed 0 ms, kicks held 112 ms". A tab the page does not play is
  heard through the same output undelayed, so only H is available (`max(0, v)`); an input
  device's sound is already in the room, so Lout does not apply and the picture runs Ldet
  late. The mode line under the sliders states what is in force, with the output latency
  it was computed from, and says by how much the picture still trails where the sound
  could not be delayed.
- **The kick queue.** Every onset is queued with a due time (declared + the H then in
  force) and kicks on the first frame at or after it — still one `kick()` per onset, still
  before that frame's steps, in the order declared (the head of the queue first, so a
  shrinking H cannot reorder them). Stop, a change of source, Reset and the return from a
  hidden tab empty it; a rebase for falling behind does not (due times are audio times).
  **The clock's target is not shifted**: simulation time's origin is arbitrary, and a
  constant offset in audio time changes nothing about the tracking — only *when* a kick is
  applied moved.
- **The output latency is re-read once a second** (earphones can connect mid-session): D
  moves by `setTargetAtTime` (a jump in a delay line clicks), H applies to onsets declared
  from then on. A value the browser does not report (undefined, NaN) is taken as 0.
- **The slider is a "sync trim"**, ±150 ms around the automatic value: positive makes the
  picture later, negative the sound later; it comes off whichever side is in force and
  spills to the other.
- **A tab is played only if it is a tab, and the browser silenced it.** The capture asks
  for `suppressLocalAudioPlayback: true` and `systemAudio: "exclude"`; the page plays it
  only if the VIDEO track's `getSettings().displaySurface` is `"browser"` and the audio
  track's `suppressLocalAudioPlayback` is exactly `true`. If it is not, the visitor is already
  hearing the tab, undelayed, and a second copy from the page would be an echo — so the
  page does not play it, runs uncompensated, and says so.
- **An input device is never routed to the speakers** — a microphone would feed back, and
  a loopback device is already being heard. `musicRouting` never says `play` for it, and
  the connect itself is additionally guarded on the source kind, so no later edit of the
  rule can make a microphone howl.
- **Raw audio.** Both captures ask for `echoCancellation`, `noiseSuppression` and
  `autoGainControl` off: each is tuned for speech and would flatten the transients the
  detector looks for.
- **The unused video track of a tab capture is KEPT ALIVE** (asked for at 64 px, 1 frame a
  second). Whether stopping it leaves the audio running is browser behaviour that has not
  been checked, and a dead audio track would be silence with no error; keeping it is the
  choice that cannot break the capture. To be confirmed in Chrome, after which it may be
  stopped in `musicAcquire`.
- **What ends a session**: Stop; choosing another source, device or file; the buffer
  running out; and any track of a capture ending from outside (Chrome's "Stop sharing",
  the tab closing, a device unplugged) — the page stops, stops every track (which is what
  clears Chrome's sharing bar) and says the capture ended. A source change while the
  picker is still open gives the capture straight back when it arrives (`music.gen`).
- **What will not start says why** (`musicStartError`): a capture with no audio track
  tells the visitor to pick a Chrome *tab* and switch on "Also share tab audio"; a
  cancelled or refused capture, an unplugged or busy device and a browser without the API
  each get their own sentence.
- **Every Start builds a new detector.** Its running normalisation and reference describe
  the audio it has heard; carried across a switch they would misjudge the next source.
  There is no burst at the start of a stream: the first seconds hold about the steady
  rate of onsets (measured mid-track on the two test recordings), and nothing fires in
  the first half second.
- **The device list** (`musicDevices`) shows audio *inputs* only. Browsers give no names
  until access has been allowed once, so entries read "audio input N" until the first
  successful Start, after which the list is re-read; a `devicechange` re-reads it too.
- **The sample rate is the context's** — 44.1 or 48 kHz, whatever the machine runs at; a
  stream is resampled into it. The detector's bands and windows are built from it.
- **The noise floor** (fix round). The detector's running 99th-percentile normaliser
  promotes ANY non-zero floor to full strength once the music stops — measured: noise
  40–90 dB below a rock track gives ~9 onsets a second at strength 0.9, whatever its level,
  for every source. The page therefore runs the detector with `fluxFloor` 0.3 over a 120 s
  peak-hold (`MUSIC_FLUX_FLOOR`, `MUSIC_PEAK_DECAY`): a band's normaliser is never below
  0.3 of the largest flux that band carried recently. Measured with a 30 s hold: no onsets
  from noise at −60 dB and below, 5 a second at −40 dB (against 9), and 0.1% of the music's
  own onsets lost. **Its failure modes**: a source that is noise from its first sample has
  no louder past to be measured against (a microphone in a quiet room still kicks); and
  for up to a couple of minutes after a loud passage, onsets weaker than 0.3 of it are
  missed. It is an OPTION of the python `CausalOnsetDetector` (`flux_floor`, off by
  default, so `detect_onsets_causal` is bitwise what it was) and of `MusicDetector`.
- **Starting and stopping** are one state machine (`music.starting`, `music.gen`): `starting`
  stays true across every await of `musicStart` — the picker, the decode, `ctx.resume()` —
  and the generation is re-checked after the last one, so a second click or a change of
  source in any of those windows cannot orphan a source. Anything thrown after a capture
  was granted stops its tracks. Stop and Start both reset the clock state, so a BEHIND
  indicator cannot outlive its session. The run button reads **Start / Stop**: it is a
  full stop (a file starts again from the top), not a pause.
- **Hidden is what the document says, not what the frame gap suggests.** The page notes
  `visibilitychange` to hidden and tells the clock (`resumed`), which re-anchors quietly
  and drops the kick queue. A long gap between frames on a VISIBLE page is a device that
  cannot keep up, and lights BEHIND.
- **A non-finite state** stops the page with a message (Reset, then Start) instead of
  leaving a clock that returns "no steps" for ever.
- **Two more colormaps, on this page only**: `magma` (matplotlib's, the map of the offline
  |u| movies; a degree-6 fit, max channel error 0.0226) and `brat` (black → #8ACE00, exact
  and linear). They are `MUSIC_CMAPS` in `music.js`, registered by the page through
  `cmapExtend` (common.js) before anything is built; the display card's colormap select
  lists them after the shared four, and the speed display starts on magma. The shared
  table, the other pages' menus and their emitted WGSL are untouched: with nothing
  registered `cmapNames()` is `CMAP_NAMES` and `cmapWGSL()` returns `CMAP_WGSL` itself.
  The canvas (`colorize`), the colour bar and a saved PNG's bar (`cmapRGB`) read the same
  coefficients. Regenerate with `python webgpu/gen_musiccmap.py`.
- **Fixed colour scale** (box "fixed colour scale", ON by default; slider "scale max
  (× amp)", logarithmic 0.25–4, default 1.5). The shared display divides by its own
  maximum, recomputed every frame, so a flow that is dying away in a quiet passage just
  renormalises and the picture stays at full brightness. With the box ticked the full-scale
  value is `m · amp` instead (`amp` is the kick-amplitude slider): 0 … m·amp for a
  magnitude, ±m·amp for a signed field, and anything beyond it saturates (`dispX` and
  `cmap` clamp, neither wraps). It applies to whatever field the card shows and there is
  no per-field scaling: `amp` is a velocity, so the multiple is dimensionless for |u|, |b|
  and the Elsasser magnitudes and only a number that sets the range for the potentials,
  vorticity and current. σ_c / σ_r keep their own pinned ±1.
  How: nothing shared changed. The range colorize divides by is the chain's 4-byte
  `maxVal` buffer, which `maxFinal` writes every render. The page wraps `render` on its
  own solver INSTANCE (`musicScaleInstall`, in `rebuild`): before each card's render
  `musicScaleApply` points that chain's `maxFinal` bind group at a scratch buffer
  (`D.buf.musicAuto`, freed with the chain) and queues a 4-byte write of `m · amp` into
  `maxVal`. The same kernels run the same number of times; the autoscale still gets
  computed and lands in the scratch buffer, which the readout quotes ("the autoscale
  would be …", read back twice a second; `musicDebug.scale` under `?musicdebug`). The
  colour bar, a saved PNG's bar and a recording all read `maxVal` or the canvas, so they
  show the same range. Either slider or the box redraws at once with no rebuild and no
  step. Box off: the original bind group is in place, nothing is written, and the render
  is the class's own, dispatch for dispatch.

**`?musicdebug`** keeps a record on `window.musicDebug` for measuring sync and latency in
a real browser:

- `kicks[]`: `tOnset` (audio s; the analysis-window centre of the peak frame), `tFire`
  (when the detector declared it), `tApplied` (when `kick()` returned), `band`,
  `strength`, `used`, `dropped`, and on the click track `sched = {t, kind, dt}`, the
  nearest scheduled click. Audio times are seconds since the source started.
- `frames[]`: `a` (audio s), `target`, `simT`, `lag` (seconds of audio; + is behind), `n`,
  `dt`, `e`, `rebased`, `kicks`, `fresh`, `lagAfter`. **`lag` is taken BEFORE the frame's
  steps**, from the previous frame's readback, so a steady run reads about one frame
  (measured: 14.5 ms median at 56.8 fps) by construction; `lagAfter` is where the frame is
  expected to end, and is the number to hold against "within a frame of the audio".
- `summary(skip = 5)`: frames, rebases, lag min/mean/max, steps per frame, frames at the
  cap, and per band the kick count, the mean energy jump, and the latency from the onset
  and from the scheduled click.
- `audio` (also in `summary().audio`; refreshed whenever the sync is recomputed): `kind`,
  `label`, `sampleRate`, `baseLatency`, `outputLatency` (as the browser reports it),
  `outputLatencyUsedMs` (what the sync used: 0 for an input device or an unreported
  value), `delayMs` (D), `holdMs` (H), `trimMs`, `ldetMs`, `residualMs` (how far the
  picture still trails where the sound could not be delayed; 0 when the sync is exact),
  `played`, `compensated`, `suppressed`, `surface`, `mode` (the line shown),
  `trackSettings`, `videoTrackAlive`, `nfft`, `bins`.
- Each kick also has `tDue`. `latencySeconds` in `summary()` is `tApplied − tOnset`, so it
  INCLUDES the hold-back: on Bluetooth it reads ≈ 0.039 + `holdMs`/1000, by design.

**`?musicdebug=energy`** additionally reads the energy back after *every* kick (`eBefore`,
`eAfter`): that is the only way to get one kick's own energy jump — kick energy enters no
accumulator — and it costs a GPU round trip per kick, so take the timing numbers without
it.

**The gate is `devtools/checkmusic.js`.** What it cannot say is listed at the top of that
file and in MUSIC_PLAN's M2 entry: nothing about a real `AnalyserNode`, audible audio, a
real frame's lateness, or a GPU.

### The effects (MUSIC_EFFECTS_PLAN, step 5; every one has a control in the `effects` group)

Phase 2 of the music project: variety. Each effect was built and judged offline first
(`examples/music_turbulence.py`, gate `examples/music_turbulence_check.py`); the page ports
the rules (`music.js`) and holds them to numbers recorded from the python
(`devtools/fixtures/music_ref.json`, key `effects`). With every effect off — kicks
`elsasser`, stereo unticked, pulse 0, mean field 0 — the page kicks, steps and sounds as
the phase-1 page did, and `checkmusic.js` runs its legs 7–12 in exactly that state. Two
things are still not the phase-1 page's, by design: the solver is always built with the
mean-field `stage` kernel (at B0 = 0 it gives the plain kernel's values), and where the
context can split channels the two channel analysers are always built (dead ends: they
feed nothing). Those legs run on a fake context WITHOUT `createChannelSplitter`, which no
real browser is; leg 13 is where the splitter branch is exercised.

| control | default | what it does |
|---|---|---|
| kicks | `variety` | the kick TYPE per band (`musicKickTypesFor`): bass `dipole` (vortex pairs 1.5σ apart, which travel), `vortex` and `island` alternating in between (φ only / ψ only), treble `sheet`. Or one type for every band. |
| sheet energy | 2 | multiplies a sheet's energy (amplitude × √) |
| stereo placement, pan gain | on, 2 | a band's blobs gather at x = L(0.5 + 0.4·clip(gain·pan)), jitter 0.12 L; pan = (E_R − E_L)/(E_R + E_L) over the band's bins of the two channel analysers, read when the onset is declared. The bass is exempt. Nothing is placed until the session's two channels have differed once (`musicChannelsDiffer`): a mono source, up-mixed to two equal channels, would otherwise read pan 0 in every band and pile into the centre column (found by the review). A context without `createChannelSplitter` has no placement at all. The click track is stereo only when the box is ticked at Start. |
| pulse | 0.08 | a bass onset lifts the zoom TO 1 + pulse·strength, relaxing over 0.25 s. **Display only, and CSS**: a `transform: scale()` on the card's canvas (and arrow overlay), clipped by the card. A saved PNG or a recording is the unzoomed canvas. |
| mean field (× rms), turn, strength walks | 1.5, 30°, on | a uniform in-plane field B0. Each bass onset turns it by a random angle (rms `turn` × strength) and, if the strength walks, steps \|B0\|/rms by rms max/6 × strength, reflected into [0, max]. \|B0\| = ratio × √(2E), E from the frame's stats readback, eased over 0.1 s. |

**Solver side** (`solver2d.js`, both off unless asked; the 2D page asks for neither):

- `kick: true` now also emits `sheetKick`, the blob kernel's anisotropic twin
  (`blobKickWGSL` with `C.sheet`): two vec4 per slot, (x0, y0, σ, w) and (e_x, e_y, 0, 0),
  gaussian exp(−σ²(k² + (e·k)²)/2). `Solver.kick(list)` routes a list in which any blob has
  `ratio ≠ 1` through it (`_packKicks(list, true)`); any other list is the blob kick,
  dispatch for dispatch as before.
- `b0: true` emits a different `stage` kernel that carries the mean field in the EXACT
  linear step: with the diagonal rate l the linear operator is [[l, i k·B0], [i k·B0, l]],
  whose exponential is e^(lτ)[[cos θ, i sin θ], [i sin θ, cos θ]], θ = k·B0 τ. So the
  field's waves cost no timestep (it is not in the CFL sum) and no energy.
  `setMeanField(bx, by)` writes the uniform. One invocation per mode owns both halves, so
  the dispatch is NM/64, not 2NM/64. Needs Pm = 1. At B0 = 0 it gives the plain kernel's
  values word for word.
- **Other pages are untouched by construction**: without the flags `buildShaders` emits the
  text it always did (`checksolver2d.js` leg 2, byte-identical against 268100f for both
  pages) and `encodeStep` dispatches `stage` over the count it always did
  (`checkkick.js` leg 7). The class's new lines are recorded as allowances in
  `checksolver2d.js` (`EFFECTS_SHADERS`, `EFFECTS_SOLVER`).

Gates: `checkkick.js` leg 7 (the sheet kernel against a real-space DFT and against the
offline `make_kick`'s `potential_k`, `fixtures/kick_offline.json` key `sheet_cases`; the
mean-field kernel's free streaming, its fp64 recurrence, B0 = 0, the wiring) and
`checkmusic.js` leg 13 (the rules, then the page with its defaults on a fake stereo
context). **In a real browser** (Chrome, M1, 2026-11-21, click track, defaults): 597
steps/s, 6.3 steps a frame (max 15, never at the cap of 64), no rebase, the simulation
1.5 ms ahead of the audio on average; `rmhd2d.html` at 609 steps/s. Not tried: a stereo
FILE or a tab capture in a browser, and any device but the M1.


### The music page: the stage (the default layout; MUSIC_STAGE_PLAN)

`music.html` is laid out to be watched: the picture fills the window, the controls sit in a
bar and a drawer that fade after 3 s without activity, and the charts are gone.
`music.html?plain` is the card layout described above (`stageFlag`; `?stage` is accepted
and means nothing). What is where:

- **`stage.js`** (pure parts first: `stageFlag`, `stageFit`, `stageTiles`, `stageIdleNew` /
  `stageIdleEvent`, `stageStatusPlace`, `stageFullscreen`; then the page side). `stageBuild`
  runs after `controlsBuild` and moves `#topbar` and `#controls` into `#stageBar` /
  `#stageDrawer`; `stageAdopt` runs after `cardsLayout`, closes the chart cards and moves the
  display card's `.cvwrap` into `#stage`. `cardAspect()` in `music.html` answers
  `stageAspect()`, so the shared card code sizes the canvas for the window (times
  devicePixelRatio, long side capped at `STAGE_CAP` = 1920). A resize re-fits the canvas
  only; the solver is never rebuilt by one.
- **The whole box is always shown** (`stageFit(..., contain = true)`), centred. Where the
  window is another shape the `sides` select (`selFill`) leaves the two strips black or
  repeats the picture: `stageFrame(drawn)` copies the card's canvas onto `#stageTile`, a 2D
  canvas over the picture, at `stageTiles`' positions. A WebGPU canvas can only be read in
  the task that rendered it, so `music.html` wraps `renderCards` (this page, the stage only)
  and passes whether a card drew; on a frame with no render nothing is copied, and stale
  copies mark the card dirty so the next frame renders. The camera pulse scales
  `#stageTile` with the card's canvas. A saved PNG or a recording is the single picture.
- **The box, the grid, the background** (`grpSim` / `grpFx`, on both layouts; the grid
  defaults to auto on the stage and to standard with `?plain`):
  `selBox` square (default) or the window's shape (short side 2π, long side up to 2.4×,
  read at boot, Start, Reset and on a select change, never on a resize); `selGrid` auto /
  light 256×128 / standard 256×256 / sharp 512×256 (auto starts on standard, drops to light
  after more than 3 falling-behind rebases in a run's first 10 s, never picks sharp);
  `selBg` walking mean field / current sheets / none. The rules are pure, in `music.js`
  (`musicWindowAspect`, `musicBoxFor`, `musicGridFor`, `musicRungFor`, `musicAutoRung`,
  `musicCutoffK`, the box-aware `musicBandPlan`, `musicSheet*`), and match
  `examples/music_turbulence.py --aspect / --grid / --sheets`.
- **Current sheets** are the solver's maintained equilibrium (`eqsrc`, with the
  `eqsrcDrag: true` opt-in that lets the uniform drag coexist with it) and exclude the mean
  field (`b0`). The profile is a flux bump, strength and width from `rSheetB` / `rSheetW`,
  uploaded at Reset. Choosing them puts `magnetic |b|` on the picture, and the fixed colour
  scale's reference follows the field shown (`musicScaleRef(u, mode)`): the sheet's own peak
  / 1.5 in a field the sheet is part of (the peak field for |b| and z±, ψ0 for ψ, 2ψ0/a² for
  the current), and amp, as everywhere else, for the velocity, φ and the vorticity.
- **`traces.js`** (`musicTraces`; pure helpers `traces*`) draws on one `<canvas
  class="traces">` over `#stage` (or over the display card with `?plain&traces`): the total
  energy spectrum along the bottom and the displayed field's middle-row cut along the top,
  as lines that fade with age in the picture's colormap; tick labels behind a toggle. The
  cut follows `musicTraces.picture()` (the stage's `stagePicture`) and its repeats. Both are OFF by default; what shows by default is the **side
  panels** (`cbTrSides`), four in each of the two black strips, of the kind its select says
  (`selTrA`, `selTrB`): spectra (E_u, E_b, E+, E−, each with a k^(-5/3) guide), those
  energies against time (from the loop's own stats readback), or four signed cuts u_x, b_x,
  z+_x, z−_x along x = Lx/2 (the solver's `readCutLine`). Readbacks: the spectrum (shared with the spectrum card where one exists,
  else its own, every 6th frame), nx + 1 four-byte copies for the top cut, one cut line.

## Alfvén-wave collision: directions, placement, χ

The propagation direction is read off the *implemented* propagator, not assumed. The 3D
stage kernel applies exp(L·τ) with L = [[d, i·kzd], [i·kzd, d]] to (φ, ψ); its
eigenvectors are ζ± = φ±ψ with eigenvalues d ± i·kz, so dζ±/dt = ±i·kz·ζ±, and with the
inverse transform's e^{+i kz z} convention that is **ζ±(z,t) = ζ±(z±t, 0)**: ζ⁺ travels
toward *smaller* z, ζ⁻ toward *larger* z, both at v_A = 1. (Same statement as RMHD's
∂_t z± ∓ v_A ∂_z z± = …, with B₀ = ẑ.)

**Placement** (`common.js: packetGeom`) is symmetric about the midplane with separation
s = clamp(5·σ_z, 3Lz/8, Lz/2), i.e. z₀± = (Lz ± s)/2. That is ≥ 5 σ_z on the direct side
by construction and on the wrap-around side too, because s ≤ Lz/2 ≤ Lz − s — which is
what the **σ_z slider's Lz/12 cap** buys (5·Lz/12 ≤ Lz/2; the slider shows "increase Lz
for longer packets" when it is sitting on the cap). Overlap is therefore impossible: at
the cap the two envelopes cross at 4.4% = exp(−25/8). At the default σ_z = Lz/16 the
3Lz/8 floor binds and the placement is the historical z₀⁺ = 11Lz/16, z₀⁻ = 5Lz/16; the
head-on collision is at z = Lz/2 at **t = s/2** (= 3Lz/16 ≈ 4.7 for Lz = 8π), strictly
before the wrap-around one at z = 0 (the closure speed is 2·v_A, so collisions repeat
every Lz/2 in time). The live line under the IC controls prints s and t.

**χ.** σ_z is the knob that sets the nonlinearity: the same line shows
**χ± = a∓·k̄⊥·σ_z/v_A** with k̄⊥ = 1/σ⊥ (σ_letter = Lx/32 for the letters, the blob-width
slider for a drawing — they share a default) and v_A = 1. Two values because the
nonlinearity one packet feels is set by the *other* one's amplitude (χ⁺ ∝ z⁻), which now
matters — the ζ± amplitudes are separate sliders. k̄⊥ comes from the shared smoothing
length because the smoothing width *is* the gradient scale of a smoothed glyph, so it is
the perpendicular wavenumber of the field z± = ẑ×∇ζ±; it is an *estimate*, the packets
are not single modes (an energy-weighted k̄ of the actual glyph runs lower). χ ≪ 1 is the
weak regime: the packets pass through each other nearly unchanged and distortion
accumulates over many transits (the box is periodic, so they collide again and again);
χ ≳ 1 is strong, a single collision already shreds them. The demo starts at a ≈ 0.2,
σ_z = Lz/16, χ ≈ 1.6, and reaches χ ≈ 2.1 at the σ_z cap.

**The sinusoidal pair** (3D IC preset `sinusoidal z± packets`, FEEDBACK item 9) is the
textbook version of the same collision: one packet whose perpendicular structure is a
single mode in x, the other a single mode in y. What is *stored* is the potential, so a
field that is a pure sine needs a potential that is a cosine of the other coordinate's
sign convention — ζ⁺ = −cos(k₁ˣx)/k₁ˣ and ζ⁻ = +cos(k₁ʸy)/k₁ʸ give
**z⁺ = a⁺·ŷ·sin(k₁ˣx)** and **z⁻ = a⁻·x̂·sin(k₁ʸy)** after `icZetaFields` normalizes each
to max|∇ζ| = its own amp slider (the 1/k prefactors are what make that exact). Each field
alone is an exact ideal solution — a z⁺ with no z⁻ propagates unchanged — and together
they interact through exactly one beat, (z⁻·∇)z⁺ = a⁺a⁻k₁ˣ sin(k₁ʸy)cos(k₁ˣx)ŷ. It is a
PACKET preset in every other respect: the same `icGaussZ` + `packetGeom` envelopes, so
σ_z, the ≥ 5σ_z placement, the trackers and the meeting time above are unchanged, and its
k̄⊥ in the χ line is the mode's own 1/k₁ = Lx/2π (the one case where it is not an
estimate). It is the collision preset's *default* IC since the on-device pass (a
space-filling pair is far easier to read as isosurfaces than two localized blobs); the
letters remain an entry in the same dropdown, and neither IC's code moved.

**Plane tracking.** Because z is spectral, "the energy in plane iz" is not a k-space
partial sum: `readPlaneEnergy` forms z± = φ±ψ, inverse-transforms *along z only*
(borrowing the gradient stack as scratch — a separate submit, and submits execute in
order), and reduces one workgroup per plane with the usual perpendicular-energy weight
minus the nz² Parseval factor, so the plane values average to E±. The frame loop reads
back 2·nz floats at ~10 Hz; every display card's z-slice source can be `manual`,
`track z⁺` or `track z⁻` (in the cube view they pick the TOP face's plane instead), and
the cut chart has its own. The tracked planes and their (continuous) z coordinates are
printed in the readout. How a tracker follows its packet is one page-wide choice:

- **energy centroid** (default) — the *circular* first moment
  z̄ = arg(Σ_k E_k e^{2πik/nz})·nz/2π. Periodic by construction, smooth, and exactly
  linear in t for a packet translating at v_A (node check: max residual 1.7e-13 planes
  over 400 readbacks, implied speed 1.000000, displayed plane never steps by more than 1).
- **peak plane** — the argmax, with **10% hysteresis**: it only leaves the current plane
  when another beats it by more than that. A raw per-readback argmax is what made the
  collision displays jitter; a 5% rival now never steals the plane, a 15% one does.

Both still cost one readback per ~100 ms and that readback is a `mapAsync` round trip
inside the frame loop, so a slow device pays a pipeline stall per tracked frame
regardless of the tracker — worth checking on-device if the collision preset still feels
uneven with the centroid.

**Lz** is a free parameter of the 3D grid (selector: 2π/4π/8π/16π), with `64²×128` and
`64²×256` added to the resolution presets. Everything kz-derived already read `p.Lz` —
kz/kzd/z_diss_k in `makeGrid`, `parKfac` for the parallel-spectrum abscissa, the
`autoZDiss` cutoff, the IC's z coordinate — and no generated WGSL depends on Lz at all
(verified: identical bytes at 2π/8π/16π for five grids). The one thing Lz does not
stretch is the cube-face view, which always draws the box as a cube.

## Expected long-time behavior

Defaults match `examples/forced-turbulence-2D.ipynb` (hyper=2, diss=1e-5, eps=(0.15,0.15),
fshell=(1,3)). On timescales much longer than the notebook's t=20, 2D RMHD piles magnetic
energy up at the box scale (inverse cascade of the mean-square flux function): verified
with the JAX solver itself at 128², E_mag grows roughly linearly (≈1 at t=20 → ≈5 at
t=140) while E_kin saturates, and the flow organizes into large coherent structures.
Stable large vortices/islands at late times are physics, not a solver bug — there is no
large-scale friction term to absorb the inverse cascade. Stronger forcing or a shell
closer to the box scale gets there proportionally faster. hyper=1 requires much larger
diss to resolve (k_η ≈ (ε/diss³)^¼ must stay below nx/3, e.g. diss ≳ 1.5e-3 at 512²,
ε=1); underresolved laplacian runs develop grid-scale pileup.

## Limitations

- fp32: fine for eyeballs and demos, expect slow energy-budget drift over long runs;
  not a substitute for the JAX solver for science runs.
- Resolutions 128/256/512 — n_x, since 2D also offers rectangular boxes: the wide 4π × 2π,
  where n_y is n_x/4, and the large 8π × 8π, which `chain` runs at 256². A
  1024-point line is exactly the WebGPU minimum workgroup-storage limit, so the 2D page
  asks for the adapter's own limit at boot and caps the longest line there (`NMAX_LINE`);
  a resolution a box cannot run is disabled in the select rather than silently clamped —
  which as of the 8π × 8π box disables nothing, every box having fy ≤ 1, but the rule is
  about the box table and not its current contents. See
  SPEC §8. Only 2D boxes are rectangular; the 3D perpendicular plane stays square.
- Elsasser forcing only, no snapshots.
- The perpendicular spectrum dispatches `nbins` = the smaller of the two axis dealias
  cutoffs in units of kunit = min(2π/L_x, 2π/L_y) (`floor(min(nx,ny)/3)` on a square box), but
  `round(|k|/kunit)` reaches that value in the corners of the dealias ellipse, so those
  few modes are binned nowhere: `sum(bins)` is a hair below the total energy (1.5% at
  16², far less at production resolutions). Long-standing, cosmetic.
