# MUSIC_PLAN — live music drives the 2D demo

> **STATUS: drafted 2026-11-21; M0 measured; M1 (solver) committed (14ed94b); M2 (page, clock,
> click track and file) committed (c960488), runs in Chrome on the M1 and looked in sync by
> eye; M3 (tab capture, input devices) and M4 (latency compensation) implemented the same
> day, uncommitted, awaiting one fresh review and a real-browser pass. The page is unlinked
> and pruned from the deploy.** Phase M0 was three
> measurements that decided the design (results below).
> **M0.a answered 2026-11-21: tab audio capture works** for both the Spotify and the
> Pandora web players (Alfred, Chrome on the M1) — tab capture is the primary route.
> **M0.b measured 2026-11-21** (numbers under Phases): a causal detector reproduces the
> offline one exactly on the click track and catches 84–96% of its strong onsets on real
> music; the poll rate is the sensitive ingredient, the running normalisation is not.
> **M0.c measured on the laptop 2026-11-21**: ~600 steps/s at 256² with the display on,
> against a need of 240–330 — about 2× headroom. M0.b's gate passed (Alfred accepted the
> causal detector on the side-by-side movie). Open: the phone, auto-diss against the
> offline rule, and the latency policy. Decisions so far are at the end.

## Why

`examples/music_turbulence.py` (offline, JAX; uncommitted as of this draft) maps a track to
2D RMHD: an onset in a frequency band is one impulsive kick of gaussian blobs, drawn
independently for z+ and z-, with the band setting the blob width. Rendered to a movie and
muxed with the audio it works — Alfred's verdict on a full 512² render of a rock track was
"works very well", threshold 0.3 preferred over 0.5. This plan is the live version: the 2D
app follows whatever is playing on the laptop (Spotify, Pandora, a file), in real time.

The offline script stays the reference implementation and the way to make a shareable
movie. The live mode does not replace it.

## What the offline version established (do not re-derive)

- **The map**: per-band spectral-flux onset → one kick; kick = a fresh set of blobs at
  uniformly random positions, random sign, one set per Elsasser channel; band → blob width
  σ, log-spaced from 0.45 (box 2π) down to 3.7 grid spacings; blob count ∝ 1/σ from 3 in
  the bass band; every blob in a kick has peak |∇z| = `amp` × onset strength.
- **Band count comes from the grid**, not the music: neighbouring widths closer than
  ×1.8 look the same (a blob's spectrum k³e^(−k²σ²) is as wide as its peak k), so
  N ≈ 1 + ln(σ_max/σ_min)/ln 1.8 — **4 bands at 256²**, 5 at 512².
- **A kick is a state edit, not a forcing term.** Amplitude is a velocity increment,
  independent of dt.
- **A uniform drag is required.** Without it 2D MHD decays far slower than a turnover
  (a pure z+ structure does not decay nonlinearly at all), energy integrates over ~10
  beats and the tempo washes out. `memory` = the energy e-folding time, 1.0 s of audio.
- **Display |u|, not vorticity.** A bass kick changes the frame 7–13× more than the
  baseline in |u| against 1.5× in ω.
- **Under-dissipation looks like a CFL problem and is not one.** The hyperviscous rate has
  to be ~50·amp at the dealiasing cutoff; at ~5 the tail piles up and no timestep fixes it.
- **Numbers to design against** (rate 1 time unit per second of audio, amp 1, memory 1 s):
  256² needs **~330 steps per second of audio** on a dense track (220 on the click track);
  512² needs 570–660. Kick rate at threshold 0.3 on that track: ~15/s over 5 bands.

## The three hard parts

1. **Audio in.** A web page cannot read another app's audio without the visitor granting a
   capture. Three routes, all ending in a `MediaStream` → `AudioContext`:
   - *tab audio* — `getDisplayMedia({audio: true})`, the visitor picks the tab playing the
     web player. No install. **Confirmed 2026-11-21 (M0.a): the Spotify and Pandora web
     players' audio comes through** in Chrome on the M1. The primary route.
   - *input device* — `getUserMedia` with a device picker: a loopback driver (BlackHole)
     for anything on the machine including desktop apps, or the microphone.
   - *file* — `<input type=file>` → `decodeAudioData`. Deterministic; the test source.
2. **Causality.** The offline detector normalises each band by its 99th percentile over
   the whole track, subtracts a *centred* moving mean and picks local maxima with 70 ms of
   look-ahead. Live it needs a running normalisation and either a look-ahead (latency) or
   threshold-crossing (jitter in the strength).
3. **The clock.** Simulation time must equal `rate` × audio time. The app's loop steps "as
   many as fit in a frame" under an adaptive `stepsPerFrame` controller, and dt is adaptive
   on the GPU, so simulation time is neither steered nor known until the stats readback.

## Design

### Solver (`physics.js`, `solver2d.js`)

- **`blobKick` kernel — new, one dispatch per kick.** The same analytic k-space evaluation
  as `blobBuildWGSL` (weight w = P·2πσ²·nx·ny/(Lx·Ly), P = `icBlobPeak(amp, σ)`, dealias
  mask inside the kernel, k = 0 skipped), but it ADDS to the fields buffer —
  δφ = (z+ + z−)/2, δψ = (z+ − z−)/2 — instead of writing the forcing buffer. Its own
  `KICK_MAX` (32 per channel: 15 needed at 256², 30 at 512²) and its own blob buffer.
  **`blobBuild`, `BLOB_FORCE_MAX = 8` and the spacebar mode are not touched**; the kick is
  dispatched between steps, never inside one (the LSRK registers are between-step clean).
- **Drag as part of the linear operator**, not a per-frame multiply: `gridB.y` =
  −diss·k^(2·hyper) − μ/2, μ = 1/(memory·rate), so the existing exact exponential carries
  it and no WGSL changes. Two things to check before relying on that: at Pm ≠ 1 `_dissLin`
  multiplies the φ half's exponent by Pm, which would scale the drag too — music mode
  either pins Pm = 1 or the drag gets its own lane; and the dissipation-rate readout will
  count drag as dissipation (correct, but say so in the readout). μ = 0 outside music mode,
  so every existing number is unchanged.
- **Dissipation**: music mode sets `diss` from the offline rule (3·amp/(n/6)⁴ at hyper 2)
  instead of the slider/auto-diss, or auto-diss is shown to already satisfy it (M0.c).

### Audio front end (new `music.js`, loaded only by `rmhd2d.html`)

- `AudioContext` created on the enabling click (autoplay policy). Source node →
  `AnalyserNode` (`fftSize` 2048 — 46 ms at 44.1 kHz, the offline window;
  `smoothingTimeConstant = 0`), polled once per frame with `getFloatFrequencyData`.
- **v1 polls on the frame**, so the hop is the frame interval (16.7 ms, jittered) against
  the offline 10 ms. An `AudioWorklet` with a fixed hop is the upgrade if M2 shows the
  jitter matters; it means shipping an FFT in the worklet, so not first.
- **Detector** (pure function of the last few spectra, no DOM — testable under node):
  per band, flux = Σ max(0, ΔdB)/scale over the band's bins; normalised by a running high
  quantile over the last ~20 s (seeded so the first seconds are not all onsets); threshold
  above a *trailing* mean; fire on the first frame after a local maximum (one frame of
  look-ahead) with a per-band refractory gap of 70 ms. Output `{band, strength}`.
- The analyser's spectrum is in dB with a Blackman window; the offline one is
  log1p-compressed with a Hann window and a whole-track maximum. **They will not agree
  onset-for-onset.** The comparison that matters is M0.b's, between the offline detector
  and a causal one written in Python first.
- **Nothing leaves the browser.** No audio, spectrum or onset is sent anywhere, and
  ANALYTICS_PLAN's events must not grow an audio-derived field. Say so in `docs.html`.

### Clock (`common.js: loop`, music mode only)

- Target simulation time = t₀ + `rate` × (`audioCtx.currentTime` − a₀). Each frame:
  `n = ceil((target − simT) / dt_last)`, capped at a per-device maximum; `simT` and
  `dt_last` come from the stats readback the loop already does every frame.
  **Caveat found in M1's review:** after a kick `dt_last` is a PRE-kick dt (the solver
  recomputes its own on the next step, the readback has not caught up), so this `n` is
  wrong for the first frame after a kick — see M1's "What M2 inherits".
- **This replaces the `stepsPerFrame` controller while music mode is on; it does not tune
  it.** LOOPLAT_PLAN's post-mortem is why: the three failed controllers all steered on a
  period that was their own output. Here the reference is the audio clock, which the loop
  cannot influence.
- **Falling behind**: if `target − simT` exceeds ~0.25 s of audio the device cannot keep
  up — rebase (t₀ ← simT) and show a "behind" indicator instead of spiralling. A device that rebases
  continuously needs a lower `rate` or resolution; that is reported, never auto-changed.
- Kicks are applied at frame boundaries (≤ 17 ms quantisation at 60 Hz; the offline runs
  measured ≤ 17 ms worst-case lateness and looked right).
- A hidden tab stops rAF and therefore the simulation while the music plays on; on return,
  rebase.

### Latency

Sound → picture delay is roughly half the analysis window (23 ms) + one frame of
look-ahead + one frame to present: an estimated 40–80 ms, near the edge of noticeable.
- *File and tab sources* can remove it: the page plays the audio itself through a
  `DelayNode` set to the measured latency (tab capture with
  `suppressLocalAudioPlayback`), so picture and sound line up.
- *Loopback and microphone* cannot (the sound is already out); they get the raw latency.
Measure before building the delay (M2).

### UI — its own page, `music.html` (decided 2026-11-21)

Music mode is a separate page beside `rmhd2d.html` and `rmhd3d.html`, linked from the same
top line, not a control group inside the 2D app. It loads `common.js`, `physics.js`,
`solver2d.js` and `music.js`, and fixes what the 2D app leaves adjustable: N = 256,
hyper 2, Pm 1, display |u|, no OU or spacebar forcing, dissipation from the offline rule.
Its own controls are few: source (file / tab / input device), threshold (default 0.3),
memory, amp, rate, and a per-band activity strip so a silent or saturated band is visible.
Consequences: the drag and the kick never appear in `rmhd2d.html` (which keeps solving
unmodified RMHD), nothing has to be pinned and restored, and `pages.yml`'s staging and
`docs.html` each gain one page.

## Phases and gates

**Process (changed 2026-11-21, agreed with Alfred).** M1 went implementer → fresh adversarial
reviewer → fix round, the standing flow, and it paid: the reviewer found the stale-dt hazard
and 13 real gate holes. But gate and mutation runs were most of the wall time, and what
remains is page glue and audio plumbing, where a mistake shows when the page is opened. So
from M2 on: implementer only, gates for the pure logic, no mutation quota (M2's one round
was already in flight), and **one fresh reviewer pass after M4** over everything since
14ed94b, aimed at the clock and the detector port. Alfred tried M2's page on the real GPU
(M1, Chrome) with the click track and a file on 2026-11-21: "looks good", in sync by eye.

**M0 — measure, no `webgpu/` code.**
- a. **DONE 2026-11-21.** Chrome on the M1, a tab playing the Spotify web player, then
  Pandora, captured with "Also share tab audio": audio comes through for both (Alfred,
  with a throwaway `getDisplayMedia` + `AnalyserNode` level-meter page). Tab capture is
  the documented route; loopback is the fallback for desktop apps. System-audio capture
  on macOS was not tested and is not needed.
- b. **MEASURED 2026-11-21; Alfred's verdict on the movies pending.**
  `detect_onsets_causal` in `examples/music_turbulence.py` (`--causal`): the last 2048
  samples under a Blackman window once per 60 Hz frame, peak-hold compression reference
  (30 s), trailing 20 s 99th-percentile normalisation, trailing 0.2 s mean, a peak declared
  one frame late, 70 ms refractory. Against `detect_onsets`, 4 bands, threshold 0.3,
  onsets matched per band within 30 ms:

  | signal | offline /s | causal /s | recall | precision | recall, strength ≥ 0.8 | ≥ 0.95 |
  |---|---|---|---|---|---|---|
  | click track, 30 s | 7.2 | 7.2 | 0.99–1.00 | 1.00 | — | — |
  | Goldberg var. 1, 115 s | 18.6 | 14.7 | 0.61–0.83 by band (0.73 pooled) | 0.80–0.99 | 0.92 | 0.96 |
  | rock track, 251 s | 10.9 | 9.9 | 0.65–0.87 by band (0.71 pooled) | 0.74–0.79 | 0.84 | 0.91 |

  Matched onsets differ in time by +2 to +6 ms mean, 5–10 ms scatter; matched strengths
  agree in the mean (ratio 0.94–1.02) and correlate at 0.5–0.7. The page knows about an
  onset **40 ms** after its window centre (half a window + one frame).
  What the disagreement is made of (rock track, first 60 s): whole-clip instead of running
  normalisation changes nothing (recall 0.68 vs 0.69) — **the causal normalisation is
  free**; the hop is what matters — **30 Hz polling drops recall to 0.44**, 100 Hz raises
  it only to 0.72 at lower precision (0.64). So ~30% of the weaker onsets are simply
  picked differently by two equally arbitrary peak-pickers, and that part is not worth
  chasing. Consequence for the design: a page that renders below 60 Hz loses onsets, so
  either music mode requires 60 Hz or the detector moves to an `AudioWorklet` with a fixed
  hop (decide in M2, with the measured frame rate).
  Side by side (40 s, 256², same settings; blob positions differ because the two onset
  lists consume the RNG differently): `examples/data/music/m0b_offline_left_causal_right.mp4`.
  Gate: Alfred prefers neither, or the difference is understood. **PASSED 2026-11-21: accepted.**
- c. **LAPTOP MEASURED 2026-11-21** (M1, Chrome, `apple metal-3`, `?bench` at 256²,
  `cflEvery` 4): whole step 1.475 ms median (1.465 min), i.e. 678 steps/s of pure GPU;
  the running app reads **~600 steps/s** with the display on. Need, from the offline runs
  of the rock track at 256²: 290 steps per second of audio with the offline detector and
  4 bands, 236 with the causal one, 330 on an 8 s clip with 3 bands — so **1.8–2.5×
  headroom**, over the ≥ 400 target. (FFTPERF_PLAN's older ~525 is superseded for this
  machine.) Per kernel the step is still transform-bound: `rowsC2R` 148 µs and `colsInv`
  86 µs of the 1475.
  **Still open:** the phone; and whether auto-diss lands near the offline rule
  (3·amp/(n/6)⁴ at hyper 2).

**M1 — solver.** `blobKick` + drag. **IMPLEMENTED 2026-11-21, through one adversarial
review round the same day, committed as 14ed94b; nothing has run on a real GPU.** Developer notes:
`webgpu/README.md`, "The impulsive kick and the drag".

*What landed.*
- `physics.js`: `KICK_MAX` = 32 and `blobKickWGSL`. `solver2d.js`: the gated emission, and
  on `Solver` the parameters `mu` / `kick`, `_packKicks`, `kick(list)`, `_dragLin`,
  `setDrag(mu)`, and the `_cflStale` mark `step` honours. `gen_kickfixture.py` +
  `devtools/fixtures/kick_offline.json`.
- `devtools/checkkick.js`, 201 rows, all green: the emitted kernel against a real-space
  DFT and against the offline `make_kick`'s own output (max |δ|/peak ≤ 1.4e-7); k = 0 and
  the out-of-band modes untouched bit for bit; the Elsasser map and additivity exact;
  `kick()`'s upload / encoder / dispatch / submit and a snapshot that nothing else about
  the solver moves; the stale-CFL mark as kick-then-step dispatch logs; the drag column,
  its refusals, and E(T)/E(0) = e^(−μT) through the real `stage` kernel to 9e-8 — at
  hyper 1–4, on a rectangular grid, and in the music configuration (`kick: true` + `mu`).
- Existing gates all green with the same verdict rows as before, and both pages' full
  kernel dumps byte-identical to HEAD; a never-kicked solver's dispatch log and uploaded
  bytes are identical to HEAD's (one-time out-of-tree comparison, 4 configurations).
  `checksolver2d` is fully green, 34 rows: the new JavaScript is its fourth allowance
  (`MUSIC_SHADERS` / `MUSIC_SOLVER`, 11 blocks), and in the same pass the `makeGrid` row
  that f1f5f23 had left red since 2026-11-13 was restored by recording that commit's two
  lines as their own allowance (`F1F5_GRID`) — shown live on a scratch clone.

*Differences from the plan as drafted.*
1. **The kernel is emitted only for `kick: true`** (as `srcInit` is for `eqsrc`), so
   `rmhd2d.html` emits exactly the kernels it did and no gate's `ADDED` list changed —
   the "kick never appears in the 2D app" decision, made literal.
2. **The drag is applied by the Solver to `makeGrid`'s column, not inside `makeGrid`.**
   Same numbers (`fl32(lin) − μ/2`).
3. **`mu ≠ 0` with `eqsrc` is refused as well as `pm ≠ 1`** — the maintained-flux source
   reads the same column.
4. **`kick()` also refreshes the energy scalars, counts itself in `solver.nkicks`, and
   marks the CFL stale** (review F1, below).
5. **Overflow policy**: first `KICK_MAX` per channel kept, the rest dropped and the count
   returned (`{used, dropped}`).
6. **Validation**: every blob is checked before anything is packed — any list position,
   zero amplitude included — on what is actually uploaded: non-finite input, σ ≤ 0, and
   an fp32-infinite weight or σ all throw. `x`, `y` are packed modulo the box.
7. **`@workgroup_size(64, 1, 1)`**, spelled out, because the devtools interpreter runs a
   one-argument `(64)` four times per invocation id and this kernel accumulates.
8. **Out-of-band modes and k = 0 are "untouched", not "exactly zero"**: an early return,
   which is the stronger statement. k = 0 is inert, not zeroed.
9. **The fixture stops at σ = 0.5**: the offline gaussian is minimum-image, the kernel's
   fully periodic; they differ by 3e-6 of the peak at σ = 0.625 and 3e-9 at 0.5.
10. **The Nyquist lines are covered by the cut, not by a reality check**: no retained mode
    lies on them; the ky = 0 row is exactly hermitian.

*The review round (2026-11-21).* A fresh reviewer mutation-tested the gate. No wrong
number; one hazard, one poison channel, and fourteen surviving mutants that were all holes
in what the gate observed. Fixed:
- **F1, hazard — a kick left a stale dt.** `step(cflEvery)` recomputes dt only at
  `nsteps % cflEvery == 0`, so up to `cflEvery − 1` steps ran on the pre-kick dt; after
  silence that is the quiescent floor `cfl·dx/0.1`, ~10·amp times the CFL dt. Now `kick`
  sets `_cflStale` and the next `step` recomputes dt regardless, then clears it. A solver
  that is never kicked is on the unchanged schedule.
- **F5, poison — `{sigma: 1e13, amp: 1}`** packed an infinite fp32 weight and the kernel
  wrote NaN to every mode. The packed weight and σ are now checked in fp32, and positions
  are wrapped so a huge finite `x` cannot overflow the phase either.
- **F2/F3/F6/F7, gate holes**: the gate now wraps the command encoder (a `clearBuffer` or
  copy inside the kick, or a pass never ended, is visible), snapshots the solver around a
  kick, builds the music configuration and hyper 3/4 and a rectangular grid for every drag
  path, checks the dispatch count where NM/64 is not an integer, and refuses bad blobs at
  every list position including at zero amplitude. All sixteen re-applied survivor
  mutations are killed.

*What M2 inherits from this — obligations, not suggestions.*
- **Construct the music solver with `kick: true`, `epsP: 0`, `epsM: 0`, `pm: 1`,
  `eqsrc` off.** The `Solver` defaults are `epsP = epsM = 0.15`: left alone, the OU shell
  forcing and its lagged scales run underneath the kicks.
- **One `kick()` per onset, never a merged list.** `KICK_MAX` = 32 per channel fits one
  band (15 at 256², 30 at 512²); two onsets in one frame merged can need 60. Check the
  returned `dropped` and treat non-zero as a bug.
- **The clock formula in Design uses a stale dt after a kick.** `n = ceil((target − simT)
  / dt_last)` takes `dt_last` from the last readback, which is the PRE-kick dt until one
  step has run — the solver now recomputes its own dt on that step, but the page's `n`
  was computed from the old one and over-estimates the advance per step. After a kick,
  take one step, re-read, then compute `n` (or bound `n` by the post-kick CFL).
- **M0.c's ~600 steps/s was measured at `cflEvery` 4.** Each kick now forces one extra
  CFL reduction; at ~15 kicks/s against ~300 steps/s that is small, but the headroom
  figure should be re-measured with kicks at the real rate, and at `cflEvery` 1 if the
  page uses it.
- **Key redraws and the stats cache on `nkicks` as well as `nsteps`** (`common.js`
  `stateMark`): a kick changes the state without a step.
- **The energy jump per kick must be differenced from two readbacks.** Kick energy enters
  no accumulator; with the drag on, `sc[6]`/`sc[7]` are hyperviscous + μE.
- **Amplitude convention**: a lone ζ⁺ blob of amplitude `amp` has peak |u| = |b| = amp/2
  (the offline script's convention too — the page's `amp` slider means the same thing).
- **`setDrag(0)` on a fresh solver can move the column by one ulp** (it rebuilds through
  `refreshDissipation`'s fp32-k² formula); construction at `mu = 0` cannot.

**M2 — the page, the clock, the click track and a file.** **IMPLEMENTED 2026-11-21,
uncommitted.** Alfred has since tried it in Chrome on the M1's GPU with the click track and
a file and reports it looked in sync; the three NUMBERS below (lag, energy jump, latency)
have not been read off yet, and nothing else on the browser-only list is checked. Developer notes:
`webgpu/README.md`, "The music page".

*Decisions for M2* (made by the coordinating session while Alfred was away, 2026-11-21,
and recorded here as theirs, not his — each is cheap to revisit):
1. `music.html` is its own page on the shared framework with a reduced control set.
   Fixed: N = 256, a 2π box, hyper 2, Pm 1, `epsP = epsM = 0`, no spacebar blobs, a solver
   built with `kick: true` and `mu = 1/(memory·rate)`, `diss = 3·amp/(n/6)⁴` recomputed on
   an amp change, |u| displayed. **Not linked from any page**; linking it is Alfred's.
2. The logic is `music.js`, pure, so node can run it; the audio glue is a thin layer.
3. The clock is the audio clock, through a hook in the shared loop that is null on the
   other pages; one `kick()` per onset; rebase past 0.25 s of lag; `stateMark` keyed on
   `nkicks`; `cflEvery` as the other pages use it.
4. Energy per kick is differenced from readbacks; a `?musicdebug` record.
5. Warm-up and running normalisation as in the python.

*What landed.*
- `webgpu/music.js` (pure): `musicBandPlan`, `MusicDetector`, `musicShouldPush`,
  `musicRng`, `musicBlobsFor`, `musicKickList`, `musicClickTrack`, `musicClock`.
- `webgpu/music.html`: the control spec, `uiParams`, Start/Stop/Reset, the audio graph
  (source → analyser → destination, fftSize 2048, smoothing 0), `musicStep` (detector →
  kicks → clock), the band strip, the BEHIND indicator, `?musicdebug` /
  `?musicdebug=energy`.
- `webgpu/common.js`: `stepHook` (null elsewhere), `stateMark` appending `/k<nkicks>` only
  when a solver has been kicked, and `_forcingRecord` knowing a kicked run. Every existing
  page's emitted WGSL is byte-identical to 14ed94b, and `checkidle`'s 54 legs are
  unchanged.
- `examples/music_turbulence.py`: `detect_onsets_causal`'s per-frame core is now the class
  `CausalOnsetDetector` (takes magnitude spectra), which `MusicDetector` ports line for
  line. Its output is bitwise what it was at 94030e1 on the click track, the Goldberg
  variation and the rock track (4 bands / 0.3 and 3 bands / 0.15).
- `webgpu/gen_musicfixture.py` → `devtools/fixtures/music_ref.json` (279 kB; regenerates
  byte for byte): `band_plan` at eleven argument sets, and `CausalOnsetDetector` on a
  stored 8 s test signal under eleven option sets.
- `devtools/checkmusic.js`, 226 rows, all green: bandPlan == the python's (σ to 1e-14,
  counts exact); the detector gives the python's onsets — same frame, same band, strength
  to 1e-15 — in all eleven option sets; the click track's kicks, snares and hats are
  found in the bass, middle and top bands at their scheduled times in the page's
  configuration; the clock's count, cap, look-ahead rule and every rebase; and the page
  booted and DRIVEN under the stub with a fake `AudioContext` whose analyser computes the
  true spectrum of the page's own click track — one `kick()` per onset with the band's σ
  and count per channel, kicks before steps, the steps taken equal to the clock's,
  simulation time within a frame of the audio over 12 s, rebase and indicator when it
  cannot keep up, Stop, restart, bpm change, a file source, a file that fails to decode.
  Mutation-tested once by the implementer: 131 one-line mutations of `music.js`, the
  page glue and the loop hook, 123 killed; the 8 survivors each became a row, but those rows
  were NOT re-run against their mutants (the round was cut short by agreement, and there is
  no separate reviewer round for M2 — one fresh review follows M4).
- `.github/workflows/pages.yml`: `music.html` and `music.js` are pruned from the deployed
  site, as `game.html` is. **This is the implementer's call, not a recorded decision** —
  without it the next push to main would publish an unlinked development page. Delete the
  two lines to ship.

*Deviations from the plan and from the M2 decisions.*
1. **`n` is not always `ceil((target − simT)/dt)`.** Two additions, both in `musicClock`
   and both gated: (a) a step that would overshoot the target by more than one frame is
   not taken yet (`n` rounds down) — at rest dt is six frames long, and rounding up left
   the simulation that far *ahead*, where a kick then hung frozen on screen until the
   audio caught up; (b) a frame that kicked takes at least one step when behind at all,
   since the dt in hand is a pre-kick one and the solver recomputes it on that step. This
   is "let the next frame's feedback absorb the stale dt", with the one-step floor that
   makes there be feedback. Nothing loops on a stale dt.
2. **The detector is fed at ≈ 60 Hz, not once per frame**, on displays faster than that
   (`musicShouldPush`): its windows are counted in frames. At 60 Hz it is once per frame.
3. **The debug energies need `?musicdebug=energy`.** A readback per kick costs a GPU round
   trip each, which would distort the timing numbers the plain flag is for.
4. **The parity fixture is at 11025 Hz / nfft 512**, not the page's 44100 / 2048: the same
   46 ms window and the same code path at a quarter of the bins, which is what keeps the
   committed fixture at 279 kB. The page's configuration is exercised by the click-track
   leg and the end-to-end leg instead (no python reference there — the gate computes the
   spectra).
5. **The JS click track is its own synthesiser**, not a port of the offline one (which
   uses numpy's RNG and scipy filters): same instruments and schedule, no kick-less
   closing quarter (it loops), seamless.
6. **Blob positions come from mulberry32**, not numpy's stream: a live session is not
   comparable blob for blob with an offline movie. Distribution and rule are the same.
7. **A saved `params.json` now says the run was kick-driven** (`_forcingRecord`, a small
   addition to shared code keyed on the solver's own `g.kick`).
8. Stop restarts a file from its beginning on the next Start (no pause/resume position).

*Measured on a real GPU, 2026-11-21* (the coordinating session, through browser
automation: the COMMITTED M2 page, a `git archive` of c960488 served from a scratch
directory, Chrome on the M1, tab visible, click track at 120 bpm).
- **Sync** (`?musicdebug`, 67.2 s, 3736 frames = 56.8 fps; context at 48000 Hz,
  `baseLatency` 5.3 ms, `outputLatency` reported 0): `rebases` 0, `framesAtCap` 0. `lag`
  over the whole run: mean 0.0159 s, min 0.0061, max 0.0685; after the first 10 s: min
  0.0061, 5% 0.0119, median 0.0145, 95% 0.0294, max 0.0685. Never negative. 68 of 3736
  frames exceed two frames (0.034 s), the earliest at 0.14–0.44 s (start-up). Steps per
  frame mean 4.43 (2–14): **246 steps/s against the ~600 available.**
- **How to read that lag.** `lag` is `target − simT` with `simT` from the PREVIOUS frame's
  readback, i.e. it is taken BEFORE the frame's steps: the simulation ended the last frame
  on its target, the audio clock has since moved one frame on, so a steady run reads about
  one frame (0.0145 s median at 56.8 fps is 0.82 of one) by construction. The steps then
  taken bring it to the target, and the frame is rendered after them. So the picture is
  not a frame behind the audio clock; the record was. The M2 gate "within a frame" is met
  in the median and not at the 95th percentile (1.8 frames) *in that pre-step number* — the
  tail is frames that followed a long one. From this round the record also carries
  `lagAfter` (the lag expected once the frame's steps are taken, at the dt in hand;
  `summary().lagAfterSeconds`), which is the number to hold against the gate; it has not
  been measured on a GPU yet. **An option, not taken:** the count could aim half a frame
  past the target so that the pre-step lag centres on zero, but that puts the rendered
  state half a frame AHEAD of the audio clock on every frame, and ahead of a kick that is
  about to land — the thing the look-ahead rule exists to prevent. Leave it.
- **Latency**, onset → `kick()` returning: band 0 (the kick drum) `latencySeconds` mean
  0.0388 (0.0380–0.0433), `latencyFromScheduled` mean 0.0374 (0.0267–0.0773); band 1
  0.0388 / 0.0372; band 3 (hats) from scheduled 0.0460 (0.0407–0.0793). About 38 ms plus
  presentation, as estimated: **M4's 50 ms default delay is the right size.**
- **Energy jump per kick** (`?musicdebug=energy`, 23.2 s, rebases 0), mean (min–max):
  band 0 **0.361** (0.147–0.579) over 46 kicks, band 1 0.130 over 46, band 2 0.102 over
  23, band 3 0.055 over 66. Offline click track: 0.30 / 0.24 / 0.16 / 0.08. The bass jump
  is +20% of the offline figure, inside the plan's ~20%; the mid bands are lower (a
  different synthesiser and detector).
- **Not measured:** a file source, a 120 Hz display, a hidden-tab return, the save and
  record buttons.

*What only a real browser can say — unverified except where noted (it runs, and looked
in sync by eye, 2026-11-21).*
- That WebGPU runs `blobKick` at all (M1 never ran on a GPU either): blobs appear in |u|
  when the click track's kick sounds.
- That a real `AnalyserNode`'s spectrum is what the gate's is: Blackman window, 1/N
  scaling, dB, smoothing 0, the window ending "now". The detector's behaviour on real
  audio rests on that.
- Sync: simulation time against `rate` × audio time over 60 s.
- The energy jump per bass kick on the click track, against the offline 0.30.
- Sound → picture latency, including the audio output latency and one frame of display.
- That audio is audible, starts on the click (autoplay policy), and that
  `decodeAudioData` takes an mp3 / m4a.
- The frame rate the page actually holds at 256² with kicks, whether the 64-step cap or
  the BEHIND indicator ever engages on the M1, and what a 120 Hz display does.
- A hidden tab's return; the save / record buttons on this page.

  *How to measure the three numbers M2 asks for* (Chrome on the M1):
  `cd webgpu && python -m http.server 8766`, open
  `http://localhost:8766/music.html?musicdebug`, leave the source on **click track**,
  press **Start**, wait 65 s, and in the console evaluate `musicDebug.summary()`.
  - *Sync*: `lagSeconds.min` / `.max` should stay within about ±1/60 s (0.017) with
    `rebases` 0 and `framesAtCap` 0; `musicDebug.frames` has it frame by frame (`a`,
    `target`, `simT`, `lag`, `n`).
  - *Latency*: `bands[0].latencyFromScheduled.mean` is seconds from a kick drum's scheduled
    time to `kick()` returning — expect ≈ 0.04 (23 ms half window + one frame + up to one
    more). What is *heard* is later than scheduled by `musicDebug.audio.outputLatency`, and
    what is *seen* later than `kick()` by about a frame; perceived lag ≈ that mean
    − `outputLatency` + 0.017.
  - *Energy jump*: reload as `…/music.html?musicdebug=energy`, Start, wait 30 s,
    `musicDebug.summary().bands[0].energyJump.mean` — the offline click-track figure is
    0.30 at these settings (different kick synthesiser and detector, so "near", not equal).

*What M3 must do.*
- Tab capture and the input-device picker are two more ways to obtain a `MediaStream`;
  `musicStart` already separates "get a buffer/source" from "analyser, detector, clock".
  A stream source has no buffer, no `onended` in the same sense and no known duration.
- With a live stream **audio time is `ctx.currentTime` only** — there is no schedule to
  compare against, so `sched` stays null and latency is `tApplied − tOnset`.
- Re-measure M0.c's headroom with kicks at a real rate (each kick forces one extra CFL
  reduction) and decide the detector's home: on the frame as now, or an `AudioWorklet`
  with a fixed hop if the frame rate measured in a browser is not a steady 60.
- The delay compensation of M4 belongs between the source and the destination only — the
  analyser must keep reading the undelayed signal.
- Decide whether the page ships: remove the `pages.yml` prune and link it, or keep it a
  development build.

**M3 — live sources, and M4 — latency compensation.** **IMPLEMENTED together 2026-11-21,
uncommitted, awaiting the one fresh review. Nothing in this entry has run in a real
browser** — the checklist at its end is what to do there. Developer notes:
`webgpu/README.md`, "The music page".

*Process.* By agreement these two phases ran under a lighter rule than M1/M2: gates for
the pure logic and for anything that could be silently wrong, no mutation round, the
existing gates run once at the end.

*What landed.*
- `music.js` (pure): `MUSIC_SOURCES`, `MUSIC_DELAY_DEFAULT` / `_MAX`,
  `musicTabConstraints`, `musicInputConstraints`, `musicRouting`, `musicStartError`.
- `music.html`: the source select gains "another tab's audio" and "audio input device";
  `musicAcquire` (one function per way of obtaining a source), `musicStart` /
  `musicStop` with a generation counter for anything still awaiting, the device list,
  the delay slider, the mode line, `musicDebug.audio`.
- The graph is `source → analyser` (a dead end, undelayed) plus, only where
  `musicRouting` allows, `source → DelayNode → speakers`. M2's `analyser → speakers`
  connection is gone.
- `devtools/checkmusic.js`: 295 rows (was 226), all green — legs 6b and 10.

*Decisions taken in this round* (the coordinating session's, with Alfred's agreement to
the lighter process; each cheap to revisit):
1. **A captured tab is played by the page only if its audio track reports
   `suppressLocalAudioPlayback === true`.** Otherwise it is analysed and not played, the
   page runs uncompensated and says so. No double audio in either case.
2. **An input device is never played.** Uncompensated by construction.
3. **Default delay 50 ms, slider 0–150 ms**, applied live.
4. **No `docs.html` section, no nav link, no `?demo=` preset**: the page is unpublished.

*Implementer's choices, flagged for the browser pass.*
1. **The tab capture's video track is kept alive**, requested at 64 px and 1 frame a
   second. I believe stopping it leaves the audio running in Chrome, but could not check,
   and the failure would be silent. If the browser pass shows stopping it is safe, stop it
   in `musicAcquire` (one line) and drop `videoTrackAlive` from the worry list.
2. **The analyser is left unconnected** on its output side. The Web Audio spec allows it
   and it is the common way to analyse a microphone; if a browser did not pull an
   unconnected analyser the page would show no kicks at all for input devices and
   unsilenced tabs — visible at once.
3. **`getSettings().suppressLocalAudioPlayback`** is where suppression is read from. If
   Chrome applies the constraint without reporting it there, a silenced tab would be
   treated as unsilenced: the tab would be SILENT (suppressed) and the page would not play
   it either. That is the one combination to look for.
4. A microphone's noise floor is not gated (see the README's "known limit").

*Still browser-only after this round.* Everything in M2's list, plus:
- that tab capture delivers audio through this code path (M0.a tested a throwaway page);
- whether `suppressLocalAudioPlayback` takes effect, and is reported, in Chrome on macOS;
- whether the unused video track can be stopped;
- that an unconnected analyser is processed;
- device labels appearing after permission; a loopback driver (BlackHole) as an input;
- that the 50 ms default actually lines the picture up with the sound, by eye and by
  `summary()`;
- Chrome's "Stop sharing" bar ending the session cleanly; a tab closed mid-capture.

*The real-browser checklist* (Chrome on the M1; serve with
`cd webgpu && python -m http.server 8766`, open
`http://localhost:8766/music.html?musicdebug`). After each Start, the line under the
sliders is the mode line; `musicDebug.audio` in the console is the record.
1. **Click track.** Source "click track" → Start. Mode line: "latency compensated (sound
   delayed 50 ms): the page plays this source, delayed". `musicDebug.audio`: `kind
   "click"`, `played true`, `compensated true`, `delayMs 50`, `suppressed null`,
   `sampleRate` 44100 or 48000. Move "sound delay" to 0 and to 150: at 0 the picture
   should trail the click slightly, at 150 lead it; `delayMs` follows. After 65 s,
   `musicDebug.summary()`: `lagSeconds` within ±0.017, `rebases` 0, `framesAtCap` 0;
   `bands[0].latencyFromScheduled.mean` ≈ 0.04, and perceived ≈ that − `delayMs`/1000 −
   `audio.outputLatency` + 0.017.
2. **File.** Source "audio file" → choose an mp3 → Start. Same mode line; `kind "file"`,
   `label` the file name. It stops by itself at the end of the file.
3. **Tab.** Play something in the Spotify web player in another tab. Source "another
   tab's audio" → Start → in Chrome's picker choose **Chrome Tab**, the Spotify tab,
   "Also share tab audio" ON → Share. Then read `musicDebug.audio.suppressed`:
   - `true`: the mode line says "…the tab is silenced and the page plays its audio,
     delayed". You should hear the music ONCE, coming from the page. `played true`,
     `delayMs 50`.
   - `false`: "NOT latency compensated: this browser did not silence the captured tab…".
     You should hear the music ONCE, from the tab. `played false`, `delayMs 0`.
   - **Report if you hear it twice, or not at all** — those are the two ways this can be
     wrong (choice 3 above).
   Also read `videoTrackAlive` (expected `true`) and `trackSettings`. Then press Chrome's
   "Stop sharing": the page should stop and say "the tab capture ended".
4. **Tab, wrong pick.** Start again and choose a *window* or turn the audio toggle off:
   the page should not start and should say to pick a Chrome TAB and switch on "Also share
   tab audio". Start again and press Cancel: "the tab capture was cancelled or not
   allowed".
5. **Input device.** Source "audio input device": the list shows "default input" and
   nameless entries. Start → allow the microphone → names appear. Mode line: "NOT latency
   compensated: an input device is never played back…". **Nothing should come out of the
   speakers from the page** (clap: the picture kicks, no echo). `kind "input"`, `played
   false`, `delayMs 0`. If BlackHole is installed, pick it and route the system output
   through it.
6. **Switching.** While the tab capture is live, change the source to "click track": the
   sharing bar should go away and the page stop. Start: no flurry of kicks in the first
   second.
7. **To settle choice 1** (optional): in the console during a live tab capture,
   `music.audio.stream.getVideoTracks()[0].stop()` — if the kicks carry on, the video
   track can be stopped in the code.

*What is left after M4.* The single fresh review of M3/M4; the browser pass above; the
decision to ship (remove the `pages.yml` prune, link the page, write its `docs.html`
section); the noise-floor guard for microphones; the detector on an `AudioWorklet` if a
device cannot hold 60 Hz; 512² live, which needs about five times the M1's step rate.

## Decisions (Alfred, 2026-11-21)

1. **Resolution: 256² for live use.** Accepted; M0.c shows ~2× headroom on the M1.
2. **Drag: music mode only**, and music mode is **its own page** (`music.html`), so the
   2D app is untouched.
3. **Latency policy: open.** Proposal: offer every source, and cancel the latency where
   the page plays the audio itself (file, tab) — see Latency above.
4. **The offline script is committed** (`5de685a`, branch `music-turbulence`), and the
   causal detector is **accepted** on the side-by-side movie (M0.b gate passed).

## Not in scope

- Recording a live session with its audio: `Recorder` writes video-only MP4, and the
  offline script already makes synced movies.
- 3D.
- Tempo/beat tracking, pitch → direction, stereo → imbalance. Onsets per band proved
  enough; add nothing until the live version is judged against the offline one.
- Browsers other than desktop Chrome for tab capture. Safari and phones get file and
  microphone only (to be confirmed in M3, not designed around).
