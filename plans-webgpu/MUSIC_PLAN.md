# MUSIC_PLAN — live music drives the 2D demo

> **STATUS: drafted 2026-11-21; M0 measured; M1 committed (14ed94b), M2 (c960488), M3 + M4
> (cc92edd). A review-and-fix round — the reviewer's nine findings, automatic sync from the
> audio output's latency (Bluetooth), a detector noise floor, two colormaps — is implemented
> and uncommitted. The page runs in Chrome on the M1; it is unlinked and pruned from the
> deploy.** Phase M0 was three
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

## Real-browser record after M4 (2026-11-21, M1, Chrome)

- **Tab capture: Alfred, by eye and ear — "works extremely well".** His debug record:
  `kind "tab"`, **`suppressed true`**, `played true`, `compensated true`, `delayMs 50`,
  mode "the tab is silenced and the page plays its audio, delayed", 48000 Hz, nfft 2048,
  `videoTrackAlive true`, track `deviceId …?local_echo=false`, `outputLatency 0.168`. So
  **Chrome honours `suppressLocalAudioPlayback` and reports it in `getSettings()`** — the
  compensated tab path is the live one, and the "no sound at all" failure did not occur.
- **Microphone: not assessed** (a loud room). The missing noise-floor guard is the known
  gap there.
- **Click track on cc92edd, 22 s, by automation:** plays through the delay, mode line
  "latency compensated (sound delayed 50 ms)", 0 rebases, 0 frames at the cap, kick
  latency 0.039 s. `lagAfterSeconds` mean −0.0023, min −0.0066, max 0 — after a frame's
  steps the simulation is within a step of the audio clock, which confirms that the
  ~15 ms `lag` in the M2 record is the pre-step reading and not an offset in the picture.
- **Output latency — was OPEN, now built (fix round, below).** That run reported
  `outputLatency` 0.168 s where the M2 run reported 0: the device was AirPods (Bluetooth),
  so the 0.168 s is real, and with the 50 ms sound delay of cc92edd on top the picture LED
  the sound by roughly 168 + 50 − 39 − 17 ≈ 160 ms in the session Alfred judged good. A
  DelayNode can only delay sound; when the output is already later than the detection,
  the kicks must wait instead. See "Review and fix round", part B.

## Review and fix round (2026-11-21, after cc92edd) — committed as f4c92f2

One fresh reviewer read M3/M4 and mutation-tested the gate; Alfred used the page on
Bluetooth earphones. **No wrong number was found in the detector or in the clock's
tracking.** What was found were state-machine and policy bugs, each reproduced under the
stub, plus one design gap (output latency). Light process, by agreement: gates where
cheap and where a failure would be silent, the affected gates run once, **no mutation
round** — none of the rows added here has been mutation-tested, and there is no further
review.

### A. The review's findings

| | finding | what changed | gated? |
|---|---|---|---|
| F1 | a second Start (or a source change) while `ctx.resume()` was pending orphaned a source that played for ever | `music.starting` stays true across every await of `musicStart`; the generation is re-checked after the LAST one; a stale acquisition is given back | yes — a `resume()` promise the leg holds, click and tab |
| F2 | at `rate` < 0.37 the hold-back and the 0.25 s rebase deadlocked: no steps, BEHIND lit falsely | the hold-back applies only while the simulation is less than half of `lagMax` behind; being AHEAD is allowed `lagMax` plus one step before a rebase | yes — pure, and on the page: silence at rate 0.25, 0.35 and 2 |
| F3 | BEHIND latched across sessions | Stop and Start reset `music.clock` and `music.behind` | yes |
| F4 | a device that cannot keep up never lit BEHIND (a long frame was excused as a hidden tab) | "hidden" comes from `visibilitychange` (`resumed`), not from the frame gap; a long gap while visible is behind | yes — both |
| F5 | a granted capture leaked if anything threw afterwards | `musicAcquire` stops the stream on any throw after the grant; `musicStart` stops whatever it acquired and did not hand to a session | yes — tab and input, at two points each |
| F6 | the tab path never checked the capture was a tab | `systemAudio: "exclude"` in the request; the page plays a capture only if its video track reports `displaySurface === "browser"` | yes |
| F7 | after music, any noise floor became ~8 kicks/s at strength 0.9 | a noise floor on the normaliser (below) | yes — parity with the python on two option sets, and behaviour on signal + noise |
| F8 | with `cfl_every` 4 the simulation can end a frame up to ~50 ms ahead in a decaying flow | **noted, not changed.** dt is frozen for four steps while the flow slows, so `n·dt` overshoots the target; bounded by one `cfl_every` block, self-correcting at the next CFL step. `cfl_every` 1 on the page's slider removes it at the cost of a reduction per step | no |
| F9 | docs named `lagSeconds` for the within-a-frame criterion; `pages.yml` said "these two lines" of four; the run button said Pause for a full stop; a non-finite state froze silently | the text names `lagAfterSeconds`; the comment says which line; the button reads Start / Stop; a non-finite readback stops the page with a message | button and non-finite: yes; the two texts: no |

Reviewer's surviving mutations closed with rows: the look-ahead bound scaling with
`rate`; Stop clearing `music.behind`; the dropped-blob counter accumulating; an input
device with no audio track; another file chosen mid-play; ANY track ending the session; a
stale session's track not stopping its successor; the detector's history trimming (a
600-frame run against an untrimmed twin).

**Two more changes to the clock that F2 required**, beyond the reviewer's suggested bound:
the ahead-rebase threshold became `lagMax` plus one step (with the bound alone, the single
step taken at rate 0.25 lands 0.27 s ahead, rebases, and the simulation runs three times
too fast); and a frame that kicked now takes its one step even when the simulation is
ahead (at a low rate it can be ahead by most of a step when a kick lands, and the kicked
flow must not wait for the audio to catch up — the step is a small, freshly computed one).

**The noise floor (F7) — what was chosen and how it fails.** Each band keeps a peak-hold
of its own raw flux decaying over `peak_decay_s`; its normaliser is the running 99th
percentile as before, but never less than `flux_floor` times that peak. Off by default in
the python `CausalOnsetDetector` (`flux_floor = 0`: `detect_onsets_causal` is bitwise
what it was at 94030e1, re-checked), on in the page at **0.3 over 120 s**. Measured (rock
track 30 s, then 60 s of white noise, 4 bands, threshold 0.3, a 30 s hold):

| noise level re the music | −40 dB | −60 dB | −75 dB | −90 dB |
|---|---|---|---|---|
| kicks/s without the floor | 8.8 | 9.4 | 9.3 | 9.5 |
| kicks/s with 0.3 | 4.9 | 0.0 | 0.0 | 0.0 |

On the music itself it removes 0.1% of the onsets (rock and Goldberg, 90 s each). The
page's 120 s hold was chosen so the protection lasts minutes, not one; **the cost figure
was measured at 30 s and not re-measured at 120 s.** Failure modes: (1) a source that is
noise from its first sample — a microphone in a quiet room — has no louder past, so it
still kicks; (2) after a loud passage, onsets weaker than 0.3 of it are missed for up to a
couple of minutes; (3) noise only 40 dB down is halved, not removed.

### B. Output latency: automatic sync

`musicSync` (pure, in `music.js`). Two delays the page did not choose — the detection
latency `MUSIC_LDET` = 0.039 s (measured) + one frame = 0.0557 s, and the output latency
`Lout` = `AudioContext.outputLatency` — and two it can add: D on the sound (a DelayNode,
only where the page plays the source) and H on the kicks (a queue). A hit entering the
analyser at 0 is heard at D + Lout and seen at Ldet + H. With `v = Lout − Ldet + trim`:

| routing | rule | laptop speakers (Lout ≈ 0) | AirPods (Lout 0.168) |
|---|---|---|---|
| the page plays it (click, file, silenced tab) | v ≥ 0: H = v, D = 0; v < 0: D = −v, H = 0 | D 56 ms, H 0 | D 0, H 112 ms |
| a tab the page does not play | H = max(0, v); D unavailable | H 0, picture 56 ms late | H 112 ms, exact |
| an input device | Lout does not apply (v = −Ldet + trim); H = max(0, v) | H 0, picture 56 ms late | the same |

- **Trim** is signed: positive makes the picture later, negative the sound later. It
  enters `v`, so it comes off whichever side is in force and spills to the other; both
  are ≥ 0 by construction and capped at 1 s.
- **The queue.** Onsets are queued with `due = declared + H` and applied on the first
  frame at or after it: one `kick()` each, before that frame's steps, head of the queue
  first. Cleared by Stop, a change of source, Reset and the return from a hidden tab; a
  behind-rebase leaves it alone.
- **The clock's target is not shifted** — agreed with the coordinating session's reading:
  a constant offset in audio time does not change the tracking, and simulation time's
  origin is arbitrary. Only when a kick is applied moved.
- `outputLatency` is re-read once a second; D moves by `setTargetAtTime`.
- The slider is now "sync trim (ms)", ±150, default 0. The 50 ms default delay of cc92edd
  is gone: on speakers the automatic value is 56 ms, the same thing by another route.

### C. Colormaps (Alfred's request)

`magma` (matplotlib's — the offline |u| movies' map; a degree-6 least-squares fit to its
256-point table, **max channel error 0.0226**, against 0.017 for the shared viridis) and
`brat` (black → #8ACE00, linear and exact). They are a music-page extension: `MUSIC_CMAPS`
in `music.js`, registered through a new opt-in `cmapExtend` in `common.js`; with nothing
registered `cmapNames()` is `CMAP_NAMES` and `cmapWGSL()` returns `CMAP_WGSL` itself, so
the other pages' menus and emitted WGSL are unchanged. The speed display starts on magma;
the user picks either from the display card's colormap select, where they follow the
shared four. Canvas, colour bar and a saved PNG's bar read one table.

### What stays browser-only after this round

- Everything about `outputLatency` as a NUMBER: that Chrome's value for a device is the
  delay actually heard (it is an estimate; the trim exists for that), that it updates when
  the output device changes mid-session, and what it reads on speakers on this machine
  (the M2 run said 0).
- That `setTargetAtTime` on the delay is click-free, and that a kick hold of ~110 ms looks
  right on Bluetooth.
- `displaySurface` being reported as `"browser"` for a tab in Chrome — if it is not, a
  tab would be analysed and NOT played while still silenced: **no sound at all**. Look
  for that first.
- `document.onvisibilitychange` firing as assumed.
- The noise floor on a real microphone and on a tab left silent for minutes.
- How magma and brat look.

**Checklist** (serve as before, open `…/music.html?musicdebug`):
1. **Laptop speakers, click track.** Start. Mode line: "sound delayed 56 ms, kicks held
   0 ms (output latency 0 ms, detection 56 ms): the page plays this source" (the first
   number drops by whatever `outputLatency` Chrome reports). `musicDebug.audio`:
   `delayMs ≈ 55.7 − outputLatencyUsedMs`, `holdMs 0`, `residualMs 0`, `trimMs 0`,
   `ldetMs 55.7`. The button reads Stop.
2. **AirPods, click track.** Connect them BEFORE Start. Mode line: "sound delayed 0 ms,
   kicks held ~112 ms (output latency ~168 ms, …)". `musicDebug.audio`: `outputLatency
   ≈ 0.168`, `delayMs 0`, `holdMs ≈ 1000·outputLatency − 55.7`, `residualMs 0`.
   `musicDebug.kicks.slice(-5)`: `tDue − tFire ≈ holdMs/1000`, `tApplied ≥ tDue`. By eye
   and ear the hit and the blobs should coincide; if not, move "sync trim" and report the
   value that fixes it.
3. **Switch output mid-session** (disconnect the AirPods while it plays): within about a
   second the mode line and `musicDebug.audio` change to the speakers' numbers, with no
   click in the sound.
4. **Tab on AirPods** (the session that showed the problem): the numbers of 2, with
   `kind "tab"`, `suppressed true`, **`surface "browser"`**, `played true`. If `surface`
   is anything else you will hear NOTHING — report it.
5. **Slow rate.** Rate slider to 0.25, click track: slow motion, no BEHIND.
6. **Hidden tab.** Switch tabs for 20 s and back: no BEHIND, no burst of kicks.
7. **Colormaps.** The display card's colormap select lists afmhot, viridis, RdBu,
   grayscale, magma, brat; it starts on magma; the colour bar follows the choice; a saved
   PNG shows the same map.
8. **Stop / Start.** Stop while BEHIND is lit: the next Start begins without it.

## Dissipation order: k⁸ (changed 2026-11-21, Alfred's request)

Where this plan says "hyper 2" and "3·amp/(n/6)⁴", the page and the offline script now use
**hyper 4 (k⁸)** with the rule restated as **rate 48·amp at the dealiasing cutoff n/3**
(`diss = 48·amp/(n/3)^(2·hyper)`; at hyper 2 that is the old number exactly). Measured
offline, 256², 14 s click track, final spectrum normalised to k = 10:

| dissipation | rate = 3 at k | E(40) | E(60) | E(84) | bass jump |
|---|---|---|---|---|---|
| k⁴, rate 48 at cutoff (old default) | 43 | 2.3e-2 | 1.0e-3 | 2.1e-5 | 0.35 |
| k⁸, rate 200 at cutoff | 51 | 6.0e-2 | 2.8e-3 | 3.2e-6 | 0.33 |
| **k⁸, rate 48 at cutoff (now)** | 60 | 6.8e-2 | 1.1e-2 | 2.1e-4 | 0.33 |
| k⁸, rate 12 at cutoff | 72 | 7.2e-2 | 2.1e-2 | 3.9e-3 | 0.33 |

Rate 12 flattens at the cutoff (the pile-up again); 48 still falls ×50 over the last 24
wavenumbers. One short run, judged on the spectrum only — not on dense music, and not by
eye. The solver's drag was already gated at hyper 4 (M1 review). The step-rate and sync
measurements in this plan were taken at k⁴; offline the step count rises ~5%.

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
