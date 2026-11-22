# MUSIC_EFFECTS_PLAN — phase 2 of the music project: variety

> **STATUS: step 1 (E1 + E2 offline) built 2026-11-21, movies rendered, awaiting Alfred's
> verdict — see "Step 1 record" at the end. Nothing else built.** Drafted 2026-11-21. Alfred accepted it the same day as the
> initial plan, with three decisions (end of file): the E1 type table as the starting
> point, the bass exempt from stereo placement, and the mean field (E4) rather than the
> pan.** Phase 1 (MUSIC_PLAN.md) is published:
> `examples/music_turbulence.py` offline, `webgpu/music.html` live. Read that plan's
> "What the offline version established" and "Decisions" first; nothing there is reopened
> here.

## Why

Phase 1 maps one thing — an onset in a frequency band — to one thing: a set of isotropic
gaussian blobs at uniformly random positions, independent for z+ and z-. It follows the
tempo well (Alfred: "works very well", "tab capture works extremely well"), and after a
minute it looks the same whatever is playing. This phase adds variety on three time scales:

| scale | what changes | effects |
|---|---|---|
| one onset | what a kick looks like and where it lands | E1 kick types, E2 stereo placement |
| a beat | the camera | E3 pulse |
| a section (10 s+) | the background the kicks land in | E4 in-plane mean field |

E5 (zoom-out re-gridding) and E6 (sustained sound) are optional and come last.

## Rules carried over from phase 1

- **Offline first.** Each effect lands in `examples/music_turbulence.py`, Alfred judges a
  movie, and only then is it ported to the page. The script stays the reference; where
  `music.js` ports a rule, `devtools/checkmusic.js` holds it to a fixture written from the
  python (`gen_musicfixture.py`).
- **Every effect is a switch, off by default** until Alfred has seen it. With all of them
  off the script and the page must produce what they produce today (the fixtures and the
  byte pins are the proof).
- **Kicks stay state edits**, never a forcing term; adaptive dt, kicks as events in `t`.
- Lighter process (agreed 2026-11-21): one implementer per phase, gates run by the
  implementer, one adversarial review at the end of the whole plan, not per phase.
- `examples/data/music/beths.wav` is Alfred's commercial track: use it locally, never send
  it to an outside service. The click track and the Open Goldberg recording are the
  shareable test inputs. Run the script with `TARANIS_PRECISION=32`.

## E1 — kick types per band

A band currently owns a width σ and a blob count. It gains a **type**. Types that need no
shader change, because `Solver.kick` already takes a per-channel list
(`{x, y, sigma, amp, pol}`) and `phi += (z+ + z-)/2`, `psi += (z+ - z-)/2`:

| type | list | what it is |
|---|---|---|
| `elsasser` | independent sets for z+ and z- | today's kick |
| `vortex` | the SAME blobs in both channels | velocity only (psi untouched) |
| `island` | the same blobs, opposite sign in z- | magnetic only (phi untouched) |
| `dipole` | `vortex` pairs: two opposite-signed blobs ~1.5σ apart, random orientation | a self-propelling velocity dipole |

One type needs the shader:

- `sheet`: an anisotropic gaussian, widths (σ, r·σ) with r ~ 4 and a random orientation θ.
  In `blobKickWGSL` the envelope `exp(-σ²k²/2)` becomes
  `exp(-(σ_a² k_a² + σ_b² k_b²)/2)` with (k_a, k_b) the wavevector rotated by θ, so a kick
  entry grows from one `vec4` to two. Normalise so the peak gradient ACROSS the sheet is
  `amp` (potential peak `amp·σ·√e`, as for a blob of the narrow width). An isotropic entry
  must give bit-identical output to today's shader; record the WGSL change as an allowance
  in `checksolver2d.js` and extend `checkkick.js` + `gen_kickfixture.py` with a sheet row.

A plane-wave type was considered and dropped: one wave in one channel is an exact nonlinear
solution, so it sits as static stripes.

Default assignment to try first (accepted by Alfred as the starting point; all other combinations reachable by a
`--kick-types` list, one entry per band, low to high): bass `dipole`, low-mid `vortex`,
high-mid `island`, treble `sheet`. With 5–6 bands repeat the middle two.

Check for each type, offline, at fp64: the untouched potential stays exactly zero from
rest (`vortex`, `island`), the energy added by one kick against the closed form, and that
a `dipole` moves along its axis at the expected speed for a few turnover times.

## E2 — stereo placement

Each band's blobs land where the band sits in the stereo image instead of uniformly in x.

- Onset detection stays on the mono mix, unchanged, so the detector fixtures hold.
- At an onset, pan = (E_R − E_L)/(E_R + E_L) for that band, from the two channels' band
  magnitudes in the same analysis frame. Offline `load_wav` has to keep both channels; the
  page needs the two channels separately (check how the capture graph feeds the detector —
  a mono downmix upstream of it would have to become a splitter).
- x centre = L·(0.5 + 0.4·g·pan), clipped; each blob is jittered about it by a gaussian of
  width 0.12 L; y stays uniform. `g` is a gain (default 1, try 2–3: most mixes are narrow).
- **The lowest band is exempt** (Alfred, 2026-11-21) and keeps uniform placement: bass is
  centred in nearly every mix, so it would land in a central column. A `--stereo-bands`
  choice can override which bands take part.
- Mono input: pan = 0 always, which would put everything in the centre — detect identical
  channels and fall back to uniform placement.

Optional second axis, only if x works: the band's spectral centroid, as a fraction of the
band's log-frequency range, sets y the same way.

## E3 — camera: pulse (display only)

**The pan is not being built** (Alfred, 2026-11-21: the mean field, E4, is the better of
the two). For the record: a uniform velocity in a periodic box is a Galilean boost, so it
would have been a display scroll with no effect on the dynamics.

- **Pulse:** zoom s(t) ≥ 1 about the display centre, kicked to 1 + a·strength (a ~ 0.08)
  on a bass onset and relaxing with a ~0.25 s time constant. The box is periodic, so the
  zoomed view just samples the tiling; no edge handling.
- The arrows overlay, the recorder and the fixed colour scale must all see the same
  transform or none of it; arrows are off by default on the music page, so the simplest
  correct choice is to disable them while the camera is on.

## The slow trigger (used by E4)

Something that fires at section changes, a few times a minute, causally:

- per-band log energy, smoothed over 2 s;
- novelty = distance between its mean over the last 4 s and over the 4 s before that,
  normalised by a trailing 60 s percentile;
- a trigger is a local maximum above a threshold, at most one per 8 s.

Fallback if this proves fiddly on real tracks: every 16th bass onset. Offline, use the
same causal form as the page (as `--causal` does for onsets), so both show the same thing.
Port to `music.js` with a fixture, like `MusicDetector`.

## E4 — in-plane mean field

A uniform in-plane field B0 is NOT a symmetry: z+ streams along −B0 and z- along +B0 at
the Alfvén speed (check the sign against the bracket conventions, do not assume), and they
interact only where they pass through each other. With independent per-channel kicks, each
hit visibly splits into two counter-streaming sets. Direction is redrawn on the slow
trigger and rotates to its new value over ~2 s; magnitude is a setting, a multiple of the
kick amplitude (start at 0.5: with a 1 s memory a blob then crosses ~1/12 of the box before
it fades — raise until the streaming is obvious).

- **Offline:** the script already owns its step loop and passes `rhs` and `set_timestep`
  to the stepper, so wrap both — no change to `taranis/`. The extra term is linear and
  k-local: (B0·∇) coupling phi and psi, i.e. a constant added to ∇ψ in the brackets. Add
  |B0|/dx to the CFL rate. B0(t) is known in advance offline: pass it as a function of
  `state.t` (a traced value), not a closed-over constant, or every change recompiles.
- **Page:** a uniform (B0x, B0y) added to ∇ψ in the bracket kernel, and to the CFL
  reduction. This touches pinned WGSL: record the allowance, and B0 = 0 must reproduce
  today's fields bit for bit. Mark `_cflStale` when B0 changes.
- **Gate (both sides):** one z+ blob from rest with B0 on and no z- translates rigidly at
  the Alfvén speed in the right direction, shape unchanged to dissipation; the same for
  z-, opposite direction; total energy unchanged by the term.

## E5 — zoom-out re-gridding (optional; decide after E3's pulse)

A true zoom INWARDS is not available: a magnified sub-window of a periodic box is not
periodic. Zooming OUT is exact. The display shows a growing piece of the tiling, and at a
factor 2 the state is replaced by its own 2×2 tiling — mode k moves to 2k, modes above half
the dealiasing cutoff are dropped — and the display snaps back to 1. Kicks stay at
box-relative scales, so old structure shrinks steadily into the dissipation range while new
kicks arrive large.

Costs to weigh before building: the tiling's repetition is visible between factors 1 and 2;
each re-grid discards the upper half of the spectrum (energy drops — measure it); on the
page it is a new k-space remap kernel. Build offline only, show Alfred, then decide.

## E6 — sustained sound (optional)

Only onsets act, so pads and held notes give nothing. A weak stream of small kicks at a
rate proportional to each band's smoothed energy would fill those passages. Risk: it blurs
the beat, which is the one thing phase 1 does well. Offline experiment only, behind a
switch, judged against the same passage without it.

## Order and hand-off

1. **E1 + E2 offline** (kick list and placement only; `sheet` needs `make_kick` extended).
   Movie on the click track, the Goldberg excerpt and locally on the Beths track.
2. **E3 pulse offline.** Movie.
3. **Slow trigger + E4 offline.** Movie.
4. Alfred picks what stays and the defaults.
5. **Port the survivors to the page**: `music.js` rules with fixtures, the `sheet` shader
   and the B0 term with their allowances, controls on `music.html`, README sections.
6. E5 / E6 experiments if still wanted.
7. One adversarial review of the whole phase; then merge and publish at Alfred's word.

## Decisions (Alfred, 2026-11-21)

- The E1 type table is the starting assignment.
- The bass is exempt from stereo placement.
- Mean field (E4), not the pan.

## Open questions for Alfred

- Whether the effects get individual controls on the page or one "variety" preset switch
  (not needed before step 5).

## Step 1 record (2026-11-21): E1 + E2 offline

In `examples/music_turbulence.py`; gate `examples/music_turbulence_check.py`
(`TARANIS_PRECISION=64`, all pass). Switches: `--kick-types` (`variety`, one
name, or a list per band), `--sheet-ratio`, `--stereo`, `--stereo-gain`, `--stereo-bands`,
and `--click-stereo` (a stereo click track: snare left, hats alternating — the test input
with known pans).

**Switches off = before**, checked once against the pre-change tree: three click-track
configurations (offline detector, `--causal`, 64² with 3 bands and memory 0.4) at fp32 and
fp64 — final fields, energy trace and step count bitwise, the mp4 files byte-identical —
and `gen_kickfixture.py` / `gen_musicfixture.py` regenerate both fixtures unchanged.

What was decided while building (each is a one-line change if Alfred wants otherwise):

- **`sheet` is an Elsasser-type kick** (independent anisotropic sets in z+ and z-); the
  plan's table did not say. Orientation uniform in [0, π), ratio 4.
- **Sheets are built in k space** (the periodised gaussian's analytic transform — the
  page kernel's form), isotropic lists stay in real space exactly as before. A 6-column
  list at ratio 1 reproduces the 4-column one to 1e-15, so the port can use one form.
- **Counts follow energy.** A sheet carries (r + 1/r)/2 = 2.1 blobs' energy, so a band's
  count is divided by that (treble at 256²: 15 blobs → 7 sheets). A dipole is `nblob`
  PAIRS and carries 1.5× a blob's energy per pair (closed form
  2[1 − (1 − q)e^(−q)], q = d²/4σ²) — not compensated, so the bass kick is 1.5× stronger
  under `variety`.
- **A dipole travels PERPENDICULAR to the line joining its two blobs**, at 0.43 of the
  naive estimate (the partner's speed at a blob's centre, 0.80·amp at 1.5σ): 0.34·amp
  measured over three turnover times, steady to ±10%, energy kept. The initial rate
  matches 2D Euler evaluated independently in numpy to 4 digits. So a bass dipole at
  σ = 0.45, amp 1 crosses ~0.05 of the box in the 1 s memory — visible as a drift, not a
  flight.
- **Pan** uses band ENERGY (Σ|X|²) in one Hann frame centred on the onset, per channel,
  computed after detection (the detectors are untouched). An instrument at amplitude
  gains (1 − p, 1 + p) therefore reads 2p/(1 + p²). It is the pan of everything sounding
  in the band at that instant, not of the onset alone.
- **Measured pans** (offline detector, threshold 0.3, 4 bands): Goldberg var. 1, 45 s —
  every band mean +0.06…+0.13, std 0.21–0.24; Beths 30–90 s — bass −0.29 ± 0.18,
  low-mid +0.05 ± 0.24, high-mid +0.14 ± 0.22, treble +0.34 ± 0.14. So the bass of that
  track is NOT centred, and gain 2 is what the movies use.
- `examples/data/music/beths.wav` is MONO; the stereo movies use
  `beths_stereo_30-90s.wav`, cut locally from the mp3. The page's capture graph was not
  looked at (step 5).
- Found and fixed on the way: since 4258643 the script raised `NameError` for every
  colormap except `brat`/`brat_r` (`get_cmap` used an import local to `run`).

Movies (256², fp32, in `examples/data/music/`, not in git): `e12_click_variety_stereo`,
`e12_goldberg_variety_stereo` (gain 2), and on the Beths excerpt `e12_beths_phase1`
(switches off), `e12_beths_variety_only`, `e12_beths_stereo_only` (gain 2),
`e12_beths_variety_stereo` (gain 2).

**Sheet round 2 (2026-11-21, Alfred: the sheets were not visible after the start).** Not
dissipation: a treble sheet (σ = 0.091 at 256²) sits at k ≈ 11–14, where the k⁸ rate is
~2e-5/s; the rate reaches 1 at k ≈ 53. At 7 sheets a treble kick added 3% to the energy
(bass 18%). Changed at his instruction: the count is back to `nblob` (the energy
compensation above is gone), `--sheet-energy` multiplies a sheet's energy (amplitude by
its square root), and a second type `sheet-island` (one set, sign flipped in z-: psi only)
exists beside the Elsasser `sheet`. With 15 sheets at `--sheet-energy 2` the treble kick
adds 11–12% on the Beths excerpt, 20% on the click track. Movies, all with stereo gain 2:
`e12b_{beths,click}_v1_sheet_elsasser`, `e12b_{beths,click}_v2_sheet_magnetic`. The
`e12_*` movies above predate this (7 sheets, energy 1). Gate: 49 checks.

**Alfred, 2026-11-21: the stereo placement "looks good"; of the two sheet variants,
variant 1 (Elsasser sheets, 15 per kick, `--sheet-energy 2`) is the better.** The defaults
are his to fix at step 4; `sheet-island` stays in the script as an available type.

## Step 2 record (2026-11-21): E3 pulse offline

`--pulse [a]` (0.08 if no value; 0 = off, the default), `--pulse-tau` (0.25 s),
`--pulse-band` (0). Two pure functions in the script: `pulse_zoom` (zoom per frame from
one band's onsets: an onset lifts the zoom TO 1 + a·strength and never lowers it, the
excess decays as exp(−t/τ), causal) and `zoom_image` (bilinear, periodic wrap, about the
image centre; exactly the input at zoom 1). The zoom is applied to the normalised frame,
so the colour scale is computed on the unzoomed field. Gate: 10 new checks (59 in all) —
the zoom law, the sampling against an analytic periodic field at 1.08, 1.5 and 0.5, and
that `run(pulse=)` leaves fields and energy bitwise while the movie changes. Pulse off:
the three baseline movies are still byte-identical.

On the Beths excerpt at a = 0.08 (3.8 bass onsets/s, median strength 0.71): mean zoom
1.034, 5–95% 1.004–1.067, above 1.02 on 79% of frames — at that onset rate the camera
rarely gets back to 1, so it reads as a wobble about ~1.03 more than as separate hits.
A shorter τ or a higher threshold for the pulse alone are the knobs if that is the verdict.

Movies (variety, stereo gain 2, sheet energy 2): `e3_beths_pulse0.08`,
`e3_beths_pulse0.15`, `e3_click_pulse0.08`.

**Found: the script is not run-to-run reproducible at 256² on a long track, switches off
included** (two identical 12 s Beths runs: same step count, energies differ from frame 5,
final fields differ; also on a143b2e). At 128² and 64² on the click track runs repeat
bitwise, which is where every "bitwise" statement in these records was made.
**Cause, localised:** one solver step at 256² differs between calls by ~1e-8 (fp32) under
XLA's multi-threaded CPU backend and repeats exactly with
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`; the kick,
the transforms alone and `observe` repeat exactly either way. So it is the solver's
compiled step under threading, not the script. **Left as is (Alfred, 2026-11-21: fine not
to be bitwise)**; the flag is the way to get a repeatable run if one is ever needed. Consequence: two
256² movies of the same settings are different realisations after ~10 s (the three Beths
movies of step 2 and `e12b_beths_v1` share E to 4 digits at 10 s and differ at 30 s), so
compare effects on such movies by eye, not frame by frame.

**Alfred, 2026-11-21: the pulse movie "looks good".**

## Step 3 record (2026-11-21): slow trigger + E4 mean field, offline

**Slow trigger** — `SectionDetector` (per-frame class, fed the same spectra as
`CausalOnsetDetector`, so it ports the same way) and `detect_sections`. As planned, with
these specifics: band level = log10 of band energy over a 30 s peak-hold of the total,
floored at −60 dB, 2 s exponential smoothing; novelty = rms over bands of (mean of the last
4 s − mean of the 4 s before), in decades; normalised by the **75th** percentile over 60 s
but by no less than **0.1 decade** (the absolute floor: steady noise never fires);
trigger = local maximum above **1.5**, one frame late, at most one per 8 s. It fires
~5 s after the change (measured 5.3 s on a synthetic balance flip) and never in the first
8 s. `--section-threshold` moves the 1.5; `--section-every N` is the fallback (every N-th
bass onset).

How it does on real tracks, which I cannot judge by ear:
- Beths, whole track (251 s): 7 triggers, at 43, 69, 83, 133, 146, 171, 239 s.
- Goldberg variations 1–4, 50 s of each back to back (true boundaries at 50, 100, 150 s):
  4 triggers at 32, 55, 84, 133 s — it catches the first boundary and misses the other
  two; dynamics INSIDE a variation move the band levels more (novelty peaks ~0.95 decade)
  than two of the boundaries do (0.4–0.5). Band energy is a weak section cue for solo
  piano. At percentile 90 / threshold 1.0 (first try) the counts were 7 and 11.

**E4** — `--mean-field [b]` (0.5 if no value, in units of amp; 0 = off). `mean_field()`
builds B0(t) of fixed magnitude whose angle turns by a random 45–180° either way over 2 s
at each trigger (its own RNG stream, so the kicks keep theirs), as a jax function of
`state.t`; `Stepper(b0=)` wraps `rhs` and `set_timestep`, `taranis/` untouched.

- **Sign, derived and measured:** B0 = ẑ×∇ψ0 adds ∂tφ = (B0·∇)ψ, ∂tψ = (B0·∇)φ, so
  ∂t z± = ±(B0·∇)z±: **z+ streams along −B0, z− along +B0**, as the plan guessed.
- Gate (fp64, 128², no dissipation): one z+ blob after t = 2 at |B0| = 0.5 equals the
  initial fields shifted by −B0·t to 1.4e-6 (the other direction: error 1.9), z− likewise
  along +B0; energy change −2.8e-6; under a field turning by 2 rad the displacement is
  ∫B0 dt to 1e-6; with both channels overlapping and nonlinear, energy to −6e-7.
- CFL: |B0|(1/dx + 1/dy) is added to the rate (an upper bound, direction-independent),
  and the script's "slow flow → fixed dt" test counts 2|B0| as speed. Same blob to t = 2:
  204 adaptive steps with B0 = 0.5, 102 without.

Gate: 74 checks. Switches off: the three baseline movies still byte-identical.

Movies (variety, stereo gain 2, sheet energy 2, pulse 0.08): `e4_click_b0.5` (30 s,
`--section-every 8`), `e4_beths_b0.5` and `e4_beths_b1.0` on `beths_stereo_30-150s.wav`
(120 s, detector).

Rendered: on the 120 s Beths excerpt the detector redraws the direction at 39, 53, 92,
103, 116 s (69, 83, 122, 133, 146 s of the track; the whole-track run gave 69, 83, 133,
146 there — the normalisation depends on the history it has seen). **Cost for the page:**
the field's CFL rate raises the step count from ~360 to **495 steps per second of audio at
B0 = 0.5 and 590 at B0 = 1.0** (256², this excerpt), against the ~600 steps/s the M1 was
measured to manage live — B0 = 1 leaves no headroom there. The bass beat contrast is
unchanged by the field (+0.14, +0.15).

**Strong-field round (2026-11-21).** Alfred on `e4_beths_b0.5`/`b1.0`: streaming visible
only early on, the direction changes not visible. Reason: the developed flow has peak
speed ~3.5 and rms ~0.9 (units of amp), so B0 ≤ 1 is at or below the flow itself.
Rendered `e4_beths_b3` and `e4_beths_b6` (first 60 s of the excerpt, redraws at 39 and
53 s): 960 and 1560 steps per second of audio. In single frames of the B0 = 6 movie the
structures are elongated along one direction, and that direction differs between 30 s and
50 s. Energy and beat contrast are unchanged by the field (E ≈ 0.42 at 30 s in both; bass
contrast +0.13). At these strengths the page cannot take the field through the CFL limit
(~600 steps/s live on the M1): it would have to go into the exact linear propagator
instead (L gains the off-diagonal ±i k·B0 pair — the putzer2 form the 2D page does not
have today).

### Bass-hit random walk and the exact linear step (2026-11-21)

Alfred's verdict on the strong-field round: the field is clearly visible at 3 and 6, but
the section trigger does not switch at sensible moments, and a fixed 1 is nearly isotropic
once the energy has built up. His replacement, built offline:

- `--field-walk [deg]` (30): each bass onset (the pulse band) turns the field by a random
  angle, rms `deg` times the onset strength; relaxes to the new value over 0.1 s.
- `--mean-field R` is then |B0| in units of the rms fluctuation sqrt(2E) at the time, so
  the field keeps its hold as the energy builds (fluctuations are ~0.9 after the buildup,
  which is why a fixed 1 looked isotropic and 3 did not).
- `--ratio-walk s`: the ratio also random-walks by rms `s` per full-strength onset,
  reflected into [0, R].
- The field is held constant over each advance (at most one frame) and passed as a traced
  argument (`Stepper(b0="driven")`), which is how the page would do it with a uniform.
- **It is in the exact linear step** (Alfred: essential): the off-diagonal pair
  `i k.B0` joins the solver's L as a putzer2 operator built per call with
  `propagators.build`, nothing in `taranis/` changed. `"driven-explicit"` keeps the RHS +
  CFL version for comparison. The section-trigger mode is still explicit.

Cost, 256^2, first 60 s of the local excerpt, steps per second of audio: ratio 2 explicit
738, exact 367; ratio walk 0-3 explicit 621, exact 368 -- the exact step costs the same
number of steps as no field at all (~360). Wall time per step is higher (the putzer2
coefficients are evaluated per stage), so offline wall time did not drop; on the page the
equivalent is a 2x2 complex rotation per mode in the linear kernel.
Gate: `driven_field_gates` (78 checks in all). Movies: `e4_beths_walk_r2.mp4`,
`e4_beths_walk_r0-3.mp4` (exact), `*_explicit.mp4` (the same with the explicit field).

## Step 4 (Alfred, 2026-11-21) and step 5 record: the port to the page

**Step 4, Alfred's decisions.** Everything stays and everything is settable on the page,
with these defaults: kicks `variety` with sheet energy 2; stereo placement on, gain 2;
pulse 0.08; mean field on as the bass-hit walk (direction and strength), upper ratio 1.5
(0-2 and 0-1 "both have their qualities"); the section trigger is dropped from the page and
stays in the offline script. The mean field goes into the page solver's exact linear step,
on condition that the other pages' performance is not affected.

**Step 5, built (uncommitted).**

- `solver2d.js` / `physics.js`: `sheetKick` (the anisotropic twin of `blobKick`, under the
  same `kick: true`) and a `b0: true` variant of the `stage` kernel with the field in the
  exact linear step (a 2x2 rotation per mode; NM invocations). `setMeanField`. The 2D page
  asks for neither flag: its emitted kernels are byte-identical (checksolver2d leg 2) and
  its stage dispatch count is unchanged.
- `music.js`: `musicKickTypesFor`, `musicKickBlobs` (the six types, `xc`), `musicPan`,
  `musicPanX`, `musicPulseHit/Zoom`, `musicWalkInit/Hit/Relax/Field`, the stereo click
  track. `musicKickList` without its new argument is the phase-1 list, draw for draw.
- `music.html`: an `effects` control group; two channel analysers off a splitter; the pan
  read when an onset is declared; the pulse as a CSS scale of the card's canvas; the field
  set every frame from the energy readback.
- Fixtures: `kick_offline.json` gained `sheet_cases`, `music_ref.json` gained `effects`;
  everything they held before is unchanged (checked by comparing the JSON).
- Gates: checkkick leg 7, checkmusic leg 13 (38 rows), allowances in checksolver2d. The
  legacy music legs run with the effects switched off through the page's own controls.
- Real browser (Chrome, M1, click track, defaults): 597 steps/s, 6.3 steps a frame, no
  rebase, no frame at the step cap; types, placement, pulse and field all observed live
  through `?musicdebug`. `rmhd2d.html`: 609 steps/s.

**Differences from the offline script, deliberate.** The pan is read from the analysers'
current Blackman window when the onset is declared, not from a Hann frame centred on it.
The pulse is a CSS zoom about the centre (no periodic wrap needed at zoom >= 1) and is not
in a recording. The walk's RNG is mulberry32 + Box-Muller, not numpy's. The ratio step is
fixed at range/6 per full-strength hit (the rule used for the 0-2 and 0-1 renders).

**Not done.** A stereo file or a tab capture in a real browser; any device but the M1; the
user-facing docs page; step 6 (E5/E6) and step 7 (the adversarial review).

## Step 6 record (2026-11-22): E5 zoom-out re-gridding, offline

Alfred: E6 (sustained sound) is not being built; E5 is worth seeing, offline first.

- `--zoom-out [T]` (20): the picture shows a window of the tiling that grows as 2^(t/T)
  about the display centre (`zoom_image` at s < 1, with a gaussian blur matched to the
  coarser sampling); every T seconds `regrid` replaces the state by
  g(x) = f(2x - x_c)/2 -- mode k to 2k with the phase of the display's centre pixel -- and
  the window is the box again. The 1/2 keeps the VELOCITIES, so the picture of |u| does not
  jump and every surviving mode keeps its energy. Modes that land past n/2 or outside the
  dealiasing cut are dropped. It multiplies with the pulse's zoom.
- Gate `regrid_gates` (85 checks in all): g against f sampled at 2x - x_c to 8e-15, energy
  to 2e-16, the picture across a re-grid, what is dropped, a short run.
- Measured, 256^2, first 60 s of the local excerpt with the page's default effects: each
  re-grid keeps 97-99% of the energy (T = 10: 0.978 0.983 0.979 0.987 0.972; T = 20: 0.982
  0.987), so the dropped upper half of the spectrum is a small loss. Steps unchanged
  (22,200-22,300 for 60 s). Bass beat contrast +0.14, as without it.
- Known seam: just before a re-grid the picture is the blurred half-zoom, just after it is
  the sharp new state.
- Movies: `e5_beths_zoomout20.mp4`, `e5_beths_zoomout10.mp4`. Uncommitted.
- For the page, if wanted: a k-space remap kernel (k to 2k, phase, mask) and a render that
  can show the tiling at zoom < 1 -- the CSS scale used for the pulse cannot, since it has
  only the one box to draw.
