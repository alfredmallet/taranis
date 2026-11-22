# MUSIC_EFFECTS_PLAN — phase 2 of the music project: variety

> **STATUS: drafted 2026-11-21, nothing built. Alfred accepted it the same day as the
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
