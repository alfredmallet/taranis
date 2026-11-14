# Numerics notes

The *reasoning* behind the numerical choices in `taranis`. CLAUDE.md states the rules;
this file says why they are what they are, so neither has to do both. Written for someone
reading or extending the solver, not for someone running it (for that see
`docs/performance.md`).

## Fourier conventions

`grids.fft/ifft` are unnormalized `rfft2`, so an O(1) real field has O(nx·ny)
coefficients. This matters whenever an amplitude is synthesized directly in k-space
rather than transformed into it: the stochastic forcing multiplies its noise by
`nx·ny` (`shared_physics.ou_update`) precisely so that the resulting real-space forcing
is resolution-independent.

`rfft2` keeps `kx` two-sided and `ky` non-negative. Reality of the underlying field is
therefore *not* a per-coefficient property — it is a constraint **between** `(kx, ky)` and
`(−kx, ky)`, and it only bites on the two rows where both are represented: `ky = 0` and
`ky = ky_Nyquist`. Anything that writes k-space directly has to impose it explicitly.
Anything that gets there through `ifft`/`fft` gets it for free.

### One normalization for every energy-like quantity

All the `perp_*` reductions in `shared_physics`, and both spectra plus `energy` in
`diagnostics`, share one convention: multiply by the `ky`-doubling factor `kgrid.yfac`
(2 everywhere except the `ky = 0` and Nyquist rows, which are not duplicated), sum, then
divide by `nz·(nx·ny)²`. The consequence relied on in `tests/test_diagnostics.py` is that
the integral of either spectrum equals `diagnostics.energy` exactly, to round-off. A new
diagnostic that invents its own normalization silently stops being comparable to the
forcing power and the dissipation rate.

### Gradients are a tuple, one transform at a time

Every equation set's `grad_func` returns a **NamedTuple** of real-space `(2, nz, nx, ny)`
gradients — `rmhd.RMHDGrads(gphi, gpsi, gvort, gjpar)`, `gdi.GDIGrads(gphi, gN, gvort)` —
one per field, not a stacked `(nfields, 2, …)` array. It is a tuple, so positional
consumers are unaffected; the names label the order. The reason is memory, not taste. An
inverse FFT is an opaque kernel, so its input must be materialised whole: a
single batched `ifft` of `stack([i·kx·f, i·ky·f])` over all four RMHD fields forces an
8 u k-space array to be live *alongside* the ~8 u of real-space output it produces. Going
one field — in fact one component — at a time makes the k-space peak 2 u instead, and
returning a tuple is what keeps the output side from paying the cost back: any
`stack`/`concatenate` of the results would be a second full copy of the real-space set.

`shared_physics.grad_fields` is the shared implementation (it returns the plain tuple; the
recipe names it); `shared_physics.GRAD_CHUNK` (a module constant read at trace time,
deliberately not a `Parameters` attribute) sets how many fields share one inverse
transform. Its output is bitwise identical at every chunk size — this is a scheduling
knob only — and the default is 1. The arithmetic is unchanged either way; what changes is
which intermediates XLA has to keep alive.

Ordering does **not** buy anything further. At the RHS peak all eight real gradient
components are live, and the brackets only need six at a time (`gvort` and `gjpar` are each
consumed once), but no source-level reordering reaches that: retiring `gvort` early would
need the `gjpar` inverse transform scheduled after `bracket(gphi, gvort)`, and no statement
order creates that dependency. Measured at exactly 0.000 u across every reordering tried,
including a contract-breaking one — the ~2 u belongs to XLA's scheduler.

## Precision model

`jax_enable_x64` is turned on unconditionally at import (`taranis/__init__.py`) — x64
availability and *field* precision are no longer the same question. `TARANIS_PRECISION`
(`"32"`/`"64"`, default `"32"`, read exactly once by `taranis/_precision.py` at import)
sets `_precision.ftype`/`ctype`, the dtypes that `fields`/`forcing_state`/`forcing_scale`
are pinned to; it does not touch `SimulationState.t`, which is float64 at *both* field
precisions. This exists to fix a real failure mode: at fp32, `t + dt == t` exactly once
`t/dt > 1/eps32 ≈ 1.7e7`, so `while state.t < t_end` never terminates, and even before
that point roundoff random-walks `t` by `~sqrt(N)·eps32`. With x64 always on, `t` can be
fp64 for free while every field array stays at the dtype `TARANIS_PRECISION` selects — an
x64-enabled JAX process compiles an all-fp32/complex64 op graph to the same kernels it
always did; the flag only changes *default* dtypes and the promotion of any 64-bit input
that leaks in. The entire risk of the design is exactly that leak, which is why
`physics/__init__.py::construct_rhs` asserts `state.fields.dtype == _precision.ctype` and
`state.forcing_state.dtype == _precision.ctype` on every trace (`tests/test_precision_dtypes.py`
is the regression net; a pre-change reference lives in `tests/data/`).

Four rules govern any new code that touches both `t` and fields:

1. **Python scalars are weak-typed.** `0.5*fields`, IMEX tableau entries (plain python
   floats from the scheme tuples, e.g. `(b[k]*dt)*z`) do not upcast anything — leave them
   as bare python floats.
2. **Bare `jnp.array(...)`/`np.array(...)` (no dtype) are STRONG float64 under x64** and
   poison every field-math op they touch — this is the dangerous case. Any such array
   that ever multiplies fields (kgrid arrays, LSRK coefficient tables, …) must be pinned
   explicitly to `ftype`/`ctype` at construction, not patched after the fact.
3. **`jax.random.*` draws need an explicit `dtype=ftype`.** Without it, a draw under x64
   is float64 — not just the wrong dtype for fields, but a *different bitstream* than the
   float32 draw it replaces. Pinning the dtype is what keeps the fp32 RNG stream bitwise
   identical to before x64 was unconditional (`shared_physics`'s two `jax.random.normal`
   calls do this).
4. **Any `t`-derived quantity must downcast to `ftype` before it re-enters field math.**
   There is exactly one such path today: `run.py::_advance_forcing`'s
   `dt = (new_state.t - prev_t).astype(_precision.ftype)`, which feeds the OU decay
   factor multiplying complex64/128 `forcing_state`. A future time-dependent term func
   that reads `t` must do the same at the point it touches fields — it will not be caught
   by rule 2's pins because `t` itself is legitimately fp64.

**`t` is fp64 storage over fp32-rounded increments in the IF steppers** (recorded
2026-08-22, pre-existing). `lsrk_advance`'s scan path builds its stage coefficients at
field precision (`timestepping.py` ~114–139: `gammas_arr = jnp.array(scheme.gammas,
dtype=_precision.ftype)`), so under `TARANIS_PRECISION=32` `t + gammas_arr[i]*dt` adds an
fp32 product to the fp64 `t`; the unrolled path multiplies by the bare python `gamma`, and
`rk44`/the IMEX steppers advance by `dt` itself — every non-scan path is the fp64
sequential sum. Six steps at `dt = 0.01` therefore end at

| path | final `t` |
|---|---|
| lsrk54, `lsrk_scan=True` | 0.05999999830964953 |
| lsrk33, `lsrk_scan=True` | 0.05999999865889549 |
| lsrk54 / lsrk33, `lsrk_scan=False`; rk44; imexcb2/3e/3c/3f | 0.060000000000000005 |

so `lsrk_scan` True and False disagree in `t` at fp32, and `t` is a live comparator in any
bitwise reference recorded there — not a formality.

### The fp32 spectral noise shelf

Storing a field at complex64 quantizes every Fourier coefficient to relative precision
`eps32 ≈ 6e-8`, so a coefficient's *energy* (`~|f|²`) carries a relative roundoff floor of
`eps32² ≈ 4e-15` against the peak of the spectrum, `E_peak`: below `~eps²·E_peak` in
absolute terms, a perpendicular spectrum is showing fp32 storage/arithmetic roundoff, not
physics. Don't fit a dissipation range below that floor, and don't mistake a flattening
spectrum there for a bottleneck or condensate effect — check the same run at
`TARANIS_PRECISION=64` before drawing a physics conclusion from spectral content that close
to `eps²·E_peak`. This is a separate mechanism from the z-stencil error below (which is
about `k∥`-resolution, not overall field storage) and from the fp64 budget-closure gap
fixed by promoting the *reductions* rather than the fields themselves (Appendix B of
`plans/PRECISION_PLAN.md`).

### The fp32 z-stencil error floor

For a mode with parallel wavenumber `k∥` stored at fp32, the 4th-order first-derivative
stencil (`shared_physics.z_derivatives`) has relative error

```
(k∥·dz)⁴/30                     truncation
+ eps/(k∥·dz)                   roundoff (dominated by fp32 STORAGE quantization of f —
                                 upcasting only the stencil arithmetic recovers nothing)
```

Both terms are relative to that mode's own amplitude — the stencil acts per `(kx,ky)`
coefficient, so the error floor is a property of each perpendicular mode independently,
not of the field as a whole. The two terms trade off oppositely in `k∥·dz`, so the sum is
minimized at

```
k∥·dz ≈ (7.5·eps)^(1/5)
```

which is `≈0.05` at fp32 (`eps = 6e-8`), giving an irreducible relative-error floor of
`~1e-6`: refining `dz` *below* this makes `∂∥` **worse**, because the roundoff term grows
as `1/(k∥·dz)`. At fp64 the same balance puts the sweet spot at `k∥·dz ≈ 1e-3` with a
floor of `~1e-13` — invisible in any practical run.

Three consequences:

- **Critical balance puts the worst conditioning on the outer scale.** Since the error is
  per-perp-mode relative to that mode's own amplitude, and critical balance scales
  `k∥ ∝ k⊥^(2/3)`, the outer-scale modes (smallest `k⊥`, smallest `k∥`) sit furthest from
  the `k∥·dz ≈ 0.05` optimum for a fixed `dz` — they are the modes most likely to be
  under-resolved in `k∥·dz` even when the inertial range is fine.
- **The spurious content lands at grid-scale `k_z`,** with amplitude `~eps·|f|` — exactly
  the wavenumber `z_diss`'s hyperdissipation targets. Keep `z_diss` on at fp32; it is
  what removes this roundoff-sourced grid-scale content rather than letting it
  accumulate.
- **`z_spectral` sidesteps the cancellation entirely.** `params.z_spectral` makes `∂∥`
  the exact `±i·k_z` diagonal of `L`, applied as a unitary phase by `apply_exp` — there is
  no finite-difference truncation/roundoff trade-off at all, so it is the fp32-robust
  parallel formulation. It is currently `dims==3`, `size==1` only.

Practical rule: at fp32, size `nz` so that the outer-scale `k∥·dz ≳ 0.05` — below that,
adding z resolution actively degrades the parallel derivative rather than improving it.

## Dealiasing

The 2/3 rule, applied as an ellipse in *mode-index* space with semi-axes `nx/3`, `ny/3`
(`grids.dealias_mask`). Box lengths cancel out of the condition entirely, so the mask
depends only on resolution.

The elliptical region is a strict subset of the box rule `|n_x| < nx/3 and |n_y| < ny/3`,
and any subset of a dealiased set is dealiased, so this is valid — slightly conservative
at the corners, which is the price of a mask that is isotropic in index space.

`run.initialize` applies the mask to the initial condition as well as the evolution
applying it to the nonlinear term. Without that, energy in an IC beyond the cutoff would
never be removed (the linear terms and the integrating factor do not truncate) and would
alias for the whole run.

## z discretization

Fourth-order centered differences for `∂/∂z`, plus a five-point `∂⁴/∂z⁴` hyperdissipation
term, both in `shared_physics.z_derivatives`. Together they need a two-plane halo, which
is what fixes `width=2` throughout `comms.halo_exchange` and `rmhd.halo_start`.

**The semi-discrete dispersion relation.** A fourth-order centered first derivative does
not reproduce `ik` exactly; for a mode `e^{ikz}` it gives `i·k_eff` with

```
k_eff = (8 sin(k·dz) − sin(2·k·dz)) / (6·dz)
```

This is why an Alfvén wave in the discrete system travels at `k_eff`, not 1. It is the
difference between a test that measures *spatial* accuracy and one that measures
*temporal* accuracy: `tests/test_advection.py` compares against `cos(z + t)` and sees the
O(dz⁴) dispersion error, while `tests/test_time_order.py` and
`tests/test_scheme_equivalence.py` compare against `cos(z + k_eff·t)`, which the
semi-discrete system solves *exactly*, leaving only the time-integration error.

**Hyperdissipation stability.** The coefficient is `z_diss·(dz/2)⁴`, and the five-point
`∂⁴/∂z⁴` stencil has maximum eigenvalue `16/dz⁴`, so the damping rate at the grid scale is
`z_diss·(dz/2)⁴·16/dz⁴ = z_diss` — independent of resolution. That is why
`rmhd.set_timestep` can bound the timestep with a bare `max(..., z_diss)` term.

## Spectral z (`params.z_spectral`)

An optional single-process mode (`dims=3`, `size==1`, never `comm_backend="jax"`) in which the
z axis is Fourier rather than finite-difference. `grids.fft/ifft` become `rfftn`/`irfftn` over
`(z, x, y)`, so `state.fields` keeps its `(nfields, nz, nkx, nky)` shape and axis 1 simply means
`kz`. Everything downstream follows from three consequences:

- **The parallel operator is k-local**, so it leaves the RHS entirely and joins `L`:
  `∂ₜφ = i·kz·ψ`, `∂ₜψ = i·kz·φ` — the exact spectral form of `rmhd.FDLinearTerm`'s `∂ψ/∂z` in the
  (vorticity/−k⊥²) equation and `∂φ/∂z` in the ψ equation. The 2×2 putzer2 backend gives
  eigenvalues `±i·kz`, i.e. `z± = φ±ψ` propagating at unit speed, *exactly*: no wave CFL and no
  dispersion error, which is why `rmhd.set_timestep` drops both its `1/dz` and `z_diss` entries
  there and `halo_start`/`FDLinearTerm` are skipped (plain `if` on the static parameter).
  `eqpars['z_diss_k']` adds an optional `−z_diss_k·kz⁴` diagonal; it is a truncation choice, not
  a stability need, and is rejected outside this mode.
- **`i·kz` needs the same Nyquist fix as gdi's `ky_deriv`** (`rmhd._kz_deriv`). The reality
  constraint of the layout is now `F(−kx, −kz, ky) = conj(F(kx, kz, ky))` on the `ky = 0` and
  Nyquist rows — `propagators._check_hermitian_compatible` mirrors both axes — and at the
  self-conjugate `kz` Nyquist plane a bare `i·kz` violates it. The 2/3 `kz` cut removes that
  plane from every nonlinear and IC path anyway, so zeroing it is inconsequential.
- **Parseval, once, everywhere.** The z-FFT is unnormalized like the perpendicular one, so the
  z-average `(1/nz)Σ_z` becomes `(1/nz²)Σ_kz`: `shared_physics.perp_reduce` divides by `nz²`
  instead of `nz` in this mode and every energy-like quantity built on it — `diagnostics.energy`,
  `perpspec`, the forcing power — stays numerically identical to the real-z computation of the
  same physical state (`tests/test_z_spectral.py::test_forcing_power_parseval_matches_real_z`
  measures ~1e-16 relative). `parspec` becomes a plain sum over `kz` (its own `1/(nx·ny·nz)²`
  is unchanged: it always worked in the transformed variable).

The forcing's `A·cos(2πz/Lz) + B·sin(2πz/Lz)` envelope is scattered exactly rather than
evaluated: the unnormalized z-FFT of that envelope is `(A − iB)·nz/2` at `kz` index `+1` and
`(A + iB)·nz/2` at index `−1` (`nz−1`), and nothing anywhere else. Reality comes for free —
`A`, `B` are already kx-Hermitian on the `ky = 0`/Nyquist rows, and mirroring `kz` there swaps
the two planes while conjugating. The scatter uses `.at[].add` so the degenerate `nz = 2` case,
where the two planes coincide, still sums correctly.

Dealiasing gains a plain 2/3 cut in `kz` multiplied onto the perpendicular ellipse
(`grids.dealias_mask`): the brackets are pointwise in z, so products alias in `kz` too. The
product mask is a subset of the full 3D box rule, hence valid, and both the `* kgrid.dealias` in
the nonlinear terms and `run.initialize`'s IC masking pick it up with no physics-code change.

Snapshots are **not** cross-mode compatible (identical shapes, different meaning): `z_spectral`
is recorded in `params.json` and `params.save`'s differing-record check is the only guard.

## Time integration

Dissipation is not an RHS term. It is applied as an integrating factor: the perpendicular
hyperdissipation operator `L = −diss·k⊥^(2·hyper)` is diagonal in k-space, so
`exp(L·dt)` can be applied exactly rather than integrated.

For the low-storage schemes each stage advances the fields by `exp(L·dt·γᵢ)` and the
accumulator `delta` by the same factor. Since `Σγᵢ = 1` for both tables, a run with no
nonlinear term reproduces `exp(L·dt)` per step exactly. Dropping the factor from
`delta` is a silent bug on any dissipative problem and invisible on a non-dissipative one
— `tests/test_time_order.py` exists to catch exactly that, and its header records the
mutation test that proves it does.

**Scan and unrolled stage loops are not bitwise identical.** `params.lsrk_scan` selects a
`lax.scan` stage loop or a Python-unrolled one. They are algebraically the same, but XLA
fuses the two graph shapes differently, so at fp64 they diverge at round-off (measured
max|diff| ~2e-15 for lsrk33 and ~7e-15 for lsrk54 after 20 steps, under jax 0.6.2 on CPU;
the size is machine- and version-dependent). Earlier notes in this repo claimed bitwise
identity — that claim was wrong, and `tests/test_scheme_equivalence.py` asserts agreement
to a relative 1e-13 instead.

### The linear propagator (`taranis/propagators.py`)

The integrating factor above is one instance of a general hook. An equation set may
declare `linear_matrix_func(kgrid, params) → L` in its `EquationRecipe`; `setup_kgrids`
builds it once (`propagators.build`) and stores it on the `K_Grids` in the single slot
`lin`, an operator pytree — a NamedTuple of the backend's arrays carrying the methods
(`DiagonalOperator(L)`, `Putzer2Operator(L, m, s)`, `SeparableL(dperp, dz, kz)`, or
`IdentityOperator()` when a recipe declares none) — and the timesteppers only ever call
`apply_exp(arr, τ)` and — from the IMEX schemes —
`solve_shifted(arr, a) = (I − a·L)⁻¹ arr`. No stepper sees a matrix.

**Sign convention, fixed repo-wide:** `L` is defined by `∂ₜf = L f + N(f)`, so the
propagator is `exp(L·τ)` and a damped mode has `Re λ(L) < 0`. (RMHD's `L` is exactly the
old `hdiss`; the `∂ₜf + M f = 0` convention of the earlier gdi branch is *not* used here.)

The backend is selected by what `linear_matrix_func` returns: a dense array dispatches on
its shape, which carries an optional z/kz axis (broadcast, size 1, whenever the operator
is perpendicular-only); a `SeparableL` NamedTuple selects the separable backend.

- **diagonal**, `(nfields, nz-or-1, nkx, nky)`: `exp(L·τ)·arr`, elementwise.
- **putzer2**, `(2, 2, nz-or-1, nkx, nky)`: the closed form
  `exp(Lτ) = e^{mτ}[cosh(sτ)·I + (sinh(sτ)/s)·(L − m·I)]` with `m = tr L/2` and
  `s = sqrt(m² − det L)` (Putzer/Sylvester). No eigendecomposition and no eigenvector
  storage, and it is smooth through the defective points where an eigenbasis does not
  exist. `s` is formed once at setup (`kgrid.lin.s`) and a stage evaluates the two
  **eigenvalue exponentials** `e± = exp((m ± s)τ)`, from which
  `C = e^{mτ}cosh(sτ) = (e⁺ + e⁻)/2` and `S = e^{mτ}sinh(sτ)/s = (e⁺ − e⁻)/(2s)`, and
  `exp(Lτ) = C·I + S·(L − m·I)`. That is 2 complex exps and 1 complex divide per mode.
  The prefactor is folded into the exponents deliberately (since 2026-11-13): the earlier
  `w = e^{sτ}` form multiplied `e^{mτ}` onto `cosh`/`sinh` afterwards, and for a stiffly
  damped mode `cosh(sτ)` overflows (|Re sτ| ~ 710 fp64 / 88 fp32) while `e^{mτ}`
  underflows — `inf·0 = NaN` for a bounded exponential. `e±` are the propagator's own
  eigenvalues, so they overflow only if the solution does. **This was reachable under
  adaptive dt**: the exponent is evaluated at every mode, the never-populated corners
  beyond the dealias cut included (`0·NaN = NaN`), where `k^{2h}` is ~91× (hyper 3) or
  ~414× (hyper 4) its value at the cut. Measured before the fix, z_spectral RMHD 64²×8,
  `hyper=4`, `diss=(ν, 0)`, lsrk54: non-finite after one step at `ν·k_d⁸·dt ≈ 2.7` (fp32)
  and `≈ 27` (fp64). Cost of the change, interleaved A/B on the M1 at fp64
  (`bench/putzer2_step.py`): GDI-IF unchanged, z_spectral ν ≠ η 10–15% faster.
- **separable**, `SeparableL(dperp, dz, kz)`: the exact factorised form for
  `L = d·I + i·kz·σx` with real `d = dperp(k⊥) + dz(kz)` — z_spectral RMHD at ν = η.
  Derivation and properties below.

Two traps in the 2×2 form, both guarded in `tests/test_linear_propagator.py`:

- **everything is complex.** `s²` is negative for any oscillatory mode and complex in
  general; a real `sqrt` NaNs silently. `cosh(sτ)` and `sinh(sτ)/s` are *even* in `s`, so
  they are single-valued functions of `s²` and the branch of the sqrt does not matter.
- **`sinh(sτ)/s` is 0/0 at a defective mode**, so below a cutoff on `|z²|`, `z = sτ`
  (`1e-4` at fp64, `1e-2` at fp32) `S` switches to the series
  `S = C·τ·tanh(z)/z = C·τ·(1 − z²/3 + 2z⁴/15 − 17z⁶/315)` — `C` is a sum and accurate
  there, so no third exponential is needed. The branch is taken with a `jnp.where` on a
  *safe* denominator so the unused branch cannot NaN.

  The cutoffs are ~100× wider in `|z²|` than the removable singularity alone would need,
  because the branch is an *accuracy* branch: `e⁺ + e⁻` is a sum of two nearly equal
  numbers, but `e⁺ − e⁻` is a **difference** of two nearly equal numbers, and that
  cancellation grows as z shrinks (relative error ~ eps/|z|² in the quotient; the same
  cancellation the earlier `w − 1/w` had, measured there as 5.6e-15 (fp64) and 3.2e-7
  (fp32) at these boundaries). On the series side the truncation `62|z|⁸/2835` is
  2.2e-10 at `|z| = 1e-1` (fp32's boundary) and 2.2e-18 at `1e-2`, so the branch is
  covered at both precisions. Both coefficients stay *even* in `s` (`s → −s` swaps `e⁺`
  and `e⁻`), so the branch-independence above is undisturbed.

`setup_kgrids` also asserts that `L(−kx, ky) = conj(L(kx, ky))` on the `ky = 0` and
Nyquist rows: without it the exponential would break the reality constraint of the rfft2
layout (see "Fourier conventions").

`scaled(c)` returns the propagator of `c·L`. `lsrk_advance` uses it to pre-scale by `dt`
so a stage exponent is formed as `exp((L·dt)·γ)` — the floating-point op order of the
pre-propagator code, which keeps the LSRK schemes bitwise identical to it on RMHD.

`exp_op(τ)` returns `exp(L·τ)` itself as an `ExpOp` pytree (`DiagonalExp`, `Putzer2Exp`,
`SeparableExp`, `IdentityExp`) with `.apply(arr)`; `apply_exp(arr, τ)` is literally
`exp_op(τ).apply(arr)`.
This is what the hoisted path stores: the exponent depends only on `τ = γ·dt`, so over a
block of steps sharing one dt (fixed dt, or a `cfl_every` block) `run.py` forms each stage's
ExpOp once (`timestepping.stage_exp_ops`) and the stepper only applies it — the same
numbers in the same op order, bitwise equal to the per-stage evaluation at fp64
(`tests/test_hoist_propagator.py`). The cost it removes is the per-stage coefficient
evaluation — the 2 complex exps + 1 divide of the putzer2 form above, or the
separable backend's real `exp`/`cos`/`sin`. It is worth the most on putzer2, which is the
only backend where the ExpOps are full-grid arrays and so the only one where hoisting has a
memory price; docs/performance.md has the per-device numbers.

### The Elsasser-separable backend (z_spectral RMHD, ν = η)

RMHD under `z_spectral` with `eqpars["diss"] = (ν, η)`, hyper order h and optional
`z_diss_k` = zd has, in the `∂ₜf = L f + N` convention on fields (φ, ψ),

```
L = [[ −ν·k⊥^{2h} − zd·kz⁴ ,        i·kz         ],
     [        i·kz          , −η·k⊥^{2h} − zd·kz⁴ ]]
```

with `kz` the Nyquist-zeroed `rmhd._kz_deriv`. At ν = η both diagonal entries are the
same real `d(k⊥,kz) = dperp(k⊥) + dz(kz)`, so `L = d·I + i·kz·σx` — and the two terms
commute (`d` is a scalar multiple of the identity at each mode), so the exponential
factorises EXACTLY, with no splitting error:

```
exp(L·τ) = e^{dperp·τ} · e^{dz·τ} · [cos(kz·τ)·I + i·sin(kz·τ)·σx]
```

using `σx² = I`. "Elsasser" records where the exactness comes from: σx is diagonalised
by `z± = φ ± ψ` with eigenvalues `d ± i·kz` — two independent damped waves — and the
eigenvectors are the constant vectors (1, ±1), independent of k. The implementation
stays in (φ, ψ) (the Elsasser transform would cost two full-grid adds per apply) and
stores three small REAL arrays — `SeparableL`'s `dperp` (nkx,nky), `dz` and `kz` (nz,1,1)
— against `Putzer2Operator`'s 6 u of resident complex full-grid `L`/`m`/`s`. Per stage the
propagator forms `P = exp(dperp·τ)`, `c = e^{dz·τ}cos(kz·τ)`, `s = e^{dz·τ}sin(kz·τ)`
and applies `out₀ = P·(c·a₀ + i·(s·a₁))`, `out₁ = P·(i·(s·a₀) + c·a₁)`, the `i·` written
as the real swap `x + iy → −y + ix`. The backend is hoistable: its whole stage stack is
`nstage·(nkx·nky + 2·nz)` reals, and what hoisting amortises here is the per-stage
exp/cos/sin evaluation, not full-grid streaming.

**ν ≠ η limit.** Write `d± = (d_φ ± d_ψ)/2`; then `L = d₊·I + d₋·σz + i·kz·σx`, and σz
does not commute with σx: the eigenvalues become `d₊ ± sqrt(d₋² − kz²)`, the eigenvectors
depend on kz and the dissipation contrast, and the constant rotation no longer
diagonalises L. That closed form is exactly what putzer2 evaluates, so ν ≠ η stays on
putzer2 (selection at setup, comparing `diss[0] == diss[1]` as python floats). GDI's L is
genuinely full-grid and is untouched.

**`solve_shifted`, and the no-pole property.** From
`(A·I + iB·σx)(A·I − iB·σx) = (A² + B²)·I`:

```
(I − a·L)⁻¹ = [(1 − a·d)·I + i·a·kz·σx] / ((1 − a·d)² + (a·kz)²)
```

`d ≤ 0` everywhere (ν, zd ≥ 0), so for a > 0 the denominator is ≥ (1 − a·d)² ≥ 1: the
separable `solve_shifted` divides by nothing smaller than 1. This retires, for this path,
one of the two documented NaN traps above — putzer2's `det(I − a·L) = 0` at λ = 1/a
(reachable at large fixed dt). Overflow cannot happen either: `e^{dperp·τ}` and
`e^{dz·τ}` are exponentials of negative reals — they underflow to 0, never overflow
(putzer2 had a cosh/sinh overflow trap here until its exponents were combined, above).

**Reality, and why the setup check is analytic.** The rfft layout's self-conjugate rows
require `L(−kx, −kz, ky) = conj(L(kx, kz, ky))` there, which for the separable form is
three elementary conditions: `dperp` mirror-symmetric in kx (true by construction — it
depends on kx²), `dz` mirror-symmetric in kz (kz⁴), and `kz` mirror-ANTIsymmetric
(`kz[−j] = −kz[j]`, so `i·kz → conj(i·kz)`). The antisymmetry check subsumes the
Nyquist rule: the mirror maps index 0 and nz/2 to themselves, forcing kz = 0 at both —
`kz[0] = 0` holds anyway and `kz[nz/2]` is zeroed by `_kz_deriv`. These are checked on
plane-sized arrays at setup; the dense operator (4 u, and transiently doubled by
`_check_hermitian_compatible`'s host-numpy mirror) is never materialised — the
operators' `dense()` method exists for tests and validation only (`particles/fields.py`
dispatches on `rmhd.linear_matrix`'s return, not on `dense()`).
The same `_kz_deriv` array feeds the off-diagonal and the kz⁴ term, exactly as the dense
path does, which is why `SeparableL.dense()` reproduces the dense L bitwise.

## Stochastic forcing

An Ornstein-Uhlenbeck process on a shell of perpendicular wavenumbers, rescaled every step
to inject power at a fixed target rate.

### Reality symmetrization: divide by √2, not 2

On the `ky = 0` and Nyquist rows the noise has to be made Hermitian in `kx`. The obvious
`(c[k] + conj(c[−k]))/2` is wrong, and wrong in two different ways at once:

- for a paired `(kx, −kx)` it averages two independent complex Gaussians, halving the
  variance;
- for the self-conjugate points `kx = 0` and `kx = Nyquist` it produces `Re(c)`, whose
  target variance is that of a *real* Gaussian, not a complex one.

Dividing the symmetrized combination by `√2` restores both at once. Using `/2` instead
anisotropically underforces exactly the modes that vary only in x.

### Cap the scale factor, never floor the denominator

The scale factor is always **clipped** to `±forcing_scale_max`, and the denominator is
never floored. Flooring `P` away from zero misbehaves whenever `P` is small but nonzero
(enormous or sign-flipped scales) and has no principled floor value, because `P` carries
units. Capping the scale bounds the worst case regardless of what `P` does. That part is
unchanged; what changed is what gets clipped.

### Normalize against the forcing's own self-energy (behaviour change 2026-08-08)

Over one step of length `dt` the force `s·f_raw` acting on the fields `z` injects

```
ΔE = s·P·dt + ½·s²·F₂·dt²,     P = ⟨∇z·∇f_raw⟩,   F₂ = ⟨|∇f_raw|²⟩
```

— the linear cross term **plus** the forcing's self term. `safe_scale(target, P)` keeps
only the first: `s = target/P`. That is right in saturation, and wrong from rest. From a
quiescent start `P = 0` exactly, so `safe_scale` pinned at `±forcing_scale_max` and the
first forced step injected `½·s_max²·F₂·dt²` — a quantity in which **`forcing_power` does
not appear at all**, because ε enters the forcing *only* through the scale and the clamp
erased it. Measured at 64², `fshell=(1,3)`, `cfl_safety=0.5` (where the quiescent velocity
floor gives `dt = 0.491`): `E` after step 2 was `1.074e+01` for `ε_tot = 1e-4, 1e-3, 1e-2`
and `1e-1` alike, identical to four digits, and at small dissipation that kick *was* the
final state. Because `ΔE ∝ ½s_max²F₂dt²` with `⟨|f|²⟩ ∝ (1 − e^{−2dt/τ})`, it scales as
`dt³`: measured 1.07e1 / 8.19e-2 / 7.08e-4 at `dt` = 0.49 / 0.098 / 0.020, versus the
predicted ratio `5³ = 125`.

`selfnorm_scale(target, P, F₂, dt)` (`shared_physics.py`) instead solves

```
½·F₂·dt·s² + P·s − target = 0        →        s = [−P + √(P² + 2·F₂·dt·target)] / (F₂·dt)
```

so `ΔE = target·dt` **exactly**, and then clips to `±forcing_scale_max` as before. Limits:
`F₂·dt → 0` or `|P|` large gives `s → target/P` (saturation is unchanged); `P → 0` gives
`s → √(2·target/(F₂·dt))` (the first kick is `target·dt` on the nose). The expression is
continuous in `P` through zero, which matters because the same pinning recurred *mid-run*
whenever `P` fluctuated through zero with a weak field — a fix special-cased on `t = 0`
would not have covered that. The clip survives as a last-resort safety: it engages if
`F₂·dt` underflows while `P ≈ 0`, or under a strongly adverse `P < 0` (there
`s ~ 2|P|/(F₂·dt)`; a clipped step under-injects, transiently even net-negative through
the linear term). Measured margins in developed runs sit more than an order of magnitude
from the clip.

Three details worth knowing:

- **The positive root, not the sign-following one.** `safe_scale` follows `sign(P)`, so an
  adverse phase flips the force to keep injecting — a rectification of the O-U process. The
  positive root keeps `s > 0` and lets the exact solve absorb an adverse linear term
  instead (still hitting the target unless the clip engages).
- **Numerically, the root is evaluated in two branches.** `(−P + √D)/(F₂dt)` cancels
  catastrophically for `P > 0` in the saturated regime `2F₂·dt·target ≪ P²`, and the
  conjugate form `2·target/(P + √D)` cancels for `P < 0`. Each is used where it does not
  cancel; both are algebraically identical.
- **`dt` is lagged by one step.** `run._advance_forcing` passes the dt of the step just
  completed, since that is what is known when the scale is stored. This is one more
  `O(dt/τ)`-class approximation of exactly the kind `forcing_norm_per_step` already makes,
  and it is *exact* under `cfl_every > 1`, where dt is frozen for the block.

**Dated behaviour change.** Runs from before **2026-08-08** spin up differently from a
quiescent (or weak-field) start: they take an ε-independent `O(½s_max²F₂dt²)` kick on step
2 that later runs do not, and at low dissipation that kick can dominate the saturated
state. Saturated statistics are affected only at `O(s²F₂dt/target)`. Same class of
non-comparability as the 2026-07-31 elsasser factor-2 note below. `tests/test_forcing_spinup.py`
pins the new behaviour; `tests/data/forcing_spinup_reference.npz` records the old one.

**The per-stage path (`forcing_norm_per_step=False`) keeps `safe_scale`.** `ForcingTerm`
has no `dt` — the RHS interface is deliberately dt-agnostic, `dt` is a stepper-local chosen
once per step (and frozen per block under `cfl_every`), and recomputing it inside the term
would repeat the CFL allreduce every stage *and* give the wrong value inside a block.
Rather than thread `dt` through `construct_rhs` and every stepper, that path gets a
mitigation instead: its clip is tightened to `min(s_max, √(2·target/(F₂·dt_q)))`, with
`dt_q` the **params-static** bound on the quiescent step length (`params.dt` when
`adaptive_timestep=False`, where it is exact; otherwise
`cfl_safety·min(dx,dy)/QUIESCENT_EPS`, which `rmhd.set_timestep` applies as an explicit
`jnp.minimum` ceiling — so `dt ≤ dt_q` holds by construction for every state, not by
keeping two expressions in sync).
That bounds the from-rest kick at `~target·dt_q` instead of `~½s_max²F₂dt_q²`, and only
binds when `P` is small, i.e. near-quiescent states. **From-rest starts that need exact
normalization should use the production default.**

### Power conventions

`forcing_power` (momentum mode) and both entries of `forcing_power_elsasser` are in the
same units: a contribution to the **total** energy injection rate. Total `dE/dt` is
`forcing_power` in momentum mode and `ε₊ + ε₋` in elsasser mode, so `(p/2, p/2)` matches
`forcing_power = p`.

This needs a factor of 2 inside `rmhd._forcing_scale_from`, because with
`E_± = ⟨|∇z_±|²⟩/2` the total energy is `E = (E₊ + E₋)/2`, not `E₊ + E₋`. The cross-helicity
injection rate is then `dH_c/dt = ε₊ − ε₋`. Before 2026-07-31 the elsasser entries meant
half this; runs from before that date inject twice as much at the same numbers.

### Per-step normalization

`forcing_norm_per_step` (production default) computes the normalization scale once per
step from the step's starting fields and reuses it across the RK sub-stages, instead of
renormalizing at every stage. The realized injection then tracks the target to O(dt/τ)
rather than exactly. Worth about +8% at fp64 (see `docs/performance.md`).

### 2D momentum forcing is pure hydrodynamics

In 2D with `forcing_mode="momentum"` from a quiescent start, `ψ` stays *exactly* zero —
bitwise, not approximately. Its only source in 2D is the `{φ, ψ}` bracket, which vanishes
identically when `ψ = 0`, and momentum forcing drives only `φ`. Use `forcing_mode="elsasser"`
for actual 2D MHD from rest.

## Test particles: fields seen by a particle

The reference derivation for `taranis/particles/fields.py` (design and phasing:
`plans/TESTPART_PLAN.md`). Charged test particles are pushed in Cartesian `(E, B)`, so the
only question this section answers is what `(E, B)` the RMHD fields `(φ, ψ)` *mean*.

**Conventions.** `shared_physics.bracket(a, b) = a_x b_y − a_y b_x = ẑ·(∇a × ∇b)`, and
`rmhd.NonlinearTerm` builds `∂ₜω + {φ, ω} = {ψ, j_z}` with `ω = ∇²φ`, `j_z = ∇²ψ` — i.e.
`∂ₜω + u·∇ω = b·∇j_z` — exactly when

```
u   = ẑ×∇φ = (−∂_yφ, ∂_xφ)
b_⊥ = ẑ×∇ψ = (−∂_yψ, ∂_xψ)
```

Those two identifications fix the potentials: `b_⊥ = ∇×(A_z ẑ) = −ẑ×∇A_z`, so `A_z = −ψ`,
and `u = E×B/B² = ẑ×∇(Φ/B₀)` gives `Φ = B₀φ`. Hence

```
B   = B₀ẑ + ẑ×∇ψ
E_⊥ = −∇_⊥Φ           = −B₀∇φ
E_z = −∂_zΦ − ∂_t A_z = −B₀∂_zφ + ∂ψ/∂t
```

**The parallel-gradient terms cancel.** The code's induction equation is
`∂ψ/∂t = −{φ, ψ} + ∂_zφ + η∇²ψ (+ f_ψ)` — the `∂_zφ` piece being `rmhd.FDLinearTerm`'s
z-stencil (or, under `z_spectral`, the `±i·kz` off-diagonal of `rmhd.linear_matrix`),
absent in 2D. Its coefficient is exactly **1**, not `B₀`: taranis writes RMHD in units
where the guide field (equivalently `v_A`) is 1, so the cancellation below needs `B₀ = 1`
in 3D — see "`B₀` in 3D" at the end of this section. Substituting it into `E_z` above kills
the `∂_zφ` terms exactly:

```
E_z = −{φ, ψ} + η∇²ψ (+ f_ψ)        the physics terms, identical in 2D and 3D
```

so the field assembly is dimension-independent in the physics: the same three pieces serve
2D and 3D. Note the sign: `E_z = +∂ψ/∂t`, because `A_z = −ψ` in this convention.

**The finite-difference-z filter is part of the resistive piece.** In 3D without
`z_spectral`, `rmhd.FDLinearTerm` adds `−z_diss·(dz/2)⁴ ∂_z⁴ψ` to `∂ψ/∂t` alongside the
Alfvén stencil (`physics/rmhd.py`, `FDLinearTerm`). Like the perpendicular hyperdissipation
it is an artificial EMF rather than a physics term, and it is the *only* other linear
non-ideal term acting on ψ — so `ez_resistive` is defined as the full linear non-ideal EMF:
the ψ diagonal of `rmhd.linear_matrix` **plus** that filter whenever `dims == 3` and
`z_spectral` is off (`fields._psi_non_ideal`; the Alfvén half of `FDLinearTerm` is the
`∂_zφ` term that has already cancelled out of `E_z`, and stays out). Under `z_spectral` the
analogous `−z_diss_k·kz⁴` is already in that diagonal. With the filter folded in, full-mask
`E_z` equals the `∂ψ/∂t` the discrete ψ obeys, to round-off, in 2D and in **both** 3D modes;
it stays one piece and one work column, so `FIELD_PIECES`, `WORK_PIECES` and the checkpoint
layout are unchanged.

**Cross-check against `E = −u×B`** (ideal Ohm, which is what the ideal piece must be):
`(u×B)_z = u_x b_y − u_y b_x = (−∂_yφ)(∂_xψ) − (∂_xφ)(−∂_yψ) = {φ, ψ}`, so
`−(u×B)_z = −{φ, ψ}`, the ideal `E_z` above; and `−u×B₀ẑ = −(ẑ×∇φ)×B₀ẑ = −B₀∇φ = E_⊥`.
Running that backwards, the E×B drift of `(E_⊥, B₀ẑ)` is `ẑ×∇φ = u` — which is what kernel
gate 2 pins numerically.

**What pins which sign.** Kernel gate 2 (`tests/test_particles_kernel.py`) pins the
*pusher's* Lorentz-force convention only: it drives uniform analytic `(E, B)` arrays, so it
never touches the field assembly. The assembly's *relative* signs are pinned by the ideal-Ohm
identity `E·B = 0`, which holds exactly for the raw (undealiased) ideal `E_z`
(`E_x b_x + E_y b_y + B₀·(−{φ,ψ}) = 0` identically, since both terms are `B₀{φ,ψ}` with
opposite signs) — asserted in `tests/test_particles_coupled.py`. That fixes `b_⊥` relative to
`E_⊥` and `E_z`, and `B_z = +B₀`; it does not fix the overall sign of `b_⊥` and `E_z` against
`ψ`, which is what gate 4's canonical `p_z = m v_z − qψ` invariant does.

**The canonical p_z invariant.** In 2D, z is ignorable (∂_z = 0 on every field), so the
canonical momentum conjugate to z, `p_z = m v_z + qA_z = m v_z − qψ` (using `A_z = −ψ`), is
conserved exactly along a particle orbit — for ANY time-dependent `(φ, ψ)`, not just frozen
fields. Newton's law gives `m dv_z/dt = q(E_z + (v×B)_z)`. `B_⊥ = ∇×(A_zẑ)` with `A_z`
z-independent makes `(v×B)_z = v_xB_y − v_yB_x = −(v_x∂_xA_z + v_y∂_yA_z) = −v·∇A_z`
identically (algebra alone, no dynamics), and `E_z = −∂_zΦ − ∂_tA_z = −∂_tA_z` since
`Φ = B₀φ` has no z-dependence either. So `m dv_z/dt = q(−∂_tA_z − v·∇A_z) = −q·dA_z/dt`
(the total derivative of `A_z(x(t),y(t),t)` along the orbit), i.e. `d/dt(m v_z + qA_z) = 0`
— the standard ignorable-coordinate argument, specialized to 2D. This holds **only with the
FULL mask** (`ez_resistive=ez_forcing=True`): `E_z` must equal the true `∂ψ/∂t` the discrete
ψ obeys, piece for piece, or the cancellation is incomplete and `p_z` drifts by exactly the
omitted acceleration (the default ideal-only ensembles are the worked example — see the
plan's gate 4). Because the invariant needs both `b_⊥` (through `(v×B)_z`) and `E_z` at
their correct relative AND absolute sign against `ψ`, it is the gate that pins the absolute
sign left open above (gate 4). With fields frozen over a step (the run.py push convention),
the per-step defect in the identity is the time-discretization error of an otherwise-exact
cancellation, `O(dt²)` per step, so the accumulated drift over a fixed physical time is
`O(dt)` — consistent with the KDK push being exact in space but first-order in how it
samples the field's time dependence.

The argument needs z ignorable, so **in 3D `p_z` is conserved only for z-independent
fields** — an embedded-2D configuration (a 3D run whose φ, ψ have no z structure, in either
z mode) is what carries gate 4 over, and it is also the sharpest test that the z axis of the
gather and of the field assembly is wired correctly: the trilinear gather must reproduce the
2D bilinear one exactly, and the `∂_zφ` cancellation must be exact rather than merely small.
In a genuinely 3D field the drift of `p_z` is physics — that parallel energization is
unbounded is precisely why Phase B exists (`plans/TESTPART_PLAN.md` §5).

`E_∥ = E·b̂` differs from `E_z` by `b_⊥·E_⊥/B₀` corrections; the pusher is Cartesian and
never needs it.

### The pieces, and why `∂ψ/∂t` is dealiased

`particle_fields` returns the piece decomposition rather than a summed `E`, because which
pieces a particle sees is a per-ensemble choice (`FIELD_PIECES` /
`FIELD_MASK_DEFAULTS`, assembled by `assemble_stacked` — `assemble` is its summed `(E, B)`
form; all mask logic is static python):

- **ideal**, `−{φ, ψ}`. On by default.
- **resistive**, `L_ψ ψk` with `L_ψ` the ψ diagonal of `rmhd.linear_matrix`, plus the
  finite-difference-z `∂_z⁴` filter in FD-z 3D (above) — `η j_z` at
  `hyper=1`, its hyper-resistive analogue otherwise. Only the diagonal: under `z_spectral`
  the full 2×2 `L` also carries the `±i·kz` off-diagonal, which is the `B₀∂_zφ` term that
  has already cancelled out of `E_z`, so `propagators.apply_L` is the wrong tool here.
  Off by default.
- **forcing**, `f_ψ` (the scaled elsasser envelope's ψ half; identically zero in momentum
  mode). Off by default.

`PFields.ex/ey` hold `E_⊥/B₀ = −∇φ`, not `E_⊥` itself: `B₀` enters in exactly one place,
`assemble_stacked(pf, mask, B0)`, which scales `(ex, ey)` by `B₀` and sets `B_z = B₀`. Physically
`E_⊥ = −B₀∇φ` as derived above; storing it per unit `B₀` keeps a single knob.

Defaults are the ideal-Ohm particle: for a collisionless test particle the resistive piece
is a fluid-closure/numerical-regularization artifact and the forcing piece is a stirring
EMF, so neither is physical `E` — but the three pieces together are exactly `∂ψ/∂t` (2D; see
the FD-z caveat above), and an `ez_resistive = ez_forcing = True` ensemble run alongside a
default one measures the resistive-acceleration difference on identical fields.

**The ideal piece must be dealiased.** The discrete ψ obeys

```
∂ₜ ψk = dealias·NL_ψ,k + F_ψ,k + L_ψ ψk
```

(`rmhd.NonlinearTerm` applies `kgrid.dealias`; the propagator applies `L` unmasked). The
raw pointwise bracket `−{φ, ψ}` has spectral content out to `2k_c` that *never enters ψ*,
so the field a particle should see is

```
E_z,ideal = ifft(dealias · fft(−{φ, ψ}))
```

not the raw product. This is not cosmetic: with the raw bracket the gate-7 comparison
against a centered difference of ψ across a step stalls at the (dt-independent) size of
the discarded beyond-cutoff content instead of converging at O(dt²), and gate 4's exact
2D invariant `p_z = m v_z − qψ` stops being exact. The cost is one extra `fft`+`ifft` pair,
which is the whole reason to state it here rather than rediscover it.

**Transform budget of `particle_fields`**: 4 `ifft`s for the `(∂_x, ∂_y)(φ, ψ)` gradients,
`+2` for the dealiased ideal `E_z`, `+1` when the resistive piece is requested, `+1` when
the forcing piece is — so ≤8 per step, against the ~30–36 of a 3-stage RHS.

### Work bookkeeping

Heating attribution is *measured*, not inferred: `ParticleState.w` carries, per particle,
the cumulative work per unit mass done by each electric piece
(`fields.WORK_PIECES = (eperp, ez_ideal, ez_resistive, ez_forcing)`, the electric members
of `FIELD_PIECES`). The split is exact, not a quadrature.

Write one Boris half-kick with `h = q·dt_k/(2m)`: `v⁻ = v + hE`, a rotation `v⁻ → v⁺` with
`|v⁺| = |v⁻|`, then `v_new = v⁺ + hE`. Then

```
½|v_new|² = ½|v⁺|² + h E·v⁺ + ½h²|E|²
½|v|²     = ½|v⁻|² − h E·v⁻ + ½h²|E|²
```

and `|v⁺| = |v⁻|`, so subtracting gives `½|v_new|² − ½|v|² = h E·(v⁺ + v⁻) = h E·(v_new + v)`
— the rotation drops out identically, no small-`dt_k` expansion. `boris.push_tracked`
accumulates exactly that expression restricted to each E component (`E_⊥` → the `x,y`
terms, each `E_z` piece → the `z` term with its own piece's sampled value), so summing the
columns of `w` reproduces `½|v|² − ½|v₀|²` per particle to round-off, over any number of
steps and substeps and whatever the mask. A piece an ensemble does not see contributes
exactly zero, forever. Gate 8 (`tests/test_particles_coupled.py`) asserts both, measured
at `~5e-14` of `KE₀` in *both* precisions — `w` and the push are fp64 whatever the field
precision is, so the closure does not degrade at fp32.

Two consequences worth stating: the work is credited with the *same* field sample the kick
used (any other choice would leave an `O(dt)` residual in the closure), and the piece
columns are attributions of the *pusher's* energy change, so `w_ez_*` is not by itself the
parallel heating — the rotation exchanges ⊥ and ∥ energy within a step.

**The magnetic moment uses the local field.** `state.moments`'s `mu` is
`μ = v_⊥B²/(2|B|)` with
`v_⊥B² = |v|² − (v·B)²/|B|²` and `B = B₀ẑ + b_⊥` *sampled at the particle*, not
`v_x²+v_y²/(2B₀)`. In RMHD `|B|` differs from `B₀` only at `O(ε²)` while the field
*direction* tilts at `O(ε)`, so the ẑ-referred perpendicular energy and the true one differ
at first order in `b_⊥/B₀` — the same effect that makes the naive grad-B drift wrong by an
O(1) factor (below). `push_tracked` returns the B sample of its last half-kick precisely so
`μ` costs no extra gather.

### E∥ projection and the amplitude parameter

**The gathered field has a numerical E∥.** On the grid the ideal pieces are exactly
orthogonal: `E_⊥·b_⊥ + B₀·(−{φ,ψ}) = 0` identically (above). At the particle they are not.
`interp.gather` interpolates each component of `E` and `B` independently, and the bilinear
interpolant of a product is not the product of the interpolants, so the sampled
`Ē·B̄ = O(dx²)·|E||B|` — a parallel electric field of purely numerical origin, which
accelerates particles along `B` and contaminates exactly the quantity a heating run
measures. Measured on a developed 64² state: `rms(Ē·B̄)/rms(|Ē||B̄|) ≈ 1.4·10⁻³`
(gate 9, `tests/test_particles_coupled.py`).

The per-ensemble flag `epar_project` removes it the way Xia, Perez, Chandran & Quataert
2013 (ApJ 776, 90, their eq. 21) do, at every half-kick, per particle:

```
Ẽ = Ē − (Ē·B̄) B̄/|B̄|²        so  Ẽ·B̄ = 0 exactly (4·10⁻¹⁷ of |E||B| measured)
```

`boris.project_perp` is the kernel; `push_tracked(..., project=True)` applies it to the
sample *before* the kick and before the work columns are formed, so `w` credits the
projected field and the closure identity above stays exact. The projection is defined only
for the exact ideal-Ohm mask (`bperp = eperp = ez_ideal = True`, the non-ideal `E_z` pieces
off) and any other mask is a config error: with `ez_resistive`/`ez_forcing` on, `E∥` is
partly *real* (that is the whole point of the paired resistive-split ensembles) and
projecting would delete it; without the three ideal pieces there is nothing to project. It
also mixes the `E_z` pieces, which is the second reason it needs a single one.

**`B₀` is the RMHD amplitude parameter, `B₀ = 1/ε`.** Nondimensionalize the real particle
equation in RMHD units — `u = ε v_A ũ`, `ψ = ε B₀ L ψ̃`, `t = L t̃/(ε v_A)`, velocities in
flow units `ε v_A`:

```
dv̂/dt̃ = Ω̂ [ −∇φ̃ + ε(∂_t ψ̃) ẑ + v̂ × (ẑ + ε b̃_⊥) ],      Ω̂ = ΩL/(ε v_A)
```

while the code pushes `qm·B₀·[ −∇φ + (1/B₀)(∂_tψ) ẑ + v × (ẑ + b_⊥/B₀) ]`. The two are
identical with `qm·B₀ = Ω̂` and `B₀ = 1/ε`. So `B₀` is not a free unit: `B₀ = 1` means
`δB/B₀ ~ 1`, an amplitude no RMHD ordering supports (Xia et al. cap their runs at
`δB_rms/B₀ ≤ 0.47` and flag even that). Production runs `B₀ ~ 10` with `q/m` scaled by
`1/B₀` so that `qm·B₀` — hence `ρ = v_⊥/(qm·B₀)`, `Ω·dt` and `ξ` — is unchanged; the one
thing that does change is `β_i = v_⊥²/v_A² = v_th²/B₀²` (0.01 for `v_th = 1`, `B₀ = 10`).
`B₀` is therefore a PER-ENSEMBLE key (the top-level one is only its default), so one run
can hold several amplitudes; with every field piece off it changes no orbit at all
(gate 9's control).

**`B₀` in 3D is pinned to 1.** In 2D nothing ties `B₀` to the solver: `∂_z ≡ 0`, so
`E_z = ∂ψ/∂t` and `−(u×B)_z = −{φ,ψ}` hold for any `B₀`, and `E·B = 0` because both terms
carry one factor of `B₀`. In 3D the induction equation ties it. The particle's
`E_z = −B₀∂_zφ + ∂ψ/∂t` reduces to the ideal `−{φ,ψ}` only if the solver supplies `+B₀∂_zφ`
— and taranis supplies `+1·∂_zφ` (`rmhd.linear_matrix`'s off-diagonal is `1j·kz`;
`FDLinearTerm` returns a bare `df_dz`), i.e. the solver is written in units where the guide
field, equivalently `v_A`, is 1. Any other `B₀` therefore leaves
`(1 − B₀)∂_zφ` in `E_z`: `E·B ≠ 0`, the field is no longer ideal-Ohm, and no ensemble
setting can repair it. `Parameters` rejects `B₀ ≠ 1` whenever `dims == 3`.

Nothing is lost. The amplitude parameter in 3D is `ε = δb_⊥/B₀ = rms|∇ψ|` (with `B₀ = 1`),
set by the run itself through the forcing amplitude and the box: RMHD at `v_A = 1` on a box
`L_z` is, under `z → B₀z`, the same system as `v_A = B₀` on a box `B₀L_z`, so lowering `ε`
means weakening the forcing and/or elongating the box — the RMHD ordering `L_z/L_⊥ ~ 1/ε`
made explicit. `B₀` stays the free per-ensemble knob in 2D, where the run itself has no
opinion about the guide-field strength.

**Heating is measured in the local-B frame.** `MOMENTS` carries `vperpB2` and `vparB2` —
`v_∥B² = (v·B)²/|B|²` and `v_⊥B² = |v|² − v_∥B²` with the pusher's own `B` sample — beside
the ẑ-referred `vperp2`/`vz2`, and `diagnostics.particles.heating_rate` halves all four.
Perpendicular heating is the local-B one (Xia et al. §4.1): the field direction tilts at
`O(ε)` while `|B|` varies only at `O(ε²)`, so the ẑ-referred split mixes ⊥ and ∥ energy at
first order in `b_⊥/B₀` — the same effect as the `μ` and grad-B remarks above.

### Dimensionless groups

`q/m` carries the scale interpretation of the gyration: `Ω = qm·B₀` and
`ρ = v_⊥/(qm·B₀)`, so at `B₀ = 1` (the default) `Ω = q/m` and `ρ = v_⊥ m/q`. `B₀` itself is
the amplitude parameter `1/ε` (above), not a unit choice. Three groups govern whether a
test-particle run means anything:

- `ρ/dx` — gyroradius against the grid, i.e. whether the fields the particle samples are
  resolved. Baseline `ρ ≈ 2–4 dx`.
- `Ω/ω_nl` — gyration against eddy turnover; the stochastic-heating parameter
  `ξ = δu(ρ)/v_⊥` is the same statement locally.
- `v_th/v_A` — thermal speed against the Alfvén speed.

At 256² (`dx ≈ 0.0245`) the baseline is `Ω = qm·B₀ ≈ 10–20`, i.e. 60–120 solver steps per
gyration: Boris at the solver `dt` is fully resolved with no substepping. `v_th/v_A` is
`v_th/B₀`.

**Guiding-centre formulas do not transfer naively.** Kernel gate 3
(`tests/test_particles_kernel.py`, derivation in its header) measures the perpendicular drift
in an RMHD `b_⊥ = ẑ×∇ψ` field and finds **2× the textbook `v_∇B`** (3× for a `v_z(0) = 0`
launch): the field *direction* shears at `O(ε)` while `|B| = √(B₀² + b_⊥²)` varies only at
`O(ε²)`, so shear terms enter the drift at the same order as the `|B|` gradient. Relevant
wherever a guiding-centre expression is used on these fields — including the `μ = v_⊥²/2|B|`
tracking of `plans/TESTPART_PLAN.md` §5.

## GDI (2D, `physics/gdi.py`)

Normalization `ρₛ = cₛ = Ωᵢ = 1` (eqs 5.4-5.5 of docs/"GDI_nonlinear_equations (10).pdf"),
fields `(N, φ)` in that order — `N = δn/n₀`, `φ = eφ_electrostatic/T_e`. The 3D model's
parallel-diffusion term `D_∥ ∂²/∂z²` is replaced by a perp-k-local closure,
`γ_∥(k) = gpar_fac · ν_in · k⊥²` (`gpar_fac` scales eq 4.3's current-closure floor; in eq
4.7's parametrization α = 1 + gpar_fac, so the default `gpar_fac=1` gives α=2, the
minimum-γ_∥ result quoted at eq 4.6) — valid until P4b adds a real kz axis.

**Deriving `L`.** `physics/gdi.linear_matrix` derives `L` directly from the scalar PDEs
(5.1)-(5.2), with `γ_∥` substituted for `D_∥ ∂²/∂z²`; the result agrees entry-by-entry
with the paper's eq (5.3) matrix under `L = -M` (full derivation and the (3.7)
factorization: docs/gdi_linear_matrix_note.tex):

```
L[N,N]     = -γ_∥ - diss·k⊥^(2·hyper)
L[N,φ]     =  γ_∥ + i·ky_deriv/Ln
L[φ,N]     =  gpar_fac·ν_in - i·ky_deriv·ν_in·v0·inv_ksq
L[φ,φ]     = -ν_in - gpar_fac·ν_in - diss·k⊥^(2·hyper)
```

`tests/test_gdi_linear.py` cross-checks this `L`'s eigenvalues against an independently
re-derived exact quadratic (eq 3.7, ω-form), and against eq (2.8)'s collisional (2.9) and
inertial/Keskinen-Ossakow (2.11) asymptotic limits at `γ_∥ = 0` — the dispersion
relations are the arbiter for the sign convention.

**The Nyquist-row fix.** `ky_deriv` is `kgrid.ky` with the last (Nyquist) row zeroed, used
everywhere an `i·ky` cross term appears in `L`. A bare `i·ky` there breaks the propagator
setup's `L(-kx,ky) = conj(L(kx,ky))` reality check at that row: this is the standard
spectral-differentiation subtlety that odd-order derivatives are ill-defined at the
self-conjugate Nyquist point (Trefethen). RMHD's nonlinear terms never hit this because
the 2/3 dealiasing mask removes the Nyquist row from every nonlinear/IC path; `L` is
applied unconditionally by the exact propagator (no dealiasing), so it needs the explicit
fix. Physically inconsequential (a single edge row with no resolved dynamics).

**dt ceiling.** The propagator applies `L` exactly, so it imposes no *stability* limit —
but the E×B CFL alone (mirroring `rmhd.set_timestep`) doesn't bound the timestep needed for
IF-RK to correctly weight the nonlinear forcing of a stiffly-damped mode (`γΔt ≲ 1`, see
plans/GDI_PLAN.md's "Design background"). `gdi.set_timestep` therefore also caps `dt` at
`lin_dt_safety / max|Re λ(L)|`, computed once with plain numpy at trace time
(`gdi._max_re_lambda`, `functools.lru_cache`d on `Parameters`' identity hash), with
`lin_dt_safety` (`params.eqpars`, default 0.5) a separate safety knob from `cfl_safety`.

The max is taken over the **dealiased** (2/3-rule) region only, not the whole k-grid.
`γ_∥ = gpar_fac·ν_in·k⊥²` (the density equation's electron-adiabaticity relaxation rate) is
genuinely unbounded in `k⊥` — real physics, not a bug — so including the grid-cutoff modes
would make `max|Re λ|`, and therefore `dt`, scale with resolution and `diss` for no
accuracy benefit: those modes receive *zero* nonlinear forcing (`run.initialize` masks the
IC, `NonlinearTerm` masks every step), so the "misweighted nonlinear forcing of a stiff
mode" concern the ceiling exists for cannot apply there, and the propagator applies `L`
there exactly regardless of how large `γ_∥` gets.

**Energy budget.** `diagnostics/gdi.py`'s `energy_budget` extends eq (3.18) (which is derived at `v0=0,
1/Ln=0` — no drive) with the Ln/Pedersen drive terms and the perp-hyperdissipation sink,
since the bracket/advection nonlinearity conserves `E = ½⟨N² + |∇⊥φ|²⟩` exactly under
periodic BCs — so the full linear operator's contraction against `(N, φ)` accounts for the
*entire* measured `dE/dt` of a nonlinear run. Measured closure in
`tests/test_gdi_linear.py::test_energy_budget_closure_nonlinear`: relative error ~4e-8 at
fp64 over a short random-IC run (centered-difference `dE/dt` vs the budget's `total`).

## Compressible MHD (`physics/cmhd.py`, plans/CMHD_PLAN.md)

Status: this section is the Phase-C0 derivation record — C1 implements against it, and
the C1 gates are built FROM the expressions here. Scope per the plan: `dims==3` +
`z_spectral=True`, single process, polytropic closure, isotropic hyperdissipation as the
whole of `L`, all wave physics explicit under IF-LSRK (or CB-IMEX; both families are
valid on a pure-dissipation diagonal `L`).

### Variables, units, equations

Seven fields, state order `(ρ, u_x, u_y, u_z, B_x, B_y, B_z)`. Density is in units of
the mean density (`⟨ρ⟩ = ρ₀ = 1` by convention); `u` and `B` share one velocity unit —
Alfvén units, `B_code = B/√(μ₀ρ₀)` — so the local Alfvén speed is `|B|/√ρ` and a uniform
field of magnitude `B₀` has `v_A = B₀`. Lengths are box units, `p = K·ρ^γ` with
`K = c_s0²/γ` so that `c_s(ρ=1) = c_s0` (isothermal `γ = 1`: `p = c_s0²·ρ`,
`c_s = c_s0` everywhere). Plasma beta about the background: `β = 2p(1)/B₀² = 2c_s0²/(γ·v_A²)`
(isothermal: `β = 2c_s0²/v_A²`).

The ideal equations, in exactly the form the RHS computes them:

```
∂ρ/∂t = −∇·(ρu)                                        (flux form)
∂u/∂t = −(∇×u)×u − ∇(|u|²/2 + h(ρ)) + (∇×B)×B/ρ        (rotational form)
∂B/∂t = ∇×(u×B)                                        (curl form)
```

using the vector identity `u·∇u = (∇×u)×u + ∇(|u|²/2)` and the barotropic pressure
force `(1/ρ)∇p = ∇h`, with the specific enthalpy `h′(ρ) = c_s²(ρ)/ρ`,
`c_s²(ρ) = γKρ^(γ−1)`:

```
h(ρ) = c_s0²·ρ^(γ−1)/(γ−1)      γ > 1
h(ρ) = c_s0²·ln ρ               γ = 1   (the default)
```

The branch on `γ == 1` is trace-time python (`params` is static). Note the fp trap the
plan records: `ρ**(γ−1)` at non-integer exponent is NaN for `ρ < 0` — that loud failure
is intended, never abs()/clip it.

Dissipation is NOT in the RHS: `L = −diss_f·(k⊥² + k_z²)^hyper` per field (uniform
`diss` → the broadcast leading-axis-1 diagonal), applied exactly by the propagator.
Two honesty notes. First, damping `u` rather than `ρu` is not a conservative viscous
stress — total momentum is not conserved by the sink (nor gated; it is a small-scale
absorber, same philosophy as RMHD's hyperdissipation). Second, `k_z` enters `L` through
the same Nyquist-zeroed helper as every z-derivative in the module (one rule, no
exceptions); for the even power this is optional — `k_z²` is reality-safe at the Nyquist
plane — but the 2/3 `k_z` cut removes that plane from every IC and nonlinear path
anyway, so the uniform rule is inconsequential and simpler to audit.

### The background lives at k = 0, and is preserved bitwise

There is no background/fluctuation split and no linear wave term: `B₀` and `ρ₀` are the
`k = 0` modes of the evolved fields, and the waves emerge from the quadratic terms
(`u×B` contains `u×B₀`, `ρu` contains `ρ₀u`, …). Both `k = 0` modes are preserved
**bitwise**, not just to round-off, by construction: at `k = 0` the flux form is
`−i·0·(ρu)^ = 0` exactly, the curl is a difference of exact zeros (fp `0.0·x = ±0.0`
for finite `x` — signed zero, but `y + (±0.0) = y` bitwise for every nonzero `y`, and
scaling by `dt` and the RK coefficients keeps zeros zeros), the dealias mask is 1 there,
and the propagator factor is `exp(−0.0) = 1.0` exactly. So mass (`ρ̂(0)`) and each
mean-B component are bitwise invariants of the full discrete step, and the C1 gate
asserts them bitwise. (The one signed-zero edge — a k=0 component stored as `−0.0`
flipping to `+0.0`, value-equal but not bitwise — is unreachable through fft-built ICs,
and the gates build ICs in real space regardless.) The mean of `u` is NOT preserved —
`⟨(∇×u)×u⟩` and `⟨(∇×B)×B/ρ⟩` do not vanish — and should not be: the continuum
invariant is `∫ρu`, which the discrete advective-form step only conserves to scheme
accuracy (not gated).

### Ideal invariants and their discrete status

`E = ∫ [ ρ|u|²/2 + |B|²/2 + ρe(ρ) ]` with the specific internal energy from
`e′(ρ) = p/ρ²`:

```
ρe(ρ) = p/(γ−1) = c_s0²·ρ^γ/(γ(γ−1))     γ > 1
ρe(ρ) = c_s0²·ρ·ln ρ                     γ = 1   (can be negative; only drift matters)
```

(Consistency check: `h = e + p/ρ` — for `γ = 1` the two differ by the constant `c_s0²`
per unit mass, i.e. a multiple of the conserved mass, which shifts `E` by a constant.)

| invariant | discrete status | gate |
|---|---|---|
| mass `ρ̂(k=0)` | bitwise (above) | C1, bitwise |
| mean B | bitwise (above) | C1, bitwise |
| `k·B̂` | round-off random walk, no systematic source | C1, tolerance `~ε·√N_steps` scale, no secular trend |
| `E` (ν = 0) | O(dt^p) + non-polynomial aliasing residual | C1, dt-convergence order + smallness — NEVER round-off |
| cross helicity `∫u·B` (ν = 0) | same class as `E` (ideal invariant of barotropic MHD) | C1, same form |
| magnetic helicity | conserved in continuum; needs `A` to measure | not gated |

The div-B argument, precisely: `∂B̂ = ik×Ê` gives `k·∂B̂` as a pairwise-cancelling sum
(`k_x k_y Ê_z` appears twice with opposite signs, etc.) — zero analytically, but the two
copies are grouped differently in fp (`k_x·(k_y·Ê_z)` vs `k_y·(k_x·Ê_z)`) and round
differently, so each evaluation deposits `O(ε_mach·|k|²|Ê|)` into `k·∂B̂` (the products
are `k_i k_j Ê`; the dimensionless gate quantity `|k·B̂|/(|k||B̂|)` carries one power of
`|k|`). Nothing amplifies or reads
it: the exponential propagator and the dealias mask scale all three components of `B̂`
identically (they commute with `k·`), so div B random-walks at machine epsilon and
nothing more. Any future induction-equation term must also be a curl or k-locally
divergence-free — this is a standing rule.

For the C2 energy budget: the functional derivatives are `δE/δρ = |u|²/2 + h(ρ)`,
`δE/δu = ρu`, `δE/δB = B`, so the hyperdissipation sink is

```
ε = −⟨ ρu·(D_u u) + B·(D_B B) + (|u|²/2 + h)·(D_ρ ρ) ⟩,   D_f = the field's diagonal of L
```

and `energy_budget`'s closure is `dE/dt + ε ≈ 0` to the same order-plus-aliasing class
as the `E` gate — never round-off. Note the `D_ρ` term: mass diffusion does `pdV`-like
work through `h`; omitting it is the likely first bug in a budget implementation. (At
γ = 1 the strict derivative is `d(ρe)/dρ = c_s0²(ln ρ + 1)` = `h + c_s0²`; the constant
multiplies `⟨D_ρ ρ⟩`, whose k = 0 mode is exactly zero since `L(0) = 0`, so it drops out
of `ε` — use `h` and don't chase the discrepancy.)

### Linear waves: dispersion relations and eigenvectors

Linearize about `ρ = 1`, `B = B₀ẑ` (the gates use ẑ WLOG — the code has no preferred
axis; EBM later distinguishes x), modes `∝ exp(i(k·x − ωt))` with `k = k(sinθ, 0, cosθ)`,
`k_∥ = k·ẑ = k·cosθ`, `v_A = B₀`, `c = c_s0`:

```
ω δρ = k·δu
ω δu = c² k δρ − (k×δB)×B₀
ω δB = −k_∥B₀ δu + (k·δu) B₀ẑ
```

**Alfvén branch**: `δu ∝ ŷ` (perpendicular to the (k, B₀) plane), `δρ = 0`,
`ω² = k_∥²v_A²`, and `δB_y = −(k_∥B₀/ω)·δu_y` — transcribe THIS form into the gates
(with signed `k_∥` and `ω = +k_∥v_A` it reads `δB = −δu`); the `∓` shorthands are
convention-ambiguous.

**Magnetosonic branches**: `δu = (δu_x, 0, δu_z)` in the (k, B₀) plane;

```
ω⁴ − ω²k²(c² + v_A²) + c²v_A²k²k_∥² = 0
ω² = ½k²(c² + v_A²)·[ 1 ± √(1 − 4c²v_A²cos²θ/(c² + v_A²)²) ]     (+ fast, − slow)
```

with eigenvector (derived from the z-momentum equation, which has NO linear Lorentz
force — `[(k×δB)×B₀]·ẑ ≡ 0`):

```
δu_z/δu_x = c²k_x k_z / (ω² − c²k_z²)
δρ   = (k_x δu_x + k_z δu_z)/ω
δB_x = −k_z B₀ δu_x/ω        δB_y = 0        δB_z = +k_x B₀ δu_x/ω
```

(`k·δB = 0` identically — check it when transcribing.) The `δu_z/δu_x` expression is
singular exactly where `δu_x → 0`: at `θ = 0` do not use the ratio, use the polarizations
directly. Degenerate limits the gates must assert explicitly:

- `θ = 0`: the branches decouple into a sound wave (`δu ∥ ẑ`, `ω = ±kc`, `δB = 0`) and a
  transverse `x`-polarized wave at `ω = ±kv_A` (degenerate with the Alfvén branch). Which
  of "fast"/"slow" is which depends on `c` vs `v_A` — at the target `β = 0.3`
  (`c ≈ 0.39·v_A`) the sound wave is the SLOW branch. Label by speed, not by name.
- `θ = π/2`: slow → `ω = 0`; fast → the magnetosonic `ω = ±k√(c² + v_A²)`. The `ω = 0`
  eigenspace is DEGENERATE (it contains the pressure-balanced `δu_z ẑ` member — the
  actual `k_z → 0⁺` slow-branch limit — and the `δB_y`/`δu_y` shear members too); the C1
  gate uses the static member `δu = 0`, `δB = δB_z ẑ` with total-pressure balance
  `c²δρ + B₀δB_z = 0`, chosen because a pressure-force sign error moves it at O(1). The
  Alfvén `δB_y = −(k_∥B₀/ω)δu_y` is 0/0 at `θ = π/2`; the limit is `δB = −δu` (exact at
  every `k_z > 0` for the + root, hence in the limit). Both were re-derived in the C1
  adversarial review.

Gate ICs are built in REAL space from these eigenvectors (`cos/sin(k·x)` with the
analytic polarization) at amplitude small enough that the `O(ε²)` quadratic self-
interaction sits below the measurement tolerance — never written directly into k-space
(the rfftn reality constraint is then satisfied by construction).

### Timestep

`set_timestep` bounds the angle-maximised fast phase speed by
`c_f(x) = √(c_s²(ρ) + |B|²/ρ)` pointwise (from the dispersion relation,
`ω²/k² ≤ c² + v_A²`), and returns `cfl_safety / max_i max_x [ (|u_i| + c_f)/d_i ]` over
the three directions (`d_z = params.dz = Lz/nz`, set unconditionally for every
`dims==3` run in config.py — z_spectral does not remove it), clamped by the
fixed-dt ceiling as in RMHD. No
quiescent floor is needed, but the reason is a GRID-MAX argument, not pointwise: at
γ = 1, `c_f ≥ c_s0` pointwise; at γ > 1, `c_s(ρ) = c_s0·ρ^((γ−1)/2)` falls below `c_s0`
wherever ρ < 1 — but mass conservation pins `max_x ρ ≥ ⟨ρ⟩ = 1` (a bitwise invariant,
above), so `max_x c_f ≥ c_s0` and the CFL denominator is bounded away from zero for any
reachable state. At the first
production target (`β = 0.3`, `δB/B₀ = 1`) the CFL is Alfvénic — `c_s < v_A`, so
explicit acoustics cost nothing over z_spectral RMHD at the same `B₀`.

### Aliasing and the transform budget

The quadratic nonlinearities are exactly dealiased by the existing 3D 2/3 mask. `h(ρ)`,
`ln ρ` and `1/ρ` are not polynomial: no finite padding dealiases them, every spectral
compressible code accepts the residual, and at smooth `ρ` it sits inside the
time-discretization error the conservation gates already budget for — which is why those
gates assert convergence ORDER plus absolute smallness and never round-off.

Transform tally per RHS evaluation (the C1 review checklist; C2 measures the real cost):
inverse — `ρ`, `u`(3), `B`(3), `ω = ∇×u`(3), `j = ∇×B`(3), both curls formed k-locally
first = **13**; forward — the combined curl force `(j×B)/ρ − ω×u`(3, summed in real
space and transformed ONCE), the combined scalar `|u|²/2 + h`(1), `u×B`(3), `ρu`(3)
= **10**; total **23** 3-D FFTs, vs 10 for z_spectral RMHD — expect ~2–2.5× the RMHD
step, to be measured, not quoted.

### Expanding box (Phase C3, plans/CMHD_PLAN.md §5)

Source of truth: Squire, Chandran & Meyrand 2020 (ApJL 891, L2; arXiv:2001.08422),
eqs (1)–(3). Their comoving frame: `a(t) = 1 + ȧt` with constant `ȧ`, radial axis x,
transverse (y, z) expanding, comoving gradient `∇̃ = (∂_x, a⁻¹∂_y, a⁻¹∂_z)`. In taranis
code units (4π absorbed, Alfvén units) their system is exactly our §"Variables" set with
`∇ → ∇̃` plus three expansion terms:

```
∂ρ/∂t = −∇̃·(ρu) − 2(ȧ/a)ρ
∂u/∂t = −(∇̃×u)×u − ∇̃(|u|²/2 + h) + (∇̃×B)×B/ρ − (ȧ/a)·T·u      T = diag(0,1,1)
∂B/∂t = ∇̃×(u×B) − (ȧ/a)·Λ·B                                     Λ = diag(2,1,1)
```

The rotational-form identity survives the anisotropic metric — for any DIAGONAL scaled
derivative `∂̃ᵢ = cᵢ∂ᵢ`, `[(∇̃×u)×u]ᵢ = (u·∇̃)uᵢ − ∂̃ᵢ(u²/2)` by the same ε-tensor
algebra, so the combined-gradient trick is unchanged. Λ = diag(2,1,1) is flux
conservation: B_x threads the transverse area (∝ a²) so B_x ∝ a⁻²; B_⊥ threads one
transverse and one radial direction (∝ a) so B_⊥ ∝ a⁻¹; with ρ ∝ a⁻² the normalized
amplitudes obey `B_x/√ρ ∝ a⁻¹`, `B_⊥/√ρ ∝ a⁰` — Squire et al.'s stated scalings.

**Closure: EBM is isothermal-only** (`γ = 1` enforced when expansion is on). Squire et
al.'s closure is isothermal with a prescribed cooling law, and pinning to their closure
is the whole justification — the restriction is scope, not mathematics: EBM preserves
`D_t(p/ρ^γ) = 0` exactly (the expansion sources cancel — combine Dong et al. 2014 eqs
8 and 11), so a barotropic γ > 1 would drop in with `h = c_s0²·a^(−2(γ−1))·ρ′^(γ−1)/(γ−1)`,
still gradient-form, γ = 5/3 reproducing the a^(−4/3) cooling automatically; deferred,
not blocked. The cooling law is
`c_s²(t) = c_s0²·a^(−q)`, `q ≥ 0`, default `q = 4/3` (their Athena++ "adiabatic"
mimic, i.e. c_s² ∝ ρ^(2/3) on the background ρ ∝ a⁻²; `q = 0` is constant temperature).
Their Snoopy-run `c_s² = 0.35 t⁻¹` fit is not supported — it is dimensional and
box-specific. So `h(ρ, t) = c_s²(t)·ln ρ`. Background β on a radial field runs as
`β_x = 2c_s²ρ/B_x² ∝ a^(2−q)` (grows at q = 4/3); combine the scalings per configuration
rather than quoting a universal β(t).

**Rescaled variables are the implementation form** (the Grappin-family trick, worked out
for the curl discretization): evolve

```
ρ′ = a²ρ        B′ = (a²B_x, a·B_y, a·B_z)        u unscaled
```

Substituting kills the −2(ȧ/a)ρ and −(ȧ/a)Λ·B terms identically, and the three
equations become

```
∂ρ′/∂t = −∇̃·(ρ′u)                                    (flux form in k̃, unchanged shape)
∂u/∂t  = (ideal RHS with ∇̃, h = c_s²(t)·ln ρ′) − (ȧ/a)·T·u
∂B′/∂t = ∇×E′        E′ = ((u×B)_x, a(u×B)_y, a(u×B)_z),  ∇ the STATIC comoving curl
```

(the uniform `−2 c_s²(t) ln a` from `ln ρ = ln ρ′ − 2 ln a` dies under the gradient).
The induction identity is the load-bearing step: with `A = diag(a², a, a)`,
`A·(∇̃×E) = ∇×E′` component by component —

```
x: a²·a⁻¹(∂_yE_z − ∂_zE_y) = ∂_y(aE_z) − ∂_z(aE_y)   ✓
y: a·(a⁻¹∂_zE_x − ∂_xE_z)  = ∂_zE_x − ∂_x(aE_z)      ✓
z: a·(∂_xE_y − a⁻¹∂_yE_x)  = ∂_x(aE_y) − ∂_yE_x      ✓
```

so `∂_tB̂′ = iK×Ê′` with STATIC integer-grid wavenumbers K. Consequences, all inherited
verbatim from the non-expanding argument: `K·B̂′` is a round-off random walk with no
systematic source (the physical divergence is `k̃·B̂ = a⁻²·K·B̂′`), and the k = 0 modes
of ρ′ and B′ are BITWISE invariants of the discrete step — equivalently the raw
backgrounds track ρ ∝ a⁻², B_x ∝ a⁻², B_⊥ ∝ a⁻¹ exactly. The k = 0 velocity is NOT an
invariant in general — the mean Reynolds/Maxwell stresses source it, exactly as the
non-expanding section says — but ON A SPATIALLY UNIFORM STATE (which stays uniform
exactly) it obeys the pure ODE `du(0)/dt = −(ȧ/a)T·u(0)`: `u_x(0)` bitwise (T_x = 0)
and `u_⊥(0) ∝ a⁻¹` to scheme order. That uniform-IC pair is the exact-ODE gate, and the
stage-time gate: its measured convergence order collapses if a stepper ever stops
setting stage-correct t. Validation enforces `ȧ > 0`; `a(t) ≤ 0` itself is not
validatable (t is traced at every use site), so the doctored-restart hole
(`t < −1/ȧ`) is recorded as a comment at the guard, nothing more.

**Where a(t) enters the graph** (everything trace-time-gated on `"expansion" in
params.eqpars` — off, every factor is the literal identity and the graph must be BITWISE
the non-EBM graph, which is a C3 gate): `k̃ = (k_x, k_y/a, k_z/a)` wherever grad and the
term funcs form derivatives — EXCEPT the induction curl, which stays on static K per the
identity above; the elementwise `B̂ = A⁻¹B̂′`, `ρ̂ = ρ̂′/a²` unscalings in
grad; the E′ scaling; `c_s²(t)`; the `−(ȧ/a)T·u` term (the ONLY new additive term); and
the CFL, which uses physical spacings `(d_x, a·d_y, a·d_z)` with speeds from the
UNPRIMED fields and c_s(t). a(t) comes from `grads.t` — cast to `_precision.ftype`
before it touches field math (`CMHDGrads.t` is deliberately float64). The transform
tally is unchanged at 23: every rescaling is elementwise, no new transforms.

Two recorded truncation choices: dissipation acts on the PRIMED fields at COMOVING k
(static diagonal L, exactly the §"Variables" operator — the physical-wavenumber
alternative needs ν(t) or per-block L rebuilds and is rejected); and the 2/3 mask is
exactly correct as is, because products are formed on the comoving grid and alias in
comoving mode indices — the mask never needs to know about a(t).

**WKB gate**: a small-amplitude Alfvén wave propagating along x on a `B_x` background at
`ȧ/a ≪ ω` conserves wave action: velocity amplitude `δu ∝ a^(−1/2)` (equivalently
`δb = δB/√ρ ∝ a^(−1/2)`), hence raw `δB_⊥ ∝ a^(−3/2)` and the normalized
`δB_⊥/B_x ∝ a^(+1/2)` — the amplitude growth that drives switchback formation, and the
C3 gate measures the −1/2 exponent over a factor 4 in a (a decade buys no extra
discrimination for ~4× the steps) with a tolerance budgeted from the O(ȧ/(aω)) WKB
corrections.

### The ln ρ density variable (Phase C4, plans/CMHD_PLAN.md §5)

An opt-in variable change for positivity: `eqpars["density_var"] = "lnrho"` evolves
`s = ln ρ` in field 0 instead of ρ (default `"rho"`, byte-identical to the C1–C3 solver).
Motivation and the conservation trade: transonic turbulence has near-lognormal density,
so `s` is the smooth near-Gaussian field where ρ has decade-spanning valleys — the
spectral representation improves exactly where the ρ-form rings and crashes — and
`ρ = e^s > 0` is structural, retiring the NaN channel entirely.

**Continuity in s** (divide the flux form by ρ = e^s):

```
∂s/∂t = −u·∇s − ∇·u
```

exactly equivalent in the continuum. **Mass conservation changes discrete class, by
necessity, not by accident**: in ρ-variables mass is the single coefficient `ρ̂(k=0)`
with an exactly-zero RHS projection (bitwise for free); in s-variables the conserved
functional `∫e^s` is nonlinear in the evolved field and there is NO linear substitute —
`⟨s⟩` is not conserved even in the continuum (`d⟨s⟩/dt = ⟨s∇·u⟩`; Jensen forces
`⟨s⟩ ≈ −σ_s²/2` downward as fluctuations grow at fixed mass). RK schemes preserve linear
invariants exactly and nonlinear ones to scheme order — for forward Euler the one-step
mass residual is the SIGN-DEFINITE `(dt²/2)∫ρ(∂_t s)²` (a systematic source, not a
random walk); lsrk54 leaves an O(dt⁵)-per-step, non-sign-definite residual — and the
spatial product rule `e^s∇s = ∇(e^s)` fails for interpolants (non-polynomial, the 2/3
rule cannot make it exact). So mass joins the class energy is already in: gated by drift
order + smallness, never round-off, with the mean-B bitwise and div-B round-off gates
UNCHANGED (induction is untouched). On a spatially uniform state every RHS term is an
exact fp zero (`ik̃ŝ`, `ik̃û` vanish identically), so `ŝ(k=0)` is bitwise there — the
uniform-state gates survive; turbulent `⟨s⟩` drifts, as the physics requires.

**Enthalpy and forces.** `h(s)`:

```
h = c_s²·s                              γ = 1  (LINEAR: the pressure force is the
                                               k-local −c_s²·ik̃ ŝ (a vector), no
                                               transform; the implementation folds
                                               c_s²ŝ into the gradient scalar before
                                               the −ik̃ multiply — algebraically
                                               identical by distributivity)
h = c_s0²·e^((γ−1)s)/(γ−1)              γ > 1  (pointwise, as before)
```

The Lorentz `1/ρ = e^(−s)` is now the ONLY non-polynomial term at γ = 1 (the ρ-form
carries both ln ρ and 1/ρ). `c_s(s)`: constant at γ = 1; `c_s0·e^((γ−1)s/2)` at γ > 1
(`set_timestep` uses it and `v_A² = |B|²e^(−s)`; the γ > 1 grid-max no-floor argument
weakens — max ρ ≥ 1 now holds only to the mass-drift order — but at the isothermal
default `c_s` is constant and the floor is exact).

**Energy functional and the budget trap.** `E = ∫ e^s|u|²/2 + |B|²/2 + ρe(s)` with
`ρe = c_s²·e^s·s` (γ = 1) or `c_s0²·e^(γs)/(γ(γ−1))` (γ > 1). Functional derivatives:

```
δE/δu = e^s·u        δE/δB = B
δE/δs = e^s·(|u|²/2 + h + c_s²)     γ = 1
δE/δs = e^s·(|u|²/2 + h)            γ > 1
```

Note the INVERSION of the ρ-form's constant rule: there the γ = 1 constant multiplied
`⟨D_ρρ⟩` whose k = 0 mode is exactly zero and dropped out; here it multiplies
`⟨e^s·D_s s⟩`, whose mean is NOT zero — **the +c_s² term must be kept in the γ = 1
budget sink**, and omitting it (by analogy with the ρ-form note) is the likely first
bug of this phase. γ > 1 is the clean branch this time. These budget expressions are
NON-EXPANDING: under EBM, `dE/dt` also carries `∂_t c_s²(t)` and the expansion work, so
the `dE/dt + ε ≈ 0` closure applies only with expansion off (true of the ρ-form budget
too — the C2 diagnostics predate EBM).

**Dissipation** acts on s (field-0 row of the same diagonal L): mass-affecting through
the variance (⟨s⟩ untouched at k = 0, `∫e^s` not), undershoot-proof (diffusion of s
cannot drive ρ ≤ 0), equally non-physical, still budget-accounted via `δE/δs`.

**EBM composes cleanly**: `ln ρ′ = s + 2 ln a ≡ s′`, and `∂_t s′ = −u·∇̃s′ − ∇̃·u` —
the −2(ȧ/a) expansion term (spatially uniform in s-variables) cancels identically
against `d(2 ln a)/dt`, the exact analogue of the ρ′ rescaling. Raw
`e^(−s) = a²·e^(−s′)`; `h`'s uniform `−2c_s²(t)ln a` dies under the gradient as before.
γ = 1 only, per the EBM scope pin.

**Transform tally: 24** (one more than the ρ-form): inverse — s(1, for `e^(−s)` and the
γ > 1 h), ∇s(3), u(3), B(3), ω(3), j(3) = 16; forward — combined curl force(3),
`|u|²/2` scalar(1; h joins it only at γ > 1), u×B(3), `u·∇s`(1) = 8; `∇·u = ik̃·û` is
k-local. `ρu`'s three forward transforms are gone.

**Snapshots/IC**: field 0 changes meaning; `density_var` rides `eqpars` into
params.json, so the differing-record check blocks cross-mode restarts; `initialize`'s
user function must return s in field 0 in this mode (document at the helper). No
migration of old snapshots — a log-converter is trivial if ever wanted.

## FLR-MHD (`physics/flrmhd.py`, plans/FLRMHD_PLAN.md)

Status: this section is the derivation record of rungs F0–F3: the recipe in 2D and
`z_spectral` (F1), the O-U forcing (F3) and `diagnostics/flrmhd.py` (F2) implement
against it, and their gates are built FROM the expressions here. FLR-MHD is the low-β finite-Larmor-radius extension of
RMHD of Meyrand, Squire, Schekochihin & Dorland (2021, JPP 87, 535870301;
arXiv:2009.02828), valid on both sides of `k⊥ρ_i ~ 1`; it is the model in which the
helicity barrier was found. Scope: `eqtype="FLRMHD"`, two fields `(φ, ψ)` — the SAME
variables and bracket conventions as the RMHD recipe — `dims==2` and `dims==3` +
`z_spectral`, single process.

### Equations in Gaussian units and the map to (φ, ψ)

Gaussian (CGS) units throughout this subsection. Background: `B₀ = B₀ẑ` (G), species
densities `n_0a` (cm⁻³) and temperatures `T_0a` (erg), ion charge `q_i = Ze`,
`τ = T_0i/T_0e`, and

```
Ω_i = Z e B₀/(m_i c)                    ρ_i = c·√(2 m_i T_0i)/(Z e B₀)   (v_th,i = √(2T_0i/m_i))
v_A = B₀/√(4π n_0i m_i)                 n_0e = Z n_0i
```

The fields are the electrostatic potential `ϕ` (statV), the parallel vector potential
`A∥` (G·cm) and the electron density perturbation `δn_e/n_0e`. With the E×B flow
`u⊥ = (c/B₀) ẑ×∇⊥ϕ` and the field-line tilt `b⊥ = −(1/B₀) ẑ×∇⊥A∥` (note the sign), the
paper's Eqs. (2.1)–(2.3), dissipation omitted, are

```
(∂/∂t + u⊥·∇⊥) δn_e/n_0e = −(c/(4π e n_0e)) (∂/∂z + b⊥·∇⊥) ∇⊥²A∥
(∂/∂t + u⊥·∇⊥) A∥        = −c ∂ϕ/∂z + (c T_0e/e) (∂/∂z + b⊥·∇⊥) δn_e/n_0e
δn_e/n_0e                = −(Z/τ) (1 − Γ̂₀) eϕ/T_0e
```

where `Γ̂₀ = I₀(α̂)·exp(−α̂)`, `α̂ = −ρ_i²∇⊥²/2`, i.e. `Γ₀(α)` at `α = k⊥²ρ_i²/2` in
Fourier space. Limits: `1 − Γ₀ → k⊥²ρ_i²/2` (`k⊥ρ_i ≪ 1`, RMHD) and `→ 1` (`k⊥ρ_i ≫ 1`,
electron RMHD / kinetic Alfvén waves). The third equation is quasineutrality with the
gyrokinetic ion response; the first is electron continuity (parallel electron flow =
parallel current), the second the parallel electron momentum equation (Ohm's law with
the isothermal electron pressure gradient along the perturbed field).

Code variables (lengths in box units, velocities in units of `v_A`, so `v_A = 1`):

```
φ  = c ϕ/B₀                    stream function of u⊥ = ẑ×∇φ             (cm²/s)
ψ  = −A∥/√(4π m_i n_0i)        flux function,  b⊥ = δB⊥/√(4π m_i n_0i) = ẑ×∇ψ  (cm²/s)
n̂  = Ω_i δn_e/n_0e             density in vorticity units               (1/s)
```

— `φ`, `ψ` exactly as in the RMHD recipe (`A_z = −ψ`). With the bracket
`{a,b} = ∂ₓa ∂_y b − ∂_y a ∂ₓb` (the paper's `[a,b]`): `u⊥·∇f = {φ,f}` and
`v_A b⊥·∇f = {ψ,f}`. Three coefficient identities do the rest:

```
(Z/τ) e B₀/(c T_0e)            = 2/(ρ_i² Ω_i)             (quasineutrality)
Ω_i c √(4π m_i n_0i)/(4π e n_0e) = v_A                     (continuity)
c T_0e/(e Ω_i √(4π m_i n_0i))  = (Z ρ_i²/(2τ)) v_A ≡ ρ_s² v_A      (Ohm)
```

`ρ_s² = Zρ_i²/(2τ) = Z T_0e/(m_i Ω_i²)` is the ion sound radius squared. So, at
`v_A = 1`,

```
n̂_k = −G(k⊥) φ_k ,          G(k⊥) = (2/ρ_i²)·(1 − Γ₀(k⊥²ρ_i²/2))
∂ₜn̂ = −{φ, n̂} + {ψ, ∇⊥²ψ} + ∂_z ∇⊥²ψ
∂ₜψ = −{φ_e, ψ} + ∂_z φ_e ,     φ_e = φ − ρ_s² n̂ = (1 + ρ_s²G) φ
```

Two structural facts the implementation rests on:

- `{φ, n̂} = {φ_e, n̂}` identically (`{n̂, n̂} = 0`), so both brackets are advected by the
  same potential `φ_e` and the RHS needs gradients of exactly four fields
  `(φ_e, ψ, n̂, j = ∇⊥²ψ)` — the transform count of RMHD (8 inverse + 2 forward).
- FLR-MHD is RMHD with three k-local substitutions: vorticity `∇⊥²φ → n̂ = −Gφ`,
  advecting potential `φ → φ_e` in both brackets, inversion `1/k⊥² → 1/G`.

Only `ρ_i` (in box units) and `Z/τ` enter the dynamics. `Ω_i` drops out: it is needed
only to convert `n̂` back to `δn_e/n_0e = n̂/Ω_i`, an amplitude of the RMHD ordering
(like `B₀` for the test particles). eqpars: `rho_i`, `tau`, optional `Z` (default 1),
`diss`, `hyper`, optional `z_diss_k` and `z_hyper` (the latter warns at `setup_kgrids`
when set away from its default where it is dead: 2D, or `z_diss_k == 0`).

### The FLR factors and their small-α evaluation

`flrmhd.factors` (the recipe's `kgrid_func`, stored once in `kgrid.eq` by
`setup_kgrids`) builds six real `(nkx, nky)` arrays from one function,
`S(α) = (1 − Γ₀(α))/α` with `α = k⊥²ρ_i²/2`:

```
G     = k⊥²·S                       → k⊥²   (α → 0);    → 2/ρ_i²        (α → ∞)
inv_G = (1/k⊥²)/S ,  0 at k⊥ = 0
pe    = 1 + ρ_s²G = 1 + (Z/τ)·α·S    → 1;                → 1 + Z/τ
a     = k⊥²/G = 1/S                  → 1;                → α
κ     = √(G·pe)/k⊥ = √(S·pe)         → 1
v_ph  = k⊥·√(pe/G) = √(pe/S)         → 1;                → (1/2 + Z/(2τ))^{1/2} k⊥ρ_i
```

Everything is a function of `S` and `α` alone, so no factor is ever a product of two
separately rounded ones (`a` is `1/S`, never `k⊥²·inv_G`), and `k⊥ = 0` needs no special
case beyond `inv_G`'s zero: `S(0) = 1` gives `pe = a = κ = v_ph = 1` there.

`1 − I₀(α)e^{−α}` cancels at small α (relative error ≈ ε/α: 1e-8 at α = 1e-8), so `S`
has two branches, switched at `α = 1`:

```
α < 1 :  S = Σ_{n≥1} (−1)^{n+1} c_n α^{n−1} ,   c_n = (2n)!/(2ⁿ (n!)³)
         = 1 − (3/4)α + (5/12)α² − (35/192)α³ + (21/320)α⁴ − …      (28 terms, Horner)
α ≥ 1 :  S = (1 − i0e(α))/α                      (jax.scipy.special.i0e)
```

Measured against 40-digit `mpmath` (jax 0.10.0, fp64): the series is within 2.2e-16 of
the reference on `[1e-14, 1]` from 24 terms on, the closed form within 2.2e-16 on
`[1, 1e3]`. The series argument is clamped to the switch point inside the `where`, so
the unselected branch stays finite and its tangent is finite under the overrides seam.
The factors are computed in float64 at every `TARANIS_PRECISION` and cast to the field
real dtype. The leading corrections, used by the gates:
`G = k⊥²(1 − (3/8)k⊥²ρ_i² + …)` and `v_ph² = 1 + (3/4 + Z/τ)·k⊥²ρ_i²/2 + …`.

`v_ph` above is the paper's `v_ph = (k⊥ρ_i/√2)·[1/(1 − Γ₀) + Z/τ]^{1/2}`. Reference
values at `Z/τ = 2`: `v_ph(k⊥ρ_i = 0.1, 1, 10) = 1.007, 1.552, 12.37`.

### Linear waves and the generalized Elsasser potentials

Linearizing about rest (`n̂ = −Gφ`, so `∂ₜφ = −(1/G)∂ₜn̂`):

```
∂ₜφ = i·kz·a·ψ ,     ∂ₜψ = i·kz·pe·φ          ⇒   ω² = kz²·a·pe = kz²·v_ph²
```

In Gaussian units `ω = ± k_z v_A v_ph(k⊥)`: the Alfvén wave at `k⊥ρ_i ≪ 1`, the kinetic
Alfvén wave `ω ≈ k_z v_A k⊥ρ_i (1/2 + Z/(2τ))^{1/2}` at `k⊥ρ_i ≫ 1`. With `χ = κφ`:

```
∂ₜχ = i·kz·v_ph·ψ ,   ∂ₜψ = i·kz·v_ph·χ
Θ± = χ ± ψ = κφ ± ψ ,        ∂ₜΘ± = ± i·kz·v_ph·Θ±
```

`Θ±` are the paper's generalized Elsasser potentials,
`Θ±_k = −Ω_i (v_ph/k⊥²) δn_e/n_0e ∓ A∥/√(4π m_i n_0i)`, under the map above
(`−(v_ph/k⊥²) n̂ = (v_ph G/k⊥²) φ = κφ`). A `Θ⁺` Fourier coefficient advances as
`exp(+i·kz·v_ph·t)`, i.e. `Θ⁺` propagates toward **−z** — as RMHD's `z⁺ = φ + ψ` does
under `∂ₜψ = ∂_zφ`.

Under `z_spectral` the waves live in the linear operator, on `(φ, ψ)`:

```
L = [[ d_φ ,        i·kz·a(k⊥) ],
     [ i·kz·pe(k⊥) ,    d_ψ    ]] ,      d_f = −ν_f·k⊥^{2h} − ν_z·kz^{2h_z}
```

with `kz` the Nyquist-zeroed `rmhd._kz_deriv`, `(ν_φ, ν_ψ) = eqpars["diss"]`,
`h = hyper`, `ν_z = z_diss_k`, `h_z = z_hyper` (default 2 = RMHD's `kz⁴`; 3 is the
paper's `ν_6z kz⁶`). Because `n̂ = −Gφ` is k-local, the paper's dissipation operator on
`δn_e` is this same diagonal on `φ`. `L` is a dense 2×2, so the backend is putzer2
(`s² = ((d_φ − d_ψ)/2)² − kz²v_ph²`), exact for any `ν_φ ≠ ν_ψ` and hoistable; the
separable backend does not apply because the phase `kz·v_ph(k⊥)·τ` is full-grid.
`L(−kx,−kz,ky) = conj L(kx,kz,ky)` holds (`a`, `pe` depend on `k⊥²` only). In 2D there
is no `kz` and `L` is the dissipation diagonal, exactly as for 2D RMHD.

With `ν_φ = ν_ψ` — the paper's and the production case — `s² = −kz²v_ph²` vanishes on
the whole `kz = 0` plane (and the zeroed Nyquist plane), not only at `k⊥ = 0` as in
RMHD. `apply_exp` is safe there (the Taylor branch of `sinh(sτ)/s`); the
`jnp.sqrt(s²)` in `putzer2_precompute` has an infinite derivative at 0, so the tangent
of `lin.s` — and of `apply_exp` — along ANY overridden parameter is non-finite on those
planes: the kgrid is rebuilt under the trace, and `0·∞` is NaN even for a parameter
`s²` does not depend on there (measured 2026-11-11 for every live key — `rho_i`, `tau`,
`Z`, `diss`, `z_diss_k`, `Lx`, `Lz`: exactly the `kz = 0` and Nyquist planes of `lin.s`,
and both fields of `apply_exp` on them). It is not FLR-specific either: RMHD with a
pair-shaped equal `diss` override (the putzer2 backend at `ν = η`) has `s² = −kz²` and
the same non-finite tangent on the same planes (measured). Nothing reachable today hits it — the stability harness's `J v = apply_L(v) +
jvp(N)` and its `_dparam_action` go through `apply_L`, which never reads `s`, so
`dJ/dp` is finite on those modes for every live key (gated explicitly) — but a
parameter derivative taken THROUGH `apply_exp` (the learned-closure step) must fix it
first, in `propagators.py`. Recorded, not fixed.

CB-IMEX is rejected for FLRMHD + `z_spectral` (`flrmhd.check_scheme`, called by the
`run.py` drivers): the standing no-L-stable-solve-on-a-wave-L rule, and this `L` is
stiffer than RMHD's by `v_ph`. In 2D, where `L` is pure dissipation, IMEX is allowed.

### Invariants and their discrete status

Volume averages on the shared `perp_reduce` normalization. Energy and generalized
helicity (the paper's `E = ¼Σ(|k⊥Θ⁺|² + |k⊥Θ⁻|²)`, `H = ¼Σ(|k⊥Θ⁺|² − |k⊥Θ⁻|²)/v_ph`):

```
E = ½ Σ_k [ G·pe·|φ_k|² + k⊥²|ψ_k|² ]
  = ½⟨G|φ|²⟩ + (ρ_s²/2)⟨n̂²⟩ + ½⟨|∇ψ|²⟩          (E×B/ion + electron pressure + magnetic)
  = ½⟨|∇χ|² + |∇ψ|²⟩ = E⁺ + E⁻ ,       E± = ¼⟨|∇Θ±|²⟩
H = Σ_k G·Re(φ_k* ψ_k) = −⟨n̂ψ⟩ = Σ_k (E⁺_k − E⁻_k)/v_ph(k⊥)
```

In Gaussian units `E` is the energy per unit mass in units of `v_A²` (multiply by
`n_0i m_i v_A² = B₀²/4π` for erg/cm³). `ρ_i → 0`: `E → ½⟨|∇φ|² + |∇ψ|²⟩` and
`H → ⟨∇φ·∇ψ⟩ = ⟨u·b⟩`, the RMHD energy and cross-helicity. Per mode
`v_ph|H_k| ≤ E_k`, with equality for a pure `Θ⁺` or `Θ⁻` state: `H = ±E/v_ph`.

**`E⁺` and `E⁻` are NOT separately conserved at finite `ρ_i`.** Per mode
`E⁺_k − E⁻_k = v_ph(k⊥)·H_k`, so

```
E⁺ − E⁻ = Σ_k v_ph(k⊥)·H_k
```

and the conserved quantities are `E = E⁺ + E⁻` and `H = Σ_k H_k` — the UNWEIGHTED sum.
The nonlinearity conserves `Σ_k H_k` while moving `H_k` across `k⊥`, i.e. between
wavenumbers of different `v_ph`, so the `v_ph`-weighted sum `E⁺ − E⁻` changes: a
forward transfer of positive `H` toward `k⊥ρ_i ≳ 1` (larger `v_ph`) raises `E⁺ − E⁻`
and hence, `E` being fixed, converts `E⁻` into `E⁺`. Only at `ρ_i = 0` (`v_ph ≡ 1`,
`E⁺ − E⁻ = H`) are the two Elsasser energies separate invariants, as in RMHD. Measured
on the gate states: `Σ_k T₊` is 7e-4 … 8e-3 of `Σ|T_E|` where `Σ_k T_E` and `Σ_k T_H`
are at round-off. So `E±` have no flux in the strict sense — `Π±` at the last bin edge
is not a residual, and channel-resolved budgets carry an exchange term.

Continuum conservation: `dE/dt = −⟨φ_e ∂ₜn̂⟩ − ⟨∇⊥²ψ ∂ₜψ⟩` and
`dH/dt = −⟨ψ ∂ₜn̂ + n̂ ∂ₜψ⟩` vanish term by term by the cyclic bracket identity
`⟨a{b,c}⟩ = ⟨b{c,a}⟩`, the self-adjointness of `G`, and integration by parts in z.

Discrete status: the 2/3-dealiased bracket is a Galerkin truncation, so both quadratic
invariants are conserved by the semi-discrete nonlinear RHS to round-off (gated at
1e-13 relative to `Σ|T|`), and by the linear operator at `ν = 0` exactly
(`exp(Lτ)` is unitary in the `(χ, ψ)` norm). Time stepping then drifts both; under
`z_spectral` the only drift is the nonlinear term's. What is known about the drift
ORDER: a Runge–Kutta scheme of order `p` conserves a quadratic invariant of a nonlinear
system only to its local order, so the drift over a fixed time is asymptotically
`O(dt^p)`. The extra order sometimes quoted for even `p` (RK4 on an oscillator has
`|R|² = 1 − (ωdt)⁶/72`) is a property of LINEAR skew dynamics only — classical RK4 on
the rigid body converges to order 4. What is measured (F1.7, fp64, dt = 0.01 … 0.00125
at fixed T, E and H, 2D and `z_spectral`): `lsrk33` 2.89–3.10; `rk44` 4.9–5.05 with no
trend over the range tried; `lsrk54` 4.83–5.15 at the gate's dt, falling under
refinement (4.88 → 4.60 in the F1 review's sweep). The order-5 behaviour of the two
fourth-order schemes is therefore pre-asymptotic. An interpretation, not an established
result: at these dt the drift is dominated by the near-linear part of the error, whose
leading term is conservative, and the genuinely nonlinear `O(dt⁴)` term has a small
constant. The gate asserts `p − 0.4 ≤ order ≤ p + 1.4` plus an absolute bound
(relative drift ≤ 1e-6 at the finer dt), not an order of `p + 1`.

Dissipation rates (`h`, `h_z` as above):
`ε_⊥ = Σ k⊥^{2h}[ν_φ G pe|φ|² + ν_ψ k⊥²|ψ|²]`, `ε_z = ν_z Σ kz^{2h_z}·2E_k`,
`ε_H,⊥ = (ν_φ + ν_ψ) Σ k⊥^{2h} H_k`, `ε_H,z = 2ν_z Σ kz^{2h_z} H_k`.

### The ρ_i = 0 branch

`eqpars["rho_i"] = 0` is a supported value and IS RMHD: `flrmhd.factors` branches at
trace time and returns `(k⊥², 1/k⊥², 1, 1, 1, 1)` — `G` and `inv_G` are the kgrid's own
`ksq` and `inv_ksq` arrays — so the recipe is term for term `rmhd.NonlinearTerm` and,
in `(φ, ψ)`, RMHD's `L`. The graphs still differ (multiplies by the unit factors; the
dense putzer2 `L` where RMHD at ν = η selects the separable backend), so agreement with
the RMHD recipe is gated at 1e-12, with the bitwise status measured and recorded in the
plan's landing note rather than assumed. `rho_i` is overridable through the stability
seam only when the static value is > 0.

### Timestep

Linearizing about uniform `u₀`, `b₀`: `ω = k⊥·u₀ ± v_ph(k⊥)(kz + k⊥·b₀)`. A
perturbation is carried along the perturbed field at `v_ph`, not at 1, so the nonlinear
CFL rate is `(|u| + v_ph,max|b⊥|)·k_max`:

```
max_vy = max(|∂ₓφ_e| + v_ph,max|∂ₓψ|) ,   max_vx = max(|∂_yφ_e| + v_ph,max|∂_yψ|)
dt = min( cfl_safety / max(max_vx/dx, max_vy/dy) ,  rmhd._quiescent_dt )
```

`v_ph,max = flrmhd._vph_max(params)` is a setup-time python float: the maximum of
`v_ph` over the 2/3-DEALIASED perpendicular modes (`v_ph` is monotonic, so it is the
value at the dealias edge; `(2.28, 2.74, 6.3, 12.4)` at `k_max ρ_i = (1.7, 2.1, 5, 10)`,
`Z/τ = 2`). `∇φ_e` (not `∇φ`) is what the recipe has and is the conservative choice.
At `ρ_i = 0`, `v_ph,max = 1` and this is `rmhd.set_timestep`. Under `z_spectral` the
linear waves are exact at any dt; a finite-difference-z path would add `v_ph,max/dz`.
`_vph_max` is timestep-only and is NOT part of the overrides seam.

### Forcing in the FLR energy (rung F3)

In `(χ, ψ) = (κφ, ψ)` the FLR energy is RMHD's `½⟨|∇χ|² + |∇ψ|²⟩` and `Θ± = χ ± ψ`.
The O-U envelopes are therefore defined as forcing on `χ` (`forcing_mode="momentum"`) or
on `Θ±` (`"elsasser"`):

```
∂ₜχ ∋ f_χ ,  ∂ₜψ ∋ f_ψ          momentum:  f_χ = s·f_raw ,           f_ψ = 0
                                 elsasser:  f_χ = ½(s₊f₊ + s₋f₋) ,   f_ψ = ½(s₊f₊ − s₋f₋)
∂ₜφ ∋ f_φ = f_χ/κ(k⊥)
```

`flrmhd.ForcingTerm` is literally `rmhd.ForcingTerm` evaluated on the state
`(κφ, ψ)` with its `φ` row divided by `κ`, and `flrmhd.forcing_scale` is
`rmhd._forcing_scale_from(stack([κφ, ψ]), f_raw, …)`: `selfnorm_scale`, `safe_scale`,
the cap, the factor 2 of the elsasser entries and the RNG stream are RMHD's, unchanged.
That is correct because the energy is the same quadratic form in `Θ±`: over one
forward-Euler step

```
ΔE± = dt·½⟨∇Θ±·∇(s±f±)⟩ + dt²·¼⟨|∇(s±f±)|²⟩ = ½(s±P±·dt + ½s±²F₂±·dt²) ,
P± = ⟨∇Θ±·∇f±⟩ ,   F₂± = ⟨|∇f±|²⟩
```

and the scale solves `s±P±dt + ½s±²F₂±dt² = 2ε±dt`, i.e. `ΔE± = ε±dt` exactly and
`ε = ε₊ + ε₋`. In the momentum mode `P = ⟨∇χ·∇f_raw⟩`, `F₂ = ⟨|∇f_raw|²⟩` and
`ΔE = ε·dt`. In terms of the evolved field the same statement reads
`ΔE = dt·Σ G·pe·Re(φ*f_φ) + dt·Σ k⊥²Re(ψ*f_ψ) + dt²·E[f_φ, f_ψ]` — the form the gate
checks, with the weights built independently. At `ρ_i = 0`, `κ = 1` and a forced FLRMHD
run is the forced RMHD run (same seed: `forcing_state` and `forcing_key` bitwise).
In Gaussian units `ε` is a power per unit mass in units of `v_A³/L₀` (`L₀` the length
unit; multiply by `n_0i m_i v_A³/L₀` for erg cm⁻³ s⁻¹).

**Helicity injection.** `H = Σ_k (E⁺_k − E⁻_k)/v_ph(k⊥)`, so the forcing injects
`ε_H = Σ_k (ε₊,k − ε₋,k)/v_ph(k⊥)`. The O-U normalization controls the two channel
TOTALS `ε±`, not their distribution over the shell, so `ε_H` is controlled up to the
shell's spread of `v_ph`. With the shell mean

```
1/v̄ = Σ_shell w_k/v_ph(k) / Σ_shell w_k ,     w_k = yfac·k⊥²
```

(`w_k` is the weight of mode k in the O-U self-energy `F₂`, the envelopes being
statistically uniform over the shell), a requested imbalance `σ_ε = ε_H/ε` is

```
ε± = ε·(1 ± σ_ε·v̄)/2                  flrmhd.elsasser_powers(eps, sigma_eps, params)
```

— the pair to pass as `forcing_power_elsasser`. `|σ_ε|·v̄ ≤ 1` is required (`ε± ≥ 0`;
a ValueError otherwise): the largest imbalance a shell can inject is `1/v̄`, which is 1
only in the RMHD limit. At the boundary itself one channel is zero; the bound is
compared with a few ulp of slack and that channel clamped at 0, because the product
`|σ_ε|·v̄` of a `σ_ε` meant to be `1/v̄` rounds to `1 − 1.1e-16`, `1` or `1 + 2.2e-16`.
The shell the mean runs over is `grids.forcing_shell(params)` — the function
`setup_kgrids` fills `kgrid.fmask` with, on the FIELD-precision `k⊥`, so `v̄` is taken
over the modes that are really forced. That matters at fp32 in a box that is not `2π`:
there the field-precision `k⊥/k_unit` of an on-edge mode can round below the integer
shell edge and the mode is not forced (the float64 shell differs from the solver's in 6
of the 20 box/shell combinations of the gate, with `v̄` off by up to 23%).

Exact when `v_ph` is constant over the shell. Otherwise it is the TIME MEAN of `ε_H`
that is controlled, to within the shell's `v_ph` spread (0.7% at `k_min ρ_i = 0.1`,
`fshell = (1, 2)`; measured mean `ε_H/ε` off the request by 7e-5 in 2D and 7.5e-4 under
`z_spectral` over 200 forcing times) — NOT the instantaneous value. The O-U
normalization fixes each channel's TOTAL per step, cross term plus self term; where in
the shell that total lands is not fixed, and the two terms are distributed differently
over it. On most steps both cross terms are positive and `ε_H/ε` sits near the request.
On the steps where an envelope is anti-correlated with its field — a cross term `s±P±`
NEGATIVE, 13–18% of the steps in the review's runs — the scale is large, the cross and
self terms are separately much larger than their sum, and their different
`1/v_ph`-weightings no longer cancel: those steps carry the largest excursions (rms
0.05–0.11 about 0.6, extremes −0.36 and 1.69 in the F3.4 runs). The mean converges on
the forcing correlation time `forcing_tau` (each envelope decorrelates on it, so the
average over `N` forcing times has `~N^(-1/2)` of the per-step scatter): a diagnostic
`ε_H/ε` needs a window of many `forcing_tau`, and a single step's value means nothing.
`diagnostics.flrmhd.injection` reports the realised value exactly. The momentum mode
forces `Θ⁺` and `Θ⁻` with the SAME envelope and has no control of `ε_H` at all.

Everything else is the RMHD forcing: `forcing_norm_per_step`, the lagged scale and its
restart rule (bitwise at both settings), `forcing_shell_noise`, the `cfl_every > 1`
overshoot from rest. The negative-damping forcing of the paper is not built (plan
rung F3b, deferred).

### Diagnostics, flux and budget conventions (rung F2, `diagnostics/flrmhd.py`)

**`diagnostics.energy` / `perpspec` / `parspec` (the RMHD ones, re-exported at the top
level) are WRONG on an FLRMHD state at `ρ_i > 0`** — they return `½⟨|∇φ|²⟩`,
`½⟨|∇ψ|²⟩`, the RMHD energy of the same arrays, silently: the state has RMHD's shape
and dtype. Use `diagnostics.flrmhd.energy` (which reduces to them at `ρ_i = 0`).

Everything is on the shared `perp_reduce` volume-average normalization (ky-doubling,
`/nz·(nx·ny)²`, `nz²` under `z_spectral`), in code units (`v_A = 1`, lengths in box
units); the weights are `kgrid.eq`'s, so `ρ_i = 0` is RMHD here too. Every function
refuses a kgrid without FLR factors (another equation set's) and one carrying
overrides; a kgrid built from a DIFFERENT FLRMHD `Parameters` (another `ρ_i`, `τ`, `Z`
or box on the same grid shape) cannot be detected and gives silently inconsistent
numbers — pass `setup_kgrids(params)` of the same `params`.

- `energy` → `dict(E_ion = ½⟨G|φ|²⟩, E_e = (ρ_s²/2)⟨n̂²⟩ = ½Σ(pe − 1)G|φ|², E_mag =
  ½⟨|∇ψ|²⟩, E_plus, E_minus = ¼⟨|∇Θ±|²⟩, total)`; `total = E_ion + E_e + E_mag =
  E_plus + E_minus`. `helicity` → `H = Σ G·Re(φ*ψ)`. `imbalance` → `dict(sigma_H =
  H/E, sigma_c = (E⁺ − E⁻)/(E⁺ + E⁻))`: the two agree only when the energy sits at
  `k⊥ρ_i ≪ 1` (`v_ph ≈ 1`); the paper's `σ_c` is the second.
- `perp_spectra` → `(k, E_plus(k), E_minus(k), E_H(k))`, spectral densities on
  `_binned` bins of width `dk = bin_factor·k_unit`: `Σ(E_plus + E_minus)·dk = E` and
  `Σ E_H·dk = H`, with `E_H(k)` binned from the per-mode `(E⁺_k − E⁻_k)/v_ph(k) =
  G·Re(φ*ψ)` — the `1/v_ph` is per MODE, never per bin. The bins run to
  `max(perpspec's k_max, the largest dealiased k⊥)`: identical to `perpspec`'s in a
  square box, extended in a non-square one, where `perpspec`'s range stops short of
  the dealiased band along the longer-wavenumber axis and the sums would not close.
- `kperp_kz_spectrum` → `(k, kz, E(kz, k), D(kz, k))`: energy and dissipation per
  `(|kz|, k⊥ bin)` (NOT divided by bin widths: `ΣE = total`, `ΣD = ε_⊥ + ε_z`, and the
  sum over `kz` of `E` is `(E_plus + E_minus)·dk` of `perp_spectra`), `kz ≥ 0` with the
  `±kz` planes folded. Single process; `dims=2` returns one `kz = 0` row.
  **`z_spectral`-only** (asserted): it reads axis 1 as `kz` and its normalization is the
  `z_spectral` one (`nz²`); a finite-difference-z path (rung F6) must z-transform first.
- `dissipation` → `dict(eps_perp, eps_z, epsH_perp, epsH_z)`, the formulas of
  "Invariants" above with the solver's own `kz` (Nyquist-zeroed, as in `L`).
- `injection` → `dict(eps_plus, eps_minus, eps, eps_H, self_plus, self_minus, self,
  self_H)` from one call of the solver's `flrmhd.ForcingTerm` `F = (f_φ, f_ψ)`: the
  `eps*` are the instantaneous cross terms (`eps_± = ½⟨∇Θ±·∇F_Θ±⟩`, `F_Θ± = κf_φ ±
  f_ψ`; `eps_H = Σ G·Re(φ*f_ψ + ψ*f_φ)`), and the `self*` are the second-order
  coefficients such that the step of length `dt` that starts from this state injects

  ```
  ΔE = eps·dt + self·dt² ,     ΔH = eps_H·dt + self_H·dt²
  ```

  up to the scheme's `O(dt)` relative difference from forward Euler (the self term
  needs the step's `dt`, which a state does not carry). What `self*` is depends on the
  normalization path, because it is the injection's first-order change DURING the step:
  - `forcing_norm_per_step=True` (default): `F` is frozen over the step (the scale read
    is `state.forcing_scale`, the one the NEXT step uses), so `self*` are the quadratic
    forms of `F` — `E±[F]`, `E[F]`, `H[F] = Σ(E⁺_k[F] − E⁻_k[F])/v_ph` — and `ΔE` is
    the per-step target exactly.
  - `forcing_norm_per_step=False`: the scale is re-solved at EVERY stage as
    `target/P(f_stage)`, so the instantaneous power is the target throughout the step
    and the step injects `eps·dt`: **`self* = 0`**. Adding the quadratic form of `F`
    there — the per-step formula — overshoots `ΔE` at first order (measured 2.5e-2,
    1.2e-2, 5.3e-3 at `dt = 0.004, 0.002, 0.001`; the cross term alone is the target
    to round-off). The exception is a channel whose scale sits at its cap
    (`rmhd._scale_epilogue`: `|target/P| > √(2·target/(F₂·dt_q))`, i.e. `P ≈ 0` — a
    start from rest, or an envelope momentarily orthogonal to the field): the capped
    scale does not depend on the state, `F` IS frozen over the step, and that
    channel's quadratic form is its self term (from rest at fixed dt the capped step
    injects `self·dt² = target·dt`). `injection` recognises a capped channel by its
    cross term missing the target (relative `√eps` of the field precision; momentum
    mode has one scale, so both channels together). With that rule the run-integrated
    sum closes to 7.5e-4, 8.8e-5, 1.2e-4 (E, 2D gate case) at those `dt` with 12–16% of the steps
    capped. `ΔH` has no such exact statement in this mode (`eps_H` is not pinned
    during the step): it is first order, 3.2e-3 … 4.4e-4.
- `transfer` → `(k, T_plus(k), T_minus(k), T_H(k))` and `fluxes` → `(K, Pi, Pi_plus,
  Pi_minus, Pi_H)`, from ONE call of the solver's own `flrmhd.NonlinearTerm` `N` (so
  the diagnostic cannot drift from the solver):

  ```
  T±(k) = ½k⊥² Re(Θ±* N_Θ±) ,  N_Θ± = κN_φ ± N_ψ ,   T_E = T₊ + T₋ ,   T_H(k) = (T₊ − T₋)/v_ph(k)
  Π_X(K) = −Σ_{k⊥ < K} T_X(k)
  ```

  `Σ_k T_E = Σ_k T_H = 0` (round-off; `Σ_k T₊` and `Σ_k T₋` are separately nonzero at
  `ρ_i > 0`). **Π > 0 is transfer toward small scales.**
  `T` is returned as a spectral density (like the spectra); `fluxes` returns `K` = the
  UPPER bin edges and `Π(K)` the minus cumulative sum over the modes with `k⊥ < K`
  (bins are `[lo, hi)`), so `Π` at the last edge is the round-off residual of the
  conservation law. Under `z_spectral` at `ν = 0` the linear operator changes no
  `E_k`, `H_k`, so `d/dt Σ_{k<K} E_k = −Π(K)` — the gate, against the stepper.
- **Sub-filter transfer, `transfer(..., kcut=)` / `fluxes(..., kcut=)`** (one extra
  RHS): with `f^<` the state low-passed to `k⊥ < kcut`, `T^sf_X(k) = T_X[f](k) −
  T_X[f^<](k)` for `k⊥ < kcut` and 0 above — the rate at which interactions involving
  at least one sub-filter mode change the retained band's `E±`, `H` at wavenumber k:
  the a-priori target of a closure at filter scale `kcut`. Because the truncated
  system is itself conservative in `E` and `H` (`Σ_k T_X[f^<] = 0` for `X = E, H`),
  `Σ_{k<kcut} T^sf_X = −Π_X(kcut)`: `fluxes(kcut=)` returns the cumulative sub-filter
  flux, which equals the full `Π(kcut)`, `Π_H(kcut)` at and beyond `K = kcut` when
  `kcut` is a bin edge. The retained band is `k⊥ < kcut` STRICTLY, the bins' own
  `[lo, hi)` convention: in a square box at the default `bin_factor = 2` modes sit
  exactly on the edges (`(2,0)`, `(0,4)`, …) and belong to the bin above. NOT per channel: FLR-MHD conserves `E` and `H`, not `E⁺` and
  `E⁻` separately (`H = Σ(E⁺_k − E⁻_k)/v_ph(k)` is not `E⁺ − E⁻`), so `Σ_k T±` does
  not vanish, `Π±` at the last edge is not a round-off residual, and the channel
  sub-filter fluxes differ from the full `Π±(kcut)`; only at `ρ_i = 0` do they agree.
- `budget` → `dict(eps, eps_perp, eps_z, dEdt, eps_H, epsH_perp, epsH_z, dHdt,
  closure_E, closure_H)`: `dEdt = ε − ε_⊥ − ε_z`, `dHdt = ε_H − ε_H,⊥ − ε_H,z`
  (instantaneous; `ε`, `ε_H` are `injection`'s cross terms), all `E` sinks
  non-negative, the `H` rates carrying the sign of `H_k`. `closure_X` is the rate of
  `X` under the solver's own full right-hand side (`apply_L` + `construct_rhs`, forcing
  included) minus `dXdt`: round-off by construction; nonzero means the formulas here
  and the solver's operators disagree. Over a run, `ΔE = Σ(dEdt·dt + self·dt²)` up
  to the scheme's `O(dt)` difference from forward Euler (gated by its order).
- `spectral_break(k, E)` → `dict(k_break, slope_lo, slope_hi, resid)`: least-squares
  continuous two-slope fit of `log E` against `log k` (host numpy), the break scanned
  over a fine grid inside the fit range. The sensitivity of `k*` to the estimator on
  real spectra is not established.

Under `comm_backend="jax"` (not available to FLRMHD until FD-z lands) every z-reducing
call here would need `comms.shard_call`, as the RMHD diagnostics do.

## Linear stability: parameter derivatives (`taranis/overrides.py`, AUTODIFF rung 2)

The harness linearizes `dt f = L(p) f + N(f; p)` and differentiates J with respect to a
continuous parameter p by forward-over-forward jvp through `grids.setup_kgrids(params,
overrides=p)` — so the tangent reaches L (`kgrid.lin`), the wavenumbers N reads, and the
coefficients at the RHS read sites (`cs0`, the FD-z `dz`) in one pass. Two conventions:

- **A box length moves k with the state held in Fourier space.** x₀ is its coefficient
  array; changing Lx rescales every kx = 2π·ikx/Lx and leaves the coefficients alone, i.e.
  x₀ is stretched with the box. That is the derivative of the *family* of states "same
  profile per grid index", not of a fixed physical profile x₀(x): for the latter, add the
  state direction dx₀ = ∂x₀/∂L (a combined `dx0` + `dparams` direction; dJ is linear in the
  direction, so the two parts add).
- **A mode block is an index.** Block iky has ky = 2π·iky/Ly, so at fixed iky
  d/dk = (dLy/dk) d/dLy with Ly = 2π·iky/k: `dparams={"Ly": -Ly/k}`. The dealias-kept set
  is index space (independent of L), so the block's rows do not change along the direction.

Under FD-z, `Lz` enters through the stencil's dz = Lz/nz: the first-derivative part of J
scales as 1/Lz, while the filter `z_diss·(dz/2)⁴·∂_z⁴` is Lz-free (= z_diss·S₄/16), so at
x₀ = 0 dJ/dLz = −(the cross-field ∂_z part of J)/Lz exactly (a gate).

## Reading 2D MHD results

Energy cascades **forward** in 2D MHD — the opposite of 2D hydrodynamics' inverse cascade
— but `⟨ψ²⟩` inverse-cascades regardless. A plateaued energy spectrum is therefore not on
its own evidence of a saturated state; check that `⟨ψ²⟩` has stopped climbing too.

Per the zeroth law of turbulence, and given adequate resolution, `diss` sets the
time-to-saturation and the dissipation-range cutoff — not the saturated amplitude.
