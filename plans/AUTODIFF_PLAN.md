# Autodiff plan — linear stability by autodiff: eigenvalues, their sensitivities, and beyond

**Status: DRAFT v2** (v1 2026-08-15; rewritten 2026-09-27 around an eigenvalue harness —
Alfred's decision: the old rung 1, a growth rate fitted from a nonlinear run and
differentiated through the stepper, is REPLACED by the eigenvalue version below).
Each rung that touches `taranis/` lands via the standing flow (implementer subagent →
fresh adversarial review with mutation testing).

**Ordering decision (Alfred, 2026-09-28):** general machinery FIRST (rung 0 + 0b: any
equation set, any equilibrium, a robust eigensolver), validated against EXACT theory —
no-shear tearing with an analytic Δ′ and the analytic inner-layer dispersion relation
(rung 1a) — and only then the shear case against the Julia eigencode (rung 1b-ref) and
the more complex equilibria. The Julia reference (committed 2597c78) waits for that.

## Goal

Show that taranis's autodiff does physics a hand-written code cannot easily do:

1. **Rung 0 — the eigen-harness.** The exact Jacobian of the discrete RHS, J = L + N′(x₀),
   from `jax.jvp` / `jax.jacfwd`, and its spectrum. No linearized code is ever written.
2. **Rung 1 — validation.** Classical tearing, then shear-flow tearing (Mallet, Eriksson,
   Swisdak & Juno, JPP 91, E146 (2025) — "the paper" below) against an independent
   eigencode.
3. **Rung 1b — eigenvalue sensitivities.** dγ/dp from first-order perturbation theory
   (left and right eigenvectors, mixed jvp); marginal curves by Newton on the eigenvalue.
4. **Rung 2 — physics parameters and k** through an overrides seam.
5. **Rung 2b — new science** the paper's 1D eigencode cannot reach.
6. **Rung 3 — beyond the linear phase** (saturated islands; statistical steady states).

## Why forward mode, mostly (decision, unchanged from v1)

Objectives are scalars with O(1)–O(5) parameters; `jax.jvp` costs ~2–3× an RHS and O(1)
memory. `simulate`'s `lax.while_loop` is not reverse-differentiable; `block_of_steps` is a
fixed-length `lax.scan` and is. **The one reverse-mode use** is Jᵀ·u (left eigenvectors,
rung 1b): a `vjp` / `jax.linear_transpose` of ONE RHS evaluation — no trajectory storage.
Reverse mode through time stays rejected (rung 3 notes).

## Core identity (read before implementing anything)

`physics.construct_rhs` returns N(f) ONLY: the k-local linear part lives in `kgrid.lin`
and the steppers apply it (`dt f = L f + N(f)`). So the Jacobian at an equilibrium x₀ is

    J v = kgrid.lin.apply_L(v) + jvp(N, (x₀,), (v,))[1]

— forgetting the first term silently drops all dissipation. Use the operator's
`apply_L` method, never `kgrid.lin`'s arrays (CLAUDE.md rule). Under `z_spectral`,
`apply_L` also carries the ±i·kz Alfvén term, which is correct here (it IS part of J).

For a 1D-in-x equilibrium (every bracket of two functions of x vanishes, so any
φ₀(x), ψ₀(x) is an exact ideal steady state), J is block-diagonal in ky. For ky ≠ 0 the
block acting on one ky column (kx two-sided, inside the 2/3 mask) is **complex-linear** —
no reality constraint couples it to anything — so it is an ordinary complex matrix of
size 2·n_kx,kept. Box: Ly = 2π/k, `ny = 8` (the mode survives the 2/3 cut).
Dealias-masked modes are spurious zero eigenvalues: restrict to the kept set.

Frozen-equilibrium assumption: resistive decay of x₀ is ignored (standard; the paper's
eigencode makes the same assumption). N(x₀) ≠ 0 at O(η) — irrelevant to J.

## Common protocol (all rungs)

- `TARANIS_PRECISION=64`. `dims=2` unless a rung says otherwise. Unforced.
- **FD gate** (acceptance, every rung): J·v against the central difference
  [N(x₀+εv) − N(x₀−εv)]/2ε (+ `apply_L`) at ≥2 ε, showing the expected O(ε²) convergence,
  for random v and for the computed eigenvector.
- Residual gate on every reported eigenpair: ‖Jv − λv‖/(max(|λ|, scale)·‖v‖) printed
  alongside λ (`scale` ≳ ‖J‖₂, `EigResult.scale`; a bare /|λ| is meaningless for the
  ν = 0 null eigenvalue). `shift_invert` raises on a near-singular J − σI and on any
  returned pair failing this — it never returns garbage silently.
- Grid convergence of every reported γ (nx doubling), since the periodic box is uniform.

## Rung 0 — the eigen-harness (repo change: new module + gate tests)

Module `taranis/stability.py` (name open; plain functions, no Parameters mutation, no
new state fields; imports physics, never imported by the solver). Contents:

- `jvp_operator(x0, kgrid, params)` → the matrix-free `v ↦ J v` above (jitted).
- `ky_block_matrix(x0, kgrid, params, iky)` → dense J restricted to one ky column, by
  `jacfwd`/vmapped jvp over the kept-mode basis. Valid only for a ky-independent x₀
  (assert it: x₀'s ky≠0 content is zero).
- `eig_dense(Jblock)` → full spectrum + left/right eigenvectors (numpy/scipy `eig`,
  host side; this is what validates "fastest mode" claims, see rung 1).
- `shift_invert(J, sigma, k)` → a few eigenvalues nearest σ: dense LU for mid-size blocks
  (up to the measured memory envelope), matrix-free GMRES inner solves for 2D equilibria.
- `transpose_operator(...)` → u ↦ Jᴴu via `vjp` (for rung 1b and non-normal checks).

Gates, `tests/test_stability.py` (both bootstrap/footer conventions; fp64):

1. FD gate above, 1D and 2D x₀.
2. **Exact operator, no equilibrium:** x₀ = 0 ⇒ J = L exactly; eigenvalues −η k², −ν k²
   (and hyper variants) to round-off. 3D `z_spectral` too, where L carries ±i·kz: the
   eigenvalues are the damped Alfvén waves ±i kz − ηk² (ν = η) exactly.
3. **Independent Fourier transcription (the differential cross-check):** for
   ψ₀ = A cos(qx), φ₀ = αA cos(qx) (q a box harmonic), the ky-column block of J is
   banded — it couples kx only to kx ± q — with entries written down by hand from the
   linearized RMHD equations in Fourier space (a test-local transcription sharing
   nothing with `NonlinearTerm` but the sign conventions of CLAUDE.md "Test particles").
   Assert the dense `ky_block_matrix` equals it entrywise at round-off (away from the
   dealias edge, where the harness's truncation is the defined behaviour). This is the
   gate with teeth against factor/sign errors in the linearization. (A uniform in-plane
   field is not available: ψ linear in x is not periodic.)
4. `ky_block_matrix` against the full `jvp_operator` on a random ky-column vector.
5. `transpose_operator` against the conjugate transpose of the dense block.
6. The mutation-testing review ([[mutation-testing-gate-suites]]): one-line mutations to
   `NonlinearTerm` / `linear_matrix` signs and factors must each fail a gate.

## Rung 0b — general machinery (before any science)

Rung 0 as first landed is RMHD/ky-block-centric. 0b makes it equation- and
geometry-agnostic, and makes the eigensolver robust:

- `jvp_operator` works for ANY registered recipe (RMHD, GDI, CMHD: same J = apply_L +
  jvp(N) identity) and any x₀ (1D, 2D, 3D; FD-z and `z_spectral`). Real-vs-complex
  linearity stated per case: the full operator on rfft2 state is only R-linear; the
  complex-linear reduction is a property of a symmetry (a ky-independent x₀), never
  assumed.
- **Two eigensolver families, both matrix-free**, chosen per problem:
  (i) **propagator Arnoldi**: Arnoldi on v ↦ e^{JT}v, where e^{JT} is integrated with the
  solver's own IF steppers on the linearized system (L exact through `apply_exp`, N′ by
  jvp) — the largest-|μ| eigenvalues ARE the fastest-growing modes, stiffness never
  enters, needs no linear solve; cost ~ (Krylov dim) × (T/dt) linear steps, measured;
  (ii) **shift-invert** (dense LU where the block fits, GMRES otherwise, with (L−σ)⁻¹ —
  diagonal in k — as the preconditioner). Every solver has maxiter/tol and RAISES on
  non-convergence (rung 0's first shift-invert hung for 7h47m at n≈5000 — never again).
- The dense ky-block path stays as the fast special case for 1D equilibria.
- **Generality gates with exact answers, one per equation set:** RMHD tearing (rung 1a,
  below); CMHD about a uniform state (the fast/slow/Alfvén dispersion relation the CMHD
  linear gates already use); GDI's analytic linear growth (GDI_PLAN theory). Each run
  through the matrix-free path, not a hand-built matrix.

**Rung 0b landed (7b83f9d + review fixes 87eb72a, 2026-09-28).** Measured outcomes that
change later rungs: the ky-averaged preconditioner is exact for 1D x₀ but GMRES fails
near the slow-mode cluster once a y-perturbation ε ≳ 0.1–0.3 (cos x sheet, 32–64×16), and
(L−σ)⁻¹ never helps at ν = 0 — so **propagator Arnoldi is the 2D-equilibrium workhorse**
(rung 2b item 3), shift-invert the 1D one. Propagator error is O(dt^p) and exactly
independent of T; its default residual_tol is 1e-4. Uniform dense grid envelope on the
laptop: nx ≤ 2048 (n ≈ 2500, ~75 s eig); S=1e5 at ka=0.1 needs nx ≈ 2e4 → matrix-free.

## Rung 1 — validation (notebooks, no repo changes beyond rung 0/0b)

### 1a. No-shear tearing against EXACT theory

Equilibrium Ψ₀ = sech²x (f = −2 tanh sech², the paper's profile at α = 0), localized so
the periodic box is harmless for Lx ≫ 1/k. Its outer equation
ψ″ = (k² + 4 − 12 sech²x)ψ is Pöschl–Teller ℓ = 3, solved in closed form by ladder
operators (ψ = A₃A₂A₁e^{−κx}, Aℓ = −d/dx + ℓ tanh x, κ² = k² + 4; checked 2026-09-28,
ODE residual 4e-15):

    Δ′a = 2(5 − k²a²)(k²a² + 3) / (k²a² √(k²a² + 4))     → 15/(ka)² + 1/8 as ka → 0,

marginal at exactly ka = √5. Note f′(0) = −2 (not 1): every layer formula uses |f′(0)|.
For the finite box, also derive the periodic-image correction or show it is below the
tolerance at the chosen Lx.

Targets (ν = 0, `hyper=1`):
1. The **analytic inner-layer dispersion relation** (Coppi et al. 1976 / Ara et al.
   1978 form, Γ-function ratio in Λ = γ/(k|f′(0)|)^{2/3}η^{1/3}): the implementer
   transcribes it from the literature WITH citation and verifies numerically that it
   reproduces both limits — FKR γ = [Γ(1/4)/(2πΓ(3/4))]^{4/5} Δ′^{4/5} η^{3/5}
   (k|f′(0)|)^{2/5} and Coppi γ = η^{1/3}(k|f′(0)|)^{2/3} — before any comparison.
2. γ_eig(k, S) → γ_theory(k, S) as S grows: the RATIO → 1 with a measured finite-S
   correction exponent (it is an asymptotic theory — agreement is a trend, not a
   tolerance at fixed S). Report where the uniform grid runs out.
3. γ_max ∝ S^{−1/2}, k_max ∝ S^{−1/4}, with the prefactors the dispersion relation gives.
4. The ideal marginal point: Re λ changes sign near ka = √5 with a finite-S shift that
   shrinks with S.
5. Cross-check γ(η), γ(k) against `examples/tearing-mode-2D.ipynb` /
   `tearing-growth-vs-k.ipynb` (time-domain measurements of the same solver, their
   equilibrium) — the eigenvalue must agree to their fit error.
Also the Harris-like periodic family B_y = tanh(sin x/a)/tanh(1/a) (locally Harris,
Δ′a = 2(1/ka − ka)) as a second exact-Δ′ case (two sheets per period: double tearing at
low k — report the mode pair).

### 1b. Shear-flow tearing against the paper's eigencode

Setup exactly as the paper §7: a = v_Ay = 1, ν = 0 (`diss=(0, η)`, `hyper=1`),
Ψ₀ = sech²x (so f = ∂xΨ₀ = −2 tanh sech², n = 2, Δ′a ≈ 15/(ka)²), Φ₀ = αΨ₀, S = 1/η.
Also f = tanh (n = 1) via the periodic family above.

**Reference: Alfred's Julia eigencode**
`/Users/alfy/Documents/current_projects/sheartearing/sheet_instabilities.jl`
(Alfred: "several years old, I don't think it is very good"). What it is (read
2026-09-27): 2nd-order 3-point FD on a geometrically stretched grid (`vargrid(dxmin,
xlim, eps)`), outer boundary a Robin condition imitating e^{−K|x|}, generalized problem
A v = γ B v in (ψ, ∇²φ), solved by shift-invert Arnoldi (KrylovKit) about a theory
`goodguess`, selecting `:SR` of 1/(γ−σ) — i.e. **the eigenvalue just below the guess,
not provably the fastest mode**. `scan_ar` and `scan_S` are broken (undefined `ik`,
`whichf`, wrong arity); `eigenmode`, `scan_K`, `scan_K_ar`, `scan_S_K` look usable.
Its strength is the stretched grid: S to 1e16 cheaply, which a uniform periodic grid
cannot match.

Protocol:
- Vendor a trimmed copy (eigenmode + vargrid + f/ddf + goodguess, provenance header) as
  `tests/reference/shear_tearing_eigen.jl`; a generator `tests/_gen_shear_tearing_reference.py`
  drives julia and writes `tests/data/shear_tearing_reference.npz` (force-added), with
  the Julia code's OWN convergence study (dxmin, eps, xlim halved/doubled) recorded
  alongside each point. Never regenerate to make a comparison pass.
- Overlap window (the uniform-grid envelope is MEASURED in rung 0, not assumed; first
  guess): S ∈ {1e3, 1e4, 1e5}, ka ∈ [0.1, 1], α ∈ {0, 0.3, 0.6, 0.8, 0.95}. Box
  Lx ≥ 15/k so periodic images move Δ′ by < e^{−15}; nx resolving δ_in (paper eq 5.15)
  by ≥ 10 points.
- Acceptance: agreement to the larger of the two codes' measured convergence errors.
  Where they disagree, the convergence studies decide which is wrong. The full dense
  spectrum settles whether the Julia `:SR`-below-guess selection ever missed the
  fastest mode — report it either way.
- Science checks inside the window: the paper's figs 2 and 3 trends (γ suppressed with α,
  k_max increasing with α), eigenfunction δ_in (paper's Ψ″ = max/4 width) vs eq 5.15.
  The asymptotic S = 1e12 scalings are the eigencode's job, not ours.

Deliverable: `examples/tearing-eigen.ipynb` (1a + 1b).

**Outcome (2026-09-29): `examples/tearing-shear-eigen.ipynb` +
`examples/tearing_shear_run.py`, gate `tests/test_tearing_shear.py`.** Same pencil as the Julia
code, term for term; the spectrum is even in α (2e-14). sech²: 90/90 points agree (rel.diff
5e-8–1.1e-4, ≤0.21 of the reference's bar). tanh: the plan's Harris-like family carries an
O(a²) shape error in Δ′ (up to 1.6e-2) that shows in γ, so it was replaced by an exact-tanh
sheet PAIR compared by its mean (image coupling antisymmetric → mean exact to 1e-11); 30/30
unstable points agree (0.00–0.42 bars), 6 stable points have no growing mode. Where the codes
differ, the Julia x_lim = 10 outer box is the error (re-running it at x_lim = 40 moves it onto
taranis). Dense spectra: the `:SR` selection never missed the fastest mode in the window. The
cost of tanh at α = 0.8 is set by an ideal Doppler-shifted Alfvén resonance, not the resistive
layer (sech² converges at far fewer points per resonance distance — cause open): it needs nx = 65536 at S ≥ 1e4 in a reduced box Lx = max(20/k, 40). Review 2026-09-29
found the S = 1e4, α = 0.8 tanh rows under-resolved at the old 16384 cap; rerun, now converged
to ≤2e-8. Open: S = 1e5, ka = 0.7, α = 0.8 is not ladder-converged at 65536 (last change 2.6e-4).
Paper trends hold: γ_max(α)/γ_max(0) = 0.955, 0.80, 0.58, 0.28 (α = 0.3–0.95), k_max grows with α.

## Rung 1b — eigenvalue sensitivities and marginal curves

For a simple eigenvalue λ with right v and left w (Jᴴw = λ̄w):

    dλ/dp = wᴴ (∂J/∂p) v / (wᴴ v)

(∂J/∂p)·v is a mixed second derivative: jvp in p of the jvp in x. Any parameter that
enters through x₀ is free — for shear tearing that is **α** (tangent ∂x₀/∂α = (Ψ₀, 0)),
and the sheet width a. Targets:

- **Running exponents** d lnγ / d ln(1−α²) against the paper's 1/2 (constant-Ψ),
  2/3 (nonconstant-Ψ), and 4/7 at the maximum (fig 3), as smooth curves instead of fitted
  slopes. FD-gated (dγ/dα vs [γ(α+h) − γ(α−h)]/2h).
- **Marginal curve**: Newton on Re λ(a) = 0 at fixed k (second derivative by
  forward-over-forward) for the classical family — must land on ka = 1 (Δ′ = 0 for
  Harris). The headline validation of the sensitivity machinery.
- Non-normality diagnostic: |wᴴv| (the eigenvalue condition number) vs α — shear makes J
  strongly non-normal; report where the eigenvalue becomes ill-conditioned.

**Rung 1b machinery landed (2026-09-28).**
`stability.djvp_operator` / `dj_operator` / `dj_matrix` (dJ/ds along dx₀, forward-over-forward,
fields / mode block / real coords; block invariance of dx₀ asserted), `left_eigenvector`
(shift_invert on Jᴴ, match-checked), `eigenvalue_sensitivity` (dλ/ds, κ, refusal when
κ·δ/gap > 1e-3 or residuals > 1e-8). Gates `tests/test_stability_sensitivity.py`: dJ vs FD
exact for RMHD/GDI, O(h²) for CMHD; sech² shear tearing (α = 0.8, 48 points, S = 50) dγ/dα =
−0.2391 vs eigenvalue FD at O(h²) (err 1.4e-7 at h = 1e-3), κ = 3.7 — mild non-normality at
this S, but the right-vector-only formula is already 0.21 off; CMHD uniform state dλ along
dB₀, du₀, dρ₀ against the hand-differentiated closed form to 5e-15. Findings: the default
left-solve offset had to be 1e-3·‖J‖ (at 1e-6 unpreconditioned GMRES stalls above
rtol 1e-12); a real ky-block eigenvalue is exactly double in real coordinates, so
sensitivities of 1D equilibria belong in the ky block.

**Rung 1b science (2026-09-29): `examples/tearing-sensitivity.ipynb` +
`examples/tearing_sensitivity_run.py`, gate `tests/test_tearing_sensitivity.py`.** dγ/dα vs
eigenvalue FD at O(h²) (orders 1.83–2.007) up to S = 1e5, α = 0.99. Running exponents
d lnγ/d ln(1−α²): constant-ψ reaches 1/2 (0.496 at S = 1e6); nonconstant-ψ (ka = 0.1) plateaus
at 0.51, 0.61, 0.646 (the last not grid-checked) for S = 1e3, 1e4, 1e5, rising toward 2/3; every curve leaves its asymptote
once δ_in/a ≳ 0.1–0.2 (S ≤ 1e4; at S ≥ 1e5 the curves are cut by refusals first). Ridge: p_ridge crosses 4/7 and peaks at ≈0.59, falling with S (consistent
with 4/7 from above); integrated it reproduces 1b-ref's γ_max ratios to ≤3e-3. Marginal curve:
Newton in the sheet width is LINEAR (ratio 0.750) because the root has multiplicity 4
(γ ∝ SΔ′⁴ in the suppression band); with the measured multiplicity it lands on ka* = √5
(S = 1e3; to a few ×1e-6 — the last digits are estimator-dependent), so the plan's "second derivative by forward-over-forward" was not needed. κ runs from
~10 to >1e6 as α → 1 and S grows; the right-vector-only formula is off by 10–70× at ka = 0.1 and
wrong in sign at small α for ka ≥ 1. Limit: the harness refuses (κδ/gap > 1e-3) at ka ≥ 1,
1−α² ≲ 4e-3, S = 1e5 — the inner GMRES stalls at δ ~ 1e-11, not the formula.

## Rung 2 — physics parameters and k via an overrides seam (unchanged design + k)

`Parameters` stays static; an optional `overrides` pytree (fixed structure, traced values)
reaches the read sites through one `getp(params, overrides, key)` helper. `overrides=None`
is a Python branch — the compiled graph must be literally today's (standing refactor
reference gate). `params.save` stamps active overrides. Static/structural parameters are
excluded by construction.

New in v2: **k as an override.** k enters through `kgrid.ky` (a pytree array built from
the static `Ly`); the seam must let `setup_kgrids`-equivalent arrays be rebuilt from a
traced Ly for the stability harness only (`ksq`, `inv_ksq`, the `lin` operator). The
implementer decides between a traced-Ly kgrid builder and a documented notebook-local
construction; the solver path is untouched either way.

Targets: dγ/dη → the local exponent dlnγ/dlnS along the dispersion curve (FKR 3/5 →
Coppi 1/3 crossover without a sweep); **k_max by Newton on ∂γ/∂k = 0** — smooth
k_max(α), k_max(S) curves replacing the paper's grid-scanned (jagged) figs 3–4, and the
S^{−3/7}, S^{−1/7} scalings (eq 8.1, 5.13) inside our S window.

**Rung 2 machinery landed (2026-09-29); the science targets are NEXT.** Decision on k: a
traced-Ly kgrid builder — `grids.setup_kgrids(params, overrides=...)` is still the only
kgrid constructor and takes traced values; no notebook-local construction. `taranis/overrides.py`
(key table per eqtype, `getp`, `validate`, `to_record`), `K_Grids.overrides`, read sites in
rmhd (`linear_matrix`, `FDLinearTerm` via `z_derivatives(dz=)`), gdi (`linear_matrix`), cmhd
(`linear_matrix`, `cs0` in `NonlinearTerm`); `propagators.build` skips the Hermitian VALUE
checks on a traced L; `run.py` rejects an overridden kgrid; `Parameters.save(overrides=)` /
`load_overrides` / `from_snapshot` warning; `overrides=` on every stability operator and
solver, `dparams=` on `djvp_operator`/`dj_operator`/`dj_matrix`. d/dk of block iky =
`dparams={"Ly": -Ly/k}`. Gates `tests/test_stability_overrides.py` (details in CLAUDE.md):
J at the static point is bitwise J(None) everywhere except FD-z (1.5e-16: python-float vs
XLA dz arithmetic); J at a moved point == J of a `Parameters` built there (0.0 at fp64);
dJ/dp vs FD for every live key in RMHD 2D (block + real coords), 3D FD-z, z_spectral
separable and putzer2, GDI 2D/3D, CMHD rho (γ=1) / lnrho (γ=5/3) — order 2.00 or round-off
for affine keys; exact x₀ = 0 derivatives of L; dλ/dp against closed-form dispersion
relations (Alfvén, GDI eq 3.7, CMHD fast/slow/Alfvén) to ≤1.4e-9; tearing (S = 50, α = 0.8)
dγ/dη = 1.910, dγ/dν = −2.176, dγ/dk = 0.0147 vs eigenvalue FD at O(h²). Mutation-tested
(record in the implementer's report). Refactor reference and gate 6 unchanged (None graph).

**Rung 2 science (2026-10-01): `examples/tearing-kmax.ipynb` + `examples/tearing_rung2_run.py`,
gate `tests/test_tearing_rung2.py`** (13 tests; kills all 54 one-line mutations of the run
script). Campaign 1159 cached solves, ~2.5 h on the M1 (one restart after the continuation fix
4e181a0: seeds held to [k*/2, 2k*], NEWTON_MAXIT 16). ∂_η, ∂_k vs eigenvalue FD at order 2.000
(4 points, κ ≤ 1e3). q_S(k) = d lnγ/d lnS along the dispersion curve, one sensitivity per point:
at α = 0 it converges onto the exact relation's analytic exponent (gap ≤ 0.005 for ka ≤ 1 at
S = 1e5); with shear the large-k plateau moves from FKR −3/5 toward the shear constant-ψ −1/2
(−0.522 at α = 0.8, S = 1e5), ordered by λ/α — q_S at fixed k discriminates the regime.
k_max by Newton (secant curvature, 3–6 solves; 2n_x moves k* ≤ 6e-9): smooth k_max(α), γ_max(α)
matching 1b-ref's peak fits and 1b's ridge fits; d lnγ*/d ln(1−α²) crosses 4/7 between α = 0.9
and 0.95 and reaches 0.59 (S = 1e5, α = 0.995). k_max(S) running slopes at S = 1e5: −0.1428 /
−0.4248 (α = 0), −0.1414 / −0.4234 (α = 0.8) — the same −1/7, −3/7 with and without shear, as
the paper says. **Finding:** a second, low-k maximum (ka ≈ 0.13–0.37, δ_in/a ≈ 0.3–0.65,
d lnγ*/d lnS ≈ −0.7 → −1.0, γ* rising again as α → 1) dominates at α ≳ 0.95 and low S
(S = 1e3; S ≤ 1e4 at α = 0.995 — at 2e4 the chain sat on a local maximum, corrected by the
low-k review), so k_max(α) and k_max(S) jump between branches where both exist (S ≳ 5e3; at
S ≤ 2e3 one maximum migrates). High-S phase (α = 0, S = 2e5–1e6, 62 min): slopes −0.14302 / −0.42744 at
S = 1e6, γ*/γ_th = 0.9978; the k slope's 1.7e-4 overshoot of −1/7 is the decaying finite-S
correction to k* (k*/k_th − 1: 1.2e-3 → 5.6e-4 over 1e5–1e6), not a different exponent.

**Low-k branch identified (2026-10-02): `examples/tearing-lowk-branch.ipynb` +
`examples/tearing_lowk_run.py`, gate `tests/test_tearing_lowk.py`** (27 tests; implementer
76/77 mutations killed, #21 equivalent; fresh reviewer 36/36 after 8 new gates). Not a box/grid
artifact (S = 1e4, α = 0.995, ka = 0.179: γ = 3.376429e-3; Julia eigencode 3.376433e-3). It is the
α → 1 end of the tearing branch: at α = 1 (Z⁻ = 0) a real, Δ′-driven mode (cutoff ka = √5)
survives only for ν ≠ η — with ν = η, ‖δz⁻‖ decays, so no EIGENMODE grows (a modal statement,
not an energy bound; δz⁺ can grow transiently) — with γ* ∝ S^−0.66, k* ∝ S^−0.174 at ν = 0. For
α > 1 it becomes ideal KH (q_S → 0 at α = 1.02, 1.1). This contradicts, for the finite-S
eigenproblem, the paper's "completely stabilized" as the flow becomes Alfvénic. Dominance window
(two maxima of equal height, box-corrected): 1 − α_c² = 2.84e-2, 1.67e-2, 9.91e-3, 5.01e-3,
3.01e-3 at S = 5e3–1e5, ≈ 16.6 S^−0.749 (local slopes still drifting); α_c reproduced exactly by
an independent dense route. Needs Pm ≪ 1 (survives Pm = 0.1, gone at Pm = 1; one point).
Open: theory for the α = 1 exponents; α_c beyond S = 1e5; α_c(Pm).

## Rung 2b — new science (each its own notebook; order open)

1. **Viscosity** (the paper's stated future work): γ(Pm, α, k), `diss=(ν, η)`.
   **DONE 2026-10-02: `examples/tearing-viscosity.ipynb` + `examples/tearing_viscosity_run.py`,
   gate `tests/test_tearing_viscosity.py`** (19 tests; implementer 82/82 mutations killed after 5
   new gates, fresh reviewer 34/36 + 2 equivalent after 8 new gates). Exact α = 0 visco-resistive
   layer relation Δ′δ_η = F(Λ, Pm) (6th-order BVP, Richardson in s_max); limits B&L (Pm → 0) and
   Pm^−1/6 F_v (Pm → ∞), I_V = π6^−1/3 Γ(5/6)/Γ(7/6) = 2.10359 (derived twice), C_VC = 1.52554;
   n = 2 asymptotes k_max ∝ S^−1/7 Pm^1/14, γ_max ∝ S^−3/7 Pm^−2/7. Eigenvalues converge onto the
   relation (Pm = 100, S = 1e5: d ln k*/d ln Pm 0.0609, d ln γ*/d ln Pm −0.282 vs 0.0625/−0.278).
   **Shear destabilises at large Pm**: S = 1e4, Pm = 100, γ*(α)/γ*(0) = 1.72, 2.45, 1.88 at
   α = 0.3, 0.6, 0.8 (2.449 confirmed by an independent dense route, grid/box/global/parity
   checked); the small-α coefficient c₂ changes sign between Pm = 3 and 30 and grows roughly like
   −Pm. A viscous stability boundary below α = 1 (bisection): 1 − α_c² = 0.0592 / 0.0619 at
   ν = 1e-2, 0.466 / 0.468 at ν = 0.1 — depends mainly on ν in this window. Open: a layer theory
   with both shear and viscosity; the weak complex pairs at ka ≤ 0.1, Pm ≥ 10.
   **Ofman, Chen, Morrison & Steinolfson 1991 (Phys. Fluids B 3, 1364) comparison — DONE 2026-10-02:
   section 7 of `examples/tearing-viscosity.ipynb` + `examples/tearing_ofman_run.py`, gate
   `tests/test_tearing_ofman.py`** (11 tests; implementer 60/60 one-line mutations killed, 5 of them after new gates;
   promoted from the untracked scratch study in `examples/data/tearing-ofman/scratch-2026-10-02/`). Their setup:
   Harris sheet (our exact tanh pair, Lx = 40), S = 1e6, ka = 0.5, S_v = S/Pm; flows G = V tanh(Rx) (R = 1,
   V = α is our u₀ = αb₀) and the jet V[sech(Rx) − 1]. One Parameters per (nx, Lx), diss and Ly through the
   overrides seam (gated against directly built Parameters). Per point: shift-invert ladder nx = 8192–131072
   (262144 for the three slowest inviscid points), stopped at 1e-6. Viscous points converge to ≤ 2e-9; inviscid
   shear to ≤ 4e-7, except α = 0.4 (1.2e-5), α = 0.8 (5.1e-5) and R = 0.73 (5.9e-6). The faster pair member is
   reported: the even mode everywhere except the R = 6 jet at S_v ≤ 1e3. The pair split is ~5e-4.
   ~3.9 h of solves (3 h wall, two to three processes).
   - No flow (fig 7): γ = 2.4818e-4 (Pm 0), 2.1320e-4 (0.01), 1.5926e-4 (0.1), 1.0964e-4 (1), 7.3911e-5 (10),
     4.9371e-5 (100), 3.2640e-5 (1e3), 2.1249e-5 (1e4). The exact layer relation (Harris Δ′a = 3, f′ = 1) is
     0.6–8.5% above; the gap grows with Pm as the outer region turns viscous. Least-squares slope over their
     S_v = 1e2–1e7: 0.1751 (theirs 0.168; the layer relation's own is 0.1686). The local slope runs 0.186 → 0.127,
     crossing 1/6 near S_v ~ 1e6.
   - Field-aligned shear, γ(α)/γ(0) at α = 0.2/0.4/0.6/0.8. Pm = 1: 0.9860/0.9417/0.8573/0.7020, so eq. 13c
     ((1−α²)^1/3) holds to ≤ 1.3%. **Pm = 100: 1.386/2.379/3.603/4.445 vs 13c's 0.880/0.724/0.589/0.445 — 13c
     has the wrong sign at large Pm.** The effect starts where 13c's Pm-factor does (α²Pm ~ 4). The 3.6029 is
     reproduced by a dense eigvals route at nx = 4096 (to 3e-8) and at Lx = 60 (3.602951). This is the
     fixed-k Harris confirmation of the sech² k_max destabilisation. Pm = 0: 1.217/1.398/1.419/1.201; C&M 1989
     (JPP eq. 1.6 as printed) is 0.2–20% high. Its k-dependence fails a second k: at ka = 0.25 γ(0.25)/γ(0.5) =
     1.372, against 1.581 as printed and 1.118 with (ka)^1/2.
   - Non-aligned tanh flow, V = 1: inviscid (fig 6) γ = 2.725/4.191/5.765/6.757/5.508e-4 at R = 0.1/0.3/0.5/
     0.73/0.9, peaking at 0.73 as theirs. S-exponents at R = 0.73 (fig 5a): −0.5002, −0.5245. Pm = 1 (fig 8):
     γ(Q) has an effective exponent of 0.30 over Q = 0.02–0.14 and turns over at Q ≈ 0.2. So their Q^1/3 holds
     roughly on their window (the scratch's "not reproduced" was too strong); exact Q^1/3 (0.345) is found along
     R = 1.
   - Jet, sheet frame (the box mean restored as a Doppler shift, −ik⟨u⟩; the scratch had left it in, Im ≈ −0.32,
     and its inviscid R = 0.44 point failed). R = 0.44: γ = 2.831e-4 + 1.8e-6i inviscid, converged to 2e-9.
     S_v slopes 0.175/0.171, the same as no flow. R = 2.5 / 6: 4.525e-3 / 1.959e-2 (match their plot); their
     plotted Im parts are a few times smaller than ours (unresolved). R = 6 vs S_v: 1.63e-2 (1e4), 4.06e-3 (1e3,
     odd member) — no stabilisation by 1e3. At S_v = 1e2 an oscillating mode 2.71e-2 + 0.780i dominates. It is
     box- and grid-independent, growing for ν ≥ 1e-2 and damped at ν ≤ 3e-3: viscously destabilised and
     unidentified.
   - Open: a theory for the Pm ≫ 1 shear destabilisation; the S_v = 1e2 jet mode; Chen & Morrison 1990b's
     derivation of 13c (not checked). Our JPP paper cites neither this paper nor Chen & Morrison 1990b (PoF B 2,
     2575; 13c's source).
2. **Non-proportional profiles**: u₀(x) ≠ αb₀(x) (e.g. different widths, offset shear
   layer). Still 1D, still the dense per-ky path.
3. **2D equilibria** — where the harness beats any 1D code. ky blocks couple; matrix-free
   shift-invert Arnoldi at ~1e5 unknowns, preconditioned by the dense per-ky blocks of
   the y-averaged equilibrium. Candidates: Fadeev/Kelvin–Stuart island chain
   ψ₀ = ln(cosh y + ε cos x) (coalescence instability; J₀ = F(ψ₀) exactly steady), finite
   sheets with ends, shear + field-aligned flow (φ₀ = G(ψ₀) is steady).
4. **3D oblique tearing with shear** (FD-z or `z_spectral`): resonance
   k_y B_y(x) + k_z = 0 at k-dependent surfaces; per-(ky,kz) blocks for 1D equilibria.

## Rung 3a — saturated island amplitude (unchanged from v1)

Converged tangent through `block_of_steps` past saturation (dx*/dp), or implicit
differentiation at the fixed point with GMRES on `jvp_operator` (rung 0 provides it).
Validation: W_sat ∝ Δ′. Frozen-equilibrium caveat: needs an equilibrium-sustaining source
term (E₀ = ηJ₀ as a `physics.Term` with an `active` predicate) for anything run to
saturation — design it here, not before.

## Rung 3b — statistical steady state, and the paper's turbulence claim

Step zero (v1): λ₁ by Benettin renormalization of a state tangent through
`block_of_steps`; count positive exponents m. Then the v1 options (ensemble of
short-window tangents; forward NILSS; FDT cross-check) — unchanged, reading list below.

**Science target (new in v2):** the paper's §8 argument that imbalance puts α near 1
(1−α² ~ δz⁻/δz⁺) so γ_tr τ_nl ∝ (1−α²)^{−1/2}. Take current sheets from imbalanced RMHD
turbulence (elsasser forcing with unequal `forcing_power_elsasser`), linearize around
them (finite-time tangent growth — the base is not steady), and measure tearing onset
against the local α. No 1D eigencode can do this.

**State of the art (checked 2026-08-15):** iGENE (Phys. Plasmas 33, 083901 (2026),
arXiv:2605.03086) reverse-differentiates time-averaged GK fluxes with truncated windows
(15–50% of FD, "directionally correct"); gyaradax (arXiv:2604.06085). The gap is a
controlled statistical-sensitivity estimator in plasma turbulence.

**Reading list (Alfred) before committing to a 3b approach:**

- [ ] Lea, Allen & Haine, Tellus A 52, 523 (2000).
- [ ] Ruelle, Commun. Math. Phys. 187, 227 (1997); Nonlinearity 22, 855 (2009).
- [ ] Eyink, Haine & Lea, Nonlinearity 17, 1867 (2004).
- [ ] Wang, Hu & Blonigan, JCP 267, 210 (2014); Ni & Wang, JCP 347, 56 (2017).
- [ ] Baladi, ICM proceedings (2014).
- [ ] iGENE (arXiv:2605.03086); gyaradax (arXiv:2604.06085).

## Order of work

1. Rung 0 module + gates (implementer), adversarial review with mutations.
2. Julia reference generator + convergence study — DONE (2597c78; 126 points, all
   converged; finding: at S=1e12, ka ≲ 1e-2 the stretched-grid code has a spurious
   eps-proportional mode above tearing — irrelevant in our window, never read a dense
   spectrum of it as "fastest mode" there).
3. Rung 0b (general machinery + generality gates), review.
4. Rung 1a notebook (no-shear tearing vs exact theory), then 1b (shear vs Julia
   reference), then rung 1b sensitivities — DONE 2026-09-29.
4. Overrides seam + reference gate — machinery DONE 2026-09-29; rung 2 science DONE 2026-10-01.
5. Rung 2b items (item 1 viscosity DONE 2026-10-02; 2–4 next); λ₁/m notebook; 3a; 3b.
