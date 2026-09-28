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

## Rung 2b — new science (each its own notebook; order open)

1. **Viscosity** (the paper's stated future work): γ(Pm, α, k), `diss=(ν, η)`.
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
   reference), then rung 1b sensitivities.
4. Overrides seam + reference gate; rung 2 targets.
5. Rung 2b items; λ₁/m notebook; 3a; 3b.
