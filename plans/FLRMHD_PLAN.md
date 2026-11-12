# FLRMHD_PLAN — FLR-MHD (helicity-barrier physics) as a new equation set

Status: **F0–F3 implemented, uncommitted (2026-11-12)** — F0/F1 landed 2026-11-11 and
passed an adversarial review "with required fixes"; those fixes and rungs F3 (forcing) and
F2 (diagnostics) landed 2026-11-12, were reviewed, and their review fixes landed the same
day. **F7: the run script and notebook exist and pass a smoke run; NO production run has
been made and no F7 criterion evaluated.** Landing notes with measured numbers and
deviations: §6. F3b, F4, F5, F6 are deferred. Where a landing note and an earlier section
disagree, the landing note is what was built. The draft was written by a planning subagent for session review; every
point the author could not verify is tagged **[UNVERIFIED]** or **[DECISION FOR ALFRED]**
and collected in §12. Tags used throughout: **[PAPER]** = read from the arXiv source of
Meyrand, Squire, Schekochihin & Dorland 2021 (JPP 87, 535870301; arXiv:2009.02828, TeX
source `HelicityPlugs.tex`), **[DERIVED]** = derived here from [PAPER] equations,
**[CHECKED]** = additionally confirmed by the scratch numerical check of §2.8.

Execution follows the house flow (CMHD_PLAN §9, AUTODIFF_PLAN "Common protocol"): one
implementer per rung, a FRESH adversarial reviewer with its own mutations before a rung
is recorded as landed, docs changed first ("change the docs first, then the module"),
dated landing notes appended to §6 of this file.

## 1. Motivation and scope

FLR-MHD is the minimal fluid model that is asymptotically valid on both sides of
k_perp rho_i ~ 1 for low-beta Alfvenic turbulence, and the model in which the helicity
barrier was found. It is step 1 of a three-step programme:

1. **(this plan)** FLR-MHD in taranis, with the diagnostics (energy/helicity fluxes,
   generalized Elsasser spectra, perpendicular and parallel dissipation) needed to
   reproduce the barrier and to serve as truth data;
2. RMHD + a hand-built helicity-conserving cutoff closure (§11 sketches what it needs);
3. a learned closure trained through the differentiable solver (§11).

Scope of step 1, fixed here:

- New `eqtype="FLRMHD"`, `nfields=2`, evolved fields `(phi, psi)` — the SAME variables
  and bracket conventions as the RMHD recipe (§3.1, §3.2).
- `dims==2` and `dims==3 + z_spectral=True` (single process) in the core rungs F1–F4;
  FD-z (and with it mpi4jax and the `"jax"` multi-GPU backend) is rung F6, gated on need.
- O-U forcing through the existing machinery, generalized to the FLR energy (rung F3).
- Test particles: rejected (already: `Parameters._validate_compat` requires
  `eqtype=="RMHD"`, and `particles/fields.py:56` asserts it). No change.
- `rho_i = 0` is a supported value and IS RMHD (trace-time branch), so the paper's
  "imbalanced RMHD" control runs through the same recipe, forcing and diagnostics.
- Existing RMHD/GDI/CMHD paths stay byte-identical (§3.1, §8).

## 2. The model

### 2.1 Equations in Gaussian units [PAPER]

Symbols as in the paper: `n_0a`, `T_0a` background density/temperature of species a,
`tau = T_0i/T_0e`, `q_i = Ze`, `Omega_i = Z e B_0/(m_i c)`,
`rho_i = c sqrt(2 m_i T_0i)/(Z e B_0)` (thermal speed defined with the factor 2),
`v_A = B_0/sqrt(4 pi n_0i m_i)`, `B_0 = B_0 zhat`. Paper Eqs. (2.1)–(2.3)
(labels `eq: FLRMHD n`, `FLRMHD A`, `FLRMHD phi`), dissipation omitted:

    (d/dt + u_perp . grad_perp) dn_e/n_0e
          = -(c/(4 pi e n_0e)) (d/dz + b_perp . grad_perp) grad_perp^2 A_par
    (d/dt + u_perp . grad_perp) A_par
          = -c d(varphi)/dz + (c T_0e/e) (d/dz + b_perp . grad_perp) dn_e/n_0e
    dn_e/n_0e = -(Z/tau) (1 - Gamma_0_hat) e varphi/T_0e

with `u_perp = (c/B_0) zhat x grad_perp varphi` (E×B flow),
`b_perp = -(1/B_0) zhat x grad_perp A_par` (note the minus sign), and
`Gamma_0_hat = I_0(alpha) exp(-alpha)`, `alpha = -rho_i^2 grad_perp^2/2`, i.e.
`alpha = k_perp^2 rho_i^2/2` in Fourier space. Limits [PAPER]:
`1 - Gamma_0 -> k_perp^2 rho_i^2/2` (k_perp rho_i << 1, RMHD) and `-> 1`
(k_perp rho_i >> 1, ERMHD). The paper's dissipation is
`D = nu_6perp grad_perp^6 + nu_6z d_z^6`, the same operator on `dn_e` and on `A_par`.

Linear modes [PAPER]: `omega = +- k_z v_ph(k_perp) v_A` with

    v_ph(k_perp) = (k_perp rho_i/sqrt 2) [ 1/(1 - Gamma_0) + Z/tau ]^(1/2)
                 -> 1                                   (k_perp rho_i << 1)
                 -> (1/2 + Z/(2 tau))^(1/2) k_perp rho_i   (k_perp rho_i >> 1)

Generalized Elsasser potentials [PAPER]:

    Theta^+-_k = -Omega_i (v_ph/k_perp^2) dn_e/n_0e  -+  A_par/sqrt(4 pi m_i n_0i)

Invariants [PAPER]: `E = (1/4) sum_k (|k_perp Theta^+|^2 + |k_perp Theta^-|^2)`,
`H = (1/4) sum_k (|k_perp Theta^+|^2 - |k_perp Theta^-|^2)/v_ph(k_perp)`; inequality
`v_ph |E_H(k)| <= E(k)`; injection imbalance `sigma_eps = |eps_H|/eps`. All runs in
the paper use `Z = 1`, `tau = 0.5`.

### 2.2 Normalisation and code variables [DERIVED, CHECKED]

Taranis RMHD conventions (from `rmhd.NonlinearTerm`/`FDLinearTerm`, CLAUDE.md):
`u = zhat x grad phi`, `b_perp = zhat x grad psi` in Alfven units, `A_z = -psi`,
`bracket(a,b) = d_x a d_y b - d_y a d_x b` (same as the paper's `[P,Q]`), and

    d_t (lap phi) = -{phi, lap phi} + {psi, lap psi} + d_z lap psi
    d_t psi       = -{phi, psi} + d_z phi

Map (Gaussian -> code, with `v_A = 1`):

    phi   = c varphi/B_0                    (stream function of u_perp;  cm^2/s)
    psi   = -A_par/sqrt(4 pi m_i n_0i)      (flux function in velocity units)
    nhat  = Omega_i dn_e/n_0e               (density in vorticity units; 1/s)

Using `c T_0e/(e sqrt(4 pi m_i n_0i)) = (Z/tau) v_A Omega_i rho_i^2/2` and
`c/(4 pi e n_0e) = v_A/(Omega_i sqrt(4 pi m_i n_0i))` (n_0e = Z n_0i), the three
equations become

    nhat_k = -G(k_perp) phi_k,   G = (2/rho_i^2)(1 - Gamma_0(k_perp^2 rho_i^2/2))
    d_t nhat = -{phi, nhat} + {psi, lap psi} + d_z lap psi
    d_t psi  = -{phi_e, psi} + d_z phi_e,     phi_e = phi - rho_s^2 nhat = (1 + rho_s^2 G) phi

with `rho_s^2 = Z rho_i^2/(2 tau)` (the ion sound radius squared; `rho_s = rho_i` at the
paper's `Z=1, tau=1/2`, matching its remark). Two structural facts the design rests on:

- `{phi, nhat} = {phi_e, nhat}` identically (`{nhat, nhat} = 0`) [CHECKED to 1e-15], so
  the RHS needs gradients of exactly four fields `(phi_e, psi, nhat, j = lap psi)` —
  **the same transform count as RMHD** (8 inverse + 2 forward).
- FLR-MHD is RMHD with three k-local substitutions: vorticity `lap phi -> nhat = -G phi`,
  advecting potential in BOTH brackets `phi -> phi_e`, inversion `inv_ksq -> 1/G`.

Only two dimensionless parameters enter the dynamics: `rho_i` (in box units) and
`Z/tau`. `Omega_i` drops out; it is needed only to convert `nhat` back to
`dn_e/n_0e = nhat/Omega_i` (an RMHD-ordering amplitude, like B_0 for particles).

k-local factors (all `(nkx, nky)`, real):

    G     = (2/rho_i^2)(1 - Gamma_0(alpha))        -> ksq         (rho_i -> 0);  -> 2/rho_i^2
    pe    = 1 + rho_s^2 G                          -> 1;          -> 1 + Z/tau
    a     = ksq/G     (L's [phi,psi] factor)       -> 1;          -> ksq rho_i^2/2
    kappa = sqrt(G pe)/k_perp                      -> 1
    v_ph  = k_perp sqrt(pe/G) = kappa a = pe/kappa -> 1

`v_ph` above equals the paper's expression to 4e-16 [CHECKED]. Reference values at
`Z/tau = 2`: `v_ph(k rho_i = 0.1, 1, 10) = 1.007, 1.552, 12.37`.

### 2.3 Linear dispersion and eigenmodes [DERIVED, CHECKED]

Linearizing about rest: `d_t phi = i kz a psi`, `d_t psi = i kz pe phi`, so
`omega^2 = kz^2 a pe = kz^2 v_ph^2`. With `chi = kappa phi`:

    d_t chi = i kz v_ph psi,   d_t psi = i kz v_ph chi
    Theta^+- = chi +- psi = kappa phi +- psi,    d_t Theta^+- = +- i kz v_ph Theta^+-

`Theta^+- = kappa phi +- psi` is the paper's definition under the map of §2.2
(`-Omega_i (v_ph/k^2) dn_e/n_0e = (v_ph G/k^2) phi = kappa phi`). `Theta^+` propagates
toward **-z** (as RMHD's `z^+ = phi + psi` does under `d_t psi = d_z phi`).

### 2.4 Invariants in code variables [DERIVED, CHECKED]

    E   = (1/2) sum_k [ G pe |phi_k|^2 + ksq |psi_k|^2 ]
        = (1/2) < G|phi|^2 >  +  (rho_s^2/2) < nhat^2 >  +  (1/2) < |grad psi|^2 >
        = (1/2) < |grad chi|^2 + |grad psi|^2 >  =  E^+ + E^-,   E^+- = (1/4)<|grad Theta^+-|^2>
    H   = sum_k G Re(phi_k^* psi_k) = -< nhat psi >
        = sum_k (ksq/v_ph) Re(chi_k^* psi_k) = sum_k (E^+_k - E^-_k)/v_ph(k)

`rho_i -> 0`: `E -> (1/2)<|grad phi|^2 + |grad psi|^2>` and `H -> <grad phi . grad psi>
= <u . b>` — exactly `diagnostics.rmhd.energy` and the cross-helicity
`tests/test_forcing_modes.py` uses. Conservation proofs (continuum): `dE/dt =
-<phi_e d_t nhat> - <lap psi d_t psi>` and `dH/dt = -<psi d_t nhat + nhat d_t psi>`
vanish term by term by the cyclic bracket identity, self-adjointness of `G`, and
integration by parts in z. Per-mode: `v_ph |H_k| <= E_k` (AM-GM on the two E terms).

**Discrete status.** The 2/3-dealiased bracket is a Galerkin truncation, so both
quadratic invariants are conserved by the semi-discrete RHS to round-off: the scratch
check gives `sum_k T_E(k)/sum|.| = -2.6e-17`, `sum_k T_H = -3.7e-18`, and the linear
terms contribute `1e-17`. Time stepping then gives a drift at the scheme's order. Under
`z_spectral` the linear waves are in L and applied exactly (unitary at nu = 0), so the
only drift is from the nonlinear term. Under FD-z the wave term is an explicit centred
stencil: skew-symmetric (conserves at the semi-discrete level), drift at scheme order.

### 2.5 Dissipation [PAPER + DERIVED]

Because `nhat = -G phi` is k-local, the paper's operator on `dn_e` is the same diagonal
on `phi`. With `D_f = -nu_f k_perp^(2h) - nu_z kz^(2 h_z)`:
`dE_k/dt = 2 D E_k` for equal coefficients; in general
`eps_perp = sum_k k_perp^(2h) [nu_phi G pe|phi|^2 + nu_psi ksq|psi|^2]`,
`eps_z = nu_z sum_k kz^(2h_z) 2 E_k`, and helicity
`eps_H,perp = (nu_phi + nu_psi) sum_k k_perp^(2h) H_k`, `eps_H,z = 2 nu_z sum kz^(2h_z) H_k`.
The paper uses `h = 3`, `h_z = 3` (kz^6, in `k_z` not `k_par`) and reports that the
`k* rho_i ~ (1 - sigma_c)^(1/4)` law is robust to the parallel order.

### 2.6 The exact rho_i -> 0 limit against the RMHD recipe [DERIVED]

At `rho_i = 0`: `G = ksq`, `1/G = inv_ksq`, `pe = 1`, `a = 1`, `kappa = 1`, and the
FLRMHD recipe is term-for-term `rmhd.NonlinearTerm` + `rmhd.linear_matrix`. Leading
corrections are O(rho_i^2): `G = ksq (1 - (3/8) ksq rho_i^2 + ...)`,
`v_ph^2 = 1 + (3/4 + Z/tau) k_perp^2 rho_i^2/2 + ...` (from
`1 - Gamma_0(alpha) = alpha - (3/4) alpha^2 + (5/12) alpha^3 - (35/192) alpha^4 +
(21/320) alpha^5 - ...`, checked against `i0e` at alpha = 0.05 to the next-term size).

### 2.7 Timestep-relevant speeds [DERIVED]

Linearizing about a uniform `u_0`, `b_0`: `omega = k_perp.u_0 +- v_ph(k_perp)(kz +
k_perp.b_0)`. So the nonlinear CFL rate is `(|u| + v_ph,max |b_perp|) k_max`, not
RMHD's `(|u| + |b_perp|) k_max`, and the explicit-FD-z linear rate is `v_ph,max/dz`.
`v_ph,max` is the max over DEALIASED modes: at k_max rho_i = (1.7, 2.1, 5, 10) and
Z/tau = 2 it is (2.28, 2.74, 6.3, 12.4).

### 2.8 What the scratch check covered

`<scratchpad>/flrmhd-plan/work/check.py` (numpy/scipy, 48^2, rho_i = 0.35, Z/tau = 2;
not committed): nonlinear E and H transfer sums at round-off; linear E/H rates at
round-off; the `{phi,nhat} = {phi_e,nhat}` identity; `v_ph` vs the paper's formula; the
two forms of E and of H; a wrong sign on the `rho_s^2` term breaks energy conservation at
7.5e-2 relative (so the gate has teeth); `1 - i0e(alpha)` loses digits at small alpha
(relative error 1.3e-8 at alpha = 1e-8) — hence the series branch of §4.2.

## 3. Design decisions

### 3.1 New eqtype, not an option on RMHD

`eqtype="FLRMHD"` with its own module `physics/flrmhd.py`. Reasons: (i) RMHD's
byte-identity is then structural — `physics/rmhd.py` is not edited at all, and no RMHD
read site gains a branch; (ii) the registry conventions key on eqtype (diagnostics
module name, overrides table, particle rejection, `params.json` cross-restart
protection — FLRMHD and RMHD states have the same shape and dtype, so the
differing-record check on `eqtype`/`eqpars` is the only guard, as for `z_spectral`);
(iii) the eqpars schema differs (`rho_i`, `tau`, `Z`, `z_hyper`). `flrmhd.py` imports
and calls rmhd helpers where the code is literally the same (`rmhd._kz_deriv`,
`rmhd.fd_linear_active`, `rmhd._forcing_scale_from`, `rmhd._quiescent_dt`) — precedent:
`diagnostics/gdi.py` imports underscore names from `physics/gdi.py`.

How byte-identity of existing paths is guaranteed and gated: §8.

### 3.2 Evolved variables: (phi, psi)

Evolve `phi` (not `nhat`, not `Theta^+-`). Then the state means what an RMHD state
means, `rho_i = 0` is literally RMHD, `initialize` ICs and the O-U forcing layout carry
over, and `1/G` is benign (G > 0 for k != 0, bounded by 2/rho_i^2). Rejected:
`(nhat, psi)` (no gain; loses the RMHD correspondence); `Theta^+-` (L becomes a complex
DiagonalOperator, but only for nu_phi = nu_psi, the diagonal backend is never hoisted so
it pays a complex full-grid exp per stage, and every diagnostic/IC needs a conversion).

### 3.3 Where the FLR operators live: one new `K_Grids` slot filled by `setup_kgrids`

Add ONE optional slot `K_Grids.eq: Optional[tuple] = None` and ONE optional recipe field
`EquationRecipe.kgrid_func: Optional[Callable] = None` (`(kgrid, params) -> NamedTuple
of arrays`). `setup_kgrids` calls it (when not None) after the base arrays exist and
BEFORE `_attach_linear_operator`, so `linear_matrix` can read `kgrid.eq`. For FLRMHD it
returns `FLRFactors(G, inv_G, pe, a, kappa, vph)`. This keeps `setup_kgrids` the sole
constructor and `K_Grids` a dumb container; `kgrid_specs`/`_kgrid_to_global` treat the
slot like `lin` (a replicated pytree). For RMHD/GDI/CMHD the slot is `None` — a leafless
pytree, so no jitted graph changes.

Alternative considered: compute the factors from `kgrid.ksq` inside `grad`/the term
(they are `(nkx,nky)`, so the cost is ~1/nz of a field op in 3D). Rejected because in 2D
a Bessel evaluation per RHS is a visible cost, the hoisting of it would rest on XLA's
LICM, and the overrides seam would need a read-site branch in every consumer instead of
one in `kgrid_func`. **[DECISION FOR ALFRED]**: the slot touches `grids.py` (shared);
if a zero-shared-edit rung F1 is preferred, the alternative works and can be swapped
later without changing any gate.

### 3.4 The linear operator

**z_spectral (production for <= 256^3, single device).** L in `(phi, psi)`:

    L = [[ d_phi,        i kz a(k_perp) ],
         [ i kz pe(k_perp),   d_psi     ]],   d_f = -nu_f ksq^h - nu_z kz^(2 h_z)

with `kz = rmhd._kz_deriv` (Nyquist plane zeroed) and `a = pe = 1` at `k_perp = 0`
(continuity with RMHD; those modes carry no energy). This is a dense 5-d L, i.e. the
existing **Putzer2Operator**, with no propagator change: `s^2 = m^2 - det =
((d_phi-d_psi)/2)^2 - kz^2 v_ph^2`, the hermitian check passes (`L(-kx,-kz,ky) =
conj L`), hoistable, exact for any nu_phi != nu_psi.

`SeparableL` cannot be generalized cheaply: its saving is that `cos(kz tau)` is
`(nz,1,1)`; here the phase is `kz v_ph(k_perp) tau`, which is full-grid. What CAN be
saved is Putzer2's generality — rung F5 (optional) adds a `DispersiveL` backend for
`nu_phi == nu_psi`: `exp(L tau) = e^(d tau) [cos(theta) I + i sin(theta) [[0, a/v_ph],
[pe/v_ph, 0]]]`, `theta = kz v_ph tau`, i.e. two REAL full-grid arrays per stage
(1 u) instead of Putzer2's four complex ones (4 u), and real trig instead of complex
exp. It is a measured-performance rung, not a correctness one; F1–F4 ship on Putzer2.

Hoisting: Putzer2 is hoisted whenever dt is frozen. With adaptive `cfl_every=1`
nothing is hoisted and every stage evaluates the complex exps; production runs from
developed states should use `cfl_every` 4–8 (the standing overshoot caveat applies).
Memory: +4 u per stage (lsrk54 ~44 u, lsrk33 ~35 u, from the nu != eta RMHD numbers).

**IMEX**: CB-IMEX on FLRMHD + z_spectral is rejected in `_check_supported` (the L is
wave-dominated and stiffer than RMHD's; the standing rule). Allowed in 2D and FD-z,
where L is the dissipation diagonal.

**2D**: no kz; L is `DiagonalOperator(-diss ksq^h)`, exactly as RMHD 2D.

**FD-z (rung F6).** L = dissipation diagonal (unhoisted, byte-for-byte the RMHD
structure); the wave term is an explicit `Term`, the analogue of `rmhd.FDLinearTerm`:
`stack([a*df_dz[1], pe*df_dz[0]]) - z_diss (dz/2)^4 d4f_dz4` — the fields are k-space in
perp and real in z, so `a`, `pe` multiply the stencil output directly. Halo and
predicates reuse `rmhd.halo_start`/`fd_linear_active`.

**Timestep constraint per path** (quantified with §2.7):

| path | linear limit | nonlinear limit |
|---|---|---|
| z_spectral, IF | none (exact) | `(max|grad phi_e| + v_ph,max max|grad psi|)/dx` |
| FD-z, explicit | `v_ph,max/dz` | same |
| 2D | none | same |

At the paper's parameters (§10: k_max rho_i ~ 1.7–2.1) `v_ph,max` ~ 2.3–2.7: FD-z's
linear dt is ~2.5x smaller than RMHD's, and the nonlinear dt 1.5–2.5x smaller on every
path. A KAW-range study (k_max rho_i = 10) costs 12x in dt on every path and ANOTHER
factor through `v_ph,max/dz` only on FD-z. z_spectral is production; FD-z exists for
resolutions that need more than one device.

### 3.5 set_timestep

`set_timestep(grads, params)` sees only grads and params, so `v_ph,max` is a setup-time
python float `flrmhd._vph_max(params)` (lru_cache, numpy rebuild of the dealiased
k-grid — the `gdi._max_re_lambda` pattern). Then

    max_vy = max(|gphi_e[0]| + vmax |gpsi[0]|),  max_vx = max(|gphi_e[1]| + vmax |gpsi[1]|)
    max_all = max(max_vx/dx, max_vy/dy);  FD-z: also vmax/dz and z_diss
    dt = min(cfl_safety/allreduce_max(max_all), rmhd._quiescent_dt(params))

`gphi_e` (not `gphi`) is what the recipe has and is the conservative choice. The
`dt <= _quiescent_dt` invariant the per-stage forcing cap relies on is kept. At
`rho_i = 0`, `vmax = 1` and this is `rmhd.set_timestep` exactly.

### 3.6 Dissipation knobs

eqpars `diss` (scalar or `(nu_phi, nu_psi)`), `hyper` (int >= 1), `z_diss_k` (>= 0,
z_spectral only), and NEW `z_hyper` (int >= 1, default 2 = RMHD's kz^4; **3 matches the
paper's nu_6z kz^6**). Meyrand et al.: `h = 3`, `h_z = 3`, equal coefficients on both
fields, values "chosen based on resolution" (not tabulated in the paper — [UNVERIFIED],
§12). FD-z has only the existing grid-tied `z_diss (dz/2)^4 d_z^4` filter (a kz^6
stencil needs halo width 3 — out of scope; the paper states the break law is insensitive
to the parallel order). The barrier saturates through `eps_z`, so `nu_z` is a physics
parameter of the saturated state, not a numerical convenience: it must be recorded and
scanned (`E_sat ~ (eps/nu_6z)^(1/4) (k* L)^(-5/8) (v_A L)^(3/2)`; the arXiv source prints
the exponent as -1/4 but the prose and figure say `nu_6z^(-1/4)`, and re-deriving from
the paper's own balance gives +1/4 on `eps/nu` — [DERIVED], flagged in §12).

### 3.7 Forcing

**Mapping.** In `(chi, psi) = (kappa phi, psi)` the FLR energy is RMHD's
`(1/2)<|grad chi|^2 + |grad psi|^2>` and `Theta^+- = chi +- psi`. So the O-U envelopes
are defined as forcing ON `chi` (momentum mode) or ON `Theta^+-` (elsasser mode), and

    forcing_scale(state, kgrid, params, dt) =
        rmhd._forcing_scale_from(stack([kappa*phi, psi]), f_raw, kgrid, params, dt)
    ForcingTerm = rmhd-style (f_chi, f_psi), then f_phi = f_chi/kappa

`selfnorm_scale`, `safe_scale`, the cap and the factor 2 (`E = (E^+ + E^-)`, entries of
`forcing_power_elsasser` are contributions to total dE/dt) are reused UNCHANGED: `P =
<grad Theta . grad f_raw>` and `F2 = <|grad f_raw|^2>` are exactly the FLR-energy cross
and self terms, because the energy is the same quadratic form in `Theta^+-`. Nothing in
`shared_physics` or `run.py` changes; the RNG stream is the RMHD one, so at `rho_i = 0`
a forced FLRMHD run reproduces the forced RMHD run (gate F3.1).

**Energy AND helicity injection.** `eps = eps_+ + eps_-` exactly (per step, as in RMHD).
The helicity rate is `eps_H = sum_k (eps_+,k - eps_-,k)/v_ph(k)`, so with shell-mean
`1/vbar = sum_shell w_k/v_ph(k) / sum_shell w_k` (`w_k = yfac ksq`, the O-U self-energy
weights), `eps_H ~= (eps_+ - eps_-)/vbar` and

    eps_+- = eps (1 +- sigma_eps vbar)/2          (helper flrmhd.elsasser_powers(eps, sigma_eps, params))

This is exact when `v_ph` is constant over the shell; the realised `eps_H(t)` fluctuates
by the shell's `v_ph` spread: 0.1% at `k_min rho_i = 0.02`, ~1% at 0.1 (fshell (1,2)).
The realised value is DIAGNOSED exactly every step (§3.8), which is what the science
needs. Realizability: `sigma_eps <= 1/v_ph(k_f)` for non-negative `eps_-` — not binding
at the paper's parameters (v_ph(k_f) <= 1.03).

**Recommendation**: ship this (rung F3). An exact two-constraint `(eps, eps_H)` solve,
or the paper's forcing — negative damping on the large-scale modes of `n_e` and `A_par`
at `0 < k_perp <= 4 pi/L`, `|k_z| = 2 pi/L` [PAPER], whose exact formula is not given —
is rung F3b, **deferred unless Alfred wants forcing-identical comparison with TURBO**
[DECISION FOR ALFRED]. A deterministic negative-damping mode would also be the natural
forcing for step 3 (no RNG in the differentiated graph); sketch in §11.

Trap carried over: `cfl_every > 1` from a quiescent forced start overshoots; start
from a developed state. The paper's IC is random large-scale with `E = 10 eps tau_A`.

### 3.8 Diagnostics (`diagnostics/flrmhd.py`)

Plain imports, diagnostics -> physics, listed in `diagnostics/__init__.py::__all__`, no
top-level re-exports. Everything on the shared `perp_reduce` normalisation (ky-doubling,
`/ nz (nx ny)^2`, `nz^2` under z_spectral), so numbers are comparable with
`diagnostics.rmhd` and `forcing_power`. Functions:

- `energy(state,kgrid,params) -> dict(E_ion=(1/2)<G|phi|^2>, E_e=(rho_s^2/2)<nhat^2>,
  E_mag, E_plus, E_minus, total)`; `helicity(...) -> H`; `sigma_c = H/E` is reported but
  named `sigma_H` in code (equal to sigma_c only when large scales dominate [PAPER]).
- `perp_spectra(...) -> (k, E_plus(k), E_minus(k), E_H(k))`, binned by `_binned` to
  `perpspec`'s kmax; `sum E_+ + E_- = E`, `sum E_H = H`.
- `kperp_kz_spectrum(...)` (size==1; z-FFT under FD-z like `parspec`): 2D `E(k_perp,kz)`
  and the dissipation spectrum of the paper's Fig. 3(c).
- `dissipation(...) -> dict(eps_perp, eps_z, epsH_perp, epsH_z)` (§2.5 formulas).
- `injection(state,kgrid,params) -> dict(eps_plus, eps_minus, eps, eps_H)`:
  instantaneous cross terms `s P` from `state.forcing_state/forcing_scale` (the per-step
  self term `(1/2) s^2 F2 dt` is returned separately, since it needs the step's dt).
- `transfer(state,kgrid,params) -> (k, T_plus(k), T_minus(k), T_H(k))` and
  `fluxes(...) -> (k, Pi(k), Pi_plus, Pi_minus, Pi_H(k))`: one call of the solver's own
  `flrmhd.NonlinearTerm` (so the diagnostic cannot drift from the solver),
  `T_+-(k) = (1/2) ksq Re(Theta^+-* N_Theta^+-)` with `N_Theta^+- = kappa N_phi +- N_psi`,
  `T_H(k) = (T_+ - T_-)/v_ph` per mode, `Pi_X(K) = -sum_{k_perp < K} T_X`. Sign: Pi > 0
  is transfer to small scales. Cost: one RHS.
- `budget(...)`: `dE/dt = eps - eps_perp - eps_z`, `dH/dt` likewise, with `closure`.
- `spectral_break(k, E_plus)`: broken-power-law fit for `k*` (the paper uses astropy's
  fit; a simple two-slope least squares is enough — [UNVERIFIED] how sensitive `k*` is
  to the estimator).

Under `comm_backend="jax"` these z-reducing calls must be made inside
`comms.shard_call` (standing trap).

### 3.9 Backends, dims, particles, stability, overrides

- `dims==2`: supported from F1 (single process, as all 2D).
- `dims==3`: z_spectral (size 1, non-sharded — what `Parameters` already enforces) in
  F1–F5; FD-z + mpi4jax + `"jax"` in F6. `flrmhd._check_supported` raises
  `NotImplementedError` for FD-z until F6 (as GDI does).
- Particles: rejected by existing validation. Nothing to do. (FLR E_par exists — a
  future plan, not this one.)
- Stability harness: works on any recipe; J v = `apply_L` + jvp(N). Gates in F1.
  Tearing with FLR/KAW physics becomes available for free.
- Overrides (`overrides._table("FLRMHD")`): `Lx, Ly, Lz`, `rho_i` (live only when the
  static `rho_i > 0`; the `rho_i == 0` branch is trace-time), `tau`, `Z`, `diss`
  (scalar/pair), `z_diss_k` (z_spectral). Static: `hyper`, `z_hyper`. All FLR read
  sites are in `kgrid_func` and `linear_matrix`, both inside
  `if kgrid.overrides is not None:` + `getp`; the NonlinearTerm reads `kgrid.eq` arrays
  only and needs no branch. `_vph_max` is timestep-only and is NOT overridden.

## 4. Module design

### 4.1 Files

New: `taranis/physics/flrmhd.py`, `taranis/diagnostics/flrmhd.py`,
`tests/test_flrmhd_operators.py`, `test_flrmhd_linear.py`, `test_flrmhd_conservation.py`,
`test_flrmhd_forcing.py`, `test_flrmhd_diagnostics.py`, (F6) `test_flrmhd_fdz.py`,
`examples/helicity_barrier_run.py` + `examples/helicity-barrier.ipynb`,
`bench/flrmhd_perf.py`.

Edited (shared, each with a byte-identity argument in §8): `taranis/physics/__init__.py`
(import, registry entry, `kgrid_func` field appended with default None),
`taranis/grids.py` (`eq` slot appended with default None; the hook call;
`kgrid_specs`/`_kgrid_to_global` tree branch), `taranis/overrides.py` (table + static
eqpars entries), `taranis/diagnostics/__init__.py` (import + `__all__`), and — NOT
foreseen by this draft — `taranis/run.py`: rung F1 added `_check_scheme`, called by
`block_of_steps`, `simulate` and `simulate_scan`, which forwards to the recipe's optional
`scheme_check_func` (FLRMHD: CB-IMEX on the z_spectral wave L is rejected there, the
scheme not being a `Parameters` attribute). It is plain python on static arguments, so
no graph changes; **a stepper called directly bypasses it** (only `run.py`'s three
drivers and `stability.propagator_*`, which has its own rejection, are guarded). NOT
edited: `rmhd.py`, `shared_physics.py`, `propagators.py` (until F5), `config.py`,
`timestepping.py`, `stability.py`, `snapshot_io.py`.

### 4.2 `physics/flrmhd.py`

    class FLRGrads(NamedTuple):   gphie, gpsi, gn, gjpar        # each (2,nz,nx,ny)
    class FLRFactors(NamedTuple): G, inv_G, pe, a, kappa, vph   # each (nkx,nky), ftype

    _EQPARS_REQUIRED = ("rho_i", "tau", "diss", "hyper")
    _EQPARS_OPTIONAL = ("Z", "z_diss_k", "z_hyper")     # Z=1.0, z_diss_k=0.0, z_hyper=2
    _eqpars(params)            # validate: rho_i >= 0, tau > 0, Z > 0, diss >= 0, unknown keys raise
    _check_supported(params)   # FD-z until F6; IMEX + z_spectral; (forcing allowed)
    one_minus_gamma0(alpha)    # jnp float64: series through alpha^5..7 below a cutoff, 1 - i0e above
    factors(kgrid, params)     # the kgrid_func; rho_i == 0 -> (ksq, inv_ksq, ones, ones, ones, ones)
    linear_matrix(kgrid, params)
    grad(state, kgrid, params) # grad_fields((pe*phi, psi, -G*phi, -ksq*psi))
    set_timestep(grads, params)
    NonlinearTerm(state, grads, kgrid, params, halo=None)
        NL_n   = bracket(gpsi, gjpar) - bracket(gphie, gn)
        NL_psi = -bracket(gphie, gpsi)
        -> stack([-inv_G*fft(NL_n), fft(NL_psi)]) * dealias
    FDLinearTerm (F6), ForcingTerm, forcing_scale, forcing_active = rmhd.forcing_active
    _vph_max(params), elsasser_powers(eps, sigma_eps, params)

`one_minus_gamma0`: computed in float64 (x64 is always on) and cast to `ftype` by
`factors`; `jax.scipy.special.i0e` (present in the installed jax; scipy is NOT a taranis
dependency). The series/closed-form switch point is chosen so both branches agree to
<= 4 eps at the switch (gate F1.2); `a = alpha/(1-Gamma_0)` and `kappa` are formed from
the series directly at small alpha, never as `ksq * inv_G`.

Registry entry:

    "FLRMHD": EquationRecipe(set_timestep_func=flrmhd.set_timestep,
        term_funcs=(Term(flrmhd.NonlinearTerm),
                    Term(flrmhd.FDLinearTerm, active=rmhd.fd_linear_active),   # F6
                    Term(flrmhd.ForcingTerm, active=rmhd.forcing_active)),
        grad_func=flrmhd.grad, nfields=2, forcing_scale_func=flrmhd.forcing_scale,
        halo_start_func=rmhd.halo_start, linear_matrix_func=flrmhd.linear_matrix,
        kgrid_func=flrmhd.factors)

## 5. Rungs and gates

Tolerances are fp64 unless stated; every test file runs in both `make test` sessions or
carries a precision marker with the measured fp32 floor in its docstring. House test
form (bootstrap header, `script_main` footer, `fresh_params` for dict-valued kwargs).
"Catches" names the mutation each gate is designed to kill (§7 has the full table).

### F0 — derivation into the docs (docs only)

Add `docs/numerics.md` § "FLR-MHD (`physics/flrmhd.py`, plans/FLRMHD_PLAN.md)" with
subsections: *Equations in Gaussian units and the map to (phi, psi)* (§2.1–2.2 here);
*The FLR factors and their small-alpha evaluation*; *Linear waves and the generalized
Elsasser potentials*; *Invariants and their discrete status*; *The rho_i = 0 branch*;
*Timestep*; *Forcing in the FLR energy*; *Flux and dissipation diagnostics, sign
conventions*. Gate: an adversarial reviewer re-derives §2.2–2.4 from the paper
independently, sign by sign, and reruns/extends the scratch check; the paper's
`Theta^+-`, `E`, `H`, `v_ph` must be recovered. Resolve the §12 decisions with Alfred.

### F1 — core: factors, recipe, 2D + z_spectral, unforced

Gates:

1. **Existing paths untouched** — `tests/test_refactor_reference.py` (fields, t, HLO
   histogram, 12 configs), gate 6/6b/6c, the spinup and fp32 references, and full
   `make test`, run ON THE RECORDING HOST (`Alfreds-MacBook-Pro.local`; a print-skip is
   not a pass). Catches: any leak of the new slot/hook into existing graphs.
2. **Factors** (`test_flrmhd_operators.py`): `1 - Gamma_0(alpha)` against an independent
   reference — the quadrature `Gamma_0 = (1/pi) int_0^pi exp(-alpha(1 - cos t)) dt`
   (spectrally convergent; agrees with `i0e` to 1e-16 in the scratch check) for
   alpha in [1e-3, 1e3], and against the series for alpha in [1e-14, 1e-2] — relative
   error <= 1e-13; continuity across the switch <= 4 eps; `G -> ksq` relative error
   `<= (3/4) alpha (1 + 0.1)` (asserts the leading correction's COEFFICIENT, not just
   smallness); `G -> 2/rho_i^2` at alpha = 1e3 within `1/sqrt(2 pi alpha)`; `v_ph`
   equals the paper's closed form to 1e-13 at Z/tau in {2, 1, 0.5}; the `rho_i = 0`
   branch returns arrays BITWISE equal to `kgrid.ksq`, `kgrid.inv_ksq`, ones. Catches:
   alpha = k^2 rho^2 (missing 1/2), `I_0` without the exponential, Pade `1/(1+alpha)`
   substituted for Gamma_0, `tau/Z` for `Z/tau`.
3. **rho_i = 0 reproduces RMHD** (`test_flrmhd_linear.py`): same IC, same params, RMHD
   vs FLRMHD(`rho_i=0`), (a) one RHS evaluation, (b) 20 lsrk54 steps, in 2D 64^2
   (decaying, hyper=2) and 3D z_spectral 32^2 x 8 with nu_phi = nu_psi (RMHD on
   SeparableL, FLRMHD on Putzer2) and nu_phi != nu_psi (both Putzer2). Assert relative
   L2 difference <= 1e-12 in every case and REPORT whether it is bitwise. Not asserted
   bitwise because the graphs differ (multiplies by `pe = 1`, a different propagator
   backend in the equal-diss case): the standing experience (CLAUDE.md, GRAD_CHUNK and
   `"serial"` notes) is that such differences move 3D results by an ulp through XLA
   fusion. Catches: every bracket sign/operand error visible in the RMHD limit, a wrong
   `inv_G` placement, a transposed L.
4. **Small-rho_i convergence**: single linear mode, `omega^2/kz^2 - 1` against
   `(3/4 + Z/tau) k^2 rho_i^2/2` for rho_i halved three times: measured order 2.00 +-
   0.05 and coefficient within 2%. Catches: a wrong O(rho_i^2) coefficient that F1.3
   cannot see (e.g. `rho_s^2` off by 2).
5. **KAW dispersion, exact** (z_spectral, the `test_alfven_dispersion_exact` pattern):
   IC pure `Theta^+` (so `psi = kappa phi`) at one k_perp per run, k_perp rho_i in
   {0.01, 0.1, 0.5, 1, 2, 5, 10} (rho_i varied at fixed grid), Z/tau in {2, 1},
   amplitude 1e-8, nu = 0, fixed dt, 20 steps, and once on a non-cubic box with
   ky != 0 and kx != 0 (so 2 pi/L factors cannot cancel). Assert: (a) measured phase
   rate of each coefficient equals `+kz v_ph` with v_ph from the INDEPENDENT closed
   form (quadrature Gamma_0), error <= 1e-11 relative; (b) the `Theta^-` projection
   stays <= 1e-12 of `Theta^+` (eigenvector); (c) the sign: `Theta^+` phase advances as
   `exp(+i kz v_ph t)`, `Theta^-` as `exp(-i ...)`; (d) with nu > 0 and `z_diss_k > 0`,
   `z_hyper` in {2, 3}: decay rate `nu ksq^h + nu_z kz^(2 h_z)` to 1e-12. Catches:
   dropped Gamma_0 in L, `a`/`pe` swapped (eigenvector test b fails, dispersion alone
   would not), wrong kappa, `Theta^+-` swapped, `z_hyper` ignored.
6. **Semi-discrete conservation, exact** (`test_flrmhd_conservation.py`): broadband
   random IC spanning k rho_i from 0.2 to 3, 2D 64^2 and z_spectral 32^2 x 8; evaluate
   the jitted RHS once and assert `|sum_k T_E|/sum_k|T_E| <= 1e-13` and the same for
   `T_H`, with T formed in the TEST from first principles (weights `G pe`, `ksq`, `G`
   built from the independent Gamma_0), not from `diagnostics.flrmhd`. Catches: sign of
   the `rho_s^2 {psi, nhat}` term (measured 7.5e-2 in scratch), `phi` for `phi_e` in
   the Ohm bracket only, dealias omitted on one term.
7. **Time-discrete conservation**: same ICs, nu = 0, dt and dt/2, 40 steps: drift order
   in [p - 0.4, p + 0.6] for lsrk33 (p=3), rk44 and lsrk54 (p=4), and absolute relative
   drift <= 1e-6 at the finer dt, for E and for H separately.
8. **Invariant values on known states**: single-k pure `Theta^+`: `H = E/v_ph(k)`
   exactly (1e-13) and pure `Theta^-`: `H = -E/v_ph`; rho_i = 0: `H` equals
   `perp_inner_product(phi, psi)` bitwise-or-1e-15 and `E` equals `diagnostics.energy`
   summed. (Computed with test-local formulas here; the diagnostics module repeats this
   in F2.) Catches: a factor 2 or sign in H, `v_ph` vs `1/v_ph`.
9. **Timestep** (`test_set_timestep_matches_the_cfl_bound` pattern): `set_timestep` on a
   constructed state equals the closed form with `v_ph,max` from an independent
   evaluation; `_vph_max` equals the max of `kgrid.eq.vph` over the dealias mask;
   discriminator — a KAW-dominated state at k_max rho_i = 10 runs 200 steps stably at
   `cfl_safety`, and with `_vph_max` monkeypatched to 1.0 it blows up or loses energy
   conservation by > 100x. `dt <= _quiescent_dt` (the `test_quiescent_dt.py` property).
10. **Stability harness**: `J(x0 = 0)` equals `kgrid.lin.dense()` entrywise (2D and
    z_spectral); `ky_block` eigenvalues at x0 = 0 equal `d +- i kz v_ph`; the FD gate of
    AUTODIFF "Common protocol" on a random x0; overrides independent route — J at
    `overrides={"rho_i": r, "tau": t}` equals J of a `Parameters` built with those values
    (the standing check that catches a read site that bypasses `getp`); `dJ/d rho_i`
    against central FD at O(h^2).
11. **Plumbing**: eqpars round trip through `params.json`; RMHD-snapshot -> FLRMHD
    restart is blocked by the differing-record check; unforced restart bitwise;
    rejections (unknown/missing keys, rho_i < 0, tau <= 0, FD-z, IMEX + z_spectral,
    particles); reality preserved over a run; `lsrk_scan` and `hoist_propagator` on/off
    agree (bitwise where the existing hoist tests are, tolerance otherwise);
    `test_equation_interface.py` extended with `FLRGrads` order.

### F2 — diagnostics

Gates (`test_flrmhd_diagnostics.py`): spectra sum to totals (1e-13); `E_plus + E_minus
= total = E_ion + E_e + E_mag`; F1.8's known-state identities through the module;
`T_H(k) = (T_+ - T_-)/v_ph` per bin on a single-shell state; `sum T = 0` (1e-13);
**flux against the stepper**: unforced, nu = 0, `E(<K)` for six K from two states one
small step apart, centred difference equals `-Pi(K)` at the midpoint to O(dt^2)
(assert order ~2 over dt halvings and 1e-6 at the finest) — independent of the
diagnostic's own formula, catches the flux sign and any weight error; **dissipation
against exact decay**: tiny amplitude, nu > 0, nu_z > 0, `dE/dt` from the closed-form
exponential equals `-(eps_perp + eps_z)` to 1e-11, same for H; budget closure on a
forced dissipative run <= O(dt). Mutation targets: factor 2 in `H` or `eps_H`, `1/v_ph`
dropped from `E_H(k)`, `nu_phi` weighting `ksq` instead of `G pe`, flux sign.

### F3 — forcing

Gates (`test_flrmhd_forcing.py`):

1. **rho_i = 0 forced equals forced RMHD**: momentum and elsasser, 2D and z_spectral,
   50 steps, same seed: fields <= 1e-12 relative, `forcing_state` and `forcing_key`
   bitwise (the RNG stream is untouched by construction).
2. **The scale solves the FLR quadratic**: at rho_i with k_f rho_i ~ 1.5 (so kappa and
   v_ph are far from 1), recompute `P`, `F2` in the test with explicit `G pe` weights
   and assert `s P dt + (1/2) s^2 F2 dt^2 = target dt` to 1e-12 for both channels.
   Catches: kappa omitted or inverted in `forcing_scale` or in `f_phi = f_chi/kappa`.
3. **Injection rates**: quiescent start, nu = 0, elsasser `(eps_+, eps_-)` from
   `elsasser_powers(eps=0.8, sigma_eps=0.4)` at k_f rho_i ~ 1.5 (vbar ~ 2): `E(t)/t`
   within 30% of eps (the existing gate's band, for the same RK-vs-forward-Euler
   reason) and `H(t)/t` within 30% of `sigma_eps eps` — at vbar ~ 2 a dropped `1/v_ph`
   or a factor 2 leaves the band, and the sign catches swapped channels. Sharper,
   deterministic companion: `diagnostics.flrmhd.injection` integrated over the run
   equals the measured `Delta E` and `Delta H` to 1e-3 (first-order in dt/tau_f; assert
   the convergence order under dt halving rather than a tight number).
4. **Realised sigma_eps**: at the paper-like `k_min rho_i = 0.1`, the time-mean of
   `eps_H/eps` over 200 forcing times is within 2% of the request.
5. **Forced restart bitwise** at `forcing_norm_per_step` True and False (the gate-6c /
   `test_restart_is_bitwise_at_default_norm` pattern), z_spectral.
6. `cfl_every=4` from a developed forced state stays finite and within 5% of
   `cfl_every=1` energy after 100 steps.

### F3b — exact (eps, eps_H) control / negative-damping forcing [deferred, §3.7]

Only if requested. Spec then: `c_+-` from the 2x2 `eps = 2 c_+ E^+_sh + 2 c_- E^-_sh`,
`eps_H = 2 c_+ H^+_sh - 2 c_- H^-_sh` over the forced shell, stored in
`state.forcing_scale`; gates: injected rates equal targets to O(dt), restart bitwise.
This reconstruction of the paper's forcing is [UNVERIFIED].

### F4 — the three-run physics check at toy size (still in `make test -m slow`)

z_spectral 32^2 x 32, fp64, k_min rho_i = 0.2, hyper = 3, z_hyper = 3, eps = 1:
(i) imbalanced FLR (sigma_eps = 0.88), (ii) balanced FLR, (iii) imbalanced
rho_i = 0, each ~10 tau_A. Assertions are bands calibrated by the implementer and then
frozen with the measured values in the docstring: budget closure
`|dE/dt - (eps - eps_perp - eps_z)| <= 2% eps` time-averaged in all three;
(ii) and (iii) reach `eps_perp + eps_z >= 0.7 eps`. **[UNVERIFIED]** whether a barrier
is visible at 32^2 at all; if it is not, F4 keeps only the budget and control
assertions and the barrier itself is gated in F7. Target runtime < 5 min.

### F5 — performance (optional `DispersiveL`; measurements mandatory)

`bench/flrmhd_perf.py`, interleaved A/B on a quiet machine, against z_spectral RMHD
(separable and nu != eta Putzer2) at 128^2 x 16 and 64^2 x 64: ms/step and compiled
memory in u; the FFT count read from the optimized HLO (**expected 10 per RHS, equal to
RMHD** — gate: FLRMHD's fft opcode count equals RMHD nu != eta's). Expected step cost
1.0–1.1x RMHD-Putzer2. If Putzer2's hoisted memory or unhoisted time is the binding
cost at the F7 sizes, implement `propagators.DispersiveL(dperp, dz, kz, vph, r01, r10)`
(§3.4) with the same op-order discipline as `SeparableL`, gated by: `dense()` equals
the Putzer2 L; `apply_exp` agrees with Putzer2 to 1e-13; hermitian check; hoisted ==
unhoisted bitwise at fp64; all F1 gates rerun on it; memory probe shows the predicted
~3 u/stage saving. Results into `docs/performance.md` "FLR-MHD".

### F6 — FD-z, mpi4jax and `"jax"` (when a run needs more than one device)

`flrmhd.FDLinearTerm`, `_check_supported` relaxed, `set_timestep` gains `vmax/dz`.
Gates (`test_flrmhd_fdz.py`): FD-z converges to z_spectral at 4th order in dz (the
`test_fd_z_converges_to_spectral_at_fourth_order` pattern) on a KAW at k rho_i = 1;
KAW frequency error O(dz^4) with the 4th-order stencil's known coefficient; energy and
helicity drift at scheme order; rho_i = 0 FD-z equals RMHD FD-z (<= 1e-12; the L here
is the same DiagonalOperator, so report bitwise); `"serial"` vs 4-fake-device `"jax"`
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`) agree to tolerance;
script-mode `mpirun -n 4` on Savio/Delta; halo width 2 unchanged. The diagnostics'
`shard_call` requirement is tested. `kperp_kz_spectrum` stays size==1.

### F7 — science gate: the helicity barrier (example script + notebook, not `make test`)

`examples/helicity_barrier_run.py` (resumable, profile env var smoke/m1/full, sidecar
time series of `E, H, eps_perp, eps_z, eps, eps_H` every step-block and spectra/fluxes
at snapshot cadence) + notebook. Parameters follow the paper: cubic box, `Z=1`,
`tau=0.5`, sigma_eps = 0.88, forcing shell `k_perp <= 2 (2 pi/L)`, `|kz| = 2 pi/L`,
hyper = z_hyper = 3, IC energy `10 eps tau_A`; the three runs of the paper's Fig. 3
(imbalanced FLR, balanced FLR, imbalanced rho_i = 0) at the same eps. Pass criteria
(qualitative physics, quantitative bands):

1. Controls saturate: balanced FLR and imbalanced RMHD reach `dE/dt ~ 0` with
   `eps_perp + eps_z = eps` within 15% (time-averaged), `eps_perp >> eps_z` for the
   balanced case.
2. Imbalanced FLR does not: over the pseudo-stationary phase E grows, with
   `dE/dt = eps - eps_perp - eps_z` closing to 5%, and `eps_perp/eps` falls to within a
   factor 2 of `1 - sigma_eps = 0.12` (the paper's `eps_perp ~ 2 eps^-`), while the
   RMHD control at the same time has `eps_perp/eps > 0.5`.
3. Flux blocked: time-averaged `Pi(k)/eps < 0.3` for `k > 2 k*` while `Pi/eps ~ 1`
   (within 30%) holds across the same k range in the balanced run.
4. Spectral break: `E^+(k)` steepens at `k* rho_i < 1` to a slope steeper than -3
   below the break scale (paper: ~ -3.8 against ~ -1.67), `E^-` shows no such break, and
   `k* rho_i` vs `(1 - sigma_c)^(1/4)` agrees within a factor 2 as `sigma_c(t)` grows.
5. (stretch, M1-feasible at 64^2) `E_sat` rises as `nu_z` falls, consistent with
   `nu_z^(-1/4)` within the scatter, where the RMHD control's `E_sat` does not.

Sizes and costs: §10. The reviewer's job at F7 is to try to make the barrier go away
with a numerical explanation (resolution, `nu_perp` above the paper's `nu_crit`,
forcing-shell width, cfl_every) before it is recorded as reproduced.

## 6. Landing notes

(one dated entry per rung: commit, reviewer verdict, measured gate numbers, deviations
from this plan.)

### F0 + F1 — implemented 2026-11-11 (uncommitted; adversarial review pending)

Host `Alfreds-MacBook-Pro.local`, jax 0.10.0, python 3.11.5 — the recording host of the
frozen references. Numbers are fp64 unless marked.

**Files.** New: `taranis/physics/flrmhd.py`, `tests/_flrmhd_testing.py` (numpy-only
independent references: quadrature Gamma_0, paper-form factors, test-local E/H),
`tests/test_flrmhd_operators.py` (F1.2, F1.9, F1.10, F1.11), `tests/test_flrmhd_linear.py`
(F1.3–F1.5), `tests/test_flrmhd_conservation.py` (F1.6, F1.6b, F1.7, F1.8). Edited:
`docs/numerics.md` (§ "FLR-MHD", F0), `taranis/grids.py` (`K_Grids.eq`,
`_attach_equation_arrays`, `_TREE_KGRID_FIELDS`), `taranis/physics/__init__.py`
(`kgrid_func`, `scheme_check_func`, the registry entry), `taranis/overrides.py` (FLRMHD
table, `_Key.positive`), `taranis/run.py` (`_check_scheme`, see deviations),
`tests/test_equation_interface.py` (`FLRGrads` order). `rmhd.py` untouched.

**Gates (measured / tolerance).**

| gate | result |
|---|---|
| F1.1 existing paths | `test_refactor_reference.py`: 2 passed, 60 checks, fields + t + HLO bitwise, fp64 and fp32, NOT host-skipped. `test_particles_coupled.py`: 20 passed (fp64), 10 passed + 10 fp64-marked skips (fp32), gate 6a compared bitwise against both npz files. No reference regenerated. |
| F1.2 factors | `1 − Γ₀` vs expm1 quadrature on [1e-14, 1e3]: 3.3e-16 (1e-13); vs exact-rational series on [1e-14, 1e-2]: 0 (1e-13); branches at the switch α = 1: 0.00 eps, S across the switch 1.00 eps (4 eps); `(1 − G/k²)/α` ∈ [0.7459, 0.7500] (0.75 ± 10%); `Γ₀(1e3)` = 1.26172e-2 vs (2πα)^-1/2 = 1.26157e-2; all six factors vs paper form at Z/τ ∈ {2, 1, 0.5, 4.29}, kρ_i ≤ 14: ≤ 5.3e-16 (1e-13; fp32 5e-7); `ρ_i = 0` branch bitwise. |
| F1.3 ρ_i = 0 vs RMHD | 1e-12 asserted. **Bitwise** (RHS, set_timestep, 20 lsrk54 steps, t): 2D 64²; z_spectral ν_φ ≠ ν_ψ (fixed and adaptive dt). **Not bitwise**: z_spectral ν_φ = ν_ψ (RMHD on SeparableL, FLRMHD on putzer2): RHS bitwise, 20 steps 4.0e-15 (4.8e-15 with `z_diss_k`). fp32: same pattern, 3.2e-6 in the two non-bitwise cases. L == RMHD's dense L bitwise in every case. |
| F1.4 small ρ_i | order 2.000 (Z/τ = 2), 2.001 (0.5) (2.00 ± 0.05); coefficient 2.7500/2.7501 vs 2.75 and 1.2500/1.2501 vs 1.25 (2%). |
| F1.5 KAW dispersion | phase rate vs `kz·v_ph` (quadrature): ≤ 2.2e-15 over 14 (kρ_i, Z/τ) cases, Θ⁻, and two non-cubic modes (1e-11); |F| − 1 ≤ 6.2e-15, Θ∓ projection ≤ 7.4e-16 (1e-12); decay rate `ν k^2h + ν_z kz^(2h_z)`, h ∈ {1,2}, h_z ∈ {2,3}: ≤ 5.2e-14 (1e-12). fp32 tolerance 2e-5. |
| F1.6 semi-discrete | `ΣT_E/Σ|T_E|` ≤ 4.6e-17, `ΣT_H/Σ|T_H|` ≤ 3.6e-17 (1e-13), 2D 64² and z_spectral 32²×8, two Z/τ each; fp32 ≤ 4.8e-8. Linear operator: per-mode E, H rates 5.6e-17 at ν = 0; `dE/dt = −(ε_⊥+ε_z)`, `dH/dt = −(ε_H,⊥+ε_H,z)` to 1.1e-16. |
| F1.6b paper variables | `∂φ/∂t`, `∂ψ/∂t` vs the Gaussian-unit transcription (δn_e/n_0e, A∥, u·∇, b·∇; Z = 2, τ = 0.7, A = 4 and Z = 1, τ = 0.5): ≤ 1.1e-15 (1e-12), 2D, non-square 2D, z_spectral; fp32 ≤ 5.6e-7. The FLR terms are 0.83–0.91 of that RHS. |
| F1.7 time-discrete | see deviation 1. lsrk33 2.98/3.10 (2D E/H), 2.97/2.89 (3D); rk44 5.00/4.99, 4.98/5.02; lsrk54 4.97/5.09, 4.90/4.90. Finer-dt drift ≤ 4.9e-7 (1e-6). fp64 only. |
| F1.8 known states | pure Θ±: `H v_ph/E ∓ 1` ≤ 4.4e-16 (1e-13); ρ_i = 0: E vs `diagnostics.energy` 4.4e-16, H vs `perp_inner_product` 4.4e-16; `v_ph|H_k| ≤ E_k` everywhere. |
| F1.9 timestep | `set_timestep` vs closed form 1.1e-16 (2D), 3.3e-16 (z_spectral); `_vph_max` vs independent 2.2e-16; zero state == `_quiescent_dt` exactly. Discriminator at k_max ρ_i = 10: 200 steps, energy drift 1.57e-5; with `_vph_max → 1` drift 12.1 (7.7e5×). |
| F1.10 stability | `J(0) == lin.dense()` 0.0; vs docs' L ≤ 1.3e-16; eigenvalues `d ± i kz v_ph` < 1e-13; FD gate < 1e-9; overrides independent route 0.0 (fp32 tol 5e-5); `dJ/d ln p` vs central FD order 2.00 for every live key (8 keys × 3 configs). **`dJ/dρ_i` is finite on all 70 rows including the 14 kz = 0 rows** (|·| = 1.8 on them). |
| F1.11 plumbing | eqpars round trip; RMHD record blocks an FLRMHD save (and a changed `rho_i` too); restart bitwise (2D, z_spectral, cfl_every = 2); 30+ rejections; reality 1e-12; `lsrk_scan` on/off < 1e-13; hoist on/off: bitwise in 7 of 8 cells, 4.0e-18 in z_spectral fixed-dt lsrk54. |

**The `s² = 0` tangent (§14 item 2): it does not bite `dJ`.** `J v = apply_L(v) + jvp(N)`
never reads `lin.s`, so the NaN stays in a dead tangent. Measured directly on a
ν_φ = ν_ψ z_spectral kgrid: the tangent of `lin.s` along `rho_i` is non-finite on 1088 of
4352 entries (exactly the kz = 0 and Nyquist planes), and `d/dρ_i` of
`lin.apply_exp(x, 0.1)` is non-finite on 2176 entries. Nothing in the harness or solver
differentiates through `apply_exp` with respect to a parameter today, so nothing was
changed; a rung that does (step 3) must fix it first. The fix that keeps every existing
graph is for the recipe to hand `propagators.build` a prebuilt `Putzer2Operator` with
`s = i·kz·v_ph` (ν_φ = ν_ψ) or a guarded sqrt — an edit to `propagators.py`/`grids.py`,
deliberately not made here.

**Deviations from the plan.**

1. **F1.7 window.** The plan's `[p − 0.4, p + 0.6]` fails for the two fourth-order
   schemes: the invariants drift at order 5, not 4 (a conserved quadratic invariant drifts
   at `p + 1` under an even-order scheme). Flat at 4.83–5.15 over dt = 0.01 … 0.00125. The
   gate asserts the measured order (3, 5, 5) ± 0.3 and still requires ≥ p − 0.4. "40
   steps" is read as 40 at dt and 80 at dt/2 (same T). Recorded in docs/numerics.md.
2. **CB-IMEX rejection is not in `_check_supported(params)`** — the scheme is not a
   `Parameters` attribute. It is `flrmhd.check_scheme(params, scheme)`, reached through a
   new optional recipe field `scheme_check_func` that `run.py`'s three drivers
   (`block_of_steps`, `simulate`, `simulate_scan`) call in plain python. This edits
   `run.py`, which §4.1 lists as not edited. A stepper called directly bypasses it;
   `stability.propagator_*` has its own rejection.
3. **Hoisted vs unhoisted is not bitwise in every FLRMHD cell** (one cell at 4e-18), where
   the RMHD putzer2 cells are; asserted at 1e-14 with the bitwise status printed.
4. **Mutation table corrections**: `inv_ksq` for `inv_G` is invisible to F1.5 (it is a
   nonlinear-term mutation; caught by F1.6 at 2.4e-2 and F1.6b), and a dropped final
   dealias mask is invisible to F1.6 (the state is zero outside the mask; caught by F1.3
   at 0.42 and F1.6b).
5. Series switch at α = 1 with 28 terms (§4.2 left the point open); `_vph_max` evaluates
   under `jax.ensure_compile_time_eval()` because `set_timestep` calls it inside a trace;
   `overrides._Key` gained `positive` (tau, Z > 0); `rho_i`, `tau`, `Z` are all dead keys
   when the static `rho_i == 0`.
6. Forcing raises `NotImplementedError` (F3) and no forcing hooks are registered.

**Not done / not verified**: the F1.1 list's "spinup and fp32 references" were not run as
separate files (only the two frozen-reference files, per the session's memory rules, plus
whatever the final full `make test` covered — see the session report); no adversarial
review yet; no `make test` baseline was taken before the change.

### F1 review fixes — 2026-11-12 (uncommitted)

The adversarial review of F0/F1 found no bug in `flrmhd.py`; it found coverage holes and
one wrong explanation. Each new gate was confirmed against the reviewer's mutation
(applied, gate run, reverted).

| fix | what changed | mutation now caught (magnitude) |
|---|---|---|
| 1. CFL y-direction | `test_set_timestep_matches_the_cfl_bound`: every box non-square with `dx != dy` (0.90, 1.67, 2.33), and two anisotropic states per case (ψ(x) only, ψ(y) only) so each directional rate binds in turn (by 230–940×) | `vmax` dropped from `max_vy_eff`: dt off by 5.4×–11× on the ψ(x) states; from `max_vx_eff`: 5.4×–11× on ψ(y); `dx ↔ dy` swap: 8%–130% in all nine states |
| 2. `_vph_max` at Z ≠ 1 | a `Z = 2, τ = 0.7` case in the same gate | `Z/tau → 1.0/tau`: `_vph_max` off by 20%, dt by 20–25% |
| 3. `Z` default | `test_Z_defaults_to_one`: eqpars without `Z` — factors vs the paper form at `1/τ`, `overrides.static_value(params, "Z") == 1.0`, `_eqpars`, `_vph_max` | default `2.0`: factors off by 0.57, `_vph_max` by 25%, static value 2.0 |
| 4. F1.7 explanation | docs ("Invariants and their discrete status") and the docstring now state what is known (asymptotic drift order `p`; the extra order is a property of linear skew dynamics) and what is measured (lsrk33 2.89–3.10; rk44 4.9–5.05, no trend; lsrk54 4.83–5.15 at the gate's dt, falling 4.88 → 4.60 under refinement in the review's sweep), with the near-linear-error-dominance reading labelled as an interpretation. Assertion: `p − 0.4 ≤ order ≤ p + 1.4` plus drift ≤ 1e-6 at the finer dt | — (measured this run: 2.98/3.10, 5.00/4.99, 4.97/5.09 in 2D; 2.97/2.89, 4.98/5.02, 4.90/4.90 z_spectral) |
| 5. `lin.s` tangent | docs sentence corrected: non-finite tangent of `lin.s`/`apply_exp` on the kz = 0 and Nyquist planes at ν_φ = ν_ψ for ANY overridden key — measured for `rho_i, tau, Z, diss, z_diss_k, Lx, Lz` (288 of 1152 entries of `lin.s`, both fields of `apply_exp` on those planes, 16²×8) and for RMHD with a pair-shaped equal `diss` override. Not fixed (needs `propagators.py`); nothing reachable hits it | — |
| 6. small items | `z_hyper` set away from 2 where it is dead (2D, or z_spectral with `z_diss_k == 0`) warns once per `setup_kgrids` (`flrmhd._warn_dead_knobs`, rank 0; silent when the overrides seam supplies `z_diss_k`), gated by `test_dead_z_hyper_warns`; header status and §4.1 updated (`run.py` gained `_check_scheme`; a stepper called directly bypasses it) | — |

**Still open from the review.** `grids._TREE_KGRID_FIELDS`'s `"eq"` entry (the
`kgrid_specs`/`_kgrid_to_global` branch for the `"jax"` backend) is UNEXERCISED: FLRMHD
cannot run on that backend until FD-z exists. **Rung F6 must gate it** (serial vs
4-fake-device `"jax"`). The `lin.s` tangent hazard is recorded for the learned-closure
step.

`tests/test_flrmhd_conservation.py` rerun at fp32 after the threshold edits: 4 passed, 1
fp64-marked skip.

### F3 — forcing, implemented 2026-11-12 (uncommitted; adversarial review pending)

**Files.** `taranis/physics/flrmhd.py` (`_chi_psi`, `forcing_scale`, `ForcingTerm`,
`forcing_active`, `elsasser_powers`; the F1 `NotImplementedError` on forcing removed),
`taranis/physics/__init__.py` (registry: the forcing `Term` and `forcing_scale_func`),
`tests/test_flrmhd_forcing.py` (new), `tests/_flrmhd_testing.py` (independent Elsasser
forms and shell mean), `tests/test_flrmhd_operators.py` (rejections updated), docs.
`ForcingTerm` is `rmhd.ForcingTerm` on the state `(κφ, ψ)` with its φ row divided by κ;
`forcing_scale` is `rmhd._forcing_scale_from` on `(κφ, ψ)`. `rmhd.py`,
`shared_physics.py`, `run.py` untouched by this rung.

| gate | fp64 (tolerance) | fp32 |
|---|---|---|
| F3.1 ρ_i = 0 vs forced RMHD, 50 steps, 3 grids × 3 forcing settings | fields: **bitwise** in 2D and z_spectral ν_φ ≠ ν_ψ; 1.6–1.8e-14 at ν_φ = ν_ψ (RMHD on SeparableL) (1e-12). `forcing_state`, `forcing_key` bitwise in all 9 | bitwise in the same 6; 5.1–8.4e-6 (1e-4) |
| F3.2 the scale solves the FLR quadratic (κ on the shell 0.77–1.19, v̄ 2.6; 2D non-square Z/τ = 2.86 and z_spectral; momentum + elsasser; dt 0.02, 0.003) | forward-Euler `dE = target·dt`: ≤ 4.3e-15; per channel ≤ 3.8e-15; `sP dt + ½s²F₂dt²` with explicit `G·pe` weights ≤ 1.1e-14 (1e-12); `E(f + dt F) − E(f)` ≤ 2.1e-13 (1e-10); per-stage path ≤ 1.3e-15 | ≤ 3.9e-7 (2e-4) |
| elsasser_powers | `(ε₊ − ε₋)/ε = σ_ε v̄` vs independent shell mean < 1e-13; `|σ_ε| v̄ > 1` a ValueError; ρ_i = 0 → `(0.75, 0.25)` | same |
| F3.3 rates from rest (v̄ = 2.07, σ_ε = ±0.4, ε = 0.8; 30% band) | E/t: 0.780, 0.784, 0.779, 0.781; H/t: +0.324, +0.326, −0.311, +0.308 against ±0.320 | 0.780–0.784; +0.320, +0.322, −0.317, +0.314 |
| F3.3 companion: Σ injection vs measured ΔE, ΔH | O-U forcing, dt = 0.005: 1.7e-4 / 6.8e-6 (2D), 1.5e-3 / 1.4e-3 (z_spectral) (5e-3). Frozen envelope, dt = 0.01, 0.005, 0.0025: E 4.4e-4 → 1.2e-4, order 0.97; H 1.6e-4 → 1.7e-5, order 1.61 (2D); E 3.2e-3 → 6.9e-4, order 1.10; H 3.0e-3 → 6.8e-4, order 1.07 (z_spectral) (order in [0.7, 2.3], ≤ 1e-3 at the finest) | fp64 only |
| F3.4 realised σ_ε at k_min ρ_i = 0.1, 200 τ_f | mean ε_H/ε off the request by 7.2e-5 (2D), 7.5e-4 (z_spectral) (plan 2%; also asserted at 0.5%) | 9.2e-5, 8.9e-4 |
| F3.5 forced restart, z_spectral, both norm settings | fields, forcing_state, forcing_scale, forcing_key, t bitwise | bitwise |
| F3.6 `cfl_every=4` vs 1 from a developed state, 100 steps | energy rel 1.0e-4 (2D), 2.9e-4 (z_spectral) (5%) | 3.1e-4, 1.3e-4 |
| extra: 2D momentum from rest | ψ exactly 0, H exactly 0, E/t = 0.677 vs 0.7 | same |

**Deviations and findings.**

1. **§3.7's claim that the realised `ε_H(t)` "fluctuates by the shell's v_ph spread" is
   wrong for the instantaneous value.** The TIME MEAN is controlled to that level
   (7e-5 … 9e-4 measured). Per step, the cross term `sP·dt` and the self term
   `½s²F₂dt²` are separately much larger than their sum whenever the envelope is
   anti-correlated with the field, and they are distributed differently over the shell:
   the per-step `ε_H/ε` had rms 0.05–0.11 about 0.6 with excursions to −0.36 and 1.69.
   Docs corrected; the gate asserts the mean only and prints the range.
2. **F3.3 companion.** "Assert the convergence order" is not possible with the O-U
   stream: it draws one noise field per step, so runs at different dt see different
   forcing (2D errors 3.5e-4, 6.6e-5, 1.7e-4 at dt = 0.02, 0.01, 0.005). The order is
   measured with a frozen envelope (one kick, `forcing_tau = 1e8`); the O-U run is
   asserted at 5e-3 only. z_spectral reaches the plan's 1e-3 at dt = 0.0025, not at
   0.005 (the kinetic-Alfvén rotation of Θ within a step is the first-order term).
3. F3.3 runs at fixed dt = 0.01: from rest the adaptive dt is the quiescent ceiling
   (~1), one step of which overshoots `t_end` before any forcing acts.
4. F3.2's per-stage check uses `dt = 1e-5` on its `Parameters`: that path caps the scale
   at `√(2·target/(F₂dt))` (`rmhd._scale_epilogue`), which binds on a one-kick envelope
   at dt = 0.01. The cap itself is RMHD's code and is not re-gated here.
5. `elsasser_powers` raises when `|σ_ε|·v̄ > 1` (the plan only noted realizability).
6. F3b not built, as agreed.

**Mutations (each applied, the named gate run, reverted).**

| mutation | caught by (magnitude) |
|---|---|
| κ missing in `forcing_scale` | F3.2: dE off by 0.66 (24 checks) |
| κ missing in `f_φ = f_χ/κ` | F3.2: 1.1–5.9 |
| κ inverted in `f_φ` | F3.2: 2.6–13 |
| κ missing in `_chi_psi` (both sites, consistently) | F3.2: 0.66 (28 checks) |
| `f_ψ → −f_ψ` in `ForcingTerm` | F3.2: per channel 0.88 and 4.0 |
| `ε₊ ↔ ε₋` in `elsasser_powers` (Θ± swapped) | F3.3 (H rate sign), elsasser_powers (rel 2.0) |
| `1/v_ph` dropped (`vbar = 1`) | F3.3: H/t = 0.46–0.49 of the request; F3.4's 0.5% check; elsasser_powers 0.52 |
| factor 2 on `σ_ε v̄` | F3.3, elsasser_powers (1.0) |
| arithmetic for harmonic shell mean | elsasser_powers only: 1.8e-2 (invisible to F3.3/F3.4, as it must be at 2%) |
| `yfac` dropped from the shell weights | elsasser_powers only: 1.2e-2 |

### F2 — diagnostics, implemented 2026-11-12 (uncommitted; adversarial review pending)

**Files.** `taranis/diagnostics/flrmhd.py` (new: `energy`, `helicity`, `imbalance`,
`perp_spectra`, `kperp_kz_spectrum`, `dissipation`, `injection`, `transfer`, `fluxes`,
`budget`, `spectral_break`), `taranis/diagnostics/__init__.py` (import + `__all__`, no
re-exports), `tests/test_flrmhd_diagnostics.py` (new), docs ("Diagnostics, flux and
budget conventions", including the warning that `diagnostics.energy`/`perpspec` are
RMHD's and silently wrong on an FLRMHD state).

| gate | fp64 (tolerance) | fp32 |
|---|---|---|
| F2.1 energies/helicity vs test-local forms; `E₊ + E₋ = total = E_ion + E_e + E_mag`; each piece vs its definition; spectra sums; bin-by-bin vs an independent numpy binning (3 configs: 2D non-square Z/τ = 2.86, 2D square, z_spectral non-cubic Z/τ = 0.5) | all ≤ 6.7e-16 (1e-13) | ≤ 2.3e-7 (2e-5) |
| ρ_i = 0 reduction | `E_ion`, `E_mag`, H, spectra == `diagnostics.rmhd` (1e-13), `E_e == 0` exactly, same bins as `perpspec` in a square box | same at 2e-5 |
| F2.2 known states (pure Θ±, Z/τ ∈ {2, 2.86}, kρ_i 0.2–7, non-square box) | `H v_ph/E = ±1`, `σ_c = ±1`, `E_H(k) v_ph = ±E(k)` (1e-13) | 5e-6 |
| F2.3 transfers bin-by-bin vs first principles (G-form `T_H`); `ΣT_E = ΣT_H = 0`; two-shell state `T_H v_ph = T₊ − T₋` | ≤ 3.2e-17 of Σ|T|; sums ≤ 9.5e-17 (1e-13); two-shell 1.4e-15, 4.1e-16 (1e-12) | ≤ 8.8e-9; 2.1e-7 |
| F2.4 flux vs the stepper, six K, X = E, E⁺, E⁻, H, h = 2e-4, 1e-4, 5e-5 | order 2.00 (1.99–2.00) in all eight; error at the finest h 7.1e-8 … 2.1e-7 (2D), 2.0e-8 … 4.6e-8 (z_spectral) of max|Π| (order 2 ± 0.3, 1e-6) | fp64 only |
| F2.5 sub-filter transfer (`kcut` on a bin edge and inside a bin) | vs first principles ≤ 4.4e-16; `−ΣT^sf = Π(kcut)` for E and H ≤ 1.1e-15; equals the full Π, Π_H at and beyond kcut 5.8e-16; `ΣT[f^<] = 0` ≤ 1.7e-16 | ≤ 6.4e-7 |
| F2.6 dissipation vs exact decay (2D hyper 2; z_spectral hyper 1/z_hyper 3 and hyper 2/z_hyper 2) | `d/dt` of the closed-form E(t), H(t) vs `−(ε_⊥ + ε_z)`, `−(ε_H,⊥ + ε_H,z)`: ≤ 6.7e-16 (1e-11); stepped E(t), H(t) vs the exponentials after 20 steps ≤ 5.9e-13 (1e-10); ν_φ ≠ ν_ψ formulas ≤ 8.9e-16 | ≤ 5.7e-7; 1.9e-6 |
| F2.7 injection vs independent rates; instantaneous budget closure (forced, dissipative, both modes, 3 configs) | ≤ 3.2e-16; closure ≤ 2.9e-16 of the rates (1e-12) | 1.0e-7; 7.7e-8 |
| F2.7 run-integrated budget, frozen envelope, dt = 8e-3, 4e-3, 2e-3 | ΔE: 2.3e-3 → 4.4e-4, order 1.20 (2D); 3.3e-3 → 8.1e-4, order 1.00 (z_spectral). ΔH: 1.2e-3 → 1.3e-4 (2D); 3.8e-4, 4.4e-4, 2.8e-4 (z_spectral, no clean order). Units: injected + dissipated. Asserted ≤ 1.0·dt at every dt, ≤ 5e-3 at the finest, E order in [0.7, 2.3] | fp64 only |
| F2.8 `kperp_kz_spectrum` sums, rows, bins; `spectral_break` on synthetic spectra | sums 1e-13; k* within 2%, slopes within 0.02 | same |

**Deviations and findings.**

1. **`E⁺` and `E⁻` are not separately conserved in FLR-MHD** (`H` is
   `Σ(E⁺_k − E⁻_k)/v_ph(k)`, not `E⁺ − E⁻`): `Σ_k T₊` is 7e-4 … 8e-3 of `Σ|T_E|` on the
   test states, `Π±` at the last edge is not a residual, and the per-channel sub-filter
   flux differs from the full `Π±(kcut)` by 3–5% of max|Π|. The identities gated are for
   E and H only. The plan did not say otherwise but §3.8/§11 did not flag it.
2. **Bin range.** Spectra run to `max(perpspec's k_max, largest dealiased k⊥)`, not to
   `perpspec`'s `k_max` as §3.8 says: identical in a square box, but in a non-square box
   `perpspec`'s range stops short of the dealiased band and the sums do not close
   (1.4e-2 of E on the test state). `perpspec` itself has that property; not changed.
3. **F2.4 step sizes.** At h = 1e-3 on an amplitude-0.5 state the O(h²) error is
   2e-5 … 5e-4, not 1e-6; the gate uses h = 2e-4 … 5e-5 and amplitude 0.2, where the
   plan's 1e-6 is met with order 2.00.
4. **F2.6** uses amplitude 1e-11 (at 1e-8 the nonlinear–dissipative interplay shows at
   1e-9 in the stepped comparison). The ν_φ ≠ ν_ψ closure against the solver's `L` is
   asserted at 1e-10 of the dissipation rates: it is the round-off of the cancelling
   wave terms, which are 10³–10⁴ times larger (measured 1.4e-12).
5. **F2.7 integrated budget.** "≤ O(dt)" is asserted as a bound with a stated constant
   plus the E order; the H error under z_spectral has no clean order (terms of both
   signs nearly cancel). Stiff sinks dominate the left-Riemann error (z_diss_k = 0.01:
   2.4e-2 at dt = 2e-3), so the gate uses rates ≲ 1.
6. `fluxes` returns `K` = the UPPER bin edges; `transfer`/`fluxes` both take `kcut=`
   (the sub-filter transfer is `T[f] − T[f^<]` on `k⊥ < kcut`); `injection` returns
   the cross terms AND the self-term coefficients (`dE = eps·dt + self·dt²`) rather
   than taking a dt; `budget`'s closure is against the solver's own full RHS; added
   `imbalance` (σ_H and σ_c). Every public function rejects a non-FLRMHD or overridden
   kgrid by name.
7. `kperp_kz_spectrum` has no FD-z branch (FD-z is F6); the per-`(kz, k⊥)` arrays are
   plain sums, not densities. `spectral_break`'s sensitivity on real spectra is
   untested (synthetic only).

**Mutations (each applied to `diagnostics/flrmhd.py`, gates run, reverted).**

| mutation | caught by (magnitude) |
|---|---|
| factor 2 in H | F2.1 (rel 1.0), 9 of 12 tests |
| sign of H | F2.1 (2.0) |
| `1/v_ph` dropped from the H density | F2.1 (1.5), F2.2 |
| `v_ph` for `1/v_ph` | F2.1 (6.5) |
| Θ± swapped | F2.1 (H 2.0, E± 0.34 of E) |
| κ missing in Θ | F2.1 (E₊ + E₋ vs total 2.4e-2, H 1.1e-3) |
| flux sign | F2.4 ONLY (error 2.0 of max|Π|, order 0) |
| cumulative sum from the wrong end | F2.3 (last entry not a residual: 1.4e-2), F2.4, F2.5 |
| K = bin centres | F2.3, F2.4, F2.5 |
| dissipation weight `ksq` for `G·pe` | F2.6 (0.21; 0.46 at ν_φ ≠ ν_ψ), closure |
| factor 2 in `ε_H,⊥`; factor ½ in `ε_H,z` | F2.6 (1.0; 0.5) |
| `z_hyper` ignored (kz⁴) | F2.6 (0.94), F2.8 |
| `E_e` coefficient ×2; `E_e` missing from total | F2.1 (0.19) |
| sub-filter mask dropped / moved | F2.5 (0.85; 0.42 of max|Π|) |
| self term ×2 | F2.7 (0.66) |
| `ε_z` sign in `budget` | F2.6/F2.7 closure (z_spectral cases only — invisible in 2D, where ε_z = 0) |
| `yfac` dropped | F2.1 (0.45) |
| ±kz not folded | F2.8 (0.47) |
| bins not extended | F2.1 (1.4e-2, non-square box only) |
| `eps_plus ↔ eps_minus` in `injection` | F2.7 (2.9e-2) |
| `apply_L` dropped from `budget`'s closure | F2.6/F2.7 (1.0) |

**Byte-identity (run after all edits, recording host, nothing regenerated).**
`test_refactor_reference.py`: 2 passed, 60 checks, fp64 and fp32, not host-skipped.
`test_particles_coupled.py`: 20 passed (fp64); 10 passed + 10 fp64-marked skips (fp32).
`test_forcing_modes.py`, `test_forcing_norm_per_step.py`, `test_forcing_smoke.py`: all
passed at both precisions. `rmhd.py` has no diff.

**Not run / not verified.** Full `make test` (left to the session); any test file other
than the FLRMHD five, `test_equation_interface.py`, the two frozen-reference files and
the three RMHD forcing files; `comm_backend="jax"`; the F1.1 "spinup and fp32
references"; `forcing_shell_noise=True` with FLRMHD; forced FLRMHD under `simulate`'s
snapshot path beyond F3.5's `simulate_scan`; momentum-mode helicity injection beyond the
2D quiescent case.

### F2/F3 review fixes — 2026-11-12 (uncommitted)

The adversarial review of F2/F3 found no physics or code bug; it found coverage holes and
two small defects. Every new gate was confirmed against the reviewer's mutation (applied,
gate run, file restored and its checksum compared — identical every time).

| # | fix | mutation now caught (magnitude) |
|---|---|---|
| 1 | F3.2 gains `_registry_checks`: `construct_rhs(equation_registry["FLRMHD"])` forced minus its unforced twin `== flrmhd.ForcingTerm` (6e-17 fp64, ≤ 1.6e-7 fp32; both normalization paths, 8 cells) and `forcing_scale_func is flrmhd.forcing_scale`. The O-U half of the F3.3 companion (`test_integrated_injection_matches_the_measured_change`) now runs at BOTH precisions (fp32: 3.0e-4 / 3.6e-5 in 2D, 9.7e-4 / 9.2e-4 z_spectral, against 5e-3); the order measurement is its own fp64 test | `rmhd.ForcingTerm` registered, run at fp32: 8 registry checks fail at 8.4e-2 … 1.9e-1 (tol 2e-3); Σinjection vs measured ΔE/ΔH 9.2e-3 … 1.7e-2 (tol 5e-3) |
| 2 | `injection` under `forcing_norm_per_step=False` — see below. New `test_injection_per_stage_normalization` (both precisions) and `test_run_integrated_injection_per_stage` (fp64) | F halved in that mode: 0.5 on every `eps*`, closure 1.0, capped self 0.75. Per-step formula restored (always add the quadratic form): `self == 0` fails in all 6 cells, run-integrated ΔE off by 2.5e-2 → 5.3e-3 against 0.5·dt |
| 3 | `grids.py`: the shell is now `grids._forcing_shell(ksq, params)` (called by `setup_kgrids`) and `grids.forcing_shell(params)`; `elsasser_powers` takes its shell from the latter (`flrmhd._forcing_shell_vbar`). `test_elsasser_powers_shell_is_the_solvers`: `shell == kgrid.fmask` for 5 shells in 4 boxes (Lx > Ly, Ly > Lx, 3D, square), both precisions; at fp32 the float64 shell differs from the solver's in 6 of the 20 combinations, with v̄ off by up to 23% | shell rebuilt with `kunit = 2π/Lx`: fails in the Ly > Lx box (v̄ off by 0.73). The pre-fix float64 shell, at fp32: fails in 2 boxes (v̄ off by 1.1e-2 and 0.23) |
| 4 | `test_subfilter_edge_convention_at_default_bins`: square 2π box, `bin_factor = 2.0`, kcut = 2 and 4 with 3 modes exactly on each: `Pi_sf(end) == Pi(kcut)` (≤ 1e-15 fp64, ≤ 1.7e-6 fp32) | `< kcut → <= kcut`: 7.6e-2, 2.3e-1 (2D), 0.87, 0.45 (z_spectral) of max|Π| |
| 5 | F2.8: every `(|kz|, k⊥)` cell of `E` and `D` against `np.histogram2d` of the per-mode `E_k`, `D_k` (≤ 7e-17); `kperp_kz_spectrum` asserts `dims == 2 or z_spectral` (tested with a stand-in Parameters) | `D` with the kx axis reversed (kz marginals and total intact): 8e-2 of ε, seen by nothing else in 2D. `E` replaced by the outer product of its marginals: 2.0e-2 of E, seen by nothing else. Assert removed: caught |
| 6 | `spectral_break(kmin=)`: a forcing-range bump at k ≤ 2.5 pulls the free fit to k* = 1.77; with `kmin = 3` it returns 9.01 (true 9) | `kmin` ignored: k* = 1.77, slopes +1.36 / −3.41 |
| 7 | `test_dead_z_hyper_warns`: `overrides={"rho_i": …}` and `{"tau", "diss"}` with a dead `z_hyper` still warn | early return on ANY overrides: both checks fail (no warning) |
| 8 | `elsasser_powers`: `|σ_ε|·v̄` compared with 8 ulp of slack, the vanishing channel clamped at 0. Gate: σ_ε = ±1/v̄ and one ulp above, 80 parameter sets — 62 of 160 have the product above 1 | bare `> 1.0`: raises in those 62; clamp removed: negative ε₋ |
| 9 | `_check_kgrid`: FLRMHD `Parameters` + RMHD kgrid is a ValueError naming the missing FLR factors (all 10 functions); docstring and docs state that a kgrid from a DIFFERENT FLRMHD `Parameters` cannot be detected | `kgrid.eq is None` test dropped: `AttributeError` instead |
| 10 | The ν_φ ≠ ν_ψ closure at fp32 was a bare 2e-2. Measured floors: 1.5e-7, 4.2e-7, 4.9e-7 of the summed |per-mode rates| and 1.5e-7, 7.6e-4, 2.0e-6 of the dissipation rates. Now asserted at 4e-6 and 4e-3 (fp64: 1e-14 and 1e-10). The second cannot go below ~1e-3 in the hyper = 1 z_spectral case: the cancelling wave terms are 2e3 × ε_⊥ | 1% on L's ψ dissipation, fp32: 5.1e-3 and 7.8e-4 (old threshold passed both) |

**Item 2, what is correct under per-stage normalization.** The scale is re-solved at
every stage as `target/P(f_stage)`, so the instantaneous power IS the target throughout
the step and the step injects `eps·dt`: there is no self term. The documented
`dE = eps·dt + self·dt²` is the per-step path's (F frozen over the step) and overshoots
here at first order (measured 2.5e-2, 1.2e-2, 5.3e-3 at dt = 0.004, 0.002, 0.001 in 2D;
1.6e-2, 8.2e-3, 4.0e-3 z_spectral). The exception is a channel whose scale sits at its
cap (`rmhd._scale_epilogue`; P ≈ 0 — a start from rest, or an envelope momentarily
orthogonal to the field: 12–20% of the steps in the frozen-envelope gate): the capped
scale is state-independent, F is frozen and that channel's quadratic form IS its self
term (from rest at fixed dt, `self·dt² = target·dt` exactly, gated at 1e-16). So
`injection` now returns `self* = 0` in that mode except for a capped channel, recognised
by its cross term missing the target (relative √eps of the precision; momentum mode has
one scale). With that rule the run-integrated sum closes to 7.5e-4, 8.8e-5, 1.2e-4 (2D)
and 3.3e-4, 4.5e-5, 3.1e-6 (z_spectral) for E; cross-only (no cap rule) was −1.2e-2 …
−2.4e-3 in the scratch measurement. ΔH has no exact statement in this mode (first order:
3.2e-3 … 1.7e-4). `self_H` is now formed per channel (`ΣE⁺_k[F]/v_ph − ΣE⁻_k[F]/v_ph`).
Docs rewritten accordingly; the per-step path is unchanged.

**Docs.** "Invariants": `E⁺ − E⁻ = Σ_k v_ph(k)·H_k`, the nonlinearity conserves `Σ H_k`
while moving `H_k` across `k⊥`, so `E±` are not separately conserved at finite ρ_i.
"Forcing": the instantaneous `ε_H/ε` scatters (largest on the 13–18% of steps where a
cross term is negative — the review's figure), only the time mean is controlled, and it
converges on `forcing_tau`; the solver's shell; the σ_ε = 1/v̄ boundary.

**Deviations.** (i) `taranis/grids.py` was edited (`_perp_ksq`, `_forcing_shell`,
`forcing_shell`): the mask EXPRESSION is unchanged and only moved, so `kgrid.fmask` is
what it was — proved by the frozen references below and the RMHD forcing files. (ii) The
"shell differs in 20 of 112 combinations" census of the review was not reproduced; the
gate's own census is 6 of 20. (iii) `tests/_flrmhd_testing.shell_vbar` takes an optional
mask, and `test_elsasser_powers` now averages over the solver's.

**Tests (one process at a time, recording host `Alfreds-MacBook-Pro.local`).** fp64:
forcing 12 passed, diagnostics 15, operators 16, linear 3, conservation 5. fp32: forcing
11 passed + 1 fp64-marked skip, diagnostics 12 + 3, operators 12 + 4, linear 3,
conservation 4 + 1. `test_refactor_reference.py`: 2 passed, 60 checks, at fp64 and at
fp32, not host-skipped, nothing regenerated. `test_particles_coupled.py`: 20 passed
(fp64), 10 passed + 10 fp64-marked skips (fp32). `test_forcing_modes/_smoke/
_norm_per_step` + `test_equation_interface`: 23 passed (fp64); the three forcing files 17
passed (fp32). `rmhd.py` has no diff. Full `make test` not run (left to the session).

### F7 — run script and notebook, 2026-11-12: SCRIPT AND SMOKE ONLY, PRODUCTION RUNS NOT DONE

**No pass criterion F7.1–F7.5 has been evaluated.** What exists is the machinery.

**Files.** `examples/helicity_barrier_run.py` (resumable driver; profiles
smoke / m1 / full by `TARANIS_HB_PROFILE`; runs `imb` / `bal` / `rmhd` selectable by
argv[1] or `TARANIS_HB_RUN`; output root argv[2] / `TARANIS_HB_ROOT`, default
`examples/data/helicity-barrier`, gitignored) and `examples/helicity-barrier.ipynb`
(committed without outputs; reads whatever exists, computes the F7 verdict lines, labels
every smoke number as not physics).

**Design.** Own block loop on `run.block_of_steps` (the time series needs a row per
block, which `simulate_scan` does not emit): each jitted block scans sub-blocks of
`cfl_every` steps, accumulating `injection`'s `(eps + self·dt)·dt` at each sub-block
start (every step when `cfl_every = 1`), then the block-end energies, helicity and
dissipation. Two phases with their own snapshot directories, because `cfl_every` is part
of the recorded `Parameters`: `spinup/` at `cfl_every = 1` for 1 τ_A (0.5 smoke), then
`main/` at `cfl_every = 4` (2 smoke). Snapshots on an absolute schedule (multiples of
`t_snap`), each followed by its spectra file: `E±(k)`, `E_H(k)`, `Π`, `Π±`, `Π_H`, the
`(kz, k⊥)` energy and dissipation spectra, and the sub-filter transfers and cumulative
fluxes at `k_c` (bin edge nearest 0.5/ρ_i, in [3, 1/ρ_i): 5 for m1). Restart: newest
snapshot, time-series rows past it dropped, spectra past it removed.

**Choices not in the paper (all env knobs, recorded in meta.json) — [DECISION FOR
ALFRED].** `eps = 1/(10 τ_A)` so that `E_0 = 10 ε τ_A = 1` (a critically balanced outer
scale; only `E_0/(ε τ_A)` is fixed by the paper). `ν = ε^(1/3) k_d^(-16/3)`,
`k_d = 0.7·N/3` (m1: 1.38e-7); `ν_z·(N/3)⁶ = 1` (m1: 1.06e-8). `forcing_tau = 1`.
`fshell = (0.5, 2.1)` — every `0 < k⊥ ≤ 2` with no mode on an edge. IC: random φ, ψ on
`0 < k⊥ ≤ 2`, `kz ∈ {0, ±1}`, statistically balanced. **None of ν, ν_z has been checked
against a developed m1 state** (only 0.5 τ_A was stepped, for timing).

**Smoke evidence (24³, k_min ρ_i = 0.2, 2 τ_A; fp64).** Three runs: 550 / 610 / 330
steps, 26 s wall for all three, peak RSS 1.33 GB, 2.4M on disk for the three. Restart:
(a) a rerun of a finished run is a no-op; (b) SIGKILL mid-`main` (35 rows on disk, last
at t = 6.50; newest snapshot at t = 4.72) then rerun; (c) three invocations under
`TARANIS_HB_WALL=9`. In (b) and (c) the final `timeseries.txt` has 56 rows, strictly
increasing t, no duplicate, contiguous step counts, and is IDENTICAL value for value to
the uninterrupted run's. Realised injection `inj_E/ε` = 0.9988, 1.0021, 1.0012 and
`inj_H/inj_E` = 0.869, −0.010, 0.875 (requests 0.88, 0, 0.88; 2 τ_A is ~12 forcing
times); block budget `ΔE − (inj_E − D)Δt` nets to −1.0%, +0.2%, −1.4% of ε·T with
dissipation 2–10 ε. The notebook executes without error on this output (5 figures) and
on an empty root. The smoke grid has 4 bins in the break-fit range, below the notebook's
6-bin minimum, so the `k*` fit and the `k*ρ_i` scatter are NOT drawn on smoke output;
that code path ran once without error with the minimum temporarily at 4 (Agg backend).

**m1 cost, MEASURED (M1 laptop, `<scratch>/flrmhd-f7/m1_cost.py`: the script's own
block function on the 64³ configuration from its IC, nothing written; after compile).**

| | fp64 `imb` | fp64 `bal` | fp64 `rmhd` | fp32 `imb` |
|---|---|---|---|---|
| spinup phase, `cfl_every = 1`, 200 steps | 71.9 ms/step | 73.0 | 73.0 | 49.3 |
| main phase, `cfl_every = 4`, 400 steps | 58.2 ms/step | 58.6 | 57.9 | 34.5 |
| dt, mean over the main-phase steps (last block) | 4.49e-3 (4.87e-3) | 4.50e-3 (4.88e-3) | 8.63e-3 (1.02e-2) | 4.02e-3 (3.71e-3) |
| peak RSS | 1.17 GB | 1.05 GB | 1.03 GB | 0.83 GB |

Compile 2.2 s per phase; `spectra()` 0.25 s per snapshot. These dt are from the first
0.5 τ_A (1.0 for `rmhd`), where the IC is still decaying (E/(ε τ_A) 10 → 8): **dt has
not settled**, and in the barrier run E is expected to grow, shrinking dt further.
At the measured main-phase rate and dt: FLR runs ~1290–1400 steps per τ_A = **75–81 s
per τ_A, 30 τ_A in 38–41 min, 200 τ_A in 4.2–4.5 h** each (fp64); `rmhd` ~620–730 steps
per τ_A = 36–42 s per τ_A, 30 τ_A in ~20 min, 200 τ_A in ~2.2 h. A factor-3 growth of
the rms amplitude in `imb` would triple its figures. Plan §10's estimate (70 ms/step,
1400 steps/τ_A, 100 s/τ_A) stands to within 25%; its "memory < 0.3 GB" does not (1.2 GB
resident, mostly the runtime).

**Not done / not verified.** The m1 and full runs; any F7 pass criterion; adequacy of
ν and ν_z at m1; convergence in `cfl_safety` and `cfl_every` (§9); the `full` profile
beyond parameter construction (not even instantiated); fp32 beyond the timing; the
ν_z scan (F7.5: `TARANIS_HB_NUZ` with a separate `TARANIS_HB_ROOT` per value); the
zero-padding refinement helper of §10; an adversarial review of this rung.

## 7. Mutation table (what a reviewer should try, and the gate that must fail)

| mutation | caught by |
|---|---|
| sign of a bracket in `NL_n` or `NL_psi` | F1.3 (O(1) vs RMHD), F1.6 |
| `phi` instead of `phi_e` in the Ohm bracket (drops `-rho_s^2{psi,nhat}`) | F1.6 (E transfer nonzero); invisible to F1.3 |
| sign of the `rho_s^2` term (`pe = 1 - rho_s^2 G`) | F1.6 (7.5e-2), F1.5, F1.2 |
| Gamma_0 dropped / replaced (`G = ksq`, Pade, missing exp, alpha without 1/2) | F1.2, F1.4 (coefficient), F1.5 |
| `Z/tau` inverted, or `rho_s^2 = rho_i^2/tau` (factor 2) | F1.4, F1.5 at two Z/tau |
| `a` and `pe` swapped in L (same eigenvalues!) | F1.5(b) eigenvector, F1.3 is blind at rho_i = 0, F1.7 (H drifts) |
| `inv_ksq` left in place of `inv_G` | F1.5, F1.6 |
| Nyquist kz not zeroed | `propagators._check_hermitian_compatible` at setup; reality gate F1.11 |
| `z_hyper` ignored (kz^4 always) | F1.5(d) |
| `v_ph,max` dropped from the CFL | F1.9 discriminator |
| factor 2 or sign in `H`; `v_ph` for `1/v_ph` | F1.8, F2 known states, F3.3 |
| `Theta^+-` swapped in forcing or spectra | F1.5(c), F3.3 sign |
| kappa missing in `forcing_scale` or in `f_phi = f_chi/kappa` | F3.2; invisible to F3.1 |
| flux sign / cumulative sum from the wrong end | F2 flux-vs-stepper |
| dissipation weight `ksq` for `G pe` | F2 exact decay |
| a read site bypassing `getp` | F1.10 independent route |
| new slot leaking into RMHD graphs | F1.1 HLO histograms |

Blind spots stated so the reviewer attacks them: every rho_i = 0 gate is blind to
FLR-only terms; dispersion alone is blind to `a <-> pe`; conservation alone is blind to
any mutation applied consistently to solver AND diagnostics (hence the test-local
weights in F1.6 and the independent Gamma_0 everywhere).

## 8. Byte-identity of the existing paths

Shared edits and why each leaves existing graphs alone: (1) `EquationRecipe.kgrid_func`
appended with default None — existing registry entries are unchanged tuples; (2)
`K_Grids.eq` appended with default None — None is a leafless pytree, `kgrid_specs` and
`_kgrid_to_global` already map None to None, and no code constructs `K_Grids`
positionally (only `grids.py:117/152/163`, all keyword); (3) `setup_kgrids` calls the
hook only when the recipe has one; (4) `overrides._table` gains an `elif`; (5) the
registry gains a key. `rmhd.py` is imported from, never edited. Proof obligation = gate
F1.1 on the recording host, both precisions, with NO reference regenerated; a red frozen
gate is a bug in the rung by definition. One thing to check rather than assume:
`functools.lru_cache`/jit caches keyed on `K_Grids` structure are per-process, so the
only persistent artifacts are the references themselves.

## 9. Risks and traps

- `1 - i0e(alpha)` cancellation at small alpha (measured). The series branch is not
  optional, and `a`, `kappa` must not be formed as products of rounded factors.
- fp32: `G`, `a` are computed in fp64 and cast; KAW phases `kz v_ph tau` at k rho_i = 10
  and large kz lose fp32 digits in Putzer2's complex exp — gate F1.5 reports the fp32
  floor rather than assuming one.
- Putzer2 at `k_perp = 0`: with `a = pe = 1` there, `s = i kz` as in RMHD (no new
  `s = 0` set). If the implementer instead sets `a(0) = 0`, `s^2 = 0` on a whole kz line
  and the `sqrt` jvp guard noted in `putzer2_precompute` becomes live for the harness.
- IF accuracy: linear KAWs are exact at any dt, but nonlinear coupling of modes with
  `omega dt >~ 1` is misweighted; the CFL of §3.5 keeps `omega dt` O(1) only through
  `v_ph,max |b_perp|`, not through `kz v_ph`. Check F7 convergence in `cfl_safety`.
- `cfl_every > 1` + hoisting is the efficient path but overshoots from rest.
- The saturated state is unphysical by the paper's own account (it relies on parallel
  hyperdissipation breaking the ordering). The pseudo-stationary phase is the science
  target; do not tune `nu_z` to make saturation "look right".
- `sigma_c ~ H/E` only when large scales dominate; report both `H/E` and
  `(E^+ - E^-)/(E^+ + E^-)`.
- Forcing differs from the paper's (O-U vs negative damping), so E(t) curves are not
  expected to match trace-for-trace; the gates of F7 are forcing-agnostic by design.

## 10. Cost

**Per step.** 10 transforms per RHS, as RMHD; the only extras are four `(nkx,nky)`
multiplies. So FLRMHD/z_spectral ~ RMHD nu != eta z_spectral (Putzer2). Measured anchors
(docs/performance.md, M1, fp64, lsrk54): RMHD z_spectral separable 55.8 ms/step at
128^2 x 16 and 237 ms/step at 256^2 x 16, i.e. ~0.21–0.23 us per grid point per step;
Putzer2 hoisted memory ~44 u. FLRMHD numbers are **estimates until F5** [UNVERIFIED].

**The paper's rho_i.** The paper writes "rho_i = 0.02 L_perp" and "0.1 L_perp". Its own
numbers — `k_crit = 22.4 (2 pi/L_perp)` where `1/v_ph = 0.88`; "rho_i lies only slightly
above the resolution cutoff" at N_perp = 256; `k* rho_i ~ 0.15` inside the box — are
consistent only with `rho_i (2 pi/L_perp) = 0.02` (then `k_crit rho_i = 0.448`,
`1/v_ph = 0.885`), not with `rho_i/L_perp = 0.02` (which would put `1/v_ph = 0.88` at
mode 3.7, where instead `k rho_i = 2.8` and `1/v_ph = 0.28`). This plan therefore uses **`k_min rho_i = 0.02` and `0.1`**, i.e.
`rho_i = 0.02`, `0.1` in a 2 pi box. [DERIVED inference, flagged in §12 — it changes
`v_ph,max` from ~13 to ~2.3 and the cost by the same factor.]

**dt.** `dt ~ cfl_safety dx/(max|grad phi_e| + v_ph,max max|grad psi|)`. With
b_rms ~ 1 (critical balance in a cubic box), peak/rms ~ 3, cfl_safety 0.5: 64^2:
`dt ~ 4e-3` (~1400 steps per tau_A = 2 pi); 128^2: ~2800; 256^2: ~5600 steps per tau_A.
Amplitudes grow in the barrier runs, so these are optimistic by up to ~3x late on.

| run | grid | k_min rho_i | steps / tau_A | wall | where |
|---|---|---|---|---|---|
| smoke (F4) | 32^2 x 32 | 0.2 | ~700 | minutes | M1, `make test -m slow` |
| **M1 science (paper Fig. 1b setup)** | 64^2 x 64 | 0.1 | ~1400 | ~70 ms/step: ~100 s per tau_A, 30 tau_A in ~1 h, 200 tau_A (saturation) in ~6 h; x3 runs; nu_z scan x5 | M1, 16 GB (memory < 0.3 GB) |
| M1 stretch | 128^3 | 0.04–0.05 | ~2800 | ~0.5 s/step: ~25 min per tau_A, 30 tau_A in ~12 h; saturation not reachable | M1 (u = 17 MB, ~0.8 GB) |
| paper Fig. 3 | 256^3 | 0.02 | ~5600 | M1: ~4–5 s/step = ~7 h per tau_A — NOT a laptop run | one GPU; memory 44 u = 6 GB fp64 / 3 GB fp32 |
| beyond | 512^3+ | 0.02 | ~11000 | — | single A100 in fp32 z_spectral plausible on memory (lsrk33 ~35 u x 0.54 GB = 19 GB) [UNVERIFIED]; otherwise F6 + `"jax"` |

So: the barrier's qualitative signature (gates F7.1–F7.4, and the nu_z scaling F7.5) is
an M1 job at the paper's own low-resolution setup (N_perp = 64, N_z <= 256,
k_min rho_i = 0.1). The 256^3 time histories to saturation (200 tau_A ~ 1.1e6 steps)
and anything showing the re-emerging KAW range need a GPU allocation; single-GPU
z_spectral ms/step at 256^3 has not been measured for any equation set, so no SU
estimate is given here [UNVERIFIED]. The paper's recursive refinement (restart at twice
the resolution) is a k-space zero-padding helper on a snapshot — small, proposed as part
of F7's script rather than the core.

## 11. What steps 2–3 will need from this (sketch only)

- **One recipe for truth and closure.** Because `rho_i = 0` FLRMHD is RMHD with the
  same forcing and diagnostics, step 2's "RMHD + cutoff closure" should run as FLRMHD
  at `rho_i = 0` with an extra `Term(flrmhd.ClosureTerm, active=closure_active)` (and,
  for a k-local LINEAR closure, entries in the Putzer2 L — it takes any 2x2). The
  comparison FLR-truth vs closure is then the same code path with one switch.
- **Truth-data products** (define in F7's script so the first production runs already
  store them): time series of `E^+-`, `H`, `eps_perp`, `eps_z`, `eps`, `eps_H`;
  `E^+-(k_perp)`, `E_H(k_perp)`, `Pi(k)`, `Pi_H(k)`, `Pi_+-(k)` at snapshot cadence;
  2D `(k_perp, kz)` dissipation spectra; and, for a filter scale `k_c < 1/rho_i`,
  the sub-filter transfers `T_E^{>k_c}`, `T_H^{>k_c}` into the retained band (one extra
  RHS with a masked state — add `fluxes(..., kcut=)`), which is the a-priori target a
  closure must reproduce: energy leaving at ~`2 eps^-` with ~zero net helicity removal.
- **A helicity-aware closure needs `H` and `eps_H` conventions fixed now** (§2.4, §2.5)
  and the per-mode identity `T_H = (T_+ - T_-)/v_ph`; at `rho_i = 0` this is the
  cross-helicity budget.
- **Traced closure coefficients (step 3).** `Parameters` is static and identity-hashed,
  and the existing `overrides` seam is deliberately rejected by `run.py` and by forcing.
  Step 3 needs a sibling seam that the solver ACCEPTS: a `K_Grids.closure` pytree
  (coefficients theta, possibly traced, fixed structure) read only by `ClosureTerm` /
  the closure's L entries behind `if kgrid.closure is not None:`, built by
  `setup_kgrids(params, closure=theta)`, with `closure=None` byte-identical — the same
  discipline as `overrides.getp`, same gates (refactor reference; independent route).
  A k-local linear closure in L must keep the hermitian check and stays hoistable only
  if theta is frozen over the block.
- **Differentiating through the solver**: fixed dt (adaptive dt and `cfl_every`
  collectives are not differentiable choices), short windows from stored truth states
  (reverse mode through `lax.scan` stores every step; forward mode is fine for a handful
  of coefficients, which is where a hand-built closure with 2–5 parameters sits),
  `donate_argnums` off or states re-materialised per window, and deterministic forcing:
  the O-U stream is reproducible from the stored `forcing_key`, but a deterministic
  negative-damping forcing (F3b) removes RNG from the graph altogether.
- **Loss definitions** want the F2 diagnostics as differentiable jnp functions (they
  are, except `_binned`'s histogram — binning with fixed integer shells is
  differentiable in the weights; check before relying on it).

## 12. Unverified points and decisions (for the reviewing session)

Decisions for Alfred:

1. §3.3: `K_Grids.eq` slot + `kgrid_func` recipe hook (edits `grids.py`,
   `physics/__init__.py`) vs computing the factors in-graph (no shared edits).
2. §3.7/F3b: is O-U generalized-Elsasser forcing with `eps_H` exact only to the shell's
   `v_ph` spread (0.1–1%) acceptable, or is the paper's negative-damping forcing wanted?
3. §3.4: reject CB-IMEX for FLRMHD + z_spectral outright (recommended), or only warn?
4. `z_hyper` default 2 (RMHD-compatible) with 3 set explicitly for paper-matching runs.
5. FD-z (F6) deferred until a multi-device run is actually scheduled.
6. eqpars spelling `rho_i`, `tau`, `Z` (only `Z/tau` and `rho_i` matter dynamically).

Not verified:

- **Paper's rho_i normalisation** (§10): inferred `k_min rho_i = 0.02` from internal
  consistency of three statements; the paper's literal text says "0.02 L_perp".
- **E_sat scaling exponent** (§3.6): the arXiv TeX prints `(eps/nu_6z)^(-1/4)`; prose,
  figure description and my re-derivation give `nu_6z^(-1/4)`. Published JPP version
  not checked.
- **Paper's numerical values**: `eps`, `nu_6perp`, `nu_6z` per run are not tabulated in
  the text read (only `1/nu_6perp ~ 3e10` for the reference run and the sweep range);
  the forcing's exact functional form is described in words only; the precise
  definition of the flux `Pi(k)` used in the paper's figures is not written out (the
  one in §3.8 is the standard shell-transfer definition).
- **Squire et al. 2022 (Nature Astronomy)** was NOT read; nothing here depends on it.
  Zocco & Schekochihin 2011 and Schekochihin et al. 2019 (the derivation of FLR-MHD)
  were NOT read: the equations are taken from Meyrand et al. as printed, and the
  normalisation, invariants in (phi, psi), `rho_s^2 = Z rho_i^2/(2 tau)` and the
  four-gradient form are my derivation (numerically checked, §2.8, in 2D with a single
  kz phase factor — not in a 3D time-stepped run).
- JPP volume/article number quoted in the header is from memory, not from the source.
- Whether `rho_i = 0` FLRMHD is BITWISE RMHD in any configuration (gate F1.3 asserts
  1e-12 and reports).
- Every cost number for FLRMHD (ms/step, memory, GPU feasibility at 256^3/512^3) is
  scaled from RMHD measurements; the dt estimates assume b_rms ~ 1 and peak/rms ~ 3.
- Whether the barrier is visible at 32^2 (F4) or cleanly at 64^2 x 64 within ~30 tau_A
  on this forcing (the paper's N_perp = 64 runs used N_z up to 256 and long times).
- Sensitivity of `k*` to the break estimator; fp32 floors of the dispersion gate;
  `jax.scipy.special.i0e` accuracy at alpha > 700 and its jvp (needed by the
  `rho_i` override) were not tested beyond a single evaluation.
- The claim that no test or tool constructs `K_Grids` positionally was checked by grep
  over `taranis/ tests/ bench/ examples/*.py`, not notebooks.
- The stability harness's compatibility with a recipe carrying `kgrid.eq` under
  `setup_kgrids(overrides=)` forward-over-forward (`dparams`) is expected, not tested.

## 13. Execution rules digest (for implementer agents)

As CMHD_PLAN §9, plus: own scratch subdirectory per agent, never `pkill` by pattern;
`rmhd.py` is read-only for this plan; any red frozen reference means stop and report;
performance numbers only from same-session interleaved A/B; physics written in Gaussian
units first in the docs (the user's standing rule), then in code variables; plotted
magnetic fields in tesla; every rung ends with docs + a dated note in §6 + a fresh
adversarial review with the reviewer's own mutations from §7 and at least three not
listed there.

## 14. Session review of this draft (2026-11-11, main session; not an adversarial rung review)

Checked by hand, independently of the scratch script:

- §2.2 re-derived from the Gaussian equations of §2.1: `nhat = -G phi` with
  `G = (2/rho_i^2)(1 - Gamma_0)`, the continuity coefficient `= v_A`, the pressure
  coefficient `= rho_s^2 = Z rho_i^2/(2 tau)`, both evolution equations and their signs,
  `v_ph^2 = a pe`, `Theta^+- = kappa phi +- psi`, and the (phi, psi) forms of E and H all
  agree with the draft.
- The `rho_i` reading (§10): `1/v_ph = 0.88` needs `k rho_i ~ 0.45` (small-alpha expansion
  `v_ph^2 ~ 1 + 2.75 alpha` at Z/tau = 2), so `k_crit = 22.4 (2 pi/L)` gives
  `rho_i = 0.02 L/(2 pi)`. The draft's reading stands; still worth one line from Alfred.
- `E_sat`: the paper's own balance (`nu_6z l_par^-6 Z+^2 ~ eps`, `l_par0 ~ L v_A/E^(1/2)`)
  gives `E^4 ∝ eps/nu_6z`, so `nu_6z^(-1/4)` at fixed eps is wrong as printed in TeX line
  624 and right in the prose; §3.6's `+1/4` on `eps/nu_6z` is correct.
- Code assumptions: `rmhd._forcing_scale_from(fields, f_raw, kgrid, params, dt)` takes the
  fields as an argument, so §3.7's `stack([kappa*phi, psi])` reuse is real;
  `K_Grids.overrides` is the precedent for an optional None-default slot (§3.3);
  `Putzer2Operator._coeffs` has a Taylor branch at `s*tau -> 0`.

Changes requested before F0 is closed:

1. **Add an independent-transcription gate to F1** (F1.6b): the nonlinear RHS at finite
   `rho_i` against a test-local numpy transcription written in the PAPER's variables
   (`dn_e/n_0e`, `A_par`, Gaussian coefficients), 1e-12 on a broadband state. As drafted,
   the finite-`rho_i` nonlinear term is pinned only by conservation plus the two limits.
2. **§9's `s = 0` note is incomplete.** With `nu_phi == nu_psi` — the paper's and the
   production case — `s^2 = -kz^2 v_ph^2` vanishes on the whole `kz = 0` plane (and the
   zeroed Nyquist plane), not only on the `k_perp = 0` line. `apply_exp` is safe there
   (Taylor branch); the `jnp.sqrt(s2)` tangent in `putzer2_precompute` is not, so F1.10's
   `dJ/d rho_i` gate must include `kz = 0` modes and assert finiteness explicitly.
3. **Shortest path to the science** is F0, F1, F2, F3, F7. F4 is a regression convenience,
   F5 and F6 stay deferred until a measurement or a multi-device run asks for them.

**Decided by Alfred, 2026-11-11**: §12 decisions 1–6 as recommended below — the
`K_Grids.eq` slot, O-U forcing first (F3b deferred), CB-IMEX + z_spectral rejected,
`z_hyper` default 2, FD-z deferred, eqpars `rho_i`/`tau`/`Z`. Alfred also confirmed the
§10 reading of the paper's `rho_i` (`rho_i = 0.02 L/(2 pi)`, i.e. `k_min rho_i = 0.02`).
Execution order agreed: F0, F1, F2, F3, F7; F4–F6 deferred.

**Bessel evaluation (measured 2026-11-11, jax 0.10.0, M1, against mpmath at 40 digits).**
Alfred flagged problems with jax's built-in Bessels; those are in
`jax.scipy.special.bessel_jn` (fixed start order, overflow — documented in
`~/gridfire/code/kinetic-dispersion/jaxbessel/README.md`), not in the order-0 function
this plan needs. On alpha in [1e-12, 1e6]:

| | `jax.scipy.special.i0e` | `jaxbessel.ive_table(x, 0)` |
|---|---|---|
| max rel. error fp64 / fp32 | 6.7e-16 / 2.8e-7 | 8.2e-16 / 5.0e-7 |
| first derivative, max rel. error | 4.4e-12 | 2.7e-10 |
| second derivative (forward-over-forward) | finite, 16 digits at x = 1 | finite, 15 digits |
| non-finite outputs | 0 | 0 |
| 256x129 array, jitted | 0.68 ms | 11.7 ms |

So `i0e` is adequate for `Gamma_0` and the plan keeps it: no new dependency. Both lose
`1 - Gamma_0` to cancellation equally (1e-10 at alpha ~ 1e-6, 3e-4 below), so the
small-alpha branch of §4.2 is needed with either. jaxbessel does offer a cancellation-free
route, `1 - Gamma_0 = 2 sum_{k>=1} Ie_k` (4e-16 for alpha < 1 with N = 200; it needs
N ~ sqrt(80 alpha) so use `1 - Ie_0` above alpha ~ 1), which would replace the hand
series if a later model needs `Gamma_n` for n > 0 and takes the dependency anyway. For
gate F1.2 the quadrature reference must be written as
`1 - Gamma_0 = -(1/pi) int_0^pi expm1(-alpha (1 - cos t)) dt` so the reference itself
does not cancel.

Recommendations on §12's decisions as originally put: take the `K_Grids.eq` slot; reject
CB-IMEX with z_spectral; keep `z_hyper` default 2; defer FD-z; ship O-U forcing first and
build F3b (negative damping) only when a trace-for-trace comparison with the paper's or
with hybrid-kinetic runs is wanted, or when step 3 needs an RNG-free graph.
