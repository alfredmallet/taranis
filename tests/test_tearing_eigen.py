# plans/AUTODIFF_PLAN.md rung 1a gate: no-shear resistive tearing of Psi0 = sech^2(x) through
# the eigen-harness (taranis/stability.py), against exact theory and the Julia reference.
# The science lives in examples/tearing-eigen.ipynb (data: examples/tearing_eigen_run.py);
# this file is its fast standing gate and shares no code with it.
#   (i)   the exact outer-region Delta'a = 2(5 - k^2)(k^2 + 3)/(k^2 sqrt(k^2 + 4)) (Poschl-Teller
#         l = 3) against a numerical shooting solution of psi'' = (k^2 + f''/f) psi, with f''/f
#         itself formed by finite differences of f = -2 tanh sech^2 (not the closed form).
#   (ii)  S = 1e3, ka = 1, nx = 512, Lx = 20: the leading eigenvalue of the ky block is real,
#         is the fastest mode of the whole block spectrum, has a round-off residual, and equals
#         Alfred's Julia eigencode (tests/data/shear_tearing_reference.npz, alpha = 0) to that
#         reference's recorded RELATIVE error `gamma_err`, taken about its Richardson-
#         extrapolated value `gamma_richardson` (the base-grid `gamma` differs from it by about
#         gamma_err). Observed: 3e-7 relative here (6e-7 grid-converged), so a
#         1e-5 regression pin is added.
#   (iii) the ideal marginal point: at S = 1e3 the fastest mode is unstable at ka = 2.0 and
#         nothing in the block spectrum grows at ka = 2.5 (theory: Delta' = 0 at ka = sqrt 5).
# fp64 only. pytest: `pytest tests/test_tearing_eigen.py`; script: `python tests/test_tearing_eigen.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import jax.numpy as jnp

import taranis as jr
from taranis import stability

_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "shear_tearing_reference.npz")
_LX = 20.0
_NX = 512


def _dprime_exact(ka):
    q = ka*ka
    return 2.0*(5.0 - q)*(q + 3.0)/(q*np.sqrt(q + 4.0))


def _f(x):
    return -2.0*np.tanh(x)/np.cosh(x)**2


def _pot(x, h=1e-3):
    # f''/f by a 4th-order central difference of f itself; at x -> 0 (0/0) the regular
    # limit f''/f -> -8 (f = -2x + (8/3)x^3 + ...) is used below x = 1e-6
    if x < 1e-6:
        return -8.0
    d2 = (-_f(x + 2*h) + 16*_f(x + h) - 30*_f(x) + 16*_f(x - h) - _f(x - 2*h))/(12*h*h)
    return d2/_f(x)


def _dprime_shoot(ka, X=30.0):
    k2 = ka*ka
    kap = np.sqrt(k2 + _pot(X))
    sol = solve_ivp(lambda x, y: [y[1], (k2 + _pot(x))*y[0]], [X, 0.0], [1.0, -kap],
                    rtol=1e-12, atol=1e-300, method="DOP853")
    return 2.0*sol.y[1, -1]/sol.y[0, -1]


def _spectrum(S, ka, nx=_NX, Lx=_LX):
    # the ky = k column of J about Psi0 = sech^2(x - Lx/2) (+ its two nearest periodic images,
    # so the periodic box sees no kink), Phi0 = 0; nu = 0, eta = 1/S
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/ka, dims=2, cfl_safety=0.5,
                           forcing=False, eqpars={"diss": (0.0, 1.0/S), "hyper": 1})
    kgrid = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    p0 = sum(1.0/np.cosh(x - Lx/2 - m*Lx)**2 for m in (-1, 0, 1))
    p0 = jnp.asarray(p0).reshape(1, -1, 1)
    x0 = jr.initialize(lambda X, Y: jnp.stack([0*p0 + 0*Y, p0 + 0*Y]), params).fields
    B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
    return stability.eig_dense(B), idx, params


def _psi_real(params, idx, v):
    out = np.zeros((2, params.nx), dtype=complex)
    out[idx[:, 0], idx[:, 2]] = v
    return np.fft.ifft(out[1])


@pytest.mark.fp64
def test_dprime_formula_against_shooting():
    with checks() as c:
        for ka in (0.3, 1.0, 2.0):
            exact, shot = _dprime_exact(ka), _dprime_shoot(ka)
            rel = abs(shot/exact - 1.0)
            print(f"  ka = {ka}: Delta'a exact {exact:.12f}  shooting {shot:.12f}  rel {rel:.1e}")
            c.check(f"Delta'(ka={ka}) formula == shooting to 1e-8", rel < 1e-8, f"rel = {rel:.2e}")
        c.check("Delta' vanishes at ka = sqrt 5", abs(_dprime_exact(np.sqrt(5.0))) < 1e-14)
        c.check("small-ka limit 15/(ka)^2 + 1/8",
                abs(_dprime_exact(1e-3) - (15e6 + 0.125)) < 1e-4)


@pytest.mark.fp64
def test_growth_rate_matches_julia_reference():
    ref = np.load(_REF)
    sel = np.nonzero((ref["whichf"] == 2) & (ref["S"] == 1e3) & (ref["ka"] == 1.0)
                     & (ref["alpha"] == 0.0))[0]
    assert len(sel) == 1
    i = int(sel[0])
    g_ref = complex(ref["gamma_richardson"][i]).real
    g_err = float(ref["gamma_err"][i])                 # relative
    g_base = complex(ref["gamma"][i]).real
    r, idx, params = _spectrum(1e3, 1.0)
    lead = r.values[0]
    psi = _psi_real(params, idx, r.right[:, 0])
    i0 = params.nx//2
    parity = (np.linalg.norm(psi - psi[(2*i0 - np.arange(params.nx)) % params.nx])
              / np.linalg.norm(psi))
    rel = abs(lead.real/g_ref - 1.0)
    print(f"  taranis gamma = {lead:.10f}, residual {r.residuals[0]:.1e}; Julia (Richardson) "
          f"{g_ref:.8f} +- {g_err:.1e} (relative), base grid {g_base:.8f}; rel. diff {rel:.1e}")
    with checks() as c:
        c.check("leading eigenvalue is real and positive (the tearing mode)",
                lead.real > 0.05 and abs(lead.imag) < 1e-12*r.scale)
        c.check("it is the fastest mode (eig_dense sorts by Re)",
                np.all(r.values[1:].real < lead.real))
        c.check("tearing parity: psi even about the sheet", parity < 1e-10, f"{parity:.1e}")
        c.check("round-off residual", r.residuals[0] < 1e-12, f"{r.residuals[0]:.1e}")
        c.check("gamma equals the Julia eigencode to its recorded (relative) error",
                rel <= g_err, f"rel. diff = {rel:.2e}")
        c.check("... and to 1e-5 (regression pin; observed 3e-7)", rel <= 1e-5,
                f"rel. diff = {rel:.2e}")


@pytest.mark.fp64
def test_marginal_point_brackets_sqrt5():
    r_lo, _, _ = _spectrum(1e3, 2.0)
    r_hi, _, _ = _spectrum(1e3, 2.5)
    g_lo, g_hi = r_lo.values[0], r_hi.values[0]
    print(f"  S = 1e3: fastest mode at ka = 2.0: {g_lo:.6e}; at ka = 2.5: {g_hi:.3e}")
    with checks() as c:
        c.check("ka = 2.0 (< sqrt 5): an unstable real tearing mode",
                g_lo.real > 1e-3 and abs(g_lo.imag) < 1e-12*r_lo.scale)
        c.check("ka = 2.5 (> sqrt 5): no eigenvalue grows (max Re at the round-off of the "
                "nu = 0 null cluster)", g_hi.real < 1e-12*r_hi.scale, f"{g_hi.real:.2e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
