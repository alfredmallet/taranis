# plans/AUTODIFF_PLAN.md rung 1b-ref gate: resistive tearing of Psi0 = sech^2(x) WITH
# FIELD-ALIGNED SHEAR FLOW Phi0 = alpha Psi0 through the eigen-harness (taranis/stability.py),
# against Alfred's Julia eigencode (tests/data/shear_tearing_reference.npz; Mallet, Eriksson,
# Swisdak & Juno, JPP 91, E146 (2025)). The science lives in examples/tearing-shear-eigen.ipynb
# (data: examples/tearing_shear_run.py); this file is its fast standing gate. It IMPORTS that
# script (examples/ is put on sys.path; nothing there is a package) so the notebook's own
# helpers are gated, not a transcription of them: (iv) below, and T.psi0_tanhpair in (v).
#   (i)  S = 1e3, (ka, alpha) = (1, 0.8) and (0.5, 0.95), nx = 512: the fastest eigenvalue of the
#        WHOLE ky-block spectrum (eig_dense) is real, has a round-off residual and the tearing
#        symmetry of a shear-flow mode (psi(-x) = conj psi(x) once psi is real at the sheet: the
#        pi-rotation symmetry of the equilibrium, which maps the ky block to its conjugate), and
#        equals the Julia code's Richardson-extrapolated `gamma_richardson` to the reference's
#        recorded RELATIVE error `gamma_err` (the plan's acceptance). Not the base-grid `gamma`:
#        for 104 of the 120 unstable reference rows gamma_err IS |gamma - gamma_richardson|/|gamma|
#        (the Richardson term dominates it), so an exact code sits exactly one bar from `gamma`
#        (printed here in units of the bar: 1.00). Observed 2.2e-7 and 4.4e-7 relative to the
#        Richardson value (grid-converged: the notebook's nx ladders give the same digits), so a
#        2e-6 regression pin is added.
#   (ii) the spectrum is even in alpha (y -> -y, phi -> -phi is a symmetry of RMHD), and the
#        growth rate is suppressed by the flow: gamma(0.95) < gamma(0.8) < gamma(0) at ka = 1.
#   (iii) the SIGN of alpha, which no eigenvalue can see (ii): the fastest-mode eigenfunction at the
#        reference's stored row (S, ka, alpha) = (1e4, 0.3, 0.8), nx = 1024, normalized as the Julia
#        code does, against its stored eigenfunction over |x| < 1.4. alpha -> -alpha maps
#        (psi, phi) -> (conj psi, -conj phi), flipping Im psi and Re phi (Re phi(0) = 0.14 here):
#        observed 7e-3 as is, 0.75 / 1.4 (psi / phi) with the sign flipped.
#   (iv) the notebook's read-back helpers on hand-built ladder records (pure numpy, both
#        precisions): _mean2 / pair_value = (mean, half split) of the TWO LARGEST real parts;
#        point_value = (value, last ladder change) for sech^2 and (pair mean,
#        max(change of the pair MEAN between the last two rungs, half split)) for the tanh pair,
#        falling back to the ladder's own err with fewer than two rungs. Each branch of the max
#        is made the winner once, so dropping either side fails.
#   (v)  the tanh PAIR (T.psi0_tanhpair, whichf = 1), S = 1e3, ka = 0.5, Lx = 40 (kLx = 20), nx =
#        1024, alpha = 0: the two fastest eigenvalues are real, growing, and of opposite parity
#        under x -> -x (kx -> -kx on both fields, the pair's reflection symmetry: the even/odd
#        split of the two sheets' modes); pair_value's MEAN equals the Julia code's
#        Richardson-extrapolated isolated-sheet value to its recorded error (observed 2.0e-6
#        relative, bar 7.6e-5; pinned at 1e-5) while EACH member alone misses it by more than the
#        bar (observed 2.1-2.2e-4: the pair coupling, which the mean cancels at first order), and
#        the half split bounds the mean's error (the notebook's box-error claim). And, at nx = 512,
#        alpha = +-0.8: the pair is even in alpha (both members to 1e-12 of ||J||) and keeps its
#        parity split under the flow.
# fp64 only for (i)-(iii) and (v). pytest: `pytest tests/test_tearing_shear.py`; script:
# `python tests/test_tearing_shear.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import importlib
import os
import sys

import numpy as np
import pytest

import jax.numpy as jnp

import taranis as jr
from taranis import stability

_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "shear_tearing_reference.npz")
_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
_NX = 512
_POINTS = ((1.0, 0.8, 20.0), (0.5, 0.95, 30.0))        # (ka, alpha, Lx = max(15/k, 20))


def _run_script(name):
    # the notebook's data script as a module (examples/ is not a package; tearing_shear_run
    # itself imports tearing_eigen_run as T from the same directory)
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    return importlib.import_module(name)


def _spectrum(S, ka, alpha, Lx, nx=_NX):
    # the ky = k column of J about psi = Psi0 = sech^2(x - Lx/2) (+ its two nearest periodic
    # images), phi = alpha Psi0; nu = 0, eta = 1/S
    params = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/ka, dims=2, cfl_safety=0.5,
                           forcing=False, eqpars={"diss": (0.0, 1.0/S), "hyper": 1})
    kgrid = jr.setup_kgrids(params)
    x = np.arange(nx)*Lx/nx
    p0 = sum(1.0/np.cosh(x - Lx/2 - m*Lx)**2 for m in (-1, 0, 1))
    p0 = jnp.asarray(p0).reshape(1, -1, 1)
    x0 = jr.initialize(lambda X, Y: jnp.stack([alpha*p0 + 0*Y, p0 + 0*Y]), params).fields
    B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
    return stability.eig_dense(B), idx, params


def _psi(params, idx, v):
    out = np.zeros((2, params.nx), dtype=complex)
    out[idx[:, 0], idx[:, 2]] = v
    psi = np.fft.ifft(out[1])
    i0 = params.nx//2
    return psi*np.conj(psi[i0])/abs(psi[i0]), i0


def _ref_row(ref, ka, alpha):
    sel = np.nonzero((ref["whichf"] == 2) & (ref["S"] == 1e3) & np.isclose(ref["ka"], ka)
                     & np.isclose(ref["alpha"], alpha))[0]
    assert len(sel) == 1 and bool(ref["unstable"][sel[0]])
    return int(sel[0])


@pytest.mark.fp64
def test_shear_growth_rates_match_julia_reference():
    ref = np.load(_REF)
    with checks() as c:
        for ka, alpha, Lx in _POINTS:
            i = _ref_row(ref, ka, alpha)
            g_rich = complex(ref["gamma_richardson"][i]).real
            g_base = complex(ref["gamma"][i]).real
            g_err = float(ref["gamma_err"][i])
            r, idx, params = _spectrum(1e3, ka, alpha, Lx)
            lead = r.values[0]
            psi, i0 = _psi(params, idx, r.right[:, 0])
            refl = (2*i0 - np.arange(params.nx)) % params.nx
            sym = np.linalg.norm(psi[refl] - np.conj(psi))/np.linalg.norm(psi)
            rel_r, rel_b = abs(lead.real/g_rich - 1), abs(lead.real/g_base - 1)
            tag = f"(ka={ka}, alpha={alpha})"
            print(f"  {tag}: taranis {lead:.10f} (residual {r.residuals[0]:.1e}, next "
                  f"{r.values[1]:.2e}); Julia Richardson {g_rich:.10f}, base {g_base:.10f}, "
                  f"rel. err {g_err:.1e}; rel. diff {rel_r:.1e} (to base: {rel_b/g_err:.3f} bars)")
            c.check(f"{tag} fastest eigenvalue is real and growing (the tearing mode)",
                    lead.real > 1e-2 and abs(lead.imag) < 1e-12*r.scale)
            c.check(f"{tag} it is the fastest mode of the whole block (eig_dense sorts by Re)",
                    np.all(r.values[1:].real < lead.real))
            c.check(f"{tag} shear-tearing symmetry psi(-x) = conj psi(x)", sym < 1e-10,
                    f"{sym:.1e}")
            c.check(f"{tag} round-off residual", r.residuals[0] < 1e-12, f"{r.residuals[0]:.1e}")
            c.check(f"{tag} equals the Julia Richardson value to its recorded relative error",
                    rel_r <= g_err, f"{rel_r:.2e} vs {g_err:.1e}")
            c.check(f"{tag} ... and to 2e-6 of the Richardson value (regression pin)",
                    rel_r <= 2e-6, f"{rel_r:.2e}")


@pytest.mark.fp64
def test_eigenfunction_pins_sign_of_alpha():
    ref = np.load(_REF)
    sel = np.nonzero((ref["whichf"] == 2) & (ref["S"] == 1e4) & np.isclose(ref["ka"], 0.3)
                     & np.isclose(ref["alpha"], 0.8))[0]
    assert len(sel) == 1
    pid = int(sel[0])
    assert pid in [int(p) for p in ref["eigfn_pids"]]
    nx, Lx = 1024, 50.0
    r, idx, params = _spectrum(1e4, 0.3, 0.8, Lx, nx=nx)
    out = np.zeros((2, nx), dtype=complex)
    out[idx[:, 0], idx[:, 2]] = r.right[:, 0]
    psi, phi = np.fft.ifft(out[1]), np.fft.ifft(out[0])
    i0 = nx//2
    ph = np.conj(psi[i0])/abs(psi[i0])/max(np.abs(psi).max(), np.abs(phi).max())
    psi, phi = psi*ph, phi*ph
    x = np.arange(nx)*Lx/nx - Lx/2
    xj, pj, fj = (ref[f"eigfn_{pid}_{n}"] for n in ("x", "psi", "phi"))
    w = np.abs(xj) < 1.4

    def at_julia(f):
        return np.interp(xj[w], x, f.real) + 1j*np.interp(xj[w], x, f.imag)
    dpsi, dphi = np.abs(at_julia(psi) - pj[w]).max(), np.abs(at_julia(phi) - fj[w]).max()
    print(f"  (S, ka, alpha) = (1e4, 0.3, 0.8), nx = {nx}: max |taranis - Julia| psi {dpsi:.1e}, "
          f"phi {dphi:.1e}; Re phi(0) {phi[i0].real:+.4f} vs Julia {fj[np.argmin(np.abs(xj))].real:+.4f}")
    with checks() as c:
        c.check("the eigenfunction matches the Julia code's, signs included (same sign of alpha)",
                max(dpsi, dphi) < 2e-2, f"{dpsi:.1e}, {dphi:.1e}")


@pytest.mark.fp64
def test_spectrum_even_in_alpha_and_shear_suppresses():
    spec, scale = {}, 0.0
    for alpha in (0.0, 0.8, -0.8, 0.95):
        r, _, _ = _spectrum(1e3, 1.0, alpha, 20.0)
        spec[alpha] = r.values
        scale = max(scale, r.scale)
    # the whole spectrum, compared as sorted real and sorted imaginary parts (the nu = 0 null
    # cluster's round-off scatter makes an elementwise pairing ambiguous)
    dre = np.max(np.abs(np.sort(spec[0.8].real) - np.sort(spec[-0.8].real)))
    dim = np.max(np.abs(np.sort(spec[0.8].imag) - np.sort(spec[-0.8].imag)))
    dtop = abs(spec[0.8][0] - spec[-0.8][0])
    g0, g8, g95 = (spec[a][0].real for a in (0.0, 0.8, 0.95))
    print(f"  ka = 1: gamma(0) = {g0:.6f}, gamma(0.8) = {g8:.6f}, gamma(0.95) = {g95:.6f}; "
          f"alpha -> -alpha: fastest {dtop:.1e}, whole spectrum (Re, Im) {dre:.1e}, {dim:.1e} "
          f"(||J|| ~ {scale:.1e})")
    with checks() as c:
        c.check("the fastest eigenvalue is even in alpha", dtop < 1e-12, f"{dtop:.1e}")
        # the ~0 cluster of the nu = 0 phi modes is (near-)defective, so round-off moves it by
        # up to ~sqrt(eps)||J||, not eps||J||: observed 7e-10 at ||J|| = 4.2
        c.check("the whole spectrum is even in alpha (to the null cluster's sqrt(eps)||J||)",
                max(dre, dim) < 1e-8*scale, f"{max(dre, dim):.1e}")
        c.check("the flow suppresses tearing: gamma(0.95) < gamma(0.8) < gamma(0)",
                g95 < g8 < g0)


def test_notebook_pair_and_point_value_helpers():
    R = _run_script("tearing_shear_run")
    vals = np.array([0.20 + 1e-14j, -0.5, 0.30 - 1e-14j, 1e-16])    # unsorted, complex, extras
    with checks() as c:
        g, half = R._mean2(vals)
        c.check("_mean2: mean and half split of the TWO LARGEST real parts",
                np.isclose(g, 0.25, rtol=0, atol=1e-15)
                and np.isclose(half, 0.05, rtol=0, atol=1e-15),
                f"{g}, {half}")
        c.check("pair_value reads the pair from res['rec']['values']",
                R.pair_value({"rec": {"values": vals}}) == R._mean2(vals))
        g, e = R.point_value("sech2", {"value": 0.123 + 1e-15j, "err": 3e-8,
                                       "rec": {"values": vals}})
        c.check("point_value(sech2) = (Re value, ladder err)", g == 0.123 and e == 3e-8,
                f"{g}, {e}")

        def pair(ladder, err=0.0):
            return R.point_value("tanhpair", {"rec": {"values": vals}, "err": err,
                                              "vals": [np.array(v) for v in ladder]})
        # half split (0.05) wins over the last change of the pair MEAN (0.26 -> 0.25)
        g, e = pair([[0.9, 0.1], [0.27, 0.25, -1.0], [0.20, -0.3, 0.30]])
        c.check("point_value(tanh): value is the pair mean; half split wins", g == 0.25 and
                np.isclose(e, 0.05, rtol=0, atol=1e-15), f"{g}, {e}")
        # the LAST mean change (0.35 -> 0.25) wins; the first (0.5 -> 0.35) is not the one used
        g, e = pair([[0.9, 0.1], [0.40, 0.30], [0.20, -0.3, 0.30]])
        c.check("point_value(tanh): the last change of the pair MEAN wins when larger",
                np.isclose(e, 0.10, rtol=0, atol=1e-15), f"{e}")
        # the MEAN's change (0.35 -> 0.25), not a member's (top 0.5 -> 0.3, other 0.2 -> 0.2)
        g, e = pair([[0.50, 0.20], [0.20, -0.3, 0.30]])
        c.check("point_value(tanh): it is the MEAN that is differenced, not a member",
                np.isclose(e, 0.10, rtol=0, atol=1e-15), f"{e}")
        g, e = pair([[0.20, 0.30]], err=0.2)
        c.check("point_value(tanh): fewer than two rungs -> max(res['err'], half)",
                np.isclose(e, 0.2, rtol=0, atol=1e-15), f"{e}")
        g, e = pair([], err=0.01)
        c.check("point_value(tanh): ... and the half split when that err is smaller",
                np.isclose(e, 0.05, rtol=0, atol=1e-15), f"{e}")


def _tanhpair(alpha, nx, Lx=40.0, S=1e3, ka=0.5):
    # eig_dense of the ky block about the notebook's tanh pair (T.make_x0 -> T.psi0_tanhpair),
    # and the parity (+-1) of the two fastest eigenvectors under kx -> -kx on both fields
    T = _run_script("tearing_eigen_run")
    params = T.make_params(S, ka, nx, Lx)
    kgrid = jr.setup_kgrids(params)
    x0 = T.make_x0(params, "tanhpair", alpha=alpha)
    B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
    r = stability.eig_dense(B)
    lut = -np.ones((2, nx), dtype=np.int64)
    lut[idx[:, 0], idx[:, 2]] = np.arange(len(idx))
    mir = lut[idx[:, 0], (-idx[:, 2]) % nx]
    assert np.all(mir >= 0)
    par = [np.vdot(r.right[:, j], r.right[mir, j])/np.vdot(r.right[:, j], r.right[:, j])
           for j in (0, 1)]
    return r, par


@pytest.mark.fp64
def test_tanh_pair_even_odd_mean_matches_julia():
    R = _run_script("tearing_shear_run")
    ref = np.load(_REF)
    i = R.ref_index(ref, 1, 1e3, 0.5, 0.0)
    assert bool(ref["unstable"][i])
    g_rich, g_err = complex(ref["gamma_richardson"][i]).real, float(ref["gamma_err"][i])
    r, par = _tanhpair(0.0, 1024)
    top = r.values[:2]
    mean, half = R.pair_value({"rec": {"values": r.values}})
    rel = abs(mean/g_rich - 1)
    rel_m = [abs(z.real/g_rich - 1) for z in top]
    print(f"  tanh pair S = 1e3, ka = 0.5, Lx = 40, nx = 1024: members {top[0]:.10f}, "
          f"{top[1]:.10f} (parity {par[0].real:+.3f}, {par[1].real:+.3f}); mean {mean:.10f}, half "
          f"split {half:.2e}; Julia Richardson {g_rich:.10f} (rel. err {g_err:.1e}): mean "
          f"{rel:.1e}, members {rel_m[0]:.1e}, {rel_m[1]:.1e}")
    with checks() as c:
        c.check("the two fastest are real and growing (the pair), the third is not",
                all(z.real > 1e-3 and abs(z.imag) < 1e-12*r.scale for z in top)
                and r.values[2].real < 1e-10*r.scale, f"{r.values[:3]}")
        c.check("one member even, one odd under x -> -x",
                sorted(p.real for p in par) == pytest.approx([-1.0, 1.0], abs=1e-10)
                and max(abs(p.imag) for p in par) < 1e-10, f"{par}")
        c.check("the pair MEAN equals the Julia Richardson value to its recorded error",
                rel <= g_err, f"{rel:.2e} vs {g_err:.1e}")
        c.check("... and to 1e-5 relative (regression pin; observed 2.0e-6)", rel <= 1e-5,
                f"{rel:.2e}")
        c.check("each member ALONE misses the Julia value by more than its error bar",
                min(rel_m) > g_err, f"{rel_m}")
        c.check("the half split bounds the mean's error (the notebook's box-error bar)",
                abs(mean - g_rich) <= half, f"{abs(mean - g_rich):.1e} vs {half:.1e}")
    # the flow at nx = 512: even in alpha, parity split kept
    rp, pp = _tanhpair(0.8, 512)
    rm, pm = _tanhpair(-0.8, 512)
    d = max(abs(rp.values[j] - rm.values[j]) for j in (0, 1))
    print(f"  alpha = +-0.8, nx = 512: members {rp.values[0]:.8f}, {rp.values[1]:.8f}; "
          f"|diff| to -alpha {d:.1e} (||J|| ~ {rp.scale:.1e})")
    with checks() as c:
        c.check("the pair is even in alpha", d < 1e-12*rp.scale, f"{d:.1e}")
        c.check("both members are real and growing with the flow",
                all(z.real > 1e-3 and abs(z.imag) < 1e-12*rp.scale for z in rp.values[:2]))
        c.check("the flow keeps the even/odd split (both signs of alpha)",
                all(sorted(p.real for p in q) == pytest.approx([-1.0, 1.0], abs=1e-10)
                    for q in (pp, pm)), f"{pp}, {pm}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
