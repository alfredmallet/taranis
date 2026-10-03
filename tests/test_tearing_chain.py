# plans/AUTODIFF_PLAN.md rung 2b item 3: the Fadeev / Kelvin-Stuart island chain
# psi0 = ln(cosh x + eps cos y) (sheet normal x, chain along y) through the autodiff harness --
# examples/tearing-chain.ipynb, data from examples/tearing_chain_run.py, which this file IMPORTS
# (examples/ is put on sys.path). dims = 2, nu = 0, eta = 1/S, Ly = 4 pi (two chain periods),
# fp64, small grids.
#   (i)   equilibrium: the eps = 0 limit IS the rung 1b-ref tanh pair (bitwise), the analytic
#         current lap psi0 = (1 - eps^2) e^(-2 psi0) at both sheets, d psi0/d eps by FD, the
#         harmonic projection, N(x0) at round-off/image level (and NOT for a non-equilibrium);
#   (ii)  exact structure: x0 = 0 gives J = L entrywise (-eta k^2 on psi, 0 on phi); J commutes
#         with T, R, P, PR (and with flow only with T, PR); the sector bases; the sector spectra
#         partition the Bloch spectrum and the full-space eigenvectors carry the parities;
#   (iii) eps -> 0: the sector eigenvalues ARE the ky-block (iky = 1, k = 1/2) Harris-pair
#         eigenvalues, also of rung 1a's independently built tanh-pair state; the degenerate
#         pair splits at first order, d lam(R+)/d eps = -d lam(R-)/d eps;
#   (iv)  sensitivities: d lam/d eps (state direction), d lam/d alpha, d lam/d eta (overrides
#         seam) against eigenvalue finite differences; J at overrides == J of a Parameters
#         built there; d lam/d alpha = 0 at alpha = 0 (the R symmetry);
#   (v)   the matrix-free route: sector_eigs == the dense sector top (O(dt^4)), with no signs ==
#         stability.propagator_eigs, loud on a bad dt, refuses residual_tol > 1e-4;
#   (vi)  eigenfunction helpers (R+ is psi even / phi odd about the X-point, in real space),
#         the cache, and the cached headline numbers (skipped when the data are absent);
#   (viii) review gates: make_data jobs write exactly the keys the readers read; dense_point record contents
#         (leading mode, n_grow, a refused sensitivity recorded as nan); ivp_point's window.
# pytest: `pytest tests/test_tearing_chain.py`; script: `python tests/test_tearing_chain.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import functools
import importlib
import os
import sys
import tempfile

import numpy as np
import pytest

import jax.numpy as jnp

import taranis as jr
from taranis import stability as st

# TARANIS_CHAIN_EXAMPLES: an alternative examples directory (mutation testing runs on a scratch copy)
_EXAMPLES = os.environ.get("TARANIS_CHAIN_EXAMPLES",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))


def _run_script(name="tearing_chain_run"):
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    return importlib.import_module(name)


@functools.lru_cache(maxsize=None)
def _setup(S=100.0, nx=64, ny=16, Lx=20.0, Ly=None):
    C = _run_script()
    p = C.make_params(S, nx, ny, Lx=Lx, Ly=C.LY if Ly is None else Ly)
    return p, jr.setup_kgrids(p)


def _dense_top(C, p, kg, eps, signs, alpha=0.0, T_sign=-1, overrides=None):
    sec = C.Sector(kg, p, T_sign, signs)
    x0 = C.make_x0(p, eps, alpha)
    return sec, x0, C.sector_dense(x0, sec, overrides=overrides)


def _richardson(fd_h, fd_h2):
    # central differences at h and h/2 -> O(h^4)
    return (4*fd_h2 - fd_h)/3


def _lap(p, f):
    # spectral Laplacian of a real (nx, ny) field
    kx = 2*np.pi*np.fft.fftfreq(p.nx, p.Lx/p.nx)
    ky = 2*np.pi*np.fft.fftfreq(p.ny, p.Ly/p.ny)
    return np.real(np.fft.ifft2(-(kx[:, None]**2 + ky[None, :]**2)*np.fft.fft2(f)))


# ============================================================ (i) equilibrium

@pytest.mark.fp64
def test_equilibrium_profile():
    C = _run_script()
    T = C.T
    Lx, nx, ny = 40.0, 512, 64
    p = C.make_params(100.0, nx, ny, Lx=Lx)
    x, y = C.grid(p)
    X, Y = np.meshgrid(x, y, indexing="ij")
    with checks() as c:
        p0 = C.psi_chain(x, y, Lx, 0.0)
        c.check("eps = 0 is rung 1b-ref's tanh pair, bitwise",
                np.array_equal(p0, np.broadcast_to(T.psi0_tanhpair(x, Lx)[:, None], p0.shape)))
        eps = 0.4
        ps = C.psi_chain(x, y, Lx, eps)
        lap = _lap(p, ps)
        n1, n2 = np.abs(x - Lx/4) < 5, np.abs(x - 3*Lx/4) < 5
        j1 = C.chain_current(x[n1] - Lx/4, y, eps)
        j2 = C.chain_current(x[n2] - 3*Lx/4, y, eps)
        e1, e2 = np.abs(lap[n1] - j1).max(), np.abs(lap[n2] + j2).max()
        c.check(f"lap psi0 = (1 - eps^2)/(cosh + eps cos)^2 at sheet 1 ({e1:.1e}) and minus it at "
                f"sheet 2 ({e2:.1e})", e1 < 1e-8 and e2 < 1e-8)
        c.check("... = (1 - eps^2) e^(-2 psi0) on the isolated chain (the F(psi0) of the docstring)",
                np.allclose(j1, (1 - eps**2)*np.exp(-2*np.log(np.cosh(X[n1] - Lx/4) + eps*np.cos(Y[n1]))),
                            rtol=1e-13, atol=0))
        c.check("the chain is periodic in y with period 2 pi and even in y",
                np.allclose(ps[:, :ny//2], ps[:, ny//2:], atol=1e-14)
                and np.allclose(ps[:, 1:], ps[:, :0:-1], atol=1e-14))
        d = C.psi_chain(x, y, Lx, eps, deriv=True)
        e = [np.abs((C.psi_chain(x, y, Lx, eps + h) - C.psi_chain(x, y, Lx, eps - h))/(2*h) - d).max()
             for h in (2e-4, 1e-4)]
        c.check(f"d psi0/d eps vs central FD: {e[0]:.1e} -> {e[1]:.1e}, O(h^2)", e[1] < 3e-8 and 3.5 < e[0]/e[1] < 4.5)
        c.check("d psi0/d eps = cos y/(cosh + eps cos) near sheet 1 (images: e^-15)",
                np.allclose(d[n1], np.cos(Y[n1])/(np.cosh(X[n1] - Lx/4) + eps*np.cos(Y[n1])), rtol=0, atol=2e-6))


@pytest.mark.fp64
def test_make_x0():
    C = _run_script()
    p, kg = _setup()
    x0 = np.asarray(C.make_x0(p, 0.3, alpha=0.5))
    iky = np.arange(p.ny//2 + 1)
    with checks() as c:
        c.check("x0 carries chain harmonics only (odd iky exactly zero)", np.all(x0[..., iky % 2 == 1] == 0))
        c.check("phi0 = alpha psi0 bitwise", np.array_equal(x0[0], 0.5*x0[1]))
        da = np.asarray(C.make_x0(p, 0.3, alpha=0.5, direction="alpha"))
        c.check("d x0/d alpha = (psi0, 0)", np.array_equal(da[0], x0[1]) and np.all(da[1] == 0))
        h = 1e-5
        fd = (np.asarray(C.make_x0(p, 0.3 + h, 0.5)) - np.asarray(C.make_x0(p, 0.3 - h, 0.5)))/(2*h)
        de = np.asarray(C.make_x0(p, 0.3, 0.5, direction="eps"))
        c.check(f"d x0/d eps vs FD ({np.abs(fd - de).max()/np.abs(de).max():.1e})",
                np.abs(fd - de).max() < 1e-7*np.abs(de).max())
        with pytest.raises(ValueError):
            C.make_x0(p, 0.3, direction="bogus")
        with pytest.raises(ValueError):
            C.chain_periods(C.make_params(1.0, 16, 16, Ly=5.0))


@pytest.mark.fp64
def test_chain_is_steady():
    # N(x0) = 0 for psi0 = ln(cosh x + eps cos y) (ideal steady); the discrete residual is the
    # images' e^(-Lx/2) plus the spectral truncation (the chain is analytic in a strip of width
    # acosh(1/eps) in y, so ln cosh is not band-limited)
    C = _run_script()
    with checks() as c:
        out = {}
        for Lx, nx, ny in ((20.0, 256, 64), (30.0, 384, 64), (40.0, 512, 64), (40.0, 512, 16)):
            p = C.make_params(100.0, nx, ny, Lx=Lx)
            kg = jr.setup_kgrids(p)
            out[(Lx, ny)] = C.n_of_x0(p, C.make_x0(p, 0.3), kg)
            if (Lx, ny) == (40.0, 64):
                na = C.n_of_x0(p, C.make_x0(p, 0.3, alpha=0.7), kg)
                # a NON-equilibrium of the same size: the chain with cos 2y (lap psi != F(psi))
                x, y = C.grid(p)
                bad = C.psi_chain(x, 2*np.asarray(y), Lx, 0.3)
                nb = C.n_of_x0(p, C._to_fields(p, 0*bad, bad), kg)
        c.check(f"Lx = 40, ny = 64: |N(x0)| = {out[(40.0, 64)]:.1e} of |x0| (round-off level)",
                out[(40.0, 64)] < 5e-11)
        c.check(f"... with field-aligned flow alpha = 0.7: {na:.1e}", na < 5e-11)
        r = out[(20.0, 64)]/out[(30.0, 64)]
        c.check(f"Lx = 20 -> 30: N(x0) {out[(20.0, 64)]:.1e} -> {out[(30.0, 64)]:.1e}, ratio {r:.0f} ~ "
                f"e^(5) = 148 (the image coupling e^(-Lx/2))", 50 < r < 500)
        c.check(f"ny = 16 truncation is visible: {out[(40.0, 16)]:.1e}", out[(40.0, 16)] > 1e3*out[(40.0, 64)])
        c.check(f"a non-equilibrium (cos 2y chain) is NOT steady: {nb:.1e}", nb > 1e-3)


# ============================================================ (ii) exact structure

@pytest.mark.fp64
def test_zero_state_J_is_L():
    C = _run_script()
    p, kg = _setup()
    eta = 1.0/100.0
    with checks() as c:
        for T_sign, signs in ((-1, {"R": 1, "P": -1}), (1, {"R": -1, "P": 1}), (-1, {})):
            sec = C.Sector(kg, p, T_sign, signs)
            J = C.sector_matrix(jnp.zeros_like(C.make_x0(p, 0.0)), sec)
            # the diagonal L in the Bloch coordinates, built from kx, ky by hand
            co = sec.coords
            kx = 2*np.pi*np.fft.fftfreq(p.nx, p.Lx/p.nx)
            ky = 2*np.pi*np.arange(p.ny//2 + 1)/p.Ly

            def diag(idx):
                f, _, ix, iy = np.unravel_index(idx, co.shape)
                return np.where(f == 1, -eta*(kx[ix]**2 + ky[iy]**2), 0.0)
            Ld = np.concatenate([diag(co.re), diag(co.im)])
            Lm = np.diag(Ld) if sec.E is None else (sec.E.T @ (Ld[:, None]*sec.E.toarray()))
            c.check(f"{sec.label()}: J(x0 = 0) == L ({np.abs(J - Lm).max():.1e})", np.abs(J - Lm).max() < 1e-15)


def _perm_op(C, sec_coords, name, p):
    perm, sign = C.signed_perm(sec_coords, C.MAPS[name], p)
    return lambda u: np.bincount(perm, weights=sign*u, minlength=len(u))


@pytest.mark.fp64
def test_symmetries_commute():
    C = _run_script()
    p, kg = _setup()
    rng = np.random.default_rng(3)
    with checks() as c:
        for alpha in (0.0, 0.4):
            x0 = C.make_x0(p, 0.5, alpha)
            for T_sign in (-1, 1):
                co = C.bloch_coords(kg, p, T_sign)
                op, _ = st.real_operator(x0, kg, p, coords=co)
                for name in ("R", "P", "PR", "XR"):
                    M = _perm_op(C, co, name, p)
                    u = rng.standard_normal(co.n)
                    Ju = op.matvec(u)
                    err = np.abs(op.matvec(M(u)) - M(Ju)).max()/np.abs(Ju).max()
                    if alpha == 0 or name in ("PR", "XR"):
                        c.check(f"alpha={alpha} T{T_sign:+d}: J {name} = {name} J ({err:.1e})", err < 1e-13)
                    else:
                        c.check(f"alpha={alpha}: {name} alone is NOT a symmetry ({err:.1e})", err > 1e-3)
        # T: J maps the T = -1 coordinates into themselves (the T = +1 part of J u is zero)
        x0 = C.make_x0(p, 0.5, 0.4)
        co, full = C.bloch_coords(kg, p, -1), st.real_coords(kg, p)
        f = co.unpack(jnp.asarray(rng.standard_normal(co.n)))
        g = np.asarray(st.jvp_operator(x0, kg, p)(f))
        even = np.arange(p.ny//2 + 1) % 2 == 0
        c.check(f"J preserves the period-doubling (T = -1) sector ({np.abs(g[..., even]).max():.1e})",
                np.abs(g[..., even]).max() < 1e-13*np.abs(g).max())
        c.check("map_T is (-1)^iky on Ly = 4 pi",
                np.allclose(np.asarray(C.map_T(np.ones((2, 1, p.nx, p.ny//2 + 1)), p))[0, 0, 0],
                            (-1.0)**np.arange(p.ny//2 + 1)))
        c.check("the Bloch sectors split the real coordinates", co.n + C.bloch_coords(kg, p, 1).n == full.n)


@pytest.mark.fp64
def test_maps_in_real_space():
    # the maps' conventions checked on real fields, independently of the Fourier formulas
    C = _run_script()
    p, kg = _setup(nx=32, ny=16)
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((p.nx, p.ny)), rng.standard_normal((p.nx, p.ny))
    f = np.stack([np.fft.rfft2(a), np.fft.rfft2(b)])[:, None]
    back = lambda g: (np.fft.irfft2(np.asarray(g)[0, 0], s=(p.nx, p.ny)),
                      np.fft.irfft2(np.asarray(g)[1, 0], s=(p.nx, p.ny)))
    with checks() as c:
        phi, psi = back(C.map_R(f, p))
        ry = (-np.arange(p.ny)) % p.ny
        c.check("R: (phi, psi)(x, y) -> (-phi, psi)(x, -y)",
                np.allclose(phi, -a[:, ry], atol=1e-12) and np.allclose(psi, b[:, ry], atol=1e-12))
        phi, psi = back(C.map_P(f, p))
        sx = (np.arange(p.nx) - p.nx//2) % p.nx
        c.check("P: (phi, psi)(x, y) -> (phi, -psi)(x - Lx/2, y)",
                np.allclose(phi, a[sx], atol=1e-12) and np.allclose(psi, -b[sx], atol=1e-12))
        phi, psi = back(C.map_XR(f, p))
        c.check("XR: (phi, psi)(x, y) -> (phi, psi)(Lx/2 - x, -y)",
                np.allclose(phi, a[(p.nx//2 - np.arange(p.nx)) % p.nx][:, ry], atol=1e-12)
                and np.allclose(psi, b[(p.nx//2 - np.arange(p.nx)) % p.nx][:, ry], atol=1e-12))
        phi, psi = back(C.map_T(f, p))
        sy = (np.arange(p.ny) - p.ny//2) % p.ny
        c.check("T: a shift by Ly/2 = 2 pi in y",
                np.allclose(phi, a[:, sy], atol=1e-12) and np.allclose(psi, b[:, sy], atol=1e-12))


@pytest.mark.fp64
def test_sector_basis():
    C = _run_script()
    p, kg = _setup()
    with checks() as c:
        for T_sign in (-1, 1):
            co = C.bloch_coords(kg, p, T_sign)
            dims = 0
            for R in (1, -1):
                for P in (1, -1):
                    E = C.sector_basis(co, p, {"R": R, "P": P})
                    dims += E.shape[1]
                    Ed = E.toarray()
                    ok = np.abs(Ed.T @ Ed - np.eye(E.shape[1])).max() < 1e-14
                    for name, s in (("R", R), ("P", P)):
                        M = _perm_op(C, co, name, p)
                        ok &= max(np.abs(M(Ed[:, j]) - s*Ed[:, j]).max() for j in range(0, E.shape[1], 7)) < 1e-15
                    c.check(f"T{T_sign:+d} R{R:+d} P{P:+d}: orthonormal, M E = s E (m = {E.shape[1]})", ok)
            c.check(f"T{T_sign:+d}: the four (R, P) sectors fill the Bloch sector ({dims} = {co.n})", dims == co.n)
        with pytest.raises(ValueError):
            C.signed_perm(C.bloch_coords(kg, p, -1), lambda f, p_: 2*f, p)
        p1 = C.make_params(100.0, 64, 16, Lx=20.0, Ly=2*np.pi)
        kg1 = jr.setup_kgrids(p1)
        with pytest.raises(ValueError, match="no T = -1 sector"):
            C.bloch_coords(kg1, p1, -1)
        c.check("one chain period: T = +1 is all the real coordinates", C.bloch_coords(kg1, p1, 1).n == st.real_coords(kg1, p1).n)


@pytest.mark.fp64
def test_sectors_partition_the_spectrum():
    C = _run_script()
    p, kg = _setup(nx=48, ny=16)
    x0 = C.make_x0(p, 0.3)
    full = C.Sector(kg, p, -1, {})
    d = C.sector_dense(x0, full)
    vals = []
    for R in (1, -1):
        for P in (1, -1):
            sec = C.Sector(kg, p, -1, {"R": R, "P": P})
            vals.append(C.sector_dense(x0, sec)["values"])
    allv = np.sort_complex(np.concatenate(vals))
    with checks() as c:
        c.check(f"union of the four sector spectra == the T = -1 spectrum ({len(allv)} eigenvalues)",
                len(allv) == len(d["values"])
                and np.abs(np.sort(allv.real)[-8:] - np.sort(d["values"].real)[-8:]).max() < 1e-10)
        # the leading full-space eigenvectors are R and P eigenvectors (simple eigenvalues)
        for i in range(3):
            pr, pp = C.parity(full, d["right"][:, i], "R"), C.parity(full, d["right"][:, i], "P")
            c.check(f"T- eigenvector {i} (lam = {d['values'][i].real:.6f}): R parity {pr:+.12f}, P parity {pp:+.12f}",
                    abs(abs(pr) - 1) < 1e-9 and abs(abs(pp) - 1) < 1e-9)
            secv = vals[(0 if pr > 0 else 2) + (0 if pp > 0 else 1)]
            c.check("... and its eigenvalue is in that sector's spectrum",
                    np.abs(secv - d["values"][i]).min() < 1e-10)


# ============================================================ (iii) eps -> 0

@pytest.mark.fp64
def test_eps0_is_the_harris_pair_ky_block():
    C = _run_script()
    T = C.T
    p, kg = _setup(nx=128, ny=16)
    with checks() as c:
        tops = {}
        for R in (1, -1):
            for P in (1, -1):
                tops[(R, P)] = _dense_top(C, p, kg, 0.0, {"R": R, "P": P})[2]["values"][0]
        x0 = C.make_x0(p, 0.0)
        B, _ = st.ky_block_matrix(x0, kg, p, 1)
        blk = np.sort(st.eig_dense(B).values.real)[::-1][:2]
        # rung 1a's independently built tanh-pair state (T.make_x0) on the same grid (the 2/3
        # mask is ELLIPTICAL in index space, so the kept kx of iky = 1 depend on ny)
        b1, _ = st.ky_block_matrix(T.make_x0(p, "tanhpair"), kg, p, 1)
        one = np.sort(st.eig_dense(b1).values.real)[::-1][:2]
        c.check(f"R+ == R- at eps = 0 (degenerate cos/sin pair): {tops[(1, 1)].real:.12f}",
                abs(tops[(1, 1)] - tops[(-1, 1)]) < 1e-12 and abs(tops[(1, -1)] - tops[(-1, -1)]) < 1e-12)
        mine = np.sort([tops[(1, 1)].real, tops[(1, -1)].real])[::-1]
        c.check(f"P+/P- sector tops {mine} == ky block iky = 1 {blk}", np.abs(mine - blk).max() < 1e-12)
        c.check(f"... == rung 1a's tanh-pair block {one}", np.abs(mine - one).max() < 1e-12)
        c.check("P+ is the slower (even-about-the-midpoint) pair member here", tops[(1, 1)].real < tops[(1, -1)].real)


@pytest.mark.fp64
def test_degenerate_pair_splits_linearly():
    # at eps = 0 the R+/R- pair is exactly degenerate; d J/d eps couples ky = +-1/2 only, so the
    # first-order splitting is +-c: d lam(R+)/d eps = -d lam(R-)/d eps, and lam(eps) = lam0 +-
    # c eps + O(eps^2)
    C = _run_script()
    p, kg = _setup(nx=128, ny=16)
    dx = C.make_x0(p, 0.0, direction="eps")
    with checks() as c:
        d = {}
        for R in (1, -1):
            sec, x0, dn = _dense_top(C, p, kg, 0.0, {"R": R, "P": 1})
            d[R] = C.sector_sensitivity(x0, sec, dn, dx0=dx).dlam
        c.check(f"d lam/d eps: R+ {d[1].real:.10f}, R- {d[-1].real:.10f}: equal and opposite",
                abs(d[1] + d[-1]) < 1e-9*abs(d[1]) and abs(d[1]) > 0.1)
        c.check("R+ (coalescence) is the one that grows", d[1].real > 0)
        lam0 = _dense_top(C, p, kg, 0.0, {"R": 1, "P": 1})[2]["values"][0].real
        errs = []
        for e in (2e-3, 1e-3):
            le = _dense_top(C, p, kg, e, {"R": 1, "P": 1})[2]["values"][0].real
            errs.append(abs(le - lam0 - d[1].real*e))
        c.check(f"lam(eps) - lam0 - c eps is O(eps^2): {errs[0]:.2e} -> {errs[1]:.2e}",
                3.0 < errs[0]/errs[1] < 5.0)


# ============================================================ (iv) sensitivities

@pytest.mark.fp64
def test_deps_against_eigenvalue_fd():
    C = _run_script()
    p, kg = _setup(nx=96, ny=32)
    sig = {"R": 1, "P": 1}
    eps = 0.25
    sec, x0, dn = _dense_top(C, p, kg, eps, sig)
    s = C.sector_sensitivity(x0, sec, dn, dx0=C.make_x0(p, eps, direction="eps"))
    fd = []
    for h in (4e-3, 2e-3):
        lp = _dense_top(C, p, kg, eps + h, sig)[2]["values"][0]
        lm = _dense_top(C, p, kg, eps - h, sig)[2]["values"][0]
        fd.append((lp - lm)/(2*h))
    errs = [abs(f - s.dlam) for f in fd]
    ext = abs(_richardson(*fd) - s.dlam)
    with checks() as c:
        c.check(f"d lam/d eps = {s.dlam.real:.8f} (kappa {s.kappa:.2f}); FD error {errs[0]:.1e} -> "
                f"{errs[1]:.1e} (O(h^2)), Richardson {ext:.1e}",
                3.0 < errs[0]/errs[1] < 5.0 and ext < 1e-7*abs(s.dlam))


@pytest.mark.fp64
def test_deta_through_the_overrides_seam():
    C = _run_script()
    p, kg = _setup(nx=96, ny=32)
    sig = {"R": 1, "P": -1}
    eps, eta = 0.25, 0.01
    sec, x0, dn = _dense_top(C, p, kg, eps, sig)
    s = C.sector_sensitivity(x0, sec, dn, dparams={"diss": (0.0, 1.0)})
    with checks() as c:
        J1 = C.sector_matrix(x0, sec, overrides={"diss": (0.0, 0.013)})
        p2 = C.make_params(1/0.013, 96, 32, Lx=20.0)
        sec2 = C.Sector(jr.setup_kgrids(p2), p2, -1, sig)
        J2 = C.sector_matrix(x0, sec2)
        c.check(f"J at overrides diss = (0, 0.013) == J of Parameters(S = 1/0.013) ({np.abs(J1 - J2).max():.1e})",
                np.abs(J1 - J2).max() < 1e-13)
        fd = []
        for h in (2e-4, 1e-4):
            lp = C.sector_dense(x0, sec, overrides={"diss": (0.0, eta + h)})["values"][0]
            lm = C.sector_dense(x0, sec, overrides={"diss": (0.0, eta - h)})["values"][0]
            fd.append((lp - lm)/(2*h))
        errs = [abs(f - s.dlam) for f in fd]
        ext = abs(_richardson(*fd) - s.dlam)
        c.check(f"d lam/d eta = {s.dlam.real:.6f}; FD (overrides) error {errs[0]:.1e} -> {errs[1]:.1e}, "
                f"Richardson {ext:.1e}", 3.0 < errs[0]/errs[1] < 5.0 and ext < 1e-7*abs(s.dlam))


@pytest.mark.fp64
def test_dalpha():
    C = _run_script()
    p, kg = _setup(nx=96, ny=32)
    sig = {"PR": 1, "XR": 1}
    eps = 0.25
    with checks() as c:
        for i, P in enumerate((1, -1)):
            a0 = _dense_top(C, p, kg, eps, C.PR[i])[2]["values"][0]
            b0 = _dense_top(C, p, kg, eps, {"R": 1, "P": P})[2]["values"][0]
            c.check(f"C.PR[{i}] is the coalescence sector R+ P{P:+d} at alpha = 0 ({a0.real:.10f} vs {b0.real:.10f})",
                    abs(a0 - b0) < 1e-12)
        sec, x0, dn = _dense_top(C, p, kg, eps, sig, alpha=0.0)
        s0 = C.sector_sensitivity(x0, sec, dn, dx0=C.make_x0(p, eps, 0.0, direction="alpha"))
        c.check(f"d lam/d alpha = {abs(s0.dlam):.1e} at alpha = 0 (R maps alpha -> -alpha within PR+ XR+)",
                abs(s0.dlam) < 1e-10)
        a = 0.4
        sec, x0, dn = _dense_top(C, p, kg, eps, sig, alpha=a)
        s = C.sector_sensitivity(x0, sec, dn, dx0=C.make_x0(p, eps, a, direction="alpha"))
        fd = []
        for h in (4e-3, 2e-3):
            lp = _dense_top(C, p, kg, eps, sig, alpha=a + h)[2]["values"][0]
            lm = _dense_top(C, p, kg, eps, sig, alpha=a - h)[2]["values"][0]
            fd.append((lp - lm)/(2*h))
        errs = [abs(f - s.dlam) for f in fd]
        ext = abs(_richardson(*fd) - s.dlam)
        c.check(f"alpha = 0.4: d lam/d alpha = {s.dlam.real:.6f}; FD error {errs[0]:.1e} -> {errs[1]:.1e}, "
                f"Richardson {ext:.1e}", 3.0 < errs[0]/errs[1] < 5.0 and ext < 1e-7*abs(s.dlam)
                and abs(s.dlam) > 1e-3)


# ============================================================ (v) the matrix-free route

@pytest.mark.fp64
def test_sector_eigs():
    C = _run_script()
    p, kg = _setup(nx=96, ny=32)
    x0 = C.make_x0(p, 0.3)
    with checks() as c:
        sec = C.Sector(kg, p, -1, {"R": 1, "P": 1})
        dn = C.sector_dense(x0, sec)
        r = C.sector_eigs(x0, sec, T=20.0, dt=0.05, k=1, ncv=12)
        c.check(f"sector_eigs {r['values'][0].real:.10f} == dense {dn['values'][0].real:.10f} "
                f"(residual {r['residuals'][0]:.1e})", abs(r["values"][0] - dn["values"][0]) < 1e-7)
        full = C.Sector(kg, p, -1, {})
        r1 = C.sector_eigs(x0, full, T=20.0, dt=0.05, k=1, ncv=12)
        r2 = st.propagator_eigs(x0, kg, p, T=20.0, dt=0.05, k=1, ncv=12, coords=full.coords,
                                v0=np.random.default_rng(0).standard_normal(full.m))
        c.check(f"no signs: sector_eigs == stability.propagator_eigs ({r1['values'][0]} vs {r2.values[0]})",
                abs(r1["values"][0] - r2.values[0]) < 1e-12
                and abs(r1["residuals"][0] - r2.residuals[0]) <= 1e-6*r2.residuals[0])
        tops = [C.sector_dense(x0, C.Sector(kg, p, -1, {"R": R, "P": P}))["values"][0]
                for R in (1, -1) for P in (1, -1)]
        c.check(f"the fastest T- mode is the fastest of the four sector tops {np.round(np.real(tops), 6)}",
                abs(r1["values"][0] - max(tops, key=lambda z: z.real)) < 1e-7)
        with pytest.raises(ValueError):
            C.sector_eigs(x0, sec, T=20.0, dt=0.05, k=1, residual_tol=1e-3)
        with pytest.raises(RuntimeError):
            C.sector_eigs(x0, sec, T=20.0, dt=2.0, k=1, ncv=12)


# ============================================================ (vi) helpers, cache, headline

@pytest.mark.fp64
def test_mode_fields_are_coalescence():
    C = _run_script()
    p, kg = _setup(nx=96, ny=32)
    sec, x0, dn = _dense_top(C, p, kg, 0.3, {"R": 1, "P": 1})
    phi, psi = C.mode_fields(sec, dn["right"][:, 0])
    ry = (-np.arange(p.ny)) % p.ny
    sx = (np.arange(p.nx) - p.nx//2) % p.nx
    sy = (np.arange(p.ny) - p.ny//2) % p.ny
    with checks() as c:
        c.check("max |psi1| = 1", abs(np.abs(psi).max() - 1) < 1e-14)
        phm, psm = C.mode_fields(sec, -dn["right"][:, 0])
        c.check("... for either sign of the eigenvector (normalised by max |psi|, not max psi)",
                abs(np.abs(psm).max() - 1) < 1e-14 and np.allclose(np.abs(psm), np.abs(psi), atol=1e-12))
        c.check("R+: psi1 even, phi1 odd about the X-point y = 0 (islands at y = +-pi move oppositely)",
                np.allclose(psi, psi[:, ry], atol=1e-10) and np.allclose(phi, -phi[:, ry], atol=1e-10)
                and np.abs(phi).max() > 1e-3)
        c.check("T-: antiperiodic over one chain period", np.allclose(psi, -psi[:, sy], atol=1e-10))
        c.check("P+: (phi, psi)(x + Lx/2) = (phi, -psi)", np.allclose(psi[sx], -psi, atol=1e-10)
                and np.allclose(phi[sx], phi, atol=1e-10))
        c.check("parity() reads the sector signs", abs(C.parity(sec, dn["right"][:, 0], "R") - 1) < 1e-12
                and abs(C.parity(sec, dn["right"][:, 0], "P") - 1) < 1e-12)


@pytest.mark.fp64
def test_cache():
    C = _run_script()
    calls = []

    def comp():
        calls.append(1)
        return dict(a=np.arange(3.0))

    def bad():
        raise RuntimeError("boom")
    with tempfile.TemporaryDirectory() as root, checks() as c:
        r1 = C.cached(comp, root=root, S=100, eps=0.3, signs={"R": 1})
        r2 = C.cached(comp, root=root, S=100.0, eps=0.3, signs={"R": 1.0})
        c.check("written once, read back (int/float keys equal)", len(calls) == 1 and np.array_equal(r2["a"], r1["a"]))
        c.check("a different key recomputes", C.cached(comp, root=root, S=100, eps=0.31, signs={"R": 1}) and len(calls) == 2)
        with pytest.raises(KeyError):
            C.cached(None, root=root, S=1)
        with pytest.raises(RuntimeError):
            C.cached(bad, root=root, S=2)
        with pytest.raises(RuntimeError, match="cached failure"):
            C.cached(comp, root=root, S=2)
        c.check("a failure is not recomputed", len(calls) == 2)
        def verr():
            calls.append(1)
            raise ValueError("not a solver failure")
        with pytest.raises(ValueError):
            C.cached(verr, root=root, S=3)
        with pytest.raises(ValueError):
            C.cached(verr, root=root, S=3)
        c.check("a non-RuntimeError is not cached as a failure", len(calls) == 4)
        c.check("no temporary files left", not any(f.endswith(".tmp.npz") for f in os.listdir(os.path.join(root, "solves"))))


# ============================================================ (vi b) run-module helpers

@pytest.mark.fp64
def test_theory_helpers():
    C = _run_script()
    from scipy.optimize import brentq
    with checks() as c:
        for eps in (1e-4, 0.1, 0.5, 0.9):
            # the separatrix psi = ln(1 + eps) crosses the O-point line y = pi at x = w
            w = brentq(lambda x: np.log(np.cosh(x) - eps) - np.log1p(eps), 1e-12, 10.0, xtol=1e-15)
            c.check(f"island_halfwidth({eps}) = {C.island_halfwidth(eps):.12f} == separatrix root {w:.12f}",
                    abs(C.island_halfwidth(eps) - w) < 1e-10)
            jO = C.chain_current(np.array([0.0]), np.array([np.pi]), eps)[0, 0]
            c.check(f"current_O({eps}) == lap psi0 at (0, pi)", abs(C.current_O(eps) - jO) < 1e-12*jO)
        c.check("w -> 2 sqrt(eps) as eps -> 0", abs(C.island_halfwidth(1e-8)/(2e-4) - 1) < 1e-6)
        c.check("gamma_harris == T.gamma_disp(1/2, eta, 1, 2(1/ka - ka))",
                C.gamma_harris(1e3) == C.T.gamma_disp(0.5, 1e-3, 1.0, 3.0))
        for nx, ny in ((256, 128), (512, 256)):
            dt = C.prop_dt(nx, ny)
            k = np.hypot(2*np.pi/C.LX0*(nx//3), 2*np.pi/C.LY*(ny//3))
            c.check(f"prop_dt({nx}, {ny}) = {dt:.5f}: min(0.05, 1.6/|k|max)", abs(dt - min(0.05, 1.6/k)) < 1e-15)
        # ideal_extrapolation recovers a synthetic g_inf + a S^-q exactly, and predicts e1
        gi, a, q = 0.1, 0.05, 0.37
        g = lambda S: gi + a*S**-q
        e = lambda S: -q*a*S**-q/g(S)
        r = C.ideal_extrapolation(g(1e2), g(1e3), e(1e3))
        c.check(f"ideal_extrapolation: {r}", abs(r[0] - gi) < 1e-12 and abs(r[1] - q) < 1e-10
                and abs(r[2] - e(1e2)) < 1e-12)
        c.check("ideal_extrapolation: nan when gamma does not decrease with S",
                all(np.isnan(C.ideal_extrapolation(0.1, 0.2, -0.1))) and all(np.isnan(C.ideal_extrapolation(0.1, 0.1, -0.1))))
        # the notebook's q reach 0.66 (eps = 0.75): the bracket must cover them
        q2 = 0.66
        g2_ = lambda S: gi + a*S**-q2
        r2_ = C.ideal_extrapolation(g2_(1e2), g2_(1e3), -q2*a*1e3**-q2/g2_(1e3))
        c.check(f"ideal_extrapolation at q = 0.66: {r2_}", abs(r2_[0] - gi) < 1e-12 and abs(r2_[1] - q2) < 1e-10)


def _tiny(C):
    # small configuration overrides for the cached-point machinery
    return dict(GRID={100.0: (64, 16)}, SPLIT_NX={100.0: 64}, FD_H=(4e-3, 2e-3), S_LIST=(100.0,))


class _patched:
    def __init__(self, mod, **attrs):
        self.mod, self.attrs, self.saved = mod, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.mod, k, v)
        return False


@pytest.mark.fp64
def test_cached_points():
    C = _run_script()
    with tempfile.TemporaryDirectory() as root, _patched(C, **_tiny(C)), checks() as c:
        sg = {"R": 1, "P": 1}
        d = C.dense_point(100.0, 0.3, sg, root=root)
        p, kg = _setup(nx=64, ny=16)
        sec = C.Sector(kg, p, -1, sg)
        x0 = C.make_x0(p, 0.3)
        dn = C.sector_dense(x0, sec)
        c.check("dense_point values == sector_dense", np.abs(d["values"] - dn["values"][:12]).max() < 1e-12)
        s_eta = C.sector_sensitivity(x0, sec, dn, dparams={"diss": (0.0, 1.0)}).dlam
        s_eps = C.sector_sensitivity(x0, sec, dn, dx0=C.make_x0(p, 0.3, direction="eps")).dlam
        c.check("dense_point d_eta, d_eps == sector_sensitivity", abs(d["d_eta"] - s_eta) < 1e-12*abs(s_eta)
                and abs(d["d_eps"] - s_eps) < 1e-12*abs(s_eps))
        c.check("dense_point stores the mode (psi max 1) and counts growing modes",
                abs(np.abs(d["psi"]).max() - 1) < 1e-6 and int(d["n_grow"]) >= 1)
        c.check("read back with compute=False", C.dense_point(100.0, 0.3, sg, root=root, compute=False)["n"] == d["n"])
        with pytest.raises(KeyError):
            C.dense_point(100.0, 0.31, sg, root=root, compute=False)
        pp = C.prop_point(100.0, 0.3, sg, (64, 16), root=root, dt=0.05)
        c.check(f"prop_point {pp['values'][0].real:.10f} == dense {d['values'][0].real:.10f}",
                abs(pp["values"][0] - d["values"][0]) < 1e-7)
        sp = C.split_point(100.0, root=root)
        c.check("split_point: R+/R- degenerate, opposite d lam/d eps, both P",
                abs(sp["lam_+1+1"] - sp["lam_-1+1"]) < 1e-12 and abs(sp["d_+1-1"] + sp["d_-1-1"]) < 1e-9
                and abs(sp["d_+1+1"] + sp["d_-1+1"]) < 1e-9)
        base, fd = C.fd_point(100.0, 0.3, 0.4, root=root)
        for k in ("eps", "eta", "alpha"):
            ext = (4*fd[f"fd_{k}_0.002"] - fd[f"fd_{k}_0.004"])/3
            c.check(f"fd_point {k}: Richardson FD {ext.real:.8f} vs autodiff {base['d_' + k].real:.8f}",
                    abs(ext - base[f"d_{k}"]) < 1e-6*abs(base[f"d_{k}"]))
        rows = C.eps_table(100.0, eps_list=(0.3,), root=root)
        c.check("eps_table: no row without both pair members", rows == [])
        C.dense_point(100.0, 0.3, {"R": 1, "P": -1}, root=root)
        # a solve cached as a FAILURE (eps = 0.6, P+) is skipped by the readers, not raised
        with pytest.raises(RuntimeError):
            C.cached(
                lambda: (_ for _ in ()).throw(RuntimeError("synthetic")), root=root, kind="dense", S=100.0, eps=0.6,
                alpha=0.0, T_sign=-1, signs=C._sig(sg), nx=64, ny=16, Lx=C.LX0, sens=["eps", "eta"])
        with pytest.raises(RuntimeError, match="cached failure"):
            C.dense_point(100.0, 0.6, sg, root=root, compute=False)
        rows = C.eps_table(100.0, eps_list=(0.3, 0.6), root=root)
        c.check("eps_table: the mean of the two members, missing eps skipped",
                len(rows) == 1 and abs(rows[0]["mean"] - 0.5*(rows[0]["lam_p"] + rows[0]["lam_m"]).real) < 1e-15)
        a_, b_ = (C.dense_point(100.0, 0.3, sg_, root=root, compute=False) for sg_ in C.RP)
        r0 = rows[0]
        c.check("eps_table: lam_p/lam_m are the P+ / P- tops, split = |dlam|/mean",
                r0["lam_p"] == a_["values"][0] and r0["lam_m"] == b_["values"][0]
                and abs(r0["split"] - abs(r0["lam_p"] - r0["lam_m"])/r0["mean"]) < 1e-12)
        c.check("eps_table: d_eps, d_eta are member means; kappa and n_grow are (P+, P-) tuples; mode from P+",
                all(r0[f"d_{k}"] == 0.5*(a_[f"d_{k}"] + b_[f"d_{k}"]).real for k in ("eps", "eta"))
                and a_["d_eps"] != b_["d_eps"]
                and r0["kappa_eps"] == (float(a_["kappa_eps"]), float(b_["kappa_eps"]))
                and r0["n_grow"] == (int(a_["n_grow"]), int(b_["n_grow"]))
                and np.array_equal(r0["psi"], a_["psi"]) and not np.array_equal(a_["psi"], b_["psi"]))
        c.check("other readers are empty on an empty root",
                C.alt_table(100.0, root=root) == [] and C.harm_table(root=root) == [] and C.alpha_table(root=root) == []
                and C.conv_table(root=root) == {} and C.box_table(root=root) == {} and C.fd_table(root=root) == [])
        c.check("split_table reads the split point", [r["S"] for r in C.split_table(root=root)] == [100.0]
                if C.S_SPLIT[0] == 100.0 else True)


@pytest.mark.fp64
def test_ivp_growth_is_the_eigenvalue():
    # the independent route on a tiny grid: the nonlinear solver's perturbation growth equals
    # the frozen-x0 eigenvalue of that sector (S = 1e4: the x0 drift over the run is ~1e-3)
    C = _run_script()
    sg = {"R": 1, "P": -1}
    t, d = C.ivp_growth(1e4, 0.3, sg, (64, 16), dt=0.02, amp=1e-9, t_end=60.0, every=100)
    p = C.make_params(1e4, 64, 16, Lx=C.LX0)
    kg = jr.setup_kgrids(p)
    lam = C.sector_dense(C.make_x0(p, 0.3), C.Sector(kg, p, -1, sg))["values"][0].real
    sel = t >= 30
    g = np.polyfit(t[sel], np.log(d[sel]), 1)[0]
    with checks() as c:
        c.check(f"IVP growth {g:.5f} vs eigenvalue {lam:.5f}", abs(g/lam - 1) < 5e-3)
        rel = d[-1]/float(np.linalg.norm(np.asarray(C.make_x0(p, 0.3))))
        c.check(f"the perturbation stayed linear: |delta|/|x0| = {rel:.1e} at the end", rel < 1e-4)


@pytest.mark.fp64
def test_jobs_and_make_data():
    C = _run_script()
    with checks() as c:
        n = {ph: len(C._jobs(ph)) for ph in C.PHASES}
        c.check(f"job counts {n}", n["eps"] == 2*len(C.S_LIST)*len(C.EPS) and n["split"] == len(C.S_SPLIT)
                and n["alpha"] == 2*len(C.ALPHAS)*len(C.ALPHA_S) and n["harm"] == 4*len(C.EPS_HARM)
                and n["conv"] == 2*sum(len(g) for g in C.CONV.values()))
        with pytest.raises(ValueError):
            C._jobs("bogus")
        seen, logs = [], []
        fake = [(f"j{i}", (lambda i=i: seen.append(i) or {"values": np.array([i + 0j])})) for i in range(5)]
        fake.insert(3, ("bad", lambda: (_ for _ in ()).throw(RuntimeError("x"))))
        with _patched(C, _jobs=lambda name: list(fake)):
            C.make_data(phases=["eps"], log=logs.append, stride=(1, 2))
            c.check(f"stride (1, 2) takes j1, bad, j4 -- a failure does not stop the phase: {seen}", seen == [1, 4])
            c.check("a failed job is logged, the phase INCOMPLETE",
                    any("FAILED bad" in l for l in logs) and "INCOMPLETE" in logs[-1])
            seen.clear()
            C.make_data(wall_budget=-1.0, phases=["eps"], log=logs.append)
            c.check("past the deadline nothing runs", seen == [])


@pytest.mark.fp64
def test_readers():
    # every reader against the records it assembles, on tiny grids in a temporary root
    C = _run_script()
    g = (64, 16)
    cfg = dict(GRID={100.0: g, 1000.0: g}, EPS_ALT=(0.3,), EPS_HARM=(0.3,), ALPHAS=(0.0, 0.4), EPS_ALPHA=0.3,
               HARM_BOX=((20.0, g),), DCONV=((100.0, 0.3, g),), BOX=((100.0, 0.3, 30.0, (96, 16)),),
               CONV={(100.0, 0.3): (g,)}, CONV_ALPHA=((100.0, 0.3, 0.4, g),), SMALL_GRID={100.0: g},
               EPS_SMALL=(1e-3,), S_SPLIT=(100.0,), SPLIT_NX={100.0: 64})
    with tempfile.TemporaryDirectory() as root, _patched(C, **cfg), checks() as c:
        rec = {}
        for sg in C.RM:
            rec[str(sg)] = C.dense_point(100.0, 0.3, sg, sens=(), root=root)
        a = C.alt_table(100.0, root=root)
        c.check("alt_table: the R- pair mean", len(a) == 1 and abs(a[0]["mean"] - 0.5*(rec[str(C.RM[0])]["values"][0]
                + rec[str(C.RM[1])]["values"][0]).real) < 1e-15)
        hs = [C.dense_point(1e3, 0.3, sg, T_sign=1, sens=(), root=root) for sg in C.RP + C.RM]
        h = C.harm_table(root=root)
        c.check("harm_table: four T+ tops in RP + RM order", len(h) == 1
                and all(h[0]["tops"][i] == hs[i]["values"][0] for i in range(4)))
        al = [C.dense_point(100.0, 0.3, sg, alpha=a_, sens=("alpha", "eta"), root=root) for a_ in (0.0, 0.4) for sg in C.PR]
        at = C.alpha_table(root=root, S=100.0)
        c.check("alpha_table: PR+/PR- (XR+) means and their d/d alpha", len(at) == 2
                and abs(at[1]["mean"] - 0.5*(al[2]["values"][0] + al[3]["values"][0]).real) < 1e-15
                and abs(at[1]["d_alpha"] - 0.5*(al[2]["d_alpha"] + al[3]["d_alpha"]).real) < 1e-15
                and abs(at[0]["d_alpha"]) < 1e-10)
        c.check("alpha_table: d_eta is the member mean, im the larger |Im|",
                abs(at[1]["d_eta"] - 0.5*(al[2]["d_eta"] + al[3]["d_eta"]).real) < 1e-15 and al[2]["d_eta"] != al[3]["d_eta"]
                and at[1]["im"] == max(abs(al[2]["values"][0].imag), abs(al[3]["values"][0].imag)))
        hb = C.dense_point(1e3, 0.6, {"R": -1, "P": -1}, T_sign=1, grid_=g, Lx=20.0, sens=(), root=root)
        c.check("harm_box_table", list(C.harm_box_table(root=root)) == [20.0]
                and C.harm_box_table(root=root)[20.0]["values"][0] == hb["values"][0])
        dc = [C.dense_point(100.0, 0.3, sg, grid_=g, sens=(), root=root) for sg in C.RP]
        c.check("dconv_table", abs(C.dconv_table(root=root)[(100.0, 0.3)]
                                   - 0.5*(dc[0]["values"][0] + dc[1]["values"][0]).real) < 1e-15)
        bx = [C.dense_point(100.0, 0.3, sg, grid_=(96, 16), Lx=30.0, sens=(), root=root) for sg in C.RP]
        c.check("box_table (Lx = 30 at the same dx)", abs(C.box_table(root=root)[30.0]["mean"]
                                                         - 0.5*(bx[0]["values"][0] + bx[1]["values"][0]).real) < 1e-15)
        pp = [C.prop_point(100.0, 0.3, sg, g, root=root) for sg in C.RP]
        ct = C.conv_table(root=root)
        c.check("conv_table: the propagator pair == the dense pair",
                abs(ct[(100.0, 0.3, g)]["mean"] - C.dconv_table(root=root)[(100.0, 0.3)]) < 1e-7
                and ct[(100.0, 0.3, g)]["res"] <= 1e-4
                and ct[(100.0, 0.3, g)]["res"] == max(pp[0]["residuals"][0], pp[1]["residuals"][0]))
        pa = [C.prop_point(100.0, 0.3, sg, g, alpha=0.4, root=root) for sg in C.PR]
        c.check("conv_table: the alpha point == the dense alpha pair",
                abs(C.conv_table(root=root)[(100.0, 0.3, g, 0.4)]["mean"] - at[1]["mean"]) < 1e-7)
        for e in (0.0, 1e-3):
            for sg in C.RP:
                C.dense_point(100.0, e, sg, grid_=g, sens=("eps",), root=root)
            for sg in C.RM:
                C.dense_point(100.0, e, sg, grid_=g, sens=(), root=root)
        sm = C.small_table(100.0, root=root)
        c.check("small_table: R+ above, R- below lam0, with d lam(R+)/d eps > 0", len(sm) == 2
                and sm[1]["plus"] > sm[0]["plus"] > sm[1]["minus"] and sm[0]["d_plus"] > 0
                and abs(sm[0]["plus"] - sm[0]["minus"]) < 1e-12)
        C.split_point(100.0, root=root)
        st_ = C.split_table(root=root)
        rec = st_[0]["rec"]
        c.check("split_table: c(R+) = -c(R-), each the P-mean of its own record entries",
                len(st_) == 1 and abs(st_[0]["c"] + st_[0]["c_minus"]) < 1e-9 and abs(st_[0]["c"] - sm[0]["d_plus"]) < 1e-9
                and st_[0]["c_minus"] == 0.5*(rec["d_-1+1"] + rec["d_-1-1"]).real
                and st_[0]["c"] == 0.5*(rec["d_+1+1"] + rec["d_+1-1"]).real
                and st_[0]["lam0"] == 0.5*(rec["lam_+1+1"] + rec["lam_+1-1"]).real)


# ============================================================ (vii) the cached headline numbers

@pytest.mark.fp64
def test_headline_numbers():
    C = _run_script()
    if not os.path.isdir(os.path.join(C.DATA_ROOT, "solves")):
        pytest.skip("examples/data/tearing-chain not generated (python examples/tearing_chain_run.py)")
    ET = {S: C.eps_table(S) for S in C.S_LIST}
    # a reader change that drops points must not turn this gate into a skip
    assert all(len(ET[S]) == len(C.EPS) for S in C.S_LIST), {S: len(v) for S, v in ET.items()}
    with checks() as c:
        r3 = {r["eps"]: r for r in ET[1e3]}
        r2 = {r["eps"]: r for r in ET[1e2]}
        c.check(f"S = 1e3, eps = 0.3: coalescence gamma = {r3[0.3]['mean']:.6f} (P-mean)", abs(r3[0.3]["mean"] - 0.126095) < 1e-5)
        c.check("eps = 0: the P-mean is the Harris pair's (1D, nx = 512: 0.0131306 / 0.0384408)",
                abs(r3[0.0]["mean"]/0.0131306 - 1) < 2e-3 and abs(r2[0.0]["mean"]/0.0384408 - 1) < 2e-3)
        for S, rows in ET.items():
            g = [r["mean"] for r in rows]
            c.check(f"S = {S:g}: gamma increases with eps", all(np.diff(g) > 0))
            c.check(f"S = {S:g}: one growing eigenvalue in each R+ sector", all(r["n_grow"] == (1, 1) for r in rows))
        c.check("gamma decreases with S at every eps", all(r2[e]["mean"] > r3[e]["mean"] for e in C.EPS))
        sS = {e: -r3[e]["d_eta"]/1e3/r3[e]["mean"] for e in C.EPS}
        c.check(f"S = 1e3 exponents s_S: eps = 0 {sS[0.0]:+.3f} (resistive), eps >= 0.2 within 0.05 of 0 (ideal): "
                f"{[round(sS[e], 3) for e in C.EPS if e >= 0.2]}", sS[0.0] < -0.4 and all(abs(sS[e]) < 0.05 for e in C.EPS if e >= 0.2))
        se = {e: e*r3[e]["d_eps"]/r3[e]["mean"] for e in C.EPS if e > 0}
        c.check(f"s_eps = d ln gamma/d ln eps ~ 1/2 at eps = 0.03-0.2 (S = 1e3): {[round(se[e], 3) for e in (0.03, 0.1, 0.2)]}",
                all(0.4 < se[e] < 0.6 for e in (0.03, 0.1, 0.2)))
        sp = C.split_table()
        c.check(f"split: c(R+) = -c(R-) at every S, c grows with S: {[round(r['c'], 4) for r in sp]}",
                len(sp) == len(C.S_SPLIT) and all(abs(r["c"] + r["c_minus"]) < 1e-9*abs(r["c"]) for r in sp)
                and all(np.diff([r["c"] for r in sp]) > 0))
        for S in C.S_LIST:
            alt = {r["eps"]: r["mean"] for r in C.alt_table(S)}
            c.check(f"S = {S:g}: alternating mode decreases monotonically with eps and is stabilised by eps = 0.3 "
                    f"({alt.get(0.3, np.nan):.1e}: the neutral cluster)", len(alt) == len(C.EPS_ALT) and abs(alt[0.3]) < 1e-8
                    and all(np.diff([alt[e] for e in C.EPS_ALT]) < 0))
        hb = C.harm_box_table()
        c.check(f"T+ sliding mode is a box artefact: gamma(Lx = 20)/gamma(Lx = 30) = "
                f"{hb[20.0]['values'][0].real/hb[30.0]['values'][0].real:.1f}",
                len(hb) == 3 and hb[20.0]["values"][0].real > 10*hb[30.0]["values"][0].real)
        at = C.alpha_table(S=1e3)
        # NB this band gates the PRODUCTION-grid caches (256x128), which run 2-10% low at alpha >= 0.6; the resolved ratios
        # (review, independent single-sheet code) are 0.671/0.400/0.226 at alpha = 0.6/0.8/0.9, i.e. 5-19% above 1 - alpha^2
        rat = [r["mean"]/at[0]["mean"]/(1 - r["alpha"]**2) for r in at]
        c.check(f"alpha: gamma(alpha)/gamma(0)/(1 - alpha^2) = {np.round(rat, 4)} (within 0 .. +8%)",
                len(at) == len(C.ALPHAS) and all(0.999 < q < 1.08 for q in rat) and abs(at[0]["d_alpha"]) < 1e-9)
        bx = C.box_table()
        c.check(f"box: Lx = 20/30/40 P-means {r3[0.3]['mean']:.6f} {bx[30.0]['mean']:.6f} {bx[40.0]['mean']:.6f} within 3e-3",
                max(abs(bx[L]["mean"]/r3[0.3]["mean"] - 1) for L in (30.0, 40.0)) < 3e-3)
        iv = C.ivp_point(S=np.inf, compute=False)
        dn = C.dense_point(np.inf, C.IVP["eps"], C.IVP["signs"], grid_=C.IVP["grid"], sens=(), compute=False)
        # reviewer's independent route (2026-10-03): a single-sheet eigencode sharing no code with taranis (isolated
        # chain, walls at |x| = 25, sinh-mapped 4th-order FD in x, cos/sin Galerkin in y, sparse shift-invert;
        # scratchpad item3_review/indep/fadeev.py), converged to ~1e-6: gamma = 0.126205 at S = 1e3, eps = 0.3.
        # The P-mean at the resolved 512x256 grid matches it; each member alone is ~3% off.
        g512 = C.conv_table()[(1e3, 0.3, (512, 256))]
        c.check(f"512x256 P-mean {g512['mean']:.6f} vs isolated-sheet 0.126205: within 2e-4; members "
                f"{g512['lam_p'].real:.4f}/{g512['lam_m'].real:.4f} are not",
                abs(g512["mean"]/0.126205 - 1) < 2e-4 and min(abs(g512["lam_p"].real/0.126205 - 1),
                                                            abs(g512["lam_m"].real/0.126205 - 1)) > 0.02)
        c.check(f"eta = 0: nonlinear-solver growth {iv['gamma']:.6f} vs frozen eigenvalue {dn['values'][0].real:.6f}",
                abs(iv["gamma"]/dn["values"][0].real - 1) < 3e-3)
        for r in C.fd_table():
            for k in ("eps", "eta", "alpha"):
                key = f"fd_{k}_{C.FD_H[1]:g}"
                if key in r["fd"]:
                    ext = (4*r["fd"][key] - r["fd"][f"fd_{k}_{C.FD_H[0]:g}"])/3
                    rel = abs(ext - r["base"][f"d_{k}"])/abs(r["base"][f"d_{k}"])
                    c.check(f"FD vs autodiff, eps={r['eps']} alpha={r['alpha']} d/d{k}: {rel:.1e}", rel < 1e-6)
        cv = C.conv_table()
        pts = [k for k in cv if len(k) == 3 and k[:2] == (1e3, 0.3)]
        c.check(f"S = 1e3, eps = 0.3: {len(pts)} grids up to 512x256 within 2e-3 of the dense value, residuals < 1e-10",
                len(pts) == 4 and all(abs(cv[k]["mean"]/r3[0.3]["mean"] - 1) < 2e-3 and cv[k]["res"] < 1e-10 for k in pts))


# ============================================================ (viii) reviewer gates (rung 2b item 3 review, 2026-10-03)

@pytest.mark.fp64
def test_jobs_write_what_the_readers_read():
    # every cache key a reader (and the notebook) asks for is one some make_data job writes, and vice versa:
    # a job whose sens / T_sign / Lx / alpha / S drifted from its reader would otherwise be caught only on a
    # full regeneration (the headline gate reads the existing caches)
    C = _run_script()
    written, read = set(), set()

    def fake(compute, root=None, recompute=False, **cfg):
        txt = C._key(**cfg)[1]
        if compute is None:
            read.add(txt)
            raise KeyError(txt)
        written.add(txt)
        return {}
    with _patched(C, cached=fake):
        for ph in C.PHASES:
            for _, fn in C._jobs(ph):
                fn()
        for S in C.S_LIST:
            C.eps_table(S)
            C.alt_table(S)
        for S in C.SMALL_GRID:
            C.small_table(S)
        for S in C.ALPHA_S:
            C.alpha_table(S=S)
        C.harm_table(); C.split_table(); C.fd_table(); C.conv_table(); C.harm_box_table()
        C.dconv_table(); C.box_table()
        for S in C.IVP_S:
            with pytest.raises(KeyError):
                C.ivp_point(S=S, compute=False)
            with pytest.raises(KeyError):
                C.dense_point(S, C.IVP["eps"], C.IVP["signs"], grid_=C.IVP["grid"], sens=(), compute=False)
    # fd_table reads its fd record only after the base dense record: read the fd keys with the base stubbed
    with _patched(C, cached=fake, dense_point=lambda *a, **k: {}):
        for S, e, a_ in C.FD_POINTS:
            with pytest.raises(KeyError):
                C.fd_point(S, e, a_, compute=False)
    with checks() as c:
        c.check(f"every read key is written ({len(read)} read, {len(written)} written; missing "
                f"{sorted(read - written)[:2]})", read <= written)
        c.check(f"every written key is read (orphans {sorted(written - read)[:2]})", written <= read)


@pytest.mark.fp64
def test_dense_point_record_contents():
    # the stored mode is the LEADING eigenvector, n_grow is the full-spectrum count above 1e-9, and fd_point's
    # alpha = 0 base is the R+ P+ (P = +1) member the notebook names
    C = _run_script()
    with tempfile.TemporaryDirectory() as root, _patched(C, **_tiny(C)), checks() as c:
        sg = {"R": 1, "P": 1}
        d = C.dense_point(100.0, 0.3, sg, root=root)
        p, kg = _setup(nx=64, ny=16)
        sec = C.Sector(kg, p, -1, sg)
        dn = C.sector_dense(C.make_x0(p, 0.3), sec)
        phi, psi = C.mode_fields(sec, dn["right"][:, 0])
        c.check("stored phi, psi == mode_fields of the leading right vector",
                np.allclose(d["psi"], psi, atol=1e-6) and np.allclose(d["phi"], phi, atol=1e-6))
        c.check(f"n_grow == #(Re lam > 1e-9) of the full sector spectrum ({int(d['n_grow'])})",
                int(d["n_grow"]) == int(np.sum(dn["values"].real > 1e-9)))
        al_ = C.dense_point(100.0, 0.1, {"R": -1, "P": 1}, root=root, sens=())
        c.check(f"n_grow counts a slow growing mode too (alternating, eps = 0.1: {al_['values'][0].real:.2e})",
                0 < al_["values"][0].real < 1e-2 and int(al_["n_grow"]) == 1)
        # eps = 0 without the R label: the coalescence/alternating pair is exactly degenerate in this sector, so
        # eigenvalue_sensitivity refuses -- recorded as nan + the message, never raised, never a number
        r0 = C.dense_point(100.0, 0.0, {"P": 1}, root=root, sens=("eps",))
        c.check("a refused sensitivity is recorded (nan, refused_eps)", np.isnan(r0["d_eps"]) and "refused_eps" in r0
                and abs(r0["values"][0] - r0["values"][1]) < 1e-10)
        base, fd = C.fd_point(100.0, 0.3, 0.0, root=root)
        c.check("fd_point(alpha = 0) is the R+ P+ record", base["values"][0] == d["values"][0])
        for k in ("eps", "eta"):
            ext = (4*fd[f"fd_{k}_0.002"] - fd[f"fd_{k}_0.004"])/3
            c.check(f"fd_point(alpha = 0) {k}: Richardson FD vs autodiff", abs(ext - base[f"d_{k}"]) < 1e-6*abs(base[f"d_{k}"]))


@pytest.mark.fp64
def test_ivp_point_tiny():
    # ivp_point's slope is the fit over the configured window (and the window is part of the key)
    C = _run_script()
    cfg = dict(C.IVP, grid=(64, 16), t_end=40.0, every=100, window=(20.0, 40.0))
    with tempfile.TemporaryDirectory() as root, _patched(C, IVP=cfg), checks() as c:
        r = C.ivp_point(root=root)
        t, d = r["t"], r["delta"]
        sel = (t >= 20.0) & (t <= 40.0)
        c.check(f"gamma {float(r['gamma']):.6f} == window fit", abs(r["gamma"] - np.polyfit(t[sel], np.log(d[sel]), 1)[0]) < 1e-12)
        c.check("... and differs from the all-time fit (the window matters)",
                abs(r["gamma"] - np.polyfit(t, np.log(d), 1)[0]) > 1e-4)
        with pytest.raises(KeyError):
            C.ivp_point(root=root, compute=False, window=(10.0, 40.0))


if __name__ == "__main__":
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
