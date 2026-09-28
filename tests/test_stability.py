# taranis.stability (plans/AUTODIFF_PLAN.md rung 0): the Jacobian J = L + N'(x0) of the
# discrete RMHD RHS by autodiff, its ky blocks, real-coordinate form, adjoint and eigensolvers.
# Gates (the plan's numbering):
#   1. FD gate: J v against [N(x0+e v) - N(x0-e v)]/2e + L v, for a 1D and a 2D x0, random v
#      and a computed eigenvector. RMHD's N is exactly quadratic in 2D, so the central
#      difference is EXACT up to round-off (no O(e^2) term to observe) and the one-sided
#      difference's error is exactly e*N(v): both are asserted, plus the O(e) slope.
#   2. x0 = 0 => J = L: ky-block eigenvalues -nu k^2h, -eta k^2h in 2D; +-i kz - eta k_perp^2
#      under z_spectral; the 4th-order FD-z Alfven stencil (+ its d4/dz4 filter) under FD-z.
#   3. independent Fourier transcription of the linearized RMHD block about
#      psi0 = A cos(qx), phi0 = alpha A cos(qx) (derivation in the test's docstring).
#   4. ky_block_matrix against the full jvp_operator on a random ky-column vector (2D, 3D
#      FD-z, 3D z_spectral), the complex-linearity of a ky column (and its absence on the
#      full fields space), and the matrix-free ky_block_operator.
#   5. transpose_operator against the conjugate transpose of the dense block; the real
#      inner-product adjoint identity on the full space; real_operator's rmatvec against M^T.
# Plus: RealCoords round trips, eig_dense's left/right pairs, shift_invert's dense-LU and
# GMRES paths (ky block and real coordinates) against the dense spectrum.
# fp64 only (round-off tolerances), except test_zero_state_smoke_both_precisions.
# pytest: `pytest tests/test_stability.py`. Script: `python tests/test_stability.py`.
from _rmhd_testing import bootstrap, checks, ctx, fit_order, make_state

bootstrap()

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import aslinearoperator

import jax.numpy as jnp

import taranis as jr
from taranis import _precision, stability
from taranis.physics import construct_rhs, equation_registry

_RNG_SEED = 20260927


def _params(**kw):
    # ctx() caches (params, kgrid); kgrids are rebuilt by the tests with setup_kgrids
    return ctx(**kw)[0]


def _p2d(**kw):
    base = dict(dims=2, nx=24, ny=8, Lx=2*np.pi, Ly=2*np.pi/0.5, diss=(0.01, 0.01), hyper=1)
    base.update(kw)
    return _params(**base)


def _x0(params, ic):
    return jr.initialize(ic, params).fields


def _tearing_ic(x, y):
    # cos x current sheet pair: phi0 = 0.3 psi0, a shear-tearing state
    psi = jnp.cos(x) + 0*y
    return jnp.stack([0.3*psi, psi])


def _twod_ic(x, y):
    phi = 0.5*jnp.sin(x)*jnp.cos(2*y) + 0.2*jnp.cos(x + y)
    psi = jnp.cos(x)*jnp.cos(y) + 0.3*jnp.sin(2*x + y)
    return jnp.stack([phi, psi])


def _rand(shape, rng):
    return rng.standard_normal(shape) + 1j*rng.standard_normal(shape)


def _embed(idx, u, shape, iky):
    v = np.zeros(shape, dtype=complex)
    v[idx[:, 0], idx[:, 1], idx[:, 2], iky] = u
    return v


def _gather(idx, g, iky):
    return np.asarray(g)[idx[:, 0], idx[:, 1], idx[:, 2], iky]


def _nonlinear(params):
    # N(fields): the solver's RHS terms, built here without taranis.stability
    rhs = construct_rhs(equation_registry[params.eqtype])
    state = make_state(params)
    kgrid = jr.setup_kgrids(params)
    return lambda f: np.asarray(rhs(state._replace(fields=jnp.asarray(f, dtype=_precision.ctype)),
                                    kgrid, params)[0])


def _match(a, b):
    # max |a_i - b_perm(i)| over the optimal pairing
    cost = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].max())


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


# ------------------------------------------------------------------------------ gate 1

@pytest.mark.fp64
def test_fd_gate():
    rng = np.random.default_rng(_RNG_SEED)
    cases = [("1D x0", _p2d(), _tearing_ic, "block"),
             ("2D x0", _p2d(nx=12, ny=12, Ly=2*np.pi), _twod_ic, "real")]
    with checks() as c:
        for name, params, ic, eigpath in cases:
            kgrid = jr.setup_kgrids(params)
            x0 = np.asarray(_x0(params, ic))
            J = stability.jvp_operator(x0, kgrid, params)
            N = _nonlinear(params)
            Lv = lambda v: np.asarray(kgrid.lin.apply_L(jnp.asarray(v)))
            if eigpath == "block":
                B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
                vec = _embed(idx, stability.eig_dense(B).right[:, 0], x0.shape, 1)
            else:
                M, coords = stability.real_matrix(x0, kgrid, params)
                w, V = np.linalg.eig(M)
                vec = np.asarray(coords.unpack(jnp.asarray(V[:, np.argmax(w.real)].real)))
            scale = np.abs(x0).max()
            for vname, v in (("random v", _rand(x0.shape, rng)), ("eigenvector", vec)):
                v = v*scale/np.abs(v).max()
                Jv = np.asarray(J(v))
                for eps in (1e-3, 1e-4):
                    fd = Lv(v) + (N(x0 + eps*v) - N(x0 - eps*v))/(2*eps)
                    e = _relerr(fd, Jv)
                    c.check(f"{name}, {vname}: central FD matches J v at eps={eps:g} "
                            f"(rel {e:.2e})", e < 1e-9, f"rel {e:.3e}")
                N0, Nv, Jn = N(x0), N(v), Jv - Lv(v)
                errs = []
                for eps in (1e-2, 1e-3, 1e-4):
                    fwd = (N(x0 + eps*v) - N0)/eps - Jn
                    errs.append(np.linalg.norm(fwd))
                    e = np.linalg.norm(fwd - eps*Nv)/np.linalg.norm(Jn)
                    c.check(f"{name}, {vname}: one-sided FD error is exactly eps*N(v) at "
                            f"eps={eps:g} (rel {e:.2e})", e < 1e-8, f"rel {e:.3e}")
                order = fit_order([1e-2, 1e-3, 1e-4], errs)
                c.check(f"{name}, {vname}: one-sided FD converges at O(eps) (order "
                        f"{order:.3f})", abs(order - 1.0) < 0.02, f"order {order:.4f}")


# ------------------------------------------------------------------------------ gate 2

@pytest.mark.fp64
def test_zero_state_is_L():
    with checks() as c:
        for hyper in (1, 2):
            nu, eta = 0.03, 0.01
            params = _p2d(nx=24, Lx=3.0, Ly=2.0, diss=(nu, eta), hyper=hyper)
            kgrid = jr.setup_kgrids(params)
            x0 = np.zeros((2, 1, params.nx, params.ny//2 + 1), dtype=complex)
            for iky in (1, 2):
                B, idx = stability.ky_block_matrix(x0, kgrid, params, iky)
                kx = np.fft.fftfreq(params.nx)*params.nx*2*np.pi/params.Lx
                ksq = kx[idx[:, 2]]**2 + (iky*2*np.pi/params.Ly)**2
                want = np.where(idx[:, 0] == 0, -nu, -eta)*ksq**hyper
                offd = np.abs(B - np.diag(np.diag(B))).max()
                c.check(f"2D hyper={hyper} iky={iky}: block is diagonal", offd == 0.0,
                        f"max off-diagonal {offd:.3e}")
                err = np.abs(np.diag(B) - want).max()/np.abs(want).max()
                c.check(f"2D hyper={hyper} iky={iky}: diagonal is -diss k^2h (rel {err:.1e})",
                        err < 1e-14, f"rel {err:.3e}")
                ev = stability.eig_dense(B).values
                err = _match(ev, want)/np.abs(want).max()
                c.check(f"2D hyper={hyper} iky={iky}: eigenvalues -nu k^2h, -eta k^2h",
                        err < 1e-14, f"rel {err:.3e}")

        # z_spectral, nu == eta: damped Alfven waves +-i kz - eta k_perp^2 exactly
        eta = 0.02
        params = _params(dims=3, nx=12, ny=8, nz=8, Lx=2*np.pi, Ly=4*np.pi, Lz=3.0,
                     z_spectral=True, diss=(eta, eta), hyper=1)
        kgrid = jr.setup_kgrids(params)
        x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
        B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        kx = np.fft.fftfreq(params.nx)*params.nx*2*np.pi/params.Lx
        kz = np.fft.fftfreq(params.nz)*params.nz*2*np.pi/params.Lz
        one = idx[idx[:, 0] == 0]            # one (kz, kx) per mode; the pair gives +-
        kperp2 = kx[one[:, 2]]**2 + (2*np.pi/params.Ly)**2
        want = np.concatenate([1j*kz[one[:, 1]] - eta*kperp2, -1j*kz[one[:, 1]] - eta*kperp2])
        err = _match(stability.eig_dense(B).values, want)/np.abs(want).max()
        c.check(f"z_spectral: eigenvalues +-i kz - eta k_perp^2 (rel {err:.1e})", err < 1e-13,
                f"rel {err:.3e}")

        # FD-z: the 4th-order stencil i(8 sin t - sin 2t)/(6 dz) couples phi <-> psi, the
        # filter adds -z_diss (dz/2)^4 (6 - 8 cos t + 2 cos 2t)/dz^4, t = kz dz
        params = _params(dims=3, nx=12, ny=8, nz=8, Lx=2*np.pi, Ly=4*np.pi, Lz=3.0,
                     comm_backend="serial", z_diss=0.7, diss=(eta, eta), hyper=1)
        kgrid = jr.setup_kgrids(params)
        B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        dz = params.dz
        kxs = sorted(set(idx[:, 2]))
        th = 2*np.pi*np.arange(params.nz)/params.nz
        D1 = (8*np.sin(th) - np.sin(2*th))/(6*dz)
        D4 = (6 - 8*np.cos(th) + 2*np.cos(2*th))/dz**4
        want = []
        for ix in kxs:
            base = -eta*(kx[ix]**2 + (2*np.pi/params.Ly)**2) - 0.7*(dz/2)**4*D4
            want.extend(base + 1j*D1)
            want.extend(base - 1j*D1)
        err = _match(stability.eig_dense(B).values, np.array(want))/np.abs(want).max()
        c.check(f"FD-z: eigenvalues are the stencil's +-i D1(kz) - eta k^2 - filter "
                f"(rel {err:.1e})", err < 1e-13, f"rel {err:.3e}")


def test_zero_state_smoke_both_precisions():
    # any precision: at x0 = 0 the operator is L, and the ky block is its diagonal
    params = _p2d(diss=(0.03, 0.01))
    kgrid = jr.setup_kgrids(params)
    x0 = jnp.zeros((2, 1, params.nx, params.ny//2 + 1), dtype=_precision.ctype)
    rng = np.random.default_rng(_RNG_SEED)
    v = jnp.asarray(_rand(x0.shape, rng), dtype=_precision.ctype)
    tol = 1e-13 if _precision.precision == "64" else 1e-5
    with checks() as c:
        Jv = stability.jvp_operator(x0, kgrid, params)(v)
        c.check("J v dtype is the field dtype", Jv.dtype == _precision.ctype, str(Jv.dtype))
        e = _relerr(Jv, kgrid.lin.apply_L(v))
        c.check("x0 = 0: J v == L v", e < tol, f"rel {e:.3e}")
        B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        L = np.asarray(kgrid.lin.L)
        want = L[idx[:, 0], idx[:, 1], idx[:, 2], 1]
        e = _relerr(np.diag(B), want)
        c.check("x0 = 0: ky block diagonal == L", e < tol, f"rel {e:.3e}")


# ------------------------------------------------------------------------------ gate 3

def _cosine_block(idx, params, iky, A, alpha, q, nu, eta, hyper):
    # Hand transcription (shares nothing with rmhd.NonlinearTerm but the conventions). With
    # {a,b} = a_x b_y - a_y b_x, J = lap psi, w = lap phi, the RMHD RHS is
    #     d_t w = {psi, J} - {phi, w},      d_t psi = -{phi, psi}     (+ L),
    # and phi_k is evolved: d_t phi_k = -(1/k^2) (d_t w)_k. Linearize about psi0(x),
    # phi0(x) with dphi, dpsi ~ exp(i ky y): {a0(x), b} = a0' i ky b, {a, b0(x)} = -i ky a b0'.
    #     d(d_t w)   = i ky [psi0' dJ - J0' dpsi - phi0' dw + w0' dphi]
    #     d(d_t psi) = i ky [psi0' dphi - phi0' dpsi]
    # psi0 = A cos qx, phi0 = alpha psi0, s = sin qx: psi0' = -A q s, J0' = A q^3 s,
    # phi0' = -alpha A q s, w0' = alpha A q^3 s; with dJ = -k'^2 dpsi, dw = -k'^2 dphi on an
    # input mode k'^2 = kx'^2 + ky^2:
    #     d(d_t w)   = i ky A q s g,  g = (k'^2 - q^2)(dpsi - alpha dphi)
    #     d(d_t psi) = i ky A q s h,  h = alpha dpsi - dphi
    # and (s g)_kx = (g_{kx-q} - g_{kx+q})/(2i), so each output kx couples to kx -+ q with
    # weights +-(ky A q/2). The phi row carries the extra -1/k^2 of the output mode; L adds
    # -nu k^2h (phi) and -eta k^2h (psi) on the diagonal.
    nx = params.nx
    kx = np.fft.fftfreq(nx)*nx*2*np.pi/params.Lx
    ky = iky*2*np.pi/params.Ly
    qi = int(round(q*params.Lx/(2*np.pi)))
    pos = {(int(f), int(ix)): r for r, (f, _, ix) in enumerate(idx)}
    B = np.zeros((len(idx), len(idx)), dtype=complex)
    c0 = ky*A*q/2
    for r, (fo, _, ixo) in enumerate(idx):
        k2o = kx[ixo]**2 + ky**2
        B[r, r] += -(nu if fo == 0 else eta)*k2o**hyper
        for sign, shift in ((+1.0, -qi), (-1.0, +qi)):
            ixi = (ixo + shift) % nx
            if (0, ixi) not in pos:
                continue                     # outside the kept set: the truncation
            assert abs(kx[ixi] - (kx[ixo] + shift*2*np.pi/params.Lx)) < 1e-12
            k2i = kx[ixi]**2 + ky**2
            if fo == 0:
                w = -(1.0/k2o)*c0*sign*(k2i - q**2)
                B[r, pos[(1, ixi)]] += w
                B[r, pos[(0, ixi)]] += -alpha*w
            else:
                B[r, pos[(1, ixi)]] += c0*sign*alpha
                B[r, pos[(0, ixi)]] += -c0*sign
    return B


@pytest.mark.fp64
def test_fourier_transcription():
    nu, eta, hyper = 0.02, 0.005, 2
    params = _p2d(nx=32, Lx=4*np.pi, Ly=2*np.pi/0.4, diss=(nu, eta), hyper=hyper)
    kgrid = jr.setup_kgrids(params)
    q, A = 1.5, 0.7                      # box harmonic 3 of Lx = 4 pi
    with checks() as c:
        for alpha in (0.0, 0.6, 1.0):
            ic = lambda x, y, a=alpha: jnp.stack([a*A*jnp.cos(q*x) + 0*y, A*jnp.cos(q*x) + 0*y])
            x0 = _x0(params, ic)
            for iky in (1, 2):
                B, idx = stability.ky_block_matrix(x0, kgrid, params, iky)
                H = _cosine_block(idx, params, iky, A, alpha, q, nu, eta, hyper)
                err = np.abs(B - H).max()/np.abs(H).max()
                c.check(f"alpha={alpha} iky={iky}: dense block == hand transcription "
                        f"entrywise (max rel {err:.1e}, n={len(idx)})", err < 1e-13,
                        f"max rel {err:.3e}")
                band = np.abs(H - np.diag(np.diag(H))).max()
                c.check(f"alpha={alpha} iky={iky}: coupling is nontrivial", band > 1e-3,
                        f"{band:.3e}")


# ------------------------------------------------------------------------------ gate 4

def _gate4_cases():
    def ic3(x, y, z):
        return jnp.stack([0.3*jnp.sin(x)*(1 + 0.2*jnp.cos(z)) + 0*y,
                          jnp.cos(x)*(1 + 0.3*jnp.sin(z)) + 0*y])
    box3 = dict(dims=3, nx=12, ny=8, nz=8, Lx=2*np.pi, Ly=4*np.pi, Lz=2*np.pi,
                diss=(0.01, 0.02), hyper=1)
    return [("2D", _p2d(), _tearing_ic),
            ("3D FD-z", _params(comm_backend="serial", z_diss=0.5, **box3), ic3),
            ("3D z_spectral", _params(z_spectral=True, **box3), ic3)]


@pytest.mark.fp64
def test_block_matches_full_operator():
    rng = np.random.default_rng(_RNG_SEED)
    with checks() as c:
        for name, params, ic in _gate4_cases():
            kgrid = jr.setup_kgrids(params)
            x0 = _x0(params, ic)
            J = stability.jvp_operator(x0, kgrid, params)
            for iky in (1, 2):
                B, idx = stability.ky_block_matrix(x0, kgrid, params, iky)
                u = _rand(len(idx), rng)
                v = _embed(idx, u, x0.shape, iky)
                Jv = np.asarray(J(v))
                Bu = B @ u
                e = _relerr(_gather(idx, Jv, iky), Bu)
                c.check(f"{name} iky={iky}: block @ u == column of J v (rel {e:.1e})",
                        e < 1e-13, f"rel {e:.3e}")
                rest = Jv.copy()
                rest[idx[:, 0], idx[:, 1], idx[:, 2], iky] = 0
                e = np.linalg.norm(rest)/np.linalg.norm(Bu)
                c.check(f"{name} iky={iky}: J v has nothing outside the kept column "
                        f"(rel {e:.1e})", e < 1e-13, f"rel {e:.3e}")
                e = _relerr(np.asarray(J(1j*v)), 1j*Jv)
                c.check(f"{name} iky={iky}: complex-linear on the column, J(iv) == iJ(v) "
                        f"(rel {e:.1e})", e < 1e-14, f"rel {e:.3e}")
                op, _ = stability.ky_block_operator(x0, kgrid, params, iky)
                e = _relerr(op.matvec(u), Bu)
                c.check(f"{name} iky={iky}: ky_block_operator matvec == block @ u",
                        e < 1e-13, f"rel {e:.3e}")
        # contrast: on the whole fields space (self-conjugate rows included) J is only
        # real-linear, which is why dense blocks are ky columns or real coordinates
        params = _p2d(nx=12, ny=12, Ly=2*np.pi)
        kgrid = jr.setup_kgrids(params)
        x0 = _x0(params, _twod_ic)
        J = stability.jvp_operator(x0, kgrid, params)
        v = _rand(x0.shape, rng)
        e = _relerr(np.asarray(J(1j*v)), 1j*np.asarray(J(v)))
        c.check(f"2D x0, full fields space: J(iv) != iJ(v) (rel {e:.2f})", e > 1e-2,
                f"rel {e:.3e}")
        with pytest.raises(ValueError, match="ky-independent"):
            stability.ky_block_matrix(x0, kgrid, params, 1)
        c.check("ky_block_matrix rejects a 2D x0", True)
        for bad in (0, params.ny//2):
            with pytest.raises(ValueError, match="iky"):
                stability.ky_block_index(kgrid, params, bad)
        c.check("ky_block_index rejects the self-conjugate rows", True)


# ------------------------------------------------------------------------------ gate 5

@pytest.mark.fp64
def test_transpose_operator():
    rng = np.random.default_rng(_RNG_SEED)
    with checks() as c:
        for name, params, ic in _gate4_cases():
            kgrid = jr.setup_kgrids(params)
            x0 = _x0(params, ic)
            JH = stability.transpose_operator(x0, kgrid, params)
            B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
            u = _rand(len(idx), rng)
            want = B.conj().T @ u
            e = _relerr(_gather(idx, JH(_embed(idx, u, x0.shape, 1)), 1), want)
            c.check(f"{name}: transpose_operator == block^H on the column (rel {e:.1e})",
                    e < 1e-13, f"rel {e:.3e}")
            op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
            e = _relerr(op.rmatvec(u), want)
            c.check(f"{name}: ky_block_operator rmatvec == block^H u", e < 1e-13,
                    f"rel {e:.3e}")

        params = _p2d(nx=12, ny=12, Ly=2*np.pi)
        kgrid = jr.setup_kgrids(params)
        x0 = _x0(params, _twod_ic)
        J = stability.jvp_operator(x0, kgrid, params)
        JH = stability.transpose_operator(x0, kgrid, params)
        u, v = _rand(x0.shape, rng), _rand(x0.shape, rng)
        lhs = np.vdot(u, np.asarray(J(v))).real
        rhs = np.vdot(np.asarray(JH(u)), v).real
        e = abs(lhs - rhs)/abs(lhs)
        c.check(f"2D x0: Re<u, J v> == Re<J^H u, v> on the full space (rel {e:.1e})",
                e < 1e-13, f"rel {e:.3e}")
        M, coords = stability.real_matrix(x0, kgrid, params)
        op, _ = stability.real_operator(x0, kgrid, params, coords)
        w = rng.standard_normal(coords.n)
        e = _relerr(op.rmatvec(w), M.T @ w)
        c.check(f"2D x0: real_operator rmatvec == M^T w (rel {e:.1e})", e < 1e-13,
                f"rel {e:.3e}")
        e = _relerr(op.matvec(w), M @ w)
        c.check(f"2D x0: real_operator matvec == M w (rel {e:.1e})", e < 1e-13,
                f"rel {e:.3e}")


# ---------------------------------------------------------------- coordinates / solvers

@pytest.mark.fp64
def test_real_coords_round_trip():
    rng = np.random.default_rng(_RNG_SEED)
    with checks() as c:
        for name, params in (("2D", _p2d(nx=12, ny=12, Ly=2*np.pi)),
                             ("3D z_spectral", _params(dims=3, nx=12, ny=8, nz=6, Lx=2*np.pi,
                                                   Ly=2*np.pi, Lz=2*np.pi, z_spectral=True,
                                                   diss=(0.01, 0.01), hyper=1))):
            kgrid = jr.setup_kgrids(params)
            coords = stability.real_coords(kgrid, params)
            real = rng.standard_normal((2, params.nz, params.nx, params.ny))
            f = jr.grids.fft(jnp.asarray(real), params)*jr.grids.dealias_mask(params)
            f = f.astype(_precision.ctype)
            back = coords.unpack(coords.pack(f))
            e = float(jnp.abs(back - f).max()/jnp.abs(f).max())
            c.check(f"{name}: unpack(pack(f)) == f for a real dealiased field (rel {e:.1e})",
                    e < 1e-15, f"rel {e:.3e}")
            u = rng.standard_normal(coords.n)
            e = float(np.abs(np.asarray(coords.pack(coords.unpack(jnp.asarray(u)))) - u).max())
            c.check(f"{name}: pack(unpack(u)) == u", e < 1e-15, f"{e:.3e}")
            g = coords.unpack(jnp.asarray(u))
            again = jr.grids.fft(jr.grids.ifft(g, params), params)
            e = float(jnp.abs(again - g).max()/jnp.abs(g).max())
            c.check(f"{name}: unpack(u) is the transform of a real field (rel {e:.1e})",
                    e < 1e-14, f"rel {e:.3e}")
            # a real field has one real degree of freedom per kept mode of the FULL
            # (two-sided ky) transform
            ix = np.fft.fftfreq(params.nx)*params.nx
            iy = np.fft.fftfreq(params.ny)*params.ny
            full = ((ix[:, None]/(params.nx/3))**2 + (iy[None, :]/(params.ny/3))**2 < 1).sum()
            if params.z_spectral:
                iz = np.fft.fftfreq(params.nz)*params.nz
                full = full*int((np.abs(iz) < params.nz/3).sum())
            else:
                full = full*params.nz
            c.check(f"{name}: n = nfields * kept real degrees of freedom ({coords.n})",
                    coords.n == 2*full, f"n={coords.n}, want {2*full}")


@pytest.mark.fp64
def test_eig_dense_and_shift_invert():
    params = _p2d(diss=(0.01, 0.01))
    kgrid = jr.setup_kgrids(params)
    x0 = _x0(params, _tearing_ic)
    with checks() as c:
        B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
        r = stability.eig_dense(B)
        c.check("eig_dense: sorted by decreasing real part",
                np.all(np.diff(r.values.real) <= 0))
        c.check(f"eig_dense: right residuals (max {r.residuals.max():.1e})",
                r.residuals.max() < 1e-10, f"{r.residuals.max():.3e}")
        lres = np.linalg.norm(r.left.conj().T @ B - r.values[:, None]*r.left.conj().T, axis=1)
        lres = lres/np.maximum(np.abs(r.values), 1e-300)
        c.check(f"eig_dense: left pairs w^H J = lambda w^H (max {lres.max():.1e})",
                lres.max() < 1e-10, f"{lres.max():.3e}")
        lead = r.values[0]
        c.check(f"cos x shear tearing at k = 0.5 is unstable (gamma = {lead.real:.5f})",
                lead.real > 0)
        sigma = lead + 0.01 + 0.01j
        want = r.values[np.argsort(np.abs(r.values - sigma))[:2]]
        dense = stability.shift_invert(B, sigma, k=2)
        e = np.abs(dense.values - want).max()/abs(lead)
        c.check(f"shift_invert dense LU: 2 nearest eigenvalues (rel {e:.1e}, residual "
                f"{dense.residuals.max():.1e})", e < 1e-10 and dense.residuals.max() < 1e-10,
                f"rel {e:.3e}, residual {dense.residuals.max():.3e}")
        op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
        mf = stability.shift_invert(op, sigma, k=2, tol=1e-12, restart=len(idx))
        e = np.abs(mf.values - want).max()/abs(lead)
        c.check(f"shift_invert GMRES on the ky block: same eigenvalues (rel {e:.1e}, "
                f"residual {mf.residuals.max():.1e})", e < 1e-9 and mf.residuals.max() < 1e-9,
                f"rel {e:.3e}, residual {mf.residuals.max():.3e}")
        # the real-coordinate operator of the whole state holds the ky block's spectrum
        rop, coords = stability.real_operator(x0, kgrid, params)
        full = stability.shift_invert(rop, lead + 1e-3, k=1, tol=1e-12, restart=coords.n)
        e = np.abs(full.values[0] - lead)/abs(lead)
        c.check(f"shift_invert GMRES, real coords, 1D x0: finds the ky-block growth rate "
                f"(rel {e:.1e})", e < 1e-9, f"rel {e:.3e}")

        # a 2D x0: matrix-free real coordinates against the dense real matrix
        params = _p2d(nx=12, ny=12, Ly=2*np.pi)
        kgrid = jr.setup_kgrids(params)
        x0 = _x0(params, _twod_ic)
        M, coords = stability.real_matrix(x0, kgrid, params)
        ev = np.linalg.eigvals(M)
        top = ev[np.argmax(ev.real)]
        sigma = top + 0.02 + 0.01j
        want = ev[np.argsort(np.abs(ev - sigma))[:2]]
        rop, _ = stability.real_operator(x0, kgrid, params, coords)
        mf = stability.shift_invert(rop, sigma, k=2, tol=1e-12, restart=coords.n)
        e = _match(mf.values, want)/abs(top)
        c.check(f"shift_invert GMRES, real coords, 2D x0: 2 eigenvalues nearest sigma (rel "
                f"{e:.1e}, residual {mf.residuals.max():.1e}, n={coords.n})",
                e < 1e-9 and mf.residuals.max() < 1e-9,
                f"rel {e:.3e}, residual {mf.residuals.max():.3e}")


def test_shift_invert_fails_loudly():
    # a near-degenerate cluster that k cuts through: ARPACK must raise, not spin
    rng = np.random.default_rng(_RNG_SEED)
    n = 200
    V = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
    D = np.concatenate([[1.0], 1e-9*(rng.standard_normal(n - 1) + 1j*rng.standard_normal(n - 1))])
    A = V @ np.diag(D) @ np.linalg.inv(V)
    with checks() as c:
        with pytest.raises(RuntimeError, match="ARPACK did not converge"):
            stability.shift_invert(A, 0.9, k=6, arpack_maxiter=2)
        c.check("shift_invert raises on ARPACK non-convergence", True)
        op = aslinearoperator(rng.standard_normal((60, 60)))
        with pytest.raises(RuntimeError, match="GMRES did not reach"):
            stability.shift_invert(op, 0.1, k=1, restart=2, maxiter=1)
        c.check("shift_invert raises on GMRES non-convergence", True)


def test_rejects_unsupported():
    with checks() as c:
        params = _p2d(forcing=True, forcing_power=1.0, fshell=(1, 3))
        kgrid = jr.setup_kgrids(params)
        x0 = jnp.zeros((2, 1, params.nx, params.ny//2 + 1), dtype=_precision.ctype)
        with pytest.raises(ValueError, match="UNFORCED"):
            stability.jvp_operator(x0, kgrid, params)
        c.check("forced Parameters are rejected", True)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
