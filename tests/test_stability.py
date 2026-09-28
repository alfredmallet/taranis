# taranis.stability (plans/AUTODIFF_PLAN.md rung 0): the Jacobian J = L + N'(x0) of the
# discrete RMHD RHS by autodiff, its ky blocks, real-coordinate form, adjoint and eigensolvers.
# Gates (the plan's numbering):
#   1. FD gate: J v against [N(x0+e v) - N(x0-e v)]/2e + L v, for a 1D and a 2D x0 in 2D and
#      a z-dependent x0 under FD-z and z_spectral, random v and a computed eigenvector. RMHD's
#      N is linear + quadratic, so the central difference is EXACT up to round-off (no O(e^2)
#      term to observe) and the one-sided difference's error is exactly e*Q(v), Q the
#      quadratic part: both are asserted, plus the O(e) slope.
#   2. x0 = 0 => J = L: in 2D the ky block is diagonal -nu k^2h, -eta k^2h; in 3D it equals
#      the analytic L ENTRYWISE -- z_spectral separable (nu == eta), putzer2 (nu != eta),
#      z_diss_k != 0, and FD-z's periodic 4th-order Alfven stencil plus its d4/dz4 filter.
#   3. independent Fourier transcription of the linearized RMHD block about
#      psi0 = A cos(qx), phi0 = alpha A cos(qx) (derivation in the test's docstring), and its
#      GDI twin about phi0 = A cos(qx), N0 = B cos(qx) (L included; the one home for the
#      hand transcriptions of a bracket linearization).
#   4. ky_block_matrix against the full jvp_operator on a random ky-column vector (2D, 3D
#      FD-z, 3D z_spectral), the complex-linearity of a ky column (and its absence on the
#      full fields space), and the matrix-free ky_block_operator.
#   5. transpose_operator against the conjugate transpose of the dense block; the real
#      inner-product adjoint identity on the full space; real_operator's rmatvec against M^T.
# Plus: RealCoords round trips, eig_dense's left/right pairs (residuals measured against
# ||J||, so a nu = 0 null vector is gated too), shift_invert's dense-LU and GMRES paths (ky
# block and real coordinates) against the dense spectrum, and shift_invert raising rather
# than returning a wrong pair (sigma on an eigenvalue, loose solves, ARPACK non-convergence).
# fp64 only (round-off tolerances), except test_zero_state_smoke_both_precisions.
# pytest: `pytest tests/test_stability.py`. Script: `python tests/test_stability.py`.
from _rmhd_testing import bootstrap, checks, ctx, fit_order, fresh_params, make_state

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


def _ic3(x, y, z):
    # ky-independent, z-dependent
    return jnp.stack([0.3*jnp.sin(x)*(1 + 0.2*jnp.cos(z)) + 0*y,
                      jnp.cos(x)*(1 + 0.3*jnp.sin(z)) + 0*y])


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
    box3 = dict(dims=3, nx=12, ny=8, nz=8, Lx=2*np.pi, Ly=4*np.pi, Lz=2*np.pi,
                diss=(0.01, 0.02), hyper=1)
    cases = [("1D x0", _p2d(), _tearing_ic, "block"),
             ("2D x0", _p2d(nx=12, ny=12, Ly=2*np.pi), _twod_ic, "real"),
             ("3D FD-z x0(x,z)", _params(comm_backend="serial", z_diss=0.5, **box3), _ic3, "block"),
             ("3D z_spectral x0(x,z)", _params(z_spectral=True, **box3), _ic3, "block")]
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
                # Q(v): the quadratic part of N (FD-z carries a linear part in N too)
                N0, Nv, Jn = N(x0), (N(v) + N(-v))/2, Jv - Lv(v)
                errs = []
                for eps in (1e-2, 1e-3, 1e-4):
                    fwd = (N(x0 + eps*v) - N0)/eps - Jn
                    errs.append(np.linalg.norm(fwd))
                    e = np.linalg.norm(fwd - eps*Nv)/np.linalg.norm(Jn)
                    c.check(f"{name}, {vname}: one-sided FD error is exactly eps*Q(v) at "
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

        # 3D: the block against the analytic L entrywise. z_spectral: diagonal
        # -diss_f k_perp^2h - z_diss_k kz^4, phi <-> psi coupling +i kz. FD-z: the periodic
        # stencil (8 (f[iz+1] - f[iz-1]) - (f[iz+2] - f[iz-2]))/(12 dz) of the OTHER field, and
        # the filter -z_diss (dz/2)^4 (1, -4, 6, -4, 1)/dz^4 on the same field.
        box3 = dict(dims=3, nx=12, ny=8, nz=8, Lx=2*np.pi, Ly=4*np.pi, Lz=3.0)
        for name, diss, hyper, kw in (
                ("z_spectral separable", (0.02, 0.02), 1, dict(z_spectral=True)),
                ("z_spectral putzer2", (0.03, 0.01), 1, dict(z_spectral=True)),
                ("z_spectral separable z_diss_k", (0.02, 0.02), 2,
                 dict(z_spectral=True, eqpars=dict(z_diss_k=0.4))),
                ("z_spectral putzer2 z_diss_k", (0.03, 0.01), 1,
                 dict(z_spectral=True, eqpars=dict(z_diss_k=0.4))),
                ("FD-z", (0.03, 0.01), 1, dict(comm_backend="serial", z_diss=0.7))):
            params = fresh_params(diss=diss, hyper=hyper, **box3, **kw)
            kgrid = jr.setup_kgrids(params)
            x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
            for iky in (1, 2):
                B, idx = stability.ky_block_matrix(x0, kgrid, params, iky)
                H = _analytic_L_block(idx, params, iky, diss, hyper)
                err = np.abs(B - H).max()/np.abs(H).max()
                c.check(f"{name} iky={iky}: zero-state block == analytic L entrywise "
                        f"(rel {err:.1e}, n={len(idx)})", err < 1e-13, f"rel {err:.3e}")


def _analytic_L_block(idx, params, iky, diss, hyper):
    kx = np.fft.fftfreq(params.nx)*params.nx*2*np.pi/params.Lx
    kz = np.fft.fftfreq(params.nz)*params.nz*2*np.pi/params.Lz
    ky = iky*2*np.pi/params.Ly
    zdk = params.eqpars.get("z_diss_k", 0.0)
    pos = {(int(f), int(iz), int(ix)): r for r, (f, iz, ix) in enumerate(idx)}
    H = np.zeros((len(idx), len(idx)), dtype=complex)
    for r, (f, iz, ix) in enumerate(idx):
        H[r, r] = -diss[f]*(kx[ix]**2 + ky**2)**hyper
        if params.z_spectral:
            H[r, r] += -zdk*kz[iz]**4
            H[r, pos[(1 - f, iz, ix)]] += 1j*kz[iz]
        else:
            for s, w in ((1, 8.0), (-1, -8.0), (2, -1.0), (-2, 1.0)):
                H[r, pos[(1 - f, (iz + s) % params.nz, ix)]] += w/(12*params.dz)
            for s, w in ((0, 6.0), (1, -4.0), (-1, -4.0), (2, 1.0), (-2, 1.0)):
                H[r, pos[(f, (iz + s) % params.nz, ix)]] += \
                    -params.z_diss*(params.dz/2)**4*w/params.dz**4
    return H


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


def _gdi_cosine_block(idx, params, iky, A, B, q):
    """Hand transcription of GDI's ky-column block about phi0 = A cos(qx), N0 = B cos(qx).

    Shares nothing with gdi.NonlinearTerm / gdi.linear_matrix but the conventions. Fields
    (N, phi) (gdi.py header, docs/numerics.md "GDI"), {a,b} = a_x b_y - a_y b_x, w = lap phi
    the vorticity; the nonlinear RHS (plans/GDI_PLAN.md: brackets {phi, N}, {phi, w}) is
        d_t N = -{phi, N},      d_t w = -{phi, w},
    with phi_k evolved: d_t phi_k = -(1/k^2) (d_t w)_k. Every bracket of two functions of x
    vanishes, so (phi0, N0) is an exact ideal steady state. Linearize with dphi, dN ~
    exp(i ky y), using {a0(x), b} = a0' i ky b and {a, b0(x)} = -i ky a b0':
        d(d_t N) = -{phi0, dN} - {dphi, N0} = i ky [-phi0' dN + N0' dphi]
        d(d_t w) = -{phi0, dw} - {dphi, w0} = i ky [-phi0' dw + w0' dphi]
    With s = sin qx: phi0' = -A q s, N0' = -B q s, w0 = -q^2 phi0 so w0' = A q^3 s, and on an
    input mode dw = -k'^2 dphi (k'^2 = kx'^2 + ky^2):
        d(d_t N) = i ky q s [A dN - B dphi]
        d(d_t w) = i ky q s A (q^2 - k'^2) dphi
    (s g)_kx = (g_(kx-q) - g_(kx+q))/(2i), so output kx couples to kx -+ q with weight
    +-(ky q/2); the phi row carries the extra -1/k^2 of the OUTPUT mode. The block of J adds
    L's 2x2 per mode, from docs/numerics.md "GDI" (gamma_par = gpar_fac nu_in k^2):
        L[N,N] = -gamma_par - diss k^2h        L[N,phi]   = gamma_par + i ky/Ln
        L[phi,N] = gpar_fac nu_in - i ky nu_in v0/k^2
        L[phi,phi] = -nu_in - gpar_fac nu_in - diss k^2h
    """
    ep = params.eqpars
    nu, g, Ln, v0 = ep["nu_in"], ep["gpar_fac"], ep["Ln"], ep["v0"]
    diss, hyper = ep["diss"], ep["hyper"]
    nx = params.nx
    kx = np.fft.fftfreq(nx)*nx*2*np.pi/params.Lx
    ky = iky*2*np.pi/params.Ly
    qi = int(round(q*params.Lx/(2*np.pi)))
    pos = {(int(f), int(ix)): r for r, (f, _, ix) in enumerate(idx)}
    H = np.zeros((len(idx), len(idx)), dtype=complex)
    c0 = ky*q/2
    for r, (fo, _, ixo) in enumerate(idx):
        k2o = kx[ixo]**2 + ky**2
        gp = g*nu*k2o
        if fo == 0:                           # N row
            H[r, pos[(0, ixo)]] += -gp - diss*k2o**hyper
            H[r, pos[(1, ixo)]] += gp + 1j*ky/Ln
        else:                                 # phi row
            H[r, pos[(0, ixo)]] += g*nu - 1j*ky*nu*v0/k2o
            H[r, pos[(1, ixo)]] += -nu - g*nu - diss*k2o**hyper
        for sign, shift in ((+1.0, -qi), (-1.0, +qi)):
            ixi = (ixo + shift) % nx
            if (0, ixi) not in pos:
                continue                     # outside the kept set: the truncation
            assert abs(kx[ixi] - (kx[ixo] + shift*2*np.pi/params.Lx)) < 1e-12
            k2i = kx[ixi]**2 + ky**2
            if fo == 0:
                H[r, pos[(0, ixi)]] += c0*sign*A
                H[r, pos[(1, ixi)]] += -c0*sign*B
            else:
                H[r, pos[(1, ixi)]] += -(1.0/k2o)*c0*sign*A*(q**2 - k2i)
    return H


@pytest.mark.fp64
def test_gdi_fourier_transcription():
    # GDI's twin of gate 3: N' != 0 with an exact answer (the GDI eigenvalue gates in
    # test_stability_general.py sit at x0 = 0, where N' vanishes)
    params = fresh_params(dims=2, nx=16, ny=16, Lx=2*np.pi, Ly=2*np.pi, eqtype="GDI",
                          comm_backend="serial",
                          eqpars=dict(Ln=1.0, nu_in=0.3, v0=2.0, diss=0.01, hyper=1,
                                      gpar_fac=0.1))
    kgrid = jr.setup_kgrids(params)
    A, B, q = 0.7, 0.4, 2.0                  # phi0 = A cos 2x, N0 = B cos 2x
    x0 = _x0(params, lambda x, y: jnp.stack([B*jnp.cos(q*x) + 0*y, A*jnp.cos(q*x) + 0*y]))
    with checks() as c:
        for iky in (1, 3):
            J, idx = stability.ky_block_matrix(x0, kgrid, params, iky)
            H = _gdi_cosine_block(idx, params, iky, A, B, q)
            err = np.abs(J - H).max()/np.abs(H).max()
            c.check(f"GDI iky={iky}: dense block == hand transcription entrywise (max rel "
                    f"{err:.1e}, n={len(idx)})", err < 1e-13, f"max rel {err:.3e}")
            H0 = _gdi_cosine_block(idx, params, iky, 0.0, 0.0, q)
            for rows, name in ((idx[:, 0] == 0, "N"), (idx[:, 0] == 1, "phi")):
                band = np.abs((H - H0)[rows]).max()
                c.check(f"GDI iky={iky}: the {name} rows' N' coupling is nontrivial "
                        f"({band:.2f})", band > 1e-2, f"{band:.3e}")


# ------------------------------------------------------------------------------ gate 4

def _gate4_cases():
    box3 = dict(dims=3, nx=12, ny=8, nz=8, Lx=2*np.pi, Ly=4*np.pi, Lz=2*np.pi,
                diss=(0.01, 0.02), hyper=1)
    return [("2D", _p2d(), _tearing_ic),
            ("3D FD-z", _params(comm_backend="serial", z_diss=0.5, **box3), _ic3),
            ("3D z_spectral", _params(z_spectral=True, **box3), _ic3)]


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
        ky1 = np.zeros(x0.shape, dtype=complex)
        ky1[1, 0, 1, 1] = 1e-6
        with pytest.raises(ValueError, match="ky-independent"):
            stability.ky_block_matrix(ky1, kgrid, params, 1)
        c.check("ky_block_matrix rejects an x0 whose only ky != 0 content is at ky = 1", True)
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
        lres = lres/np.maximum(np.abs(r.values), r.scale)
        c.check(f"eig_dense: left pairs w^H J = lambda w^H (max {lres.max():.1e})",
                lres.max() < 1e-10, f"{lres.max():.3e}")
        lead = r.values[0]
        c.check(f"cos x shear tearing at k = 0.5 is unstable (gamma = {lead.real:.5f})",
                lead.real > 0)
        sigma = lead + 0.01 + 0.01j
        want = r.values[np.argsort(np.abs(r.values - sigma))[:2]]
        dense = stability.shift_invert(B, sigma, k=2)
        _check_sorted(c, "shift_invert dense LU", dense, sigma)
        e = np.abs(dense.values - want).max()/abs(lead)
        c.check(f"shift_invert dense LU: 2 nearest eigenvalues (rel {e:.1e}, residual "
                f"{dense.residuals.max():.1e})", e < 1e-10 and dense.residuals.max() < 1e-10,
                f"rel {e:.3e}, residual {dense.residuals.max():.3e}")
        op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
        mf = stability.shift_invert(op, sigma, k=2, tol=1e-12, restart=len(idx))
        _check_sorted(c, "shift_invert GMRES ky block", mf, sigma)
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
        _check_sorted(c, "shift_invert GMRES real coords", mf, sigma)
        e = _match(mf.values, want)/abs(top)
        c.check(f"shift_invert GMRES, real coords, 2D x0: 2 eigenvalues nearest sigma (rel "
                f"{e:.1e}, residual {mf.residuals.max():.1e}, n={coords.n})",
                e < 1e-9 and mf.residuals.max() < 1e-9,
                f"rel {e:.3e}, residual {mf.residuals.max():.3e}")


def _check_sorted(c, name, res, sigma):
    d = np.abs(res.values - sigma)
    c.check(f"{name}: values sorted by |lambda - sigma|", np.all(np.diff(d) >= 0), f"{d}")


def _nu0_block():
    # nu = 0 cos x current sheet: the ky block has exactly one lambda ~ 0 (a null vector)
    params = _params(dims=2, nx=64, ny=8, Lx=2*np.pi, Ly=4*np.pi, diss=(0.0, 0.01), hyper=1)
    kgrid = jr.setup_kgrids(params)
    x0 = _x0(params, lambda x, y: jnp.stack([0*x + 0*y, jnp.cos(x) + 0*y]))
    B, idx = stability.ky_block_matrix(x0, kgrid, params, 1)
    return B, idx, x0, kgrid, params


@pytest.mark.fp64
def test_nu0_residuals_and_sigma_on_an_eigenvalue():
    B, idx, x0, kgrid, params = _nu0_block()
    norm2 = np.linalg.norm(B, 2)

    def absres(res):
        V = res.right
        return np.linalg.norm(B @ V - V*res.values[None, :], axis=0)/np.linalg.norm(V, axis=0)
    with checks() as c:
        r = stability.eig_dense(B)
        i0 = int(np.argmin(np.abs(r.values)))
        c.check(f"nu = 0 block has a null vector (|lambda| = {abs(r.values[i0]):.1e})",
                abs(r.values[i0]) < 1e-12, f"{abs(r.values[i0]):.3e}")
        c.check(f"eig_dense scale bounds ||J||_2 ({r.scale:.3f} >= {norm2:.3f})",
                norm2 <= r.scale*(1 + 1e-12) and r.scale < 2*norm2, f"{r.scale}")
        c.check(f"eig_dense, nu = 0: every right residual is small, the null vector's too (max "
                f"{r.residuals.max():.1e}, null {r.residuals[i0]:.1e})",
                r.residuals.max() < 1e-12, f"{r.residuals}")

        # sigma exactly on / within round-off of the zero eigenvalue: the dense LU is singular
        for sigma in (0.0, r.values[i0], 1e-14):
            with pytest.raises(RuntimeError, match="numerically singular"):
                stability.shift_invert(B, sigma, k=6)
        c.check("dense shift_invert raises at sigma on the null eigenvalue", True)
        with pytest.raises(RuntimeError, match="numerically singular"):
            stability.shift_invert(np.diag(np.arange(1.0, 30.0)), 3.0, k=3)
        c.check("dense shift_invert raises at sigma exactly on an eigenvalue", True)

        # near it: either a RuntimeError, or pairs that are eigenpairs by an independent check
        for sigma in (1e-12, 1e-11, 1e-10, 1e-8, 1e-6):
            try:
                res = stability.shift_invert(B, sigma, k=6)
            except RuntimeError as err:
                c.check(f"sigma={sigma:g}: raised ({str(err)[:60]}...)", True)
                continue
            a = absres(res).max()
            want = r.values[np.argsort(np.abs(r.values - sigma))[:6]]
            e = np.abs(res.values - want).max()
            c.check(f"sigma={sigma:g}: returned pairs are eigenpairs (|Jv - lambda v|/|v| "
                    f"{a:.1e}, values off by {e:.1e})", a < 1e-8*norm2 and e < 1e-8,
                    f"abs residual {a:.3e}, values {e:.3e}")
            _check_sorted(c, f"sigma={sigma:g}", res, sigma)
        res = stability.shift_invert(B, 1e-6, k=6)
        c.check(f"sigma=1e-6 returns the null vector (max residual {res.residuals.max():.1e})",
                abs(res.values[0]) < 1e-12 and res.residuals.max() < 1e-12,
                f"{res.values}, {res.residuals}")

        # the matrix-free path at sigma = 0: raises
        op, _ = stability.ky_block_operator(x0, kgrid, params, 1)
        with pytest.raises(RuntimeError):
            stability.shift_invert(op, 0.0, k=6, restart=len(idx))
        c.check("GMRES shift_invert raises at sigma on the null eigenvalue", True)

        # the residual guard itself: GMRES solves looser than tol cannot pass
        gamma = r.values[0].real
        with pytest.raises(RuntimeError, match="not eigenpairs"):
            stability.shift_invert(op, gamma + 1e-3, k=6, restart=len(idx), gmres_rtol=1e-5)
        c.check("GMRES shift_invert raises when its solves are looser than tol", True)
        with pytest.raises(RuntimeError, match="not eigenpairs"):
            stability.shift_invert(B, gamma + 1e-3, k=6, residual_tol=1e-20)
        c.check("dense shift_invert raises over residual_tol", True)
        res = stability.shift_invert(B, gamma + 1e-3, k=6)
        c.check(f"sigma = gamma + 1e-3: residuals small (max {res.residuals.max():.1e}), the "
                f"~0 eigenvalue's included", res.residuals.max() < 1e-12, f"{res.residuals}")


def test_shift_invert_sorts_by_distance():
    # random dense matrices, where ARPACK's own order is NOT by |lambda - sigma|
    import scipy.linalg
    import scipy.sparse.linalg as spla
    rng = np.random.default_rng(_RNG_SEED)
    n, k = 60, 8
    raw_unsorted = 0
    with checks() as c:
        for trial in range(6):
            A = rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n))
            sigma = complex(rng.standard_normal(), rng.standard_normal())
            lu = scipy.linalg.lu_factor(A - sigma*np.eye(n))
            inv = spla.LinearOperator((n, n), dtype=complex,
                                      matvec=lambda b, lu=lu: scipy.linalg.lu_solve(lu, b))
            mu = spla.eigs(inv, k=k, which="LM", tol=1e-12, v0=np.ones(n))[0]
            raw_unsorted += bool(np.any(np.diff(np.abs(1/mu)) < 0))
            res = stability.shift_invert(A, sigma, k=k, v0=np.ones(n))
            _check_sorted(c, f"trial {trial}", res, sigma)
            ev = np.linalg.eigvals(A)
            want = ev[np.argsort(np.abs(ev - sigma))[:k]]
            e = np.abs(res.values - want).max()
            c.check(f"trial {trial}: the {k} nearest eigenvalues, in order (err {e:.1e})",
                    e < 1e-8, f"{e:.3e}")
        c.check(f"discriminating: ARPACK's raw order was unsorted in {raw_unsorted}/6 trials",
                raw_unsorted > 0)


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
        with pytest.raises(RuntimeError, match="GMRES did not reach"):
            stability.shift_invert(aslinearoperator(np.diag(np.arange(1.0, 30.0))), 3.0, k=3)
        c.check("GMRES shift_invert raises at sigma exactly on an eigenvalue", True)


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
