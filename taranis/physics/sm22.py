# Squire & Mallet 2022 constant-|B| growth rule (J. Plasma Phys. 88, eqs 2.1-2.2), fully
# spectral: dims=2 (the 2.5D problem: 3 components varying in (x,y)) or dims=3 with
# z_spectral. Derivation and conventions: docs/numerics.md "SM22".
#
#   dt B = dB + curl(grad(phi) x B),     dB = B - <B>,
#   phi:  P_band[ B . P_band curl(grad(phi) x B) ] = P_band[ -B . dB - (gamma/2)(|B|^2 - <|B|^2>) ]
#
# i.e. d|B|^2/dt = 0 (up to a constant) is imposed GALERKIN-wise on the retained 2/3 band --
# the band the dealiased RHS lives in -- and phi is sought in that band. Band -> band makes
# the system square (a collocation constraint on the full grid is overdetermined for a
# band-limited phi, and BiCGSTAB would stall on it). The out-of-band part of |B|^2 is the
# honest unresolved burden, measured by diagnostics.sm22 (CLAUDE.md aw_papers rule 1).
#
# Rules:
# - The +dB growth is LINEAR and k-local: it is L = +1 on every k != 0 (0 at k = 0), so the
#   integrating factor treats it exactly and <B> (the k = 0 mode, carried in the fields like
#   CMHD's background) is preserved bitwise. Optional hyperdissipation joins L.
# - Induction is curl form, so k.dB is a pairwise-cancelling sum (div B stays at round-off).
# - The phi solve lives in grad (set_timestep needs max|grad phi|, and set_timestep sees only
#   grads). The Krylov vectors are k-space coefficients on the band minus k = 0; the
#   preconditioner is k-local and free.
# - The last phi is the solver aux (EquationRecipe.aux_init_func): the warm start threaded
#   through every stage and step by the IF steppers. It is not snapshotted.
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .. import grids
from .. import comms
from .. import _precision


class SM22Grads(NamedTuple):
    B: jnp.ndarray            # real-space B, (3, nz, nx, ny)
    u: tuple                  # real-space grad(phi) components; 0.0 for d/dz in dims=2
    phi: jnp.ndarray          # k-space phi on the band, (nz, nkx, nky)
    aux: jnp.ndarray          # = phi: the next solve's warm start (EquationRecipe contract)
    its: jnp.ndarray          # Krylov iterations taken
    res: jnp.ndarray          # final relative residual (band, k-space norm)


_EQPARS_KEYS = ("gamma", "maxiter", "rtol", "fixed_iters", "precond", "diss", "hyper", "dtmax")
_PRECONDS = ("scalar", "tensor")


def _check_supported(params):
    if params.eqtype != "SM22":
        raise ValueError(f"physics.sm22 was called with eqtype={params.eqtype!r}")
    if params.spatial_dimensions == 3 and not params.z_spectral:
        raise NotImplementedError("SM22 in 3D requires z_spectral=True (the solve is fully "
                                  "spectral; there is no finite-difference-z path)")
    if params.size != 1:
        raise NotImplementedError(f"SM22 is single-process only, but this process is one of "
                                  f"{params.size}")
    if params.forcing:
        raise NotImplementedError("SM22 has no forcing: use forcing=False")


def _eqpars(params):
    ep = params.eqpars
    bad = sorted(set(ep) - set(_EQPARS_KEYS))
    if bad:
        raise ValueError(f"SM22 eqpars: unknown keys {bad}; allowed {list(_EQPARS_KEYS)}")
    f64 = _precision.ftype == jnp.float64
    precond = ep.get("precond", "scalar")
    if precond not in _PRECONDS:
        raise ValueError(f"SM22 eqpars['precond'] must be one of {_PRECONDS}, got {precond!r}")
    hyper = ep.get("hyper", 8)
    if int(hyper) != hyper or hyper < 1:
        raise ValueError(f"SM22 eqpars['hyper'] must be an int >= 1, got {hyper!r}")
    diss = float(ep.get("diss", 0.0))
    if diss < 0:
        raise ValueError(f"SM22 eqpars['diss'] must be >= 0, got {diss!r}")
    return dict(gamma=float(ep.get("gamma", 0.0)),
                maxiter=int(ep.get("maxiter", 100)),
                rtol=float(ep.get("rtol", 1e-9 if f64 else 1e-4)),
                fixed_iters=bool(ep.get("fixed_iters", False)),
                precond=precond, diss=diss, hyper=int(hyper),
                dtmax=float(ep.get("dtmax", 0.05)))


# ------------------------------------------------------------------ k-space helpers

def _kvec(kgrid, params):
    # (kx, ky, kz) broadcastable onto (nz, nkx, nky); kz is None in dims=2. kz is
    # Nyquist-zeroed exactly as cmhd._kz_deriv (reality of i*kz on the rfftn mirror rows).
    kz = None
    if params.spatial_dimensions == 3:
        kz = kgrid.kz
        if params.nz % 2 == 0:
            kz = kz.at[params.nz // 2].set(0.0)
    return kgrid.kx, kgrid.ky, kz


def _band(kgrid, params):
    # the retained band minus k = 0: where phi lives and where the constraint is imposed
    mask = jnp.broadcast_to(kgrid.dealias, kgrid.dealias.shape)
    if params.spatial_dimensions == 2:
        mask = mask[None]
    return mask.at[(0,) * mask.ndim].set(False).astype(_precision.ftype)


def _ksq(kgrid, params):
    kx, ky, kz = _kvec(kgrid, params)
    k2 = kgrid.ksq
    return k2 if kz is None else k2 + kz * kz


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _curl_k(ek, kx, ky, kz):
    # i k x E^, k-local; kz None = d/dz 0
    if kz is None:
        return jnp.stack([1j*ky*ek[2], -1j*kx*ek[2], 1j*(kx*ek[1] - ky*ek[0])])
    return jnp.stack([1j*(ky*ek[2] - kz*ek[1]), 1j*(kz*ek[0] - kx*ek[2]),
                      1j*(kx*ek[1] - ky*ek[0])])


def _grad_real(fk, kx, ky, kz, params):
    # real-space gradient of a k-space scalar; the d/dz slot is the scalar 0.0 in dims=2
    gx = grids.ifft(1j*kx*fk, params)
    gy = grids.ifft(1j*ky*fk, params)
    gz = 0.0 if kz is None else grids.ifft(1j*kz*fk, params)
    return (gx, gy, gz)


def _induction_k(u, B, kx, ky, kz, params):
    # curl(u x B)^, unmasked: 3 forward transforms (the curl is k-local)
    ek = jnp.stack([grids.fft(c, params) for c in _cross(u, B)])
    return _curl_k(ek, kx, ky, kz)


# ------------------------------------------------------------------ Krylov

def _kdot(a, b, yfac):
    # the real inner product of two real fields from their rfft coefficients (up to a
    # constant): Re sum yfac conj(a) b
    return jnp.sum(yfac * jnp.real(jnp.conj(a) * b))


def _bicgstab(A, b, x0, M, K, rtol, fixed, yfac):
    # right-preconditioned BiCGSTAB (as jax_constantB/sm22_v2.bicgstab), iterates FROZEN once
    # ||r|| <= rtol ||b|| (masked updates with safe denominators). fixed=True: exactly K
    # iterations in a fori_loop -- reverse-differentiable, the adjoint/jacrev path.
    # fixed=False: while_loop with early exit (forward runs; forward-mode differentiable).
    f64 = _precision.ftype == jnp.float64
    tiny = 1e-250 if f64 else 1e-30
    dot = lambda a_, b_: _kdot(a_, b_, yfac)
    r = b - A(x0)
    rh = r
    z = jnp.zeros_like(b)
    one = jnp.ones((), _precision.ftype)
    bn2 = dot(b, b) * rtol**2

    def sd(a_, d_, go):
        ok = go & (jnp.abs(d_) > tiny)
        return jnp.where(ok, a_ / jnp.where(ok, d_, 1.0), 0.0)

    def body(c):
        x, r, p, v, rho, al, om, i = c
        go = dot(r, r) > bn2
        rhon = dot(rh, r)
        be = sd(rhon, rho, go) * sd(al, om, go)
        pn = r + be * (p - om * v)
        y = M(pn); vn = A(y)
        aln = sd(rhon, dot(rh, vn), go)
        s = r - aln * vn
        zz = M(s); t = A(zz)
        omn = sd(dot(t, s), dot(t, t), go)
        xn = x + aln * y + omn * zz
        rn = s - omn * t
        sel = lambda new, old: jnp.where(go, new, old)
        return (sel(xn, x), sel(rn, r), sel(pn, p), sel(vn, v), sel(rhon, rho),
                sel(aln, al), sel(omn, om), i + go.astype(jnp.int32))

    c0 = (x0, r, z, z, one, one, one, jnp.zeros((), jnp.int32))
    if fixed:
        c = jax.lax.fori_loop(0, K, lambda _, c: body(c), c0)
    else:
        c = jax.lax.while_loop(lambda c: (c[7] < K) & (dot(c[1], c[1]) > bn2), body, c0)
    x, r, its = c[0], c[1], c[7]
    res = jnp.sqrt(dot(r, r) / jnp.maximum(dot(b, b), tiny))
    return x, its, res


def _precond(B, kgrid, params, band, method):
    # k-local inverse of the constant-coefficient symbol of phi -> B.curl(grad phi x B):
    # |B|^2 k^2 - (B.k)^2. "tensor" uses <|B|^2> and <B_i B_j> (exact for uniform B);
    # "scalar" is sm22_v2's m k^2, m = <|B|^2> - sum_active <B_i^2>/d, its angle average
    # over the d grid directions that vary. Both are stop_gradient'ed (a preconditioner
    # only changes the path, not the solution).
    B = jax.lax.stop_gradient(B)
    kx, ky, kz = _kvec(kgrid, params)
    k2 = _ksq(kgrid, params)
    mB2 = jnp.mean(B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
    act = 2 if kz is None else 3
    if method == "scalar":
        m = mB2 - sum(jnp.mean(B[i]*B[i]) for i in range(act)) / act
        den = m * k2
    else:
        ks = (kx, ky) if kz is None else (kx, ky, kz)
        kBk = sum(ks[i]*ks[j]*jnp.mean(B[i]*B[j]) for i in range(act) for j in range(act))
        den = mB2 * k2 - kBk
        den = jnp.maximum(den, 1e-3 * mB2 * k2)
    return band * jnp.where(k2 > 0, 1.0 / jnp.where(k2 > 0, den, 1.0), 0.0)


def solve_phi(B, dBsq_rhs_k, phi0, kgrid, params):
    # phi^ on the band from P_band[B . P_dealias curl(grad phi x B)] = rhs^ (k-space, band)
    ep = _eqpars(params)
    kx, ky, kz = _kvec(kgrid, params)
    band = _band(kgrid, params)
    dealias = kgrid.dealias

    def A(pk):
        u = _grad_real(pk, kx, ky, kz, params)
        ck = _induction_k(u, B, kx, ky, kz, params) * dealias
        c = [grids.ifft(ck[i], params) for i in range(3)]
        return band * grids.fft(B[0]*c[0] + B[1]*c[1] + B[2]*c[2], params)

    P = _precond(B, kgrid, params, band, ep["precond"])
    M = lambda r: P * r
    return _bicgstab(A, dBsq_rhs_k, phi0, M, ep["maxiter"], ep["rtol"], ep["fixed_iters"],
                     kgrid.yfac)


# ------------------------------------------------------------------ recipe functions

def linear_matrix(kgrid, params):
    # L = +1 (the growth of dB = B - <B>) on every k != 0, 0 at k = 0 (<B> fixed bitwise),
    # minus the optional hyperdissipation diss*(k^2/k_c^2)^hyper, k_c the band edge
    # (2 pi (n/3)/L along the coarsest direction), so diss is a rate at the band edge and
    # the operator is resolution-scaled like a filter. Shape (1, nz, nkx, nky).
    _check_supported(params)
    ep = _eqpars(params)
    k2 = _ksq(kgrid, params)
    if params.spatial_dimensions == 2:
        k2 = jnp.broadcast_to(k2, k2.shape)[None]
    grow = jnp.where(k2 > 0, 1.0, 0.0)
    L = grow
    if ep["diss"] > 0:
        edges = [2*jnp.pi*(params.nx/3)/params.Lx, 2*jnp.pi*(params.ny/3)/params.Ly]
        if params.spatial_dimensions == 3:
            edges.append(2*jnp.pi*(params.nz/3)/params.Lz)
        kc2 = min(edges)**2
        L = L - ep["diss"] * (k2 / kc2)**ep["hyper"]
    return L[None].astype(_precision.ftype)


def aux_init(state, kgrid, params):
    # cold start: phi = 0 on the band
    return jnp.zeros(state.fields.shape[1:], dtype=_precision.ctype)


def grad(state, kgrid, params, aux=None):
    # B (3 inverse transforms), the Galerkin phi solve, and u = grad phi (d inverse)
    _check_supported(params)
    ep = _eqpars(params)
    kx, ky, kz = _kvec(kgrid, params)
    fk = state.fields
    B = jnp.stack([grids.ifft(fk[i], params) for i in range(3)])
    axes = tuple(range(1, B.ndim))
    dB = B - jnp.mean(B, axis=axes, keepdims=True)
    q = B[0]*dB[0] + B[1]*dB[1] + B[2]*dB[2]
    if ep["gamma"] != 0.0:
        Bsq = B[0]*B[0] + B[1]*B[1] + B[2]*B[2]
        q = q + 0.5*ep["gamma"]*(Bsq - jnp.mean(Bsq))
    rhs_k = -_band(kgrid, params) * grids.fft(q, params)
    phi0 = aux_init(state, kgrid, params) if aux is None else aux
    phi, its, res = solve_phi(B, rhs_k, phi0, kgrid, params)
    u = _grad_real(phi, kx, ky, kz, params)
    return SM22Grads(B=B, u=u, phi=phi, aux=phi, its=its, res=res)


def NonlinearTerm(state, grads, kgrid, params, halo=None):
    # P_dealias curl(grad phi x B): 3 forward transforms
    kx, ky, kz = _kvec(kgrid, params)
    return _induction_k(grads.u, grads.B, kx, ky, kz, params) * kgrid.dealias


def set_timestep(grads, params):
    # dt = min(dtmax, cfl_safety * min spacing / max|grad phi|) -- sm22_v2's rule
    ep = _eqpars(params)
    u = grads.u
    u2 = u[0]*u[0] + u[1]*u[1] + (0.0 if params.spatial_dimensions == 2 else u[2]*u[2])
    umax = comms.allreduce_max(jnp.sqrt(jnp.max(u2)), params)
    h = min(params.dx, params.dy, params.dz) if params.spatial_dimensions == 3 \
        else min(params.dx, params.dy)
    dt = jnp.minimum(ep["dtmax"], params.cfl_safety * h / jnp.maximum(umax, 1e-30))
    if not params.adaptive_timestep:
        dt = jnp.minimum(dt, params.dt)
    return dt
