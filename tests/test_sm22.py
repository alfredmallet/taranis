# SM22 constant-|B| growth rule (physics/sm22.py, docs/numerics.md "SM22"). Gates:
#
#   1. LINEAR GROWTH. At tiny amplitude an Alfvenic (dB _|_ Bbar, div-free) IC has phi =
#      O(A^2), so dB grows as e^t: gates L's +1 on k != 0 and 0 at k = 0.
#   2. THE CONSTRAINT. The assembled RHS keeps |B|^2 uniform on the retained band:
#      P_band(B . dB/dt) = 0 to the Krylov tolerance -- the Galerkin statement the solve
#      imposes, checked on the production rhs (L f + N), not on the solver's own residual.
#   3. INVARIANTS over a nonlinear run: <B> (the k = 0 mode) bitwise, div B at round-off,
#      in-band |B|^2 error flat (gamma = 0), fields stay inside the 2/3 band.
#   4. 2.5D EMBEDDING. dims=3 z_spectral with a z-independent IC reproduces dims=2 and stays
#      exactly z-independent.
#   5. SOLVER AUX. The warm start changes the iteration count, not the answer (to the
#      tolerance); rk44 and lsrk54 both thread it and return (state, aux); IMEX rejects it.
#   6. REVERSE MODE through fixed_iters=True (fp64): jax.grad of the amplitude after a few
#      fixed-dt steps against central differences.
#
# Single process (SM22 is size==1). pytest, or `python tests/test_sm22.py`.
from _rmhd_testing import bootstrap

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, grids
from taranis import diagnostics as D
from taranis.physics import equation_registry, construct_rhs, sm22
from taranis.run import init_aux, block_of_steps
from taranis.timestepping import get_scheme

FP64 = _precision.ftype == jnp.float64
REC = equation_registry["SM22"]
RHS = construct_rhs(REC)
TH = np.deg2rad(30.0)


def _params(N=32, dims=2, nz=1, **kw):
    eqpars = kw.pop("eqpars", {})
    extra = dict(nz=nz, Lz=1.0, z_spectral=True) if dims == 3 else {}
    return jr.Parameters(N, N, 1.0, 1.0, 0.5, eqtype="SM22", dims=dims, eqpars=eqpars,
                         **extra, **kw)


def _seed2d(N, A0, kcut=1, seed=2):
    # sm22_grow2d.seed_state (jax_constantB): dB = curl(A_x xhat) in the (l, z, ign) frame,
    # mean field at 30 deg -- Alfvenic (dB _|_ Bbar) and div-free
    rng = np.random.default_rng(seed)
    x = np.arange(N) / N
    L, Z = np.meshgrid(x, x, indexing="ij")
    Ax = np.zeros((N, N))
    for ml in range(-kcut, kcut + 1):
        for mz in range(-kcut, kcut + 1):
            if ml == 0 and mz == 0:
                continue
            a, p = rng.normal(), rng.uniform(0, 2*np.pi)
            Ax += a * np.cos(2*np.pi*(ml*L + mz*Z) + p)
    k = 2*np.pi*np.fft.fftfreq(N, 1.0/N)
    KL, KZ = np.meshgrid(k, k, indexing="ij")
    Ah = np.fft.fft2(Ax)
    Axl = np.real(np.fft.ifft2(1j*KL*Ah)); Axz = np.real(np.fft.ifft2(1j*KZ*Ah))
    s, c = np.sin(TH), np.cos(TH)
    dB = np.stack([-s*Axz, s*Axl, -c*Axz])
    dB *= A0 / np.sqrt(np.mean(np.sum(dB**2, axis=0)))
    return dB + np.array([c, 0.0, -s])[:, None, None]


def _init2d(p, A0=0.05):
    B = _seed2d(p.nx, A0)
    return jr.initialize(lambda x, y: jnp.asarray(B, dtype=_precision.ftype)[:, None], p)


def _stepper(p, kg, name="lsrk54", dt=None):
    stepper, scheme = get_scheme(name)
    return jax.jit(lambda s, a: stepper(s, kg, p, RHS, REC.set_timestep_func, scheme, dt,
                                        aux=a))


def _inband_rms(st, kg, p):
    B = grids.ifft(st.fields, p)
    qk = sm22._band(kg, p) * grids.fft(jnp.sum(B*B, axis=0), p)
    return float(jnp.sqrt(jnp.mean(grids.ifft(qk, p)**2)))


def test_linear_growth_is_exp_t():
    p = _params(N=16, adaptive_timestep=False, dt=0.05)
    kg = jr.setup_kgrids(p)
    st = _init2d(p, A0=1e-5)
    A0 = float(D.sm22.amplitude(st, p))
    step = _stepper(p, kg, dt=0.05)
    aux = init_aux(st, kg, p)
    for _ in range(20):
        st, aux = step(st, aux)
    # phi = O(A^2) -> relative correction O(A) = 1e-5 over t = 1
    ratio = float(D.sm22.amplitude(st, p)) / A0
    assert abs(ratio / np.exp(float(st.t)) - 1) < (1e-4 if FP64 else 1e-3), ratio


def test_constraint_on_the_band():
    p = _params(N=32)
    kg = jr.setup_kgrids(p)
    st = _init2d(p, A0=0.3)
    aux = init_aux(st, kg, p)
    nl, grads = RHS(st, kg, p, aux)
    dBdt = kg.lin.apply_L(st.fields) + nl
    B = grads.B
    q = sum(B[i] * grids.ifft(dBdt[i], p) for i in range(3))
    qk = sm22._band(kg, p) * grids.fft(q, p)
    # compare with the size of the uncancelled source B . dB
    Bm = B - jnp.mean(B, axis=(1, 2, 3), keepdims=True)
    src = sm22._band(kg, p) * grids.fft(sum(B[i]*Bm[i] for i in range(3)), p)
    rel = float(jnp.sqrt(jnp.sum(jnp.abs(qk)**2) / jnp.sum(jnp.abs(src)**2)))
    assert float(grads.res) < (1e-8 if FP64 else 2e-4)
    assert rel < (1e-8 if FP64 else 2e-4), rel


def test_invariants_over_a_nonlinear_run():
    p = _params(N=32)
    kg = jr.setup_kgrids(p)
    st = _init2d(p, A0=0.2)
    k0 = np.asarray(st.fields[:, 0, 0, 0])
    d0 = {k: float(v) for k, v in D.sm22.diagnostics(st, kg, p).items()}
    inband0 = _inband_rms(st, kg, p)
    step = _stepper(p, kg)
    aux = init_aux(st, kg, p)
    for _ in range(15):
        st, aux = step(st, aux)
    d = {k: float(v) for k, v in D.sm22.diagnostics(st, kg, p).items()}
    assert d["A"] > 1.5 * d0["A"]
    np.testing.assert_array_equal(np.asarray(st.fields[:, 0, 0, 0]), k0)
    assert d["divB"] < (1e-11 if FP64 else 1e-4)
    # in-band |B|^2 fluctuation, ABSOLUTE (Berr divides by <|B|^2>, which grows with A): the
    # constraint freezes it (gamma = 0) up to the time-integration error (3e-8 here,
    # converging with dt)
    assert abs(_inband_rms(st, kg, p) / inband0 - 1) < (1e-6 if FP64 else 1e-3)
    outside = np.asarray(jnp.abs(st.fields) * (1 - kg.dealias))
    assert outside.max() == 0.0


def test_2p5d_embedding():
    N = 16
    p2 = _params(N=N)
    p3 = _params(N=N, dims=3, nz=4)
    B = _seed2d(N, 0.2)
    s2 = jr.initialize(lambda x, y: jnp.asarray(B, dtype=_precision.ftype)[:, None], p2)
    s3 = jr.initialize(lambda x, y, z: jnp.broadcast_to(
        jnp.asarray(B, dtype=_precision.ftype)[:, None], (3, z.shape[0], N, N)), p3)
    kg2, kg3 = jr.setup_kgrids(p2), jr.setup_kgrids(p3)
    st2, st3 = _stepper(p2, kg2), _stepper(p3, kg3)
    a2, a3 = init_aux(s2, kg2, p2), init_aux(s3, kg3, p3)
    for _ in range(10):
        s2, a2 = st2(s2, a2)
        s3, a3 = st3(s3, a3)
    B2 = np.asarray(grids.ifft(s2.fields, p2))
    B3 = np.asarray(grids.ifft(s3.fields, p3))
    tol = 1e-12 if FP64 else 1e-4
    assert np.abs(B3 - B2).max() < tol
    assert np.abs(B3 - B3[:, :1]).max() < tol


def test_solver_aux_warm_start():
    p = _params(N=32)
    kg = jr.setup_kgrids(p)
    st = _init2d(p, A0=0.3)
    aux = init_aux(st, kg, p)
    for name in ("rk44", "lsrk54"):
        out = _stepper(p, kg, name)(st, aux)
        assert isinstance(out, tuple) and len(out) == 2
    s1, a1 = _stepper(p, kg)(st, aux)
    cold = REC.grad_func(s1, kg, p)
    warm = REC.grad_func(s1, kg, p, a1)
    assert int(warm.its) < int(cold.its)
    rel = float(jnp.max(jnp.abs(warm.phi - cold.phi)) / jnp.max(jnp.abs(cold.phi)))
    assert rel < (1e-7 if FP64 else 1e-2), rel
    stepper, scheme = get_scheme("imexcb3e")
    with pytest.raises(NotImplementedError):
        stepper(st, kg, p, RHS, REC.set_timestep_func, scheme, aux=aux)
    # block_of_steps: (state, aux) in -> (state, aux) out
    s, a = block_of_steps((st, aux), kg, p, 2, *reversed(get_scheme("lsrk54")))
    assert s.fields.shape == st.fields.shape and a.shape == aux.shape
    # the stability harness's overrides seam is rejected on the (state, aux) path too
    with pytest.raises(ValueError, match="overrides"):
        block_of_steps((st, aux), kg._replace(overrides={"Lx": 1.0}), p, 1,
                       *reversed(get_scheme("lsrk54")))


@pytest.mark.fp64
def test_reverse_mode_fixed_iters():
    if not FP64:
        pytest.skip("fp64 only: FD at fp32 is at the noise floor")
    p = _params(N=16, adaptive_timestep=False, dt=0.05,
                eqpars=dict(fixed_iters=True, maxiter=40, rtol=1e-12))
    kg = jr.setup_kgrids(p)
    B = jnp.asarray(_seed2d(16, 0.3))
    dB = jnp.asarray(_seed2d(16, 0.3, seed=7)) - jnp.mean(jnp.asarray(_seed2d(16, 0.3, seed=7)),
                                                          axis=(1, 2), keepdims=True)
    stepper, scheme = get_scheme("lsrk54")

    def J(eps):
        f = grids.fft((B + eps*dB)[:, None], p) * kg.dealias
        st = jr.initialize(lambda x, y: B[:, None], p)._replace(fields=f.astype(_precision.ctype))
        aux = init_aux(st, kg, p)
        for _ in range(4):
            st, aux = stepper(st, kg, p, RHS, REC.set_timestep_func, scheme, 0.05, aux=aux)
        return D.sm22.amplitude(st, p)

    g = float(jax.grad(J)(0.0))
    h = 1e-4
    fd = (float(J(h)) - float(J(-h))) / (2*h)
    assert abs(g - fd) < 1e-6 * max(abs(fd), 1e-3), (g, fd)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
