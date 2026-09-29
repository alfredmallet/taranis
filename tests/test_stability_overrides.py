# taranis.stability rung 2 (plans/AUTODIFF_PLAN.md "Rung 2"): the overrides seam
# (taranis/overrides.py) -- continuous physics parameters and the box lengths evaluated at a
# point other than the Parameters' own, and dJ/dp through the kgrid build (dparams=).
#   1. overrides=None is today's graph: the refactor reference and gate 6 are the standing
#      proof (not re-run here). Here: J at overrides = the static values (every live key, and
#      the empty dict) equals J at overrides=None, per configuration.
#   2. Independent route: J at overrides = a perturbed point (every live key moved at once)
#      equals J of a Parameters BUILT with those values through the static path. This is what
#      catches a read site that ignores its override (the FD gate cannot: its reference runs
#      through the same read sites).
#   3. FD gate: dj_matrix(dparams={key: p}) (d/dln p) against the central difference of J at
#      overrides p(1 +- h), O(h^2) (or round-off where J is affine in p), for EVERY live key in
#      RMHD 2D (ky block and real coordinates), RMHD 3D FD-z, RMHD z_spectral separable and
#      putzer2, GDI 2D and 3D, CMHD rho (gamma=1) and lnrho (gamma=5/3); box lengths move k
#      in L and N. Plus a direction whose shape differs from the point's (scalar diss against
#      a (nu, eta) / (D_rho, nu, eta) direction: the point is broadcast).
#   4. Exact: at x0 = 0 J = L, so dJ/d(diss) = -k^(2 hyper) and dJ/dz_diss_k = -kz^4
#      entrywise; FD-z dJ/dz_diss = -S4/16 and dJ/dLz = -(the cross-field d/dz part of J)/Lz.
#      dlambda/dp by eigenvalue_sensitivity(dparams) against the d/dp of CLOSED-FORM
#      dispersion relations: z_spectral Alfven +-i kz (both backends; dlambda/dLz = -+i kz/Lz
#      + 4 z_diss_k kz^4/Lz), GDI eq (3.7) (2D and 3D), CMHD fast/slow/Alfven about a uniform
#      state (d/dcs0 and the box lengths, both density variables).
#   5. Tearing: dgamma/deta and dgamma/dk (dparams={"Ly": -Ly/k}) against eigenvalue FD at
#      O(h^2); propagator_eigs(overrides=) against eig_dense at the overridden point.
#   6. Rejections; Parameters.save(overrides=) / load_overrides / from_snapshot.
#   7. (review) Every other entry point honours overrides= (jvp/transpose/djvp operators,
#      ky_block/real/dj operators, diagonal and ky-averaged preconditioners,
#      propagator_operator); the (1,) diss shape and CMHD lnrho at gamma = 1 by the
#      independent route; a partly-NaN pair; from_snapshot does not leak the stamp.
# Tolerances split by precision where the test runs at both.
# pytest: `pytest tests/test_stability_overrides.py`. Script:
# `python tests/test_stability_overrides.py`.
from _rmhd_testing import bootstrap, checks, fit_order, fresh_params, make_state

bootstrap()

import dataclasses
import functools
import json
import os
import tempfile
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import taranis as jr
from taranis import _precision, overrides, run, stability

FP64 = _precision.precision == "64"
TWO_PI = 2*np.pi


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))/np.linalg.norm(np.asarray(b)))


def _pair(a, b):
    # index into b paired with each entry of a (optimal assignment)
    cost = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    r, c = linear_sum_assignment(cost)
    return c[np.argsort(r)]


# ------------------------------------------------------------------------ configurations

_BASE = dict(cfl_safety=0.5, dt=0.01, adaptive_timestep=False, comm_backend="serial")
_CTOR_KEYS = ("Lx", "Ly", "Lz", "z_diss")

_CFG = {
    "rmhd2d": dict(nx=16, ny=8, Lx=TWO_PI, Ly=4*np.pi, dims=2, eqtype="RMHD",
                   eqpars=dict(diss=(0.02, 0.01), hyper=1)),
    "rmhd2d_real": dict(nx=8, ny=8, Lx=TWO_PI, Ly=1.3*TWO_PI, dims=2, eqtype="RMHD",
                        eqpars=dict(diss=(0.02, 0.01), hyper=1)),
    "rmhd_fdz": dict(nx=8, ny=8, nz=8, Lx=TWO_PI, Ly=TWO_PI, Lz=3*np.pi, dims=3, z_diss=0.4,
                     eqtype="RMHD", eqpars=dict(diss=(0.02, 0.01), hyper=1)),
    "rmhd_zsep": dict(nx=8, ny=8, nz=8, Lx=TWO_PI, Ly=TWO_PI, Lz=3*np.pi, dims=3,
                      z_spectral=True, eqtype="RMHD",
                      eqpars=dict(diss=0.02, hyper=1, z_diss_k=1e-3)),
    "rmhd_zput": dict(nx=8, ny=8, nz=8, Lx=TWO_PI, Ly=TWO_PI, Lz=3*np.pi, dims=3,
                      z_spectral=True, eqtype="RMHD",
                      eqpars=dict(diss=(0.02, 0.01), hyper=1, z_diss_k=1e-3)),
    "gdi2d": dict(nx=12, ny=8, Lx=TWO_PI, Ly=TWO_PI, dims=2, eqtype="GDI",
                  eqpars=dict(Ln=1.0, nu_in=0.3, v0=2.0, diss=0.01, hyper=1, gpar_fac=0.1)),
    "gdi3d": dict(nx=8, ny=8, nz=6, Lx=TWO_PI, Ly=TWO_PI, Lz=4*np.pi, dims=3, z_spectral=True,
                  eqtype="GDI", eqpars=dict(Ln=1.0, nu_in=0.3, v0=2.0, diss=0.01, hyper=1,
                                            gpar_fac=0.05, D_par=0.5)),
    "cmhd_rho": dict(nx=6, ny=6, nz=6, Lx=TWO_PI, Ly=TWO_PI, Lz=TWO_PI, dims=3,
                     z_spectral=True, eqtype="CMHD",
                     eqpars=dict(cs0=0.8, diss=0.05, hyper=2, gamma=1.0)),
    "cmhd_lnrho": dict(nx=6, ny=6, nz=6, Lx=TWO_PI, Ly=TWO_PI, Lz=TWO_PI, dims=3,
                       z_spectral=True, eqtype="CMHD",
                       eqpars=dict(cs0=0.8, diss=0.05, hyper=2, gamma=5/3,
                                   density_var="lnrho")),
}

# the keys each configuration must expose (overridable(params) is asserted against these)
_KEYS = {
    "rmhd2d": ["Lx", "Ly", "diss"], "rmhd2d_real": ["Lx", "Ly", "diss"],
    "rmhd_fdz": ["Lx", "Ly", "Lz", "diss", "z_diss"],
    "rmhd_zsep": ["Lx", "Ly", "Lz", "diss", "z_diss_k"],
    "rmhd_zput": ["Lx", "Ly", "Lz", "diss", "z_diss_k"],
    "gdi2d": ["Ln", "Lx", "Ly", "diss", "gpar_fac", "nu_in", "v0"],
    "gdi3d": ["D_par", "Ln", "Lx", "Ly", "Lz", "diss", "gpar_fac", "nu_in", "v0"],
    "cmhd_rho": ["Lx", "Ly", "Lz", "cs0", "diss"], "cmhd_lnrho": ["Lx", "Ly", "Lz", "cs0", "diss"],
}


def _make(name, values=None):
    # Parameters of configuration `name`, with `values` {key: value} set through the ctor /
    # eqpars -- the STATIC route the overrides seam must agree with
    kw = dict(_BASE, **_CFG[name])
    eq = dict(kw.pop("eqpars"))
    for k, v in (values or {}).items():
        if k in _CTOR_KEYS:
            kw[k] = v
        else:
            eq[k] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jr.Parameters(eqpars=eq, **kw)


def _x0(name, params):
    # a ky-INDEPENDENT x0 (so a ky block is exact) with x (and z) structure, except
    # rmhd2d_real (fully 2D, real coordinates)
    eqtype, d3 = params.eqtype, params.spatial_dimensions == 3
    if name == "rmhd2d_real":
        ic = lambda x, y: jnp.stack([0.5*jnp.sin(x)*jnp.cos(y) + 0.2*jnp.cos(x + y),
                                     jnp.cos(x)*jnp.cos(y) + 0.3*jnp.sin(2*x + y)])
        return np.asarray(jr.initialize(ic, params).fields)
    if not d3:
        ic = lambda x, y: jnp.stack([0.4*jnp.sin(x) + 0*y, jnp.cos(x) + 0.3*jnp.sin(2*x) + 0*y])
        return np.asarray(jr.initialize(ic, params).fields)
    kz = TWO_PI/params.Lz
    if eqtype in ("RMHD", "GDI"):
        ic = lambda x, y, z: jnp.stack([0.4*jnp.sin(x)*jnp.cos(kz*z) + 0*y,
                                        jnp.cos(x) + 0.3*jnp.sin(kz*z) + 0*y])
        return np.asarray(jr.initialize(ic, params).fields)
    lnrho = params.eqpars.get("density_var", "rho") == "lnrho"

    def ic(x, y, z):
        rho = 1.3 + 0.2*jnp.cos(x) + 0.1*jnp.sin(kz*z) + 0*y
        f0 = jnp.log(rho) if lnrho else rho
        return jnp.stack([f0, 0.1*jnp.sin(kz*z) + 0*x + 0*y, 0.05*jnp.cos(x) + 0*y + 0*z,
                          0.07 + 0*x + 0*y + 0*z, 0.3 + 0.1*jnp.cos(kz*z) + 0*x + 0*y,
                          0.4 + 0.1*jnp.sin(x) + 0*y + 0*z, 1.0 + 0*x + 0*y + 0*z])
    return np.asarray(jr.initialize(ic, params).fields)


def _space(name):
    return {} if name == "rmhd2d_real" else {"iky": 1}


@functools.lru_cache(maxsize=None)
def _case(name):
    # (params, kgrid, x0): cached so each configuration compiles once per session; the
    # objects are shared -- never mutate them (x0 is a numpy array, never a SimulationState)
    params = _make(name)
    return params, jr.setup_kgrids(params), _x0(name, params)


def _J(params, kgrid, x0, space, ov=None):
    if "iky" in space:
        return stability.ky_block_matrix(x0, kgrid, params, space["iky"], ikx=space.get("ikx"),
                                         iz=space.get("iz"), overrides=ov)[0]
    return stability.real_matrix(x0, kgrid, params, overrides=ov)[0]


def _dJ(params, kgrid, x0, space, dparams, ov=None):
    return stability.dj_matrix(x0, None, kgrid, params, iky=space.get("iky"),
                               ikx=space.get("ikx"), iz=space.get("iz"), dparams=dparams,
                               overrides=ov)[0]


def _static_point(params):
    return {k: np.asarray(overrides.static_value(params, k), dtype=float)
            for k in overrides.overridable(params)}


def _to_ctor(v):
    v = np.asarray(v, dtype=float)
    return tuple(float(a) for a in v) if v.ndim else float(v)


# ------------------------------------------------------------------------ gates 1 and 2

def test_static_point_and_independent_route():
    # any precision. Gate 1: J at overrides = its own static values (and at {}) is J at
    # overrides=None -- bitwise at fp64 except FD-z, whose stencil dz = Lz/nz and filter
    # z_diss*(dz/2)^4 are python floats on the None path and XLA arithmetic on the
    # overridden one (integer_pow vs C pow: ulp-level). Gate 2: J at a perturbed point
    # (every key at once) equals J of a Parameters built with those values.
    tol1 = 1e-14 if FP64 else 2e-6
    tol2 = 1e-12 if FP64 else 2e-5
    with checks() as c:
        for name in _CFG:
            params, kgrid, x0 = _case(name)
            space = _space(name)
            c.check(f"{name}: overridable keys {overrides.overridable(params)}",
                    overrides.overridable(params) == _KEYS[name])
            J0 = _J(params, kgrid, x0, space)
            pt = _static_point(params)
            for label, ov in (("static point", pt), ("{}", {})):
                J1 = _J(params, kgrid, x0, space, ov)
                e = _relerr(J1, J0)
                exact = FP64 and name != "rmhd_fdz"
                ok = np.array_equal(J1, J0) if exact else e < tol1
                c.check(f"{name}: J(overrides={label}) == J(None) "
                        f"({'bitwise' if exact else f'rel {e:.1e}'})", ok, f"rel {e:.3e}")
            moved = {k: v*(1.0 + 0.05*(i + 1)) for i, (k, v) in enumerate(sorted(pt.items()))}
            p2 = _make(name, {k: _to_ctor(v) for k, v in moved.items()})
            J2 = _J(p2, jr.setup_kgrids(p2), x0, space)
            J2o = _J(params, kgrid, x0, space, moved)
            e = _relerr(J2o, J2)
            e0 = _relerr(J0, J2)
            c.check(f"{name}: J(overrides=moved point) == J(Parameters at that point) (rel "
                    f"{e:.1e}; the move itself changes J by {e0:.1e})",
                    e < tol2 and e0 > 1e3*tol2, f"rel {e:.3e}, move {e0:.3e}")
        # the other diss shapes each read site takes, through both routes (a scalar against a
        # (nu, eta) static and back; CMHD's (D_rho, nu, eta) -> 7-field expansion)
        for name, d in (("rmhd2d", 0.03), ("rmhd_zsep", (0.03, 0.017)), ("rmhd_zput", 0.025),
                        ("cmhd_rho", (0.04, 0.06, 0.03)), ("cmhd_lnrho", (0.04, 0.06, 0.03))):
            params, kgrid, x0 = _case(name)
            space = _space(name)
            p2 = _make(name, {"diss": d})
            J2 = _J(p2, jr.setup_kgrids(p2), x0, space)
            e = _relerr(_J(params, kgrid, x0, space, {"diss": d}), J2)
            c.check(f"{name}: J(overrides diss={d}) == J(Parameters diss={d}) (rel {e:.1e})",
                    e < tol2, f"rel {e:.3e}")


def test_rmhd_zspectral_backend_follows_override_shape():
    # the RMHD z_spectral backend is picked from the diss override's SHAPE: a scalar is
    # nu == eta (separable), a pair is the general putzer2 -- the same operator either way
    with checks() as c:
        params, kgrid, x0 = _case("rmhd_zsep")
        from taranis import grids, propagators
        c.check("static diss scalar -> separable", isinstance(kgrid.lin, propagators.SeparableL))
        kg_s = grids.setup_kgrids(params, overrides={"diss": 0.03})
        kg_p = grids.setup_kgrids(params, overrides={"diss": (0.03, 0.03)})
        kg_l = grids.setup_kgrids(params, overrides={"Ly": 5.0})
        c.check("scalar diss override -> separable", isinstance(kg_s.lin, propagators.SeparableL))
        c.check("pair diss override -> putzer2", isinstance(kg_p.lin, propagators.Putzer2Operator))
        c.check("no diss override -> the static choice (separable)",
                isinstance(kg_l.lin, propagators.SeparableL))
        p2, kg2, _ = _case("rmhd_zput")
        kg2s = grids.setup_kgrids(p2, overrides={"diss": 0.01})
        c.check("zput with a scalar diss override -> separable",
                isinstance(kg2s.lin, propagators.SeparableL))
        Js = _J(params, kgrid, x0, {"iky": 1}, {"diss": 0.03})
        Jp = _J(params, kgrid, x0, {"iky": 1}, {"diss": (0.03, 0.03)})
        e = _relerr(Jp, Js)
        c.check(f"separable and putzer2 at nu = eta give the same J (rel {e:.1e})",
                e < (1e-13 if FP64 else 1e-5), f"{e:.3e}")


def test_override_dtypes():
    # any precision (discriminating at fp32): overrides are cast to the FIELD real dtype, so a
    # strong float64 value never upcasts an fp32 graph -- every kgrid array and N stay at
    # the session's field precision
    from taranis import grids
    with checks() as c:
        for name, ov in (("rmhd_fdz", {"Lx": np.float64(6.0), "Lz": np.float64(9.0),
                                       "diss": np.array([0.01, 0.02]), "z_diss": 0.3}),
                         ("rmhd_zsep", {"Ly": np.float64(5.0), "z_diss_k": np.float64(2e-3)}),
                         ("gdi3d", {"Ln": np.float64(1.5), "D_par": np.float64(0.4),
                                    "Lz": np.float64(11.0)}),
                         ("cmhd_lnrho", {"cs0": np.float64(0.7), "Lx": np.float64(6.5),
                                         "diss": np.array([0.01, 0.02, 0.03])})):
            params, _, x0 = _case(name)
            kg = grids.setup_kgrids(params, overrides=ov)
            ok_dt = {np.dtype(_precision.ftype), np.dtype(_precision.ctype), np.dtype(bool)}
            bad = [str(a.dtype) for a in jax.tree.leaves(kg)
                   if hasattr(a, "dtype") and np.dtype(a.dtype) not in ok_dt
                   and not jnp.issubdtype(a.dtype, jnp.integer)]
            c.check(f"{name}: every overridden kgrid array at field precision ({bad or 'ok'})",
                    not bad)
            n = stability._nonlinear(params)(jnp.asarray(x0, dtype=_precision.ctype), kg)
            c.check(f"{name}: N at the overridden kgrid is {n.dtype}",
                    n.dtype == _precision.ctype)


# ------------------------------------------------------------------------------ gate 3

def _fd_errors(params, kgrid, x0, space, point, key, direction, hs):
    # rel error of dJ/ds (s: point + s*direction along key) against the central difference
    dJ = _dJ(params, kgrid, x0, space, {key: direction}, point)
    errs = []
    for h in hs:
        pp, pm = dict(point), dict(point)
        pp[key] = np.asarray(point[key]) + h*np.asarray(direction)
        pm[key] = np.asarray(point[key]) - h*np.asarray(direction)
        fd = (_J(params, kgrid, x0, space, pp) - _J(params, kgrid, x0, space, pm))/(2*h)
        errs.append(_relerr(dJ, fd))
    return errs, float(np.linalg.norm(dJ))


def _fd_ok(errs, hs):
    # fp64: round-off (J affine in the key) or O(h^2); fp32: one h, a loose bound
    if not FP64:
        return errs[0] < 2e-3, f"err {errs[0]:.1e}"
    if errs[-1] < 1e-9:
        return True, f"errs {errs[0]:.1e}, {errs[1]:.1e} (affine: round-off)"
    order = fit_order(hs, errs)
    return (1.7 < order < 2.3 and errs[-1] < 1e-4,
            f"errs {errs[0]:.1e}, {errs[1]:.1e}, order {order:.2f}")


def test_fd_gate_every_key():
    hs = (2e-3, 1e-3) if FP64 else (1e-2,)
    with checks() as c:
        for name in _CFG:
            params, kgrid, x0 = _case(name)
            space = _space(name)
            pt = _static_point(params)
            # the point carries EVERY live key (one compile per configuration); direction
            # d/dln p along each key in turn
            for key in sorted(pt):
                errs, nrm = _fd_errors(params, kgrid, x0, space, pt, key, pt[key], hs)
                ok, msg = _fd_ok(errs, hs)
                c.check(f"{name}: dJ/dln {key} == central FD ({msg}; |dJ| {nrm:.2e})",
                        ok and nrm > 0, msg)
        # a direction whose shape differs from the point's: the point is broadcast
        for name, key, d in (("rmhd_zsep", "diss", (0.02, 0.0)), ("rmhd2d", "diss", 0.015),
                             ("cmhd_rho", "diss", (0.0, 0.05, 0.0))):
            params, kgrid, x0 = _case(name)
            pt = {key: np.asarray(overrides.static_value(params, key), dtype=float)}
            errs, nrm = _fd_errors(params, kgrid, x0, _space(name), pt, key, np.asarray(d), hs)
            ok, msg = _fd_ok(errs, hs)
            c.check(f"{name}: point {pt[key].tolist()} against direction {d} ({msg})",
                    ok and nrm > 0, msg)


def test_state_and_parameter_directions():
    # a combined direction (dx0, dparams) is the sum of its parts (dJ is linear in the
    # direction), and the pure-state dJ at an overridden point is the state FD of J there.
    # Keys that enter N (RMHD/GDI N is quadratic: a key entering only L, like Ln or the
    # FD-z Lz, leaves the state derivative unchanged -- correctly)
    tol = 1e-12 if FP64 else 1e-5
    hs = (2e-3, 1e-3) if FP64 else (1e-2,)
    with checks() as c:
        for name, key in (("rmhd_fdz", "Lx"), ("cmhd_lnrho", "cs0"), ("gdi2d", "Ly")):
            params, kgrid, x0 = _case(name)
            sp = _space(name)
            dx0 = np.roll(x0, 1, axis=2)*0.5          # ky-independent like x0
            ov = {key: float(overrides.static_value(params, key))*1.2}
            dp = {key: 1.0}
            both = stability.dj_matrix(x0, dx0, kgrid, params, dparams=dp, overrides=ov, **sp)[0]
            st = stability.dj_matrix(x0, dx0, kgrid, params, overrides=ov, **sp)[0]
            pa = stability.dj_matrix(x0, None, kgrid, params, dparams=dp, overrides=ov, **sp)[0]
            e = _relerr(both, st + pa)
            c.check(f"{name}: dJ along (dx0, d{key}) == along dx0 + along d{key} (rel {e:.1e}; "
                    f"parts {np.linalg.norm(st):.2e}, {np.linalg.norm(pa):.2e})",
                    e < tol and np.linalg.norm(st) > 0 and np.linalg.norm(pa) > 0, f"{e:.3e}")
            errs = []
            for h in hs:
                fd = (_J(params, kgrid, x0 + h*dx0, sp, ov)
                      - _J(params, kgrid, x0 - h*dx0, sp, ov))/(2*h)
                errs.append(_relerr(st, fd))
            ok, msg = _fd_ok(errs, hs)
            # and it IS at the overridden point: it differs from the static point's
            st0 = stability.dj_matrix(x0, dx0, kgrid, params, **sp)[0]
            moved = _relerr(st0, st)
            c.check(f"{name}: state dJ at overrides {ov} == state FD there ({msg}; differs from "
                    f"the static point's by {moved:.1e})", ok and moved > 1e-3, msg)


# ------------------------------------------------------------------------------ gate 4

def _block_k(params, idx, iky, Ly=None):
    # (kx, ky, kz) of every row (field, iz, ikx) of a ky block (kz: 0 unless z_spectral)
    n = lambda i, m: np.where(i > m//2, i - m, i)
    kx = TWO_PI*n(idx[:, 2], params.nx)/params.Lx
    ky = TWO_PI*iky/(params.Ly if Ly is None else Ly)*np.ones(len(idx))
    kz = (TWO_PI*n(idx[:, 1], params.nz)/params.Lz if params.z_spectral
          else np.zeros(len(idx)))
    return kx, ky, kz


def test_exact_at_x0_zero():
    # J = L at x0 = 0: the parameter derivatives of L, entrywise
    tol = 1e-12 if FP64 else 1e-5
    with checks() as c:
        for name, dirs in (("rmhd2d", ((1.0, 0.0), (0.0, 1.0))),
                           ("rmhd_fdz", ((1.0, 0.0),)),
                           ("rmhd_zput", ((1.0, 0.0), (0.0, 1.0))),
                           ("rmhd_zsep", (1.0,))):
            params, kgrid, _ = _case(name)
            x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
            idx = stability.ky_block_index(kgrid, params, 1)
            kx, ky, kz = _block_k(params, idx, 1)
            ksq = kx*kx + ky*ky
            for d in dirs:
                dJ = _dJ(params, kgrid, x0, {"iky": 1}, {"diss": d})
                w = np.broadcast_to(np.asarray(d, dtype=float), (2,))[idx[:, 0]]
                want = np.diag(-w*ksq)
                e = np.abs(dJ - want).max()/np.abs(want).max()
                c.check(f"{name}: dJ/d(diss) along {d} == diag(-k_perp^2) (rel {e:.1e})",
                        e < tol, f"{e:.3e}")
            if params.z_spectral:
                dJ = _dJ(params, kgrid, x0, {"iky": 1}, {"z_diss_k": 1.0})
                want = np.diag(-kz**4)
                e = np.abs(dJ - want).max()/np.abs(want).max()
                c.check(f"{name}: dJ/dz_diss_k == diag(-kz^4) (rel {e:.1e})", e < tol, f"{e:.3e}")
        # FD-z: the filter -z_diss (dz/2)^4 d4/dz4 is Lz-free (= -z_diss S4/16), the d/dz
        # (cross-field) part scales as 1/dz = nz/Lz
        params, kgrid, _ = _case("rmhd_fdz")
        x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
        idx = stability.ky_block_index(kgrid, params, 1)
        J0 = _J(params, kgrid, x0, {"iky": 1})
        f, iz, ix = idx[:, 0], idx[:, 1], idx[:, 2]
        same_f = f[:, None] == f[None, :]
        dzi = (iz[None, :] - iz[:, None]) % params.nz
        s4 = np.select([dzi == 0, (dzi == 1) | (dzi == params.nz - 1),
                        (dzi == 2) | (dzi == params.nz - 2)], [6.0, -4.0, 1.0], 0.0)
        want = np.where(same_f & (ix[:, None] == ix[None, :]), -s4/16.0, 0.0)
        dJ = _dJ(params, kgrid, x0, {"iky": 1}, {"z_diss": 1.0})
        e = np.abs(dJ - want).max()/np.abs(want).max()
        c.check(f"FD-z: dJ/dz_diss == -S4/16 (rel {e:.1e})", e < tol, f"{e:.3e}")
        dJ = _dJ(params, kgrid, x0, {"iky": 1}, {"Lz": 1.0})
        want = np.where(same_f, 0.0, -J0/params.Lz)
        e = np.abs(dJ - want).max()/np.abs(want).max()
        c.check(f"FD-z: dJ/dLz == -(cross-field d/dz part of J)/Lz, same-field blocks 0 (rel "
                f"{e:.1e})", e < tol and np.abs(want).max() > 0, f"{e:.3e}")


def _eig_sens(c, label, J, dJs, want_fn, keys, p0, tol, res_tol):
    # eigenvalue_sensitivity on every eigenvalue of J along each key, against the d/dp of the
    # closed form want_fn(values) (5-point stencil on the closed form: an independent route)
    ev = stability.eig_dense(J)
    want0 = want_fn(p0)
    perm = _pair(want0, ev.values)
    e = np.abs(ev.values[perm] - want0).max()/np.abs(want0).max()
    c.check(f"{label}: eigenvalues == closed form (rel {e:.1e})", e < tol, f"{e:.3e}")
    for key in keys:
        hp = 1e-4*abs(p0[key])
        stencil = []
        for s in (-2, -1, 1, 2):
            p = dict(p0)
            p[key] = p0[key] + s*hp
            stencil.append(want_fn(p))
        dwant = (stencil[0] - 8*stencil[1] + 8*stencil[2] - stencil[3])/(12*hp)
        got = np.array([stability.eigenvalue_sensitivity(
            J, dJs[key], ev.values[j], ev.right[:, j], ev.left[:, j], spectrum=ev.values,
            residual_tol=res_tol).dlam for j in perm])
        e = np.abs(got - dwant).max()/max(np.abs(dwant).max(), 1e-300)
        c.check(f"{label}: dlambda/d{key} == d/d{key} of the closed form (rel {e:.1e}, "
                f"max |dlambda| {np.abs(dwant).max():.2e})",
                e < tol and np.abs(dwant).max() > 0, f"{e:.3e}")


_RHO0, _U0, _B0 = 1.3, np.array([0.1, -0.05, 0.07]), np.array([0.3, 0.4, 1.0])


def _cmhd_exact(k, cs0, gamma, D, hyper):
    # docs/numerics.md "Compressible MHD" linear waves about (rho0, u0, B0)
    k2 = k @ k
    c2 = cs0*cs0*_RHO0**(gamma - 1.0)
    vA = _B0/np.sqrt(_RHO0)
    s = c2 + vA @ vA
    kva = k @ vA
    wf2 = 0.5*(k2*s + np.sqrt((k2*s)**2 - 4*k2*c2*kva**2))
    ws2 = k2*c2*kva**2/wf2
    damp = -D*k2**hyper
    dop = -1j*(k @ _U0)
    waves = [damp + dop + sg*1j*w for w in (np.sqrt(wf2), np.sqrt(ws2), abs(kva))
             for sg in (1, -1)]
    return np.array(waves + [damp])


def _gdi_exact(kx, ky, kz, ep):
    # eq (3.7) (as tests/test_stability_general.py)
    ksq = kx*kx + ky*ky
    nu, g = ep["nu_in"], ep.get("gpar_fac", 0.0)*ep["nu_in"]*ksq + ep.get("D_par", 0.0)*kz*kz
    ws, wp = ky/ep["Ln"], ky*nu*ep["v0"]
    roots = np.roots([ksq, 1j*(nu*ksq + g*(1 + ksq)), -g*nu*ksq + wp*ws + 1j*g*(ws - wp)])
    return -1j*roots - ep["diss"]*ksq**ep["hyper"]


def _mode_k(p, params, ix, iy, iz):
    kx = TWO_PI*ix/p["Lx"]
    ky = TWO_PI*iy/p["Ly"]
    kz = TWO_PI*iz/p["Lz"] if params.spatial_dimensions == 3 else 0.0
    return kx, ky, kz


def test_exact_dispersion_sensitivities():
    tol = 1e-8 if FP64 else 1e-5
    res_tol = 1e-8 if FP64 else 1e-4
    with checks() as c:
        # z_spectral RMHD at x0 = 0, one (kx, ky, kz) mode: lambda = m +- sqrt(q^2 - kz^2)
        for name in ("rmhd_zsep", "rmhd_zput"):
            params, kgrid, _ = _case(name)
            x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
            sp = {"iky": 1, "ikx": 1, "iz": 1}
            nu_eta = np.broadcast_to(np.asarray(params.eqpars["diss"], float), (2,))
            p0 = dict(Lx=params.Lx, Ly=params.Ly, Lz=params.Lz, nu=nu_eta[0], eta=nu_eta[1],
                      zk=params.eqpars["z_diss_k"])

            def want(p, params=params):
                kx, ky, kz = _mode_k(p, params, 1, 1, 1)
                kp2 = kx*kx + ky*ky
                dphi, dpsi = -p["nu"]*kp2 - p["zk"]*kz**4, -p["eta"]*kp2 - p["zk"]*kz**4
                q = np.sqrt(complex(((dphi - dpsi)/2)**2 - kz*kz))
                return np.array([(dphi + dpsi)/2 + q, (dphi + dpsi)/2 - q])
            J = _J(params, kgrid, x0, sp)
            dJs = {k: _dJ(params, kgrid, x0, sp, {k: 1.0}) for k in ("Lx", "Ly", "Lz")}
            dJs["zk"] = _dJ(params, kgrid, x0, sp, {"z_diss_k": 1.0})
            keys = ["Lx", "Ly", "Lz", "zk"]
            if name == "rmhd_zput":
                dJs["nu"] = _dJ(params, kgrid, x0, sp, {"diss": (1.0, 0.0)})
                dJs["eta"] = _dJ(params, kgrid, x0, sp, {"diss": (0.0, 1.0)})
                keys += ["nu", "eta"]
            _eig_sens(c, f"{name} Alfven mode", J, dJs, want, keys, p0, tol, res_tol)
            if name == "rmhd_zsep":
                # the prompt's form: dlambda/dLz = -+ i kz/Lz + 4 z_diss_k kz^4/Lz
                ev = stability.eig_dense(J)
                kz = TWO_PI/params.Lz
                for j in range(2):
                    s = stability.eigenvalue_sensitivity(J, dJs["Lz"], ev.values[j],
                                                         ev.right[:, j], ev.left[:, j],
                                                         spectrum=ev.values,
                                                         residual_tol=res_tol)
                    sg = np.sign(ev.values[j].imag)
                    w = -sg*1j*kz/params.Lz + 4*p0["zk"]*kz**4/params.Lz
                    e = abs(s.dlam - w)/abs(w)
                    c.check(f"rmhd_zsep: dlambda/dLz == {'-' if sg > 0 else '+'}i kz/Lz + "
                            f"4 zk kz^4/Lz (rel {e:.1e})", e < tol, f"{e:.3e}")
        # GDI at x0 = 0, one mode
        for name, sp, ixyz in (("gdi2d", {"iky": 1, "ikx": 1}, (1, 1, 0)),
                               ("gdi3d", {"iky": 1, "ikx": 1, "iz": 1}, (1, 1, 1))):
            params, kgrid, _ = _case(name)
            x0 = np.zeros((2, params.nz, params.nx, params.ny//2 + 1), dtype=complex)
            keys = [k for k in overrides.overridable(params)
                    if k != "Lz" or params.spatial_dimensions == 3]
            p0 = {k: float(overrides.static_value(params, k)) for k in keys}

            def want(p, params=params):
                kx, ky, kz = _mode_k(p, params, *ixyz)
                ep = dict(params.eqpars)
                ep.update({k: v for k, v in p.items() if k not in _CTOR_KEYS})
                return _gdi_exact(kx, ky, kz, ep)
            J = _J(params, kgrid, x0, sp)
            dJs = {k: _dJ(params, kgrid, x0, sp, {k: 1.0}) for k in keys}
            _eig_sens(c, f"{name} mode {ixyz}", J, dJs, want, keys, p0, tol, res_tol)
        # CMHD about a uniform state, one mode, both density variables
        for name, gamma in (("cmhd_rho", 1.0), ("cmhd_lnrho", 5/3)):
            params, kgrid, _ = _case(name)
            lnrho = name == "cmhd_lnrho"
            vals = [np.log(_RHO0) if lnrho else _RHO0, *_U0, *_B0]
            x0 = np.asarray(jr.initialize(
                lambda x, y, z: jnp.stack([v + 0*x + 0*y + 0*z for v in vals]), params).fields)
            sp = {"iky": 1, "ikx": 1, "iz": 1}
            keys = ["Lx", "Ly", "Lz", "cs0", "diss"]
            p0 = {k: float(overrides.static_value(params, k)) for k in keys}

            def want(p, params=params, gamma=gamma):
                k = np.array(_mode_k(p, params, 1, 1, 1))
                return _cmhd_exact(k, p["cs0"], gamma, p["diss"], params.eqpars["hyper"])
            J = _J(params, kgrid, x0, sp)
            dJs = {k: _dJ(params, kgrid, x0, sp, {k: 1.0}) for k in keys}
            _eig_sens(c, f"{name} uniform state", J, dJs, want, keys, p0, tol, res_tol)


# ------------------------------------------------------------------------------ gate 5

_TEAR = dict(nx=48, Lx=16.0, k=0.5, eta=0.02, alpha=0.8)


def _tearing():
    t = _TEAR
    params = fresh_params(dims=2, nx=t["nx"], ny=8, Lx=t["Lx"], Ly=TWO_PI/t["k"],
                          diss=(t["eta"], t["eta"]), hyper=1, comm_backend="serial")
    kgrid = jr.setup_kgrids(params)
    psi = lambda x, y: 1/jnp.cosh(x - t["Lx"]/2)**2 + 0*y
    x0 = np.asarray(jr.initialize(lambda x, y: jnp.stack([t["alpha"]*psi(x, y), psi(x, y)]),
                                  params).fields)
    return params, kgrid, x0


def _tearing_mode(J):
    ev = stability.eig_dense(J)
    j = int(np.argmax(ev.values.real))
    return ev, j


@pytest.mark.fp64
def test_tearing_eta_and_k_sensitivity():
    t = _TEAR
    with checks() as c:
        params, kgrid, x0 = _tearing()
        J = _J(params, kgrid, x0, {"iky": 1})
        ev, j = _tearing_mode(J)
        lam = ev.values[j]
        c.check(f"shear tearing is unstable (gamma {lam.real:.5f})", lam.real > 0)
        for label, dparams, point_of in (
                ("deta", {"diss": (0.0, 1.0)},
                 lambda h: {"diss": (t["eta"], t["eta"] + h)}),
                ("dnu", {"diss": (1.0, 0.0)},
                 lambda h: {"diss": (t["eta"] + h, t["eta"])}),
                ("dk", {"Ly": -params.Ly/t["k"]},
                 lambda h: {"Ly": TWO_PI/(t["k"] + h)})):
            dJ = _dJ(params, kgrid, x0, {"iky": 1}, dparams)
            s = stability.eigenvalue_sensitivity(J, dJ, lam, ev.right[:, j], ev.left[:, j],
                                                 spectrum=ev.values)
            hs = (1e-4, 5e-5) if label != "dk" else (1e-3, 5e-4)
            errs = []
            for h in hs:
                lp = [_tearing_mode(_J(params, kgrid, x0, {"iky": 1}, point_of(sg*h)))
                      for sg in (1, -1)]
                vp = [ev_.values[np.argmin(np.abs(ev_.values - lam))] for ev_, _ in lp]
                fd = (vp[0] - vp[1])/(2*h)
                errs.append(abs(s.dlam - fd)/abs(fd))
            order = fit_order(hs, errs)
            c.check(f"tearing dlambda/{label} = {s.dlam:.6f} == eigenvalue FD (errs "
                    f"{errs[0]:.1e}, {errs[1]:.1e}, order {order:.2f}; kappa {s.kappa:.1f})",
                    errs[-1] < 1e-5 and (errs[-1] < 1e-10 or 1.7 < order < 2.3),
                    f"errs {errs}, order {order:.2f}")
        # propagator_eigs at an overridden point: exp(J T) AND the residuals at that point
        ov = {"diss": (0.03, 0.03)}
        Jov = _J(params, kgrid, x0, {"iky": 1}, ov)
        ev_ov, j_ov = _tearing_mode(Jov)
        r = stability.propagator_eigs(x0, kgrid, params, T=8.0, dt=0.125, k=3, iky=1,
                                      overrides=ov)
        e = abs(r.values[0] - ev_ov.values[j_ov])
        moved = abs(ev_ov.values[j_ov] - lam)
        c.check(f"propagator_eigs(overrides) leading == eig_dense at the overridden point "
                f"(abs {e:.1e}; the point moves gamma by {moved:.1e}; residual "
                f"{r.residuals[0]:.1e})", e < 1e-6 and moved > 1e-3, f"{e:.3e}")


# ------------------------------------------------------------------------------ gate 6

def test_rejections():
    # any precision: no tolerances
    with checks() as c:
        p2, kg2, x2 = _case("rmhd2d")
        pf, kgf, _ = _case("rmhd_fdz")
        ps, kgs, _ = _case("rmhd_zsep")
        pg, kgg, _ = _case("gdi2d")
        pc, kgc, _ = _case("cmhd_rho")
        from taranis import grids
        bad = [
            (p2, {"hyper": 2}, "static RMHD parameter"),
            (p2, {"nx": 32}, "static Parameters argument"),
            (p2, {"cfl_safety": 0.1}, "static Parameters argument"),
            (pc, {"gamma": 1.4}, "static CMHD parameter"),
            (pc, {"density_var": 1.0}, "static CMHD parameter"),
            (pg, {"lin_dt_safety": 0.5}, "static GDI parameter"),
            (p2, {"eta": 0.1}, "unknown key"),
            (p2, {"cs0": 1.0}, "unknown key"),
            (p2, {"Lz": 3.0}, "does not enter"),
            (p2, {"z_diss": 0.1}, "does not enter"),
            (pf, {"z_diss_k": 0.1}, "does not enter"),
            (ps, {"z_diss": 0.1}, "does not enter"),
            (pg, {"D_par": 0.1}, "does not enter"),
            (p2, {"diss": 0.01 + 0.0j}, "must be real"),
            (p2, {"diss": np.array([0.01, 0.02], dtype=complex)}, "must be real"),
            (p2, {"diss": (0.01, 0.02, 0.03)}, "shape"),
            (pc, {"diss": (0.01, 0.02)}, "shape"),
            (p2, {"Ly": (1.0, 2.0)}, "shape"),
            (p2, {"Lx": 0.0}, "must be > 0"),
            (p2, {"Ly": -3.0}, "must be > 0"),
            (pc, {"cs0": 0.0}, "must be > 0"),
            (p2, {"diss": np.nan}, "finite"),
            (p2, {"Lx": np.inf}, "finite"),
            (p2, [("diss", 0.01)], "must be a dict"),
        ]
        for params, ov, msg in bad:
            with pytest.raises(ValueError, match=msg):
                grids.setup_kgrids(params, overrides=ov)
            with pytest.raises(ValueError, match=msg):
                stability.ky_block_matrix(np.zeros((params.nfields, params.nz, params.nx,
                                                    params.ny//2 + 1), complex),
                                          jr.setup_kgrids(params) if params is not p2 else kg2,
                                          params, 1, overrides=ov)
            c.check(f"{params.eqtype} overrides={ov!r} rejected ({msg})", True)
        # a direction may have any sign, but not a dead / static key
        stability.dj_matrix(x2, None, kg2, p2, iky=1, dparams={"Lx": -1.0})
        c.check("a negative box-length DIRECTION is accepted", True)
        with pytest.raises(ValueError, match="does not enter"):
            stability.dj_matrix(x2, None, kg2, p2, iky=1, dparams={"Lz": 1.0})
        with pytest.raises(ValueError, match="static"):
            stability.dj_matrix(x2, None, kg2, p2, iky=1, dparams={"hyper": 1.0})
        c.check("dparams validated like overrides", True)
        # an overridden kgrid: never into the solver, never overridden twice
        kg_ov = grids.setup_kgrids(p2, overrides={"diss": 0.05})
        state = make_state(p2)
        with pytest.raises(ValueError, match="carries overrides"):
            run.block_of_steps(state, kg_ov, p2, 1, *_scheme("lsrk33"))
        with pytest.raises(ValueError, match="carries overrides"):
            jr.simulate(make_state(p2), kg_ov, p2, 1.0, 1.0, None, save=False)
        with pytest.raises(ValueError, match="carries overrides"):
            jr.simulate_scan(make_state(p2), kg_ov, p2, 1, 1.0, 1.0, None, save=False)
        c.check("run.py entry points reject an overridden kgrid", True)
        with pytest.raises(ValueError, match="already carries overrides"):
            stability.ky_block_matrix(x2, kg_ov, p2, 1, overrides={"diss": 0.01})
        with pytest.raises(ValueError, match="plain setup_kgrids"):
            stability.dj_matrix(x2, None, kg_ov, p2, iky=1, dparams={"diss": 1.0})
        c.check("overrides= / dparams= on an already-overridden kgrid are refused", True)
        # ... while an overridden kgrid passed with overrides=None IS that point
        Ja = _J(p2, kg_ov, x2, {"iky": 1})
        Jb = _J(p2, kg2, x2, {"iky": 1}, {"diss": 0.05})
        c.check("an overridden kgrid with overrides=None is J at its point",
                np.array_equal(Ja, Jb))
        # forcing and the jax backend
        pforced = fresh_params(dims=2, nx=16, ny=16, forcing=True, comm_backend="serial")
        with pytest.raises(ValueError, match="unforced"):
            grids.setup_kgrids(pforced, overrides={})
        pj = fresh_params(dims=2, nx=16, ny=8, comm_backend="serial")
        pj.runtime = dataclasses.replace(pj.runtime, backend="jax")
        with pytest.raises(ValueError, match="non-sharded"):
            grids.setup_kgrids(pj, overrides={})
        c.check("forcing and comm_backend='jax' rejected by setup_kgrids(overrides=)", True)
        # a traced value cannot be recorded
        with pytest.raises(ValueError, match="traced"):
            jax.jit(lambda v: overrides.to_record(p2, {"diss": v}) and v)(0.01)
        c.check("to_record refuses a traced value", True)


def _scheme(name):
    from taranis import timestepping
    stepper, scheme = timestepping.get_scheme(name)
    return scheme, stepper


def test_save_record_round_trip():
    # any precision
    with checks() as c, tempfile.TemporaryDirectory() as d:
        params, kgrid, x0 = _case("rmhd2d")
        ov = {"diss": (0.03, 0.015), "Ly": 5.0}
        params.save(d, overrides=ov)
        path = os.path.join(d, "params.json")
        with open(path) as f:
            rec = json.load(f)
        c.check(f"save stamps _overrides ({rec.get('_overrides')})",
                rec.get("_overrides") == {"Ly": 5.0, "diss": [0.03, 0.015]})
        back = jr.Parameters.load_overrides(d)
        c.check(f"load_overrides round-trips ({back})", back == {"Ly": 5.0, "diss": (0.03, 0.015)})
        c.check("J at the loaded point == J at the original point",
                np.array_equal(_J(params, kgrid, x0, {"iky": 1}, back),
                               _J(params, kgrid, x0, {"iky": 1}, ov)))
        before = open(path).read()
        params.save(d, overrides={"Ly": 5.0, "diss": [0.03, 0.015]})
        c.check("identical re-save is a no-op", open(path).read() == before)
        with pytest.raises(ValueError, match="different parameters"):
            params.save(d, overrides={"diss": (0.03, 0.02), "Ly": 5.0})
        with pytest.raises(ValueError, match="different parameters"):
            params.save(d)
        c.check("re-save with different / without overrides is a hard error", True)
        with pytest.raises(ValueError, match="static"):
            params.save(d, overrides={"hyper": 2})
        c.check("save validates the overrides", True)
        with pytest.warns(UserWarning, match="NOT part of the returned Parameters"):
            p2 = jr.Parameters.from_snapshot(d)
        c.check("from_snapshot warns and does NOT fold the overrides in",
                tuple(p2.eqpars["diss"]) == (0.02, 0.01) and p2.Ly == params.Ly)
        with pytest.raises(TypeError, match="load_overrides"):
            jr.Parameters.from_snapshot(d, overrides=jr.Parameters.load_overrides(d))
        c.check("from_snapshot(overrides=) is a TypeError naming load_overrides", True)
    with checks() as c, tempfile.TemporaryDirectory() as d:
        params, _, _ = _case("rmhd2d")
        params.save(d)
        c.check("no overrides: no _overrides key; load_overrides -> None",
                "_overrides" not in json.load(open(os.path.join(d, "params.json")))
                and jr.Parameters.load_overrides(d) is None)
        params.save(d, overrides={})
        c.check("overrides={} records nothing (a no-op re-save)", True)
        with pytest.raises(ValueError, match="different parameters"):
            params.save(d, overrides={"diss": 0.5})
        c.check("adding overrides to an un-stamped record is a hard error", True)


# ------------------------------------------------------------------ gate 7 (review additions)

def test_every_entry_point_honours_overrides():
    # any precision. Every operator/solver entry point is its own `overrides=` gate (the
    # matrices above are only two of them): each, at a moved point, equals the same entry
    # point handed the explicitly overridden kgrid (overrides=None -- J at its point, gate 6),
    # and differs from the static point. djvp_operator / dj_operator are checked against
    # dj_matrix and against an FD of jvp_operator.
    from taranis import grids
    tol = 1e-12 if FP64 else 1e-5
    rng = np.random.default_rng(3)
    with checks() as c:
        for name in ("rmhd2d", "rmhd2d_real"):
            params, kgrid, x0 = _case(name)
            sp = _space(name)
            ov = {"Ly": params.Ly*1.15, "diss": (0.035, 0.022)}
            kg_ov = grids.setup_kgrids(params, overrides=ov)
            shape = x0.shape

            def cmp(label, got, want, base):
                e, m = _relerr(got, want), _relerr(base, want)
                c.check(f"{name}: {label} honours overrides (rel {e:.1e}; the point moves it "
                        f"by {m:.1e})", e < tol and m > 1e-3, f"rel {e:.3e}, move {m:.3e}")

            v = jnp.asarray(rng.standard_normal(shape) + 1j*rng.standard_normal(shape))
            v = v*(jnp.asarray(kgrid.dealias))            # dealias-kept: a real-field-like probe
            dx0 = np.roll(x0, 1, axis=2)*0.5
            for label, f in (("jvp_operator", stability.jvp_operator),
                             ("transpose_operator", stability.transpose_operator)):
                cmp(label, f(x0, kgrid, params, overrides=ov)(v), f(x0, kg_ov, params)(v),
                    f(x0, kgrid, params)(v))
            cmp("djvp_operator (state direction)",
                stability.djvp_operator(x0, dx0, kgrid, params, overrides=ov)(v),
                stability.djvp_operator(x0, dx0, kg_ov, params)(v),
                stability.djvp_operator(x0, dx0, kgrid, params)(v))
            # a parameter direction on the fields array: J is affine in diss, so the central
            # difference of jvp_operator is exact to round-off
            h = 1e-3
            fd = (stability.jvp_operator(x0, kgrid, params, overrides=dict(
                ov, diss=(0.035, 0.022 + h)))(v) - stability.jvp_operator(
                x0, kgrid, params, overrides=dict(ov, diss=(0.035, 0.022 - h)))(v))/(2*h)
            dv = stability.djvp_operator(x0, None, kgrid, params, dparams={"diss": (0.0, 1.0)},
                                         overrides=ov)(v)
            e = _relerr(dv, fd)
            c.check(f"{name}: djvp_operator(dparams) == FD of jvp_operator (rel {e:.1e})",
                    e < (1e-9 if FP64 else 2e-3) and float(jnp.linalg.norm(dv)) > 0, f"{e:.3e}")
            if "iky" in sp:
                J = _J(params, kgrid, x0, sp, ov)
                J0 = _J(params, kgrid, x0, sp)
                n = J.shape[0]
                u = rng.standard_normal(n) + 1j*rng.standard_normal(n)
                op, _ = stability.ky_block_operator(x0, kgrid, params, sp["iky"], overrides=ov)
                cmp("ky_block_operator matvec", op.matvec(u), J @ u, J0 @ u)
                cmp("ky_block_operator rmatvec", op.rmatvec(u), J.conj().T @ u,
                    J0.conj().T @ u)
                pre = stability.diagonal_preconditioner(kgrid, params, iky=sp["iky"],
                                                        overrides=ov)(0.3)
                pre1 = stability.diagonal_preconditioner(kg_ov, params, iky=sp["iky"])(0.3)
                pre0 = stability.diagonal_preconditioner(kgrid, params, iky=sp["iky"])(0.3)
                cmp("diagonal_preconditioner", pre.matvec(u), pre1.matvec(u), pre0.matvec(u))
                prop = dict(T=0.5, dt=0.125, iky=sp["iky"])
                P = stability.propagator_operator(x0, kgrid, params, overrides=ov, **prop)[0]
                P1 = stability.propagator_operator(x0, kg_ov, params, **prop)[0]
                P0 = stability.propagator_operator(x0, kgrid, params, **prop)[0]
                cmp("propagator_operator", P.matvec(u), P1.matvec(u), P0.matvec(u))
                dp = {"Ly": 1.0}
                dJ = _dJ(params, kgrid, x0, sp, dp, ov)
                dJ0 = _dJ(params, kgrid, x0, sp, dp)
                dop, _ = stability.dj_operator(x0, None, kgrid, params, iky=sp["iky"],
                                               dparams=dp, overrides=ov)
                cmp("dj_operator (parameter direction)", dop.matvec(u), dJ @ u, dJ0 @ u)
            else:
                M = _J(params, kgrid, x0, sp, ov)
                M0 = _J(params, kgrid, x0, sp)
                n = M.shape[0]
                u = rng.standard_normal(n)
                op, _ = stability.real_operator(x0, kgrid, params, overrides=ov)
                cmp("real_operator matvec", op.matvec(u), M @ u, M0 @ u)
                cmp("real_operator rmatvec", op.rmatvec(u), M.T @ u, M0.T @ u)
                ka = stability.ky_averaged_preconditioner(x0, kgrid, params, overrides=ov)(0.3)
                ka1 = stability.ky_averaged_preconditioner(x0, kg_ov, params)(0.3)
                ka0 = stability.ky_averaged_preconditioner(x0, kgrid, params)(0.3)
                cmp("ky_averaged_preconditioner", ka.matvec(u), ka1.matvec(u), ka0.matvec(u))


def test_review_extra_routes():
    # any precision: independent-route checks for what gates 1-2 leave out -- the (1,) diss
    # shape every RMHD/CMHD read site takes, and CMHD lnrho at gamma = 1 (the k-local
    # cs0^2 s pressure, a read site of its own that neither CMHD configuration reaches)
    tol = 1e-12 if FP64 else 2e-5
    with checks() as c:
        for name, d in (("rmhd2d", (0.03,)), ("rmhd_zsep", (0.03,)), ("cmhd_rho", (0.04,))):
            params, kgrid, x0 = _case(name)
            space = _space(name)
            p2 = _make(name, {"diss": d})
            e = _relerr(_J(params, kgrid, x0, space, {"diss": d}),
                        _J(p2, jr.setup_kgrids(p2), x0, space))
            c.check(f"{name}: J(overrides diss={d}) == J(Parameters diss={d}) (rel {e:.1e})",
                    e < tol, f"rel {e:.3e}")
        kw = dict(_BASE, **_CFG["cmhd_lnrho"])
        eq = dict(kw.pop("eqpars"), gamma=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = jr.Parameters(eqpars=eq, **kw)
            p2 = jr.Parameters(eqpars=dict(eq, cs0=0.95), **kw)
        x0 = _x0("cmhd_lnrho", p1)
        J2 = _J(p2, jr.setup_kgrids(p2), x0, {"iky": 1})
        J1 = _J(p1, jr.setup_kgrids(p1), x0, {"iky": 1})
        Jo = _J(p1, jr.setup_kgrids(p1), x0, {"iky": 1}, {"cs0": 0.95})
        e, m = _relerr(Jo, J2), _relerr(J1, J2)
        c.check(f"cmhd lnrho gamma=1: J(overrides cs0) == J(Parameters cs0) (rel {e:.1e}; "
                f"move {m:.1e})", e < tol and m > 1e-3, f"{e:.3e}")
        # a partly non-finite value is non-finite
        p, _, _ = _case("rmhd2d")
        from taranis import grids
        with pytest.raises(ValueError, match="finite"):
            grids.setup_kgrids(p, overrides={"diss": (0.01, np.nan)})
        c.check("a pair with one NaN is rejected", True)
        # from_snapshot pops the stamp: it is not reported as an unknown ctor parameter
        with tempfile.TemporaryDirectory() as dd:
            p.save(dd, overrides={"diss": 0.02})
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                jr.Parameters.from_snapshot(dd)
            unk = [str(x.message) for x in w if "unknown parameters" in str(x.message)]
            c.check(f"from_snapshot: the stamp is not an unknown parameter ({unk or 'ok'})",
                    not unk)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
