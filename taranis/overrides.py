# The overrides seam (plans/AUTODIFF_PLAN.md "Rung 2"): traced values for a fixed set of
# continuous physics parameters, for the linear-stability harness ONLY.
#
# Parameters is static and identity-hashed; it is never traced. An `overrides` dict
# {key: value} (fixed structure: its keys and the SHAPE of each value are static, the values
# may be tracers) rides in the kgrid -- K_Grids.overrides, set only by
# grids.setup_kgrids(params, overrides=...) -- and every physics read site of an overridable
# parameter goes through ONE helper, getp(params, kgrid.overrides, key), behind a plain python
# `if kgrid.overrides is not None` branch. overrides=None is therefore today's graph, op for
# op (the refactor reference and gate 6 are the standing proof). The solver never sees an
# overridden kgrid: run.py's entry points reject one.
#
# Overridable = continuous, enters J, and read nowhere as a trace-time python value. Anything
# that sets a shape or a trace-time branch (nx, dims, z_spectral, hyper, gamma, density_var,
# expansion, eqtype, precision...) or does not enter J (cfl_safety, dt, lin_dt_safety, the
# forcing knobs) is STATIC: naming one is a ValueError, as is an unknown key, a key that is
# dead in this configuration (Lz in 2D, D_par in 2D GDI, z_diss_k under FD-z, z_diss under
# z_spectral), a complex or non-finite value, or a value whose shape the read site cannot take.
#
# Per equation set (every key ALSO reaches J through the kgrid when it is a box length):
#   all      Lx, Ly (kx, ky, ksq, inv_ksq and L), Lz (dims=3: kz under z_spectral; dz of the
#            FD-z stencil and filter otherwise)
#   RMHD     diss (scalar, or (nu, eta)), z_diss_k (z_spectral), z_diss (FD-z)
#   GDI      Ln, nu_in, v0, gpar_fac, diss (scalar), D_par (dims=3)
#   CMHD     cs0, diss (scalar, or (D_rho, nu, eta))
# Values are cast to the FIELD real dtype (_precision.ftype) by setup_kgrids, like every other
# kgrid array: a strong float64 scalar must not upcast an fp32 field graph.
import inspect

import numpy as np

_LENGTHS = ("Lx", "Ly", "Lz")

# the eqpars keys each equation set knows that are STATIC (why, for the error message)
_STATIC_EQPARS = {
    "RMHD": {"hyper": "the dissipation exponent: an integer power fixed at trace time"},
    "GDI": {"hyper": "the dissipation exponent: an integer power fixed at trace time",
            "lin_dt_safety": "a timestep ceiling only; it does not enter J"},
    "CMHD": {"hyper": "the dissipation exponent: an integer power fixed at trace time",
             "gamma": "the polytropic index selects trace-time branches (gamma == 1)",
             "density_var": "selects the evolved density variable (a trace-time branch)",
             "expansion": "the expanding box is time-dependent and structural"},
}


def _static_rmhd(params, key):
    from .physics import rmhd
    if key == "diss":
        return rmhd._diss_hyper(params)[0]
    if key == "z_diss_k":
        return rmhd._z_diss_k(params)
    return params.z_diss                                  # "z_diss"


_GDI_INDEX = {"Ln": 0, "nu_in": 1, "v0": 2, "gpar_fac": 3, "diss": 4, "D_par": 6}


def _static_gdi(params, key):
    from .physics import gdi
    return gdi._eqpars(params)[_GDI_INDEX[key]]


def _static_cmhd(params, key):
    from .physics import cmhd
    cs0, diss, _, _ = cmhd._eqpars(params)
    return cs0 if key == "cs0" else diss


def _is3d(p):
    return p.spatial_dimensions == 3


def _fdz(p):
    return p.spatial_dimensions == 3 and not p.z_spectral


def _zspec(p):
    return p.spatial_dimensions == 3 and p.z_spectral


class _Key:
    # one overridable parameter: its static value, the value shapes its read site takes,
    # and the configurations it is live in (why not, otherwise)
    def __init__(self, static, shapes, live=None, dead_reason=""):
        self.static, self.shapes, self.live, self.dead_reason = static, shapes, live, dead_reason


def _length(key):
    return _Key(lambda p, k=key: getattr(p, k), lambda p: ((),),
                live=_is3d if key == "Lz" else None,
                dead_reason="dims=2 has no z axis" if key == "Lz" else "")


def _table(eqtype):
    t = {k: _length(k) for k in _LENGTHS}
    if eqtype == "RMHD":
        t["diss"] = _Key(_static_rmhd, lambda p: ((), (1,), (p.nfields,)))
        t["z_diss_k"] = _Key(_static_rmhd, lambda p: ((),), live=_zspec,
                             dead_reason="z_diss_k is the z_spectral kz^4 filter")
        t["z_diss"] = _Key(_static_rmhd, lambda p: ((),), live=_fdz,
                           dead_reason="z_diss is the finite-difference-z filter (dims=3, "
                                       "z_spectral=False)")
    elif eqtype == "GDI":
        for k in ("Ln", "nu_in", "v0", "gpar_fac", "diss"):
            t[k] = _Key(_static_gdi, lambda p: ((),))
        t["D_par"] = _Key(_static_gdi, lambda p: ((),), live=_is3d,
                          dead_reason="D_par is the 3D parallel closure")
    elif eqtype == "CMHD":
        t["cs0"] = _Key(_static_cmhd, lambda p: ((),))
        t["diss"] = _Key(_static_cmhd, lambda p: ((), (1,), (3,)))
    return t


def overridable(params):
    # the keys live in this configuration, sorted
    return sorted(k for k, e in _table(params.eqtype).items() if e.live is None or e.live(params))


def static_value(params, key):
    # the Parameters' own value of an overridable key, as the physics read site parses it
    entry = _table(params.eqtype).get(key)
    if entry is None:
        raise KeyError(f"overrides: {key!r} is not an overridable {params.eqtype} parameter")
    return entry.static(params, key)


def getp(params, overrides, key):
    """THE read-site helper: overrides[key] when an overrides dict carries key, else the
    static value (the one the unmodified read site uses). Read sites call it only inside
    their `if kgrid.overrides is not None` branch, so overrides=None never reaches it."""
    if overrides is not None and key in overrides:
        return overrides[key]
    return static_value(params, key)


def _ctor_args():
    from .config import Parameters
    return set(inspect.signature(Parameters.__init__).parameters) - {"self", "runtime"}


def _is_tracer(v):
    import jax
    return isinstance(v, jax.core.Tracer)


def validate(params, overrides, direction=False):
    """Check an overrides dict against params: keys overridable and live here, values real
    with a shape the read site takes, box lengths and cs0 > 0 where concrete (not for a
    `direction`, a tangent in parameter space, which may have any sign). Returns a new dict of
    FIELD-precision real jnp arrays (tracers stay tracers); None passes through."""
    if overrides is None:
        return None
    import jax.numpy as jnp
    from . import _precision
    if not isinstance(overrides, dict):
        raise ValueError(f"overrides must be a dict {{key: value}}, got "
                         f"{type(overrides).__name__}")
    table = _table(params.eqtype)
    static_eq = _STATIC_EQPARS.get(params.eqtype, {})
    live = overridable(params)
    out = {}
    for key, value in overrides.items():
        if key not in table:
            if key in static_eq:
                raise ValueError(f"overrides: {key!r} is a static {params.eqtype} parameter "
                                 f"({static_eq[key]}) and cannot be overridden; overridable "
                                 f"here: {live}")
            if key in _ctor_args():
                raise ValueError(f"overrides: {key!r} is a static Parameters argument (it sets "
                                 f"a shape or a trace-time branch, or does not enter the "
                                 f"Jacobian) and cannot be overridden; overridable here: {live}")
            raise ValueError(f"overrides: unknown key {key!r} for eqtype {params.eqtype!r}; "
                             f"overridable here: {live}")
        entry = table[key]
        if entry.live is not None and not entry.live(params):
            raise ValueError(f"overrides: {key!r} does not enter this configuration "
                             f"({entry.dead_reason}); overridable here: {live}")
        if np.iscomplexobj(value) or (hasattr(value, "dtype")
                                      and jnp.issubdtype(value.dtype, jnp.complexfloating)):
            raise ValueError(f"overrides: {key!r} must be real, got a complex value")
        shape = tuple(np.shape(value))
        if shape not in entry.shapes(params):
            raise ValueError(f"overrides: {key!r} has shape {shape}; its read site takes "
                             f"{list(entry.shapes(params))}")
        arr = jnp.asarray(value).astype(_precision.ftype)
        if not _is_tracer(arr) and not bool(np.all(np.isfinite(np.asarray(arr)))):
            raise ValueError(f"overrides: {key!r} must be finite, got {value!r}")
        if (key in _LENGTHS or key == "cs0") and not direction and not _is_tracer(arr):
            if not bool(np.all(np.asarray(arr) > 0)):
                raise ValueError(f"overrides: {key!r} must be > 0, got {value!r}")
        out[key] = arr
    return out


def to_record(params, overrides):
    # overrides as plain JSON (floats / lists of floats), for Parameters.save. Validated; a
    # traced value cannot be recorded (save is a host-side call about CONCRETE values)
    ov = validate(params, overrides)
    rec = {}
    for key in sorted(ov):
        if _is_tracer(ov[key]):
            raise ValueError(f"Parameters.save: overrides[{key!r}] is a traced value; record "
                             f"the concrete point the study ran at")
        # from the value as passed (not the ftype cast): an fp32 session records what the
        # caller asked for
        a = np.asarray(overrides[key], dtype=float)
        rec[key] = a.tolist() if a.ndim else float(a)
    return rec
