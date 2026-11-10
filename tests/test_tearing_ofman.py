# plans/AUTODIFF_PLAN.md rung 2b item 1: the comparison with Ofman, Chen, Morrison & Steinolfson
# 1991 (Phys. Fluids B 3, 1364) -- the last section of examples/tearing-viscosity.ipynb, data from
# examples/tearing_ofman_run.py, which this file IMPORTS (examples/ is put on sys.path).
# Setup: their Harris sheet as the exact periodic tanh pair, flows V tanh(R x) / V[sech(R x) - 1],
# diss = (Pm/S, 1/S) and Ly through the overrides seam, dims = 2, ky block, fp64. Small grids.
#   (i)   equilibrium: the pair profile, the tanh flow's shape/sign/R scaling, R = 1 == alpha Psi0
#         bitwise, the jet's zero at the sheets and its mean, phi' == u - <u>, the band;
#   (ii)  theory helpers against hand values: Delta', eq. 13c, C&M 1989 (JPP eq. 1.6), Porcelli,
#         the layer relation (Pm = 0 == B&L, inverse of the tables, -> Porcelli as Pm grows), Q,
#         slopes, the table roots;
#   (iii) independent routes: J at overrides == J of a Parameters built there; solve() == dense
#         eigenvalues; the Doppler restoration by box independence (Lx moves <u>, not physics);
#   (iv)  the ladder/cache/reader logic on small grids (keys, convergence stop, cap, retries,
#         read-back, KeyError when uncached), the small-S physics checks (eigenvalue vs the layer
#         relation, eq. 13c at Pm = 1 and its wrong sign at Pm = 100);
#   (v)   the cached headline numbers (skipped when examples/data/tearing-ofman has not been run).
# pytest: `pytest tests/test_tearing_ofman.py`; script: `python tests/test_tearing_ofman.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import functools
import importlib
import os
import sys
import tempfile

import numpy as np
import pytest

import taranis as jr
from taranis import stability

_EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _run_script(name="tearing_ofman_run"):
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    return importlib.import_module(name)


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


def _quiet(*a, **k):
    pass


# ============================================================ (i) equilibrium

def test_pair_and_flow_profiles():
    O = _run_script()
    U = O.U
    Lx, nx = 40.0, 4096
    x = np.arange(nx)*Lx/nx
    near1 = np.abs(x - Lx/4) < 4
    near2 = np.abs(x - 3*Lx/4) < 4
    with checks() as c:
        B = O.by_pair(x, Lx)
        c.check("by_pair == rung 1b-ref's U.by_tanhpair_a bitwise", np.array_equal(B, U.by_tanhpair_a(x, Lx, 1.0)))
        c.check("B = tanh(x - Lx/4) at sheet 1, -tanh(x - 3Lx/4) at sheet 2 (to e^-Lx/2)",
                np.abs(B[near1] - np.tanh(x[near1] - Lx/4)).max() < 1e-7
                and np.abs(B[near2] + np.tanh(x[near2] - 3*Lx/4)).max() < 1e-7)
        c.check("<B> = 0", abs(B.mean()) < 1e-13)
        for V_, R in ((1.0, 0.73), (0.6, 1.0), (1.0, 2.5)):
            u = O.flow_profile(x, Lx, "tanh", V_, R)
            c.check(f"tanh V={V_} R={R}: V tanh(R(x - x_s)) at sheet 1, -V tanh at sheet 2",
                    np.abs(u[near1] - V_*np.tanh(R*(x[near1] - Lx/4))).max() < 1e-5
                    and np.abs(u[near2] + V_*np.tanh(R*(x[near2] - 3*Lx/4))).max() < 1e-5)
            i = nx//4
            dudx = (u[i + 1] - u[i - 1])/(2*Lx/nx)
            c.check(f"G'(x_s) = V R ({dudx:.6f})", abs(dudx - V_*R) < 5e-4*V_*R)
            c.check("tanh flow mean below round-off", abs(u.mean()) < 1e-13)
        phi, psi, um = O.equilibrium_fields(nx, Lx, "tanh", 0.6, 1.0)
        c.check("R = 1: phi == 0.6 psi BITWISE (u_0 = alpha b_0 exactly), u_mean = 0",
                np.array_equal(phi, 0.6*psi) and um == 0.0)
        c.check("psi = int B_y", np.array_equal(psi, U._fft_integral(B, Lx)))
        phi2, _, _ = O.equilibrium_fields(nx, Lx, "tanh", 0.6, 0.9)
        c.check("R != 1: phi != alpha psi", np.abs(phi2 - 0.6*psi).max() > 1e-2)
        phi0, psi0, um0 = O.equilibrium_fields(nx, Lx, "none", 0.7, 0.3)
        c.check("no flow: phi = 0, u_mean = 0", not phi0.any() and um0 == 0.0)
        for V_, R in ((1.0, 0.44), (0.5, 6.0)):
            u = O.flow_profile(x, Lx, "sech", V_, R)
            ex = V_*(1/np.cosh(R*(x - Lx/4)) - 1)
            c.check(f"jet V={V_} R={R}: u(x_s) = 0 at both sheets",
                    abs(u[nx//4]) < 1e-15 and abs(u[3*nx//4]) < 1e-15)
            c.check("jet SAME sign at both sheets (u(x_s2 + d) == u(x_s1 + d))",
                    np.allclose(u[near2], u[near1], atol=1e-14, rtol=0))
            close = np.abs(x - Lx/4) < 2/R
            c.check("jet = V[sech(R(x - x_s)) - 1] near the sheet (tails ~ 2 sech(R Lx/4))",
                    np.abs(u[close] - ex[close]).max() < 3*V_/np.cosh(R*Lx/4))
            phi, psi, um = O.equilibrium_fields(nx, Lx, "sech", V_, R)
            jet_s = sum(1/np.cosh(R*(m*Lx)) + 1/np.cosh(R*(Lx/2 + m*Lx)) for m in range(-2, 3))
            mean_th = V_*(2*np.pi/(R*Lx) - jet_s)       # each sech integrates to pi/R
            c.check(f"u_mean = V(2 pi/(R Lx) - jet(x_s)) ({um:.12f} vs {mean_th:.12f})", abs(um - mean_th) < 1e-9)
            kx = 2*np.pi*np.fft.fftfreq(nx, d=Lx/nx)
            dphi = np.real(np.fft.ifft(1j*kx*np.fft.fft(phi)))
            c.check("phi' == u - u_mean (spectrally)", np.abs(dphi - (u - um)).max() < 1e-10)
        try:
            O.flow_profile(x, Lx, "jet", 1.0, 1.0)
            c.check("bad flow raises", False)
        except ValueError:
            c.check("bad flow raises", True)
        b0 = O.T.band_width("tanhpair", Lx)
        c.check("band: tanh-pair band at R <= 1 and for no flow, widened by R above",
                O.band(Lx, "none", 6.0) == b0 and O.band(Lx, "tanh", 0.73) == b0
                and O.band(Lx, "sech", 6.0) == int(np.ceil(6*(b0 - 2))) + 2 and O.band(Lx, "sech", 6.0) > 5*b0)


# ============================================================ (ii) theory helpers

@functools.lru_cache(maxsize=1)
def _theory_roots():
    # scratch copies of the two table roots (the viscosity notebook's and ours): the theory gates
    # never write into examples/data (a missing table is computed into the copy)
    import shutil
    O = _run_script()
    out = []
    for real in (O.V.DATA_ROOT_FULL, O.DATA_ROOT_FULL):
        tmp = tempfile.mkdtemp(prefix="taranis_ofman_theory_")
        os.makedirs(os.path.join(tmp, "theory"))
        src = os.path.join(real, "theory")
        if os.path.isdir(src):
            for f in os.listdir(src):
                shutil.copy(os.path.join(src, f), os.path.join(tmp, "theory", f))
        out.append(tmp)
    return tuple(out)


def test_theory_helpers_hand_values():
    O = _run_script()
    T, V = O.T, O.V
    vroot, oroot = _theory_roots()
    with _patched(V, DATA_ROOT_FULL=vroot), _patched(O, DATA_ROOT_FULL=oroot):
        if not os.path.exists(V._table_path(1.0, True, vroot)):
            V.theory_table(1.0, True, vroot)
        for f in os.listdir(os.path.join(vroot, "theory")):      # Pm = 1e3 must be "ours"
            if f.startswith("F_i_Pm1.000000000000e+03"):
                os.remove(os.path.join(vroot, "theory", f))
        _theory_checks(O, T, V)


def _theory_checks(O, T, V):
    with checks() as c:
        c.check("Harris Delta' a = 2(1/ka - ka) = 3 at ka = 0.5; 7.5 at 0.25",
                O.dprime_a(0.5) == 3.0 and abs(O.dprime_a(0.25) - 7.5) < 1e-15)
        c.check("13c: 1 at g = 0 for every Pm", all(O.ratio_13c(0.0, p) == 1.0 for p in (0.01, 1.0, 100.0)))
        c.check("13c at Pm = 1 is (1 - g^2)^(1/3)", abs(O.ratio_13c(0.6, 1.0) - 0.64**(1/3)) < 1e-15)
        c.check("13c hand values (Pm=100 g=0.2: 0.8803240304; Pm=0.01 g=0.8: 0.4450353915)",
                abs(O.ratio_13c(0.2, 100.0) - 0.8803240304007952) < 1e-13
                and abs(O.ratio_13c(0.8, 0.01) - 0.445035391533413) < 1e-13)
        c.check("13c symmetric under Pm -> 1/Pm only through (1-Pm)^2/(4Pm): Pm=100 and 0.01 equal",
                abs(O.ratio_13c(0.5, 100.0) - O.ratio_13c(0.5, 0.01)) < 1e-15)
        cm = (2*np.pi)**-0.5*np.sqrt(3.0)*np.sqrt(0.2*0.96)*1e-3
        c.check(f"C&M 1989 (JPP eq 1.6) at g=0.2, S=1e6: {cm:.10e}", abs(O.gamma_cm89(0.2, 1e6) - cm) < 1e-18
                and abs(O.gamma_cm89(0.2, 1e6) - 3.027759026424192e-4) < 1e-15)
        c.check("C&M has no explicit k: only through Delta'",
                abs(O.gamma_cm89(0.2, 1e6, ka=0.25)/O.gamma_cm89(0.2, 1e6) - np.sqrt(2.5)) < 1e-14)
        por = 3.0*1e-5*(1e-4)**(-1/6)*0.5**(1/3)/2.103592074
        c.check(f"Porcelli at S=1e6, Pm=100 = {por:.8e}", abs(O.gamma_porcelli(1e6, 100.0)/por - 1) < 1e-9)
        c.check("I_V = pi 6^(-1/3) Gamma(5/6)/Gamma(7/6)", abs(V.I_V - 2.1035920740) < 1e-9)
        c.check("layer Pm = 0 == B&L exact relation (T.gamma_disp, f' = 1, Delta' = 3)",
                O.gamma_layer(1e6, 0.0) == T.gamma_disp(0.5, 1e-6, 1.0, 3.0))
        for Pm in (0.1, 100.0, 1e4):
            g = O.gamma_layer(1e6, Pm)
            lam = g/((1e-6)**(1/3)*0.5**(2/3))
            lnF, _ = V.F_table(lam, Pm, root=O.theory_root(Pm))
            dd = 3.0*(1e-6/0.5)**(1/3)
            c.check(f"layer Pm={Pm:g}: F(Lam, Pm) == Delta' delta_eta ({np.exp(lnF):.10f} vs {dd:.10f})",
                    abs(np.exp(lnF)/dd - 1) < 1e-9)
        g = [O.gamma_layer(1e6, p) for p in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4)]
        c.check("layer gamma decreases monotonically with Pm", all(a > b for a, b in zip(g[:-1], g[1:])))
        r = [O.gamma_layer(S, 1e4)/O.gamma_porcelli(S, 1e4) for S in (1e6, 1e8)]
        c.check(f"layer -> Porcelli in the viscous constant-psi limit ({r[0]:.5f} at S=1e6, {r[1]:.5f} at 1e8)",
                0.9 < r[0] < r[1] < 1.0 and r[1] > 0.97)
        c.check("Q = ka F'(1 - (V R)^2)", abs(O.q_of(1.0, 0.73) - 0.5*(1 - 0.73**2)) < 1e-15
                and O.q_of(0.6, 1.0, ka=0.25) == 0.25*(1 - 0.36))
        xs = np.array([1e2, 1e3, 1e5, 1e7])
        mid, sl = O.local_slopes(xs, 3*xs**0.17)
        c.check("local_slopes on a power law", np.allclose(sl, 0.17, atol=1e-12, rtol=0)
                and np.allclose(mid, [np.sqrt(1e5), np.sqrt(1e8), 1e6]))
        c.check("lsq_slope on a power law", abs(O.lsq_slope(xs, 2*xs**-0.5) + 0.5) < 1e-12)
        _, sl2 = O.local_slopes([1, 2, 4], [1, 2, 8])
        c.check("local_slopes picks up curvature", np.allclose(sl2, [1, 2]))
        c.check("theory_root: the viscosity notebook's tables for Pm = 1, ours for 1e3",
                O.theory_root(1.0) == V.DATA_ROOT_FULL and O.theory_root(1e3) == O.DATA_ROOT_FULL)


# ============================================================ (iii) independent routes

@pytest.mark.fp64
def test_overrides_equal_direct_parameters():
    O = _run_script()
    nx, Lx = 128, 20.0
    with checks() as c:
        for (S, Pm, ka, flow, V_, R) in ((1e3, 0.0, 0.5, "none", 0.0, 1.0), (1e4, 10.0, 0.5, "tanh", 0.5, 0.73),
                                         (3e3, 1.0, 0.3, "sech", 1.0, 2.0)):
            Bo, idx = O.block_matrix(S, Pm, ka, flow, V_, R, Lx, nx)
            p = jr.Parameters(nx=nx, ny=8, Lx=Lx, Ly=2*np.pi/ka, dims=2, cfl_safety=0.5, forcing=False,
                              eqpars={"diss": (Pm/S, 1.0/S), "hyper": 1})
            phi, psi, um = O.equilibrium_fields(nx, Lx, flow, V_, R)
            x0 = O.U._fields(p, phi, psi)
            Bd, idx2 = stability.ky_block_matrix(x0, jr.setup_kgrids(p), p, 1)
            Bd = Bd - 1j*ka*um*np.eye(Bd.shape[0])
            err = np.abs(Bo - Bd).max()/np.abs(Bd).max()
            c.check(f"{flow} S={S:.0e} Pm={Pm} ka={ka}: J(overrides) == J(direct Parameters) ({err:.1e})",
                    err < 1e-13 and np.array_equal(idx, idx2))
        c.check("make_params' placeholder diss is not the physics one", O.DIFF_PLACEHOLDER == (0.0, 1e-6))


@pytest.mark.fp64
def test_solve_matches_dense_and_parity():
    O = _run_script()
    S, Pm, ka, Lx, nx = 1e3, 1.0, 0.5, 20.0, 512
    with checks() as c:
        for flow, V_, R in (("none", 0.0, 1.0), ("tanh", 0.5, 1.0), ("sech", 1.0, 2.0)):
            top = O.dense_top(S, Pm, ka, flow, V_, R, Lx, nx, 4)
            c.check(f"{flow}: dense_top returns nkeep eigenvalues, by decreasing Re",
                    len(top) == 4 and np.all(np.diff(top.real) <= 0))
            lam = O.pick(top)
            r = O.solve(S, Pm, ka, flow, V_, R, Lx, nx, complex(lam.real*1.03, lam.imag), k=2)
            got = O.pick(r["values"])
            c.check(f"{flow}: shift-invert == dense eigenvalue ({got:.10e} vs {lam:.10e})",
                    abs(got - lam) < 1e-9*abs(lam))
            other = [v for v in r["values"] if abs(v - got) > 1e-12]
            c.check(f"{flow}: the pair partner is dense's second eigenvalue",
                    len(other) == 1 and np.min(np.abs(top - other[0])) < 1e-9*abs(lam))
            par = r["parity"]
            c.check(f"{flow}: parities are +1 and -1 ({par})", np.allclose(sorted(par.real), [-1, 1], atol=1e-6))
            j = int(np.argmax(np.real(r["values"])))
            c.check(f"{flow}: the faster member is EVEN about the midpoint", abs(par[j] - 1) < 1e-6)
            c.check(f"{flow}: u_mean recorded", (r["u_mean"] != 0.0) == (flow == "sech"))
        r3 = O.solve(S, Pm, ka, "none", 0.0, 1.0, Lx, nx, 1.03*O.pick(O.dense_top(S, Pm, ka, "none", 0.0, 1.0, Lx, nx, 2)), k=3)
        c.check("solve returns k eigenvalues", len(r3["values"]) == 3 and len(r3["parity"]) == 3)


@pytest.mark.fp64
def test_doppler_sign_by_box_independence():
    # <u> depends on Lx, the physics (to the pair coupling) does not: the restored true lambda must
    # agree between boxes, the uncorrected one must not
    O = _run_script()
    S, Pm, ka = 1e3, 1.0, 0.5
    res = {}
    for Lx, nx in ((20.0, 256), (30.0, 384)):
        lam = O.pick(O.dense_top(S, Pm, ka, "sech", 1.0, 2.0, Lx, nx, 3))
        um = O.equilibrium_fields(nx, Lx, "sech", 1.0, 2.0)[2]
        res[Lx] = (lam, um)
    (l1, u1), (l2, u2) = res[20.0], res[30.0]
    raw1, raw2 = l1 + 1j*ka*u1, l2 + 1j*ka*u2
    with checks() as c:
        c.check(f"<u> differs between boxes ({u1:.4f}, {u2:.4f})", abs(u1 - u2) > 0.03)
        c.check(f"true Im lambda box-independent ({l1.imag:.6f}, {l2.imag:.6f})", abs(l1.imag - l2.imag) < 2e-4)
        c.check("without the shift Im moves by k Delta<u>", abs(abs(raw1.imag - raw2.imag) - ka*abs(u1 - u2)) < 2e-4)
        c.check("Re box-independent to the pair coupling", abs(l1.real/l2.real - 1) < 2e-2)
        op = stability.ky_block_operator(*(lambda p: (O._x0(p, "sech", 1.0, 2.0)[0], O._kgrid(256, 20.0), p))(
            O.make_params(256, 20.0)), 1, overrides=O.V.overrides(ka, S, Pm))[0]
        v = np.random.default_rng(1).standard_normal(op.shape[0]) + 0j
        d = O.doppler(op, ka, u1)
        c.check("doppler(op) == op - i k <u> I", np.allclose(d.matvec(v), op.matvec(v) - 1j*ka*u1*v, atol=1e-13, rtol=0))
        c.check("doppler(op, 0) is op itself", O.doppler(op, ka, 0.0) is op)


@pytest.mark.fp64
def test_fd_route_against_taranis():
    # the fully independent route (one sheet, non-periodic, stretched finite differences, no taranis) against
    # taranis' dense block (the exact pair, Lx = 40) at S = 1e4: the field-aligned shear at Pm = 100, and the
    # jet in the sheet frame, whose Im part independently gates the Doppler restoration (without it Im moves
    # by k<u> ~ 0.4). The FD route with Ofman's printed nu W'''' instead of nu (d^2 - k^2)^2 W is measurably
    # further from taranis (whose viscosity is nu nabla^4) at every point.
    O = _run_script()
    S, ka, Lx, nx = 1e4, 0.5, 40.0, 1024
    got = {}
    with checks() as c:
        for name, Pm, flow, V_, R, tol in (("fa", 100.0, "tanh", 0.6, 1.0, 5e-4), ("nf", 100.0, "none", 0.0, 1.0, 5e-4),
                                           ("jet", 1.0, "sech", 1.0, 2.0, 2e-4)):
            lam = O.pick(O.dense_top(S, Pm, ka, flow, V_, R, Lx, nx, 4))
            hint = complex(lam.real*1.03, lam.imag)
            f, f4 = (O.fd_single_sheet(S, Pm, ka, flow, V_, R, hint, 3000, visc=v) for v in ("full", "d4"))
            g, g4 = (complex(v[np.argmin(np.abs(v - lam))]) for v in (f, f4))
            got[name] = (lam, g)
            c.check(f"{name}: FD {g:.6e} == taranis {lam:.6e} ({abs(g/lam - 1):.1e}; pair coupling ~3e-4)",
                    abs(g/lam - 1) < tol)
            c.check(f"{name}: the FD route's eigenvalues are sorted by Re", np.all(np.diff(f.real) <= 0))
            c.check(f"{name}: nu W'''' alone is further from taranis ({abs(g4/lam - 1):.1e})", abs(g4/lam - 1) > 5*abs(g/lam - 1))
        (lf, gf), (ln, gn) = got["fa"], got["nf"]
        c.check(f"Pm = 100 shear ratio FD {gf.real/gn.real:.6f} vs taranis {lf.real/ln.real:.6f}",
                abs((gf.real/gn.real)/(lf.real/ln.real) - 1) < 1e-4)
        lj, gj = got["jet"]
        um = O.equilibrium_fields(nx, Lx, "sech", 1.0, 2.0)[2]
        c.check(f"jet Im: FD {gj.imag:.6e} vs taranis {lj.imag:.6e}, against a Doppler error k<u> = {ka*abs(um):.3f}",
                abs(gj.imag - lj.imag) < 1e-5 and ka*abs(um) > 0.1)
        for bad in (dict(visc="d2"), dict(flow="jet")):
            try:
                O.fd_single_sheet(S, 1.0, ka, bad.get("flow", "none"), 0.0, 1.0, 1e-3, 200, visc=bad.get("visc", "full"))
                c.check(f"bad {bad} raises", False)
            except ValueError:
                c.check(f"bad {bad} raises", True)


# ============================================================ (iv) ladder, cache, readers, small-S physics

def test_cfg_keys_points_and_nx_rules():
    O = _run_script()
    T = O.T
    with checks() as c:
        c.check("cfg_of canonicalises no flow (V = 0, R = 1)", O.cfg_of(Pm=1.0, V_=0.3, R=0.5) == O.cfg_of(Pm=1.0))
        c.check("cfg_of defaults: S0, ka 0.5, Lx 40", O.cfg_of() == (O.S0, 0.0, 0.5, "none", 0.0, 1.0, 40.0))
        try:
            O.cfg_of(flow="jet")
            c.check("bad flow raises", False)
        except ValueError:
            c.check("bad flow raises", True)
        base = O.cfg_of(Pm=1.0, flow="tanh", V_=0.6, R=1.0)
        variants = [base, O.cfg_of(Pm=100.0, flow="tanh", V_=0.6), O.cfg_of(Pm=1.0, flow="tanh", V_=0.6, R=0.73),
                    O.cfg_of(Pm=1.0, flow="tanh", V_=0.4), O.cfg_of(Pm=1.0, flow="sech", V_=0.6),
                    O.cfg_of(S=1e5, Pm=1.0, flow="tanh", V_=0.6), O.cfg_of(Pm=1.0, ka=0.25, flow="tanh", V_=0.6),
                    O.cfg_of(Pm=1.0, flow="tanh", V_=0.6, Lx=60.0)]
        keys = {T._key(nx=8192, k=2, attempt=0, **O._keys(cc))[0] for cc in variants}
        c.check("cache keys distinct across (Pm, R, V, flow, S, ka, Lx)", len(keys) == len(variants))
        k0 = T._key(nx=8192, k=2, attempt=0, **O._keys(base))[0]
        c.check("... and across nx, k and attempt",
                len({k0, T._key(nx=16384, k=2, attempt=0, **O._keys(base))[0],
                     T._key(nx=8192, k=4, attempt=0, **O._keys(base))[0],
                     T._key(nx=8192, k=2, attempt=1, **O._keys(base))[0]}) == 4)
        c.check("nxs_for(40) == NXS, (60) rounds up to powers of two, cap filters",
                O.nxs_for(40.0) == O.NXS and all(n & (n - 1) == 0 for n in O.nxs_for(60.0))
                and all(n >= m*1.5 for n, m in zip(O.nxs_for(60.0), O.NXS))
                and O.nxs_for(40.0, 32768) == tuple(n for n in O.NXS if n <= 32768))
        c.check("NX_MAX caps every ladder (Lx = 80 drops the doubled top rung)",
                max(O.nxs_for(80.0)) == O.NX_MAX and len(O.nxs_for(80.0)) == len(O.NXS) - 1)
        hi = O.cfg_of(Pm=O.HI[0][0], flow=O.HI[0][1], V_=O.HI[0][2], R=O.HI[0][3])
        c.check("HI points: one more rung (NX_HI), only at S0/KA/LX",
                O.is_hi(hi) and not O.is_hi(O.cfg_of(S=1e5, Pm=hi[1], flow=hi[3], V_=hi[4], R=hi[5]))
                and not O.is_hi(O.cfg_of(Pm=1.0, flow="tanh", V_=hi[4])) and not O.is_hi(hi[:6] + (60.0,))
                and not O.is_hi(O.cfg_of(Pm=hi[1], ka=0.25, flow=hi[3], V_=hi[4], R=hi[5]))
                and O.nxs_for(40.0, hi=True) == O.NXS + (O.NX_HI,) and O.nxs_for(40.0, 65536, True) == O.nxs_for(40.0, 65536))
        c.check("ladder uses the HI rule", len([p for p in O.fa_points() + O.rinv_points() if O.is_hi(p)]) == len(O.HI))
        c.check("nx_seed_for", O.nx_seed_for(40.0) == O.NX_SEED and O.nx_seed_for(80.0) == 2*O.NX_SEED)
        c.check("_pow2_at_least", [O._pow2_at_least(n) for n in (1024, 1025, 3072, 4096.0)] == [1024, 2048, 4096, 4096])
        lists = [O.nf_points(), O.fa_points(), O.rinv_points(), O.rvisc_points(), O.slope_points(),
                 O.jet_points(), O.box_points(), O.cmk_points()]
        c.check("each phase's points are distinct", all(len(set(l)) == len(l) for l in lists))
        main = set(O.nf_points()) | set(O.fa_points()) | set(O.jet_points())
        c.check("shared points only by design: the Lx = 40 box points are main-campaign points, the S0 "
                "slope point is the R = 0.73 inviscid one",
                {p for p in O.box_points() if p[6] == O.LX} <= main and O.slope_points()[-1] in O.rinv_points())
        c.check("fa points: tanh, R = 1, V = alpha", all(p[3] == "tanh" and p[5] == 1.0 and p[4] in O.ALPHA_FA
                                                          for p in O.fa_points()))
        c.check("slope points end at S0, tanh R_SLOPE V = 1, inviscid",
                O.slope_points()[-1][0] == O.S0 and all(p[1] == 0.0 and p[5] == O.R_SLOPE for p in O.slope_points()))
        c.check("box points come in (LX, BOX_LX) pairs", [p[6] for p in O.box_points()] == [O.LX, O.BOX_LX]*len(O.BOX))
        c.check("theory_pms: every positive Pm of the campaign",
                set(O.theory_pms()) == {p for p in set(O.PM_NF) | set(O.PM_FA) | {j[0] for j in O.JET}
                                        | {b[0] for b in O.BOX} if p > 0})
        c.check("pick: largest growing Re, None when nothing grows",
                O.pick([1e-3 + 0.1j, 2e-3 - 0.2j, -1.0, 1e-12]) == 2e-3 - 0.2j and O.pick([-1e-3, 1e-10]) is None
                and O.pick([1e-3 + 0.8j, 2e-3 + 0j]) == 2e-3)


def test_rung_retry_ladder_and_failures():
    O = _run_script()
    seen = []

    def fake_cached(kind, root=None, compute=None, attempt=None, **cfg):
        seen.append((attempt, cfg["nx"], cfg["k"]))
        if attempt < 2:
            raise RuntimeError("(cached failure) GMRES")
        if attempt == 2:
            return dict(values=np.array([-1e-3 + 0j, 1e-12 + 0j]), parity=np.array([1, -1]))
        return dict(values=np.array([5e-3 + 0j, 4.9e-3 + 0j]), parity=np.array([1, -1]))
    with _patched(O.U, cached=fake_cached), checks() as c:
        rec, lam = O.rung(O.cfg_of(), 1024, 4e-3 + 0j, 2, root="x", log=_quiet)
        c.check("refusals and a non-growing result skipped: attempt 3 returned", lam == 5e-3 and seen[-1][0] == 3)

    def never(kind, root=None, compute=None, attempt=None, **cfg):
        raise RuntimeError("(cached failure) GMRES")
    with _patched(O.U, cached=never), checks() as c:
        try:
            O.rung(O.cfg_of(), 1024, 4e-3 + 0j, 2, root="x", log=_quiet)
            c.check("all attempts fail -> RuntimeError", False)
        except RuntimeError as e:
            c.check("all attempts fail -> RuntimeError", "no growing eigenvalue" in str(e))

    sigmas = []

    def fake_solve(S, Pm, ka, flow, V_, R, Lx, nx, sigma, k=2, rtol_mult=1.0):
        sigmas.append((nx, sigma, k, rtol_mult))
        lam = 1e-3*(1 + 1.0/nx**2)               # converges like nx^-2; partner listed FIRST
        return dict(values=np.array([0.99*lam, lam]), residuals=np.array([2e-12, 1e-12]),
                    parity=np.array([-1.0 + 0j, 1.0 + 0j]), u_mean=0.0, seconds=1.0)
    root = tempfile.mkdtemp(prefix="taranis_ofman_ladder_")
    seed_nx = []

    def fake_dense_top(*a, **k):
        seed_nx.append((a[6], a[7]))                # (Lx, nx) of the seed grid
        return np.array([2e-3 + 1e-4j, 1e-4 + 0j])
    with _patched(O, solve=fake_solve, dense_top=fake_dense_top,
                  NXS=(16, 32, 64, 128, 256)), checks() as c:
        r = O.ladder(O.cfg_of(), root, _quiet)
        c.check("seconds = the sum over the rungs", r["seconds"] == float(len(r["rows"])))
        n0 = len(sigmas)
        O.ladder(O.cfg_of(Lx=80.0), root, _quiet)
        del sigmas[n0:]
        c.check(f"the dense seed is on nx_seed_for(Lx) ({seed_nx})",
                seed_nx[0] == (40.0, O.nx_seed_for(40.0)) and seed_nx[-1] == (80.0, O.nx_seed_for(80.0))
                and O.nx_seed_for(80.0) != O.NX_SEED)
        c.check(f"never within CONV_TOL = 1e-6 on this ladder: runs to the top, not converged (nx {r['nx']})",
                r["nx"] == 256 and r["converged"] is False and len(r["rows"]) == 5)
        nxs = [row["nx"] for row in r["rows"]]
        exp_conv = abs(1/nxs[-1]**2 - 1/nxs[-2]**2)/(1 + 1/nxs[-1]**2)
        c.check(f"conv = |last - previous|/|last| ({r['conv']:.3e})", abs(r["conv"] - exp_conv) < 1e-12)
        c.check("first rung with SEED_K, later with LADDER_K", sigmas[0][2] == O.SEED_K
                and all(s[2] == O.LADDER_K for s in sigmas[1:]))
        c.check("first shift 3% above the dense seed, later 3% above the previous rung",
                abs(sigmas[0][1] - (2e-3*1.03 + 1e-4j)) < 1e-15 and abs(sigmas[1][1] - r["rows"][0]["lam"]*1.03) < 1e-15)
        c.check("partner, parity, split, residual of the REPORTED member recorded",
                abs(r["rows"][-1]["partner"] - 0.99*r["lam"]) < 1e-15 and r["parity"] == 1
                and abs(r["split"] - 0.01) < 1e-12 and r["rows"][-1]["residual"] == 1e-12)
        c.check("seed recorded", r["seed"] == 2e-3 + 1e-4j)
        r2 = O.ladder(O.cfg_of(), root, _quiet, compute=False)
        c.check("read-back from the cache identical", r2["lam"] == r["lam"] and r2["nx"] == r["nx"])
        try:
            O.ladder(O.cfg_of(Pm=3.0), root, _quiet, compute=False)
            c.check("uncached -> KeyError", False)
        except KeyError:
            c.check("uncached -> KeyError", True)
        with _patched(O, CONV_TOL=1e-3):
            r3 = O.ladder(O.cfg_of(Pm=5.0), root, _quiet)
            c.check(f"looser CONV_TOL stops earlier ({r3['nx']})", r3["nx"] == 64 and r3["converged"])
        def slow_solve(S, Pm, ka, flow, V_, R, Lx, nx, sigma, k=2, rtol_mult=1.0):
            lam = 1e-3*(1 + 0.0655/nx**2)             # last rung-to-rung change 3e-6: in (CONV_TOL, 10 CONV_TOL)
            return dict(values=np.array([0.99*lam, lam]), residuals=np.array([1e-12, 1e-12]),
                        parity=np.array([-1.0 + 0j, 1.0 + 0j]), u_mean=0.0, seconds=1.0)
        with _patched(O, solve=slow_solve):
            r6 = O.ladder(O.cfg_of(Pm=19.0), root, _quiet)
            c.check(f"converged means conv <= CONV_TOL exactly (conv {r6['conv']:.2e} at the top)",
                    O.CONV_TOL < r6["conv"] < 10*O.CONV_TOL and r6["nx"] == 256 and r6["converged"] is False)
        r4 = O.ladder(O.cfg_of(Pm=7.0), root, _quiet, cap=64)
        c.check("cap stops the ladder", r4["nx"] == 64 and not r4["converged"])
        real_rung = O.rung

        def rung_failing_at(nxbad):
            def f(c_, nx, hint, k, root=None, log=None, compute=True):
                if nx >= nxbad:
                    raise RuntimeError("no growing eigenvalue (fake)")
                return real_rung(c_, nx, hint, k, root, log, compute)
            return f
        with _patched(O, rung=rung_failing_at(64)):
            r5 = O.ladder(O.cfg_of(Pm=11.0), root, _quiet)
            c.check("a mid-ladder failure stops the ladder at the last good rung", r5["nx"] == 32 and not r5["converged"])
        with _patched(O, rung=rung_failing_at(16)):
            try:
                O.ladder(O.cfg_of(Pm=13.0), root, _quiet)
                c.check("a first-rung failure raises RuntimeError", False)
            except RuntimeError:
                c.check("a first-rung failure raises RuntimeError", True)
        with _patched(O, dense_top=lambda *a, **k: np.array([-1e-3 + 0j])):
            try:
                O.ladder(O.cfg_of(Pm=9.0), root, _quiet)
                c.check("no growing seed -> RuntimeError", False)
            except RuntimeError:
                c.check("no growing seed -> RuntimeError", True)


@pytest.mark.fp64
def test_make_data_and_readers_on_small_grids():
    O = _run_script()
    root = tempfile.mkdtemp(prefix="taranis_ofman_readers_")
    with _patched(O, S0=1e3, NXS=(256, 512), NX_SEED=256, PM_NF=(10.0, 30.0), PM_FA=(10.0,), ALPHA_FA=(0.3,),
                  DENSE=((10.0, 0.3),), NX_INDEP=512), checks() as c:
        c.check("dense_results empty before the dense phase", O.dense_results(root) == {})
        done = O.make_data(root, phases=("nf", "fa", "dense"), log=_quiet)
        dr = O.dense_results(root)
        top = {a: O.dense_top(1e3, 10.0, 0.5, "tanh" if a else "none", a, 1.0, 40.0, 512) for a in (0.0, 0.3)}
        c.check("dense_results: the fastest growing dense eigenvalue at alpha and at 0",
                list(dr) == [(10.0, 0.3)] and all(dr[(10.0, 0.3)][a] == O.pick(top[a]) for a in (0.0, 0.3))
                and dr[(10.0, 0.3)][0.3] != dr[(10.0, 0.3)][0.0])
        c.check("make_data on nf, fa: done", done)
        c.check("a capped, unconverged ladder leaves its phase INCOMPLETE",
                O.make_data(root, phases=("nf",), log=_quiet, cap=256) is False)

        def failing_ladder(*a, **k):
            raise RuntimeError("fake refusal")
        with _patched(O, ladder=failing_ladder):
            c.check("an incomplete phase is not hidden by a later complete one",
                    O.make_data(root, phases=("nf", "dense"), log=_quiet) is False)
        nf = O.nf_results(root)
        c.check("nf_results keyed by Pm", sorted(nf) == [10.0, 30.0])
        fa = O.fa_results(root)
        c.check("fa_results adds alpha = 0 from the no-flow points", sorted(fa) == [(10.0, 0.0), (10.0, 0.3)]
                and fa[(10.0, 0.0)] is not None and O.gamma(fa[(10.0, 0.0)]) == O.gamma(nf[10.0]))
        sr = O.shear_ratios(root=root)
        c.check("shear_ratios = gamma(alpha)/gamma(0)",
                list(sr) == [(10.0, 0.3)] and sr[(10.0, 0.3)] == O.gamma(fa[(10.0, 0.3)])/O.gamma(fa[(10.0, 0.0)]))
        c.check("readers skip uncached points", O.rinv_results(root) == {} and O.jet_results(root) == {})
        try:
            O.make_data(root, phases=("bogus",), log=_quiet)
            c.check("unknown phase raises", False)
        except ValueError:
            c.check("unknown phase raises", True)
        c.check("split_branches: fastest oscillating / fastest rest",
                O.split_branches([0.1 + 0.5j, 0.2 + 0.9j, 0.05 + 0.1j, 0.3 - 0.2j]) == (0.2 + 0.9j, 0.3 - 0.2j)
                and np.isnan(O.split_branches([0.1 + 0.1j])[0]) and O.split_branches([1 + 1j], im=2.0)[1] == 1 + 1j
                and O.split_branches([0.3 + 0.1j, 0.05 + 2.0j, 0.1 - 0.5j, 0.01 + 0j]) == (0.1 - 0.5j, 0.3 + 0.1j))
        with _patched(O, OSC_PM=(0.0, 100.0), OSC_NX=256):
            osc = O.osc_scan(root)
            B, _ = O.block_matrix(1e3, 100.0, O.KA, "sech", 1.0, 6.0, O.LX, 256)
            c.check("osc_scan: {Pm: split of the dense R = 6 jet spectrum}", sorted(osc) == [0.0, 100.0]
                    and osc[100.0] == O.split_branches(np.linalg.eigvals(B)) and osc == O.osc_scan(root, compute=False))
        c.check("expired budget -> incomplete", O.make_data(root, wall_budget=-1.0, phases=("rinv",), log=_quiet) is False)


def test_reader_keys():
    # every reader's key layout, on a fake ladder that returns its configuration (and raises
    # KeyError/RuntimeError for the points it is told are missing)
    O = _run_script()
    missing = {O.cfg_of(Pm=O.PM_NF[1]), O.rinv_points()[0]}

    def fake_ladder(c, root=None, log=None, compute=True, cap=None, nxs=None):
        assert compute is False
        if c in missing:
            raise (KeyError if c == O.rinv_points()[0] else RuntimeError)("missing")
        return dict(cfg=c, lam=1e-4 + 0j)
    with _patched(O, ladder=fake_ladder), checks() as c:
        nf = O.nf_results("r")
        c.check("nf: Pm -> record, missing skipped", sorted(nf) == sorted(set(O.PM_NF) - {O.PM_NF[1]})
                and all(nf[p]["cfg"][1] == p and nf[p]["cfg"][3] == "none" for p in nf))
        fa = O.fa_results("r")
        c.check("fa: (Pm, alpha) -> record", all(r["cfg"][1] == k[0] and r["cfg"][4] == k[1] for k, r in fa.items())
                and {(p, a) for p in O.PM_FA for a in O.ALPHA_FA} <= set(fa))
        ri = O.rinv_results("r")
        c.check("rinv: R -> record, Pm = 0, missing skipped", sorted(ri) == sorted(O.R_INV[1:])
                and all(r["cfg"][5] == R and r["cfg"][1] == 0.0 for R, r in ri.items()))
        rv = O.rvisc_results("r")
        c.check("rvisc: R -> record at Pm = 1", sorted(rv) == sorted(O.R_VISC)
                and all(r["cfg"][5] == R and r["cfg"][1] == 1.0 for R, r in rv.items()))
        sl = O.slope_results("r")
        c.check("slope: S -> record", sorted(sl) == sorted(O.S_SLOPE + (O.S0,)) and all(r["cfg"][0] == S for S, r in sl.items()))
        jt = O.jet_results("r")
        c.check("jet: (Pm, R) -> sech record", sorted(jt) == sorted(O.JET)
                and all(r["cfg"][1] == k[0] and r["cfg"][5] == k[1] and r["cfg"][3] == "sech" for k, r in jt.items()))
        bx = O.box_results("r")
        c.check("box: (Pm, flow, V, R, Lx) -> record", len(bx) == 2*len(O.BOX)
                and all(r["cfg"][1] == k[0] and r["cfg"][3:7] == k[1:] for k, r in bx.items()))
        ck = O.cmk_results("r")
        c.check("cmk: (ka, flow, V) -> record", len(ck) == 2*len(O.CMK)
                and all(r["cfg"][2] == k[0] and r["cfg"][3] == k[1] and r["cfg"][4] == k[2] for k, r in ck.items()))
        c.check("jet points: sech, V = 1", all(p[3] == "sech" and p[4] == 1.0 for p in O.jet_points()))
        c.check("cmk points: Pm = 0, ka and Lx from CMK, no flow + tanh alpha",
                [(p[2], p[6], p[3], p[4]) for p in O.cmk_points()]
                == [(ka, Lx, f, v) for ka, Lx, al in O.CMK for f, v in (("none", 0.0), ("tanh", al))]
                and all(p[1] == 0.0 for p in O.cmk_points()))
        c.check("rvisc/rinv points: tanh V = 1", all(p[3] == "tanh" and p[4] == 1.0 for p in O.rinv_points() + O.rvisc_points()))


@pytest.mark.fp64
def test_small_S_layer_relation_and_13c():
    # S = 1e4, Lx = 20 (pair coupling e^-5): the no-flow eigenvalue approaches the exact layer
    # relation from below as S grows; eq. 13c's (1 - g^2)^(1/3) is approached at Pm = 1 as S grows
    # while at Pm = 100 the eigen ratio is > 1 where 13c is < 1 (measured S = 1e3 .. 3e4)
    O = _run_script()
    root = tempfile.mkdtemp(prefix="taranis_ofman_smallS_")
    vroot, oroot = _theory_roots()
    with _patched(O, NXS=(1024, 2048, 4096), NX_SEED=512, DATA_ROOT_FULL=oroot), \
            _patched(O.V, DATA_ROOT_FULL=vroot), checks() as c:
        g = {}
        for S in (1e3, 1e4):
            for Pm in (0.0, 1.0, 100.0):
                g[(S, Pm, 0.0)] = O.gamma(O.ladder(O.cfg_of(S=S, Pm=Pm, Lx=20.0), root, _quiet))
            for Pm in (1.0, 100.0):
                g[(S, Pm, 0.6)] = O.gamma(O.ladder(O.cfg_of(S=S, Pm=Pm, flow="tanh", V_=0.6, Lx=20.0), root, _quiet))
        for Pm in (0.0, 1.0):
            r3, r4 = (g[(S, Pm, 0.0)]/O.gamma_layer(S, Pm) for S in (1e3, 1e4))
            c.check(f"Pm={Pm}: gamma/layer {r3:.4f} (S=1e3) -> {r4:.4f} (1e4), below 1", r3 < r4 < 1.0 and r4 > 0.9)
        e3, e4 = (g[(S, 1.0, 0.6)]/g[(S, 1.0, 0.0)] for S in (1e3, 1e4))
        th = O.ratio_13c(0.6, 1.0)
        c.check(f"Pm=1: gamma(0.6)/gamma(0) {e3:.4f} -> {e4:.4f} approaches 13c {th:.4f}",
                abs(e4 - th) < abs(e3 - th) and e4 < 1.0)
        for S in (1e3, 1e4):
            e = g[(S, 100.0, 0.6)]/g[(S, 100.0, 0.0)]
            c.check(f"Pm=100, S={S:.0e}: eigen ratio {e:.3f} > 1 > 13c {O.ratio_13c(0.6, 100.0):.3f} (wrong sign)",
                    e > 1.5 and O.ratio_13c(0.6, 100.0) < 0.7)


# ============================================================ (v) the cached headline numbers

HEADLINE = {       # the full campaign's numbers (2026-10-02), see the notebook's section 7
    "nf": {0.0: 0.00024817735734997625, 0.01: 0.00021319760395802358, 0.1: 0.00015926434631768758, 1.0: 0.00010963516465873797, 10.0: 7.391069396802931e-05, 100.0: 4.93714391986277e-05, 1000.0: 3.26397519628507e-05, 10000.0: 2.1248982511285716e-05},
    "fig7_fit": 0.17507836242089003,
    "fa": {(0.0, 0.2): 1.2174536436771681, (0.0, 0.4): 1.398157618978516, (0.0, 0.6): 1.4185560827511008, (0.0, 0.8): 1.2008892456840017, (1.0, 0.2): 0.9860432351592314, (1.0, 0.4): 0.9416945287843146, (1.0, 0.6): 0.8572616277283265, (1.0, 0.8): 0.7019704027755198, (100.0, 0.2): 1.3861123106996895, (100.0, 0.4): 2.378886011216597, (100.0, 0.6): 3.602909808788638, (100.0, 0.8): 4.444910686047552},
    "dense_tol": 1e-6,
}

def _full(O, fn):
    # the readers already skip uncached points; anything else they raise is a bug, not "no data"
    return fn(O.DATA_ROOT_FULL)


UNCONVERGED = {("fa", (0.0, 0.4)), ("fa", (0.0, 0.8)), ("rinv", 0.73), ("rinv", 0.9),
               ("slope", 1e6)}     # the notebook's "*" rows (the S0 slope point IS the R = 0.73 one)


@pytest.mark.fp64
def test_cached_headline_numbers():
    O = _run_script()
    if O.PROFILE != "full":
        pytest.skip("headline numbers are those of the full profile")
    if not os.path.isdir(os.path.join(O.DATA_ROOT_FULL, "solves")):
        pytest.skip("examples/data/tearing-ofman not generated (python examples/tearing_ofman_run.py)")
    # once the cache exists, a point that does not read back is a FAILURE: a reader, threshold or key
    # change that drops points must not turn this gate into a skip
    nf, fa = _full(O, O.nf_results), _full(O, O.fa_results)
    ri, slp, jt = _full(O, O.rinv_results), _full(O, O.slope_results), _full(O, O.jet_results)
    bx, ck, rv, dn = _full(O, O.box_results), _full(O, O.cmk_results), _full(O, O.rvisc_results), _full(O, O.dense_results)
    HEAD = HEADLINE
    with checks() as c:
        sizes = dict(nf=(len(nf), len(O.PM_NF)), fa=(len(fa), len(O.PM_FA)*(len(O.ALPHA_FA) + 1)),
                     rinv=(len(ri), len(O.R_INV)), rvisc=(len(rv), len(O.R_VISC)), slope=(len(slp), len(O.S_SLOPE) + 1),
                     jet=(len(jt), len(O.JET)), box=(len(bx), 2*len(O.BOX)), cmk=(len(ck), 2*len(O.CMK)),
                     dense=(len(dn), len(O.DENSE)))
        c.check(f"every campaign point reads back from the cache {sizes}", all(a == b for a, b in sizes.values()))
        unconv = ({("fa", k) for k, r in fa.items() if not r["converged"]}
                  | {("rinv", k) for k, r in ri.items() if not r["converged"]}
                  | {(n, k) for n, d in (("nf", nf), ("rvisc", rv), ("slope", slp), ("jet", jt), ("box", bx), ("cmk", ck))
                     for k, r in d.items() if not r["converged"]})
        c.check(f"the unconverged points are exactly the documented ones ({sorted(map(str, unconv))})",
                unconv == UNCONVERGED)
        for Pm, gref in HEAD["nf"].items():
            r = nf[Pm]
            c.check(f"fig 7 Pm={Pm:g}: gamma = {O.gamma(r):.6e} (recorded {gref:.6e}), conv {r['conv']:.1e}",
                    abs(O.gamma(r)/gref - 1) < 1e-6 and r["converged"])
        rl = [O.gamma(nf[p])/O.gamma_layer(O.S0, p) for p in O.PM_NF]
        c.check(f"eigen/layer in [{min(rl):.4f}, {max(rl):.4f}], decreasing with Pm",
                all(a > b for a, b in zip(rl[:-1], rl[1:])) and 0.9 < min(rl) and max(rl) < 1.0)
        pm = [p for p in O.PM_NF if 1e-1 <= p <= 1e4]
        sl = O.lsq_slope([O.S0/p for p in pm], [O.gamma(nf[p]) for p in pm])
        c.check(f"fit over S_v = 1e2..1e7: {sl:.4f} (recorded {HEAD['fig7_fit']:.4f}; theirs 0.168)",
                abs(sl - HEAD["fig7_fit"]) < 1e-4)
        sr = O.shear_ratios(fa)
        for (Pm, a), ref in HEAD["fa"].items():
            r = fa[(Pm, a)]
            c.check(f"Pm={Pm:g} alpha={a}: ratio {sr[(Pm, a)]:.4f} (recorded {ref:.4f}), conv {r['conv']:.1e}",
                    abs(sr[(Pm, a)] - ref) < 2e-4*ref)
        c.check("Pm = 1: 13c holds to <= 1.5% at every alpha",
                all(abs(sr[(1.0, a)]/O.ratio_13c(a, 1.0) - 1) < 0.015 for a in O.ALPHA_FA))
        c.check("Pm = 100: eigen ratio > 1 and increasing, 13c < 1 and decreasing",
                all(sr[(100.0, a)] > 1 > O.ratio_13c(a, 100.0) for a in O.ALPHA_FA)
                and all(sr[(100.0, a)] < sr[(100.0, b)] for a, b in zip(O.ALPHA_FA[:-1], O.ALPHA_FA[1:])))
        if (100.0, 0.6) in dn:
            d = dn[(100.0, 0.6)]
            dr = d[0.6].real/d[0.0].real
            c.check(f"independent dense route (nx {O.NX_INDEP}): Pm=100 ratio {dr:.4f} vs ladder "
                    f"{sr[(100.0, 0.6)]:.4f}", abs(dr/sr[(100.0, 0.6)] - 1) < HEAD["dense_tol"])
        k0, k1 = (100.0, "none", 0.0, 1.0), (100.0, "tanh", 0.6, 1.0)
        if all(k + (L,) in bx for k in (k0, k1) for L in (O.LX, O.BOX_LX)):
            rb = O.gamma(bx[k1 + (O.BOX_LX,)])/O.gamma(bx[k0 + (O.BOX_LX,)])
            c.check(f"second box Lx={O.BOX_LX:g}: Pm=100 ratio {rb:.5f} vs {sr[(100.0, 0.6)]:.5f}",
                    abs(rb/sr[(100.0, 0.6)] - 1) < 1e-3)
        if len(ri) == len(O.R_INV):
            c.check("inviscid R sweep: R = 0.73 is the grid maximum (near their peak at ~0.7)", max(ri, key=lambda R: O.gamma(ri[R])) == 0.73)
        if len(slp) == len(O.S_SLOPE) + 1:
            _, e = O.local_slopes(sorted(slp), [O.gamma(slp[S]) for S in sorted(slp)])
            c.check(f"fig 5a S-exponents {e} (recorded -0.5002, -0.5245)",
                    abs(e[0] + 0.5002) < 2e-4 and abs(e[1] + 0.5245) < 2e-4)
        if (0.0, 0.44) in jt:
            r = jt[(0.0, 0.44)]
            c.check(f"inviscid R = 0.44 jet solved in the sheet frame ({complex(r['lam']):.4e}, conv {r['conv']:.0e})",
                    r["converged"] and abs(complex(r["lam"]).imag) < 1e-5 and abs(O.gamma(r)/2.83147e-4 - 1) < 1e-5)
        if (1e4, 6.0) in jt:
            c.check("R = 6, S_v = 1e2: the oscillating mode dominates", complex(jt[(1e4, 6.0)]["lam"]).imag > 0.5)
        qa = [O.q_of(a, 1.0) for a in O.ALPHA_FA]
        _, e = O.local_slopes(qa[::-1], [O.gamma(fa[(1.0, a)]) for a in O.ALPHA_FA][::-1])
        c.check(f"Pm = 1 along R = 1: d ln gamma/d ln Q = {e} ~ 1/3 (13c)", all(0.33 < x < 0.36 for x in e))
        for (Pm, a) in ((0.0, 0.6), (0.0, 0.8)):
            c.check(f"HI point Pm={Pm} alpha={a} reached NX_HI", fa[(Pm, a)]["nx"] == O.NX_HI)
        # the fully independent FD route (one sheet, no box): the headline ratios, the R = 6 jet at S_v = 1e2
        fd = _full(O, O.fd_results)
        c.check(f"every FD point cached and converged in N ({ {k: f'{v[1]:.0e}' for k, v in fd.items()} })",
                len(fd) == len(O.FD) and all(v[1] < 2e-6 for v in fd.values()))
        rat = lambda a_, b_: fd[a_][0].real/fd[b_][0].real
        c.check(f"FD Pm = 100 ratio {rat('fa100', 'nf100'):.6f} vs ladder {sr[(100.0, 0.6)]:.6f}",
                abs(rat("fa100", "nf100")/sr[(100.0, 0.6)] - 1) < 5e-5)
        c.check(f"FD Pm = 1 ratio {rat('fa1', 'nf1'):.6f} vs ladder {sr[(1.0, 0.6)]:.6f}",
                abs(rat("fa1", "nf1")/sr[(1.0, 0.6)] - 1) < 5e-5)
        c.check(f"the wrong sign survives Ofman's printed nu W'''': FD d4 ratio {rat('fa100_d4', 'nf100_d4'):.4f}",
                abs(rat("fa100_d4", "nf100_d4")/rat("fa100", "nf100") - 1) < 5e-3)
        lo = complex(jt[(1e4, 6.0)]["lam"])
        c.check(f"S_v = 1e2 jet mode: FD (one sheet, L = 20) {fd['osc'][0]:.6e} vs ladder {lo:.6e}; L = 30 identical",
                abs(fd["osc"][0]/lo - 1) < 1e-3 and abs(fd["osc_L30"][0]/fd["osc"][0] - 1) < 1e-6)
        c.check(f"... and still growing with nu W'''' ({fd['osc_d4'][0]:.4e})", fd["osc_d4"][0].real > 0.02)
        c.check(f"S_v = 1e2: no growing eigenvalue near the KH branch in the FD route ({fd['kh1e2'][2]})",
                np.all(fd["kh1e2"][2].real < 0))
        osc = _full(O, lambda root: O.osc_scan(root, compute=False))
        c.check("R = 6 dense scan: the KH branch grows at S_v >= 1e3 and is stable at S_v = 3e2 and 1e2, where only "
                "the oscillating branch can grow",
                all(osc[p][1].real > 0 for p in (0.0, 100.0, 1e3)) and all(osc[p][1].real < 0 for p in (3e3, 1e4))
                and osc[3e3][0].real < 0 and osc[1e4][0].real > 0.02)


if __name__ == "__main__":
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
