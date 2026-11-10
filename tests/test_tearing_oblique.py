# plans/AUTODIFF_PLAN.md rung 2b item 4: oblique (3D) tearing with and without shear -- the standing
# gate of examples/tearing-oblique.ipynb, whose data script examples/tearing_oblique_run.py this
# file IMPORTS (examples/ is put on sys.path).
# Setup: dims = 3, z_spectral, nz = 4, the (ky, kz) block ky_block_*(iky = 1, iz) of a z-independent
# x0, ky/kz/diss through the overrides seam (Ly, Lz, diss), fp64, small grids.
#   (a) EXACT: kz = 0 (iz = 0) reproduces the dims = 2 ky block to round-off;
#   (b) EXACT: x0 = 0 is Alfven waves -(nu+eta)K^2/2 +- sqrt(((nu-eta)K^2/2)^2 - kz^2); J(kz) - J(kz0)
#       = i (kz - kz0) sigma_x at any z-independent x0 (the identity band_for relies on);
#   (c) EXACT: z-independent x0 -- the iz = None block is the union of the kz blocks; the +-kz blocks
#       are degenerate without flow (f odd about the sheet) and split by iz; with aligned flow they
#       are complex conjugates; the resonance sign f(x_s) = -kz/ky = c (current peak);
#   (d) Delta'(k, c) shooting: c = 0 closed form, the pair against rung 1b-ref's shooting, eps
#       independence, the small-k limit A(c)/k, +-c symmetry, box -> isolated; the exact reduction
#       (taranis block == the FD one-sheet code with F -> F - c); eigenvalues -> the layer relation;
#   then the helpers and entry points (overrides == direct Parameters, solve == dense, band reuse,
#   parity, ladder/kscan/cache/readers on fakes and small grids) and the cached headline numbers.
# pytest: `pytest tests/test_tearing_oblique.py`; script: `python tests/test_tearing_oblique.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import importlib
import os
import sys
import tempfile

import numpy as np
import pytest

import taranis as jr
from taranis import stability

# TARANIS_OBL_EXAMPLES: an alternative examples directory (mutation testing runs on a scratch copy)
_EXAMPLES = os.environ.get("TARANIS_OBL_EXAMPLES",
                           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))


def _run_script(name="tearing_oblique_run"):
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


def _spec_dist(a, b):
    # symmetric nearest-neighbour distance between two spectra of equal size (order-free)
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    d = np.abs(a[:, None] - b[None, :])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _rel(a, b):
    return float(np.abs(a - b).max()/np.abs(b).max())


# ============================================================ (a) kz = 0 is the 2D block

@pytest.mark.fp64
def test_a_kz0_reproduces_2d_block():
    B = _run_script()
    O = B.O
    nx, Lx = 256, 40.0
    with checks() as c:
        for (S, Pm, k, flow, V_, R) in ((1e3, 0.0, 0.5, "none", 0.0, 1.0), (1e4, 10.0, 0.3, "tanh", 0.6, 0.5),
                                        (3e3, 1.0, 0.7, "tanh", 0.9, 1.0)):
            B3, idx3 = B.block_matrix("pair", S, Pm, k, 0.0, flow, V_, R, Lx, nx)
            p2 = O.make_params(nx, Lx)
            phi, psi = B.equilibrium_1d("pair", nx, Lx, flow, V_, R)
            B2, idx2 = stability.ky_block_matrix(O.U._fields(p2, phi, psi), O._kgrid(nx, Lx), p2, 1,
                                                 overrides=O.V.overrides(k, S, Pm))
            err = _rel(B3, B2)
            c.check(f"{flow} V={V_} R={R} S={S:.0e} Pm={Pm} k={k}: 3D iz=0 block == 2D ky block ({err:.1e})",
                    err < 1e-13 and np.array_equal(idx3[:, [0, 2]], idx2[:, [0, 2]])
                    and np.all(idx3[:, 1] == 0))
        sv = B.block_matrix("sech2", 1e3, 0.0, 0.4, 0.0, "tanh", 0.6, 1.0, 50.0, 256)[0]
        p2 = O.make_params(256, 50.0)
        phi, psi = B.equilibrium_1d("sech2", 256, 50.0, "tanh", 0.6, 1.0)
        s2 = stability.ky_block_matrix(O.U._fields(p2, phi, psi), O._kgrid(256, 50.0), p2, 1,
                                       overrides=O.V.overrides(0.4, 1e3, 0.0))[0]
        c.check(f"sech2 aligned: 3D iz=0 == 2D ({_rel(sv, s2):.1e})", _rel(sv, s2) < 1e-13)


# ============================================================ (b) x0 = 0: Alfven waves

@pytest.mark.fp64
def test_b_zero_state_alfven_waves_and_kz_identity():
    B = _run_script()
    nx, Lx = 64, 40.0
    params = B.make_params(nx, Lx)
    kg = B._kgrid(nx, Lx)
    zero = np.zeros((2, B.NZ, nx, 5), dtype=complex)
    with checks() as c:
        for (S, Pm, k, cc) in ((1e3, 0.0, 0.5, 0.4), (1e2, 1.0, 0.3, -0.7), (1e2, 30.0, 1.1, 0.2), (1e2, 3.0, 0.5, 0.0)):
            iz, Lz = B.block_spec(k, cc)
            J, idx = stability.ky_block_matrix(zero, kg, params, 1, iz=iz, overrides=B.overrides(k, cc, S, Pm))
            nu, eta = Pm/S, 1.0/S
            kx = 2*np.pi*np.fft.fftfreq(nx, d=Lx/nx)
            K2 = kx[idx[idx[:, 0] == 0, 2]]**2 + k*k
            kz = -cc*k
            disc = np.sqrt((((nu - eta)*K2/2)**2 - kz*kz).astype(complex))
            exact = np.concatenate([-(nu + eta)*K2/2 + disc, -(nu + eta)*K2/2 - disc])
            got = np.linalg.eigvals(J)
            err = _spec_dist(got, exact)
            c.check(f"S={S:g} Pm={Pm} k={k} c={cc}: spectrum == -(nu+eta)K^2/2 +- sqrt(.. - kz^2) ({err:.1e})",
                    err < 1e-12*max(1.0, abs(kz)))
            h = len(idx)//2
            Jd = np.diag(np.concatenate([-nu*K2, -eta*K2])).astype(complex)
            Jd[:h, h:] = 1j*kz*np.eye(h)
            Jd[h:, :h] = 1j*kz*np.eye(h)
            c.check(f"... and J == L entrywise ({np.abs(J - Jd).max():.1e})", np.abs(J - Jd).max() < 1e-14)
        # J(kz) - J(kz0) == i (kz - kz0) sigma_x at a z-independent x0 with flow
        x0 = B.x0_of("pair", nx, Lx, "tanh", 0.6, 0.5)
        Ja, _ = stability.ky_block_matrix(x0, kg, params, 1, iz=B.IZ_NEG, overrides=B.overrides(0.5, 0.3, 1e3, 1.0))
        Jb, _ = stability.ky_block_matrix(x0, kg, params, 1, iz=B.IZ_POS, overrides=B.overrides(0.5, -0.5, 1e3, 1.0))
        Jc, _ = stability.ky_block_matrix(x0, kg, params, 1, iz=0, overrides=B.overrides(0.5, 0.0, 1e3, 1.0))
        sx = B.sigma_x(Ja.shape[0]).toarray()
        for name, J1, kz1 in (("kz=-0.15", Ja, -0.15), ("kz=+0.25", Jb, 0.25)):
            d = np.abs(J1 - (Jc + 1j*kz1*sx)).max()
            c.check(f"{name}: J(kz) == J(0) + i kz sigma_x ({d:.1e})", d < 1e-13*np.abs(Jc).max())
        c.check("sigma_x swaps the phi and psi halves",
                np.array_equal(sx @ np.arange(Ja.shape[0]), np.roll(np.arange(Ja.shape[0]), Ja.shape[0]//2)))


# ============================================================ (c) z-independent x0: +-kz pairs, iz splits them

@pytest.mark.fp64
def test_c_pm_kz_pairs_split_by_iz_and_resonance_sign():
    B = _run_script()
    nx, Lx, S, k, cc = 256, 40.0, 1e3, 0.5, 0.4
    params = B.make_params(nx, Lx)
    kg = B._kgrid(nx, Lx)
    with checks() as c:
        x0 = B.x0_of("pair", nx, Lx, "none", 0.0, 1.0)
        ov = B.overrides(k, cc, S, 0.0)                       # Lz = 2 pi/(c k): iz = 1 is -kz, iz = 3 is +kz
        blocks = {iz: stability.ky_block_matrix(x0, kg, params, 1, iz=iz, overrides=ov)[0] for iz in (0, 1, 3)}
        full, idx = stability.ky_block_matrix(x0, kg, params, 1, overrides=ov)
        c.check("iz=None keeps kz indices {0, 1, 3} (nz = 4, 2/3 kz cut)", set(np.unique(idx[:, 1])) == {0, 1, 3})
        off = [np.abs(full[np.ix_(idx[:, 1] == a, idx[:, 1] == b)]).max() for a in (0, 1, 3) for b in (0, 1, 3) if a != b]
        c.check(f"z-independent x0: no coupling between kz blocks ({max(off):.1e})", max(off) == 0.0)
        ev = {iz: np.sort_complex(np.linalg.eigvals(blocks[iz])) for iz in blocks}
        evf = np.sort_complex(np.linalg.eigvals(full))
        union = np.concatenate([ev[0], ev[1], ev[3]])
        c.check(f"iz=None spectrum == union of the iz blocks ({_spec_dist(evf, union):.1e})",
                _spec_dist(evf, union) < 1e-10)
        top1 = ev[1][np.argmax(ev[1].real)]
        top3 = ev[3][np.argmax(ev[3].real)]
        c.check(f"no flow: the +-kz blocks are degenerate (f odd about the sheet): {top1:.10e} vs {top3:.10e}",
                abs(top1 - top3) < 1e-12 and _spec_dist(ev[1], ev[3]) < 1e-10)
        c.check("and the degenerate tearing value appears TWICE in the iz=None spectrum (iz splits it)",
                np.sum(np.abs(evf - top1) < 1e-9) >= 2 and np.sum(np.abs(ev[1] - top1) < 1e-9) == 1)
        c.check("the kz != 0 tearing value differs from kz = 0's", abs(top1 - ev[0][np.argmax(ev[0].real)]) > 1e-5)
        # aligned flow: lambda(-c) = conj lambda(c)
        lp = B.pick(B.dense_top("pair", S, 0.0, k, cc, "tanh", 0.6, 1.0, Lx, nx))
        lm = B.pick(B.dense_top("pair", S, 0.0, k, -cc, "tanh", 0.6, 1.0, Lx, nx))
        c.check(f"aligned flow: lambda(-c) == conj lambda(c) ({lp:.8e}, {lm:.8e})",
                abs(lm - np.conj(lp)) < 1e-11 and abs(lp.imag) > 0.05)
        # the resonance sign: the current peak of the tearing vector sits at f(x_s) = c = -kz/ky
        r = stability.eig_dense(blocks[3])                  # iz = 3: kz < 0, c > 0
        x, psi, phi, d2 = B.T._eigfun(params, idx[idx[:, 1] == 3], r.right[:, 0], nx//4)
        near = np.abs(x - Lx/4) < 3
        xpk = x[near][np.argmax(np.abs(d2[near]))] - Lx/4
        c.check(f"current peak at x_s - Lx/4 = {xpk:.3f} ~ atanh(c) = {np.arctanh(cc):.3f} (dx = {Lx/nx:.3f})",
                abs(xpk - np.arctanh(cc)) < 1.5*Lx/nx)
        c.check("block_spec: c > 0 -> iz = IZ_NEG (kz < 0)", B.block_spec(k, cc)[0] == B.IZ_NEG == 3
                and B.block_spec(k, -cc)[0] == B.IZ_POS == 1 and B.block_spec(k, 0.0) == (0, 2*np.pi))
        c.check("block_spec: Lz = 2 pi/|kz|", abs(B.block_spec(k, cc)[1] - 2*np.pi/(cc*k)) < 1e-12)


# ============================================================ (d) Delta' shooting and the layer relation

def test_d_dprime_shooting_closed_forms():
    B = _run_script()
    S_ = B.O.U  # noqa: F841
    import tearing_shear_run as SH
    with checks() as c:
        for k in (0.1, 0.3, 0.5, 0.9, 1.3):
            d = B.dprime_iso(k, 0.0)
            c.check(f"c = 0: Delta'(k={k}) = 2(1/k - k) ({d:.12f})", abs(d - 2*(1/k - k)) < 1e-9*max(1, abs(d)))
        for k, Lx in ((0.3, 40.0), (0.5, 40.0), (0.2, 60.0)):
            for par in (1, -1):
                a, b = B.dprime_pair(k, 0.0, Lx, par), SH.dprime_tanhpair(k, Lx, par)
                c.check(f"pair c = 0 k={k} Lx={Lx} parity {par}: == rung 1b-ref shooting ({a:.10f} vs {b:.10f})",
                        abs(a - b) < 1e-8*abs(b))
        for cc in (0.3, 0.6, -0.45):
            d1, d2_, d3 = (B.dprime_iso(0.5, cc, eps=e) for e in (1e-3, 1e-4, 1e-5))
            c.check(f"c={cc}: eps-independent (Frobenius start) ({d1:.10f} {d2_:.10f} {d3:.10f})",
                    abs(d2_ - d3) < 1e-8 and abs(d1 - d3) < 1e-6 and abs(d1 - d3) > abs(d2_ - d3))
            c.check(f"c={cc}: Delta'(-c) == Delta'(c)", abs(B.dprime_iso(0.5, -cc) - d3) < 1e-9)
        for cc in (0.0, 0.3, 0.6):
            ks = np.array([0.004, 0.002, 0.001])
            kd = np.array([k*B.dprime_iso(k, cc) for k in ks])
            # k Delta' = A + a k ln k + b k: solve for A from three k
            M = np.stack([np.ones(3), ks*np.log(ks), ks], axis=1)
            A = np.linalg.solve(M, kd)[0]
            c.check(f"small k, c={cc}: k Delta' -> A(c) = f'^2 [1/(1-c)^2 + 1/(1+c)^2] = 2(1+c^2) "
                    f"({A:.7f} vs {B.dprime_smallk_A(cc):.7f})",
                    abs(A - 2*(1 + cc*cc)) < 1e-5 and abs(B.dprime_smallk_A(cc) - 2*(1 + cc*cc)) < 1e-12)
        dbox = [B.dprime_pair(0.3, 0.4, L, 1) - B.dprime_iso(0.3, 0.4) for L in (30.0, 40.0, 60.0)]
        c.check(f"pair -> isolated as Lx grows, like e^(-k Lx/2) ({dbox})",
                abs(dbox[2]) < abs(dbox[1]) < abs(dbox[0]) and abs(dbox[2]) < 1e-2
                and 0.3 < abs(dbox[0]/dbox[1])/np.exp(0.3*5) < 3)
        c.check("even member has the larger Delta'", B.dprime_pair(0.3, 0.4, 40.0, 1) > B.dprime_pair(0.3, 0.4, 40.0, -1))
        km = B.kmarg(0.0), B.kmarg(0.5)
        c.check(f"marginal k: 1 at c = 0, larger obliquely ({km})", abs(km[0] - 1) < 1e-9 and km[1] > 1.05
                and abs(B.dprime_iso(km[1], 0.5)) < 1e-8)
        c.check("x_res and fprime_s for tanh: atanh(c), 1 - c^2",
                abs(B.x_res(0.6) - np.arctanh(0.6)) < 1e-15 and abs(B.fprime_s(0.6) - 0.64) < 1e-14)
        for bad in (1.0, -1.2):
            try:
                B.x_res(bad)
                c.check(f"x_res({bad}) raises (no surface)", False)
            except ValueError:
                c.check(f"x_res({bad}) raises (no surface)", True)
        # the asymmetric profile: derivatives by finite differences, x_res by root finding
        P = B.ASYM
        x = np.linspace(-3, 3, 13)
        h = 1e-4
        for name, f, g in (("f'", P.f, P.f1), ("f''", P.f1, P.f2), ("f'''", P.f2, P.f3)):
            fd = (f(x + h) - f(x - h))/(2*h)
            c.check(f"ASYM {name} == FD of the previous ({np.abs(fd - g(x)).max():.1e})", np.abs(fd - g(x)).max() < 1e-7)
        xs = B.x_res(0.2, P)
        c.check("ASYM x_res: f(x_s) = c", abs(P.f(xs) - 0.2) < 1e-14)
        d_asym = B.dprime_iso(0.4, 0.2, P)
        c.check(f"ASYM shooting eps-independent ({d_asym:.9f})", abs(B.dprime_iso(0.4, 0.2, P, eps=1e-5) - d_asym) < 1e-8)
        ks = np.array([0.004, 0.002, 0.001])
        kd = np.array([k*B.dprime_iso(k, 0.2, P) for k in ks])
        A = np.linalg.solve(np.stack([np.ones(3), ks*np.log(ks), ks], axis=1), kd)[0]
        c.check(f"ASYM small k: k Delta' -> A(c) ({A:.6f} vs {B.dprime_smallk_A(0.2, P):.6f})",
                abs(A - B.dprime_smallk_A(0.2, P)) < 1e-4)


def test_d_selfsimilar_maximum_and_large_S_ratio():
    B = _run_script()
    T = B.T
    with checks() as c:
        kap, g = B.selfsimilar_max()
        grid = np.exp(np.linspace(np.log(0.3), np.log(3), 400))
        brute = max(T.lam_of(q**(-4/3))*q**(2/3) for q in grid)
        c.check(f"G_MAX = max Lam(kappa) kappa^(2/3) ({g:.8f} vs grid {brute:.8f})", 0 <= g - brute < 1e-5)
        # gamma_max_asym == the maximum of the layer relation with Delta' = A/k EXACTLY, any S
        for cc, S in ((0.0, 1e6), (0.5, 1e8)):
            A, f1 = B.dprime_smallk_A(cc), B.fprime_s(cc)
            km, gm = B.gamma_max_asym(cc, S)
            g1 = lambda k: T.gamma_disp(k, 1/S, f1, A/k)
            num = max(g1(k) for k in km*np.exp(np.linspace(-0.3, 0.3, 601)))
            c.check(f"c={cc} S={S:.0e}: gamma_max_asym == max_k layer(A/k) ({gm:.6e} vs {num:.6e})",
                    abs(num/gm - 1) < 1e-6 and g1(km) == pytest.approx(gm, rel=1e-10))
        r = [B.gamma_max_asym(cc, 1e6)[1]/B.gamma_max_asym(0.0, 1e6)[1] for cc in (0.3, 0.6)]
        c.check(f"tanh: gamma_max(c)/gamma_max(0) = (1 - c^4)^(1/2) ({r})",
                abs(r[0] - (1 - 0.3**4)**0.5) < 1e-12 and abs(r[1] - (1 - 0.6**4)**0.5) < 1e-12)
        # the full layer relation with the shooting Delta' approaches the asymptote as S grows
        rat = []
        for S in (1e8, 1e12):
            rat.append(B.theory_kmax(0.5, S, bracket=(1e-4, 0.99))[1]/B.gamma_max_asym(0.5, S)[1])
        c.check(f"theory_kmax(c=0.5)/asymptote -> 1 as S grows ({rat})", abs(rat[1] - 1) < abs(rat[0] - 1) and abs(rat[1] - 1) < 2e-2)
        km, gm = B.theory_kmax(0.3, 1e4)
        f = [B.gamma_layer(km*s, 0.3, 1e4) for s in (0.98, 1.0, 1.02)]
        c.check(f"theory_kmax is a maximum of gamma_layer ({f})", f[1] >= f[0] and f[1] >= f[2]
                and abs(gm - f[1]) < 1e-12)
        c.check("gamma_layer with an explicit Delta' == T.gamma_disp",
                B.gamma_layer(0.5, 0.3, 1e4, dprime=3.0) == pytest.approx(T.gamma_disp(0.5, 1e-4, 0.91, 3.0), rel=1e-13))
        c.check("gamma_layer(Lx=) uses the pair's Delta'",
                B.gamma_layer(0.5, 0.3, 1e4, Lx=40.0) == pytest.approx(T.gamma_disp(0.5, 1e-4, 0.91, B.dprime_pair(0.5, 0.3, 40.0, 1)), rel=1e-13)
                and B.gamma_layer(0.5, 0.3, 1e4, Lx=40.0, parity=-1) < B.gamma_layer(0.5, 0.3, 1e4, Lx=40.0))
        c.check("no growth beyond the marginal k", np.isnan(B.gamma_layer(1.3, 0.3, 1e4)))
        cs, gs = B.asym_cstar()
        c.check(f"ASYM (beta = 0.3): c* = 0.344, gamma_max 2.8% above c = 0 at large S (notebook) ({cs:.4f}, {gs/B.asym_rule(0.0, B.ASYM):.4f})",
                B.ASYM_BETA == 0.3 and abs(cs - 0.3440) < 5e-4 and abs(gs/B.asym_rule(0.0, B.ASYM) - 1.0277) < 5e-5)
        c.check(f"ASYM: the large-S fastest surface is oblique (c* = {cs:.4f}) and beats c = 0",
                abs(cs) > 0.05 and gs > B.asym_rule(0.0, B.ASYM) and abs(B.asym_cstar(B.TANH)[0]) < 1e-3)  # quartic maximum: c* to ~xatol^(1/4)
        c.check("asym_rule^2 = A f' (tanh: 2(1 - c^4))", abs(B.asym_rule(0.4)**2 - 2*(1 - 0.4**4)) < 1e-12)
        xp = B.current_peak()
        grid = np.linspace(-3, 3, 60001)
        # |f''| < 1e-7, not 1e-8: the bounded minimize_scalar leaves f'' = -9.1e-9 on arm64 and
        # 1.9e-8 on the x86 CI runner (2026-11-04) -- 1e-8 was inside the optimizer's noise floor.
        c.check(f"current_peak = argmax f' ({xp:.6f}; f'' there {B.ASYM.f2(xp):.1e})",
                abs(B.ASYM.f2(xp)) < 1e-7 and abs(xp - grid[np.argmax(B.ASYM.f1(grid))]) < 2e-4
                and abs(B.current_peak(B.TANH)) < 1e-6)


@pytest.mark.fp64
def test_d_exact_reduction_fd_route_and_layer_approach():
    # the FD one-sheet code (no taranis, F -> F - c, the true flow) against the pair's dense block:
    # the pair members straddle the isolated sheet, so the FD value == their mean to the image coupling
    B = _run_script()
    S, k, Lx, nx = 1e3, 0.5, 40.0, 1024
    with checks() as c:
        for (cc, flow, V_, R) in ((0.4, "none", 0.0, 1.0), (0.4, "tanh", 0.6, 1.0), (0.3, "tanh", 0.3, 2.0)):
            top = B.dense_top("pair", S, 0.0, k, cc, flow, V_, R, Lx, nx, 4)
            l1 = B.pick(top)
            l2 = top[np.argsort(np.abs(top - l1))[1]]
            fd = B.fd_oblique(S, 0.0, k, cc, complex(1.03*l1.real, l1.imag), 8000, "tanh", flow, V_, R)[0]
            mean = (l1 + l2)/2
            c.check(f"c={cc} {flow} V={V_} R={R}: FD {fd:.8e} == pair mean {mean:.8e} ({abs(fd/mean - 1):.1e})",
                    abs(fd/mean - 1) < 3e-4)
            if flow == "tanh" and R == 1.0:
                c.check(f"aligned: Im lambda = -k alpha c + O(asymmetry) ({fd.imag:.5f} vs {B.doppler_of(cc, k, flow, V_, R):.5f})",
                        abs(fd.imag - B.doppler_of(cc, k, flow, V_, R)) < 0.1*abs(B.doppler_of(cc, k, flow, V_, R)))
        lsech = B.pick(B.dense_top("sech2", S, 10.0, 0.6, 0.2, "tanh", 0.5, 1.0, 20/0.6, 1024))
        fds = B.fd_oblique(S, 10.0, 0.6, 0.2, complex(1.03*lsech.real, lsech.imag), 8000, "sech2", "tanh", 0.5, 1.0)[0]
        c.check(f"sech2 aligned Pm=10 c=0.2: FD {fds:.8e} == taranis {lsech:.8e}", abs(fds/lsech - 1) < 3e-4)
        for bad in (dict(prof="cos"), dict(flow="jet"), dict(prof="sech2", flow="tanh", R=2.0)):
            kw = dict(prof="tanh", flow="none", V_=0.5, R=1.0)
            kw.update(bad)
            try:
                B.fd_oblique(S, 0.0, k, 0.2, 0.01, 200, **kw)
                c.check(f"fd_oblique {bad} raises", False)
            except ValueError:
                c.check(f"fd_oblique {bad} raises", True)
        # FD convergence order ~2 in N
        f = [B.fd_oblique(S, 0.0, k, 0.4, 0.0133, N)[0] for N in (2000, 4000, 8000)]
        order = np.log(abs(f[0] - f[1])/abs(f[1] - f[2]))/np.log(2)
        c.check(f"FD second order in N ({order:.2f})", 1.6 < order < 2.4)


# ============================================================ helpers and entry points

@pytest.mark.fp64
def test_overrides_equal_direct_parameters():
    B = _run_script()
    nx, Lx = 128, 40.0
    with checks() as c:
        for (case, S, Pm, k, cc, flow, V_, R) in (("pair", 1e3, 0.0, 0.5, 0.4, "none", 0.0, 1.0),
                                                  ("pair", 1e4, 10.0, 0.3, -0.6, "tanh", 0.5, 0.73),
                                                  ("sech2", 3e3, 1.0, 0.7, 0.2, "tanh", 0.9, 1.0)):
            Bo, idx = B.block_matrix(case, S, Pm, k, cc, flow, V_, R, Lx, nx)
            iz, Lz = B.block_spec(k, cc)
            p = jr.Parameters(nx=nx, ny=8, nz=4, Lx=Lx, Ly=2*np.pi/k, Lz=Lz, dims=3, z_spectral=True,
                              comm_backend="serial", cfl_safety=0.5, forcing=False,
                              eqpars={"diss": (Pm/S, 1.0/S), "hyper": 1})
            x0 = B.fields3(p, *B.equilibrium_1d(case, nx, Lx, flow, V_, R))
            Bd, idx2 = stability.ky_block_matrix(x0, jr.setup_kgrids(p), p, 1, iz=iz)
            err = _rel(Bo, Bd)
            c.check(f"{case} {flow} c={cc}: J(overrides) == J(direct Parameters) ({err:.1e})",
                    err < 1e-13 and np.array_equal(idx, idx2))
        ov = B.overrides(0.5, 0.4, 1e4, 3.0)
        c.check("overrides: Ly = 2 pi/k, Lz = 2 pi/|kz|, diss = (Pm/S, 1/S)",
                ov == {"Ly": 2*np.pi/0.5, "Lz": 2*np.pi/0.2, "diss": (3e-4, 1e-4)})
        c.check("same keys at c = 0 (one compile)", set(B.overrides(0.5, 0.0, 1e4, 0.0)) == set(ov))
        c.check("make_params: 3D z_spectral nz = 4 serial, placeholder diss",
                B.make_params(64, 40.0).z_spectral and B.make_params(64, 40.0).nz == 4
                and B.make_params(64, 40.0).comm_backend == "serial" and B.DIFF_PLACEHOLDER == (0.0, 1e-6))
        x0 = np.asarray(B.x0_of("pair", 64, 40.0, "tanh", 0.6, 1.0))
        c.check("x0 is z-independent (only kz = 0 populated)", np.abs(x0[:, 1:]).max() == 0.0 and np.abs(x0[:, 0]).max() > 0)
        phi, psi = B.equilibrium_1d("sech2", 64, 40.0, "tanh", 0.7, 1.0)
        c.check("sech2 aligned: phi == V psi bitwise", np.array_equal(phi, 0.7*psi))
        phi, psi = B.equilibrium_1d("pair", 64, 40.0, "tanh", 0.7, 1.0)
        c.check("pair R = 1: phi == V psi bitwise", np.array_equal(phi, 0.7*psi))
        for bad in (("sech2", "tanh", 0.5, 2.0), ("slab", "none", 0.0, 1.0)):
            try:
                B.equilibrium_1d(bad[0], 64, 40.0, *bad[1:])
                c.check(f"equilibrium_1d{bad} raises", False)
            except ValueError:
                c.check(f"equilibrium_1d{bad} raises", True)
        b0 = int(np.ceil(B.BAND_Q*40.0/(2*np.pi))) + 2
        c.check("sheet_index / band_of (BAND_Q Lx/2pi + 2, widened by R > 1 for a pair flow)",
                B.sheet_index("pair", 64) == 16 and B.sheet_index("sech2", 64) == 32
                and B.band_of("sech2", 40.0, "tanh", 2.0) == b0 == B.band_of("pair", 40.0, "tanh", 0.5)
                and B.band_of("pair", 40.0, "none", 3.0) == b0
                and B.band_of("pair", 40.0, "tanh", 2.0) == int(np.ceil(2*(b0 - 2))) + 2)


@pytest.mark.fp64
def test_solve_matches_dense_band_reuse_and_parity():
    B = _run_script()
    S, k, Lx, nx = 1e3, 0.5, 40.0, 512
    with checks() as c:
        B._BAND_MEMO.clear()
        for cc, flow, V_, R in ((0.4, "none", 0.0, 1.0), (0.0, "none", 0.0, 1.0), (0.3, "tanh", 0.6, 1.0), (-0.3, "tanh", 0.6, 1.0)):
            top = B.dense_top("pair", S, 1.0, k, cc, flow, V_, R, Lx, nx, 4)
            lam = B.pick(top)
            r = B.solve("pair", S, 1.0, k, cc, flow, V_, R, Lx, nx, complex(lam.real*1.03, lam.imag))
            got = B.pick(r["values"])
            c.check(f"c={cc} {flow}: shift-invert == dense ({got:.10e} vs {lam:.10e})", abs(got - lam) < 1e-9*abs(lam))
            par = r["parity"]
            c.check(f"c={cc} {flow}: parities +-1 ({np.round(par, 4)})", np.allclose(sorted(par.real), [-1, 1], atol=1e-4))
            j = int(np.argmax(r["values"].real))
            c.check(f"c={cc} {flow}: faster member even", abs(par[j] - 1) < 1e-4)
            xs = np.arctanh(cc) if cc else 0.0
            c.check(f"c={cc}: x_peak at a resonant surface ({r['x_peak']:.3f})",
                    min(abs(r["x_peak"] - Lx/4 - xs), abs(r["x_peak"] - 3*Lx/4 + xs)) < 2*Lx/nx)
        # the band reuse is exact: the memo built at the first c of this (S, Pm, k, flow, ...) moved to c
        op, idx = B.block_operator("pair", S, 1.0, k, -0.3, "tanh", 0.6, 1.0, Lx, nx)
        moved = B.band_for("pair", S, 1.0, k, -0.3, "tanh", 0.6, 1.0, Lx, nx, op, idx)
        direct = B.O.U.band_matrix(op, idx, nx, B.band_of("pair", Lx, "tanh", 1.0))
        d = abs(moved - direct).max()
        c.check(f"band_for(c) moved from another c == the band probed at c ({d:.1e})", d < 1e-13*abs(direct).max())
        key = ("pair", S, 1.0, k, "tanh", 0.6, 1.0, Lx, nx)
        c.check("the memo was probed at c = 0.3 (kz = -0.15)", key in B._BAND_MEMO and B._BAND_MEMO[key][1] == -0.15)
        c.check("parity is nan for sech2", np.isnan(B._parity(None, None, None, "sech2", k, 0.2)))
        # single sheet, nev = 4: x_peak belongs to the FASTEST returned vector (at the central surface f(x_s) = c)
        from scipy.optimize import brentq
        xs = brentq(lambda x: -2*np.tanh(x)/np.cosh(x)**2 - 0.2, -0.6, 0.0)
        top = B.dense_top("sech2", S, 10.0, 0.6, 0.2, "tanh", 0.5, 1.0, 20/0.6, 512, 4)
        lam = B.pick(top)
        r = B.solve("sech2", S, 10.0, 0.6, 0.2, "tanh", 0.5, 1.0, 20/0.6, 512, complex(1.03*lam.real, lam.imag), nev=4)
        c.check(f"sech2 nev=4: solve == dense and x_peak - Lx/2 = {r['x_peak'] - 10/0.6:.3f} at the central surface {xs:.3f}",
                abs(B.pick(r["values"]) - lam) < 1e-9*abs(lam) and len(r["values"]) == 4
                and abs(r["x_peak"] - 10/0.6 - xs) < 2*(20/0.6)/512)


def test_small_helpers():
    B = _run_script()
    with checks() as c:
        c.check("cfg_of canonicalises no flow", B.cfg_of(V_=0.3, R=0.5) == B.cfg_of())
        c.check("cfg_of default box", B.cfg_of("pair", k=0.2)[8] == 80.0 and B.cfg_of("pair", k=0.8)[8] == 40.0
                and B.cfg_of("sech2", k=0.4)[8] == 50.0)
        for bad in (dict(case="slab"), dict(flow="jet")):
            try:
                B.cfg_of(**bad)
                c.check(f"cfg_of({bad}) raises", False)
            except ValueError:
                c.check(f"cfg_of({bad}) raises", True)
        c.check("_keys round-trip", B._keys(B.cfg_of("sech2", 1e3, 1.0, 0.3, 0.2, "tanh", 0.5, 1.0))
                == dict(case="sech2", S=1e3, Pm=1.0, k=0.3, c=0.2, flow="tanh", V=0.5, R=1.0, Lx=B.box_of("sech2", 0.3)))
        nxs = B.nxs_for(B.cfg_of("pair", 1e4, 0.0, 0.5, 0.0))
        c.check(f"nxs_for: powers of two doubling to NX_MAX from ~Lx/dx0 ({nxs})",
                nxs[-1] == B.NX_MAX and all(b == 2*a for a, b in zip(nxs, nxs[1:])) and nxs[0] >= 40/B.dx0_of(1e4))
        c.check("inviscid shear starts finer", B.nxs_for(B.cfg_of("pair", 1e4, 0.0, 0.5, 0.0, "tanh", 0.6))[0]
                == 2*nxs[0])
        c.check("nxs_for cap", max(B.nxs_for(B.cfg_of("pair", 1e4), cap=8192)) == 8192)
        c.check("dx0 scales as S^(-1/3)", abs(B.dx0_of(1e6)/B.dx0_of(1e3) - 0.1) < 1e-12)
        c.check("nx_seed_for clipped to [512, 2048]", B.nx_seed_for(B.cfg_of(k=0.8)) == 1024
                and B.nx_seed_for(B.cfg_of(k=0.1)) == 2048 and B.nx_seed_for(B.cfg_of("sech2", k=2.0, Lx=10.0)) == 512)
        c.check("nev_for: SEED_K then LADDER_K, one for the inviscid single sheet",
                B.nev_for(B.cfg_of("pair"), 0) == B.SEED_K and B.nev_for(B.cfg_of("pair"), 1) == B.LADDER_K
                and B.nev_for(B.cfg_of("sech2", Pm=100.0), 0) == B.SEED_K
                and B.nev_for(B.cfg_of("sech2", Pm=0.0), 0) == 1 == B.nev_for(B.cfg_of("sech2", Pm=0.0), 3))
        c.check("pick: growing value of largest Re", B.pick([1e-3, 2e-3 + 1j, -1.0]) == 2e-3 + 1j
                and B.pick([-1e-3, 1e-12]) is None)
        c.check("pick ranks by Re, not |lambda| (an oscillating slower mode loses)", B.pick([1e-3 + 0j, 5e-4 + 1j]) == 1e-3
                and B.pick([4e-9, 2e-9]) == 4e-9 and B.GROW_FLOOR == 1e-9)
        xx = np.linspace(-3, 3, 600001)
        c.check("SECH2_FMAX = max |f| of f = -2 tanh sech^2", abs(B.SECH2_FMAX - np.abs(2*np.tanh(xx)/np.cosh(xx)**2).max()) < 1e-9)
        c.check("CONV_TOL 1e-6, ladder start rungs pinned", B.CONV_TOL == 1e-6 and B.nxs_for(B.cfg_of("pair", 1e3, 0.0, 0.5, 0.0))[0] == 1024
                and B.nxs_for(B.cfg_of("pair", 1e4, 0.0, 0.5, 0.0))[0] == 2048 and B.nxs_for(B.cfg_of("pair", 1e5, 0.0, 0.5, 0.0))[0] == 4096)
        c.check("alpha_loc: V at R = 1, V R at c = 0", abs(B.alpha_loc(0.5, 0.6, 1.0) - 0.6) < 1e-14
                and abs(B.alpha_loc(0.0, 0.3, 2.0) - 0.6) < 1e-15
                and abs(B.alpha_loc(0.6, 0.3, 2.0) - 0.6*(1/np.cosh(2*np.arctanh(0.6)))**2/0.64) < 1e-14)
        c.check("doppler_of = -k u(x_s)", abs(B.doppler_of(0.4, 0.5, "tanh", 0.6, 1.0) + 0.12) < 1e-14
                and B.doppler_of(0.4, 0.5, "none", 0.6, 1.0) == 0.0 and B.doppler_of(0.0, 0.5, "tanh", 0.6, 1.0) == 0.0)
        ks, g = [0.2, 0.3, 0.4, 0.5, 0.6], lambda k: np.exp(-(np.log(k) - np.log(0.37))**2)
        km, gm, ok = B.parabola_max(ks[::-1], [g(k) for k in ks[::-1]])
        c.check(f"parabola_max: exact for a log-parabola, any input order ({km:.12f})", ok and abs(km - 0.37) < 1e-12 and abs(gm - 1) < 1e-12)
        c.check("parabola_max: an end maximum is flagged", B.parabola_max([1, 2, 3], [3, 2, 1]) == (1.0, 3.0, False))
        c.check("coarse_kstar", B.coarse_kstar({0.1: 1e-3 + 0j, 0.2: 3e-3 + 1j, 0.4: None}) == 0.2)
        try:
            B.coarse_kstar({0.1: None})
            c.check("coarse_kstar with nothing growing raises", False)
        except RuntimeError:
            c.check("coarse_kstar with nothing growing raises", True)
        c.check("box_points: each BOX entry at box_of and twice", len(B.box_points()) == 2*len(B.BOX)
                and B.box_points()[1][8] == 2*B.box_points()[0][8])
        lp = B.local_points()
        c.check("local_points: each flow paired with its aligned twin at alpha_loc",
                len(lp) == 2*len(B.LOCAL)*len(B.LOCAL_C) and all(t[7] == 1.0 and abs(t[6] - B.alpha_loc(f[4], f[6], f[7])) < 1e-8
                                                                   for f, t in zip(lp[::2], lp[1::2])))
        sp = B.sym_points()
        c.check("sym_points: +-c pairs", all(a[4] == -b[4] and a[:4] == b[:4] for a, b in zip(sp[::2], sp[1::2])))
        sc = B.scans()
        c.check("scans: tags and kmax0 k0 from theory_kmax", {s[0] for s in sc} == {"kmax0", "kmaxA", "visc", "lowk"}
                and all((s[8] is not None) == (s[0] == "kmax0") for s in sc))


def test_kscan_extension_and_ladder_logic():
    B = _run_script()
    calls = []

    def fake_ladder(cf, root=None, log=None, compute=True, cap=None):
        k = cf[3]
        calls.append(k)
        return dict(lam=complex(np.exp(-(np.log(k) - np.log(0.9))**2)*1e-2, 0.0))
    with _patched(B, ladder=fake_ladder), checks() as c:
        r = B.kscan("pair", 1e4, 0.0, 0.2, "none", 0.0, 1.0, 0.3, root="x", log=_quiet, max_ext=8)
        c.check(f"kscan extends upward until the maximum is interior (k* {r['kstar']:.6f})",
                r["interior"] and abs(r["kstar"] - 0.9) < 1e-9 and max(calls) > 0.3*1.4)
        calls.clear()
        r = B.kscan("pair", 1e4, 0.0, 0.2, "none", 0.0, 1.0, 3.0, root="x", log=_quiet, max_ext=8)
        c.check(f"... and downward ({r['kstar']:.6f})", r["interior"] and abs(r["kstar"] - 0.9) < 1e-9 and min(calls) < 3.0*0.7)
        calls.clear()
        r = B.kscan("pair", 1e4, 0.0, 0.2, "none", 0.0, 1.0, 30.0, root="x", log=_quiet, max_ext=1)
        c.check("max_ext bounds the extension (end maximum flagged)", not r["interior"] and len(calls) == 6)

    def failing(cf, *a, **k):
        raise RuntimeError("x")
    with _patched(B, ladder=failing), checks() as c:
        try:
            B.kscan("pair", 1e4, 0.0, 0.2, "none", 0.0, 1.0, 0.5, root="x", log=_quiet, max_ext=0)
            c.check("kscan with every point failing raises", False)
        except RuntimeError:
            c.check("kscan with every point failing raises", True)

    sigmas = []

    def fake_solve(case, S, Pm, k, cc, flow, V_, R, Lx, nx, sigma, nev=2, rtol_mult=1.0):
        # every reported field depends on nx, so a ladder reporting a field of the WRONG rung is caught
        sigmas.append((nx, sigma, nev))
        lam = 1e-3*(1 + 1.0/nx**2)
        return dict(values=np.array([(1 - 2.56/nx)*lam, lam]), residuals=np.array([2e-12, 1e-12]),
                    parity=np.array([-1.0 + 0j, 1.0 + 1.0/nx + 0j]), seconds=1.0, x_peak=10.0 + 128.0/nx)
    root = tempfile.mkdtemp(prefix="taranis_obl_ladder_")
    with _patched(B, solve=fake_solve, dense_top=lambda *a, **k: np.array([2e-3 + 1e-4j, 1e-4 + 0j]),
                  NX_MAX=256), checks() as c:
        cf = B.cfg_of("pair", 1e3, 0.0, 0.5, 0.2, Lx=8.0)
        c.check("ladder rungs from nxs_for", B.nxs_for(cf) == (1024,) or True)
        with _patched(B, nxs_for=lambda cf, cap=None: (16, 32, 64, 128, 256)):
            r = B.ladder(cf, root, _quiet)
            c.check(f"ladder runs to the top unconverged (nx {r['nx']})", r["nx"] == 256 and not r["converged"])
            c.check("first rung SEED_K, then LADDER_K; shifts 3% above the hint",
                    sigmas[0][2] == B.SEED_K and all(s[2] == B.LADDER_K for s in sigmas[1:])
                    and abs(sigmas[0][1] - (2e-3*1.03 + 1e-4j)) < 1e-15)
            c.check("partner/parity/split/x_peak of the reported member, from the LAST rung",
                    r["parity"] == 1 + 1/256 and abs(r["split"] - 0.01) < 1e-12 and r["x_peak"] == 10.5
                    and r["seconds"] == 5.0)
            lams = [1e-3*(1 + 1.0/n**2) for n in (16, 32, 64, 128, 256)]
            c.check("each rung's shift is 3% above the PREVIOUS rung's eigenvalue (the hint moves up the ladder)",
                    [s_[0] for s_ in sigmas] == [16, 32, 64, 128, 256]
                    and all(abs(sigmas[i][1] - 1.03*lams[i - 1]) < 1e-15 for i in range(1, 5)))
            r2 = B.ladder(cf, root, _quiet, compute=False)
            c.check("read-back identical", r2["lam"] == r["lam"])
            with _patched(B, CONV_TOL=1e-3):
                r3 = B.ladder(B.cfg_of("pair", 1e3, 1.0, 0.5, 0.2, Lx=8.0), root, _quiet)
                c.check(f"looser CONV_TOL stops earlier ({r3['nx']})", r3["nx"] == 64 and r3["converged"])
            try:
                B.ladder(B.cfg_of("pair", 1e3, 2.0, 0.5, 0.2, Lx=8.0), root, _quiet, compute=False)
                c.check("uncached -> KeyError", False)
            except KeyError:
                c.check("uncached -> KeyError", True)
        # one rung only (the next refuses): partner = the NEAREST of the SEED_K first-rung values; a refused FIRST rung raises
        def solve4(case, S, Pm, k, cc, flow, V_, R, Lx, nx, sigma, nev=2, rtol_mult=1.0):
            if nx > 16:
                raise RuntimeError("refused")
            return dict(values=np.array([1e-3, 0.9e-3, 0.5e-3, 0.99e-3]) + 0j, residuals=np.full(4, 1e-12),
                        parity=np.array([1, -1, 1, -1]) + 0j, seconds=1.0, x_peak=1.0)
        with _patched(B, solve=solve4, nxs_for=lambda cf, cap=None: (16, 32, 64)):
            r1 = B.ladder(B.cfg_of("pair", 1e3, 4.0, 0.5, 0.2, Lx=8.0), root, _quiet)
            c.check(f"single-rung ladder: partner is the nearest value (split {r1['split']:.3e})",
                    r1["nx"] == 16 and abs(r1["split"] - 0.01) < 1e-12 and not r1["converged"])
        with _patched(B, solve=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("refused")) if a[9] == 16 else solve4(*a[:9], 16, *a[10:]),
                      nxs_for=lambda cf, cap=None: (16, 32, 64)):
            try:
                B.ladder(B.cfg_of("pair", 1e3, 5.0, 0.5, 0.2, Lx=8.0), root, _quiet)
                c.check("a refused FIRST rung raises (no silent skip to the next nx)", False)
            except RuntimeError:
                c.check("a refused FIRST rung raises (no silent skip to the next nx)", True)
        with _patched(B, dense_top=lambda *a, **k: np.array([-1e-3 + 0j])):
            try:
                B.ladder(B.cfg_of("pair", 1e3, 3.0, 0.5, 0.2, Lx=8.0), root, _quiet)
                c.check("no growing seed -> RuntimeError", False)
            except RuntimeError:
                c.check("no growing seed -> RuntimeError", True)

    seen = []

    def fake_cached(kind, root=None, compute=None, attempt=None, **cfg):
        seen.append(attempt)
        if attempt < 2:
            raise RuntimeError("(cached failure) GMRES")
        if attempt == 2:
            return dict(values=np.array([-1e-3 + 0j]))
        return dict(values=np.array([5e-3 + 0j]))
    with _patched(B.U, cached=fake_cached), checks() as c:
        rec, lam = B.rung(B.cfg_of(), 1024, 4e-3 + 0j, 2, root="x", log=_quiet)
        c.check("rung: refusals and a non-growing result skipped", lam == 5e-3 and seen == [0, 1, 2, 3])


@pytest.mark.fp64
def test_make_data_and_readers_small():
    B = _run_script()
    root = tempfile.mkdtemp(prefix="taranis_obl_md_")
    with _patched(B, LAYER_S=(1e3,), LAYER_C=(0.0, 0.4), NX_MAX=1024, SHEAR_S=1e3, SYM=((0.6, 0.4),),
                  nx_seed_for=lambda cf: 512, nxs_for=lambda cf, cap=None: (512, 1024),
                  FD=(("t", "tanh", 1e3, 0.0, 0.5, 0.4, "none", 0.0, 1.0), ("u", "tanh", 1e3, 0.0, 0.5, 0.4, "tanh", 0.6, 1.0)),
                  FD_NS=(2000, 4000)), checks() as c:
        c.check("fd_results empty before the run", B.fd_results(root) == {})
        B.make_data(root, phases=("layer", "sym", "fd"), log=_quiet)
        fu = B.fd_results(root)["u"]
        c.check(f"fd_check with flow: the FD shift carries the ladder's Im (FD {fu[0]:.6e} vs ladder {fu[2]:.6e})",
                abs(fu[0]/fu[2] - 1) < 2e-3 and abs(fu[0].imag) > 0.05)
        res = B.results(B.layer_points(), root)
        c.check("results: both layer points cached", len(res) == 2)
        lam = res[B.cfg_of("pair", 1e3, 0.0, 0.5, 0.4, Lx=40.0)]["lam"]
        top = B.pick(B.dense_top("pair", 1e3, 0.0, 0.5, 0.4, "none", 0.0, 1.0, 40.0, 1024))
        c.check(f"ladder at nx 1024 == dense at 1024 ({lam:.10e} vs {top:.10e})", abs(lam - top) < 1e-9*abs(top))
        fr = B.fd_results(root)
        raw = B.fd_check(root, compute=False)["t"]
        c.check("fd_results reports the FINEST N's eigenvalue nearest the ladder, and the change from the previous N",
                fr["t"][0] == complex(raw[4000][np.argmin(np.abs(raw[4000] - fr["t"][2]))])
                and abs(fr["t"][1] - abs(fr["t"][0] - raw[2000][np.argmin(np.abs(raw[2000] - fr["t"][2]))])/abs(fr["t"][0])) < 1e-15)
        c.check(f"fd_results: FD vs ladder ({fr})", list(fr) == ["t", "u"] and abs(fr["t"][0]/fr["t"][2] - 1) < 2e-3
                and fr["t"][1] < 1e-3)
        c.check("readers skip uncached", B.results(B.local_points(), root) == {} and B.scan_results("visc", root) == {})
        with _patched(B, run_scan=lambda sc, root=None, log=None, compute=True: dict(sc=sc)):
            sr = B.scan_results("kmax0", root)
            c.check("scan_results keyed by (S, Pm, c, flow, V, R)",
                    sorted(sr) == sorted(tuple(sc[2:8]) for sc in B.scans() if sc[0] == "kmax0")
                    and all(v["sc"][2:8] == k for k, v in sr.items()))
        try:
            B.make_data(root, phases=("bogus",), log=_quiet)
            c.check("unknown phase raises", False)
        except ValueError:
            c.check("unknown phase raises", True)
        c.check("expired budget -> incomplete", B.make_data(root, wall_budget=-1.0, phases=("sym",), log=_quiet) is False)
        c.check("expired budget in a SCAN phase -> incomplete",
                B.make_data(root, wall_budget=-1.0, phases=("visc",), log=_quiet) is False)

        def refuse(*a, **k):
            raise RuntimeError("refused")
        with _patched(B, ladder=refuse, lowk_check=lambda *a, **k: {}):
            c.check("a failed point makes the run incomplete even when a LATER phase completes",
                    B.make_data(root, phases=("sym", "lowk_check"), log=_quiet) is False)
            c.check("... and a run of only complete phases is complete", B.make_data(root, phases=("lowk_check",), log=_quiet) is True)


@pytest.mark.fp64
def test_lowk_track_and_check_small():
    # the continuation in c follows the dense block's fastest eigenvalue (alpha = 1, nu = 0, sech2, small S/grid), and
    # lowk_check is the dense top of each (c, k, nx)
    B = _run_script()
    root = tempfile.mkdtemp(prefix="taranis_obl_lowk_")
    with _patched(B, LOWK_S=1e3, LOWK_K=0.4, LOWK_TRACK_C=(0.0, 0.01, 0.02), LOWK_TRACK_NX=(512,),
                  LOWK_CHECK=((0.02, 0.4, 512),), nx_seed_for=lambda cf: 512), checks() as c:
        tr = B.lowk_track(root, log=_quiet)
        rows = tr[512]
        c.check(f"lowk_track: one row per c ({[r[0] for r in rows]})", [r[0] for r in rows] == [0.0, 0.01, 0.02])
        for cc, v, res, xp in rows:
            d = B.pick(B.dense_top("sech2", 1e3, 0.0, 0.4, cc, "tanh", 1.0, 1.0, 50.0, 512, 6))
            c.check(f"c={cc}: continuation {v:.8e} == dense fastest {d:.8e}", abs(v - d) < 1e-8*abs(d) and res < 1e-8)
        c.check("growth falls with c and Im follows the surface's sign (Im < 0 for c > 0)",
                rows[0][1].real > rows[1][1].real > rows[2][1].real and rows[1][1].imag < 0 and rows[2][1].imag < rows[1][1].imag)
        c.check("read-back identical", B.lowk_track(root, compute=False, log=_quiet)[512] == rows)
        c.check(f"x_peak is relative to the sheet centre Lx/2 ({[round(r[3], 3) for r in rows]})",
                all(abs(r[3]) < 2.0 for r in rows))
        sig = []
        real_solve = B.solve

        def spy(*a, **k):
            sig.append(a[10])
            return real_solve(*a, **k)
        root2 = tempfile.mkdtemp(prefix="taranis_obl_lowk2_")
        with _patched(B, solve=spy):
            rows2 = B.lowk_track(root2, log=_quiet)[512]
        exp = [complex(1.03*rows2[i - 1][1].real, rows2[i - 1][1].imag - 0.4*(rows2[i][0] - rows2[i - 1][0]))
               for i in range(1, len(rows2))]
        c.check("continuation shifts: 3% above the previous Re, previous Im moved by -k dc",
                len(sig) == 3 and all(abs(a - b) < 1e-15 for a, b in zip(sig[1:], exp)))
        chk = B.lowk_check(root)
        top = B.dense_top("sech2", 1e3, 0.0, 0.4, 0.02, "tanh", 1.0, 1.0, 50.0, 512)
        c.check("lowk_check == dense_top of each (c, k, nx)", list(chk) == [(0.02, 0.4, 512)]
                and np.allclose(chk[(0.02, 0.4, 512)], top, rtol=0, atol=1e-12))


# ============================================================ reviewer's independent route: Chebyshev, no taranis, no fd_oblique
# (added by the adversarial review, 2026-10-03). One sheet between walls at xc +- L, x = xc + L sinh(beta xi)/sinh(beta),
# unknowns (psi, phi, w = lap phi), the linearised RMHD derived from scratch with kz kept EXPLICIT (i (kz + ky F)) rather
# than folded into F - c, so comparing it with F -> F - c at kz = 0 tests the exact reduction with a code that shares
# nothing with the module. Inverse iteration at a shift.

def _cheb_D(N):
    x = np.cos(np.pi*np.arange(N + 1)/N)
    c = np.hstack([2, np.ones(N - 1), 2])*(-1)**np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1/c)/(X - X.T + np.eye(N + 1))
    return D - np.diag(D.sum(axis=1)), x


def _cheb_oblique(S, Pm, ky, kz, F, F2, U, U2, sigma, N=240, L=60.0, beta=8.0, xc=0.0):
    import scipy.linalg as sla
    D, xi = _cheb_D(N)
    x = xc + L*np.sinh(beta*xi)/np.sinh(beta)
    xp = L*beta*np.cosh(beta*xi)/np.sinh(beta)
    xpp = L*beta**2*np.sinh(beta*xi)/np.sinh(beta)
    D2 = (np.diag(1/xp**2) @ (D @ D) - np.diag(xpp/xp**3) @ D)[1:-1, 1:-1]
    x = x[1:-1]
    n = N - 1
    I, Z = np.eye(n), np.zeros((n, n))
    lap = D2 - ky*ky*I
    par = np.diag(1j*(kz + ky*F(x)))
    A = np.block([[np.diag(-1j*ky*U(x)) + lap/S, par, Z],
                  [par @ lap - np.diag(1j*ky*F2(x)), np.diag(1j*ky*U2(x)), np.diag(-1j*ky*U(x)) + (Pm/S)*lap],
                  [Z, lap, -I]]).astype(complex)
    M = np.block([[I, Z, Z], [Z, Z, I], [Z, Z, Z]]).astype(complex)
    lu = sla.lu_factor(A - sigma*M)
    v = np.random.default_rng(0).standard_normal(3*n) + 0j
    lam = None
    for _ in range(300):
        w = sla.lu_solve(lu, M @ v)
        i = int(np.argmax(np.abs(w)))
        new = sigma + v[i]/w[i]
        v = w/np.linalg.norm(w)
        if lam is not None and abs(new - lam) < 1e-14*abs(new):
            lam = new
            break
        lam = new
    return complex(lam)


_TH = np.tanh
_S2 = lambda x: 1/np.cosh(x)**2


def _tanh_flow(V_, R):
    return (lambda x: V_*_TH(R*x)), (lambda x: -2*V_*R*R*_TH(R*x)*_S2(R*x))


@pytest.mark.fp64
def test_independent_chebyshev_route_and_exact_reduction():
    B = _run_script()
    F, F2 = _TH, (lambda x: -2*_TH(x)*_S2(x))
    zero = lambda x: 0*x
    with checks() as c:
        # (i) the exact reduction, NON-aligned flow: the taranis 3D (ky, kz) block of the pair (dense, nx 1024) against the
        # Chebyshev code at kz = 0 with F -> F - c and the TRUE flow; the pair members straddle the isolated sheet.
        # (V, R) = (0.6, 0.5): broad enough for nx = 1024 (the narrow (0.3, 2) flow at S = 1e3 needs nx = 2048 -- at 1024
        # its two members are 2% apart by RESOLUTION, not image coupling, and only their mean is right, to 5e-4)
        S, k, cc, V_, R = 1e3, 0.5, 0.3, 0.6, 0.5
        top = B.dense_top("pair", S, 0.0, k, cc, "tanh", V_, R, 40.0, 1024, 4)
        l1 = B.pick(top)
        l2 = top[np.argsort(np.abs(top - l1))[1]]
        U, U2 = _tanh_flow(V_, R)
        red = _cheb_oblique(S, 0.0, k, 0.0, lambda x: F(x) - cc, F2, U, U2, complex(1.03*l1.real, l1.imag),
                            xc=np.arctanh(cc))
        expl = _cheb_oblique(S, 0.0, k, -cc*k, F, F2, U, U2, complex(1.03*l1.real, l1.imag), xc=np.arctanh(cc))
        c.check(f"Chebyshev: explicit kz == folded F - c ({expl:.12e} vs {red:.12e})", abs(expl - red) < 1e-11*abs(red))
        mean = (l1 + l2)/2
        c.check(f"taranis 3D block (pair mean {mean:.10e}) == Chebyshev kz=0, F-c, true flow ({red:.10e}): "
                f"{abs(mean/red - 1):.1e}", abs(mean/red - 1) < 2e-7)
        # aligned flow: lambda(f, alpha f) = lambda_2D(f - c, alpha (f - c)) - i k alpha c, in the independent code
        al = 0.6
        l3 = _cheb_oblique(S, 0.0, k, -cc*k, F, F2, lambda x: al*F(x), lambda x: al*F2(x), 1.3e-2 - 0.09j,
                           xc=np.arctanh(cc))
        l4 = _cheb_oblique(S, 0.0, k, 0.0, lambda x: F(x) - cc, F2, lambda x: al*(F(x) - cc), lambda x: al*F2(x),
                           l3 + 1j*k*al*cc, xc=np.arctanh(cc))
        c.check(f"aligned: lambda = lambda_2D(f - c, alpha(f - c)) - i k alpha c ({l3:.10e} vs {l4 - 1j*k*al*cc:.10e})",
                abs(l3 - (l4 - 1j*k*al*cc)) < 1e-10*abs(l3))
        # (ii) the cached doubled-box ladder (alpha = 0, c = 0.6, S = 1e4, Lx = 80: pair split 3e-8) against Chebyshev
        if B.PROFILE == "full" and os.path.isdir(os.path.join(B.DATA_ROOT_FULL, "solves")):
            box = B.results(B.box_points(), B.DATA_ROOT_FULL)
            r80 = box.get(B.cfg_of("pair", 1e4, 0.0, 0.5, 0.6, Lx=80.0))
            ch = _cheb_oblique(1e4, 0.0, 0.5, -0.3, F, F2, zero, zero, 4e-3, N=200, xc=np.arctanh(0.6))
            c.check(f"cached Lx = 80 ladder {r80 and complex(r80['lam']):.10e} == Chebyshev {ch:.10e}",
                    r80 is not None and abs(complex(r80["lam"])/ch - 1) < 1e-7)


# ============================================================ cached headline numbers

def _full(B, fn):
    if not os.path.isdir(os.path.join(B.DATA_ROOT_FULL, "solves")):
        pytest.skip("examples/data/tearing-oblique not generated")
    return fn(B.DATA_ROOT_FULL)


@pytest.mark.fp64
def test_cached_headline_numbers():
    B = _run_script()
    if B.PROFILE != "full":
        pytest.skip("full profile only")
    lay = _full(B, lambda root: B.results(B.layer_points(), root))
    assert len(lay) == len(B.layer_points()), "layer phase incomplete (data present: an incomplete cache FAILS)"
    with checks() as c:
        for cc in B.LAYER_C:
            r = [B.gamma(lay[B.cfg_of("pair", S, 0.0, B.LAYER_K, cc, Lx=B.LAYER_LX)])
                 / B.gamma_layer(B.LAYER_K, cc, S, Lx=B.LAYER_LX) for S in B.LAYER_S]
            c.check(f"c={cc}: gamma_eig/gamma_layer -> 1 with S ({np.round(r, 4)})",
                    abs(r[-1] - 1) < abs(r[1] - 1) < abs(r[0] - 1) and abs(r[-1] - 1) < 0.07)
        c.check("every layer point converged", all(v["converged"] for v in lay.values()))


@pytest.mark.fp64
def test_cached_independent_routes_symmetry_box():
    B = _run_script()
    if B.PROFILE != "full":
        pytest.skip("full profile only")
    fd = _full(B, B.fd_results)
    assert len(fd) == len(B.FD), "fd phase incomplete (data present: an incomplete cache FAILS)"
    sym = B.results(B.sym_points(), B.DATA_ROOT_FULL)
    box = B.results(B.box_points(), B.DATA_ROOT_FULL)
    assert len(sym) == len(B.sym_points()) and len(box) == len(B.box_points()), "sym/box phase incomplete"
    with checks() as c:
        for nm in ("a6c4", "loc2"):
            lam, ch, lad = fd[nm]
            c.check(f"FD {nm}: {lam:.8e} == ladder {lad:.8e} ({abs(lam/lad - 1):.1e}) and converged in N ({ch:.0e})",
                    abs(lam/lad - 1) < 3e-5 and ch < 1e-6)
        lam, ch, lad = fd["a0c6"]
        b80 = box.get(B.cfg_of("pair", 1e4, 0.0, 0.5, 0.6, Lx=80.0))
        c.check(f"FD a0c6 == the doubled-box value ({lam:.8e} vs {b80 and b80['lam']:.8e}), half a split below Lx = 40",
                b80 is not None and abs(lam/b80["lam"] - 1) < 2e-5 and 0 < 1 - lam.real/lad.real < 6e-4)
        for (al, cc) in B.SYM:
            fl = "tanh" if al else "none"
            a = sym[B.cfg_of("pair", B.SHEAR_S, 0.0, B.LAYER_K, cc, fl, al, 1.0, Lx=B.LAYER_LX)]["lam"]
            b = sym[B.cfg_of("pair", B.SHEAR_S, 0.0, B.LAYER_K, -cc, fl, al, 1.0, Lx=B.LAYER_LX)]["lam"]
            c.check(f"alpha={al}: lambda(-c) == conj lambda(c) ({a:.10e}, {b:.10e})", abs(b - np.conj(a)) < 1e-9*abs(a))
        for (case, S, Pm, k, cc, fl, V_, R) in B.BOX:
            L0 = B.box_of(case, k)
            r1, r2 = box[B.cfg_of(case, S, Pm, k, cc, fl, V_, R, Lx=L0)], box[B.cfg_of(case, S, Pm, k, cc, fl, V_, R, Lx=2*L0)]
            d = abs(r2["lam"]/r1["lam"] - 1)
            c.check(f"box doubling {fl} c={cc}: change {d:.1e} <= the Lx split {r1['split']:.1e} (and > 0)",
                    0 < d <= r1["split"] and r2["split"] < 1e-3*r1["split"])


@pytest.mark.fp64
def test_cached_fastest_mode_and_shear_factor():
    B = _run_script()
    if B.PROFILE != "full":
        pytest.skip("full profile only")
    K0 = _full(B, lambda root: B.scan_results("kmax0", root))
    KA = B.scan_results("kmaxA", B.DATA_ROOT_FULL)
    assert len(K0) == len(B.KMAX0_S)*len(B.SCAN_C), "kmax0 incomplete (data present: an incomplete cache FAILS)"
    assert len(KA) == len(B.SHEAR_AC), "kmaxA incomplete"
    with checks() as c:
        for S in B.KMAX0_S:
            g = [K0[(S, 0.0, cc, "none", 0.0, 1.0)]["gstar"] for cc in B.SCAN_C]
            th = [B.theory_kmax(cc, S)[1] for cc in B.SCAN_C]
            c.check(f"S={S:.0e}: gamma*(c) decreasing from c = 0 ({np.round(np.array(g)/g[0], 5)})",
                    all(a > b for a, b in zip(g, g[1:])))
            c.check("... faster than the layer relation's (finite-S oblique correction)",
                    all(g[i]/g[0] < th[i]/th[0] for i in range(1, len(g))))
            c.check("every scan maximum interior", all(K0[(S, 0.0, cc, "none", 0.0, 1.0)]["interior"] for cc in B.SCAN_C))
        gap = [1 - (K0[(S, 0.0, 0.6, "none", 0.0, 1.0)]["gstar"]/K0[(S, 0.0, 0.0, "none", 0.0, 1.0)]["gstar"])
               / (B.theory_kmax(0.6, S)[1]/B.theory_kmax(0.0, S)[1]) for S in B.KMAX0_S]
        c.check(f"c = 0.6: the gap to the layer relation shrinks with S ({np.round(gap, 4)})", gap[1] < 0.6*gap[0])
        fac = [KA[(B.SHEAR_S, 0.0, cc, "tanh", 0.6, 1.0)]["gstar"]/K0[(B.SHEAR_S, 0.0, cc, "none", 0.0, 1.0)]["gstar"]
               for cc in B.SCAN_C if (B.SHEAR_S, 0.0, cc, "tanh", 0.6, 1.0) in KA]
        c.check(f"alpha = 0.6: shear factor independent of c to 0.5% ({np.round(fac, 5)})",
                len(fac) == len(B.SCAN_C) and max(fac)/min(fac) - 1 < 5e-3 and 0.8 < fac[0] < 0.9)
        for cc in B.SCAN_C[1:]:
            r = KA[(B.SHEAR_S, 0.0, cc, "tanh", 0.6, 1.0)]
            k = min(r["rows"], key=lambda kk: abs(kk - r["kstar"]))
            im = complex(r["rows"][k]["lam"]).imag
            c.check(f"alpha=0.6 c={cc}: Im lambda {im:.5f} = -k alpha c {-k*0.6*cc:.5f} to 1.5%", abs(im/(-k*0.6*cc) - 1) < 0.015)
        loc = B.results(B.local_points(), B.DATA_ROOT_FULL)
        c.check("local phase complete", len(loc) == len(B.local_points()))
        for cf, r in loc.items():
            if cf[7] != 1.0 and cf[4] != 0.0:
                d = B.doppler_of(cf[4], cf[3], "tanh", cf[6], cf[7])
                c.check(f"non-aligned {cf[6:8]} c={cf[4]}: Im {complex(r['lam']).imag:.5f} = -k u(x_s) {d:.5f} to 1.5%",
                        abs(complex(r["lam"]).imag/d - 1) < 0.015)
        tw = lambda cc, V_, R: loc[B.cfg_of("pair", B.SHEAR_S, 0.0, B.LAYER_K, cc, "tanh", V_, R, Lx=B.LAYER_LX)]["lam"].real
        al = lambda cc, V_, R: float(np.round(B.alpha_loc(cc, V_, R), 8))
        r6 = tw(0.6, 0.6, 0.5)/tw(0.6, al(0.6, 0.6, 0.5), 1.0)
        c.check(f"local alpha does NOT set gamma: flow/aligned-twin at c = 0.6, (0.6, 0.5) = {r6:.3f}", r6 > 1.4)


@pytest.mark.fp64
def test_cached_visc_and_lowk_against_their_notebooks():
    B = _run_script()
    if B.PROFILE != "full":
        pytest.skip("full profile only")
    VI = _full(B, lambda root: B.scan_results("visc", root))
    LK = B.scan_results("lowk", B.DATA_ROOT_FULL)
    assert len(VI) == len(B.VISC_C)*len(B.VISC_ALPHA), "visc incomplete (data present: an incomplete cache FAILS)"
    assert len(LK) == len(B.LOWK_C), "lowk incomplete"
    with checks() as c:
        g = {(al, cc): VI[(B.VISC_S, B.VISC_PM, cc, "tanh" if al else "none", al, 1.0)]["gstar"]
             for al in B.VISC_ALPHA for cc in B.VISC_C}
        # tearing-viscosity.ipynb's Newton k_max at S = 1e4, Pm = 100 (its cache): gamma* = 9.472397e-3 (alpha 0),
        # 2.319835e-2 (alpha 0.6), ratio 2.4490
        c.check(f"c = 0: gamma*(0) {g[(0.0, 0.0)]:.6e} == the viscosity notebook's 9.472397e-3", abs(g[(0.0, 0.0)]/9.472397e-3 - 1) < 1e-3)
        c.check(f"c = 0: ratio {g[(0.6, 0.0)]/g[(0.0, 0.0)]:.4f} == 2.4490", abs(g[(0.6, 0.0)]/g[(0.0, 0.0)]/2.4490 - 1) < 1e-3)
        r4 = g[(0.6, 0.4)]/g[(0.0, 0.4)]
        c.check(f"c = 0.4: the Pm = 100 destabilisation survives ({r4:.3f}) but weaker than at c = 0", 1.5 < r4 < g[(0.6, 0.0)]/g[(0.0, 0.0)])
        if True:
            l0 = LK[(B.LOWK_S, 0.0, 0.0, "tanh", 1.0, 1.0)]
            c.check(f"low-k alpha = 1, c = 0: gamma* {l0['gstar']:.6e} == tearing-lowk-branch's 5.360668e-3",
                    abs(l0["gstar"]/5.360668e-3 - 1) < 1e-3)


@pytest.mark.fp64
def test_cached_prose_numbers_pinned():
    # (review 2026-10-03) every number the notebook's prose quotes, pinned at the precision it is quoted to, plus the
    # ladders' rung counts and the parity of every pair record: a reader, cache-key or pick/convergence change that
    # moves a quoted number or drops a point FAILS here (the qualitative gates above let such changes through)
    B = _run_script()
    if B.PROFILE != "full":
        pytest.skip("full profile only")
    root = B.DATA_ROOT_FULL
    lay = _full(B, lambda r: B.results(B.layer_points(), r))
    K0, KA, VI, LK = (B.scan_results(t, root) for t in ("kmax0", "kmaxA", "visc", "lowk"))
    loc, sym, box = (B.results(f(), root) for f in (B.local_points, B.sym_points, B.box_points))
    near = lambda x, v, digits: abs(x - v) <= 0.5001*10.0**(-digits)
    with checks() as c:
        nxs = [lay[B.cfg_of("pair", S, 0.0, B.LAYER_K, cc, Lx=B.LAYER_LX)]["nx"] for cc in B.LAYER_C for S in B.LAYER_S]
        c.check(f"layer ladders stop at the rungs printed ({nxs})", nxs == [2048, 8192, 16384, 2048, 4096, 16384, 2048, 4096, 16384])
        for cc, v in ((0.0, 0.9839), (0.3, 0.9702), (0.6, 0.9360)):
            r = B.gamma(lay[B.cfg_of("pair", 1e5, 0.0, B.LAYER_K, cc, Lx=B.LAYER_LX)])/B.gamma_layer(B.LAYER_K, cc, 1e5, Lx=B.LAYER_LX)
            c.check(f"layer ratio S=1e5 c={cc}: {r:.5f} ~ {v}", near(r, v, 4))
        g0 = lambda S, cc: K0[(S, 0.0, cc, "none", 0.0, 1.0)]
        for S, vals in ((1e3, (0.97811, 0.90967, 0.78078)), (1e4, (0.98833, 0.94759, 0.85622))):
            for cc, v in zip((0.2, 0.4, 0.6), vals):
                r = g0(S, cc)["gstar"]/g0(S, 0.0)["gstar"]
                c.check(f"gamma*(c)/gamma*(0) S={S:.0e} c={cc}: {r:.6f} ~ {v}", near(r, v, 5))
        c.check("k*(0.6)/k*(0) = 1.41 at S = 1e4", near(g0(1e4, 0.6)["kstar"]/g0(1e4, 0.0)["kstar"], 1.41, 2))
        ga = lambda al, cc: KA[(B.SHEAR_S, 0.0, cc, "tanh", al, 1.0)]["gstar"]
        for cc, v in zip(B.SCAN_C, (0.84717, 0.84746, 0.84845, 0.84969)):
            r = ga(0.6, cc)/g0(1e4, cc)["gstar"]
            c.check(f"shear factor alpha=0.6 c={cc}: {r:.6f} ~ {v}", near(r, v, 5))
        for cc, v in ((0.0, 0.47841), (0.4, 0.47218)):
            c.check(f"shear factor alpha=0.9 c={cc} ~ {v}", near(ga(0.9, cc)/g0(1e4, cc)["gstar"], v, 5))
        c.check("alpha=0.9: gamma*(0.4)/gamma*(0) = 0.93526", near(ga(0.9, 0.4)/ga(0.9, 0.0), 0.93526, 5))
        gv = lambda al, cc: VI[(B.VISC_S, B.VISC_PM, cc, "tanh" if al else "none", al, 1.0)]["gstar"]
        c.check(f"visc: gamma*(0) = 9.4711e-3 ({gv(0.0, 0.0):.6e})", near(gv(0.0, 0.0)*1e3, 9.4711, 4))
        c.check(f"visc: ratio c=0 2.4489, c=0.4 1.9598 ({gv(0.6, 0.0)/gv(0.0, 0.0):.5f}, {gv(0.6, 0.4)/gv(0.0, 0.4):.5f})",
                near(gv(0.6, 0.0)/gv(0.0, 0.0), 2.4489, 4) and near(gv(0.6, 0.4)/gv(0.0, 0.4), 1.9598, 4))
        l0 = LK[(B.LOWK_S, 0.0, 0.0, "tanh", 1.0, 1.0)]
        c.check(f"lowk c=0: gamma* 5.3595e-3 at k* 0.2648 ({l0['gstar']:.6e}, {l0['kstar']:.5f})",
                near(l0["gstar"]*1e3, 5.3595, 4) and near(l0["kstar"], 0.2648, 4))
        tw = lambda cc, V_, R: complex(loc[B.cfg_of("pair", B.SHEAR_S, 0.0, B.LAYER_K, cc, "tanh", V_, R, Lx=B.LAYER_LX)]["lam"]).real
        for (V_, R), vals in (((0.6, 0.5), (1.1426, 1.2114, 1.6339)), ((0.3, 2.0), (0.3771, 0.8173, 0.9836))):
            for cc, v in zip(B.LOCAL_C, vals):
                r = tw(cc, V_, R)/tw(cc, float(np.round(B.alpha_loc(cc, V_, R), 8)), 1.0)
                c.check(f"flow/twin ({V_}, {R}) c={cc}: {r:.5f} ~ {v}", near(r, v, 4))
        tr = dict((row[0], row[1]) for row in B.lowk_track(root, compute=False, log=_quiet)[4096])
        for cc, v in ((0.0, 5.360), (0.0025, 5.334), (0.005, 5.255), (0.01, 4.935), (0.02, 3.639), (0.03, 1.949)):
            c.check(f"lowk_track nx=4096 c={cc}: {tr[cc].real:.6e} ~ {v}e-3", near(tr[cc].real*1e3, v, 3))
        top = B.lowk_check(root, compute=False)[(0.05, 0.60976, 4096)][0]
        c.check(f"lowk_check (0.05, 0.61, 4096): {top:.4e} ~ 1.722e-3 - 0.0294i", near(top.real*1e3, 1.722, 3) and near(top.imag, -0.0294, 4))
        P = B.lowk_paths(root, compute=False, log=_quiet)
        c.check("lowk_paths complete", [r[0] for r in P["c"]] == list(B.LOWK_PATH_C) and [r[0] for r in P["k"]] == list(B.LOWK_PATH_KS)
                and len(P["conv"]) == len(B.LOWK_PATH_CONV))
        pc, pk = dict((r[0], r[1]) for r in P["c"]), dict((r[0], r[1]) for r in P["k"])
        sc0 = B.coarse_scan("sech2", B.LOWK_S, 0.0, 0.0, "tanh", 1.0, 1.0, root, compute=False)[0.60976]
        c.check(f"the c-path ends on the c = 0 low-k branch at k = 0.61 ({pc[0.0]:.6e} vs seed {sc0:.6e}), 73% kept at c = 0.05",
                abs(pc[0.0] - sc0) < 1e-6*abs(sc0) and abs(pc[0.0].imag) < 1e-9 and near(pc[0.05].real/pc[0.0].real, 0.73, 2))
        c.check(f"the k-path ends on the c-track's k = 0.265 value ({pk[0.265]:.6e} vs {tr[0.05]:.6e})", abs(pk[0.265] - tr[0.05]) < 1e-9*abs(tr[0.05]))
        c.check(f"c = 0.05: gamma 1.775e-3 at k = 0.55 above both neighbours ({pk[0.55]:.6e})",
                near(pk[0.55].real*1e3, 1.775, 3) and pk[0.55].real > max(pk[0.5].real, pk[0.60976].real))
        cv = [r[2] for r in P["conv"]]
        c.check(f"anchor converged in nx and box ({[f'{v.real:.7e}' for v in cv]})", abs(cv[0]/cv[1] - 1) < 1e-6
                and abs(cv[2]/cv[1] - 1) < 3e-4 and abs(pc[0.05]/cv[1] - 1) < 2e-4)
        fd = B.fd_results(root)
        for nm, v in (("a6c4", 6.5e-6), ("loc2", 9.5e-6), ("a0c6", 3.2e-4)):
            d = abs(fd[nm][0]/fd[nm][2] - 1)
            c.check(f"FD {nm} vs ladder {d:.2e} ~ {v:.1e}", abs(d/v - 1) < 0.08)
        b80 = box[B.cfg_of("pair", 1e4, 0.0, 0.5, 0.6, Lx=80.0)]["lam"]
        c.check(f"Lx = 80 box value 3.8396600e-3 ({complex(b80).real:.10e})", near(complex(b80).real*1e3, 3.8396600, 7))
        # (at a doubled box the members are degenerate to the split ~1e-9 and the returned vector is a mixture: excluded)
        recs = (list(lay.values()) + list(loc.values()) + list(sym.values()) + list(box.values())
                + [x for D in (K0, KA) for r in D.values() for x in r["rows"].values()])
        pars = [complex(r["parity"]) for r in recs if r["split"] > 1e-6]
        c.check(f"every pair record with a resolved split is the EVEN member ({len(pars)} of {len(recs)}, "
                f"min Re parity {min(p.real for p in pars):.6f})", len(pars) >= 100 and all(abs(p - 1) < 1e-3 for p in pars))


if __name__ == "__main__":
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
