# Eigen-solves and resumable data generation for examples/tearing-shear-eigen.ipynb:
# plans/AUTODIFF_PLAN.md rung 1b-ref -- resistive tearing with FIELD-ALIGNED SHEAR FLOW through
# the autodiff eigen-harness (taranis/stability.py), against Alfred's Julia eigencode
# (tests/reference/shear_tearing_eigen.jl; data tests/data/shear_tearing_reference.npz, whose
# generator header documents every key). The paper: Mallet, Eriksson, Swisdak & Juno, JPP 91,
# E146 (2025), sections 2 and 7.
#
# Setup (every number below): dims=2, nu = 0 (diss = (0, eta)), hyper = 1, a = v_Ay = 1,
# S = 1/eta, ky block (Ly = 2 pi/k, ny = 8, iky = 1), fp64. Equilibrium psi = Psi0(x),
# phi = alpha*Psi0(x) (u = z x grad phi = alpha b_perp: field-aligned flow). With taranis's
# brackets the linearized ky block is EXACTLY the Julia code's
#     gamma psi = eta (D2 - k^2) psi - i alpha k f psi + i k f phi,
#     gamma (D2 - k^2) phi = i k [f (D2 - k^2) psi - f'' psi] - i alpha k [f (D2 - k^2) phi - f'' phi]
# (f = Psi0'), i.e. the same sign of alpha; the spectrum is anyway even in alpha (y -> -y,
# phi -> -phi is a symmetry of RMHD), which the notebook checks.
#   whichf = 2: Psi0 = sech^2 x (+ its two nearest periodic images, rung 1a's box), Lx = 15/k.
#   whichf = 1: f = tanh x is not periodic. The plan's Harris-like family tanh(sin x/a)/tanh(1/a)
#     carries an O(a^2) SHAPE error in Delta' (0.05-1.5 % at affordable boxes -- measured here,
#     `harris_dprime_table`), far above the reference's 3e-5..1e-3 error bars, so the comparison
#     uses T.psi0_tanhpair instead: B_y = tanh(x - Lx/4) and -tanh(x - 3Lx/4), exact to e^(-Lx),
#     whose only deviation from an isolated sheet is the e^(-k Lx/2) coupling of the two sheets.
#     That coupling splits the tearing mode into an even/odd PAIR with Delta'_+- = Delta'_iso +- d
#     at first order, so the pair MEAN is the isolated-sheet value to O(d^2), and half the split
#     is quoted as its (conservative) box error. Both pair members are converged together
#     (shift-invert k = 2).
#
# SOLVER: rung 1a's machinery, imported from tearing_eigen_run (T): shift-invert on the
# matrix-free ky block with band-LU-preconditioned GMRES, nx ladders to 1e-7 (T.ladder), every
# solve cached (T.cached, keyed by configuration + alpha). The first S of each (ka, alpha) is
# seeded by a dense spectrum (T.cached("dense")); later S by continuation (previous S's
# converged value times the alpha = 0 theory ratio). Dense spectra on the seed grid certify the
# fastest mode (`dense` phase).
#
# RESOLUTION (measured, notebook section 2): for sech^2 the flow WIDENS the layer and the nx
# ladders end at the same or smaller nx as alpha grows (<= 16384 everywhere, Lx = 15/k). The
# tanh pair with flow is the expensive case (its far field carries a uniform flow +-alpha along
# a uniform field): at S = 1e3, ka = 0.5, alpha = 0.8 the pair members are 14 % apart at
# dx = 0.039 (Lx = 40), 6 % at dx = 0.029 and 4e-6 at dx = 0.007 (Lx = 60), so its alpha = 0.8 ladders are capped (NX_CAP_TANH) and the pair MEAN
# (which converges much faster than either member) is what is compared.
#
# `python examples/tearing_shear_run.py [wall_seconds] [phase ...]` runs make_data to completion
# (resumable; phases can be split across processes since every solve is its own cache file).
import os
os.environ.setdefault("TARANIS_PRECISION", "64")

import sys
import time

import numpy as np

import tearing_eigen_run as T

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "tearing-shear")
REF_PATH = os.path.join(HERE, "..", "tests", "data", "shear_tearing_reference.npz")

S_REF = (1e3, 1e4, 1e5)
KA_REF = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
ALPHA_REF = (0.0, 0.3, 0.6, 0.8, 0.95)
ALPHA_F1 = (0.0, 0.8)
KA_EXTRA = (0.15, 0.4, 0.6, 0.85)          # extra ka for k_max(alpha) (sech^2 only)
NX_CAP = 65536
# tanh pair nx cap per alpha (Lx = 30/k: ~2x sech^2's nx per dx). alpha = 0 ran uncapped (its
# ladders converge by nx = 32768); at alpha = 0.8 one nx = 16384 solve at ka = 0.1 is ~30 min, so
# those ladders stop there and the pair MEAN's last change is the quoted error
NX_CAP_TANH = {0.0: 65536, 0.8: 16384}
# tanh pair, S = 1e5, alpha = 0.8: the Doppler-shifted Alfven resonance gamma + i k (alpha - 1) f = 0
# sits at complex distance d = gamma/(k (1 - alpha) |f'|) = 0.064 .. 0.0055 (ka = 0.1 .. 0.7) from
# the sheet and must be resolved (nx ~ 20 Lx/d); these rows run in a SMALLER box,
# Lx = max(20/k, 40) (the pair MEAN is box-independent to 1e-8 relative between kLx = 20 and 40,
# section 2), with a larger nx cap. S = 1e4 too, all ka (at ka = 0.1 GMRES also fails at every
# shift at nx = 8192 in the Lx = 30/k box).
# Every cap is 65536 (= NX_CAP, which family() also applies): a 131072 solve is hours and most
# of the M1's memory. S = 1e5, ka = 0.7 wants nx ~ 20 Lx/d ~ 1.5e5 and is left NOT
# ladder-converged at 65536 (last change of the mean 2.6e-4; its increments shrink ~40x per
# rung, so the true error is ~6e-6 -- an extrapolation) -- an open item, not a silent cap.
TANH_HI = {(1e5, 0.8): (20.0, 65536),
           (1e4, 0.8): (20.0, 65536)}         # nx ~ 2-4.5e4 (at Lx = 30/k, nx <= 16384 they were
                                              # under-resolved -- review 2026-09-29)
TOL = 1e-7
TANH_C = 30.0                              # tanh pair: Lx = TANH_C/k (sheet spacing 15/k)
CASES = {2: "sech2", 1: "tanhpair"}
BOX_KA = (0.1, 0.3, 1.0)                   # box study (sech^2, S = 1e3)
BOX_ALPHA = (0.6, 0.95)
BOX_C = (10.0, 15.0, 20.0, 30.0)
TBOX_C = (20.0, 30.0, 40.0)                # tanh-pair box study (S = 1e3, ka = 0.3, 0.5)
HARRIS_C = 15.0                            # Harris-like demo: a = pi ka/HARRIS_C (sheet
HARRIS_KA = (0.3, 0.5, 0.7)                # spacing pi/a = HARRIS_C/k)


# ============================================================ reference

def load_ref(path=REF_PATH):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def ref_index(ref, whichf, S, ka, alpha):
    sel = np.nonzero((ref["whichf"] == whichf) & (ref["S"] == S) & np.isclose(ref["ka"], ka)
                     & np.isclose(ref["alpha"], alpha))[0]
    assert len(sel) == 1, (whichf, S, ka, alpha)
    return int(sel[0])


# ============================================================ boxes, seeds

def lx_of(case, ka, c=None):
    if case == "sech2":
        return T.lx_for(ka) if c is None else max(c/ka, 10.0)
    return max((TANH_C if c is None else c)/ka, 40.0)


def nx_dense(Lx):
    return 1024 if Lx <= 40 else 2048


def gamma_theory0(case, ka, S):
    # the alpha = 0 exact dispersion relation (used ONLY to scale continuation seeds)
    if case == "sech2":
        return T.gamma_theory(ka, S)
    return T.gamma_disp(ka, 1.0/S, 1.0, T.dprime_harris_local(ka))


def top_real(values, tol=1e-8):
    # the largest-Re eigenvalue that is real (|Im| <= tol |lam|) and positive, else None
    for z in values:
        z = complex(z)
        if z.real > 0 and abs(z.imag) <= tol*abs(z) + 1e-13:
            return z
    return None


def family(case, ka, alpha, S_list, root=None, log=print, deadline=np.inf, nxcap=NX_CAP,
           Lx=None, a=1.0, lxc=None, nxcap_tanh=None):
    # ascending-S ladders at one (ka, alpha): the first S seeded by a dense spectrum, each later
    # S by continuation (previous converged value times the alpha = 0 theory ratio), starting
    # at nx = max(dense nx, previous final nx / 2). Returns ({S: ladder result}, complete).
    root = root or DATA_ROOT
    Lx = Lx or lx_of(case, ka, lxc)
    nev = 1 if case == "sech2" else 2
    if case != "sech2":
        nxcap = min(nxcap, nxcap_tanh or NX_CAP_TANH.get(alpha, 16384))
    target = "tearing" if nev == 1 else "pair"
    out, prev = {}, None
    for S in S_list:
        if time.time() > deadline:
            return out, False
        if prev is None:
            nxd = nx_dense(Lx)
            d = T.cached("dense", case, S, ka, nxd, Lx, a=a, root=root, alpha=alpha)
            lam = top_real(d["values"])
            if lam is None:
                log(f"  {case} ka={ka} alpha={alpha} S={S:.0e}: no real growing mode in the "
                    f"dense spectrum (top {d['values'][:3]})")
                out[S] = dict(ok=False, nx=[], lam=[], value=np.nan, err=np.inf,
                              converged=False, rec=None)
                continue
            sigma0 = lam + (0.03 if nev == 1 else 0.1)*abs(lam)
            nx0 = nxd
        else:
            Sp, res = prev
            gt, gtp = gamma_theory0(case, ka, S), gamma_theory0(case, ka, Sp)
            scale = gt/gtp if np.isfinite(gt) and np.isfinite(gtp) else (Sp/S)**0.5
            sigma0 = res["value"].real*scale*1.03
            nx0 = max(nx_dense(Lx), res["nx"][-1]//2)
        res = T.ladder(case, S, ka, Lx, nx0, sigma0, nxcap=nxcap, target=target, a=a,
                       root=root, log=log, nev=nev, alpha=alpha, tol=TOL)
        out[S] = res
        if res["ok"]:
            prev = (S, res)
    return out, True


# ============================================================ phases

def _main(root, log, deadline, which=(2,), alphas=None, kas=None):
    out, done = {}, True
    for whichf in which:
        case = CASES[whichf]
        for alpha in (alphas or (ALPHA_REF if whichf == 2 else ALPHA_F1)):
            for ka in (kas or KA_REF):
                hi = [S for S in S_REF if (S, alpha, ka) in TANH_HI or (S, alpha) in TANH_HI] \
                    if whichf == 1 else []
                fam, ok = family(case, ka, alpha, [S for S in S_REF if S not in hi], root, log,
                                 deadline)
                for S in hi:
                    c, cap = TANH_HI.get((S, alpha, ka), TANH_HI.get((S, alpha)))
                    f2, ok2 = family(case, ka, alpha, (S,), root, log, deadline, lxc=c,
                                     nxcap_tanh=cap)
                    fam.update(f2)
                    ok &= ok2
                out[(whichf, ka, alpha)] = fam
                done &= ok
                if not ok:
                    return out, False
    return out, done


def _extra(root, log, deadline):
    out = {}
    for alpha in ALPHA_REF:
        for ka in KA_EXTRA:
            fam, ok = family("sech2", ka, alpha, S_REF, root, log, deadline)
            out[(ka, alpha)] = fam
            if not ok:
                return out, False
    return out, True


def _dense(root, log, deadline):
    # the whole block spectrum on the seed grid of EVERY reference point (and the stable
    # tanh ka = 1 rows): what certifies "fastest mode"
    out = {}
    for whichf in (2, 1):
        case = CASES[whichf]
        for S in S_REF:
            for alpha in (ALPHA_REF if whichf == 2 else ALPHA_F1):
                for ka in KA_REF:
                    if time.time() > deadline:
                        return out, False
                    Lx = lx_of(case, ka)
                    out[(whichf, S, ka, alpha)] = T.cached("dense", case, S, ka, nx_dense(Lx), Lx,
                                                           root=root, alpha=alpha)
    return out, True


def _box(root, log, deadline):
    # box-size study at S = 1e3: the converged eigenvalue at Lx = c/k (sech^2) and the pair's
    # split and mean at Lx = c/k (tanh pair)
    out = {"sech2": {}, "tanhpair": {}}
    for alpha in BOX_ALPHA:
        for ka in BOX_KA:
            for c in BOX_C:
                if time.time() > deadline:
                    return out, False
                fam, _ = family("sech2", ka, alpha, (1e3,), root, log, deadline, lxc=c)
                out["sech2"][(ka, alpha, c)] = fam.get(1e3)
    for ka in (0.3, 0.5):
        for c in TBOX_C:
            if time.time() > deadline:
                return out, False
            fam, _ = family("tanhpair", ka, 0.8, (1e3,), root, log, deadline, lxc=c)
            out["tanhpair"][(ka, c)] = fam.get(1e3)
    return out, True


def harris_a(ka):
    return np.pi*ka/HARRIS_C


def _harris(root, log, deadline):
    # the plan's Harris-like family at S = 1e3 (units of its a, box 2 pi): its Delta' shape error
    # shows up in gamma, and is predicted by the alpha = 0 dispersion relation
    out = {}
    for alpha in ALPHA_F1:
        for ka in HARRIS_KA:
            if time.time() > deadline:
                return out, False
            a = harris_a(ka)
            fam, _ = family("harris", ka, alpha, (1e3,), root, log, deadline, Lx=2*np.pi, a=a)
            out[(ka, alpha)] = fam.get(1e3)
    return out, True


PHASES = ("main", "tanh", "dense", "box", "extra", "harris")


def make_data(root=None, wall_budget=3*3600.0, phases=None, log=print):
    # resumable: every eigen-solve is cached; returns True when every requested phase is done
    root = root or DATA_ROOT
    os.makedirs(root, exist_ok=True)
    deadline = time.time() + wall_budget
    done = True
    for name in phases or PHASES:
        if ":" in name:            # "main:0.8" / "tanh:0.8[:0.3]": one alpha [one ka]
            kind, al, *kk = name.split(":")
            fn = (lambda k, a, ks: lambda *args: _main(*args, which=(2 if k == "main" else 1,),
                                                       alphas=(a,), kas=ks))(
                kind, float(al), tuple(float(v) for v in kk) or None)
            _, ok = fn(root, log, deadline)
            log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'}")
            done &= ok
            continue
        fn = {"main": lambda *a: _main(*a, which=(2,)), "tanh": lambda *a: _main(*a, which=(1,)),
              "dense": _dense, "box": _box, "extra": _extra, "harris": _harris}[name]
        t0 = time.time()
        _, ok = fn(root, log, deadline)
        log(f"== phase {name}: {'complete' if ok else 'INCOMPLETE'} ({time.time() - t0:.0f} s)")
        done &= ok
        if time.time() > deadline:
            return False
    return bool(done)


# ============================================================ read-back (all cached)

def _quiet(*a, **k):
    pass


def main_results(root=None, which=(2, 1)):
    # {(whichf, ka, alpha): {S: ladder result}}
    out = {}
    for w in which:
        out.update(_main(root or DATA_ROOT, _quiet, np.inf, which=(w,))[0])
    return out


def extra_results(root=None):
    return _extra(root or DATA_ROOT, _quiet, np.inf)[0]


def dense_results(root=None):
    return _dense(root or DATA_ROOT, _quiet, np.inf)[0]


def box_results(root=None):
    return _box(root or DATA_ROOT, _quiet, np.inf)[0]


def harris_results(root=None):
    return _harris(root or DATA_ROOT, _quiet, np.inf)[0]


def _mean2(values):
    v = np.sort(np.real(values))[::-1][:2]
    return 0.5*(v[0] + v[1]), 0.5*abs(v[0] - v[1])


def pair_value(res):
    # (isolated-sheet estimate, its box error) from a tanh-pair ladder: the mean of the two
    # (real) pair members, and half their split
    return _mean2(res["rec"]["values"])


def point_value(case, res):
    # (gamma, absolute error) for one converged ladder: sech^2 -> the eigenvalue and the last
    # ladder change; tanh pair -> the pair mean and max(ladder change, half split)
    if case == "sech2":
        return res["value"].real, res["err"]
    # tanh pair -> the pair MEAN, whose ladder change is its grid error (the members converge
    # much more slowly than their mean while the Doppler resonance is under-resolved), and the
    # half split as box error
    g, half = pair_value(res)
    means = [_mean2(v)[0] for v in res.get("vals", [])]
    err = abs(means[-1] - means[-2]) if len(means) >= 2 else res["err"]
    return g, max(err, half)


# ============================================================ theory helpers

def tanhpair_pot(x, Lx):
    # f''/f for T.psi0_tanhpair's B_y (regular at the sheets, where it -> -2)
    ms = range(-2, 3)
    f = -1.0 + sum(np.tanh(x - Lx/4 - m*Lx) - np.tanh(x - 3*Lx/4 - m*Lx) for m in ms)
    def g(u):             # -2 tanh sech^2, written so it cannot overflow
        e = np.exp(-2.0*np.abs(u))
        return -2.0*np.sign(u)*(1 - e)/(1 + e)*4.0*e/(1.0 + e)**2
    fpp = sum(g(x - Lx/4 - m*Lx) - g(x - 3*Lx/4 - m*Lx) for m in ms)
    return -2.0/np.cosh(x - Lx/4)**2 if abs(f) < 1e-7 else fpp/f


def dprime_tanhpair(k, Lx, parity):
    # Delta' at the first sheet (x = Lx/4) of the tearing mode even (+1) / odd (-1) about the
    # midpoint x = Lx/2: shoot psi'' = (k^2 + f''/f) psi from Lx/2 to Lx/4+
    from scipy.integrate import solve_ivp
    y0 = [1.0, 0.0] if parity > 0 else [0.0, 1.0]
    with np.errstate(invalid="ignore"):     # solve_ivp's first-step estimate at psi(Lx/2) = 0
        sol = solve_ivp(lambda x, y: [y[1], (k*k + tanhpair_pot(x, Lx))*y[0]], [Lx/2, Lx/4],
                        y0, rtol=1e-12, atol=1e-300, method="DOP853")
    return 2.0*sol.y[1, -1]/sol.y[0, -1]


def harris_dprime_table(kas=(0.1, 0.2, 0.3, 0.5, 0.7), cs=(10.0, 15.0, 20.0)):
    # the Harris-like family's Delta'_+- a against the isolated tanh 2(1/ka - ka), at sheet
    # spacing c/k (a = pi ka/c): rows (ka, c, rel_even, rel_odd)
    rows = []
    for ka in kas:
        dl = T.dprime_harris_local(ka)
        for c in cs:
            a = np.pi*ka/c
            dp, dm = [T.dprime_harris(ka/a, a, s)*a for s in (1, -1)]
            rows.append((ka, c, dp/dl - 1, dm/dl - 1))
    return rows


# ============================================================ Julia adjudication
# Where the two codes differ by more than taranis's own error, the vendored Julia code
# (tests/reference/shear_tearing_eigen.jl, UNCHANGED; the npz is never regenerated) is re-run
# here with a wider convergence study than the reference's -- its outer box xlim as well as
# (dxmin, eps), with a Richardson step at every xlim -- so the convergence studies decide.
# Needs julia + KrylovKit (a project dir JL_PROJECT, set up with the generator's
# _prepare_project); results cached in DATA_ROOT/julia_study.tsv.

JL_PROJECT = os.path.join(DATA_ROOT, "jlenv")
JL_SOURCE = os.path.join(HERE, "..", "tests", "reference", "shear_tearing_eigen.jl")
JL_LEVELS = (1.0, 0.5, 0.25)        # joint (dxmin, eps) refinement factors
_JL_DRIVER = r'''
include(ARGS[1])
using Printf
for line in eachline(ARGS[2])
    p = split(line)
    whichf = parse(Int, p[1]); S, K, ar, dxmin, xlim, eps, sr, si = parse.(Float64, p[2:9])
    t = time()
    vals, vecs, info = eigenmode_shift(S, K, ar, dxmin, xlim, eps, whichf, complex(sr, si), 200)
    A, B, x, dx = operators(S, K, ar, dxmin, xlim, eps, whichf)
    res = residual(S, K, ar, dxmin, xlim, eps, whichf, vals[1], vecs[1])
    open(ARGS[3], "a") do io
        println(io, join([p[1:8]; [@sprintf("%.17g", v) for v in (real(vals[1]), imag(vals[1]),
                length(x), res, time() - t)]], "\t"))
    end
end
'''


def _jl_key(row):
    # (whichf, S, ka, alpha, dxmin, xlim, eps): one Julia solve
    return tuple(float(v) for v in row[:7])


def julia_study(jobs, path=None, log=print):
    # jobs: list of (whichf, S, ka, alpha, dxmin, xlim, eps, sigma). Returns {key: (gamma, N,
    # residual, seconds)}, running only the jobs missing from the cache file.
    import subprocess
    import tempfile
    path = path or os.path.join(DATA_ROOT, "julia_study.tsv")
    have = {}
    if os.path.exists(path):
        for line in open(path):
            p = line.split("\t")
            have[_jl_key(p)] = (complex(float(p[8]), float(p[9])), int(float(p[10])),
                                float(p[11]), float(p[12]))
    todo = [j for j in jobs if _jl_key(j) not in have]
    if todo:
        with tempfile.TemporaryDirectory() as tmp:
            jf, df = os.path.join(tmp, "jobs.txt"), os.path.join(tmp, "driver.jl")
            with open(jf, "w") as fh:
                for (w, S, ka, al, dxm, xl, ep, sg) in todo:
                    fh.write(f"{w} {S!r} {ka!r} {al!r} {dxm!r} {xl!r} {ep!r} "
                             f"{complex(sg).real!r} {complex(sg).imag!r}\n")
            with open(df, "w") as fh:
                fh.write(_JL_DRIVER)
            log(f"julia: {len(todo)} solves")
            subprocess.run(["julia", f"--project={JL_PROJECT}", df, JL_SOURCE, jf, path],
                           check=True)
        return julia_study(jobs, path, log)
    return have


def julia_xlim_jobs(ref, whichf, S, ka, alpha, xlims=(10.0, 20.0, 40.0), levels=JL_LEVELS):
    i = ref_index(ref, whichf, S, ka, alpha)
    dxmin, eps = float(ref["dxmin"][i, 0]), float(ref["eps"][i, 0])
    sg = complex(ref["gamma"][i])
    return [(whichf, S, ka, alpha, dxmin*f, xl, eps*f, sg) for xl in xlims for f in levels]


def julia_xlim_table(ref, whichf, S, ka, alpha, xlims=(10.0, 20.0, 40.0), levels=JL_LEVELS,
                     log=print):
    # {xlim: (gammas per level, Richardson from the last two levels (2nd order), N per level)}
    jobs = julia_xlim_jobs(ref, whichf, S, ka, alpha, xlims, levels)
    got = julia_study(jobs, log=log)
    out = {}
    for xl in xlims:
        gs, Ns = [], []
        for j in jobs:
            if j[5] == xl:
                g, N, _, _ = got[_jl_key(j)]
                gs.append(g.real)
                Ns.append(N)
        out[xl] = (np.array(gs), gs[-1] + (gs[-1] - gs[-2])/3.0, Ns)
    return out


if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3*3600.0
    phases = tuple(sys.argv[2:]) or None
    t_end = time.time() + budget

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)
    while time.time() < t_end and not make_data(wall_budget=t_end - time.time(), phases=phases,
                                                log=log):
        pass
