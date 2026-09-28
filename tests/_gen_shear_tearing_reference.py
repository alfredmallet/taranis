# Generator for tests/data/shear_tearing_reference.npz -- the independent reference for
# plans/AUTODIFF_PLAN.md rung 1b ("Shear-flow tearing against the paper's eigencode"):
# growth rates of resistive tearing with field-aligned shear flow, Mallet, Eriksson,
# Swisdak & Juno, JPP 91, E146 (2025) §7 setup (a = v_Ay = 1, nu = 0, S = 1/eta,
# Psi0 = sech^2 x i.e. f = -2 tanh sech^2 [whichf=2]; plus f = tanh [whichf=1]).
#
# It drives Alfred's Julia eigencode, vendored as tests/reference/shear_tearing_eigen.jl
# (provenance and every change from the original in that file's header), through ONE
# julia subprocess, and records for every (S, ka, alpha, whichf) point:
#
#   * the Julia code's OWN convergence study -- four grids: "base", dxmin halved, eps
#     halved, xlim doubled -- and the relative change of gamma under each;
#   * TWO eigenvalues per grid: "si" = Alfred's protocol exactly (shift-invert about
#     goodguess, KrylovKit :SR of 1/(gamma - guess) -- the eigenvalue just below the
#     guess, NOT provably the fastest mode), and "fast" = the fastest mode: the
#     max-Re eigenvalue of the DENSE spectrum on the base grid, polished on each grid by
#     shift-invert :LM about it (the eigenvalue nearest that shift);
#   * the top of the dense spectrum (eigen(B \ A), the original's `bruteforce`) on every
#     grid whose matrix is small enough (2N <= DENSE_MAX_2N), so whether the :SR
#     selection ever missed the fastest mode is recorded, not assumed;
#   * the measured inner-layer width (the paper's §7 definition: full width where the
#     real part of psi'' is >= max/4) and how many grid points it holds;
#   * eigenfunctions (x, psi, phi) of the fastest mode for a few representative points;
#   * an audit block: the operator-level truncation order of the 3-point stencil on the
#     geometric grid and the eigenvalue's observed order under joint (dxmin, eps)
#     refinement, and a deliberately-too-low guess showing what :SR then returns.
#
# THE REFERENCE VALUE a comparison should use is `gamma_fast` on the base grid, with
# `gamma_err` = max(conv_err, richardson_err) as its RELATIVE error bar: conv_err is the
# max relative change over the plan's three variants; richardson_err = (4/3) x the change
# under joint (dxmin, eps) halving (variant "both_half"), the base grid's error at the
# observed 2nd order. gamma_richardson is the extrapolated value.
# Never regenerate to make a taranis comparison pass (plan: "the convergence studies
# decide which is wrong").
#
# Not a test module (leading underscore keeps pytest away). Needs julia (>= 1.9) and the
# KrylovKit package; nothing in taranis is imported. Run from anywhere:
#   python tests/_gen_shear_tearing_reference.py [out.npz] [--julia-project DIR] [--workdir DIR]
# Without --julia-project a throw-away project is created in a temp dir with KrylovKit
# pinned to KRYLOVKIT_VERSION (offline from the local depot first, online as fallback);
# its Project.toml/Manifest.toml text is stored in the npz. `julia` is taken from
# $TARANIS_JULIA, else PATH. If julia or KrylovKit is unavailable the script PRINTS a
# skip reason and exits 0 without writing anything (the reference is host-agnostic
# data; a host without julia simply cannot regenerate it).
#
# GENERATED 2026-09-27/28 (julia 1.9.4, KrylovKit 0.8.1; see the npz metadata), in two
# invocations: the first died at point 100 (tanh, ka = 1: the shift landed exactly on
# the alpha = 0 null eigenvalue -> singular LU; now handled by shift_safe) and was
# resumed with --workdir, recomputing nothing already written. `date` is the final
# assembly; per-grid solve times are in `seconds`.
#
# NPZ KEYS (P = number of points, V = len(VARIANTS) = 5, in VARIANTS order)
#   whichf, S, ka, alpha (P,)            the point; `unstable` (P,) False = no tearing
#                                         mode (tanh at ka = 1) -- exclude those rows
#   gamma (P,) complex                    THE reference: fastest mode, base grid
#   gamma_err (P,)                        its relative error bar (see above)
#   conv_err, richardson_err, gamma_richardson (P,)
#   gamma_fast, gamma_si (P,V) complex    fastest mode / Alfred's :SR result per grid
#   relchange_fast, relchange_si (P,V-1)  |g_variant - g_base| / |g_base|
#   si_agrees (P,V), fast_matches_dense (P,V)   bool, AGREE_RTOL
#   dense_top3 (P,V,3), dense_done (P,V), n_pos_dense (P,V; -1 = not done)
#   si_converged, fast_converged, si_numops, si_normres, si_resid, fast_resid (P,V)
#   dxmin, xlim, eps, N, guess, din_meas, n_in_layer, seconds (P,V)
#   delta_est, delta_in_515, delta_noflow (P,)
#   eigfn_pids; eigfn_<pid>_x / _psi / _phi   fastest-mode eigenfunction, base grid,
#                                         normalized: psi(0) real > 0, max|[psi;phi]| = 1
#   audit_order(+_points), audit_stencil, audit_lowguess(+_point)  (column docs below)
#   julia_version, package_versions, julia_project_toml, julia_manifest_toml,
#   jl_source (the vendored .jl text), grid_rule, date, hostname, host_platform,
#   python_version, git_commit, wall_seconds
import datetime
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JL_SOURCE = os.path.join(HERE, "reference", "shear_tearing_eigen.jl")
KRYLOVKIT_VERSION = "0.8.1"

S_VALUES = (1e3, 1e4, 1e5)
KA_VALUES = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
ALPHA_VALUES = (0.0, 0.3, 0.6, 0.8, 0.95)
ALPHA_VALUES_F1 = (0.0, 0.8)

# Base grid: the paper's xlim and eps (resistive.ipynb: xlim = 10, eps = 0.02); dxmin
# from the inner-layer estimate so the layer holds >= 20 points (recorded: n_in_layer).
XLIM = 10.0
EPS = 0.02
POINTS_PER_DELTA = 50      # dxmin = delta_est / POINTS_PER_DELTA
DENSE_MAX_2N = 1600        # dense eigen(B \ A) on a grid only when 2N <= this
KRYLOV_MAXITER = 200       # the paper's maxiter
KRYLOV_SEED = 1234
GAMMA_FLOOR = 1e-5         # smallest real tearing rate in the window is ~6e-4
AGREE_RTOL = 1e-6          # si vs fastest mode: "agree" threshold (relative)

VARIANTS = ("base", "dxmin_half", "eps_half", "xlim_double", "both_half")
PLAN_VARIANTS = 3          # conv_err = max over VARIANTS[1:1+PLAN_VARIANTS] (the plan's three)

# (whichf, S, ka, alpha) whose fastest-mode eigenfunction is stored.
EIGENFUNCTION_POINTS = (
    (2, 1e4, 0.3, 0.0),
    (2, 1e4, 0.3, 0.8),
    (2, 1e4, 0.1, 0.95),
    (2, 1e5, 1.0, 0.6),
    (1, 1e4, 0.5, 0.8),
)

# audit: joint (dxmin, eps) refinement ~ uniform refinement of the grid's index map.
AUDIT_POINTS = ((2, 1e4, 0.3, 0.0), (2, 1e4, 0.3, 0.8), (2, 1e5, 0.1, 0.95))
AUDIT_LEVELS = 5           # factors 1, 1/2, 1/4, 1/8, 1/16 on BOTH dxmin and eps
AUDIT_LOWGUESS = (2, 1e4, 0.3, 0.8)   # :SR run with guess = 0.5 * fastest gamma


def delta_noflow(S, ka):
    """Alfred's estimate_delta_noflow (Coppi (S k)^-1/3 / FKR S^-2/5 k^-3/5)."""
    return (S * ka) ** (-1.0 / 3.0) if ka < S ** -0.25 else S ** -0.4 * ka ** -0.6


def delta_in_515(S, ka, alpha):
    """Paper eq 5.15: delta_in ~ (alpha/(1-alpha^2))^(1/3) (ka)^(-1/3) S^(-1/3); inf at
    alpha = 0 (the shear-modified layer does not exist there)."""
    if alpha == 0.0:
        return np.inf
    return (alpha / (1.0 - alpha ** 2)) ** (1.0 / 3.0) * ka ** (-1.0 / 3.0) * S ** (-1.0 / 3.0)


def delta_est(S, ka, alpha):
    return min(delta_noflow(S, ka), delta_in_515(S, ka, alpha))


def base_dxmin(S, ka, alpha):
    return delta_est(S, ka, alpha) / POINTS_PER_DELTA


def variant_grid(variant, dxmin, xlim=XLIM, eps=EPS):
    return {"base": (dxmin, xlim, eps),
            "dxmin_half": (dxmin / 2, xlim, eps),
            "eps_half": (dxmin, xlim, eps / 2),
            "xlim_double": (dxmin, 2 * xlim, eps),
            # joint refinement ~ a uniform refinement of the grid's index map, where the
            # stencil is 2nd order (audit): gives a Richardson estimate of the base error
            "both_half": (dxmin / 2, xlim, eps / 2)}[variant]


def grid_size(dxmin, xlim, eps):
    """N = length(vargrid(dxmin, xlim, eps)[1]) (the Julia formula)."""
    n = int(np.floor(np.log(xlim * eps / dxmin + 1) / np.log(1 + eps))) + 1
    return 2 * n - 1


def points():
    """(whichf, S, ka, alpha) of every recorded point, in npz row order."""
    out = [(2, S, ka, a) for S in S_VALUES for ka in KA_VALUES for a in ALPHA_VALUES]
    out += [(1, S, ka, a) for S in S_VALUES for ka in KA_VALUES for a in ALPHA_VALUES_F1]
    return out


def reference_path():
    return os.path.join(HERE, "data", "shear_tearing_reference.npz")


# --------------------------------------------------------------------------------------
# The julia driver. Reads the jobs file, writes one results row per (point, variant),
# eigenfunction binaries, and the audit rows. All numbers "%.17g".
_DRIVER = r'''
include(ARGS[1])
using Printf
const JOBS = ARGS[2]; const OUT = ARGS[3]; const VECDIR = ARGS[4]
const MAXIT = parse(Int, ARGS[5]); const SEED = parse(Int, ARGS[6])
const DENSEMAX = parse(Int, ARGS[7])

g17(x) = @sprintf("%.17g", x)

function topdense(S,K,ar,dxmin,xlim,eps,whichf)
    E = bruteforce(S,K,ar,dxmin,xlim,eps,whichf)
    ev = sort(E.values, by=real, rev=true)
    return ev
end

function deltain_meas(psi, x, dx, K)
    # Alfred's deltain: dd() = D2 with the Robin rows, applied to real(psi);
    # full width where it is >= max/4. Returns (width, points inside).
    N = length(x)
    ld=2.0*(dx[2:end]./dx[1:end-1])./(dx[2:end].^2+dx[1:end-1].*dx[2:end])
    ud=2.0./(dx[2:end].^2+dx[1:end-1].*dx[2:end])
    d=-2.0*(1.0.+dx[2:end]./dx[1:end-1])./(dx[2:end].^2+dx[1:end-1].*dx[2:end])
    d[1]=d[1]+2.0*((dx[2]/dx[1])/(1+K*dx[1]))/(dx[2]^2+dx[1]*dx[2])
    d[end]=d[end]+2.0/(1+K*dx[N+1])/(dx[N+1]^2+dx[N]*dx[N+1])
    DD = Tridiagonal(ld[2:end], d, ud[1:end-1])
    y = DD * real.(psi)
    m = maximum(y)
    i1 = findfirst(y .>= m/4); i2 = findlast(y .>= m/4)
    return x[i2]-x[i1], i2-i1+1
end

hdr = ["pid","variant","N","dxmin","xlim","eps","guess",
       "si_re","si_im","si_conv","si_numops","si_normres","si_resid",
       "fast_re","fast_im","fast_conv","fast_resid",
       "dense","d1_re","d1_im","d2_re","d2_im","d3_re","d3_im","n_pos_dense",
       "din_meas","n_in_layer","seconds"]
if !isfile(OUT * ".audit_done")
    for sfx in (".audit_order", ".audit_stencil", ".audit_lowguess"); rm(OUT * sfx; force=true); end
end
# resume: points already complete (all variants written) in an existing OUT are skipped
done = Set{Int}()
if isfile(OUT) && filesize(OUT) > 0
    cnt = Dict{Int,Int}()
    for l in Iterators.drop(eachline(OUT), 1)
        p = parse(Int, split(l, "\t")[1]); cnt[p] = get(cnt, p, 0) + 1
    end
    NV = parse(Int, ARGS[8])
    for (p, c) in cnt; c == NV && push!(done, p); end
    # drop partial rows so a half-written point is recomputed cleanly
    keep = [l for (i, l) in enumerate(eachline(OUT)) if i == 1 || parse(Int, split(l, "\t")[1]) in done]
    open(OUT, "w") do fh; for l in keep; println(fh, l); end; end
    io = open(OUT, "a")
    println(stderr, "resuming: $(length(done)) points already done")
else
    io = open(OUT, "w")
    println(io, join(hdr, "\t")); flush(io)
end

# shift-invert about sigma; if sigma is (to round-off) an exact eigenvalue -- e.g. the
# exact discrete null mode at alpha = 0 (psi = 0, phi = delta at x = 0, where f = 0),
# which the dense solve returns as 0 when no mode is unstable -- A - sigma B is
# singular: nudge sigma by 1e-9 (1 + i) and retry. The nearest eigenvalue is unchanged.
function shift_safe(S,K,ar,dxmin,xlim,eps,whichf,sigma,miter; seed)
    try
        return eigenmode_shift(S,K,ar,dxmin,xlim,eps,whichf,sigma,miter; seed=seed)
    catch e
        e isa SingularException || rethrow()
        return eigenmode_shift(S,K,ar,dxmin,xlim,eps,whichf,sigma+1e-9*(1+im),miter; seed=seed)
    end
end
for line in eachline(JOBS)
    startswith(line, "#") && continue
    t = split(line)
    kind = t[1]
    if kind == "point"
        pid = parse(Int, t[2]); whichf = parse(Int, t[3])
        pid in done && continue
        S = parse(Float64, t[4]); K = parse(Float64, t[5]); ar = parse(Float64, t[6])
        savevec = parse(Int, t[7])
        grids = [(parse(Float64,t[8+3j]), parse(Float64,t[9+3j]), parse(Float64,t[10+3j])) for j in 0:div(length(t)-7,3)-1]
        guess = goodguess(S,K,ar,whichf)
        # sigma for the fastest-mode polish: max-Re eigenvalue of the dense base grid
        (dxmin, xlim, eps) = grids[1]
        evb = topdense(S,K,ar,dxmin,xlim,eps,whichf)
        sigma = evb[1]
        for (j, (dxmin, xlim, eps)) in enumerate(grids)
            t0 = time()
            x, dx = vargrid(dxmin, xlim, eps); N = length(x)
            gs, vs, is = eigenmode(S,K,ar,dxmin,xlim,eps,whichf,guess,MAXIT; seed=SEED)
            rs = residual(S,K,ar,dxmin,xlim,eps,whichf,gs,vs)
            vals, vecs, ifa = shift_safe(S,K,ar,dxmin,xlim,eps,whichf,sigma,MAXIT; seed=SEED)
            gf = vals[1]; vf = vecs[1]
            rf = residual(S,K,ar,dxmin,xlim,eps,whichf,gf,vf)
            if j == 1
                ev = evb; dn = 1
            elseif 2N <= DENSEMAX
                ev = topdense(S,K,ar,dxmin,xlim,eps,whichf); dn = 1
            else
                ev = fill(complex(NaN, NaN), 3); dn = 0
            end
            npos = dn == 1 ? count(z -> real(z) > 1e-9, ev) : -1
            din, nin = deltain_meas(vf[1:N], x, dx, K)
            row = [string(pid), string(j-1), string(N), g17(dxmin), g17(xlim), g17(eps), g17(guess),
                   g17(real(gs)), g17(imag(gs)), string(is.converged), string(is.numops), g17(is.normres[1]), g17(rs),
                   g17(real(gf)), g17(imag(gf)), string(ifa.converged), g17(rf),
                   string(dn), g17(real(ev[1])), g17(imag(ev[1])), g17(real(ev[2])), g17(imag(ev[2])),
                   g17(real(ev[3])), g17(imag(ev[3])), string(npos),
                   g17(din), string(nin), g17(time()-t0)]
            println(io, join(row, "\t")); flush(io)
            if savevec == 1 && j == 1
                open(joinpath(VECDIR, "vec_$(pid).bin"), "w") do fh
                    write(fh, x); write(fh, real.(vf)); write(fh, imag.(vf))
                end
            end
        end
        println(stderr, "point $pid done"); flush(stderr)
    elseif kind in ("audit_order", "audit_stencil", "audit_lowguess") && isfile(OUT * ".audit_done")
        continue
    elseif kind == "audit_order"
        # joint refinement of (dxmin, eps): fastest-mode gamma at each level
        aid = parse(Int, t[2]); whichf = parse(Int, t[3])
        S = parse(Float64, t[4]); K = parse(Float64, t[5]); ar = parse(Float64, t[6])
        dxmin = parse(Float64, t[7]); xlim = parse(Float64, t[8]); eps = parse(Float64, t[9])
        nlev = parse(Int, t[10])
        sigma = topdense(S,K,ar,dxmin,xlim,eps,whichf)[1]
        open(OUT * ".audit_order", "a") do fa
            for l in 0:nlev-1
                fac = 0.5^l
                vals, vecs, ifa = shift_safe(S,K,ar,dxmin*fac,xlim,eps*fac,whichf,sigma,MAXIT; seed=SEED)
                sigma = vals[1]
                N = length(vargrid(dxmin*fac,xlim,eps*fac)[1])
                println(fa, join([string(aid), string(l), string(N), g17(dxmin*fac), g17(eps*fac),
                                  g17(real(vals[1])), g17(imag(vals[1])), string(ifa.converged)], "\t"))
            end
        end
        println(stderr, "audit_order $aid done"); flush(stderr)
    elseif kind == "audit_stencil"
        # operator truncation: D2 applied to g = sech^2 on interior points, max error
        dxmin = parse(Float64, t[2]); xlim = parse(Float64, t[3]); eps = parse(Float64, t[4])
        x, dx = vargrid(dxmin, xlim, eps); N = length(x)
        g(x) = sech(x)^2; g2(x) = 2*sech(x)^2*(2*tanh(x)^2 - sech(x)^2)
        hl = dx[2:N-1]; hr = dx[3:N]    # spacings around interior points 2..N-1
        y = g.(x)
        D2y = 2.0 .* (y[1:N-2] ./ (hl .* (hl .+ hr)) .- y[2:N-1] ./ (hl .* hr) .+ y[3:N] ./ (hr .* (hl .+ hr)))
        err = abs.(D2y .- g2.(x[2:N-1]))
        open(OUT * ".audit_stencil", "a") do fa
            println(fa, join([g17(dxmin), g17(eps), string(N), g17(maximum(err)),
                              g17(maximum(err[abs.(x[2:N-1]) .< 0.1])),
                              g17(maximum(err[abs.(x[2:N-1]) .> 1.0]))], "\t"))
        end
    elseif kind == "audit_lowguess"
        whichf = parse(Int, t[2]); S = parse(Float64, t[3]); K = parse(Float64, t[4]); ar = parse(Float64, t[5])
        dxmin = parse(Float64, t[6]); xlim = parse(Float64, t[7]); eps = parse(Float64, t[8])
        gfrac = parse(Float64, t[9])
        gmax = topdense(S,K,ar,dxmin,xlim,eps,whichf)[1]
        gs, vs, is = eigenmode(S,K,ar,dxmin,xlim,eps,whichf,gfrac*real(gmax),MAXIT; seed=SEED)
        open(OUT * ".audit_lowguess", "a") do fa
            println(fa, join([g17(gfrac*real(gmax)), g17(real(gmax)), g17(imag(gmax)),
                              g17(real(gs)), g17(imag(gs)), string(is.converged)], "\t"))
        end
    end
end
close(io)
touch(OUT * ".audit_done")
'''

_JL_ENV_SETUP = r'''
using Pkg
try
    withenv("JULIA_PKG_OFFLINE" => "true") do
        Pkg.add(name="KrylovKit", version=ARGS[1])
    end
catch err
    println(stderr, "offline add failed ($err); trying online")
    Pkg.add(name="KrylovKit", version=ARGS[1])
end
using KrylovKit
'''


def _skip(msg):
    print(f"SKIP _gen_shear_tearing_reference: {msg}")
    sys.exit(0)


def _julia_bin():
    jl = os.environ.get("TARANIS_JULIA") or shutil.which("julia")
    if not jl:
        _skip("no julia on PATH and $TARANIS_JULIA unset")
    if not (os.path.isfile(jl) and os.access(jl, os.X_OK)):
        _skip(f"julia binary {jl!r} is not an executable file")
    return jl


def _prepare_project(julia, project):
    if project is None:
        project = tempfile.mkdtemp(prefix="shear_tearing_jlenv_")
        r = subprocess.run([julia, f"--project={project}", "-e", _JL_ENV_SETUP,
                            KRYLOVKIT_VERSION], capture_output=True, text=True)
        if r.returncode != 0:
            _skip(f"could not install KrylovKit {KRYLOVKIT_VERSION}:\n{r.stderr[-2000:]}")
    r = subprocess.run([julia, f"--project={project}", "-e", "using KrylovKit"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        _skip(f"KrylovKit not importable in {project}:\n{r.stderr[-2000:]}")
    return project


def _versions(julia, project):
    code = ('using Pkg; println(VERSION); '
            'for (u, p) in Pkg.dependencies(); p.is_direct_dep && println(p.name, " ", p.version); end')
    r = subprocess.run([julia, f"--project={project}", "-e", code],
                       capture_output=True, text=True, check=True)
    lines = r.stdout.strip().splitlines()

    def _read(name):
        p = os.path.join(project, name)
        return open(p).read() if os.path.exists(p) else ""
    return lines[0], "\n".join(lines[1:]), _read("Project.toml"), _read("Manifest.toml")


def _write_jobs(path, pts):
    with open(path, "w") as fh:
        for pid, (whichf, S, ka, a) in enumerate(pts):
            dxm = base_dxmin(S, ka, a)
            grids = [variant_grid(v, dxm) for v in VARIANTS]
            save = int((whichf, S, ka, a) in EIGENFUNCTION_POINTS)
            fh.write(" ".join(["point", str(pid), str(whichf), repr(S), repr(ka), repr(a),
                               str(save)] + [repr(g) for grid in grids for g in grid]) + "\n")
        for aid, (whichf, S, ka, a) in enumerate(AUDIT_POINTS):
            fh.write(" ".join(["audit_order", str(aid), str(whichf), repr(S), repr(ka),
                               repr(a), repr(base_dxmin(S, ka, a)), repr(XLIM), repr(EPS),
                               str(AUDIT_LEVELS)]) + "\n")
        for lev in range(5):
            fac = 0.5 ** lev
            # (a) joint refinement, (b) dxmin only, (c) eps only
            for dxm, eps in ((1e-3 * fac, EPS * fac), (1e-3 * fac, EPS), (1e-3, EPS * fac)):
                fh.write(f"audit_stencil {dxm!r} {XLIM!r} {eps!r}\n")
        whichf, S, ka, a = AUDIT_LOWGUESS
        fh.write(" ".join(["audit_lowguess", str(whichf), repr(S), repr(ka), repr(a),
                           repr(base_dxmin(S, ka, a)), repr(XLIM), repr(EPS), "0.5"]) + "\n")


def _read_tsv(path, ncol=None):
    rows = [ln.rstrip("\n").split("\t") for ln in open(path) if ln.strip()]
    return rows


def main(argv):
    out_path = None
    project = None
    work = None
    it = iter(argv)
    for a in it:
        if a == "--julia-project":
            project = next(it)
        elif a == "--workdir":      # reuse / resume a previous run's work directory
            work = next(it)
        else:
            out_path = a
    out_path = out_path or reference_path()

    julia = _julia_bin()
    project = _prepare_project(julia, project)
    julia_version, pkg_versions, project_toml, manifest_toml = _versions(julia, project)
    print(f"julia {julia_version}; {pkg_versions}; project {project}")

    pts = points()
    work = work or tempfile.mkdtemp(prefix="shear_tearing_run_")
    print(f"work dir {work} (pass --workdir {work} to resume after an interruption)")
    jobs = os.path.join(work, "jobs.txt")
    res = os.path.join(work, "results.tsv")
    vecdir = os.path.join(work, "vecs")
    os.makedirs(vecdir, exist_ok=True)
    _write_jobs(jobs, pts)
    driver = os.path.join(work, "driver.jl")
    with open(driver, "w") as fh:
        fh.write(_DRIVER)

    t0 = time.time()
    subprocess.run([julia, f"--project={project}", driver, JL_SOURCE, jobs, res, vecdir,
                    str(KRYLOV_MAXITER), str(KRYLOV_SEED), str(DENSE_MAX_2N),
                    str(len(VARIANTS))], check=True)
    wall = time.time() - t0

    rows = _read_tsv(res)
    hdr, body = rows[0], rows[1:]
    col = {h: i for i, h in enumerate(hdr)}
    npt, nv = len(pts), len(VARIANTS)
    assert len(body) == npt * nv, (len(body), npt * nv)

    def grab(name, dtype=float):
        a = np.empty((npt, nv), dtype=dtype)
        for r in body:
            a[int(r[col["pid"]]), int(r[col["variant"]])] = dtype(r[col[name]]) \
                if dtype is not bool else r[col[name]] in ("1", "true")
        return a

    def cplx(re, im):
        return grab(re) + 1j * grab(im)

    out = {}
    P = np.array(pts, dtype=float)
    out["whichf"] = P[:, 0].astype(int)
    out["S"] = P[:, 1]
    out["ka"] = P[:, 2]
    out["alpha"] = P[:, 3]
    out["variants"] = np.array(VARIANTS)
    out["delta_est"] = np.array([delta_est(S, k, a) for _, S, k, a in pts])
    out["delta_in_515"] = np.array([delta_in_515(S, k, a) for _, S, k, a in pts])
    out["delta_noflow"] = np.array([delta_noflow(S, k) for _, S, k, a in pts])
    for name in ("dxmin", "xlim", "eps", "guess", "si_normres", "si_resid", "fast_resid",
                 "din_meas", "seconds"):
        out[name] = grab(name)
    for name in ("N", "si_numops", "n_in_layer", "n_pos_dense"):
        out[name] = grab(name, int)
    out["si_converged"] = grab("si_conv", int)
    out["fast_converged"] = grab("fast_conv", int)
    out["dense_done"] = grab("dense", int).astype(bool)
    out["gamma_si"] = cplx("si_re", "si_im")
    out["gamma_fast"] = cplx("fast_re", "fast_im")
    out["dense_top3"] = np.stack([cplx("d1_re", "d1_im"), cplx("d2_re", "d2_im"),
                                  cplx("d3_re", "d3_im")], axis=-1)

    gf = out["gamma_fast"].real
    gs = out["gamma_si"].real
    out["relchange_fast"] = np.abs(gf[:, 1:] - gf[:, :1]) / np.abs(gf[:, :1])
    out["relchange_si"] = np.abs(gs[:, 1:] - gs[:, :1]) / np.abs(gs[:, :1])
    out["conv_err"] = out["relchange_fast"][:, :PLAN_VARIANTS].max(axis=1)
    # both_half is a 2x refinement of the index map at observed order 2 (audit_order):
    # base error ~ (4/3) |gamma_both - gamma_base|, extrapolated value (4 g_both - g)/3
    jb = VARIANTS.index("both_half")
    out["gamma_richardson"] = (4 * out["gamma_fast"][:, jb] - out["gamma_fast"][:, 0]) / 3
    out["richardson_err"] = (4.0 / 3.0) * out["relchange_fast"][:, jb - 1]
    # the error bar a comparison should use
    out["gamma_err"] = np.maximum(out["conv_err"], out["richardson_err"])
    out["gamma"] = out["gamma_fast"][:, 0]      # THE reference value (base grid)
    # False where no tearing mode exists on the base grid: the dense spectrum has no
    # Re > 1e-9 eigenvalue, or the fastest one is below GAMMA_FLOOR (tanh at ka = 1,
    # Delta' = 0: the dense top is the exact alpha=0 null mode or a +1e-8..1e-7 marginal
    # mode). gamma and its relative errors are meaningless there.
    out["unstable"] = (out["n_pos_dense"][:, 0] > 0) & (out["gamma"].real > GAMMA_FLOOR)
    out["si_agrees"] = (np.abs(out["gamma_si"] - out["gamma_fast"])
                        <= AGREE_RTOL * np.abs(out["gamma_fast"]))
    d1 = out["dense_top3"][..., 0]
    out["fast_matches_dense"] = np.where(
        out["dense_done"], np.abs(out["gamma_fast"] - d1) <= AGREE_RTOL * np.abs(d1), True)

    # eigenfunctions
    ef_ids = [i for i, p in enumerate(pts) if p in EIGENFUNCTION_POINTS]
    out["eigfn_pids"] = np.array(ef_ids)
    for pid in ef_ids:
        raw = np.fromfile(os.path.join(vecdir, f"vec_{pid}.bin"), dtype=np.float64)
        N = int(out["N"][pid, 0])
        x, vre, vim = raw[:N], raw[N:3 * N], raw[3 * N:]
        v = vre + 1j * vim
        out[f"eigfn_{pid}_x"] = x
        out[f"eigfn_{pid}_psi"] = v[:N]
        out[f"eigfn_{pid}_phi"] = v[N:]

    # audits
    ao = np.array(_read_tsv(res + ".audit_order"), dtype=float)
    out["audit_order_points"] = np.array(AUDIT_POINTS, dtype=float)
    out["audit_order"] = ao   # cols: aid, level, N, dxmin, eps, gamma_re, gamma_im, conv
    ast = np.array(_read_tsv(res + ".audit_stencil"), dtype=float)
    out["audit_stencil"] = ast  # cols: dxmin, eps, N, maxerr, maxerr|x|<0.1, maxerr|x|>1
    out["audit_lowguess_point"] = np.array(AUDIT_LOWGUESS, dtype=float)
    out["audit_lowguess"] = np.array(_read_tsv(res + ".audit_lowguess"), dtype=float)
    # cols: guess, gamma_fast_re, gamma_fast_im, gamma_si_re, gamma_si_im, conv

    # metadata
    out["julia_version"] = np.array(julia_version)
    out["package_versions"] = np.array(pkg_versions)
    out["julia_project_toml"] = np.array(project_toml)
    out["julia_manifest_toml"] = np.array(manifest_toml)
    out["jl_source"] = np.array(open(JL_SOURCE).read())
    out["grid_rule"] = np.array(
        f"base: xlim={XLIM}, eps={EPS}, dxmin=min(delta_noflow, delta_in_515)/"
        f"{POINTS_PER_DELTA}; variants dxmin/2, eps/2, 2*xlim; dense when 2N<="
        f"{DENSE_MAX_2N}; maxiter={KRYLOV_MAXITER}, seed={KRYLOV_SEED}")
    out["date"] = np.array(datetime.datetime.now().isoformat(timespec="seconds"))
    out["hostname"] = np.array(platform.node())
    out["host_platform"] = np.array(platform.platform())
    out["python_version"] = np.array(platform.python_version())
    try:
        out["git_commit"] = np.array(subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=HERE).strip())
    except Exception:
        out["git_commit"] = np.array("unknown")
    out["wall_seconds"] = np.array(wall)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    print(f"wrote {out_path} ({wall:.0f} s of julia)")
    _summary(out)
    return out_path


def _summary(o):
    print()
    print(" f      S    ka  alpha        gamma      conv_err  rich_err  N_base  n_layer  si==fast  npos")
    for i in range(len(o["S"])):
        print(f" {o['whichf'][i]} {o['S'][i]:6.0e} {o['ka'][i]:5.2f} {o['alpha'][i]:5.2f} "
              f"{o['gamma'][i].real:12.6e}  {o['conv_err'][i]:9.2e}  {o['richardson_err'][i]:8.2e}  "
              f"{o['N'][i,0]:6d}  "
              f"{o['n_in_layer'][i,0]:7d}  {str(o['si_agrees'][i].all()):>8}  "
              f"{o['n_pos_dense'][i,0]:4d}")


if __name__ == "__main__":
    main(sys.argv[1:])
