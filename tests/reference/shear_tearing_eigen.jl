# shear_tearing_eigen.jl -- vendored reference eigencode for resistive tearing with
# field-aligned shear flow (Mallet, Eriksson, Swisdak & Juno, JPP 91, E146 (2025),
# eqs 2.6-2.7 with a = v_Ay = 1, nu = 0, S = 1/eta).
#
# PROVENANCE
#   Source : /Users/alfy/Documents/current_projects/sheartearing/sheet_instabilities.jl
#            (Alfred Mallet; file mtime 2024-08-13; sha256
#            20c8118885df05b6f6d9c17eb5dcb4fcc39fe3926489edc51e7ef50f2c98239f).
#            The code behind the paper's figures (resistive.ipynb: S = 1e12,
#            dxmin = 1e-6, xlim = 10, eps = 0.02, whichf = 2, maxiter = 200).
#   Vendored: 2026-09-27, for plans/AUTODIFF_PLAN.md rung 1b (the taranis eigen-harness
#            is validated against it). Driven by tests/_gen_shear_tearing_reference.py.
#
# WHAT WAS KEPT VERBATIM (the discretization is the reference; do not "improve" it)
#   f1/ddf1 (tanh), f2/ddf2 (-2 tanh sech^2), choosef, vargrid, the 3-point
#   nonuniform second-derivative coefficients (ld, d, ud), the Robin outer boundary
#   rows, the (A, B) assembly (every matrix entry expression is character-for-character
#   the original's), goodguess (resistive branch), estimate_delta_noflow, DeltaPrime,
#   the eigenvector normalization (phase of psi at x=0, max-abs = 1).
#
# WHAT WAS CHANGED, AND WHY
#   1. Dropped: Plots, LaTeXStrings, BenchmarkTools, say(), all scan_* drivers, the
#      collisionless/combined variants, outersol, searchsortednearest. Reason: no
#      plotting deps; the drivers live in the python generator. (Original scan_ar and
#      scan_S are broken anyway: undefined ik/whichf, wrong eigenmode arity.)
#   2. The (A, B) assembly, duplicated verbatim in the original's eigenmode and
#      bruteforce, is factored into ONE function `operators`, called by both, so the
#      shift-invert and dense solves provably see the same matrices.
#   3. BUG FIX (original bruteforce): it referenced an undefined `f`/`ddf` (no
#      `whichf` argument, no choosef call) and so could not run. The copy takes
#      `whichf` and uses choosef like eigenmode. Its return value (the raw
#      `eigen(B \ A)` factorization) is unchanged.
#   4. eigenmode: the shifted matrix (A - guess*B) is LU-factorized ONCE and reused in
#      every Krylov application (the original re-factorized it on each `\`). Same
#      factorization, same solves -- a speed change only.
#   5. eigenmode: the Krylov start vector is drawn from a seeded MersenneTwister(seed)
#      (default seed 1234) instead of the global RNG, so the reference is
#      reproducible. The original's `rand(ComplexF64, 2N)` distribution is kept.
#   6. eigenmode_shift: NEW helper, not in the original. The same shift-invert
#      operator but selecting :LM of 1/(gamma - sigma) (the eigenvalue NEAREST a
#      complex shift sigma), used by the generator to polish the fastest mode found by
#      the dense solve. The original's `:SR` selection (eigenmode) is kept unchanged
#      and is what "Alfred's protocol" means in the reference npz.
#   7. residual: NEW helper, ||A v - gamma B v|| / ||gamma B v||.
#
# UNKNOWNS AND EQUATIONS
#   Unknown vector v = [psi(x_1..x_N); phi(x_1..x_N)], generalized problem
#   A v = gamma B v with B = diag(I, D2 - K^2):
#     gamma psi          = (D2 - K^2) psi / S - i ar K f psi + i K f phi
#     gamma (D2-K^2) phi = i K [f (D2-K^2) psi - f'' psi] - i ar K [f (D2-K^2) phi - f'' phi]
#   (paper 2.6-2.7; note the psi equation keeps the eta*K^2 term the paper drops.)
#   ar = alpha (Phi0 = alpha Psi0), K = k a.

using LinearAlgebra
using SparseArrays
using KrylovKit
using Random

f1(x)=tanh(x)
ddf1(x)=-2*tanh(x)*(sech(x))^2

f2(x)=-2*tanh(x)*(sech(x))^2
ddf2(x)=8*tanh(x)*(sech(x))^2 * (3*(sech(x))^2-1)

function choosef(whichf)
    if whichf==1
        f=f1
        ddf=ddf1
    elseif whichf==2
        f=f2
        ddf=ddf2
    else
        println("whichf should be ==1 or ==2, using f1")
        f=f1
        ddf=ddf1
    end
    return f,ddf
end

# Geometric grid, symmetric about x=0 (x[n] == 0): spacing dxmin at the centre, growing
# by (1+eps) per cell out to ~xlim. x has 2n-1 points; dx has 2n entries, dx[i] =
# x[i]-x[i-1] for interior i, dx[1] == dx[2n] are the ghost spacings past the ends.
function vargrid(dxmin,xlim,eps)
    if eps > 0.0
        n = convert(Int64,floor(log(xlim*eps/dxmin+1)/log((1+eps))))+1
    else
        n = convert(Int64,xlim/dxmin)+1
    end

    dxmax = dxmin*(1+eps)^n
    x=zeros(2*n-1)
    dx=zeros(2*n)
    dx[n+1]=dxmin
    dx[n]=dxmin
    for i in (n+1):2*n-1
        dx[i+1]=dx[i]*(1+eps)
        x[i]=x[i-1]+dx[i]
    end
    for i in n-1:-1:1
        dx[i]=dx[2*n+1-i]
        x[i]=-x[2*n-i]
    end

    return x,dx
end

# The (A, B) pencil. Every entry expression below is verbatim from the original
# eigenmode/bruteforce (change 2 in the header).
function operators(S,K,ar,dxmin,xlim,eps,whichf)
    f,ddf=choosef(whichf)
    x,dx=vargrid(dxmin,xlim,eps)

    N=length(x)

    ld=2.0*(dx[2:end]./dx[1:end-1])./(dx[2:end].^2+dx[1:end-1].*dx[2:end])
    ud=2.0./(dx[2:end].^2+dx[1:end-1].*dx[2:end])
    d=-2.0*(1.0.+dx[2:end]./dx[1:end-1])./(dx[2:end].^2+dx[1:end-1].*dx[2:end])
    d[1]=d[1]+2.0*((dx[2]/dx[1])/(1+K*dx[1]))/(dx[2]^2+dx[1]*dx[2])
    d[end]=d[end]+2.0/(1+K*dx[N+1])/(dx[N+1]^2+dx[N]*dx[N+1])

    B_ud=vcat(zeros(ComplexF64,N),ud[1:end-1])
    B_ld=vcat(zeros(ComplexF64,N),ld[2:end])
    B_d=vcat(ones(ComplexF64,N),d.-K^2)
    B=Tridiagonal(B_ld,B_d,B_ud)
    I_11=vcat(2:N,1:N,1:N-1)
    J_11=vcat(1:N-1,1:N,2:N)
    A_11=vcat(ld[2:N]/S,(d.-K^2)/S -im*ar*K*f.(x),ud[1:N-1]/S)
    I_12=1:N
    J_12=N+1:2*N
    A_12=im*K*f.(x)
    I_21=vcat(N+2:2*N,N+1:2*N,N+1:2*N-1)
    J_21=vcat(1:N-1,1:N,2:N)
    A_21=vcat(im*K*f.(x[2:N]).*ld[2:N],im*K*f.(x).*(d.-K^2)-im*K*ddf.(x),im*K*f.(x[1:N-1]).*ud[1:N-1])
    I_22=vcat(N+2:2*N,N+1:2*N,N+1:2*N-1)
    J_22=vcat(N+1:2*N-1,N+1:2*N,N+2:2*N)
    A_22=vcat(-im*ar*K*f.(x[2:N]).*ld[2:N],-im*ar*K*f.(x).*(d.-K^2)+im*ar*K*ddf.(x),-im*ar*K*f.(x[1:N-1]).*ud[1:N-1])
    Imap=vcat(I_11,I_12,I_21,I_22)
    Jmap=vcat(J_11,J_12,J_21,J_22)
    Amap=vcat(A_11,A_12,A_21,A_22)
    A=dropzeros(sparse(Imap,Jmap,Amap))
    return A,B,x,dx
end

function _normalize(v,N)
    zeropos=convert(Int64,(N+1)/2)
    return v*exp(-im*angle.(v[zeropos]))/maximum(abs.(v))
end

# Original's eigenmode: shift-invert Arnoldi about `guess`, selecting :SR of
# mu = 1/(gamma - guess), i.e. the eigenvalue with Re(gamma) < Re(guess) that
# maximizes (Re guess - Re gamma)/|gamma - guess|^2 -- NOT provably the fastest mode.
# Returns (gamma, eigenvector [psi; phi], KrylovKit ConvergenceInfo).
function eigenmode(S,K,ar,dxmin,xlim,eps,whichf,guess,miter; seed=1234)
    A,B,x,dx=operators(S,K,ar,dxmin,xlim,eps,whichf)
    N=length(x)
    F=lu(A-guess*sparse(B))
    rng=MersenneTwister(seed)
    sol=eigsolve(v->F\(B*v),rand(rng,ComplexF64,2*N),1,:SR,maxiter=miter)
    sol[2][1]=_normalize(sol[2][1],N)
    return 1/sol[1][1]+guess,sol[2][1],sol[3]
end

# NEW (change 6): eigenvalue nearest a (complex) shift sigma: :LM of 1/(gamma - sigma).
function eigenmode_shift(S,K,ar,dxmin,xlim,eps,whichf,sigma,miter; seed=1234, howmany=1)
    A,B,x,dx=operators(S,K,ar,dxmin,xlim,eps,whichf)
    N=length(x)
    F=lu(A-sigma*sparse(B))
    rng=MersenneTwister(seed)
    sol=eigsolve(v->F\(B*v),rand(rng,ComplexF64,2*N),howmany,:LM,maxiter=miter)
    vals=[1/m+sigma for m in sol[1]]
    vecs=[_normalize(v,N) for v in sol[2]]
    return vals,vecs,sol[3]
end

# Original's bruteforce, with the undefined-f bug fixed (change 3): full dense spectrum.
function bruteforce(S,K,ar,dxmin,xlim,eps,whichf)
    A,B,x,dx=operators(S,K,ar,dxmin,xlim,eps,whichf)
    return eigen(Matrix(B) \ Matrix(A))
end

# NEW (change 7).
function residual(S,K,ar,dxmin,xlim,eps,whichf,gamma,v)
    A,B,x,dx=operators(S,K,ar,dxmin,xlim,eps,whichf)
    Bv=B*v
    return norm(A*v-gamma*Bv)/norm(gamma*Bv)
end

function estimate_delta_noflow(S,K)
    if K < S^-0.25
        (S*K)^(-1.0/3.0)
    else
        S^-0.4 * K^-0.6
    end
end

function DeltaPrime(K)
    2.0*(1.0 ./K.-K)
end

function goodguess(S,K,ar,whichf)
    if whichf!=1
        Ktr=S^(-1.0/7.0)*(1-ar^2)^(-1.0/7.0)
        #Ktr=S^(-1.0/7.0)
        if ar<1
            if K<=Ktr
                gg=10.0*(1-ar^2)^(2.0/3.0)*S^(-1.0/3.0)*K^(2.0/3.0)
            else
                gg=5.0*S^(-0.5)*(1-ar^2)^0.5 * K^-0.5
            end
        elseif ar==1.0
            gg=0.1*S^-0.5
        else
            gg=0.2*ar
        end
    else
        if ar<1
            gg=S^(-0.5)*(1-ar^2)^0.5
        elseif ar==1.0
            gg=0.1*S^-0.5
        else
            gg=0.2*ar
        end
    end
    gg

end
