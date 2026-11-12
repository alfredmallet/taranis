#!/bin/bash
# Step-time benchmark for taranis on NCSA Delta: 1/2/4 A100 strong scaling of the jax
# (shard_map/NCCL over NVLink) backend at the fp64 production grid, with the mpi4jax
# cross-check backend, an fp32 twin, the latency-hiding flag off, unrolled stages, one HLO
# audit and one profile.  Submit as-is from ~/taranis:
#
#     sbatch slurms/delta_bench.sh
#
# then `grep ms/step delta_bench_<jobid>.out` is the table.  This is the like-for-like
# counterpart of slurms/bench_phase3_a5000.sh (Savio 4xA5000 fp32, docs/performance.md "GPU")
# and slurms/bench_xla_flags_2080.sh (16x2080Ti flag matrix): same bench/bench_phase1.py,
# same 3d_forced lsrk54 elsasser nps cfl1 case, same nb/nr step counts, so the rows are
# directly comparable.  Ladder rung 4 (slurms/delta_forced_turbulence.sh, job 23738721,
# 2026-11-11) gave only an upper bound -- 55.3 ms/step at 512^2x128 fp64 INCLUDING the first
# compile and 20 orbax save-waits -- because the production driver prints no step count; this
# job measures the compiled steady-state step on its own.
#
# Cost: one exclusive gpuA100x4 node = 4 SU/h.  ~22 cases x (compile ~20-40 s + 120 steps at
# ~50 ms) ~ 15-20 min -> ~1.5 SU; walltime below is a cap.  The 1- and 2-GPU rows idle the
# other GPUs -- that is the price of one queue wait for the whole curve (an exclusive 4-GPU
# node is the scarce resource here, docs/DELTA_GPU_SETUP.md section 4 rung 2).
#
# NOTE the persistent compilation cache (RMHD_COMPILATION_CACHE) is deliberately NOT set:
# cases differ by XLA_FLAGS and precision, and a cache keyed on shapes alone would serve one
# case's binary to another.  Every case pays its own compile.
#
# STATUS 2026-11-11: written from the two Savio scripts + the verified Delta env/launch block
# of slurms/delta_backend_check.sh; not yet run.  Record the table in docs/performance.md
# "NCSA Delta" and fix docs/DELTA_GPU_SETUP.md section 3 when it has.
#
#SBATCH --job-name=delta_bench
#SBATCH --account=bihz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=delta_bench_%j.out
#SBATCH --error=delta_bench_%j.err

set -uo pipefail   # not -e: one crashed case must not abort the matrix

# ------------------------------------------------------------------ env (verified block)
# Identical to slurms/delta_backend_check.sh (job 23738568 ALL PASS) and
# slurms/delta_forced_turbulence.sh (job 23738721).  Cray PE: reset, never purge.
module reset
module load nvhpc-hpcx-cuda12-openmpi4/26.5 miniforge3-python
TARANIS_ENV=${TARANIS_ENV:-/projects/bihz/$USER/jax_gpu}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TARANIS_ENV"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
# `which python` after activate has echoed the BASE miniforge python in one job (23738721)
# and the env's in another (23738568) -- so every launch below uses the explicit $PY, never
# a bare `python`.  The echo stays as a record of which it was this time.
echo "python=$(which python)  (informational; launches use PY=$TARANIS_ENV/bin/python)"

NVLIBS=$("$TARANIS_ENV/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"

# MPI transport (probe job 23491994): HPC-X's UCX pml cannot reach peer ranks on Slingshot;
# OpenMPI's ob1 + shared memory + tcp works.  coll/ucc and hcoll off (THREAD_MULTIPLE noise).
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_coll_hcoll_enable=0
MPI_MODE=${MPI_MODE:-pmix}   # pmi2 aborts in MPI_Init_thread with this HPC-X build

# NO NCCL_P2P_DISABLE: Delta's A100s are NVLinked and rung 2 (jobs 23492171, 23738568) showed
# NCCL choosing P2P/CUMEM on every channel with nothing set.
# Single node: no aws-ofi-nccl / NCCL_SOCKET_IFNAME (those are the multi-node rung-5 lines).

# Production XLA flag (slurms/delta_forced_turbulence.sh sets the same): the latency-hiding
# scheduler, worth 1.31x at 16 GPUs / ~1.02x at 4 on Savio.  The "nolhs" rows clear it.
LHS_FLAG="--xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_FLAGS="$LHS_FLAG"
export RMHD_REQUIRE_GPU=1    # abort a case where a rank silently fell back to CPU

# mpi4jax rows: host-staged transport by default.  HPC-X IS CUDA-aware (docs/DELTA_GPU_SETUP.md
# section 5 open item), so one row below tries MPI4JAX_USE_CUDA_MPI=1 -- if it crashes or
# hangs the timeout below ends it and the matrix continues; nothing production depends on it.
export MPI4JAX_USE_CUDA_MPI=0

PY=$TARANIS_ENV/bin/python
REPO=$HOME/taranis
BENCH=$REPO/bench/bench_phase1.py
AUDIT=$REPO/bench/hlo_audit.py
export RMHD_PKG=$REPO        # bench prints pkg= so the imported package is verifiable

NX=${NX:-512}; NZ=${NZ:-128} # the production grid; 0.27 GB/GPU per state at fp64 on 4 GPUs
STEPS="nb20 nr5"             # 20 warm + 100 timed steps per case
PASSES=${PASSES:-2}          # two passes so run-to-run spread is visible before a delta is believed
CASE_TIMEOUT=${CASE_TIMEOUT:-600}

# ---------------------------------------------------------------------------- launchers
# launch <ngpu> <script> <args...>; run <ngpu> <bench args...> is launch with $BENCH.  srun
# inside the allocation with --ntasks=n; the other GPUs sit idle for the 1- and 2-GPU rows.
# jax backend: NO --gpu-bind -- every node GPU stays visible and comms._local_device_ids pins
# each task to its node-local rank's ordinal (so a 1-task run uses GPU 0 and must print
# global_device_count=1).  mpi4jax backend: --gpu-bind=single:1, as the backend check uses.
# Do NOT set CUDA_VISIBLE_DEVICES by hand; Slurm owns it.  Precision is read at IMPORT, so
# TARANIS_PRECISION is set per case (default 64 -- the A100 is here for full-rate fp64).
# XLA_FLAGS given as a `VAR=... run ...` prefix reaches the srun'd python (bash exports a
# command-prefix assignment to the children of a function).
#
# GPU SCOPING FOR THE 1- AND 2-GPU ROWS.  At size 1 the jax backend does not initialise
# jax.distributed at all (comms.init_backend), so a 1-task run sees every GPU Slurm leaves
# visible and would silently shard the single process over all four -- a "1 GPU" row that is
# really 4.  Savio's scripts hid this with a step-level --gres; here the step gets --gpus=$n
# when n is below the node's GPU count, so the step's CUDA_VISIBLE_DEVICES holds exactly n
# devices (1 -> the nvis==1 branch; 2 -> split by node-local rank).  The 4-GPU line stays the
# bare, verified one (jobs 23738568/23738721).  CHECK the printed global_device_count against
# the row's GPU count before believing any 1- or 2-GPU number.
launch() {
    local n=$1; shift
    local bind="--gpu-bind=single:1" scope=""
    case "$*" in *backend=jax*) bind="";; esac
    [ "$n" -lt "${SLURM_GPUS_ON_NODE:-4}" ] && scope="--gpus=$n"
    timeout -s KILL "$CASE_TIMEOUT" \
    srun --mpi="$MPI_MODE" --ntasks="$n" --cpus-per-task="$SLURM_CPUS_PER_TASK" $scope $bind \
         "$PY" -u "$@" 2>&1 | grep -v "bit precision" || true
}
run() { local n=$1; shift; launch "$n" "$BENCH" "$@"; }

echo "=== config: grid=${NX}^2x$NZ steps='$STEPS' passes=$PASSES node=$(hostname) ==="
echo "===         XLA_FLAGS='$XLA_FLAGS'  mpi=$MPI_MODE  ob1/vader/tcp  no P2P workaround ==="
"$PY" -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"

# ------------------------------------------------------------------- phase 1: HLO audit
# Compile only: are the halo ppermutes and the two allreduces async, and does XLA schedule
# anything inside their -start/-done windows under the production flag?  `grep AUDIT`.
echo
echo "############ phase 1: HLO audit, 4 GPUs, lhs on vs off (compile only) ############"
export TARANIS_PRECISION=64
XLA_FLAGS="$LHS_FLAG" launch 4 "$AUDIT" nx$NX nz$NZ backend=jax label=lhs   --dump="$SLURM_SUBMIT_DIR/hlo_delta_$SLURM_JOB_ID"
XLA_FLAGS=""          launch 4 "$AUDIT" nx$NX nz$NZ backend=jax label=nolhs --dump="$SLURM_SUBMIT_DIR/hlo_delta_$SLURM_JOB_ID"

# ---------------------------------------------------------------- phase 2: timed matrix
echo
echo "############ phase 2: timed matrix ($PASSES passes) ############"
for pass in $(seq 1 "$PASSES"); do
    echo "=== pass $pass: jax backend, fp64, lhs on -- the production configuration ==="
    export TARANIS_PRECISION=64
    run 1 j1      3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
    run 2 j2      3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
    run 4 j4      3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
    echo "=== pass $pass: jax backend, 4 GPUs, variants ==="
    XLA_FLAGS="" \
    run 4 j4nolhs 3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS          # flag off
    run 4 j4unr   3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax unroll $STEPS   # lsrk_scan=False
    run 4 j4U     3d        donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS          # unforced: the forcing allreduce's cost
    export TARANIS_PRECISION=32
    run 4 j4f32   3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS          # fp32 twin: what fp64 costs on an A100
    run 1 j1f32   3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
    export TARANIS_PRECISION=64
    echo "=== pass $pass: mpi4jax backend, fp64 (the cross-check backend) ==="
    run 1 g1      3d_forced donate nx$NX nz$NZ nps cfl1 halo_late $STEPS
    run 4 g4      3d_forced donate nx$NX nz$NZ nps cfl1 halo_late $STEPS
done

echo
echo "=== mpi4jax with CUDA-aware MPI (HPC-X), 4 GPUs, one shot -- section 5 open item ==="
MPI4JAX_USE_CUDA_MPI=1 run 4 g4cuda 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late $STEPS

# -------------------------------------------------------------------- phase 3: profile
# One traced production case (rank 0 only; RMHD_PROFILE_ALL=1 traces every rank).  Reading
# it: docs/SAVIO_GPU_SETUP.md "Reading the profile" -- collectives on their own stream
# overlapping FFT kernels = real latency hiding; collectives sandwiched between kernels with
# idle gaps = serialization.
echo
echo "############ phase 3: profile, j4 production case ############"
export RMHD_PROFILE_DIR=$SLURM_SUBMIT_DIR/prof_delta_$SLURM_JOB_ID
run 4 j4prof 3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS --profile

# ------------------------------------------------------------------------------ wrap-up
echo
echo "############ compact results ############"
echo "--- AUDIT lines ---"
grep -h "^AUDIT" "$SLURM_SUBMIT_DIR/delta_bench_${SLURM_JOB_ID}.out" 2>/dev/null || echo "(grep AUDIT on this file)"
echo
echo "--- timing lines (label case nx nz ranks [tags] steps steps/s ms/step backend plat dev) ---"
grep -h "ms/step" "$SLURM_SUBMIT_DIR/delta_bench_${SLURM_JOB_ID}.out" 2>/dev/null || echo "(grep ms/step on this file)"
echo
echo "HLO dumps: $SLURM_SUBMIT_DIR/hlo_delta_$SLURM_JOB_ID/"
echo "profile:   $RMHD_PROFILE_DIR/"
echo "done."
