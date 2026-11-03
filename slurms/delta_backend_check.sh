#!/bin/bash
#SBATCH --job-name=backend_delta
#SBATCH --account=bihz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=backend_delta_%j.out
#SBATCH --error=backend_delta_%j.err

# Ladder rung 3 (docs/DELTA_GPU_SETUP.md §4): the Savio T9 correctness battery
# (slurms/test_backend_jax_gpu.sh) on one Delta gpuA100x4 node -- comm_backend="jax"
# (shard_map/NCCL) vs "mpi4jax" on 4 GPUs, same seed, plus a cross-backend restart in both
# directions, preceded by the bare NCCL repro (rung 2) as phase 0.  fp64 (full rate on
# A100, and it makes the tolerance sharp).  ~3 SU.
# STATUS 2026-11-03: drafted, not yet run.

module reset
module load nvhpc-hpcx-cuda12-openmpi4/26.5 miniforge3-python
TARANIS_ENV=${TARANIS_ENV:-/projects/bihz/$USER/jax_gpu}     # UNVERIFIED path, see doc §2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TARANIS_ENV"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
echo "python=$(which python)"

NVLIBS=$("$TARANIS_ENV/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"

MPI_MODE=${MPI_MODE:-pmix}   # pmix_v5 and pmi2 both exist here; flip if MPI_Init aborts

# One-task compute-node probe: what the job python sees BEFORE any phase runs.
srun --mpi=$MPI_MODE --ntasks=1 --cpus-per-task=16 "$TARANIS_ENV/bin/python" -c "import os,sys,nvidia,jax; print('probe PYTHONPATH=',os.environ.get('PYTHONPATH')); print('probe CVD=',os.environ.get('CUDA_VISIBLE_DEVICES')); print('probe nvidia=',nvidia.__path__); print('probe devices=',jax.devices())" || true

PY=$TARANIS_ENV/bin/python
REPO=$HOME/taranis
DRIVER="$REPO/tests/test_backend_jax_mpi.py"
OUT=/work/hdd/bihz/$USER/taranis/test_backend_delta   # small, but keep output off home/projects
NRANK=4

export TARANIS_PRECISION=64
export RMHD_PKG=$REPO
export RMHD_NX=64
export RMHD_NZ=32
export RMHD_TEND=1.0

# Two launch modes, one per backend (Savio §4): mpi4jax needs one visible GPU per task;
# the jax/NCCL backend must see every node GPU and pins itself by node-local rank.
LAUNCH_MPI4JAX="srun --mpi=$MPI_MODE --ntasks=$NRANK --cpus-per-task=$SLURM_CPUS_PER_TASK --gpu-bind=single:1"
LAUNCH_JAX="srun --mpi=$MPI_MODE --ntasks=$NRANK --cpus-per-task=$SLURM_CPUS_PER_TASK"

# Hang forensics: NCCL bring-up per rank to stderr, hangs become exit 124 after 15 min,
# python stacks every 120 s while hung.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
TMO="timeout 900"
export RMHD_DEBUG_HANG=1

# NO NCCL_P2P_DISABLE (Savio-only PCIe workaround; Delta's A100s are NVLinked).
# NO XLA_FLAGS: correctness run, no scheduler variable in the tolerance comparison.

rm -rf "$OUT"

# Phase 0 = ladder rung 2: NCCL alone, no taranis imports (bench/nccl_repro.py header has the
# knob ladder).  Every rank must print "psum OK" with NO NCCL_P2P_DISABLE.  A 4-GPU exclusive
# allocation is the scarce resource on gpuA100x4, so rungs 2 and 3 share one queue wait.
echo "=== phase 0: NCCL repro, 4 ranks, no taranis (expect 4x psum OK) ==="
time timeout -s KILL 600 $LAUNCH_JAX "$PY" -u "$REPO/bench/nccl_repro.py"
echo "phase 0 exit=$?"

echo "=== phase 1: fresh run, backend=mpi4jax (${NRANK} ranks) ==="
time $TMO $LAUNCH_MPI4JAX "$PY" -u "$DRIVER" mpi4jax "$OUT/mpi4jax"

echo "=== phase 2: fresh run, backend=jax, same seed ==="
time $TMO $LAUNCH_JAX "$PY" -u "$DRIVER" jax "$OUT/jax"

echo "=== phase 3: compare mpi4jax vs jax (expect rel < 1e-12) ==="
"$PY" -u "$DRIVER" --compare "$OUT/mpi4jax" "$OUT/jax"

echo "=== phase 4a: jax backend restarting from the mpi4jax-written snapshot ==="
time $TMO $LAUNCH_JAX "$PY" -u "$DRIVER" jax "$OUT/xr_jax" "$OUT/mpi4jax"

echo "=== phase 4b: mpi4jax restarting from the jax-written snapshot ==="
time $TMO $LAUNCH_MPI4JAX "$PY" -u "$DRIVER" mpi4jax "$OUT/xr_mpi4jax" "$OUT/jax"

echo "=== phase 5: compare the two cross-backend restarts ==="
RMHD_CMP_TOL=1e-10 "$PY" -u "$DRIVER" --compare "$OUT/xr_jax" "$OUT/xr_mpi4jax"

echo "=== done: every phase above must end in ALL PASS ==="
