#!/bin/bash
# Production multi-GPU forced 3D RMHD on NCSA Delta: comm_backend="jax" (shard_map/NCCL)
# over 4 A100s (NVLink, full-rate fp64) on one gpuA100x4 node.  Submit as-is:
#
#     sbatch slurms/delta_forced_turbulence.sh
#
# and resubmit the same script to continue the run -- the driver restarts from the newest
# snapshot in TARANIS_SNAPDIR.  Setup and the Delta-vs-Savio differences:
# docs/DELTA_GPU_SETUP.md; the reasoning behind every export: docs/SAVIO_GPU_SETUP.md §4.
#
# STATUS 2026-11-03: drafted, not yet run (docs/DELTA_GPU_SETUP.md §4 is the ladder).
#
# Charging is per GPU-hour on RESERVED resources: 4 A100s = the whole node = 4 SU/h on
# gpuA100x4 (half that on gpuA100x4-preempt, which the resubmit driver tolerates).  For more
# nodes change --nodes and keep --ntasks-per-node=4 --gpus-per-node=4 (one task per GPU);
# nz must stay divisible by the TOTAL GPU count; multi-node also wants the Slingshot NCCL
# lines below.
#
#SBATCH --job-name=turb_delta
#SBATCH --account=bihz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=turb_delta_%j.out
#SBATCH --error=turb_delta_%j.err

set -uo pipefail

# Cray PE: reset, never purge.  No anaconda module on Delta; miniforge3-python is the base.
module reset
module load nvhpc-hpcx-cuda12-openmpi4/26.5 miniforge3-python
TARANIS_ENV=${TARANIS_ENV:-/projects/bihz/$USER/jax_gpu}     # UNVERIFIED path, see doc §2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$TARANIS_ENV"

unset PYTHONPATH               # harmless here; kept so the Savio and Delta scripts diff cleanly
export PYTHONNOUSERSITE=1
echo "python=$(which python)"

# jax's CUDA plugin sometimes fails to dlopen the pip-bundled nvidia libs by bare soname
# inside a job -- put every nvidia/*/lib dir on the path.  Echoed as proof it ran.
NVLIBS=$("$TARANIS_ENV/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"

# NO NCCL_P2P_DISABLE here: that export worked around broken PCIe peer-to-peer on Savio's
# savio4_gpu nodes.  Delta's A100s are NVLinked -- bench/nccl_repro.py (ladder rung 2) is
# the check.  Do not copy the Savio line over.

# Multi-node only (Slingshot 11): the user guide's NCCL recipe.  The module exists
# (aws-ofi-nccl/1.14.2 and 1.19.2, survey 2026-11-03); UNVERIFIED that jax's pip-bundled
# NCCL picks the plugin up -- uncomment at ladder rung 5 and check NCCL_DEBUG=INFO for
# 'NET/OFI' (plugin in use) rather than 'NET/Socket'.
# module load aws-ofi-nccl/1.19.2
# export NCCL_SOCKET_IFNAME=hsn
# export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,NET

# Overlap collectives with compute: 1.31x at 16 GPUs, ~1.02x at 4, never worth omitting.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true}"

# Persistent JIT cache so a resubmitted continuation skips the first block's compile.
export RMHD_COMPILATION_CACHE=$HOME/.taranis_jit_cache

# Field precision, read at IMPORT time.  A100 fp64 is full rate -- this is why we are here.
export TARANIS_PRECISION=64

# The run.  Start at the Savio production grid (0.27 GB/GPU at fp64 -- small); doc §3 sizes
# the step up (1024^2 x 256 fits one node, 1024^2 x 512 does not).
export TARANIS_BACKEND=jax
export TARANIS_NX=512          # nx = ny
export TARANIS_NZ=128          # MUST be divisible by the total GPU count
export TARANIS_TEND=10.0
export TARANIS_TSNAP=0.5
export TARANIS_SCHEME=lsrk54
# Snapshots go to /work/hdd (1 TB, 2.55M-file quota): orbax writes many small files, and
# /projects and home carry a 750k-file cap.  Survey 2026-11-03: home 100 GB, /projects/bihz
# 500 GB (the env lives there), /work/hdd/bihz 1 TB, /work/nvme/bihz 500 GB.
export TARANIS_SNAPDIR=${TARANIS_SNAPDIR:-/work/hdd/bihz/$USER/taranis/turb_delta}

PY=$TARANIS_ENV/bin/python
REPO=$HOME/taranis
DRIVER=$REPO/examples/multigpu_forced_turbulence.py

# srun, NEVER mpirun.  --mpi=pmix (pmix_v5); pmi2 is listed by `srun --mpi=list` but aborts
# in MPI_Init_thread with this HPC-X build (probe case F).
# MPI transport (job 23491994, 2026-11-03, slurms/delta_mpi_transport_probe.sh): HPC-X's
# default UCX layer cannot reach peer ranks on a Slingshot node ("MPI_INIT has failed because
# at least one MPI process is unreachable"), and every UCX_TLS restriction and --mpi=pmi2 fail
# too.  OpenMPI's own ob1 with shared memory (vader) + tcp works; tcp is what multi-node will
# use (MPI is control plane only on the jax backend).  coll/ucc otherwise prints "UCC library
# doesn't support MPI_THREAD_MULTIPLE" x ranks and falls back -- disable it (and hcoll).
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_coll_ucc_enable=0
export OMPI_MCA_coll_hcoll_enable=0

MPI_MODE=${MPI_MODE:-pmix}

# NO --gpu-bind (the jax/NCCL backend needs every node GPU visible to every process;
# comms._local_device_ids pins each task to its node-local rank's ordinal).  Delta's own
# examples show --gpu-bind=closest -- do not add it for this backend.  Do NOT set
# CUDA_VISIBLE_DEVICES by hand; Slurm owns it.
time srun --mpi="$MPI_MODE" --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     "$PY" -u "$DRIVER"
