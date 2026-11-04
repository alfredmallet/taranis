#!/bin/bash
#SBATCH --job-name=mpi_probe_delta
#SBATCH --account=bihz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=mpi_probe_delta_%j.out
#SBATCH --error=mpi_probe_delta_%j.err

# Which MPI transport does HPC-X OpenMPI accept on a Delta GPU node?  Job 23491350
# (2026-11-03) died in MPI_Init on every multi-rank phase -- "at least one MPI process is
# unreachable from another": the default UCX layer knows no Slingshot fabric and does not
# fall back to shared memory/TCP by itself.  This job runs 4 ranks (the production layout
# on one node) on ONE GPU's share of a SHARED node, so it starts in minutes, not hours, and
# tries one OMPI_MCA setting per case.  Read the .out: the first case with four "rank i of
# 4 sum 4" lines and exit=0 is what goes into the production scripts' exports.  ~0.25 SU.
# docs/DELTA_GPU_SETUP.md §3.

module reset
module load nvhpc-hpcx-cuda12-openmpi4/26.5
export PYTHONNOUSERSITE=1
PY=${TARANIS_ENV:-/projects/bihz/$USER/jax_gpu}/bin/python
LOGDIR=$HOME/mpi_test/$SLURM_JOB_ID
mkdir -p "$LOGDIR"
echo "python=$PY  logs=$LOGDIR  node=$(hostname)  ntasks=$SLURM_NTASKS"

echo "=== ompi_info: transport components ==="
ompi_info | grep -E "MCA (pml|btl|mtl):"
echo "=== ucx_info -d: devices/transports seen ==="
ucx_info -d 2>&1 | grep -E "^#\s+(Transport|Device):" | sort | uniq -c | head -40

C='from mpi4py import MPI; c=MPI.COMM_WORLD; print("rank", c.Get_rank(), "of", c.Get_size(), "sum", c.allreduce(1), flush=True)'

run() {  # run <case-name> VAR=val ...   (MPI_MODE may be set in the caller's environment)
  local name=$1; shift
  echo "=== $name   [${*:-defaults}]   --mpi=${MPI_MODE:-pmix} ==="
  env "$@" timeout -s KILL 90 srun --mpi="${MPI_MODE:-pmix}" --ntasks="$SLURM_NTASKS" \
      --cpus-per-task="$SLURM_CPUS_PER_TASK" "$PY" -c "$C" > "$LOGDIR/$name.log" 2>&1
  echo "exit=$?"
  grep -E "^rank|unreachable|rror" "$LOGDIR/$name.log" | sort | head -12
}

run A_default
run B_ob1_vader_tcp    OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
run C_ucx_sm_tcp       OMPI_MCA_pml=ucx UCX_TLS=sm,tcp,self
run D_ucx_posix_tcp    OMPI_MCA_pml=ucx UCX_TLS=posix,sysv,tcp,self
run E_ob1_vader_only   OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
MPI_MODE=pmi2 run F_pmi2_ob1_vader_tcp OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
# why the default fails, for the record (long log; only the summary lines are echoed)
run G_default_verbose  OMPI_MCA_pml_base_verbose=100 OMPI_MCA_btl_base_verbose=100

echo "=== done.  Full per-case logs in $LOGDIR ==="
