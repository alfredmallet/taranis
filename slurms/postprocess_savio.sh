#!/bin/bash
#SBATCH --job-name=postproc
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=postproc_%j.out
#SBATCH --error=postproc_%j.err
#
# Post-process a run's snapshots ON Savio (energy time series + spectra), so only a few
# KB..MB of text/npz/png leave the cluster instead of the snapshot tree.
#
# Why savio3_htc and not savio3: savio3 is per-NODE charged (32 cores at 1 SU/core-hr held
# for the whole job even if one process uses 4), while the *_htc partitions are per-CORE
# (savio3_htc 2.67 SU/core-hr, savio4_htc 3.67 -- docs-research-it "Service Units").  Four
# cores for an hour here is ~11 SU; the same hour on one savio3 node is 32 SU.
#
# Size CORES BY MEMORY, not by compute: per-core partitions hand out memory per core
# (savio3_htc nodes are 384 GB / 40 cores = 9.6 GB/core; savio4_htc 4 GB/core, 8 GB/core
# with -C savio4_m512).  Measured peak RSS of the script (M1, flat layout, one snapshot at a
# time, 2026-11-10): 3.9x the in-memory state with parspec, 3.0x without, plus ~0.5 GB of
# jax baseline.  State = 2 fields x nz x nx x (ny/2+1) complex:
#     512^3  fp32   1.1 GB state ->  4.6 GB peak (3.7 GB --no-parspec),  ~5 s/snapshot
#    1024^3  fp32   8.6 GB state -> ~34 GB peak (~26 GB --no-parspec),  ~40 s/snapshot + read
#    1024^3  fp64  17.2 GB state -> ~68 GB peak (~52 GB --no-parspec)
# So 1024^3 fp32 wants ~48 GB with margin: 6 cores on savio3_htc (~16 SU/hr), or 12 cores on
# savio4_htc (~44 SU/hr).  XLA:CPU multithreads the FFTs, so the extra cores are not idle.
#
# Never on a login node, never on a GPU partition (fp64 is 1/32 rate there and the SU rate
# is higher), and never on `--ntasks` > 1: the script reads the full z range on ONE process
# (load_snapshot unions the per-rank slabs -- docs/checkpointing.md "Resharding").
#
# Usage:  sbatch slurms/postprocess_savio.sh /global/scratch/users/$USER/runs/my_run [out_dir]
#         TARANIS_PRECISION must match the run's (64 for production fp64 runs; the script
#         refuses on a mismatch).  Set PRECISION below or export it before sbatch.

SNAP_PATH=${1:?usage: sbatch postprocess_savio.sh SNAP_PATH [OUT_DIR]}
OUT_DIR=${2:-$SNAP_PATH/analysis}
PRECISION=${TARANIS_PRECISION:-64}

module purge
module load anaconda3 gcc openmpi
source activate jax_cpu          # the CPU env from docs/SAVIO_CPU_SETUP.md -- works on
export PYTHONNOUSERSITE=1        # snapshots from GPU runs too (same on-disk layout)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_PLATFORMS=cpu         # harmless in jax_cpu; needed if you use the jax_gpu env
export TARANIS_PRECISION=$PRECISION

cd ~/taranis
# one rank: mpi4py is installed in this env, so launch through mpirun as everywhere on Savio
mpirun -n 1 python examples/snapshot_timeseries.py "$SNAP_PATH" --out "$OUT_DIR"
