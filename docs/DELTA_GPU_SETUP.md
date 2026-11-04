# taranis on NCSA Delta GPUs — setup and running multi-GPU

**Status (2026-11-03): drafted from the Delta user guide and a login-node survey; NOT yet run
on a compute node.** Every step marked UNVERIFIED below is a guess that the bring-up ladder
(§4) is meant to settle; fix the text as each rung passes. The reasoning behind every
environment export and launch flag is in `docs/SAVIO_GPU_SETUP.md` §3–§6 and is not
repeated here — this file carries only what Delta does differently, plus a complete recipe.

Production template once §4 passes: `slurms/delta_forced_turbulence.sh` +
`examples/multigpu_forced_turbulence.py` (the same driver as Savio; everything it needs is
an environment variable set in the job script).

## 0. Why Delta

Savio's FCA caps a user at 4 A5000s (24 GB, fp64 at 1/32 rate) with full-rate fp64 only on
two V100s. Delta's `gpuA100x4` nodes carry four 40 GB A100 SXMs with NVLink between every
pair — full-rate fp64 — 64 EPYC cores and 256 GB host memory, with 100 such nodes and a 48 h
walltime. That is the "fp64 GPU production" hardware `docs/performance.md` says the
`comm_backend="jax"` path needs. Inter-node is Slingshot 11.

## 1. Account, partitions, budget

Login-node survey, 2026-11-03 (`accounts`, `sinfo`, `srun --mpi=list`, `module spider`):

- Account: **`bihz-delta-gpu`**, 1000 GPU-hours deposited. Charging is per GPU-hour times a
  partition factor, on RESERVED resources. 1 SU = 1 GPU + 16 cores + 62.5 GB on the 4-way
  nodes, so a 4-GPU job is the whole node and costs 4 SU per wall-clock hour on `gpuA100x4`.

| Partition | GPU | per node | factor (SU / GPU·h) | notes |
|---|---|---|---|---|
| `gpuA100x4` | A100 SXM 40 GB, NVLink | 4 | 1.0 | **production, fp64** |
| `gpuA100x4-interactive` | same | 4 | 2.0 | 1 h, 4 nodes max — twice the price, use sbatch instead |
| `gpuA100x4-preempt` | same | 4 | 0.5 | 10 min guaranteed; the resubmit-to-continue driver suits it |
| `gpuA40x4` | A40 PCIe 48 GB, no NVLink | 4 | 0.5 | fp32 only (fp64 at 1/32) |
| `gpuA100x8` | A100 SXM 40 GB | 8 | 1.5 | 6 nodes |
| `gpuH200x8` | H200 141 GB | 8 | 3.0 | 1 node max — not for us |

Budget arithmetic for the ladder: the 4-GPU correctness job (§4 rung 3, 40 min) is ~3 SU; a
48 h production run on one node is 192 SU; a 1 h 16-GPU scaling test is 16 SU. Node sharing
is the DEFAULT on Delta, so always ask for the whole node (`--exclusive --mem=0`) when taking
all four GPUs — it costs the same and keeps host memory for orbax. Default walltime is 30
min and default memory 1000 MB/core: set both explicitly.

Queue state on 2026-11-03: 63/100 A100 nodes backfill-planned, 25 allocated, 11 mixed. Keep
test walltimes short so they backfill.

Slurm supports `--mpi=pmix` (pmix_v5), `pmi2` and `cray_shasta`.

## 2. Build the `jax_gpu` env (login node, once)

Delta is a Cray Programming Environment (Lmod + CPE): use `module reset`, not `module purge`
(purge drops the craype defaults the compiler wrappers need). There is NO anaconda module;
`miniforge3-python` is the conda base (conda 25.7, base `/sw/rh9.4/python/miniforge3`). The only OpenMPI is NVIDIA's HPC-X bundle, built with
the nvhpc compilers — force gcc through OpenMPI's own `OMPI_CC` override so mpi4py/mpi4jax
compile with gcc, not `nvc` (confirmed: `OMPI_CC=gcc mpicc --version` → GCC 14.2.1). The bundle
loads its own `cudatoolkit` (12.9, matching the `jax[cuda12]` wheels) — do NOT also load
`cuda/12.9`: Lmod refuses the conflict and, because a load line is all-or-nothing, silently
drops `miniforge3-python` with it, after which `pip` is the system Python 3.9's (2026-11-03,
first attempt). Avoid the cuda13 bundles. Run the recipe in stages and check each `which`.

Storage (survey 2026-11-03, `quota`): home `/u/$USER` 100 GB / 750k files; `/projects/bihz`
500 GB / 750k files (shared by the allocation; `/projects/bihz/$USER` exists); `/work/hdd/bihz`
1 TB / 2.55M files; `/work/nvme/bihz` 500 GB. The env goes in `/projects/bihz/$USER/jax_gpu`.
**Snapshots go under `/work/hdd/bihz/$USER`** — orbax writes many small files and the
750k-file caps on home and projects are the binding limit, not bytes.

```bash
module reset
module load gcc-native/14 nvhpc-hpcx-cuda12-openmpi4/26.5 miniforge3-python
module list; which conda mpicc nvcc    # all three must resolve before going on
export OMPI_CC=gcc OMPI_CXX=g++      # HPC-X's mpicc wraps nvc by default
export PYTHONNOUSERSITE=1
export TARANIS_ENV=/projects/bihz/$USER/jax_gpu

# first time only: miniforge3-python's conda is not shell-initialised. `conda init bash`,
# `source ~/.bashrc`, then turn OFF base auto-activation so login shells and jobs do not
# get base conda's python ahead of the module stack (the Savio PYTHONPATH lesson again):
#   conda init bash && source ~/.bashrc && conda config --set auto_activate_base false
conda create -p "$TARANIS_ENV" python=3.12 -y
conda activate "$TARANIS_ENV"

pip install -U "jax[cuda12]"
# from source against THIS mpicc, no cached wheel — same two rules as Savio §2
MPICC=$(which mpicc) python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py
# Delta's DEFAULT environment loads cudatoolkit/26.5_13.2, so CUDA_HOME, CPATH and the
# CRAY_CUDATOOLKIT_* variables point at CUDA 13.2 while the HPC-X cuda12 bundle puts 12.9 on
# PATH/LD_LIBRARY_PATH (12.9 first). mpi4jax's detection then lands on .../26.5/compilers/
# and the link fails with "cannot find -lcudart" (2026-11-03). Override CUDA_HOME for the
# build (runtime is fine: the jax plugin wants .so.12 sonames, which only the 12.9 tree has):
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/cuda/12.9
export CUDA_PATH=$CUDA_HOME CUDA_ROOT=$CUDA_HOME
python -m pip install --no-cache-dir --no-binary=mpi4jax -v mpi4jax 2>&1 | grep -E "CUDA INFO|error|Successfully"
# the CUDA INFO line must name cuda/12.9/include and cuda/12.9/lib64
pip install orbax-checkpoint tensorstore numpy matplotlib pytest
cd ~/taranis && pip install -e .

timeout 30 python -c "import jax; print('jax', jax.__version__)"
timeout 30 python -c "import mpi4py; print(mpi4py.__version__, mpi4py.get_config())"   # no MPI_Init; 4.x has no 'mpicc' key
```

**Do not `from mpi4py import MPI` (or import mpi4jax) on a login node**: the HPC-X singleton
`MPI_Init` hangs there (2026-11-03) — the login nodes lack the fabric the bundle probes — and
OpenMPI swallows SIGTERM, so plain `timeout` does not rescue the shell: always
`timeout -s KILL` around anything that initialises MPI.
The MPI and CUDA-bridge checks are part of the rung-1 job below, where `srun` supplies PMIx.

Versions that resolved on 2026-11-03: python 3.12.16, mpi4py 4.1.2, mpi4jax 0.9.1.post2, numpy
2.5.4, scipy 1.18.1, and jax/jaxlib 0.12.0.1 — which mpi4jax 0.9.1 does not support (its
ceiling is 0.11.2; it warns at import). Pin jax and rebuild mpi4jax against it:

```bash
pip install "jax[cuda12]==0.11.2"
python -m pip install --no-cache-dir --force-reinstall --no-deps --no-binary=mpi4jax mpi4jax   # same CUDA_HOME/OMPI_CC exports as above
```

**Check `which pip` before every install.** With base auto-activation on, a fresh login shell
sits in the read-only system `(base)` env and pip fails with `Permission denied:
/sw/rh9.4/python/miniforge3/...` (2026-11-03) — harmless, but nothing is installed. Filter pip
output with `grep -i` — it prints `ERROR` in capitals.

Still newer than Savio's 0.10.2 and the laptop's 0.10.0; a tolerance difference in rung 3 would
be the first sign it matters.
If `python=3.12` is refused by any wheel, drop to 3.11 (Savio's). If the CUDA bridge import
fails, see Savio §2 (`CUDA_ROOT`, rerun with `-v`). UNVERIFIED: whether `miniforge3-python`'s
`conda` needs `conda init`/`source $(conda info --base)/etc/profile.d/conda.sh` inside a
batch script — the job scripts source that file explicitly, which works either way.

Single-GPU visibility check (inside an allocation — login nodes have no GPUs). This is
§4 rung 1, ~0.2 SU:

```bash
conda deactivate            # or the miniforge module warns about the forwarded CONDA_PREFIX
PY=/projects/bihz/$USER/jax_gpu/bin/python
srun -A bihz-delta-gpu -p gpuA100x4 --gpus-per-node=1 --ntasks=1 --cpus-per-task=16 \
     --mem=60G -t 00:10:00 --mpi=pmix --pty \
  bash -c "module reset; module load nvhpc-hpcx-cuda12-openmpi4/26.5; export PYTHONNOUSERSITE=1;
           $PY -c 'import jax; print(jax.default_backend(), jax.devices())';
           timeout -s KILL 60 $PY -c 'from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0], MPI.COMM_WORLD.Get_size())'; echo mpi exit=\$?;
           timeout -s KILL 60 $PY -c 'import mpi4jax._src.xla_bridge.mpi_xla_bridge_cuda; print(\"CUDA bridge OK\")'; echo bridge exit=\$?"
```

`bash -c` never reads `.bashrc`, so `conda activate` fails inside it ("Run 'conda init' before
'conda activate'", 2026-11-03) — call the env's python by absolute path, or source
`$(conda info --base)/etc/profile.d/conda.sh` first as the job scripts do.

Expect three lines: `cuda [CudaDevice(id=0)]`, an `Open MPI v4.1…` line ending in `1`, and
`CUDA bridge OK`. If the MPI line is killed (exit 137) under `--mpi=pmix`, rerun with `pmi2`
— that decides `MPI_MODE` for every job script. `cpu [...]` means the plugin did not load — read the stderr,
and try the `NVLIBS` block from the job scripts (every pip-bundled `nvidia/*/lib` on
`LD_LIBRARY_PATH`).

## 3. What differs from Savio at launch

Everything in Savio §4 still applies (one task per GPU, `srun` never `mpirun`, no
`--gpu-bind` for the jax backend, `--gpu-bind=single:1` for mpi4jax, never set
`CUDA_VISIBLE_DEVICES` by hand, `XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true`
on anything multi-GPU, `TARANIS_PRECISION` exported before the process starts). The
differences:

| Savio | Delta | why |
|---|---|---|
| `--gres=gpu:A5000:4 --qos=...` | `--gpus-per-node=4`, no QoS | Delta's idiom; no FCA gres-type rule |
| `--cpus-per-task=4` | `--cpus-per-task=16` | 16 cores per GPU is the charge unit; the user guide's own 4-GPU example |
| per-core charging | `--exclusive --mem=0` | node sharing is default; whole node costs the same 4 SU/h |
| `module purge; module load anaconda3 gcc openmpi` | `module reset; module load nvhpc-hpcx-cuda12-openmpi4/26.5 miniforge3-python` | CPE stack, no anaconda |
| `unset PYTHONPATH` | not needed | no anaconda module to leak it — kept harmlessly in the scripts |
| `export NCCL_P2P_DISABLE=1` | **DROPPED** | that was a Savio PCIe ACS bug; here the GPUs are NVLinked. Confirm with `bench/nccl_repro.py` (rung 2) before trusting any timing |
| `--mpi=pmix` | `--mpi=pmix` (confirmed, rung 1) | both exist; HPC-X's PMIx client talks to Slurm's pmix_v5. `pmi2` stays as the `MPI_MODE` fallback in the scripts, unused |
| `TARANIS_PRECISION=32` | **`64`** | A100 fp64 is full rate; that is the point of being here |
| multi-node on TCP | `module load aws-ofi-nccl/1.19.2` + `NCCL_SOCKET_IFNAME=hsn` (module confirmed present; UNVERIFIED with jax's pip NCCL) | the user guide's PyTorch example for Slingshot 11; look for `NET/OFI` in `NCCL_DEBUG=INFO` output, `NET/Socket` means the plugin was not picked up (`NCCL_NET_PLUGIN`). Single-node runs need none of it |
| `savio4_gpu` is the only pool | `gpuA100x4-preempt` at half price | the driver already restarts from the newest snapshot, so preemption costs one block |

`comms._local_device_ids` splits Slurm's visible-device list by node-local rank, so
`--gpus-per-node=4` with no binding gives each of the four tasks `CUDA_VISIBLE_DEVICES=0,1,2,3`
and the pinning works unchanged. Delta's own examples show `--gpu-bind=closest`; with the jax
backend that is the Savio "invalid device ordinal" failure — do not add it.

**Sizing at fp64 on 40 GB.** One state array is `2·nz_local·nx·(nx/2+1)·16` bytes; budget
~10× per GPU (Savio §5). On 4 A100s: 512²×128 is 0.27 GB/GPU (tiny — the Savio production
grid, now at fp64), 1024²×256 is 2.2 GB/GPU → ~22 GB with workspace and fits; 1024²×512 does
not on one node. Expect roughly the Savio fp32 step time at the same grid (A100 fp64 moves
twice the bytes at twice the bandwidth), so ~80 ms/step at 512²×128 and ~0.6 s/step at
1024²×256 — UNVERIFIED, measure. `nz` must be divisible by the total GPU count.

## 4. Bring-up ladder

Each rung is cheap; do not skip one. Record what each printed next to its UNVERIFIED items.

1. **One GPU, serial backend** (~0.2 SU): the §2 visibility `srun`. Settles the plugin load,
   the conda activation inside a job, and whether `NVLIBS` is needed. **PASSED 2026-11-03**
   (job 23491131): `gpu [CudaDevice(id=0)]` with no `NVLIBS`, `Open MPI v4.1.9a1` size 1 under
   `--mpi=pmix` (so `MPI_MODE=pmix` is confirmed), `CUDA bridge OK`. mpi4jax 0.9.1.post2
   warned that its newest supported jax is 0.11.2 against the installed 0.12.0.1 — jax was
   pinned to 0.11.2 and mpi4jax rebuilt before rung 3 (done 2026-11-03: `0.11.2 0.11.2`), so
   that the mpi4jax reference the backend check compares against is a supported pairing.
2. **NCCL alone, 4 GPUs** (~1 SU): `salloc -A bihz-delta-gpu -p gpuA100x4 -N1 -n4 -c16
   --gpus-per-node=4 --exclusive --mem=0 -t 0:20:00`, then the module lines and
   `srun --mpi=pmix --ntasks=4 $PY bench/nccl_repro.py`. Must print `psum OK` with NO
   `NCCL_P2P_DISABLE`. **A 4-GPU exclusive allocation is the scarce resource** (2026-11-03:
   the 1-GPU rung-1 `srun` started instantly on a shared node; the exclusive `salloc` queued
   behind 100 fully planned nodes), so the same check is also phase 0 of
   `slurms/delta_backend_check.sh` — one queue wait for rungs 2 and 3 together. If it hangs, walk the knob ladder in that file's header — but on NVLink
   hardware a hang points at the PMIx/coordinator layer first: try `MPI_MODE=pmi2`, then
   `RMHD_REPRO_NOMPI=1`.
   **First attempt FAILED 2026-11-03 (job 23491350, node gpua090)**: every phase (the NCCL
   repro included) died ~5 s in with `MPI_INIT has failed because at least one MPI process is
   unreachable from another` — HPC-X's default UCX transport finds no fabric it knows on a
   Slingshot node and does not fall back to shared memory/TCP by itself; a single rank (rung 1)
   never needs a transport, which is why it passed. The plugin/probe side was fine (all four
   GPUs visible, no PYTHONPATH). Fix: force a transport via `OMPI_MCA_*` — selected on a
   SHARED node with `--ntasks=2 --gpus-per-node=1` (instant start) rather than another
   exclusive-node queue wait: `sbatch slurms/delta_mpi_transport_probe.sh` (4 ranks, 1 GPU,
   ~0.25 SU) tries the candidate `OMPI_MCA_pml`/`btl`/`UCX_TLS` settings in one job and
   prints a per-case `exit=`; the winner goes into the job scripts' exports.
3. **Backend correctness, 4 GPUs** (~3 SU): `sbatch slurms/delta_backend_check.sh` — the five
   phase mpi4jax vs jax battery with cross-backend restarts, fp64, 64²×32. Every phase must
   end in `ALL PASS`. This is the only real exercise of `comms.Runtime` bringing up
   `jax.distributed`, the `shard_call` boundary and the global snapshot layout on this
   machine.
4. **Production, one node** (4 SU/h): `sbatch slurms/delta_forced_turbulence.sh` at its
   default 512²×128 fp64 for an hour. Check the per-rank header lines (`platform=cuda`, one
   distinct local device each, `global_devices=4`), then the ms/step against the §3
   estimate and `bench/hlo_audit.py` for the latency-hiding overlap. Then raise `TARANIS_NX`
   / `TARANIS_NZ`.
5. **Multi-node** (16 SU/h at 4 nodes): `--nodes=4 --ntasks-per-node=4 --gpus-per-node=4`,
   with the Slingshot NCCL plugin lines uncommented. Compare 4→16 scaling with Savio's 4.15×
   on TCP (docs/performance.md).

## 5. Open items

- Everything tagged UNVERIFIED above.
- `tests/test_backend_jax_mpi.py` and the `RMHD_PKG` A/B mechanism were written for
  `$HOME/taranis`; the Delta scripts assume the same checkout location.
- mpi4jax on HPC-X: HPC-X IS CUDA-aware, unlike Savio's OpenMPI (Savio §7), so
  `MPI4JAX_USE_CUDA_MPI=1` may remove the host staging on the mpi4jax backend. Irrelevant
  to the production `"jax"` path (MPI is control plane only there); worth one A/B if the
  mpi4jax backend is ever used for more than the correctness cross-check.
- No Delta CPU-partition recipe (the `mpi4jax` CPU production path) has been attempted.
