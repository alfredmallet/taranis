# taranis on NCSA Delta GPUs — setup and running multi-GPU

**Status (2026-11-10): rungs 1–3 of §4 PASSED on compute nodes — 1-GPU serial, 4-GPU NCCL with
NVLink P2P and no Savio workarounds, and the full mpi4jax-vs-jax backend battery with
cross-backend restarts (job 23738568) — and rung 4, one-node production at 512²×128 fp64
(job 23738721, 2026-11-11): 6580 steps in 364 s, ≤55 ms/step including compile, 21
snapshots on /work/hdd. Production on one Delta node WORKS. Next: a clean ms/step + 1/2/4-GPU
scaling with `bench/bench_phase1.py`, then rung 5 multi-node (aws-ofi-nccl, UNRUN).** Every step marked UNVERIFIED below is a guess that the bring-up ladder
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
# --no-build-isolation is REQUIRED: mpi4jax's extension takes its XLA FFI headers from the
# jaxlib pip can see at build time, and pip's isolated build env resolves the NEWEST jaxlib
# (0.12.x, FFI API 0.5) regardless of what the runtime env holds (0.11.2, API 0.4) — see
# rung 3 below (job 23492171). PIP_CONSTRAINT=<jax==0.11.2 pin> was tried 2026-11-10 with
# pip 26.2.2 and did NOT reach the build env (it still installed jaxlib-0.12.0.1 there). So
# build in the env itself, after installing mpi4jax's build deps into it:
pip install "setuptools>=82.0.1" wheel "nanobind>=2.0.0"
# same CUDA_HOME/CUDA_PATH/CUDA_ROOT and OMPI_CC exports as above — without CUDA_HOME the
# link fails with "cannot find -lcudart" (2026-11-03 and again 2026-11-10)
python -m pip install --no-cache-dir --no-build-isolation --force-reinstall --no-deps --no-binary=mpi4jax -v mpi4jax 2>&1 | grep -E "jaxlib/include|CUDA INFO|error|Successfully"
# every -I.../jaxlib/include must be under $TARANIS_ENV/lib/python3.12/site-packages, never
# /tmp/pip-build-env-*. CUDA INFO must name EITHER cuda/12.9 OR the env's own
# site-packages/nvidia/cuda_runtime (libs [':libcudart.so.12'], rpath into the env) — the
# latter is what the 2026-11-10 rebuild resolved, and is the same libcudart the jax CUDA
# plugin loads, so it is the better outcome. Verified building both bridges 2026-11-10.
python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"   # 0.11.2 0.11.2
```

Then, on a compute node (1 GPU, shared, instant — never on a login node, §2), verify the
rebuilt extension against the runtime jax BEFORE spending an exclusive-node queue wait:
`srun ... $PY -c 'import mpi4jax, jax; print(jax.devices())'`. A header mismatch is NOT a
warning: mpi4jax registers its FFI handlers during XLA backend init, so a bad build raises
`XLA FFI handler registration for mpi_*_ffi ... handler's API version (0.5) is incompatible
with the framework's API version (0.4)` from `jax.devices()` and poisons EVERY backend in the
process — CUDA and CPU, the `"jax"` comm backend included (`_mpi_compat` imports mpi4jax at
module scope whenever it is importable). The rung-1 `CUDA bridge OK` import check does not
catch this (the mismatch surfaces at backend init, not at import). **The
`--no-build-isolation` rebuild PASSED this check 2026-11-10**: `0.11.2 0.11.2` on the login
node, `0.11.2 [CudaDevice(id=0)]` with no FFI error from the 1-GPU shared-node `srun`.

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
| `--mpi=pmix` | `--mpi=pmix` only | HPC-X's PMIx client talks to Slurm's pmix_v5; `pmi2` aborts in `MPI_Init_thread` (probe case F) |
| OpenMPI transport defaults | `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp`, `coll_ucc_enable=0`, `coll_hcoll_enable=0` | HPC-X's default UCX layer finds no reachable peer on a Slingshot node (job 23491350; probe cases A/C/D/G all fail, `UCX_TLS` restrictions included). ob1+vader is intra-node shared memory, tcp carries multi-node; MPI is control plane only on the jax backend so this costs nothing there. ucc/hcoll off silences a per-rank `MPI_THREAD_MULTIPLE` error from a plugin that falls back anyway |
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
1024²×256 was the guess. **Measured 2026-11-11 (rung 4, job 23738721): ≤55.3 ms/step at
512²×128 fp64 on 4 A100s** — 6580 steps in 364.0 s wall INCLUDING the first compile and 20
orbax save-waits of 1.1–2.0 s, so the steady-state step is somewhere under ~50 ms; faster than
Savio's 4×A5000 at the same grid in fp32 (77 ms/step, docs/performance.md). The 1024²×256
figure is still a guess. `nz` must be divisible by the total GPU count.

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
   prints a per-case `exit=`. **Result (job 23491994, gpua002)**: default, `ucx` with
   `UCX_TLS=sm,tcp,self` or `posix,sysv,tcp,self`, and `--mpi=pmi2` all fail; `ob1` with
   `self,vader,tcp` and with `self,vader` pass (4× `rank i of 4 sum 4`). `ucx_info -d` does list
   posix/sysv/cma/tcp/cuda transports, so the UCX failure is in HPC-X's pml, not a missing
   transport — not pursued. Exports baked into both job scripts.
   **PASSED 2026-11-03 (job 23492171, node gpua061)** as phase 0 of the resubmitted backend
   check: all four ranks `init ok ... global=4`, then `psum OK got=10.0 want=10.0 PASS` ×4,
   `phase 0 exit=0`, in 29.5 s wall. NCCL 2.32.3+cuda12.9 on driver 13.0 chose
   `P2P/CUMEM/read` on every channel (`isAllDirectP2p 1`) with NO `NCCL_P2P_DISABLE` — the
   Savio PCIe workaround is confirmed unnecessary here. The `WatchTasksAsync failed ...
   Connection refused` warnings on stderr come after the PASS lines, at coordinator teardown,
   and are harmless.
3. **Backend correctness, 4 GPUs** (~3 SU): `sbatch slurms/delta_backend_check.sh` — the five
   phase mpi4jax vs jax battery with cross-backend restarts, fp64, 64²×32. Every phase must
   end in `ALL PASS`. This is the only real exercise of `comms.Runtime` bringing up
   `jax.distributed`, the `shard_call` boundary and the global snapshot layout on this
   machine.
   **First attempt FAILED 2026-11-03 (job 23492171, gpua061, same job as the rung-2 PASS)**:
   phases 1, 2, 4a, 4b all died in 7–14 s before any stepping, every rank with
   `jax.errors.JaxRuntimeError: INVALID_ARGUMENT: XLA FFI handler registration for
   mpi_<op>_ffi on platform CUDA ... handler's API version (0.5) is incompatible with the
   framework's API version (0.4)` for every mpi4jax op, then `RuntimeError: Unable to
   initialize backend 'cpu'` from `jax.devices()` (`comms.get_mesh`, and the test's own
   header print). Phases 3 and 5 then tracebacked on `os.listdir` of the never-written
   `/work/hdd/bihz/alfyma/taranis/test_backend_delta/mpi4jax` (a secondary symptom; the
   compare should say "no output from phase N" instead — not yet changed). Cause: the
   2026-11-03 mpi4jax rebuild after the jax pin compiled its extension against jaxlib 0.12.x
   headers, which pip's isolated build env pulled in, while the runtime jaxlib is 0.11.2 —
   so the pin was right and the rebuild recipe was wrong. The `"jax"` backend phases fell
   with it because the broken handlers register during backend init for every platform.
   Not a Delta/NCCL/MPI problem: phase 0 in the same job proves the GPU, NCCL, PMIx and
   transport layers. Fix: the `--no-build-isolation` rebuild + 1-GPU `jax.devices()` check in
   §2, both done and passed 2026-11-10.
   **PASSED 2026-11-10 (job 23738568, node gpua034)** — every phase `ALL PASS`, fp64, 64²×32:
   phase 0 `psum OK` ×4 again; phase 1 mpi4jax headers `platform=gpu
   local_devices=[CudaDevice(id=0)] global_devices=1` on all four ranks (one visible GPU per
   rank is what `--gpu-bind=single:1` is for); phase 2 jax headers `global_devices=4` with
   distinct `CudaDevice(id=0..3)`; phase 3 rel 1.6e-16…3.7e-16 (tol 1e-12), final t
   1.0048195168440754 identical; phases 4a/4b restarted cross-backend from each other's
   snapshots on `/work/hdd`; phase 5 rel 5.8e-16…8.1e-16 (tol 1e-10), final t
   2.0074275650375384 identical. ~28 s per 4-rank phase wall (mostly compile; 64² is tiny).
   stderr noise to ignore: the `WatchTasksAsync ... Connection refused`/`CANCELLED` lines at
   each phase's coordinator teardown, mpi4jax's `Not using CUDA-enabled MPI` warning (HPC-X
   IS CUDA-aware — §5 open item), and orbax's `Waiting for previous save to complete took
   1.8 s` (snapshot cadence of the test, not a production setting). The `"jax"` comm backend,
   `comms.Runtime`'s `jax.distributed` bring-up under PMIx, the `shard_call` boundary and the
   global snapshot layout are therefore all verified on Delta.
4. **Production, one node** (4 SU/h): `sbatch slurms/delta_forced_turbulence.sh` at its
   default 512²×128 fp64 for an hour. Check the per-rank header lines (`platform=gpu`, one
   distinct local device each, `global_devices=4`), then the ms/step against the §3
   estimate and `bench/hlo_audit.py` for the latency-hiding overlap. Then raise `TARANIS_NX`
   / `TARANIS_NZ`.
   **PASSED 2026-11-11 (job 23738721)** at the defaults, from rest: headers `[rank i/4]
   backend=jax platform=gpu local_devices=[CudaDevice(id=i)] global_devices=4 precision=64
   grid=512^2x128` with distinct ids, `z split 4 ways: nz_local = 32`, `nblock=20`, `lsrk54`;
   `Ending simulation at t = 10.024…`, `E_kin=4.5655e-01 E_mag=4.7630e-01`, snapshots 0–20 on
   `/work/hdd`, `wall 364.0 s`, `DONE`, 7m02s job wall; ~0.4 SU. `.err` holds only the known
   noise (Lmod reset warning, orbax `Waiting for previous save` 1.1–2.0 s at each of the 20
   snapshots — the test cadence `TARANIS_TSNAP=0.5` is one snapshot per ~16 s, not a
   production setting — and the `WatchTasksAsync` teardown lines). **Step count and ms/step
   come from the per-block `t` lines** `simulate_scan` prints (one per scanned block of
   `nblock=20` steps): 329 blocks = 6580 steps → 55.3 ms/step including compile and save-waits
   (§3). dt went from the quiescent 0.0128 in the first block to ~0.0013 once the turbulence
   developed (median 0.0013 over the run). The clean steady-state number and the 1/2/4-GPU
   NVLink scaling curve come from **`sbatch slurms/delta_bench.sh`** (written 2026-11-11, UNRUN;
   ~1.5 SU): `bench/bench_phase1.py` on the same 3d_forced lsrk54 case as the Savio tables —
   jax at 1/2/4 GPUs fp64, the 4-GPU variants (latency-hiding flag off, unrolled stages,
   unforced, fp32 twin), mpi4jax at 1/4 (+ one CUDA-aware-MPI shot), an HLO audit and a
   profile. `grep ms/step` the `.out`; record in docs/performance.md "NCSA Delta" and fix §3.
   One oddity: this job's `python=$(which python)` header echoed the BASE miniforge python
   (`/sw/rh9.4/python/miniforge3/bin/python`) after `conda activate`, where job 23738568's
   identical lines echoed the env's — harmless because every launch uses the explicit
   `PY=$TARANIS_ENV/bin/python`, never bare `python`; keep it that way.
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
