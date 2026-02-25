# Montage Pegasus + DFTracer

## Overview

This folder hosts the Montage Pegasus workflow integrated with DFTracer for I/O and application tracing. It follows the [DFTracer pegasus_montage tutorial](https://github.com/LLNL/dftracer/blob/develop/docs/pegasus_montage.rst).

Components are built/installed via **Spack where possible** (Python, AstroPy, Ant, OpenJDK) for consistency with other workload subprojects; externals from `packages.yaml` are used when available. Condor and Pegasus are installed from official tarballs (no Spack packages available). Montage is built from source (submodule). DFTracer patches can be applied later via `dftracer/apply_montage_patches.py` when tracing is needed.

## Layout

- `.spack-env/`, `.venv/`: Spack and Python environments
- `montage/`: Montage toolkit source (submodule)
- `montage-workflow-v3/`: Pegasus workflow DAX generator (submodule)
- `pegasus/`: Pegasus source for building pegasus-mpi-cluster (submodule)
- `install/`: Consolidated Condor + Pegasus + Montage binaries
- `logs/`: build/run/trace logs
- `output/`: workflow-generated mosaics
- `dftracer/`: DFTracer patches and knob config

## Source Pinning

- **montage**: `v6.0`
- **montage-workflow-v3**: latest from main
- **pegasus**: latest from main (for pegasus-mpi-cluster build)

## Spack Dependencies

Installed via Spack (see `spack.yaml`):

- `python@3.12`
- `py-astropy`
- `ant` (for pegasus-mpi-cluster)
- `openjdk`

## Quick Start

```bash
source ~/spack/share/spack/setup-env.sh
cd montage-pegasus

# 1. Spack + submodules + venv + Condor + Pegasus
./install.sh

# 2. Build Montage and pegasus-mpi-cluster
./build.sh

# 3. Run the workflow (starts Condor if needed, runs preflight checks)
./run.sh
```

`run.sh` automatically starts Condor if not running, runs preflight checks (Pegasus, Montage, network), and invokes `pegasus-configure-glite` when needed.

**Cleanup** — Use `./clean.sh` when you want to fully tear down: remove Pegasus workflows (`pegasus-remove`), stop Condor (`condor_off -master`), wipe workflow data (`montage-workflow-v3/work`, `montage-workflow-v3/data`), and reset Condor spool/execute dirs. Run this before a clean re-run or when debugging. It is idempotent.

## DFTracer Integration

To trace Montage with DFTracer, apply the patch first: run `dftracer/apply_montage_patches.py` (with `MONTAGE_SRC`, `DFTRACER_INCLUDE`, `DFTRACER_LIB` set), then rebuild. The workflow sets `DFTRACER_ENABLE=1`, `DFTRACER_BIND_SIGNALS=0` (for Pegasus), and writes traces to `logs/traces/<run_id>/`.

Traces can be analyzed with DFAnalyzer (available in the venv).

## Platform Notes

`install.sh` detects Ubuntu/RHEL and selects Condor/Pegasus tarballs. Override via:

- `CONDOR_PEGASUS_PLATFORM`: e.g. `ubuntu22`, `rhel8`
- `CONDOR_TARBALL_URL`, `PEGASUS_BINARY_URL`, `PEGASUS_WORKER_URL`

## Deploying on Tuolumne (LLNL)

Tuolumne is an LLNL HPC cluster (CORAL2, 1,100+ nodes, 96 cores/node, AMD MI300A). It uses **Flux** as its sole batch scheduler (no Slurm `sbatch`). The current Quick Start is designed for single-node Condor; below are notes for scaling to Tuolumne.

### Tuolumne Basics

- **Scheduler**: Flux (use `flux batch`, `flux run`, `flux alloc`, `flux jobs -A`)
- **Queues**: `pbatch` (batch, 256 nodes/job max, 24h), `pdebug` (interactive, 16 nodes/user, 1h)
- **Storage**: `/p/lustre5` (Lustre, shared across nodes)
- **Architecture**: x86_64, AMD EPYC, Cray compilers, TOSS 4
- **Docs**: [Tuolumne](https://hpc.llnl.gov/hardware/compute-platforms/tuolumne), [Flux Quick Start](https://hpc.llnl.gov/banks-jobs/running-jobs/flux-quick-start-guide)

### Why Flux Differs

HTCondor's BLAH/glite layer (used by Pegasus to submit to clusters) supports SLURM, PBS, SGE, LSF—**not Flux**. So the standard Pegasus+Condor+glite path does not apply directly.

### Option A: pegasus-mpi-cluster (recommended)

Run the entire workflow as one MPI job. No Condor, no glite.

1. Set `pegasus.code.generator=PMC` in pegasus.properties so Pegasus emits a single PMC DAG instead of Condor submit files.
2. Build pegasus-mpi-cluster (we already do this in `build.sh`); ensure MPI is available on Tuolumne (e.g. Cray MPICH or similar).
3. Create a Flux batch script that allocates nodes and runs PMC:

   ```bash
   #!/bin/bash
   #flux: -N 2
   #flux: -q pbatch
   #flux: -t 2h
   #flux: -B YOUR_BANK

   cd $PEGASUS_SCRATCH_DIR  # or your workflow dir on /p/lustre5
   flux run -N 2 -n 96 pegasus-mpi-cluster workflow.dag
   ```

4. Submit: `flux batch montage_pmc.cmd`

Pegasus PMC mode is experimental; you may need to adapt the generated PBS-style script to Flux directives. See [pegasus-mpi-cluster manpage](https://pegasus.isi.edu/documentation/manpages/pegasus-mpi-cluster.html).

### Option B: Flux allocation + Condor pool (Magpie-style)

Run inside a Flux allocation and treat the allocated nodes as a Condor pool.

1. Submit a Flux job to get N nodes (e.g. `flux batch -N 4 alloc.cmd` where `alloc.cmd` keeps a shell or runs a setup script).
2. On the first node, start `condor_master`; on all nodes, start `condor_startd` so they join the pool.
3. Use site catalog `style: condor`, `universe: vanilla`—jobs run on Condor workers (the same N nodes). No glite needed.
4. Use shared scratch/storage on `/p/lustre5` so data is visible to all nodes.

This mirrors [pegasus-isi/pegasus-llnl](https://github.com/pegasus-isi/pegasus-llnl), which uses Magpie+SLURM on Catalyst. For Tuolumne, replace `sbatch` with `flux batch` and provision the pool manually or via a Magpie-like harness.

### Option C: Slurm-based LC cluster (e.g. Corona, Catalyst)

The [DFTracer pegasus_montage tutorial](https://github.com/LLNL/dftracer/blob/develop/docs/pegasus_montage.rst) targets **LC Corona** (Slurm). On Slurm clusters:

1. Run `pegasus-configure-glite` for Slurm.
2. Use site catalog `style: glite`, `grid_resource: batch slurm`.
3. Run Condor on a login or dedicated node; Pegasus submits via glite to Slurm.

The pegasus-llnl repo provides [SLURM examples](https://github.com/pegasus-isi/pegasus-llnl/tree/master/examples) (e.g. `mpi-hw-slurm`, `diamond-slurm`) on Catalyst—adaptable to other Slurm-based LC systems, but not directly to Tuolumne’s Flux-only setup.

### Checklist for Tuolumne

- [ ] Use `/p/lustre5` for scratch and output (shared FS).
- [ ] Load MPI module if needed for pegasus-mpi-cluster (e.g. Cray default).
- [ ] Set bank with `-B` in flux batch; check with `flux account view-user $USER`.
- [ ] For DFTracer: Montage must be built with DFTracer; workflow env sets `DFTRACER_ENABLE=1`, etc., as in Quick Start.

## References

- [DFTracer pegasus_montage.rst](https://github.com/LLNL/dftracer/blob/develop/docs/pegasus_montage.rst) — LC Corona/SLURM setup
- [pegasus-isi/pegasus-llnl](https://github.com/pegasus-isi/pegasus-llnl) — Pegasus + Magpie on LLNL Catalyst (SLURM)
- [pegasus-isi/hpc-examples](https://github.com/pegasus-isi/hpc-examples) — Pegasus Jupyter examples for HPC
- [Montage](http://montage.ipac.caltech.edu)
- [Pegasus WMS](https://pegasus.isi.edu)
