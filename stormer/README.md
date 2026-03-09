# Stormer + DFTracer

## Overview

This folder hosts the real Stormer training workload as a standalone project.
It intentionally excludes the DLIO surrogate and keeps only the application,
its model code, the local utility modules it depends on, and the
`data-preparation/` scripts needed for the ERA5 HDF5 dataset.

DFTracer instrumentation is already integrated in the real training code. When
`DFTRACER_ENABLE=1`, the app emits per-rank trace files from the training path
itself. This standalone repo does not include Mofka, analyzer, diagnoser, or
optimizer services.
For standalone runs, `run.sh` also auto-enables the local DFTracer preload
library when available, so intercepted POSIX calls are captured without any
extra `LD_PRELOAD` setup.

The standalone default is intentionally simple:
- local `.venv`
- pip-installed `dftracer`
- no external DFSuite runtime/services
- single-rank training is supported directly
- multi-rank launcher integration can be added later as an optional path

## Layout

- `.venv/`: local Python environment created by `install.sh`
- `models/`: Stormer model code
- `utils/`: local logging, MPI, precision, torch, and DFTracer helpers
- `data-preparation/`: dataset download/regrid/preprocess scripts
- `logs/`: run/build diagnostics and DFTracer trace files
- `output/`: workload output directories
- `install.sh`, `setup_env.sh`, `build.sh`, `run.sh`, `prepare_data.sh`
- `install_tuo.sh`, `setup_env_tuo.sh`, `build_tuo.sh`, `run_tuo.sh`
- `requirements.txt`, `requirements-data.txt`
- `AGENTS.md`: agent-focused operational playbook

## Logs vs Output Convention

- `logs/` is for diagnostics and traces:
  - build/install logs
  - launcher stdout/stderr captures
  - environment snapshots
  - DFTracer trace files
- `output/` is reserved for workload-generated artifacts and future model output.

## Quick Start

```bash
cd stormer
./install.sh
./build.sh
DATA_FOLDER=/path/to/era5-hdf5 ./run.sh
DATA_FOLDER=/path/to/era5-hdf5 DFTRACER_ENABLE=1 ./run.sh
```

On Tuolumne, use the provided wrappers:

```bash
./install_tuo.sh
./build_tuo.sh
DATA_FOLDER=/p/lustre5/izzet/datasets/era5/hdf5 ./run_tuo.sh
DATA_FOLDER=/p/lustre5/izzet/datasets/era5/hdf5 DFTRACER_ENABLE=1 ./run_tuo.sh
```

`install_tuo.sh` loads `cray-python/3.11.7`, `gcc/13.3.1`, and `rocm/6.3.1`,
then installs the known-good traced stack:
- `torch 2.9.1+rocm6.3`
- `torchvision 0.24.1+rocm6.3`
- `dftracer 1.0.15`

This is the current working Tuolumne combination for Stormer with DFTracer
preload and POSIX capture enabled.
`run_tuo.sh` defaults to a one-rank Flux launch shape on Tuolumne.

## What `install.sh` does

- Creates `.venv` with the selected Python interpreter
- Installs the standalone training dependencies from `requirements.txt`
- Optionally installs data-preparation extras with `--with-data-tools`

## What `build.sh` does

- Activates the workload environment
- Compiles all Python files with `py_compile`
- Imports `train.py` as a smoke test

## What `run.sh` does

- Activates the workload environment
- Creates per-run directories under `logs/runs/<run_id>/` and `output/<run_id>/`
- Launches the real Stormer training app through `flux run`, `mpiexec`, or
  directly
- Preserves the in-app DFTracer instrumentation and writes trace files to
  `logs/runs/<run_id>/`

The default `run.sh` does not assume MPI or a particular cluster. The
`*_tuo.sh` wrappers are the supported Tuolumne entrypoints in this repo.

## Data Preparation

Install the optional tools and then run:

```bash
./install.sh --with-data-tools
./prepare_data.sh
```

`prepare_data.sh` supports the staged WeatherBench2 -> regrid -> HDF5 flow using
the scripts in `data-preparation/`. Normalization constants default to the
published upstream files.
