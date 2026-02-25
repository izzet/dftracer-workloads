# E3SM-IO + DFTracer

## Overview

This folder hosts a reproducible E3SM-IO workload setup with DFTracer hooks.
Scripts are now aligned to E3SM-IO upstream build/run flow (`autoreconf`,
`configure`, `make`, `mpiexec ... e3sm-io/src/e3sm_io ...`).

## Layout

- `.spack-env/`: local Spack environment data.
- `.venv/`: local Python environment for DFTracer tooling.
- `spack.yaml`: workload dependencies.
- `requirements.txt`: Python dependencies (includes `dftracer`).
- `e3sm-io/`: upstream E3SM-IO source submodule at workload root.
- `logs/`: build/run/trace logs and run metadata.
- `output/`: benchmark-generated output files.
- `harness/`: optional, when present — quick-loop reproducer artifacts.
- `dftracer/`: optional, when present — knobs and adapter config.
- `install.sh`, `setup_env.sh`, `build.sh`, `run_*.sh`: entry-point scripts.
- `AGENTS.md`: agent-focused operational playbook.

## Logs vs Output Convention

- `logs/` is for execution telemetry and diagnostics:
  - build/install logs
  - run stdout/stderr captures
  - JSON run summaries
  - trace metadata and trace artifacts
- `output/` is for workload-generated benchmark/model output files (`.nc`, `.h5`, `.bp`, etc.).
- Do not mix meanings: scripts should write diagnostics to `logs/` and workload result files to `output/`.

## Source Pinning

`install.sh` will automatically add/init/sync the source submodule and pin it
to the stable tag below. Manual command (optional):

```bash
git submodule add https://github.com/Parallel-NetCDF/E3SM-IO.git e3sm-io/e3sm-io
```

Pin to a stable tag/commit and record it here.

- Recommended initial pin (update if you choose a newer stable tag):
  - `v.1.2.0`

## Dependency Pin Rationale (DeepWiki-verified)

Pinned in `spack.yaml` to align with E3SM-IO docs/CI compatibility guidance:

- `pnetcdf@1.14.0` (E3SM-IO requires >=1.10.0; CI uses 1.14.0)
- `hdf5@1.14.6+mpi` (E3SM-IO requires >=1.14.0; later 1.14.x avoids older caveats)
- `netcdf-c@4.9.2+mpi` (E3SM-IO requires >=4.9.0 due to dimension-scale bug in older versions)
- `adios2@2.8.3+mpi` (E3SM-IO docs/CI examples use 2.8.3)

## Quick Start

```bash
source ~/spack/share/spack/setup-env.sh
cd e3sm-io
./install.sh
./build.sh
./run.sh
./run.sh --dftracer-enable 1
```

### What `build.sh` does

- Runs `autoreconf -i` in `e3sm-io/`
- Configures with Spack-provided libraries (at minimum PnetCDF)
- Builds and installs into local `install/`

### Python environment source

`install.sh` creates `.venv` using the Python installed in the active Spack
environment (`spack location -i python`), not the system Python.

### What `run.sh` does

- Unified runner: use `--dftracer-enable 0` for baseline, `1` for traced run.
- Runs the benchmark executable: `e3sm-io/src/e3sm_io`
- Uses provided small dataset: `e3sm-io/datasets/f_case_866x72_16p.nc`
- Writes benchmark output files under `output/`.
- Default run mode: API `pnetcdf`, layout `canonical`, ranks `16`, records `2`.

Override runtime settings with env vars, for example:

```bash
NP=8 E3SM_API=pnetcdf E3SM_LAYOUT=canonical E3SM_RECORDS=1 ./run.sh
./run.sh --dftracer-enable 1
```

## Notes (Runs, Hiccups, Achievements)

- Runs:
  - Build/run scripts now execute concrete E3SM-IO commands using upstream
    datasets and options.
- Hiccups:
  - DeepWiki pages for this repo were not directly retrievable in this runtime,
    so scripts were grounded in upstream `docs/INSTALL.md`.
- Achievements:
  - Replaced placeholder scripts with concrete autotools build and real
    benchmark execution flow.
