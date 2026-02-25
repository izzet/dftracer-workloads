# WRF + DFTracer

## Overview

This folder hosts a reproducible WRF workload setup with DFTracer I/O tracing
and online knob actuation targets (output frequency, quilting servers, I/O format).

WRF is compiled and installed via Spack (`wrf@4.6.1 build_type=dmpar
compile_type=em_b_wave pnetcdf=true`), which handles WRF's complex configure
system automatically.  The test case is the fully **idealized baroclinic wave
(`em_b_wave`)** — no real input data required, runs in minutes, and exercises
WRF's full I/O stack (history + restart writes).

## Layout

- `.spack-env/`: local Spack environment data (created by `install.sh`).
- `.venv/`: local Python virtual environment for DFTracer tooling.
- `spack.yaml`: Spack specs — `wrf@4.6.1` + `python@3.11`.
- `requirements.txt`: Python deps (`dftracer`, `pyyaml`).
- `wrf/`: WRF source git submodule (pinned to `v4.6.1`, reference only).
- `logs/`: build/run/trace outputs and run metadata.
- `output/`: WRF-generated output files (`wrfout_*`, `wrfrst_*`).
- `run/`: per-case template directories (created by `prepare_case.sh`) and
  per-run working directories (`run/runs/<run_id>/`).
- `harness/`: optional fast-loop I/O reproducer cases.
- `dftracer/knobs.yaml`: online knob registry.
- `install.sh`, `build.sh`, `setup_env.sh`, `prepare_case.sh`, `run*.sh`: scripts.
- `AGENTS.md`: agent-facing operational playbook.

## Logs vs Output Convention

- `logs/` — execution telemetry: build logs, run stdout/stderr, JSON run
  summaries, trace artifacts (`logs/runs/<run_id>/trace*.pfw`).
- `output/<run_id>/` — WRF-generated scientific outputs (`wrfout_d01_*`,
  `wrfrst_*`, `wrfinput_*`).
- Never mix them.

## Source Pinning and `spack develop`

`spack.yaml` uses a `develop:` block pointing at the local submodule:

```yaml
develop:
  wrf:
    spec: wrf@4.6.1 build_type=dmpar compile_type=em_b_wave pnetcdf=true
    path: ./wrf
```

This means Spack **compiles WRF from `wrf/wrf/`** (the submodule checkout) instead
of downloading a tarball.  Benefits:

- Edit WRF source (e.g., add runtime knob-change hooks in Fortran) and
  `./build.sh` will do an **incremental rebuild** — only changed files recompiled.
- Spack still manages all dependencies and drives `./configure` + `./compile`
  with the correct library paths and compiler flags.
- You never need to manually set `NETCDF`, `HDF5`, `JASPER_DIR`, etc.

```bash
# Manual submodule add (install.sh does this automatically):
git submodule add https://github.com/wrf-model/WRF.git wrf/wrf
```

Pinned tag: **`v4.6.1`** (aligns with Spack preferred version).

### Source-edit workflow

```bash
# 1. Edit WRF source (e.g. add a knob-change hook):
#    vim wrf/share/module_io.F
#    vim wrf/frame/module_configure.F

# 2. Incremental rebuild:
./build.sh          # runs spack install (incremental), updates build_info.env

# 3. Refresh the run directory with the new binaries:
./prepare_case.sh   # re-symlinks wrf.exe / ideal.exe

# 4. Run:
./run.sh --dftracer-enable 1
```

## Dependency Pin Rationale

Spack `wrf@4.6.1` resolves all WRF dependencies automatically:

- OpenMPI (dmpar build type requires MPI)
- NetCDF-C + NetCDF-Fortran (WRF primary I/O)
- HDF5+MPI (backend for NetCDF-4 format)
- parallel-netcdf / pnetcdf (enabled via `pnetcdf=true` variant)
- Jasper + libpng (compression support)
- gfortran 13.x (external, from system gcc)

## Quick Start

```bash
source ~/spack/share/spack/setup-env.sh
cd wrf
./install.sh                   # ~15-30 min: Spack concretize + WRF compile + Python venv
./build.sh                     # verifies install, writes logs/build_info.env
./prepare_case.sh              # sets up run/em_b_wave/ from the Spack WRF install
./run.sh                       # baseline WRF run (no tracing)
./run.sh --dftracer-enable 1   # traced run with DFTracer + dfanalyzer post-processing
```

### What `install.sh` does

1. Adds/syncs git submodule `wrf/wrf` pinned to `v4.6.1`.
2. Runs `spack -e . concretize -f && spack -e . install` — this compiles WRF
   with `build_type=dmpar compile_type=em_b_wave pnetcdf=true`.
3. Creates `.venv` using Spack Python and installs `requirements.txt`.

### What `build.sh` does

Runs `spack -e . install` against the local WRF source (incremental rebuild),
then writes `logs/build_info.env` with:
- `WRF_PREFIX` — Spack install prefix
- `WRF_EXE`, `IDEAL_EXE` — paths to compiled executables
- `WRF_RUN_DIR`, `WRF_BWAVE_DIR` — run/data directories

Use `./build.sh` after every source edit to get an incremental recompile.
After `./build.sh`, run `./prepare_case.sh` to refresh the symlinks.

### What `prepare_case.sh` does

- Reads `logs/build_info.env`.
- Creates `run/em_b_wave/` with support files (TBL, namelists) from the install.
- Symlinks `wrf.exe` and `ideal.exe` into the case directory.
- Uses the `em_b_wave` namelist from the Spack install (or submodule) if
  available, otherwise generates a working fallback.

### What `run.sh` does

- Copies the case template to `run/runs/<run_id>/`.
- Runs `ideal.exe` to generate initial conditions.
- Runs `mpiexec -n NP wrf.exe`.
- Moves `wrfout_*` / `wrfrst_*` to `output/<run_id>/`.
- Optionally wraps with DFTracer `LD_PRELOAD` and runs `dfanalyzer` on traces.
- Writes `logs/runs/<run_id>/config.json` run summary.

```bash
./run.sh                        # baseline, 4 ranks
./run.sh --np 8                 # baseline, 8 ranks
./run.sh --dftracer-enable 1    # traced run
NP=8 ./run.sh                   # override via env var
```

## DFTracer Integration

DFTracer is attached via `LD_PRELOAD` (`libdftracer_preload.so`), which
intercepts POSIX file I/O.  This captures:

- All `wrfout_d01_*` history file writes (key I/O pain point).
- `wrfrst_*` restart file writes.
- `wrfinput_d01` / `wrfbdy_d01` reads by `wrf.exe`.
- Any NetCDF internal HDF5/PnetCDF calls that reach POSIX.

Trace artifacts land in `logs/runs/<run_id>/trace*.pfw`.

## I/O Knobs (DFTracer Targets)

Defined in `dftracer/knobs.yaml`.  These can be changed between runs or at
output-interval safe points by editing `run/<case>/namelist.input`:

| Knob | `namelist.input` key | Section | Notes |
|------|---------------------|---------|-------|
| Output frequency | `history_interval` | `&time_control` | minutes |
| Frames per file | `frames_per_outfile` | `&time_control` | timesteps/file |
| Restart interval | `restart_interval` | `&time_control` | minutes |
| History I/O format | `io_form_history` | `&time_control` | 2=NetCDF4, 11=pnetcdf |
| Quilting tasks | `nio_tasks_per_group` | `&namelist_quilt` | 0=quilting off |
| Quilting groups | `nio_groups` | `&namelist_quilt` | |

## Notes (Runs, Hiccups, Achievements)

- Runs:
  - Spack-based build system replaces placeholder scripts.
  - em_b_wave idealized case used for fast DFTracer iteration.
- Hiccups:
  - Spack WRF compilation is slow (~15-30 min); run `install.sh` once.
  - Exact WRF install layout (where executables land) may vary; `build.sh`
    searches multiple candidate paths.
  - If `build_info.env` shows `IDEAL_EXE=<not found>`, check that
    `compile_type=em_b_wave` is in `spack.yaml` and re-run `install.sh`.
- Achievements:
  - Full script scaffold with real WRF build/run commands.
  - DFTracer LD_PRELOAD integration matching e3sm-io pattern.
  - Online knob registry aligned to WRF `namelist.input` parameters.
