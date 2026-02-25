# LAMMPS + DFTracer

## Overview

This folder hosts a reproducible LAMMPS workload setup with DFTracer I/O tracing,
focused on restart/dump I/O behavior and I/O mode comparisons (posix vs mpiio).

LAMMPS is compiled and installed via Spack (`lammps+mpi`), which handles
dependencies automatically.  The test case is **Lennard-Jones melt** — minimal,
self-contained, runs in seconds, and exercises LAMMPS's dump and restart I/O.

## Layout

- `.spack-env/`: local Spack environment data (created by `install.sh`).
- `.venv/`: local Python virtual environment for DFTracer tooling.
- `spack.yaml`: Spack specs — `lammps+mpi` + `python@3.12`.
- `requirements.txt`: Python deps (`dftracer`, `dftracer-analyzer`, `pyyaml`).
- `lammps/`: LAMMPS source git submodule (pinned to `stable_29Aug2024_update4`).
- `logs/`: build/run/trace outputs and run metadata.
- `output/`: LAMMPS-generated output files (dump.*, restart.*).
- `run/`: case template (`run/lj_melt/`) and per-run working dirs (`run/runs/<run_id>/`).
- `harness/`: optional reduced reproducer cases.
- `dftracer/knobs.yaml`: online knob registry.
- `install.sh`, `build.sh`, `setup_env.sh`, `prepare_case.sh`, `run.sh`: scripts.
- `AGENTS.md`: agent-facing operational playbook.

## Logs vs Output Convention

- `logs/` — execution telemetry: build logs, run stdout/stderr, JSON run
  summaries, trace artifacts (`logs/runs/<run_id>/trace*.pfw`).
- `output/<run_id>/` — LAMMPS-generated outputs (dump.melt.lammpstrj,
  restart.lj_melt.*).
- Never mix them.

## Source Pinning and `spack develop`

`spack.yaml` uses a `develop:` block pointing at the local submodule:

```yaml
develop:
  lammps:
    spec: lammps+mpi
    path: ./lammps
```

This means Spack **compiles LAMMPS from `lammps/lammps/`** (the submodule checkout)
instead of downloading a tarball.  Benefits:

- Edit LAMMPS source (e.g., add runtime knob-change hooks) and `./build.sh` will do
  an **incremental rebuild**.
- Spack still manages all dependencies.

```bash
# Manual submodule add (install.sh does this automatically):
git submodule add https://github.com/lammps/lammps.git lammps/lammps
```

Pinned tag: **`stable_29Aug2024_update4`**.

## Quick Start

```bash
source ~/spack/share/spack/setup-env.sh
cd lammps
./install.sh                   # ~10-20 min: Spack concretize + LAMMPS compile + Python venv
./build.sh                     # verifies install, writes logs/build_info.env
./prepare_case.sh              # sets up run/lj_melt/ from the Spack LAMMPS install
./run.sh                       # baseline LAMMPS run (no tracing)
./run.sh --dftracer-enable 1   # traced run with DFTracer
```

### What `install.sh` does

1. Adds/syncs git submodule `lammps/lammps` pinned to `stable_29Aug2024_update4`.
2. Runs `spack -e . concretize -f && spack -e . install` — compiles LAMMPS from
   local source.
3. Creates `.venv` using Spack Python and installs `requirements.txt`.

### What `build.sh` does

Runs `spack -e . install` against the local LAMMPS source (incremental rebuild),
then writes `logs/build_info.env` with:
- `LAMMPS_PREFIX` — Spack install prefix
- `LMP_EXE` — path to compiled lmp executable

Use `./build.sh` after every source edit to get an incremental recompile.
After `./build.sh`, run `./prepare_case.sh` to refresh the symlinks.

### What `prepare_case.sh` does

- Reads `logs/build_info.env`.
- Creates `run/lj_melt/` with input script (`in.lj_melt`) and symlinks `lmp`.
- Input script is a Lennard-Jones melt — self-contained, runs in seconds.

### What `run.sh` does

- Copies the case template to `run/runs/<run_id>/`.
- Runs `mpiexec -n NP lmp -in in.lj_melt`.
- Moves `dump.*` / `restart.*` to `output/<run_id>/`.
- Optionally wraps with DFTracer `LD_PRELOAD` and runs `dfanalyzer` on traces.
- Writes `logs/runs/<run_id>/config.json` run summary.

```bash
./run.sh                        # baseline, 4 ranks
./run.sh --np 8                 # baseline, 8 ranks
./run.sh --dftracer-enable 1     # traced run
NP=8 ./run.sh                   # override via env var
```

## DFTracer Integration

DFTracer is attached via `LD_PRELOAD` (`libdftracer_preload.so`), which
intercepts POSIX file I/O.  This captures:

- Dump file writes (`dump.melt.lammpstrj`).
- Restart file writes (`restart.lj_melt.*`).

Trace artifacts land in `logs/runs/<run_id>/trace*.pfw`.

## I/O Knobs (DFTracer Targets)

Defined in `dftracer/knobs.yaml`.  Applied via `prepare_case.sh`:

| Knob              | `prepare_case.sh` flag   | Notes                      |
|-------------------|--------------------------|----------------------------|
| Dump interval     | `--dump-interval N`      | Steps between dump outputs |
| Restart interval  | `--restart-interval N`   | Steps between restarts     |
| I/O mode          | (future)                 | posix vs mpiio             |

## Notes

- Runs: Spack-based build system, LJ melt for fast DFTracer iteration.
- If `gcc@13.3.0` in `spack.yaml` does not match your system, edit the
  `packages.gcc.externals` section or remove it.
