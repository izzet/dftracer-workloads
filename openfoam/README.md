# OpenFOAM + DFTracer

## Overview

This folder hosts a reproducible OpenFOAM workload with DFTracer hooks,
focused on demonstrating parallel I/O patterns:

- **Uncollated output** (default): creates one `processorN/` directory per
  rank — classic file-count explosion / metadata storm at scale.
- **Collated output**: a single `processorsN/` directory with reduced
  file-count but different write characteristics.

Switching between these modes at runtime is the primary knob (see
`dftracer/knobs.yaml`).

The benchmark case is **pitzDaily** (backward-facing step, turbulent RANS),
drawn directly from the OpenFOAM 12 tutorials bundled with the Spack install.

## Layout

- `.spack-env/` — local Spack environment data (gitignored).
- `.venv/`      — Python venv for DFTracer tooling (gitignored).
- `spack.yaml`  — Spack environment spec (openfoam-org@12 + python@3.11).
- `requirements.txt` — Python deps (dftracer, pyyaml).
- `logs/`       — run/build/trace logs and JSON run summaries (gitignored).
- `output/`     — benchmark output (per-run case dirs, field files) (gitignored).
- `harness/`    — cavity quick-iteration reproducer (see harness/README.md).
- `dftracer/`   — knob registry (knobs.yaml).
- `install.sh`, `setup_env.sh`, `build.sh`, `run.sh` — entry-point scripts.
- `AGENTS.md`   — agent-focused operational playbook.

## Logs vs Output convention

- `logs/` — diagnostics: build/run logs, JSON summaries, DFTracer trace files.
- `output/` — workload artifacts: per-run OpenFOAM case directories, field data.

## Source pinning and patching

OpenFOAM is added as a **git submodule** at `openfoam/openfoam/` (tag `version-12`)
and built via **`spack develop`**: Spack builds from the submodule source tree
rather than from a tarball, giving full source edit freedom for future patches
(e.g. adding an online knob-reception hook as a `functionObject`).

```bash
git submodule add https://github.com/OpenFOAM/OpenFOAM-12.git openfoam/openfoam
```

The `spack.yaml` `develop:` section pins this:

```yaml
develop:
  openfoam-org:
    spec: openfoam-org@12
    path: openfoam    # relative: openfoam/openfoam/
```

After making changes to the OpenFOAM source:

```bash
# Full rebuild via Spack
spack -e /path/to/openfoam install --only package openfoam-org

# Or build directly (faster; environment must be active)
source setup_env.sh
cd openfoam && ./Allwmake -j$(nproc)
```

## Dependency pin rationale

- `openfoam-org@12` — Spack's preferred stable version as of early 2026;
  version 13 is not yet in the Spack package repo.
- `python@3.11` — broadly compatible, consistent with other workloads.
- MPI, CMake, flex, and scotch are pulled in as transitive Spack dependencies.

## Quick start

```bash
source ~/spack/share/spack/setup-env.sh
cd openfoam
./install.sh          # ~30-60 min on first build; installs OpenFOAM via Spack
./build.sh            # copies pitzDaily tutorial, runs blockMesh
./run.sh              # 4-rank uncollated run (no tracing)
./run.sh --dftracer-enable 1   # traced run with DFTracer LD_PRELOAD
```

## Knob experiments

```bash
# Collated I/O (single processorsN/ directory instead of processorN/)
./run.sh --io-mode collated --np 8

# Increase output frequency to stress metadata
./run.sh --io-mode uncollated --write-interval 10 --np 8

# Collated run with DFTracer tracing
./run.sh --io-mode collated --dftracer-enable 1 --np 4

# All defaults listed in run.sh --help
./run.sh --help
```

## What run.sh does

1. Sources `setup_env.sh` (Spack env + OpenFOAM env + Python venv).
2. Copies `output/cases/pitzDaily/` to a fresh `output/runs/<run_id>/pitzDaily/`.
3. Patches `system/controlDict` with the requested `writeInterval`, `endTime`,
   and `fileHandler` (`uncollated` or `collated`).
4. Writes `system/decomposeParDict` (scotch, NP subdomains).
5. Runs `decomposePar -force` (if NP > 1).
6. Runs `foamRun [-parallel]` (with or without `LD_PRELOAD` for DFTracer).
7. Writes `logs/runs/<run_id>/config.json` (per DFTRACER.md schema).

## I/O signature (what to look for in DFTracer traces)

| mode        | file pattern                          | expected finding         |
|-------------|---------------------------------------|--------------------------|
| uncollated  | N × `processorK/TIME/FIELD` per step  | `metadata_storm` at scale|
| collated    | 1 × `processorsN/TIME/FIELD` per step | `write_latency_spike`    |

## Harness (quick-iteration)

See `harness/README.md` for a minimal cavity case that completes in seconds
and is useful for testing DFTracer setup and analysis scripts.

## Notes (Runs, Hiccups, Achievements)

- Runs:
  - All scripts are concrete (no placeholders).
  - Uses `foamRun` (OF 12+); falls back to `application` entry from controlDict
    for older OF layouts.
- Hiccups:
  - OpenFOAM 13 is not yet in Spack; using openfoam-org@12.
  - `setup_env.sh` sources OpenFOAM's `etc/bashrc` to populate `$FOAM_TUTORIALS`
    and `$WM_PROJECT_DIR`; this may emit harmless warnings about existing vars.
  - `scotch` decomposition is built into OpenFOAM by default; no extra Spack
    spec is needed.
  - First `spack install` (building OF from source via spack develop) takes ~1h.
    Subsequent incremental builds are much faster.
- Achievements:
  - Using `spack develop` + git submodule: full source edit freedom for future
    knob-registration patches, while Spack still manages all dependencies.
  - Full reproducible pipeline from Spack install → blockMesh → parallel run
    → DFTracer tracing with the collated/uncollated I/O knob wired up.
