# AGENTS: openfoam

## Scope

Operate only within `openfoam/` unless explicitly instructed.

## Required Local Layout

- `.spack-env/`, `.venv/` (created by install.sh, gitignored)
- `openfoam/` — OpenFOAM-12 git submodule (source; build artifacts gitignored)
- `spack.yaml` (openfoam-org@12 with develop: section, python@3.12)
- `requirements.txt` (dftracer, pyyaml)
- Root scripts: `install.sh`, `setup_env.sh`, `build.sh`, `run.sh`
- `logs/`, `output/` (created at runtime, gitignored)
- `README.md`, `AGENTS.md`

## Standard Workflow

```bash
source ~/spack/share/spack/setup-env.sh
cd openfoam
./install.sh          # installs openfoam-org@12 via Spack + Python venv
./build.sh            # copies pitzDaily tutorial + runs blockMesh
./run.sh              # baseline run (4 ranks, uncollated)
./run.sh --dftracer-enable 1   # traced run (4 ranks, uncollated)
```

## Knob experiments

```bash
# Change I/O mode
./run.sh --io-mode collated --np 4
./run.sh --io-mode uncollated --np 8 --write-interval 10

# Full traced collated run
./run.sh --io-mode collated --dftracer-enable 1
```

## Key Contracts

- `setup_env.sh` activates Spack env AND sources `$FOAM_INSTALL/etc/bashrc`.
  `run.sh` must source `setup_env.sh`.
- `run.sh` is the unified runner; use `--dftracer-enable 0|1` for baseline vs traced.
- Logs and run metadata are written under `logs/runs/<run_id>/`.
- Per-run case copies go under `output/runs/<run_id>/pitzDaily/`.
- `.venv` is created by `install.sh` using Python from the active Spack env.

## Logs vs Output Convention

- `logs/`: build log, per-run stdout/stderr, `config.json` summaries, trace files.
- `output/`: base case template (`output/cases/pitzDaily/`) and per-run case dirs.

## OpenFOAM Version Notes

- Version: `openfoam-org@12` (version 13 not yet in Spack).
- Source: git submodule at `openfoam/openfoam/` pinned to tag `version-12`.
- Build: `spack develop` — Spack builds OpenFOAM FROM the submodule source.
  `spack.yaml` has a `develop:` section with `path: openfoam` (relative).
- Solver runner: `foamRun` (OF 11+).
  `run.sh` falls back to the `application` entry from `system/controlDict`
  for compatibility with older OF layouts.
- Tutorial path (after install):
  `$(spack location -i openfoam-org)/tutorials/incompressibleFluid/pitzDaily/`

## Patching OpenFOAM source

Edit files under `openfoam/openfoam/` freely.  Then rebuild:

```bash
# Via Spack (recommended; preserves install path)
spack -e <workload-root> install --only package openfoam-org

# Or directly (faster; environment must already be active)
source setup_env.sh
cd openfoam && ./Allwmake -j$(nproc)
```

The in-source build artifacts (`openfoam/platforms/`, etc.) are gitignored.
Commit only the source changes, not build products.

## Knobs Registry

When present, knobs live in `dftracer/knobs.yaml`:
- `output_mode`: `uncollated` (default) | `collated`
  → maps to `run.sh --io-mode`
- `write_interval`: int, default 50
  → maps to `run.sh --write-interval`
- `collated_buffer_mb`: int, advisory hint for buffer sizing
  (not yet wired to a concrete OF setting; future work)

Only apply knob updates at the `between_runs` or `between_write_intervals`
safe points documented in `knobs.yaml`.

## Common Issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FOAM_TUTORIALS not set` | `etc/bashrc` not sourced | Check `setup_env.sh` found `openfoam-org` install |
| `foamRun: command not found` | Spack env not activated | Run `source setup_env.sh` |
| `pitzDaily: not found` | `build.sh` not run | Run `./build.sh` |
| DFTracer lib missing | dftracer not in venv | Re-run `./install.sh` |
| `scotch: decomposition failed` | scotch not in OpenFOAM build | Change method to `hierarchical` in `run.sh` decomposeParDict template |
| Submodule empty or missing | submodule not initialised | `git submodule update --init openfoam/openfoam` or re-run `./install.sh` |
| Build changes not reflected | old cached Spack build | `spack -e . install --only package openfoam-org` |
| `mpi.h: No such file or directory` | WM_MPLIB=SYSTEMMPI needs MPI_ROOT | Changed to SYSTEMOPENMPI in `openfoam/etc/bashrc`; uses `mpicc --showme` autodetect |
