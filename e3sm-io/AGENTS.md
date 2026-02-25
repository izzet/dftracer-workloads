# AGENTS: e3sm-io

## Scope

Operate only within `e3sm-io/` unless explicitly asked otherwise.

## Required Local Layout

- `.spack-env/`, `.venv/`, `e3sm-io/`, `logs/`
- `spack.yaml`, `requirements.txt`, `.gitignore`
- `install.sh`, `build.sh`, `setup_env.sh`, `run.sh`
- `README.md`, `AGENTS.md`

## Standard Workflow

```bash
source ~/spack/share/spack/setup-env.sh
cd e3sm-io
./install.sh
./build.sh
./run.sh
./run.sh --dftracer-enable 1
```

## Script Contracts

- `install.sh`: idempotent setup of `.spack-env` + `.venv`.
  - `.venv` must be created using the active Spack environment's Python.
- `setup_env.sh`: single source of activation logic for Spack + Python envs.
- `build.sh`: runs `autoreconf/configure/make` from `e3sm-io/`, then installs under `install/`.
- `run.sh`: unified runner; must call `setup_env.sh`; use `--dftracer-enable 0|1`; write logs under `logs/`.

## Logs vs Output Convention

- `logs/`: diagnostics only (script logs, stdout/stderr captures, JSON summaries, trace metadata/artifacts).
- `output/`: benchmark-generated data products only.
- Agent rule: never write workload output files into `logs/`, and never store diagnostics as primary files in `output/`.

## Knobs

When present, knobs live in `dftracer/knobs.yaml`.
Only use allowed values and make changes at safe boundaries (between runs, or
validated points supported by workload configuration).

## Common Issues

- Missing source: add submodule at `e3sm-io/`.
- Missing envs: run `./install.sh`.
- Build failure: inspect `logs/build.log`.
- Runtime failure: check `e3sm-io/datasets/f_case_866x72_16p.nc` exists and tune `NP`.

## Commit Hygiene

- Keep generated artifacts under ignored paths (`logs/`, `build/`, `install/`).
- Do not commit large generated data.
