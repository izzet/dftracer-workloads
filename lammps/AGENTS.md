# AGENTS: lammps

## Scope

Work inside `lammps/` only unless explicitly instructed.

## Required Local Layout

- `.spack-env/`, `.venv/`, `lammps/` (submodule), `logs/`, `run/`, `output/`
- `spack.yaml`, `requirements.txt`, `.gitignore`
- `install.sh`, `build.sh`, `setup_env.sh`, `prepare_case.sh`, `run.sh`
- `README.md`, `AGENTS.md`
- `logs/build_info.env` (written by `build.sh` after install)

## Standard Workflow

```bash
source ~/spack/share/spack/setup-env.sh
cd lammps
./install.sh                   # one-time: Spack LAMMPS compile (~10-20 min) + Python venv
./build.sh                     # verify install, write logs/build_info.env
./prepare_case.sh              # set up run/lj_melt/ case directory
./run.sh                       # baseline run (no tracing)
./run.sh --dftracer-enable 1   # DFTracer traced run
```

## Script Contracts

- `install.sh`: idempotent.  Adds submodule, runs `spack -e . concretize + install`
  (first full LAMMPS compile from local source via `spack develop`), creates `.venv`.
- `setup_env.sh`: single activation point for Spack env + Python venv.  Must be
  sourced before any run scripts.
- `build.sh`: **incremental rebuild** after source edits — runs `spack -e . install`,
  then writes `logs/build_info.env`.
- `prepare_case.sh`: reads `build_info.env`, creates `run/lj_melt/`, writes
  `in.lj_melt` input script, symlinks `lmp` executable.
- `run.sh`: unified runner; handles baseline and DFTracer modes via `--dftracer-enable`.
- All scripts and `prepare_case.sh` must source `setup_env.sh`.

## spack develop Workflow

`spack.yaml` has a `develop:` section:
```yaml
develop:
  lammps:
    spec: lammps+mpi
    path: ./lammps
```
Spack builds LAMMPS **from `lammps/lammps/`** — never downloads a tarball.
After any source change the loop is:
```bash
# edit lammps/src/...  (or wherever)
./build.sh                     # incremental rebuild
./prepare_case.sh              # refresh symlinks in run/lj_melt/
./run.sh --dftracer-enable 1   # test
```

## Logs vs Output Convention

- `logs/`: all diagnostics (build logs, run stdout/stderr, JSON summaries,
  trace `.pfw` files, `build_info.env`, `env.txt`).
- `output/<run_id>/`: LAMMPS scientific output files only (`dump.*`, `restart.*`).
- `run/runs/<run_id>/`: per-run working directory (case copy).  Gitignored.

## Knobs

When present, knobs live in `dftracer/knobs.yaml`.  They map to `prepare_case.sh` flags and
the generated `in.lj_melt` input script:

| Knob             | Input script / prepare_case       |
|-------------------|----------------------------------|
| `dump_interval`   | `dump N all atom N dump.melt...` |
| `restart_interval`| `restart N restart.lj_melt`      |
| `io_mode`        | (future: posix vs mpiio)         |

Apply knob updates at the `between_restarts` or `between_runs` safe points only.

## Common Issues

| Symptom                    | Cause                      | Fix                          |
|---------------------------|----------------------------|------------------------------|
| `lmp not found` in run.sh | Spack LAMMPS not installed | Re-run `./install.sh`         |
| Case directory not found  | prepare_case.sh not run yet| Run `./prepare_case.sh`      |
| libdftracer_preload.so not found | dftracer not in .venv | Check `.venv` and requirements.txt |
| Spack concretize fails    | Compiler not detected      | Check gcc externals in spack.yaml |
| LAMMPS run crashes        | MPI/rank mismatch          | Reduce `NP` (default 4)      |

## DFTracer LD_PRELOAD Path

`run.sh` uses `find` to resolve the path dynamically.  Override if needed:

```bash
DFTRACER_PRELOAD_LIB=/path/to/lib ./run.sh --dftracer-enable 1
```

## Commit Hygiene

- Never commit `.venv/`, `.spack-env/`, `build/`, `install/`, `run/`, `output/`, `logs/`.
- Do not commit `logs/build_info.env` (machine-specific paths).
