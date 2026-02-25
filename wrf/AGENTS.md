# AGENTS: wrf

## Scope

Work inside `wrf/` only unless explicitly instructed.

## Required Local Layout

- `.spack-env/`, `.venv/`, `wrf/` (submodule), `logs/`, `run/`, `output/`
- `spack.yaml`, `requirements.txt`, `.gitignore`
- `install.sh`, `build.sh`, `setup_env.sh`, `prepare_case.sh`, `run*.sh`
- `README.md`, `AGENTS.md`
- `logs/build_info.env` (written by `build.sh` after install)

## Standard Workflow

```bash
source ~/spack/share/spack/setup-env.sh
cd wrf
./install.sh                   # one-time: Spack WRF compile (~15-30 min) + Python venv
./build.sh                     # verify install, write logs/build_info.env
./prepare_case.sh              # set up run/em_b_wave/ case directory
./run.sh                       # baseline run (no tracing)
./run.sh --dftracer-enable 1   # DFTracer traced run
```

## Script Contracts

- `install.sh`: idempotent.  Adds submodule, runs `spack -e . concretize + install`
  (first full WRF compile from local source via `spack develop`), creates `.venv`.
- `setup_env.sh`: single activation point for Spack env + Python venv.  Must be
  sourced before any run scripts.
- `build.sh`: **incremental rebuild** after source edits — runs `spack -e . install`
  (only recompiles changed files), then writes `logs/build_info.env`.
- `prepare_case.sh`: reads `build_info.env`, creates `run/em_b_wave/`, copies
  support files and namelists, symlinks executables.
- `run.sh`: unified runner; handles baseline and DFTracer modes via `--dftracer-enable`.
- All scripts and `prepare_case.sh` must source `setup_env.sh`.

## spack develop Workflow

`spack.yaml` has a `develop:` section:
```yaml
develop:
  wrf:
    spec: wrf@4.6.1 ...
    path: ./wrf      # relative to spack.yaml = the git submodule
```
Spack builds WRF **from `wrf/wrf/`** — never downloads a tarball.
After any source change the loop is:
```bash
# edit wrf/share/module_io.F  (or wherever)
./build.sh                     # incremental rebuild
./prepare_case.sh              # refresh symlinks in run/em_b_wave/
./run.sh --dftracer-enable 1   # test
```

## Key WRF Source Files for Knob Hooks

When adding runtime knob support, focus on:

| File | Purpose |
|------|---------|
| `wrf/frame/module_configure.F` | Namelist reading; good place to poll a knob file |
| `wrf/share/module_io.F` | I/O dispatch; intercept history/restart writes |
| `wrf/share/output_wrf.F` | History output logic; control field sets |
| `wrf/main/wrf.F` | Main time loop; safe-point for knob checks |

## Logs vs Output Convention

- `logs/`: all diagnostics (build logs, run stdout/stderr, JSON summaries,
  trace `.pfw` files, `build_info.env`, `env.txt`).
- `output/<run_id>/`: WRF scientific output files only (`wrfout_*`, `wrfrst_*`,
  `wrfinput_*`).
- `run/runs/<run_id>/`: per-run working directory (case copy).  Gitignored.

## Knobs

When present, knobs live in `dftracer/knobs.yaml`.  They map to `namelist.input` parameters
in the prepared case directory.  Safe to change between runs; do NOT change
during a running simulation.

Key knobs:
- `history_interval_minutes` → `namelist.input` `history_interval`
- `frames_per_outfile` → `namelist.input` `frames_per_outfile`
- `io_form_history` → `namelist.input` `io_form_history` (2=NetCDF4, 11=pnetcdf)
- `quilt_servers` → `namelist.input` `nio_tasks_per_group`

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `wrf.exe not found` in build.sh | Spack WRF not installed | Re-run `./install.sh` |
| `IDEAL_EXE=<not found>` in build_info.env | Wrong compile_type in spack.yaml | Check `compile_type=em_b_wave` |
| `Case directory not found` in run.sh | prepare_case.sh not run yet | Run `./prepare_case.sh` |
| `libdftracer_preload.so not found` | dftracer not installed in .venv | Check `.venv` and requirements.txt |
| WRF run crashes with MPI error | Too many ranks for grid size | Reduce `NP` (default 4; min 1) |
| `rsl.error.*` files in work dir | WRF runtime error | Check `logs/runs/<run_id>/output.log` |
| Spack concretize fails | Compiler not detected | Check gcc externals in spack.yaml |

## DFTracer LD_PRELOAD Path

The default path for `libdftracer_preload.so` is:

```
.venv/lib/python3.12/site-packages/dftracer/lib/libdftracer_preload.so
```

`run.sh` uses `find` to resolve the path dynamically (handles minor version changes automatically).

If your Python version differs, override:

```bash
DFTRACER_PRELOAD_LIB=/path/to/lib ./run.sh --dftracer-enable 1
```

## Commit Hygiene

- Never commit `.venv/`, `.spack-env/`, `build/`, `install/`, `run/`, `output/`, `logs/`.
- Do not commit `wrf/wrf/test/em_real/` (large input data).
- Do not commit `logs/build_info.env` (machine-specific paths).
