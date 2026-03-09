# AGENTS: stormer

## Scope

Work inside `stormer/` only unless explicitly instructed otherwise.

## Required Local Layout

- `.venv/`, `logs/`, `output/`
- `models/`, `utils/`, `data-preparation/`
- `requirements.txt`, `requirements-data.txt`, `.gitignore`
- `install.sh`, `setup_env.sh`, `build.sh`, `run.sh`, `prepare_data.sh`
- `install_tuo.sh`, `setup_env_tuo.sh`, `build_tuo.sh`, `run_tuo.sh`
- `README.md`, `AGENTS.md`

## Standard Workflow

```bash
cd stormer
./install.sh
./build.sh
DATA_FOLDER=/path/to/era5-hdf5 ./run.sh
DATA_FOLDER=/path/to/era5-hdf5 DFTRACER_ENABLE=1 ./run.sh
```

For Tuolumne, prefer:

```bash
./install_tuo.sh
./build_tuo.sh
DATA_FOLDER=/p/lustre5/izzet/datasets/era5/hdf5 ./run_tuo.sh
```

## Script Contracts

- `install.sh`: idempotent `.venv` creation and dependency installation
- `setup_env.sh`: single activation point for modules, `.venv`, and `PYTHONPATH`
- `build.sh`: import and syntax smoke tests only; no external build products
- `run.sh`: unified real-app runner; must call `setup_env.sh`; writes diagnostics
  under `logs/`
- `*_tuo.sh`: Tuolumne-specific wrappers that only set modules and launch defaults
- Tuolumne wrappers should prefer `gcc/13.3.1` so pip-installed `dftracer` can
  build locally
- `prepare_data.sh`: orchestrates dataset download, regrid, preprocess, and
  normalization-constant setup using `data-preparation/`

## Real App Only

- This workload intentionally excludes DLIO-specific files and configs.
- Preserve the application-integrated DFTracer instrumentation in `train.py`.
- Do not wire Mofka, DFAnalyzer, DFDiagnoser, or DFOptimizer into the default
  standalone scripts.
- Do not reintroduce `dlio_reader.py`, `dlio_config/`, or DLIO comparison scripts
  unless explicitly requested.

## Logs vs Output Convention

- `logs/`: diagnostics, launcher stdout/stderr, run metadata, trace files
- `output/`: workload artifacts and future model output only

## Common Issues

- Missing `.venv`: run `./install.sh`
- Missing dataset path: pass `DATA_FOLDER=/path/...` to `run.sh`
- Missing module environment on LC: set `STORMER_MODULES`
- Wrong launcher for cluster: use the appropriate wrapper such as `run_tuo.sh`
- `dftracer` source build fails on Tuolumne: ensure the Tuolumne wrappers load
  `gcc/13.3.1`
- Import failure in `build.sh`: inspect `logs/build.log`
- No DFTracer traces: ensure `DFTRACER_ENABLE=1` and `dftracer` is installed
- Missing POSIX events with `DFTRACER_ENABLE=1`: verify the runner exported the
  local DFTracer preload library in `logs/runs/<run_id>/env.txt`

## Commit Hygiene

- Never commit `.venv/`, `logs/`, `output/`, or generated datasets
- Keep large `.h5`, `.nc`, and `.npz` files out of the repo
