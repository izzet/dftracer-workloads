# AGENTS: montage-pegasus

## Scope

Operate only within `montage-pegasus/` unless explicitly asked otherwise.

## Required Local Layout

- `.spack-env/`, `.venv/`, `montage/`, `montage-workflow-v3/`, `pegasus/`, `logs/`
- `spack.yaml`, `requirements.txt`, `.gitignore`
- `install.sh`, `build.sh`, `setup_env.sh`, `run.sh`
- `README.md`, `AGENTS.md`

## Standard Workflow

```bash
source ~/spack/share/spack/setup-env.sh
cd montage-pegasus
./install.sh
# Start Condor (see README)
./build.sh
./run.sh
```

## Script Contracts

- `install.sh`: Spack env (uses externals from packages.yaml when available), git submodules, Python venv, Condor + Pegasus from tarballs
- `build.sh`: Builds Montage and pegasus-mpi-cluster (no DFTracer patch by default; apply later when needed)
- `setup_env.sh`: Activates Spack env + venv, sets PATH/LD_LIBRARY_PATH for Montage, Pegasus, DFTracer
- `run.sh`: Runs montage-workflow.py then pegasus-run

## DFTracer

- When `dftracer/` is present, apply patch via `dftracer/apply_montage_patches.py` when tracing is needed
- `DFTRACER_BIND_SIGNALS=0` required for Pegasus (per docs)

## Common Issues

- Missing montage/Pegasus/Condor: run `install.sh`
- Condor not running: `condor_master`, `condor_status`
- Pegasus not in PATH: ensure `install/bin` and `install/sbin` are in PATH (setup_env.sh does this)
- Platform tarballs: set `CONDOR_PEGASUS_PLATFORM` or URLs via env
