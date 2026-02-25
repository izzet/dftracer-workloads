# E3SM-IO Online I/O Opportunities

This note captures practical opportunities for runtime I/O inspection and tuning.

## Current App Reality

- The app parses knobs once at startup (`-n`, `-d`, `-r`, `-s`, `-m`, `-t`, `-o`).
- There is no built-in dynamic reconfiguration channel in the current source.
- The record loop (`rec_no`) is a natural synchronization boundary, but currently not used for external control updates.

## Where Online Optimization Is Feasible

1. **Record-boundary control points**
   - Add a rank-0 control check at record boundaries.
   - Broadcast updated values with `MPI_Bcast`.
   - Apply only knobs that are safe to change mid-run.

2. **Synchronization cadence**
   - `ncmpi_wait_all` dominates wall time in current traces.
   - Candidate knob: wait/flush every `k` records instead of every record.

3. **Write ordering behavior**
   - The code already has ordering variants (`two_buf` behavior in varn path).
   - This can be promoted to a runtime-updatable knob at safe boundaries.

4. **Chunked phase switching**
   - Some knobs are not safe to flip within an open output stream (`varn` vs `vard`, MPI/PnetCDF info hints).
   - Use chunked execution windows where each chunk can re-open with new settings.

## MOFKA Integration Direction

- Use MOFKA as control plane for optimization strategy messages.
- Suggested control flow:
  1. Rank 0 receives strategy updates.
  2. Rank 0 validates and timestamps update.
  3. Broadcast to all ranks.
  4. Apply at next safe checkpoint.
  5. Log applied strategy + outcomes into run metadata.

## What `run.sh` Supports Today

- DFTracer enable/disable and metadata toggle.
- Per-run logs and environment capture (`logs/runs/<timestamp>/env.txt`).
- `dfanalyzer` post-run analysis output at `logs/runs/<timestamp>/dfanalyzer_output.txt`.
- Optional `dfanalyzer` checkpoints under `tmp/dfanalyzer_<timestamp>`.

## Metric Interpretation Note

- Benchmark-reported bandwidth in `output.log` and DFAnalyzer-reported POSIX bandwidth
  are both correct, but they use different denominators.
- Benchmark bandwidth is end-to-end phase throughput (includes synchronization,
  waiting, and metadata/open/close overheads) and should be the primary optimization
  objective.
- DFAnalyzer POSIX bandwidth is operation-active throughput and is best used for
  diagnosis and root-cause analysis.

## How To Actualize Opportunities

1. **Scale to expose real bottlenecks**
   - Increase workload duration (`-r` records), process count, and decomposition size.
   - Target runs where I/O is a significant part of wall time, not tiny smoke tests.

2. **Create stable control windows**
   - Treat record boundaries (or chunk boundaries) as control points.
   - Require a minimum window length to avoid noisy decisions.

3. **Introduce one online knob first**
   - Start with `wait_every_k_records` because it directly targets wait/sync overhead.
   - Keep all other behavior fixed to isolate effect.

4. **Use MOFKA for closed-loop updates**
   - Rank 0 consumes strategy messages, validates values, broadcasts to all ranks,
     and applies only at safe points.
   - Persist strategy version and apply time in run metadata.

5. **Expand to chunk-level mode switching**
   - For unsafe in-place knobs (`varn`/`vard`, MPI/PnetCDF hints), switch only between
     chunks that reopen output context.

6. **Use a two-metric decision rule**
   - Optimize for benchmark end-to-end throughput.
   - Use DFAnalyzer metrics to explain why changes helped or hurt.

## Next Minimal Code Change

- Add one runtime knob in app code: `wait_every_k_records`.
- Evaluate impact with DFTracer + dfanalyzer in repeated runs before adding more knobs.
