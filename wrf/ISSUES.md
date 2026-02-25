# WRF DFTracer Integration Issues

## Summary: Almost No Data I/O Visible in Traces

DFTracer (LD_PRELOAD) captures only ~19 POSIX ops with zero data size. Large NetCDF outputs (wrfout, wrfrst, wrfinput) are not traced.

---

## 1. wrf.exe: MPI-IO Bypasses LD_PRELOAD

**Root cause:** WRF uses NetCDF-4 (HDF5) with `io_form_history=2`, `io_form_restart=2`, etc. In parallel (4 ranks), HDF5 uses the **MPI-IO driver** — all data I/O goes through `MPI_File_*` calls. DFTracer's preload only wraps libc (open/read/write) and stdio (fopen/fread/fwrite). MPI-IO bypasses libc entirely, so those calls are invisible.

**Evidence:** wrf.exe trace shows only "start" and "end" events (6 total), no I/O operations.

**Possible remedies:**
- Use DFTracer/Brahma built with MPI-IO support (`BRAHMA_BUILD_WITH_MPI=ON`) if available
- Use Darshan or another tracer that instruments MPI-IO
- Run WRF with a different I/O backend that uses POSIX (if supported)

---

## 2. ideal.exe: POSIX I/O Exists but Is Not Traced

**Evidence (strace):** ideal.exe uses POSIX `openat()` + `write()` for wrfinput_d01 (~87 KB):

```
openat(AT_FDCWD, "wrfinput_d01", O_RDWR|O_CREAT|O_TRUNC, 0666) = 26
write(26, "&TIME_CONTROL\n RUN_DAYS=5...", 67331) = 67331
write(26, "&DOMAINS\n...", 3925) = 3925
...
```

**But** DFTracer trace for ideal.exe shows only: fork, fopen for `.ncrc`, `.daprc`, `.dodsrc`. No open/write for wrfinput_d01, input_jet, namelist.output, etc.

**Likely causes:**
1. **Path filtering:** `openat(AT_FDCWD, "wrfinput_d01", ...)` passes a relative path. DFTracer may check this against `DFTRACER_DATA_DIR` (absolute paths); if matching fails before path resolution, the operation is filtered out.
2. **Fork behavior:** ideal.exe forks; child process tracer state may not be properly initialized, so child I/O is not logged.
3. **MPI singleton:** ideal.exe spawns orted; MPI startup can complicate preload behavior.

---

## 3. Would Explicit DFTracer Init (C/C++ Integration) Help?

- **wrf.exe:** No. I/O would still go through MPI-IO; LD_PRELOAD cannot intercept that regardless of init.
- **ideal.exe:** Possibly, if the issue is preload not inheriting correctly in forked/MPI children. Explicit init would guarantee tracer setup in all processes. Would not fix path-filtering logic.

---

## 4. Experiments to Try

- Verify path filtering: try `DFTRACER_DATA_DIR` with an absolute path that definitely includes the run directory.
- Test ideal.exe standalone (no MPI) if possible to isolate fork/path effects.

## 5. Experiment Results (2026-02-18)

### DFTRACER_DATA_DIR=all

**Result:** Segfault in ideal.exe (pid 57489) during Step 1.

**Debug output (with `libdftracer_preload_dbg.so` + `LOG_LEVEL=DEBUG`):**

- With `DATA_DIR=all`, DFTracer sets `trace_all_files=1` and traces many more files (e.g. `/proc/cpuinfo`, `/etc/openmpi/...`, `/proc/self/maps`).
- ideal.exe runs for ~5 seconds then segfaults.
- **Last debug lines before crash:**
  ```
  write Calling function write
  is_traced Calling POSIXDFTracer.is_traced for write and fd 22 trace 0
  Segmentation fault (core dumped)
  ```
- fd 22 is likely a **pipe or socket** (MPI/orted communication). Even with `trace_all_files=1`, `is_traced` returns 0 for it (pipes/sockets may be excluded).
- **Hypothesis:** The segfault may occur when the tracer handles `write()` to fd 22: either (a) in the tracer after `is_traced` returns, (b) when invoking the original `write`, or (c) from state corruption due to tracing all files (larger fd map, more edge cases). A bug when tracing pipe/socket fds is plausible.

**Where fd 22 comes from (strace of ideal.exe):**

- ideal.exe is MPI-linked; when run it spawns `orted` (Open MPI daemon). orted creates pipes for process communication.
- strace shows `pipe2([5,6], ...)`, `pipe2([9,10], ...)`, `pipe2([11,12], ...)`, `pipe2([14,15], ...)`, `pipe2([23,24], ...)`.
- fd 22 is **not** created by pipe() directly; it is assigned by `openat()` when it is the next free fd. At different times fd 22 refers to:
  - `/sys/fs/cgroup/cgroup.controllers`, `/sys/class/block/...`, `/sys/devices/...` (hwloc/topology probing)
  - `/proc/self/fd/N` (O_PATH)
  - Library opens (libX11, libxml2, etc.)
- The write/read interleaving on fd 22 in the debug log (write→read→write→read) matches **pipe** semantics: MPI/orted parent–child communication. By the time of the crash (~5s in), fd 22 is most likely the **write end of a pipe** to orted.
- Pipes have no path (created via pipe/pipe2); DFTracer's fd→path map would not have an entry. `is_traced` returns 0 (excluded). The crash could be in the tracer's handling of this edge case.

### DFTRACER_LOG_LEVEL=DEBUG (with default path-based DATA_DIR)

**Result:** Run completes successfully. No extra diagnostic output observed.

The pip-installed DFTracer appears to be a release build: Brahma logging macros (BRAHMA_LOG_DEBUG, BRAHMA_LOG_INFO, etc.) are compiled as no-ops in `brahma/logging.h`. So `DFTRACER_LOG_LEVEL=DEBUG` does not emit visible debug logs. To get DEBUG output, DFTracer would need to be rebuilt from source with logging enabled.

---

## 6. input_jet Full Path Patch Causes Segfault with DFTracer

**Context:** `input_jet` is hardcoded as a relative path in `module_initialize_ideal.F` (`OPEN(unit=10, file='input_jet', ...)`). DFTracer filters relative paths, so `input_jet` is not traced. To fix this, we tried patching WRF to use a full path.

**Attempted approaches (both cause segfault):**

1. **`get_environment_variable('WRF_INPUT_JET_PATH', ...)`** — Read full path from env var. Segfault when ideal.exe runs with `LD_PRELOAD` (DFTracer).
2. **File-based `.wrf_input_jet_path`** — Read full path from a file in the run directory. Same segfault with DFTracer.

**Observation:** Without the input_jet patch, DFTracer + ideal.exe runs successfully. The segfault is specific to the combination of (a) the patched `read_input_jet` code and (b) DFTracer's LD_PRELOAD.

**Current status:** input_jet patch reverted. Traced runs capture wrfinput_d01, wrfout, wrfrst (full paths from namelist injection) but not input_jet.

**Possible next steps:**
- Report to DFTracer: segfault when ideal.exe uses `get_environment_variable` or file-based path resolution under LD_PRELOAD.
- Custom LD_PRELOAD wrapper that rewrites `open("input_jet")` to full path.
- Check if DFTracer can resolve relative paths under cwd when path matches `DFTRACER_DATA_DIR`.
