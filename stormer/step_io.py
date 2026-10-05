"""Cheap per-step DataLoader-worker read accounting from /proc/<pid>/io (opt-in, STORMER_STEP_IO=1).

Independent of the analyzer/SHM plane.  `rchar` counts bytes returned to the process by read()-family calls
(includes page-cache hits and, for HDF5 on Lustre, the bytes the client handed to the reader); `read_bytes` is
storage-layer bytes as the kernel accounts them (often 0 for network filesystems, kept as reported).
"""
import os


def child_pids(pid=None):
    """Direct children of `pid` (default: this process) via /proc/<pid>/task/*/children."""
    pid = os.getpid() if pid is None else pid
    out = set()
    try:
        for tid in os.listdir("/proc/%d/task" % pid):
            try:
                with open("/proc/%d/task/%s/children" % (pid, tid)) as f:
                    out.update(int(x) for x in f.read().split())
            except (OSError, ValueError):
                pass
    except OSError:
        pass
    return sorted(out)


def read_io(pids):
    """(sum rchar, sum read_bytes, n_alive) over `pids`; vanished pids are skipped."""
    rchar = rbytes = alive = 0
    for p in pids:
        try:
            with open("/proc/%d/io" % p) as f:
                for line in f:
                    k, _, v = line.partition(":")
                    if k == "rchar":
                        rchar += int(v)
                    elif k == "read_bytes":
                        rbytes += int(v)
            alive += 1
        except (OSError, ValueError):
            pass
    return rchar, rbytes, alive
