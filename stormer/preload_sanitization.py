"""Keep a rank's already-mapped preload out of later exec children.

The worker-POSIX gate starts a rank with DFTracer in ``LD_PRELOAD``.  Ordinary
DataLoader workers are forked, so they inherit that mapping without needing the
loader environment variable.  An exec child, however, would load a fresh
interposer from that variable and can claim an unrostered lane.
"""

import os


_PRELOAD_ONLY_ENV = ("LD_PRELOAD", "DFTRACER_PRELOAD_LIB")


def remove_preload_from_exec_environment(environ=None):
    """Remove loader-only entries only for the opt-in worker-POSIX rank.

    The mapped library and its configuration remain in this process and are
    copied by ``fork``.  Do not remove ``DFTRACER_SHM_PREFIX``,
    ``DFTRACER_INIT``, ``DFTRACER_INC_METADATA``, or ``LD_LIBRARY_PATH``:
    those describe the already-mapped runtime rather than causing a future
    executable to load the interposer.
    """
    environ = os.environ if environ is None else environ
    if environ.get("STORMER_SCOPE_PRELOAD_TO_WORKERS", "0") != "1":
        return {}
    removed = {}
    for name in _PRELOAD_ONLY_ENV:
        value = environ.pop(name, None)
        if value is not None:
            removed[name] = value
    return removed
