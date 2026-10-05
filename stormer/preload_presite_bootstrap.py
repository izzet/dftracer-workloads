#!/usr/bin/env python3
"""Enter StorMer only after removing a mapped preload from future execs.

``LD_PRELOAD`` has done its one required job before this interpreter begins:
the DFTracer interposer is already mapped.  ``-S`` prevents Python from
running ``site``/``sitecustomize`` before this small bootstrap removes the
loader variables.  That matters on the Cray Python path, whose startup imports
can exec ``uname -p``.  Forked DataLoader workers retain the mapped library.
"""

import os
import sys


_PRELOAD_ONLY_ENV = ("LD_PRELOAD", "DFTRACER_PRELOAD_LIB")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("preload presite bootstrap requires the train.py path")
    if os.environ.get("STORMER_SCOPE_PRELOAD_TO_WORKERS") != "1":
        raise SystemExit("preload presite bootstrap requires worker-POSIX scope")

    removed = {name: os.environ.pop(name) for name in _PRELOAD_ONLY_ENV
               if name in os.environ}
    if set(removed) != set(_PRELOAD_ONLY_ENV):
        raise SystemExit("preload presite bootstrap missing loader environment")

    # Match direct ``python train.py`` import semantics after the environment
    # is clean.  Calling site.main() explicitly is required because -S skips
    # it, including the venv site-packages and its sitecustomize hook.
    script = os.path.abspath(argv[0])
    sys.path[0] = os.path.dirname(script)
    import site
    site.main()
    import runpy
    sys.argv = [script, *argv[1:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
