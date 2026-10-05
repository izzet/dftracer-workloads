"""Offline contract tests for the worker-POSIX exec-child preload boundary."""

import mmap
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preload_sanitization import remove_preload_from_exec_environment


def test_default_scope_preserves_environment():
    env = {"LD_PRELOAD": "/selected/libdftracer_preload.so"}
    assert remove_preload_from_exec_environment(env) == {}
    assert env["LD_PRELOAD"] == "/selected/libdftracer_preload.so"


def test_opt_in_scope_removes_only_loader_entries():
    env = {
        "STORMER_SCOPE_PRELOAD_TO_WORKERS": "1",
        "LD_PRELOAD": "/selected/libdftracer_preload.so:/system/libextra.so",
        "DFTRACER_PRELOAD_LIB": "/selected/libdftracer_preload.so",
        "DFTRACER_SHM_PREFIX": "unique-prefix",
        "DFTRACER_INIT": "PRELOAD",
        "DFTRACER_INC_METADATA": "1",
        "LD_LIBRARY_PATH": "/selected/lib64",
    }
    removed = remove_preload_from_exec_environment(env)
    assert removed == {
        "LD_PRELOAD": "/selected/libdftracer_preload.so:/system/libextra.so",
        "DFTRACER_PRELOAD_LIB": "/selected/libdftracer_preload.so",
    }
    assert "LD_PRELOAD" not in env and "DFTRACER_PRELOAD_LIB" not in env
    assert env["DFTRACER_SHM_PREFIX"] == "unique-prefix"
    assert env["DFTRACER_INIT"] == "PRELOAD"
    assert env["DFTRACER_INC_METADATA"] == "1"
    assert env["LD_LIBRARY_PATH"] == "/selected/lib64"


def test_fork_keeps_mapped_state_while_exec_sees_no_loader_entries(tmp_path):
    """Fork preserves already-mapped state; a later exec inherits the clean env.

    This is a loader-semantics test, not a substitute for the native SHM
    emission recipe documented in STORMER-WORKER-POSIX-GATE.md.
    """
    saved = dict(os.environ)
    mapping = mmap.mmap(-1, 1)
    mapping[0:1] = b"L"
    read_fd, write_fd = os.pipe()
    try:
        os.environ.update({
            "STORMER_SCOPE_PRELOAD_TO_WORKERS": "1",
            "LD_PRELOAD": "/selected/libdftracer_preload.so",
            "DFTRACER_PRELOAD_LIB": "/selected/libdftracer_preload.so",
            "DFTRACER_SHM_PREFIX": "unique-prefix",
        })
        remove_preload_from_exec_environment()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            os.write(write_fd, mapping[0:1])
            os._exit(0)
        os.close(write_fd)
        assert os.read(read_fd, 1) == b"L"
        assert os.waitpid(pid, 0)[1] == 0
        result = subprocess.run(
            ["/usr/bin/env"], env=dict(os.environ), check=True,
            capture_output=True, text=True,
        )
        assert "LD_PRELOAD=" not in result.stdout
        assert "DFTRACER_PRELOAD_LIB=" not in result.stdout
    finally:
        os.environ.clear()
        os.environ.update(saved)
        mapping.close()
        for fd in (read_fd, write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_sanitization_precedes_h5py_import_and_launcher_records_it():
    root = Path(__file__).resolve().parents[3]
    train = (root / "optimization/dftracer_workloads/stormer/train.py").read_text()
    launcher = (root / "optimization/run_stormer_flux.sh").read_text()
    assert train.index("remove_preload_from_exec_environment") < train.index("import h5py")
    assert 'loader_kwargs["multiprocessing_context"] = "fork"' in train
    assert "rank-presite-bootstrap-fork-inherit-v2" in launcher
    assert "preload_exec_removed_ld_preload" in launcher
    assert "preload_exec_removed_dftracer_preload_lib" in launcher
    assert "-S -u ${STORMER_ROOT}/preload_presite_bootstrap.py ${STORMER_ROOT}/train.py" in launcher
