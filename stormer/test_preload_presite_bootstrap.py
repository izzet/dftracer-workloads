"""Regression test for the rank startup boundary before train.py can run."""

import os
from pathlib import Path
import subprocess
import sys


BOOTSTRAP = Path(__file__).with_name("preload_presite_bootstrap.py")


def test_presite_bootstrap_clears_loader_entries_before_sitecustomize(tmp_path):
    marker = tmp_path / "sitecustomize-env.txt"
    (tmp_path / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['BOOTSTRAP_MARKER']).write_text("
        "repr((os.environ.get('LD_PRELOAD'), os.environ.get('DFTRACER_PRELOAD_LIB'))))\n"
    )
    train = tmp_path / "train.py"
    train.write_text("from pathlib import Path\nPath(__file__).with_name('train-ran').write_text('ok')\n")
    env = os.environ.copy()
    env.update({
        "STORMER_SCOPE_PRELOAD_TO_WORKERS": "1",
        # Empty avoids asking the dynamic loader for a test-only library while
        # still proving that the bootstrap removes an already-present entry.
        "LD_PRELOAD": "",
        "DFTRACER_PRELOAD_LIB": "/preloaded/libdftracer_preload.so",
        "PYTHONPATH": str(tmp_path),
        "BOOTSTRAP_MARKER": str(marker),
    })
    completed = subprocess.run(
        [sys.executable, "-S", str(BOOTSTRAP), str(train), "--sentinel"],
        env=env, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert marker.read_text() == "(None, None)"
    assert (tmp_path / "train-ran").read_text() == "ok"


def test_presite_bootstrap_fails_closed_without_rank_scope(tmp_path):
    train = tmp_path / "train.py"
    train.write_text("raise AssertionError('must not run')\n")
    env = os.environ.copy()
    env.pop("STORMER_SCOPE_PRELOAD_TO_WORKERS", None)
    completed = subprocess.run(
        [sys.executable, "-S", str(BOOTSTRAP), str(train)],
        env=env, text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert "requires worker-POSIX scope" in completed.stderr
