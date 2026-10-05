"""Focused tests for the opt-in per-step fetch / worker-IO instruments (STORMER_STEP_FETCH, STORMER_STEP_IO)."""
import os, re, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import step_io
SRC = open(os.path.join(HERE, "train.py")).read()


def test_flags_default_off_and_gate_the_new_lines():
    assert 'os.environ.get("STORMER_STEP_FETCH", "0") == "1"' in SRC
    assert 'os.environ.get("STORMER_STEP_IO", "0") == "1"' in SRC
    # both require STORMER_STEP_WALL (existing flag) and every emit sits behind its own flag
    assert re.search(r'_sf_on = _sw_on and os\.environ\.get\("STORMER_STEP_FETCH"', SRC)
    assert re.search(r'_sio_on = _sw_on and os\.environ\.get\("STORMER_STEP_IO"', SRC)
    assert re.search(r'if _sf_on and _sf_fetch:\n\s+log\.info\("STEP_SERIES_FETCH', SRC)
    assert re.search(r'if _sio_on and _sio_r:\n\s+log\.info\("STEP_SERIES_IO', SRC)
    assert re.search(r'if _sf_on:\n\s+_sf_fetch\.append', SRC)
    assert re.search(r'if _sio_on:\n\s+import step_io', SRC)


def test_existing_step_series_line_unchanged():
    assert 'log.info("STEP_SERIES rank=%d epoch=%d n=%d wall_ms=%s loss=%s",' in SRC
    assert "[repr(round(x, 3)) for x in _sw_wall], [repr(x) for x in _sw_loss])" in SRC


def test_fetch_measured_after_previous_loss_sync():
    i = SRC.index("_sf_fetch.append((time.perf_counter() - _sw_prev)")
    j = SRC.index("_sw_prev = _sw_now")            # previous step's end (after float(loss))
    assert SRC.index("_sw_now = time.perf_counter()") < j
    assert j != i  # the fetch sample is taken at batch arrival, _sw_prev is updated at step end


def test_child_pids_and_read_io_count_reads():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"x" * (4 << 20)); path = f.name
    try:
        code = ("import sys,time\nd=open(%r,'rb').read()\nsys.stdout.write('go\\n');sys.stdout.flush()\ntime.sleep(5)" % path)
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
        try:
            assert p.stdout.readline().strip() == b"go"
            kids = step_io.child_pids()
            assert p.pid in kids
            rchar, rbytes, alive = step_io.read_io([p.pid, 2 ** 22 + 12345])   # second pid does not exist
            assert alive == 1 and rchar >= (4 << 20), (rchar, alive)
        finally:
            p.kill(); p.wait()
    finally:
        os.unlink(path)


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
