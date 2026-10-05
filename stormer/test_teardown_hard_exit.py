"""Behavioral test of the post-finalize teardown block in train.py main() (fix A')."""
import os, re, sys, textwrap, types
SRC = open(os.path.join(os.path.dirname(__file__), "train.py")).read()
m = re.search(r"    pfwlogger\.finalize\(\)\n(.*?)\n\n\nif __name__", SRC, re.S)
BLOCK = textwrap.dedent("    pfwlogger.finalize()\n" + m.group(1))


class Exit(Exception):
    pass


def run(env, dist_initialized=True):
    calls = []
    dist = types.SimpleNamespace(barrier=lambda: calls.append("barrier"),
                                 destroy_process_group=lambda: calls.append("destroy"))
    mpi = types.SimpleNamespace(finalize=lambda: calls.append("mpi_finalize"))
    fh = types.SimpleNamespace(dump_traceback_later=lambda *a, **k: calls.append(("arm", a, k)),
                               cancel_dump_traceback_later=lambda: calls.append("cancel"))
    osx = types.SimpleNamespace(environ=env, _exit=lambda c: (calls.append(("exit", c)), (_ for _ in ()).throw(Exit()))[0])
    ns = dict(pfwlogger=types.SimpleNamespace(finalize=lambda: calls.append("pfw")), dist=dist, MPIUtils=mpi,
              faulthandler=fh, os=osx, sys=sys, dist_initialized=dist_initialized)
    try:
        exec(BLOCK, ns)
    except Exit:
        pass
    return calls


def names(c):
    return [x if isinstance(x, str) else x[0] for x in c]


def test_hard_exit_before_destroy_and_mpi_finalize():
    c = run({"STORMER_HARD_EXIT_AFTER_FINALIZE": "1"})
    n = names(c)
    assert n == ["pfw", "arm", "barrier", "cancel", "exit"], n
    assert ("exit", 0) in c and "destroy" not in n and "mpi_finalize" not in n
    arm = [x for x in c if not isinstance(x, str) and x[0] == "arm"][0]
    assert arm[1] == (120.0,) and arm[2] == {"repeat": False, "exit": False}


def test_default_path_unchanged_order():
    n = names(run({}))
    assert n == ["pfw", "arm", "barrier", "destroy", "mpi_finalize", "cancel"], n
    assert "exit" not in n


def test_no_dist_no_barrier():
    n = names(run({"STORMER_HARD_EXIT_AFTER_FINALIZE": "1"}, dist_initialized=False))
    assert n == ["pfw", "arm", "cancel", "exit"], n


def test_timeout_zero_disables_dump():
    n = names(run({"STORMER_TEARDOWN_STACK_SEC": "0", "STORMER_HARD_EXIT_AFTER_FINALIZE": "1"}))
    assert "arm" not in n and n[-1] == "exit"


def test_exception_before_block_keeps_nonzero():
    # the block sits after trainer.train()/finalize inside main(); an exception there never reaches it
    i = SRC.index("trainer.train()"); j = SRC.index("pfwlogger.finalize()", i)
    assert "os._exit" not in SRC[i:j]
