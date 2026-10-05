#!/usr/bin/env python3
"""Offline control-flow tests; the real MPI check is the two-rank fixture."""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(__file__))

from loop_exit_consensus import (LoopExitConsensus, PaddedRankSyncIterator,  # noqa: E402
                                 UnequalLoaderExhaustion, wrap_step_iterator)


class _FakeConsensus:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def should_stop(self, local_stop):
        self.calls.append(local_stop)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def test_disabled_path_is_exact_passthrough():
    assert list(wrap_step_iterator(iter([10, 20]), None)) == [(10, False), (20, False)]


def test_equal_loader_stops_without_padding():
    consensus = _FakeConsensus([False, False, True])
    assert list(PaddedRankSyncIterator(iter([10, 20]), consensus)) == [(10, False), (20, False)]
    assert consensus.calls == [False, False, True]


def test_unequal_loader_raises_before_extra_update():
    consensus = _FakeConsensus([False, False,
                                UnequalLoaderExhaustion("UNEQUAL_LOADER_EXHAUSTION")])
    completed_updates = 0
    try:
        for _, is_padding in PaddedRankSyncIterator(iter([10, 20]), consensus):
            assert not is_padding
            completed_updates += 1
    except UnequalLoaderExhaustion:
        pass
    else:
        raise AssertionError("expected unequal exhaustion")
    assert completed_updates == 2
    assert consensus.calls == [False, False, True]


def test_collective_reports_mixed_exhaustion_on_every_rank():
    class FakeMPI:
        SUM = object()
    sys.modules["mpi4py"] = types.SimpleNamespace(MPI=FakeMPI)

    class FakeComm:
        def allreduce(self, value, op):
            assert op is FakeMPI.SUM
            return 1

    class FakeMPIUtils:
        def size(self):
            return 2

        def is_mpi(self):
            return True

        def comm_world(self):
            return FakeComm()

    try:
        for local_stop in (False, True):
            try:
                LoopExitConsensus(FakeMPIUtils()).should_stop(local_stop)
            except UnequalLoaderExhaustion as exc:
                assert "UNEQUAL_LOADER_EXHAUSTION" in str(exc)
            else:
                raise AssertionError("mixed exhaustion must raise on every rank")
    finally:
        del sys.modules["mpi4py"]


if __name__ == "__main__":
    test_disabled_path_is_exact_passthrough()
    test_equal_loader_stops_without_padding()
    test_unequal_loader_raises_before_extra_update()
    test_collective_reports_mixed_exhaustion_on_every_rank()
    print("ALL PASS")
