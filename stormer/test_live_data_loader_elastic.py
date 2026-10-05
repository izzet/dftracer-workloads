"""CPU tests for the elastic LiveDataLoader (in-place worker/prefetch resize).

Run:  python test_live_data_loader_elastic.py   (or pytest)
"""
import os
import sys
import time

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_data_loader import LiveDataLoader, _ElasticIter  # noqa: E402
from live_sampler import LiveSampler  # noqa: E402


class _DS(Dataset):
    def __init__(self, n=400, delay=0.002):
        self.n, self.delay = n, delay

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(self.delay)
        return torch.tensor([i], dtype=torch.int64), os.getpid()


def _collate(b):
    return torch.cat([x[0] for x in b]), [x[1] for x in b]


def _make(workers=2, pf=2, n=400, elastic=True, delay=0.002, ds=None, keyed=False):
    os.environ["STORMER_ELASTIC_LOADER"] = "1" if elastic else "0"
    ds = ds if ds is not None else _DS(n, delay)
    s = LiveSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=7)
    ld = LiveDataLoader(ds, s, batch_size=1, num_workers=workers, prefetch_factor=pf,
                        collate_fn=_collate,
                        generator_factory=(lambda e: torch.Generator().manual_seed(e)) if keyed else None)
    ld.set_epoch(0)
    return ld, s


def _expected(ld):
    return [int(i) for i in ld._sampler._indices]


def _children():
    import multiprocessing as mp
    return {p.pid for p in mp.active_children()}


def _drive(ld, schedule, total):
    """Consume `total` batches, applying (step -> (workers, pf)) from `schedule`."""
    got, pids = [], set()
    for step, (b, p) in enumerate(ld):
        got.append(int(b[0]))
        pids.update(p)
        if step in schedule:
            w, pf = schedule[step]
            ld.reconfigure(num_workers=w, prefetch_factor=pf)
            ld.apply_pending()
        if len(got) >= total:
            break
    return got, pids


def test_order_preserved_under_resizes():
    ld, _ = _make(workers=2, pf=2, n=400)
    exp = _expected(ld)
    sched = {20: (6, 2), 60: (3, 2), 100: (3, 5), 140: (8, 1), 200: (2, 2), 260: (4, 3)}
    got = []
    for step, (b, _) in enumerate(ld):
        got.append(int(b[0]))
        if step in sched:
            ld.reconfigure(num_workers=sched[step][0], prefetch_factor=sched[step][1])
            ld.apply_pending()
            assert "workers_after" in ld.last_apply, ld.last_apply
            assert ld.observed_state() == sched[step], (ld.observed_state(), sched[step])
    assert got == exp, "sample order changed"
    assert len(got) == 400


def test_standby_pool_admits_and_parks_without_forking():
    import multiprocessing as mp
    os.environ["STORMER_ELASTIC_POOL_MAX"] = "8"
    try:
        ld, _ = _make(workers=2, pf=2, n=800)
        it = iter(ld)
        for _ in range(10):
            next(it)
        assert len(mp.active_children()) == 8, len(mp.active_children())      # one fork event: the whole pool
        assert ld.observed_state() == (2, 2)
        ld.reconfigure(num_workers=6)
        assert ld.apply_pending()
        assert ld.last_apply["forked"] == 0 and ld.last_apply["admitted_from_standby"] == 4
        assert ld.last_apply["wall_s"] < 0.05, ld.last_apply
        for _ in range(10):
            next(it)
        assert len(mp.active_children()) == 8 and ld.observed_state() == (6, 2)
        ld.reconfigure(num_workers=10)                                       # beyond the pool: fork only the missing 2
        ld.apply_pending()
        assert ld.last_apply["forked"] == 2, ld.last_apply
        for _ in range(10):
            next(it)
        assert len(mp.active_children()) == 10
        ld.reconfigure(num_workers=3)
        ld.apply_pending()
        assert ld.last_apply["parked"] == 7 and ld.last_apply["forked"] == 0
        for _ in range(60):
            next(it)
        assert len(mp.active_children()) == 10 and ld.observed_state() == (3, 2)   # nothing terminated mid-epoch
        # only the active workers served the last 60 batches
        seen = set()
        for _ in range(60):
            b, p = next(it)
            seen.update(p)
        assert len(seen) <= 3, seen
    finally:
        os.environ.pop("STORMER_ELASTIC_POOL_MAX", None)


class _EpochDS(_DS):
    """Item content depends on the epoch the WORKER was told (the keyed-protocol property)."""
    def __init__(self, n=200, delay=0.001):
        super().__init__(n, delay)
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = int(e)

    def __getitem__(self, i):
        time.sleep(self.delay)
        return torch.tensor([i * 10 + self.epoch], dtype=torch.int64), os.getpid()


def test_persistent_pool_one_fork_event_follows_epoch_and_discards_stale():
    import multiprocessing as mp
    os.environ["STORMER_ELASTIC_POOL_MAX"] = "6"
    try:
        ds = _EpochDS(200)
        ld, _ = _make(workers=3, pf=2, ds=ds, keyed=True)
        pids_by_epoch = []
        for epoch in range(3):
            ld.set_epoch(epoch)
            exp = [i * 10 + epoch for i in ld._sampler._indices]
            got, pids = [], set()
            for step, (b, p) in enumerate(ld):
                got.append(int(b[0]))
                pids.update(p)
                if epoch == 1 and step == 40:
                    ld.reconfigure(num_workers=5, prefetch_factor=3)
                    ld.apply_pending()
                if epoch < 2 and step == 120:      # leave the epoch early, as the 901-step cap does
                    break
            assert got == exp[:len(got)], "epoch %d stream wrong (stale or wrong-epoch items)" % epoch
            assert len(got) in (121, 200)
            pids_by_epoch.append(pids)
        assert len(mp.active_children()) == 6, len(mp.active_children())     # forked once, never rebuilt
        allp = set().union(*pids_by_epoch)
        assert len(allp) <= 6
        assert ld.observed_state() == (5, 3)
    finally:
        os.environ.pop("STORMER_ELASTIC_POOL_MAX", None)


def test_persistent_matches_per_epoch_rebuild_stream():
    outs = []
    for keep in ("1", "0"):
        os.environ["STORMER_LOADER_PERSISTENT"] = keep
        ds = _EpochDS(150)
        ld, _ = _make(workers=2, pf=2, ds=ds, keyed=True)
        seq = []
        for epoch in range(2):
            ld.set_epoch(epoch)
            for step, (b, _) in enumerate(ld):
                seq.append(int(b[0]))
                if step == 90:
                    break
        outs.append(seq)
    os.environ.pop("STORMER_LOADER_PERSISTENT", None)
    assert outs[0] == outs[1]


def test_matches_rebuild_loader_stream():
    a, _ = _make(workers=3, pf=2, n=300, elastic=True)
    b, _ = _make(workers=3, pf=2, n=300, elastic=False)
    ea, eb = _expected(a), _expected(b)
    assert ea == eb
    sched = {10: (5, 2), 80: (2, 4)}
    ga, _ = _drive(a, sched, 300)
    gb, _ = _drive(b, sched, 300)
    assert ga == gb == ea


def test_rebuild_fallback_for_zero_workers_and_epochs():
    ld, _ = _make(workers=2, n=120)
    got = [int(b[0]) for b, _ in ld]
    assert len(got) == 120
    ld.reconfigure(num_workers=4)
    ld.set_epoch(1)
    it = iter(ld)
    next(it)
    assert ld.observed_state() == (4, 2)


def test_resize_at_epoch_end_is_deferred_and_next_epoch_uses_new_width():
    ld, _ = _make(workers=2, pf=2, n=40, delay=0.0)
    it = iter(ld)
    for _ in range(40):
        next(it)
    ld.reconfigure(num_workers=5)
    assert ld.apply_pending()
    assert ld.last_apply["mode"] in ("deferred_to_next_epoch", "elastic")
    ld.set_epoch(1)
    it2 = iter(ld)
    next(it2)
    assert ld.observed_state() == (5, 2)


def test_worker_error_still_propagates_after_resize():
    class Bad(_DS):
        def __getitem__(self, i):
            if i == 5:
                raise ValueError("boom")
            return super().__getitem__(i)
    os.environ["STORMER_ELASTIC_LOADER"] = "1"
    ds = Bad(200)
    s = LiveSampler(ds, num_replicas=1, rank=0, shuffle=False, seed=0)
    ld = LiveDataLoader(ds, s, batch_size=1, num_workers=2, prefetch_factor=2, collate_fn=_collate)
    ld.set_epoch(0)
    it = iter(ld)
    next(it)
    ld.reconfigure(num_workers=4)
    ld.apply_pending()
    try:
        for _ in range(20):
            next(it)
    except ValueError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("worker exception lost")


def test_random_resize_stress_keeps_order_and_exits_clean():
    import random
    import multiprocessing as mp
    for seed in range(4):
        rnd = random.Random(seed)
        os.environ["STORMER_ELASTIC_POOL_MAX"] = str(rnd.choice([0, 4, 9]))
        ld, _ = _make(workers=3, pf=2, n=700, delay=0.0005)
        exp = _expected(ld)
        got = []
        for step, (b, _) in enumerate(ld):
            got.append(int(b[0]))
            if step % rnd.randint(7, 25) == 0:
                ld.reconfigure(num_workers=rnd.randint(1, 9), prefetch_factor=rnd.randint(1, 5))
                ld.apply_pending()
        assert got == exp, "order changed (seed %d)" % seed
    os.environ.pop("STORMER_ELASTIC_POOL_MAX", None)
    time.sleep(0.3)
    assert not mp.active_children(), mp.active_children()


def test_take_handoff_s_separates_blocked_wait_from_handoff_and_resets():
    """Delivery family: blocked-queue wait and handoff mechanics are measured separately, then reset."""
    ld, _ = _make(workers=2, pf=2, n=60, delay=0.02)
    it = iter(ld)
    next(it)
    w1, h1 = ld.take_handoff_s()
    assert w1 > 0.0 and h1 >= 0.0               # the first batch had to wait for a worker
    assert ld.take_handoff_s() == (0.0, 0.0)    # reset by the read
    for _ in range(20):
        next(it)
    w, h = ld.take_handoff_s()
    assert w > 0.0 and 0.0 <= h < w             # handoff is the non-blocked part of next()
    stock, _ = _make(workers=2, pf=2, n=60, elastic=False)
    iter(stock)
    assert stock.take_handoff_s() is None       # stock iterator is not instrumented: no delivery split
    time.sleep(0.3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print("PASS", name, "%.1fs" % (time.time() - t))
