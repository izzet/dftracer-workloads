"""CPU tests for the comm second-domain seam: post-apply ACKs, hook survival across DDP
rebuild, rebuild invariants, and one-step gradient equivalence.  No GPU, dataset, or train import.

Run from this directory: python -m unittest test_comm_second_domain
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import apply_ack
import comm_equivalence
import comm_hook
import rebuild_invariants


class _Ctx:
    def __init__(self):
        self.acks = []

    def ack_action(self, plan, old, new, status="applied", reason="", **kw):
        self.acks.append((plan, old, new, status, reason))


def _model():
    torch.manual_seed(7)
    return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Tanh(), torch.nn.Linear(128, 64))


class ApplyAckTests(unittest.TestCase):
    W = {"num_workers": 4, "bucket_cap_mb": 400}

    def test_all_ranks_confirm_gives_applied_with_observed_value(self):
        ctx = _Ctx()
        obs = {"num_workers": 4, "bucket_cap_mb": 400}
        flags = apply_ack.local_flags(self.W, obs)
        out = apply_ack.ack_pending(ctx, [("p1", "num_workers", 8), ("p2", "bucket_cap_mb", 25)],
                                    flags, self.W, obs)
        self.assertEqual(out, [("num_workers", "applied"), ("bucket_cap_mb", "applied")])
        self.assertEqual(ctx.acks[0][1:3], (8, 4))

    def test_one_rank_mismatch_fails_only_that_knob(self):
        ctx = _Ctx()
        good = apply_ack.local_flags(self.W, {"num_workers": 4, "bucket_cap_mb": 400})
        bad = apply_ack.local_flags(self.W, {"num_workers": None, "bucket_cap_mb": 400})
        reduced = [min(a, b) for a, b in zip(good, bad)]          # what Allreduce(MIN) yields
        out = apply_ack.ack_pending(ctx, [("p1", "num_workers", 8), ("p2", "bucket_cap_mb", 25)],
                                    reduced, self.W, {"num_workers": 4, "bucket_cap_mb": 400})
        self.assertEqual(out, [("num_workers", "failed"), ("bucket_cap_mb", "applied")])
        self.assertIn("apply_not_confirmed_on_all_ranks", ctx.acks[0][4])

    def test_receipt_gap_is_logged(self):
        with self.assertLogs("apply_ack", level="INFO") as cm:
            apply_ack.ack_pending(_Ctx(), [("p", "num_workers", 1)],
                                  apply_ack.local_flags({"num_workers": 2}, {"num_workers": 2}),
                                  {"num_workers": 2}, {"num_workers": 2})
        self.assertTrue(any("effective_window_id=None" in m for m in cm.output))

    def test_withheld_knob_is_rejected_not_applied(self):
        ctx = _Ctx()
        w = {"decode_threads": 8}
        wh = frozenset(["decode_threads"])
        flags = apply_ack.local_flags(w, {"decode_threads": 1}, wh)
        out = apply_ack.ack_pending(ctx, [("p", "decode_threads", 1)], flags, w,
                                    {"decode_threads": 1}, wh)
        self.assertEqual(out, [("decode_threads", "rejected")])

    def test_failed_rebuild_sentinel_never_matches_target(self):
        flags = apply_ack.local_flags({"bucket_cap_mb": 400}, {"bucket_cap_mb": -2})
        self.assertEqual(flags[apply_ack.KNOBS.index("bucket_cap_mb")], 0)


class TrainSourceRegressionTests(unittest.TestCase):
    """train.py cannot be imported on a CPU login node; assert the wiring textually."""
    src = (Path(__file__).with_name("train.py")).read_text()

    def test_only_reject_path_acks_directly(self):
        # One rejection path, plus two `failed` acks that exist only under STORMER_CONSENSUS=unanimous (stormer-loop patch 03,
        # judge D1d): a superseded or expired bucket_cap vote must be answered so its planner reverts to the applied value.
        direct = re.findall(r"ctx\.ack_action\(", self.src)
        self.assertEqual(len(direct), 3)
        parts = self.src.split("ctx.ack_action(")
        self.assertIn('status="rejected"', parts[1][:120])
        self.assertIn('reason="consensus_superseded"', parts[2][:160])
        self.assertIn('reason="no_unanimity_within_sticky_window"', parts[3][:200])
        self.assertIn("apply_ack.ack_pending(", self.src)

    def test_rebuild_and_setup_share_the_hook_with_trace_region(self):
        self.assertEqual(self.src.count("comm_hook.make_hook_for_mode(dft_ai.comm.all_reduce"), 2)
        self.assertNotIn("def _comm_hook(", self.src)

    def test_prefetch_observation_never_substituted(self):
        self.assertNotIn("want_pf if _settled", self.src)
        self.assertIn('_obs["prefetch_factor"] = _opf if _settled else None', self.src)

    def test_step_series_is_opt_in(self):
        self.assertIn('os.environ.get("STORMER_STEP_WALL", "0") == "1"', self.src)
        self.assertIn("STEP_SERIES rank=%d epoch=%d", self.src)

    def test_rebuild_checks_invariants_and_refuses_accumulation(self):
        self.assertIn("rebuild_invariants.boundary_problem", self.src)
        self.assertIn("rebuild_invariants.state_digest", self.src)


class ProbeModeTests(unittest.TestCase):
    def test_default_blocking_and_off(self):
        os.environ.pop("STORMER_COMM_PROBE", None)
        self.assertEqual(comm_hook.comm_probe_mode(), "blocking")
        self.assertTrue(comm_hook.hook_wanted(False, False, True, 4))
        os.environ["STORMER_COMM_PROBE"] = "off"
        try:
            self.assertFalse(comm_hook.hook_wanted(True, True, True, 4))
            os.environ["STORMER_COMM_PROBE"] = "bogus"
            with self.assertRaises(ValueError):
                comm_hook.comm_probe_mode()
        finally:
            os.environ.pop("STORMER_COMM_PROBE", None)

    def test_single_rank_never_hooks(self):
        self.assertFalse(comm_hook.hook_wanted(True, True, True, 1))


class _Region:
    def __init__(self):
        self.entered = 0

    def __enter__(self):
        self.entered += 1

    def __exit__(self, *a):
        return False


def _worker(rank, world, tmp, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=os.environ["_TEST_PORT"])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(100 + rank)
    x = torch.randn(8, 64)
    loss_fn = lambda m, b: m(b).pow(2).mean()
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # (A) one-step gradient equivalence across caps (many buckets vs one bucket)
    ga = comm_equivalence.grads_for_cap(model, x, loss_fn, 0.005)
    gb = comm_equivalence.grads_for_cap(model, x, loss_fn, 400)
    cmp = comm_equivalence.compare(ga, gb)

    # (C) rebuild keeps region + byte-identical state; hook state gets events after rebuild
    region = _Region()
    ddp = DDP(model, bucket_cap_mb=0.005)
    buckets = []
    ddp.register_comm_hook(buckets, comm_hook.make_blocking_hook(region))
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss_fn(ddp, x).backward()
        opt.step()
    before_events = region.entered
    d0 = rebuild_invariants.state_digest(ddp.module, opt)
    ddp = DDP(ddp.module, bucket_cap_mb=400)
    ddp.register_comm_hook(buckets, comm_hook.make_blocking_hook(region))
    d1 = rebuild_invariants.state_digest(ddp.module, opt)
    opt.zero_grad(set_to_none=True)
    loss_fn(ddp, x).backward()
    opt.step()
    q.put((rank, cmp, before_events > 0, region.entered > before_events, d0 == d1))
    dist.destroy_process_group()


class GlooTwoRankTests(unittest.TestCase):
    def test_grad_equivalence_and_rebuild_invariants(self):
        import socket
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        os.environ["_TEST_PORT"] = str(port)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            ps = [ctx.Process(target=_worker, args=(r, 2, tmp, q)) for r in range(2)]
            [p.start() for p in ps]
            res = [q.get(timeout=120) for _ in ps]
            [p.join(30) for p in ps]
        for rank, cmp, ev_before, ev_after, same in res:
            self.assertTrue(ev_before, "hook region entered before rebuild")
            self.assertTrue(ev_after, "hook region STILL entered after rebuild")
            self.assertTrue(same, "state digest changed across rebuild")
            self.assertLessEqual(cmp["max_rel"], 1e-5, cmp)
            print("rank", rank, "grad-equivalence", cmp)


class BoundaryTests(unittest.TestCase):
    def test_accumulation_refuses_rebuild(self):
        self.assertIsNone(rebuild_invariants.boundary_problem(1))
        self.assertIsNotNone(rebuild_invariants.boundary_problem(2))

    def test_digest_detects_single_byte_change(self):
        m = _model(); o = torch.optim.Adam(m.parameters())
        a = rebuild_invariants.state_digest(m, o)
        with torch.no_grad():
            next(m.parameters()).view(-1)[0] += 1e-7
        self.assertNotEqual(a, rebuild_invariants.state_digest(m, o))


if __name__ == "__main__":
    unittest.main()
