"""Fast DDP rebuild: device-side fingerprint, cross-rank check, init_sync skip, no-op rebuild leaves the trajectory bitwise unchanged."""
import os
import socket
import sys
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(__file__))
import rebuild_invariants as ri


def _model():
    torch.manual_seed(7)
    return torch.nn.Sequential(torch.nn.Linear(64, 96), torch.nn.GELU(), torch.nn.Linear(96, 64), torch.nn.LayerNorm(64))


class _GlooComm:
    """mpi4py-shaped Allreduce over torch.distributed (gloo), ops are 'min'/'max'."""
    def Allreduce(self, send, recv, op):
        t = torch.from_numpy(np.array(send, copy=True))
        dist.all_reduce(t, op=dist.ReduceOp.MIN if op == "min" else dist.ReduceOp.MAX)
        recv[...] = t.numpy()


class FingerprintTests(unittest.TestCase):
    def test_detects_bit_change_permutation_and_param_group_change(self):
        m = _model(); o = torch.optim.Adam(m.parameters(), lr=1e-3)
        a, va = ri.fingerprint(m, o)
        self.assertEqual(a, ri.fingerprint(m, o)[0])
        with torch.no_grad():
            p = next(m.parameters()); p.view(-1)[0] += 1e-7
        b, _ = ri.fingerprint(m, o)
        self.assertNotEqual(a, b)
        with torch.no_grad():
            p.view(-1)[0] -= 1e-7
            q = list(m.parameters())[2]; q.view(-1)[[0, 1]] = q.view(-1)[[1, 0]].clone()   # swap two values
        self.assertNotEqual(a, ri.fingerprint(m, o)[0])
        m2 = _model(); o2 = torch.optim.Adam(m2.parameters(), lr=2e-3)
        self.assertNotEqual(a, ri.fingerprint(m2, o2)[0])

    def test_odd_sized_and_integer_tensors(self):
        m = torch.nn.Sequential(torch.nn.Linear(3, 5)); m.register_buffer("n", torch.tensor([1, 2, 3], dtype=torch.int16))
        a, _ = ri.fingerprint(m)
        m.n[1] = 9
        self.assertNotEqual(a, ri.fingerprint(m)[0])


def _run(rank, world, port, mode, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.manual_seed(100 + rank)
    xs = [torch.randn(8, 64) for _ in range(6)]
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ddp = DDP(model, bucket_cap_mb=0.005)
    comm, ops = _GlooComm(), ("min", "max")
    out, info = [], None
    for i, x in enumerate(xs):
        if i == 3 and mode != "none":
            if mode == "diverged" and rank == 1:
                with torch.no_grad():
                    next(ddp.module.parameters()).view(-1)[0] += 1.0       # a rank that disagrees
            if mode == "legacy":
                ddp = DDP(ddp.module, bucket_cap_mb=0.005)
            else:
                ddp, info = ri.fast_rewrap(ddp.module, opt, DDP, dict(bucket_cap_mb=0.005), comm, ops=ops)
        opt.zero_grad(set_to_none=True)
        loss = ddp(x).pow(2).mean()
        loss.backward()
        gn = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in ddp.module.parameters()))
        opt.step()
        out.append((repr(float(loss)), repr(float(gn))))
    q.put((rank, out, info and {k: v for k, v in info.items() if k != "phases"}, ri.fingerprint(ddp.module)[0]))
    dist.destroy_process_group()


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _launch(mode):
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    # both ranks must share the same port
    port = _port()
    ps = [ctx.Process(target=_run, args=(r, 2, port, mode, q)) for r in range(2)]
    [p.start() for p in ps]
    res = sorted([q.get(timeout=180) for _ in ps])
    [p.join(30) for p in ps]
    return res


class TwoRankTests(unittest.TestCase):
    def test_noop_rebuild_keeps_loss_and_grad_norm_bitwise(self):
        base = _launch("none")
        fast = _launch("fast")
        legacy = _launch("legacy")
        for r in range(2):
            self.assertEqual(base[r][1], fast[r][1], "fast no-op rebuild changed the loss/grad-norm sequence")
            self.assertEqual(base[r][1], legacy[r][1])
            self.assertEqual(base[r][3], fast[r][3])
        self.assertFalse(fast[0][2]["init_sync"], "identical ranks must skip the broadcast")
        self.assertTrue(fast[0][2]["ok"] and fast[1][2]["ok"])

    def test_diverged_rank_forces_the_broadcast(self):
        res = _launch("diverged")
        self.assertTrue(res[0][2]["init_sync"] and res[1][2]["init_sync"])      # every rank takes the same path
        self.assertEqual(res[0][3], res[1][3])                                  # and ends identical (rank 0 wins)


if __name__ == "__main__":
    unittest.main()
