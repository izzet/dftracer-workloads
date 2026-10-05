"""One-step gradient equivalence between DDP bucket caps (check A of the comm-arm gate).

`grads_for_cap` runs one forward/backward on a fresh DDP wrapper of the SAME initial weights and
returns the post-all-reduce gradients; `compare` reports max abs / max rel differences.  Exact
arithmetic makes the caps equal; ring all-reduce summation order depends on element offset within a
bucket, so ulp-level (not necessarily bitwise) differences are expected.
"""

from __future__ import annotations

import copy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def grads_for_cap(model, batch, loss_fn, cap_mb, hook=None, device_ids=None):
    m = copy.deepcopy(model)
    ddp = DDP(m, bucket_cap_mb=cap_mb, device_ids=device_ids)
    if hook is not None:
        ddp.register_comm_hook([], hook)
    ddp.zero_grad(set_to_none=True)
    loss_fn(ddp, batch).backward()
    return {n: p.grad.detach().clone().cpu() for n, p in ddp.module.named_parameters()}


def compare(ga, gb):
    max_abs = 0.0
    max_rel = 0.0
    bitwise = True
    for name, a in ga.items():
        b = gb[name]
        d = (a.double() - b.double()).abs()
        max_abs = max(max_abs, float(d.max()) if d.numel() else 0.0)
        denom = a.double().abs().clamp_min(torch.finfo(torch.float32).tiny)
        max_rel = max(max_rel, float((d / denom).max()) if d.numel() else 0.0)
        bitwise = bitwise and torch.equal(a, b)
    return {"max_abs": max_abs, "max_rel": max_rel, "bitwise_equal": bitwise}
