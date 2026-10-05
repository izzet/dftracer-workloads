"""Byte-level invariants for a live DDP rebuild (bucket_cap_mb actuation).

A rebuild only re-wraps the SAME module, so parameters, buffers, and optimizer state must be
byte-identical across it, and it must happen at an optimizer-step boundary with accumulation off
(otherwise a partial accumulation window could be lost or double-applied).
"""

from __future__ import annotations

import hashlib

import torch


def _feed(h, tensor: torch.Tensor):
    t = tensor.detach().contiguous().cpu().reshape(-1)
    h.update(str(t.dtype).encode())
    h.update(str(tuple(tensor.shape)).encode())
    h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")


def state_digest(module: torch.nn.Module, optimizer=None) -> str:
    """SHA-256 over params, buffers (name order) and optimizer state (param-index order)."""
    h = hashlib.sha256()
    for name, p in module.named_parameters():
        h.update(name.encode()); _feed(h, p)
    for name, b in module.named_buffers():
        h.update(name.encode()); _feed(h, b)
    if optimizer is not None:
        sd = optimizer.state_dict()
        for pid in sorted(sd["state"]):
            for key in sorted(sd["state"][pid]):
                v = sd["state"][pid][key]
                h.update(("%s/%s" % (pid, key)).encode())
                if torch.is_tensor(v):
                    _feed(h, v)
                else:
                    h.update(repr(v).encode())
        h.update(repr(sd["param_groups"]).encode())
    return h.hexdigest()


def boundary_problem(accumulate_grad_batches: int):
    """None if a rebuild is safe, else the reason.

    The apply window runs after a completed training step, so with accumulation off the optimizer
    step has been applied and the next step zero_grads first: no gradient is lost or double-applied.
    With accumulation on, a window boundary may fall inside an accumulation window: refuse.
    """
    if int(accumulate_grad_batches) != 1:
        return ("accumulate_grad_batches=%d != 1: rebuild could split an accumulation window"
                % accumulate_grad_batches)
    return None
