"""Opt-in, CPU-safe retained-state artifacts for StorMer trajectory tests.

This module intentionally has no training-loop, launcher, or resume integration.
It captures immutable evidence snapshots for the reviewed FP32/scaler-none,
single-rank first cell.  Archive checksums establish artifact integrity; callers
must use :func:`tensorwise_differences` for state comparison.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "stormer-trajectory-state.v1"


def _cpu_copy(value: Any) -> Any:
    """Recursively detach tensors and own their CPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module without importing distributed machinery."""
    return model.module if hasattr(model, "module") else model


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    cuda_available = bool(include_cuda and torch.cuda.is_available())
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": _cpu_copy(np.random.get_state()),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "cuda_available": cuda_available,
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if cuda_available else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("cuda_available"):
        if not torch.cuda.is_available():
            raise RuntimeError("snapshot contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _named_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    names = {id(param): name for name, param in unwrap_model(model).named_parameters()}
    state: dict[str, Any] = {}
    for parameter, value in optimizer.state.items():
        name = names.get(id(parameter))
        if name is None:
            raise ValueError("optimizer contains a parameter absent from the unwrapped model")
        state[name] = _cpu_copy(value)
    groups = []
    for group in optimizer.param_groups:
        copied = {key: _cpu_copy(value) for key, value in group.items() if key != "params"}
        try:
            copied["params"] = [names[id(parameter)] for parameter in group["params"]]
        except KeyError as exc:
            raise ValueError("optimizer param group contains a parameter absent from the model") from exc
        groups.append(copied)
    return {"state_by_name": state, "param_groups": groups}


def _state_dict_or_none(value: Any) -> Any:
    return None if value is None else _cpu_copy(value.state_dict())


def _disabled_scaler_state(scaler: Any) -> dict[str, Any]:
    """State-v1 admits the application's disabled FP32 GradScaler, never AMP."""
    if scaler is None:
        return {"present": False, "enabled": False, "state": None}
    if not hasattr(scaler, "is_enabled") or not hasattr(scaler, "state_dict"):
        raise TypeError("scaler must be None or a GradScaler-like object")
    if scaler.is_enabled():
        raise ValueError("state-v1 first cell requires an inactive FP32 scaler")
    return {"present": True, "enabled": False, "state": _cpu_copy(scaler.state_dict())}


def _nonfinite_paths(value: Any, path: str = "state") -> list[str]:
    if isinstance(value, torch.Tensor):
        return [path] if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(torch.isfinite(value).all()) else []
    if isinstance(value, np.ndarray):
        return [path] if (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.complexfloating)) and not bool(np.isfinite(value).all()) else []
    if isinstance(value, (float, np.floating)):
        return [path] if not math.isfinite(float(value)) else []
    if isinstance(value, Mapping):
        return [bad for key, item in value.items() for bad in _nonfinite_paths(item, f"{path}.{key}")]
    if isinstance(value, (list, tuple)):
        return [bad for index, item in enumerate(value) for bad in _nonfinite_paths(item, f"{path}[{index}]")]
    return []


def capture_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    completed_update_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture a standalone state-v1 payload for the reviewed FP32 first cell."""
    if isinstance(completed_update_id, bool) or not isinstance(completed_update_id, int) or completed_update_id < 0:
        raise ValueError("completed_update_id must be a nonnegative integer")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    kind = metadata.get("retention_kind")
    if completed_update_id == 0:
        if kind != "initial":
            raise ValueError("update 0 must be marked retention_kind=initial")
    elif kind != "epoch_end":
        raise ValueError("state-v1 retains only predeclared epoch-end updates")
    if completed_update_id > 0 and (
        metadata.get("observed_update") is not True or metadata.get("loss_finite") is not True
    ):
        raise ValueError("post-update retention requires an observed finite optimizer update")
    try:
        json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must be JSON-serializable") from exc
    # GPU writes from the completed update must be visible before owned CPU copies are made.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    module = unwrap_model(model)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "completed_update_id": completed_update_id,
        "metadata": copy.deepcopy(dict(metadata)),
        "model_state": _cpu_copy(module.state_dict()),
        "optimizer_state": _named_optimizer_state(module, optimizer),
        "scheduler_state": _state_dict_or_none(scheduler),
        "scaler_state": _disabled_scaler_state(scaler),
        "rng_state": capture_rng_state(),
    }
    nonfinite = _nonfinite_paths(snapshot)
    if nonfinite:
        raise ValueError("refusing nonfinite retained state: " + ", ".join(nonfinite[:3]))
    return snapshot


def _tensor_digest(value: Any, hasher: "hashlib._Hash") -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(repr(tuple(tensor.shape)).encode())
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        hasher.update(b"ndarray\0")
        hasher.update(str(value.dtype).encode())
        hasher.update(repr(tuple(value.shape)).encode())
        hasher.update(value.tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"mapping\0")
        for key in sorted(value, key=str):
            hasher.update(repr(key).encode())
            _tensor_digest(value[key], hasher)
    elif isinstance(value, (list, tuple)):
        hasher.update(b"sequence\0")
        for item in value:
            _tensor_digest(item, hasher)
    else:
        hasher.update(b"scalar\0")
        hasher.update(repr(value).encode())


def component_tensor_digests(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """Stable tensor-content digests, distinct from the serialized archive hash."""
    result = {}
    for name in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng_state"):
        digest = hashlib.sha256()
        _tensor_digest(snapshot.get(name), digest)
        result[name] = digest.hexdigest()
    return result


def tensorwise_differences(left: Any, right: Any, *, atol: float = 0.0, rtol: float = 0.0, path: str = "state") -> list[str]:
    """Return structural/tensor differences; archive bytes are deliberately irrelevant."""
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return [f"{path}: tensor dtype/shape differs"]
        if (torch.is_floating_point(left) or torch.is_complex(left)) and (
            not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all())
        ):
            return [f"{path}: nonfinite tensor"]
        if torch.is_floating_point(left) or torch.is_complex(left):
            return [] if torch.allclose(left, right, atol=atol, rtol=rtol) else [f"{path}: tensor values differ"]
        return [] if torch.equal(left, right) else [f"{path}: tensor values differ"]
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or left.shape != right.shape:
            return [f"{path}: array dtype/shape differs"]
        if (np.issubdtype(left.dtype, np.floating) or np.issubdtype(left.dtype, np.complexfloating)) and (
            not bool(np.isfinite(left).all()) or not bool(np.isfinite(right).all())
        ):
            return [f"{path}: nonfinite array"]
        return [] if np.array_equal(left, right) else [f"{path}: array values differ"]
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping):
            return [f"{path}: type differs"]
        differences = []
        if set(left) != set(right):
            differences.append(f"{path}: keys differ")
        for key in sorted(set(left) & set(right), key=str):
            differences.extend(tensorwise_differences(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}"))
        return differences
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)):
            return [f"{path}: type differs"]
        if len(left) != len(right):
            return [f"{path}: length differs"]
        return [difference for index, (a, b) in enumerate(zip(left, right))
                for difference in tensorwise_differences(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]")]
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        if not isinstance(left, (float, np.floating)) or not isinstance(right, (float, np.floating)):
            return [f"{path}: type differs"]
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return [f"{path}: nonfinite scalar"]
    return [] if type(left) is type(right) and left == right else [f"{path}: value differs"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_retained_state(destination: str | Path, snapshot: Mapping[str, Any]) -> Path:
    """Atomically publish one complete owned snapshot directory, never overwriting one."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing existing or partial retained artifact: {destination}")
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected retained-state schema")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        payload = _cpu_copy(dict(snapshot))
        serialized = io.BytesIO()
        torch.save(payload, serialized)
        state_bytes = serialized.getvalue()
        state_sha256 = _sha256(state_bytes)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "completed_update_id": payload["completed_update_id"],
            "metadata": payload["metadata"],
            "artifact": {"state_pt_sha256": state_sha256},
            "tensor_component_sha256": component_tensor_digests(payload),
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        (staging / "state.pt").write_bytes(state_bytes)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        (staging / "sha256sums.txt").write_text(
            f"{state_sha256}  state.pt\n{_sha256(manifest_bytes)}  manifest.json\n"
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def verify_retained_state(destination: str | Path) -> dict[str, Any]:
    destination = Path(destination)
    expected_files = {"state.pt", "manifest.json", "sha256sums.txt"}
    checksummed_files = {"state.pt", "manifest.json"}
    if not destination.is_dir() or set(path.name for path in destination.iterdir()) != expected_files:
        raise ValueError(f"retained artifact is partial or has unexpected files: {destination}")
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("retained artifact schema mismatch")
    checksums = {}
    for line in (destination / "sha256sums.txt").read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name not in checksummed_files or name in checksums or len(digest) != 64:
            raise ValueError("invalid retained artifact checksum ledger")
        checksums[name] = digest
    if set(checksums) != checksummed_files:
        raise ValueError("retained artifact checksum ledger is incomplete")
    for name, digest in checksums.items():
        if _sha256((destination / name).read_bytes()) != digest:
            raise ValueError(f"{name} checksum ledger mismatch")
    if checksums["state.pt"] != manifest.get("artifact", {}).get("state_pt_sha256"):
        raise ValueError("state.pt integrity hash mismatch")
    # This load is verification-only.  v1 deliberately provides no restore/resume API.
    payload = torch.load(destination / "state.pt", map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("state.pt schema mismatch")
    if payload.get("completed_update_id") != manifest.get("completed_update_id"):
        raise ValueError("manifest/payload completed-update ID mismatch")
    if payload.get("metadata") != manifest.get("metadata"):
        raise ValueError("manifest/payload metadata mismatch")
    nonfinite = _nonfinite_paths(payload)
    if nonfinite:
        raise ValueError("retained state contains nonfinite tensors: " + ", ".join(nonfinite[:3]))
    if component_tensor_digests(payload) != manifest.get("tensor_component_sha256"):
        raise ValueError("retained state tensor-content digest mismatch")
    return manifest


@contextmanager
def validation_preserving_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any = None,
) -> Iterator[None]:
    """Restore RNG/mode and reject validation mutation of retained training state."""
    mode = model.training
    rng = capture_rng_state()
    before = _cpu_copy(unwrap_model(model).state_dict())
    optimizer_before = _named_optimizer_state(model, optimizer)
    scheduler_before = _state_dict_or_none(scheduler)
    scaler_before = _disabled_scaler_state(scaler)
    try:
        yield
        after = _cpu_copy(unwrap_model(model).state_dict())
        differences = tensorwise_differences(before, after)
        differences.extend(tensorwise_differences(
            optimizer_before, _named_optimizer_state(model, optimizer), path="optimizer"
        ))
        differences.extend(tensorwise_differences(
            scheduler_before, _state_dict_or_none(scheduler), path="scheduler"
        ))
        differences.extend(tensorwise_differences(
            scaler_before, _disabled_scaler_state(scaler), path="scaler"
        ))
        if differences:
            raise RuntimeError("validation mutated model state: " + "; ".join(differences[:3]))
    finally:
        model.train(mode)
        restore_rng_state(rng)
