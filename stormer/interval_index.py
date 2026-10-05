"""Portable identity of the ordered StorMer dataset-index to HDF-file mapping.

This fingerprints the exact ``ERA5Dataset.inp_file_paths`` after its existing
sort/filter/tail rules. It does not establish full HDF content identity.
"""

import hashlib
import json
from pathlib import Path


INDEX_FINGERPRINT_VERSION = "stormer-index-fingerprint.v1"


def fingerprint_index(file_paths, root_dir):
    root = Path(root_dir)
    relative = []
    for file_path in file_paths:
        path = Path(file_path)
        if path.parent != root:
            raise ValueError(f"input outside the dataset split: {path}")
        relative.append(path.name)
    payload = json.dumps(
        [INDEX_FINGERPRINT_VERSION, relative],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return {"count": len(relative), "sha256": hashlib.sha256(payload).hexdigest()}


def verify_index(file_paths, root_dir, manifest_path, split, *,
                 intervals, data_freq, selector_version):
    if split not in ("train", "val"):
        raise ValueError(f"unsupported StorMer split: {split}")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != INDEX_FINGERPRINT_VERSION:
        raise ValueError("StorMer index manifest schema mismatch")
    for key, actual_value in (
        ("intervals_hours", list(intervals)),
        ("data_freq_hours", data_freq),
        ("selector_version", selector_version),
    ):
        if manifest.get(key) != actual_value:
            raise ValueError(f"StorMer interval contract mismatch: {key}")
    expected = manifest.get("splits", {}).get(split)
    actual = fingerprint_index(file_paths, root_dir)
    if expected != actual:
        raise ValueError(
            f"StorMer {split} index mapping differs from frozen manifest: "
            f"expected={expected}, actual={actual}"
        )
    return actual
