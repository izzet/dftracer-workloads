"""Fail-closed loading of the separately versioned ERA5 latitude metadata.

This validates the native corpus and the exact 721x1440 -> 128x256 resize
contract before StorMer uses a corrected latitude vector for its loss.  It
never changes the corpus or creates a metadata artifact.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch


SCHEMA = "stormer-era5-latitude-metadata.v2"
ARTIFACT_ID = "stormer_era5_025deg_latitude_v2"
MANIFEST_SHA256 = "2923ef960677f0d4953a041c81aecce951b8b80bda428ada7cb26892abade893"
NATIVE_SHAPE = (721, 1440)
MODEL_SHAPE = (128, 256)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError("StorMer latitude metadata: " + reason)


def _array_sha256(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _verify_recorded_bindings(source, evidence):
    """Check the bounded train/val surface and pressure samples in the manifest."""
    bindings = evidence["sampled_field_bindings_recorded_from_proposal"]
    expected = {
        ("train", "2m_temperature", "2000-01-01"),
        ("train", "geopotential_100", "2000-01-01"),
        ("validation", "2m_temperature", "2019-01-01"),
        ("validation", "geopotential_100", "2019-01-01"),
    }
    _require({(b["split"], b["field"], b["date"]) for b in bindings} == expected
             and len(bindings) == 4, "sampled field binding set mismatch")
    latitude_hash = evidence["netcdf_latitude"]["latitude_raw_bytes_sha256"]
    for binding in bindings:
        field_name = binding["field"]
        with h5py.File(source / binding["netcdf_path"], "r") as netcdf:
            _require(_array_sha256(netcdf["latitude"][:]) == latitude_hash,
                     "sampled source coordinate mismatch: " + binding["netcdf_path"])
            if field_name == "geopotential_100":
                levels = np.flatnonzero(netcdf["level"][:] == 100)
                _require(len(levels) == 1, "sampled source geopotential level mismatch")
                source_field = netcdf["geopotential"][0, int(levels[0])]
            else:
                source_field = netcdf["2m_temperature"][0]
        with h5py.File(source / binding["hdf5_path"], "r") as hdf5:
            hdf_field = hdf5["input/" + field_name][...]
        _require(tuple(source_field.shape) == NATIVE_SHAPE
                 and tuple(hdf_field.shape) == NATIVE_SHAPE,
                 "sampled field native dimensions mismatch: " + field_name)
        _require(_array_sha256(source_field) == binding["array_sha256"]
                 and _array_sha256(hdf_field) == binding["array_sha256"],
                 "sampled field binding hash mismatch: " + binding["hdf5_path"]
                 + ":" + field_name)


def load_latitude(data_folder, metadata_dir, image_shape, downsample):
    """Return verified model-row latitude degrees, or reject a mismatched cell."""
    data = Path(data_folder).resolve(strict=True)
    model_shape = tuple(int(value) for value in image_shape)
    _require(len(model_shape) == 2, "image_shape must be [height, width]")

    if not metadata_dir:
        # Keep standalone physical-coordinate corpora usable, but never admit
        # the native 0..127 index vector as a nonnegative latitude loss.
        native = np.load(data / "lat.npy", allow_pickle=False)
        _require(native.shape == (model_shape[0],), "native lat.npy shape does not match model rows")
        _require(np.all(np.isfinite(native)), "native lat.npy has nonfinite values")
        _require(np.all((-90.0 <= native) & (native <= 90.0)),
                 "native lat.npy contains values outside physical latitude degrees; provide reviewed metadata")
        return native

    artifact = Path(metadata_dir).resolve(strict=True)
    manifest_path = artifact / "manifest.json"
    _require(_sha256(manifest_path) == MANIFEST_SHA256, "reviewed v2 manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA, "schema mismatch")
    _require(manifest.get("artifact_id") == ARTIFACT_ID, "artifact identity mismatch")
    provenance = manifest["provenance"]
    resize = manifest["resize"]
    latitude = manifest["latitude"]
    _require(Path(provenance["corpus_root"]).resolve() == data, "corpus root mismatch")
    _require(bool(downsample), "reviewed metadata requires downsampling")
    _require(model_shape == MODEL_SHAPE, "reviewed metadata requires model shape 128x256")
    _require((resize["source_height"], resize["source_width"]) == NATIVE_SHAPE,
             "native resize shape mismatch")
    _require((resize["target_height"], resize["target_width"]) == MODEL_SHAPE,
             "model resize shape mismatch")
    _require(resize["mode"] == "bilinear" and resize["align_corners"] is False,
             "interpolation convention mismatch")
    _require(manifest["runtime"]["torch_version"] == torch.__version__,
             "pinned PyTorch version mismatch")
    _require(latitude["dtype"] == "float64" and latitude["shape"] == [128]
             and latitude["units"] == "degrees_north"
             and latitude["order"] == "north_to_south", "latitude contract mismatch")

    native_lat = data / "lat.npy"
    _require(_sha256(native_lat) == provenance["native_lat_npy"]["sha256"],
             "native lat.npy hash mismatch")
    for name, expected_hash in provenance["normalization_files"].items():
        _require(_sha256(data / name) == expected_hash,
                 "normalization hash mismatch: " + name)

    source = data.parent
    evidence = provenance["source_evidence"]
    for item in (evidence["netcdf_latitude"], evidence["producer"]):
        _require(_sha256(source / item["path"]) == item["sha256"],
                 "source provenance hash mismatch: " + item["path"])
    sampled = data / "train" / "2000_0000.h5"
    with h5py.File(sampled, "r") as stream:
        field = stream["input/land_sea_mask"]
        _require(tuple(field.shape) == NATIVE_SHAPE, "sampled native HDF5 shape mismatch")
        actual_hash = hashlib.sha256(field[...].tobytes()).hexdigest()
    _require(actual_hash == evidence["sampled_land_sea_mask"]["array_sha256"],
             "sampled native HDF5 field hash mismatch")
    _verify_recorded_bindings(source, evidence)

    lat_path = artifact / manifest["artifact"]["lat_npy"]
    _require(lat_path.parent == artifact and lat_path.name == "lat.npy",
             "artifact latitude filename mismatch")
    _require(_sha256(lat_path) == manifest["artifact"]["lat_npy_sha256"],
             "artifact latitude hash mismatch")
    lat = np.load(lat_path, allow_pickle=False)
    _require(lat.dtype == np.float64 and lat.shape == (128,), "artifact latitude shape/dtype mismatch")
    rows = np.arange(128, dtype=np.float64)
    expected = 90.0 - 0.25 * ((rows + 0.5) * 721.0 / 128.0 - 0.5)
    _require(np.array_equal(lat, expected), "artifact latitude values differ from reviewed formula")
    _require(np.all(np.isfinite(lat)) and np.all(np.cos(np.deg2rad(lat)) >= 0.0),
             "artifact latitude gives invalid loss weights")
    return lat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--image-height", required=True, type=int)
    parser.add_argument("--image-width", required=True, type=int)
    parser.add_argument("--downsample", choices=("on", "off"), required=True)
    args = parser.parse_args()
    lat = load_latitude(args.data_folder, args.metadata_dir,
                        (args.image_height, args.image_width), args.downsample == "on")
    print("verified StorMer latitude metadata: %s rows, artifact %s" % (len(lat), ARTIFACT_ID))


if __name__ == "__main__":
    main()
