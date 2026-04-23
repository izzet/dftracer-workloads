# Taken from https://github.com/tung-nd/stormer
# Modified by Troy Arcomano, Sam Foreman, Ray Andrew Sinurat
#
# MIT License
#
# Copyright (c) 2024 Tung Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Historical versionings:
# - Tung and Troy worked on Stormer
# - Troy and Sam worked on pure pytorch version + mpi4py
# - Ray modified the codebase for
#   - profiling I/O
#   - replicating the Stormer pipeline in DLIO Benchmark
#   - I/O Optimization
#
# This codebase is for single step pretraining only and no validation

from typing import Any
import logging
import os
import time
import contextlib
import argparse
from glob import glob
from contextlib import contextmanager
from collections.abc import Generator
import warnings

# LC HACK: work around so that "import torch" will not change CPU affinity
# see https://rzlc.llnl.gov/jira/browse/ELCAP-386
os.environ.pop("OMP_PLACES", None)
os.environ.pop("OMP_PROC_BIND", None)

# from mpi4py import MPI

import h5py
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.nn.parallel.distributed import _MixedPrecision
except ImportError:
    _MixedPrecision = None
from torch.utils.data import Dataset
import torch.distributed as dist

from models.stormer.stormer import Stormer
from models.scheduler import configure_lr_scheduler
from models.optimizer import configure_optimizers
from models.utils import replace_constant, pad
from models.metrics import lat_weighted_mse
from models.data_utils import WEIGHT_DICT

from utils.torch_utils import seed_everything
from utils.precision import setup_precision_context, get_precision_dtype
from utils.perf_tracer import PerfTracer, PERFTRACER_ENABLE, dft_ai
from utils.logging import configure_logging
from utils.mpi import MPIUtils, get_master_addr_and_port


warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger(__name__)

# MPIUtils.rank() = int(os.environ["FLUX_TASK_RANK"])
# MPI_SIZE = int(os.environ["FLUX_JOB_SIZE"])
# MPI_LOCAL_RANK = int(os.environ["FLUX_TASK_LOCAL_ID"])
# MPI_NODES = int(os.environ["FLUX_JOB_NNODES"])
GPUS = os.environ.get("ROCR_VISIBLE_DEVICES", "")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 # since we only see 1 gpu, so GPU that is observed by the process will be 0


def get_ddp_reduce_dtype(value: str | None):
    if value is None:
        return None
    value = value.strip().lower()
    if value in ("", "none", "off", "0"):
        return None
    if value in ("16", "fp16", "float16"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported STORMER_DDP_REDUCE_DTYPE={value}")

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
]

def log0(*args, level: str = "info", **kwargs):
    if MPIUtils.rank() == 0:
        getattr(log, level)(*args, **kwargs)

class ERA5Dataset(Dataset):
    def __init__(
        self,
        args,
        root_dir: str,
        variables: list[str],
        inp_transform: torch.nn.Module,
        out_transform_dict: dict[int, torch.nn.Module],
        list_intervals: list[int] = [6, 12, 24],
        data_freq: int = 6,
        year_list: list[int] | None = None,
    ):
        self.perf_tracer: PerfTracer | None = None
        self.args = args
        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform_dict = out_transform_dict
        self.steps = 1
        self.list_intervals = list_intervals
        self.data_freq = data_freq
        self.year_list = year_list
        self.year_idx_map: dict | None = None
        self.image_shape = self.args.in_img_size
        self.downsample = os.environ.get("STORMER_DOWNSAMPLE", "1") == "1"
        if year_list is not None:
            self.year_idx_map = {year: i for i, year in enumerate(year_list)}

        file_paths = glob(os.path.join(root_dir, "*.h5"))
        file_paths = sorted(file_paths)

        final_file_paths: list[str] = []
        if self.year_list is not None:
            year_str = [str(year) for year in sorted(self.year_list)]
            for f in file_paths:
                for year in year_str:
                    if year in str(f):
                        final_file_paths.append(f)
        else:
            final_file_paths = file_paths

        self.inp_file_paths = final_file_paths[
            : -(self.steps * max(list_intervals) // data_freq)
        ]  # the last few points do not have ground-truth

    def get_out_path(
        self,
        root_dir: str,
        year: int,
        inp_file_idx: int,
        steps: int,
        year_list: list[int] | None = None,
        year_idx_map: dict[int, int] | None = None,
    ):
        # year: current year
        # inp_file_idx: file index of the input in the current year
        # steps: number of steps forward
        out_file_idx = inp_file_idx + steps
        out_path = os.path.join(root_dir, f"{year}_{out_file_idx:04}.h5")
        if not os.path.exists(out_path):
            max_step_forward = 0
            for i in range(1, steps):
                out_file_idx = inp_file_idx + i
                out_path = os.path.join(root_dir, f"{year}_{out_file_idx:04}.h5")
                if os.path.exists(out_path):
                    max_step_forward = i
            remaining_steps = steps - max_step_forward
            if year_list is None or year_idx_map is None:
                next_year = year + 1
            else:
                next_year = year_list[year_idx_map[year] + 1]
            out_path = os.path.join(root_dir, f"{next_year}_{remaining_steps - 1:04}.h5")
        return out_path

    def get_data_given_path(self, path: str, variables: list[str]):
        f = h5py.File(path, "r")
        x = []
        for var in variables:
            d = f["input"][var][:]
            x.append(d.reshape(1, *d.shape[-2:]))
        f.close()
        del f

        out = torch.from_numpy(np.concatenate(x))  # (V, H_file, W_file)
        if self.downsample and out.shape[-2:] != tuple(self.image_shape):
            out = torch.nn.functional.interpolate(
                out.unsqueeze(0),  # (1, V, H_file, W_file)
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (V, H_target, W_target)
        return out
    
    def __len__(self):
        return len(self.inp_file_paths)

    @dft_ai.data.item
    def __getitem__(self, index: int):

        # 1. Get input data
        inp_path = self.inp_file_paths[index]
        inp_data = self.get_data_given_path(inp_path, self.variables)

        chosen_interval = np.random.choice(self.list_intervals)
        year, inp_file_idx = os.path.basename(inp_path).split(".")[0].split("_")
        year, inp_file_idx = int(year), int(inp_file_idx)

        # 2. Get input data
        out_path = self.get_out_path(
            self.root_dir,
            year,
            inp_file_idx,
            steps=chosen_interval // self.data_freq,
            year_list=self.year_list,
            year_idx_map=self.year_idx_map,
        )
        out = self.get_data_given_path(out_path, self.variables)

        # 3. Preprocess
        with dft_ai.data.preprocess:
            diff = out - inp_data
            interval = torch.Tensor([chosen_interval]) / 10.0
            inp_transform = self.inp_transform(inp_data).unsqueeze(0)
            diff_transform = self.out_transform_dict[chosen_interval](diff).unsqueeze(0)

        return (
            inp_transform,
            diff_transform,
            interval,
        )
    
collate_profiler = dft_ai.data.preprocess.derive(name="collate")

@collate_profiler
def collate_fn_train(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])  # B, V, H, W
    out = torch.stack([batch[i][1] for i in range(len(batch))])  # B, T, V, H, W
    interval = torch.stack([batch[i][4] for i in range(len(batch))])  # B, T
    return inp, out, interval  # type: ignore[return-value]

class DataModule:
    @dft_ai.data.init
    def __init__(
        self,
        args,
        root_dir: str,
        variables: list[str],
        intervals: list[int],
    ):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data_freq = self.args.data_freq

        self.root_dir = root_dir
        self.variables = variables
        self.intervals = intervals

        # normalization for input
        normalize_mean = dict(np.load(os.path.join(self.root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(self.root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)

        out_transforms = {}
        for intv in intervals:
            normalize_diff_std = dict(
                np.load(os.path.join(root_dir, f"normalize_diff_std_{intv}.npz"))
            )
            normalize_diff_std = np.concatenate(
                [normalize_diff_std[v] for v in variables], axis=0
            )
            out_transforms[intv] = transforms.Normalize(
                np.zeros_like(normalize_diff_std), normalize_diff_std
            )

        self.transforms = transforms.Normalize(normalize_mean, normalize_std)
        self.out_transforms = out_transforms

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.root_dir, "lon.npy"))
        return lat, lon

    def get_transforms(self):
        return self.transforms, self.out_transforms

    def val_dataloader(self, max_samples=50):
        """Create a small validation DataLoader from the val/ split."""
        val_dir = os.path.join(self.root_dir, "val")
        if not os.path.isdir(val_dir):
            return None
        dataset = ERA5Dataset(
            args=self.args,
            root_dir=val_dir,
            variables=self.variables,
            inp_transform=self.transforms,
            out_transform_dict=self.out_transforms,
            list_intervals=self.intervals,
            data_freq=self.data_freq,
        )
        # Use a small fixed subset for fast validation
        if len(dataset) > max_samples:
            indices = list(range(0, len(dataset), len(dataset) // max_samples))[:max_samples]
            dataset = torch.utils.data.Subset(dataset, indices)
        batch_size = self.batch_size
        collate_fn = collate_fn_train
        if self.args.disable_collation and batch_size == 1:
            batch_size = None
            collate_fn = None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # keep it simple for val
            pin_memory=False,
            collate_fn=collate_fn,
        )

    def dataloader(self, loader_config: dict[str, Any] | None = None):
        if loader_config is None:
            loader_config = {
                "num_workers": self.args.num_workers,
                "pin_memory": self.args.pin_memory,
                "persistent_workers": self.args.persistent_workers,
                "prefetch_factor": self.args.prefetch_factor if self.args.num_workers > 0 else None,
            }

        dataset = ERA5Dataset(
            args=self.args,
            root_dir=os.path.join(self.root_dir, "train"),
            variables=self.variables,
            inp_transform=self.transforms,
            out_transform_dict=self.out_transforms,
            list_intervals=self.intervals,
            data_freq=self.data_freq,
            year_list=None, # use all data
        )

        batch_size = self.batch_size
        collate_fn = collate_fn_train

        if self.args.disable_collation:
            if batch_size > 1:
                log0(f"Cannot disable collation since batch_size is {batch_size}", mode="warning")

            if batch_size == 1:
                batch_size = None
                collate_fn = None
                log0("Disabling collation")

        use_live = os.environ.get("DFOPTIMIZER_ENABLE", "0") == "1"

        if use_live:
            from live_sampler import LiveSampler
            from live_data_loader import LiveDataLoader

            sampler = LiveSampler(
                dataset,
                num_replicas=MPIUtils.size(),
                rank=MPIUtils.rank(),
                shuffle=True,
                seed=self.args.seed,
            )
            loader = LiveDataLoader(
                dataset,
                sampler,
                batch_size=batch_size,
                num_workers=loader_config["num_workers"],
                prefetch_factor=loader_config.get("prefetch_factor") or 2,
                pin_memory=loader_config["pin_memory"],
                persistent_workers=loader_config["persistent_workers"],
                collate_fn=collate_fn,
            )
            return loader, sampler

        if MPIUtils.size() >= 1:
            sampler = DistributedSampler(
                dataset=dataset,
                rank=MPIUtils.rank(),
                num_replicas=MPIUtils.size(),
                seed=self.args.seed,
                shuffle=True,
                drop_last=False,
            )
        else:
            sampler = None

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                sampler=sampler,
                num_workers=loader_config["num_workers"],
                pin_memory=loader_config["pin_memory"],
                persistent_workers=loader_config["persistent_workers"],
                prefetch_factor=loader_config["prefetch_factor"],
                collate_fn=collate_fn,
            ),
            sampler,
        )


class Trainer:
    def __init__(
        self,
        model: Stormer,
        datamodule: DataModule,
        args,
    ):
        self.args = args
        self.model = model
        model_id: str = self.model.__class__.__qualname__
        self.model_id = model_id.lower()

        self.datamodule = datamodule
        self.device = torch.device(f"{DEVICE_TYPE}:{DEVICE_ID}")

        self.optimizer = configure_optimizers(
            params=model.named_parameters(),
            lr=self.args.lr,
            beta_1=self.args.beta_1,
            beta_2=self.args.beta_2,
            weight_decay=self.args.weight_decay,
        )
        self.precision = self.args.precision
        self.scheduler = None

        self.epochs = self.args.epochs
        self.max_training_step = self.args.max_training_step
        self.weighted_loss = self.args.weighted_loss
        self.enable_progress_bar = self.args.enable_progress_bar
        self.variables = VARIABLES
        self.accumulate_grad_batches = self.args.accumulate_grad_batches

        self.interval_combinations = None
        self.set_lat_lon(*self.datamodule.get_lat_lon())
        self.set_transforms(*self.datamodule.get_transforms())
        self.list_train_intervals = self.datamodule.intervals

        self.sync_batchnorm = self.args.sync_batchnorm

        self.dtype = get_precision_dtype(precision=self.args.precision)

        self.scaler = None
        self.setup_model()

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_transforms(self, inp_transform, diff_transform):
        self.inp_transform = inp_transform
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)

        self.diff_transform = diff_transform
        self.reverse_diff_transform = {
            k: self.get_reverse_transform(v) for k, v in diff_transform.items()
        }

    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)

    def setup_model(self):
        # config = Runtime.config()

        self.net = self.model
        if self.sync_batchnorm and MPIUtils.size() > 1:
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)

        if DEVICE_TYPE == "cuda":
            # torch.cuda.set_device(MPI_LOCAL_RANK)
            torch.cuda.set_device(DEVICE_ID)

        # log0(f"Moving model to device {self.device}")
        # self.net.to(MPI_LOCAL_RANK)
        self.net.to(device=self.device)

        self._ddp_mixed_precision = None
        if MPIUtils.size() > 1:
            _bucket_cap = float(os.environ.get("STORMER_BUCKET_CAP_MB", "25"))
            reduce_dtype = get_ddp_reduce_dtype(os.environ.get("STORMER_DDP_REDUCE_DTYPE"))
            if reduce_dtype is not None:
                if _MixedPrecision is None:
                    raise RuntimeError("DDP mixed precision is unavailable in this torch build")
                self._ddp_mixed_precision = _MixedPrecision(reduce_dtype=reduce_dtype)
            if MPIUtils.rank() == 0:
                if self._ddp_mixed_precision is not None:
                    log.info(
                        "Setting up DDP (bucket_cap_mb=%.0f, reduce_dtype=%s)",
                        _bucket_cap,
                        str(reduce_dtype).replace("torch.", ""),
                    )
                else:
                    log.info("Setting up DDP (bucket_cap_mb=%.0f)", _bucket_cap)
            self.net = DDP(
                self.net,
                device_ids=[DEVICE_ID],
                # device_ids=[MPI_LOCAL_RANK],
                output_device=DEVICE_ID,
                bucket_cap_mb=_bucket_cap,
                mixed_precision=self._ddp_mixed_precision,
            )
        elif MPIUtils.rank() == 0:
            log.info("Running without DDP (single-process mode)")

        # --- Communication timing and optimization ---
        self._comm_timing = os.environ.get("STORMER_COMM_TIMING", "0") == "1"
        self._comm_optimize = os.environ.get("STORMER_COMM_OPTIMIZE", "0") == "1"
        self._comm_bucket_ms = []  # filled by hook, cleared per step
        self._fwd_ms = 0.0
        self._bwd_ms = 0.0
        self._bucket_cap_mb = _bucket_cap if MPIUtils.size() > 1 else 25.0
        self._comm_window_steps = []  # per-step (step_ms, comm_ms, n_buckets) for current window
        self._comm_cooldown = 0  # windows to skip after a change
        if self._ddp_mixed_precision is not None and (self._comm_timing or self._comm_optimize):
            if MPIUtils.rank() == 0:
                log.warning(
                    "COMM_TIMING/COMM_OPTIMIZE disabled because DDP mixed_precision registers its own comm hook"
                )
            self._comm_timing = False
            self._comm_optimize = False
        # Hook is registered whenever any of COMM_TIMING / COMM_OPTIMIZE / DFTracer is
        # active. DFTracer activation emits per-bucket comm.all_reduce events so the
        # analyzer can compute comm_frac alongside fetch_frac / compute_frac.
        _comm_hook_enabled = (
            self._comm_timing or self._comm_optimize or PERFTRACER_ENABLE
        )
        if _comm_hook_enabled and MPIUtils.size() > 1:
            def _comm_hook(state, bucket):
                with dft_ai.comm.all_reduce:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    work = dist.all_reduce(bucket.buffer(), async_op=True)
                    work.wait()
                    torch.cuda.synchronize()
                    state.append((time.time() - t0) * 1000)
                # Return C++ Future via Work.get_future() (DDP requirement)
                return work.get_future().then(lambda fut: fut.value()[0])
            self.net.register_comm_hook(self._comm_bucket_ms, _comm_hook)
            if MPIUtils.rank() == 0:
                _reason_bits = []
                if self._comm_timing:
                    _reason_bits.append("timing")
                if self._comm_optimize:
                    _reason_bits.append("optimize")
                if PERFTRACER_ENABLE:
                    _reason_bits.append("dftracer")
                log.info(
                    "COMM_HOOK: registered (%s, all-reduce serialized)",
                    "+".join(_reason_bits) or "default",
                )

        if MPIUtils.rank() == 0:
            log.info("Setting up Scaler")

        self.scaler = torch.GradScaler(  # only for float16 stablility
            self.device.type,
            enabled=(self.dtype == torch.float16 and torch.cuda.is_available()),
        )

    def make_loader(self, epoch: int):
        # With LiveDataLoader, reuse the same loader across epochs
        # (set_epoch resets the sampler; the loader object persists for mid-epoch reconfigure)
        if not hasattr(self, "_live_loader"):
            self._live_loader, self._live_sampler = self.datamodule.dataloader()

        from live_data_loader import LiveDataLoader
        if isinstance(self._live_loader, LiveDataLoader):
            self._live_loader.set_epoch(epoch)
            log0(
                "make_loader: epoch=%s num_workers=%s prefetch_factor=%s (LiveDataLoader)",
                epoch, self._live_loader.num_workers, self._live_loader.prefetch_factor,
            )
            return self._live_loader, self._live_sampler

        # Standard DataLoader: recreate each epoch
        train_loader, train_sampler = self.datamodule.dataloader()
        log0(
            "make_loader: epoch=%s num_workers=%s prefetch_factor=%s pin_memory=%s persistent_workers=%s",
            epoch,
            self.args.num_workers,
            self.args.prefetch_factor,
            self.args.pin_memory,
            self.args.persistent_workers,
        )
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        return train_loader, train_sampler

    def configure_scheduler(self, n_steps: int):
        return configure_lr_scheduler(
            self.optimizer,
            max_epochs=self.epochs,
            n_steps=n_steps,
            warmup_epochs=10,
            warmup_start_lr=1e-8,
            eta_min=1e-8,
            num_nodes=MPIUtils.num_nodes(),
            num_devices=self.args.ngpus_per_node,
        )

    def forward_train(self, x: torch.Tensor, variables, interval, *args, **kwargs):
        if "stormer" in self.model_id:
            padded_x, pad_size = pad(x=x, patch_size=self.model.patch_size)
            norm_pred_diff = self.net(padded_x, variables, interval)[
                :, :, pad_size:
            ]  # diff in the normalized space
        else:
            norm_pred_diff = self.net(x)

        return replace_constant(norm_pred_diff, variables)

    @dft_ai.device.transfer
    def send_to_device(self, batch):
        x, gt_diff, interval_tensors = batch
        x = x.to(device=self.device)  # , non_blocking=True)
        gt_diff = gt_diff.to(device=self.device)  # , non_blocking=True)
        interval_tensors = interval_tensors.to(device=self.device)  # , non_blocking=True)
        return x, gt_diff, interval_tensors

    def autocast_context_manager(self):
        return setup_precision_context(
            dtype=self.dtype,
            device_type=DEVICE_TYPE,
            mixed=True,
            enabled=True,
            # enabled=Distributed.strategy() != DistributedStrategy.DEEPSPEED,
        )

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

    @dft_ai.compute.forward
    def forward_step(self, batch: Any):
        x, gt_diff, interval = batch

        with self.forward_context():
            pred_diff = self.forward_train(x, self.variables, interval)

            loss_dict = lat_weighted_mse(
                pred_diff,
                gt_diff,
                self.variables,
                self.lat,
                weighted=self.weighted_loss,
                weight_dict=WEIGHT_DICT,
            )

        loss = loss_dict["w_mse_aggregate"]
        return loss, loss.detach()

    @dft_ai.compute.backward
    def backward_step(self, loss, batch_idx: int, n_accum_steps: int):
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Only step optimizer at end of accumulation window
        if batch_idx % n_accum_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

    @dft_ai.compute
    def training_step(self, batch: Any, batch_idx: int):
        n_accum_steps = self.accumulate_grad_batches

        # Only zero gradients at start of accumulation window
        if ((batch_idx - 1) % n_accum_steps) == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Apply same sync context to forward + backward
        if batch_idx % n_accum_steps != 0 and hasattr(self.net, "no_sync"):
            sync_context = self.net.no_sync()
        else:
            sync_context = contextlib.nullcontext()

        with sync_context:
            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                _t0 = time.time()

            loss, loss_item = self.forward_step(batch)

            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                self._fwd_ms = (time.time() - _t0) * 1000

            # Skip NaN losses to prevent gradient divergence at scale
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad(set_to_none=True)
                return float("nan")

            # Normalize loss for gradient accumulation (keeps effective LR constant)
            if n_accum_steps > 1:
                loss = loss / n_accum_steps

            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                _t1 = time.time()

            self.backward_step(loss, batch_idx, n_accum_steps)

            if self._comm_timing or self._comm_optimize:
                torch.cuda.synchronize()
                self._bwd_ms = (time.time() - _t1) * 1000

        return loss_item.cpu().item()

    def _setup_optimizer_context(self):
        """Initialize dfoptimizer context for knob registration and plan reception."""
        self._optimizer_ctx = None
        if os.environ.get("DFOPTIMIZER_ENABLE", "0") != "1":
            return
        try:
            from dfoptimizer.runtime import optimizer_context
            group_file = os.environ.get("DFTRACER_MOFKA_GROUP_FILE", "")
            self._optimizer_ctx = optimizer_context(
                namespace="stormer",
                group_file=group_file,
                topic_plans="optimizer_plans",
                topic_acks="optimizer_acks",
                topic_registry="optimizer_registry",
            )
            log0("Optimizer context started (group_file=%s)", group_file)
        except Exception as e:
            log0("WARNING: optimizer context setup failed: %s", e)

    def _poll_and_apply_plans(self, train_loader):
        """Drain pending optimizer actions and reconfigure LiveDataLoader."""
        ctx = self._optimizer_ctx
        if ctx is None or ctx._noop:
            return
        from live_data_loader import LiveDataLoader
        if not isinstance(train_loader, LiveDataLoader):
            return
        actions = ctx.drain_actions_for("stormer.make_loader", at="window_boundary")
        if not actions:
            return
        new_w = None
        new_pf = None
        for action in actions:
            knob_name = action.knob_id
            if knob_name.startswith("stormer."):
                knob_name = knob_name[len("stormer."):]
            if knob_name == "num_workers":
                new_w = int(action.new_value)
                ctx.ack_action(action, train_loader.num_workers, new_w)
            elif knob_name == "prefetch_factor":
                new_pf = int(action.new_value)
                ctx.ack_action(action, train_loader.prefetch_factor, new_pf)
            else:
                ctx.ack_action(action, None, None, status="rejected")
        train_loader.reconfigure(num_workers=new_w, prefetch_factor=new_pf)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Run validation on held-out data (rank 0 only to avoid file contention)."""
        if MPIUtils.rank() != 0:
            # Non-rank-0 processes wait at barrier
            if MPIUtils.size() > 1:
                dist.barrier()
            return
        if not hasattr(self, '_val_loader'):
            try:
                self._val_loader = self.datamodule.val_dataloader(max_samples=30)
                log.info("Val loader created: %s (%d samples)",
                         "OK" if self._val_loader else "None",
                         len(self._val_loader.dataset) if self._val_loader else 0)
            except Exception as e:
                log.info("WARNING: val_dataloader failed: %s", e)
                self._val_loader = None
        if self._val_loader is None:
            if MPIUtils.size() > 1:
                dist.barrier()
            return
        # Use the unwrapped model for single-rank eval
        model = self.net.module if isinstance(self.net, DDP) else self.net
        model.eval()
        val_losses = []
        try:
            for batch in self._val_loader:
                batch = self.send_to_device(batch)
                x, gt_diff, interval = batch
                with self.autocast_context_manager():
                    pred_diff = self.forward_train(x, self.variables, interval)
                    loss_dict = lat_weighted_mse(
                        pred_diff, gt_diff, self.variables, self.lat,
                        weighted=self.weighted_loss, weight_dict=WEIGHT_DICT,
                    )
                val_losses.append(loss_dict["w_mse_aggregate"].item())
        except Exception as e:
            log.info("WARNING: validation step failed: %s", e)
        model.train()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("nan")
        log.info("VAL_LOSS epoch=%d val_loss=%.4f (n=%d samples)", epoch, avg_val_loss, len(val_losses))
        if MPIUtils.size() > 1:
            dist.barrier()
        return avg_val_loss

    def _rebuild_ddp(self, new_bucket_cap_mb):
        """Rebuild DDP with a new bucket_cap_mb. Must be called outside backward pass."""
        if not isinstance(self.net, DDP):
            return
        module = self.net.module
        old_cap = self._bucket_cap_mb
        self._bucket_cap_mb = new_bucket_cap_mb
        self.net = DDP(
            module,
            device_ids=[DEVICE_ID],
            output_device=DEVICE_ID,
            bucket_cap_mb=new_bucket_cap_mb,
            mixed_precision=self._ddp_mixed_precision,
        )
        # Re-register comm hook on new DDP wrapper
        if (self._comm_timing or self._comm_optimize) and MPIUtils.size() > 1:
            def _comm_hook(state, bucket):
                torch.cuda.synchronize()
                t0 = time.time()
                work = dist.all_reduce(bucket.buffer(), async_op=True)
                work.wait()
                torch.cuda.synchronize()
                state.append((time.time() - t0) * 1000)
                return work.get_future().then(lambda fut: fut.value()[0])
            self.net.register_comm_hook(self._comm_bucket_ms, _comm_hook)
        log0("COMM_OPT: rebuilt DDP bucket_cap_mb=%.0f→%.0f", old_cap, new_bucket_cap_mb)

    def _comm_optimize_window(self):
        """Evaluate comm metrics for the completed window and tune knobs."""
        if not self._comm_optimize or not self._comm_window_steps:
            return
        if self._comm_cooldown > 0:
            self._comm_cooldown -= 1
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: cooldown=%d, skipping", self._comm_cooldown)
            return

        # Skip first window (MIOpen warmup distorts metrics)
        if not hasattr(self, '_comm_window_count'):
            self._comm_window_count = 0
        self._comm_window_count += 1
        if self._comm_window_count == 1:
            self._comm_window_steps.clear()
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: skipping first window (warmup)")
            return

        n = len(self._comm_window_steps)
        avg_step = sum(r[0] for r in self._comm_window_steps) / n
        avg_comm = sum(r[1] for r in self._comm_window_steps) / n
        avg_buckets = sum(r[2] for r in self._comm_window_steps) / n
        losses = [r[3] for r in self._comm_window_steps if not (r[3] != r[3])]  # filter NaN
        avg_loss = sum(losses) / len(losses) if losses else float("nan")
        comm_frac = avg_comm / avg_step if avg_step > 0 else 0
        self._comm_window_steps.clear()

        if MPIUtils.rank() == 0:
            log.info(
                "COMM_OPT: window avg_step=%.0fms avg_comm=%.0fms comm_frac=%.2f "
                "buckets=%.0f bucket_cap=%.0f accum=%d avg_loss=%.4f",
                avg_step, avg_comm, comm_frac, avg_buckets,
                self._bucket_cap_mb, self.accumulate_grad_batches, avg_loss,
            )

        if comm_frac < 0.70:
            if MPIUtils.rank() == 0:
                log.info("COMM_OPT: comm_frac=%.2f < 0.70, no action needed", comm_frac)
            return

        # Stage 1: Increase bucket_cap_mb (pure systems, no semantics impact)
        if avg_buckets > 15 and self._bucket_cap_mb < 300:
            new_cap = min(self._bucket_cap_mb * 4, 300)
            self._rebuild_ddp(new_cap)
            self._comm_cooldown = 1  # wait 1 window to observe effect
            return

        # Stage 2: Increase accumulation (higher leverage, changes sync cadence)
        if self.accumulate_grad_batches < 4:
            old_accum = self.accumulate_grad_batches
            self.accumulate_grad_batches = min(old_accum + 1, 4)
            # Clear stale gradients from previous accum window to prevent
            # the next no-zero-grad step from accumulating on old gradients
            self.optimizer.zero_grad(set_to_none=True)
            self._comm_cooldown = 1
            log0("COMM_OPT: accumulate_grad_batches=%d→%d", old_accum, self.accumulate_grad_batches)
            return

        if MPIUtils.rank() == 0:
            log.info("COMM_OPT: knobs exhausted (bucket_cap=%.0f, accum=%d)",
                     self._bucket_cap_mb, self.accumulate_grad_batches)

    @dft_ai.pipeline.train
    def train(self):
        self._setup_optimizer_context()

        # Register knobs with optimizer (if context is active)
        if self._optimizer_ctx and not self._optimizer_ctx._noop:
            try:
                from dfoptimizer.runtime import knob
                from dfoptimizer.runtime.knob import knob_def_from_dict
                _knob_max_workers = int(os.environ.get("STORMER_KNOB_MAX_WORKERS", "8"))
                knob_defs = {
                    "num_workers": knob(
                        default=self.args.num_workers,
                        range=(0, _knob_max_workers),
                        type=int,
                        responds_to={
                            "reader_parallelism": {
                                "direction": "increase",
                                "step_mode": "evidence",
                                "min_persistence": 2,
                                "cooldown_windows": 3,
                                "apply_when": "window_boundary",
                            },
                        },
                    ),
                    "prefetch_factor": knob(
                        default=self.args.prefetch_factor,
                        range=(1, 16),
                        type=int,
                        responds_to={
                            "dataloader_prefetch": {
                                "direction": "increase",
                                "step_mode": "evidence",
                                "min_persistence": 2,
                                "cooldown_windows": 3,
                                "apply_when": "window_boundary",
                            },
                        },
                    ),
                }
                built_defs = {
                    k: knob_def_from_dict(f"stormer.{k}", v, target_function="stormer.make_loader")
                    for k, v in knob_defs.items()
                }
                self._optimizer_ctx.register_knobs(
                    "stormer.make_loader",
                    built_defs,
                    current_values={
                        "num_workers": self.args.num_workers,
                        "prefetch_factor": self.args.prefetch_factor,
                    },
                )
                log0("Registered knobs: num_workers=%s, prefetch_factor=%s",
                     self.args.num_workers, self.args.prefetch_factor)
            except Exception as e:
                log0("WARNING: knob registration failed: %s", e)

        train_loader, train_sampler = self.make_loader(epoch=0)
        self.n_steps = len(train_sampler)
        self.scheduler = self.configure_scheduler(n_steps=self.n_steps)
        log0("Will run %s steps", self.n_steps)

        # Window manages analysis boundaries for the streaming pipeline.
        # Emits window.start/window.stop control events at the configured
        # cadence so the analyzer knows when to drain and analyze.
        _window = None
        _n_steps_eff = self.max_training_step if self.max_training_step > 0 else self.n_steps
        _window_every_n = max(_n_steps_eff // 5, 1)
        try:
            from dfoptimizer.runtime import Window
            _window = Window(PerfTracer.get_instance(), max_every_n=self.n_steps)
            _window.update_cadence(_window_every_n)
            log0("Window initialized (every_n=%s)", _window.every_n)
        except Exception as e:
            log0("WARNING: Window init failed (no streaming analysis): %s", e)

        for epoch in dft_ai.pipeline.epoch.iter(
            range(self.epochs),
            include_iter=False,
        ):
            dft_ai.pipeline.epoch.start(metadata=True)
            pbar = None
            try:
                log0(f"Epoch: {epoch}")
                train_loader, train_sampler = self.make_loader(epoch=epoch)

                self.net.train()

                if self.enable_progress_bar:
                    import tqdm.auto as tqdm

                    pbar = tqdm.tqdm(
                        desc=f"epoch {epoch}",
                        total=len(train_loader),
                        position=MPIUtils.rank(),
                        leave=True,
                        disable=MPIUtils.rank() != 0,
                        bar_format="{l_bar}{bar}|{n}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                    )

                step = 1
                dft_ai.update(epoch=epoch, step=step)
                collate_profiler.update(epoch=epoch, step=step)

                if self.enable_progress_bar:
                    from tqdm.contrib.logging import logging_redirect_tqdm
                    logging_ctx = logging_redirect_tqdm
                else:
                    from utils.context import noop_context
                    logging_ctx = noop_context

                # Comm timing accumulators for epoch summary
                _ct_steps = []

                with logging_ctx():
                    _t_fetch0 = time.time()
                    for batch in dft_ai.dataloader.fetch.iter(train_loader):
                        if self._comm_timing or self._comm_optimize:
                            _fetch_ms = (time.time() - _t_fetch0) * 1000

                        if self.max_training_step != -1 and step > self.max_training_step:
                            log0(f"Training step limit reached: {self.max_training_step}")
                            break

                        if _window is not None:
                            _window.start()

                        if self._comm_timing or self._comm_optimize:
                            torch.cuda.synchronize()
                            _t_step0 = time.time()

                        batch = self.send_to_device(batch)
                        loss = self.training_step(batch, step)

                        if self._comm_timing or self._comm_optimize:
                            torch.cuda.synchronize()
                            step_ms = (time.time() - _t_step0) * 1000
                            comm_ms = sum(self._comm_bucket_ms)
                            n_buckets = len(self._comm_bucket_ms)
                            self._comm_bucket_ms.clear()
                            fwd_ms = self._fwd_ms
                            bwd_ms = self._bwd_ms
                            other_ms = step_ms - fwd_ms - bwd_ms
                            total_ms = _fetch_ms + step_ms
                            _loss_for_ct = loss if isinstance(loss, float) else float("nan")
                            _ct_steps.append((step_ms, fwd_ms, bwd_ms, comm_ms, n_buckets, other_ms, _loss_for_ct, _fetch_ms))
                            # Feed per-step data to comm optimizer (include loss for convergence tracking)
                            _loss_val = loss if isinstance(loss, float) else float("nan")
                            self._comm_window_steps.append((step_ms, comm_ms, n_buckets, _loss_val))
                            if MPIUtils.rank() == 0 and self._comm_timing:
                                log.info(
                                    "COMM_TIMING e=%d s=%d total=%.0f fetch=%.0f step=%.0f fwd=%.0f bwd=%.0f comm=%.0f(%d buckets) other=%.0f comm_pct=%.1f%% fetch_pct=%.1f%%",
                                    epoch, step, total_ms, _fetch_ms, step_ms, fwd_ms, bwd_ms, comm_ms, n_buckets,
                                    other_ms,
                                    100 * comm_ms / total_ms if total_ms > 0 else 0,
                                    100 * _fetch_ms / total_ms if total_ms > 0 else 0,
                                )
                            _t_fetch0 = time.time()

                        if _window is not None:
                            _window.stop()

                        if pbar:
                            pbar.set_postfix({"loss": loss})
                            pbar.update()

                        step += 1
                        dft_ai.update(epoch=epoch, step=step)
                        collate_profiler.update(epoch=epoch, step=step)

                        # Poll optimizer at window boundaries for mid-epoch reconfiguration
                        if step % _window_every_n == 0:
                            self._poll_and_apply_plans(train_loader)
                            self._comm_optimize_window()
            finally:
                # Epoch-level comm timing summary
                if (self._comm_timing or self._comm_optimize) and _ct_steps and MPIUtils.rank() == 0:
                    n = len(_ct_steps)
                    avg_step = sum(r[0] for r in _ct_steps) / n
                    avg_fwd = sum(r[1] for r in _ct_steps) / n
                    avg_bwd = sum(r[2] for r in _ct_steps) / n
                    avg_comm = sum(r[3] for r in _ct_steps) / n
                    avg_other = sum(r[5] for r in _ct_steps) / n
                    avg_fetch = sum(r[7] for r in _ct_steps if len(r) > 7) / n
                    avg_total = avg_fetch + avg_step
                    epoch_losses = [r[6] for r in _ct_steps if len(r) > 6 and not (r[6] != r[6])]
                    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
                    log.info(
                        "COMM_TIMING EPOCH %d SUMMARY (%d steps): "
                        "avg_total=%.0fms avg_fetch=%.0fms avg_step=%.0fms avg_fwd=%.0fms avg_bwd=%.0fms "
                        "avg_comm=%.0fms avg_other=%.0fms comm_pct=%.1f%% fetch_pct=%.1f%% "
                        "avg_loss=%.4f",
                        epoch, n, avg_total, avg_fetch, avg_step, avg_fwd, avg_bwd, avg_comm, avg_other,
                        100 * avg_comm / avg_total if avg_total > 0 else 0,
                        100 * avg_fetch / avg_total if avg_total > 0 else 0,
                        avg_loss,
                    )

                # Validation at end of epoch
                self.validate_epoch(epoch)

                if _window is not None:
                    _window.flush()
                if pbar:
                    pbar.close()
                dft_ai.pipeline.epoch.stop(metadata=True)

    def finalize(self):
        return None

def get_args():
    # Benchmark settings
    parser = argparse.ArgumentParser(
        description="Stormer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-folder", default="outputs", type=str)
    parser.add_argument("--log-folder", default=None, type=str)
    parser.add_argument("--ngpus-per-node", type=int, help="NGPUs per node (required)")
    # dataset
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--data-freq", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--intervals", type=int, nargs="+", required=True, help="List of intervals (required)")
    # dataloader
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--disable-collation", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    # optimizer
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta-1", type=float, default=0.9)
    parser.add_argument("--beta-2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    # scheduler
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of warmup epochs (default: 10)")
    parser.add_argument("--warmup-start-lr", type=float, default=1e-8, help="Starting learning rate for warmup (default: 1e-8)")
    parser.add_argument("--eta-min", type=float, default=1e-8, help="Minimum learning rate (default: 1e-8)")
    # model
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-img-size", type=int, nargs='+', required=True, help="Input image size as a list or tuple of integers")
    # parser.add_argument("--variables", type=str, nargs='+', required=True, help="List of variable names as strings")
    parser.add_argument("--patch-size", type=int, default=2, help="Patch size (default: 2)")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size (default: 1024)")
    parser.add_argument("--depth", type=int, default=24, help="Depth (default: 24)")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of heads (default: 16)")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP ratio (default: 4.0)")
    # trainer
    parser.add_argument("--precision", type=str, default="16", help="Precision (str, int, or None). Default: 16")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--max-training-step", type=int, default=-1, help="Maximum training steps (default: -1 for unlimited)")
    parser.add_argument("--weighted-loss", action="store_true", default=True, help="Use weighted loss (default: True)")
    parser.add_argument("--enable-progress-bar", action="store_true", default=True, help="Enable progress bar (default: True)")
    parser.add_argument("--sync-batchnorm", action="store_true", default=False, help="Synchronize batch normalization across ranks")
    parser.add_argument("--no-sync-batchnorm", dest="sync_batchnorm", action="store_false")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Accumulate gradient batches (default: 1)")

    args = parser.parse_args()
    return args

def main() -> None:
    MPIUtils.initialize()
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    log_folder = args.log_folder or args.output_folder
    os.makedirs(log_folder, exist_ok=True)
    configure_logging(output_dir=log_folder)

    dftracer_logger = None  # set by pfwlogger below

    if DEVICE_TYPE != "cuda":
        raise RuntimeError("Stormer training requires a CUDA/ROCm GPU environment.")

    torch.cuda.set_device(DEVICE_ID)

    hostname, port = get_master_addr_and_port(port=23456, set_env=True)

    if MPIUtils.rank() == 0:
        log.info("Master addresses = %s:%d", hostname, port)

    dist_initialized = False
    if MPIUtils.size() > 1:
        from datetime import timedelta
        _nccl_timeout = int(os.environ.get("STORMER_NCCL_TIMEOUT", "1800"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=_nccl_timeout),
            world_size=MPIUtils.size(),
            rank=MPIUtils.rank(),
            device_id=DEVICE_ID,
        )
        dist.barrier()
        dist_initialized = True

    # ngpus_per_node = torch.cuda.device_count()
    # log.info("NGPUS %s", ngpus_per_node)
    # for i in range(torch.cuda.device_count()):
    #     print(f"Rank {MPIUtils.rank()} -> Device {i}: {torch.cuda.get_device_name(i)}")
    #     properties = torch.cuda.get_device_properties(i)
    #     print(f"  Total Memory: {properties.total_memory / (1024**3):.2f} GB")
    #     print(f"  Compute Capability: {properties.major}.{properties.minor}")
    # return

    if MPIUtils.rank() == 0:
        log.info("Arguments")
        for key, value in vars(args).items():
            log.info(f" {key}: {value}")

    if MPIUtils.rank() == 0:
        log.info("DFTracer init: DFTRACER_ENABLE=%s WRITER_TYPE=%s GROUP_FILE=%s data_dir=%s",
                 os.environ.get("DFTRACER_ENABLE"), os.environ.get("DFTRACER_WRITER_TYPE"),
                 os.environ.get("DFTRACER_MOFKA_GROUP_FILE"), args.data_folder)
    pfwlogger = PerfTracer.initialize_log(
        logfile=f"{log_folder}/trace-{MPIUtils.rank() + 1}-of-{MPIUtils.size()}.pfw",
        data_dir=args.data_folder,
        process_id=MPIUtils.rank(),
    )
    dftracer_logger = pfwlogger

    with dft_ai:
        model = Stormer(
            in_img_size=args.in_img_size,
            variables=VARIABLES,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
        )

        datamodule = DataModule(
            args=args,
            root_dir=args.data_folder,
            variables=VARIABLES,
            intervals=args.intervals,
        )

        trainer = Trainer(
            args=args,
            model=model,
            datamodule=datamodule
        )

        trainer.train()
        trainer.finalize()

    pfwlogger.finalize()
    if dftracer_logger is not None:
        dftracer_logger.finalize()
    if dist_initialized:
        dist.barrier()
        dist.destroy_process_group()
    MPIUtils.finalize()


if __name__ == "__main__":
    main()
