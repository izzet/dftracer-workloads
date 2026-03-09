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
from utils.perf_tracer import PerfTracer, dft_ai
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
            d = f["input"][var][:].reshape(1, *self.image_shape)
            x.append(d)
        f.close()
        del f

        return torch.from_numpy(np.concatenate(x))
    
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

    def dataloader(self, loader_config: dict[str, Any] | None = None):
        if loader_config is None:
            loader_config = {
                "num_workers": self.args.num_workers,
                "pin_memory": self.args.pin_memory,
                "persistent_workers": self.args.persistent_workers,
                "prefetch_factor": self.args.prefetch_factor,
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

        batch_size = self.batch_size
        collate_fn = collate_fn_train

        if self.args.disable_collation:
            if batch_size > 1:
                log0(f"Cannot disable collation since batch_size is {batch_size}", mode="warning")

            if batch_size == 1:
                batch_size = None
                collate_fn = None
                log0("Disabling collation")

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                # worker_init_fn=dataset.worker_init,
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

        if MPIUtils.size() > 1:
            if MPIUtils.rank() == 0:
                log.info("Setting up DDP")
            self.net = DDP(
                self.net,
                device_ids=[DEVICE_ID],
                # device_ids=[MPI_LOCAL_RANK],
                output_device=DEVICE_ID,
            )
        elif MPIUtils.rank() == 0:
            log.info("Running without DDP (single-process mode)")

        if MPIUtils.rank() == 0:
            log.info("Setting up Scaler")

        self.scaler = torch.GradScaler(  # only for float16 stablility
            self.device.type,
            enabled=(self.dtype == torch.float16 and torch.cuda.is_available()),
        )

    def make_loader(self, epoch: int):
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
            loss, loss_item = self.forward_step(batch)
            self.backward_step(loss, batch_idx, n_accum_steps)

        return loss_item.cpu().item()

    @dft_ai.pipeline.train
    def train(self):
        train_loader, train_sampler = self.make_loader(epoch=0)
        self.n_steps = len(train_sampler)
        self.scheduler = self.configure_scheduler(n_steps=self.n_steps)
        log0("Will run %s steps", self.n_steps)

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

                with logging_ctx():
                    for batch in dft_ai.dataloader.fetch.iter(train_loader):
                        if self.max_training_step != -1 and step > self.max_training_step:
                            log0(f"Training step limit reached: {self.max_training_step}")
                            break

                        batch = self.send_to_device(batch)
                        loss = self.training_step(batch, step)
                        # loss = self.simulate_training_step(batch, step)

                        if pbar:
                            pbar.set_postfix({"loss": loss})
                            pbar.update()

                        step += 1

                        # if step % 1000 == 0:
                        #     log0("Running steps %s", step)

                        dft_ai.update(epoch=epoch, step=step)
                        collate_profiler.update(epoch=epoch, step=step)
            finally:
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
    parser.add_argument("--sync-batchnorm", action="store_true", default=True, help="Synchronize batch normalization (default: True)")
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

    if DEVICE_TYPE != "cuda":
        raise RuntimeError("Stormer training requires a CUDA/ROCm GPU environment.")

    torch.cuda.set_device(DEVICE_ID)

    hostname, port = get_master_addr_and_port(port=23456, set_env=True)

    if MPIUtils.rank() == 0:
        log.info("Master addresses = %s:%d", hostname, port)

    dist_initialized = False
    if MPIUtils.size() > 1:
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            # init_method=f"tcp://{hostname}:{port}",
            # timeout=3600,
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

    pfwlogger = PerfTracer.initialize_log(
        logfile=f"{log_folder}/trace-{MPIUtils.rank() + 1}-of-{MPIUtils.size()}.pfw",
        data_dir=args.data_folder,
        process_id=MPIUtils.rank(),
    )

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
    if dist_initialized:
        dist.barrier()
        dist.destroy_process_group()
    MPIUtils.finalize()


if __name__ == "__main__":
    main()
