"""LiveDataLoader for StorMer — supports mid-epoch num_workers reconfiguration.

Wraps a standard PyTorch DataLoader + LiveSampler.  When reconfigure() is
called (e.g. by the optimizer), the current DataLoader is torn down and a new
one is created with updated num_workers / prefetch_factor, resuming iteration
from the exact sample position.

Usage in the training loop:

    loader = LiveDataLoader(dataset, sampler, ...)
    loader.set_epoch(epoch)

    for batch in loader:
        train_step(batch)
        if step % poll_interval == 0:
            plan = poll_plans()
            if plan:
                loader.reconfigure(num_workers=plan["num_workers"], ...)
"""

import logging
from torch.utils.data import DataLoader, Dataset

from live_sampler import LiveSampler

log = logging.getLogger(__name__)


class LiveDataLoader:

    def __init__(
        self,
        dataset: Dataset,
        sampler: LiveSampler,
        *,
        batch_size: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        collate_fn=None,
    ):
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._collate_fn = collate_fn

        self._inner: DataLoader | None = None
        self._inner_iter = None
        self._samples_yielded: int = 0
        self._pending_reconfigure: bool = False

    def set_epoch(self, epoch: int):
        self._sampler.set_epoch(epoch)
        self._samples_yielded = 0
        self._pending_reconfigure = False
        self._inner = self._build_inner()
        self._inner_iter = None

    def _build_inner(self) -> DataLoader:
        self._sampler.advance_to(self._samples_yielded)
        kwargs = {}
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = max(self.prefetch_factor, 1)
            if self._persistent_workers:
                kwargs["persistent_workers"] = True
        return DataLoader(
            self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self.num_workers,
            pin_memory=self._pin_memory,
            collate_fn=self._collate_fn,
            drop_last=False,
            **kwargs,
        )

    def reconfigure(
        self,
        num_workers: int | None = None,
        prefetch_factor: int | None = None,
    ):
        changed = False
        if num_workers is not None and num_workers != self.num_workers:
            self.num_workers = num_workers
            changed = True
        if prefetch_factor is not None and prefetch_factor != self.prefetch_factor:
            self.prefetch_factor = prefetch_factor
            changed = True
        if changed:
            self._pending_reconfigure = True

    def _apply_reconfigure(self):
        old_w = self._inner.num_workers if self._inner else "?"
        # Tear down old DataLoader (kills worker processes)
        del self._inner
        self._inner = self._build_inner()
        self._inner_iter = iter(self._inner)
        self._pending_reconfigure = False
        log.info(
            f"LiveDataLoader: reconfigured workers {old_w} -> {self.num_workers}, "
            f"prefetch={self.prefetch_factor}, "
            f"resuming from sample {self._samples_yielded}"
        )

    def __len__(self) -> int:
        return len(self._sampler)

    def __iter__(self):
        self._inner_iter = iter(self._inner)
        return self

    def __next__(self):
        if self._pending_reconfigure:
            self._apply_reconfigure()

        try:
            batch = next(self._inner_iter)
        except StopIteration:
            raise

        self._samples_yielded += self._batch_size or 1
        return batch
