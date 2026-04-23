"""Position-tracking sampler for mid-epoch DataLoader reconfiguration.

Wraps a DistributedSampler's index list and supports advance_to() / reset()
so that a newly-created DataLoader can resume from the exact sample position
after a num_workers change.
"""

from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class LiveSampler(Sampler):

    def __init__(self, dataset, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0):
        self._dist_sampler = DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank,
            shuffle=shuffle, seed=seed, drop_last=False,
        )
        self._indices: list[int] = []
        self._offset: int = 0
        self._epoch: int = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        self._dist_sampler.set_epoch(epoch)
        self._indices = list(self._dist_sampler)
        self._offset = 0

    @property
    def remaining(self) -> int:
        return len(self._indices) - self._offset

    def advance_to(self, offset: int):
        self._offset = min(offset, len(self._indices))

    def reset(self):
        self._offset = 0

    def __len__(self) -> int:
        return self.remaining

    def __iter__(self):
        for i in range(self._offset, len(self._indices)):
            yield self._indices[i]
