import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.seed import make_generator, seed_everything, seed_worker


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_reproducible():
    seed_everything(5)
    a = _draw()
    seed_everything(5)
    assert a == _draw()
    seed_everything(6)
    assert a != _draw()


class _Idx(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        return i, random.random()


def _order(seed, workers=0):
    loader = DataLoader(
        _Idx(),
        batch_size=3,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=workers,
    )
    return [int(i) for batch in loader for i in batch[0]]


def test_dataloader_shuffle_reproducible():
    assert _order(1) == _order(1)
    assert _order(1) != _order(2)


def test_seed_worker_sets_python_seed():
    torch.manual_seed(3)
    seed_worker(0)
    a = random.random()
    seed_worker(0)
    assert a == random.random()
