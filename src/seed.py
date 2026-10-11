"""Reproducibility helpers: global seeding and DataLoader determinism."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)  # legacy global state: transforms/libraries may use it
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive python/numpy seeds from torch's."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
