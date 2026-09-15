"""Explicit PCG64 generator construction and deterministic seed derivation."""

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

from backend.app.models.simulation import RandomEngine


def create_rng(
    seed: int | SeedSequence,
    engine: RandomEngine = RandomEngine.PCG64,
) -> Generator:
    """Construct an isolated generator; global NumPy RNG state is never used."""

    if engine is not RandomEngine.PCG64:
        raise ValueError(f"Unsupported random engine: {engine}")
    seed_sequence = seed if isinstance(seed, SeedSequence) else SeedSequence(seed)
    return Generator(PCG64(seed_sequence))


def spawn_child_seeds(root_seed: int, count: int) -> list[int]:
    """Derive deterministic child seeds for independent convergence streams."""

    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    if count < 0:
        raise ValueError("count must be non-negative")

    root = SeedSequence(root_seed)
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in root.spawn(count)]
