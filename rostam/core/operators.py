"""Crossover and mutation for assignment vectors (pure Python, mirrors the
custom operators from legacy Operators.py)."""

from __future__ import annotations

import numpy as np

from rostam.core.individual import Individual


def uniform_crossover(p1: Individual, p2: Individual, rng: np.random.Generator):
    """Per-gene 50/50 swap between two parents -> two children."""
    c1, c2 = p1.copy(), p2.copy()
    mask = rng.random(len(p1)) < 0.5
    for t in range(len(p1)):
        if mask[t]:
            c1.genes[t], c2.genes[t] = c2.genes[t], c1.genes[t]
    return c1, c2


def block_crossover(p1: Individual, p2: Individual, rng: np.random.Generator):
    """Copy a random contiguous gene block from p1 into p2's child and vice versa."""
    n = len(p1)
    start, end = sorted(rng.integers(0, n, size=2).tolist())
    c1, c2 = p1.copy(), p2.copy()
    c1.genes[start:end + 1] = p2.genes[start:end + 1]
    c2.genes[start:end + 1] = p1.genes[start:end + 1]
    return c1, c2


def mutate(ind: Individual, robot_num: int, mutation_rate: float,
           rng: np.random.Generator) -> None:
    """In-place mutation: per-gene reassignment to a random robot, plus one
    assignment swap. `mutation_rate` is the per-individual probability."""
    if rng.random() > mutation_rate or not ind.genes:
        return
    n = len(ind)
    n_flip = max(1, int(0.1 * n))
    for t in rng.integers(0, n, size=n_flip):
        ind.genes[t] = int(rng.integers(0, robot_num))
    a, b = rng.integers(0, n, size=2)
    ind.genes[a], ind.genes[b] = ind.genes[b], ind.genes[a]
