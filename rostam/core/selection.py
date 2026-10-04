"""Selection operators: FPS, RBS, tournament, truncation (pure Python)."""

from __future__ import annotations

import numpy as np

from rostam.core.individual import Individual


def _score(ind: Individual) -> float:
    return ind.fitness if ind.fitness is not None else float("inf")


def tournament_selection(population, k: int = 3, rng=None) -> Individual:
    rng = rng or np.random.default_rng()
    idx = rng.integers(0, len(population), size=k)
    contenders = [population[i] for i in idx]
    return min(contenders, key=_score).copy()


def fps_selection(population, rng=None) -> Individual:
    """Fitness-proportional selection (minimisation -> invert scores)."""
    rng = rng or np.random.default_rng()
    scores = np.array([_score(i) for i in population], dtype=float)
    inv = 1.0 / (scores + 1e-12)
    total = inv.sum()
    if total <= 0:
        return population[int(rng.integers(len(population)))].copy()
    probs = inv / total
    i = int(rng.choice(len(population), p=probs))
    return population[i].copy()


def rbs_selection(population, rng=None) -> Individual:
    """Rank-based selection (best rank = lowest fitness)."""
    rng = rng or np.random.default_rng()
    order = sorted(range(len(population)), key=lambda i: _score(population[i]))
    ranks = np.empty(len(population), dtype=float)
    for rank_pos, i in enumerate(order):
        ranks[i] = rank_pos + 1.0
    probs = ranks / ranks.sum()   # higher rank -> higher probability
    i = int(rng.choice(len(population), p=probs))
    return population[i].copy()


def truncation_selection(population, rng=None) -> Individual:
    return min(population, key=_score).copy()
