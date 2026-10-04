"""Population diversity metrics and AIS-style random injection."""

from __future__ import annotations

from typing import List

import numpy as np

from rostam.core.individual import Individual


def hamming_distance(a: Individual, b: Individual) -> int:
    return sum(1 for x, y in zip(a.genes, b.genes) if x != y)


def population_diversity(population: List[Individual]) -> float:
    """Mean pairwise Hamming distance normalised by chromosome length."""
    n = len(population)
    if n < 2 or not population[0].genes:
        return 0.0
    total, pairs = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += hamming_distance(population[i], population[j])
            pairs += 1
    return total / pairs / len(population[0].genes)


def inject_random(population: List[Individual], num_tasks: int, robot_num: int,
                  count: int, rng: np.random.Generator) -> None:
    """Replace the `count` worst individuals with fresh random ones (in place)."""
    if count <= 0:
        return
    ranked = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)
    for i in ranked[:count]:
        genes = rng.integers(0, robot_num, size=num_tasks).tolist()
        fresh = Individual(genes)
        fresh.fitness, fresh.makespan = float("inf"), float("inf")
        population[i] = fresh
