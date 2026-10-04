"""
Selection Operators
===================
Parent and survivor selection methods used in RoSTAM's EA.

All selection methods return copies of individuals (not references)
to avoid unintended mutation of the source population.
"""

from __future__ import annotations

import random
import copy
from typing import List


# ---------------------------------------------------------------------------
# Parent Selection
# ---------------------------------------------------------------------------

def select_fps(population: List, k: int = 2) -> List:
    """
    Fitness Proportionate Selection (FPS / Roulette Wheel).

    Individuals with lower fitness (minimisation) have higher selection
    probability. Fitness values are inverted for the roulette wheel.

    Parameters
    ----------
    population : List[creator.Individual]
        Current population with valid fitness values.
    k : int
        Number of individuals to select.

    Returns
    -------
    List[creator.Individual]
        k selected individuals (deep copies).
    """
    fitnesses = [ind.fitness.values[0] for ind in population]
    # Invert because we minimise (lower fitness = better)
    max_fit = max(fitnesses) + 1e-9
    inverted = [max_fit - f + 1e-9 for f in fitnesses]
    total = sum(inverted)
    selected = []
    for _ in range(k):
        pick = random.uniform(0, total)
        cumulative = 0.0
        for ind, w in zip(population, inverted):
            cumulative += w
            if cumulative >= pick:
                selected.append(copy.deepcopy(ind))
                break
        else:
            selected.append(copy.deepcopy(population[-1]))
    return selected


def select_rbs(population: List, k: int = 2) -> List:
    """
    Rank-Based Selection (RBS).

    Ranks individuals by fitness (best = rank 1) and assigns selection
    probability proportional to rank. Less susceptible to fitness dominance
    than FPS, preserving population diversity better in early generations.

    Parameters
    ----------
    population : List[creator.Individual]
    k : int

    Returns
    -------
    List[creator.Individual]
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    n = len(sorted_pop)
    ranks = list(range(n, 0, -1))  # rank n for best, rank 1 for worst
    total = sum(ranks)
    selected = []
    for _ in range(k):
        pick = random.uniform(0, total)
        cumulative = 0.0
        for ind, rank in zip(sorted_pop, ranks):
            cumulative += rank
            if cumulative >= pick:
                selected.append(copy.deepcopy(ind))
                break
        else:
            selected.append(copy.deepcopy(sorted_pop[0]))
    return selected


def select_tournament(population: List, k: int = 2, tournament_size: int = 2) -> List:
    """
    Binary (or k-way) Tournament Selection.

    Randomly samples `tournament_size` individuals and returns the best.
    Repeated k times. Simple, effective, and computationally cheap.

    Parameters
    ----------
    population : List[creator.Individual]
    k : int
        Number of individuals to return.
    tournament_size : int
        Size of each tournament.

    Returns
    -------
    List[creator.Individual]
    """
    selected = []
    for _ in range(k):
        aspirants = random.choices(population, k=tournament_size)
        winner = min(aspirants, key=lambda ind: ind.fitness.values[0])
        selected.append(copy.deepcopy(winner))
    return selected


# ---------------------------------------------------------------------------
# Survivor Selection
# ---------------------------------------------------------------------------

def select_truncation(population: List, n: int) -> List:
    """
    Truncation Selection (Elitist).

    Returns the n best individuals from the population (by fitness).
    Used for elitism — preserving the top solutions across generations.

    Parameters
    ----------
    population : List[creator.Individual]
    n : int
        Number of elite individuals to retain.

    Returns
    -------
    List[creator.Individual]
        The n best individuals (deep copies).
    """
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    return [copy.deepcopy(ind) for ind in sorted_pop[:n]]


def select_mu_plus_lambda(
    parents: List, offspring: List, mu: int
) -> List:
    """
    (μ + λ) Survivor Selection.

    Combines parents and offspring and selects the best mu individuals.
    Classic steady-state strategy that never discards good solutions.

    Parameters
    ----------
    parents : List[creator.Individual]
    offspring : List[creator.Individual]
    mu : int
        Target population size.

    Returns
    -------
    List[creator.Individual]
    """
    combined = parents + offspring
    return select_truncation(combined, mu)
