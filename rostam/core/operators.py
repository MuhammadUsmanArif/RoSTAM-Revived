"""
Variation Operators — Crossover and Mutation
=============================================
All operators work on RoSTAM's two-part chromosome:
  Part 1: permutation of sub-task IDs
  Part 2: robots_assign (list of task-count integers per robot)

Crossover operators recombine Part 1 (permutation) and regenerate Part 2.
Mutation operators perturb both parts.
"""

from __future__ import annotations

import random
import copy
from typing import List, Tuple

from .chromosome import clone_individual, _distribute_tasks_evenly


# ---------------------------------------------------------------------------
# Crossover Operators
# ---------------------------------------------------------------------------

def crossover_order(
    parent1, parent2
) -> Tuple:
    """
    Order Crossover (OX) for permutation chromosomes.

    Selects a random segment from parent1 and fills remaining positions
    with genes from parent2 in their original order. Preserves relative
    ordering of elements not in the segment.

    Parameters
    ----------
    parent1, parent2 : creator.Individual
        Parent chromosomes. Modified IN PLACE.

    Returns
    -------
    Tuple of (child1, child2) — same objects as input, modified in place.
    """
    size = len(parent1)
    if size < 2:
        return parent1, parent2

    # Choose crossover window
    cx_start = random.randint(0, size - 2)
    cx_end = random.randint(cx_start + 1, size - 1)

    def _ox(p1, p2):
        child_genes = [None] * size
        # Copy segment from p1
        segment = set()
        for i in range(cx_start, cx_end + 1):
            child_genes[i] = p1[i]
            segment.add(p1[i])
        # Fill remaining positions from p2 in order
        p2_remaining = [g for g in p2 if g not in segment]
        idx = 0
        for i in range(size):
            if child_genes[i] is None:
                child_genes[i] = p2_remaining[idx]
                idx += 1
        return child_genes

    child1_genes = _ox(parent1, parent2)
    child2_genes = _ox(parent2, parent1)

    parent1[:] = child1_genes
    parent2[:] = child2_genes

    # Regenerate Part 2 after crossover
    parent1.robots_assign = _distribute_tasks_evenly(
        len(parent1), len(parent1.robots_assign)
    )
    parent2.robots_assign = _distribute_tasks_evenly(
        len(parent2), len(parent2.robots_assign)
    )

    # Invalidate fitness
    del parent1.fitness.values
    del parent2.fitness.values

    return parent1, parent2


def crossover_carter(
    parent1, parent2
) -> Tuple:
    """
    Carter 2006 Permutation Crossover.

    For each position, select the gene from parent1 if it would not
    create a duplicate, otherwise take the next available gene from parent2.
    Tends to preserve adjacency information from parent1 better than OX.

    Parameters
    ----------
    parent1, parent2 : creator.Individual
        Modified IN PLACE.

    Returns
    -------
    Tuple of (child1, child2).
    """
    def _carter_cx(p1, p2):
        size = len(p1)
        p2_queue = list(p2)
        child = []
        used = set()
        for gene in p1:
            if gene not in used:
                child.append(gene)
                used.add(gene)
            else:
                # Find next unused gene from p2
                while p2_queue and p2_queue[0] in used:
                    p2_queue.pop(0)
                if p2_queue:
                    next_gene = p2_queue.pop(0)
                    child.append(next_gene)
                    used.add(next_gene)
        return child

    parent1[:] = _carter_cx(parent1, parent2)
    parent2[:] = _carter_cx(parent2, parent1)

    parent1.robots_assign = _distribute_tasks_evenly(
        len(parent1), len(parent1.robots_assign)
    )
    parent2.robots_assign = _distribute_tasks_evenly(
        len(parent2), len(parent2.robots_assign)
    )

    del parent1.fitness.values
    del parent2.fitness.values

    return parent1, parent2


# ---------------------------------------------------------------------------
# Mutation Operators
# ---------------------------------------------------------------------------

def mutate_swap(individual, mut_prob: float) -> Tuple:
    """
    Swap Mutation.

    For each gene position, with probability mut_prob, swap it with
    a randomly chosen other position. Exploitative — makes small local
    changes to the tour ordering.

    Parameters
    ----------
    individual : creator.Individual
        Modified IN PLACE.
    mut_prob : float
        Per-gene probability of triggering a swap.

    Returns
    -------
    Tuple of (individual,)  — DEAP convention.
    """
    size = len(individual)
    for i in range(size):
        if random.random() < mut_prob:
            j = random.randint(0, size - 1)
            individual[i], individual[j] = individual[j], individual[i]
    del individual.fitness.values
    return (individual,)


def mutate_inverse(individual, mut_prob: float) -> Tuple:
    """
    Inverse (Inversion) Mutation.

    With probability mut_prob, select a random sub-segment and reverse it.
    More explorative than swap — can make larger structural changes.

    Parameters
    ----------
    individual : creator.Individual
        Modified IN PLACE.
    mut_prob : float
        Probability of applying inversion.

    Returns
    -------
    Tuple of (individual,).
    """
    if random.random() < mut_prob:
        size = len(individual)
        if size < 2:
            return (individual,)
        i = random.randint(0, size - 2)
        j = random.randint(i + 1, size - 1)
        individual[i:j + 1] = individual[i:j + 1][::-1]
    del individual.fitness.values
    return (individual,)


def mutate_robot_assign(individual, robot_num: int, mut_prob: float = 0.3) -> Tuple:
    """
    Robot Assignment Mutation.

    With probability mut_prob, randomly redistribute the task counts
    across robots (Part 2 of the chromosome). This explores different
    workload distributions while keeping the task ordering (Part 1) intact.

    Parameters
    ----------
    individual : creator.Individual
        Modified IN PLACE.
    robot_num : int
        Number of robots.
    mut_prob : float
        Probability of triggering the mutation.

    Returns
    -------
    Tuple of (individual,).
    """
    if random.random() < mut_prob:
        individual.robots_assign = _distribute_tasks_evenly(
            len(individual), robot_num
        )
        del individual.fitness.values
    return (individual,)


def apply_variation(
    child1,
    child2,
    robot_num: int,
    cx_prob: float,
    mut_prob: float,
    generation: int = 0,
) -> Tuple:
    """
    Apply crossover then mutation to a pair of children.

    Alternates between swap and inverse mutation based on a coin flip,
    which provides complementary exploration / exploitation balance.

    Parameters
    ----------
    child1, child2 : creator.Individual
        Children to vary. Modified IN PLACE.
    robot_num : int
        Number of robots (needed for robot-assign regeneration).
    cx_prob : float
        Crossover probability.
    mut_prob : float
        Per-gene mutation probability.
    generation : int
        Current generation (reserved for future annealing schedules).

    Returns
    -------
    Tuple of (child1, child2).
    """
    # Crossover
    if random.random() < cx_prob:
        child1, child2 = crossover_carter(child1, child2)

    # Mutation — alternate between swap and inverse
    if random.random() < 0.5:
        child1, = mutate_swap(child1, mut_prob)
        child2, = mutate_swap(child2, mut_prob)
    else:
        child1, = mutate_inverse(child1, mut_prob)
        child2, = mutate_inverse(child2, mut_prob)

    return child1, child2
