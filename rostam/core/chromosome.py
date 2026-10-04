"""Population initialisation and tour extraction (pure NumPy/Python)."""

from __future__ import annotations

from typing import List

import numpy as np

from rostam.core.individual import Individual


def random_individual(num_tasks: int, robot_num: int, rng: np.random.Generator) -> Individual:
    """Uniformly random assignment of tasks to active robots."""
    genes = rng.integers(0, robot_num, size=num_tasks).tolist()
    return Individual(genes)


def random_population(num_tasks: int, robot_num: int, size: int,
                      rng: np.random.Generator) -> List[Individual]:
    return [random_individual(num_tasks, robot_num, rng) for _ in range(size)]


def get_robot_tours(individual: Individual) -> List[List[int]]:
    """Return, per robot, the list of task indices assigned to it (gene order)."""
    tours: List[List[int]] = []
    num_robots = (max(individual.genes) + 1) if individual.genes else 0
    for r in range(num_robots):
        tours.append([t for t, g in enumerate(individual.genes) if g == r])
    return tours
