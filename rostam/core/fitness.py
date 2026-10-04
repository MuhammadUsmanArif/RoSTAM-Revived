"""Fitness functions: per-robot route cost, makespan, feasibility.

Convention
----------
dist_matrix  : (num_tasks, num_tasks) pairwise task travel distances
start_matrix : (num_robots, num_tasks) distance from robot i's start to task j
speed_matrix : (robot_num,) speeds of the ACTIVE robots

Each robot visits its assigned tasks in nearest-neighbour order starting
from its own start position; its cost is total travel time (distance/speed).
The objective is the makespan: max cost across robots (minimised).
"""

from __future__ import annotations

from typing import List

import numpy as np

from rostam.core.individual import Individual


def is_infeasible(individual: Individual, robot_num: int) -> bool:
    """Infeasible if any task is assigned to a non-existent/failed robot."""
    return individual.violations > 0


def count_violations(individual: Individual, robot_num: int) -> int:
    return sum(1 for g in individual.genes if g < 0 or g >= robot_num)


def per_robot_costs(individual: Individual, dist_matrix: np.ndarray,
                    start_matrix: np.ndarray, robot_num: int,
                    speed_matrix: np.ndarray) -> List[float]:
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    start_matrix = np.asarray(start_matrix, dtype=float)
    costs = [0.0] * robot_num
    for r in range(robot_num):
        tasks = [t for t, g in enumerate(individual.genes) if g == r]
        if not tasks:
            continue
        remaining = list(tasks)
        current = -1  # -1 means "at start position"
        cost = 0.0
        speed = float(speed_matrix[r]) if r < len(speed_matrix) and speed_matrix[r] > 0 else 1.0
        while remaining:
            if current == -1:
                d = [float(start_matrix[r, t]) for t in remaining]
            else:
                d = [float(dist_matrix[current, t]) for t in remaining]
            i = int(np.argmin(d))
            nxt = remaining.pop(i)
            cost += d[i] / speed
            current = nxt
        costs[r] = cost
    return costs


def makespan(individual: Individual, dist_matrix: np.ndarray,
             start_matrix: np.ndarray, robot_num: int,
             speed_matrix: np.ndarray) -> float:
    costs = per_robot_costs(individual, dist_matrix, start_matrix, robot_num, speed_matrix)
    return max(costs) if robot_num else 0.0


def evaluate(individual: Individual, dist_matrix: np.ndarray,
             start_matrix: np.ndarray, robot_num: int,
             speed_matrix: np.ndarray, penalty_value: float = 0.0) -> float:
    """Set fitness/makespan/feasibility on the individual and return fitness."""
    individual.violations = count_violations(individual, robot_num)
    individual.feasible = individual.violations == 0
    individual.makespan = makespan(individual, dist_matrix, start_matrix, robot_num, speed_matrix)
    individual.fitness = individual.makespan + (penalty_value * individual.violations
                                                if not individual.feasible else 0.0)
    return individual.fitness
