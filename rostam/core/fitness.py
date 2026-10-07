"""Fitness functions: per-robot route cost, makespan, feasibility."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from rostam.core.individual import Individual


def count_violations(individual: Individual, robot_num: int,
                     sub_ids: Optional[List[str]] = None,
                     task_type: int = 1) -> int:
    """
    Count infeasibilities based on task type:
      task_type 1 (ST-SR-TA)  : only out-of-range robot genes
      task_type 2 (loose MR)  : above + same parent assigned to same robot
      task_type 3 (tight MR)  : above + DAG cycle count (deadlocks)
    """
    from rostam.core.penalties import count_loose_violations, count_tight_violations

    v = sum(1 for g in individual.genes if g < 0 or g >= robot_num)

    if sub_ids and task_type >= 2:
        v += count_loose_violations(individual, sub_ids)

    if sub_ids and task_type >= 3:
        v += count_tight_violations(individual, sub_ids, robot_num)

    return v


def is_infeasible(individual: Individual, robot_num: int) -> bool:
    return individual.violations > 0


def per_robot_costs(individual: Individual, dist_matrix: np.ndarray,
                    start_matrix: np.ndarray, robot_num: int,
                    speed_matrix: np.ndarray) -> List[float]:
    costs = [0.0] * robot_num
    for r in range(robot_num):
        tasks = [t for t, g in enumerate(individual.genes) if g == r]
        if not tasks:
            continue
        remaining = list(tasks)
        current = -1
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
             speed_matrix: np.ndarray, penalty_value: float = 0.0,
             sub_ids: Optional[List[str]] = None,
             task_type: int = 1) -> float:
    individual.violations = count_violations(individual, robot_num, sub_ids, task_type)
    individual.feasible = individual.violations == 0
    individual.makespan = makespan(individual, dist_matrix, start_matrix, robot_num, speed_matrix)
    individual.fitness = individual.makespan + (
        penalty_value * individual.violations if not individual.feasible else 0.0
    )
    return individual.fitness
