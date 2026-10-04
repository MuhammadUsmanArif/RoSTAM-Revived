"""
Fitness Functions
=================
All fitness evaluation logic for RoSTAM's EA.

The primary objective is to **minimise the makespan** — the time taken
by the slowest robot to complete its allocated sub-tasks.

Objective = max over all robots { TourCost(robot_r) }

where TourCost for robot r =
    depot → task_1 → task_2 → ... → task_k → depot
    (scaled by the robot's speed)

Infeasible solutions (where a robot has zero tasks but tasks remain,
or a required MR task doesn't have all slots allocated) are returned
with a penalty surcharge applied to their raw tour cost.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .chromosome import get_robot_tours


def _tour_cost_single_robot(
    sub_task_ids: List[str],
    dist_matrix: Dict[str, Dict[str, float]],
    start_matrix: Dict[str, float],
    speed: float = 1.0,
) -> float:
    """
    Compute the tour cost (time) for one robot's ordered list of sub-tasks.

    The robot starts at the depot, visits each sub-task in order,
    then returns to the depot.

    Parameters
    ----------
    sub_task_ids : List[str]
        Ordered sub-task IDs assigned to this robot.
    dist_matrix : Dict[str, Dict[str, float]]
        Pre-computed inter-sub-task distances.
    start_matrix : Dict[str, float]
        Pre-computed depot-to-sub-task distances.
    speed : float
        Robot speed multiplier. Cost = distance / speed.

    Returns
    -------
    float
        Total travel time for this robot's tour.
    """
    if not sub_task_ids:
        return 0.0

    # Depot → first task
    cost = start_matrix[sub_task_ids[0]]

    # Sequential inter-task legs
    for i in range(1, len(sub_task_ids)):
        cost += dist_matrix[sub_task_ids[i - 1]][sub_task_ids[i]]

    # Last task → depot (use start_matrix which stores task ↔ depot distance)
    cost += start_matrix[sub_task_ids[-1]]

    return cost / speed


def evaluate_makespan(
    individual,
    dist_matrix: Dict[str, Dict[str, float]],
    start_matrix: Dict[str, float],
    robot_num: int,
    speed_matrix: Optional[List[float]] = None,
) -> float:
    """
    Evaluate the makespan (max tour time across all robots) for an Individual.

    Parameters
    ----------
    individual : creator.Individual
        The chromosome to evaluate.
    dist_matrix : Dict
        Inter-sub-task distance matrix.
    start_matrix : Dict
        Depot-to-sub-task distance matrix.
    robot_num : int
        Number of robots.
    speed_matrix : List[float], optional
        Per-robot speed. Defaults to all 1.0.

    Returns
    -------
    float
        Makespan value (to be minimised).
    """
    speeds = speed_matrix or [1.0] * robot_num
    tours = get_robot_tours(individual)
    tour_costs = [
        _tour_cost_single_robot(tour, dist_matrix, start_matrix, speeds[i])
        for i, tour in enumerate(tours)
    ]
    return max(tour_costs) if tour_costs else float("inf")


def evaluate_with_penalty(
    individual,
    dist_matrix: Dict[str, Dict[str, float]],
    start_matrix: Dict[str, float],
    robot_num: int,
    penalty_value: float = 1.0,
    speed_matrix: Optional[List[float]] = None,
) -> float:
    """
    Evaluate fitness + penalty for infeasible solutions.

    An individual is infeasible if any robot has zero tasks while
    there are tasks in the chromosome (empty robot sub-tour).

    Parameters
    ----------
    individual : creator.Individual
        The chromosome to evaluate.
    dist_matrix : Dict
        Inter-sub-task distance matrix.
    start_matrix : Dict
        Depot-to-sub-task distance matrix.
    robot_num : int
        Number of robots.
    penalty_value : float
        Current adaptive penalty coefficient.
    speed_matrix : List[float], optional
        Per-robot speeds.

    Returns
    -------
    float
        Penalised fitness value (to be minimised).
    """
    raw_fitness = evaluate_makespan(
        individual, dist_matrix, start_matrix, robot_num, speed_matrix
    )
    penalty = compute_penalty(individual, robot_num, penalty_value)
    return raw_fitness + penalty


def compute_penalty(individual, robot_num: int, penalty_value: float) -> float:
    """
    Compute constraint-violation penalty for an individual.

    Penalty is proportional to the number of robots with empty tours
    multiplied by the current penalty coefficient.

    Parameters
    ----------
    individual : creator.Individual
        The chromosome to check.
    robot_num : int
        Expected number of robots.
    penalty_value : float
        Current penalty scaling coefficient.

    Returns
    -------
    float
        Penalty surcharge (0.0 if fully feasible).
    """
    tours = get_robot_tours(individual)
    empty_robots = sum(1 for tour in tours if len(tour) == 0)
    if empty_robots == 0:
        return 0.0
    return penalty_value * empty_robots * 1000.0


def is_infeasible(individual, robot_num: int) -> bool:
    """
    Return True if the individual has any empty robot sub-tour.

    Parameters
    ----------
    individual : creator.Individual
    robot_num : int

    Returns
    -------
    bool
    """
    tours = get_robot_tours(individual)
    return any(len(tour) == 0 for tour in tours)


def per_robot_costs(
    individual,
    dist_matrix: Dict[str, Dict[str, float]],
    start_matrix: Dict[str, float],
    robot_num: int,
    speed_matrix: Optional[List[float]] = None,
) -> List[float]:
    """
    Return the individual tour cost for each robot.
    Useful for detailed result logging.

    Returns
    -------
    List[float]
        One cost value per robot.
    """
    speeds = speed_matrix or [1.0] * robot_num
    tours = get_robot_tours(individual)
    return [
        _tour_cost_single_robot(tour, dist_matrix, start_matrix, speeds[i])
        for i, tour in enumerate(tours)
    ]
