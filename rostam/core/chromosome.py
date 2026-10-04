"""
Chromosome Representation
==========================
RoSTAM uses a two-part chromosome:

  Part 1 — Permutation  : ordered list of sub-task IDs
  Part 2 — Robot assign : list of integers, one per robot,
                          specifying how many consecutive sub-tasks
                          from Part 1 are allocated to that robot.

Example (3 robots, 9 sub-tasks):
  Part 1:  ['1,1', '3,1', '2,1', '5,1', '4,1', '6,1', '7,1', '9,1', '8,1']
  Part 2:  [3, 2, 4]   → robot0 gets first 3, robot1 next 2, robot2 last 4

The two parts together form one individual in the DEAP toolbox.
"""

from __future__ import annotations

import random
import copy
from typing import Dict, List, Optional

from deap import base, creator, tools


# ---------------------------------------------------------------------------
# DEAP type registration (call once per process)
# ---------------------------------------------------------------------------

def register_deap_types() -> None:
    """
    Register the Individual and Fitness types with DEAP's creator.
    Safe to call multiple times — skips if already registered.
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create(
            "FitnessMin",
            base.Fitness,
            weights=(-1.0,),
        )
    if not hasattr(creator, "Individual"):
        creator.create(
            "Individual",
            list,
            fitness=creator.FitnessMin,
            robots_assign=None,   # Part 2: list of int
        )


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _distribute_tasks_evenly(ind_size: int, robot_num: int) -> List[int]:
    """
    Distribute ind_size task slots roughly evenly across robot_num robots.
    Remainder tasks are distributed one-by-one to the first robots.

    Returns
    -------
    List[int]
        A list of length robot_num where each entry is the number of
        sub-tasks allocated to the corresponding robot.
    """
    base_count = ind_size // robot_num
    remainder = ind_size % robot_num
    assignment = [base_count] * robot_num
    for i in range(remainder):
        assignment[i] += 1
    random.shuffle(assignment)
    return assignment


def make_individual(
    sub_task_ids: List[str],
    robot_num: int,
    rng: Optional[random.Random] = None,
) -> "creator.Individual":
    """
    Create a single random Individual (chromosome).

    Parameters
    ----------
    sub_task_ids : List[str]
        All active sub-task ID strings (e.g. ['1,1', '2,1', ...]).
    robot_num : int
        Number of robots in the team.
    rng : random.Random, optional
        Local RNG instance for reproducibility. Falls back to module-level random.

    Returns
    -------
    creator.Individual
        A DEAP Individual with Part 1 (task permutation) and
        Part 2 (robots_assign) populated.
    """
    _rng = rng or random
    shuffled = _rng.sample(sub_task_ids, len(sub_task_ids))
    ind = creator.Individual(shuffled)
    ind.robots_assign = _distribute_tasks_evenly(len(ind), robot_num)
    return ind


def make_population(
    sub_task_ids: List[str],
    robot_num: int,
    pop_size: int,
    rng: Optional[random.Random] = None,
) -> List["creator.Individual"]:
    """
    Generate an initial random population.

    Parameters
    ----------
    sub_task_ids : List[str]
        Active sub-task IDs.
    robot_num : int
        Number of robots.
    pop_size : int
        Desired population size.
    rng : random.Random, optional
        Local RNG instance.

    Returns
    -------
    List[creator.Individual]
    """
    return [make_individual(sub_task_ids, robot_num, rng) for _ in range(pop_size)]


def clone_individual(ind: "creator.Individual") -> "creator.Individual":
    """Deep copy an Individual, preserving fitness and robots_assign."""
    new_ind = creator.Individual(list(ind))
    new_ind.robots_assign = list(ind.robots_assign)
    if ind.fitness.valid:
        new_ind.fitness.values = ind.fitness.values
    return new_ind


def get_robot_tours(ind: "creator.Individual") -> List[List[str]]:
    """
    Decode an Individual into per-robot sub-task lists.

    Returns
    -------
    List[List[str]]
        One list of sub-task IDs per robot.
    """
    tours = []
    cursor = 0
    for count in ind.robots_assign:
        tours.append(list(ind[cursor: cursor + count]))
        cursor += count
    return tours
