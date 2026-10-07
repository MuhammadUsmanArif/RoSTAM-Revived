"""
Task and Environment Representation
=====================================
Defines the data structures for tasks, sub-tasks (task instances), and the
overall task environment used throughout RoSTAM.

Key Concepts
------------
- A **Task** is a physical mission objective at a map coordinate.
  It has a requirement vector specifying which robot capabilities are needed.
- A **SubTask** (or task instance) is one required-robot slot for a Task.
  For an ST (single-robot) task, one SubTask is created. For an MR task with
  cardinality q, q SubTasks are created—one per required robot.
- The **TaskEnvironment** manages the full task set, sub-task dictionary,
  and the distance/cost matrix used by the fitness function.
"""

from __future__ import annotations

import random
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A single task in the environment.

    Attributes
    ----------
    task_id : int
        Unique integer identifier.
    x : float
        X coordinate on the map.
    y : float
        Y coordinate on the map.
    requirements : List[str]
        Capability tokens required by this task (e.g. ['water_jet', 'gripper']).
        Cardinality gives the number of robots needed.
    """
    task_id: int
    x: float
    y: float
    requirements: List[str] = field(default_factory=list)

    @property
    def cardinality(self) -> int:
        """Number of robots required to complete this task."""
        return len(self.requirements) if self.requirements else 1

    def distance_to(self, other: "Task") -> float:
        """Euclidean distance from this task to another."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self) -> str:
        return f"Task(id={self.task_id}, x={self.x:.2f}, y={self.y:.2f}, q={self.cardinality})"


@dataclass
class SubTask:
    """
    A single robot-slot instance of a Task.
    For an MR task with cardinality q, q SubTasks are created.

    Attributes
    ----------
    sub_id : str
        Composite key in format 'task_id,instance_index'.
    parent_task_id : int
        ID of the parent Task.
    required_capability : str
        The specific capability token this slot requires.
    x : float
        Inherited x coordinate from parent task.
    y : float
        Inherited y coordinate from parent task.
    """
    sub_id: str
    parent_task_id: int
    required_capability: str
    x: float
    y: float

    def __repr__(self) -> str:
        return f"SubTask(id={self.sub_id}, cap={self.required_capability})"


class TaskEnvironment:
    """
    Manages the complete task environment for an MRTA experiment.

    Responsibilities
    ----------------
    - Generate random task layouts.
    - Expand tasks into sub-tasks (task instances).
    - Compute and cache the inter-task distance matrix.
    - Support serialisation for reproducible experiments.

    Parameters
    ----------
    num_tasks : int
        Number of top-level tasks to generate.
    map_dim : int
        Tasks are placed in [-map_dim, map_dim] x [-map_dim, map_dim].
    task_type : int
        1 = all SR tasks (q=1)
        2 = MR loosely coupled (random q in {1,2,3})
        3 = MR tightly coupled (same as 2 but flagged for deadlock checking)
        4 = multi-tasking tasks (heterogeneous capability requirements)
    depot : Tuple[float, float]
        Home depot coordinates for all robots. Default (0, 0).
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_tasks: int,
        map_dim: int = 6,
        task_type: int = 1,
        depot: Tuple[float, float] = (0.0, 0.0),
        seed: Optional[int] = None,
    ):
        self.num_tasks = num_tasks
        self.map_dim = map_dim
        self.task_type = task_type
        self.depot_x, self.depot_y = depot
        self._rng = random.Random(seed)

        # Core data structures
        self.tasks: Dict[int, Task] = {}
        self.sub_tasks: Dict[str, SubTask] = {}
        self.dist_matrix: Dict[str, Dict[str, float]] = {}  # sub_id → sub_id → cost
        self.start_matrix: Dict[str, float] = {}            # sub_id → depot distance

        self._generate()
        self._expand_to_subtasks()
        self._compute_cost_matrix()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Generate random task positions and capability requirements."""
        CAPABILITY_POOL = [
            "water_jet", "gripper", "infrared_camera",
            "load_pusher", "gas_detector", "drill",
        ]
        for i in range(1, self.num_tasks + 1):
            x = self._rng.uniform(-self.map_dim, self.map_dim)
            y = self._rng.uniform(-self.map_dim, self.map_dim)

            if self.task_type == 1:
                # ST — single-robot tasks, one generic requirement
                reqs = ["generic"]
            elif self.task_type in (2, 3):
                # MR loosely/tightly coupled — 1-3 robots needed
                q = self._rng.randint(1, 3)
                reqs = self._rng.choices(CAPABILITY_POOL, k=q)
            else:
                # Multi-tasking — heterogeneous 1-4 requirements
                q = self._rng.randint(1, 4)
                reqs = self._rng.choices(CAPABILITY_POOL, k=q)

            self.tasks[i] = Task(task_id=i, x=x, y=y, requirements=reqs)

    def _expand_to_subtasks(self) -> None:
        """
        Expand tasks into sub-tasks (task instances).
        Each required-robot slot becomes one SubTask with key 'task_id,slot_index'.
        """
        self.sub_tasks.clear()
        for task in self.tasks.values():
            for slot_idx, cap in enumerate(task.requirements):
                sub_id = f"{task.task_id}.{slot_idx + 1}"
                self.sub_tasks[sub_id] = SubTask(
                    sub_id=sub_id,
                    parent_task_id=task.task_id,
                    required_capability=cap,
                    x=task.x,
                    y=task.y,
                )

    def _compute_cost_matrix(self) -> None:
        """
        Compute the Euclidean distance between every pair of sub-tasks
        and from the depot to each sub-task.

        The distance matrix is symmetric: dist[a][b] == dist[b][a].
        """
        ids = list(self.sub_tasks.keys())
        self.dist_matrix = {i: {} for i in ids}

        for a in ids:
            st_a = self.sub_tasks[a]
            # Depot-to-task distance
            self.start_matrix[a] = math.sqrt(
                (st_a.x - self.depot_x) ** 2 + (st_a.y - self.depot_y) ** 2
            )
            for b in ids:
                if a == b:
                    self.dist_matrix[a][b] = 0.0
                    continue
                st_b = self.sub_tasks[b]
                d = math.sqrt((st_a.x - st_b.x) ** 2 + (st_a.y - st_b.y) ** 2)
                self.dist_matrix[a][b] = d

    # ------------------------------------------------------------------
    # Dynamic Environment Operations
    # ------------------------------------------------------------------

    def remove_attempted_tasks(self, attempted_ids: List[str]) -> None:
        """
        Remove completed sub-tasks from the active sub-task dictionary.
        Used during rolling-horizon re-planning.

        Parameters
        ----------
        attempted_ids : List[str]
            Sub-task IDs that have been successfully executed.
        """
        for sid in attempted_ids:
            self.sub_tasks.pop(sid, None)

    def add_new_tasks(
        self, new_tasks: List[Task], recompute_matrix: bool = True
    ) -> None:
        """
        Add dynamically arriving tasks to the environment.

        Parameters
        ----------
        new_tasks : List[Task]
            New Task objects to add.
        recompute_matrix : bool
            Whether to recompute the full distance matrix after adding tasks.
            Set to False for batch additions where you call recompute manually.
        """
        for task in new_tasks:
            self.tasks[task.task_id] = task
            for slot_idx, cap in enumerate(task.requirements):
                sub_id = f"{task.task_id},{slot_idx + 1}"
                self.sub_tasks[sub_id] = SubTask(
                    sub_id=sub_id,
                    parent_task_id=task.task_id,
                    required_capability=cap,
                    x=task.x,
                    y=task.y,
                )
        if recompute_matrix:
            self._compute_cost_matrix()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist the environment state to disk for reproducibility."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "task_env.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, directory: str | Path) -> "TaskEnvironment":
        """Load a previously saved TaskEnvironment."""
        with open(Path(directory) / "task_env.pkl", "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        return (
            f"TaskEnvironment(tasks={len(self.tasks)}, "
            f"sub_tasks={len(self.sub_tasks)}, "
            f"task_type={self.task_type})"
        )
