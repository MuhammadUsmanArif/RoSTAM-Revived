"""Rolling-horizon windowed execution.

Tasks are revealed over time. At each window the solver allocates all
currently visible unallocated tasks; only the first `window_size` committed
tasks are executed before the horizon rolls forward and newly revealed tasks
join the pool. Mirrors the legacy Execution_followup.py window logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from rostam.core.individual import Individual

logger = logging.getLogger("rostam.window")

# A solver maps visible task indices -> Individual whose genes align to that list.
Solver = Callable[[List[int]], Individual]


@dataclass
class WindowResult:
    window_id: int
    visible_tasks: List[int]
    committed_tasks: List[int]
    per_robot_costs: List[float]
    running_makespan: float
    stats: Dict = field(default_factory=dict)


class WindowExecutor:
    def __init__(self, cfg, env, solver: Solver):
        self.cfg = cfg
        self.env = env
        self.dist = np.asarray(env.dist_matrix, dtype=float)
        self.start = np.asarray(env.start_matrix, dtype=float)
        self.solver = solver
        failed = list(getattr(cfg.environment, "failed_robots", []) or [])
        mask = np.asarray(failed, dtype=bool) if failed and len(failed) == self.start.shape[0] \
            else np.ones(self.start.shape[0], dtype=bool)
        self.robot_num = int(mask.sum())
        speeds = np.asarray(cfg.environment.speed_matrix, dtype=float)
        self.speed = speeds[mask] if speeds.size == mask.size else speeds[: self.robot_num]
        self.robot_cost = [0.0] * self.robot_num

    def run(self) -> List[WindowResult]:
        num_tasks = self.dist.shape[0]
        hidden_pct = float(getattr(self.cfg.environment, "hidden_task_percentage", 0.0) or 0.0)
        window_size = int(getattr(self.cfg.window, "window_size", 3))

        rng = np.random.default_rng(self.cfg.ea.seed)
        order = rng.permutation(num_tasks)
        n_hidden = int(round(hidden_pct * num_tasks))
        hidden = list(order[:n_hidden])
        visible = list(order[n_hidden:])

        results: List[WindowResult] = []
        window_id = 0

        while visible:
            individual = self.solver(visible)

            # Commit the first `window_size` tasks: those earliest on each robot's route
            committed: List[int] = []
            tours: List[List[int]] = [[] for _ in range(self.robot_num)]
            for local_t, robot in enumerate(individual.genes):
                tours[int(robot)].append(visible[local_t])
            for tour in tours:
                committed.extend(tour[:max(1, window_size // max(1, self.robot_num))])
            committed = committed[:window_size]

            # Advance simulated execution: add committed travel to each robot's clock
            for r in range(self.robot_num):
                for t in tours[r][:max(1, window_size // max(1, self.robot_num))]:
                    prev = -1
                    d = float(self.start[r, t]) if prev == -1 else float(self.dist[0, t])
                    self.robot_cost[r] += d / (float(self.speed[r]) or 1.0)

            makespan_now = max(self.robot_cost) if self.robot_num else 0.0
            results.append(WindowResult(window_id=window_id, visible_tasks=list(visible),
                                        committed_tasks=committed,
                                        per_robot_costs=list(self.robot_cost),
                                        running_makespan=makespan_now))
            logger.info("Window %d | visible=%d committed=%d makespan=%.3f",
                        window_id, len(visible), len(committed), makespan_now)

            # Roll the horizon: committed tasks leave, hidden tasks are revealed
            visible = [t for t in visible if t not in set(committed)]
            reveal = hidden[:window_size]
            hidden = hidden[window_size:]
            visible.extend(reveal)
            window_id += 1

        return results
