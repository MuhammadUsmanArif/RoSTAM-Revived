"""Darrah-style greedy allocation baseline.

The legacy Darrah_2015.py solved the allocation MILP remotely on NEOS
(CPLEX). This module is the dependency-free heuristic stand-in with the same
interface: robots repeatedly claim their nearest unallocated task until all
tasks are assigned. Swap in a MILP solver later if you need the exact baseline.
"""

from __future__ import annotations

import numpy as np

from rostam.core.individual import Individual


class DarrahGreedy:
    def __init__(self, dist_matrix: np.ndarray, start_matrix: np.ndarray,
                 robot_num: int, speed_matrix: np.ndarray):
        self.dist = np.asarray(dist_matrix, dtype=float)
        self.start = np.asarray(start_matrix, dtype=float)
        self.robot_num = robot_num
        self.speed = np.asarray(speed_matrix, dtype=float)

    def run(self) -> Individual:
        num_tasks = self.dist.shape[0]
        genes = [-1] * num_tasks
        current = [-1] * self.robot_num       # last visited task per robot
        cost = [0.0] * self.robot_num
        remaining = set(range(num_tasks))

        while remaining:
            best_gain, best_robot, best_task = float("inf"), -1, -1
            for r in range(self.robot_num):
                speed = float(self.speed[r]) if r < len(self.speed) and self.speed[r] > 0 else 1.0
                for t in remaining:
                    d = float(self.start[r, t]) if current[r] == -1 else float(self.dist[current[r], t])
                    gain = cost[r] + d / speed
                    if gain < best_gain - 1e-12:
                        best_gain, best_robot, best_task = gain, r, t
            if best_robot < 0:
                break
            genes[best_task] = best_robot
            cost[best_robot] = best_gain
            current[best_robot] = best_task
            remaining.discard(best_task)

        for t, g in enumerate(genes):          # safety for zero-robot edge case
            if g < 0:
                genes[t] = 0
        return Individual(genes)
