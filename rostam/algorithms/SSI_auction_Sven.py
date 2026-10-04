"""Sequential Single-Item (SSI) auction baseline.

Each task is auctioned one at a time; every robot bids its marginal cost of
inserting the task into its current route (cheapest insertion position,
travel time = distance/speed). The lowest bidder wins. Mirrors the classic
SSI scheme (Koenig et al.) used in the legacy SSI_auction_Sven.py.
"""

from __future__ import annotations

from typing import List

import numpy as np

from rostam.core.individual import Individual


def _tour_cost(tour: List[int], r: int, dist: np.ndarray, start: np.ndarray, speed: float) -> float:
    if not tour:
        return 0.0
    cost, current = 0.0, -1
    for t in tour:
        d = float(start[r, t]) if current == -1 else float(dist[current, t])
        cost += d / speed
        current = t
    return cost


class SSI_Auction:
    def __init__(self, dist_matrix: np.ndarray, start_matrix: np.ndarray,
                 robot_num: int, speed_matrix: np.ndarray):
        self.dist = np.asarray(dist_matrix, dtype=float)
        self.start = np.asarray(start_matrix, dtype=float)
        self.robot_num = robot_num
        self.speed = np.asarray(speed_matrix, dtype=float)

    def run(self) -> Individual:
        num_tasks = self.dist.shape[0]
        tours: List[List[int]] = [[] for _ in range(self.robot_num)]
        unallocated = set(range(num_tasks))

        while unallocated:
            best_bid, best_robot, best_task = float("inf"), -1, -1
            for t in sorted(unallocated):
                for r in range(self.robot_num):
                    speed = float(self.speed[r]) if r < len(self.speed) and self.speed[r] > 0 else 1.0
                    base = _tour_cost(tours[r], r, self.dist, self.start, speed)
                    for pos in range(len(tours[r]) + 1):     # cheapest insertion
                        trial = tours[r][:pos] + [t] + tours[r][pos:]
                        bid = _tour_cost(trial, r, self.dist, self.start, speed) - base
                        if bid < best_bid - 1e-12:
                            best_bid, best_robot, best_task = bid, r, t
            if best_robot < 0:   # no active robots
                break
            tours[best_robot].append(best_task)
            unallocated.discard(best_task)

        genes = [0] * num_tasks
        for r, tour in enumerate(tours):
            for t in tour:
                genes[t] = r
        return Individual(genes)
