"""Adaptive penalty + DAG-based deadlock detection for RoSTAM."""

from __future__ import annotations
from collections import defaultdict
from typing import List, Dict
import numpy as np
from rostam.core.individual import Individual


# ── Loosely Coupled Infeasibility ────────────────────────────────────────────

def count_loose_violations(individual: Individual, sub_ids: List[str]) -> int:
    """
    Loosely coupled: no robot may be assigned more than one instance
    of the same parent task (same X in X.Y notation).
    """
    violations = 0
    robot_parent_map: Dict[int, set] = defaultdict(set)
    for idx, sid in enumerate(sub_ids):
        parent = sid.split(".")[0]
        robot = individual.genes[idx]
        if parent in robot_parent_map[robot]:
            violations += 1
        else:
            robot_parent_map[robot].add(parent)
    return violations


# ── DAG / Deadlock Detection (Tightly Coupled) ───────────────────────────────

def build_schedule_graph(individual: Individual, sub_ids: List[str],
                          robot_num: int) -> Dict[str, List[str]]:
    """
    Build directed graph G(T, E) where tasks are vertices and directed
    edges represent execution order within each robot's schedule.
    Edge t_i → t_j means robot visits t_i before t_j.
    For tightly coupled tasks, all sub-tasks of the same parent
    must be treated as the same vertex (the parent task node).
    """
    # Group sub_ids by robot, preserving gene order
    robot_schedules: Dict[int, List[str]] = defaultdict(list)
    for idx, sid in enumerate(sub_ids):
        robot = individual.genes[idx]
        parent = sid.split(".")[0]
        robot_schedules[robot].append(parent)

    # Build adjacency list — edges from consecutive tasks in each robot's plan
    graph: Dict[str, List[str]] = defaultdict(list)
    all_nodes = set()
    for robot, schedule in robot_schedules.items():
        # Deduplicate consecutive same-parent entries but keep order
        seen = []
        for node in schedule:
            if not seen or seen[-1] != node:
                seen.append(node)
            all_nodes.add(node)
        for i in range(len(seen) - 1):
            graph[seen[i]].append(seen[i + 1])

    # Ensure all nodes exist in graph
    for node in all_nodes:
        if node not in graph:
            graph[node] = []

    return dict(graph)


def count_cycles_johnson(graph: Dict[str, List[str]]) -> int:
    """
    Count number of simple cycles in a directed graph.
    Implements a simplified version of Johnson's algorithm using DFS.
    For penalty purposes — we count cycles, not enumerate them fully.
    """
    nodes = list(graph.keys())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    cycle_count = 0

    visited = [False] * n
    rec_stack = [False] * n

    def dfs(v: int) -> bool:
        nonlocal cycle_count
        visited[v] = True
        rec_stack[v] = True
        node = nodes[v]
        found_cycle = False
        for neighbour in graph.get(node, []):
            if neighbour not in node_idx:
                continue
            u = node_idx[neighbour]
            if not visited[u]:
                if dfs(u):
                    found_cycle = True
            elif rec_stack[u]:
                cycle_count += 1
                found_cycle = True
        rec_stack[v] = False
        return found_cycle

    for i in range(n):
        if not visited[i]:
            dfs(i)

    return cycle_count


def count_tight_violations(individual: Individual, sub_ids: List[str],
                            robot_num: int) -> int:
    """
    Tightly coupled: build the schedule graph and count cycles (deadlocks).
    Each cycle = one deadlock = one violation unit.
    """
    graph = build_schedule_graph(individual, sub_ids, robot_num)
    return count_cycles_johnson(graph)


# ── Adaptive Penalty ─────────────────────────────────────────────────────────

class AdaptivePenalty:
    """
    Tracks last N best solutions and adapts penalty factor:
    - All feasible   → reduce penalty (relax)
    - All infeasible → increase penalty (tighten)
    - Mixed          → hold
    """

    def __init__(self, base: float = 100.0, window: int = 10,
                 increase: float = 1.2, decrease: float = 0.9,
                 min_val: float = 1.0, max_val: float = 1e6):
        self._base = base
        self._penalty = base
        self._window = window
        self._increase = increase
        self._decrease = decrease
        self._min = min_val
        self._max = max_val
        self._history: List[bool] = []   # True = feasible

    def get_penalty(self, generation: int) -> float:
        return self._penalty

    def update(self, best: Individual, robot_num: int, generation: int) -> None:
        self._history.append(best.feasible)
        if len(self._history) > self._window:
            self._history.pop(0)
        if len(self._history) == self._window:
            if all(self._history):
                self._penalty = max(self._min, self._penalty * self._decrease)
            elif not any(self._history):
                self._penalty = min(self._max, self._penalty * self._increase)


def make_penalty(cfg) -> AdaptivePenalty:
    return AdaptivePenalty(
        base=getattr(cfg, "base_penalty", 100.0),
        window=getattr(cfg, "window", 10),
        increase=getattr(cfg, "increase_factor", 1.2),
        decrease=getattr(cfg, "decrease_factor", 0.9),
    )
