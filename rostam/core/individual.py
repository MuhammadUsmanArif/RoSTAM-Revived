"""Individual representation for RoSTAM — no external EA library.

An Individual is a task->robot assignment vector. Gene t holds the index
of the robot assigned to task t. Fitness is a plain float (makespan plus
any penalty); lower is better.
"""

from __future__ import annotations

from typing import List


class Individual:
    """A single candidate solution (task->robot assignment)."""

    __slots__ = ("genes", "fitness", "makespan", "feasible", "violations")

    def __init__(self, genes: List[int]):
        self.genes: List[int] = list(genes)
        self.fitness: float = float("inf")   # penalised score, minimised
        self.makespan: float = float("inf")  # true unpenalised makespan
        self.feasible: bool = True
        self.violations: int = 0

    def copy(self) -> "Individual":
        clone = Individual(self.genes)
        clone.fitness = self.fitness
        clone.makespan = self.makespan
        clone.feasible = self.feasible
        clone.violations = self.violations
        return clone

    def __len__(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f}, genes={self.genes})"
