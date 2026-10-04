"""
RoSTAM Genetic Algorithm Engine
=================================
The core evolutionary loop for task allocation.

This module implements the complete EA as described in the RoSTAM thesis:
  - Random population initialisation (with optional AIS injection)
  - (μ + λ) survival selection with elitism
  - Adaptive penalty management
  - Per-generation statistics logging
  - Population diversity maintenance via AIS injection

Usage
-----
    from rostam.algorithms.ga import RoSTAMGA
    from rostam.environment.task import TaskEnvironment
    from rostam.config import load_config

    cfg = load_config("config/default.yaml")
    env = TaskEnvironment(cfg.environment.num_tasks, ...)

    ga = RoSTAMGA(cfg, env)
    best = ga.run()
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import RoSTAMConfig
from ..core.chromosome import (
    register_deap_types,
    make_individual,
    make_population,
    clone_individual,
    get_robot_tours,
)
from ..core.fitness import evaluate_with_penalty, is_infeasible, per_robot_costs
from ..core.operators import apply_variation
from ..core.selection import select_fps, select_tournament, select_truncation, select_mu_plus_lambda
from ..core.penalties import PenaltyController
from ..environment.task import TaskEnvironment


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class GAResult:
    """
    Container for a completed GA run's results.

    Attributes
    ----------
    best_individual :
        The best feasible Individual found during the run.
    best_fitness : float
        Makespan of the best individual.
    per_robot_costs : List[float]
        Individual tour cost for each robot.
    robot_tours : List[List[str]]
        Decoded sub-task assignments per robot.
    generation_stats : List[Dict]
        Per-generation statistics (min, avg, max, std fitness).
    runtime_seconds : float
        Wall-clock time of the EA run.
    feasible : bool
        Whether the best solution is feasible.
    """
    best_individual: object
    best_fitness: float
    per_robot_costs: List[float]
    robot_tours: List[List[str]]
    generation_stats: List[Dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    feasible: bool = True


# ---------------------------------------------------------------------------
# Main GA Class
# ---------------------------------------------------------------------------

class RoSTAMGA:
    """
    RoSTAM Evolutionary Algorithm for Multi-Robot Task Allocation.

    Implements the full EA lifecycle:
        1. Population initialisation
        2. Fitness evaluation
        3. Parent selection (FPS)
        4. Variation (Carter crossover + swap/inverse mutation)
        5. Survivor selection (μ + λ with elitism)
        6. AIS diversity injection every N generations
        7. Adaptive penalty management

    Parameters
    ----------
    config : RoSTAMConfig
        Full experiment configuration.
    env : TaskEnvironment
        The task environment (provides sub-tasks, dist matrix, start matrix).
    attempted_tasks : List[str], optional
        Sub-task IDs already completed (for windowed / iterative operation).
    start_locations : List[int], optional
        Depot offsets per robot (for non-depot start locations).
    """

    def __init__(
        self,
        config: RoSTAMConfig,
        env: TaskEnvironment,
        attempted_tasks: Optional[List[str]] = None,
        start_locations: Optional[List[int]] = None,
    ):
        self.cfg = config
        self.env = env
        self.robot_num = config.environment.num_robots
        self.speed_matrix = config.environment.speed_matrix

        # Remove already-attempted tasks from the active sub-task set
        self._active_sub_tasks = dict(env.sub_tasks)
        for sid in (attempted_tasks or []):
            self._active_sub_tasks.pop(sid, None)

        self._sub_task_ids = list(self._active_sub_tasks.keys())

        # Reproducibility
        _seed = config.ea.seed
        self._rng = random.Random(_seed)
        if _seed is not None:
            random.seed(_seed)
            np.random.seed(_seed)

        # Register DEAP types (idempotent)
        register_deap_types()

        # Penalty controller
        self._penalty = PenaltyController(
            mode=config.penalty.mode,
            initial_value=config.penalty.initial_value,
            adaptive_window=config.penalty.adaptive_window,
            adaptive_rate=config.penalty.adaptive_rate,
            max_value=config.penalty.max_value,
        )

        self._stats: List[Dict] = []
        self._best_valid: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, warm_population: Optional[List] = None) -> GAResult:
        """
        Execute the full evolutionary loop.

        Parameters
        ----------
        warm_population : List[creator.Individual], optional
            Pre-existing population to seed from (iterative/windowed restarts).

        Returns
        -------
        GAResult
            Complete results including the best individual found.
        """
        cfg_ea = self.cfg.ea
        cfg_log = self.cfg.logging

        if not self._sub_task_ids:
            raise ValueError("No active sub-tasks to allocate. Check task removal logic.")

        start_time = time.time()

        # ── 1. Initialise population ──────────────────────────────────
        if warm_population is not None:
            pop = [clone_individual(ind) for ind in warm_population]
            logger.info("Warm-starting from provided population (size %d).", len(pop))
        else:
            pop = make_population(
                self._sub_task_ids,
                self.robot_num,
                cfg_ea.population_size,
                self._rng,
            )
            logger.info("Initialised random population (size %d).", len(pop))

        # ── 2. Evaluate initial population ────────────────────────────
        for ind in pop:
            penalty = self._penalty.get_penalty(0)
            ind.fitness.values = (
                evaluate_with_penalty(
                    ind,
                    self.env.dist_matrix,
                    self.env.start_matrix,
                    self.robot_num,
                    penalty,
                    self.speed_matrix,
                ),
            )

        self._best_valid = self._get_best_valid(pop)

        # ── 3. Main evolutionary loop ─────────────────────────────────
        for gen in range(cfg_ea.num_generations):
            offspring: List = []

            # Parent selection and variation
            for _ in range(0, cfg_ea.population_size, 2):
                p1, p2 = select_fps(pop, k=2)
                c1 = clone_individual(p1)
                c2 = clone_individual(p2)

                c1, c2 = apply_variation(
                    c1, c2,
                    self.robot_num,
                    cfg_ea.crossover_prob,
                    cfg_ea.mutation_prob,
                    generation=gen,
                )
                # Evaluate offspring
                penalty = self._penalty.get_penalty(gen)
                c1.fitness.values = (
                    evaluate_with_penalty(
                        c1, self.env.dist_matrix, self.env.start_matrix,
                        self.robot_num, penalty, self.speed_matrix
                    ),
                )
                c2.fitness.values = (
                    evaluate_with_penalty(
                        c2, self.env.dist_matrix, self.env.start_matrix,
                        self.robot_num, penalty, self.speed_matrix
                    ),
                )
                offspring.extend([c1, c2])

            # Re-evaluate parents with current penalty (penalty may have changed)
            for ind in pop:
                ind.fitness.values = (
                    evaluate_with_penalty(
                        ind, self.env.dist_matrix, self.env.start_matrix,
                        self.robot_num, self._penalty.get_penalty(gen), self.speed_matrix
                    ),
                )

            # ── AIS Diversity Injection ───────────────────────────────
            ais_fresh = []
            if gen % cfg_ea.ais_injection_freq == 0:
                ais_fresh = make_population(
                    self._sub_task_ids, self.robot_num,
                    cfg_ea.ais_inject_count, self._rng
                )
                for ind in ais_fresh:
                    ind.fitness.values = (
                        evaluate_with_penalty(
                            ind, self.env.dist_matrix, self.env.start_matrix,
                            self.robot_num, 0.0, self.speed_matrix
                        ),
                    )

            # ── Survival Selection (μ + λ) ────────────────────────────
            combined = pop + offspring + ais_fresh
            # Elitism: always keep best 1
            elite = select_truncation(combined, 1)
            rest = select_tournament(combined, cfg_ea.population_size - 1, tournament_size=2)
            pop = elite + rest

            # ── Track best valid solution ─────────────────────────────
            current_best = self._get_best_valid(pop)
            if current_best is not None:
                if (
                    self._best_valid is None
                    or current_best.fitness.values[0] < self._best_valid.fitness.values[0]
                ):
                    self._best_valid = clone_individual(current_best)

            # ── Penalty update ────────────────────────────────────────
            gen_best = min(pop, key=lambda ind: ind.fitness.values[0])
            self._penalty.update(gen_best, self.robot_num, gen)

            # ── Statistics ───────────────────────────────────────────
            stats = self._compute_stats(pop, gen)
            self._stats.append(stats)

            if gen % cfg_log.log_frequency == 0:
                logger.info(
                    "Gen %4d | min=%.3f avg=%.3f max=%.3f | penalty=%.3f",
                    gen, stats["min"], stats["avg"], stats["max"],
                    self._penalty.get_penalty(gen),
                )

        # ── 4. Package results ────────────────────────────────────────
        runtime = time.time() - start_time
        best = self._best_valid or min(pop, key=lambda ind: ind.fitness.values[0])

        costs = per_robot_costs(
            best, self.env.dist_matrix, self.env.start_matrix,
            self.robot_num, self.speed_matrix
        )
        tours = get_robot_tours(best)
        feasible = not is_infeasible(best, self.robot_num)

        logger.info(
            "Run complete | best_fitness=%.4f | feasible=%s | runtime=%.1fs",
            max(costs), feasible, runtime,
        )

        return GAResult(
            best_individual=best,
            best_fitness=max(costs),
            per_robot_costs=costs,
            robot_tours=tours,
            generation_stats=self._stats,
            runtime_seconds=runtime,
            feasible=feasible,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_best_valid(self, population: List) -> Optional[object]:
        """Return the best feasible individual, or None if none exist."""
        valid = [ind for ind in population if not is_infeasible(ind, self.robot_num)]
        if not valid:
            return None
        return min(valid, key=lambda ind: ind.fitness.values[0])

    def _compute_stats(self, population: List, gen: int) -> Dict:
        """Compute summary statistics for the current generation."""
        fitnesses = [ind.fitness.values[0] for ind in population]
        return {
            "generation": gen,
            "min": float(np.min(fitnesses)),
            "avg": float(np.mean(fitnesses)),
            "max": float(np.max(fitnesses)),
            "std": float(np.std(fitnesses)),
        }
