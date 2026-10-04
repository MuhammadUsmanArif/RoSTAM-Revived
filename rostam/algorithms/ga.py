"""RoSTAM evolutionary engine — custom implementation, no DEAP.

Faithful to the legacy GA_solo.py loop: random init -> evaluate -> select ->
crossover -> mutate -> AIS injection -> adaptive penalty, with elitism.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from rostam.core import fitness as fit
from rostam.core.chromosome import get_robot_tours, random_population
from rostam.core.diversity import inject_random
from rostam.core.individual import Individual
from rostam.core.operators import block_crossover, mutate, uniform_crossover
from rostam.core.penalties import make_penalty
from rostam.core.selection import tournament_selection

logger = logging.getLogger("rostam.ga")


@dataclass
class GAResult:
    best_individual: Individual
    best_fitness: float                 # true (unpenalised) makespan
    per_robot_costs: List[float]
    robot_tours: List[List[int]]
    generation_stats: List[Dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    feasible: bool = True


class RoSTAMGA:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.dist_matrix = np.asarray(env.dist_matrix, dtype=float)
        self.start_matrix = np.asarray(env.start_matrix, dtype=float)

        # Active robots come from the failed_robots mask in the config.
        failed = list(getattr(cfg.environment, "failed_robots", []) or [])
        if failed and len(failed) == self.start_matrix.shape[0]:
            mask = np.asarray(failed, dtype=bool)
        else:
            mask = np.ones(self.start_matrix.shape[0], dtype=bool)
        self.robot_num = int(mask.sum())
        speeds = np.asarray(cfg.environment.speed_matrix, dtype=float)
        self.speed_matrix = speeds[mask] if speeds.size == mask.size else speeds[: self.robot_num]
        self.start_matrix = self.start_matrix[mask] if mask.size == np.asarray(env.start_matrix).shape[0] else self.start_matrix[: self.robot_num]

        self._penalty = make_penalty(cfg.penalty)
        self._stats: List[Dict] = []
        self._rng = np.random.default_rng(cfg.ea.seed)

    # ------------------------------------------------------------------
    def run(self) -> GAResult:
        start_time = time.time()
        cfg = self.cfg
        num_tasks = self.dist_matrix.shape[0]
        pop_size = cfg.ea.population_size

        def score_all(population: List[Individual], penalty: float) -> None:
            for ind in population:
                fit.evaluate(ind, self.dist_matrix, self.start_matrix,
                             self.robot_num, self.speed_matrix, penalty)

        # ── 1. Initial population ─────────────────────────────────────
        pop = random_population(num_tasks, self.robot_num, pop_size, self._rng)
        score_all(pop, self._penalty.get_penalty(0))

        best_valid: Optional[Individual] = None

        # ── 2. Generational loop ──────────────────────────────────────
        for gen in range(cfg.ea.num_generations):
            pop.sort(key=lambda ind: ind.fitness)

            # AIS injection: refresh diversity every N generations
            if gen > 0 and cfg.ea.ais_injection_freq and gen % cfg.ea.ais_injection_freq == 0:
                inject_random(pop, num_tasks, self.robot_num, cfg.ea.ais_inject_count, self._rng)

            # Track the best *feasible* individual ever seen
            current_best_valid = self._get_best_valid(pop)
            if current_best_valid is not None:
                if best_valid is None or current_best_valid.fitness < best_valid.fitness:
                    best_valid = current_best_valid.copy()

            # ── Reproduction ─────────────────────────────────────────
            elite = pop[0].copy()
            children: List[Individual] = [elite]
            rng = self._rng
            while len(children) < pop_size:
                p1 = tournament_selection(pop, k=3, rng=rng)
                p2 = tournament_selection(pop, k=3, rng=rng)
                if rng.random() < cfg.ea.crossover_prob:
                    c1, c2 = uniform_crossover(p1, p2, rng)
                    if rng.random() < 0.5:
                        c1, c2 = block_crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                mutate(c1, self.robot_num, cfg.ea.mutation_prob, rng)
                mutate(c2, self.robot_num, cfg.ea.mutation_prob, rng)
                children.extend([c1, c2])
            pop = children[:pop_size]

            # ── Evaluation with current penalty, then penalty update ──
            score_all(pop, self._penalty.get_penalty(gen))
            gen_best = min(pop, key=lambda ind: ind.fitness)
            self._penalty.update(gen_best, self.robot_num, gen)

            # ── Statistics ────────────────────────────────────────────
            fits = [ind.fitness for ind in pop]
            stats = {"generation": gen,
                     "min": float(np.min(fits)), "avg": float(np.mean(fits)),
                     "max": float(np.max(fits)), "std": float(np.std(fits))}
            self._stats.append(stats)

            if gen % cfg.logging.log_frequency == 0:
                logger.info("Gen %4d | min=%.3f avg=%.3f max=%.3f | penalty=%.3f",
                            gen, stats["min"], stats["avg"], stats["max"],
                            self._penalty.get_penalty(gen))

        # ── 3. Package results (true makespan, not penalised score) ───
        runtime = time.time() - start_time
        best = best_valid if best_valid is not None else min(pop, key=lambda ind: ind.fitness)

        costs = fit.per_robot_costs(best, self.dist_matrix, self.start_matrix,
                                    self.robot_num, self.speed_matrix)
        tours = get_robot_tours(best)

        logger.info("Run complete | best_makespan=%.4f | feasible=%s | runtime=%.1fs",
                    max(costs) if costs else 0.0, best.feasible, runtime)

        return GAResult(
            best_individual=best,
            best_fitness=max(costs) if costs else 0.0,
            per_robot_costs=costs,
            robot_tours=tours,
            generation_stats=self._stats,
            runtime_seconds=runtime,
            feasible=best.feasible,
        )

    # ------------------------------------------------------------------
    def _get_best_valid(self, population: List[Individual]) -> Optional[Individual]:
        valid = [ind for ind in population if not fit.is_infeasible(ind, self.robot_num)]
        if not valid:
            return None
        return min(valid, key=lambda ind: ind.fitness)
