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
    best_fitness: float
    per_robot_costs: List[float]
    robot_tours: List[List[int]]
    generation_stats: List[Dict] = field(default_factory=list)
    runtime_seconds: float = 0.0
    feasible: bool = True


class RoSTAMGA:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

        # ── Build a consistent integer index over sub-tasks ──────────────
        # task.py stores dist_matrix as Dict[str, Dict[str, float]]
        # and start_matrix as Dict[str, float].
        # fitness.py needs numpy arrays: dist (n×n), start (robots×n).
        self.sub_ids = sorted(env.sub_tasks.keys())
        n = len(self.sub_ids)
        sid_to_idx = {sid: i for i, sid in enumerate(self.sub_ids)}

        # Convert nested dict → (n × n) numpy array
        dist = np.zeros((n, n), dtype=float)
        for a, row in env.dist_matrix.items():
            ia = sid_to_idx[a]
            for b, d in row.items():
                dist[ia][sid_to_idx[b]] = d
        self.dist_matrix = dist

        # depot-to-task distances as 1D array (n,)
        depot_dists = np.array(
            [env.start_matrix[sid] for sid in self.sub_ids], dtype=float
        )

        # ── Active robots ────────────────────────────────────────────────
        self.robot_num = int(getattr(cfg.environment, "robot_count", 3))
        self.task_type = getattr(cfg.environment, "task_type", 1)

        # All robots start at the same depot → replicate row robot_num times
        # start_matrix shape: (robot_num, n)
        self.start_matrix = np.tile(depot_dists, (self.robot_num, 1))

        # Speed matrix (robot_num,)
        speeds = getattr(cfg.environment, "speed_matrix", None)
        if speeds is not None and len(speeds) >= self.robot_num:
            self.speed_matrix = np.asarray(speeds[:self.robot_num], dtype=float)
        else:
            self.speed_matrix = np.ones(self.robot_num, dtype=float)

        self._penalty = make_penalty(cfg.penalty)
        self._stats: List[Dict] = []
        self._rng = np.random.default_rng(cfg.ea.seed)

    # ------------------------------------------------------------------
    def run(self) -> GAResult:
        start_time = time.time()
        cfg = self.cfg
        num_tasks = len(self.sub_ids)
        pop_size = cfg.ea.population_size

       def score_all(population: List[Individual], penalty: float) -> None:
            for ind in population:
                fit.evaluate(ind, self.dist_matrix, self.start_matrix,
                     self.robot_num, self.speed_matrix, penalty,
                     sub_ids=self.sub_ids,
                     task_type=self.cfg.environment.task_type)

        # ── 1. Initial population ────────────────────────────────────────
        pop = random_population(num_tasks, self.robot_num, pop_size, self._rng)
        score_all(pop, self._penalty.get_penalty(0))

        best_valid: Optional[Individual] = None

        # ── 2. Generational loop ─────────────────────────────────────────
        for gen in range(cfg.ea.num_generations):
            pop.sort(key=lambda ind: ind.fitness)

            if gen > 0 and cfg.ea.ais_injection_freq and gen % cfg.ea.ais_injection_freq == 0:
                inject_random(pop, num_tasks, self.robot_num, cfg.ea.ais_inject_count, self._rng)

            current_best_valid = self._get_best_valid(pop)
            if current_best_valid is not None:
                if best_valid is None or current_best_valid.fitness < best_valid.fitness:
                    best_valid = current_best_valid.copy()

            # ── Reproduction ─────────────────────────────────────────────
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

            score_all(pop, self._penalty.get_penalty(gen))
            gen_best = min(pop, key=lambda ind: ind.fitness)
            self._penalty.update(gen_best, self.robot_num, gen)

            fits = [ind.fitness for ind in pop]
            stats = {
                "generation": gen,
                "min": float(np.min(fits)), "avg": float(np.mean(fits)),
                "max": float(np.max(fits)), "std": float(np.std(fits)),
            }
            self._stats.append(stats)

            if gen % cfg.logging.log_frequency == 0:
                logger.info("Gen %4d | min=%.3f avg=%.3f max=%.3f | penalty=%.3f",
                            gen, stats["min"], stats["avg"], stats["max"],
                            self._penalty.get_penalty(gen))

        # ── 3. Package results ───────────────────────────────────────────
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
