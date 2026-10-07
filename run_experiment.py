"""
RoSTAM — Main Experiment Entry Point
======================================
Run a complete MRTA experiment using the RoSTAM framework.

Usage
-----
    python run_experiment.py                          # uses default config
    python run_experiment.py --config my_config.yaml  # custom config
    python run_experiment.py --config my_config.yaml --save-env  # save environment for reuse

Example
-------
    python run_experiment.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        description="RoSTAM — Multi-Robot Task Allocation Framework"
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML configuration file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--save-env",
        action="store_true",
        help="Save the generated task environment to disk for reuse.",
    )
    parser.add_argument(
        "--load-env",
        type=str,
        default=None,
        help="Load a previously saved TaskEnvironment from this directory.",
    )
    parsed = parser.parse_args(args)

    # ── Load configuration ────────────────────────────────────────────
    from rostam.config import load_config
    cfg = load_config(parsed.config)
    _setup_logging(cfg.logging.level)
    logger = logging.getLogger("rostam.main")

    logger.info("Starting RoSTAM experiment: %s", cfg.experiment.name)
    logger.info("Problem distribution : %s", cfg.experiment.problem_distribution)
    logger.info("Robots: %d  |  Tasks: %d  |  Generations: %d",
                cfg.environment.num_robots,
                cfg.environment.num_tasks,
                cfg.ea.num_generations)

    # ── Build or load task environment ───────────────────────────────
    from rostam.environment.task import TaskEnvironment

    if parsed.load_env:
        logger.info("Loading task environment from: %s", parsed.load_env)
        env = TaskEnvironment.load(parsed.load_env)
    else:
        env = TaskEnvironment(
            num_tasks=cfg.environment.num_tasks,
            map_dim=cfg.environment.map_dim,
            task_type=cfg.environment.task_type,
            seed=cfg.ea.seed,
        )
        logger.info("Generated environment: %s", env)

    if parsed.save_env:
        out_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
        env.save(out_dir)
        logger.info("Environment saved to: %s", out_dir)

    # ── Run GA ────────────────────────────────────────────────────────
    from rostam.algorithms.ga import RoSTAMGA

    ga = RoSTAMGA(cfg, env)
    result = ga.run()
    sub_ids = ga.sub_ids  # index → dot-notation mapping
    named_tours = [
    [sub_ids[idx] for idx in tour]
    for tour in result.robot_tours
]

    # ── Save results ──────────────────────────────────────────────────
    out_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": cfg.experiment.name,
        "problem_distribution": cfg.experiment.problem_distribution,
        "best_makespan": result.best_fitness,
        "per_robot_costs": result.per_robot_costs,
        "feasible": result.feasible,
        "runtime_seconds": result.runtime_seconds,
        "robot_tours": named_tours,   # was result.robot_tours
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to: %s", summary_path)

    # ── Console summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  RoSTAM Experiment: {cfg.experiment.name}")
    print("=" * 60)
    print(f"  Best makespan  : {result.best_fitness:.4f}")
    print(f"  Feasible       : {result.feasible}")
    print(f"  Runtime        : {result.runtime_seconds:.1f}s")
    print(f"  Per-robot costs: {[f'{c:.3f}' for c in result.per_robot_costs]}")
    print("-" * 60)
    for i, tour in enumerate(named_tours):
        print(f"  Robot {i+1}: {tour}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
