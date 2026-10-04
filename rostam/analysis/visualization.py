"""Convergence, window and allocation plots (matplotlib)."""

from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")           # safe headless backend
import matplotlib.pyplot as plt


def plot_convergence(stats_list: List[List[dict]], labels: Optional[List[str]] = None,
                     save_path: Optional[str] = None, show: bool = False) -> None:
    labels = labels or [f"run {i+1}" for i in range(len(stats_list))]
    plt.figure(figsize=(8, 5))
    for stats, label in zip(stats_list, labels):
        gens = [s["generation"] for s in stats]
        plt.plot(gens, [s["min"] for s in stats], label=f"{label} (best)")
        plt.plot(gens, [s["avg"] for s in stats], "--", alpha=0.6, label=f"{label} (avg)")
    plt.xlabel("Generation"); plt.ylabel("Fitness (lower is better)")
    plt.title("RoSTAM convergence"); plt.legend(); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_window_makespans(window_results, save_path: Optional[str] = None,
                          show: bool = False) -> None:
    ids = [w.window_id for w in window_results]
    mk = [w.running_makespan for w in window_results]
    plt.figure(figsize=(8, 5))
    plt.bar(ids, mk)
    plt.xlabel("Window"); plt.ylabel("Running makespan")
    plt.title("Windowed execution"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_allocation(env, individual, save_path: Optional[str] = None,
                    show: bool = False) -> None:
    """Scatter tasks coloured by robot and draw each robot's route.

    Requires env to expose `task_positions` (num_tasks x 2 array) and
    `start_positions` (num_robots x 2). Silently skips if unavailable.
    """
    pos = getattr(env, "task_positions", None)
    starts = getattr(env, "start_positions", None)
    if pos is None:
        return
    plt.figure(figsize=(7, 7))
    num_robots = (max(individual.genes) + 1) if individual.genes else 0
    cmap = plt.get_cmap("tab10")
    for r in range(num_robots):
        tour = [t for t, g in enumerate(individual.genes) if g == r]
        if not tour:
            continue
        pts = [pos[t] for t in tour]
        xs, ys = zip(*pts)
        plt.scatter(xs, ys, color=cmap(r), label=f"robot {r+1}")
        plt.plot(xs, ys, color=cmap(r), alpha=0.4)
        if starts is not None:
            plt.scatter([starts[r][0]], [starts[r][1]], marker="s", color=cmap(r))
    plt.legend(); plt.title("Task allocation"); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
