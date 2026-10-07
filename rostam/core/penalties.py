"""Penalty strategies for infeasible individuals (fixed / dynamic / adaptive)."""

from __future__ import annotations


class FixedPenalty:
    def __init__(self, value: float = 1.0):
        self.value = value

    def update(self, best_individual, robot_num: int, gen: int) -> None:
        pass

    def get_penalty(self, gen: int) -> float:
        return self.value


class DynamicPenalty:
    """Penalty grows linearly every `window` generations."""

    def __init__(self, initial: float = 1.0, window: int = 10,
                 rate: float = 0.1, max_value: float = 20.0):
        self.initial, self.window, self.rate, self.max_value = initial, max(1, window), rate, max_value

    def update(self, best_individual, robot_num: int, gen: int) -> None:
        pass

    def get_penalty(self, gen: int) -> float:
        return min(self.initial + self.rate * (gen // self.window), self.max_value)


class AdaptivePenalty:
    """Increases when the current best is infeasible, eases off when feasible.

    Mirrors the adaptive scheme from legacy penalties.py: penalty is nudged by
    `adaptive_rate` (as a fraction) every generation, clamped to [floor, max].
    """

    def __init__(self, initial: float = 1.0, window: int = 10,
                 rate: float = 0.15, max_value: float = 20.0):
        self.value = initial
        self.floor = initial
        self.window = max(1, window)
        self.rate = rate
        self.max_value = max_value

    def update(self, best_individual, robot_num: int, gen: int) -> None:
        if best_individual is None:
            return
        if getattr(best_individual, "violations", 0) > 0:
            self.value *= (1.0 + self.rate)
        else:
            self.value *= (1.0 - self.rate)
            self.value = max(self.value, self.floor)
        self.value = min(self.value, self.max_value)

    def get_penalty(self, gen: int) -> float:
        return self.value


def make_penalty(cfg_penalty) -> FixedPenalty | DynamicPenalty | AdaptivePenalty:
    mode = cfg_penalty.mode
    if mode == "fixed":
        return FixedPenalty(cfg_penalty.initial_value)
    if mode == "dynamic":
        return DynamicPenalty(cfg_penalty.initial_value, cfg_penalty.adaptive_window,
                              cfg_penalty.adaptive_rate, cfg_penalty.max_value)
    return AdaptivePenalty(cfg_penalty.initial_value, cfg_penalty.adaptive_window,
                           cfg_penalty.adaptive_rate, cfg_penalty.max_value)
