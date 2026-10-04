"""Robot and RobotTeam — heterogeneity metadata (speeds, capabilities, failures)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Robot:
    id: int
    name: str = ""
    speed: float = 1.0
    position: Tuple[float, float] = (0.0, 0.0)
    capabilities: frozenset = frozenset()
    failed: bool = False

    @property
    def is_active(self) -> bool:
        return not self.failed


@dataclass
class RobotTeam:
    robots: List[Robot] = field(default_factory=list)

    @property
    def active_robots(self) -> List[Robot]:
        return [r for r in self.robots if r.is_active]

    @property
    def robot_num(self) -> int:
        return len(self.active_robots)

    @classmethod
    def from_config(cls, env_cfg) -> "RobotTeam":
        """Build from cfg.environment (num_robots, speed_matrix, failed_robots)."""
        n = env_cfg.num_robots
        speeds = list(env_cfg.speed_matrix) + [1.0] * n
        failed = list(env_cfg.failed_robots) + [1] * n
        robots = [Robot(id=i, name=f"robot_{i}", speed=float(speeds[i]),
                        failed=(int(failed[i]) == 0)) for i in range(n)]
        return cls(robots=robots)
