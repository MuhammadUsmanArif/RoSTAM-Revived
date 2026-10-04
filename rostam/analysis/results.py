"""Results consolidation: in-memory records -> pandas DataFrame / Excel / JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class ResultsCollector:
    def __init__(self):
        self.records: List[Dict] = []

    def add(self, **record) -> None:
        self.records.append(record)

    def add_ga_result(self, algorithm: str, experiment: str, result) -> None:
        self.add(algorithm=algorithm, experiment=experiment,
                 best_makespan=result.best_fitness, feasible=result.feasible,
                 runtime_seconds=result.runtime_seconds,
                 per_robot_costs=result.per_robot_costs)

    def add_window_results(self, algorithm: str, experiment: str, window_results) -> None:
        for w in window_results:
            self.add(algorithm=algorithm, experiment=experiment, window_id=w.window_id,
                     running_makespan=w.running_makespan,
                     committed_count=len(w.committed_tasks))

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)

    def to_excel(self, path: str, sheet_name: str = "results") -> None:
        df = self.to_dataframe()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    def to_json(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.records, f, indent=2, default=str)
