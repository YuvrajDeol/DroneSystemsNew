from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DataLogger:
    out_dir: Path
    _rows: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def log(self, **fields: Any) -> None:
        self._rows.append(dict(fields))

    def to_csv(self, filename: str = "sim_log.csv") -> Path:
        path = self.out_dir / filename
        pd.DataFrame(self._rows).to_csv(path, index=False)
        return path

