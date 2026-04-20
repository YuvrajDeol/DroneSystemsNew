from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SeaState:
    """Tiny placeholder for sea/environment configuration."""

    sea_state: int = 3
    wind_mps: float = 5.0

