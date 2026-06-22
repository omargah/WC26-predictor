# -*- coding: utf-8 -*-
"""
Configuración central del proyecto.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    env_root = os.environ.get("MUNDIAL_2026_ROOT")

    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = get_project_root()


def get_paths(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT

    paths = {
        "root": root,
        "data": root / "data",
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "features": root / "data" / "features",
        "fixtures": root / "data" / "fixtures",
        "predictions": root / "data" / "predictions",
        "models": root / "models",
        "reports": root / "reports",
        "docs": root / "docs",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

WORLD_CUP_TOURNAMENT_NAME = "FIFA World Cup"
WORLD_CUP_YEAR = 2026
