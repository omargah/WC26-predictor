# -*- coding: utf-8 -*-
"""
src/config.py

Configuración central del proyecto.

Este archivo evita repetir rutas en todos lados. La raíz del proyecto se
resuelve así:

1. Si existe la variable de entorno MUNDIAL_2026_ROOT, se usa esa ruta.
2. Si no existe, se usa la carpeta raíz del repo.

Fuente principal:
    https://raw.githubusercontent.com/martj42/international_results/master/results.csv

La razón de usar esta fuente es evitar depender del snapshot de KaggleHub
que tiene "2017" en el nombre. Aquí la fecha real se verificará con
date.max() cada vez que corramos el reporte de frescura.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Devuelve la raíz del proyecto.
    """

    env_root = os.environ.get("MUNDIAL_2026_ROOT")

    if env_root:
        return Path(env_root).expanduser().resolve()

    # src/config.py -> src/ -> raíz del repo
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = get_project_root()


def get_paths(project_root: str | Path | None = None) -> dict:
    """
    Devuelve todas las rutas usadas por el proyecto y las crea si no existen.
    """

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
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


# ============================================================
# Fuente viva de resultados internacionales
# ============================================================

HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

WORLD_CUP_TOURNAMENT_NAME = "FIFA World Cup"
WORLD_CUP_YEAR = 2026