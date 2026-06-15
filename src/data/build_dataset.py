
# -*- coding: utf-8 -*-
"""
Construcción del dataset limpio de partidos.

Este módulo conecta:
    1. carga de datos crudos;
    2. limpieza y estandarización;
    3. guardado de datos procesados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.data.load_raw import load_raw_matches, inspect_raw_schema
from src.data.clean_matches import clean_matches, save_clean_matches


def build_clean_matches_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    filename: Optional[str] = None,
) -> dict:
    """
    Construye el dataset limpio de partidos históricos.
    """

    df_raw, raw_path = load_raw_matches(raw_dir, filename=filename)
    raw_schema = inspect_raw_schema(df_raw)

    df_clean = clean_matches(df_raw)
    output_paths = save_clean_matches(df_clean, processed_dir)

    summary = {
        "raw_path": raw_path,
        "raw_rows": raw_schema["n_rows"],
        "raw_columns": raw_schema["n_columns"],
        "clean_rows": len(df_clean),
        "clean_columns": len(df_clean.columns),
        "output_paths": output_paths,
        "date_min": df_clean["date"].min(),
        "date_max": df_clean["date"].max(),
        "n_teams": len(set(df_clean["home_team"]) | set(df_clean["away_team"])),
        "n_tournaments": df_clean["tournament"].nunique()
        if "tournament" in df_clean.columns
        else None,
    }

    return summary
