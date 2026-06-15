
# -*- coding: utf-8 -*-
"""
Limpieza y estandarización de partidos históricos.

Este módulo convierte datasets de diferentes fuentes a un esquema canónico.

Esquema canónico mínimo:
    date
    home_team
    away_team
    home_goals
    away_goals
    tournament
    neutral
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.aliases import apply_team_aliases


COLUMN_CANDIDATES = {
    "date": [
        "date", "fecha", "match_date", "utcDate", "utc_date", "Date", "Fecha"
    ],
    "home_team": [
        "home_team", "equipo_local", "home", "local", "HomeTeam",
        "homeTeam", "team_home", "Equipo_Local"
    ],
    "away_team": [
        "away_team", "equipo_visitante", "away", "visitante", "AwayTeam",
        "awayTeam", "team_away", "Equipo_Visitante"
    ],
    "home_goals": [
        "home_score", "goles_local", "home_goals", "score_home",
        "HomeGoals", "FTHG", "goles_local_ft"
    ],
    "away_goals": [
        "away_score", "goles_visitante", "away_goals", "score_away",
        "AwayGoals", "FTAG", "goles_visitante_ft"
    ],
    "tournament": [
        "tournament", "competicion", "competition", "torneo", "Competicion",
        "league", "cup", "event"
    ],
    "neutral": [
        "neutral", "es_neutral", "is_neutral", "Neutral"
    ],
    "city": [
        "city", "ciudad", "venue_city", "sede_ciudad"
    ],
    "country": [
        "country", "pais", "venue_country", "sede_pais"
    ],
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Busca la primera columna existente de una lista de candidatos.
    """

    existing = set(df.columns)

    for col in candidates:
        if col in existing:
            return col

    lower_map = {str(col).lower(): col for col in df.columns}

    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    return None


def standardize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas de distintas fuentes al esquema canónico.
    """

    df_out = df.copy()
    rename_map = {}

    for canonical, candidates in COLUMN_CANDIDATES.items():
        found = _find_column(df_out, candidates)

        if found is not None:
            rename_map[found] = canonical

    df_out = df_out.rename(columns=rename_map)

    return df_out


def ensure_required_columns(df: pd.DataFrame) -> None:
    """
    Verifica que existan las columnas mínimas necesarias.
    """

    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Faltan columnas mínimas después de estandarizar: "
            + ", ".join(missing)
            + ". Revisa los nombres de columnas del archivo raw."
        )


def coerce_match_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte fechas, goles y neutralidad a tipos consistentes.
    """

    df_out = df.copy()

    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce")

    df_out["home_goals"] = pd.to_numeric(df_out["home_goals"], errors="coerce")
    df_out["away_goals"] = pd.to_numeric(df_out["away_goals"], errors="coerce")

    if "neutral" not in df_out.columns:
        df_out["neutral"] = 0
    else:
        df_out["neutral"] = (
            df_out["neutral"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": 1,
                "1": 1,
                "yes": 1,
                "si": 1,
                "sí": 1,
                "neutral": 1,
                "false": 0,
                "0": 0,
                "no": 0,
            })
            .fillna(0)
            .astype(int)
        )

    if "tournament" not in df_out.columns:
        df_out["tournament"] = "Unknown"

    if "city" not in df_out.columns:
        df_out["city"] = np.nan

    if "country" not in df_out.columns:
        df_out["country"] = np.nan

    return df_out


def drop_invalid_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina registros que no sirven para entrenamiento.
    """

    df_out = df.copy()

    df_out = df_out.dropna(
        subset=["date", "home_team", "away_team", "home_goals", "away_goals"]
    )

    df_out = df_out[df_out["home_team"].astype(str).str.strip() != ""]
    df_out = df_out[df_out["away_team"].astype(str).str.strip() != ""]
    df_out = df_out[df_out["home_team"] != df_out["away_team"]]

    return df_out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados básicos generados al mezclar o exportar fuentes.
    """

    df_out = df.copy()

    subset = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    df_out = df_out.drop_duplicates(subset=subset)

    return df_out


def add_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas del marcador.

    result_1x2:
        0 = gana local
        1 = empate
        2 = gana visitante
    """

    df_out = df.copy()

    df_out["home_goals"] = df_out["home_goals"].astype(int)
    df_out["away_goals"] = df_out["away_goals"].astype(int)

    df_out["total_goals"] = df_out["home_goals"] + df_out["away_goals"]
    df_out["goal_diff"] = df_out["home_goals"] - df_out["away_goals"]

    df_out["home_win"] = (df_out["home_goals"] > df_out["away_goals"]).astype(int)
    df_out["draw"] = (df_out["home_goals"] == df_out["away_goals"]).astype(int)
    df_out["away_win"] = (df_out["home_goals"] < df_out["away_goals"]).astype(int)

    df_out["result_1x2"] = np.select(
        [
            df_out["home_goals"] > df_out["away_goals"],
            df_out["home_goals"] == df_out["away_goals"],
            df_out["home_goals"] < df_out["away_goals"],
        ],
        [0, 1, 2],
        default=np.nan,
    ).astype(int)

    return df_out


def clean_matches(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de limpieza de partidos.
    """

    df = standardize_match_columns(df_raw)
    ensure_required_columns(df)

    df = coerce_match_types(df)
    df = apply_team_aliases(df, home_col="home_team", away_col="away_team")

    df = drop_invalid_matches(df)
    df = remove_duplicates(df)
    df = add_result_columns(df)

    keep_cols = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_diff",
        "result_1x2",
        "home_win",
        "draw",
        "away_win",
        "tournament",
        "neutral",
        "city",
        "country",
    ]

    available_cols = [col for col in keep_cols if col in df.columns]
    df = df[available_cols].copy()

    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_clean_matches(df: pd.DataFrame, output_dir: str | Path) -> dict:
    """
    Guarda el dataset limpio en Parquet y CSV.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "matches_clean.parquet"
    csv_path = output_dir / "matches_clean.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    return {
        "parquet": parquet_path,
        "csv": csv_path,
    }
