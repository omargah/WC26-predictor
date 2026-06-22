# -*- coding: utf-8 -*-
"""
Limpieza y normalización del dataset.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def assign_competition_weight(tournament) -> float:
    if pd.isna(tournament):
        return 0.70

    t = str(tournament).lower()

    if t in ("fifa world cup", "world cup"):
        return 1.00
    if "world cup qualification" in t or "world cup qualifier" in t:
        return 0.85
    if "uefa euro" in t or "european championship" in t:
        return 0.95
    if "copa américa" in t or "copa america" in t:
        return 0.90
    if "nations league" in t:
        return 0.80
    if "gold cup" in t:
        return 0.80
    if "african cup" in t or "africa cup" in t or "afcon" in t:
        return 0.80
    if "asian cup" in t:
        return 0.75
    if "friendly" in t:
        return 0.60

    return 0.70


def _make_match_id(row: pd.Series) -> str:
    raw = f"{row['fecha']}|{row['equipo_local']}|{row['equipo_visitante']}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _result_1x2(goles_local, goles_visitante):
    if pd.isna(goles_local) or pd.isna(goles_visitante):
        return np.nan
    if goles_local > goles_visitante:
        return "L"
    if goles_local < goles_visitante:
        return "V"
    return "E"


def build_clean_matches(
    df_raw: pd.DataFrame,
    source_id: str = "international_results_martj42",
) -> pd.DataFrame:
    df = df_raw.copy()

    required = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError("Faltan columnas en el CSV crudo: " + ", ".join(missing))

    df = df.rename(
        columns={
            "date": "fecha",
            "home_team": "equipo_local",
            "away_team": "equipo_visitante",
            "home_score": "goles_local",
            "away_score": "goles_visitante",
            "tournament": "torneo",
            "city": "ciudad",
            "country": "pais_sede",
        }
    )

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df[df["fecha"].notna()].copy()

    df["neutral"] = (
        df["neutral"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            }
        )
        .fillna(0)
        .astype(int)
    )

    df["goles_local"] = pd.to_numeric(df["goles_local"], errors="coerce")
    df["goles_visitante"] = pd.to_numeric(df["goles_visitante"], errors="coerce")

    df["is_played"] = (
        df["goles_local"].notna()
        &
        df["goles_visitante"].notna()
    )

    total_goals = df["goles_local"] + df["goles_visitante"]

    df["over_2_5"] = np.where(
        df["is_played"],
        (total_goals >= 3).astype(float),
        np.nan,
    )

    df["btts"] = np.where(
        df["is_played"],
        (
            (df["goles_local"] > 0)
            &
            (df["goles_visitante"] > 0)
        ).astype(float),
        np.nan,
    )

    df["resultado_1x2"] = df.apply(
        lambda r: _result_1x2(r["goles_local"], r["goles_visitante"]),
        axis=1,
    )

    df["peso_competicion"] = df["torneo"].apply(assign_competition_weight)
    df["source_id"] = source_id
    df["match_id"] = df.apply(_make_match_id, axis=1)

    keep_cols = [
        "match_id",
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "goles_local",
        "goles_visitante",
        "resultado_1x2",
        "over_2_5",
        "btts",
        "torneo",
        "ciudad",
        "pais_sede",
        "neutral",
        "peso_competicion",
        "source_id",
        "is_played",
    ]

    df = df[keep_cols].sort_values("fecha").reset_index(drop=True)

    before = len(df)

    df = df.drop_duplicates(
        subset=["fecha", "equipo_local", "equipo_visitante"]
    ).reset_index(drop=True)

    removed = before - len(df)

    if removed > 0:
        print(f"[cleaners] Se removieron {removed} duplicados exactos.")

    return df


def validate_clean_matches(df: pd.DataFrame) -> list[str]:
    problems = []

    if df["fecha"].isna().any():
        problems.append("Hay fechas nulas después de la limpieza.")

    played = df[df["is_played"]].copy()

    if (played["goles_local"] < 0).any():
        problems.append("Hay goles locales negativos.")

    if (played["goles_visitante"] < 0).any():
        problems.append("Hay goles visitantes negativos.")

    duplicated_ids = df.duplicated(subset=["match_id"]).sum()

    if duplicated_ids > 0:
        problems.append(f"Hay {duplicated_ids} match_id duplicados.")

    return problems
