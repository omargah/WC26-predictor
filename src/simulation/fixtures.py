# -*- coding: utf-8 -*-
"""
src/simulation/fixtures.py

Carga y auditoría del fixture de fase de grupos.

Objetivo:
    - listar partidos jugados;
    - listar partidos pendientes;
    - agregar grupo;
    - agregar lambdas pre-partido para partidos jugados y pendientes.

Para partidos jugados del Mundial 2026:
    Se intenta usar phase03_validation_predictions.csv, porque eso representa
    una predicción pre-partido dentro del split temporal.

Para partidos pendientes:
    Se usa phase03_pending_predictions.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_paths
from src.simulation.groups import TEAM_TO_GROUP


PREDICTION_COLUMNS = [
    "match_id",
    "lambda_local",
    "lambda_visitante",
    "prob_local",
    "prob_empate",
    "prob_visitante",
    "prob_over_2_5",
    "prob_btts_si",
    "marcador_mas_probable",
    "prob_marcador_mas_probable",
]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df


def load_worldcup_group_fixture_v2(paths: dict | None = None) -> pd.DataFrame:
    """
    Carga todos los partidos de fase de grupos del Mundial 2026.

    Devuelve un DataFrame con:
        - datos reales del fixture;
        - grupo;
        - estado jugado/pendiente;
        - predicciones disponibles.
    """

    if paths is None:
        paths = get_paths()

    all_path = paths["features"] / "modeling_dataset_all.parquet"

    if not all_path.exists():
        raise FileNotFoundError(
            f"No existe {all_path}. Ejecuta primero Fase 2."
        )

    df_all = pd.read_parquet(all_path)
    df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce")

    df_wc = df_all[
        (df_all["torneo"] == "FIFA World Cup")
        &
        (df_all["fecha"] >= "2026-01-01")
    ].copy()

    df_wc = df_wc.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    df_wc["group"] = df_wc["equipo_local"].map(TEAM_TO_GROUP)

    missing_group = df_wc[df_wc["group"].isna()].copy()

    if len(missing_group) > 0:
        missing_teams = sorted(
            set(missing_group["equipo_local"].dropna())
            |
            set(missing_group["equipo_visitante"].dropna())
        )

        raise ValueError(
            "Hay equipos del fixture sin grupo definido: "
            + ", ".join(missing_teams)
        )

    val_path = paths["predictions"] / "phase03_validation_predictions.csv"
    pending_path = paths["predictions"] / "phase03_pending_predictions.csv"

    df_val = _read_csv_if_exists(val_path)
    df_pending = _read_csv_if_exists(pending_path)

    pred_frames = []

    if not df_val.empty:
        keep = [c for c in PREDICTION_COLUMNS if c in df_val.columns]
        tmp = df_val[keep].copy()
        tmp["prediction_source"] = "validation_pre_match"
        pred_frames.append(tmp)

    if not df_pending.empty:
        keep = [c for c in PREDICTION_COLUMNS if c in df_pending.columns]
        tmp = df_pending[keep].copy()
        tmp["prediction_source"] = "pending_model_prediction"
        pred_frames.append(tmp)

    if pred_frames:
        df_pred = pd.concat(pred_frames, ignore_index=True)
        df_pred = df_pred.drop_duplicates(subset=["match_id"], keep="first")
    else:
        df_pred = pd.DataFrame(columns=["match_id", "prediction_source"])

    df_wc = df_wc.merge(
        df_pred,
        on="match_id",
        how="left",
        suffixes=("", "_pred"),
    )

    df_wc["fixture_status"] = np.where(
        df_wc["is_played"],
        "played",
        "pending",
    )

    df_wc["has_model_prediction"] = (
        df_wc["lambda_local"].notna()
        &
        df_wc["lambda_visitante"].notna()
    )

    return df_wc


def split_group_fixture(df_wc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Separa partidos jugados y pendientes.
    """

    played = df_wc[df_wc["is_played"]].copy()
    pending = df_wc[~df_wc["is_played"]].copy()

    return {
        "played": played,
        "pending": pending,
        "all": df_wc.copy(),
    }


def save_group_fixture_audit(
    df_wc: pd.DataFrame,
    paths: dict | None = None,
) -> dict[str, Path]:
    """
    Guarda auditorías de partidos de grupo.
    """

    if paths is None:
        paths = get_paths()

    reports = paths["reports"]
    predictions = paths["predictions"]

    reports.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)

    all_path = reports / "phase05_v2_group_fixture_all.csv"
    played_path = reports / "phase05_v2_group_fixture_played.csv"
    pending_path = reports / "phase05_v2_group_fixture_pending.csv"

    df_wc.to_csv(all_path, index=False, encoding="utf-8")
    df_wc[df_wc["is_played"]].to_csv(played_path, index=False, encoding="utf-8")
    df_wc[~df_wc["is_played"]].to_csv(pending_path, index=False, encoding="utf-8")

    return {
        "all": all_path,
        "played": played_path,
        "pending": pending_path,
    }
