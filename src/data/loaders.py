# -*- coding: utf-8 -*-
"""
Descarga y verificación del dataset histórico.

Este módulo distingue dos fechas importantes:

1. fecha_maxima_dataset_completo:
   Última fecha que aparece en el CSV, incluyendo partidos futuros del fixture.

2. fecha_maxima_partido_jugado:
   Última fecha con marcador real disponible.

La segunda es la que importa para saber si los resultados reales están
actualizados conforme avanza el Mundial.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import HISTORICAL_RESULTS_URL, WORLD_CUP_TOURNAMENT_NAME, get_paths


def download_historical_results(
    project_root: str | Path | None = None,
    force_refresh: bool = False,
) -> Path:
    """
    Descarga results.csv y lo guarda en data/raw/.
    """

    paths = get_paths(project_root)

    raw_path = paths["raw"] / "international_results.csv"
    meta_path = paths["raw"] / "international_results_meta.json"

    if raw_path.exists() and not force_refresh:
        return raw_path

    print("[loaders] Descargando dataset desde GitHub...")
    print(f"[loaders] URL: {HISTORICAL_RESULTS_URL}")

    urllib.request.urlretrieve(HISTORICAL_RESULTS_URL, raw_path)

    meta = {
        "source_url": HISTORICAL_RESULTS_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_path),
    }

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[loaders] Guardado en: {raw_path}")

    return raw_path


def load_raw_results(
    project_root: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Carga el CSV crudo de resultados internacionales.
    """

    raw_path = download_historical_results(
        project_root=project_root,
        force_refresh=force_refresh,
    )

    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def add_is_played_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columna is_played usando home_score y away_score.
    """

    df = df.copy()

    df["is_played"] = (
        df["home_score"].notna()
        &
        df["away_score"].notna()
    )

    return df


def load_world_cup_2026_fixtures(
    project_root: str | Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Filtra solo partidos del Mundial 2026.
    """

    df = load_raw_results(
        project_root=project_root,
        force_refresh=force_refresh,
    )

    mask = (
        (df["tournament"] == WORLD_CUP_TOURNAMENT_NAME)
        &
        (df["date"] >= "2026-01-01")
    )

    df_wc = df.loc[mask].sort_values("date").reset_index(drop=True)
    df_wc = add_is_played_column(df_wc)

    return df_wc


def get_data_freshness_report(
    project_root: str | Path | None = None,
    force_refresh: bool = False,
) -> dict:
    """
    Reporte para verificar si el dataset está actualizado.

    La fecha importante para resultados reales es:
        fecha_maxima_partido_jugado

    La fecha máxima del dataset completo puede ser futura porque el CSV
    también trae fixture.
    """

    df = load_raw_results(
        project_root=project_root,
        force_refresh=force_refresh,
    )

    df = add_is_played_column(df)

    df_played = df[df["is_played"]].copy()

    df_wc = load_world_cup_2026_fixtures(
        project_root=project_root,
        force_refresh=False,
    )

    df_wc_played = df_wc[df_wc["is_played"]].copy()
    df_wc_pending = df_wc[~df_wc["is_played"]].copy()

    if len(df_wc_pending) > 0:
        proximo_partido_mundial_2026 = str(df_wc_pending["date"].min().date())
    else:
        proximo_partido_mundial_2026 = None

    report = {
        "fecha_minima_dataset_completo": str(df["date"].min().date()),
        "fecha_maxima_dataset_completo_incluye_fixture": str(df["date"].max().date()),
        "fecha_maxima_partido_jugado": str(df_played["date"].max().date()),
        "total_partidos_historicos": int(len(df)),
        "total_partidos_jugados": int(df["is_played"].sum()),
        "total_partidos_pendientes": int((~df["is_played"]).sum()),
        "columnas_dataset": list(df.columns.drop("is_played")),
        "partidos_mundial_2026_total": int(len(df_wc)),
        "partidos_mundial_2026_jugados": int(df_wc["is_played"].sum()),
        "partidos_mundial_2026_pendientes": int((~df_wc["is_played"]).sum()),
        "fecha_maxima_mundial_2026_jugado": (
            str(df_wc_played["date"].max().date())
            if len(df_wc_played) > 0
            else None
        ),
        "proximo_partido_mundial_2026": proximo_partido_mundial_2026,
    }

    return report


if __name__ == "__main__":
    report = get_data_freshness_report(force_refresh=True)

    print()
    print("=" * 80)
    print("REPORTE DE FRESCURA DEL DATASET")
    print("=" * 80)

    for key, value in report.items():
        print(f"{key}: {value}")
