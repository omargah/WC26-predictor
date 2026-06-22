# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from src.config import get_paths
from src.data.features import build_phase02_features


def main() -> None:
    paths = get_paths()

    print()
    print("=" * 80)
    print("FASE 2 — FEATURES SIN LEAKAGE")
    print("=" * 80)

    result = build_phase02_features(
        matches_clean_path=paths["processed"] / "matches_clean.parquet",
        output_dir=paths["features"],
        reports_dir=paths["reports"],
        start_year=2010,
    )

    report = result["report"]
    df_all = result["df_all"]
    df_train = result["df_train"]
    df_pending = result["df_pending"]

    print()
    print("[1] Dataset de features construido")
    print(f"Filas totales desde 2010: {len(df_all):,}")
    print(f"Filas de entrenamiento jugadas: {len(df_train):,}")
    print(f"Filas pendientes para predicción: {len(df_pending):,}")
    print(f"Rango total: {df_all['fecha'].min()} → {df_all['fecha'].max()}")
    print(f"Rango entrenamiento: {df_train['fecha'].min()} → {df_train['fecha'].max()}")

    print()
    print("[2] Verificación anti-leakage")
    for key, value in report["leakage_report"].items():
        print(f"{key}: {value}")

    print()
    print("[3] Archivos generados")
    for key, value in report["outputs"].items():
        print(f"{key}: {value}")

    print()
    print("[4] Próximos partidos pendientes con features")
    show_cols = [
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "torneo",
        "ciudad",
        "pais_sede",
        "neutral",
        "elo_local_pre",
        "elo_visitante_pre",
        "diff_elo_pre",
        "goals_for_avg_5_L",
        "goals_for_avg_5_V",
        "goals_against_avg_5_L",
        "goals_against_avg_5_V",
        "h2h_matches",
    ]

    show_cols = [c for c in show_cols if c in df_pending.columns]

    if len(df_pending) > 0:
        print(df_pending[show_cols].head(12).to_string(index=False))
    else:
        print("No hay partidos pendientes.")

    print()
    print("=" * 80)
    print("FASE 2 COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
