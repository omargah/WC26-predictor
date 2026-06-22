# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd

from src.config import get_paths


def main() -> None:
    paths = get_paths()

    all_path = paths["features"] / "modeling_dataset_all.parquet"
    train_path = paths["features"] / "modeling_dataset_train.parquet"
    pending_path = paths["features"] / "modeling_dataset_pending.parquet"
    report_path = paths["reports"] / "phase02_features_report.json"
    null_path = paths["reports"] / "phase02_feature_null_report.csv"

    if not all_path.exists():
        raise FileNotFoundError("No existe modeling_dataset_all.parquet. Ejecuta primero scripts/build_features.py")

    df_all = pd.read_parquet(all_path)
    df_train = pd.read_parquet(train_path)
    df_pending = pd.read_parquet(pending_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    null_report = pd.read_csv(null_path)

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 2 — FEATURES")
    print("=" * 80)

    print(f"Filas totales: {len(df_all):,}")
    print(f"Filas train:   {len(df_train):,}")
    print(f"Filas pending: {len(df_pending):,}")

    print()
    print("Leakage report:")
    for k, v in report["leakage_report"].items():
        print(f"  {k}: {v}")

    print()
    print("Top 20 features con más nulos en train:")
    print(null_report.head(20).to_string(index=False))

    print()
    print("Últimos 10 partidos de entrenamiento:")
    cols_train = [
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "goles_local",
        "goles_visitante",
        "resultado_1x2",
        "elo_local_pre",
        "elo_visitante_pre",
        "diff_elo_pre",
    ]
    cols_train = [c for c in cols_train if c in df_train.columns]
    print(df_train[cols_train].tail(10).to_string(index=False))

    print()
    print("Primeros 10 partidos pendientes:")
    cols_pending = [
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
    ]
    cols_pending = [c for c in cols_pending if c in df_pending.columns]
    print(df_pending[cols_pending].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
