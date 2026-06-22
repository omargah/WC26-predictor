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

    metrics_path = paths["reports"] / "phase03_metrics.json"
    report_path = paths["reports"] / "phase03_report.json"
    val_path = paths["predictions"] / "phase03_validation_predictions.csv"
    pending_path = paths["predictions"] / "phase03_pending_predictions.csv"

    if not metrics_path.exists():
        raise FileNotFoundError("No existe phase03_metrics.json. Ejecuta primero train_phase03_poisson_dc.py")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    df_val = pd.read_csv(val_path)

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 3 — MODELO POISSON + DIXON-COLES")
    print("=" * 80)

    print()
    print("[1] Resumen del entrenamiento")
    for key in [
        "n_rows_total_played",
        "n_rows_fit",
        "n_rows_validation",
        "date_min_total",
        "date_max_total",
        "date_max_fit",
        "date_min_validation",
        "date_max_validation",
        "n_features",
    ]:
        print(f"{key}: {report.get(key)}")

    print()
    print("[2] Métricas")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()
    print("[3] Últimas 15 predicciones de validación")
    cols_val = [
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "goles_local",
        "goles_visitante",
        "lambda_local",
        "lambda_visitante",
        "prob_local",
        "prob_empate",
        "prob_visitante",
        "marcador_mas_probable",
        "pred_resultado_1x2",
        "resultado_1x2",
    ]
    cols_val = [c for c in cols_val if c in df_val.columns]
    print(df_val[cols_val].tail(15).to_string(index=False))

    if pending_path.exists():
        df_pending = pd.read_csv(pending_path)

        print()
        print("[4] Primeras 15 predicciones pendientes")
        cols_pending = [
            "fecha",
            "equipo_local",
            "equipo_visitante",
            "lambda_local",
            "lambda_visitante",
            "prob_local",
            "prob_empate",
            "prob_visitante",
            "prob_over_2_5",
            "prob_btts_si",
            "marcador_mas_probable",
        ]
        cols_pending = [c for c in cols_pending if c in df_pending.columns]
        print(df_pending[cols_pending].head(15).to_string(index=False))
    else:
        print()
        print("[4] No existe todavía phase03_pending_predictions.csv")

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 3 COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
