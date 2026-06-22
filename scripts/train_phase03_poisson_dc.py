# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd

from src.config import get_paths
from src.models.poisson_dc import save_model, train_evaluate_and_refit


def main() -> None:
    paths = get_paths()

    features_path = paths["features"] / "modeling_dataset_train.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"No existe {features_path}. Ejecuta primero la Fase 2: python scripts/build_features.py"
        )

    print()
    print("=" * 80)
    print("FASE 3 — ENTRENAMIENTO POISSON + DIXON-COLES")
    print("=" * 80)

    print()
    print("[1] Cargando dataset de entrenamiento...")
    df_train_all = pd.read_parquet(features_path)
    df_train_all["fecha"] = pd.to_datetime(df_train_all["fecha"], errors="coerce")

    print(f"Filas disponibles: {len(df_train_all):,}")
    print(f"Rango: {df_train_all['fecha'].min()} → {df_train_all['fecha'].max()}")

    print()
    print("[2] Entrenando y evaluando con split temporal...")
    result = train_evaluate_and_refit(
        df_train_all=df_train_all,
        test_ratio=0.20,
        min_feature_coverage=0.30,
        alpha=0.001,
    )

    eval_model = result["eval_model"]
    final_model = result["final_model"]
    df_val_pred = result["validation_predictions"]
    metrics = result["metrics"]
    metadata = result["metadata"]

    print()
    print("[3] Métricas de validación")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()
    print("[4] Guardando modelo y reportes...")

    model_final_path = paths["models"] / "poisson_dc_base.joblib"
    model_eval_path = paths["models"] / "poisson_dc_eval.joblib"

    save_model(final_model, model_final_path)
    save_model(eval_model, model_eval_path)

    pred_path = paths["predictions"] / "phase03_validation_predictions.csv"
    df_val_pred.to_csv(pred_path, index=False, encoding="utf-8")

    metrics_path = paths["reports"] / "phase03_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path = paths["reports"] / "phase03_report.json"
    report_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    feature_path = paths["reports"] / "phase03_features_used.txt"
    feature_path.write_text(
        "\n".join(result["features"]) + "\n",
        encoding="utf-8",
    )

    print(f"Modelo final:       {model_final_path}")
    print(f"Modelo evaluación:  {model_eval_path}")
    print(f"Predicciones val:   {pred_path}")
    print(f"Métricas JSON:      {metrics_path}")
    print(f"Reporte JSON:       {report_path}")
    print(f"Features usadas:    {feature_path}")

    print()
    print("=" * 80)
    print("FASE 3 ENTRENAMIENTO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
