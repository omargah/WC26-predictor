# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths
from src.models.poisson_dc import load_model, predict_dataframe


def main() -> None:
    paths = get_paths()

    model_path = paths["models"] / "poisson_dc_base.joblib"
    pending_path = paths["features"] / "modeling_dataset_pending.parquet"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Ejecuta primero: python scripts/train_phase03_poisson_dc.py"
        )

    if not pending_path.exists():
        raise FileNotFoundError(
            f"No existe {pending_path}. Ejecuta primero: python scripts/build_features.py"
        )

    print()
    print("=" * 80)
    print("FASE 3 — PREDICCIÓN DE PARTIDOS PENDIENTES")
    print("=" * 80)

    model = load_model(model_path)
    df_pending = pd.read_parquet(pending_path)
    df_pending["fecha"] = pd.to_datetime(df_pending["fecha"], errors="coerce")

    df_pred = predict_dataframe(model, df_pending)

    out_path = paths["predictions"] / "phase03_pending_predictions.csv"
    df_pred.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Partidos pendientes predichos: {len(df_pred):,}")
    print(f"Archivo: {out_path}")

    cols = [
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "ciudad",
        "pais_sede",
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

    cols = [c for c in cols if c in df_pred.columns]

    print()
    print("Primeras predicciones pendientes:")
    print(df_pred[cols].head(20).to_string(index=False))

    print()
    print("=" * 80)
    print("PREDICCIÓN PENDIENTES COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
