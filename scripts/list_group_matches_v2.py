# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths
from src.simulation.fixtures import load_worldcup_group_fixture_v2, save_group_fixture_audit


def main() -> None:
    paths = get_paths()

    df_wc = load_worldcup_group_fixture_v2(paths)
    out_paths = save_group_fixture_audit(df_wc, paths)

    print()
    print("=" * 90)
    print("FASE 5 V2 — AUDITORÍA DE PARTIDOS DE GRUPO")
    print("=" * 90)

    print(f"Total partidos grupo: {len(df_wc)}")
    print(f"Jugados:              {int(df_wc['is_played'].sum())}")
    print(f"Pendientes:           {int((~df_wc['is_played']).sum())}")
    print(f"Con predicción:       {int(df_wc['has_model_prediction'].sum())}")

    print()
    print("-" * 90)
    print("PARTIDOS YA JUGADOS")
    print("-" * 90)

    cols_played = [
        "group",
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "goles_local",
        "goles_visitante",
        "prediction_source",
        "lambda_local",
        "lambda_visitante",
    ]

    cols_played = [c for c in cols_played if c in df_wc.columns]

    print(
        df_wc[df_wc["is_played"]][cols_played]
        .sort_values(["fecha", "group"])
        .to_string(index=False)
    )

    print()
    print("-" * 90)
    print("PARTIDOS PENDIENTES")
    print("-" * 90)

    cols_pending = [
        "group",
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "ciudad",
        "pais_sede",
        "neutral",
        "prediction_source",
        "lambda_local",
        "lambda_visitante",
        "prob_local",
        "prob_empate",
        "prob_visitante",
    ]

    cols_pending = [c for c in cols_pending if c in df_wc.columns]

    print(
        df_wc[~df_wc["is_played"]][cols_pending]
        .sort_values(["fecha", "group"])
        .to_string(index=False)
    )

    print()
    print("-" * 90)
    print("ARCHIVOS GENERADOS")
    print("-" * 90)

    for k, v in out_paths.items():
        print(f"{k}: {v}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
