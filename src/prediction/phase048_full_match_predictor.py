
# -*- coding: utf-8 -*-
"""
FASE 4.8 — Predictor completo de partido.

Combina:
    - Fase 4: goles Poisson + Dixon-Coles.
    - Fase 4.7: córners/tarjetas.

Modos de córners/tarjetas:
    legacy:
        reproduce valores benchmark del primer Colab cuando existan.

    joblib:
        usa models/corners_cards_poisson_stable.joblib.

    auto:
        intenta joblib y si falla usa benchmark legacy.
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.prediction.phase04_predict_match import predecir_partido_manual
from src.models.phase047_corners_cards_model import (
    predecir_corners_tarjetas_experimental,
    imprimir_corners_tarjetas,
)


def predecir_partido_completo(
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    torneo: str = "FIFA World Cup",
    fase: str = "Group Stage",
    ciudad: str = "TBD",
    estadio: str = "TBD",
    pais_sede: str = "TBD",
    neutral: int = 1,
    project_root: str | Path = ".",
    verbose: bool = True,
    save: bool = True,
    candidate_dates: list[str] | None = None,
    corners_cards_mode: str = "legacy",
) -> dict:
    """
    Predicción completa:
        goles + córners + tarjetas.

    corners_cards_mode:
        "legacy", "joblib" o "auto".
    """

    project_root = Path(project_root)

    corners_cards_mode = str(corners_cards_mode).lower().strip()

    if corners_cards_mode not in {"legacy", "joblib", "auto"}:
        raise ValueError(
            "corners_cards_mode debe ser 'legacy', 'joblib' o 'auto'."
        )

    goles = predecir_partido_manual(
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        torneo=torneo,
        fase=fase,
        ciudad=ciudad,
        estadio=estadio,
        pais_sede=pais_sede,
        neutral=neutral,
        project_root=project_root,
        verbose=False,
        save=False,
        candidate_dates=candidate_dates,
    )

    if corners_cards_mode == "legacy":
        prefer_joblib = False
    else:
        prefer_joblib = True

    corners_cards = predecir_corners_tarjetas_experimental(
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        project_root=project_root,
        candidate_dates=candidate_dates,
        prefer_joblib=prefer_joblib,
    )

    if corners_cards_mode == "joblib" and corners_cards.get("source") == "legacy_benchmark_compatible":
        corners_cards = {
            "available": False,
            "source": "joblib_requested_but_fallback_returned",
            "reason": "Se pidió modo joblib, pero el módulo cayó a benchmark legacy.",
        }

    resultado = {
        **goles,
        "corners_cards_mode": corners_cards_mode,
        "corners_cards_available": bool(corners_cards.get("available", False)),
        "corners_cards_source": corners_cards.get("source"),
        "corners_cards_reason": corners_cards.get("reason"),
    }

    if corners_cards.get("available", False):
        resultado.update({
            "corners_home": corners_cards["corners_home"],
            "corners_away": corners_cards["corners_away"],
            "corners_total": corners_cards["corners_total"],
            "cards_home": corners_cards["cards_home"],
            "cards_away": corners_cards["cards_away"],
            "cards_total": corners_cards["cards_total"],
            "corners_markets": corners_cards["corners_markets"],
            "cards_markets": corners_cards["cards_markets"],
        })

    if verbose:
        print("=" * 90)
        print("PREDICCIÓN COMPLETA — GOLES + CÓRNERS + TARJETAS")
        print("=" * 90)
        print(f"{equipo_local} vs {equipo_visitante}")
        print(f"Fecha: {fecha_partido}")
        print(f"Torneo: {torneo}")
        print(f"Fase: {fase}")
        print(f"Ciudad: {ciudad}")
        print(f"Estadio: {estadio}")
        print(f"País sede: {pais_sede}")
        print(f"Neutral: {neutral}")
        print(f"Modo córners/tarjetas: {corners_cards_mode}")

        print("\n------------------------------------------------------------------------------------------")
        print("GOLES")
        print("------------------------------------------------------------------------------------------")
        print(f"Goles esperados {equipo_local}: {goles['lambda_home']:.3f}")
        print(f"Goles esperados {equipo_visitante}: {goles['lambda_away']:.3f}")
        print(f"Total goles esperado: {goles['lambda_total']:.3f}")

        print("\nProbabilidades 1X2:")
        print(f"  Gana {equipo_local}: {goles['prob_home']:.2%}")
        print(f"  Empate: {goles['prob_draw']:.2%}")
        print(f"  Gana {equipo_visitante}: {goles['prob_away']:.2%}")

        print("\nMercados goles:")
        print(f"  Over 1.5: {goles['over_1_5']:.2%}")
        print(f"  Under 1.5: {goles['under_1_5']:.2%}")
        print(f"  Over 2.5: {goles['over_2_5']:.2%}")
        print(f"  Under 2.5: {goles['under_2_5']:.2%}")
        print(f"  Over 3.5: {goles['over_3_5']:.2%}")
        print(f"  Under 3.5: {goles['under_3_5']:.2%}")

        print("\nAmbos anotan:")
        print(f"  BTTS Sí: {goles['btts_yes']:.2%}")
        print(f"  BTTS No: {goles['btts_no']:.2%}")

        print("\nMarcador más probable:")
        print(f"  {goles['top_score']} → {goles['top_score_prob']:.2%}")

        print("\nTop marcadores:")
        for score in goles["top_10_scores"]:
            print(f"  {score['score']}: {score['prob']:.2%}")

        imprimir_corners_tarjetas(
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
            pred=corners_cards,
        )

        print("\n------------------------------------------------------------------------------------------")
        print("NOTA")
        print("------------------------------------------------------------------------------------------")
        print("Goles: modelo actualizado Poisson + Dixon-Coles.")
        print("Córners/tarjetas: depende de corners_cards_mode.")
        print("legacy = reproduce primer Colab; joblib = modelo recuperado real; auto = joblib con fallback.")

    if save:
        out_path = project_root / "data" / "predictions" / "manual_full_match_predictions_updated.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_row = resultado.copy()

        if "top_10_scores" in save_row:
            save_row["top_10_scores"] = json.dumps(save_row["top_10_scores"], ensure_ascii=False)

        if "corners_markets" in save_row:
            save_row["corners_markets"] = json.dumps(save_row["corners_markets"], ensure_ascii=False)

        if "cards_markets" in save_row:
            save_row["cards_markets"] = json.dumps(save_row["cards_markets"], ensure_ascii=False)

        df_save = pd.DataFrame([save_row])

        if out_path.exists():
            old = pd.read_csv(out_path)
            df_save = pd.concat([old, df_save], ignore_index=True)

        df_save.to_csv(out_path, index=False, encoding="utf-8")

        if verbose:
            print(f"\nPredicción guardada en: {out_path}")

    return resultado
