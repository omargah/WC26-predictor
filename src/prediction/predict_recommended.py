# -*- coding: utf-8 -*-

# ============================================================
# Predictor recomendado
# ============================================================
# Goles:
#   Modelo actualizado Poisson + Dixon-Coles.
#
# Córners/tarjetas:
#   Modo legacy compatible.
#
# Motivo:
#   El joblib real de córners/tarjetas funciona, pero para los
#   partidos benchmark está usando dataset básico y faltan features
#   avanzadas. Por eso se deja como experimental.
# ============================================================

from __future__ import annotations

from pathlib import Path

from src.prediction.phase048_full_match_predictor import predecir_partido_completo


def predecir_partido_completo_recomendado(
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
) -> dict:
    """
    Predicción completa recomendada.

    Usa:
        corners_cards_mode="legacy"

    Esta es la configuración estable para reportes comparables
    con el primer Colab.
    """

    return predecir_partido_completo(
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
        verbose=verbose,
        save=save,
        candidate_dates=candidate_dates,
        corners_cards_mode="legacy",
    )
