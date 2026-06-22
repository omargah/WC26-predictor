# -*- coding: utf-8 -*-
"""
src/simulation/match_engine.py

Motor de simulación de partidos.

Modos de uso para partidos de grupo:

    played_policy="fixed"
        Si el partido ya se jugó, usa el resultado real.
        Si está pendiente, lo simula con lambdas del modelo.

    played_policy="resimulate"
        Simula todos los partidos, incluso los ya jugados.
        Para partidos ya jugados usa preferentemente predicción de validación
        temporal, no el resultado real.

    played_policy="evaluate"
        Para partidos ya jugados, simula muchas veces contra su predicción
        pre-partido y permite comparar contra el resultado real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.poisson_dc import score_matrix


VALID_PLAYED_POLICIES = {
    "fixed",
    "resimulate",
    "evaluate",
}


def sample_score_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    rng: np.random.Generator,
    max_goals: int = 10,
) -> tuple[int, int]:
    """
    Simula marcador exacto usando matriz Poisson-Dixon-Coles.
    """

    mat = score_matrix(
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        max_goals=max_goals,
        rho=-0.075,
        use_dixon_coles=True,
    )

    flat_index = rng.choice(mat.size, p=mat.ravel())
    gh, ga = np.unravel_index(flat_index, mat.shape)

    return int(gh), int(ga)


def result_label(gh: int, ga: int) -> str:
    if gh > ga:
        return "L"
    if gh < ga:
        return "V"
    return "E"


def simulate_group_match_row(
    row: pd.Series,
    rng: np.random.Generator,
    played_policy: str = "fixed",
) -> dict:
    """
    Simula o fija un partido de fase de grupos según la política elegida.
    """

    if played_policy not in VALID_PLAYED_POLICIES:
        raise ValueError(
            f"played_policy inválida: {played_policy}. "
            f"Usa una de: {sorted(VALID_PLAYED_POLICIES)}"
        )

    is_played = bool(row["is_played"])

    home = row["equipo_local"]
    away = row["equipo_visitante"]

    if is_played and played_policy == "fixed":
        gh = int(row["goles_local"])
        ga = int(row["goles_visitante"])
        simulation_source = "real_fixed_result"

    else:
        if pd.isna(row.get("lambda_local")) or pd.isna(row.get("lambda_visitante")):
            raise ValueError(
                f"No hay lambdas para simular {home} vs {away}. "
                f"prediction_source={row.get('prediction_source')}"
            )

        gh, ga = sample_score_from_lambdas(
            lambda_home=float(row["lambda_local"]),
            lambda_away=float(row["lambda_visitante"]),
            rng=rng,
        )

        if is_played:
            simulation_source = "simulated_played_match_counterfactual"
        else:
            simulation_source = "simulated_pending_match"

    return {
        "match_id": row["match_id"],
        "group": row.get("group"),
        "fecha": str(pd.to_datetime(row["fecha"]).date()),
        "home": home,
        "away": away,
        "home_goals": gh,
        "away_goals": ga,
        "result_1x2": result_label(gh, ga),
        "is_played_real_life": is_played,
        "simulation_source": simulation_source,
        "prediction_source": row.get("prediction_source"),
        "lambda_home": None if pd.isna(row.get("lambda_local")) else float(row.get("lambda_local")),
        "lambda_away": None if pd.isna(row.get("lambda_visitante")) else float(row.get("lambda_visitante")),
        "real_home_goals": None if pd.isna(row.get("goles_local")) else float(row.get("goles_local")),
        "real_away_goals": None if pd.isna(row.get("goles_visitante")) else float(row.get("goles_visitante")),
    }


def simulate_group_fixture_once(
    df_wc: pd.DataFrame,
    rng: np.random.Generator,
    played_policy: str = "fixed",
) -> pd.DataFrame:
    """
    Simula una vez todos los partidos de grupo.
    """

    rows = []

    for _, row in df_wc.sort_values(["fecha", "match_id"]).iterrows():
        rows.append(
            simulate_group_match_row(
                row=row,
                rng=rng,
                played_policy=played_policy,
            )
        )

    return pd.DataFrame(rows)


def simulate_single_match_many(
    row: pd.Series,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simula muchas veces un partido, aunque ya se haya jugado.

    Útil para evaluación histórica o contrafactual.
    """

    rng = np.random.default_rng(seed)

    rows = []

    for i in range(n):
        sim = simulate_group_match_row(
            row=row,
            rng=rng,
            played_policy="resimulate",
        )
        sim["simulation_id"] = i + 1
        rows.append(sim)

    return pd.DataFrame(rows)
