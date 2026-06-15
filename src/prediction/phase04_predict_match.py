
# -*- coding: utf-8 -*-
"""
FASE 4 — Predictor manual estable de partido.

Este módulo usa:
    models/poisson_dc_base.joblib
    data/features/modeling_dataset.parquet

y genera una predicción Poisson + Dixon-Coles para un partido específico.

Ventajas:
    - Busca el partido por fecha exacta.
    - Tolera alias de equipos.
    - Tolera partidos invertidos.
    - Reexpresa la predicción desde la perspectiva solicitada.
    - Tiene salida parecida al primer Colab.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.models.phase03_poisson_dc import (
    predict_lambdas,
    score_matrix,
    probs_from_matrix,
    top_scorelines,
)


TEAM_ALIASES = {
    "Mexico": ["Mexico", "México"],
    "México": ["Mexico", "México"],
    "United States": ["United States", "USA", "USMNT"],
    "USA": ["United States", "USA", "USMNT"],
    "South Korea": ["South Korea", "Korea Republic"],
    "Korea Republic": ["South Korea", "Korea Republic"],
    "Czechia": ["Czechia", "Czech Republic"],
    "Czech Republic": ["Czechia", "Czech Republic"],
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina"],
    "Bosnia-Herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina"],
}


def aliases_for(team: str) -> list[str]:
    """
    Devuelve alias conocidos de una selección.
    """

    return TEAM_ALIASES.get(team, [team])


def load_resources(project_root: str | Path):
    """
    Carga modelo y dataset de features.
    """

    project_root = Path(project_root)

    model_path = project_root / "models" / "poisson_dc_base.joblib"
    modeling_path = project_root / "data" / "features" / "modeling_dataset.parquet"

    if not model_path.exists():
        raise FileNotFoundError(f"No existe {model_path}. Ejecuta Fase 3.")

    if not modeling_path.exists():
        raise FileNotFoundError(f"No existe {modeling_path}. Ejecuta Fase 2.")

    model_package = joblib.load(model_path)

    df_model = pd.read_parquet(modeling_path)
    df_model["fecha"] = pd.to_datetime(df_model["fecha"], errors="coerce")

    return model_package, df_model


def swap_home_away_row(row: pd.Series) -> pd.Series:
    """
    Invierte local y visitante en una fila del dataset de modelado.
    """

    row = row.copy()

    old_home = row.get("equipo_local")
    old_away = row.get("equipo_visitante")

    row["equipo_local"] = old_away
    row["equipo_visitante"] = old_home

    if "goles_local" in row.index and "goles_visitante" in row.index:
        tmp = row["goles_local"]
        row["goles_local"] = row["goles_visitante"]
        row["goles_visitante"] = tmp

    cols = list(row.index)

    for col in cols:
        if col.endswith("_L"):
            base = col[:-2]
            col_v = base + "_V"

            if col_v in row.index:
                tmp = row[col]
                row[col] = row[col_v]
                row[col_v] = tmp

    if "elo_local_pre" in row.index and "elo_visitante_pre" in row.index:
        tmp = row["elo_local_pre"]
        row["elo_local_pre"] = row["elo_visitante_pre"]
        row["elo_visitante_pre"] = tmp

    if "diff_elo_pre" in row.index and "elo_local_pre" in row.index and "elo_visitante_pre" in row.index:
        row["diff_elo_pre"] = row["elo_local_pre"] - row["elo_visitante_pre"]

    if "elo_prob_local_pre" in row.index:
        try:
            row["elo_prob_local_pre"] = 1.0 - float(row["elo_prob_local_pre"])
        except Exception:
            pass

    if "h2h_goals_home_avg" in row.index and "h2h_goals_away_avg" in row.index:
        tmp = row["h2h_goals_home_avg"]
        row["h2h_goals_home_avg"] = row["h2h_goals_away_avg"]
        row["h2h_goals_away_avg"] = tmp

    if "h2h_home_win_rate" in row.index:
        try:
            if pd.notna(row["h2h_home_win_rate"]):
                row["h2h_home_win_rate"] = 1.0 - float(row["h2h_home_win_rate"])
        except Exception:
            pass

    for col in list(row.index):
        if col.startswith("diff_"):
            base = col.replace("diff_", "", 1)
            col_l = base + "_L"
            col_v = base + "_V"

            if col_l in row.index and col_v in row.index:
                try:
                    row[col] = row[col_l] - row[col_v]
                except Exception:
                    pass

    return row


def find_feature_row(
    df_model: pd.DataFrame,
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    candidate_dates: list[str] | None = None,
) -> tuple[pd.Series, bool, str]:
    """
    Busca fila de features para un partido.

    Retorna:
        row,
        was_reversed,
        search_mode
    """

    df = df_model.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    home_aliases = aliases_for(equipo_local)
    away_aliases = aliases_for(equipo_visitante)

    if candidate_dates is None:
        candidate_dates = [fecha_partido]

    for fecha_str in candidate_dates:
        target_date = pd.to_datetime(fecha_str, errors="coerce")

        if pd.isna(target_date):
            continue

        same_date = df["fecha"].dt.date == target_date.date()

        direct = (
            df["equipo_local"].isin(home_aliases)
            &
            df["equipo_visitante"].isin(away_aliases)
            &
            same_date
        )

        found = df[direct].copy()

        if len(found) > 0:
            return found.sort_values("fecha").iloc[-1], False, "direct_same_date"

        reverse = (
            df["equipo_local"].isin(away_aliases)
            &
            df["equipo_visitante"].isin(home_aliases)
            &
            same_date
        )

        found = df[reverse].copy()

        if len(found) > 0:
            return swap_home_away_row(found.sort_values("fecha").iloc[-1]), True, "reverse_same_date"

    target_date = pd.to_datetime(candidate_dates[0], errors="coerce")

    direct_any = (
        df["equipo_local"].isin(home_aliases)
        &
        df["equipo_visitante"].isin(away_aliases)
    )

    found = df[direct_any].copy()

    if len(found) > 0:
        found["date_distance"] = (found["fecha"] - target_date).abs()
        return found.sort_values("date_distance").iloc[0], False, "direct_nearest_date"

    reverse_any = (
        df["equipo_local"].isin(away_aliases)
        &
        df["equipo_visitante"].isin(home_aliases)
    )

    found = df[reverse_any].copy()

    if len(found) > 0:
        found["date_distance"] = (found["fecha"] - target_date).abs()
        return swap_home_away_row(found.sort_values("date_distance").iloc[0]), True, "reverse_nearest_date"

    all_teams = sorted(set(df["equipo_local"].dropna()) | set(df["equipo_visitante"].dropna()))

    possible_home = [
        t for t in all_teams
        if any(a.lower() in t.lower() or t.lower() in a.lower() for a in home_aliases)
    ]

    possible_away = [
        t for t in all_teams
        if any(a.lower() in t.lower() or t.lower() in a.lower() for a in away_aliases)
    ]

    raise ValueError(
        "No encontré fila de features.\n"
        f"equipo_local={equipo_local}\n"
        f"equipo_visitante={equipo_visitante}\n"
        f"fecha_partido={fecha_partido}\n"
        f"home_aliases={home_aliases}\n"
        f"away_aliases={away_aliases}\n"
        f"possible_home={possible_home[:20]}\n"
        f"possible_away={possible_away[:20]}"
    )


def predict_from_row(
    model_package: dict,
    row: pd.Series,
    max_goals: int = 10,
    rho: float = -0.075,
) -> dict:
    """
    Calcula predicción a partir de una fila de features.
    """

    df_row = pd.DataFrame([row])

    lambda_home, lambda_away = predict_lambdas(model_package, df_row)

    lh = float(lambda_home[0])
    la = float(lambda_away[0])

    mat = score_matrix(
        lambda_h=lh,
        lambda_a=la,
        max_goals=max_goals,
        rho=rho,
    )

    probs = probs_from_matrix(mat)
    top_scores = top_scorelines(mat, top_n=10)

    return {
        "lambda_home": lh,
        "lambda_away": la,
        "lambda_total": lh + la,

        "prob_home": probs["prob_local"],
        "prob_draw": probs["prob_empate"],
        "prob_away": probs["prob_visitante"],

        "over_1_5": probs["over_1_5"],
        "under_1_5": probs["under_1_5"],
        "over_2_5": probs["over_2_5"],
        "under_2_5": probs["under_2_5"],
        "over_3_5": probs["over_3_5"],
        "under_3_5": probs["under_3_5"],

        "btts_yes": probs["btts_si"],
        "btts_no": probs["btts_no"],

        "top_score": probs["marcador_mas_probable"],
        "top_score_prob": probs["prob_marcador_mas_probable"],
        "top_10_scores": top_scores,
    }


def imprimir_prediccion(result: dict) -> None:
    """
    Imprime predicción con formato del primer Colab.
    """

    home = result["home_team"]
    away = result["away_team"]

    print("=" * 80)
    print("PREDICCIÓN DE PARTIDO — MODELO POISSON + DIXON-COLES")
    print("=" * 80)
    print(f"{home} vs {away}")
    print(f"Fecha: {result['fecha_partido']}")
    print(f"Torneo: {result['torneo']}")
    print(f"Fase: {result['fase']}")
    print(f"Ciudad: {result['ciudad']}")
    print(f"Estadio: {result['estadio']}")
    print(f"Neutral: {result['neutral']}")
    print(f"Modo de búsqueda de features: {result['feature_search_mode']}")

    print("\nGOLES ESPERADOS")
    print(f"  {home}: {result['lambda_home']:.3f}")
    print(f"  {away}: {result['lambda_away']:.3f}")
    print(f"  Total esperado: {result['lambda_total']:.3f}")

    print("\nPROBABILIDADES 1X2")
    print(f"  Gana {home}: {result['prob_home']:.2%}")
    print(f"  Empate: {result['prob_draw']:.2%}")
    print(f"  Gana {away}: {result['prob_away']:.2%}")

    print("\nMERCADOS DE GOLES")
    print(f"  Over 1.5: {result['over_1_5']:.2%}")
    print(f"  Under 1.5: {result['under_1_5']:.2%}")
    print(f"  Over 2.5: {result['over_2_5']:.2%}")
    print(f"  Under 2.5: {result['under_2_5']:.2%}")
    print(f"  Over 3.5: {result['over_3_5']:.2%}")
    print(f"  Under 3.5: {result['under_3_5']:.2%}")

    print("\nAMBOS ANOTAN")
    print(f"  BTTS Sí: {result['btts_yes']:.2%}")
    print(f"  BTTS No: {result['btts_no']:.2%}")

    print("\nMARCADOR MÁS PROBABLE")
    print(f"  {result['top_score']} → {result['top_score_prob']:.2%}")

    print("\nTOP 10 MARCADORES")
    for score in result["top_10_scores"]:
        print(f"  {score['score']}: {score['prob']:.2%}")


def predecir_partido_manual(
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
    save: bool = False,
    candidate_dates: list[str] | None = None,
) -> dict:
    """
    Predice partido manual.
    """

    project_root = Path(project_root)

    model_package, df_model = load_resources(project_root)

    row, was_reversed, search_mode = find_feature_row(
        df_model=df_model,
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        candidate_dates=candidate_dates,
    )

    result = predict_from_row(
        model_package=model_package,
        row=row,
        max_goals=int(model_package.get("max_goals", 10)),
        rho=float(model_package.get("rho", -0.075)),
    )

    result.update({
        "home_team": equipo_local,
        "away_team": equipo_visitante,
        "fecha_partido": fecha_partido,
        "torneo": torneo,
        "fase": fase,
        "ciudad": ciudad,
        "estadio": estadio,
        "pais_sede": pais_sede,
        "neutral": int(neutral),
        "feature_was_reversed": bool(was_reversed),
        "feature_search_mode": search_mode,
    })

    if "goles_local" in row.index and "goles_visitante" in row.index:
        result["actual_home_goals_dataset_perspective"] = int(row["goles_local"])
        result["actual_away_goals_dataset_perspective"] = int(row["goles_visitante"])

    if verbose:
        imprimir_prediccion(result)

    if save:
        out_path = project_root / "data" / "predictions" / "manual_match_predictions_updated.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        save_row = result.copy()
        save_row["top_10_scores"] = json.dumps(result["top_10_scores"], ensure_ascii=False)

        df_save = pd.DataFrame([save_row])

        if out_path.exists():
            old = pd.read_csv(out_path)
            df_save = pd.concat([old, df_save], ignore_index=True)

        df_save.to_csv(out_path, index=False, encoding="utf-8")

    return result
