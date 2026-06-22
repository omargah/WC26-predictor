# -*- coding: utf-8 -*-
"""
src/prediction/match.py

FASE 4 -- Predictor manual de partidos.

Permite predecir:
    - partidos pendientes ya presentes en el fixture;
    - partidos ya jugados, con advertencia;
    - partidos manuales no presentes en el fixture.

Regla:
    Si el partido ya está en modeling_dataset_all.parquet, se usan sus features
    pre-partido ya calculadas.

    Si no existe, se crea una fila sintética y se calculan features usando
    únicamente el historial previo a la fecha indicada.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_paths
from src.data.cleaners import assign_competition_weight
from src.data.features import (
    add_differential_features,
    add_dynamic_elo,
    add_h2h_features,
    add_team_state_features,
    attach_recent_features,
    build_team_match_history,
    ensure_datetime,
)
from src.models.poisson_dc import load_model, predict_dataframe


TEAM_ALIASES = {
    # México
    "mexico": "Mexico",
    "méxico": "Mexico",
    "mex": "Mexico",

    # Estados Unidos
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "estados unidos": "United States",

    # Canadá
    "canada": "Canada",
    "canadá": "Canada",

    # Corea
    "south korea": "South Korea",
    "corea del sur": "South Korea",
    "korea republic": "South Korea",

    # Sudáfrica
    "south africa": "South Africa",
    "sudafrica": "South Africa",
    "sudáfrica": "South Africa",

    # Chequia
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "república checa": "Czech Republic",
    "republica checa": "Czech Republic",

    # Bosnia
    "bosnia": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia y herzegovina": "Bosnia and Herzegovina",

    # Costa de Marfil
    "ivory coast": "Ivory Coast",
    "costa de marfil": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",

    # Curazao
    "curacao": "Curaçao",
    "curaçao": "Curaçao",

    # DR Congo
    "dr congo": "DR Congo",
    "rd congo": "DR Congo",
    "congo dr": "DR Congo",
    "congo rd": "DR Congo",
    "democratic republic of congo": "DR Congo",

    # Países comunes
    "england": "England",
    "inglaterra": "England",
    "france": "France",
    "francia": "France",
    "germany": "Germany",
    "alemania": "Germany",
    "spain": "Spain",
    "españa": "Spain",
    "argentina": "Argentina",
    "brazil": "Brazil",
    "brasil": "Brazil",
    "portugal": "Portugal",
    "japan": "Japan",
    "japón": "Japan",
    "netherlands": "Netherlands",
    "países bajos": "Netherlands",
    "paises bajos": "Netherlands",
    "switzerland": "Switzerland",
    "suiza": "Switzerland",
    "morocco": "Morocco",
    "marruecos": "Morocco",
    "croatia": "Croatia",
    "croacia": "Croatia",
    "colombia": "Colombia",
    "uruguay": "Uruguay",
    "belgium": "Belgium",
    "bélgica": "Belgium",
    "belgica": "Belgium",
}


def normalize_text(x: str) -> str:
    return str(x).strip().lower()


def canonical_team_name(team: str, known_teams: set[str]) -> str:
    """
    Convierte alias a nombre canónico del dataset.
    """

    raw = str(team).strip()
    key = normalize_text(raw)

    if key in TEAM_ALIASES:
        return TEAM_ALIASES[key]

    lower_map = {normalize_text(t): t for t in known_teams}

    if key in lower_map:
        return lower_map[key]

    # Búsqueda flexible: si el usuario escribe parte del nombre.
    partial_matches = [
        original for normalized, original in lower_map.items()
        if key in normalized or normalized in key
    ]

    if len(partial_matches) == 1:
        return partial_matches[0]

    return raw


def make_manual_match_id(home: str, away: str, date: str) -> str:
    raw = f"manual|{date}|{home}|{away}"
    return "manual_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def load_known_teams(paths: dict) -> set[str]:
    """
    Carga equipos conocidos desde matches_clean.parquet.
    """

    clean_path = paths["processed"] / "matches_clean.parquet"

    if not clean_path.exists():
        return set()

    df = pd.read_parquet(clean_path)

    teams = set(df["equipo_local"].dropna().unique()) | set(df["equipo_visitante"].dropna().unique())

    return teams


def find_existing_match(
    df_features_all: pd.DataFrame,
    home: str,
    away: str,
    date: str | None = None,
) -> pd.DataFrame:
    """
    Busca un partido ya existente en features.

    Si no se da fecha:
        - prioriza pendientes;
        - si no hay pendientes, toma el más reciente.
    """

    df = df_features_all.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    mask = (
        (df["equipo_local"] == home)
        &
        (df["equipo_visitante"] == away)
    )

    if date is not None:
        d = pd.to_datetime(date).normalize()
        mask = mask & (df["fecha"].dt.normalize() == d)

    found = df.loc[mask].copy()

    if found.empty:
        return found

    if date is not None:
        return found.sort_values("fecha").tail(1)

    pending = found[~found["is_played"]].copy()

    if not pending.empty:
        return pending.sort_values("fecha").head(1)

    return found.sort_values("fecha").tail(1)


def find_validation_prediction(
    paths: dict,
    home: str,
    away: str,
    date,
) -> pd.DataFrame:
    """
    Busca predicción de validación temporal si existe.

    Esto es útil para partidos ya jugados dentro del periodo de validación.
    """

    val_path = paths["predictions"] / "phase03_validation_predictions.csv"

    if not val_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(val_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    d = pd.to_datetime(date).normalize()

    mask = (
        (df["equipo_local"] == home)
        &
        (df["equipo_visitante"] == away)
        &
        (df["fecha"].dt.normalize() == d)
    )

    return df.loc[mask].copy()


def build_manual_feature_row(
    paths: dict,
    home: str,
    away: str,
    date: str,
    tournament: str,
    city: str,
    country: str,
    neutral: int,
) -> pd.DataFrame:
    """
    Construye una fila de features para un partido que no existe en el fixture.

    Usa el historial limpio completo, agrega una fila futura sin marcador
    y calcula features pre-partido.
    """

    clean_path = paths["processed"] / "matches_clean.parquet"

    if not clean_path.exists():
        raise FileNotFoundError(
            f"No existe {clean_path}. Ejecuta primero Fase 1."
        )

    df = pd.read_parquet(clean_path)
    df = ensure_datetime(df, "fecha")

    date_ts = pd.to_datetime(date)

    match_id = make_manual_match_id(home, away, str(date_ts.date()))

    peso = assign_competition_weight(tournament)

    manual_row = {
        "match_id": match_id,
        "fecha": date_ts,
        "equipo_local": home,
        "equipo_visitante": away,
        "goles_local": np.nan,
        "goles_visitante": np.nan,
        "resultado_1x2": np.nan,
        "over_2_5": np.nan,
        "btts": np.nan,
        "torneo": tournament,
        "ciudad": city,
        "pais_sede": country,
        "neutral": int(neutral),
        "peso_competicion": float(peso),
        "source_id": "manual_prediction",
        "is_played": False,
    }

    df_aug = pd.concat(
        [df, pd.DataFrame([manual_row])],
        ignore_index=True,
    )

    df_aug = df_aug.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    team_history = build_team_match_history(df_aug)
    team_state = add_team_state_features(team_history)

    df_model = add_dynamic_elo(df_aug)
    df_model = attach_recent_features(df_model, team_state)
    df_model = add_h2h_features(df_model)
    df_model = add_differential_features(df_model)

    row = df_model[df_model["match_id"] == match_id].copy()

    if row.empty:
        raise RuntimeError("No se pudo construir la fila manual de features.")

    return row


def format_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def safe_float(x) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def parse_top_scores(value) -> list[dict]:
    if isinstance(value, list):
        return value

    try:
        return json.loads(value)
    except Exception:
        return []


def predict_match(
    home: str,
    away: str,
    date: str | None = None,
    tournament: str = "FIFA World Cup",
    city: str = "Manual",
    country: str = "Unknown",
    neutral: int = 1,
) -> dict:
    """
    Predice un partido manual o de fixture.
    """

    paths = get_paths()

    model_path = paths["models"] / "poisson_dc_base.joblib"
    features_all_path = paths["features"] / "modeling_dataset_all.parquet"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo {model_path}. Ejecuta Fase 3 primero."
        )

    if not features_all_path.exists():
        raise FileNotFoundError(
            f"No existe {features_all_path}. Ejecuta Fase 2 primero."
        )

    known_teams = load_known_teams(paths)

    home_c = canonical_team_name(home, known_teams)
    away_c = canonical_team_name(away, known_teams)

    df_all = pd.read_parquet(features_all_path)
    df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce")

    existing = find_existing_match(
        df_features_all=df_all,
        home=home_c,
        away=away_c,
        date=date,
    )

    model = load_model(model_path)

    source_type = None
    warning = None

    if not existing.empty:
        row = existing.copy()
        source_type = "fixture_or_existing_dataset"

        if bool(row.iloc[0]["is_played"]):
            val_pred = find_validation_prediction(
                paths=paths,
                home=home_c,
                away=away_c,
                date=row.iloc[0]["fecha"],
            )

            if not val_pred.empty:
                pred = val_pred.copy()
                source_type = "temporal_validation_prediction"
                warning = (
                    "Este partido ya fue jugado. Se encontró predicción de "
                    "validación temporal, por lo que sirve como evaluación histórica."
                )
            else:
                pred = predict_dataframe(model, row)
                warning = (
                    "Este partido ya fue jugado. La predicción usa el modelo final "
                    "reentrenado con todos los partidos jugados, por lo que NO debe "
                    "interpretarse como evaluación honesta."
                )
        else:
            pred = predict_dataframe(model, row)
            warning = None

    else:
        if date is None:
            raise ValueError(
                "El partido no existe en el fixture/features. Para un partido manual "
                "debes indicar --date YYYY-MM-DD."
            )

        row = build_manual_feature_row(
            paths=paths,
            home=home_c,
            away=away_c,
            date=date,
            tournament=tournament,
            city=city,
            country=country,
            neutral=neutral,
        )

        pred = predict_dataframe(model, row)
        source_type = "manual_synthetic_match"
        warning = (
            "Partido no encontrado en el fixture. Se construyó una fila manual "
            "con features calculadas desde el historial previo."
        )

    r = pred.iloc[0].to_dict()

    top_scores = parse_top_scores(r.get("top_10_marcadores", "[]"))

    result = {
        "source_type": source_type,
        "warning": warning,
        "fecha": str(pd.to_datetime(r["fecha"]).date()),
        "equipo_local": r["equipo_local"],
        "equipo_visitante": r["equipo_visitante"],
        "torneo": r.get("torneo"),
        "ciudad": r.get("ciudad"),
        "pais_sede": r.get("pais_sede"),
        "neutral": int(r.get("neutral", 0)),
        "is_played": bool(r.get("is_played", False)),
        "goles_local_real": safe_float(r.get("goles_local")),
        "goles_visitante_real": safe_float(r.get("goles_visitante")),
        "lambda_local": float(r["lambda_local"]),
        "lambda_visitante": float(r["lambda_visitante"]),
        "prob_local": float(r["prob_local"]),
        "prob_empate": float(r["prob_empate"]),
        "prob_visitante": float(r["prob_visitante"]),
        "prob_over_2_5": float(r.get("prob_over_2_5", np.nan)),
        "prob_under_2_5": float(r.get("prob_under_2_5", np.nan)),
        "prob_btts_si": float(r.get("prob_btts_si", np.nan)),
        "prob_btts_no": float(r.get("prob_btts_no", np.nan)),
        "marcador_mas_probable": r.get("marcador_mas_probable"),
        "prob_marcador_mas_probable": float(r.get("prob_marcador_mas_probable", np.nan)),
        "top_10_marcadores": top_scores,
        "pred_resultado_1x2": r.get("pred_resultado_1x2"),
    }

    return result


def print_prediction(result: dict) -> None:
    """
    Imprime una predicción en formato legible.
    """

    print()
    print("=" * 80)
    print("PREDICTOR MANUAL — MUNDIAL 2026")
    print("=" * 80)

    print()
    print(f"Partido: {result['equipo_local']} vs {result['equipo_visitante']}")
    print(f"Fecha:   {result['fecha']}")
    print(f"Torneo:  {result.get('torneo')}")
    print(f"Sede:    {result.get('ciudad')}, {result.get('pais_sede')}")
    print(f"Neutral: {result.get('neutral')}")
    print(f"Fuente:  {result['source_type']}")

    if result.get("warning"):
        print()
        print("ADVERTENCIA:")
        print(result["warning"])

    if result["is_played"]:
        print()
        print("Resultado real:")
        print(
            f"{result['equipo_local']} {int(result['goles_local_real'])} - "
            f"{int(result['goles_visitante_real'])} {result['equipo_visitante']}"
        )

    print()
    print("-" * 80)
    print("GOLES ESPERADOS")
    print("-" * 80)
    print(f"λ local:      {result['lambda_local']:.3f}")
    print(f"λ visitante:  {result['lambda_visitante']:.3f}")

    print()
    print("-" * 80)
    print("PROBABILIDADES 1X2")
    print("-" * 80)
    print(f"Gana {result['equipo_local']}:      {format_pct(result['prob_local'])}")
    print(f"Empate:                     {format_pct(result['prob_empate'])}")
    print(f"Gana {result['equipo_visitante']}:  {format_pct(result['prob_visitante'])}")

    print()
    print("-" * 80)
    print("MERCADOS DERIVADOS")
    print("-" * 80)
    print(f"Over 2.5:       {format_pct(result['prob_over_2_5'])}")
    print(f"Under 2.5:      {format_pct(result['prob_under_2_5'])}")
    print(f"BTTS Sí:        {format_pct(result['prob_btts_si'])}")
    print(f"BTTS No:        {format_pct(result['prob_btts_no'])}")

    print()
    print("-" * 80)
    print("MARCADOR MÁS PROBABLE")
    print("-" * 80)
    print(
        f"{result['marcador_mas_probable']} "
        f"({format_pct(result['prob_marcador_mas_probable'])})"
    )

    print()
    print("-" * 80)
    print("TOP 10 MARCADORES")
    print("-" * 80)

    for item in result["top_10_marcadores"][:10]:
        print(f"{item['score']:>5}  {format_pct(item['prob'])}")

    print()
    print("=" * 80)


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Predice un partido usando el modelo Poisson + Dixon-Coles."
    )

    parser.add_argument("--home", required=True, help="Equipo local")
    parser.add_argument("--away", required=True, help="Equipo visitante")
    parser.add_argument("--date", default=None, help="Fecha YYYY-MM-DD. Obligatoria si el partido no está en fixture.")
    parser.add_argument("--tournament", default="FIFA World Cup", help="Torneo para partido manual.")
    parser.add_argument("--city", default="Manual", help="Ciudad para partido manual.")
    parser.add_argument("--country", default="Unknown", help="País sede para partido manual.")
    parser.add_argument("--neutral", type=int, default=1, help="1 neutral, 0 no neutral.")
    parser.add_argument("--json", action="store_true", help="Imprime JSON en lugar de formato humano.")

    args = parser.parse_args()

    result = predict_match(
        home=args.home,
        away=args.away,
        date=args.date,
        tournament=args.tournament,
        city=args.city,
        country=args.country,
        neutral=args.neutral,
    )

    if args.json:
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print_prediction(result)


if __name__ == "__main__":
    main_cli()
