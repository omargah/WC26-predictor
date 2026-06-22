# -*- coding: utf-8 -*-
"""
src/simulation/knockout_engine_full.py

Motor KO completo para Tournament V2.

A diferencia del primer knockout_engine.py, este módulo permite pasar
resultados KO previos para recalcular features ronda por ronda.

Flujo:
    grupos simulados/fijos
    + KO previos ya jugados
    + KO actual sintético
    -> features actualizadas
    -> lambdas Poisson-Dixon-Coles
    -> neutralización por doble orientación
    -> 90 min + prórroga + penales
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import get_paths
from src.data.cleaners import assign_competition_weight
import src.data.features as feature_builder
from src.models.poisson_dc import load_model, predict_lambdas, score_matrix


@dataclass
class KnockoutConfig:
    tournament: str = "FIFA World Cup"
    city: str = "Knockout"
    country: str = "Neutral"
    neutral: int = 1
    max_goals_90: int = 10
    max_goals_extra_time: int = 6
    rho: float = -0.075
    extra_time_minutes: float = 30.0
    extra_time_intensity: float = 0.85
    penalty_strength_scale: float = 0.40
    penalty_min_prob: float = 0.40
    penalty_max_prob: float = 0.60


ROUND_DATES = {
    "Round of 32": "2026-06-28",
    "Round of 16": "2026-07-04",
    "Quarterfinal": "2026-07-09",
    "Semifinal": "2026-07-14",
    "Third Place": "2026-07-18",
    "Final": "2026-07-19",
}


def stable_match_id(raw: str) -> str:
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def result_1x2(gf: int, ga: int) -> str:
    if gf > ga:
        return "L"
    if gf < ga:
        return "V"
    return "E"


def over_2_5(gf: int, ga: int) -> float:
    return float((gf + ga) >= 3)


def btts(gf: int, ga: int) -> float:
    return float((gf > 0) and (ga > 0))


def build_modeling_dataset_compatible(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Usa la API real disponible en src.data.features.
    """

    df = df_clean.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.sort_values("fecha").reset_index(drop=True)

    if hasattr(feature_builder, "build_modeling_dataset"):
        return feature_builder.build_modeling_dataset(df)

    if hasattr(feature_builder, "add_dynamic_elo"):
        df = feature_builder.add_dynamic_elo(df)
    else:
        raise ImportError(
            "src.data.features no tiene add_dynamic_elo ni build_modeling_dataset."
        )

    if (
        hasattr(feature_builder, "build_team_match_history")
        and hasattr(feature_builder, "add_team_state_features")
        and hasattr(feature_builder, "attach_recent_features")
    ):
        team_history = feature_builder.build_team_match_history(df)
        team_state = feature_builder.add_team_state_features(team_history)

        try:
            df = feature_builder.attach_recent_features(df, team_state)
        except TypeError:
            df = feature_builder.attach_recent_features(df)

    elif hasattr(feature_builder, "add_recent_form_features"):
        df = feature_builder.add_recent_form_features(df)

    else:
        raise ImportError(
            "No encontré funciones de forma reciente en src.data.features."
        )

    if hasattr(feature_builder, "add_h2h_features"):
        df = feature_builder.add_h2h_features(df)

    if hasattr(feature_builder, "add_differential_features"):
        df = feature_builder.add_differential_features(df)

    if hasattr(feature_builder, "check_temporal_leakage"):
        leakage = feature_builder.check_temporal_leakage(df)

        if leakage.get("leakage_total", 0) > 0:
            raise ValueError(f"Se detectó leakage temporal: {leakage}")

    return df


def load_clean_matches(paths: dict | None = None) -> pd.DataFrame:
    if paths is None:
        paths = get_paths()

    path = paths["processed"] / "matches_clean.parquet"

    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero Fase 1.")

    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["match_id"] = df["match_id"].astype(str)

    return df


def apply_group_scenario_to_clean_matches(
    df_clean: pd.DataFrame,
    df_group_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sustituye los 72 partidos de grupo por el escenario actual.
    """

    df = df_clean.copy()
    df["match_id"] = df["match_id"].astype(str)

    group_matches = df_group_matches.copy()
    group_matches["match_id"] = group_matches["match_id"].astype(str)

    for _, row in group_matches.iterrows():
        match_id = str(row["match_id"])
        gh = int(row["home_goals"])
        ga = int(row["away_goals"])

        mask = df["match_id"] == match_id

        if mask.sum() != 1:
            raise RuntimeError(
                f"match_id={match_id}: esperada 1 fila, encontradas {int(mask.sum())}."
            )

        df.loc[mask, "goles_local"] = float(gh)
        df.loc[mask, "goles_visitante"] = float(ga)
        df.loc[mask, "resultado_1x2"] = result_1x2(gh, ga)
        df.loc[mask, "over_2_5"] = over_2_5(gh, ga)
        df.loc[mask, "btts"] = btts(gh, ga)
        df.loc[mask, "is_played"] = True

    return df.sort_values("fecha").reset_index(drop=True)


def ko_results_to_played_rows(
    results: pd.DataFrame,
    round_label: str,
    config: KnockoutConfig,
) -> pd.DataFrame:
    """
    Convierte resultados KO ya simulados a filas jugadas para recalcular features.

    Para partidos decididos por penales, el marcador usado para features es el
    marcador tras prórroga. Es decir, puede quedar empate en goles aunque haya
    ganador por penales.
    """

    date = pd.to_datetime(ROUND_DATES.get(round_label, "2026-07-01"))

    rows = []

    for _, r in results.iterrows():
        team_a = str(r["team_a"])
        team_b = str(r["team_b"])

        ga = int(r["goals_a_total"])
        gb = int(r["goals_b_total"])

        raw = f"{date.date()}|{round_label}|{int(r['match_number'])}|{team_a}|{team_b}|PLAYED_KO_V2"

        rows.append(
            {
                "match_id": stable_match_id(raw),
                "fecha": date,
                "equipo_local": team_a,
                "equipo_visitante": team_b,
                "goles_local": float(ga),
                "goles_visitante": float(gb),
                "resultado_1x2": result_1x2(ga, gb),
                "over_2_5": over_2_5(ga, gb),
                "btts": btts(ga, gb),
                "torneo": config.tournament,
                "ciudad": config.city,
                "pais_sede": config.country,
                "neutral": int(config.neutral),
                "peso_competicion": assign_competition_weight(config.tournament),
                "source_id": "simulation_ko_v2_played",
                "is_played": True,
                "ko_round": round_label,
                "decided_by": r.get("decided_by"),
                "winner": r.get("winner"),
            }
        )

    return pd.DataFrame(rows)


def build_synthetic_ko_rows(
    bracket: pd.DataFrame,
    round_label: str,
    config: KnockoutConfig,
) -> pd.DataFrame:
    """
    Construye ambos sentidos de cada partido KO:
        A vs B
        B vs A
    """

    date = pd.to_datetime(ROUND_DATES.get(round_label, "2026-07-01"))

    rows = []

    for _, r in bracket.iterrows():
        for orientation in ["AB", "BA"]:
            if orientation == "AB":
                home = str(r["team_a"])
                away = str(r["team_b"])
            else:
                home = str(r["team_b"])
                away = str(r["team_a"])

            raw = f"{date.date()}|{round_label}|{int(r['match_number'])}|{home}|{away}|SYNTH_KO_V2"

            rows.append(
                {
                    "match_id": stable_match_id(raw),
                    "fecha": date,
                    "equipo_local": home,
                    "equipo_visitante": away,
                    "goles_local": np.nan,
                    "goles_visitante": np.nan,
                    "resultado_1x2": np.nan,
                    "over_2_5": np.nan,
                    "btts": np.nan,
                    "torneo": config.tournament,
                    "ciudad": config.city,
                    "pais_sede": config.country,
                    "neutral": int(config.neutral),
                    "peso_competicion": assign_competition_weight(config.tournament),
                    "source_id": "simulation_ko_v2_current",
                    "is_played": False,
                    "ko_round": round_label,
                    "ko_match_number": int(r["match_number"]),
                    "orientation": orientation,
                }
            )

    return pd.DataFrame(rows)


def build_directed_knockout_lambdas(
    bracket: pd.DataFrame,
    df_group_matches: pd.DataFrame,
    prior_ko_results: list[tuple[str, pd.DataFrame]] | None = None,
    paths: dict | None = None,
    config: KnockoutConfig | None = None,
    round_label: str = "Round of 32",
) -> pd.DataFrame:
    if paths is None:
        paths = get_paths()

    if config is None:
        config = KnockoutConfig()

    if prior_ko_results is None:
        prior_ko_results = []

    clean = load_clean_matches(paths)
    scenario_clean = apply_group_scenario_to_clean_matches(clean, df_group_matches)

    prior_rows = []

    for prior_round_label, prior_df in prior_ko_results:
        if prior_df is not None and len(prior_df) > 0:
            prior_rows.append(
                ko_results_to_played_rows(
                    results=prior_df,
                    round_label=prior_round_label,
                    config=config,
                )
            )

    synthetic = build_synthetic_ko_rows(
        bracket=bracket,
        round_label=round_label,
        config=config,
    )

    frames = [scenario_clean]

    if prior_rows:
        frames.extend(prior_rows)

    frames.append(synthetic)

    augmented = pd.concat(frames, ignore_index=True).sort_values("fecha").reset_index(drop=True)

    modeled = build_modeling_dataset_compatible(augmented)

    ko_features = modeled[modeled["source_id"] == "simulation_ko_v2_current"].copy()

    if len(ko_features) != len(synthetic):
        raise RuntimeError(
            f"Se esperaban {len(synthetic)} filas KO actuales y salieron {len(ko_features)}."
        )

    model_package = load_model(paths["models"] / "poisson_dc_base.joblib")

    lambda_home, lambda_away = predict_lambdas(model_package, ko_features)

    ko_features["lambda_home"] = lambda_home
    ko_features["lambda_away"] = lambda_away
    ko_features["round"] = round_label

    keep = [
        "match_id",
        "round",
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "ko_match_number",
        "orientation",
        "neutral",
        "lambda_home",
        "lambda_away",
        "elo_local_pre",
        "elo_visitante_pre",
        "diff_elo_pre",
        "h2h_matches",
        "L_form_goals_for_avg",
        "V_form_goals_for_avg",
        "L_form_goals_against_avg",
        "V_form_goals_against_avg",
    ]

    keep = [c for c in keep if c in ko_features.columns]

    return ko_features[keep].reset_index(drop=True)


def build_neutralized_predictions(
    bracket: pd.DataFrame,
    directed_lambdas: pd.DataFrame,
    round_label: str,
) -> pd.DataFrame:
    lookup = {}

    for _, r in directed_lambdas.iterrows():
        lookup[(str(r["equipo_local"]), str(r["equipo_visitante"]), int(r["ko_match_number"]))] = r

    rows = []

    for _, r in bracket.iterrows():
        match_number = int(r["match_number"])
        a = str(r["team_a"])
        b = str(r["team_b"])

        ab = lookup[(a, b, match_number)]
        ba = lookup[(b, a, match_number)]

        lambda_a = 0.5 * (float(ab["lambda_home"]) + float(ba["lambda_away"]))
        lambda_b = 0.5 * (float(ab["lambda_away"]) + float(ba["lambda_home"]))

        row = r.to_dict()
        row.update(
            {
                "round": round_label,
                "lambda_a_90": float(lambda_a),
                "lambda_b_90": float(lambda_b),
                "lambda_a_home_orientation": float(ab["lambda_home"]),
                "lambda_b_away_orientation": float(ab["lambda_away"]),
                "lambda_b_home_orientation": float(ba["lambda_home"]),
                "lambda_a_away_orientation": float(ba["lambda_away"]),
                "neutralization_method": "average_both_directions",
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def sample_score(
    lambda_a: float,
    lambda_b: float,
    rng: np.random.Generator,
    max_goals: int,
    rho: float,
) -> tuple[int, int]:
    mat = score_matrix(
        float(lambda_a),
        float(lambda_b),
        max_goals=max_goals,
        rho=rho,
        use_dixon_coles=True,
    )

    flat = rng.choice(mat.size, p=mat.ravel())
    a, b = np.unravel_index(flat, mat.shape)

    return int(a), int(b)


def penalty_win_probability_a(
    lambda_a: float,
    lambda_b: float,
    config: KnockoutConfig,
) -> float:
    diff = float(lambda_a) - float(lambda_b)
    p = 1.0 / (1.0 + np.exp(-config.penalty_strength_scale * diff))

    return float(np.clip(p, config.penalty_min_prob, config.penalty_max_prob))


def simulate_knockout_match(
    row: pd.Series,
    rng: np.random.Generator,
    config: KnockoutConfig,
) -> dict:
    team_a = str(row["team_a"])
    team_b = str(row["team_b"])

    la = float(row["lambda_a_90"])
    lb = float(row["lambda_b_90"])

    ga90, gb90 = sample_score(la, lb, rng, config.max_goals_90, config.rho)

    gaet = 0
    gbet = 0

    total_a = ga90
    total_b = gb90

    went_et = False
    went_pen = False
    decided_by = "90"
    penalty_winner = None

    if total_a == total_b:
        went_et = True

        et_scale = (config.extra_time_minutes / 90.0) * config.extra_time_intensity

        gaet, gbet = sample_score(
            la * et_scale,
            lb * et_scale,
            rng,
            config.max_goals_extra_time,
            config.rho,
        )

        total_a += gaet
        total_b += gbet

        if total_a != total_b:
            decided_by = "ET"

    p_pen_a = penalty_win_probability_a(la, lb, config)

    if total_a == total_b:
        went_pen = True
        decided_by = "PEN"

        if rng.random() < p_pen_a:
            winner = team_a
            loser = team_b
            penalty_winner = team_a
        else:
            winner = team_b
            loser = team_a
            penalty_winner = team_b

    else:
        if total_a > total_b:
            winner = team_a
            loser = team_b
        else:
            winner = team_b
            loser = team_a

    return {
        "round": str(row["round"]),
        "match_number": int(row["match_number"]),
        "slot_a": row.get("slot_a"),
        "slot_b": row.get("slot_b"),
        "team_a": team_a,
        "team_b": team_b,
        "group_a": row.get("group_a"),
        "group_b": row.get("group_b"),
        "lambda_a_90": la,
        "lambda_b_90": lb,
        "goals_a_90": ga90,
        "goals_b_90": gb90,
        "goals_a_et": gaet,
        "goals_b_et": gbet,
        "goals_a_total": total_a,
        "goals_b_total": total_b,
        "went_to_extra_time": went_et,
        "went_to_penalties": went_pen,
        "p_penalty_a": p_pen_a,
        "penalty_winner": penalty_winner,
        "winner": winner,
        "loser": loser,
        "decided_by": decided_by,
    }


def simulate_knockout_round_once(
    bracket: pd.DataFrame,
    df_group_matches: pd.DataFrame,
    prior_ko_results: list[tuple[str, pd.DataFrame]] | None = None,
    seed: int = 42,
    paths: dict | None = None,
    config: KnockoutConfig | None = None,
    round_label: str = "Round of 32",
) -> dict[str, pd.DataFrame]:
    if paths is None:
        paths = get_paths()

    if config is None:
        config = KnockoutConfig()

    if prior_ko_results is None:
        prior_ko_results = []

    rng = np.random.default_rng(seed)

    directed = build_directed_knockout_lambdas(
        bracket=bracket,
        df_group_matches=df_group_matches,
        prior_ko_results=prior_ko_results,
        paths=paths,
        config=config,
        round_label=round_label,
    )

    neutral = build_neutralized_predictions(
        bracket=bracket,
        directed_lambdas=directed,
        round_label=round_label,
    )

    results = []

    for _, row in neutral.iterrows():
        results.append(simulate_knockout_match(row, rng, config))

    return {
        "directed_lambdas": directed,
        "neutral_predictions": neutral,
        "results": pd.DataFrame(results),
    }
