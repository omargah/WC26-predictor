# -*- coding: utf-8 -*-
"""
src/simulation/knockout_engine.py

Motor de eliminatorias V2.

Objetivo:
    Simular partidos de eliminación directa de forma seria:

        - usa el modelo Poisson-Dixon-Coles entrenado;
        - actualiza el estado pre-KO con los resultados de grupo simulados;
        - construye partidos neutrales;
        - neutraliza el orden team_a/team_b;
        - simula 90 minutos;
        - si hay empate, simula prórroga;
        - si sigue empate, simula penales.

Importante:
    Para una simulación de torneo completa, este módulo debe llamarse
    ronda por ronda. Después de cada ronda, los resultados KO pueden
    agregarse como partidos jugados antes de construir features de la
    siguiente ronda.
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
    """
    Configuración del motor KO.

    extra_time_minutes:
        Duración reglamentaria de la prórroga.

    extra_time_intensity:
        Factor adicional para reflejar que la prórroga suele tener menor
        ritmo que 30 minutos normales. Si quisieras proporcionalidad pura,
        usa 1.0.

    penalty_strength_scale:
        Cuánto influye la diferencia de lambdas en la tanda de penales.

    penalty_min_prob / penalty_max_prob:
        Cota para evitar que los penales se vuelvan casi deterministas.
    """

    match_date: str = "2026-06-28"
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


def _stable_match_id(raw: str) -> str:
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _result_1x2(gf: int, ga: int) -> str:
    if gf > ga:
        return "L"
    if gf < ga:
        return "V"
    return "E"


def _over_2_5(gf: int, ga: int) -> float:
    return float((gf + ga) >= 3)


def _btts(gf: int, ga: int) -> float:
    return float((gf > 0) and (ga > 0))



def build_modeling_dataset_compatible(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Construye features usando la API disponible en src.data.features.

    Compatibilidad:
        - si existe build_modeling_dataset(), la usa directamente;
        - si no existe, arma el pipeline con las funciones modulares
          disponibles en el archivo local de features.py.
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

    # Ruta modular usada en versiones más completas de Fase 2.
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

    # Ruta compacta usada en versiones anteriores.
    elif hasattr(feature_builder, "add_recent_form_features"):
        df = feature_builder.add_recent_form_features(df)

    else:
        raise ImportError(
            "No encontré funciones de forma reciente en src.data.features: "
            "ni attach_recent_features ni add_recent_form_features."
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
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero Fase 1."
        )

    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    return df


def apply_group_scenario_to_clean_matches(
    df_clean: pd.DataFrame,
    df_group_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inserta en el dataset limpio los resultados de grupo de una simulación.

    Para played_policy='fixed':
        Los partidos ya jugados mantienen resultado real.
        Los pendientes reciben marcador simulado.

    Para played_policy='resimulate':
        Incluso los partidos ya jugados reciben marcador contrafactual.

    Resultado:
        todos los 72 partidos de grupo quedan como is_played=True antes
        de construir features de eliminatorias.
    """

    df = df_clean.copy()
    df["match_id"] = df["match_id"].astype(str)

    required = [
        "match_id",
        "home",
        "away",
        "home_goals",
        "away_goals",
    ]

    missing = [c for c in required if c not in df_group_matches.columns]

    if missing:
        raise ValueError(
            "Faltan columnas en df_group_matches: " + ", ".join(missing)
        )

    group_matches = df_group_matches.copy()
    group_matches["match_id"] = group_matches["match_id"].astype(str)

    for _, row in group_matches.iterrows():
        match_id = str(row["match_id"])
        gh = int(row["home_goals"])
        ga = int(row["away_goals"])

        mask = df["match_id"] == match_id

        if mask.sum() != 1:
            raise RuntimeError(
                f"Se esperaba encontrar exactamente 1 fila para match_id={match_id}; "
                f"encontradas={int(mask.sum())}."
            )

        df.loc[mask, "goles_local"] = float(gh)
        df.loc[mask, "goles_visitante"] = float(ga)
        df.loc[mask, "resultado_1x2"] = _result_1x2(gh, ga)
        df.loc[mask, "over_2_5"] = _over_2_5(gh, ga)
        df.loc[mask, "btts"] = _btts(gh, ga)
        df.loc[mask, "is_played"] = True

    return df.sort_values("fecha").reset_index(drop=True)


def build_synthetic_ko_rows(
    pairs: list[tuple[str, str]],
    config: KnockoutConfig,
    round_label: str = "Round of 32",
) -> pd.DataFrame:
    """
    Construye filas sintéticas de partidos KO neutrales.

    Cada par se interpreta como:
        equipo_local = team_a
        equipo_visitante = team_b

    Se crean sin marcador porque son filas para predecir lambdas.
    """

    rows = []

    date = pd.to_datetime(config.match_date)

    for home, away in pairs:
        raw = f"{date.date()}|{round_label}|{home}|{away}|KO_V2"
        match_id = _stable_match_id(raw)

        rows.append(
            {
                "match_id": match_id,
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
                "source_id": "simulation_ko_v2",
                "is_played": False,
                "ko_round": round_label,
                "ko_pair_home": home,
                "ko_pair_away": away,
            }
        )

    return pd.DataFrame(rows)


def directed_pairs_from_bracket(bracket: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Devuelve ambos sentidos para cada cruce:
        A vs B y B vs A

    Esto permite neutralizar la ventaja artificial de aparecer como team_a.
    """

    pairs = []

    for _, row in bracket.iterrows():
        a = str(row["team_a"])
        b = str(row["team_b"])

        pairs.append((a, b))
        pairs.append((b, a))

    # preservar orden sin duplicados
    out = []
    seen = set()

    for p in pairs:
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out


def build_directed_knockout_lambdas(
    bracket: pd.DataFrame,
    df_group_matches: pd.DataFrame,
    paths: dict | None = None,
    config: KnockoutConfig | None = None,
    round_label: str = "Round of 32",
) -> pd.DataFrame:
    """
    Construye lambdas para ambos sentidos de cada cruce KO.

    Metodología:
        1. carga matches_clean;
        2. sustituye resultados de grupo por la simulación actual;
        3. agrega filas KO sintéticas neutrales;
        4. recalcula features;
        5. predice lambdas con el modelo entrenado.
    """

    if paths is None:
        paths = get_paths()

    if config is None:
        config = KnockoutConfig()

    clean = load_clean_matches(paths)
    scenario_clean = apply_group_scenario_to_clean_matches(
        df_clean=clean,
        df_group_matches=df_group_matches,
    )

    pairs = directed_pairs_from_bracket(bracket)

    synthetic_ko = build_synthetic_ko_rows(
        pairs=pairs,
        config=config,
        round_label=round_label,
    )

    augmented = pd.concat(
        [scenario_clean, synthetic_ko],
        ignore_index=True,
    ).sort_values("fecha").reset_index(drop=True)

    modeled = build_modeling_dataset_compatible(augmented)

    ko_features = modeled[modeled["source_id"] == "simulation_ko_v2"].copy()

    if len(ko_features) != len(synthetic_ko):
        raise RuntimeError(
            f"Se esperaban {len(synthetic_ko)} filas KO modeladas "
            f"y salieron {len(ko_features)}."
        )

    model_package = load_model(paths["models"] / "poisson_dc_base.joblib")

    lambda_home, lambda_away = predict_lambdas(model_package, ko_features)

    ko_features["lambda_home"] = lambda_home
    ko_features["lambda_away"] = lambda_away
    ko_features["round"] = round_label

    cols = [
        "match_id",
        "round",
        "fecha",
        "equipo_local",
        "equipo_visitante",
        "neutral",
        "lambda_home",
        "lambda_away",
        "elo_local_pre",
        "elo_visitante_pre",
        "diff_elo_pre",
        "h2h_matches",
        "h2h_goals_home_avg",
        "h2h_goals_away_avg",
        "L_form_goals_for_avg",
        "V_form_goals_for_avg",
        "L_form_goals_against_avg",
        "V_form_goals_against_avg",
    ]

    cols = [c for c in cols if c in ko_features.columns]

    return ko_features[cols].reset_index(drop=True)


def build_neutralized_knockout_predictions(
    bracket: pd.DataFrame,
    directed_lambdas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Neutraliza lambdas usando ambos sentidos del partido.

    Para A vs B:
        predicción A local vs B visitante:
            lambda_A_home, lambda_B_away

        predicción B local vs A visitante:
            lambda_B_home, lambda_A_away

    Neutralización:
        lambda_A = promedio(lambda_A_home, lambda_A_away)
        lambda_B = promedio(lambda_B_away, lambda_B_home)
    """

    lookup = {}

    for _, row in directed_lambdas.iterrows():
        home = str(row["equipo_local"])
        away = str(row["equipo_visitante"])

        lookup[(home, away)] = row

    rows = []

    for _, row in bracket.iterrows():
        a = str(row["team_a"])
        b = str(row["team_b"])

        if (a, b) not in lookup:
            raise RuntimeError(f"Falta lambda dirigida para {a} vs {b}.")

        if (b, a) not in lookup:
            raise RuntimeError(f"Falta lambda dirigida para {b} vs {a}.")

        ab = lookup[(a, b)]
        ba = lookup[(b, a)]

        lambda_a = 0.5 * (float(ab["lambda_home"]) + float(ba["lambda_away"]))
        lambda_b = 0.5 * (float(ab["lambda_away"]) + float(ba["lambda_home"]))

        rows.append(
            {
                **row.to_dict(),
                "lambda_a_90": float(lambda_a),
                "lambda_b_90": float(lambda_b),
                "lambda_a_home_orientation": float(ab["lambda_home"]),
                "lambda_b_away_orientation": float(ab["lambda_away"]),
                "lambda_b_home_orientation": float(ba["lambda_home"]),
                "lambda_a_away_orientation": float(ba["lambda_away"]),
                "neutralization_method": "average_both_directions",
            }
        )

    return pd.DataFrame(rows)


def sample_score(
    lambda_a: float,
    lambda_b: float,
    rng: np.random.Generator,
    max_goals: int,
    rho: float,
) -> tuple[int, int]:
    mat = score_matrix(
        lambda_a,
        lambda_b,
        max_goals=max_goals,
        rho=rho,
        use_dixon_coles=True,
    )

    flat = rng.choice(mat.size, p=mat.ravel())
    ga, gb = np.unravel_index(flat, mat.shape)

    return int(ga), int(gb)


def penalty_win_probability_a(
    lambda_a: float,
    lambda_b: float,
    config: KnockoutConfig,
) -> float:
    """
    Probabilidad de que team_a gane la tanda.

    Los penales tienen mucha varianza; por eso se acota el efecto de la
    fuerza del modelo entre penalty_min_prob y penalty_max_prob.
    """

    diff = float(lambda_a) - float(lambda_b)

    p = 1.0 / (1.0 + np.exp(-config.penalty_strength_scale * diff))

    p = float(
        np.clip(
            p,
            config.penalty_min_prob,
            config.penalty_max_prob,
        )
    )

    return p


def simulate_knockout_match(
    prediction_row: pd.Series,
    rng: np.random.Generator,
    config: KnockoutConfig | None = None,
) -> dict:
    """
    Simula un partido KO con:
        90 minutos,
        prórroga si hay empate,
        penales si persiste el empate.
    """

    if config is None:
        config = KnockoutConfig()

    team_a = str(prediction_row["team_a"])
    team_b = str(prediction_row["team_b"])

    lambda_a_90 = float(prediction_row["lambda_a_90"])
    lambda_b_90 = float(prediction_row["lambda_b_90"])

    goals_a_90, goals_b_90 = sample_score(
        lambda_a=lambda_a_90,
        lambda_b=lambda_b_90,
        rng=rng,
        max_goals=config.max_goals_90,
        rho=config.rho,
    )

    goals_a_et = 0
    goals_b_et = 0
    decided_by = "90"
    went_to_extra_time = False
    went_to_penalties = False
    penalty_winner = None

    total_a = goals_a_90
    total_b = goals_b_90

    if total_a == total_b:
        went_to_extra_time = True

        et_scale = (config.extra_time_minutes / 90.0) * config.extra_time_intensity

        goals_a_et, goals_b_et = sample_score(
            lambda_a=lambda_a_90 * et_scale,
            lambda_b=lambda_b_90 * et_scale,
            rng=rng,
            max_goals=config.max_goals_extra_time,
            rho=config.rho,
        )

        total_a += goals_a_et
        total_b += goals_b_et

        if total_a != total_b:
            decided_by = "ET"

    p_penalty_a = penalty_win_probability_a(
        lambda_a=lambda_a_90,
        lambda_b=lambda_b_90,
        config=config,
    )

    if total_a == total_b:
        went_to_penalties = True
        decided_by = "PEN"

        if rng.random() < p_penalty_a:
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
        "round": prediction_row.get("round", "Round of 32"),
        "match_number": int(prediction_row["match_number"]),
        "slot_a": prediction_row.get("slot_a"),
        "slot_b": prediction_row.get("slot_b"),
        "team_a": team_a,
        "team_b": team_b,
        "group_a": prediction_row.get("group_a"),
        "group_b": prediction_row.get("group_b"),
        "lambda_a_90": lambda_a_90,
        "lambda_b_90": lambda_b_90,
        "goals_a_90": goals_a_90,
        "goals_b_90": goals_b_90,
        "goals_a_et": goals_a_et,
        "goals_b_et": goals_b_et,
        "goals_a_total": total_a,
        "goals_b_total": total_b,
        "went_to_extra_time": went_to_extra_time,
        "went_to_penalties": went_to_penalties,
        "p_penalty_a": p_penalty_a,
        "penalty_winner": penalty_winner,
        "winner": winner,
        "loser": loser,
        "decided_by": decided_by,
    }


def simulate_knockout_round_once(
    bracket: pd.DataFrame,
    df_group_matches: pd.DataFrame,
    seed: int = 42,
    paths: dict | None = None,
    config: KnockoutConfig | None = None,
    round_label: str = "Round of 32",
) -> dict[str, pd.DataFrame]:
    """
    Simula una ronda KO completa.

    Devuelve:
        directed_lambdas:
            lambdas A-vs-B y B-vs-A.

        neutral_predictions:
            lambdas neutralizadas para cada cruce.

        results:
            resultado simulado de cada partido KO.
    """

    if paths is None:
        paths = get_paths()

    if config is None:
        config = KnockoutConfig()

    rng = np.random.default_rng(seed)

    directed = build_directed_knockout_lambdas(
        bracket=bracket,
        df_group_matches=df_group_matches,
        paths=paths,
        config=config,
        round_label=round_label,
    )

    neutral = build_neutralized_knockout_predictions(
        bracket=bracket,
        directed_lambdas=directed,
    )

    results = []

    for _, row in neutral.iterrows():
        results.append(
            simulate_knockout_match(
                prediction_row=row,
                rng=rng,
                config=config,
            )
        )

    return {
        "directed_lambdas": directed,
        "neutral_predictions": neutral,
        "results": pd.DataFrame(results),
    }
