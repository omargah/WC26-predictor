# -*- coding: utf-8 -*-
"""
src/data/features.py

FASE 2 -- Construcción de features sin leakage temporal.

Entrada:
    data/processed/matches_clean.parquet

Salidas:
    data/features/modeling_dataset_all.parquet
    data/features/modeling_dataset_train.parquet
    data/features/modeling_dataset_pending.parquet
    data/features/team_state_history.parquet

Regla central:
    Para predecir un partido en fecha T, solo usamos información de partidos
    jugados con fecha estrictamente menor que T.

Esto evita leakage temporal.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Utilidades
# ============================================================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_datetime(df: pd.DataFrame, col: str = "fecha") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ============================================================
# 1. Historial equipo-partido
# ============================================================

def build_team_match_history(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte partidos jugados a tabla equipo-partido.

    Cada partido jugado produce dos filas:
        - perspectiva del local;
        - perspectiva del visitante.

    Los partidos futuros NO entran aquí porque no tienen marcador real.
    """

    played = df_matches[df_matches["is_played"]].copy()
    played = played.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    rows = []

    for _, row in played.iterrows():
        gl = float(row["goles_local"])
        gv = float(row["goles_visitante"])

        # Perspectiva local
        rows.append(
            {
                "fecha": row["fecha"],
                "match_id": row["match_id"],
                "equipo": row["equipo_local"],
                "opponent": row["equipo_visitante"],
                "is_home": 1,
                "neutral": int(row["neutral"]),
                "torneo": row["torneo"],
                "peso_competicion": float(row["peso_competicion"]),
                "goals_for": gl,
                "goals_against": gv,
                "goal_diff": gl - gv,
                "points": 3 if gl > gv else (1 if gl == gv else 0),
                "win": 1 if gl > gv else 0,
                "draw": 1 if gl == gv else 0,
                "loss": 1 if gl < gv else 0,
                "clean_sheet": 1 if gv == 0 else 0,
                "failed_to_score": 1 if gl == 0 else 0,
                "btts": 1 if gl > 0 and gv > 0 else 0,
                "over_2_5": 1 if gl + gv >= 3 else 0,
            }
        )

        # Perspectiva visitante
        rows.append(
            {
                "fecha": row["fecha"],
                "match_id": row["match_id"],
                "equipo": row["equipo_visitante"],
                "opponent": row["equipo_local"],
                "is_home": 0,
                "neutral": int(row["neutral"]),
                "torneo": row["torneo"],
                "peso_competicion": float(row["peso_competicion"]),
                "goals_for": gv,
                "goals_against": gl,
                "goal_diff": gv - gl,
                "points": 3 if gv > gl else (1 if gl == gv else 0),
                "win": 1 if gv > gl else 0,
                "draw": 1 if gl == gv else 0,
                "loss": 1 if gv < gl else 0,
                "clean_sheet": 1 if gl == 0 else 0,
                "failed_to_score": 1 if gv == 0 else 0,
                "btts": 1 if gl > 0 and gv > 0 else 0,
                "over_2_5": 1 if gl + gv >= 3 else 0,
            }
        )

    team_history = pd.DataFrame(rows)

    if team_history.empty:
        raise ValueError("No hay partidos jugados para construir historial de equipos.")

    team_history = team_history.sort_values(["equipo", "fecha", "match_id"]).reset_index(drop=True)

    return team_history


# ============================================================
# 2. Estados de equipo después de cada partido jugado
# ============================================================

def add_team_state_features(
    team_history: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Calcula snapshots de forma del equipo DESPUÉS de cada partido jugado.

    Luego, para un partido objetivo en fecha T, se usará el último snapshot
    con fecha < T.

    Por eso no usamos shift aquí: el snapshot representa el estado después
    de ese partido. El anti-leakage se garantiza al cruzar con fecha
    estrictamente anterior.
    """

    numeric_cols = [
        "goals_for",
        "goals_against",
        "goal_diff",
        "points",
        "win",
        "draw",
        "loss",
        "clean_sheet",
        "failed_to_score",
        "btts",
        "over_2_5",
    ]

    outputs = []

    for equipo, g in team_history.groupby("equipo", sort=False):
        g = g.sort_values(["fecha", "match_id"]).copy()
        g["matches_played_prior"] = np.arange(1, len(g) + 1)

        # Rachas después del partido jugado
        unbeaten = []
        wins = []
        scoring = []

        unbeaten_streak = 0
        win_streak = 0
        scoring_streak = 0

        for _, row in g.iterrows():
            unbeaten_streak = unbeaten_streak + 1 if row["points"] > 0 else 0
            win_streak = win_streak + 1 if row["win"] == 1 else 0
            scoring_streak = scoring_streak + 1 if row["goals_for"] > 0 else 0

            unbeaten.append(unbeaten_streak)
            wins.append(win_streak)
            scoring.append(scoring_streak)

        g["unbeaten_streak_prior"] = unbeaten
        g["win_streak_prior"] = wins
        g["scoring_streak_prior"] = scoring

        for w in windows:
            for col in numeric_cols:
                g[f"{col}_avg_{w}"] = (
                    g[col]
                    .rolling(window=w, min_periods=1)
                    .mean()
                )

            g[f"points_sum_{w}"] = (
                g["points"]
                .rolling(window=w, min_periods=1)
                .sum()
            )

            weighted_points = g["points"] * g["peso_competicion"]
            g[f"weighted_points_avg_{w}"] = (
                weighted_points
                .rolling(window=w, min_periods=1)
                .mean()
            )

        outputs.append(g)

    state = pd.concat(outputs, ignore_index=True)
    state = state.sort_values(["equipo", "fecha", "match_id"]).reset_index(drop=True)

    return state


# ============================================================
# 3. Cruce de features pre-partido
# ============================================================

def attach_team_state_for_side(
    df_matches: pd.DataFrame,
    team_state: pd.DataFrame,
    team_col: str,
    side_suffix: str,
) -> pd.DataFrame:
    """
    Adjunta al partido el último estado disponible del equipo antes de la fecha.

    Se usa fecha estrictamente menor:
        last_match_date_pre < fecha

    Eso evita usar el partido actual como parte de sus propias features.
    """

    df_base = df_matches.copy()
    df_base["__row_id"] = np.arange(len(df_base))

    state_feature_cols = [
        c for c in team_state.columns
        if c not in {
            "equipo",
            "opponent",
            "match_id",
            "is_home",
            "neutral",
            "torneo",
            "peso_competicion",
            "goals_for",
            "goals_against",
            "goal_diff",
            "points",
            "win",
            "draw",
            "loss",
            "clean_sheet",
            "failed_to_score",
            "btts",
            "over_2_5",
        }
    ]

    pieces = []

    for equipo, target_team_rows in df_base.groupby(team_col, sort=False):
        target_team_rows = target_team_rows.sort_values("fecha").copy()

        state_team = team_state[team_state["equipo"] == equipo].copy()
        state_team = state_team.sort_values("fecha").copy()

        if state_team.empty:
            # Equipo sin historial previo
            tmp = target_team_rows[["__row_id"]].copy()
            for col in state_feature_cols:
                new_col = f"{col}_{side_suffix}"
                tmp[new_col] = np.nan
            pieces.append(tmp)
            continue

        state_small = state_team[state_feature_cols].copy()

        # Renombramos la fecha del estado para conservarla como última fecha previa.
        state_small = state_small.rename(columns={"fecha": f"last_match_date_pre_{side_suffix}"})

        merged = pd.merge_asof(
            target_team_rows[["__row_id", "fecha"]].sort_values("fecha"),
            state_small.sort_values(f"last_match_date_pre_{side_suffix}"),
            left_on="fecha",
            right_on=f"last_match_date_pre_{side_suffix}",
            direction="backward",
            allow_exact_matches=False,
        )

        pieces.append(merged.drop(columns=["fecha"]))

    attached = pd.concat(pieces, ignore_index=True)

    rename_map = {}
    for col in attached.columns:
        if col in {"__row_id", f"last_match_date_pre_{side_suffix}"}:
            continue
        rename_map[col] = f"{col}_{side_suffix}"

    attached = attached.rename(columns=rename_map)

    df_out = df_base.merge(attached, on="__row_id", how="left")
    df_out = df_out.drop(columns=["__row_id"])

    return df_out


def attach_recent_features(
    df_matches: pd.DataFrame,
    team_state: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adjunta features de local y visitante.
    """

    df = df_matches.copy()

    df = attach_team_state_for_side(
        df_matches=df,
        team_state=team_state,
        team_col="equipo_local",
        side_suffix="L",
    )

    df = attach_team_state_for_side(
        df_matches=df,
        team_state=team_state,
        team_col="equipo_visitante",
        side_suffix="V",
    )

    for side in ["L", "V"]:
        date_col = f"last_match_date_pre_{side}"

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[f"days_since_last_match_pre_{side}"] = (
                df["fecha"] - df[date_col]
            ).dt.days

    return df


# ============================================================
# 4. ELO dinámico pre-partido
# ============================================================

def competition_k_factor(weight: float) -> float:
    if pd.isna(weight):
        weight = 0.70
    return 20.0 * (0.75 + float(weight))


def add_dynamic_elo(
    df_matches: pd.DataFrame,
    base_elo: float = 1500.0,
    home_advantage: float = 50.0,
) -> pd.DataFrame:
    """
    Calcula ELO pre-partido.

    Si el partido ya se jugó, actualiza ratings después de registrar el ELO pre.
    Si el partido es futuro, registra el ELO pre pero no actualiza ratings.
    """

    df = df_matches.sort_values(["fecha", "match_id"]).reset_index(drop=True).copy()

    teams = sorted(set(df["equipo_local"]) | set(df["equipo_visitante"]))
    ratings = {team: base_elo for team in teams}

    elo_local_pre = []
    elo_visitante_pre = []
    elo_prob_local_pre = []

    for _, row in df.iterrows():
        home = row["equipo_local"]
        away = row["equipo_visitante"]

        rh = ratings.get(home, base_elo)
        ra = ratings.get(away, base_elo)

        neutral = int(row.get("neutral", 0))
        hfa = 0.0 if neutral == 1 else home_advantage

        expected_home = 1.0 / (1.0 + 10.0 ** (-(rh + hfa - ra) / 400.0))

        elo_local_pre.append(rh)
        elo_visitante_pre.append(ra)
        elo_prob_local_pre.append(expected_home)

        if not bool(row.get("is_played", False)):
            continue

        gh = float(row["goles_local"])
        ga = float(row["goles_visitante"])

        if gh > ga:
            score_home = 1.0
        elif gh == ga:
            score_home = 0.5
        else:
            score_home = 0.0

        margin = abs(gh - ga)
        margin_multiplier = math.log(margin + 1.0) + 1.0
        k = competition_k_factor(row.get("peso_competicion", 0.70))

        delta = k * margin_multiplier * (score_home - expected_home)

        ratings[home] = rh + delta
        ratings[away] = ra - delta

    df["elo_local_pre"] = elo_local_pre
    df["elo_visitante_pre"] = elo_visitante_pre
    df["diff_elo_pre"] = df["elo_local_pre"] - df["elo_visitante_pre"]
    df["elo_prob_local_pre"] = elo_prob_local_pre

    return df


# ============================================================
# 5. H2H pre-partido
# ============================================================

def add_h2h_features(
    df_matches: pd.DataFrame,
    max_matches: int = 5,
    decay: float = 0.35,
) -> pd.DataFrame:
    """
    Historial directo antes de cada partido.

    Solo usa partidos jugados anteriores.
    No agrega partidos futuros al historial.
    """

    df = df_matches.sort_values(["fecha", "match_id"]).reset_index(drop=True).copy()

    history_by_pair: Dict[Tuple[str, str], List[dict]] = {}
    h2h_rows = []

    for _, row in df.iterrows():
        home = row["equipo_local"]
        away = row["equipo_visitante"]
        date = row["fecha"]

        pair = tuple(sorted([home, away]))

        # Usamos solo partidos estrictamente anteriores a la fecha actual.
        hist_all = history_by_pair.get(pair, [])
        hist = [h for h in hist_all if h["fecha"] < date][-max_matches:]

        if not hist:
            h2h_rows.append(
                {
                    "h2h_matches": 0,
                    "h2h_goals_home_avg": np.nan,
                    "h2h_goals_away_avg": np.nan,
                    "h2h_total_goals_avg": np.nan,
                    "h2h_home_win_rate": np.nan,
                    "h2h_weight_sum": 0.0,
                }
            )
        else:
            goals_home_persp = []
            goals_away_persp = []
            home_wins = []
            weights = []

            for h in hist:
                years_old = max((date - h["fecha"]).days / 365.25, 0)
                w = math.exp(-decay * years_old)
                weights.append(w)

                if h["equipo_local"] == home:
                    gh = h["goles_local"]
                    ga = h["goles_visitante"]
                else:
                    gh = h["goles_visitante"]
                    ga = h["goles_local"]

                goals_home_persp.append(gh)
                goals_away_persp.append(ga)
                home_wins.append(1.0 if gh > ga else 0.0)

            weights = np.array(weights, dtype=float)

            h2h_rows.append(
                {
                    "h2h_matches": int(len(hist)),
                    "h2h_goals_home_avg": float(np.average(goals_home_persp, weights=weights)),
                    "h2h_goals_away_avg": float(np.average(goals_away_persp, weights=weights)),
                    "h2h_total_goals_avg": float(
                        np.average(np.array(goals_home_persp) + np.array(goals_away_persp), weights=weights)
                    ),
                    "h2h_home_win_rate": float(np.average(home_wins, weights=weights)),
                    "h2h_weight_sum": float(weights.sum()),
                }
            )

        if bool(row.get("is_played", False)):
            history_by_pair.setdefault(pair, []).append(
                {
                    "fecha": date,
                    "equipo_local": home,
                    "equipo_visitante": away,
                    "goles_local": float(row["goles_local"]),
                    "goles_visitante": float(row["goles_visitante"]),
                }
            )

    h2h_df = pd.DataFrame(h2h_rows)
    df_out = pd.concat([df.reset_index(drop=True), h2h_df.reset_index(drop=True)], axis=1)

    return df_out


# ============================================================
# 6. Diferenciales local - visitante
# ============================================================

def add_differential_features(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Crea diferencias local - visitante para features que tienen sufijo _L y _V.
    """

    df = df_matches.copy()

    for col in list(df.columns):
        if not col.endswith("_L"):
            continue

        base = col[:-2]
        col_v = f"{base}_V"

        if col_v not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[col_v]):
            df[f"diff_{base}"] = df[col] - df[col_v]

    return df


# ============================================================
# 7. Verificación anti-leakage
# ============================================================

def check_temporal_leakage(df_matches: pd.DataFrame) -> dict:
    """
    Verifica que las fechas de último partido previo sean menores que la fecha actual.
    """

    df = df_matches.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    report = {}

    total = 0

    for side in ["L", "V"]:
        col = f"last_match_date_pre_{side}"

        if col not in df.columns:
            report[f"leakage_{side}"] = None
            continue

        tmp = pd.to_datetime(df[col], errors="coerce")
        mask = tmp.notna() & (tmp >= df["fecha"])

        n = int(mask.sum())
        report[f"leakage_{side}"] = n
        total += n

    report["leakage_total"] = int(total)

    return report


# ============================================================
# 8. Orquestador de Fase 2
# ============================================================

def build_phase02_features(
    matches_clean_path: str | Path,
    output_dir: str | Path,
    reports_dir: str | Path,
    start_year: int = 2010,
) -> dict:
    """
    Construye features completas y separa:
        - dataset de entrenamiento;
        - dataset pendiente para predicción.
    """

    matches_clean_path = Path(matches_clean_path)
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not matches_clean_path.exists():
        raise FileNotFoundError(f"No existe: {matches_clean_path}")

    df = pd.read_parquet(matches_clean_path)
    df = ensure_datetime(df, "fecha")
    df = df.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    # Construimos estados con TODO el historial jugado.
    team_history = build_team_match_history(df)
    team_state = add_team_state_features(team_history)

    team_state_path = output_dir / "team_state_history.parquet"
    team_state.to_parquet(team_state_path, index=False)

    # ELO se calcula sobre todo el historial para que llegue bien a 2010+.
    df_model = add_dynamic_elo(df)
    df_model = attach_recent_features(df_model, team_state)
    df_model = add_h2h_features(df_model)
    df_model = add_differential_features(df_model)

    # Nos quedamos con fútbol moderno para entrenamiento/predicción operativa.
    df_model = df_model[df_model["fecha"].dt.year >= start_year].copy()
    df_model = df_model.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    leakage_report = check_temporal_leakage(df_model)

    # Split lógico:
    # - train: jugados desde start_year;
    # - pending: no jugados desde start_year, normalmente fixture Mundial 2026.
    df_train = df_model[df_model["is_played"]].copy()
    df_pending = df_model[~df_model["is_played"]].copy()

    all_path = output_dir / "modeling_dataset_all.parquet"
    train_path = output_dir / "modeling_dataset_train.parquet"
    pending_path = output_dir / "modeling_dataset_pending.parquet"

    all_csv = output_dir / "modeling_dataset_all.csv"
    train_csv = output_dir / "modeling_dataset_train.csv"
    pending_csv = output_dir / "modeling_dataset_pending.csv"

    df_model.to_parquet(all_path, index=False)
    df_train.to_parquet(train_path, index=False)
    df_pending.to_parquet(pending_path, index=False)

    df_model.to_csv(all_csv, index=False, encoding="utf-8")
    df_train.to_csv(train_csv, index=False, encoding="utf-8")
    df_pending.to_csv(pending_csv, index=False, encoding="utf-8")

    feature_cols = [
        c for c in df_model.columns
        if c not in {
            "match_id",
            "fecha",
            "equipo_local",
            "equipo_visitante",
            "goles_local",
            "goles_visitante",
            "resultado_1x2",
            "over_2_5",
            "btts",
            "torneo",
            "ciudad",
            "pais_sede",
            "source_id",
            "is_played",
        }
    ]

    null_report = (
        df_train[feature_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    null_report.columns = ["feature", "null_rate"]
    null_report_path = reports_dir / "phase02_feature_null_report.csv"
    null_report.to_csv(null_report_path, index=False, encoding="utf-8")

    report = {
        "created_at": now_iso(),
        "start_year": int(start_year),
        "input": str(matches_clean_path),
        "n_rows_all": int(len(df_model)),
        "n_rows_train_played": int(len(df_train)),
        "n_rows_pending": int(len(df_pending)),
        "date_min_all": str(df_model["fecha"].min()),
        "date_max_all": str(df_model["fecha"].max()),
        "date_min_train": str(df_train["fecha"].min()),
        "date_max_train": str(df_train["fecha"].max()),
        "n_features_candidate": int(len(feature_cols)),
        "leakage_report": leakage_report,
        "outputs": {
            "team_state_history": str(team_state_path),
            "modeling_dataset_all": str(all_path),
            "modeling_dataset_train": str(train_path),
            "modeling_dataset_pending": str(pending_path),
            "null_report": str(null_report_path),
        },
    }

    report_path = reports_dir / "phase02_features_report.json"
    report_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "df_all": df_model,
        "df_train": df_train,
        "df_pending": df_pending,
        "team_state": team_state,
        "report": report,
    }
