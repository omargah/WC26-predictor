
# -*- coding: utf-8 -*-
"""
FASE 2 — Feature engineering temporal para Mundial 2026.

Este módulo construye el dataset de modelado a partir de los partidos limpios.

Entrada principal:
    data/processed/partidos_selecciones.parquet

Salidas:
    data/features/team_match_history.parquet
    data/features/modeling_dataset_raw.parquet
    data/features/modeling_dataset.parquet
    data/features/modeling_dataset.csv
    reports/phase02_report.txt
    reports/phase02_report.json

Principio central:
    Toda feature histórica debe calcularse con información anterior al partido.
    Por eso usamos shift(1) antes de rolling(...).

Además:
    Integramos una validación strict para eliminar casos raros de leakage,
    por ejemplo selecciones con más de un partido en la misma fecha.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 1. UTILIDADES GENERALES
# ============================================================

def now_iso() -> str:
    """
    Devuelve timestamp legible para reportes.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_phase02_paths(project_root: str | Path) -> dict:
    """
    Construye las rutas que usará la Fase 2.

    Mantener las rutas aquí evita escribir paths manuales en varias partes.
    """

    root = Path(project_root)

    paths = {
        "root": root,
        "processed": root / "data" / "processed",
        "features": root / "data" / "features",
        "manual": root / "data" / "manual",
        "reports": root / "reports",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def safe_write_json(obj: dict, path: str | Path) -> None:
    """
    Guarda un diccionario como JSON con codificación UTF-8.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, indent=4, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ============================================================
# 2. CARGA DEL DATASET BASE
# ============================================================

def load_matches(project_root: str | Path) -> pd.DataFrame:
    """
    Carga partidos_selecciones.parquet.

    Si todavía no existe la capa compatible, intenta crearla desde
    matches_clean.parquet. Esto hace que la Fase 2 sea más robusta.
    """

    paths = get_phase02_paths(project_root)

    compat_path = paths["processed"] / "partidos_selecciones.parquet"
    clean_path = paths["processed"] / "matches_clean.parquet"

    if compat_path.exists():
        df = pd.read_parquet(compat_path)
    elif clean_path.exists():
        df_clean = pd.read_parquet(clean_path)
        df = create_compatibility_view(df_clean)
        df.to_parquet(compat_path, index=False)
        df.to_csv(paths["processed"] / "partidos_selecciones.csv", index=False, encoding="utf-8")
    else:
        raise FileNotFoundError(
            "No existe partidos_selecciones.parquet ni matches_clean.parquet. "
            "Ejecuta primero la Fase 1."
        )

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df[df["fecha"].notna()].copy()

    required = [
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
        "neutral",
        "peso_competicion",
        "source_id",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Faltan columnas necesarias para Fase 2: "
            + ", ".join(missing)
        )

    df = df.sort_values("fecha").reset_index(drop=True)

    return df


def assign_competition_weight(tournament) -> float:
    """
    Asigna peso de competición.

    Esta función se usa solo como respaldo si se necesita reconstruir
    partidos_selecciones desde matches_clean.
    """

    if pd.isna(tournament):
        return 0.70

    t = str(tournament).lower()

    if t in ["fifa world cup", "world cup"]:
        return 1.00

    if "world cup qualification" in t or "world cup qualifier" in t:
        return 0.85

    if "uefa euro" in t or "european championship" in t:
        return 0.95

    if "copa américa" in t or "copa america" in t:
        return 0.90

    if "nations league" in t:
        return 0.80

    if "gold cup" in t:
        return 0.80

    if "africa cup" in t or "afcon" in t:
        return 0.80

    if "asian cup" in t:
        return 0.75

    if "friendly" in t:
        return 0.60

    return 0.70


def create_compatibility_view(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Crea esquema compatible con los Colabs anteriores.

    El nuevo dataset usa columnas en inglés; la Fase 2 conserva el esquema
    anterior en español para facilitar continuidad metodológica.
    """

    import hashlib

    df = df_clean.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.rename(
        columns={
            "date": "fecha",
            "home_team": "equipo_local",
            "away_team": "equipo_visitante",
            "home_goals": "goles_local",
            "away_goals": "goles_visitante",
            "result_1x2": "resultado_1x2",
            "tournament": "torneo",
            "city": "ciudad",
            "country": "pais_sede",
        }
    )

    df["over_2_5"] = ((df["goles_local"] + df["goles_visitante"]) >= 3).astype(int)
    df["btts"] = ((df["goles_local"] > 0) & (df["goles_visitante"] > 0)).astype(int)
    df["peso_competicion"] = df["torneo"].apply(assign_competition_weight)
    df["source_id"] = "international_results_martj42"

    def make_match_id(row):
        raw = f"{row['fecha']}|{row['equipo_local']}|{row['equipo_visitante']}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    df["match_id"] = df.apply(make_match_id, axis=1)

    keep_cols = [
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
        "neutral",
        "peso_competicion",
        "source_id",
    ]

    return df[keep_cols].sort_values("fecha").reset_index(drop=True)


# ============================================================
# 3. TABLA EQUIPO-PARTIDO
# ============================================================

def build_team_match_history(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte una tabla de partidos en una tabla equipo-partido.

    Cada partido genera dos filas:
    - perspectiva del local;
    - perspectiva del visitante.

    Por qué:
        Las features de forma reciente se calculan por selección, no por
        partido. Para saber cómo venía México antes de enfrentar a otro equipo,
        necesitamos una fila histórica de México en cada partido anterior.
    """

    rows = []

    for idx, row in df_matches.iterrows():
        gl = int(row["goles_local"])
        gv = int(row["goles_visitante"])

        total_goals = gl + gv
        over25 = int(total_goals >= 3)
        btts = int(gl > 0 and gv > 0)

        # Perspectiva del equipo local
        rows.append({
            "match_id": row["match_id"],
            "match_index": idx,
            "fecha": row["fecha"],
            "equipo": row["equipo_local"],
            "opponent": row["equipo_visitante"],
            "side": "L",
            "is_home": 1,
            "neutral": int(row.get("neutral", 0)),
            "torneo": row.get("torneo", np.nan),
            "peso_competicion": float(row.get("peso_competicion", 0.70)),
            "goals_for": gl,
            "goals_against": gv,
            "goal_diff": gl - gv,
            "points": 3 if gl > gv else (1 if gl == gv else 0),
            "win": int(gl > gv),
            "draw": int(gl == gv),
            "loss": int(gl < gv),
            "clean_sheet": int(gv == 0),
            "failed_to_score": int(gl == 0),
            "btts": btts,
            "over_2_5": over25,
        })

        # Perspectiva del equipo visitante
        rows.append({
            "match_id": row["match_id"],
            "match_index": idx,
            "fecha": row["fecha"],
            "equipo": row["equipo_visitante"],
            "opponent": row["equipo_local"],
            "side": "V",
            "is_home": 0,
            "neutral": int(row.get("neutral", 0)),
            "torneo": row.get("torneo", np.nan),
            "peso_competicion": float(row.get("peso_competicion", 0.70)),
            "goals_for": gv,
            "goals_against": gl,
            "goal_diff": gv - gl,
            "points": 3 if gv > gl else (1 if gl == gv else 0),
            "win": int(gv > gl),
            "draw": int(gl == gv),
            "loss": int(gv < gl),
            "clean_sheet": int(gl == 0),
            "failed_to_score": int(gv == 0),
            "btts": btts,
            "over_2_5": over25,
        })

    team_df = pd.DataFrame(rows)

    team_df = team_df.sort_values(
        ["equipo", "fecha", "match_index"]
    ).reset_index(drop=True)

    return team_df


# ============================================================
# 4. ROLLING FEATURES SIN LEAKAGE
# ============================================================

def _streak_before(values, condition) -> list[int]:
    """
    Calcula una racha previa.

    La fila actual recibe la racha acumulada ANTES de jugar ese partido.
    Esto evita usar el resultado del partido actual como parte de su propia
    explicación.
    """

    out = []
    streak = 0

    for value in values:
        out.append(streak)

        if condition(value):
            streak += 1
        else:
            streak = 0

    return out


def add_rolling_features(team_df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """
    Calcula forma reciente con ventanas móviles.

    Punto clave:
        g[col].shift(1).rolling(...)

    El shift(1) significa:
        "para el partido actual, calcula la media usando partidos anteriores,
        no incluyendo el actual".
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

    output = []

    for equipo, g in team_df.groupby("equipo", sort=False):
        g = g.sort_values(["fecha", "match_index"]).copy()

        g["prev_matches_count"] = np.arange(len(g))
        g["last_match_date_pre"] = g["fecha"].shift(1)

        g["unbeaten_streak_pre"] = _streak_before(
            g["points"].values,
            lambda p: p > 0,
        )

        g["win_streak_pre"] = _streak_before(
            g["win"].values,
            lambda w: w == 1,
        )

        g["scoring_streak_pre"] = _streak_before(
            g["goals_for"].values,
            lambda gf: gf > 0,
        )

        for w in windows:
            for col in numeric_cols:
                g[f"{col}_avg_{w}"] = (
                    g[col]
                    .shift(1)
                    .rolling(window=w, min_periods=1)
                    .mean()
                )

            g[f"points_sum_{w}"] = (
                g["points"]
                .shift(1)
                .rolling(window=w, min_periods=1)
                .sum()
            )

            weighted_points = g["points"] * g["peso_competicion"]

            g[f"weighted_points_avg_{w}"] = (
                weighted_points
                .shift(1)
                .rolling(window=w, min_periods=1)
                .mean()
            )

        output.append(g)

    team_roll = pd.concat(output, ignore_index=True)

    team_roll = team_roll.sort_values(
        ["fecha", "match_index", "side"]
    ).reset_index(drop=True)

    return team_roll


def merge_team_features_to_matches(
    df_matches: pd.DataFrame,
    team_roll: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cruza features de equipo hacia la tabla de partidos.

    Para cada partido, agregamos:
    - features del equipo local con sufijo _L;
    - features del equipo visitante con sufijo _V.
    """

    feature_cols = [
        c for c in team_roll.columns
        if (
            c.endswith("_pre")
            or "_avg_" in c
            or c.startswith("points_sum_")
            or c.startswith("weighted_points_avg_")
            or c == "prev_matches_count"
        )
    ]

    local = (
        team_roll[team_roll["side"] == "L"][["match_id"] + feature_cols]
        .rename(columns={c: f"{c}_L" for c in feature_cols})
    )

    visitante = (
        team_roll[team_roll["side"] == "V"][["match_id"] + feature_cols]
        .rename(columns={c: f"{c}_V" for c in feature_cols})
    )

    df = df_matches.merge(local, on="match_id", how="left")
    df = df.merge(visitante, on="match_id", how="left")

    for side in ["L", "V"]:
        date_col = f"last_match_date_pre_{side}"

        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[f"days_since_last_match_pre_{side}"] = (
                pd.to_datetime(df["fecha"]) - df[date_col]
            ).dt.days

    return df


# ============================================================
# 5. ELO DINÁMICO PRE-PARTIDO
# ============================================================

def competition_k_factor(weight) -> float:
    """
    Define un K dinámico para ELO según importancia del partido.
    """

    if pd.isna(weight):
        weight = 0.70

    return 20.0 * (0.75 + float(weight))


def add_dynamic_elo(
    df: pd.DataFrame,
    base_elo: float = 1500.0,
    home_advantage: float = 50.0,
) -> pd.DataFrame:
    """
    Calcula ELO pre-partido.

    Punto clave:
        El ELO que entra como feature es el ELO ANTES de actualizar con
        el resultado del partido actual.
    """

    df = df.sort_values("fecha").reset_index(drop=True).copy()

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

        expected_home = 1.0 / (
            1.0 + 10.0 ** (-(rh + hfa - ra) / 400.0)
        )

        elo_local_pre.append(rh)
        elo_visitante_pre.append(ra)
        elo_prob_local_pre.append(expected_home)

        gh = int(row["goles_local"])
        ga = int(row["goles_visitante"])

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
# 6. H2H CON DECAIMIENTO TEMPORAL
# ============================================================

def add_h2h_features(
    df: pd.DataFrame,
    max_matches: int = 5,
    decay: float = 0.35,
) -> pd.DataFrame:
    """
    Calcula historial directo H2H antes de cada partido.

    Mejora respecto al primer Colab:
        Usamos un diccionario por pareja de equipos para no recorrer todo
        el dataset anterior en cada fila. Esto conserva la lógica, pero es
        más eficiente.
    """

    df = df.sort_values("fecha").reset_index(drop=True).copy()

    history_by_pair: Dict[Tuple[str, str], List[dict]] = {}
    h2h_rows = []

    for _, row in df.iterrows():
        home = row["equipo_local"]
        away = row["equipo_visitante"]
        date = row["fecha"]

        pair = tuple(sorted([home, away]))
        hist = history_by_pair.get(pair, [])[-max_matches:]

        if not hist:
            h2h_rows.append({
                "h2h_matches": 0,
                "h2h_goals_home_avg": np.nan,
                "h2h_goals_away_avg": np.nan,
                "h2h_total_goals_avg": np.nan,
                "h2h_home_win_rate": np.nan,
                "h2h_weight_sum": 0.0,
            })
        else:
            goals_home_perspective = []
            goals_away_perspective = []
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

                goals_home_perspective.append(gh)
                goals_away_perspective.append(ga)
                home_wins.append(1 if gh > ga else 0)

            weights = np.array(weights)

            h2h_home_avg = np.average(goals_home_perspective, weights=weights)
            h2h_away_avg = np.average(goals_away_perspective, weights=weights)

            h2h_rows.append({
                "h2h_matches": len(hist),
                "h2h_goals_home_avg": float(h2h_home_avg),
                "h2h_goals_away_avg": float(h2h_away_avg),
                "h2h_total_goals_avg": float(h2h_home_avg + h2h_away_avg),
                "h2h_home_win_rate": float(np.average(home_wins, weights=weights)),
                "h2h_weight_sum": float(weights.sum()),
            })

        current = {
            "fecha": date,
            "equipo_local": home,
            "equipo_visitante": away,
            "goles_local": int(row["goles_local"]),
            "goles_visitante": int(row["goles_visitante"]),
        }

        history_by_pair.setdefault(pair, []).append(current)

    h2h_df = pd.DataFrame(h2h_rows)
    df = pd.concat([df.reset_index(drop=True), h2h_df], axis=1)

    return df


# ============================================================
# 7. VARIABLES DIFERENCIALES LOCAL - VISITANTE
# ============================================================

def add_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea diferenciales local - visitante.

    Por qué:
        Muchos modelos funcionan mejor con diferencias de fuerza relativa:
        forma_local - forma_visitante, ELO_local - ELO_visitante, etc.
    """

    candidate_pairs = [
        "goals_for_avg_5",
        "goals_against_avg_5",
        "goal_diff_avg_5",
        "points_avg_5",
        "win_avg_5",
        "draw_avg_5",
        "loss_avg_5",
        "clean_sheet_avg_5",
        "failed_to_score_avg_5",
        "btts_avg_5",
        "over_2_5_avg_5",
        "points_sum_5",
        "weighted_points_avg_5",

        "goals_for_avg_10",
        "goals_against_avg_10",
        "goal_diff_avg_10",
        "points_avg_10",
        "win_avg_10",
        "draw_avg_10",
        "loss_avg_10",
        "clean_sheet_avg_10",
        "failed_to_score_avg_10",
        "btts_avg_10",
        "over_2_5_avg_10",
        "points_sum_10",
        "weighted_points_avg_10",

        "goals_for_avg_20",
        "goals_against_avg_20",
        "goal_diff_avg_20",
        "points_avg_20",
        "win_avg_20",
        "draw_avg_20",
        "loss_avg_20",
        "clean_sheet_avg_20",
        "failed_to_score_avg_20",
        "btts_avg_20",
        "over_2_5_avg_20",
        "points_sum_20",
        "weighted_points_avg_20",

        "prev_matches_count",
        "unbeaten_streak_pre",
        "win_streak_pre",
        "scoring_streak_pre",
        "days_since_last_match_pre",
    ]

    for base in candidate_pairs:
        col_l = f"{base}_L"
        col_v = f"{base}_V"

        if col_l in df.columns and col_v in df.columns:
            df[f"diff_{base}"] = df[col_l] - df[col_v]

    return df


# ============================================================
# 8. VALIDACIÓN STRICT ANTI-LEAKAGE
# ============================================================

def leakage_masks(df: pd.DataFrame) -> dict:
    """
    Detecta filas donde la fecha del último partido previo no es estrictamente
    anterior a la fecha del partido actual.
    """

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    masks = {}

    for side in ["L", "V"]:
        col = f"last_match_date_pre_{side}"

        if col not in df.columns:
            masks[side] = pd.Series(False, index=df.index)
            continue

        tmp_col = pd.to_datetime(df[col], errors="coerce")

        masks[side] = (
            tmp_col.notna()
            &
            (tmp_col >= df["fecha"])
        )

    return masks


def strict_remove_leakage(df_model: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Elimina filas con posible leakage temporal.

    En el primer Colab esto se hizo como parche posterior.
    Aquí lo integramos dentro de la Fase 2 para que la salida final ya sea strict.
    """

    masks = leakage_masks(df_model)

    mask_leak_L = masks["L"]
    mask_leak_V = masks["V"]
    mask_leak = mask_leak_L | mask_leak_V

    report = {
        "n_original": int(len(df_model)),
        "n_leak_L": int(mask_leak_L.sum()),
        "n_leak_V": int(mask_leak_V.sum()),
        "n_leak_total": int(mask_leak.sum()),
    }

    df_strict = df_model.loc[~mask_leak].copy().reset_index(drop=True)

    masks_after = leakage_masks(df_strict)

    report["n_final"] = int(len(df_strict))
    report["leak_after_L"] = int(masks_after["L"].sum())
    report["leak_after_V"] = int(masks_after["V"].sum())

    return df_strict, report


# ============================================================
# 9. REPORTE DE FASE 2
# ============================================================

def build_phase02_report(
    df_raw_model: pd.DataFrame,
    df_strict: pd.DataFrame,
    team_history: pd.DataFrame,
    strict_report: dict,
    paths: dict,
) -> dict:
    """
    Construye reporte de la Fase 2.
    """

    non_feature_cols = {
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
    }

    feature_cols = [
        col for col in df_strict.columns
        if col not in non_feature_cols
    ]

    null_report = (
        df_strict[feature_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    null_report.columns = ["feature", "null_rate"]

    null_report_path = paths["reports"] / "phase02_null_report.csv"
    null_report.to_csv(null_report_path, index=False)

    report = {
        "created_at": now_iso(),
        "n_matches_raw_model": int(len(df_raw_model)),
        "n_matches_strict": int(len(df_strict)),
        "n_team_match_rows": int(len(team_history)),
        "n_columns": int(len(df_strict.columns)),
        "n_feature_columns": int(len(feature_cols)),
        "date_min": str(df_strict["fecha"].min()),
        "date_max": str(df_strict["fecha"].max()),
        "strict_report": strict_report,
        "top_null_features": null_report.head(25).to_dict(orient="records"),
        "output_files": {
            "null_report": str(null_report_path),
            "team_match_history": str(paths["features"] / "team_match_history.parquet"),
            "modeling_dataset": str(paths["features"] / "modeling_dataset.parquet"),
        },
    }

    json_path = paths["reports"] / "phase02_report.json"
    txt_path = paths["reports"] / "phase02_report.txt"

    safe_write_json(report, json_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("REPORTE FASE 2 — FEATURE ENGINEERING\n")
        f.write("=" * 70 + "\n")
        f.write(f"Creado: {report['created_at']}\n")
        f.write(f"Partidos antes de strict: {report['n_matches_raw_model']}\n")
        f.write(f"Partidos strict finales: {report['n_matches_strict']}\n")
        f.write(f"Filas equipo-partido: {report['n_team_match_rows']}\n")
        f.write(f"Columnas totales: {report['n_columns']}\n")
        f.write(f"Columnas candidatas de features: {report['n_feature_columns']}\n")
        f.write(f"Rango fechas: {report['date_min']} → {report['date_max']}\n\n")

        f.write("Validación strict anti-leakage:\n")
        f.write(f"  - Leakage lado local: {strict_report['n_leak_L']}\n")
        f.write(f"  - Leakage lado visitante: {strict_report['n_leak_V']}\n")
        f.write(f"  - Filas únicas eliminadas: {strict_report['n_leak_total']}\n")
        f.write(f"  - Leakage restante local: {strict_report['leak_after_L']}\n")
        f.write(f"  - Leakage restante visitante: {strict_report['leak_after_V']}\n\n")

        f.write("Top features con mayor tasa de nulos:\n")
        for row in report["top_null_features"]:
            f.write(f"  - {row['feature']}: {row['null_rate']:.2%}\n")

    return report


# ============================================================
# 10. FUNCIÓN PRINCIPAL
# ============================================================

def run_phase02(project_root: str | Path) -> dict:
    """
    Ejecuta la Fase 2 completa.
    """

    print("=" * 80)
    print("INICIANDO FASE 2 — FEATURES TEMPORALES + ELO + H2H")
    print("=" * 80)

    paths = get_phase02_paths(project_root)

    print("\n[2.1] Cargando partidos limpios compatibles...")
    df_matches = load_matches(project_root)
    print(f"   ✓ Partidos cargados: {len(df_matches):,}")
    print(f"   ✓ Rango: {df_matches['fecha'].min()} → {df_matches['fecha'].max()}")

    print("\n[2.2] Construyendo tabla equipo-partido...")
    team_history = build_team_match_history(df_matches)
    print(f"   ✓ Filas equipo-partido: {len(team_history):,}")

    print("\n[2.3] Calculando rolling features sin leakage...")
    team_history_features = add_rolling_features(team_history, windows=(5, 10, 20))
    print("   ✓ Rolling features calculadas")

    team_history_path = paths["features"] / "team_match_history.parquet"
    team_history_features.to_parquet(team_history_path, index=False)
    print(f"   ✓ Guardado: {team_history_path}")

    print("\n[2.4] Cruzando features al dataset de partidos...")
    df_model = merge_team_features_to_matches(df_matches, team_history_features)
    print(f"   ✓ Dataset parcial: {df_model.shape}")

    print("\n[2.5] Calculando ELO dinámico pre-partido...")
    df_model = add_dynamic_elo(df_model)
    print("   ✓ ELO agregado")

    print("\n[2.6] Calculando H2H con decaimiento temporal...")
    df_model = add_h2h_features(df_model, max_matches=5, decay=0.35)
    print("   ✓ H2H agregado")

    print("\n[2.7] Creando variables diferenciales local - visitante...")
    df_model = add_differential_features(df_model)
    print(f"   ✓ Dataset con diferenciales: {df_model.shape}")

    raw_model_path = paths["features"] / "modeling_dataset_raw.parquet"
    df_model.to_parquet(raw_model_path, index=False)
    print(f"   ✓ Dataset raw guardado: {raw_model_path}")

    print("\n[2.8] Aplicando validación strict anti-leakage...")
    df_strict, strict_report = strict_remove_leakage(df_model)
    print(f"   ✓ Filas eliminadas por leakage: {strict_report['n_leak_total']}")
    print(f"   ✓ Partidos finales strict: {len(df_strict):,}")

    print("\n[2.9] Guardando dataset final de modelado...")
    out_parquet = paths["features"] / "modeling_dataset.parquet"
    out_csv = paths["features"] / "modeling_dataset.csv"

    df_strict.to_parquet(out_parquet, index=False)
    df_strict.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"   ✓ Parquet: {out_parquet}")
    print(f"   ✓ CSV: {out_csv}")

    print("\n[2.10] Construyendo reporte...")
    report = build_phase02_report(
        df_raw_model=df_model,
        df_strict=df_strict,
        team_history=team_history_features,
        strict_report=strict_report,
        paths=paths,
    )

    print(f"   ✓ Reporte: {paths['reports'] / 'phase02_report.txt'}")

    print("\n" + "=" * 80)
    print("FASE 2 COMPLETADA")
    print("=" * 80)
    print(f"Partidos finales: {len(df_strict):,}")
    print(f"Columnas finales: {len(df_strict.columns):,}")
    print(f"Dataset final: {out_parquet}")

    return {
        "matches": df_matches,
        "team_history": team_history_features,
        "modeling_dataset_raw": df_model,
        "modeling_dataset": df_strict,
        "strict_report": strict_report,
        "report": report,
        "paths": {k: str(v) for k, v in paths.items()},
    }
