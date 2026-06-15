
# -*- coding: utf-8 -*-

"""
FASE 4.8 — Predictor completo de partido

Combina:
    - Modelo de goles Poisson + Dixon-Coles
    - Modelo estable de córners
    - Modelo estable de tarjetas amarillas

Entradas:
    data/features/modeling_dataset_advanced.parquet
    data/features/team_match_history.parquet
    data/features/team_advanced_history.parquet
    models/poisson_dc_base.joblib
    models/corners_cards_poisson_stable.joblib

Salida:
    Reporte completo de partido.
"""

from pathlib import Path
from datetime import datetime
import hashlib
import math
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson

from phase03_poisson_dc import predict_from_feature_row


DEFAULT_PROJECT_ROOT = Path("/content/drive/MyDrive/Mundial_2026_Model")


# ============================================================
# UTILIDADES
# ============================================================

def make_future_match_id(equipo_local, equipo_visitante, fecha, torneo):
    raw = f"{fecha}|{equipo_local}|{equipo_visitante}|{torneo}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def assign_competition_weight(torneo, fase=None):
    torneo_l = str(torneo).lower() if torneo is not None else ""
    fase_l = str(fase).lower() if fase is not None else ""

    if "world cup" in torneo_l or "mundial" in torneo_l:
        if "final" in fase_l:
            return 1.10
        if "semi" in fase_l or "quarter" in fase_l or "octavos" in fase_l or "knockout" in fase_l:
            return 1.05
        return 1.00

    if "qualifier" in torneo_l or "qualification" in torneo_l:
        return 0.85

    if "friendly" in torneo_l:
        return 0.60

    return 0.70


def poisson_over(lambda_total, line):
    lambda_total = float(np.clip(lambda_total, 0.01, 50.0))
    k = int(np.floor(line))
    return float(poisson.sf(k, lambda_total))


def poisson_under(lambda_total, line):
    return 1.0 - poisson_over(lambda_total, line)


def market_probs(lambda_total, lines):
    out = {}

    for line in lines:
        key = str(line).replace(".", "_")
        out[f"over_{key}"] = poisson_over(lambda_total, line)
        out[f"under_{key}"] = poisson_under(lambda_total, line)

    return out


def load_assets(project_root=DEFAULT_PROJECT_ROOT):
    project_root = Path(project_root)

    df_adv = pd.read_parquet(project_root / "data/features/modeling_dataset_advanced.parquet")
    team_history = pd.read_parquet(project_root / "data/features/team_match_history.parquet")
    team_adv_history = pd.read_parquet(project_root / "data/features/team_advanced_history.parquet")

    goal_bundle = joblib.load(project_root / "models/poisson_dc_base.joblib")
    corners_cards_bundle = joblib.load(project_root / "models/corners_cards_poisson_stable.joblib")

    df_adv["fecha"] = pd.to_datetime(df_adv["fecha"], errors="coerce")
    team_history["fecha"] = pd.to_datetime(team_history["fecha"], errors="coerce")
    team_adv_history["fecha"] = pd.to_datetime(team_adv_history["fecha"], errors="coerce")

    df_adv = df_adv[df_adv["fecha"].notna()].sort_values("fecha").reset_index(drop=True)
    team_history = team_history[team_history["fecha"].notna()].sort_values("fecha").reset_index(drop=True)
    team_adv_history = team_adv_history[team_adv_history["fecha"].notna()].sort_values("fecha").reset_index(drop=True)

    return df_adv, team_history, team_adv_history, goal_bundle, corners_cards_bundle


# ============================================================
# FEATURES BASE DE GOLES
# ============================================================

def _streak_from_history(values, condition):
    streak = 0

    for v in reversed(list(values)):
        if condition(v):
            streak += 1
        else:
            break

    return streak


def compute_team_pre_features(team_history, equipo, fecha_partido, side_suffix):
    fecha_partido = pd.to_datetime(fecha_partido)

    hist = team_history[
        (team_history["equipo"] == equipo)
        & (team_history["fecha"] < fecha_partido)
    ].sort_values("fecha").copy()

    out = {}

    out[f"prev_matches_count_{side_suffix}"] = len(hist)

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

    if len(hist) == 0:
        out[f"last_match_date_pre_{side_suffix}"] = pd.NaT
        out[f"unbeaten_streak_pre_{side_suffix}"] = 0
        out[f"win_streak_pre_{side_suffix}"] = 0
        out[f"scoring_streak_pre_{side_suffix}"] = 0

        for w in [5, 10, 20]:
            for col in numeric_cols:
                out[f"{col}_avg_{w}_{side_suffix}"] = np.nan

            out[f"points_sum_{w}_{side_suffix}"] = np.nan
            out[f"weighted_points_avg_{w}_{side_suffix}"] = np.nan

        return out

    out[f"last_match_date_pre_{side_suffix}"] = hist["fecha"].max()

    out[f"unbeaten_streak_pre_{side_suffix}"] = _streak_from_history(
        hist["points"].values,
        lambda p: p > 0
    )

    out[f"win_streak_pre_{side_suffix}"] = _streak_from_history(
        hist["win"].values,
        lambda w: w == 1
    )

    out[f"scoring_streak_pre_{side_suffix}"] = _streak_from_history(
        hist["goals_for"].values,
        lambda gf: gf > 0
    )

    for w in [5, 10, 20]:
        last_w = hist.tail(w)

        for col in numeric_cols:
            out[f"{col}_avg_{w}_{side_suffix}"] = last_w[col].mean()

        out[f"points_sum_{w}_{side_suffix}"] = last_w["points"].sum()
        out[f"weighted_points_avg_{w}_{side_suffix}"] = (
            last_w["points"] * last_w["peso_competicion"]
        ).mean()

    return out


# ============================================================
# ELO Y H2H
# ============================================================

def competition_k_factor(weight):
    if pd.isna(weight):
        weight = 0.70
    return 20.0 * (0.75 + float(weight))


def compute_elo_as_of(df, fecha_partido, base_elo=1500.0, home_advantage=50.0):
    fecha_partido = pd.to_datetime(fecha_partido)

    hist = df[df["fecha"] < fecha_partido].sort_values("fecha").copy()

    teams = sorted(set(hist["equipo_local"]) | set(hist["equipo_visitante"]))
    ratings = {t: base_elo for t in teams}

    for _, row in hist.iterrows():
        home = row["equipo_local"]
        away = row["equipo_visitante"]

        rh = ratings.get(home, base_elo)
        ra = ratings.get(away, base_elo)

        neutral = int(row.get("neutral", 0))
        hfa = 0.0 if neutral == 1 else home_advantage

        expected_home = 1.0 / (1.0 + 10.0 ** (-(rh + hfa - ra) / 400.0))

        gh = row["goles_local"]
        ga = row["goles_visitante"]

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

    return ratings


def add_future_elo_features(row, df, equipo_local, equipo_visitante, fecha_partido, neutral):
    ratings = compute_elo_as_of(df, fecha_partido)

    elo_l = ratings.get(equipo_local, 1500.0)
    elo_v = ratings.get(equipo_visitante, 1500.0)

    hfa = 0.0 if int(neutral) == 1 else 50.0

    elo_prob_local = 1.0 / (1.0 + 10.0 ** (-(elo_l + hfa - elo_v) / 400.0))

    row["elo_local_pre"] = elo_l
    row["elo_visitante_pre"] = elo_v
    row["diff_elo_pre"] = elo_l - elo_v
    row["elo_prob_local_pre"] = elo_prob_local

    return row


def add_future_h2h_features(row, df, equipo_local, equipo_visitante, fecha_partido, max_matches=5, decay=0.35):
    fecha_partido = pd.to_datetime(fecha_partido)

    hist = df[
        (
            ((df["equipo_local"] == equipo_local) & (df["equipo_visitante"] == equipo_visitante))
            |
            ((df["equipo_local"] == equipo_visitante) & (df["equipo_visitante"] == equipo_local))
        )
        &
        (df["fecha"] < fecha_partido)
    ].sort_values("fecha").tail(max_matches).copy()

    if len(hist) == 0:
        row["h2h_matches"] = 0
        row["h2h_goals_home_avg"] = np.nan
        row["h2h_goals_away_avg"] = np.nan
        row["h2h_total_goals_avg"] = np.nan
        row["h2h_home_win_rate"] = np.nan
        row["h2h_weight_sum"] = 0.0
        return row

    goals_home_perspective = []
    goals_away_perspective = []
    home_wins = []
    weights = []

    for _, h in hist.iterrows():
        years_old = max((fecha_partido - h["fecha"]).days / 365.25, 0)
        w = math.exp(-decay * years_old)
        weights.append(w)

        if h["equipo_local"] == equipo_local:
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

    row["h2h_matches"] = len(hist)
    row["h2h_goals_home_avg"] = h2h_home_avg
    row["h2h_goals_away_avg"] = h2h_away_avg
    row["h2h_total_goals_avg"] = h2h_home_avg + h2h_away_avg
    row["h2h_home_win_rate"] = np.average(home_wins, weights=weights)
    row["h2h_weight_sum"] = float(weights.sum())

    return row


# ============================================================
# FEATURES AVANZADAS FUTURAS
# ============================================================

def compute_advanced_pre_features(team_adv_history, equipo, fecha_partido, side_suffix):
    fecha_partido = pd.to_datetime(fecha_partido)

    stat_cols = [
        "corners_for",
        "corners_against",
        "yellow_cards_for",
        "yellow_cards_against",
        "red_cards_for",
        "red_cards_against",
        "fouls_for",
        "fouls_against",
        "shots_for",
        "shots_against",
        "shots_on_target_for",
        "shots_on_target_against",
        "xg_for",
        "xg_against",
    ]

    hist = team_adv_history[
        (team_adv_history["equipo"] == equipo)
        & (team_adv_history["fecha"] < fecha_partido)
    ].sort_values("fecha").copy()

    if len(hist) > 0:
        hist = hist[hist[stat_cols].notna().any(axis=1)].copy()

    out = {}
    out[f"advanced_matches_count_total_{side_suffix}"] = len(hist)

    for w in [5, 10, 20]:
        last_w = hist.tail(w)

        for col in stat_cols:
            out[f"{col}_avg_{w}_{side_suffix}"] = last_w[col].mean()

        out[f"advanced_matches_count_{w}_{side_suffix}"] = len(last_w)

    return out


def add_generic_differentials(row):
    keys = list(row.index)

    for col_l in keys:
        if not col_l.endswith("_L"):
            continue

        base = col_l[:-2]
        col_v = f"{base}_V"
        diff_col = f"diff_{base}"

        if col_v in row.index and diff_col in row.index:
            try:
                row[diff_col] = row[col_l] - row[col_v]
            except Exception:
                pass

    return row


# ============================================================
# CONSTRUIR FILA FUTURA COMPLETA
# ============================================================

def build_future_match_row_full(
    df_adv,
    team_history,
    team_adv_history,
    equipo_local,
    equipo_visitante,
    fecha_partido,
    torneo="FIFA World Cup",
    fase="Group Stage",
    ciudad=None,
    estadio=None,
    pais_sede=None,
    neutral=1,
):
    fecha_partido = pd.to_datetime(fecha_partido)

    row = pd.Series(index=df_adv.columns, dtype="object")

    row["match_id"] = make_future_match_id(
        equipo_local,
        equipo_visitante,
        str(fecha_partido.date()),
        torneo
    )

    row["fecha"] = fecha_partido
    row["equipo_local"] = equipo_local
    row["equipo_visitante"] = equipo_visitante

    # Targets desconocidos
    for c in [
        "goles_local", "goles_visitante", "resultado_1x2", "over_2_5", "btts",
        "corners_local", "corners_visitante", "total_corners",
        "yellow_cards_local", "yellow_cards_visitante", "total_yellow_cards",
        "red_cards_local", "red_cards_visitante", "total_red_cards",
        "fouls_local", "fouls_visitante", "total_fouls",
        "shots_local", "shots_visitante", "total_shots",
        "shots_on_target_local", "shots_on_target_visitante", "total_shots_on_target",
        "xg_local", "xg_visitante", "total_xg",
    ]:
        if c in row.index:
            row[c] = np.nan

    row["torneo"] = torneo
    row["ciudad"] = ciudad
    row["pais_sede"] = pais_sede
    row["neutral"] = int(neutral)
    row["peso_competicion"] = assign_competition_weight(torneo, fase)

    if "source_id" in row.index:
        row["source_id"] = "manual_future_full_prediction"

    # Features base
    feats_l = compute_team_pre_features(team_history, equipo_local, fecha_partido, "L")
    feats_v = compute_team_pre_features(team_history, equipo_visitante, fecha_partido, "V")

    for k, v in feats_l.items():
        if k in row.index:
            row[k] = v

    for k, v in feats_v.items():
        if k in row.index:
            row[k] = v

    # ELO y H2H
    row = add_future_elo_features(
        row,
        df_adv,
        equipo_local,
        equipo_visitante,
        fecha_partido,
        neutral
    )

    row = add_future_h2h_features(
        row,
        df_adv,
        equipo_local,
        equipo_visitante,
        fecha_partido
    )

    # Features avanzadas
    adv_l = compute_advanced_pre_features(team_adv_history, equipo_local, fecha_partido, "L")
    adv_v = compute_advanced_pre_features(team_adv_history, equipo_visitante, fecha_partido, "V")

    for k, v in adv_l.items():
        if k in row.index:
            row[k] = v

    for k, v in adv_v.items():
        if k in row.index:
            row[k] = v

    # Diferenciales
    row = add_generic_differentials(row)

    return row


# ============================================================
# PREDICCIÓN CÓRNERS/TARJETAS
# ============================================================

def predict_pair_from_row(row, model_pack, lines):
    features = model_pack["features"]
    medians = model_pack["medians"]
    scaler = model_pack["scaler"]
    model_local = model_pack["model_local"]
    model_visitante = model_pack["model_visitante"]
    max_side_lambda = model_pack.get("max_side_lambda", 20.0)

    x_dict = {}

    for c in features:
        if c in row.index:
            x_dict[c] = row[c]
        else:
            x_dict[c] = np.nan

    X = pd.DataFrame([x_dict])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians)
    X = X.fillna(0.0)

    Xs = scaler.transform(X)

    lambda_local = float(np.clip(model_local.predict(Xs)[0], 0.01, max_side_lambda))
    lambda_visitante = float(np.clip(model_visitante.predict(Xs)[0], 0.01, max_side_lambda))
    lambda_total = lambda_local + lambda_visitante

    markets = market_probs(lambda_total, lines)

    return {
        "lambda_local": lambda_local,
        "lambda_visitante": lambda_visitante,
        "lambda_total": lambda_total,
        "markets": markets,
    }


# ============================================================
# REPORTE
# ============================================================

def imprimir_reporte_completo(result):
    partido = result["partido"]
    goles = result["goles"]
    corners = result["corners"]
    cards = result["yellow_cards"]

    probs = goles["probabilidades"]

    print("=" * 90)
    print("PREDICCIÓN COMPLETA — GOLES + CÓRNERS + TARJETAS")
    print("=" * 90)

    print(f"{partido['equipo_local']} vs {partido['equipo_visitante']}")
    print(f"Fecha: {partido['fecha']}")
    print(f"Torneo: {partido['torneo']}")
    print(f"Fase: {partido['fase']}")
    print(f"Ciudad: {partido['ciudad']}")
    print(f"Estadio: {partido['estadio']}")
    print(f"País sede: {partido['pais_sede']}")
    print(f"Neutral: {partido['neutral']}")

    print("\n" + "-" * 90)
    print("GOLES")
    print("-" * 90)
    print(f"Goles esperados {partido['equipo_local']}: {goles['lambda_local']:.3f}")
    print(f"Goles esperados {partido['equipo_visitante']}: {goles['lambda_visitante']:.3f}")
    print(f"Total goles esperado: {goles['lambda_total']:.3f}")

    print("\nProbabilidades 1X2:")
    print(f"  Gana {partido['equipo_local']}: {100 * probs['prob_local']:.2f}%")
    print(f"  Empate: {100 * probs['prob_empate']:.2f}%")
    print(f"  Gana {partido['equipo_visitante']}: {100 * probs['prob_visitante']:.2f}%")

    print("\nMercados goles:")
    print(f"  Over 1.5: {100 * probs['over_1_5']:.2f}%")
    print(f"  Under 1.5: {100 * probs['under_1_5']:.2f}%")
    print(f"  Over 2.5: {100 * probs['over_2_5']:.2f}%")
    print(f"  Under 2.5: {100 * probs['under_2_5']:.2f}%")
    print(f"  Over 3.5: {100 * probs['over_3_5']:.2f}%")
    print(f"  Under 3.5: {100 * probs['under_3_5']:.2f}%")

    print("\nAmbos anotan:")
    print(f"  BTTS Sí: {100 * probs['btts_si']:.2f}%")
    print(f"  BTTS No: {100 * probs['btts_no']:.2f}%")

    print("\nMarcador más probable:")
    print(f"  {probs['marcador_mas_probable']} → {100 * probs['prob_marcador_mas_probable']:.2f}%")

    print("\nTop marcadores:")
    for item in goles["top_marcadores"][:10]:
        print(f"  {item['score']}: {100 * item['prob']:.2f}%")

    print("\n" + "-" * 90)
    print("CÓRNERS — MODELO EXPERIMENTAL ESTABLE")
    print("-" * 90)
    print(f"Córners esperados {partido['equipo_local']}: {corners['lambda_local']:.3f}")
    print(f"Córners esperados {partido['equipo_visitante']}: {corners['lambda_visitante']:.3f}")
    print(f"Total córners esperado: {corners['lambda_total']:.3f}")

    print("\nMercados córners:")
    for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}: {100 * corners['markets'][f'over_{key}']:.2f}% | Under {line}: {100 * corners['markets'][f'under_{key}']:.2f}%")

    print("\n" + "-" * 90)
    print("TARJETAS AMARILLAS — MODELO EXPERIMENTAL ESTABLE")
    print("-" * 90)
    print(f"Tarjetas esperadas {partido['equipo_local']}: {cards['lambda_local']:.3f}")
    print(f"Tarjetas esperadas {partido['equipo_visitante']}: {cards['lambda_visitante']:.3f}")
    print(f"Total tarjetas esperado: {cards['lambda_total']:.3f}")

    print("\nMercados tarjetas:")
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        key = str(line).replace(".", "_")
        print(f"  Over {line}: {100 * cards['markets'][f'over_{key}']:.2f}% | Under {line}: {100 * cards['markets'][f'under_{key}']:.2f}%")

    print("\n" + "-" * 90)
    print("NOTA")
    print("-" * 90)
    print("El modelo de goles está entrenado con todo el histórico disponible.")
    print("Los modelos de córners y tarjetas son experimentales porque usan 290 partidos con StatsBomb Open Data.")


def append_prediction_summary(result, project_root=DEFAULT_PROJECT_ROOT):
    project_root = Path(project_root)
    out_path = project_root / "data/predictions/manual_full_match_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    partido = result["partido"]
    goles = result["goles"]
    corners = result["corners"]
    cards = result["yellow_cards"]
    probs = goles["probabilidades"]

    row = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fecha": partido["fecha"],
        "equipo_local": partido["equipo_local"],
        "equipo_visitante": partido["equipo_visitante"],
        "torneo": partido["torneo"],
        "fase": partido["fase"],
        "ciudad": partido["ciudad"],
        "estadio": partido["estadio"],
        "pais_sede": partido["pais_sede"],
        "neutral": partido["neutral"],

        "lambda_goles_local": goles["lambda_local"],
        "lambda_goles_visitante": goles["lambda_visitante"],
        "lambda_goles_total": goles["lambda_total"],
        "prob_local": probs["prob_local"],
        "prob_empate": probs["prob_empate"],
        "prob_visitante": probs["prob_visitante"],
        "over_2_5_goals": probs["over_2_5"],
        "under_2_5_goals": probs["under_2_5"],
        "btts_si": probs["btts_si"],
        "btts_no": probs["btts_no"],
        "marcador_mas_probable": probs["marcador_mas_probable"],

        "lambda_corners_local": corners["lambda_local"],
        "lambda_corners_visitante": corners["lambda_visitante"],
        "lambda_corners_total": corners["lambda_total"],

        "lambda_yellow_cards_local": cards["lambda_local"],
        "lambda_yellow_cards_visitante": cards["lambda_visitante"],
        "lambda_yellow_cards_total": cards["lambda_total"],
    }

    for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
        key = str(line).replace(".", "_")
        row[f"over_corners_{key}"] = corners["markets"][f"over_{key}"]
        row[f"under_corners_{key}"] = corners["markets"][f"under_{key}"]

    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        key = str(line).replace(".", "_")
        row[f"over_cards_{key}"] = cards["markets"][f"over_{key}"]
        row[f"under_cards_{key}"] = cards["markets"][f"under_{key}"]

    df_new = pd.DataFrame([row])

    if out_path.exists():
        df_old = pd.read_csv(out_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(out_path, index=False)

    return out_path


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def predecir_partido_completo(
    equipo_local,
    equipo_visitante,
    fecha_partido,
    torneo="FIFA World Cup",
    fase="Group Stage",
    ciudad=None,
    estadio=None,
    pais_sede=None,
    neutral=1,
    project_root=DEFAULT_PROJECT_ROOT,
    verbose=True,
    save=True,
):
    df_adv, team_history, team_adv_history, goal_bundle, cc_bundle = load_assets(project_root)

    row = build_future_match_row_full(
        df_adv=df_adv,
        team_history=team_history,
        team_adv_history=team_adv_history,
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        torneo=torneo,
        fase=fase,
        ciudad=ciudad,
        estadio=estadio,
        pais_sede=pais_sede,
        neutral=neutral,
    )

    goles = predict_from_feature_row(row, goal_bundle)

    corners = predict_pair_from_row(
        row=row,
        model_pack=cc_bundle["corners"],
        lines=cc_bundle["corners_lines"]
    )

    yellow_cards = predict_pair_from_row(
        row=row,
        model_pack=cc_bundle["yellow_cards"],
        lines=cc_bundle["cards_lines"]
    )

    result = {
        "partido": {
            "equipo_local": equipo_local,
            "equipo_visitante": equipo_visitante,
            "fecha": str(pd.to_datetime(fecha_partido).date()),
            "torneo": torneo,
            "fase": fase,
            "ciudad": ciudad,
            "estadio": estadio,
            "pais_sede": pais_sede,
            "neutral": int(neutral),
        },
        "features_row": row,
        "goles": goles,
        "corners": corners,
        "yellow_cards": yellow_cards,
    }

    if verbose:
        imprimir_reporte_completo(result)

    if save:
        out_path = append_prediction_summary(result, project_root=project_root)
        if verbose:
            print(f"\nPredicción guardada en: {out_path}")

    return result
