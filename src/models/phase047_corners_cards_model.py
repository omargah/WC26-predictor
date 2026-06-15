
# -*- coding: utf-8 -*-
"""
FASE 4.7 — Córners y tarjetas con modelo joblib real + fallback legacy.

Orden de uso:
    1. Intenta usar models/corners_cards_poisson_stable.joblib.
    2. Si falla, usa benchmarks legacy guardados.
    3. Si no hay benchmark, devuelve available=False.

Este módulo no inventa datos.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore")


# ============================================================
# Utilidades generales
# ============================================================

def _norm_team(x: str) -> str:
    return str(x).strip().lower()


def _poisson_over(lambda_total: float, line: float) -> float:
    threshold = int(math.floor(line) + 1)
    return float(1.0 - poisson.cdf(threshold - 1, float(lambda_total)))


def _markets(lambda_total: float, lines: list[float]) -> dict:
    out = {}

    for line in lines:
        key = str(line).replace(".", "_")
        over = _poisson_over(lambda_total, line)
        out[f"over_{key}"] = over
        out[f"under_{key}"] = 1.0 - over

    return out


def _safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _clip_lambda(x, low=0.05, high=20.0):
    x = _safe_float(x, default=np.nan)

    if pd.isna(x):
        return np.nan

    return float(np.clip(x, low, high))


# ============================================================
# Fallback legacy por benchmark
# ============================================================

def load_legacy_benchmarks(project_root: str | Path) -> pd.DataFrame:
    project_root = Path(project_root)
    path = project_root / "data" / "manual" / "legacy_vs_updated_model_benchmarks.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero los bloques de benchmarks."
        )

    return pd.read_csv(path)


def find_legacy_row(
    project_root: str | Path,
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str | None = None,
) -> pd.Series | None:
    df = load_legacy_benchmarks(project_root)

    home = _norm_team(equipo_local)
    away = _norm_team(equipo_visitante)

    mask = (
        df["home_team"].map(_norm_team).eq(home)
        &
        df["away_team"].map(_norm_team).eq(away)
    )

    if fecha_partido is not None and "fecha_partido" in df.columns:
        mask = mask & df["fecha_partido"].astype(str).eq(str(fecha_partido))

    found = df[mask].copy()

    if len(found) > 0:
        return found.iloc[0]

    return None


def predict_from_legacy_benchmark(
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    project_root: str | Path,
) -> dict:
    row = find_legacy_row(
        project_root=project_root,
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
    )

    if row is None:
        return {
            "available": False,
            "source": "no_legacy_benchmark",
            "reason": "No hay benchmark legacy para este partido.",
        }

    required = [
        "old_corners_home",
        "old_corners_away",
        "old_corners_total",
        "old_cards_home",
        "old_cards_away",
        "old_cards_total",
    ]

    for col in required:
        if col not in row.index or pd.isna(row[col]):
            return {
                "available": False,
                "source": "legacy_benchmark_without_corners_cards",
                "reason": "El benchmark existe, pero no contiene córners/tarjetas.",
            }

    corners_home = float(row["old_corners_home"])
    corners_away = float(row["old_corners_away"])
    corners_total = float(row["old_corners_total"])

    cards_home = float(row["old_cards_home"])
    cards_away = float(row["old_cards_away"])
    cards_total = float(row["old_cards_total"])

    return {
        "available": True,
        "source": "legacy_benchmark_compatible",
        "corners_home": corners_home,
        "corners_away": corners_away,
        "corners_total": corners_total,
        "cards_home": cards_home,
        "cards_away": cards_away,
        "cards_total": cards_total,
        "corners_markets": _markets(corners_total, [7.5, 8.5, 9.5, 10.5, 11.5]),
        "cards_markets": _markets(cards_total, [2.5, 3.5, 4.5, 5.5, 6.5]),
    }


# ============================================================
# Descubrimiento del paquete joblib
# ============================================================

def load_joblib_model(project_root: str | Path):
    project_root = Path(project_root)
    path = project_root / "models" / "corners_cards_poisson_stable.joblib"

    if not path.exists():
        return None, f"No existe {path}"

    try:
        return joblib.load(path), None
    except Exception as e:
        return None, f"No pude cargar joblib: {e}"


def walk_objects(obj, path="root", max_depth=5):
    """
    Recorre recursivamente dict/list/tuple y devuelve objetos con su ruta.
    """

    rows = [(path, obj)]

    if max_depth <= 0:
        return rows

    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.extend(walk_objects(v, f"{path}.{k}", max_depth=max_depth - 1))

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            rows.extend(walk_objects(v, f"{path}[{i}]", max_depth=max_depth - 1))

    return rows


def find_feature_list(package) -> list[str] | None:
    """
    Busca una lista de features dentro del paquete.
    """

    preferred_names = [
        "features",
        "feature_cols",
        "feature_columns",
        "selected_features",
        "features_used",
        "model_features",
        "pre_match_features",
        "feature_names",
    ]

    if isinstance(package, dict):
        for name in preferred_names:
            if name in package:
                value = package[name]
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    if all(isinstance(x, str) for x in value):
                        return list(value)

    for path, obj in walk_objects(package):
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            if all(isinstance(x, str) for x in obj):
                lower_path = path.lower()
                if "feature" in lower_path:
                    return list(obj)

    # Si algún estimador trae feature_names_in_.
    for path, obj in walk_objects(package):
        if hasattr(obj, "feature_names_in_"):
            try:
                return list(obj.feature_names_in_)
            except Exception:
                pass

    return None


def find_medians(package):
    """
    Busca medianas/fill values.
    """

    preferred_names = [
        "medians",
        "feature_medians",
        "fill_values",
        "fillna_values",
        "impute_values",
        "imputer_values",
    ]

    if isinstance(package, dict):
        for name in preferred_names:
            if name in package:
                value = package[name]
                if isinstance(value, (pd.Series, dict)):
                    return value

    for path, obj in walk_objects(package):
        lower_path = path.lower()

        if any(k in lower_path for k in ["median", "fill", "imput"]):
            if isinstance(obj, (pd.Series, dict)):
                return obj

    return None


def find_scaler(package):
    """
    Busca un scaler o transformador.
    """

    if isinstance(package, dict):
        for key in ["scaler", "x_scaler", "standard_scaler", "preprocessor", "transformer"]:
            if key in package and hasattr(package[key], "transform"):
                return package[key]

    for path, obj in walk_objects(package):
        lower_path = path.lower()
        if hasattr(obj, "transform") and any(k in lower_path for k in ["scaler", "standard", "preprocess", "transform"]):
            return obj

    return None


def find_estimators(package) -> dict:
    """
    Busca estimadores con predict y los clasifica por nombre/ruta.
    """

    estimators = {}

    for path, obj in walk_objects(package):
        if hasattr(obj, "predict"):
            estimators[path] = obj

    return estimators


def choose_model(estimators: dict, target: str):
    """
    Escoge un estimador para target:
        corners_home, corners_away, cards_home, cards_away
    usando heurística por nombre de ruta.
    """

    target = target.lower()

    if "corners" in target:
        main_tokens = ["corner", "corners", "corners_total", "corners_for"]
        avoid_tokens = []
    else:
        main_tokens = ["card", "cards", "yellow", "tarjeta", "tarjetas"]
        avoid_tokens = []

    if target.endswith("home"):
        side_tokens = ["home", "local", "_l", ".l", "for"]
        bad_side_tokens = ["away", "visitante", "_v", ".v", "against"]
    else:
        side_tokens = ["away", "visitante", "_v", ".v", "against"]
        bad_side_tokens = ["home", "local", "_l", ".l", "for"]

    candidates = []

    for path, model in estimators.items():
        p = path.lower()

        score = 0

        if any(t in p for t in main_tokens):
            score += 5

        if any(t in p for t in side_tokens):
            score += 3

        if any(t in p for t in bad_side_tokens):
            score -= 3

        if "model" in p or "poisson" in p:
            score += 1

        if score > 0:
            candidates.append((score, path, model))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    if len(candidates) == 0:
        return None, None

    return candidates[0][2], candidates[0][1]


def prepare_X(row: pd.Series, features: list[str], medians=None):
    """
    Construye X de una fila con las features esperadas por el modelo.
    """

    data = {}

    for f in features:
        data[f] = row[f] if f in row.index else np.nan

    X = pd.DataFrame([data])
    X = X.replace([np.inf, -np.inf], np.nan)

    if medians is not None:
        if isinstance(medians, pd.Series):
            X = X.fillna(medians)
        elif isinstance(medians, dict):
            X = X.fillna(medians)

    X = X.fillna(0.0)

    return X


def apply_scaler_if_possible(scaler, X):
    if scaler is None:
        return X

    try:
        return scaler.transform(X)
    except Exception:
        try:
            return scaler.transform(X.values)
        except Exception:
            return X


# ============================================================
# Features del partido
# ============================================================

def load_modeling_dataset(project_root: str | Path) -> pd.DataFrame:
    """
    Carga dataset para córners/tarjetas.

    Prioridad:
        1. modeling_dataset_advanced.parquet
        2. modeling_dataset.parquet

    Esta función devuelve el dataset preferido.
    Para búsqueda robusta por partido, usar load_modeling_datasets_for_search.
    """

    project_root = Path(project_root)

    advanced_path = project_root / "data" / "features" / "modeling_dataset_advanced.parquet"
    basic_path = project_root / "data" / "features" / "modeling_dataset.parquet"

    if advanced_path.exists():
        path = advanced_path
    elif basic_path.exists():
        path = basic_path
    else:
        raise FileNotFoundError(
            f"No existe {advanced_path} ni {basic_path}. Ejecuta primero Fase 2."
        )

    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df.attrs["source_path"] = str(path)

    return df


def load_modeling_datasets_for_search(project_root: str | Path) -> list[tuple[str, pd.DataFrame]]:
    """
    Carga datasets para búsqueda de partido.

    Devuelve lista ordenada:
        advanced primero,
        basic después.

    Esto permite que el .joblib use advanced cuando exista,
    pero no falle si un partido solo existe en basic.
    """

    project_root = Path(project_root)

    advanced_path = project_root / "data" / "features" / "modeling_dataset_advanced.parquet"
    basic_path = project_root / "data" / "features" / "modeling_dataset.parquet"

    datasets = []

    if advanced_path.exists():
        df_adv = pd.read_parquet(advanced_path)
        df_adv["fecha"] = pd.to_datetime(df_adv["fecha"], errors="coerce")
        datasets.append(("advanced", df_adv))

    if basic_path.exists():
        df_basic = pd.read_parquet(basic_path)
        df_basic["fecha"] = pd.to_datetime(df_basic["fecha"], errors="coerce")
        datasets.append(("basic", df_basic))

    if len(datasets) == 0:
        raise FileNotFoundError(
            f"No existe {advanced_path} ni {basic_path}. Ejecuta primero Fase 2."
        )

    return datasets


def team_aliases(team: str) -> list[str]:
    try:
        from src.utils.team_names import team_aliases as _team_aliases
        return _team_aliases(team)
    except Exception:
        return [str(team).strip()]

def swap_home_away_row(row: pd.Series) -> pd.Series:
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


def find_feature_row_for_match(
    project_root: str | Path,
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    candidate_dates: list[str] | None = None,
) -> tuple[pd.Series, bool, str]:
    """
    Busca una fila de features para el partido.

    Orden:
        1. advanced, fecha exacta
        2. basic, fecha exacta
        3. advanced, fecha cercana
        4. basic, fecha cercana

    Retorna:
        row,
        was_reversed,
        search_mode
    """

    datasets = load_modeling_datasets_for_search(project_root)

    home_aliases = team_aliases(equipo_local)
    away_aliases = team_aliases(equipo_visitante)

    if candidate_dates is None:
        candidate_dates = [fecha_partido]

    # --------------------------------------------------------
    # 1. Búsqueda por fecha exacta
    # --------------------------------------------------------

    for dataset_name, df in datasets:
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
                row = found.sort_values("fecha").iloc[-1]
                row["feature_dataset_source"] = dataset_name
                return row, False, f"{dataset_name}_direct_same_date"

            reverse = (
                df["equipo_local"].isin(away_aliases)
                &
                df["equipo_visitante"].isin(home_aliases)
                &
                same_date
            )

            found = df[reverse].copy()

            if len(found) > 0:
                row = swap_home_away_row(found.sort_values("fecha").iloc[-1])
                row["feature_dataset_source"] = dataset_name
                return row, True, f"{dataset_name}_reverse_same_date"

    # --------------------------------------------------------
    # 2. Búsqueda por fecha cercana
    # --------------------------------------------------------

    target_date = pd.to_datetime(candidate_dates[0], errors="coerce")

    for dataset_name, df in datasets:
        direct_any = (
            df["equipo_local"].isin(home_aliases)
            &
            df["equipo_visitante"].isin(away_aliases)
        )

        found = df[direct_any].copy()

        if len(found) > 0:
            found["date_distance"] = (found["fecha"] - target_date).abs()
            row = found.sort_values("date_distance").iloc[0]
            row["feature_dataset_source"] = dataset_name
            return row, False, f"{dataset_name}_direct_nearest_date"

        reverse_any = (
            df["equipo_local"].isin(away_aliases)
            &
            df["equipo_visitante"].isin(home_aliases)
        )

        found = df[reverse_any].copy()

        if len(found) > 0:
            found["date_distance"] = (found["fecha"] - target_date).abs()
            row = swap_home_away_row(found.sort_values("date_distance").iloc[0])
            row["feature_dataset_source"] = dataset_name
            return row, True, f"{dataset_name}_reverse_nearest_date"

    raise ValueError(
        f"No encontré fila de features para {equipo_local} vs {equipo_visitante}. "
        f"Busqué en {[name for name, _ in datasets]}."
    )


def predict_from_joblib(
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    project_root: str | Path,
    candidate_dates: list[str] | None = None,
) -> dict:
    package, error = load_joblib_model(project_root)

    if package is None:
        return {
            "available": False,
            "source": "joblib_unavailable",
            "reason": error,
        }

    estimators = find_estimators(package)

    if len(estimators) == 0:
        return {
            "available": False,
            "source": "joblib_no_estimators",
            "reason": "No encontré objetos con método predict dentro del joblib.",
        }

    model_ch, path_ch = choose_model(estimators, "corners_home")
    model_ca, path_ca = choose_model(estimators, "corners_away")
    model_yh, path_yh = choose_model(estimators, "cards_home")
    model_ya, path_ya = choose_model(estimators, "cards_away")

    missing = []

    if model_ch is None:
        missing.append("corners_home")

    if model_ca is None:
        missing.append("corners_away")

    if model_yh is None:
        missing.append("cards_home")

    if model_ya is None:
        missing.append("cards_away")

    if missing:
        return {
            "available": False,
            "source": "joblib_models_not_identified",
            "reason": f"No pude identificar modelos para: {missing}",
            "estimators_found": list(estimators.keys()),
        }

    features = find_feature_list(package)
    medians = find_medians(package)
    scaler = find_scaler(package)

    if features is None:
        # Último recurso: si el primer modelo trae n_features_in_, tomamos columnas numéricas.
        row, was_reversed, search_mode = find_feature_row_for_match(
            project_root=project_root,
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
            fecha_partido=fecha_partido,
            candidate_dates=candidate_dates,
        )

        numeric_cols = [
            c for c in row.index
            if pd.api.types.is_number(row[c])
        ]

        n_expected = getattr(model_ch, "n_features_in_", None)

        if n_expected is not None and len(numeric_cols) >= int(n_expected):
            features = numeric_cols[:int(n_expected)]
        else:
            return {
                "available": False,
                "source": "joblib_features_not_identified",
                "reason": "No pude identificar la lista de features del joblib.",
            }
    else:
        row, was_reversed, search_mode = find_feature_row_for_match(
            project_root=project_root,
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
            fecha_partido=fecha_partido,
            candidate_dates=candidate_dates,
        )

    X = prepare_X(row, features, medians=medians)
    X_model = apply_scaler_if_possible(scaler, X)

    try:
        corners_home = _clip_lambda(model_ch.predict(X_model)[0], 0.05, 25.0)
        corners_away = _clip_lambda(model_ca.predict(X_model)[0], 0.05, 25.0)
        cards_home = _clip_lambda(model_yh.predict(X_model)[0], 0.05, 15.0)
        cards_away = _clip_lambda(model_ya.predict(X_model)[0], 0.05, 15.0)
    except Exception as e:
        return {
            "available": False,
            "source": "joblib_prediction_error",
            "reason": str(e),
            "features_count": len(features),
            "model_paths": {
                "corners_home": path_ch,
                "corners_away": path_ca,
                "cards_home": path_yh,
                "cards_away": path_ya,
            },
        }

    corners_total = corners_home + corners_away
    cards_total = cards_home + cards_away

    return {
        "available": True,
        "source": "legacy_joblib_real",
        "feature_search_mode": search_mode,
        "feature_was_reversed": was_reversed,
        "features_count": len(features),
        "model_paths": {
            "corners_home": path_ch,
            "corners_away": path_ca,
            "cards_home": path_yh,
            "cards_away": path_ya,
        },
        "corners_home": corners_home,
        "corners_away": corners_away,
        "corners_total": corners_total,
        "cards_home": cards_home,
        "cards_away": cards_away,
        "cards_total": cards_total,
        "corners_markets": _markets(corners_total, [7.5, 8.5, 9.5, 10.5, 11.5]),
        "cards_markets": _markets(cards_total, [2.5, 3.5, 4.5, 5.5, 6.5]),
    }


# ============================================================
# Función pública
# ============================================================

def predecir_corners_tarjetas_experimental(
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    project_root: str | Path,
    candidate_dates: list[str] | None = None,
    prefer_joblib: bool = True,
) -> dict:
    """
    Predice córners y tarjetas.

    Si prefer_joblib=True:
        intenta joblib real primero y fallback legacy después.

    Si prefer_joblib=False:
        usa directamente benchmark legacy.
    """

    if prefer_joblib:
        pred_joblib = predict_from_joblib(
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
            fecha_partido=fecha_partido,
            project_root=project_root,
            candidate_dates=candidate_dates,
        )

        if pred_joblib.get("available", False):
            return pred_joblib

        fallback = predict_from_legacy_benchmark(
            equipo_local=equipo_local,
            equipo_visitante=equipo_visitante,
            fecha_partido=fecha_partido,
            project_root=project_root,
        )

        if fallback.get("available", False):
            fallback["joblib_attempt_source"] = pred_joblib.get("source")
            fallback["joblib_attempt_reason"] = pred_joblib.get("reason")
            return fallback

        return pred_joblib

    return predict_from_legacy_benchmark(
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        project_root=project_root,
    )


def imprimir_corners_tarjetas(
    equipo_local: str,
    equipo_visitante: str,
    pred: dict,
) -> None:
    if not pred.get("available", False):
        print("\n------------------------------------------------------------------------------------------")
        print("CÓRNERS / TARJETAS")
        print("------------------------------------------------------------------------------------------")
        print("No disponibles para este caso.")
        print(f"Fuente: {pred.get('source')}")
        print(f"Motivo: {pred.get('reason')}")
        return

    print("\n------------------------------------------------------------------------------------------")
    print("CÓRNERS — MODELO EXPERIMENTAL ESTABLE")
    print("------------------------------------------------------------------------------------------")
    print(f"Fuente: {pred.get('source')}")
    print(f"Córners esperados {equipo_local}: {pred['corners_home']:.3f}")
    print(f"Córners esperados {equipo_visitante}: {pred['corners_away']:.3f}")
    print(f"Total córners esperado: {pred['corners_total']:.3f}")

    print("\nMercados córners:")
    for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
        key = str(line).replace(".", "_")
        over = pred["corners_markets"][f"over_{key}"]
        under = pred["corners_markets"][f"under_{key}"]
        print(f"  Over {line}: {over:.2%} | Under {line}: {under:.2%}")

    print("\n------------------------------------------------------------------------------------------")
    print("TARJETAS AMARILLAS — MODELO EXPERIMENTAL ESTABLE")
    print("------------------------------------------------------------------------------------------")
    print(f"Tarjetas esperadas {equipo_local}: {pred['cards_home']:.3f}")
    print(f"Tarjetas esperadas {equipo_visitante}: {pred['cards_away']:.3f}")
    print(f"Total tarjetas esperado: {pred['cards_total']:.3f}")

    print("\nMercados tarjetas:")
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        key = str(line).replace(".", "_")
        over = pred["cards_markets"][f"over_{key}"]
        under = pred["cards_markets"][f"under_{key}"]
        print(f"  Over {line}: {over:.2%} | Under {line}: {under:.2%}")
