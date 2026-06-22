# -*- coding: utf-8 -*-
"""
src/models/poisson_dc.py

FASE 3 -- Modelo base Poisson + Dixon-Coles.

Objetivo:
    Entrenar un modelo explicable de goles esperados:
        lambda_local
        lambda_visitante

Luego convertir esas lambdas en probabilidades:
    - 1X2;
    - marcador exacto;
    - Over/Under;
    - ambos anotan.

Metodología:
    1. Split temporal para evaluación honesta.
    2. PoissonRegressor para goles local y visitante.
    3. Matriz de marcadores con ajuste Dixon-Coles.
    4. Modelo final reentrenado con todos los partidos jugados para predecir pendientes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import poisson

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


TARGET_AND_ID_COLUMNS = {
    "match_id",
    "goles_local",
    "goles_visitante",
    "resultado_1x2",
    "over_2_5",
    "btts",
    "is_played",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ============================================================
# 1. Features y split temporal
# ============================================================

def select_numeric_features(df: pd.DataFrame, min_coverage: float = 0.30) -> list[str]:
    """
    Selecciona columnas numéricas que pueden usarse como features.

    Excluye targets, identificadores y columnas no numéricas.
    """

    candidates = []

    for col in df.columns:
        if col in TARGET_AND_ID_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            coverage = df[col].notna().mean()

            if coverage >= min_coverage:
                candidates.append(col)

    useful = []

    for col in candidates:
        if df[col].nunique(dropna=True) > 1:
            useful.append(col)

    return useful


def temporal_split(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal:
        primeros 80% por fecha -> entrenamiento;
        últimos 20% por fecha -> validación.

    No usamos split aleatorio porque mezclaría pasado y futuro.
    """

    df = df.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - test_ratio))

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    return df_train, df_test


def clean_X(
    df: pd.DataFrame,
    features: list[str],
    medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Limpieza de matriz X:
        - reemplaza inf por NaN;
        - imputa NaN con medianas del set de entrenamiento;
        - si aún quedan NaN, usa 0.0.
    """

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = X.median(numeric_only=True)

    X = X.fillna(medians)
    X = X.fillna(0.0)

    return X, medians


# ============================================================
# 2. Entrenamiento Poisson
# ============================================================

def train_poisson_models(
    df_train: pd.DataFrame,
    features: list[str],
    alpha: float = 0.001,
    max_iter: int = 1000,
) -> dict:
    """
    Entrena dos modelos Poisson:
        - goles local;
        - goles visitante.
    """

    X_train, medians = clean_X(df_train, features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    y_home = df_train["goles_local"].astype(float)
    y_away = df_train["goles_visitante"].astype(float)

    model_home = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    model_away = PoissonRegressor(alpha=alpha, max_iter=max_iter)

    model_home.fit(X_scaled, y_home)
    model_away.fit(X_scaled, y_away)

    package = {
        "model_home": model_home,
        "model_away": model_away,
        "scaler": scaler,
        "features": features,
        "medians": medians,
        "alpha": alpha,
        "max_iter": max_iter,
        "rho": -0.075,
        "max_goals": 10,
        "trained_at": now_iso(),
    }

    return package


def predict_lambdas(
    model_package: dict,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predice lambdas de goles local y visitante.
    """

    features = model_package["features"]
    medians = model_package["medians"]
    scaler = model_package["scaler"]

    X, _ = clean_X(df, features, medians=medians)
    X_scaled = scaler.transform(X)

    lambda_home = model_package["model_home"].predict(X_scaled)
    lambda_away = model_package["model_away"].predict(X_scaled)

    lambda_home = np.clip(lambda_home, 0.05, 8.0)
    lambda_away = np.clip(lambda_away, 0.05, 8.0)

    return lambda_home, lambda_away


# ============================================================
# 3. Dixon-Coles y matriz de marcador
# ============================================================

def dixon_coles_tau(
    x: int,
    y: int,
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.075,
) -> float:
    """
    Factor Dixon-Coles para corregir marcadores bajos.

    Afecta principalmente:
        0-0, 1-0, 0-1, 1-1.
    """

    if x == 0 and y == 0:
        tau = 1.0 + lambda_home * lambda_away * rho
    elif x == 0 and y == 1:
        tau = 1.0 - lambda_home * rho
    elif x == 1 and y == 0:
        tau = 1.0 - lambda_away * rho
    elif x == 1 and y == 1:
        tau = 1.0 + rho
    else:
        tau = 1.0

    return max(float(tau), 0.0001)


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
    rho: float = -0.075,
    use_dixon_coles: bool = True,
) -> np.ndarray:
    """
    Matriz P(goles_local=i, goles_visitante=j).
    """

    lambda_home = float(np.clip(lambda_home, 0.05, 8.0))
    lambda_away = float(np.clip(lambda_away, 0.05, 8.0))

    goals = np.arange(max_goals + 1)

    p_home = poisson.pmf(goals, lambda_home)
    p_away = poisson.pmf(goals, lambda_away)

    mat = np.outer(p_home, p_away)

    if use_dixon_coles:
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                mat[i, j] *= dixon_coles_tau(
                    i,
                    j,
                    lambda_home=lambda_home,
                    lambda_away=lambda_away,
                    rho=rho,
                )

    mat = mat / mat.sum()

    return mat


def probs_from_matrix(mat: np.ndarray) -> dict:
    """
    Convierte matriz de marcador exacto en probabilidades de mercado.
    """

    gh, ga = np.indices(mat.shape)
    total_goals = gh + ga

    probs = {
        "prob_local": float(mat[gh > ga].sum()),
        "prob_empate": float(mat[gh == ga].sum()),
        "prob_visitante": float(mat[gh < ga].sum()),
        "prob_btts_si": float(mat[(gh > 0) & (ga > 0)].sum()),
        "prob_btts_no": float(mat[(gh == 0) | (ga == 0)].sum()),
    }

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        suffix = str(line).replace(".", "_")
        probs[f"prob_over_{suffix}"] = float(mat[total_goals > line].sum())
        probs[f"prob_under_{suffix}"] = float(mat[total_goals < line].sum())

    idx = np.unravel_index(np.argmax(mat), mat.shape)

    probs["marcador_mas_probable"] = f"{idx[0]}-{idx[1]}"
    probs["prob_marcador_mas_probable"] = float(mat[idx])

    return probs


def top_scorelines(
    mat: np.ndarray,
    top_n: int = 10,
) -> list[dict]:
    """
    Devuelve los marcadores exactos más probables.
    """

    rows = []

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rows.append(
                {
                    "score": f"{i}-{j}",
                    "home_goals": int(i),
                    "away_goals": int(j),
                    "prob": float(mat[i, j]),
                }
            )

    return sorted(rows, key=lambda r: r["prob"], reverse=True)[:top_n]


def predict_dataframe(
    model_package: dict,
    df: pd.DataFrame,
    use_dixon_coles: bool = True,
) -> pd.DataFrame:
    """
    Predice lambdas y probabilidades para todas las filas de un DataFrame.
    """

    df_out = df.copy()

    lambda_home, lambda_away = predict_lambdas(model_package, df_out)

    df_out["lambda_local"] = lambda_home
    df_out["lambda_visitante"] = lambda_away

    prob_rows = []
    top_rows = []

    max_goals = int(model_package.get("max_goals", 10))
    rho = float(model_package.get("rho", -0.075))

    for lh, la in zip(lambda_home, lambda_away):
        mat = score_matrix(
            lambda_home=lh,
            lambda_away=la,
            max_goals=max_goals,
            rho=rho,
            use_dixon_coles=use_dixon_coles,
        )

        probs = probs_from_matrix(mat)
        prob_rows.append(probs)
        top_rows.append(json.dumps(top_scorelines(mat, top_n=10), ensure_ascii=False))

    probs_df = pd.DataFrame(prob_rows)

    df_out = pd.concat(
        [df_out.reset_index(drop=True), probs_df.reset_index(drop=True)],
        axis=1,
    )

    df_out["top_10_marcadores"] = top_rows

    prob_cols = ["prob_local", "prob_empate", "prob_visitante"]
    pred_idx = df_out[prob_cols].values.argmax(axis=1)

    label_map = {0: "L", 1: "E", 2: "V"}
    df_out["pred_resultado_1x2"] = [label_map[int(i)] for i in pred_idx]

    return df_out


# ============================================================
# 4. Evaluación
# ============================================================

def evaluate_predictions(df_pred: pd.DataFrame) -> dict:
    """
    Evalúa desempeño del modelo en validación temporal.
    """

    y_true = df_pred["resultado_1x2"].astype(str).values

    label_to_int = {"L": 0, "E": 1, "V": 2}
    y_true_int = np.array([label_to_int[x] for x in y_true])

    probs = df_pred[["prob_local", "prob_empate", "prob_visitante"]].values

    y_pred = df_pred["pred_resultado_1x2"].astype(str).values

    mae_home = mean_absolute_error(df_pred["goles_local"], df_pred["lambda_local"])
    mae_away = mean_absolute_error(df_pred["goles_visitante"], df_pred["lambda_visitante"])

    rmse_home = np.sqrt(mean_squared_error(df_pred["goles_local"], df_pred["lambda_local"]))
    rmse_away = np.sqrt(mean_squared_error(df_pred["goles_visitante"], df_pred["lambda_visitante"]))

    metrics = {
        "n_validation": int(len(df_pred)),
        "mae_goles_local": float(mae_home),
        "mae_goles_visitante": float(mae_away),
        "mae_total": float(mae_home + mae_away),
        "rmse_goles_local": float(rmse_home),
        "rmse_goles_visitante": float(rmse_away),
        "accuracy_1x2": float(accuracy_score(y_true, y_pred)),
        "log_loss_1x2": float(log_loss(y_true_int, probs, labels=[0, 1, 2])),
        "lambda_local_promedio": float(df_pred["lambda_local"].mean()),
        "lambda_visitante_promedio": float(df_pred["lambda_visitante"].mean()),
        "goles_local_promedio_real": float(df_pred["goles_local"].mean()),
        "goles_visitante_promedio_real": float(df_pred["goles_visitante"].mean()),
    }

    return metrics


# ============================================================
# 5. Guardado y carga
# ============================================================

def save_model(
    model_package: dict,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, path)


def load_model(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el modelo: {path}")

    return joblib.load(path)


# ============================================================
# 6. Entrenamiento completo
# ============================================================

def train_evaluate_and_refit(
    df_train_all: pd.DataFrame,
    test_ratio: float = 0.20,
    min_feature_coverage: float = 0.30,
    alpha: float = 0.001,
) -> dict:
    """
    Entrena y evalúa el modelo.

    Devuelve:
        - modelo de evaluación;
        - modelo final reentrenado con todos los partidos jugados;
        - predicciones de validación;
        - métricas.
    """

    df_played = df_train_all[df_train_all["is_played"]].copy()
    df_played["fecha"] = pd.to_datetime(df_played["fecha"], errors="coerce")
    df_played = df_played.sort_values(["fecha", "match_id"]).reset_index(drop=True)

    df_fit, df_val = temporal_split(df_played, test_ratio=test_ratio)

    features = select_numeric_features(df_fit, min_coverage=min_feature_coverage)

    if not features:
        raise ValueError("No se seleccionaron features numéricas. Revisa la Fase 2.")

    eval_model = train_poisson_models(
        df_train=df_fit,
        features=features,
        alpha=alpha,
    )

    df_val_pred = predict_dataframe(eval_model, df_val)

    metrics = evaluate_predictions(df_val_pred)

    final_model = train_poisson_models(
        df_train=df_played,
        features=features,
        alpha=alpha,
    )

    metadata = {
        "trained_at": now_iso(),
        "test_ratio": test_ratio,
        "min_feature_coverage": min_feature_coverage,
        "alpha": alpha,
        "n_rows_total_played": int(len(df_played)),
        "n_rows_fit": int(len(df_fit)),
        "n_rows_validation": int(len(df_val)),
        "date_min_total": str(df_played["fecha"].min()),
        "date_max_total": str(df_played["fecha"].max()),
        "date_max_fit": str(df_fit["fecha"].max()),
        "date_min_validation": str(df_val["fecha"].min()),
        "date_max_validation": str(df_val["fecha"].max()),
        "n_features": int(len(features)),
        "features": features,
        "metrics": metrics,
    }

    eval_model["metadata"] = metadata
    final_model["metadata"] = metadata
    final_model["model_role"] = "final_refit_all_played"
    eval_model["model_role"] = "evaluation_temporal_split"

    return {
        "eval_model": eval_model,
        "final_model": final_model,
        "validation_predictions": df_val_pred,
        "metrics": metrics,
        "metadata": metadata,
        "features": features,
    }
