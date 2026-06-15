
# -*- coding: utf-8 -*-
"""
FASE 3 — Modelo Poisson + Dixon-Coles compatible.

Este módulo reconstruye el modelo de goles del primer Colab:
- PoissonRegressor para goles local.
- PoissonRegressor para goles visitante.
- Split temporal para validación.
- Modelo final entrenado con todo el histórico disponible.
- Matriz de marcador exacto con ajuste Dixon-Coles.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss

warnings.filterwarnings("ignore")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_paths(project_root: str | Path) -> dict:
    root = Path(project_root)
    paths = {
        "root": root,
        "features": root / "data" / "features",
        "models": root / "models",
        "reports": root / "reports",
        "predictions": root / "data" / "predictions",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_modeling_dataset(project_root: str | Path) -> pd.DataFrame:
    paths = get_paths(project_root)
    path = paths["features"] / "modeling_dataset.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero la Fase 2."
        )

    df = pd.read_parquet(path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df[df["fecha"].notna()].copy()
    df = df.sort_values("fecha").reset_index(drop=True)
    return df


def select_numeric_features(df: pd.DataFrame, min_coverage: float = 0.30) -> list[str]:
    """
    Selecciona features numéricas como en el primer Colab.

    Excluye targets y columnas identificadoras.
    """

    exclude = {
        "goles_local",
        "goles_visitante",
        "resultado_1x2",
        "over_2_5",
        "btts",
        "match_id",
    }

    features = []

    for col in df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            coverage = df[col].notna().mean()
            nunique = df[col].nunique(dropna=True)

            if coverage >= min_coverage and nunique > 1:
                features.append(col)

    return features


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.20):
    df = df.sort_values("fecha").reset_index(drop=True)
    split_idx = int(len(df) * (1.0 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def clean_X(df: pd.DataFrame, features: list[str], medians=None):
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if medians is None:
        medians = X.median(numeric_only=True)

    X = X.fillna(medians)
    X = X.fillna(0.0)

    return X, medians


def train_poisson_package(df_train: pd.DataFrame, features: list[str], alpha: float = 0.001):
    """
    Entrena modelos Poisson para goles local y visitante.
    """

    X_train, medians = clean_X(df_train, features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model_home = PoissonRegressor(alpha=alpha, max_iter=1000)
    model_away = PoissonRegressor(alpha=alpha, max_iter=1000)

    model_home.fit(X_train_scaled, df_train["goles_local"].astype(float))
    model_away.fit(X_train_scaled, df_train["goles_visitante"].astype(float))

    return {
        "model_home": model_home,
        "model_away": model_away,
        "scaler": scaler,
        "features": features,
        "medians": medians,
        "alpha": alpha,
        "trained_at": now_iso(),
    }


def predict_lambdas(model_package: dict, df: pd.DataFrame):
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


def dixon_coles_tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float = -0.075):
    """
    Ajuste Dixon-Coles para marcadores bajos.
    """

    if x == 0 and y == 0:
        tau = 1.0 + lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        tau = 1.0 - lambda_h * rho
    elif x == 1 and y == 0:
        tau = 1.0 - lambda_a * rho
    elif x == 1 and y == 1:
        tau = 1.0 + rho
    else:
        tau = 1.0

    return max(float(tau), 0.0001)


def score_matrix(lambda_h: float, lambda_a: float, max_goals: int = 10, rho: float = -0.075):
    """
    Construye matriz de marcador exacto.
    """

    lambda_h = float(np.clip(lambda_h, 0.05, 8.0))
    lambda_a = float(np.clip(lambda_a, 0.05, 8.0))

    goals = np.arange(max_goals + 1)

    p_h = poisson.pmf(goals, lambda_h)
    p_a = poisson.pmf(goals, lambda_a)

    mat = np.outer(p_h, p_a)

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] *= dixon_coles_tau(i, j, lambda_h, lambda_a, rho=rho)

    mat = mat / mat.sum()
    return mat


def probs_from_matrix(mat: np.ndarray):
    gh, ga = np.indices(mat.shape)
    total = gh + ga

    probs = {
        "prob_local": float(mat[gh > ga].sum()),
        "prob_empate": float(mat[gh == ga].sum()),
        "prob_visitante": float(mat[gh < ga].sum()),

        "over_1_5": float(mat[total >= 2].sum()),
        "under_1_5": float(mat[total <= 1].sum()),
        "over_2_5": float(mat[total >= 3].sum()),
        "under_2_5": float(mat[total <= 2].sum()),
        "over_3_5": float(mat[total >= 4].sum()),
        "under_3_5": float(mat[total <= 3].sum()),

        "btts_si": float(mat[(gh > 0) & (ga > 0)].sum()),
        "btts_no": float(mat[(gh == 0) | (ga == 0)].sum()),
    }

    idx = np.unravel_index(np.argmax(mat), mat.shape)
    probs["marcador_mas_probable"] = f"{idx[0]}-{idx[1]}"
    probs["prob_marcador_mas_probable"] = float(mat[idx])

    return probs


def top_scorelines(mat: np.ndarray, top_n: int = 10):
    rows = []

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rows.append({
                "score": f"{i}-{j}",
                "home_goals": int(i),
                "away_goals": int(j),
                "prob": float(mat[i, j]),
            })

    rows = sorted(rows, key=lambda x: x["prob"], reverse=True)
    return rows[:top_n]


def predict_dataframe(model_package: dict, df: pd.DataFrame, max_goals: int = 10, rho: float = -0.075):
    df_out = df.copy()

    lambda_home, lambda_away = predict_lambdas(model_package, df_out)

    df_out["lambda_local"] = lambda_home
    df_out["lambda_visitante"] = lambda_away

    prob_rows = []
    top_rows = []

    for lh, la in zip(lambda_home, lambda_away):
        mat = score_matrix(lh, la, max_goals=max_goals, rho=rho)
        prob_rows.append(probs_from_matrix(mat))
        top_rows.append(top_scorelines(mat, top_n=10))

    probs_df = pd.DataFrame(prob_rows)

    df_out = pd.concat(
        [df_out.reset_index(drop=True), probs_df.reset_index(drop=True)],
        axis=1,
    )

    df_out["top_10_marcadores"] = [
        json.dumps(x, ensure_ascii=False) for x in top_rows
    ]

    df_out["pred_result_1x2"] = df_out[
        ["prob_local", "prob_empate", "prob_visitante"]
    ].values.argmax(axis=1)

    return df_out


def brier_multiclass(y_true, probs):
    y_true = np.asarray(y_true).astype(int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def evaluate_predictions(df_pred: pd.DataFrame):
    probs = df_pred[["prob_local", "prob_empate", "prob_visitante"]].values
    y_true = df_pred["resultado_1x2"].astype(int).values
    y_pred = df_pred["pred_result_1x2"].astype(int).values

    total_real = df_pred["goles_local"] + df_pred["goles_visitante"]
    total_lambda = df_pred["lambda_local"] + df_pred["lambda_visitante"]

    metrics = {
        "n_train_test_matches": int(len(df_pred)),
        "test_date_min": str(df_pred["fecha"].min().date()),
        "test_date_max": str(df_pred["fecha"].max().date()),
        "mae_goles_local": float(mean_absolute_error(df_pred["goles_local"], df_pred["lambda_local"])),
        "mae_goles_visitante": float(mean_absolute_error(df_pred["goles_visitante"], df_pred["lambda_visitante"])),
        "mae_goles_total": float(mean_absolute_error(total_real, total_lambda)),
        "accuracy_1x2": float(accuracy_score(y_true, y_pred)),
        "log_loss_1x2": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "brier_multiclass_1x2": brier_multiclass(y_true, probs),
    }

    return metrics


def run_phase03(project_root: str | Path, test_ratio: float = 0.20, alpha: float = 0.001):
    print("=" * 80)
    print("INICIANDO FASE 3 — MODELO POISSON + DIXON-COLES")
    print("=" * 80)

    paths = get_paths(project_root)

    print("\n[3.1] Cargando dataset de modelado...")
    df = load_modeling_dataset(project_root)
    print(f"   ✓ Partidos cargados: {len(df):,}")
    print(f"   ✓ Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")

    print("\n[3.2] Seleccionando features numéricas disponibles...")
    features = select_numeric_features(df, min_coverage=0.30)
    print(f"   ✓ Features seleccionadas: {len(features)}")

    print("\n[3.3] Entrenando modelos Poisson para validación temporal...")
    df_train, df_test = temporal_split(df, test_ratio=test_ratio)

    validation_package = train_poisson_package(df_train, features, alpha=alpha)
    df_pred = predict_dataframe(validation_package, df_test)

    print("   ✓ Entrenamiento de validación completado")
    print(f"   ✓ Train: {len(df_train):,} partidos")
    print(f"   ✓ Test:  {len(df_test):,} partidos")
    print(f"   ✓ Corte train hasta: {df_train['fecha'].max().date()}")

    print("\n[3.4] Métricas de validación temporal:")
    metrics = evaluate_predictions(df_pred)

    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print("\n[3.5] Reentrenando modelo final con todo el histórico disponible...")
    final_package = train_poisson_package(df, features, alpha=alpha)

    final_package["phase"] = "phase03_poisson_dc_compatible"
    final_package["model_note"] = "Modelo final entrenado con todo el histórico disponible."
    final_package["validation_metrics"] = metrics
    final_package["features"] = features
    final_package["n_training_rows_final"] = int(len(df))
    final_package["date_min_final"] = str(df["fecha"].min().date())
    final_package["date_max_final"] = str(df["fecha"].max().date())
    final_package["max_goals"] = 10
    final_package["rho"] = -0.075

    print("   ✓ Modelo final entrenado")

    print("\n[3.6] Guardando modelo y resultados...")
    model_path = paths["models"] / "poisson_dc_base.joblib"
    pred_path = paths["predictions"] / "phase03_validation_predictions.csv"
    metrics_path = paths["reports"] / "phase03_metrics.csv"
    report_path = paths["reports"] / "phase03_report.txt"
    report_json_path = paths["reports"] / "phase03_report.json"

    joblib.dump(final_package, model_path)

    df_pred.to_csv(pred_path, index=False, encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    report = {
        "created_at": now_iso(),
        "n_rows_total": int(len(df)),
        "n_features": int(len(features)),
        "features": features,
        "train_rows_validation": int(len(df_train)),
        "test_rows_validation": int(len(df_test)),
        "train_date_min": str(df_train["fecha"].min().date()),
        "train_date_max": str(df_train["fecha"].max().date()),
        "test_date_min": str(df_test["fecha"].min().date()),
        "test_date_max": str(df_test["fecha"].max().date()),
        "metrics": metrics,
        "model_path": str(model_path),
        "note": "Se validó con split temporal y se guardó modelo final entrenado con todo el histórico.",
    }

    report_json_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE FASE 3 — POISSON + DIXON-COLES COMPATIBLE\n")
        f.write("=" * 70 + "\n")
        f.write(f"Creado: {report['created_at']}\n")
        f.write(f"Filas totales: {report['n_rows_total']:,}\n")
        f.write(f"Features usadas: {report['n_features']}\n")
        f.write(f"Train validación: {report['train_rows_validation']:,}\n")
        f.write(f"Test validación: {report['test_rows_validation']:,}\n")
        f.write(f"Train rango: {report['train_date_min']} → {report['train_date_max']}\n")
        f.write(f"Test rango: {report['test_date_min']} → {report['test_date_max']}\n\n")
        f.write("Métricas:\n")
        for k, v in metrics.items():
            f.write(f"  - {k}: {v}\n")
        f.write("\nNota:\n")
        f.write(report["note"] + "\n")

    print(f"   ✓ Modelo: {model_path}")
    print(f"   ✓ Métricas: {metrics_path}")
    print(f"   ✓ Predicciones validación: {pred_path}")
    print(f"   ✓ Reporte: {report_path}")

    print("\n" + "=" * 80)
    print("FASE 3 COMPLETADA")
    print("=" * 80)

    return {
        "model_package": final_package,
        "validation_predictions": df_pred,
        "metrics": metrics,
        "features": features,
        "report": report,
        "paths": {k: str(v) for k, v in paths.items()},
    }
