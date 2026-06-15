
# -*- coding: utf-8 -*-
"""
Auditoría legacy vs modelo actualizado.

Este módulo compara las predicciones del primer Colab contra las predicciones
del modelo actualizado.

No entrena modelos. No modifica resultados. Solo audita cambios.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_benchmark_table(project_root: str | Path) -> pd.DataFrame:
    """
    Carga la tabla principal de benchmarks.
    """

    project_root = Path(project_root)
    path = project_root / "data" / "manual" / "legacy_vs_updated_model_benchmarks.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero el bloque de benchmarks."
        )

    return pd.read_csv(path)


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula diferencias entre modelo actualizado y primer Colab.

    delta = new - old

    Si las columnas new_* todavía están vacías, los deltas quedan vacíos.
    """

    df = df.copy()

    metrics = [
        "lambda_home",
        "lambda_away",
        "lambda_total",
        "prob_home",
        "prob_draw",
        "prob_away",
        "over_1_5",
        "over_2_5",
        "over_3_5",
        "btts_yes",
        "corners_home",
        "corners_away",
        "corners_total",
        "cards_home",
        "cards_away",
        "cards_total",
    ]

    for metric in metrics:
        old_col = f"old_{metric}"
        new_col = f"new_{metric}"

        if old_col in df.columns and new_col in df.columns:
            df[f"delta_{metric}"] = pd.to_numeric(df[new_col], errors="coerce") - pd.to_numeric(df[old_col], errors="coerce")
            df[f"abs_delta_{metric}"] = df[f"delta_{metric}"].abs()

    return df


def flag_large_changes(
    df: pd.DataFrame,
    prob_threshold: float = 0.10,
    lambda_threshold: float = 0.35,
    corners_threshold: float = 1.25,
    cards_threshold: float = 0.75,
) -> pd.DataFrame:
    """
    Marca cambios grandes.

    Umbrales:
        prob_threshold:
            0.10 equivale a 10 puntos porcentuales.

        lambda_threshold:
            0.35 goles esperados.

        corners_threshold:
            1.25 córners esperados.

        cards_threshold:
            0.75 tarjetas esperadas.
    """

    df = df.copy()
    flags = []

    for _, row in df.iterrows():
        row_flags = []

        for metric in ["prob_home", "prob_draw", "prob_away", "over_2_5", "btts_yes"]:
            col = f"abs_delta_{metric}"
            if col in df.columns and pd.notna(row.get(col)) and row[col] > prob_threshold:
                row_flags.append(f"{metric}>{prob_threshold}")

        for metric in ["lambda_home", "lambda_away", "lambda_total"]:
            col = f"abs_delta_{metric}"
            if col in df.columns and pd.notna(row.get(col)) and row[col] > lambda_threshold:
                row_flags.append(f"{metric}>{lambda_threshold}")

        for metric in ["corners_home", "corners_away", "corners_total"]:
            col = f"abs_delta_{metric}"
            if col in df.columns and pd.notna(row.get(col)) and row[col] > corners_threshold:
                row_flags.append(f"{metric}>{corners_threshold}")

        for metric in ["cards_home", "cards_away", "cards_total"]:
            col = f"abs_delta_{metric}"
            if col in df.columns and pd.notna(row.get(col)) and row[col] > cards_threshold:
                row_flags.append(f"{metric}>{cards_threshold}")

        flags.append("; ".join(row_flags))

    df["large_change_flags"] = flags
    df["has_large_change"] = df["large_change_flags"].astype(str).str.len() > 0

    return df


def build_audit_table(project_root: str | Path) -> pd.DataFrame:
    """
    Construye tabla auditada completa.
    """

    df = load_benchmark_table(project_root)
    df = compute_deltas(df)
    df = flag_large_changes(df)

    return df


def save_audit_report(project_root: str | Path) -> Path:
    """
    Guarda el reporte de auditoría.
    """

    project_root = Path(project_root)

    df = build_audit_table(project_root)

    out_path = project_root / "reports" / "legacy_vs_updated_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False, encoding="utf-8")

    return out_path
