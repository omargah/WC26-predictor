
# -*- coding: utf-8 -*-
"""
Carga de datos crudos de partidos históricos.

Este módulo no limpia ni transforma de fondo los datos. Su única
responsabilidad es encontrar un archivo en data/raw/ y cargarlo como
DataFrame de pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".xlsx", ".xls"}


def list_raw_files(raw_dir: str | Path) -> list[Path]:
    """
    Lista archivos soportados dentro de data/raw/.
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        return []

    files = [
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(files)


def choose_raw_file(raw_dir: str | Path, filename: Optional[str] = None) -> Path:
    """
    Elige el archivo raw que será cargado.

    Si filename se especifica, busca exactamente ese archivo.
    Si no se especifica, toma el primer archivo compatible encontrado.
    """

    raw_dir = Path(raw_dir)

    if filename is not None:
        path = raw_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo solicitado: {path}")

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Formato no soportado: {path.suffix}")

        return path

    files = list_raw_files(raw_dir)

    if not files:
        raise FileNotFoundError(
            "No hay archivos raw compatibles en data/raw/. "
            "Sube o descarga un CSV, Parquet o Excel con partidos históricos."
        )

    return files[0]


def read_raw_matches(path: str | Path) -> pd.DataFrame:
    """
    Lee un archivo de partidos históricos detectando el formato por extensión.
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Formato no soportado: {suffix}")


def load_raw_matches(raw_dir: str | Path, filename: Optional[str] = None) -> tuple[pd.DataFrame, Path]:
    """
    Carga el dataset crudo de partidos.

    Returns
    -------
    tuple[pd.DataFrame, Path]
        DataFrame cargado y ruta del archivo usado.
    """

    path = choose_raw_file(raw_dir, filename=filename)
    df = read_raw_matches(path)

    return df, path


def inspect_raw_schema(df: pd.DataFrame) -> dict:
    """
    Devuelve un resumen simple del dataset crudo.
    """

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
