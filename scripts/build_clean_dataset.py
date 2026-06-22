# -*- coding: utf-8 -*-
# --- Ajuste de ruta para ejecutar scripts desde terminal ---
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Construye el dataset limpio de Fase 1.
"""

from src.config import get_paths
from src.data.loaders import load_raw_results, load_world_cup_2026_fixtures
from src.data.cleaners import build_clean_matches, validate_clean_matches


def main() -> None:
    paths = get_paths()

    print()
    print("=" * 80)
    print("FASE 1 — CONSTRUCCIÓN DEL DATASET LIMPIO")
    print("=" * 80)

    print()
    print("[1] Cargando dataset crudo desde GitHub...")
    df_raw = load_raw_results(force_refresh=True)

    print(f"Filas crudas: {len(df_raw):,}")
    print(f"Fecha mínima cruda: {df_raw['date'].min()}")
    print(f"Fecha máxima cruda: {df_raw['date'].max()}")
    print(f"Columnas crudas: {list(df_raw.columns)}")

    print()
    print("[2] Limpiando dataset completo...")
    df_clean = build_clean_matches(df_raw)

    problems = validate_clean_matches(df_clean)

    if problems:
        print()
        print("Problemas detectados:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print()
        print("Validación básica aprobada.")

    out_parquet = paths["processed"] / "matches_clean.parquet"
    out_csv = paths["processed"] / "matches_clean.csv"

    df_clean.to_parquet(out_parquet, index=False)
    df_clean.to_csv(out_csv, index=False, encoding="utf-8")

    print()
    print("[3] Dataset limpio guardado:")
    print(f"Parquet: {out_parquet}")
    print(f"CSV:     {out_csv}")
    print(f"Filas limpias: {len(df_clean):,}")
    print(f"Rango limpio: {df_clean['fecha'].min()} → {df_clean['fecha'].max()}")
    print(f"Partidos jugados: {int(df_clean['is_played'].sum()):,}")
    print(f"Partidos pendientes: {int((~df_clean['is_played']).sum()):,}")

    print()
    print("[4] Construyendo fixture limpio del Mundial 2026...")
    df_wc_raw = load_world_cup_2026_fixtures(force_refresh=False)
    df_wc_clean = build_clean_matches(df_wc_raw)

    out_wc = paths["processed"] / "worldcup_2026_fixture_clean.csv"
    df_wc_clean.to_csv(out_wc, index=False, encoding="utf-8")

    print(f"Fixture Mundial 2026: {out_wc}")
    print(f"Partidos Mundial 2026 total: {len(df_wc_clean):,}")
    print(f"Jugados: {int(df_wc_clean['is_played'].sum()):,}")
    print(f"Pendientes: {int((~df_wc_clean['is_played']).sum()):,}")

    print()
    print("=" * 80)
    print("FASE 1 COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
