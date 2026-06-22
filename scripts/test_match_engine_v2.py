# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import get_paths
from src.simulation.fixtures import load_worldcup_group_fixture_v2
from src.simulation.match_engine import simulate_group_match_row, simulate_single_match_many


def print_case(title, result):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)
    for k, v in result.items():
        print(f"{k}: {v}")


def main() -> None:
    paths = get_paths()

    df_wc = load_worldcup_group_fixture_v2(paths)

    rng = np.random.default_rng(42)

    # Caso 1: partido ya jugado fijo
    played_row = df_wc[df_wc["is_played"]].iloc[-1]
    fixed_result = simulate_group_match_row(
        row=played_row,
        rng=rng,
        played_policy="fixed",
    )

    # Caso 2: mismo partido ya jugado, pero simulado contrafactualmente
    resim_result = simulate_group_match_row(
        row=played_row,
        rng=rng,
        played_policy="resimulate",
    )

    # Caso 3: partido pendiente
    pending_row = df_wc[~df_wc["is_played"]].iloc[0]
    pending_result = simulate_group_match_row(
        row=pending_row,
        rng=rng,
        played_policy="fixed",
    )

    print()
    print("=" * 90)
    print("TEST FASE 5 V2 — MATCH ENGINE")
    print("=" * 90)

    print_case("Partido ya jugado con played_policy='fixed'", fixed_result)
    print_case("Partido ya jugado con played_policy='resimulate'", resim_result)
    print_case("Partido pendiente con played_policy='fixed'", pending_result)

    print()
    print("-" * 90)
    print("Simulación múltiple de partido ya jugado")
    print("-" * 90)

    sims = simulate_single_match_many(
        row=played_row,
        n=1000,
        seed=123,
    )

    summary = (
        sims["result_1x2"]
        .value_counts(normalize=True)
        .rename_axis("result_1x2")
        .reset_index(name="probability")
    )

    print(summary.to_string(index=False))

    print()
    print("=" * 90)
    print("TEST COMPLETADO")
    print("=" * 90)


if __name__ == "__main__":
    main()
