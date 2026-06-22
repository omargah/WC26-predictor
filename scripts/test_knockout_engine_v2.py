# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths
from src.simulation.knockout_engine import KnockoutConfig, simulate_knockout_round_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tests del motor KO V2."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    paths = get_paths()

    bracket_path = paths["predictions"] / f"phase05_v2_round_of_32_{args.played_policy}.csv"
    group_matches_path = paths["predictions"] / f"phase05_v2_group_matches_once_{args.played_policy}.csv"

    bracket = pd.read_csv(bracket_path)
    df_group_matches = pd.read_csv(group_matches_path)

    package = simulate_knockout_round_once(
        bracket=bracket,
        df_group_matches=df_group_matches,
        seed=args.seed,
        paths=paths,
        config=KnockoutConfig(match_date="2026-06-28"),
        round_label="Round of 32",
    )

    directed = package["directed_lambdas"]
    neutral = package["neutral_predictions"]
    results = package["results"]

    assert len(bracket) == 16, "R32 debe tener 16 cruces."
    assert len(directed) == 32, "Debe haber 32 predicciones dirigidas: A-B y B-A por cruce."
    assert len(neutral) == 16, "Debe haber 16 predicciones neutralizadas."
    assert len(results) == 16, "Debe haber 16 resultados KO."

    assert directed["lambda_home"].notna().all(), "Hay lambda_home nula."
    assert directed["lambda_away"].notna().all(), "Hay lambda_away nula."
    assert neutral["lambda_a_90"].notna().all(), "Hay lambda_a_90 nula."
    assert neutral["lambda_b_90"].notna().all(), "Hay lambda_b_90 nula."

    valid_decisions = {"90", "ET", "PEN"}
    assert set(results["decided_by"]).issubset(valid_decisions), "decided_by inválido."

    assert results["winner"].notna().all(), "Hay partidos sin ganador."
    assert results["loser"].notna().all(), "Hay partidos sin perdedor."
    assert (results["winner"] != results["loser"]).all(), "Hay ganador igual a perdedor."

    teams = results["team_a"].tolist() + results["team_b"].tolist()
    assert len(teams) == len(set(teams)), "Hay equipos repetidos en R32."

    assert not bracket["same_group_violation"].any(), "Hay violación de mismo grupo."

    print()
    print("=" * 90)
    print("TESTS FASE 5 V2 — KNOCKOUT ENGINE")
    print("=" * 90)

    print(f"played_policy: {args.played_policy}")
    print(f"seed:          {args.seed}")
    print("OK bracket:    16 cruces")
    print("OK directed:   32 lambdas dirigidas")
    print("OK neutral:    16 predicciones neutralizadas")
    print("OK results:    16 ganadores")
    print("OK KO:         prórroga/penales disponibles")
    print("OK grupos:     sin cruces del mismo grupo")

    print()
    print(results[[
        "match_number",
        "team_a",
        "team_b",
        "lambda_a_90",
        "lambda_b_90",
        "goals_a_total",
        "goals_b_total",
        "decided_by",
        "winner",
    ]].to_string(index=False, formatters={
        "lambda_a_90": lambda x: f"{x:.3f}",
        "lambda_b_90": lambda x: f"{x:.3f}",
    }))

    print()
    print("=" * 90)
    print("TODOS LOS TESTS DE KNOCKOUT ENGINE PASARON")
    print("=" * 90)


if __name__ == "__main__":
    main()
