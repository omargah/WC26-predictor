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
        description="Simula la Ronda de 32 V2 con Poisson-Dixon-Coles, prórroga y penales."
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

    if not bracket_path.exists():
        raise FileNotFoundError(
            f"No existe {bracket_path}. Ejecuta primero:\n"
            f"python scripts/build_round_of_32_v2.py --played-policy {args.played_policy}"
        )

    if not group_matches_path.exists():
        raise FileNotFoundError(
            f"No existe {group_matches_path}. Ejecuta primero:\n"
            f"python scripts/simulate_group_matches_v2.py --played-policy {args.played_policy} --seed {args.seed}"
        )

    bracket = pd.read_csv(bracket_path)
    df_group_matches = pd.read_csv(group_matches_path)

    config = KnockoutConfig(
        match_date="2026-06-28",
    )

    package = simulate_knockout_round_once(
        bracket=bracket,
        df_group_matches=df_group_matches,
        seed=args.seed,
        paths=paths,
        config=config,
        round_label="Round of 32",
    )

    directed = package["directed_lambdas"]
    neutral = package["neutral_predictions"]
    results = package["results"]

    out_directed = paths["predictions"] / f"phase05_v2_round_of_32_directed_lambdas_{args.played_policy}.csv"
    out_neutral = paths["predictions"] / f"phase05_v2_round_of_32_neutral_predictions_{args.played_policy}.csv"
    out_results = paths["predictions"] / f"phase05_v2_round_of_32_results_{args.played_policy}.csv"

    directed.to_csv(out_directed, index=False, encoding="utf-8")
    neutral.to_csv(out_neutral, index=False, encoding="utf-8")
    results.to_csv(out_results, index=False, encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 5 V2 — SIMULACIÓN RONDA DE 32")
    print("=" * 90)

    print(f"played_policy: {args.played_policy}")
    print(f"seed:          {args.seed}")
    print(f"bracket:       {bracket_path}")
    print(f"group_matches: {group_matches_path}")

    print()
    print("-" * 90)
    print("PREDICCIONES NEUTRALIZADAS")
    print("-" * 90)

    pred_cols = [
        "match_number",
        "team_a",
        "team_b",
        "lambda_a_90",
        "lambda_b_90",
        "neutralization_method",
    ]

    print(neutral[pred_cols].to_string(index=False, formatters={
        "lambda_a_90": lambda x: f"{x:.3f}",
        "lambda_b_90": lambda x: f"{x:.3f}",
    }))

    print()
    print("-" * 90)
    print("RESULTADOS SIMULADOS")
    print("-" * 90)

    result_cols = [
        "match_number",
        "team_a",
        "team_b",
        "goals_a_90",
        "goals_b_90",
        "goals_a_et",
        "goals_b_et",
        "goals_a_total",
        "goals_b_total",
        "decided_by",
        "winner",
        "loser",
        "went_to_extra_time",
        "went_to_penalties",
    ]

    print(results[result_cols].to_string(index=False))

    print()
    print("-" * 90)
    print("ARCHIVOS GENERADOS")
    print("-" * 90)
    print(f"directed_lambdas:     {out_directed}")
    print(f"neutral_predictions:  {out_neutral}")
    print(f"results:              {out_results}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
