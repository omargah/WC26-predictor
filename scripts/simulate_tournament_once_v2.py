# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths
from src.simulation.tournament_v2 import simulate_full_tournament_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simula una corrida completa del Mundial 2026 con Tournament V2."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    paths = get_paths()

    package = simulate_full_tournament_once(
        played_policy=args.played_policy,
        seed=args.seed,
        paths=paths,
    )

    results = package["all_results"]
    predictions = package["all_predictions"]
    summary = package["summary"]

    out_results = paths["predictions"] / f"phase05_v2_full_tournament_results_{args.played_policy}.csv"
    out_predictions = paths["predictions"] / f"phase05_v2_full_tournament_predictions_{args.played_policy}.csv"
    out_summary = paths["reports"] / f"phase05_v2_full_tournament_summary_{args.played_policy}.json"

    results.to_csv(out_results, index=False, encoding="utf-8")
    predictions.to_csv(out_predictions, index=False, encoding="utf-8")
    out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("FASE 5 V2 — TORNEO COMPLETO, UNA CORRIDA")
    print("=" * 90)

    print(f"played_policy: {args.played_policy}")
    print(f"seed:          {args.seed}")
    print()
    print(f"Campeón:       {summary['champion']}")
    print(f"Subcampeón:    {summary['runner_up']}")
    print(f"Tercer lugar:  {summary['third_place']}")
    print(f"Cuarto lugar:  {summary['fourth_place']}")
    print(f"Final decidida por: {summary['final_decided_by']}")

    print()
    print("-" * 90)
    print("RESULTADOS KO COMPLETOS")
    print("-" * 90)

    cols = [
        "round",
        "match_number",
        "team_a",
        "team_b",
        "lambda_a_90",
        "lambda_b_90",
        "goals_a_total",
        "goals_b_total",
        "decided_by",
        "winner",
        "loser",
    ]

    print(results[cols].to_string(index=False, formatters={
        "lambda_a_90": lambda x: f"{x:.3f}",
        "lambda_b_90": lambda x: f"{x:.3f}",
    }))

    print()
    print("-" * 90)
    print("ARCHIVOS GENERADOS")
    print("-" * 90)
    print(f"results:     {out_results}")
    print(f"predictions: {out_predictions}")
    print(f"summary:     {out_summary}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
