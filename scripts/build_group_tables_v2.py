# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd

from src.config import get_paths
from src.simulation.standings import RankingConfig, build_standings_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye tablas de grupo V2 con desempates por duelo directo."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
        help="Qué simulación de grupos leer.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para desempate residual.",
    )

    args = parser.parse_args()

    paths = get_paths()

    input_path = (
        paths["predictions"]
        / f"phase05_v2_group_matches_once_{args.played_policy}.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe {input_path}. Ejecuta primero:\n"
            f"python scripts/simulate_group_matches_v2.py --played-policy {args.played_policy} --seed {args.seed}"
        )

    df_matches = pd.read_csv(input_path)

    config = RankingConfig(
        conduct_score=None,
        ranking_proxy=None,
        random_seed=args.seed,
    )

    result = build_standings_package(
        df_matches=df_matches,
        config=config,
    )

    standings = result["standings"]
    best_thirds = result["best_thirds"]
    qualified = result["qualified"]

    out_standings = paths["predictions"] / f"phase05_v2_group_standings_{args.played_policy}.csv"
    out_thirds = paths["predictions"] / f"phase05_v2_best_thirds_{args.played_policy}.csv"
    out_qualified = paths["predictions"] / f"phase05_v2_qualified_{args.played_policy}.csv"

    standings.to_csv(out_standings, index=False, encoding="utf-8")
    best_thirds.to_csv(out_thirds, index=False, encoding="utf-8")
    qualified.to_csv(out_qualified, index=False, encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 5 V2 — TABLAS DE GRUPO CON DUELO DIRECTO")
    print("=" * 90)

    print(f"played_policy: {args.played_policy}")
    print(f"input:         {input_path}")

    print()
    print("-" * 90)
    print("TABLAS DE GRUPO")
    print("-" * 90)

    for group in sorted(standings["group"].unique()):
        print()
        print(f"Grupo {group}")
        cols = [
            "position",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "gf",
            "ga",
            "gd",
            "points",
        ]
        print(
            standings[standings["group"] == group][cols]
            .to_string(index=False)
        )

    print()
    print("-" * 90)
    print("MEJORES TERCEROS")
    print("-" * 90)

    cols_thirds = [
        "third_rank",
        "group",
        "team",
        "played",
        "wins",
        "draws",
        "losses",
        "gf",
        "ga",
        "gd",
        "points",
        "qualifies_as_best_third",
    ]

    print(best_thirds[cols_thirds].to_string(index=False))

    print()
    print("-" * 90)
    print("CLASIFICADOS A RONDA DE 32")
    print("-" * 90)

    cols_q = [
        "group",
        "position",
        "team",
        "points",
        "gd",
        "gf",
    ]

    print(qualified[cols_q].to_string(index=False))

    print()
    print("-" * 90)
    print("ARCHIVOS GENERADOS")
    print("-" * 90)
    print(f"standings:  {out_standings}")
    print(f"thirds:     {out_thirds}")
    print(f"qualified:  {out_qualified}")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
