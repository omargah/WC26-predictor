# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths
from src.simulation.bracket import (
    build_round_of_32,
    load_annexe_c_table,
    load_qualified_for_policy,
    third_groups_key_from_qualified,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye la ronda de 32 V2 usando Annexe C oficial."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    args = parser.parse_args()

    paths = get_paths()

    qualified = load_qualified_for_policy(args.played_policy, paths)
    annexe_c = load_annexe_c_table(paths)

    key = third_groups_key_from_qualified(qualified)
    bracket = build_round_of_32(qualified, annexe_c)

    out_path = (
        paths["predictions"]
        / f"phase05_v2_round_of_32_{args.played_policy}.csv"
    )

    bracket.to_csv(out_path, index=False, encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 5 V2 — RONDA DE 32 OFICIAL-LIKE CON ANNEXE C")
    print("=" * 90)
    print(f"played_policy:     {args.played_policy}")
    print(f"third_groups_key:  {key}")
    print(f"annexe_option:     {bracket['annexe_option'].dropna().astype(int).unique().tolist()}")
    print(f"archivo:           {out_path}")

    print()
    print("-" * 90)
    print("CRUCES RONDA DE 32")
    print("-" * 90)

    cols = [
        "match_number",
        "slot_a",
        "team_a",
        "slot_b",
        "team_b",
        "annexe_column",
        "annexe_option",
        "same_group_violation",
    ]

    print(bracket[cols].to_string(index=False))

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
