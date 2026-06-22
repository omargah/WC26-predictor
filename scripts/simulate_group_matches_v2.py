# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np

from src.config import get_paths
from src.simulation.fixtures import load_worldcup_group_fixture_v2
from src.simulation.match_engine import simulate_group_fixture_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simula una vez los partidos de grupo con política configurable."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate", "evaluate"],
        default="fixed",
        help=(
            "fixed = respeta resultados reales; "
            "resimulate = simula también partidos ya jugados; "
            "evaluate = modo para evaluación histórica."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    paths = get_paths()

    df_wc = load_worldcup_group_fixture_v2(paths)

    rng = np.random.default_rng(args.seed)

    df_sim = simulate_group_fixture_once(
        df_wc=df_wc,
        rng=rng,
        played_policy=args.played_policy,
    )

    out_path = paths["predictions"] / f"phase05_v2_group_matches_once_{args.played_policy}.csv"
    df_sim.to_csv(out_path, index=False, encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 5 V2 — SIMULACIÓN DE PARTIDOS DE GRUPO")
    print("=" * 90)

    print(f"played_policy: {args.played_policy}")
    print(f"seed:          {args.seed}")
    print(f"partidos:      {len(df_sim)}")
    print(f"archivo:       {out_path}")

    print()
    print(df_sim.to_string(index=False))

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
