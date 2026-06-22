# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse
import json
import time
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths
from src.simulation.tournament_v2 import simulate_full_tournament_once


ROUND_LEVELS = {
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarterfinal": 3,
    "Semifinal": 4,
    "Final": 5,
    "Champion": 6,
}


def reached_rounds_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la ronda máxima alcanzada por equipo en una simulación.

    Criterio:
        Todos los equipos que aparecen en R32 alcanzaron Round of 32.
        Ganadores de R32 alcanzan Round of 16.
        Ganadores de R16 alcanzan Quarterfinal.
        Ganadores de QF alcanzan Semifinal.
        Ganadores de SF alcanzan Final.
        Ganador de Final alcanza Champion.
    """

    r32 = results[results["round"] == "Round of 32"].copy()

    teams = sorted(set(r32["team_a"]).union(set(r32["team_b"])))

    reached_level = {team: ROUND_LEVELS["Round of 32"] for team in teams}

    advancement = [
        ("Round of 32", "Round of 16"),
        ("Round of 16", "Quarterfinal"),
        ("Quarterfinal", "Semifinal"),
        ("Semifinal", "Final"),
        ("Final", "Champion"),
    ]

    for source_round, target_round in advancement:
        winners = results.loc[results["round"] == source_round, "winner"].dropna().astype(str)

        for team in winners:
            reached_level[team] = max(
                reached_level.get(team, 0),
                ROUND_LEVELS[target_round],
            )

    inverse = {v: k for k, v in ROUND_LEVELS.items()}

    rows = []

    for team, level in reached_level.items():
        rows.append(
            {
                "team": team,
                "reached_level": int(level),
                "reached_round": inverse[int(level)],
            }
        )

    return pd.DataFrame(rows)


def round_probability_table(progress: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []

    teams = sorted(progress["team"].unique())

    for team in teams:
        sub = progress[progress["team"] == team]

        row = {"team": team}

        for round_name, level in ROUND_LEVELS.items():
            row[f"p_{round_name.lower().replace(' ', '_')}"] = float(
                (sub["reached_level"] >= level).mean()
            )

        row["n_simulations"] = int(n)
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["p_champion", "p_final", "p_semifinal", "p_quarterfinal"],
        ascending=False,
    ).reset_index(drop=True)

    return df


def champion_probability_table(champions: list[str], n: int) -> pd.DataFrame:
    counts = Counter(champions)

    rows = []

    for team, count in counts.items():
        rows.append(
            {
                "team": team,
                "championships": int(count),
                "champion_probability": float(count / n),
                "n_simulations": int(n),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["champion_probability", "championships"], ascending=False)
        .reset_index(drop=True)
    )


def top4_table(summaries: list[dict], n: int) -> pd.DataFrame:
    rows = []

    for s in summaries:
        rows.append(
            {
                "seed": int(s["seed"]),
                "champion": s["champion"],
                "runner_up": s["runner_up"],
                "third_place": s["third_place"],
                "fourth_place": s["fourth_place"],
                "final_decided_by": s["final_decided_by"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo V2 técnico: repite Tournament V2 ya validado."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=5)

    args = parser.parse_args()

    paths = get_paths()

    all_progress = []
    all_results = []
    summaries = []
    champions = []

    t0 = time.time()

    for i in range(args.n):
        sim_seed = args.seed + i

        package = simulate_full_tournament_once(
            played_policy=args.played_policy,
            seed=sim_seed,
            paths=paths,
        )

        results = package["all_results"].copy()
        summary = package["summary"].copy()

        results["simulation_id"] = i + 1
        results["seed"] = sim_seed

        progress = reached_rounds_from_results(results)
        progress["simulation_id"] = i + 1
        progress["seed"] = sim_seed

        summary["simulation_id"] = i + 1
        summary["seed"] = sim_seed

        all_results.append(results)
        all_progress.append(progress)
        summaries.append(summary)
        champions.append(summary["champion"])

        if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or (i + 1) == args.n):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[mc-v2] {i + 1:,}/{args.n:,} "
                f"({100 * (i + 1) / args.n:.1f}%) "
                f"| {rate:.3f} sim/s"
            )

    results_df = pd.concat(all_results, ignore_index=True)
    progress_df = pd.concat(all_progress, ignore_index=True)
    summaries_df = top4_table(summaries, args.n)

    champion_probs = champion_probability_table(champions, args.n)
    round_probs = round_probability_table(progress_df, args.n)

    elapsed = time.time() - t0

    metadata = {
        "played_policy": args.played_policy,
        "n_simulations": int(args.n),
        "base_seed": int(args.seed),
        "elapsed_seconds": float(elapsed),
        "simulations_per_second": float(args.n / elapsed if elapsed > 0 else 0.0),
        "note": (
            "Monte Carlo V2 técnico condicionado al escenario de grupos/bracket "
            "ya generado para el played_policy seleccionado."
        ),
    }

    out_results = paths["predictions"] / f"phase05_v2_mc_match_results_{args.played_policy}.csv"
    out_progress = paths["predictions"] / f"phase05_v2_mc_team_progress_{args.played_policy}.csv"
    out_champions = paths["predictions"] / f"phase05_v2_mc_champion_probabilities_{args.played_policy}.csv"
    out_rounds = paths["predictions"] / f"phase05_v2_mc_round_probabilities_{args.played_policy}.csv"
    out_top4 = paths["predictions"] / f"phase05_v2_mc_top4_{args.played_policy}.csv"
    out_metadata = paths["reports"] / f"phase05_v2_mc_metadata_{args.played_policy}.json"

    results_df.to_csv(out_results, index=False, encoding="utf-8")
    progress_df.to_csv(out_progress, index=False, encoding="utf-8")
    champion_probs.to_csv(out_champions, index=False, encoding="utf-8")
    round_probs.to_csv(out_rounds, index=False, encoding="utf-8")
    summaries_df.to_csv(out_top4, index=False, encoding="utf-8")
    out_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 5 V2 — MONTE CARLO TÉCNICO")
    print("=" * 90)
    print(f"played_policy: {args.played_policy}")
    print(f"n:             {args.n}")
    print(f"seed base:     {args.seed}")
    print(f"tiempo:        {elapsed:.2f} s")
    print(f"velocidad:     {metadata['simulations_per_second']:.4f} sim/s")

    print()
    print("-" * 90)
    print("TOP CAMPEÓN")
    print("-" * 90)
    print(champion_probs.head(20).to_string(index=False, formatters={
        "champion_probability": lambda x: f"{100*x:.2f}%",
    }))

    print()
    print("-" * 90)
    print("PROBABILIDADES POR RONDA — TOP 20 POR CAMPEÓN")
    print("-" * 90)

    cols = [
        "team",
        "p_round_of_32",
        "p_round_of_16",
        "p_quarterfinal",
        "p_semifinal",
        "p_final",
        "p_champion",
    ]

    print(round_probs[cols].head(20).to_string(index=False, formatters={
        c: (lambda x: f"{100*x:.2f}%") for c in cols if c != "team"
    }))

    print()
    print("-" * 90)
    print("ARCHIVOS GENERADOS")
    print("-" * 90)
    print(f"match_results:  {out_results}")
    print(f"team_progress:  {out_progress}")
    print(f"champions:      {out_champions}")
    print(f"round_probs:    {out_rounds}")
    print(f"top4:           {out_top4}")
    print(f"metadata:       {out_metadata}")
    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
