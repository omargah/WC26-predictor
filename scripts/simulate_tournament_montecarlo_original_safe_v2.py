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
    r32 = results[results["round"] == "Round of 32"].copy()
    teams = sorted(set(r32["team_a"]).union(set(r32["team_b"])))

    reached = {team: ROUND_LEVELS["Round of 32"] for team in teams}

    advancement = [
        ("Round of 32", "Round of 16"),
        ("Round of 16", "Quarterfinal"),
        ("Quarterfinal", "Semifinal"),
        ("Semifinal", "Final"),
        ("Final", "Champion"),
    ]

    for source_round, target_round in advancement:
        winners = results.loc[
            results["round"] == source_round,
            "winner"
        ].dropna().astype(str)

        for team in winners:
            reached[team] = max(reached.get(team, 0), ROUND_LEVELS[target_round])

    inverse = {v: k for k, v in ROUND_LEVELS.items()}

    return pd.DataFrame([
        {
            "team": team,
            "reached_level": int(level),
            "reached_round": inverse[int(level)],
        }
        for team, level in reached.items()
    ])


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")


def load_done_simulations(summary_path: Path) -> set[int]:
    if not summary_path.exists():
        return set()

    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return set()

    if "simulation_id" not in df.columns:
        return set()

    return set(df["simulation_id"].dropna().astype(int).tolist())


def build_outputs(progress_path: Path, summary_path: Path, n_done: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress = pd.read_csv(progress_path)
    summaries = pd.read_csv(summary_path)

    champions = summaries["champion"].dropna().astype(str).tolist()
    counts = Counter(champions)

    champion_rows = []

    for team, count in counts.items():
        champion_rows.append({
            "team": team,
            "championships": int(count),
            "champion_probability": float(count / n_done),
            "n_simulations": int(n_done),
        })

    champion_probs = (
        pd.DataFrame(champion_rows)
        .sort_values(["champion_probability", "championships"], ascending=False)
        .reset_index(drop=True)
    )

    round_rows = []

    for team in sorted(progress["team"].unique()):
        sub = progress[progress["team"] == team]
        row = {"team": team}

        for round_name, level in ROUND_LEVELS.items():
            col = f"p_{round_name.lower().replace(' ', '_')}"
            row[col] = float((sub["reached_level"] >= level).mean())

        row["n_simulations"] = int(n_done)
        round_rows.append(row)

    round_probs = (
        pd.DataFrame(round_rows)
        .sort_values(
            ["p_champion", "p_final", "p_semifinal", "p_quarterfinal"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return champion_probs, round_probs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo V2 original/técnico con checkpoints y resume."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    parser.add_argument("--n-total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--progress-every", type=int, default=1)

    args = parser.parse_args()

    paths = get_paths()

    run_name = args.run_name or f"original_v2_{args.played_policy}_seed{args.seed}"

    run_dir = paths["predictions"] / "mc_original_v2_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "match_results.csv"
    progress_path = run_dir / "team_progress.csv"
    summary_path = run_dir / "top4_summary.csv"
    champion_path = run_dir / "champion_probabilities.csv"
    round_path = run_dir / "round_probabilities.csv"
    metadata_path = run_dir / "metadata.json"

    done = load_done_simulations(summary_path)

    print()
    print("=" * 90)
    print("MONTE CARLO V2 ORIGINAL — CHECKPOINT/RESUME")
    print("=" * 90)
    print(f"played_policy: {args.played_policy}")
    print(f"n_total:       {args.n_total}")
    print(f"seed base:     {args.seed}")
    print(f"run_name:      {run_name}")
    print(f"run_dir:       {run_dir}")
    print(f"ya hechas:     {len(done)}")
    print("=" * 90)

    t0 = time.time()

    completed_now = 0

    for sim_id in range(1, args.n_total + 1):
        if sim_id in done:
            continue

        sim_seed = args.seed + sim_id - 1

        sim_t0 = time.time()

        package = simulate_full_tournament_once(
            played_policy=args.played_policy,
            seed=sim_seed,
            paths=paths,
        )

        results = package["all_results"].copy()
        summary = dict(package["summary"])

        results["simulation_id"] = sim_id
        results["seed"] = sim_seed

        progress = reached_rounds_from_results(results)
        progress["simulation_id"] = sim_id
        progress["seed"] = sim_seed

        summary_row = pd.DataFrame([{
            "simulation_id": sim_id,
            "seed": sim_seed,
            "played_policy": args.played_policy,
            "champion": summary["champion"],
            "runner_up": summary["runner_up"],
            "third_place": summary["third_place"],
            "fourth_place": summary["fourth_place"],
            "final_decided_by": summary["final_decided_by"],
        }])

        append_csv(results_path, results)
        append_csv(progress_path, progress)
        append_csv(summary_path, summary_row)

        completed_now += 1
        done.add(sim_id)

        elapsed_sim = time.time() - sim_t0
        elapsed_total = time.time() - t0

        if args.progress_every > 0 and (
            completed_now % args.progress_every == 0
            or len(done) == args.n_total
        ):
            print(
                f"[original-v2] sim {sim_id}/{args.n_total} "
                f"| hechas total: {len(done)} "
                f"| última: {elapsed_sim:.2f}s "
                f"| sesión: {elapsed_total/60:.1f} min"
            )

        champion_probs, round_probs = build_outputs(
            progress_path=progress_path,
            summary_path=summary_path,
            n_done=len(done),
        )

        champion_probs.to_csv(champion_path, index=False, encoding="utf-8")
        round_probs.to_csv(round_path, index=False, encoding="utf-8")

        metadata = {
            "played_policy": args.played_policy,
            "n_total_requested": int(args.n_total),
            "n_completed": int(len(done)),
            "base_seed": int(args.seed),
            "run_name": run_name,
            "method": "original_technical_v2_checkpoint_resume",
            "quality_note": (
                "Modelo original/técnico: recalcula features ronda por ronda "
                "y no usa cache fast de lambdas."
            ),
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    elapsed = time.time() - t0

    champion_probs, round_probs = build_outputs(
        progress_path=progress_path,
        summary_path=summary_path,
        n_done=len(done),
    )

    print()
    print("=" * 90)
    print("MONTE CARLO V2 ORIGINAL — RESUMEN")
    print("=" * 90)
    print(f"simulaciones completadas: {len(done)} / {args.n_total}")
    print(f"tiempo de esta sesión:    {elapsed/60:.2f} min")

    print()
    print("-" * 90)
    print("TOP CAMPEÓN")
    print("-" * 90)
    print(champion_probs.head(20).to_string(index=False, formatters={
        "champion_probability": lambda x: f"{100*x:.2f}%",
    }))

    print()
    print("-" * 90)
    print("PROBABILIDADES POR RONDA — TOP 20")
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
    print("ARCHIVOS")
    print("-" * 90)
    print(f"run_dir:       {run_dir}")
    print(f"results:       {results_path}")
    print(f"progress:      {progress_path}")
    print(f"summary:       {summary_path}")
    print(f"champions:     {champion_path}")
    print(f"round_probs:   {round_path}")
    print(f"metadata:      {metadata_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
