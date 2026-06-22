# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse
import json
import time
from itertools import combinations
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import get_paths
from src.simulation.knockout_engine import (
    KnockoutConfig,
    build_directed_knockout_lambdas,
    build_neutralized_knockout_predictions,
    simulate_knockout_match,
)


ROUND_LEVELS = {
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarterfinal": 3,
    "Semifinal": 4,
    "Final": 5,
    "Champion": 6,
}


def load_group_matches(played_policy: str, paths: dict) -> pd.DataFrame:
    path = paths["predictions"] / f"phase05_v2_group_matches_once_{played_policy}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_r32_bracket(played_policy: str, paths: dict) -> pd.DataFrame:
    path = paths["predictions"] / f"phase05_v2_round_of_32_{played_policy}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_team_group_map(r32: pd.DataFrame) -> dict[str, str]:
    mapping = {}
    for _, r in r32.iterrows():
        mapping[str(r["team_a"])] = str(r["group_a"])
        mapping[str(r["team_b"])] = str(r["group_b"])
    return mapping


def build_all_pair_bracket(r32: pd.DataFrame) -> pd.DataFrame:
    teams = sorted(set(r32["team_a"].astype(str)).union(set(r32["team_b"].astype(str))))
    team_group_map = build_team_group_map(r32)

    if len(teams) != 32:
        raise RuntimeError(f"Se esperaban 32 equipos en R32 y hay {len(teams)}.")

    rows = []

    for k, (a, b) in enumerate(combinations(teams, 2), start=1):
        rows.append(
            {
                "round": "Pair Cache",
                "match_number": 10000 + k,
                "round_order": k,
                "slot_a": "CACHE_A",
                "slot_b": "CACHE_B",
                "team_a": a,
                "team_b": b,
                "group_a": team_group_map.get(a),
                "group_b": team_group_map.get(b),
                "same_group_violation": False,
            }
        )

    return pd.DataFrame(rows)


def build_or_load_pair_cache(
    played_policy: str,
    paths: dict,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    cache_path = paths["predictions"] / f"phase05_v2_fast_pair_lambdas_{played_policy}.csv"

    if cache_path.exists() and not rebuild_cache:
        print(f"[fast-mc] usando cache existente: {cache_path}")
        return pd.read_csv(cache_path)

    print("[fast-mc] construyendo cache de lambdas para todos los pares KO...")

    group_matches = load_group_matches(played_policy, paths)
    r32 = load_r32_bracket(played_policy, paths)

    all_pair_bracket = build_all_pair_bracket(r32)

    config = KnockoutConfig(match_date="2026-06-28")

    directed = build_directed_knockout_lambdas(
        bracket=all_pair_bracket,
        df_group_matches=group_matches,
        paths=paths,
        config=config,
        round_label="Pair Cache",
    )

    neutral = build_neutralized_knockout_predictions(
        bracket=all_pair_bracket,
        directed_lambdas=directed,
    )

    neutral.to_csv(cache_path, index=False, encoding="utf-8")

    print(f"[fast-mc] cache guardado: {cache_path}")
    print(f"[fast-mc] pares cacheados: {len(neutral)}")

    return neutral


def build_pair_lookup(pair_cache: pd.DataFrame) -> dict:
    lookup = {}

    for _, r in pair_cache.iterrows():
        a = str(r["team_a"])
        b = str(r["team_b"])

        lookup[(a, b)] = {
            "lambda_a_90": float(r["lambda_a_90"]),
            "lambda_b_90": float(r["lambda_b_90"]),
        }

        lookup[(b, a)] = {
            "lambda_a_90": float(r["lambda_b_90"]),
            "lambda_b_90": float(r["lambda_a_90"]),
        }

    return lookup


def simulate_cached_match(
    team_a: str,
    team_b: str,
    match_number: int,
    round_label: str,
    pair_lookup: dict,
    rng: np.random.Generator,
    config: KnockoutConfig,
) -> dict:
    if (team_a, team_b) not in pair_lookup:
        raise RuntimeError(f"No hay lambdas cacheadas para {team_a} vs {team_b}.")

    lambdas = pair_lookup[(team_a, team_b)]

    row = pd.Series(
        {
            "round": round_label,
            "match_number": int(match_number),
            "slot_a": "",
            "slot_b": "",
            "team_a": team_a,
            "team_b": team_b,
            "group_a": "",
            "group_b": "",
            "lambda_a_90": lambdas["lambda_a_90"],
            "lambda_b_90": lambdas["lambda_b_90"],
        }
    )

    out = simulate_knockout_match(
        prediction_row=row,
        rng=rng,
        config=config,
    )

    out["round"] = round_label
    out["match_number"] = int(match_number)

    return out


def winner_of(results_by_match: dict, match_number: int) -> str:
    return str(results_by_match[int(match_number)]["winner"])


def loser_of(results_by_match: dict, match_number: int) -> str:
    return str(results_by_match[int(match_number)]["loser"])


def simulate_one_fast_tournament(
    r32: pd.DataFrame,
    pair_lookup: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    config = KnockoutConfig()

    results = []
    by_match = {}

    # R32
    for _, r in r32.sort_values("match_number").iterrows():
        sim = simulate_cached_match(
            team_a=str(r["team_a"]),
            team_b=str(r["team_b"]),
            match_number=int(r["match_number"]),
            round_label="Round of 32",
            pair_lookup=pair_lookup,
            rng=rng,
            config=config,
        )
        results.append(sim)
        by_match[int(sim["match_number"])] = sim

    # R16
    r16_pairings = [
        (89, 73, 74),
        (90, 75, 76),
        (91, 77, 78),
        (92, 79, 80),
        (93, 81, 82),
        (94, 83, 84),
        (95, 85, 86),
        (96, 87, 88),
    ]

    for new_match, prev_a, prev_b in r16_pairings:
        sim = simulate_cached_match(
            team_a=winner_of(by_match, prev_a),
            team_b=winner_of(by_match, prev_b),
            match_number=new_match,
            round_label="Round of 16",
            pair_lookup=pair_lookup,
            rng=rng,
            config=config,
        )
        results.append(sim)
        by_match[int(sim["match_number"])] = sim

    # QF
    qf_pairings = [
        (97, 89, 90),
        (98, 91, 92),
        (99, 93, 94),
        (100, 95, 96),
    ]

    for new_match, prev_a, prev_b in qf_pairings:
        sim = simulate_cached_match(
            team_a=winner_of(by_match, prev_a),
            team_b=winner_of(by_match, prev_b),
            match_number=new_match,
            round_label="Quarterfinal",
            pair_lookup=pair_lookup,
            rng=rng,
            config=config,
        )
        results.append(sim)
        by_match[int(sim["match_number"])] = sim

    # SF
    sf_pairings = [
        (101, 97, 98),
        (102, 99, 100),
    ]

    for new_match, prev_a, prev_b in sf_pairings:
        sim = simulate_cached_match(
            team_a=winner_of(by_match, prev_a),
            team_b=winner_of(by_match, prev_b),
            match_number=new_match,
            round_label="Semifinal",
            pair_lookup=pair_lookup,
            rng=rng,
            config=config,
        )
        results.append(sim)
        by_match[int(sim["match_number"])] = sim

    # Third place
    sim = simulate_cached_match(
        team_a=loser_of(by_match, 101),
        team_b=loser_of(by_match, 102),
        match_number=103,
        round_label="Third Place",
        pair_lookup=pair_lookup,
        rng=rng,
        config=config,
    )
    results.append(sim)
    by_match[int(sim["match_number"])] = sim

    # Final
    sim = simulate_cached_match(
        team_a=winner_of(by_match, 101),
        team_b=winner_of(by_match, 102),
        match_number=104,
        round_label="Final",
        pair_lookup=pair_lookup,
        rng=rng,
        config=config,
    )
    results.append(sim)
    by_match[int(sim["match_number"])] = sim

    results_df = pd.DataFrame(results)

    summary = {
        "seed": int(seed),
        "champion": str(by_match[104]["winner"]),
        "runner_up": str(by_match[104]["loser"]),
        "third_place": str(by_match[103]["winner"]),
        "fourth_place": str(by_match[103]["loser"]),
        "final_decided_by": str(by_match[104]["decided_by"]),
    }

    return results_df, summary


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
        winners = results.loc[results["round"] == source_round, "winner"].dropna().astype(str)

        for team in winners:
            reached[team] = max(reached.get(team, 0), ROUND_LEVELS[target_round])

    inverse = {v: k for k, v in ROUND_LEVELS.items()}

    return pd.DataFrame(
        [
            {
                "team": team,
                "reached_level": int(level),
                "reached_round": inverse[int(level)],
            }
            for team, level in reached.items()
        ]
    )


def champion_probability_table(champions: list[str], n: int) -> pd.DataFrame:
    counts = Counter(champions)

    rows = [
        {
            "team": team,
            "championships": int(count),
            "champion_probability": float(count / n),
            "n_simulations": int(n),
        }
        for team, count in counts.items()
    ]

    return (
        pd.DataFrame(rows)
        .sort_values(["champion_probability", "championships"], ascending=False)
        .reset_index(drop=True)
    )


def round_probability_table(progress: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []

    for team in sorted(progress["team"].unique()):
        sub = progress[progress["team"] == team]

        row = {"team": team}

        for round_name, level in ROUND_LEVELS.items():
            col = f"p_{round_name.lower().replace(' ', '_')}"
            row[col] = float((sub["reached_level"] >= level).mean())

        row["n_simulations"] = int(n)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["p_champion", "p_final", "p_semifinal", "p_quarterfinal"], ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Fast V2 con cache de lambdas KO."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--rebuild-cache", action="store_true")

    args = parser.parse_args()

    paths = get_paths()

    r32 = load_r32_bracket(args.played_policy, paths)

    pair_cache = build_or_load_pair_cache(
        played_policy=args.played_policy,
        paths=paths,
        rebuild_cache=args.rebuild_cache,
    )

    pair_lookup = build_pair_lookup(pair_cache)

    all_results = []
    all_progress = []
    summaries = []
    champions = []

    t0 = time.time()

    for i in range(args.n):
        sim_seed = args.seed + i

        results, summary = simulate_one_fast_tournament(
            r32=r32,
            pair_lookup=pair_lookup,
            seed=sim_seed,
        )

        results["simulation_id"] = i + 1
        results["seed"] = sim_seed

        progress = reached_rounds_from_results(results)
        progress["simulation_id"] = i + 1
        progress["seed"] = sim_seed

        summary["simulation_id"] = i + 1

        all_results.append(results)
        all_progress.append(progress)
        summaries.append(summary)
        champions.append(summary["champion"])

        if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or (i + 1) == args.n):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"[fast-mc-v2] {i + 1:,}/{args.n:,} "
                f"({100 * (i + 1) / args.n:.1f}%) | {rate:.2f} sim/s"
            )

    elapsed = time.time() - t0

    results_df = pd.concat(all_results, ignore_index=True)
    progress_df = pd.concat(all_progress, ignore_index=True)
    summaries_df = pd.DataFrame(summaries)

    champion_probs = champion_probability_table(champions, args.n)
    round_probs = round_probability_table(progress_df, args.n)

    metadata = {
        "played_policy": args.played_policy,
        "n_simulations": int(args.n),
        "base_seed": int(args.seed),
        "elapsed_seconds": float(elapsed),
        "simulations_per_second": float(args.n / elapsed if elapsed > 0 else 0.0),
        "method": "fast_cached_pair_lambdas",
        "note": (
            "Monte Carlo Fast V2 condicionado al escenario de grupos/bracket. "
            "Precalcula lambdas de todos los pares KO con estado post-grupos y "
            "simula el bracket usando cache."
        ),
    }

    out_results = paths["predictions"] / f"phase05_v2_fast_mc_match_results_{args.played_policy}.csv"
    out_progress = paths["predictions"] / f"phase05_v2_fast_mc_team_progress_{args.played_policy}.csv"
    out_champions = paths["predictions"] / f"phase05_v2_fast_mc_champion_probabilities_{args.played_policy}.csv"
    out_rounds = paths["predictions"] / f"phase05_v2_fast_mc_round_probabilities_{args.played_policy}.csv"
    out_top4 = paths["predictions"] / f"phase05_v2_fast_mc_top4_{args.played_policy}.csv"
    out_metadata = paths["reports"] / f"phase05_v2_fast_mc_metadata_{args.played_policy}.json"

    results_df.to_csv(out_results, index=False, encoding="utf-8")
    progress_df.to_csv(out_progress, index=False, encoding="utf-8")
    champion_probs.to_csv(out_champions, index=False, encoding="utf-8")
    round_probs.to_csv(out_rounds, index=False, encoding="utf-8")
    summaries_df.to_csv(out_top4, index=False, encoding="utf-8")
    out_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 5 V2 — MONTE CARLO FAST")
    print("=" * 90)
    print(f"played_policy: {args.played_policy}")
    print(f"n:             {args.n}")
    print(f"seed base:     {args.seed}")
    print(f"tiempo:        {elapsed:.2f} s")
    print(f"velocidad:     {metadata['simulations_per_second']:.2f} sim/s")

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
