# -*- coding: utf-8 -*-
"""
src/simulation/tournament_v2.py

Simulador completo de una corrida del Mundial 2026.

Este módulo une:
    - resultados de grupos fixed/resimulate;
    - R32 oficial-like con Annexe C;
    - KO ronda por ronda con features actualizadas;
    - final y campeón.
"""

from __future__ import annotations

import pandas as pd

from src.config import get_paths
from src.simulation.knockout_engine_full import (
    KnockoutConfig,
    simulate_knockout_round_once,
)


def load_group_matches(played_policy: str, paths: dict | None = None) -> pd.DataFrame:
    if paths is None:
        paths = get_paths()

    path = paths["predictions"] / f"phase05_v2_group_matches_once_{played_policy}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero simulate_group_matches_v2.py."
        )

    return pd.read_csv(path)


def load_round_of_32(played_policy: str, paths: dict | None = None) -> pd.DataFrame:
    if paths is None:
        paths = get_paths()

    path = paths["predictions"] / f"phase05_v2_round_of_32_{played_policy}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero build_round_of_32_v2.py."
        )

    return pd.read_csv(path)


def build_team_group_map(r32: pd.DataFrame) -> dict[str, str]:
    mapping = {}

    for _, r in r32.iterrows():
        mapping[str(r["team_a"])] = str(r["group_a"])
        mapping[str(r["team_b"])] = str(r["group_b"])

    return mapping


def winner_of(results: pd.DataFrame, match_number: int) -> str:
    sub = results[results["match_number"].astype(int) == int(match_number)]

    if len(sub) != 1:
        raise RuntimeError(
            f"No se encontró exactamente un ganador para M{match_number}."
        )

    return str(sub.iloc[0]["winner"])


def loser_of(results: pd.DataFrame, match_number: int) -> str:
    sub = results[results["match_number"].astype(int) == int(match_number)]

    if len(sub) != 1:
        raise RuntimeError(
            f"No se encontró exactamente un perdedor para M{match_number}."
        )

    return str(sub.iloc[0]["loser"])


def build_pairing_bracket(
    pairings: list[tuple[int, int, int]],
    previous_results: pd.DataFrame,
    team_group_map: dict[str, str],
    round_label: str,
) -> pd.DataFrame:
    """
    pairings:
        [(new_match, prev_match_a, prev_match_b), ...]
    """

    rows = []

    for order, (new_match, prev_a, prev_b) in enumerate(pairings, start=1):
        team_a = winner_of(previous_results, prev_a)
        team_b = winner_of(previous_results, prev_b)

        rows.append(
            {
                "round": round_label,
                "match_number": int(new_match),
                "round_order": int(order),
                "slot_a": f"W{prev_a}",
                "slot_b": f"W{prev_b}",
                "team_a": team_a,
                "team_b": team_b,
                "group_a": team_group_map.get(team_a),
                "group_b": team_group_map.get(team_b),
                "same_group_violation": False,
            }
        )

    return pd.DataFrame(rows)


def build_third_place_bracket(
    semifinals: pd.DataFrame,
    team_group_map: dict[str, str],
) -> pd.DataFrame:
    team_a = loser_of(semifinals, 101)
    team_b = loser_of(semifinals, 102)

    return pd.DataFrame(
        [
            {
                "round": "Third Place",
                "match_number": 103,
                "round_order": 1,
                "slot_a": "L101",
                "slot_b": "L102",
                "team_a": team_a,
                "team_b": team_b,
                "group_a": team_group_map.get(team_a),
                "group_b": team_group_map.get(team_b),
                "same_group_violation": False,
            }
        ]
    )


def build_final_bracket(
    semifinals: pd.DataFrame,
    team_group_map: dict[str, str],
) -> pd.DataFrame:
    team_a = winner_of(semifinals, 101)
    team_b = winner_of(semifinals, 102)

    return pd.DataFrame(
        [
            {
                "round": "Final",
                "match_number": 104,
                "round_order": 1,
                "slot_a": "W101",
                "slot_b": "W102",
                "team_a": team_a,
                "team_b": team_b,
                "group_a": team_group_map.get(team_a),
                "group_b": team_group_map.get(team_b),
                "same_group_violation": False,
            }
        ]
    )


def simulate_full_tournament_once(
    played_policy: str = "fixed",
    seed: int = 42,
    paths: dict | None = None,
) -> dict[str, pd.DataFrame | dict]:
    if paths is None:
        paths = get_paths()

    config = KnockoutConfig()

    group_matches = load_group_matches(played_policy, paths)
    r32 = load_round_of_32(played_policy, paths)

    team_group_map = build_team_group_map(r32)

    prior: list[tuple[str, pd.DataFrame]] = []

    # Ronda de 32
    r32_pkg = simulate_knockout_round_once(
        bracket=r32,
        df_group_matches=group_matches,
        prior_ko_results=prior,
        seed=seed + 32,
        paths=paths,
        config=config,
        round_label="Round of 32",
    )

    r32_results = r32_pkg["results"]
    prior.append(("Round of 32", r32_results))

    # Octavos / Round of 16
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

    r16 = build_pairing_bracket(
        pairings=r16_pairings,
        previous_results=r32_results,
        team_group_map=team_group_map,
        round_label="Round of 16",
    )

    r16_pkg = simulate_knockout_round_once(
        bracket=r16,
        df_group_matches=group_matches,
        prior_ko_results=prior,
        seed=seed + 16,
        paths=paths,
        config=config,
        round_label="Round of 16",
    )

    r16_results = r16_pkg["results"]
    prior.append(("Round of 16", r16_results))

    # Cuartos
    qf_pairings = [
        (97, 89, 90),
        (98, 91, 92),
        (99, 93, 94),
        (100, 95, 96),
    ]

    qf = build_pairing_bracket(
        pairings=qf_pairings,
        previous_results=r16_results,
        team_group_map=team_group_map,
        round_label="Quarterfinal",
    )

    qf_pkg = simulate_knockout_round_once(
        bracket=qf,
        df_group_matches=group_matches,
        prior_ko_results=prior,
        seed=seed + 8,
        paths=paths,
        config=config,
        round_label="Quarterfinal",
    )

    qf_results = qf_pkg["results"]
    prior.append(("Quarterfinal", qf_results))

    # Semifinal
    sf_pairings = [
        (101, 97, 98),
        (102, 99, 100),
    ]

    sf = build_pairing_bracket(
        pairings=sf_pairings,
        previous_results=qf_results,
        team_group_map=team_group_map,
        round_label="Semifinal",
    )

    sf_pkg = simulate_knockout_round_once(
        bracket=sf,
        df_group_matches=group_matches,
        prior_ko_results=prior,
        seed=seed + 4,
        paths=paths,
        config=config,
        round_label="Semifinal",
    )

    sf_results = sf_pkg["results"]
    prior.append(("Semifinal", sf_results))

    # Tercer lugar
    third_place = build_third_place_bracket(sf_results, team_group_map)

    third_pkg = simulate_knockout_round_once(
        bracket=third_place,
        df_group_matches=group_matches,
        prior_ko_results=prior,
        seed=seed + 3,
        paths=paths,
        config=config,
        round_label="Third Place",
    )

    third_results = third_pkg["results"]

    # Final
    final = build_final_bracket(sf_results, team_group_map)

    final_pkg = simulate_knockout_round_once(
        bracket=final,
        df_group_matches=group_matches,
        prior_ko_results=prior,
        seed=seed + 2,
        paths=paths,
        config=config,
        round_label="Final",
    )

    final_results = final_pkg["results"]

    all_results = pd.concat(
        [
            r32_results,
            r16_results,
            qf_results,
            sf_results,
            third_results,
            final_results,
        ],
        ignore_index=True,
    )

    all_predictions = pd.concat(
        [
            r32_pkg["neutral_predictions"],
            r16_pkg["neutral_predictions"],
            qf_pkg["neutral_predictions"],
            sf_pkg["neutral_predictions"],
            third_pkg["neutral_predictions"],
            final_pkg["neutral_predictions"],
        ],
        ignore_index=True,
    )

    champion = str(final_results.iloc[0]["winner"])
    runner_up = str(final_results.iloc[0]["loser"])
    third_place_team = str(third_results.iloc[0]["winner"])
    fourth_place_team = str(third_results.iloc[0]["loser"])

    summary = {
        "played_policy": played_policy,
        "seed": int(seed),
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place_team,
        "fourth_place": fourth_place_team,
        "final_match": int(final_results.iloc[0]["match_number"]),
        "final_decided_by": str(final_results.iloc[0]["decided_by"]),
    }

    return {
        "group_matches": group_matches,
        "r32_bracket": r32,
        "all_predictions": all_predictions,
        "all_results": all_results,
        "summary": summary,
    }
