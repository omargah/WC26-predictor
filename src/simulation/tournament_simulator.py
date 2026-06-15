# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.simulation.group_stage_simulator import simulate_all_groups
from src.simulation.knockout_match_simulator import simulate_knockout_match


R32_SLOTS = [
    {'match_id': 73, 'round': 'Round of 32', 'slot_a': '2A', 'slot_b': '2B', 'date': '2026-06-28', 'venue_country': 'TBD'},
    {'match_id': 74, 'round': 'Round of 32', 'slot_a': '1E', 'slot_b': '3?', 'allowed_thirds': list('ABCDF'), 'date': '2026-06-29', 'venue_country': 'TBD'},
    {'match_id': 75, 'round': 'Round of 32', 'slot_a': '1F', 'slot_b': '2C', 'date': '2026-06-29', 'venue_country': 'TBD'},
    {'match_id': 76, 'round': 'Round of 32', 'slot_a': '1C', 'slot_b': '2F', 'date': '2026-06-29', 'venue_country': 'TBD'},
    {'match_id': 77, 'round': 'Round of 32', 'slot_a': '1I', 'slot_b': '3?', 'allowed_thirds': list('CDFGH'), 'date': '2026-06-30', 'venue_country': 'TBD'},
    {'match_id': 78, 'round': 'Round of 32', 'slot_a': '2E', 'slot_b': '2I', 'date': '2026-06-30', 'venue_country': 'TBD'},
    {'match_id': 79, 'round': 'Round of 32', 'slot_a': '1A', 'slot_b': '3?', 'allowed_thirds': list('CEFHI'), 'date': '2026-06-30', 'venue_country': 'TBD'},
    {'match_id': 80, 'round': 'Round of 32', 'slot_a': '1L', 'slot_b': '3?', 'allowed_thirds': list('EHIJK'), 'date': '2026-07-01', 'venue_country': 'TBD'},
    {'match_id': 81, 'round': 'Round of 32', 'slot_a': '1D', 'slot_b': '3?', 'allowed_thirds': list('BEFIJ'), 'date': '2026-07-01', 'venue_country': 'TBD'},
    {'match_id': 82, 'round': 'Round of 32', 'slot_a': '1G', 'slot_b': '3?', 'allowed_thirds': list('AEHIJ'), 'date': '2026-07-01', 'venue_country': 'TBD'},
    {'match_id': 83, 'round': 'Round of 32', 'slot_a': '2K', 'slot_b': '2L', 'date': '2026-07-02', 'venue_country': 'TBD'},
    {'match_id': 84, 'round': 'Round of 32', 'slot_a': '1H', 'slot_b': '2J', 'date': '2026-07-02', 'venue_country': 'TBD'},
    {'match_id': 85, 'round': 'Round of 32', 'slot_a': '1B', 'slot_b': '3?', 'allowed_thirds': list('EFGIJ'), 'date': '2026-07-02', 'venue_country': 'TBD'},
    {'match_id': 86, 'round': 'Round of 32', 'slot_a': '1J', 'slot_b': '2H', 'date': '2026-07-03', 'venue_country': 'TBD'},
    {'match_id': 87, 'round': 'Round of 32', 'slot_a': '1K', 'slot_b': '3?', 'allowed_thirds': list('DEIJL'), 'date': '2026-07-03', 'venue_country': 'TBD'},
    {'match_id': 88, 'round': 'Round of 32', 'slot_a': '2D', 'slot_b': '2G', 'date': '2026-07-03', 'venue_country': 'TBD'},
]

NEXT_ROUNDS = [
    {'match_id': 89, 'round': 'Round of 16', 'from_a': 73, 'from_b': 75, 'date': '2026-07-04', 'venue_country': 'TBD'},
    {'match_id': 90, 'round': 'Round of 16', 'from_a': 74, 'from_b': 77, 'date': '2026-07-04', 'venue_country': 'TBD'},
    {'match_id': 91, 'round': 'Round of 16', 'from_a': 76, 'from_b': 78, 'date': '2026-07-05', 'venue_country': 'TBD'},
    {'match_id': 92, 'round': 'Round of 16', 'from_a': 79, 'from_b': 80, 'date': '2026-07-05', 'venue_country': 'TBD'},
    {'match_id': 93, 'round': 'Round of 16', 'from_a': 83, 'from_b': 84, 'date': '2026-07-06', 'venue_country': 'TBD'},
    {'match_id': 94, 'round': 'Round of 16', 'from_a': 81, 'from_b': 82, 'date': '2026-07-06', 'venue_country': 'TBD'},
    {'match_id': 95, 'round': 'Round of 16', 'from_a': 86, 'from_b': 88, 'date': '2026-07-07', 'venue_country': 'TBD'},
    {'match_id': 96, 'round': 'Round of 16', 'from_a': 85, 'from_b': 87, 'date': '2026-07-07', 'venue_country': 'TBD'},
    {'match_id': 97, 'round': 'Quarterfinal', 'from_a': 89, 'from_b': 90, 'date': '2026-07-09', 'venue_country': 'TBD'},
    {'match_id': 98, 'round': 'Quarterfinal', 'from_a': 93, 'from_b': 94, 'date': '2026-07-10', 'venue_country': 'TBD'},
    {'match_id': 99, 'round': 'Quarterfinal', 'from_a': 91, 'from_b': 92, 'date': '2026-07-11', 'venue_country': 'TBD'},
    {'match_id': 100, 'round': 'Quarterfinal', 'from_a': 95, 'from_b': 96, 'date': '2026-07-11', 'venue_country': 'TBD'},
    {'match_id': 101, 'round': 'Semifinal', 'from_a': 97, 'from_b': 98, 'date': '2026-07-14', 'venue_country': 'TBD'},
    {'match_id': 102, 'round': 'Semifinal', 'from_a': 99, 'from_b': 100, 'date': '2026-07-15', 'venue_country': 'TBD'},
    {'match_id': 103, 'round': 'Third Place', 'loser_a': 101, 'loser_b': 102, 'date': '2026-07-18', 'venue_country': 'TBD'},
    {'match_id': 104, 'round': 'Final', 'from_a': 101, 'from_b': 102, 'date': '2026-07-19', 'venue_country': 'United States'},
]


def assign_third_place_slots(qualified_third_groups: list[str]):
    third_slots = [s for s in R32_SLOTS if s.get('slot_b') == '3?']
    groups = list(qualified_third_groups)
    assignment = {}

    def backtrack(i, remaining):
        if i >= len(third_slots):
            return True
        slot = third_slots[i]
        allowed = slot.get('allowed_thirds', [])
        candidates = [g for g in remaining if g in allowed]
        for g in candidates:
            assignment[slot['match_id']] = '3' + g
            new_remaining = [x for x in remaining if x != g]
            if backtrack(i + 1, new_remaining):
                return True
            assignment.pop(slot['match_id'], None)
        return False

    ok = backtrack(0, groups)
    if not ok:
        # Fallback operativo: asignar por orden de terceros aunque no respete todos los allowed.
        for slot, g in zip(third_slots, groups):
            assignment[slot['match_id']] = '3' + g
        assignment['_warning'] = 'fallback_assignment_not_annex_c'

    return assignment


def resolve_slot(slot: str, qualifiers: dict):
    if slot not in qualifiers:
        raise KeyError(f'No existe clasificado para slot {slot}')
    return qualifiers[slot]


def simulate_tournament_once(
    project_root: str | Path,
    analysis_date: str | None = None,
    use_registered_results: bool = True,
    seed: int = 2026,
    scenario: str = 'base',
    corners_cards_mode: str = 'legacy',
):
    rng = np.random.default_rng(seed)

    group_stage = simulate_all_groups(
        project_root=project_root,
        analysis_date=analysis_date,
        use_registered_results=use_registered_results,
        seed=seed,
        corners_cards_mode=corners_cards_mode,
    )

    qualifiers = dict(group_stage['qualifiers'])
    best_thirds = group_stage['best_thirds']
    qualified_third_groups = list(best_thirds[best_thirds['qualifies_as_best_third']]['Grupo'])
    third_assignment = assign_third_place_slots(qualified_third_groups)

    winners = {}
    losers = {}
    bracket_rows = []

    for slot in R32_SLOTS:
        slot_a = slot['slot_a']
        slot_b = slot['slot_b']
        if slot_b == '3?':
            slot_b = third_assignment[slot['match_id']]
        home = resolve_slot(slot_a, qualifiers)
        away = resolve_slot(slot_b, qualifiers)
        result = simulate_knockout_match(
            home_team=home,
            away_team=away,
            match_date=slot['date'],
            round_name=slot['round'],
            venue_country=slot.get('venue_country', 'TBD'),
            project_root=project_root,
            scenario=scenario,
            rng=rng,
            corners_cards_mode=corners_cards_mode,
        )
        winners[slot['match_id']] = result['winner']
        losers[slot['match_id']] = result['loser']
        bracket_rows.append({'match_id': slot['match_id'], 'round': slot['round'], 'home_team': home, 'away_team': away, 'winner': result['winner'], 'method': result['method'], 'home_goals_90': result['home_goals_90'], 'away_goals_90': result['away_goals_90']})

    for slot in NEXT_ROUNDS:
        if 'from_a' in slot:
            home = winners[slot['from_a']]
            away = winners[slot['from_b']]
        else:
            home = losers[slot['loser_a']]
            away = losers[slot['loser_b']]

        result = simulate_knockout_match(
            home_team=home,
            away_team=away,
            match_date=slot['date'],
            round_name=slot['round'],
            venue_country=slot.get('venue_country', 'TBD'),
            project_root=project_root,
            scenario=scenario,
            rng=rng,
            corners_cards_mode=corners_cards_mode,
        )
        winners[slot['match_id']] = result['winner']
        losers[slot['match_id']] = result['loser']
        bracket_rows.append({'match_id': slot['match_id'], 'round': slot['round'], 'home_team': home, 'away_team': away, 'winner': result['winner'], 'method': result['method'], 'home_goals_90': result['home_goals_90'], 'away_goals_90': result['away_goals_90']})

    bracket = pd.DataFrame(bracket_rows)
    champion = winners[104]
    runner_up = losers[104]
    third_place = winners[103]

    return {
        'champion': champion,
        'runner_up': runner_up,
        'third_place': third_place,
        'group_stage': group_stage,
        'bracket': bracket,
        'third_assignment': third_assignment,
        'scenario': scenario,
        'seed': seed,
    }


def run_tournament_monte_carlo(
    project_root: str | Path,
    n_simulations: int = 1000,
    analysis_date: str | None = None,
    use_registered_results: bool = True,
    seed: int = 2026,
    scenario: str = 'base',
    corners_cards_mode: str = 'legacy',
):
    rows = []
    for i in range(int(n_simulations)):
        sim_seed = int(seed) + i
        out = simulate_tournament_once(
            project_root=project_root,
            analysis_date=analysis_date,
            use_registered_results=use_registered_results,
            seed=sim_seed,
            scenario=scenario,
            corners_cards_mode=corners_cards_mode,
        )
        rows.append({
            'simulation': i + 1,
            'seed': sim_seed,
            'scenario': scenario,
            'champion': out['champion'],
            'runner_up': out['runner_up'],
            'third_place': out['third_place'],
        })

    sims = pd.DataFrame(rows)
    champion_probs = sims['champion'].value_counts(normalize=True).reset_index()
    champion_probs.columns = ['team', 'champion_probability']
    champion_probs['champion_count'] = champion_probs['team'].map(sims['champion'].value_counts())
    champion_probs = champion_probs.sort_values('champion_probability', ascending=False).reset_index(drop=True)

    return {'simulations': sims, 'champion_probs': champion_probs}
