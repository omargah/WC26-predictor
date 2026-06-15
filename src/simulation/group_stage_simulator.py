# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.interactive.group_match_selector import GROUPS, get_group_fixtures, empty_table, apply_result_to_table
from src.interactive.group_match_selector_v2 import find_registered_result, date_label
from src.prediction.flexible_full_predictor import predecir_partido_completo_flexible
from src.simulation.match_monte_carlo import build_score_matrix


def sample_group_score(pred: dict, rng, max_goals: int = 10):
    matrix = build_score_matrix(
        lambda_home=float(pred['lambda_home']),
        lambda_away=float(pred['lambda_away']),
        rho=-0.075,
        max_goals=max_goals,
    )
    flat = matrix.ravel()
    idx = rng.choice(np.arange(len(flat)), p=flat)
    gh = int(idx // (max_goals + 1))
    ga = int(idx % (max_goals + 1))
    return gh, ga


class PredictionCache:
    def __init__(self):
        self.cache = {}

    def get_group_prediction(self, fixture: dict, project_root: str | Path, corners_cards_mode: str = 'legacy'):
        date_str = str(pd.to_datetime(fixture['date']).date())
        key = (fixture['home_team'], fixture['away_team'], date_str, int(fixture.get('neutral', 1)), corners_cards_mode)
        if key in self.cache:
            return self.cache[key]

        pred = predecir_partido_completo_flexible(
            equipo_local=fixture['home_team'],
            equipo_visitante=fixture['away_team'],
            fecha_partido=date_str,
            torneo='FIFA World Cup',
            fase='Group Stage',
            ciudad=fixture.get('city', 'TBD'),
            estadio=fixture.get('stadium', 'TBD'),
            pais_sede=fixture.get('venue_country', 'TBD'),
            neutral=int(fixture.get('neutral', 1)),
            project_root=project_root,
            verbose=False,
            save=False,
            candidate_dates=[date_str],
            corners_cards_mode=corners_cards_mode,
        )
        self.cache[key] = pred
        return pred


def sort_group_table(table: pd.DataFrame):
    table = table.copy()
    table['rank_noise'] = np.random.default_rng(123).random(len(table)) * 1e-9
    table = table.sort_values(['Pts', 'DG', 'GF', 'rank_noise'], ascending=[False, False, False, True]).reset_index(drop=True)
    table = table.drop(columns=['rank_noise'], errors='ignore')
    table['Pos'] = range(1, len(table) + 1)
    return table


def simulate_single_group(
    group: str,
    project_root: str | Path,
    analysis_date: str | None = None,
    use_registered_results: bool = True,
    rng=None,
    seed: int | None = None,
    prediction_cache: PredictionCache | None = None,
    corners_cards_mode: str = 'legacy',
):
    if rng is None:
        rng = np.random.default_rng(seed)
    if prediction_cache is None:
        prediction_cache = PredictionCache()

    fixtures = get_group_fixtures()
    fixtures = fixtures[fixtures['group'] == group].sort_values('fixture_id').reset_index(drop=True)
    table = empty_table(group)
    results = []

    for _, row in fixtures.iterrows():
        fixture = row.to_dict()
        date_str = str(pd.to_datetime(fixture['date']).date())
        source = 'simulado'

        registered = None
        if use_registered_results:
            registered = find_registered_result(project_root, fixture, analysis_date=analysis_date)

        if registered is not None:
            gh, ga, source = registered
        else:
            pred = prediction_cache.get_group_prediction(
                fixture=fixture,
                project_root=project_root,
                corners_cards_mode=corners_cards_mode,
            )
            gh, ga = sample_group_score(pred, rng=rng)
            source = 'simulado_' + date_label(fixture, analysis_date)

        table = apply_result_to_table(table, fixture['home_team'], fixture['away_team'], gh, ga)
        results.append({
            'group': group,
            'matchday': int(fixture['matchday']),
            'date': date_str,
            'home_team': fixture['home_team'],
            'away_team': fixture['away_team'],
            'home_goals': int(gh),
            'away_goals': int(ga),
            'source': source,
        })

    table = sort_group_table(table)
    table['Grupo'] = group
    return table, pd.DataFrame(results)


def rank_best_thirds(all_tables: dict):
    rows = []
    for group, table in all_tables.items():
        third = table.sort_values('Pos').iloc[2].copy()
        rows.append(third)
    thirds = pd.DataFrame(rows)
    thirds = thirds.sort_values(['Pts', 'DG', 'GF', 'Equipo'], ascending=[False, False, False, True]).reset_index(drop=True)
    thirds['third_rank'] = range(1, len(thirds) + 1)
    thirds['qualifies_as_best_third'] = thirds['third_rank'] <= 8
    return thirds


def simulate_all_groups(
    project_root: str | Path,
    analysis_date: str | None = None,
    use_registered_results: bool = True,
    seed: int = 2026,
    corners_cards_mode: str = 'legacy',
):
    rng = np.random.default_rng(seed)
    cache = PredictionCache()
    all_tables = {}
    all_results = []

    for group in sorted(GROUPS.keys()):
        table, results = simulate_single_group(
            group=group,
            project_root=project_root,
            analysis_date=analysis_date,
            use_registered_results=use_registered_results,
            rng=rng,
            prediction_cache=cache,
            corners_cards_mode=corners_cards_mode,
        )
        all_tables[group] = table
        all_results.append(results)

    results_df = pd.concat(all_results, ignore_index=True)
    best_thirds = rank_best_thirds(all_tables)

    qualifiers = {}
    for group, table in all_tables.items():
        ordered = table.sort_values('Pos').reset_index(drop=True)
        qualifiers[f'1{group}'] = ordered.iloc[0]['Equipo']
        qualifiers[f'2{group}'] = ordered.iloc[1]['Equipo']

    for _, row in best_thirds[best_thirds['qualifies_as_best_third']].iterrows():
        qualifiers[f'3{row["Grupo"]}'] = row['Equipo']

    return {
        'tables': all_tables,
        'results': results_df,
        'best_thirds': best_thirds,
        'qualifiers': qualifiers,
    }
