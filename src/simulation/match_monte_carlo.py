# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def dc_factor(i: int, j: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if i == 0 and j == 0:
        return max(0.0, 1.0 - lambda_home * lambda_away * rho)
    if i == 0 and j == 1:
        return max(0.0, 1.0 + lambda_home * rho)
    if i == 1 and j == 0:
        return max(0.0, 1.0 + lambda_away * rho)
    if i == 1 and j == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def build_score_matrix(lambda_home: float, lambda_away: float, rho: float = -0.075, max_goals: int = 10):
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            p = p * dc_factor(i, j, lambda_home, lambda_away, rho)
            matrix[i, j] = p

    total = matrix.sum()
    if total > 0:
        matrix = matrix / total

    return matrix


def approximate_ci(p: float, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan
    se = np.sqrt(max(p * (1.0 - p), 0.0) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def simulate_match_monte_carlo_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    home_team: str = 'Local',
    away_team: str = 'Visitante',
    rho: float = -0.075,
    max_goals: int = 10,
    n_simulations: int = 20000,
    seed: int = 2026,
):
    rng = np.random.default_rng(seed)

    matrix = build_score_matrix(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        rho=rho,
        max_goals=max_goals,
    )

    flat_probs = matrix.ravel()
    flat_indices = np.arange(len(flat_probs))

    draws = rng.choice(flat_indices, size=int(n_simulations), replace=True, p=flat_probs)
    home_goals = draws // (max_goals + 1)
    away_goals = draws % (max_goals + 1)

    home_win = home_goals > away_goals
    draw = home_goals == away_goals
    away_win = home_goals < away_goals
    total_goals = home_goals + away_goals
    btts = (home_goals > 0) & (away_goals > 0)

    metrics = {
        'home_team': home_team,
        'away_team': away_team,
        'n_simulations': int(n_simulations),
        'seed': int(seed),
        'lambda_home': float(lambda_home),
        'lambda_away': float(lambda_away),
        'lambda_total': float(lambda_home + lambda_away),
        'prob_home_mc': float(home_win.mean()),
        'prob_draw_mc': float(draw.mean()),
        'prob_away_mc': float(away_win.mean()),
        'over_1_5_mc': float((total_goals > 1.5).mean()),
        'over_2_5_mc': float((total_goals > 2.5).mean()),
        'over_3_5_mc': float((total_goals > 3.5).mean()),
        'btts_yes_mc': float(btts.mean()),
        'mean_home_goals_mc': float(home_goals.mean()),
        'mean_away_goals_mc': float(away_goals.mean()),
        'mean_total_goals_mc': float(total_goals.mean()),
    }

    for key in ['prob_home_mc', 'prob_draw_mc', 'prob_away_mc', 'over_2_5_mc', 'btts_yes_mc']:
        low, high = approximate_ci(metrics[key], int(n_simulations))
        metrics[key + '_ci_low'] = float(low)
        metrics[key + '_ci_high'] = float(high)

    scores = pd.DataFrame({
        'home_goals': home_goals,
        'away_goals': away_goals,
    })
    scores['score'] = scores['home_goals'].astype(str) + '-' + scores['away_goals'].astype(str)

    score_counts = (
        scores['score']
        .value_counts(normalize=False)
        .reset_index()
    )
    score_counts.columns = ['score', 'count']
    score_counts['prob_mc'] = score_counts['count'] / int(n_simulations)
    score_counts = score_counts.sort_values(['count', 'score'], ascending=[False, True]).reset_index(drop=True)

    simulations = pd.DataFrame({
        'home_goals': home_goals,
        'away_goals': away_goals,
        'total_goals': total_goals,
        'result': np.where(home_win, 'home', np.where(draw, 'draw', 'away')),
    })

    return {
        'metrics': metrics,
        'score_counts': score_counts,
        'simulations': simulations,
        'score_matrix': matrix,
    }


def simulate_match_monte_carlo_from_prediction(pred: dict, n_simulations: int = 20000, seed: int = 2026, rho: float = -0.075, max_goals: int = 10):
    return simulate_match_monte_carlo_from_lambdas(
        lambda_home=float(pred.get('lambda_home')),
        lambda_away=float(pred.get('lambda_away')),
        home_team=str(pred.get('home_team', 'Local')),
        away_team=str(pred.get('away_team', 'Visitante')),
        rho=rho,
        max_goals=max_goals,
        n_simulations=n_simulations,
        seed=seed,
    )
