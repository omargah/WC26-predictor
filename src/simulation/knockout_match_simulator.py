# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.prediction.flexible_full_predictor import predecir_partido_completo_flexible
from src.simulation.match_monte_carlo import build_score_matrix


def apply_host_adjustment(lambda_home: float, lambda_away: float, net_host_advantage: float, strength: float = 0.10):
    factor = float(np.exp(strength * float(net_host_advantage)))
    lh = float(lambda_home) * factor
    la = float(lambda_away) / factor
    return max(lh, 0.05), max(la, 0.05)


def get_net_host_advantage_safe(home_team: str, away_team: str, round_name: str, venue_country: str, scenario: str):
    try:
        from src.simulation.host_policy import get_match_host_advantage
        result = get_match_host_advantage(
            home_team=home_team,
            away_team=away_team,
            round_name=round_name,
            venue_country=venue_country,
            scenario=scenario,
        )
        return float(result.get('net_host_advantage', 0.0))
    except Exception:
        return 0.0


def sample_score_from_lambdas(lambda_home: float, lambda_away: float, rng, rho: float = -0.075, max_goals: int = 10):
    matrix = build_score_matrix(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        rho=rho,
        max_goals=max_goals,
    )
    flat = matrix.ravel()
    idx = rng.choice(np.arange(len(flat)), p=flat)
    gh = int(idx // (max_goals + 1))
    ga = int(idx % (max_goals + 1))
    return gh, ga


def penalty_win_probability(lambda_home: float, lambda_away: float):
    diff = float(lambda_home) - float(lambda_away)
    p = 0.50 + 0.06 * np.tanh(diff)
    return float(np.clip(p, 0.42, 0.58))


def simulate_penalties(home_team: str, away_team: str, lambda_home: float, lambda_away: float, rng):
    p_home = penalty_win_probability(lambda_home, lambda_away)
    home_wins = bool(rng.random() < p_home)
    winner = home_team if home_wins else away_team
    loser = away_team if home_wins else home_team

    # Marcador decorativo de penales.
    if home_wins:
        pens_home = int(rng.choice([4, 5], p=[0.35, 0.65]))
        pens_away = int(max(0, pens_home - rng.choice([1, 2], p=[0.75, 0.25])))
    else:
        pens_away = int(rng.choice([4, 5], p=[0.35, 0.65]))
        pens_home = int(max(0, pens_away - rng.choice([1, 2], p=[0.75, 0.25])))

    return {
        'winner': winner,
        'loser': loser,
        'penalties_home': pens_home,
        'penalties_away': pens_away,
        'penalty_home_win_probability': p_home,
    }


def predict_knockout_base(
    home_team: str,
    away_team: str,
    match_date: str,
    round_name: str,
    venue_country: str = 'TBD',
    project_root: str | Path = '.',
    scenario: str = 'base',
    corners_cards_mode: str = 'legacy',
):
    pred = predecir_partido_completo_flexible(
        equipo_local=home_team,
        equipo_visitante=away_team,
        fecha_partido=match_date,
        torneo='FIFA World Cup',
        fase=round_name,
        ciudad='TBD',
        estadio='TBD',
        pais_sede=venue_country,
        neutral=1,
        project_root=project_root,
        verbose=False,
        save=False,
        candidate_dates=[match_date],
        corners_cards_mode=corners_cards_mode,
    )

    net_adv = get_net_host_advantage_safe(
        home_team=home_team,
        away_team=away_team,
        round_name=round_name,
        venue_country=venue_country,
        scenario=scenario,
    )

    lh, la = apply_host_adjustment(
        pred['lambda_home'],
        pred['lambda_away'],
        net_host_advantage=net_adv,
    )

    pred = dict(pred)
    pred['lambda_home_raw'] = pred['lambda_home']
    pred['lambda_away_raw'] = pred['lambda_away']
    pred['lambda_home'] = lh
    pred['lambda_away'] = la
    pred['lambda_total'] = lh + la
    pred['scenario'] = scenario
    pred['round_name'] = round_name
    pred['venue_country'] = venue_country
    pred['net_host_advantage'] = net_adv

    return pred


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    match_date: str,
    round_name: str,
    venue_country: str = 'TBD',
    project_root: str | Path = '.',
    scenario: str = 'base',
    seed: int | None = None,
    rng=None,
    corners_cards_mode: str = 'legacy',
):
    if rng is None:
        rng = np.random.default_rng(seed)

    pred = predict_knockout_base(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        round_name=round_name,
        venue_country=venue_country,
        project_root=project_root,
        scenario=scenario,
        corners_cards_mode=corners_cards_mode,
    )

    lh = float(pred['lambda_home'])
    la = float(pred['lambda_away'])

    gh90, ga90 = sample_score_from_lambdas(lh, la, rng=rng)

    if gh90 > ga90:
        winner = home_team
        loser = away_team
        method = '90min'
        return {**pred, 'home_goals_90': gh90, 'away_goals_90': ga90, 'home_goals_et': 0, 'away_goals_et': 0, 'winner': winner, 'loser': loser, 'method': method}

    if gh90 < ga90:
        winner = away_team
        loser = home_team
        method = '90min'
        return {**pred, 'home_goals_90': gh90, 'away_goals_90': ga90, 'home_goals_et': 0, 'away_goals_et': 0, 'winner': winner, 'loser': loser, 'method': method}

    # Tiempo extra: 30 minutos aprox. Se usa una fracción conservadora de lambdas.
    et_home_lambda = lh / 3.0 * 0.90
    et_away_lambda = la / 3.0 * 0.90
    ghet, gaet = sample_score_from_lambdas(et_home_lambda, et_away_lambda, rng=rng, max_goals=5)

    total_home = gh90 + ghet
    total_away = ga90 + gaet

    if total_home > total_away:
        return {**pred, 'home_goals_90': gh90, 'away_goals_90': ga90, 'home_goals_et': ghet, 'away_goals_et': gaet, 'winner': home_team, 'loser': away_team, 'method': 'extra_time'}

    if total_home < total_away:
        return {**pred, 'home_goals_90': gh90, 'away_goals_90': ga90, 'home_goals_et': ghet, 'away_goals_et': gaet, 'winner': away_team, 'loser': home_team, 'method': 'extra_time'}

    pens = simulate_penalties(home_team, away_team, lh, la, rng)

    return {
        **pred,
        'home_goals_90': gh90,
        'away_goals_90': ga90,
        'home_goals_et': ghet,
        'away_goals_et': gaet,
        'winner': pens['winner'],
        'loser': pens['loser'],
        'method': 'penalties',
        'penalties_home': pens['penalties_home'],
        'penalties_away': pens['penalties_away'],
        'penalty_home_win_probability': pens['penalty_home_win_probability'],
    }
