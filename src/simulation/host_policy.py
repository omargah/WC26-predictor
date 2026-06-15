
# -*- coding: utf-8 -*-
"""
Política de localía para Mundial 2026.

Este módulo evita que México, Canadá o Estados Unidos arrastren una
ventaja completa durante todo el torneo.

La política distingue:
    - localía territorial real;
    - ronda del torneo;
    - escenario conservador/moderado/afición extendida.
"""

from __future__ import annotations

from dataclasses import dataclass


HOST_COUNTRIES = {
    "Mexico": "Mexico",
    "Canada": "Canada",
    "United States": "United States",
    "USA": "United States",
}


HOST_TEAMS = {"Mexico", "Canada", "United States", "USA"}


@dataclass
class HostAdvantageResult:
    team: str
    opponent: str
    round_name: str
    venue_country: str
    scenario: str
    is_host_team: bool
    is_playing_in_own_country: bool
    host_advantage_multiplier: float
    host_advantage_label: str
    explanation: str


def normalize_team(team: str) -> str:
    team = str(team).strip()

    if team == "USA":
        return "United States"

    return team


def normalize_country(country: str) -> str:
    country = str(country).strip()

    if country == "USA":
        return "United States"

    return country


def normalize_round(round_name: str) -> str:
    value = str(round_name).strip().lower()

    if value in {"group", "group stage", "fase de grupos", "grupos"}:
        return "group"

    if value in {"round of 32", "round32", "r32", "32avos", "dieciseisavos", "16avos"}:
        return "round32"

    if value in {"round of 16", "round16", "r16", "octavos"}:
        return "round16"

    if value in {"quarter-final", "quarterfinal", "quarter-finals", "quarterfinals", "cuartos"}:
        return "quarterfinal"

    if value in {"semi-final", "semifinal", "semi-finals", "semifinals", "semifinal"}:
        return "semifinal"

    if value in {"final"}:
        return "final"

    return value


def host_team_country(team: str) -> str | None:
    team = normalize_team(team)
    return HOST_COUNTRIES.get(team)


def get_host_advantage_multiplier(
    team: str,
    opponent: str,
    round_name: str,
    venue_country: str,
    scenario: str = "base",
) -> HostAdvantageResult:
    """
    Devuelve multiplicador de localía para un equipo.

    scenario:
        base:
            Política conservadora recomendada.

        moderate:
            Mantiene localía reducida hasta octavos si el anfitrión juega
            en su país.

        diaspora:
            Agrega apoyo pequeño a México/USA cuando juegan en Estados Unidos,
            pero nunca como localía completa.
    """

    team_norm = normalize_team(team)
    opponent_norm = normalize_team(opponent)
    country_norm = normalize_country(venue_country)
    round_norm = normalize_round(round_name)
    scenario_norm = str(scenario).strip().lower()

    own_country = host_team_country(team_norm)

    is_host = own_country is not None
    is_own_country = bool(is_host and own_country == country_norm)

    multiplier = 0.0
    label = "neutral"
    explanation = "Equipo no anfitrión o sede neutral."

    # --------------------------------------------------------
    # Escenario base: recomendado para simulación principal.
    # --------------------------------------------------------

    if scenario_norm == "base":
        if is_own_country and round_norm == "group":
            multiplier = 1.00
            label = "full_home_group"
            explanation = "Anfitrión en su país durante fase de grupos."

        elif is_own_country and round_norm == "round32":
            multiplier = 0.60
            label = "partial_home_round32"
            explanation = "Anfitrión en su país en 16avos; ventaja parcial."

        elif is_own_country and round_norm == "round16":
            multiplier = 0.35
            label = "reduced_home_round16"
            explanation = "Anfitrión en su país en octavos; ventaja reducida."

        else:
            multiplier = 0.0
            label = "neutral_base"
            explanation = "Desde cuartos o fuera de su país: neutral en escenario base."

    # --------------------------------------------------------
    # Escenario moderado: algo más generoso con anfitriones.
    # --------------------------------------------------------

    elif scenario_norm == "moderate":
        if is_own_country and round_norm == "group":
            multiplier = 1.00
            label = "full_home_group"
            explanation = "Anfitrión en su país durante fase de grupos."

        elif is_own_country and round_norm == "round32":
            multiplier = 0.70
            label = "partial_home_round32_moderate"
            explanation = "Anfitrión en su país en 16avos; escenario moderado."

        elif is_own_country and round_norm == "round16":
            multiplier = 0.50
            label = "reduced_home_round16_moderate"
            explanation = "Anfitrión en su país en octavos; escenario moderado."

        elif is_own_country and round_norm == "quarterfinal":
            multiplier = 0.20
            label = "small_home_quarterfinal_moderate"
            explanation = "Anfitrión en su país en cuartos; ventaja pequeña."

        else:
            multiplier = 0.0
            label = "neutral_moderate"
            explanation = "Neutral en escenario moderado."

    # --------------------------------------------------------
    # Escenario diaspora: solo para sensibilidad.
    # --------------------------------------------------------

    elif scenario_norm == "diaspora":
        if is_own_country and round_norm == "group":
            multiplier = 1.00
            label = "full_home_group"
            explanation = "Anfitrión en su país durante fase de grupos."

        elif is_own_country and round_norm == "round32":
            multiplier = 0.65
            label = "partial_home_round32_diaspora"
            explanation = "Anfitrión en su país en 16avos."

        elif is_own_country and round_norm == "round16":
            multiplier = 0.40
            label = "reduced_home_round16_diaspora"
            explanation = "Anfitrión en su país en octavos."

        elif team_norm == "Mexico" and country_norm == "United States":
            multiplier = 0.15
            label = "mexico_us_diaspora_small"
            explanation = "Apoyo de afición mexicana en EE.UU.; no es localía completa."

        elif team_norm == "United States" and country_norm == "United States":
            multiplier = 0.20
            label = "usa_us_late_small"
            explanation = "Apoyo local estadounidense tardío; ventaja pequeña y capada."

        else:
            multiplier = 0.0
            label = "neutral_diaspora"
            explanation = "Neutral en escenario diaspora."

    else:
        raise ValueError("scenario debe ser 'base', 'moderate' o 'diaspora'.")

    return HostAdvantageResult(
        team=team_norm,
        opponent=opponent_norm,
        round_name=round_norm,
        venue_country=country_norm,
        scenario=scenario_norm,
        is_host_team=is_host,
        is_playing_in_own_country=is_own_country,
        host_advantage_multiplier=float(multiplier),
        host_advantage_label=label,
        explanation=explanation,
    )


def get_match_host_advantage(
    home_team: str,
    away_team: str,
    round_name: str,
    venue_country: str,
    scenario: str = "base",
) -> dict:
    """
    Evalúa localía para ambos equipos de un partido.
    """

    home = get_host_advantage_multiplier(
        team=home_team,
        opponent=away_team,
        round_name=round_name,
        venue_country=venue_country,
        scenario=scenario,
    )

    away = get_host_advantage_multiplier(
        team=away_team,
        opponent=home_team,
        round_name=round_name,
        venue_country=venue_country,
        scenario=scenario,
    )

    return {
        "home": home,
        "away": away,
        "net_host_advantage": home.host_advantage_multiplier - away.host_advantage_multiplier,
    }
