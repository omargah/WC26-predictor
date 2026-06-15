# -*- coding: utf-8 -*-

from __future__ import annotations

TEAM_ALIASES = {
    'Mexico': ['Mexico', 'México'],
    'México': ['Mexico', 'México'],
    'United States': ['United States', 'USA', 'USMNT', 'United States of America'],
    'USA': ['United States', 'USA', 'USMNT', 'United States of America'],
    'South Korea': ['South Korea', 'Korea Republic', 'Republic of Korea'],
    'Korea Republic': ['South Korea', 'Korea Republic', 'Republic of Korea'],
    'Czechia': ['Czechia', 'Czech Republic'],
    'Czech Republic': ['Czechia', 'Czech Republic'],
    'Bosnia and Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina', 'Bosnia'],
    'Bosnia-Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina', 'Bosnia'],
    'Cape Verde': ['Cape Verde', 'Cabo Verde', 'Cape Verde Islands', 'Cabo Verde Islands'],
    'Cabo Verde': ['Cape Verde', 'Cabo Verde', 'Cape Verde Islands', 'Cabo Verde Islands'],
    'Curacao': ['Curacao', 'Curaçao'],
    'Curaçao': ['Curacao', 'Curaçao'],
    'Ivory Coast': ['Ivory Coast', "Cote d'Ivoire", "Côte d'Ivoire"],
    "Cote d'Ivoire": ['Ivory Coast', "Cote d'Ivoire", "Côte d'Ivoire"],
    "Côte d'Ivoire": ['Ivory Coast', "Cote d'Ivoire", "Côte d'Ivoire"],
    'DR Congo': ['DR Congo', 'Congo DR', 'Democratic Republic of the Congo'],
    'Congo DR': ['DR Congo', 'Congo DR', 'Democratic Republic of the Congo'],
}


def team_aliases(team: str) -> list[str]:
    team = str(team).strip()
    if team in TEAM_ALIASES:
        return TEAM_ALIASES[team]
    for canonical, aliases in TEAM_ALIASES.items():
        if team in aliases:
            return aliases
    return [team]


def canonical_team(team: str) -> str:
    team = str(team).strip()
    if team in TEAM_ALIASES:
        return team
    for canonical, aliases in TEAM_ALIASES.items():
        if team in aliases:
            return canonical
    return team
