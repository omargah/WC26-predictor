# -*- coding: utf-8 -*-
"""
src/simulation/groups.py

Grupos del Mundial 2026 usados por el simulador.

Este archivo centraliza los grupos para no repetirlos en varios scripts.
"""

GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

TEAM_TO_GROUP = {
    team: group
    for group, teams in GROUPS.items()
    for team in teams
}


def get_team_group(team: str) -> str | None:
    return TEAM_TO_GROUP.get(team)


def all_worldcup_teams() -> list[str]:
    teams = []

    for group_teams in GROUPS.values():
        teams.extend(group_teams)

    return teams
