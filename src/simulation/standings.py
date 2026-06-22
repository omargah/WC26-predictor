# -*- coding: utf-8 -*-
"""
src/simulation/standings.py

Tablas de grupo y desempates tipo FIFA para Mundial 2026.

Este módulo trabaja sobre partidos ya simulados o fijados, con columnas:

    group
    home
    away
    home_goals
    away_goals

Regla metodológica:
    - Dentro de un grupo, primero se ordena por puntos totales.
    - Si hay empate en puntos, se aplican criterios de duelo directo
      entre los equipos empatados:
          1. puntos en partidos entre empatados;
          2. diferencia de goles en partidos entre empatados;
          3. goles anotados en partidos entre empatados.
    - Si queda empate parcial, se reevalúa el subconjunto empatado.
    - Si sigue el empate, se usan criterios globales:
          diferencia de goles total,
          goles a favor total,
          victorias totales,
          ranking_proxy,
          sorteo residual reproducible.

Limitación:
    No tenemos todavía tarjetas / fair play oficial ni ranking FIFA dinámico.
    Por eso dejamos conduct_score y ranking_proxy como campos explícitos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.simulation.groups import GROUPS


@dataclass
class RankingConfig:
    """
    Configuración de desempates residuales.

    conduct_score:
        Si en el futuro agregamos tarjetas, aquí puede entrar un puntaje
        de conducta por equipo.

    ranking_proxy:
        Si en el futuro agregamos ranking FIFA, aquí puede entrar el ranking.
        Por ahora se puede dejar vacío.

    random_seed:
        Semilla para desempate residual reproducible.
    """

    conduct_score: dict[str, float] | None = None
    ranking_proxy: dict[str, float] | None = None
    random_seed: int = 42


def result_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def normalize_matches(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza tipos mínimos de la tabla de partidos simulados.
    """

    df = df_matches.copy()

    required = ["group", "home", "away", "home_goals", "away_goals"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Faltan columnas para construir standings: "
            + ", ".join(missing)
        )

    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce").astype(int)
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce").astype(int)

    return df


def build_base_table_for_group(
    group: str,
    df_group_matches: pd.DataFrame,
    teams: list[str] | None = None,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """
    Construye tabla base de un grupo antes del ordenamiento final.
    """

    if config is None:
        config = RankingConfig()

    if teams is None:
        teams = GROUPS[group]

    rows = []

    for team in teams:
        rows.append(
            {
                "group": group,
                "team": team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "points": 0,
                "conduct_score": float((config.conduct_score or {}).get(team, 0.0)),
                "ranking_proxy": float((config.ranking_proxy or {}).get(team, 0.0)),
            }
        )

    table = pd.DataFrame(rows).set_index("team")

    for _, match in df_group_matches.iterrows():
        home = match["home"]
        away = match["away"]
        gh = int(match["home_goals"])
        ga = int(match["away_goals"])

        for team in [home, away]:
            if team not in table.index:
                raise ValueError(
                    f"Equipo {team} no pertenece al grupo {group} según GROUPS."
                )

        table.loc[home, "played"] += 1
        table.loc[away, "played"] += 1

        table.loc[home, "gf"] += gh
        table.loc[home, "ga"] += ga
        table.loc[away, "gf"] += ga
        table.loc[away, "ga"] += gh

        table.loc[home, "points"] += result_points(gh, ga)
        table.loc[away, "points"] += result_points(ga, gh)

        if gh > ga:
            table.loc[home, "wins"] += 1
            table.loc[away, "losses"] += 1
        elif gh < ga:
            table.loc[away, "wins"] += 1
            table.loc[home, "losses"] += 1
        else:
            table.loc[home, "draws"] += 1
            table.loc[away, "draws"] += 1

    table["gd"] = table["gf"] - table["ga"]

    return table.reset_index()


def build_head_to_head_table(
    tied_teams: list[str],
    df_group_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye mini-tabla solo con partidos entre los equipos empatados.
    """

    tied_set = set(tied_teams)

    rows = []

    for team in tied_teams:
        rows.append(
            {
                "team": team,
                "h2h_played": 0,
                "h2h_points": 0,
                "h2h_gf": 0,
                "h2h_ga": 0,
                "h2h_gd": 0,
            }
        )

    h2h = pd.DataFrame(rows).set_index("team")

    mask = (
        df_group_matches["home"].isin(tied_set)
        &
        df_group_matches["away"].isin(tied_set)
    )

    direct_matches = df_group_matches.loc[mask].copy()

    for _, match in direct_matches.iterrows():
        home = match["home"]
        away = match["away"]
        gh = int(match["home_goals"])
        ga = int(match["away_goals"])

        h2h.loc[home, "h2h_played"] += 1
        h2h.loc[away, "h2h_played"] += 1

        h2h.loc[home, "h2h_gf"] += gh
        h2h.loc[home, "h2h_ga"] += ga
        h2h.loc[away, "h2h_gf"] += ga
        h2h.loc[away, "h2h_ga"] += gh

        h2h.loc[home, "h2h_points"] += result_points(gh, ga)
        h2h.loc[away, "h2h_points"] += result_points(ga, gh)

    h2h["h2h_gd"] = h2h["h2h_gf"] - h2h["h2h_ga"]

    return h2h.reset_index()


def _overall_sort_key(
    team: str,
    base_table: pd.DataFrame,
    residual_random: dict[str, float],
) -> tuple:
    """
    Criterios residuales cuando el duelo directo no alcanza.
    """

    r = base_table.set_index("team").loc[team]

    return (
        int(r["gd"]),
        int(r["gf"]),
        int(r["wins"]),
        float(r["conduct_score"]),
        float(r["ranking_proxy"]),
        float(residual_random[team]),
    )


def rank_tied_subset(
    tied_teams: list[str],
    df_group_matches: pd.DataFrame,
    base_table: pd.DataFrame,
    residual_random: dict[str, float],
    depth: int = 0,
    max_depth: int = 5,
) -> list[str]:
    """
    Ordena un subconjunto de equipos empatados en puntos.

    Primero intenta resolver por duelo directo.
    Si el duelo directo produce grupos aún empatados, reevalúa esos subconjuntos.
    Si ya no hay separación posible, cae a criterios globales.
    """

    if len(tied_teams) <= 1:
        return tied_teams

    if depth > max_depth:
        return sorted(
            tied_teams,
            key=lambda t: _overall_sort_key(t, base_table, residual_random),
            reverse=True,
        )

    h2h = build_head_to_head_table(tied_teams, df_group_matches)

    h2h = h2h.sort_values(
        ["h2h_points", "h2h_gd", "h2h_gf"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    h2h["h2h_key"] = list(
        zip(
            h2h["h2h_points"],
            h2h["h2h_gd"],
            h2h["h2h_gf"],
        )
    )

    unique_keys = list(dict.fromkeys(h2h["h2h_key"].tolist()))

    # Si el H2H no separa absolutamente nada, pasamos a criterios globales.
    if len(unique_keys) == 1:
        return sorted(
            tied_teams,
            key=lambda t: _overall_sort_key(t, base_table, residual_random),
            reverse=True,
        )

    ordered = []

    for key in unique_keys:
        subgroup = h2h.loc[h2h["h2h_key"] == key, "team"].tolist()

        if len(subgroup) == 1:
            ordered.extend(subgroup)
        else:
            # Reaplicación del mini-desempate al subconjunto que sigue empatado.
            sub_order = rank_tied_subset(
                tied_teams=subgroup,
                df_group_matches=df_group_matches,
                base_table=base_table,
                residual_random=residual_random,
                depth=depth + 1,
                max_depth=max_depth,
            )
            ordered.extend(sub_order)

    return ordered


def rank_group_official_like(
    group: str,
    df_group_matches: pd.DataFrame,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """
    Ordena un grupo con criterios tipo FIFA, incluyendo duelo directo.
    """

    if config is None:
        config = RankingConfig()

    df_group_matches = normalize_matches(df_group_matches)

    base = build_base_table_for_group(
        group=group,
        df_group_matches=df_group_matches,
        teams=GROUPS[group],
        config=config,
    )

    rng = np.random.default_rng(config.random_seed)
    residual_random = {
        team: float(rng.random())
        for team in base["team"].tolist()
    }

    ordered_teams = []

    # Primero se separa por puntos totales.
    points_groups = (
        base
        .sort_values(["points"], ascending=False)
        .groupby("points", sort=False)
    )

    point_values = sorted(base["points"].unique(), reverse=True)

    for points in point_values:
        tied_teams = (
            base.loc[base["points"] == points, "team"]
            .tolist()
        )

        if len(tied_teams) == 1:
            ordered_teams.extend(tied_teams)
        else:
            ordered_teams.extend(
                rank_tied_subset(
                    tied_teams=tied_teams,
                    df_group_matches=df_group_matches,
                    base_table=base,
                    residual_random=residual_random,
                )
            )

    base_indexed = base.set_index("team")
    ranked = base_indexed.loc[ordered_teams].reset_index()
    ranked["position"] = np.arange(1, len(ranked) + 1)
    ranked["tie_random_residual"] = ranked["team"].map(residual_random)

    # Columnas ordenadas
    cols = [
        "group",
        "position",
        "team",
        "played",
        "wins",
        "draws",
        "losses",
        "gf",
        "ga",
        "gd",
        "points",
        "conduct_score",
        "ranking_proxy",
        "tie_random_residual",
    ]

    return ranked[cols]


def rank_all_groups_official_like(
    df_matches: pd.DataFrame,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """
    Ordena todos los grupos.
    """

    if config is None:
        config = RankingConfig()

    df_matches = normalize_matches(df_matches)

    tables = []

    for group in sorted(GROUPS.keys()):
        df_group = df_matches[df_matches["group"] == group].copy()

        if len(df_group) == 0:
            continue

        table = rank_group_official_like(
            group=group,
            df_group_matches=df_group,
            config=config,
        )

        tables.append(table)

    if not tables:
        raise ValueError("No se generó ninguna tabla de grupo.")

    return pd.concat(tables, ignore_index=True)


def rank_best_thirds_official_like(
    standings: pd.DataFrame,
    config: RankingConfig | None = None,
) -> pd.DataFrame:
    """
    Ordena terceros de grupo.

    Aquí NO se usa duelo directo porque los terceros vienen de grupos distintos.
    """

    if config is None:
        config = RankingConfig()

    thirds = standings[standings["position"] == 3].copy()

    if thirds.empty:
        raise ValueError("No hay terceros para ordenar.")

    rng = np.random.default_rng(config.random_seed)

    thirds["third_random_residual"] = rng.random(len(thirds))

    thirds = thirds.sort_values(
        [
            "points",
            "gd",
            "gf",
            "wins",
            "conduct_score",
            "ranking_proxy",
            "third_random_residual",
        ],
        ascending=[False, False, False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    thirds["third_rank"] = np.arange(1, len(thirds) + 1)
    thirds["qualifies_as_best_third"] = thirds["third_rank"] <= 8

    return thirds


def select_qualified_teams_official_like(
    standings: pd.DataFrame,
    best_thirds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selecciona clasificados:
        - posiciones 1 y 2 de cada grupo;
        - ocho mejores terceros.
    """

    top_two = standings[standings["position"] <= 2].copy()
    qualified_thirds = best_thirds[best_thirds["qualifies_as_best_third"]].copy()

    qualified = pd.concat([top_two, qualified_thirds], ignore_index=True)

    if len(qualified) != 32:
        raise RuntimeError(
            f"Se esperaban 32 clasificados y salieron {len(qualified)}."
        )

    qualified["qualified"] = True

    return qualified.sort_values(
        ["group", "position", "third_rank"],
        na_position="last",
    ).reset_index(drop=True)


def build_standings_package(
    df_matches: pd.DataFrame,
    config: RankingConfig | None = None,
) -> dict:
    """
    Construye standings, mejores terceros y clasificados.
    """

    if config is None:
        config = RankingConfig()

    standings = rank_all_groups_official_like(df_matches, config=config)
    best_thirds = rank_best_thirds_official_like(standings, config=config)
    qualified = select_qualified_teams_official_like(standings, best_thirds)

    return {
        "standings": standings,
        "best_thirds": best_thirds,
        "qualified": qualified,
    }
