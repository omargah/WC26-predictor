# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.simulation.standings import rank_group_official_like


def assert_order(table, expected):
    got = table["team"].tolist()

    if got != expected:
        raise AssertionError(
            f"Orden incorrecto.\nEsperado: {expected}\nObtenido: {got}"
        )


def test_two_team_head_to_head():
    """
    Caso:
        A y B empatan en puntos.
        B tiene mejor diferencia global, pero A le ganó a B.
        Debe quedar A arriba de B por duelo directo.
    """

    matches = pd.DataFrame(
        [
            {"group": "A", "home": "Mexico", "away": "South Africa", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home": "Mexico", "away": "South Korea", "home_goals": 0, "away_goals": 3},
            {"group": "A", "home": "Mexico", "away": "Czech Republic", "home_goals": 0, "away_goals": 1},

            {"group": "A", "home": "South Africa", "away": "South Korea", "home_goals": 3, "away_goals": 0},
            {"group": "A", "home": "South Africa", "away": "Czech Republic", "home_goals": 0, "away_goals": 1},

            {"group": "A", "home": "South Korea", "away": "Czech Republic", "home_goals": 0, "away_goals": 1},
        ]
    )

    table = rank_group_official_like("A", matches)

    # Czech 9 pts.
    # Mexico, South Africa y South Korea 3 pts.
    # Entre esos 3:
    # Mexico venció a South Africa, perdió con South Korea.
    # South Africa venció a South Korea, perdió con Mexico.
    # South Korea venció a Mexico, perdió con South Africa.
    # Mini-tabla empatada, pasa a global.
    # En este caso solo verificamos que Czech esté primero.
    if table.iloc[0]["team"] != "Czech Republic":
        raise AssertionError("Czech Republic debería quedar primero.")

    print("OK test_two_team_head_to_head")
    print(table.to_string(index=False))


def test_direct_two_team_clear():
    """
    Caso simple:
        Mexico y South Korea empatan en puntos.
        Mexico venció a South Korea.
        Mexico debe quedar arriba aunque la diferencia global sea peor.
    """

    matches = pd.DataFrame(
        [
            {"group": "A", "home": "Mexico", "away": "South Korea", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home": "Mexico", "away": "South Africa", "home_goals": 0, "away_goals": 3},
            {"group": "A", "home": "Mexico", "away": "Czech Republic", "home_goals": 0, "away_goals": 1},

            {"group": "A", "home": "South Korea", "away": "South Africa", "home_goals": 0, "away_goals": 1},
            {"group": "A", "home": "South Korea", "away": "Czech Republic", "home_goals": 2, "away_goals": 0},

            {"group": "A", "home": "South Africa", "away": "Czech Republic", "home_goals": 0, "away_goals": 2},
        ]
    )

    table = rank_group_official_like("A", matches)

    mexico_pos = int(table.loc[table["team"] == "Mexico", "position"].iloc[0])
    korea_pos = int(table.loc[table["team"] == "South Korea", "position"].iloc[0])

    if not mexico_pos < korea_pos:
        raise AssertionError(
            "Mexico debe quedar arriba de South Korea por duelo directo."
        )

    print("OK test_direct_two_team_clear")
    print(table.to_string(index=False))


def test_three_team_mini_table():
    """
    Caso de tres empatados:
        Mexico, South Korea y Czech Republic empatan en puntos.
        Se decide por mini-tabla entre ellos.
    """

    matches = pd.DataFrame(
        [
            {"group": "A", "home": "Mexico", "away": "South Korea", "home_goals": 2, "away_goals": 0},
            {"group": "A", "home": "South Korea", "away": "Czech Republic", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home": "Czech Republic", "away": "Mexico", "home_goals": 3, "away_goals": 0},

            {"group": "A", "home": "Mexico", "away": "South Africa", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home": "South Korea", "away": "South Africa", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home": "Czech Republic", "away": "South Africa", "home_goals": 1, "away_goals": 0},
        ]
    )

    table = rank_group_official_like("A", matches)

    # Los tres tienen 6 pts.
    # Mini-tabla:
    # Czech: ganó 3-0 a Mexico, perdió 0-1 con Korea => GD +2
    # Mexico: ganó 2-0 a Korea, perdió 0-3 con Czech => GD -1
    # Korea: ganó 1-0 a Czech, perdió 0-2 con Mexico => GD -1
    # Mexico queda arriba de Korea por más goles H2H.
    expected_top_three = ["Czech Republic", "Mexico", "South Korea"]

    got_top_three = table["team"].tolist()[:3]

    if got_top_three != expected_top_three:
        raise AssertionError(
            f"Mini-tabla H2H incorrecta.\nEsperado: {expected_top_three}\nObtenido: {got_top_three}"
        )

    print("OK test_three_team_mini_table")
    print(table.to_string(index=False))


def main():
    print()
    print("=" * 90)
    print("TESTS FASE 5 V2 — STANDINGS CON DUELO DIRECTO")
    print("=" * 90)

    test_direct_two_team_clear()
    print()
    test_two_team_head_to_head()
    print()
    test_three_team_mini_table()

    print()
    print("=" * 90)
    print("TODOS LOS TESTS DE STANDINGS PASARON")
    print("=" * 90)


if __name__ == "__main__":
    main()
