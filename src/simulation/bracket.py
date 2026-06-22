# -*- coding: utf-8 -*-
"""
src/simulation/bracket.py

Construcción oficial-like de la ronda de 32 del Mundial 2026.

Este módulo usa:
    - clasificados de standings.py;
    - Annexe C extraído del reglamento FIFA;
    - estructura oficial M73-M88.

No simula partidos todavía. Solo arma cruces.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import get_paths


ANNEXE_COLUMNS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


ROUND_OF_32_TEMPLATE = [
    {
        "match_number": 73,
        "round32_order": 1,
        "slot_a": "2A",
        "slot_b": "2B",
        "annexe_column": None,
        "description": "Runner-up A v Runner-up B",
    },
    {
        "match_number": 74,
        "round32_order": 2,
        "slot_a": "1E",
        "slot_b": None,
        "annexe_column": "1E",
        "description": "Winner E v Annexe C third-place team",
    },
    {
        "match_number": 75,
        "round32_order": 3,
        "slot_a": "1F",
        "slot_b": "2C",
        "annexe_column": None,
        "description": "Winner F v Runner-up C",
    },
    {
        "match_number": 76,
        "round32_order": 4,
        "slot_a": "1C",
        "slot_b": "2F",
        "annexe_column": None,
        "description": "Winner C v Runner-up F",
    },
    {
        "match_number": 77,
        "round32_order": 5,
        "slot_a": "1I",
        "slot_b": None,
        "annexe_column": "1I",
        "description": "Winner I v Annexe C third-place team",
    },
    {
        "match_number": 78,
        "round32_order": 6,
        "slot_a": "2E",
        "slot_b": "2I",
        "annexe_column": None,
        "description": "Runner-up E v Runner-up I",
    },
    {
        "match_number": 79,
        "round32_order": 7,
        "slot_a": "1A",
        "slot_b": None,
        "annexe_column": "1A",
        "description": "Winner A v Annexe C third-place team",
    },
    {
        "match_number": 80,
        "round32_order": 8,
        "slot_a": "1L",
        "slot_b": None,
        "annexe_column": "1L",
        "description": "Winner L v Annexe C third-place team",
    },
    {
        "match_number": 81,
        "round32_order": 9,
        "slot_a": "1D",
        "slot_b": None,
        "annexe_column": "1D",
        "description": "Winner D v Annexe C third-place team",
    },
    {
        "match_number": 82,
        "round32_order": 10,
        "slot_a": "1G",
        "slot_b": None,
        "annexe_column": "1G",
        "description": "Winner G v Annexe C third-place team",
    },
    {
        "match_number": 83,
        "round32_order": 11,
        "slot_a": "2K",
        "slot_b": "2L",
        "annexe_column": None,
        "description": "Runner-up K v Runner-up L",
    },
    {
        "match_number": 84,
        "round32_order": 12,
        "slot_a": "1H",
        "slot_b": "2J",
        "annexe_column": None,
        "description": "Winner H v Runner-up J",
    },
    {
        "match_number": 85,
        "round32_order": 13,
        "slot_a": "1B",
        "slot_b": None,
        "annexe_column": "1B",
        "description": "Winner B v Annexe C third-place team",
    },
    {
        "match_number": 86,
        "round32_order": 14,
        "slot_a": "1J",
        "slot_b": "2H",
        "annexe_column": None,
        "description": "Winner J v Runner-up H",
    },
    {
        "match_number": 87,
        "round32_order": 15,
        "slot_a": "1K",
        "slot_b": None,
        "annexe_column": "1K",
        "description": "Winner K v Annexe C third-place team",
    },
    {
        "match_number": 88,
        "round32_order": 16,
        "slot_a": "2D",
        "slot_b": "2G",
        "annexe_column": None,
        "description": "Runner-up D v Runner-up G",
    },
]


def load_annexe_c_table(paths: dict | None = None) -> pd.DataFrame:
    if paths is None:
        paths = get_paths()

    path = (
        paths["data"]
        / "external"
        / "fifa_2026_annexe_c_third_place_combinations.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero:\\n"
            f"python scripts/build_annexe_c_v2.py"
        )

    df = pd.read_csv(path)

    required = ["option"] + ANNEXE_COLUMNS + ["third_groups_key"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError("Annexe C incompleto. Faltan: " + ", ".join(missing))

    if len(df) != 495:
        raise ValueError(f"Annexe C debe tener 495 filas; tiene {len(df)}.")

    return df


def slot_group(slot: str) -> str:
    return slot[-1]


def slot_position(slot: str) -> int:
    return int(slot[0])


def third_groups_key_from_qualified(qualified: pd.DataFrame) -> str:
    thirds = qualified[qualified["position"].astype(int) == 3].copy()

    groups = sorted(thirds["group"].astype(str).tolist())

    if len(groups) != 8:
        raise ValueError(
            f"Se esperaban 8 terceros clasificados y salieron {len(groups)}: {groups}"
        )

    return "".join(groups)


def get_annexe_c_row(
    qualified: pd.DataFrame,
    annexe_c: pd.DataFrame,
) -> pd.Series:
    key = third_groups_key_from_qualified(qualified)

    row = annexe_c[annexe_c["third_groups_key"] == key]

    if len(row) != 1:
        raise RuntimeError(
            f"No se encontró exactamente una fila de Annexe C para key={key}. "
            f"Filas encontradas: {len(row)}"
        )

    return row.iloc[0]


def team_for_slot(qualified: pd.DataFrame, slot: str) -> dict:
    """
    Devuelve datos del equipo correspondiente a un slot:
        1A, 2B, 3F, etc.
    """

    pos = slot_position(slot)
    group = slot_group(slot)

    sub = qualified[
        (qualified["group"].astype(str) == group)
        &
        (qualified["position"].astype(int) == pos)
    ].copy()

    if len(sub) != 1:
        raise RuntimeError(
            f"No se encontró exactamente un equipo para slot {slot}. "
            f"Encontrados: {len(sub)}"
        )

    r = sub.iloc[0]

    return {
        "slot": slot,
        "group": str(r["group"]),
        "position": int(r["position"]),
        "team": str(r["team"]),
        "points": int(r["points"]),
        "gd": int(r["gd"]),
        "gf": int(r["gf"]),
    }


def build_round_of_32(
    qualified: pd.DataFrame,
    annexe_c: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye cruces oficiales de ronda de 32.
    """

    annexe_row = get_annexe_c_row(qualified, annexe_c)

    rows = []

    for item in ROUND_OF_32_TEMPLATE:
        slot_a = item["slot_a"]

        if item["annexe_column"] is None:
            slot_b = item["slot_b"]
            annexe_option = None
            third_groups_key = None
        else:
            slot_b = str(annexe_row[item["annexe_column"]])
            annexe_option = int(annexe_row["option"])
            third_groups_key = str(annexe_row["third_groups_key"])

        a = team_for_slot(qualified, slot_a)
        b = team_for_slot(qualified, slot_b)

        same_group = a["group"] == b["group"]

        rows.append(
            {
                "round": "Round of 32",
                "match_number": item["match_number"],
                "round32_order": item["round32_order"],
                "slot_a": slot_a,
                "slot_b": slot_b,
                "team_a": a["team"],
                "team_b": b["team"],
                "group_a": a["group"],
                "group_b": b["group"],
                "position_a": a["position"],
                "position_b": b["position"],
                "annexe_column": item["annexe_column"],
                "annexe_option": annexe_option,
                "third_groups_key": third_groups_key,
                "same_group_violation": same_group,
                "description": item["description"],
            }
        )

    bracket = pd.DataFrame(rows)

    if len(bracket) != 16:
        raise RuntimeError(f"La ronda de 32 debe tener 16 partidos; tiene {len(bracket)}.")

    if bracket["same_group_violation"].any():
        bad = bracket[bracket["same_group_violation"]]
        raise RuntimeError(
            "Hay cruce de equipos del mismo grupo en R32:\\n"
            + bad.to_string(index=False)
        )

    teams = bracket["team_a"].tolist() + bracket["team_b"].tolist()

    if len(teams) != len(set(teams)):
        duplicates = sorted({t for t in teams if teams.count(t) > 1})
        raise RuntimeError(
            "Hay equipos repetidos en R32: " + ", ".join(duplicates)
        )

    return bracket


def load_qualified_for_policy(
    played_policy: str,
    paths: dict | None = None,
) -> pd.DataFrame:
    if paths is None:
        paths = get_paths()

    path = (
        paths["predictions"]
        / f"phase05_v2_qualified_{played_policy}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero:\\n"
            f"python scripts/build_group_tables_v2.py --played-policy {played_policy} --seed 42"
        )

    return pd.read_csv(path)
