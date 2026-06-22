# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths
from src.simulation.bracket import (
    ANNEXE_COLUMNS,
    build_round_of_32,
    load_annexe_c_table,
    load_qualified_for_policy,
    third_groups_key_from_qualified,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tests del bracket V2."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    args = parser.parse_args()

    paths = get_paths()

    annexe_c = load_annexe_c_table(paths)

    assert len(annexe_c) == 495, "Annexe C debe tener 495 filas."
    assert annexe_c["option"].nunique() == 495, "Opciones duplicadas."
    assert annexe_c["third_groups_key"].nunique() == 495, "Keys duplicadas."

    for col in ANNEXE_COLUMNS:
        assert col in annexe_c.columns, f"Falta columna {col}."

    qualified = load_qualified_for_policy(args.played_policy, paths)

    assert len(qualified) == 32, "Debe haber 32 clasificados."
    assert len(qualified["team"].unique()) == 32, "Hay equipos repetidos en clasificados."

    key = third_groups_key_from_qualified(qualified)

    bracket = build_round_of_32(qualified, annexe_c)

    assert len(bracket) == 16, "Debe haber 16 partidos de R32."
    assert not bracket["same_group_violation"].any(), "Hay cruce del mismo grupo."
    assert len(set(bracket["team_a"].tolist() + bracket["team_b"].tolist())) == 32, "Equipos repetidos en bracket."

    print()
    print("=" * 90)
    print("TESTS FASE 5 V2 — BRACKET")
    print("=" * 90)
    print(f"played_policy:    {args.played_policy}")
    print(f"third_groups_key: {key}")
    print("OK Annexe C:      495 opciones únicas")
    print("OK clasificados:  32 equipos únicos")
    print("OK R32:           16 partidos")
    print("OK grupos:        sin cruces del mismo grupo")
    print()
    print(bracket[["match_number", "slot_a", "team_a", "slot_b", "team_b"]].to_string(index=False))
    print()
    print("=" * 90)
    print("TODOS LOS TESTS DE BRACKET PASARON")
    print("=" * 90)


if __name__ == "__main__":
    main()
