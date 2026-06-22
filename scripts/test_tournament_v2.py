# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths
from src.simulation.tournament_v2 import simulate_full_tournament_once


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tests del Tournament V2."
    )

    parser.add_argument(
        "--played-policy",
        choices=["fixed", "resimulate"],
        default="fixed",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    paths = get_paths()

    package = simulate_full_tournament_once(
        played_policy=args.played_policy,
        seed=args.seed,
        paths=paths,
    )

    results = package["all_results"]
    predictions = package["all_predictions"]
    summary = package["summary"]

    assert len(results) == 32, f"Debe haber 32 partidos KO incluyendo tercer lugar y final; hay {len(results)}."
    assert len(predictions) == 32, f"Debe haber 32 predicciones neutralizadas; hay {len(predictions)}."

    expected_round_counts = {
        "Round of 32": 16,
        "Round of 16": 8,
        "Quarterfinal": 4,
        "Semifinal": 2,
        "Third Place": 1,
        "Final": 1,
    }

    for round_name, expected in expected_round_counts.items():
        got = int((results["round"] == round_name).sum())
        assert got == expected, f"{round_name}: esperado {expected}, obtenido {got}."

    assert summary["champion"], "No hay campeón."
    assert summary["runner_up"], "No hay subcampeón."
    assert summary["champion"] != summary["runner_up"], "Campeón y subcampeón iguales."

    assert results["winner"].notna().all(), "Hay partidos sin ganador."
    assert results["loser"].notna().all(), "Hay partidos sin perdedor."
    assert (results["winner"] != results["loser"]).all(), "Hay ganador igual a perdedor."

    valid_decisions = {"90", "ET", "PEN"}
    assert set(results["decided_by"]).issubset(valid_decisions), "decided_by inválido."

    print()
    print("=" * 90)
    print("TESTS FASE 5 V2 — TOURNAMENT")
    print("=" * 90)
    print(f"played_policy: {args.played_policy}")
    print(f"seed:          {args.seed}")
    print("OK KO total:   32 partidos")
    print("OK rondas:     16 + 8 + 4 + 2 + tercer lugar + final")
    print("OK resultados: todos con ganador")
    print("OK decisiones: 90 / ET / PEN")
    print()
    print(f"Campeón:       {summary['champion']}")
    print(f"Subcampeón:    {summary['runner_up']}")
    print(f"Tercer lugar:  {summary['third_place']}")
    print(f"Cuarto lugar:  {summary['fourth_place']}")
    print()
    print("=" * 90)
    print("TODOS LOS TESTS DE TOURNAMENT V2 PASARON")
    print("=" * 90)


if __name__ == "__main__":
    main()
