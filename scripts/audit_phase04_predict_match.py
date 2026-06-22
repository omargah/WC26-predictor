# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.match import predict_match


def run_case(home, away, date=None):
    result = predict_match(home=home, away=away, date=date)

    print()
    print("-" * 80)
    print(f"{result['equipo_local']} vs {result['equipo_visitante']} | {result['fecha']}")
    print("-" * 80)
    print(f"source_type: {result['source_type']}")
    print(f"is_played: {result['is_played']}")
    print(f"λ: {result['lambda_local']:.3f} - {result['lambda_visitante']:.3f}")
    print(
        "1X2: "
        f"{100*result['prob_local']:.2f}% / "
        f"{100*result['prob_empate']:.2f}% / "
        f"{100*result['prob_visitante']:.2f}%"
    )
    print(f"Marcador más probable: {result['marcador_mas_probable']}")
    if result.get("warning"):
        print("warning:", result["warning"])


def main():
    print()
    print("=" * 80)
    print("AUDITORÍA FASE 4 — PREDICTOR MANUAL")
    print("=" * 80)

    # Partido pendiente del fixture
    run_case("Canada", "Switzerland")

    # Partido pendiente del fixture
    run_case("Mexico", "Czech Republic")

    # Partido ya jugado, debe mostrar advertencia
    run_case("Mexico", "South Korea", date="2026-06-18")

    # Partido manual no necesariamente en fixture
    result = predict_match(
        home="Mexico",
        away="Brazil",
        date="2026-07-01",
        tournament="FIFA World Cup",
        city="Manual",
        country="Unknown",
        neutral=1,
    )

    print()
    print("-" * 80)
    print("Mexico vs Brazil manual | 2026-07-01")
    print("-" * 80)
    print(f"source_type: {result['source_type']}")
    print(f"λ: {result['lambda_local']:.3f} - {result['lambda_visitante']:.3f}")
    print(
        "1X2: "
        f"{100*result['prob_local']:.2f}% / "
        f"{100*result['prob_empate']:.2f}% / "
        f"{100*result['prob_visitante']:.2f}%"
    )
    print(f"Marcador más probable: {result['marcador_mas_probable']}")
    if result.get("warning"):
        print("warning:", result["warning"])

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 4 COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
