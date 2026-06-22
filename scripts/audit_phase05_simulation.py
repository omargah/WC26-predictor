# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd

from src.config import get_paths


def pct(x):
    return f"{100 * float(x):.2f}%"


def main() -> None:
    paths = get_paths()

    champion_path = paths["predictions"] / "phase05_champion_probabilities.csv"
    round_path = paths["predictions"] / "phase05_round_probabilities.csv"
    group_path = paths["predictions"] / "phase05_group_position_probabilities.csv"
    bracket_path = paths["predictions"] / "phase05_sample_bracket.csv"
    report_path = paths["reports"] / "phase05_simulation_report.json"

    if not champion_path.exists():
        raise FileNotFoundError(
            "No existe phase05_champion_probabilities.csv. "
            "Ejecuta primero: python scripts/simulate_tournament_phase05.py --n 1000"
        )

    champion = pd.read_csv(champion_path)
    rounds = pd.read_csv(round_path)
    groups = pd.read_csv(group_path)
    bracket = pd.read_csv(bracket_path)

    metadata = json.loads(report_path.read_text(encoding="utf-8"))

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 5 — SIMULADOR MUNDIAL 2026")
    print("=" * 80)

    print()
    print("[1] Metadata")
    for k, v in metadata.items():
        print(f"{k}: {v}")

    print()
    print("[2] Top 20 candidatos al título")
    tmp = champion.head(20).copy()
    tmp["champion_probability"] = tmp["champion_probability"].map(pct)
    print(tmp.to_string(index=False))

    print()
    print("[3] México, Canadá y Estados Unidos")
    focus = rounds[
        rounds["team"].isin(["Mexico", "Canada", "United States"])
    ].copy()

    prob_cols = [
        "round_of_32_probability",
        "round_of_16_probability",
        "quarterfinal_probability",
        "semifinal_probability",
        "final_probability",
        "champion_probability",
    ]

    for col in prob_cols:
        focus[col] = focus[col].map(pct)

    print(focus[["team", "group"] + prob_cols].to_string(index=False))

    print()
    print("[4] Posiciones de grupo — México, Canadá y Estados Unidos")
    focus_groups = groups[
        groups["team"].isin(["Mexico", "Canada", "United States"])
    ].copy()

    for col in ["pos_1_probability", "pos_2_probability", "pos_3_probability", "pos_4_probability"]:
        focus_groups[col] = focus_groups[col].map(pct)

    print(focus_groups.to_string(index=False))

    print()
    print("[5] Ejemplo de bracket simulado")
    print(bracket.head(20).to_string(index=False))

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 5 COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
