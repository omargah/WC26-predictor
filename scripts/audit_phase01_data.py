# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd

from src.config import get_paths
from src.data.loaders import get_data_freshness_report, load_world_cup_2026_fixtures


def main() -> None:
    paths = get_paths()

    report = get_data_freshness_report(force_refresh=True)
    df_wc = load_world_cup_2026_fixtures(force_refresh=False)

    played = df_wc[df_wc["is_played"]].copy()
    pending = df_wc[~df_wc["is_played"]].copy()

    reports_dir = paths["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "phase01_data_freshness_report.json"
    wc_status_path = reports_dir / "phase01_worldcup_2026_status.csv"

    report_path.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    df_wc.to_csv(wc_status_path, index=False, encoding="utf-8")

    print()
    print("=" * 80)
    print("AUDITORÍA FASE 1 — DATASET Y FIXTURE MUNDIAL 2026")
    print("=" * 80)

    for key, value in report.items():
        print(f"{key}: {value}")

    print()
    print("-" * 80)
    print("ÚLTIMOS 10 PARTIDOS JUGADOS DEL MUNDIAL 2026 EN LA FUENTE")
    print("-" * 80)

    cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    if len(played) > 0:
        print(played[cols].tail(10).to_string(index=False))
    else:
        print("No hay partidos jugados del Mundial 2026 en la fuente.")

    print()
    print("-" * 80)
    print("PRÓXIMOS 10 PARTIDOS PENDIENTES DEL MUNDIAL 2026 EN LA FUENTE")
    print("-" * 80)

    if len(pending) > 0:
        print(pending[cols].head(10).to_string(index=False))
    else:
        print("No hay partidos pendientes del Mundial 2026 en la fuente.")

    print()
    print("=" * 80)
    print("ARCHIVOS GENERADOS")
    print("=" * 80)
    print(f"Reporte JSON: {report_path}")
    print(f"Status CSV:   {wc_status_path}")


if __name__ == "__main__":
    main()
