# -*- coding: utf-8 -*-

# --- Ajuste de ruta para ejecutar scripts desde terminal ---
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
Chequeo rápido del dataset.
"""

from src.data.loaders import get_data_freshness_report


def main() -> None:
    report = get_data_freshness_report(force_refresh=True)

    print()
    print("=" * 80)
    print("REPORTE DE FRESCURA DEL DATASET")
    print("=" * 80)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("=" * 80)

    fecha_max_jugada = report["fecha_maxima_partido_jugado"]

    if fecha_max_jugada.startswith("2017"):
        print("ALERTA: los resultados jugados parecen estar cortados en 2017.")
        print("No entrenes el modelo todavía.")
    else:
        print("OK: los resultados jugados no parecen estar cortados en 2017.")
        print("Revisa fecha_maxima_partido_jugado para confirmar actualización real.")


if __name__ == "__main__":
    main()
