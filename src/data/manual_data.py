
# -*- coding: utf-8 -*-
"""
Creación de archivos manuales base.

Estos archivos funcionan como punto de partida para el pipeline.
No sustituyen una verificación oficial del fixture o de las selecciones
clasificadas, pero permiten construir el proyecto de manera ordenada.

Los archivos generados se guardan en data/manual/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.aliases import TEAM_ALIASES


def create_team_aliases_csv(output_dir: str | Path) -> Path:
    """
    Crea un CSV con los alias de selecciones.

    Este archivo permite revisar manualmente cómo se normalizan los nombres.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {"alias": alias, "canonical_name": canonical}
        for alias, canonical in sorted(TEAM_ALIASES.items())
    ]

    df = pd.DataFrame(rows)
    path = output_dir / "team_aliases_manual.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    return path


def create_wc2026_venues_csv(output_dir: str | Path) -> Path:
    """
    Crea una tabla inicial de sedes del Mundial 2026.

    Importante:
    Este archivo debe revisarse y actualizarse con la fuente oficial antes
    de correr simulaciones finales. Por ahora nos sirve como catálogo manual
    para modelar sede, país anfitrión y altitud aproximada.

    La altitud se usará con cuidado:
    no es una ventaja global del país; solo aplica si el partido se juega
    realmente en esa sede.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    venues = [
        {
            "venue_id": "MEX_MEXICO_CITY",
            "city": "Mexico City",
            "country": "Mexico",
            "host_country": "Mexico",
            "altitude_m": 2240,
            "latitude": 19.4326,
            "longitude": -99.1332,
            "notes": "Altitud relevante. Revisar estadio y fixture oficial.",
        },
        {
            "venue_id": "MEX_GUADALAJARA",
            "city": "Guadalajara",
            "country": "Mexico",
            "host_country": "Mexico",
            "altitude_m": 1566,
            "latitude": 20.6597,
            "longitude": -103.3496,
            "notes": "Altitud moderada. Revisar estadio y fixture oficial.",
        },
        {
            "venue_id": "MEX_MONTERREY",
            "city": "Monterrey",
            "country": "Mexico",
            "host_country": "Mexico",
            "altitude_m": 540,
            "latitude": 25.6866,
            "longitude": -100.3161,
            "notes": "Altitud menor que CDMX/Guadalajara.",
        },
        {
            "venue_id": "USA_LOS_ANGELES",
            "city": "Los Angeles",
            "country": "United States",
            "host_country": "United States",
            "altitude_m": 71,
            "latitude": 34.0522,
            "longitude": -118.2437,
            "notes": "Sede de baja altitud.",
        },
        {
            "venue_id": "USA_SEATTLE",
            "city": "Seattle",
            "country": "United States",
            "host_country": "United States",
            "altitude_m": 52,
            "latitude": 47.6062,
            "longitude": -122.3321,
            "notes": "Sede de baja altitud.",
        },
        {
            "venue_id": "USA_DALLAS",
            "city": "Dallas",
            "country": "United States",
            "host_country": "United States",
            "altitude_m": 131,
            "latitude": 32.7767,
            "longitude": -96.7970,
            "notes": "Sede de baja altitud.",
        },
        {
            "venue_id": "USA_HOUSTON",
            "city": "Houston",
            "country": "United States",
            "host_country": "United States",
            "altitude_m": 24,
            "latitude": 29.7604,
            "longitude": -95.3698,
            "notes": "Sede de baja altitud.",
        },
        {
            "venue_id": "USA_ATLANTA",
            "city": "Atlanta",
            "country": "United States",
            "host_country": "United States",
            "altitude_m": 320,
            "latitude": 33.7490,
            "longitude": -84.3880,
            "notes": "Sede de baja/moderada altitud.",
        },
        {
            "venue_id": "USA_MIAMI",
            "city": "Miami",
            "country": "United States",
            "host_country": "United States",
            "altitude_m": 2,
            "latitude": 25.7617,
            "longitude": -80.1918,
            "notes": "Humedad puede ser relevante si se incorpora clima.",
        },
        {
            "venue_id": "CAN_TORONTO",
            "city": "Toronto",
            "country": "Canada",
            "host_country": "Canada",
            "altitude_m": 76,
            "latitude": 43.6532,
            "longitude": -79.3832,
            "notes": "Sede de baja altitud.",
        },
        {
            "venue_id": "CAN_VANCOUVER",
            "city": "Vancouver",
            "country": "Canada",
            "host_country": "Canada",
            "altitude_m": 2,
            "latitude": 49.2827,
            "longitude": -123.1207,
            "notes": "Sede de baja altitud.",
        },
    ]

    df = pd.DataFrame(venues)
    path = output_dir / "wc2026_venues_manual.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    return path


def create_fixture_template_csv(output_dir: str | Path) -> Path:
    """
    Crea una plantilla de fixture.

    Esta plantilla será reemplazada o completada cuando se cargue el fixture
    real del Mundial 2026. Por ahora define el esquema que usará el simulador.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "match_id",
        "date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "venue_id",
        "neutral",
        "notes",
    ]

    df = pd.DataFrame(columns=columns)
    path = output_dir / "wc2026_fixture_template.csv"
    df.to_csv(path, index=False, encoding="utf-8")

    return path


def create_all_manual_files(output_dir: str | Path) -> dict:
    """
    Genera todos los archivos manuales base.

    Returns
    -------
    dict
        Diccionario con nombre lógico y ruta del archivo generado.
    """

    output_dir = Path(output_dir)

    paths = {
        "team_aliases": create_team_aliases_csv(output_dir),
        "venues": create_wc2026_venues_csv(output_dir),
        "fixture_template": create_fixture_template_csv(output_dir),
    }

    return paths
