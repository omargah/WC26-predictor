# -*- coding: utf-8 -*-

"""
Construye archivos de referencia para interfaces web/Colab.

Genera:
- data/reference/available_dates.csv
- data/reference/available_matches.csv
- data/reference/team_aliases.csv
"""

from pathlib import Path
import sys
import unicodedata
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths


ALIASES = {
    "Spain": ["Spain", "España", "Espana"],
    "Saudi Arabia": ["Saudi Arabia", "Arabia Saudita"],
    "Belgium": ["Belgium", "Bélgica", "Belgica"],
    "Iran": ["Iran", "Irán"],
    "Uruguay": ["Uruguay"],
    "Cape Verde": ["Cape Verde", "Cabo Verde"],
    "New Zealand": ["New Zealand", "Nueva Zelanda"],
    "Egypt": ["Egypt", "Egipto"],
    "Mexico": ["Mexico", "México"],
    "United States": ["United States", "USA", "Estados Unidos"],
    "Canada": ["Canada", "Canadá"],
    "Argentina": ["Argentina"],
    "Austria": ["Austria"],
    "France": ["France", "Francia"],
    "Iraq": ["Iraq", "Irak"],
    "Norway": ["Norway", "Noruega"],
    "Senegal": ["Senegal"],
    "Jordan": ["Jordan", "Jordania"],
    "Algeria": ["Algeria", "Argelia"],
    "Brazil": ["Brazil", "Brasil"],
    "Germany": ["Germany", "Alemania"],
    "England": ["England", "Inglaterra"],
    "Portugal": ["Portugal"],
    "Colombia": ["Colombia"],
    "Morocco": ["Morocco", "Marruecos"],
    "Switzerland": ["Switzerland", "Suiza"],
    "Japan": ["Japan", "Japón", "Japon"],
    "South Korea": ["South Korea", "Corea del Sur"],
    "Czechia": ["Czechia", "República Checa", "Republica Checa", "Czech Republic"],
}


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def norm_text(text: str) -> str:
    return " ".join(strip_accents(str(text)).lower().strip().split())


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    paths = get_paths()

    reference_dir = PROJECT_ROOT / "data" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)

    pred_path = paths["predictions"] / "phase03_pending_predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError(
            f"No existe {pred_path}. Ejecuta primero el predictor de pendientes de Fase 3."
        )

    df = pd.read_csv(pred_path)

    date_col = pick_col(df, ["fecha", "date", "match_date"])
    home_col = pick_col(df, ["equipo_local", "home", "home_team", "team_home", "local"])
    away_col = pick_col(df, ["equipo_visitante", "away", "away_team", "team_away", "visitante"])

    lambda_home_col = pick_col(df, ["lambda_local", "lambda_home", "lambda_goles_local", "pred_lambda_home"])
    lambda_away_col = pick_col(df, ["lambda_visitante", "lambda_away", "lambda_goles_visitante", "pred_lambda_away"])

    missing = [
        name for name, value in {
            "date_col": date_col,
            "home_col": home_col,
            "away_col": away_col,
            "lambda_home_col": lambda_home_col,
            "lambda_away_col": lambda_away_col,
        }.items()
        if value is None
    ]

    if missing:
        print("Columnas disponibles:")
        print(list(df.columns))
        raise RuntimeError("Faltan columnas necesarias: " + ", ".join(missing))

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    # Fechas disponibles
    dates = (
        work[[date_col]]
        .dropna()
        .drop_duplicates()
        .sort_values(date_col)
        .rename(columns={date_col: "date"})
    )

    dates["date"] = dates["date"].dt.date.astype(str)
    dates["label"] = dates["date"]

    # Partidos disponibles
    matches = work[[date_col, home_col, away_col, lambda_home_col, lambda_away_col]].copy()
    matches[date_col] = pd.to_datetime(matches[date_col], errors="coerce").dt.date.astype(str)

    matches = matches.rename(
        columns={
            date_col: "date",
            home_col: "home_team",
            away_col: "away_team",
            lambda_home_col: "lambda_home",
            lambda_away_col: "lambda_away",
        }
    )

    matches["match"] = matches["home_team"].astype(str) + " vs " + matches["away_team"].astype(str)
    matches["match_key"] = matches["date"] + "__" + matches["home_team"].map(norm_text) + "__vs__" + matches["away_team"].map(norm_text)
    matches["dropdown_label"] = matches["date"] + " | " + matches["match"]

    matches = matches[
        [
            "date",
            "home_team",
            "away_team",
            "match",
            "match_key",
            "dropdown_label",
            "lambda_home",
            "lambda_away",
        ]
    ].sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    # Aliases
    alias_rows = []

    teams_from_matches = sorted(
        set(matches["home_team"].astype(str)).union(set(matches["away_team"].astype(str)))
    )

    for team in teams_from_matches:
        variants = ALIASES.get(team, [team])

        for alias in variants:
            alias_rows.append(
                {
                    "canonical_team": team,
                    "alias": alias,
                    "alias_norm": norm_text(alias),
                }
            )

    alias_df = pd.DataFrame(alias_rows).drop_duplicates().sort_values(
        ["canonical_team", "alias"]
    )

    # Guardar
    dates_path = reference_dir / "available_dates.csv"
    matches_path = reference_dir / "available_matches.csv"
    aliases_path = reference_dir / "team_aliases.csv"

    dates.to_csv(dates_path, index=False, encoding="utf-8")
    matches.to_csv(matches_path, index=False, encoding="utf-8")
    alias_df.to_csv(aliases_path, index=False, encoding="utf-8")

    print()
    print("=" * 90)
    print("REFERENCIAS PARA COLAB/WEB")
    print("=" * 90)
    print(f"dates:   {dates_path} ({len(dates)} fechas)")
    print(f"matches: {matches_path} ({len(matches)} partidos)")
    print(f"aliases: {aliases_path} ({len(alias_df)} aliases)")
    print()
    print("Próximas fechas:")
    print(dates.head(10).to_string(index=False))
    print()
    print("Próximos partidos:")
    print(matches.head(10)[['date', 'match']].to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
