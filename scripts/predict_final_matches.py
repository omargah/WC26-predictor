# -*- coding: utf-8 -*-

"""
Predictor final de partidos — Mundial 2026.

Genera:
1. CSV ancho tradicional.
2. Carpeta analítica:
   - match_summary.csv
   - markets_long.csv
   - scorelines_long.csv
   - report.md
   - raw_full.csv
   - metadata.json
"""

from pathlib import Path
import sys
import argparse
import math
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import get_paths
from src.reporting.match_outputs import format_prediction_outputs

try:
    from src.models.poisson_dc import score_matrix
except Exception:
    score_matrix = None


ALIASES = {
    "espana": "Spain",
    "españa": "Spain",
    "spain": "Spain",

    "arabia saudita": "Saudi Arabia",
    "saudi arabia": "Saudi Arabia",

    "belgica": "Belgium",
    "bélgica": "Belgium",
    "belgium": "Belgium",

    "iran": "Iran",
    "irán": "Iran",

    "uruguay": "Uruguay",

    "cabo verde": "Cape Verde",
    "cape verde": "Cape Verde",

    "nueva zelanda": "New Zealand",
    "new zealand": "New Zealand",

    "egipto": "Egypt",
    "egypt": "Egypt",

    "mexico": "Mexico",
    "méxico": "Mexico",

    "estados unidos": "United States",
    "usa": "United States",
    "united states": "United States",

    "canada": "Canada",
    "canadá": "Canada",

    "argentina": "Argentina",
    "austria": "Austria",
    "france": "France",
    "francia": "France",
    "iraq": "Iraq",
    "irak": "Iraq",
    "norway": "Norway",
    "noruega": "Norway",
    "senegal": "Senegal",
    "jordan": "Jordan",
    "jordania": "Jordan",
    "algeria": "Algeria",
    "argelia": "Algeria",
}


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def norm_text(text: str) -> str:
    return " ".join(strip_accents(str(text)).lower().strip().split())


def canonical_team(name: str) -> str:
    key = norm_text(name)
    return ALIASES.get(key, str(name).strip())


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def fallback_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)

    total = mat.sum()
    if total > 0:
        mat = mat / total

    return mat


def build_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10, rho: float = -0.075) -> np.ndarray:
    if score_matrix is None:
        return fallback_score_matrix(lambda_home, lambda_away, max_goals=max_goals)

    try:
        mat = score_matrix(
            float(lambda_home),
            float(lambda_away),
            max_goals=max_goals,
            rho=rho,
            use_dixon_coles=True,
        )
    except TypeError:
        try:
            mat = score_matrix(float(lambda_home), float(lambda_away), max_goals=max_goals)
        except TypeError:
            mat = score_matrix(float(lambda_home), float(lambda_away))

    mat = np.asarray(mat, dtype=float)

    total = mat.sum()
    if total > 0:
        mat = mat / total

    return mat


def fair_odds(p: float) -> float:
    p = float(p)
    if p <= 0:
        return np.nan
    return 1.0 / p


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.2f}%"


def parse_match(text: str) -> tuple[str, str]:
    raw = str(text).strip()

    separators = [" vs ", " VS ", " v ", " V ", "|", ","]

    for sep in separators:
        if sep in raw:
            a, b = raw.split(sep, 1)
            return canonical_team(a), canonical_team(b)

    raise ValueError(
        f"No pude interpretar el partido '{text}'. Usa formato: 'Spain vs Saudi Arabia'."
    )


def resolve_date(value: str | None):
    if value is None:
        return None

    key = norm_text(value)

    if key == "today":
        return datetime.now(ZoneInfo("America/Mexico_City")).date()

    if key == "tomorrow":
        return datetime.now(ZoneInfo("America/Mexico_City")).date() + timedelta(days=1)

    return pd.to_datetime(value).date()


def load_prediction_sources(paths: dict, include_validation: bool = False) -> pd.DataFrame:
    files = []

    pending_path = paths["predictions"] / "phase03_pending_predictions.csv"

    if pending_path.exists():
        pending = pd.read_csv(pending_path).copy()
        pending = pending.assign(_prediction_source="pending")
        files.append(pending)

    if include_validation:
        validation_path = paths["predictions"] / "phase03_validation_predictions.csv"

        if validation_path.exists():
            validation = pd.read_csv(validation_path).copy()
            validation = validation.assign(_prediction_source="validation")
            files.append(validation)

    if not files:
        raise FileNotFoundError(
            "No encontré archivos de predicción. Ejecuta primero Fase 3."
        )

    return pd.concat(files, ignore_index=True, sort=False).copy()


def calculate_from_lambdas(lambda_home: float, lambda_away: float, max_goals: int = 10) -> dict:
    mat = build_score_matrix(lambda_home, lambda_away, max_goals=max_goals)

    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())

    p_1x = p_home + p_draw
    p_12 = p_home + p_away
    p_x2 = p_draw + p_away

    totals = {}

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        p_over = 0.0

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i + j > line:
                    p_over += float(mat[i, j])

        key = str(line).replace(".", "_")
        totals[f"over_{key}"] = p_over
        totals[f"under_{key}"] = 1.0 - p_over

    p_btts_yes = 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i > 0 and j > 0:
                p_btts_yes += float(mat[i, j])

    p_btts_no = 1.0 - p_btts_yes

    best_idx = np.unravel_index(np.argmax(mat), mat.shape)
    marcador_probable = f"{best_idx[0]}-{best_idx[1]}"
    prob_marcador = float(mat[best_idx])

    top_scores = []
    order = np.argsort(mat.ravel())[::-1]

    for flat_idx in order[:10]:
        i, j = np.unravel_index(flat_idx, mat.shape)
        top_scores.append(f"{i}-{j} ({100 * mat[i, j]:.2f}%)")

    market_probs = {
        "local": p_home,
        "empate": p_draw,
        "visitante": p_away,
        "1X": p_1x,
        "12": p_12,
        "X2": p_x2,
        "over_1_5": totals["over_1_5"],
        "under_1_5": totals["under_1_5"],
        "over_2_5": totals["over_2_5"],
        "under_2_5": totals["under_2_5"],
        "over_3_5": totals["over_3_5"],
        "under_3_5": totals["under_3_5"],
        "btts_yes": p_btts_yes,
        "btts_no": p_btts_no,
    }

    best_market = max(market_probs.items(), key=lambda kv: kv[1])

    out = {
        "p_local": p_home,
        "p_empate": p_draw,
        "p_visitante": p_away,
        "p_1x": p_1x,
        "p_12": p_12,
        "p_x2": p_x2,
        "p_btts_yes": p_btts_yes,
        "p_btts_no": p_btts_no,
        "marcador_probable": marcador_probable,
        "prob_marcador": prob_marcador,
        "top_10_marcadores": " | ".join(top_scores),
        "mercado_mas_probable": best_market[0],
        "prob_mercado_mas_probable": best_market[1],
    }

    out.update(totals)

    for name, p in market_probs.items():
        out[f"momio_justo_{name.lower()}"] = fair_odds(p)

    return out


def filter_requested_matches(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    date_col: str | None,
    requested_matches: list[tuple[str, str]],
    target_date,
) -> pd.DataFrame:
    work = df.copy()

    work["_home_canon"] = work[home_col].map(canonical_team)
    work["_away_canon"] = work[away_col].map(canonical_team)
    work["_home_norm"] = work["_home_canon"].map(norm_text)
    work["_away_norm"] = work["_away_canon"].map(norm_text)

    if date_col is not None:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

    selected_parts = []

    if target_date is not None:
        if date_col is None:
            raise RuntimeError("Se pidió filtrar por fecha, pero no encontré columna de fecha.")

        selected_parts.append(work[work[date_col].dt.date == target_date].copy())

    for home, away in requested_matches:
        home_c = canonical_team(home)
        away_c = canonical_team(away)

        home_n = norm_text(home_c)
        away_n = norm_text(away_c)

        sub = work[
            (work["_home_norm"] == home_n)
            & (work["_away_norm"] == away_n)
        ].copy()

        if sub.empty:
            sub = work[
                (work["_home_norm"] == away_n)
                & (work["_away_norm"] == home_n)
            ].copy()

        if sub.empty:
            selected_parts.append(pd.DataFrame([{
                "_missing_requested_match": f"{home_c} vs {away_c}",
            }]))
        else:
            selected_parts.append(sub.head(1))

    if not selected_parts:
        raise RuntimeError("No se especificó fecha ni partidos. Usa --date o --match.")

    out = pd.concat(selected_parts, ignore_index=True, sort=False).copy()

    if home_col in out.columns and away_col in out.columns:
        subset = [home_col, away_col]
        if date_col and date_col in out.columns:
            subset = [date_col, home_col, away_col]
        out = out.drop_duplicates(subset=subset, keep="first")

    return out.copy()


def build_output_dataframe(
    selected: pd.DataFrame,
    home_col: str,
    away_col: str,
    date_col: str | None,
    lambda_home_col: str,
    lambda_away_col: str,
    max_goals: int,
) -> pd.DataFrame:
    rows = []

    for _, r in selected.iterrows():
        if "_missing_requested_match" in r and pd.notna(r.get("_missing_requested_match")):
            rows.append({
                "estado": "NO_ENCONTRADO",
                "partido": r["_missing_requested_match"],
            })
            continue

        local = str(r[home_col])
        visitante = str(r[away_col])

        lh = float(r[lambda_home_col])
        la = float(r[lambda_away_col])

        calc = calculate_from_lambdas(lh, la, max_goals=max_goals)

        fecha = ""
        if date_col is not None and date_col in r.index and pd.notna(r[date_col]):
            fecha = str(pd.to_datetime(r[date_col]).date())

        row = {
            "estado": "OK",
            "source": str(r.get("_prediction_source", "")),
            "fecha": fecha,
            "local": local,
            "visitante": visitante,
            "partido": f"{local} vs {visitante}",
            "lambda_local": lh,
            "lambda_visitante": la,
        }

        row.update(calc)
        rows.append(row)

    out = pd.DataFrame(rows)

    preferred_cols = [
        "estado", "source", "fecha", "partido", "local", "visitante",
        "lambda_local", "lambda_visitante",
        "p_local", "p_empate", "p_visitante",
        "p_1x", "p_12", "p_x2",
        "over_0_5", "under_0_5",
        "over_1_5", "under_1_5",
        "over_2_5", "under_2_5",
        "over_3_5", "under_3_5",
        "over_4_5", "under_4_5",
        "p_btts_yes", "p_btts_no",
        "marcador_probable", "prob_marcador",
        "mercado_mas_probable", "prob_mercado_mas_probable",
        "momio_justo_local", "momio_justo_empate", "momio_justo_visitante",
        "momio_justo_1x", "momio_justo_12", "momio_justo_x2",
        "momio_justo_over_1_5", "momio_justo_under_1_5",
        "momio_justo_over_2_5", "momio_justo_under_2_5",
        "momio_justo_over_3_5", "momio_justo_under_3_5",
        "momio_justo_btts_yes", "momio_justo_btts_no",
        "top_10_marcadores",
    ]

    existing = [c for c in preferred_cols if c in out.columns]
    other = [c for c in out.columns if c not in existing]

    return out[existing + other].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictor final de partidos del modelo Mundial 2026."
    )

    parser.add_argument("--date", default=None, help="Fecha YYYY-MM-DD, today o tomorrow.")
    parser.add_argument("--match", action="append", default=[], help="Partido en formato 'Spain vs Saudi Arabia'. Puede repetirse.")
    parser.add_argument("--home", default=None, help="Equipo local.")
    parser.add_argument("--away", default=None, help="Equipo visitante.")
    parser.add_argument("--include-validation", action="store_true", help="Incluye predicciones de validación.")
    parser.add_argument("--max-goals", type=int, default=10, help="Máximo de goles para matriz de marcadores.")
    parser.add_argument("--save-name", default=None, help="Nombre del CSV ancho dentro de data/predictions.")
    parser.add_argument("--formatted-dir", default=None, help="Carpeta para salidas limpias. Default: data/predictions/formatted/<save_name>.")
    parser.add_argument("--no-formatted", action="store_true", help="No generar carpeta de salidas limpias.")

    args = parser.parse_args()

    paths = get_paths()

    df = load_prediction_sources(paths, include_validation=args.include_validation)

    date_col = pick_col(df, ["fecha", "date", "match_date"])
    home_col = pick_col(df, ["equipo_local", "home", "home_team", "team_home", "local"])
    away_col = pick_col(df, ["equipo_visitante", "away", "away_team", "team_away", "visitante"])

    lambda_home_col = pick_col(df, ["lambda_local", "lambda_home", "lambda_goles_local", "pred_lambda_home"])
    lambda_away_col = pick_col(df, ["lambda_visitante", "lambda_away", "lambda_goles_visitante", "pred_lambda_away"])

    missing = [
        name for name, value in {
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

    requested_matches = [parse_match(m) for m in args.match]

    if args.home and args.away:
        requested_matches.append((canonical_team(args.home), canonical_team(args.away)))

    target_date = resolve_date(args.date)

    selected = filter_requested_matches(
        df=df,
        home_col=home_col,
        away_col=away_col,
        date_col=date_col,
        requested_matches=requested_matches,
        target_date=target_date,
    )

    out = build_output_dataframe(
        selected=selected,
        home_col=home_col,
        away_col=away_col,
        date_col=date_col,
        lambda_home_col=lambda_home_col,
        lambda_away_col=lambda_away_col,
        max_goals=args.max_goals,
    )

    if args.save_name:
        save_name = args.save_name
    elif target_date is not None:
        save_name = f"final_match_predictions_{target_date}.csv"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"final_match_predictions_{stamp}.csv"

    if not save_name.endswith(".csv"):
        save_name += ".csv"

    out_path = paths["predictions"] / save_name
    out.to_csv(out_path, index=False, encoding="utf-8")

    formatted_metadata = None

    if not args.no_formatted:
        if args.formatted_dir:
            formatted_dir = Path(args.formatted_dir)
        else:
            formatted_dir = paths["predictions"] / "formatted" / Path(save_name).stem

        formatted_metadata = format_prediction_outputs(
            df=out,
            output_dir=formatted_dir,
            max_goals=args.max_goals,
        )

    print()
    print("=" * 110)
    print("PREDICTOR FINAL DE PARTIDOS — MUNDIAL 2026")
    print("=" * 110)

    display_cols = [
        "fecha", "partido",
        "lambda_local", "lambda_visitante",
        "p_local", "p_empate", "p_visitante",
        "p_1x", "p_x2",
        "over_1_5", "over_2_5", "over_3_5",
        "p_btts_yes",
        "marcador_probable", "prob_marcador",
        "mercado_mas_probable", "prob_mercado_mas_probable",
    ]

    display_cols = [c for c in display_cols if c in out.columns]

    print(out[display_cols].to_string(index=False, formatters={
        "lambda_local": lambda x: "" if pd.isna(x) else f"{float(x):.3f}",
        "lambda_visitante": lambda x: "" if pd.isna(x) else f"{float(x):.3f}",
        "p_local": pct,
        "p_empate": pct,
        "p_visitante": pct,
        "p_1x": pct,
        "p_x2": pct,
        "over_1_5": pct,
        "over_2_5": pct,
        "over_3_5": pct,
        "p_btts_yes": pct,
        "prob_marcador": pct,
        "prob_mercado_mas_probable": pct,
    }))

    print()
    print("-" * 110)
    print(f"CSV ancho guardado: {out_path}")

    if formatted_metadata:
        print(f"Carpeta analítica:  {formatted_metadata['output_dir']}")
        print("Archivos limpios:")
        print(f" - {formatted_metadata['files']['match_summary']}")
        print(f" - {formatted_metadata['files']['markets_long']}")
        print(f" - {formatted_metadata['files']['scorelines_long']}")
        print(f" - {formatted_metadata['files']['report']}")

    print("=" * 110)


if __name__ == "__main__":
    main()
