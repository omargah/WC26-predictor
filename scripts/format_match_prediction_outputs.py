# -*- coding: utf-8 -*-

"""
Formateador de predicciones de partido.

Convierte el CSV ancho del predictor final en archivos más analizables:

- match_summary.csv
- markets_long.csv
- scorelines_long.csv
- raw_full.csv
- metadata.json

Ejemplo:

python scripts/format_match_prediction_outputs.py \
  --input data/predictions/test_matches_by_date_2026_06_22.csv
"""

from pathlib import Path
import sys
import argparse
import json
import math
import re
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

try:
    from src.models.poisson_dc import score_matrix
except Exception:
    score_matrix = None


def slug(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def fair_odds(p):
    p = float(p)
    if p <= 0:
        return np.nan
    return 1.0 / p


def pct_fmt(x):
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.2f}%"


def odds_fmt(x):
    if pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def poisson_pmf(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def fallback_matrix(lambda_home, lambda_away, max_goals=10):
    mat = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)

    total = mat.sum()
    if total > 0:
        mat = mat / total

    return mat


def build_score_matrix(lambda_home, lambda_away, max_goals=10):
    if score_matrix is None:
        return fallback_matrix(lambda_home, lambda_away, max_goals=max_goals)

    try:
        mat = score_matrix(
            float(lambda_home),
            float(lambda_away),
            max_goals=max_goals,
            rho=-0.075,
            use_dixon_coles=True,
        )
    except TypeError:
        mat = score_matrix(float(lambda_home), float(lambda_away), max_goals=max_goals)

    mat = np.asarray(mat, dtype=float)

    total = mat.sum()
    if total > 0:
        mat = mat / total

    return mat


def make_match_id(row):
    date = str(row.get("fecha", ""))
    home = str(row.get("local", ""))
    away = str(row.get("visitante", ""))
    return f"{date}_{slug(home)}_vs_{slug(away)}"


def build_match_summary(df):
    rows = []

    for _, r in df.iterrows():
        match_id = make_match_id(r)

        p_local = float(r["p_local"]) if "p_local" in r else np.nan
        p_empate = float(r["p_empate"]) if "p_empate" in r else np.nan
        p_visitante = float(r["p_visitante"]) if "p_visitante" in r else np.nan

        probs = {
            "home_win": p_local,
            "draw": p_empate,
            "away_win": p_visitante,
        }

        most_likely_result = max(probs.items(), key=lambda kv: kv[1])[0]

        rows.append({
            "match_id": match_id,
            "date": r.get("fecha", ""),
            "home_team": r.get("local", ""),
            "away_team": r.get("visitante", ""),
            "match": r.get("partido", ""),
            "prediction_source": r.get("source", ""),
            "lambda_home": r.get("lambda_local", np.nan),
            "lambda_away": r.get("lambda_visitante", np.nan),

            "p_home_win": p_local,
            "p_draw": p_empate,
            "p_away_win": p_visitante,

            "fair_odds_home_win": fair_odds(p_local) if not pd.isna(p_local) else np.nan,
            "fair_odds_draw": fair_odds(p_empate) if not pd.isna(p_empate) else np.nan,
            "fair_odds_away_win": fair_odds(p_visitante) if not pd.isna(p_visitante) else np.nan,

            "most_likely_result": most_likely_result,
            "most_likely_score": r.get("marcador_probable", ""),
            "p_most_likely_score": r.get("prob_marcador", np.nan),

            "p_over_1_5": r.get("over_1_5", np.nan),
            "p_over_2_5": r.get("over_2_5", np.nan),
            "p_over_3_5": r.get("over_3_5", np.nan),
            "p_btts_yes": r.get("p_btts_yes", np.nan),

            "best_market": r.get("mercado_mas_probable", ""),
            "p_best_market": r.get("prob_mercado_mas_probable", np.nan),
        })

    return pd.DataFrame(rows)


def add_market(rows, base, market_type, market, selection, probability, line=None):
    if pd.isna(probability):
        return

    probability = float(probability)

    rows.append({
        **base,
        "market_type": market_type,
        "market": market,
        "line": line,
        "selection": selection,
        "probability": probability,
        "probability_pct": 100 * probability,
        "fair_odds": fair_odds(probability),
    })


def build_markets_long(df):
    rows = []

    for _, r in df.iterrows():
        base = {
            "match_id": make_match_id(r),
            "date": r.get("fecha", ""),
            "home_team": r.get("local", ""),
            "away_team": r.get("visitante", ""),
            "match": r.get("partido", ""),
        }

        # 1X2
        add_market(rows, base, "1X2", "Match Result", "Home", r.get("p_local", np.nan))
        add_market(rows, base, "1X2", "Match Result", "Draw", r.get("p_empate", np.nan))
        add_market(rows, base, "1X2", "Match Result", "Away", r.get("p_visitante", np.nan))

        # Doble oportunidad
        add_market(rows, base, "Double Chance", "Double Chance", "1X", r.get("p_1x", np.nan))
        add_market(rows, base, "Double Chance", "Double Chance", "12", r.get("p_12", np.nan))
        add_market(rows, base, "Double Chance", "Double Chance", "X2", r.get("p_x2", np.nan))

        # Totales
        for line in ["0_5", "1_5", "2_5", "3_5", "4_5"]:
            pretty_line = line.replace("_", ".")
            add_market(
                rows,
                base,
                "Total Goals",
                f"Over/Under {pretty_line}",
                "Over",
                r.get(f"over_{line}", np.nan),
                line=pretty_line,
            )
            add_market(
                rows,
                base,
                "Total Goals",
                f"Over/Under {pretty_line}",
                "Under",
                r.get(f"under_{line}", np.nan),
                line=pretty_line,
            )

        # BTTS
        add_market(rows, base, "BTTS", "Both Teams To Score", "Yes", r.get("p_btts_yes", np.nan))
        add_market(rows, base, "BTTS", "Both Teams To Score", "No", r.get("p_btts_no", np.nan))

    out = pd.DataFrame(rows)

    if not out.empty:
        out["rank_within_match"] = (
            out.groupby("match_id")["probability"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        out = out.sort_values(
            ["match_id", "rank_within_match", "market_type", "market", "selection"]
        ).reset_index(drop=True)

    return out


def build_scorelines_long(df, max_goals=10):
    rows = []

    for _, r in df.iterrows():
        match_id = make_match_id(r)

        lambda_home = float(r["lambda_local"])
        lambda_away = float(r["lambda_visitante"])

        mat = build_score_matrix(lambda_home, lambda_away, max_goals=max_goals)

        base = {
            "match_id": match_id,
            "date": r.get("fecha", ""),
            "home_team": r.get("local", ""),
            "away_team": r.get("visitante", ""),
            "match": r.get("partido", ""),
        }

        for home_goals in range(mat.shape[0]):
            for away_goals in range(mat.shape[1]):
                p = float(mat[home_goals, away_goals])

                if home_goals > away_goals:
                    result = "Home"
                elif home_goals < away_goals:
                    result = "Away"
                else:
                    result = "Draw"

                rows.append({
                    **base,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "score": f"{home_goals}-{away_goals}",
                    "result": result,
                    "total_goals": home_goals + away_goals,
                    "probability": p,
                    "probability_pct": 100 * p,
                    "fair_odds": fair_odds(p),
                })

    out = pd.DataFrame(rows)

    if not out.empty:
        out["score_rank_within_match"] = (
            out.groupby("match_id")["probability"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

        out = out.sort_values(
            ["match_id", "score_rank_within_match"]
        ).reset_index(drop=True)

    return out


def build_readable_report(summary, markets, scorelines):
    lines = []

    lines.append("# Match prediction report")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    for _, r in summary.iterrows():
        match_id = r["match_id"]

        lines.append(f"## {r['match']}")
        lines.append("")
        lines.append(f"- Fecha: `{r['date']}`")
        lines.append(f"- λ local: `{float(r['lambda_home']):.3f}`")
        lines.append(f"- λ visitante: `{float(r['lambda_away']):.3f}`")
        lines.append(f"- 1X2: local `{100*r['p_home_win']:.2f}%`, empate `{100*r['p_draw']:.2f}%`, visitante `{100*r['p_away_win']:.2f}%`")
        lines.append(f"- Marcador probable: `{r['most_likely_score']}` con `{100*r['p_most_likely_score']:.2f}%`")
        lines.append(f"- Over 2.5: `{100*r['p_over_2_5']:.2f}%`")
        lines.append(f"- BTTS Sí: `{100*r['p_btts_yes']:.2f}%`")
        lines.append("")

        top_markets = markets[markets["match_id"] == match_id].head(8)
        lines.append("### Mercados más probables")
        lines.append("")
        lines.append("| Mercado | Selección | Probabilidad | Momio justo |")
        lines.append("|---|---:|---:|---:|")

        for _, m in top_markets.iterrows():
            lines.append(
                f"| {m['market']} | {m['selection']} | "
                f"{100*m['probability']:.2f}% | {m['fair_odds']:.2f} |"
            )

        lines.append("")

        top_scores = scorelines[scorelines["match_id"] == match_id].head(8)
        lines.append("### Marcadores más probables")
        lines.append("")
        lines.append("| Marcador | Probabilidad | Momio justo |")
        lines.append("|---:|---:|---:|")

        for _, s in top_scores.iterrows():
            lines.append(
                f"| {s['score']} | {100*s['probability']:.2f}% | {s['fair_odds']:.2f} |"
            )

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Reformatea salidas del predictor de partidos."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV generado por scripts/predict_final_matches.py",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Carpeta de salida. Si se omite, usa data/predictions/formatted/<input_stem>.",
    )

    parser.add_argument(
        "--max-goals",
        type=int,
        default=10,
        help="Máximo de goles para scorelines_long.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("data/predictions/formatted") / input_path.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_match_summary(df)
    markets = build_markets_long(df)
    scorelines = build_scorelines_long(df, max_goals=args.max_goals)

    raw_path = output_dir / "raw_full.csv"
    summary_path = output_dir / "match_summary.csv"
    markets_path = output_dir / "markets_long.csv"
    scorelines_path = output_dir / "scorelines_long.csv"
    report_path = output_dir / "report.md"
    metadata_path = output_dir / "metadata.json"

    df.to_csv(raw_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    markets.to_csv(markets_path, index=False, encoding="utf-8")
    scorelines.to_csv(scorelines_path, index=False, encoding="utf-8")

    report_text = build_readable_report(summary, markets, scorelines)
    report_path.write_text(report_text, encoding="utf-8")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "n_matches": int(len(summary)),
        "n_market_rows": int(len(markets)),
        "n_scoreline_rows": int(len(scorelines)),
        "max_goals": int(args.max_goals),
        "files": {
            "raw_full": str(raw_path),
            "match_summary": str(summary_path),
            "markets_long": str(markets_path),
            "scorelines_long": str(scorelines_path),
            "report": str(report_path),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("SALIDA FORMATEADA PARA ANÁLISIS")
    print("=" * 90)
    print(f"input:       {input_path}")
    print(f"output_dir:  {output_dir}")
    print()
    print("Archivos:")
    print(f" - {summary_path}")
    print(f" - {markets_path}")
    print(f" - {scorelines_path}")
    print(f" - {report_path}")
    print(f" - {raw_path}")
    print()
    print("-" * 90)
    print("RESUMEN")
    print("-" * 90)
    print(summary.to_string(index=False, formatters={
        "lambda_home": lambda x: f"{float(x):.3f}",
        "lambda_away": lambda x: f"{float(x):.3f}",
        "p_home_win": pct_fmt,
        "p_draw": pct_fmt,
        "p_away_win": pct_fmt,
        "fair_odds_home_win": odds_fmt,
        "fair_odds_draw": odds_fmt,
        "fair_odds_away_win": odds_fmt,
        "p_most_likely_score": pct_fmt,
        "p_over_1_5": pct_fmt,
        "p_over_2_5": pct_fmt,
        "p_over_3_5": pct_fmt,
        "p_btts_yes": pct_fmt,
        "p_best_market": pct_fmt,
    }))
    print("=" * 90)


if __name__ == "__main__":
    main()
