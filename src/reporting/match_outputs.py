# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import math
import re

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


def fair_odds(p: float) -> float:
    p = float(p)
    if p <= 0:
        return np.nan
    return 1.0 / p


def pct_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100 * float(x):.2f}%"


def odds_fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def fallback_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
    mat = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)

    total = mat.sum()
    if total > 0:
        mat = mat / total

    return mat


def build_score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
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
        try:
            mat = score_matrix(float(lambda_home), float(lambda_away), max_goals=max_goals)
        except TypeError:
            mat = score_matrix(float(lambda_home), float(lambda_away))

    mat = np.asarray(mat, dtype=float)

    total = mat.sum()
    if total > 0:
        mat = mat / total

    return mat


def make_match_id(row: pd.Series) -> str:
    date = str(row.get("fecha", row.get("date", "")))
    home = str(row.get("local", row.get("home_team", "")))
    away = str(row.get("visitante", row.get("away_team", "")))
    return f"{date}_{slug(home)}_vs_{slug(away)}"


def build_match_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        match_id = make_match_id(r)

        p_local = float(r.get("p_local", np.nan))
        p_empate = float(r.get("p_empate", np.nan))
        p_visitante = float(r.get("p_visitante", np.nan))

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

            "fair_odds_home_win": fair_odds(p_local),
            "fair_odds_draw": fair_odds(p_empate),
            "fair_odds_away_win": fair_odds(p_visitante),

            "most_likely_result": most_likely_result,
            "most_likely_score": r.get("marcador_probable", ""),
            "p_most_likely_score": r.get("prob_marcador", np.nan),

            "p_1x": r.get("p_1x", np.nan),
            "p_12": r.get("p_12", np.nan),
            "p_x2": r.get("p_x2", np.nan),

            "p_over_1_5": r.get("over_1_5", np.nan),
            "p_over_2_5": r.get("over_2_5", np.nan),
            "p_over_3_5": r.get("over_3_5", np.nan),

            "p_btts_yes": r.get("p_btts_yes", np.nan),
            "p_btts_no": r.get("p_btts_no", np.nan),

            "best_market": r.get("mercado_mas_probable", ""),
            "p_best_market": r.get("prob_mercado_mas_probable", np.nan),
        })

    return pd.DataFrame(rows)


def _add_market(rows, base, market_type, market, selection, probability, line=None):
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


def build_markets_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df.iterrows():
        base = {
            "match_id": make_match_id(r),
            "date": r.get("fecha", ""),
            "home_team": r.get("local", ""),
            "away_team": r.get("visitante", ""),
            "match": r.get("partido", ""),
        }

        _add_market(rows, base, "1X2", "Match Result", "Home", r.get("p_local", np.nan))
        _add_market(rows, base, "1X2", "Match Result", "Draw", r.get("p_empate", np.nan))
        _add_market(rows, base, "1X2", "Match Result", "Away", r.get("p_visitante", np.nan))

        _add_market(rows, base, "Double Chance", "Double Chance", "1X", r.get("p_1x", np.nan))
        _add_market(rows, base, "Double Chance", "Double Chance", "12", r.get("p_12", np.nan))
        _add_market(rows, base, "Double Chance", "Double Chance", "X2", r.get("p_x2", np.nan))

        for line in ["0_5", "1_5", "2_5", "3_5", "4_5"]:
            pretty = line.replace("_", ".")

            _add_market(
                rows,
                base,
                "Total Goals",
                f"Over/Under {pretty}",
                "Over",
                r.get(f"over_{line}", np.nan),
                line=pretty,
            )

            _add_market(
                rows,
                base,
                "Total Goals",
                f"Over/Under {pretty}",
                "Under",
                r.get(f"under_{line}", np.nan),
                line=pretty,
            )

        _add_market(rows, base, "BTTS", "Both Teams To Score", "Yes", r.get("p_btts_yes", np.nan))
        _add_market(rows, base, "BTTS", "Both Teams To Score", "No", r.get("p_btts_no", np.nan))

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


def build_scorelines_long(df: pd.DataFrame, max_goals: int = 10) -> pd.DataFrame:
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


def build_readable_report(summary: pd.DataFrame, markets: pd.DataFrame, scorelines: pd.DataFrame) -> str:
    lines = []

    lines.append("# Reporte de predicción de partidos")
    lines.append("")
    lines.append(f"Generado: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    for _, r in summary.iterrows():
        match_id = r["match_id"]

        lines.append(f"## {r['match']}")
        lines.append("")
        lines.append(f"- Fecha: `{r['date']}`")
        lines.append(f"- λ local: `{float(r['lambda_home']):.3f}`")
        lines.append(f"- λ visitante: `{float(r['lambda_away']):.3f}`")
        lines.append(
            f"- 1X2: local `{100*r['p_home_win']:.2f}%`, "
            f"empate `{100*r['p_draw']:.2f}%`, "
            f"visitante `{100*r['p_away_win']:.2f}%`"
        )
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


def format_prediction_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    max_goals: int = 10,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = df.copy()
    summary = build_match_summary(raw)
    markets = build_markets_long(raw)
    scorelines = build_scorelines_long(raw, max_goals=max_goals)

    raw_path = output_dir / "raw_full.csv"
    summary_path = output_dir / "match_summary.csv"
    markets_path = output_dir / "markets_long.csv"
    scorelines_path = output_dir / "scorelines_long.csv"
    report_path = output_dir / "report.md"
    metadata_path = output_dir / "metadata.json"

    raw.to_csv(raw_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    markets.to_csv(markets_path, index=False, encoding="utf-8")
    scorelines.to_csv(scorelines_path, index=False, encoding="utf-8")

    report_text = build_readable_report(summary, markets, scorelines)
    report_path.write_text(report_text, encoding="utf-8")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "n_matches": int(len(summary)),
        "n_market_rows": int(len(markets)),
        "n_scoreline_rows": int(len(scorelines)),
        "max_goals": int(max_goals),
        "files": {
            "raw_full": str(raw_path),
            "match_summary": str(summary_path),
            "markets_long": str(markets_path),
            "scorelines_long": str(scorelines_path),
            "report": str(report_path),
            "metadata": str(metadata_path),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metadata
