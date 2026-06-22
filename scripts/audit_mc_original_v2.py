# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths


EXPECTED_ROUND_COUNTS = {
    "Round of 32": 16,
    "Round of 16": 8,
    "Quarterfinal": 4,
    "Semifinal": 2,
    "Third Place": 1,
    "Final": 1,
}


KEY_TEAMS = [
    "Mexico",
    "United States",
    "Canada",
    "Spain",
    "Argentina",
    "Brazil",
    "France",
    "Belgium",
    "England",
    "Colombia",
]


def pct(x):
    return f"{100 * float(x):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita resultados de Monte Carlo original V2."
    )

    parser.add_argument(
        "--run-name",
        default="fixed_original_seed42",
        help="Nombre de la corrida dentro de data/predictions/mc_original_v2_runs.",
    )

    args = parser.parse_args()

    paths = get_paths()

    run_dir = paths["predictions"] / "mc_original_v2_runs" / args.run_name

    results_path = run_dir / "match_results.csv"
    progress_path = run_dir / "team_progress.csv"
    summary_path = run_dir / "top4_summary.csv"
    champion_path = run_dir / "champion_probabilities.csv"
    round_path = run_dir / "round_probabilities.csv"
    metadata_path = run_dir / "metadata.json"

    required_files = [
        results_path,
        progress_path,
        summary_path,
        champion_path,
        round_path,
        metadata_path,
    ]

    missing = [str(p) for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Faltan archivos de la corrida:\n" + "\n".join(missing)
        )

    results = pd.read_csv(results_path)
    progress = pd.read_csv(progress_path)
    summary = pd.read_csv(summary_path)
    champions = pd.read_csv(champion_path)
    rounds = pd.read_csv(round_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_summary = summary["simulation_id"].nunique()
    n_results = results["simulation_id"].nunique()
    n_progress = progress["simulation_id"].nunique()

    errors = []
    warnings = []

    if n_summary != n_results:
        errors.append(f"summary tiene {n_summary} sims, results tiene {n_results}.")

    if n_summary != n_progress:
        errors.append(f"summary tiene {n_summary} sims, progress tiene {n_progress}.")

    # Validar 32 partidos por simulación
    matches_per_sim = results.groupby("simulation_id").size()
    bad_match_counts = matches_per_sim[matches_per_sim != 32]

    if len(bad_match_counts) > 0:
        errors.append(
            "Hay simulaciones sin 32 partidos KO: "
            + bad_match_counts.to_string()
        )

    # Validar conteos por ronda
    round_count_errors = []

    for sim_id, sub in results.groupby("simulation_id"):
        counts = sub["round"].value_counts().to_dict()

        for round_name, expected in EXPECTED_ROUND_COUNTS.items():
            got = int(counts.get(round_name, 0))

            if got != expected:
                round_count_errors.append(
                    {
                        "simulation_id": int(sim_id),
                        "round": round_name,
                        "expected": expected,
                        "got": got,
                    }
                )

    if round_count_errors:
        errors.append(
            f"Hay {len(round_count_errors)} errores de conteo por ronda."
        )

    # Validar ganadores/perdedores
    if results["winner"].isna().any():
        errors.append("Hay partidos sin winner.")

    if results["loser"].isna().any():
        errors.append("Hay partidos sin loser.")

    if (results["winner"].astype(str) == results["loser"].astype(str)).any():
        errors.append("Hay partidos con winner igual a loser.")

    valid_decisions = {"90", "ET", "PEN"}

    invalid_decisions = sorted(set(results["decided_by"].dropna().astype(str)) - valid_decisions)

    if invalid_decisions:
        errors.append(f"decided_by inválidos: {invalid_decisions}")

    # Validar campeones
    if abs(champions["champion_probability"].sum() - 1.0) > 1e-9:
        warnings.append(
            f"Las probabilidades de campeón suman {champions['champion_probability'].sum():.6f}."
        )

    # Resumen key teams
    key_rows = []

    for team in KEY_TEAMS:
        r = rounds[rounds["team"] == team]

        if r.empty:
            key_rows.append(
                {
                    "team": team,
                    "status": "NO_APARECE",
                    "p_round_of_16": None,
                    "p_quarterfinal": None,
                    "p_semifinal": None,
                    "p_final": None,
                    "p_champion": None,
                }
            )
            continue

        rr = r.iloc[0]

        key_rows.append(
            {
                "team": team,
                "status": "OK",
                "p_round_of_16": float(rr.get("p_round_of_16", 0.0)),
                "p_quarterfinal": float(rr.get("p_quarterfinal", 0.0)),
                "p_semifinal": float(rr.get("p_semifinal", 0.0)),
                "p_final": float(rr.get("p_final", 0.0)),
                "p_champion": float(rr.get("p_champion", 0.0)),
            }
        )

    key_df = pd.DataFrame(key_rows)

    # Distribución de decisiones
    decision_summary = (
        results["decided_by"]
        .value_counts()
        .rename_axis("decided_by")
        .reset_index(name="count")
    )

    decision_summary["share"] = decision_summary["count"] / len(results)

    # Resumen final
    report = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "n_simulations_summary": int(n_summary),
        "n_simulations_results": int(n_results),
        "n_simulations_progress": int(n_progress),
        "n_matches_total": int(len(results)),
        "expected_matches_total": int(32 * n_summary),
        "errors": errors,
        "warnings": warnings,
        "metadata": metadata,
    }

    reports_dir = paths["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / f"phase05_v2_original_mc_audit_{args.run_name}.json"
    key_path = reports_dir / f"phase05_v2_original_mc_key_teams_{args.run_name}.csv"
    decisions_path = reports_dir / f"phase05_v2_original_mc_decisions_{args.run_name}.csv"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    key_df.to_csv(key_path, index=False, encoding="utf-8")
    decision_summary.to_csv(decisions_path, index=False, encoding="utf-8")

    print()
    print("=" * 90)
    print("AUDITORÍA MONTE CARLO ORIGINAL V2")
    print("=" * 90)

    print(f"run_name:              {args.run_name}")
    print(f"run_dir:               {run_dir}")
    print(f"simulaciones summary:  {n_summary}")
    print(f"simulaciones results:  {n_results}")
    print(f"partidos totales:      {len(results)}")
    print(f"esperados:             {32 * n_summary}")

    print()
    print("-" * 90)
    print("ESTADO")
    print("-" * 90)

    if errors:
        print("ERRORES:")
        for e in errors:
            print(f" - {e}")
    else:
        print("OK: sin errores críticos.")

    if warnings:
        print()
        print("WARNINGS:")
        for w in warnings:
            print(f" - {w}")

    print()
    print("-" * 90)
    print("TOP 20 CAMPEÓN")
    print("-" * 90)
    print(champions.head(20).to_string(index=False, formatters={
        "champion_probability": pct,
    }))

    print()
    print("-" * 90)
    print("EQUIPOS CLAVE")
    print("-" * 90)
    print(key_df.to_string(index=False, formatters={
        "p_round_of_16": lambda x: "" if pd.isna(x) else pct(x),
        "p_quarterfinal": lambda x: "" if pd.isna(x) else pct(x),
        "p_semifinal": lambda x: "" if pd.isna(x) else pct(x),
        "p_final": lambda x: "" if pd.isna(x) else pct(x),
        "p_champion": lambda x: "" if pd.isna(x) else pct(x),
    }))

    print()
    print("-" * 90)
    print("DECISIONES KO")
    print("-" * 90)
    print(decision_summary.to_string(index=False, formatters={
        "share": pct,
    }))

    print()
    print("-" * 90)
    print("ARCHIVOS GENERADOS")
    print("-" * 90)
    print(f"report:      {report_path}")
    print(f"key teams:   {key_path}")
    print(f"decisions:   {decisions_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
