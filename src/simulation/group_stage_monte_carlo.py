
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.interactive.group_match_selector import GROUPS
from src.simulation.group_stage_simulator import (
    simulate_single_group,
    PredictionCache,
    rank_best_thirds,
)


def _simulate_all_groups_once(
    project_root: str | Path,
    analysis_date: str | None,
    use_registered_results: bool,
    seed: int,
    prediction_cache: PredictionCache,
    corners_cards_mode: str = "legacy",
):
    rng = np.random.default_rng(seed)

    all_tables = {}
    all_results = []

    for group in sorted(GROUPS.keys()):
        table, results = simulate_single_group(
            group=group,
            project_root=project_root,
            analysis_date=analysis_date,
            use_registered_results=use_registered_results,
            rng=rng,
            prediction_cache=prediction_cache,
            corners_cards_mode=corners_cards_mode,
        )

        all_tables[group] = table
        all_results.append(results)

    results_df = pd.concat(all_results, ignore_index=True)
    best_thirds = rank_best_thirds(all_tables)

    return {
        "tables": all_tables,
        "results": results_df,
        "best_thirds": best_thirds,
    }


def _extract_team_rows(simulation_id: int, sim_out: dict):
    rows = []

    best_thirds = sim_out["best_thirds"].copy()
    qualified_best_third_groups = set(
        best_thirds.loc[
            best_thirds["qualifies_as_best_third"] == True,
            "Grupo"
        ].astype(str)
    )

    for group, table in sim_out["tables"].items():
        t = table.sort_values("Pos").reset_index(drop=True).copy()

        for _, r in t.iterrows():
            pos = int(r["Pos"])
            team = r["Equipo"]

            qualify_top2 = pos <= 2
            qualify_best_third = (pos == 3) and (str(group) in qualified_best_third_groups)
            qualify_total = qualify_top2 or qualify_best_third

            rows.append(
                {
                    "simulation": simulation_id,
                    "group": group,
                    "team": team,
                    "position": pos,
                    "points": int(r["Pts"]),
                    "gf": int(r["GF"]),
                    "ga": int(r["GC"]),
                    "gd": int(r["DG"]),
                    "qualify_top2": qualify_top2,
                    "qualify_best_third": qualify_best_third,
                    "qualify_total": qualify_total,
                }
            )

    return rows


def summarize_group_monte_carlo(team_simulations: pd.DataFrame):
    df = team_simulations.copy()

    base = (
        df.groupby(["group", "team"], as_index=False)
        .agg(
            simulations=("simulation", "count"),
            avg_position=("position", "mean"),
            avg_points=("points", "mean"),
            avg_gf=("gf", "mean"),
            avg_ga=("ga", "mean"),
            avg_gd=("gd", "mean"),
            p_qualify_top2=("qualify_top2", "mean"),
            p_qualify_best_third=("qualify_best_third", "mean"),
            p_qualify_total=("qualify_total", "mean"),
        )
    )

    pos_probs = (
        pd.crosstab(
            [df["group"], df["team"]],
            df["position"],
            normalize="index",
        )
        .reset_index()
    )

    rename = {}
    for c in pos_probs.columns:
        if isinstance(c, int):
            rename[c] = f"p_pos_{c}"

    pos_probs = pos_probs.rename(columns=rename)

    for c in ["p_pos_1", "p_pos_2", "p_pos_3", "p_pos_4"]:
        if c not in pos_probs.columns:
            pos_probs[c] = 0.0

    summary = base.merge(pos_probs, on=["group", "team"], how="left")

    ordered_cols = [
        "group",
        "team",
        "simulations",
        "p_pos_1",
        "p_pos_2",
        "p_pos_3",
        "p_pos_4",
        "p_qualify_top2",
        "p_qualify_best_third",
        "p_qualify_total",
        "avg_position",
        "avg_points",
        "avg_gd",
        "avg_gf",
        "avg_ga",
    ]

    ordered_cols = [c for c in ordered_cols if c in summary.columns]
    summary = summary[ordered_cols]

    summary = summary.sort_values(
        ["group", "p_qualify_total", "p_pos_1", "avg_points"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    return summary


def run_group_stage_monte_carlo(
    project_root: str | Path,
    n_simulations: int = 100,
    analysis_date: str | None = None,
    use_registered_results: bool = True,
    seed: int = 2026,
    corners_cards_mode: str = "legacy",
    progress_callback=None,
):
    project_root = Path(project_root)
    n_simulations = int(n_simulations)

    prediction_cache = PredictionCache()

    all_team_rows = []
    all_match_rows = []

    for i in range(n_simulations):
        sim_id = i + 1
        sim_seed = int(seed) + i

        if progress_callback is not None:
            progress_callback(
                current=sim_id,
                total=n_simulations,
                message=f"Simulando fase de grupos {sim_id} de {n_simulations}",
            )

        sim_out = _simulate_all_groups_once(
            project_root=project_root,
            analysis_date=analysis_date,
            use_registered_results=use_registered_results,
            seed=sim_seed,
            prediction_cache=prediction_cache,
            corners_cards_mode=corners_cards_mode,
        )

        team_rows = _extract_team_rows(sim_id, sim_out)
        all_team_rows.extend(team_rows)

        match_rows = sim_out["results"].copy()
        match_rows["simulation"] = sim_id
        all_match_rows.append(match_rows)

    team_simulations = pd.DataFrame(all_team_rows)
    match_simulations = pd.concat(all_match_rows, ignore_index=True)

    summary = summarize_group_monte_carlo(team_simulations)

    if progress_callback is not None:
        progress_callback(
            current=n_simulations,
            total=n_simulations,
            message="Monte Carlo de grupos terminado",
        )

    return {
        "summary": summary,
        "team_simulations": team_simulations,
        "match_simulations": match_simulations,
    }


def save_group_monte_carlo_outputs(result: dict, project_root: str | Path):
    project_root = Path(project_root)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "group_stage_monte_carlo_summary.csv"
    team_path = reports_dir / "group_stage_monte_carlo_team_simulations.csv"

    result["summary"].to_csv(summary_path, index=False, encoding="utf-8")
    result["team_simulations"].to_csv(team_path, index=False, encoding="utf-8")

    return {
        "summary_path": summary_path,
        "team_simulations_path": team_path,
    }
