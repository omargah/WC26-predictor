
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.simulation.tournament_simulator import simulate_tournament_once


MAIN_ROUNDS = [
    "Round of 32",
    "Round of 16",
    "Quarterfinal",
    "Semifinal",
    "Final",
]

ROUND_ORDER = {
    "Group Stage": 0,
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarterfinal": 3,
    "Semifinal": 4,
    "Final": 5,
    "Champion": 6,
}


def _find_group_row(group_stage: dict, team: str):
    for group, table in group_stage["tables"].items():
        t = table.copy()
        mask = t["Equipo"].astype(str).str.lower() == str(team).lower()
        if mask.any():
            row = t[mask].iloc[0].to_dict()
            row["Grupo"] = group
            return row
    return None


def _team_qualified(group_stage: dict, team: str):
    qualifiers = group_stage.get("qualifiers", {})
    return str(team) in set(map(str, qualifiers.values()))


def _extract_team_path_from_bracket(bracket: pd.DataFrame, team: str):
    team = str(team)
    rows = []
    path_parts = []

    reached_round = "Group Stage"
    eliminated_round = "Group Stage"
    champion = False
    runner_up = False

    for round_name in MAIN_ROUNDS:
        br = bracket[bracket["round"] == round_name].copy()

        played = br[
            (br["home_team"].astype(str) == team) |
            (br["away_team"].astype(str) == team)
        ]

        if played.empty:
            continue

        match = played.iloc[0].to_dict()

        home = str(match["home_team"])
        away = str(match["away_team"])
        winner = str(match["winner"])

        opponent = away if home == team else home

        reached_round = round_name
        path_parts.append(f"{round_name}: {opponent}")

        rows.append(
            {
                "round": round_name,
                "match_id": match.get("match_id"),
                "opponent": opponent,
                "home_team": home,
                "away_team": away,
                "winner": winner,
                "method": match.get("method"),
                "home_goals_90": match.get("home_goals_90"),
                "away_goals_90": match.get("away_goals_90"),
            }
        )

        if winner != team:
            eliminated_round = round_name
            if round_name == "Final":
                runner_up = True
            break

        if round_name == "Final" and winner == team:
            champion = True
            eliminated_round = "Champion"

    if champion:
        max_stage = "Champion"
    elif runner_up:
        max_stage = "Final"
    else:
        max_stage = reached_round if reached_round != "Group Stage" else "Group Stage"

    return {
        "matches": pd.DataFrame(rows),
        "path_string": " → ".join(path_parts) if path_parts else "No clasificó a eliminación directa",
        "max_stage": max_stage,
        "eliminated_round": eliminated_round,
        "champion": champion,
        "runner_up": runner_up,
    }


def extract_team_simulation_result(simulation_id: int, sim_out: dict, team: str):
    group_stage = sim_out["group_stage"]
    bracket = sim_out["bracket"]

    group_row = _find_group_row(group_stage, team)

    if group_row is None:
        raise ValueError(f"No encontré al equipo {team} en las tablas de grupo.")

    qualified = _team_qualified(group_stage, team)

    base = {
        "simulation": simulation_id,
        "team": team,
        "group": group_row.get("Grupo"),
        "group_position": int(group_row.get("Pos")),
        "group_points": int(group_row.get("Pts")),
        "group_gf": int(group_row.get("GF")),
        "group_ga": int(group_row.get("GC")),
        "group_gd": int(group_row.get("DG")),
        "qualified_group": bool(qualified),
        "champion": sim_out.get("champion"),
        "runner_up_tournament": sim_out.get("runner_up"),
        "third_place_tournament": sim_out.get("third_place"),
    }

    if not qualified:
        base.update(
            {
                "max_stage": "Group Stage",
                "eliminated_round": "Group Stage",
                "is_champion": False,
                "is_runner_up": False,
                "path_string": "Eliminado en fase de grupos",
            }
        )
        return base, pd.DataFrame([])

    path = _extract_team_path_from_bracket(bracket, team)

    base.update(
        {
            "max_stage": path["max_stage"],
            "eliminated_round": path["eliminated_round"],
            "is_champion": bool(path["champion"]),
            "is_runner_up": bool(path["runner_up"]),
            "path_string": path["path_string"],
        }
    )

    path_matches = path["matches"].copy()
    if not path_matches.empty:
        path_matches["simulation"] = simulation_id
        path_matches["team"] = team

    return base, path_matches


def summarize_team_roadmap(results_df: pd.DataFrame, path_matches_df: pd.DataFrame):
    n = len(results_df)

    if n == 0:
        raise ValueError("No hay simulaciones para resumir.")

    stage_probs = (
        results_df["max_stage"]
        .value_counts(normalize=True)
        .reset_index()
    )
    stage_probs.columns = ["stage", "probability"]
    stage_probs["count"] = stage_probs["stage"].map(results_df["max_stage"].value_counts())
    stage_probs["stage_order"] = stage_probs["stage"].map(ROUND_ORDER).fillna(-1)
    stage_probs = stage_probs.sort_values("stage_order").reset_index(drop=True)

    group_position_probs = (
        results_df["group_position"]
        .value_counts(normalize=True)
        .reset_index()
    )
    group_position_probs.columns = ["group_position", "probability"]
    group_position_probs["count"] = group_position_probs["group_position"].map(
        results_df["group_position"].value_counts()
    )
    group_position_probs = group_position_probs.sort_values("group_position").reset_index(drop=True)

    key_probs = {
        "n_simulations": int(n),
        "p_champion": float(results_df["is_champion"].mean()),
        "p_runner_up": float(results_df["is_runner_up"].mean()),
        "p_qualify_group": float(results_df["qualified_group"].mean()),
        "avg_group_points": float(results_df["group_points"].mean()),
        "avg_group_gd": float(results_df["group_gd"].mean()),
    }

    for round_name in MAIN_ROUNDS:
        key = "p_reach_" + round_name.lower().replace(" ", "_")
        min_order = ROUND_ORDER[round_name]
        key_probs[key] = float(
            results_df["max_stage"].map(ROUND_ORDER).fillna(0).ge(min_order).mean()
        )

    if path_matches_df is not None and not path_matches_df.empty:
        opponent_probs = (
            path_matches_df
            .groupby(["round", "opponent"], as_index=False)
            .agg(count=("simulation", "count"))
        )
        round_counts = (
            path_matches_df
            .groupby("round")["simulation"]
            .nunique()
            .to_dict()
        )
        opponent_probs["conditional_probability"] = opponent_probs.apply(
            lambda r: r["count"] / max(round_counts.get(r["round"], 1), 1),
            axis=1,
        )
        opponent_probs["round_order"] = opponent_probs["round"].map(ROUND_ORDER)
        opponent_probs = opponent_probs.sort_values(
            ["round_order", "conditional_probability", "count"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    else:
        opponent_probs = pd.DataFrame(
            columns=["round", "opponent", "count", "conditional_probability"]
        )

    paths = (
        results_df["path_string"]
        .value_counts(normalize=False)
        .reset_index()
    )
    paths.columns = ["path", "count"]
    paths["probability"] = paths["count"] / n
    paths = paths.sort_values(["count", "path"], ascending=[False, True]).reset_index(drop=True)

    return {
        "key_probs": key_probs,
        "stage_probs": stage_probs,
        "group_position_probs": group_position_probs,
        "opponent_probs": opponent_probs,
        "common_paths": paths,
    }


def run_team_roadmap_monte_carlo(
    project_root: str | Path,
    team: str,
    n_simulations: int = 50,
    analysis_date: str | None = None,
    use_registered_results: bool = True,
    seed: int = 2026,
    scenario: str = "base",
    corners_cards_mode: str = "legacy",
    progress_callback=None,
):
    project_root = Path(project_root)
    n_simulations = int(n_simulations)

    rows = []
    path_frames = []

    for i in range(n_simulations):
        sim_id = i + 1
        sim_seed = int(seed) + i

        if progress_callback is not None:
            progress_callback(
                current=sim_id,
                total=n_simulations,
                message=f"Simulando Copa {sim_id} de {n_simulations} para {team}",
            )

        sim_out = simulate_tournament_once(
            project_root=project_root,
            analysis_date=analysis_date,
            use_registered_results=use_registered_results,
            seed=sim_seed,
            scenario=scenario,
            corners_cards_mode=corners_cards_mode,
        )

        row, path_matches = extract_team_simulation_result(
            simulation_id=sim_id,
            sim_out=sim_out,
            team=team,
        )

        rows.append(row)

        if path_matches is not None and not path_matches.empty:
            path_frames.append(path_matches)

    results_df = pd.DataFrame(rows)

    if path_frames:
        path_matches_df = pd.concat(path_frames, ignore_index=True)
    else:
        path_matches_df = pd.DataFrame([])

    summary = summarize_team_roadmap(results_df, path_matches_df)

    if progress_callback is not None:
        progress_callback(
            current=n_simulations,
            total=n_simulations,
            message=f"Ruta de {team} terminada",
        )

    return {
        "team": team,
        "scenario": scenario,
        "results": results_df,
        "path_matches": path_matches_df,
        "summary": summary,
    }


def save_team_roadmap_outputs(result: dict, project_root: str | Path):
    project_root = Path(project_root)
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    safe_team = str(result["team"]).replace(" ", "_").replace("/", "_")
    scenario = str(result.get("scenario", "base"))

    results_path = reports_dir / f"team_roadmap_{safe_team}_{scenario}_simulations.csv"
    paths_path = reports_dir / f"team_roadmap_{safe_team}_{scenario}_paths.csv"
    opponents_path = reports_dir / f"team_roadmap_{safe_team}_{scenario}_opponents.csv"

    result["results"].to_csv(results_path, index=False, encoding="utf-8")

    result["summary"]["common_paths"].to_csv(paths_path, index=False, encoding="utf-8")
    result["summary"]["opponent_probs"].to_csv(opponents_path, index=False, encoding="utf-8")

    return {
        "results_path": results_path,
        "paths_path": paths_path,
        "opponents_path": opponents_path,
    }
