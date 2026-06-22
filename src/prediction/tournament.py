# -*- coding: utf-8 -*-
"""
src/prediction/tournament.py

FASE 5 -- Simulador completo del Mundial 2026.

Versión inicial:
    - Usa resultados reales ya jugados.
    - Simula partidos pendientes de fase de grupos con Poisson-Dixon-Coles.
    - Clasifica top 2 de cada grupo + 8 mejores terceros.
    - Simula eliminatorias con un bracket técnico provisional por siembra.
    - En eliminatorias NO aplica ventaja de localía artificial.

Nota metodológica:
    El bracket de Round of 32 de esta versión es provisional.
    Sirve para validar el motor de simulación. Después se puede sustituir
    por el mapa oficial exacto de cruces.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_paths
from src.models.poisson_dc import score_matrix


GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


TEAM_TO_GROUP = {
    team: group
    for group, teams in GROUPS.items()
    for team in teams
}


ROUND_KEYS = [
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_builtin(obj):
    """
    Convierte objetos numpy/pandas a tipos JSON serializables.
    """

    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)

    return obj


def load_worldcup_with_predictions(paths: dict) -> pd.DataFrame:
    """
    Carga fixture Mundial 2026 y agrega lambdas para partidos pendientes.
    """

    all_path = paths["features"] / "modeling_dataset_all.parquet"
    pred_path = paths["predictions"] / "phase03_pending_predictions.csv"

    if not all_path.exists():
        raise FileNotFoundError(
            f"No existe {all_path}. Ejecuta Fase 2 primero."
        )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"No existe {pred_path}. Ejecuta Fase 3 primero."
        )

    df_all = pd.read_parquet(all_path)
    df_all["fecha"] = pd.to_datetime(df_all["fecha"], errors="coerce")

    df_wc = df_all[
        (df_all["torneo"] == "FIFA World Cup")
        &
        (df_all["fecha"] >= "2026-01-01")
    ].copy()

    df_pred = pd.read_csv(pred_path)
    pred_cols = [
        "match_id",
        "lambda_local",
        "lambda_visitante",
    ]

    df_wc = df_wc.merge(
        df_pred[pred_cols],
        on="match_id",
        how="left",
        suffixes=("", "_pred"),
    )

    df_wc["group"] = df_wc["equipo_local"].map(TEAM_TO_GROUP)

    missing_group = df_wc[df_wc["group"].isna()]

    if len(missing_group) > 0:
        teams = sorted(
            set(missing_group["equipo_local"].dropna())
            |
            set(missing_group["equipo_visitante"].dropna())
        )
        raise ValueError(
            "Hay equipos del fixture sin grupo en GROUPS: "
            + ", ".join(teams)
        )

    return df_wc.sort_values(["fecha", "match_id"]).reset_index(drop=True)


def build_latest_elo(paths: dict) -> dict[str, float]:
    """
    Construye rating ELO actual aproximado por equipo usando las features pre-partido.
    """

    all_path = paths["features"] / "modeling_dataset_all.parquet"

    df = pd.read_parquet(all_path)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    local = df[["fecha", "equipo_local", "elo_local_pre"]].rename(
        columns={
            "equipo_local": "team",
            "elo_local_pre": "elo",
        }
    )

    away = df[["fecha", "equipo_visitante", "elo_visitante_pre"]].rename(
        columns={
            "equipo_visitante": "team",
            "elo_visitante_pre": "elo",
        }
    )

    elos = pd.concat([local, away], ignore_index=True)
    elos = elos.dropna(subset=["team", "elo"])
    elos = elos.sort_values(["team", "fecha"])

    latest = elos.groupby("team", as_index=False).tail(1)

    return dict(zip(latest["team"], latest["elo"]))


def sample_score(
    lambda_home: float,
    lambda_away: float,
    rng: np.random.Generator,
    max_goals: int = 10,
) -> tuple[int, int]:
    """
    Simula un marcador exacto usando la matriz Poisson-Dixon-Coles.
    """

    mat = score_matrix(
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        max_goals=max_goals,
        rho=-0.075,
        use_dixon_coles=True,
    )

    flat_index = rng.choice(mat.size, p=mat.ravel())
    gh, ga = np.unravel_index(flat_index, mat.shape)

    return int(gh), int(ga)


def init_group_table(group: str) -> pd.DataFrame:
    """
    Inicializa tabla de grupo.
    """

    rows = []

    for team in GROUPS[group]:
        rows.append(
            {
                "group": group,
                "team": team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "points": 0,
            }
        )

    return pd.DataFrame(rows).set_index("team")


def update_table(
    table: pd.DataFrame,
    home: str,
    away: str,
    gh: int,
    ga: int,
) -> None:
    """
    Actualiza tabla de grupo con un resultado.
    """

    table.loc[home, "played"] += 1
    table.loc[away, "played"] += 1

    table.loc[home, "gf"] += gh
    table.loc[home, "ga"] += ga
    table.loc[away, "gf"] += ga
    table.loc[away, "ga"] += gh

    table.loc[home, "gd"] = table.loc[home, "gf"] - table.loc[home, "ga"]
    table.loc[away, "gd"] = table.loc[away, "gf"] - table.loc[away, "ga"]

    if gh > ga:
        table.loc[home, "wins"] += 1
        table.loc[home, "points"] += 3
        table.loc[away, "losses"] += 1
    elif gh < ga:
        table.loc[away, "wins"] += 1
        table.loc[away, "points"] += 3
        table.loc[home, "losses"] += 1
    else:
        table.loc[home, "draws"] += 1
        table.loc[away, "draws"] += 1
        table.loc[home, "points"] += 1
        table.loc[away, "points"] += 1


def rank_group_table(
    table: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Ordena tabla de grupo.

    Criterios simplificados:
        puntos,
        diferencia de goles,
        goles a favor,
        victorias,
        desempate aleatorio residual.

    El desempate aleatorio representa criterios no modelados como fair play.
    """

    df = table.reset_index().copy()
    df["tie_random"] = rng.random(len(df))

    df = df.sort_values(
        ["points", "gd", "gf", "wins", "tie_random"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    df["position"] = np.arange(1, len(df) + 1)

    return df


def simulate_group_stage(
    df_wc: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, list[dict]]:
    """
    Simula fase de grupos desde el estado actual.
    """

    tables = {
        group: init_group_table(group)
        for group in GROUPS
    }

    match_rows = []

    for _, row in df_wc.iterrows():
        group = row["group"]
        home = row["equipo_local"]
        away = row["equipo_visitante"]

        if bool(row["is_played"]):
            gh = int(row["goles_local"])
            ga = int(row["goles_visitante"])
            source = "real"
        else:
            if pd.isna(row["lambda_local"]) or pd.isna(row["lambda_visitante"]):
                raise ValueError(
                    f"Faltan lambdas para partido pendiente: {home} vs {away}"
                )

            gh, ga = sample_score(
                lambda_home=float(row["lambda_local"]),
                lambda_away=float(row["lambda_visitante"]),
                rng=rng,
            )
            source = "simulated"

        update_table(tables[group], home, away, gh, ga)

        match_rows.append(
            {
                "group": group,
                "fecha": str(pd.to_datetime(row["fecha"]).date()),
                "home": home,
                "away": away,
                "home_goals": gh,
                "away_goals": ga,
                "source": source,
            }
        )

    ranked_tables = {}
    all_standings = []

    for group, table in tables.items():
        ranked = rank_group_table(table, rng)
        ranked_tables[group] = ranked
        all_standings.append(ranked)

    standings = pd.concat(all_standings, ignore_index=True)

    return ranked_tables, standings, match_rows


def select_qualified(standings: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica top 2 de cada grupo + 8 mejores terceros.
    """

    top_two = standings[standings["position"] <= 2].copy()
    thirds = standings[standings["position"] == 3].copy()

    best_thirds = thirds.sort_values(
        ["points", "gd", "gf", "wins", "tie_random"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).head(8).copy()

    qualified = pd.concat([top_two, best_thirds], ignore_index=True)

    if len(qualified) != 32:
        raise RuntimeError(f"Se esperaban 32 clasificados y salieron {len(qualified)}")

    return qualified


def seed_bracket(qualified: pd.DataFrame) -> list[tuple[dict, dict]]:
    """
    Construye bracket provisional por siembra.

    No es el bracket oficial.
    Se ordena:
        1. ganadores de grupo,
        2. segundos,
        3. mejores terceros,
    y dentro de cada bloque por rendimiento de grupo.
    """

    q = qualified.sort_values(
        ["position", "points", "gd", "gf", "wins", "tie_random"],
        ascending=[True, False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    # Orden clásico de bracket sembrado para 32 equipos.
    pair_indices = [
        (0, 31),
        (15, 16),
        (7, 24),
        (8, 23),
        (3, 28),
        (12, 19),
        (4, 27),
        (11, 20),
        (1, 30),
        (14, 17),
        (6, 25),
        (9, 22),
        (2, 29),
        (13, 18),
        (5, 26),
        (10, 21),
    ]

    pairs = []

    for i, j in pair_indices:
        pairs.append((q.iloc[i].to_dict(), q.iloc[j].to_dict()))

    return pairs


def pair_consecutive(entries: list[dict]) -> list[tuple[dict, dict]]:
    """
    Empareja ganadores consecutivos para la siguiente ronda.
    """

    if len(entries) % 2 != 0:
        raise ValueError("Número impar de equipos para emparejar.")

    return [
        (entries[i], entries[i + 1])
        for i in range(0, len(entries), 2)
    ]


def elo_knockout_lambdas(
    team_a: str,
    team_b: str,
    elos: dict[str, float],
) -> tuple[float, float]:
    """
    Lambdas rápidas para eliminatorias con ELO.

    No se aplica localía. Ambos equipos se tratan como neutrales.

    Esta función evita meter ventaja artificial de sede en KO.
    """

    elo_a = float(elos.get(team_a, 1500.0))
    elo_b = float(elos.get(team_b, 1500.0))

    diff = elo_a - elo_b

    base = 1.25
    sensitivity = 0.55

    lambda_a = base * np.exp(sensitivity * diff / 400.0)
    lambda_b = base * np.exp(-sensitivity * diff / 400.0)

    lambda_a = float(np.clip(lambda_a, 0.25, 4.50))
    lambda_b = float(np.clip(lambda_b, 0.25, 4.50))

    return lambda_a, lambda_b


def penalty_win_probability(
    team_a: str,
    team_b: str,
    elos: dict[str, float],
) -> float:
    """
    Probabilidad aproximada de ganar penales si el partido queda empatado.

    Mantiene los penales cerca de 50/50, con ajuste suave por fuerza.
    """

    elo_a = float(elos.get(team_a, 1500.0))
    elo_b = float(elos.get(team_b, 1500.0))

    diff = elo_a - elo_b

    p = 1.0 / (1.0 + 10.0 ** (-diff / 1000.0))

    return float(np.clip(p, 0.35, 0.65))


def simulate_knockout_match(
    team_a: str,
    team_b: str,
    elos: dict[str, float],
    rng: np.random.Generator,
) -> dict:
    """
    Simula partido de eliminación directa.

    Si hay empate en 90 minutos, desempata por penales.
    """

    lambda_a, lambda_b = elo_knockout_lambdas(team_a, team_b, elos)

    ga, gb = sample_score(lambda_a, lambda_b, rng=rng)

    decided_by_penalties = False

    if ga > gb:
        winner = team_a
        loser = team_b
    elif ga < gb:
        winner = team_b
        loser = team_a
    else:
        decided_by_penalties = True
        p_a = penalty_win_probability(team_a, team_b, elos)

        if rng.random() < p_a:
            winner = team_a
            loser = team_b
        else:
            winner = team_b
            loser = team_a

    return {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": int(ga),
        "goals_b": int(gb),
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "winner": winner,
        "loser": loser,
        "decided_by_penalties": bool(decided_by_penalties),
    }


def simulate_knockout(
    qualified: pd.DataFrame,
    elos: dict[str, float],
    rng: np.random.Generator,
    round_counts: dict[str, dict[str, int]],
) -> tuple[list[dict], str]:
    """
    Simula todo el bracket de eliminación directa.
    """

    current_pairs = seed_bracket(qualified)

    rounds = [
        ("Round of 32", "round_of_16"),
        ("Round of 16", "quarterfinal"),
        ("Quarterfinal", "semifinal"),
        ("Semifinal", "final"),
        ("Final", "champion"),
    ]

    bracket_rows = []

    for round_label, next_stage_key in rounds:
        winners = []

        for match_number, (a, b) in enumerate(current_pairs, start=1):
            team_a = a["team"]
            team_b = b["team"]

            result = simulate_knockout_match(
                team_a=team_a,
                team_b=team_b,
                elos=elos,
                rng=rng,
            )

            winners.append(
                {
                    "team": result["winner"],
                    "source_round": round_label,
                }
            )

            bracket_rows.append(
                {
                    "round": round_label,
                    "match_number": match_number,
                    "team_a": team_a,
                    "team_b": team_b,
                    "goals_a": result["goals_a"],
                    "goals_b": result["goals_b"],
                    "lambda_a": result["lambda_a"],
                    "lambda_b": result["lambda_b"],
                    "winner": result["winner"],
                    "loser": result["loser"],
                    "decided_by_penalties": result["decided_by_penalties"],
                }
            )

        for w in winners:
            round_counts[w["team"]][next_stage_key] += 1

        if next_stage_key == "champion":
            champion = winners[0]["team"]
            return bracket_rows, champion

        current_pairs = pair_consecutive(winners)

    raise RuntimeError("No se pudo determinar campeón.")


def simulate_many(
    n_simulations: int,
    seed: int,
    paths: dict,
) -> dict:
    """
    Ejecuta Monte Carlo del torneo completo.
    """

    rng = np.random.default_rng(seed)

    df_wc = load_worldcup_with_predictions(paths)
    elos = build_latest_elo(paths)

    all_teams = [
        team
        for group in GROUPS.values()
        for team in group
    ]

    round_counts = {
        team: {key: 0 for key in ROUND_KEYS}
        for team in all_teams
    }

    champion_counts = Counter()
    position_counts = Counter()

    sample_group_matches = None
    sample_standings = None
    sample_bracket = None

    for sim in range(n_simulations):
        if sim == 0 or (sim + 1) % max(1, n_simulations // 10) == 0:
            print(
                f"[simulación] {sim + 1:,}/{n_simulations:,} "
                f"({100 * (sim + 1) / n_simulations:.1f}%)",
                flush=True,
            )

        ranked_tables, standings, group_match_rows = simulate_group_stage(
            df_wc=df_wc,
            rng=rng,
        )

        qualified = select_qualified(standings)

        for team in qualified["team"]:
            round_counts[team]["round_of_32"] += 1

        for _, row in standings.iterrows():
            key = (
                row["group"],
                row["team"],
                int(row["position"]),
            )
            position_counts[key] += 1

        bracket_rows, champion = simulate_knockout(
            qualified=qualified,
            elos=elos,
            rng=rng,
            round_counts=round_counts,
        )

        champion_counts[champion] += 1

        if sim == 0:
            sample_group_matches = group_match_rows
            sample_standings = standings.copy()
            sample_bracket = bracket_rows

    champion_rows = []

    for team in all_teams:
        champion_rows.append(
            {
                "team": team,
                "group": TEAM_TO_GROUP[team],
                "champion_count": int(champion_counts[team]),
                "champion_probability": champion_counts[team] / n_simulations,
            }
        )

    champion_probs = pd.DataFrame(champion_rows).sort_values(
        "champion_probability",
        ascending=False,
    ).reset_index(drop=True)

    round_rows = []

    for team in all_teams:
        row = {
            "team": team,
            "group": TEAM_TO_GROUP[team],
        }

        for key in ROUND_KEYS:
            row[f"{key}_probability"] = round_counts[team][key] / n_simulations

        round_rows.append(row)

    round_probs = pd.DataFrame(round_rows).sort_values(
        "champion_probability",
        ascending=False,
    ).reset_index(drop=True)

    position_rows = []

    for group, teams in GROUPS.items():
        for team in teams:
            row = {
                "group": group,
                "team": team,
            }

            for pos in [1, 2, 3, 4]:
                row[f"pos_{pos}_probability"] = (
                    position_counts[(group, team, pos)] / n_simulations
                )

            position_rows.append(row)

    group_position_probs = pd.DataFrame(position_rows).sort_values(
        ["group", "team"]
    ).reset_index(drop=True)

    return {
        "champion_probs": champion_probs,
        "round_probs": round_probs,
        "group_position_probs": group_position_probs,
        "sample_group_matches": pd.DataFrame(sample_group_matches),
        "sample_standings": sample_standings,
        "sample_bracket": pd.DataFrame(sample_bracket),
        "metadata": {
            "created_at": now_iso(),
            "n_simulations": int(n_simulations),
            "seed": int(seed),
            "mode": "current_state_real_played_plus_simulated_pending",
            "group_stage": "real played matches fixed; pending matches simulated",
            "knockout": "neutral ELO-Poisson provisional seeded bracket",
            "official_bracket": False,
            "note": (
                "El bracket de eliminatorias es provisional por siembra. "
                "No debe interpretarse como bracket oficial del Mundial 2026."
            ),
        },
    }


def save_simulation_outputs(result: dict, paths: dict) -> dict:
    """
    Guarda salidas de la simulación.
    """

    predictions_dir = paths["predictions"]
    reports_dir = paths["reports"]

    out_paths = {
        "champion_probs": predictions_dir / "phase05_champion_probabilities.csv",
        "round_probs": predictions_dir / "phase05_round_probabilities.csv",
        "group_position_probs": predictions_dir / "phase05_group_position_probabilities.csv",
        "sample_group_matches": predictions_dir / "phase05_sample_group_matches.csv",
        "sample_standings": predictions_dir / "phase05_sample_group_standings.csv",
        "sample_bracket": predictions_dir / "phase05_sample_bracket.csv",
        "metadata": reports_dir / "phase05_simulation_report.json",
    }

    result["champion_probs"].to_csv(out_paths["champion_probs"], index=False, encoding="utf-8")
    result["round_probs"].to_csv(out_paths["round_probs"], index=False, encoding="utf-8")
    result["group_position_probs"].to_csv(out_paths["group_position_probs"], index=False, encoding="utf-8")
    result["sample_group_matches"].to_csv(out_paths["sample_group_matches"], index=False, encoding="utf-8")
    result["sample_standings"].to_csv(out_paths["sample_standings"], index=False, encoding="utf-8")
    result["sample_bracket"].to_csv(out_paths["sample_bracket"], index=False, encoding="utf-8")

    out_paths["metadata"].write_text(
        json.dumps(to_builtin(result["metadata"]), indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    return out_paths


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Simula el Mundial 2026 desde el estado actual."
    )

    parser.add_argument("--n", type=int, default=1000, help="Número de simulaciones Monte Carlo.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")

    args = parser.parse_args()

    paths = get_paths()

    print()
    print("=" * 80)
    print("FASE 5 — SIMULADOR COMPLETO MUNDIAL 2026")
    print("=" * 80)
    print(f"Simulaciones: {args.n:,}")
    print(f"Seed: {args.seed}")
    print()
    print("Nota: bracket de eliminatorias provisional por siembra, no oficial.")
    print("Las eliminatorias se simulan sin ventaja artificial de localía.")

    result = simulate_many(
        n_simulations=args.n,
        seed=args.seed,
        paths=paths,
    )

    out_paths = save_simulation_outputs(result, paths)

    print()
    print("-" * 80)
    print("TOP 20 CANDIDATOS AL TÍTULO")
    print("-" * 80)
    print(result["champion_probs"].head(20).to_string(index=False))

    print()
    print("-" * 80)
    print("PROBABILIDADES DE RONDA — TOP 20 POR TÍTULO")
    print("-" * 80)
    show_cols = [
        "team",
        "group",
        "round_of_32_probability",
        "round_of_16_probability",
        "quarterfinal_probability",
        "semifinal_probability",
        "final_probability",
        "champion_probability",
    ]
    print(result["round_probs"][show_cols].head(20).to_string(index=False))

    print()
    print("-" * 80)
    print("ARCHIVOS GENERADOS")
    print("-" * 80)

    for key, value in out_paths.items():
        print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("FASE 5 COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main_cli()
