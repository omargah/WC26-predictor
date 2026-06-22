# -*- coding: utf-8 -*-

"""
Simulador de escenarios manuales — Mundial 2026.

Permite tomar la simulación base de fase de grupos, fijar marcadores manuales,
recalcular tablas, mejores terceros, Round of 32 y torneo completo.

Ejemplos:

1) Crear plantilla editable:

python scripts/simulate_scenario.py \
  --create-template data/scenarios/manual_results.csv

2) Ejecutar escenario:

python scripts/simulate_scenario.py \
  --input data/scenarios/manual_results.csv \
  --scenario-name prueba_manual \
  --seed 42
"""

from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"


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


def result_code(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "L"
    if home_goals < away_goals:
        return "V"
    return "E"


def to_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "si", "sí"}


def run(cmd: list[str]) -> None:
    print()
    print("=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if proc.returncode != 0:
        raise RuntimeError(f"Falló comando: {' '.join(cmd)}")


def load_base_group_matches() -> pd.DataFrame:
    path = PRED_DIR / "phase05_v2_group_matches_once_fixed.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre primero: python scripts/update_worldcup_state.py --scope full"
        )

    return pd.read_csv(path)


def detect_group_match_columns(df: pd.DataFrame) -> dict:
    cols = {
        "date": pick_col(df, ["fecha", "date", "match_date"]),
        "group": pick_col(df, ["group", "grupo"]),
        "home": pick_col(df, ["equipo_local", "home_team", "home", "local", "team_a"]),
        "away": pick_col(df, ["equipo_visitante", "away_team", "away", "visitante", "team_b"]),
        "home_goals": pick_col(df, ["goles_local", "home_goals", "goals_home", "goals_a", "home_score"]),
        "away_goals": pick_col(df, ["goles_visitante", "away_goals", "goals_away", "goals_b", "away_score"]),
        "result": pick_col(df, ["resultado_1x2", "resultado", "result"]),
        "is_played": pick_col(df, ["is_played", "played", "partido_jugado"]),
        "mode": pick_col(df, ["simulation_mode", "mode"]),
        "source": pick_col(df, ["source", "prediction_source"]),
    }

    required = ["date", "home", "away", "home_goals", "away_goals"]

    missing = [k for k in required if cols[k] is None]

    if missing:
        print("Columnas disponibles:")
        print(list(df.columns))
        raise RuntimeError("No pude detectar columnas requeridas: " + ", ".join(missing))

    return cols


def create_template(output_path: Path) -> None:
    df = load_base_group_matches()
    cols = detect_group_match_columns(df)

    work = df.copy()

    if cols["is_played"] is not None:
        pending = work[~work[cols["is_played"]].map(to_bool)].copy()
    else:
        pending = work.copy()

    out = pd.DataFrame({
        "date": pending[cols["date"]],
        "home_team": pending[cols["home"]],
        "away_team": pending[cols["away"]],
        "home_goals": "",
        "away_goals": "",
        "notes": "",
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")

    print()
    print("=" * 100)
    print("PLANTILLA DE ESCENARIO CREADA")
    print("=" * 100)
    print(output_path)
    print()
    print("Edita home_goals y away_goals solo en los partidos que quieras fijar.")
    print("=" * 100)


def load_manual_results(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    date_col = pick_col(df, ["date", "fecha", "match_date"])
    home_col = pick_col(df, ["home_team", "equipo_local", "home", "local"])
    away_col = pick_col(df, ["away_team", "equipo_visitante", "away", "visitante"])
    hg_col = pick_col(df, ["home_goals", "goles_local", "home_score"])
    ag_col = pick_col(df, ["away_goals", "goles_visitante", "away_score"])

    missing = [
        name for name, col in {
            "home_team": home_col,
            "away_team": away_col,
            "home_goals": hg_col,
            "away_goals": ag_col,
        }.items()
        if col is None
    ]

    if missing:
        print("Columnas disponibles en input:")
        print(list(df.columns))
        raise RuntimeError("Faltan columnas en input: " + ", ".join(missing))

    out = pd.DataFrame({
        "date": df[date_col] if date_col else "",
        "home_team": df[home_col],
        "away_team": df[away_col],
        "home_goals": pd.to_numeric(df[hg_col], errors="coerce"),
        "away_goals": pd.to_numeric(df[ag_col], errors="coerce"),
    })

    out = out.dropna(subset=["home_goals", "away_goals"]).copy()
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    out["_home_norm"] = out["home_team"].map(norm_text)
    out["_away_norm"] = out["away_team"].map(norm_text)
    out["_date_norm"] = out["date"].astype(str)

    return out


def apply_manual_results(
    base: pd.DataFrame,
    manual: pd.DataFrame,
    allow_overwrite_played: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = base.copy()
    cols = detect_group_match_columns(df)

    df["_home_norm"] = df[cols["home"]].map(norm_text)
    df["_away_norm"] = df[cols["away"]].map(norm_text)
    df["_date_norm"] = df[cols["date"]].astype(str)

    applied_rows = []

    for _, m in manual.iterrows():
        mask = (
            (df["_home_norm"] == m["_home_norm"])
            & (df["_away_norm"] == m["_away_norm"])
        )

        if str(m["_date_norm"]).strip() not in {"", "nan", "NaT"}:
            mask = mask & (df["_date_norm"] == str(m["_date_norm"]))

        candidates = df[mask]

        if len(candidates) == 0:
            raise RuntimeError(
                f"No encontré partido para escenario: {m['home_team']} vs {m['away_team']} ({m['date']})"
            )

        if len(candidates) > 1:
            raise RuntimeError(
                f"Partido ambiguo en escenario: {m['home_team']} vs {m['away_team']} ({m['date']})"
            )

        idx = candidates.index[0]

        if cols["is_played"] is not None:
            already_played = to_bool(df.loc[idx, cols["is_played"]])

            if already_played and not allow_overwrite_played:
                raise RuntimeError(
                    "Intentas sobrescribir un partido real ya jugado: "
                    f"{df.loc[idx, cols['home']]} vs {df.loc[idx, cols['away']]}. "
                    "Usa --allow-overwrite-played solo para contrafactuales."
                )

        hg = int(m["home_goals"])
        ag = int(m["away_goals"])

        df.loc[idx, cols["home_goals"]] = hg
        df.loc[idx, cols["away_goals"]] = ag

        if cols["result"] is not None:
            df.loc[idx, cols["result"]] = result_code(hg, ag)

        if cols["mode"] is not None:
            df.loc[idx, cols["mode"]] = "manual_scenario"

        if cols["source"] is not None:
            df.loc[idx, cols["source"]] = "manual_user_input"

        if cols["is_played"] is not None:
            # No lo marcamos como partido real jugado; solo queda fijado manualmente.
            df.loc[idx, cols["is_played"]] = False

        applied_rows.append({
            "date": df.loc[idx, cols["date"]],
            "home_team": df.loc[idx, cols["home"]],
            "away_team": df.loc[idx, cols["away"]],
            "home_goals": hg,
            "away_goals": ag,
            "result": result_code(hg, ag),
        })

    df = df.drop(columns=["_home_norm", "_away_norm", "_date_norm"], errors="ignore")

    applied = pd.DataFrame(applied_rows)

    return df, applied


def backup_file(path: Path, backup_dir: Path) -> Path | None:
    if not path.exists():
        return None

    backup_path = backup_dir / path.relative_to(PROJECT_ROOT)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)

    return backup_path


def restore_file(path: Path, backup_dir: Path) -> None:
    backup_path = backup_dir / path.relative_to(PROJECT_ROOT)

    if backup_path.exists():
        shutil.copy2(backup_path, path)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_report(scenario_dir: Path, scenario_name: str, applied: pd.DataFrame) -> None:
    lines = []

    lines.append(f"# Escenario: {scenario_name}")
    lines.append("")
    lines.append(f"Generado: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    lines.append("## Marcadores manuales aplicados")
    lines.append("")

    if len(applied) == 0:
        lines.append("No se aplicaron marcadores manuales. Se usó la simulación base.")
    else:
        lines.append("| Fecha | Local | Visitante | Marcador |")
        lines.append("|---|---|---|---:|")

        for _, r in applied.iterrows():
            lines.append(
                f"| {r['date']} | {r['home_team']} | {r['away_team']} | "
                f"{int(r['home_goals'])}-{int(r['away_goals'])} |"
            )

    lines.append("")
    lines.append("## Archivos generados")
    lines.append("")
    lines.append("- scenario_group_matches.csv")
    lines.append("- scenario_group_standings.csv")
    lines.append("- scenario_best_thirds.csv")
    lines.append("- scenario_qualified.csv")
    lines.append("- scenario_round_of_32.csv")
    lines.append("- scenario_full_tournament_results.csv")
    lines.append("- scenario_full_tournament_predictions.csv")
    lines.append("- scenario_full_tournament_summary.json")
    lines.append("")

    path = scenario_dir / "scenario_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def run_scenario(
    input_path: Path,
    scenario_name: str,
    seed: int,
    allow_overwrite_played: bool,
) -> None:
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

    scenario_dir = SCENARIOS_DIR / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_group_matches()
    manual = load_manual_results(input_path)

    scenario_matches, applied = apply_manual_results(
        base=base,
        manual=manual,
        allow_overwrite_played=allow_overwrite_played,
    )

    scenario_group_matches_path = scenario_dir / "scenario_group_matches.csv"
    applied_path = scenario_dir / "manual_results_applied.csv"

    scenario_matches.to_csv(scenario_group_matches_path, index=False, encoding="utf-8")
    applied.to_csv(applied_path, index=False, encoding="utf-8")

    fixed_group_path = PRED_DIR / "phase05_v2_group_matches_once_fixed.csv"

    generated_paths = [
        fixed_group_path,
        PRED_DIR / "phase05_v2_group_standings_fixed.csv",
        PRED_DIR / "phase05_v2_best_thirds_fixed.csv",
        PRED_DIR / "phase05_v2_qualified_fixed.csv",
        PRED_DIR / "phase05_v2_round_of_32_fixed.csv",
        PRED_DIR / "phase05_v2_full_tournament_results_fixed.csv",
        PRED_DIR / "phase05_v2_full_tournament_predictions_fixed.csv",
        REPORTS_DIR / "phase05_v2_full_tournament_summary_fixed.json",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        backup_dir = Path(tmp)

        for p in generated_paths:
            backup_file(p, backup_dir)

        try:
            # Sobrescribir temporalmente el archivo fixed para reutilizar los scripts oficiales.
            scenario_matches.to_csv(fixed_group_path, index=False, encoding="utf-8")

            run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_group_tables_v2.py"), "--played-policy", "fixed"])
            run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_round_of_32_v2.py"), "--played-policy", "fixed"])
            run([sys.executable, str(PROJECT_ROOT / "scripts" / "simulate_tournament_once_v2.py"), "--played-policy", "fixed", "--seed", str(seed)])

            # Copiar resultados generados al escenario.
            copy_if_exists(fixed_group_path, scenario_dir / "scenario_group_matches_used.csv")
            copy_if_exists(PRED_DIR / "phase05_v2_group_standings_fixed.csv", scenario_dir / "scenario_group_standings.csv")
            copy_if_exists(PRED_DIR / "phase05_v2_best_thirds_fixed.csv", scenario_dir / "scenario_best_thirds.csv")
            copy_if_exists(PRED_DIR / "phase05_v2_qualified_fixed.csv", scenario_dir / "scenario_qualified.csv")
            copy_if_exists(PRED_DIR / "phase05_v2_round_of_32_fixed.csv", scenario_dir / "scenario_round_of_32.csv")
            copy_if_exists(PRED_DIR / "phase05_v2_full_tournament_results_fixed.csv", scenario_dir / "scenario_full_tournament_results.csv")
            copy_if_exists(PRED_DIR / "phase05_v2_full_tournament_predictions_fixed.csv", scenario_dir / "scenario_full_tournament_predictions.csv")
            copy_if_exists(REPORTS_DIR / "phase05_v2_full_tournament_summary_fixed.json", scenario_dir / "scenario_full_tournament_summary.json")

        finally:
            # Restaurar archivos oficiales fixed originales.
            for p in generated_paths:
                restore_file(p, backup_dir)

    metadata = {
        "scenario_name": scenario_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(input_path),
        "scenario_dir": str(scenario_dir),
        "seed": int(seed),
        "n_manual_results": int(len(applied)),
        "allow_overwrite_played": bool(allow_overwrite_played),
        "files": {
            "scenario_group_matches": str(scenario_dir / "scenario_group_matches.csv"),
            "manual_results_applied": str(applied_path),
            "scenario_group_standings": str(scenario_dir / "scenario_group_standings.csv"),
            "scenario_best_thirds": str(scenario_dir / "scenario_best_thirds.csv"),
            "scenario_qualified": str(scenario_dir / "scenario_qualified.csv"),
            "scenario_round_of_32": str(scenario_dir / "scenario_round_of_32.csv"),
            "scenario_full_tournament_results": str(scenario_dir / "scenario_full_tournament_results.csv"),
            "scenario_full_tournament_predictions": str(scenario_dir / "scenario_full_tournament_predictions.csv"),
            "scenario_full_tournament_summary": str(scenario_dir / "scenario_full_tournament_summary.json"),
        },
    }

    (scenario_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    build_report(scenario_dir, scenario_name, applied)

    print()
    print("=" * 100)
    print("ESCENARIO COMPLETADO")
    print("=" * 100)
    print(f"scenario_name: {scenario_name}")
    print(f"scenario_dir:  {scenario_dir}")
    print(f"manual results: {len(applied)}")
    print()
    print("Archivos principales:")
    print(f" - {scenario_dir / 'scenario_group_standings.csv'}")
    print(f" - {scenario_dir / 'scenario_best_thirds.csv'}")
    print(f" - {scenario_dir / 'scenario_round_of_32.csv'}")
    print(f" - {scenario_dir / 'scenario_full_tournament_results.csv'}")
    print(f" - {scenario_dir / 'scenario_report.md'}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador de escenarios manuales del Mundial 2026."
    )

    parser.add_argument("--input", default=None, help="CSV con resultados manuales.")
    parser.add_argument("--scenario-name", default=None, help="Nombre de carpeta del escenario.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-overwrite-played", action="store_true")
    parser.add_argument("--create-template", default=None, help="Crea plantilla CSV y termina.")

    args = parser.parse_args()

    if args.create_template:
        create_template(Path(args.create_template))
        return

    if args.input is None:
        raise RuntimeError("Usa --input o --create-template.")

    input_path = Path(args.input)

    if args.scenario_name:
        scenario_name = args.scenario_name
    else:
        scenario_name = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_scenario(
        input_path=input_path,
        scenario_name=scenario_name,
        seed=args.seed,
        allow_overwrite_played=args.allow_overwrite_played,
    )


if __name__ == "__main__":
    main()
