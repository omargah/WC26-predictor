from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import shutil
import subprocess
import sys
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("America/Mexico_City")
TODAY = datetime.now(TZ).date().isoformat()

SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots" / "bracket" / TODAY
REPORT_SNAPSHOT_DIR = PROJECT_ROOT / "reports" / "snapshots" / "bracket" / TODAY

PREDICTIONS = PROJECT_ROOT / "data" / "predictions"
REPORTS = PROJECT_ROOT / "reports"

FILES_TO_SNAPSHOT = [
    PREDICTIONS / "phase05_v2_group_matches_once_fixed.csv",
    PREDICTIONS / "phase05_v2_group_standings_fixed.csv",
    PREDICTIONS / "phase05_v2_best_thirds_fixed.csv",
    PREDICTIONS / "phase05_v2_qualified_fixed.csv",
    PREDICTIONS / "phase05_v2_round_of_32_fixed.csv",
    PREDICTIONS / "phase05_v2_full_tournament_results_fixed.csv",
    PREDICTIONS / "phase05_v2_full_tournament_predictions_fixed.csv",
    REPORTS / "phase05_v2_full_tournament_summary_fixed.json",
]


def run(cmd):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def copy_snapshots():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    copied = []

    for src in FILES_TO_SNAPSHOT:
        if not src.exists():
            print("NO EXISTE, se omite:", src)
            continue

        if src.suffix == ".json":
            dst = REPORT_SNAPSHOT_DIR / src.name
        else:
            dst = SNAPSHOT_DIR / src.name

        shutil.copy2(src, dst)
        copied.append((src, dst))

    latest_pointer = PROJECT_ROOT / "data" / "snapshots" / "bracket" / "latest.txt"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(str(SNAPSHOT_DIR.relative_to(PROJECT_ROOT)) + "\n")

    print("\nARCHIVOS COPIADOS A SNAPSHOT:")
    for src, dst in copied:
        print("-", src.relative_to(PROJECT_ROOT), "->", dst.relative_to(PROJECT_ROOT))


def safe_read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_markdown_report():
    round32_path = PREDICTIONS / "phase05_v2_round_of_32_fixed.csv"
    ko_path = PREDICTIONS / "phase05_v2_full_tournament_results_fixed.csv"
    standings_path = PREDICTIONS / "phase05_v2_group_standings_fixed.csv"
    summary_path = REPORTS / "phase05_v2_full_tournament_summary_fixed.json"

    round32 = safe_read_csv(round32_path)
    ko = safe_read_csv(ko_path)
    standings = safe_read_csv(standings_path)

    summary = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}

    lines = []
    lines.append(f"# Actualización diaria de llaves — {TODAY}")
    lines.append("")
    lines.append("Este reporte se generó automáticamente con resultados reales disponibles y predicciones actualizadas.")
    lines.append("")

    if summary:
        lines.append("## Resumen de torneo proyectado")
        for key in ["champion", "runner_up", "third_place", "fourth_place"]:
            if key in summary:
                lines.append(f"- **{key}**: {summary[key]}")
        lines.append("")

    if not standings.empty:
        lines.append("## Líderes de grupo actuales/proyectados")
        needed = [c for c in ["group", "position", "team", "points", "gd", "gf"] if c in standings.columns]
        top = standings[standings["position"].isin([1, 2])] if "position" in standings.columns else standings
        lines.append("```text")
        lines.append(top[needed].to_string(index=False))
        lines.append("```")
        lines.append("")

    if not round32.empty:
        lines.append("## Round of 32 actualizado")
        needed = [c for c in ["match_number", "slot_a", "team_a", "slot_b", "team_b"] if c in round32.columns]
        lines.append("```text")
        lines.append(round32[needed].to_string(index=False))
        lines.append("```")
        lines.append("")

    if not ko.empty:
        lines.append("## Llave proyectada completa")
        needed = [
            c for c in [
                "round",
                "match_number",
                "team_a",
                "team_b",
                "goals_a_total",
                "goals_b_total",
                "decided_by",
                "winner",
                "loser",
            ]
            if c in ko.columns
        ]
        lines.append("```text")
        lines.append(ko[needed].to_string(index=False))
        lines.append("```")
        lines.append("")

    latest = REPORTS / "daily_bracket_latest.md"
    dated = REPORT_SNAPSHOT_DIR / "daily_bracket_report.md"

    latest.write_text("\n".join(lines))
    dated.write_text("\n".join(lines))

    print("\nREPORTES GENERADOS:")
    print("-", latest.relative_to(PROJECT_ROOT))
    print("-", dated.relative_to(PROJECT_ROOT))


def main():
    print("=" * 100)
    print("ACTUALIZACION DIARIA DE LLAVES")
    print("Fecha local CDMX:", TODAY)
    print("=" * 100)

    run([sys.executable, "scripts/update_worldcup_state.py", "--scope", "full", "--continue-on-error"])

    # Reforzamos explícitamente la reconstrucción de la llave.
    run([sys.executable, "scripts/build_group_tables_v2.py", "--played-policy", "fixed"])
    run([sys.executable, "scripts/build_round_of_32_v2.py", "--played-policy", "fixed"])
    run([sys.executable, "scripts/simulate_tournament_once_v2.py", "--played-policy", "fixed", "--seed", "42"])

    # Referencias para app/Colab y reporte maestro.
    run([sys.executable, "scripts/build_web_reference_options.py"])
    run([sys.executable, "scripts/build_project_status_report.py"])
    run([sys.executable, "scripts/audit_deployment.py"])

    copy_snapshots()
    build_markdown_report()

    print("\n" + "=" * 100)
    print("ACTUALIZACION DIARIA DE LLAVES TERMINADA")
    print("=" * 100)


if __name__ == "__main__":
    main()
