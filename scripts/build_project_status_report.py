# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_paths


def exists_status(path: Path) -> str:
    return "OK" if path.exists() else "FALTA"


def read_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_safe(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def file_row(label: str, path: Path, description: str):
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "status": exists_status(path),
        "description": description,
    }


def main():
    paths = get_paths()

    reports_dir = paths["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = paths["predictions"]
    models_dir = paths["models"]
    processed_dir = paths["processed"]
    features_dir = paths["features"]

    files = []

    # Datos
    files.append(file_row(
        "matches_clean",
        processed_dir / "matches_clean.parquet",
        "Dataset histórico limpio de partidos internacionales.",
    ))

    files.append(file_row(
        "worldcup_fixture_clean",
        processed_dir / "worldcup_2026_fixture_clean.csv",
        "Fixture limpio del Mundial 2026.",
    ))

    # Features
    files.append(file_row(
        "modeling_dataset_all",
        features_dir / "modeling_dataset_all.parquet",
        "Dataset de modelado con partidos jugados y pendientes.",
    ))

    files.append(file_row(
        "modeling_dataset_train",
        features_dir / "modeling_dataset_train.parquet",
        "Dataset de entrenamiento sin fixtures futuros.",
    ))

    files.append(file_row(
        "modeling_dataset_pending",
        features_dir / "modeling_dataset_pending.parquet",
        "Dataset de partidos pendientes para predicción.",
    ))

    # Modelo
    files.append(file_row(
        "poisson_dc_base",
        models_dir / "poisson_dc_base.joblib",
        "Modelo Poisson/Dixon-Coles entrenado.",
    ))

    files.append(file_row(
        "phase03_metrics",
        reports_dir / "phase03_metrics.json",
        "Métricas de validación del modelo de goles.",
    ))

    files.append(file_row(
        "phase03_pending_predictions",
        predictions_dir / "phase03_pending_predictions.csv",
        "Predicciones de partidos pendientes.",
    ))

    # Torneo V2
    files.append(file_row(
        "group_matches_fixed",
        predictions_dir / "phase05_v2_group_matches_once_fixed.csv",
        "Partidos de grupo simulados con resultados reales fijos.",
    ))

    files.append(file_row(
        "round_of_32_fixed",
        predictions_dir / "phase05_v2_round_of_32_fixed.csv",
        "Cruces de Round of 32 del escenario fixed.",
    ))

    files.append(file_row(
        "full_tournament_results_fixed",
        predictions_dir / "phase05_v2_full_tournament_results_fixed.csv",
        "Una simulación completa del torneo fixed.",
    ))

    files.append(file_row(
        "mc_original_fixed_results",
        predictions_dir / "mc_original_v2_runs" / "fixed_original_seed42" / "match_results.csv",
        "Resultados KO del Monte Carlo original/técnico fixed.",
    ))

    files.append(file_row(
        "mc_original_fixed_champions",
        predictions_dir / "mc_original_v2_runs" / "fixed_original_seed42" / "champion_probabilities.csv",
        "Probabilidades de campeón del Monte Carlo original/técnico fixed.",
    ))

    files.append(file_row(
        "mc_original_fixed_rounds",
        predictions_dir / "mc_original_v2_runs" / "fixed_original_seed42" / "round_probabilities.csv",
        "Probabilidades de avance por ronda del Monte Carlo original/técnico fixed.",
    ))

    files.append(file_row(
        "mc_original_fixed_metadata",
        predictions_dir / "mc_original_v2_runs" / "fixed_original_seed42" / "metadata.json",
        "Metadata de corrida Monte Carlo original/técnico fixed.",
    ))

    inventory = pd.DataFrame(files)

    # Métricas Phase 3
    phase03_metrics_path = reports_dir / "phase03_metrics.json"
    phase03_metrics = read_json_safe(phase03_metrics_path)

    # Predicciones pendientes
    pending_path = predictions_dir / "phase03_pending_predictions.csv"
    pending = read_csv_safe(pending_path)

    pending_summary = {}

    if pending is not None:
        pending_summary = {
            "n_pending_predictions": int(len(pending)),
            "columns": list(pending.columns),
        }

        for col in ["fecha", "date", "match_date"]:
            if col in pending.columns:
                dates = pd.to_datetime(pending[col], errors="coerce")
                pending_summary["min_date"] = str(dates.min().date()) if not dates.isna().all() else None
                pending_summary["max_date"] = str(dates.max().date()) if not dates.isna().all() else None
                break

    # Monte Carlo original
    mc_dir = predictions_dir / "mc_original_v2_runs" / "fixed_original_seed42"
    mc_champions = read_csv_safe(mc_dir / "champion_probabilities.csv")
    mc_rounds = read_csv_safe(mc_dir / "round_probabilities.csv")
    mc_summary = read_csv_safe(mc_dir / "top4_summary.csv")
    mc_metadata = read_json_safe(mc_dir / "metadata.json")

    mc_block = {}

    if mc_champions is not None:
        mc_block["top_champions"] = mc_champions.head(20).to_dict(orient="records")

    if mc_summary is not None:
        mc_block["n_simulations"] = int(mc_summary["simulation_id"].nunique()) if "simulation_id" in mc_summary.columns else int(len(mc_summary))

    if mc_rounds is not None:
        key_teams = [
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

        key_rows = []

        for team in key_teams:
            row = mc_rounds[mc_rounds["team"] == team]
            if row.empty:
                continue

            r = row.iloc[0].to_dict()
            key_rows.append(r)

        mc_block["key_teams_round_probabilities"] = key_rows

    mc_block["metadata"] = mc_metadata

    # Estado por fase
    phase_status = [
        {
            "phase": "Fase 1 — Datos",
            "status": "TERMINADA" if (processed_dir / "matches_clean.parquet").exists() else "INCOMPLETA",
            "objective": "Construir dataset limpio histórico + fixture Mundial 2026.",
        },
        {
            "phase": "Fase 2 — Features",
            "status": "TERMINADA" if (features_dir / "modeling_dataset_train.parquet").exists() else "INCOMPLETA",
            "objective": "Generar variables temporales sin leakage.",
        },
        {
            "phase": "Fase 3 — Modelo de goles",
            "status": "TERMINADA" if (models_dir / "poisson_dc_base.joblib").exists() else "INCOMPLETA",
            "objective": "Entrenar Poisson/Dixon-Coles y validar 1X2/goles.",
        },
        {
            "phase": "Fase 4 — Predictor de partidos",
            "status": "TERMINADA" if (predictions_dir / "phase03_pending_predictions.csv").exists() else "INCOMPLETA",
            "objective": "Calcular lambdas, 1X2, marcadores, over/BTTS para partidos.",
        },
        {
            "phase": "Fase 5 — Simulación torneo",
            "status": "TERMINADA" if (predictions_dir / "phase05_v2_round_of_32_fixed.csv").exists() else "INCOMPLETA",
            "objective": "Simular grupos, mejores terceros, Annexe C y KO.",
        },
        {
            "phase": "Fase 5B — Monte Carlo técnico",
            "status": "TERMINADA" if (mc_dir / "champion_probabilities.csv").exists() else "INCOMPLETA",
            "objective": "Estimar probabilidades de campeón y rondas con checkpoint.",
        },
        {
            "phase": "Fase 6 — Consolidación final",
            "status": "EN PROCESO",
            "objective": "Documentar, limpiar, empaquetar y dejar repo listo para portafolio.",
        },
    ]

    phase_df = pd.DataFrame(phase_status)

    final_recommendations = [
        "No integrar córners/tarjetas en esta versión base hasta tener datos confiables.",
        "Mantener Poisson/Dixon-Coles como núcleo oficial del modelo actual.",
        "Usar Monte Carlo original/técnico como referencia principal.",
        "Usar Fast Monte Carlo solo como exploración, no como resultado oficial.",
        "Construir un README formal y un reporte técnico antes del commit final.",
        "Crear un script final único para predicción de partidos individuales.",
        "Crear un script final único para simulación de torneo.",
    ]

    project_status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "phase_status": phase_status,
        "inventory": files,
        "phase03_metrics": phase03_metrics,
        "pending_summary": pending_summary,
        "montecarlo_original_fixed": mc_block,
        "final_recommendations": final_recommendations,
    }

    out_json = reports_dir / "project_status_modelo_actual.json"
    out_md = reports_dir / "project_status_modelo_actual.md"
    out_inventory = reports_dir / "project_file_inventory.csv"
    out_phases = reports_dir / "project_phase_status.csv"

    out_json.write_text(
        json.dumps(project_status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    inventory.to_csv(out_inventory, index=False, encoding="utf-8")
    phase_df.to_csv(out_phases, index=False, encoding="utf-8")

    lines = []

    lines.append("# Estado actual del modelo Mundial 2026")
    lines.append("")
    lines.append(f"Generado: `{project_status['generated_at']}`")
    lines.append("")
    lines.append("## 1. Resumen ejecutivo")
    lines.append("")
    lines.append(
        "El proyecto ya cuenta con un pipeline funcional para datos históricos, "
        "features temporales sin leakage, modelo de goles Poisson/Dixon-Coles, "
        "predicción de partidos individuales, simulación de grupos y eliminatorias "
        "del Mundial 2026, y Monte Carlo técnico con checkpoint."
    )
    lines.append("")
    lines.append("## 2. Estado por fase")
    lines.append("")
    lines.append("| Fase | Estado | Objetivo |")
    lines.append("|---|---:|---|")

    for row in phase_status:
        lines.append(f"| {row['phase']} | {row['status']} | {row['objective']} |")

    lines.append("")
    lines.append("## 3. Inventario de archivos clave")
    lines.append("")
    lines.append("| Archivo | Estado | Descripción |")
    lines.append("|---|---:|---|")

    for row in files:
        lines.append(f"| `{row['path']}` | {row['status']} | {row['description']} |")

    lines.append("")
    lines.append("## 4. Métricas del modelo de goles")
    lines.append("")

    if phase03_metrics:
        for k, v in phase03_metrics.items():
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- No se encontró `reports/phase03_metrics.json`.")

    lines.append("")
    lines.append("## 5. Predicciones pendientes")
    lines.append("")

    if pending_summary:
        lines.append(f"- Partidos pendientes predichos: `{pending_summary.get('n_pending_predictions')}`")
        lines.append(f"- Fecha mínima: `{pending_summary.get('min_date')}`")
        lines.append(f"- Fecha máxima: `{pending_summary.get('max_date')}`")
    else:
        lines.append("- No se encontró archivo de predicciones pendientes.")

    lines.append("")
    lines.append("## 6. Monte Carlo original/técnico fixed")
    lines.append("")

    if mc_block.get("n_simulations") is not None:
        lines.append(f"- Simulaciones completadas: `{mc_block['n_simulations']}`")

    if mc_champions is not None:
        lines.append("")
        lines.append("### Top campeones")
        lines.append("")
        lines.append("| Equipo | Campeonatos | Probabilidad |")
        lines.append("|---|---:|---:|")

        for _, r in mc_champions.head(20).iterrows():
            prob = 100 * float(r["champion_probability"])
            lines.append(f"| {r['team']} | {int(r['championships'])} | {prob:.2f}% |")

    if mc_rounds is not None:
        lines.append("")
        lines.append("### Equipos clave")
        lines.append("")
        lines.append("| Equipo | R16 | QF | SF | Final | Campeón |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        for team in [
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
        ]:
            row = mc_rounds[mc_rounds["team"] == team]
            if row.empty:
                continue

            r = row.iloc[0]

            lines.append(
                f"| {team} | "
                f"{100*float(r.get('p_round_of_16', 0)):.2f}% | "
                f"{100*float(r.get('p_quarterfinal', 0)):.2f}% | "
                f"{100*float(r.get('p_semifinal', 0)):.2f}% | "
                f"{100*float(r.get('p_final', 0)):.2f}% | "
                f"{100*float(r.get('p_champion', 0)):.2f}% |"
            )

    lines.append("")
    lines.append("## 7. Recomendaciones de cierre")
    lines.append("")

    for rec in final_recommendations:
        lines.append(f"- {rec}")

    lines.append("")
    lines.append("## 8. Próximo paso")
    lines.append("")
    lines.append(
        "Construir los scripts finales de uso: uno para predicción de partidos "
        "individuales y otro para simulación consolidada del torneo."
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 90)
    print("FASE 6.1 — REPORTE MAESTRO DEL MODELO ACTUAL")
    print("=" * 90)
    print(f"JSON:       {out_json}")
    print(f"Markdown:   {out_md}")
    print(f"Inventario: {out_inventory}")
    print(f"Fases:      {out_phases}")

    print()
    print("-" * 90)
    print("ESTADO POR FASE")
    print("-" * 90)
    print(phase_df.to_string(index=False))

    print()
    print("-" * 90)
    print("ARCHIVOS CLAVE")
    print("-" * 90)
    print(inventory[["label", "status", "path"]].to_string(index=False))

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
