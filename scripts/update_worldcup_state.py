# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(name, cmd, required=True, dry_run=False):
    t0 = time.time()

    print()
    print("=" * 90)
    print(f"[update] {name}")
    print("=" * 90)
    print(" ".join(cmd))

    if dry_run:
        return {
            "name": name,
            "cmd": cmd,
            "required": required,
            "status": "DRY_RUN",
            "returncode": None,
            "elapsed_seconds": 0.0,
        }

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
    )

    elapsed = time.time() - t0

    status = "OK" if proc.returncode == 0 else "ERROR"

    result = {
        "name": name,
        "cmd": cmd,
        "required": required,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
    }

    if proc.returncode != 0 and required:
        raise RuntimeError(
            f"Falló paso requerido: {name}. Return code: {proc.returncode}"
        )

    return result


def script_cmd(script_name, extra_args=None):
    script_path = PROJECT_ROOT / "scripts" / script_name

    if extra_args is None:
        extra_args = []

    return [sys.executable, str(script_path), *extra_args]


def script_exists(script_name):
    return (PROJECT_ROOT / "scripts" / script_name).exists()


def add_step(steps, name, script_name, args=None, required=True):
    if args is None:
        args = []

    if script_exists(script_name):
        steps.append(
            {
                "name": name,
                "script_name": script_name,
                "cmd": script_cmd(script_name, args),
                "required": required,
                "exists": True,
            }
        )
    else:
        steps.append(
            {
                "name": name,
                "script_name": script_name,
                "cmd": None,
                "required": required,
                "exists": False,
            }
        )


def summarize_pending_predictions():
    path = PROJECT_ROOT / "data" / "predictions" / "phase03_pending_predictions.csv"

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    df = pd.read_csv(path)

    date_col = None
    for c in ["fecha", "date", "match_date"]:
        if c in df.columns:
            date_col = c
            break

    summary = {
        "exists": True,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        summary["date_col"] = date_col
        summary["min_date"] = str(dates.min().date()) if not dates.isna().all() else None
        summary["max_date"] = str(dates.max().date()) if not dates.isna().all() else None
        summary["dates"] = sorted([str(x) for x in dates.dt.date.dropna().unique()])

    return summary


def summarize_reference_matches():
    path = PROJECT_ROOT / "data" / "reference" / "available_matches.csv"

    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
        }

    df = pd.read_csv(path)

    summary = {
        "exists": True,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    if "date" in df.columns:
        summary["dates"] = sorted(df["date"].dropna().astype(str).unique().tolist())

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Actualiza el estado local del Mundial 2026."
    )

    parser.add_argument(
        "--scope",
        choices=["core", "full"],
        default="core",
        help="core reconstruye datos/features/predicciones. full intenta reconstruir también torneo.",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Reentrena el modelo de goles. Por defecto solo usa el modelo existente.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--mc-n",
        type=int,
        default=0,
        help="Si es mayor que 0, corre Monte Carlo original seguro con n simulaciones.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo imprime los pasos, no ejecuta.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="No detenerse ante errores requeridos.",
    )

    args = parser.parse_args()

    started_at = datetime.now().isoformat(timespec="seconds")

    steps = []

    add_step(steps, "Revisar frescura de datos", "check_data_freshness.py", required=False)
    add_step(steps, "Construir dataset limpio", "build_clean_dataset.py", required=True)
    add_step(steps, "Construir features", "build_features.py", required=True)

    if args.retrain:
        add_step(steps, "Reentrenar Poisson Dixon-Coles", "train_phase03_poisson_dc.py", required=True)

    add_step(steps, "Predecir partidos pendientes", "predict_pending_phase03.py", required=True)
    add_step(steps, "Construir referencias web/Colab", "build_web_reference_options.py", required=True)
    add_step(steps, "Construir reporte maestro", "build_project_status_report.py", required=False)

    if args.scope == "full":
        add_step(steps, "Listar partidos de grupo V2", "list_group_matches_v2.py", required=False)
        add_step(
            steps,
            "Simular grupos fixed V2",
            "simulate_group_matches_v2.py",
            args=["--played-policy", "fixed", "--seed", str(args.seed)],
            required=False,
        )
        add_step(
            steps,
            "Construir tablas de grupo V2",
            "build_group_tables_v2.py",
            args=["--played-policy", "fixed"],
            required=False,
        )
        add_step(
            steps,
            "Construir Round of 32 V2",
            "build_round_of_32_v2.py",
            args=["--played-policy", "fixed"],
            required=False,
        )
        add_step(
            steps,
            "Simular torneo una vez V2",
            "simulate_tournament_once_v2.py",
            args=["--played-policy", "fixed", "--seed", str(args.seed)],
            required=False,
        )

    if args.mc_n > 0:
        run_name = f"daily_update_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        add_step(
            steps,
            f"Monte Carlo original seguro n={args.mc_n}",
            "simulate_tournament_montecarlo_original_safe_v2.py",
            args=[
                "--played-policy", "fixed",
                "--n-total", str(args.mc_n),
                "--seed", str(args.seed),
                "--run-name", run_name,
                "--progress-every", "1",
            ],
            required=False,
        )

    print()
    print("=" * 90)
    print("UPDATE WORLDCUP STATE")
    print("=" * 90)
    print(f"started_at: {started_at}")
    print(f"scope:      {args.scope}")
    print(f"retrain:    {args.retrain}")
    print(f"mc_n:       {args.mc_n}")
    print(f"dry_run:    {args.dry_run}")
    print("=" * 90)

    results = []

    for step in steps:
        if not step["exists"]:
            status = "MISSING_REQUIRED" if step["required"] else "MISSING_OPTIONAL"

            print()
            print("=" * 90)
            print(f"[update] {step['name']}")
            print("=" * 90)
            print(f"No existe scripts/{step['script_name']}")

            result = {
                "name": step["name"],
                "script_name": step["script_name"],
                "required": step["required"],
                "status": status,
                "returncode": None,
                "elapsed_seconds": 0.0,
            }

            results.append(result)

            if step["required"] and not args.continue_on_error:
                raise FileNotFoundError(f"Falta script requerido: {step['script_name']}")

            continue

        try:
            result = run_command(
                name=step["name"],
                cmd=step["cmd"],
                required=step["required"] and not args.continue_on_error,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            result = {
                "name": step["name"],
                "script_name": step["script_name"],
                "required": step["required"],
                "status": "EXCEPTION",
                "error": str(exc),
                "returncode": None,
                "elapsed_seconds": None,
            }

            results.append(result)

            if step["required"] and not args.continue_on_error:
                raise

            continue

        results.append(result)

    finished_at = datetime.now().isoformat(timespec="seconds")

    pending_summary = summarize_pending_predictions()
    references_summary = summarize_reference_matches()

    report = {
        "started_at": started_at,
        "finished_at": finished_at,
        "scope": args.scope,
        "retrain": bool(args.retrain),
        "seed": int(args.seed),
        "mc_n": int(args.mc_n),
        "dry_run": bool(args.dry_run),
        "steps": results,
        "pending_predictions": pending_summary,
        "reference_matches": references_summary,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    latest_path = reports_dir / "worldcup_state_update_latest.json"
    dated_path = reports_dir / f"worldcup_state_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    dated_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 90)
    print("UPDATE TERMINADO")
    print("=" * 90)
    print(f"latest report: {latest_path}")
    print(f"dated report:  {dated_path}")

    print()
    print("-" * 90)
    print("PREDICCIONES PENDIENTES")
    print("-" * 90)
    print(json.dumps(pending_summary, indent=2, ensure_ascii=False))

    print()
    print("-" * 90)
    print("REFERENCIAS WEB/COLAB")
    print("-" * 90)
    print(json.dumps(references_summary, indent=2, ensure_ascii=False))

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
