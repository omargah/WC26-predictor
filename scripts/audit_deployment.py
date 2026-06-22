from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "app.py",
    "requirements.txt",
    ".gitignore",
    ".streamlit/config.toml",
    "README.md",
    "LICENSE",
    "docs/STREAMLIT_APP.md",
    "docs/DEPLOYMENT.md",
    "scripts/update_worldcup_state.py",
    "scripts/simulate_scenario.py",
    "src",
    "models/poisson_dc_base.joblib",
    "data/reference/available_matches.csv",
    "data/predictions/phase03_pending_predictions.csv",
    "data/predictions/phase05_v2_group_matches_once_fixed.csv",
]

def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)

def main():
    ok = True
    print("=" * 80)
    print("AUDITORIA DE DESPLIEGUE")
    print("=" * 80)
    for rel in REQUIRED:
        exists = (ROOT / rel).exists()
        print(f"{'OK' if exists else 'FALTA':8} {rel}")
        ok = ok and exists

    print("-" * 80)
    for target in ["app.py", "scripts/update_worldcup_state.py", "scripts/simulate_scenario.py"]:
        proc = run([sys.executable, "-m", "py_compile", target])
        print(f"compile {target}: {proc.returncode}")
        if proc.stderr:
            print(proc.stderr)
        ok = ok and proc.returncode == 0

    print("=" * 80)
    print("DESPLIEGUE LISTO" if ok else "AUN FALTAN ELEMENTOS")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
