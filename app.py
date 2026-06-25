# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import date, datetime
import json
import math
import subprocess
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_MATCHES = PROJECT_ROOT / "data" / "reference" / "available_matches.csv"
GROUP_MATCHES = PROJECT_ROOT / "data" / "predictions" / "phase05_v2_group_matches_once_fixed.csv"
OFFICIAL_STANDINGS = PROJECT_ROOT / "data" / "predictions" / "phase05_v2_group_standings_fixed.csv"
OFFICIAL_THIRDS = PROJECT_ROOT / "data" / "predictions" / "phase05_v2_best_thirds_fixed.csv"
OFFICIAL_R32 = PROJECT_ROOT / "data" / "predictions" / "phase05_v2_round_of_32_fixed.csv"
OFFICIAL_KO = PROJECT_ROOT / "data" / "predictions" / "phase05_v2_full_tournament_results_fixed.csv"
OFFICIAL_SUMMARY = PROJECT_ROOT / "reports" / "phase05_v2_full_tournament_summary_fixed.json"
SCENARIOS_DIR = PROJECT_ROOT / "data" / "scenarios"


TEAM_ES = {
    "Mexico": "México",
    "South Africa": "Sudáfrica",
    "South Korea": "Corea del Sur",
    "Czech Republic": "República Checa",
    "Czechia": "República Checa",
    "Canada": "Canadá",
    "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Qatar": "Qatar",
    "Switzerland": "Suiza",
    "Brazil": "Brasil",
    "Morocco": "Marruecos",
    "Haiti": "Haití",
    "Scotland": "Escocia",
    "United States": "Estados Unidos",
    "Paraguay": "Paraguay",
    "Australia": "Australia",
    "Turkey": "Turquía",
    "Germany": "Alemania",
    "Curacao": "Curazao",
    "Curaçao": "Curazao",
    "Ivory Coast": "Costa de Marfil",
    "Ecuador": "Ecuador",
    "Netherlands": "Países Bajos",
    "Japan": "Japón",
    "Sweden": "Suecia",
    "Tunisia": "Túnez",
    "Belgium": "Bélgica",
    "Egypt": "Egipto",
    "Iran": "Irán",
    "New Zealand": "Nueva Zelanda",
    "Spain": "España",
    "Cape Verde": "Cabo Verde",
    "Saudi Arabia": "Arabia Saudita",
    "Uruguay": "Uruguay",
    "France": "Francia",
    "Senegal": "Senegal",
    "Iraq": "Irak",
    "Norway": "Noruega",
    "Argentina": "Argentina",
    "Algeria": "Argelia",
    "Austria": "Austria",
    "Jordan": "Jordania",
    "Portugal": "Portugal",
    "DR Congo": "RD Congo",
    "Uzbekistan": "Uzbekistán",
    "Colombia": "Colombia",
    "England": "Inglaterra",
    "Croatia": "Croacia",
    "Ghana": "Ghana",
    "Panama": "Panamá",
}

FLAGS = {
    "Mexico": "🇲🇽",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Czech Republic": "🇨🇿",
    "Czechia": "🇨🇿",
    "Canada": "🇨🇦",
    "Bosnia and Herzegovina": "🇧🇦",
    "Qatar": "🇶🇦",
    "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷",
    "Morocco": "🇲🇦",
    "Haiti": "🇭🇹",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸",
    "Paraguay": "🇵🇾",
    "Australia": "🇦🇺",
    "Turkey": "🇹🇷",
    "Germany": "🇩🇪",
    "Curacao": "🇨🇼",
    "Curaçao": "🇨🇼",
    "Ivory Coast": "🇨🇮",
    "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱",
    "Japan": "🇯🇵",
    "Sweden": "🇸🇪",
    "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪",
    "Egypt": "🇪🇬",
    "Iran": "🇮🇷",
    "New Zealand": "🇳🇿",
    "Spain": "🇪🇸",
    "Cape Verde": "🇨🇻",
    "Saudi Arabia": "🇸🇦",
    "Uruguay": "🇺🇾",
    "France": "🇫🇷",
    "Senegal": "🇸🇳",
    "Iraq": "🇮🇶",
    "Norway": "🇳🇴",
    "Argentina": "🇦🇷",
    "Algeria": "🇩🇿",
    "Austria": "🇦🇹",
    "Jordan": "🇯🇴",
    "Portugal": "🇵🇹",
    "DR Congo": "🇨🇩",
    "Uzbekistan": "🇺🇿",
    "Colombia": "🇨🇴",
    "England": "🏴",
    "Croatia": "🇭🇷",
    "Ghana": "🇬🇭",
    "Panama": "🇵🇦",
}


st.set_page_config(
    page_title="Mundial 2026 Predictor",
    page_icon="⚽",
    layout="wide",
)


st.markdown(
    """
<style>
:root {
    --wc26-black: #09090b;
    --wc26-ink: #111827;
    --wc26-muted: #64748b;
    --wc26-line: #e5e7eb;
    --wc26-paper: #ffffff;
    --wc26-cream: #f8f5ef;
    --wc26-red: #e11d48;
    --wc26-blue: #2563eb;
    --wc26-green: #16a34a;
    --wc26-gold: #d4a017;
    --wc26-cyan: #06b6d4;
    --wc26-purple: #7c3aed;
}

html, body, [class*="css"] {
    font-family: "Noto Sans", "Inter", "Aptos", "Helvetica Neue", Arial, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(225,29,72,0.13), transparent 28%),
        radial-gradient(circle at 92% 4%, rgba(37,99,235,0.13), transparent 30%),
        radial-gradient(circle at 55% 0%, rgba(22,163,74,0.09), transparent 25%),
        linear-gradient(180deg, #fbfaf7 0%, #f8fafc 36%, #ffffff 100%);
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #080808 0%, #111827 48%, #0f172a 100%) !important;
    color: white;
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(90deg, var(--wc26-red), var(--wc26-blue)) !important;
    color: white !important;
    border: 0 !important;
    border-radius: 999px !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.22);
}

section[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-1px);
    filter: brightness(1.07);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 18px;
    padding: 0.85rem 1rem;
    box-shadow: 0 12px 30px rgba(15,23,42,0.06);
}

button[kind="primary"] {
    background: linear-gradient(90deg, var(--wc26-red), var(--wc26-blue)) !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 800 !important;
}

.main-title {
    font-size: 2.85rem;
    font-weight: 950;
    letter-spacing: -0.055em;
    line-height: 0.95;
    margin-bottom: 0.15rem;
    color: var(--wc26-black);
}

.subtitle {
    font-size: 1.35rem;
    font-weight: 800;
    color: #334155;
    margin-bottom: 0.4rem;
}

.hero {
    position: relative;
    border-radius: 28px;
    padding: 1.6rem 1.75rem;
    margin: 0.5rem 0 1.2rem 0;
    color: white;
    overflow: hidden;
    background:
        radial-gradient(circle at 12% 20%, rgba(225,29,72,0.85), transparent 24%),
        radial-gradient(circle at 78% 14%, rgba(37,99,235,0.85), transparent 28%),
        radial-gradient(circle at 52% 95%, rgba(22,163,74,0.75), transparent 32%),
        linear-gradient(135deg, #050505 0%, #111827 56%, #0f172a 100%);
    box-shadow: 0 22px 55px rgba(15,23,42,0.22);
}

.hero:before {
    content: "26";
    position: absolute;
    right: 1.1rem;
    top: -1.1rem;
    font-size: 8.8rem;
    font-weight: 1000;
    line-height: 1;
    letter-spacing: -0.12em;
    color: rgba(255,255,255,0.10);
}

.hero-title {
    position: relative;
    font-size: 2.65rem;
    font-weight: 1000;
    letter-spacing: -0.055em;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    position: relative;
    font-size: 1.05rem;
    color: rgba(255,255,255,0.83);
    max-width: 850px;
}

.hero-badges {
    position: relative;
    display: flex;
    gap: 0.45rem;
    margin-top: 0.9rem;
    flex-wrap: wrap;
}

.hero-badge {
    border: 1px solid rgba(255,255,255,0.22);
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(8px);
    border-radius: 999px;
    padding: 0.28rem 0.72rem;
    color: white;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.035em;
}

.card {
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 22px;
    padding: 1.05rem 1.15rem;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,250,252,0.96) 100%);
    margin-bottom: 0.9rem;
    box-shadow: 0 14px 34px rgba(15,23,42,0.07);
}

.card-red { border-left: 7px solid var(--wc26-red); }
.card-blue { border-left: 7px solid var(--wc26-blue); }
.card-green { border-left: 7px solid var(--wc26-green); }
.card-gold { border-left: 7px solid var(--wc26-gold); }

.small-muted {
    color: var(--wc26-muted);
    font-size: 0.89rem;
}

.match-title {
    font-size: 1.18rem;
    font-weight: 900;
    letter-spacing: -0.025em;
    color: var(--wc26-ink);
}

.badge {
    display: inline-block;
    border-radius: 999px;
    padding: 0.22rem 0.62rem;
    background: #eef2ff;
    color: #3730a3;
    font-size: 0.78rem;
    font-weight: 850;
    letter-spacing: 0.025em;
}

.badge-live {
    background: #fee2e2;
    color: #991b1b;
}

.badge-ai {
    background: linear-gradient(90deg, #111827, #2563eb);
    color: white;
}

.match-card-26 {
    border-radius: 24px;
    background:
        linear-gradient(90deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98)),
        radial-gradient(circle at 0% 0%, rgba(225,29,72,0.10), transparent 30%);
    border: 1px solid rgba(15,23,42,0.10);
    padding: 1.05rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 14px 34px rgba(15,23,42,0.07);
}

.match-topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.6rem;
}

.match-teams {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 0.8rem;
    margin: 0.55rem 0;
}

.team-left, .team-right {
    font-size: 1.12rem;
    font-weight: 950;
    color: var(--wc26-black);
}

.team-right {
    text-align: right;
}

.score-pill {
    min-width: 92px;
    text-align: center;
    padding: 0.42rem 0.75rem;
    border-radius: 16px;
    background: #09090b;
    color: white;
    font-size: 1.18rem;
    font-weight: 950;
    box-shadow: inset 0 -2px 0 rgba(255,255,255,0.10);
}

.pred-strip {
    border-radius: 16px;
    padding: 0.65rem 0.8rem;
    background:
        linear-gradient(90deg, rgba(225,29,72,0.08), rgba(37,99,235,0.08), rgba(22,163,74,0.08));
    border: 1px solid rgba(15,23,42,0.06);
    margin-top: 0.65rem;
}

.bracket-card {
    border: 1px solid rgba(15,23,42,0.10);
    border-radius: 20px;
    padding: 0.9rem;
    margin-bottom: 0.7rem;
    background:
        linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    box-shadow: 0 10px 26px rgba(15,23,42,0.055);
}

.bracket-match-no {
    display: inline-block;
    background: #09090b;
    color: white;
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 900;
    margin-bottom: 0.4rem;
}

.bracket-team {
    padding: 0.35rem 0.5rem;
    border-radius: 12px;
    background: white;
    border: 1px solid #e5e7eb;
    margin: 0.25rem 0;
    font-weight: 850;
}

.group-chip {
    display: inline-block;
    padding: 0.18rem 0.48rem;
    border-radius: 10px;
    background: #f1f5f9;
    color: #334155;
    font-weight: 800;
    font-size: 0.78rem;
}

hr {
    border-color: rgba(15,23,42,0.08);
}

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 10px 24px rgba(15,23,42,0.04);
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------

def team_name(team):
    return f"{FLAGS.get(team, '🏳️')} {TEAM_ES.get(team, team)}"


def plain_team(team):
    return TEAM_ES.get(team, team)


def run_command(cmd):
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def pick_col(df, candidates):
    exact = {str(c): str(c) for c in df.columns}
    lower = {str(c).lower().strip(): str(c) for c in df.columns}

    for c in candidates:
        if c in exact:
            return c

    for c in candidates:
        key = c.lower().strip()
        if key in lower:
            return lower[key]

    return None


def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def score_matrix(lambda_home, lambda_away, max_goals=10):
    rows = []
    total_prob = 0.0

    for h in range(max_goals + 1):
        ph = poisson_pmf(h, lambda_home)
        for a in range(max_goals + 1):
            pa = poisson_pmf(a, lambda_away)
            p = ph * pa
            total_prob += p
            rows.append({
                "home_goals": h,
                "away_goals": a,
                "score": f"{h}-{a}",
                "prob": p,
            })

    df = pd.DataFrame(rows)

    if total_prob > 0:
        df["prob"] = df["prob"] / total_prob

    return df


def match_markets(lambda_home, lambda_away, max_goals=10):
    sm = score_matrix(lambda_home, lambda_away, max_goals=max_goals)
    sm["total_goals"] = sm["home_goals"] + sm["away_goals"]

    p_home = sm.loc[sm["home_goals"] > sm["away_goals"], "prob"].sum()
    p_draw = sm.loc[sm["home_goals"] == sm["away_goals"], "prob"].sum()
    p_away = sm.loc[sm["home_goals"] < sm["away_goals"], "prob"].sum()

    markets = []

    def add(name, p):
        p = float(p)
        markets.append({
            "mercado": name,
            "probabilidad": p,
            "probabilidad_%": round(100 * p, 2),
            "momio_justo_decimal": round((1 / p), 3) if p > 0 else None,
        })

    add("Local gana", p_home)
    add("Empate", p_draw)
    add("Visitante gana", p_away)
    add("Doble oportunidad 1X", p_home + p_draw)
    add("Doble oportunidad 12", p_home + p_away)
    add("Doble oportunidad X2", p_draw + p_away)

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        add(f"Over {line}", sm.loc[sm["total_goals"] > line, "prob"].sum())
        add(f"Under {line}", sm.loc[sm["total_goals"] < line, "prob"].sum())

    p_btts_yes = sm.loc[(sm["home_goals"] > 0) & (sm["away_goals"] > 0), "prob"].sum()
    add("BTTS Sí", p_btts_yes)
    add("BTTS No", 1 - p_btts_yes)

    markets_df = pd.DataFrame(markets).sort_values("probabilidad", ascending=False).reset_index(drop=True)

    score_df = sm.sort_values("prob", ascending=False).head(10).copy()
    score_df["probabilidad_%"] = (100 * score_df["prob"]).round(2)
    score_df["momio_justo_decimal"] = (1 / score_df["prob"]).round(3)

    return markets_df, score_df


def prediction_summary(row):
    lam_h = float(row["lambda_home"])
    lam_a = float(row["lambda_away"])

    markets, scores = match_markets(lam_h, lam_a, max_goals=10)

    top_market = markets.iloc[0]
    top_score = scores.iloc[0]

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "top_market": top_market["mercado"],
        "top_market_prob": float(top_market["probabilidad_%"]),
        "top_score": top_score["score"],
        "top_score_prob": float(top_score["probabilidad_%"]),
        "markets": markets,
        "scores": scores,
    }


def result_1x2(h, a):
    if h > a:
        return "Local"
    if h < a:
        return "Visitante"
    return "Empate"


def normalize_group_matches(df):
    df = df.copy()

    date_col = pick_col(df, ["date", "fecha", "match_date", "game_date"])
    group_col = pick_col(df, ["group", "grupo", "Group"])

    home_col = pick_col(df, [
        "home_team", "equipo_local", "local", "team_home", "home",
        "team_a", "local_team", "home_name", "local_name", "equipo_l"
    ])

    away_col = pick_col(df, [
        "away_team", "equipo_visitante", "visitante", "team_away", "away",
        "team_b", "visitor_team", "away_name", "visitante_name", "equipo_v"
    ])

    hg_col = pick_col(df, [
        "current_home_goals", "goles_local", "home_goals", "goals_home",
        "goals_a", "home_score", "score_home", "goles_a"
    ])

    ag_col = pick_col(df, [
        "current_away_goals", "goles_visitante", "away_goals", "goals_away",
        "goals_b", "away_score", "score_away", "goles_b"
    ])

    lambda_h_col = pick_col(df, [
        "lambda_home", "lambda_local", "lambda_a", "lambda_a_90",
        "xg_home", "xg_local"
    ])

    lambda_a_col = pick_col(df, [
        "lambda_away", "lambda_visitante", "lambda_b", "lambda_b_90",
        "xg_away", "xg_visitante"
    ])

    played_col = pick_col(df, [
        "is_played", "played", "partido_jugado", "real_played", "is_real_played", "is_played_real_life"
    ])

    source_col = pick_col(df, [
        "simulation_mode", "mode", "source", "prediction_source", "match_source"
    ])

    city_col = pick_col(df, ["ciudad", "city", "host_city"])
    country_col = pick_col(df, ["pais_sede", "country", "host_country"])
    stadium_col = pick_col(df, ["estadio", "stadium", "venue"])

    missing = []
    if date_col is None:
        missing.append("date/fecha")
    if home_col is None:
        missing.append("home_team/equipo_local")
    if away_col is None:
        missing.append("away_team/equipo_visitante")
    if lambda_h_col is None:
        missing.append("lambda_home/lambda_local")
    if lambda_a_col is None:
        missing.append("lambda_away/lambda_visitante")

    if missing:
        raise RuntimeError(
            "No pude detectar columnas requeridas: "
            + ", ".join(missing)
            + "\\nColumnas disponibles: "
            + ", ".join(map(str, df.columns))
        )

    out = pd.DataFrame()
    out["date"] = df[date_col].astype(str)
    out["group"] = df[group_col].astype(str) if group_col else ""
    out["home_team"] = df[home_col].astype(str)
    out["away_team"] = df[away_col].astype(str)
    out["match"] = out["home_team"] + " vs " + out["away_team"]
    out["lambda_home"] = pd.to_numeric(df[lambda_h_col], errors="coerce")
    out["lambda_away"] = pd.to_numeric(df[lambda_a_col], errors="coerce")

    if hg_col:
        out["current_home_goals"] = pd.to_numeric(df[hg_col], errors="coerce")
    else:
        out["current_home_goals"] = None

    if ag_col:
        out["current_away_goals"] = pd.to_numeric(df[ag_col], errors="coerce")
    else:
        out["current_away_goals"] = None

    if played_col:
        out["is_played_bool"] = (
            df[played_col].astype(str).str.lower().isin(["true", "1", "yes", "sí", "si"])
        )
    elif source_col:
        source = df[source_col].astype(str).str.lower()
        out["is_played_bool"] = (
            source.str.contains("real", na=False)
            | (
                source.str.contains("played", na=False)
                & ~source.str.contains("simulated|pending", na=False, regex=True)
            )
        )
    else:
        out["is_played_bool"] = False

    out["status"] = out["is_played_bool"].map({
        True: "Real jugado",
        False: "Pendiente/simulado",
    })

    out["city"] = df[city_col].astype(str) if city_col else ""
    out["country"] = df[country_col].astype(str) if country_col else ""
    out["stadium"] = df[stadium_col].astype(str) if stadium_col else ""

    out = out.dropna(subset=["lambda_home", "lambda_away"]).reset_index(drop=True)

    return out


@st.cache_data(show_spinner=False)
def load_all_matches_cached():
    if GROUP_MATCHES.exists():
        return normalize_group_matches(pd.read_csv(GROUP_MATCHES))

    if REFERENCE_MATCHES.exists():
        df = pd.read_csv(REFERENCE_MATCHES)
        df["date"] = df["date"].astype(str)
        df["group"] = ""
        df["status"] = "Pendiente"
        df["is_played_bool"] = False
        df["current_home_goals"] = None
        df["current_away_goals"] = None
        df["city"] = ""
        df["country"] = ""
        df["stadium"] = ""
        return df

    return pd.DataFrame()


def safe_read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def safe_read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def latest_update_date():
    path = PROJECT_ROOT / "reports" / "worldcup_state_update_latest.json"

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    raw = str(data.get("started_at", "")).strip()

    if not raw:
        return None

    return raw[:10]


def should_auto_update_today():
    today = date.today().isoformat()
    last = latest_update_date()
    return last != today


def auto_update_once_per_session():
    today = date.today().isoformat()

    if st.session_state.get("auto_update_done_for") == today:
        return None

    if not should_auto_update_today():
        st.session_state["auto_update_done_for"] = today
        return None

    code, out, err = run_command([
        sys.executable,
        "scripts/update_worldcup_state.py",
        "--scope",
        "full",
    ])

    st.session_state["auto_update_done_for"] = today

    st.cache_data.clear()

    return {
        "code": code,
        "stdout": out,
        "stderr": err,
        "date": today,
    }


def init_state():
    if "scenario_rows" not in st.session_state:
        st.session_state.scenario_rows = []
    if "last_scenario_name" not in st.session_state:
        st.session_state.last_scenario_name = "escenario_streamlit"


def add_scenario_row(row, home_goals, away_goals):
    item = {
        "date": str(row["date"]),
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "notes": "streamlit_manual",
    }

    key = (item["date"], item["home_team"], item["away_team"])

    st.session_state.scenario_rows = [
        r for r in st.session_state.scenario_rows
        if (r["date"], r["home_team"], r["away_team"]) != key
    ]

    st.session_state.scenario_rows.append(item)


def scenario_df():
    if st.session_state.scenario_rows:
        return pd.DataFrame(st.session_state.scenario_rows)

    return pd.DataFrame(
        columns=["date", "home_team", "away_team", "home_goals", "away_goals", "notes"]
    )


def run_scenario(scenario_name, seed, allow_overwrite_played):
    clean_name = scenario_name.strip() or "escenario_streamlit"
    input_path = SCENARIOS_DIR / f"{clean_name}_manual_results.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)

    df = scenario_df()
    df.to_csv(input_path, index=False, encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/simulate_scenario.py",
        "--input",
        str(input_path),
        "--scenario-name",
        clean_name,
        "--seed",
        str(int(seed)),
    ]

    if allow_overwrite_played:
        cmd.append("--allow-overwrite-played")

    code, out, err = run_command(cmd)

    st.session_state.last_scenario_name = clean_name

    return code, out, err, input_path


def scenario_folder(name=None):
    clean = name or st.session_state.get("last_scenario_name", "escenario_streamlit")
    return SCENARIOS_DIR / clean


def show_group_tabs(standings):
    if standings is None or standings.empty:
        st.info("No hay tablas de grupo para mostrar.")
        return

    group_col = "group" if "group" in standings.columns else None

    if group_col is None:
        st.dataframe(standings, use_container_width=True)
        return

    groups = sorted(standings[group_col].dropna().astype(str).unique())
    tabs = st.tabs([f"Grupo {g}" for g in groups])

    for tab, g in zip(tabs, groups):
        with tab:
            sub = standings[standings[group_col].astype(str) == g].copy()
            st.dataframe(sub, use_container_width=True, hide_index=True)


def match_card(row, include_prediction=True):
    pred = prediction_summary(row)

    if pd.notna(row.get("current_home_goals", None)) and pd.notna(row.get("current_away_goals", None)):
        score = f"{int(row['current_home_goals'])} - {int(row['current_away_goals'])}"
        score_label = "Marcador real/base"
    else:
        score = pred["top_score"].replace("-", " - ")
        score_label = "Marcador IA probable"

    sede = []
    if str(row.get("stadium", "")).strip() and str(row.get("stadium", "")).lower() != "nan":
        sede.append(str(row.get("stadium")))
    if str(row.get("city", "")).strip() and str(row.get("city", "")).lower() != "nan":
        sede.append(str(row.get("city")))
    if str(row.get("country", "")).strip() and str(row.get("country", "")).lower() != "nan":
        sede.append(str(row.get("country")))

    sede_text = " · ".join(sede) if sede else "Sede no disponible"
    group_text = str(row.get("group", "")).strip()
    group_badge = f"Grupo {group_text}" if group_text else "Grupo no disponible"

    status_class = "badge-live" if "Real" in str(row.get("status", "")) else "badge-ai"

    st.markdown(
        f"""
<div class="match-card-26">
  <div class="match-topline">
    <div>
      <span class="badge {status_class}">{row.get('status', '')}</span>
      <span class="group-chip">{group_badge}</span>
    </div>
    <div class="small-muted">{row['date']} · {sede_text}</div>
  </div>

  <div class="match-teams">
    <div class="team-left">{team_name(row['home_team'])}</div>
    <div>
      <div class="score-pill">{score}</div>
      <div class="small-muted" style="text-align:center; margin-top:0.2rem;">{score_label}</div>
    </div>
    <div class="team-right">{team_name(row['away_team'])}</div>
  </div>

  <div class="pred-strip">
    <span class="badge badge-ai">Pronóstico hecho con IA</span>
    <span class="small-muted">
      &nbsp; λ {pred['lambda_home']:.2f} - {pred['lambda_away']:.2f}
      · Marcador más probable: <b>{pred['top_score']}</b> ({pred['top_score_prob']:.1f}%)
      · Proposición más probable: <b>{pred['top_market']}</b> ({pred['top_market_prob']:.1f}%)
    </span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def show_prediction_details(row):
    pred = prediction_summary(row)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("λ local", f"{pred['lambda_home']:.2f}")
    c2.metric("λ visitante", f"{pred['lambda_away']:.2f}")
    c3.metric("Marcador más probable", pred["top_score"], f"{pred['top_score_prob']:.1f}%")
    c4.metric("Proposición más probable", pred["top_market"], f"{pred['top_market_prob']:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Proposiciones más probables")
        st.dataframe(
            pred["markets"].head(14)[["mercado", "probabilidad_%", "momio_justo_decimal"]],
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("#### Marcadores exactos más probables")
        st.dataframe(
            pred["scores"][["score", "probabilidad_%", "momio_justo_decimal"]],
            use_container_width=True,
            hide_index=True,
        )


def show_r32_cards(r32):
    if r32 is None or r32.empty:
        st.info("No hay Round of 32 para mostrar.")
        return

    cols = st.columns(2)

    for i, (_, row) in enumerate(r32.iterrows()):
        col = cols[i % 2]

        same_group = bool(row.get("same_group_violation", False))
        warning = "⚠️ Revisión requerida" if same_group else "Sin choque de mismo grupo"

        with col:
            st.markdown(
                f"""
<div class="bracket-card">
  <div class="bracket-match-no">M{int(row['match_number'])}</div>
  <div class="small-muted">{row.get('slot_a', '')} vs {row.get('slot_b', '')} · {warning}</div>
  <div class="bracket-team">{team_name(row['team_a'])}</div>
  <div class="small-muted" style="text-align:center; font-weight:900;">VS</div>
  <div class="bracket-team">{team_name(row['team_b'])}</div>
  <div class="small-muted">
    Origen: Grupo {row.get('group_a', '')} posición {row.get('position_a', '')}
    · Grupo {row.get('group_b', '')} posición {row.get('position_b', '')}
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

def latest_available_date(matches):
    today = date.today().isoformat()
    dates = sorted(matches["date"].astype(str).unique())

    if today in dates:
        return today

    future = [d for d in dates if d >= today]
    if future:
        return future[0]

    return dates[-1] if dates else today


init_state()

# Producción: no se ejecuta actualización pesada al abrir.
# La app carga archivos precalculados y el usuario puede actualizar manualmente desde la barra lateral.
boot_update_result = None
matches_all = load_all_matches_cached()

if matches_all.empty:
    st.error("No se encontraron datos. Ejecuta primero: python scripts/update_worldcup_state.py --scope full")
    st.stop()


st.markdown(
    """
<div class="hero">
  <div class="hero-title">Mundial 2026 Predictor</div>
  <div class="hero-subtitle">
    Pronóstico hecho con IA · Simula partidos, escenarios de grupo, terceros lugares,
    Round of 32, eliminatorias y campeón proyectado.
  </div>
  <div class="hero-badges">
    <span class="hero-badge">México · USA · Canadá</span>
    <span class="hero-badge">Poisson / Dixon-Coles</span>
    <span class="hero-badge">Escenarios contrafactuales</span>
    <span class="hero-badge">Campeón proyectado</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Actualización diaria")

    auto_update_enabled = st.checkbox(
        "Actualizar automáticamente al abrir si cambió el día",
        value=False,
        help="En producción queda apagado por defecto para que la app cargue rápido. Actívalo solo si quieres ejecutar update_worldcup_state.py --scope full.",
    )

    last_update = latest_update_date()
    st.caption(f"Última actualización detectada: {last_update or 'no disponible'}")

    if auto_update_enabled:
        with st.spinner("Revisando actualización diaria..."):
            auto_result = auto_update_once_per_session()

        if auto_result is not None:
            if auto_result["code"] == 0:
                st.success("Actualización automática completada.")
            else:
                st.error("Falló la actualización automática.")

            with st.expander("Ver salida de actualización automática"):
                st.code(auto_result["stdout"])
                if auto_result["stderr"]:
                    st.code(auto_result["stderr"])

    if st.button("Actualizar datos y torneo", type="primary"):
        with st.spinner("Actualizando resultados reales y reconstruyendo torneo..."):
            code, out, err = run_command([
                sys.executable,
                "scripts/update_worldcup_state.py",
                "--scope",
                "full",
            ])

        st.cache_data.clear()

        if code == 0:
            st.success("Actualización completada.")
        else:
            st.error("Falló la actualización.")

        with st.expander("Ver salida"):
            st.code(out)
            if err:
                st.code(err)

    st.divider()

    st.header("Escenario")

    scenario_name = st.text_input("Nombre del escenario", value=st.session_state.last_scenario_name)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

    allow_overwrite_played = st.checkbox(
        "Permitir modificar partidos ya jugados",
        value=False,
        help="Actívalo para escenarios contrafactuales. No modifica los datos oficiales permanentemente.",
    )

    auto_recalc = st.checkbox(
        "Recalcular tablas al simular",
        value=True,
        help="Usa los scripts oficiales para recalcular grupos, terceros, Round of 32 y torneo.",
    )

    if st.button("Limpiar escenario"):
        st.session_state.scenario_rows = []
        st.success("Escenario limpiado.")

    st.caption(f"Marcadores en escenario actual: {len(st.session_state.scenario_rows)}")


tabs = st.tabs([
    "🏠 Hoy",
    "🔍 Partido individual",
    "🧩 Escenario",
    "📊 Tablas + 32avos",
    "🧬 Bracket",
    "🏆 Campeón",
    "🧪 Modelo vs realidad",
])


# ---------------------------------------------------------------------
# Tab 1: Hoy
# ---------------------------------------------------------------------
with tabs[0]:
    st.header("Marcadores y pronósticos del día")

    default_day = latest_available_date(matches_all)
    day = st.selectbox(
        "Día a revisar",
        sorted(matches_all["date"].astype(str).unique()),
        index=sorted(matches_all["date"].astype(str).unique()).index(default_day),
        key="today_selector",
    )

    today_matches = matches_all[matches_all["date"] == day].copy()

    if today_matches.empty:
        st.info("No hay partidos para este día.")
    else:
        st.write(f"Partidos encontrados: **{len(today_matches)}**")

        for _, row in today_matches.iterrows():
            match_card(row)

    st.markdown("### Escenario actual")
    st.dataframe(scenario_df(), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Tab 2: Partido individual
# ---------------------------------------------------------------------
with tabs[1]:
    st.header("Análisis de partido individual")

    c1, c2, c3 = st.columns([1, 1, 3])

    with c1:
        mode = st.radio(
            "Tipo de partido",
            ["Pendientes", "Jugados", "Todos"],
            index=2,
            horizontal=False,
        )

    pool = matches_all.copy()

    if mode == "Pendientes":
        pool = pool[~pool["is_played_bool"]].copy()
    elif mode == "Jugados":
        pool = pool[pool["is_played_bool"]].copy()

    with c2:
        dates = sorted(pool["date"].astype(str).unique())
        selected_date = st.selectbox("Fecha", dates, key="individual_date")

    sub = pool[pool["date"] == selected_date].copy()

    with c3:
        labels = {}
        for idx, row in sub.iterrows():
            score = ""
            if pd.notna(row.get("current_home_goals", None)) and pd.notna(row.get("current_away_goals", None)):
                score = f" | {int(row['current_home_goals'])}-{int(row['current_away_goals'])}"
            label = f"{row['match']} | {row.get('status', '')}{score}"
            labels[label] = idx

        if not labels:
            st.warning("No hay partidos para esta fecha con el filtro seleccionado.")
            st.stop()

        label = st.selectbox("Partido", list(labels.keys()), key="individual_match")

    if label is None or label not in labels:
        st.warning("Selecciona un partido válido.")
        st.stop()

    row = pool.loc[labels[label]]

    st.markdown("### Pronóstico hecho con IA")
    show_prediction_details(row)

    st.markdown("### Información del partido")

    info_cols = st.columns(4)
    info_cols[0].metric("Grupo", str(row.get("group", "")))
    info_cols[1].metric("Estado", str(row.get("status", "")))
    info_cols[2].metric("Fecha", str(row.get("date", "")))
    info_cols[3].metric("Sede", str(row.get("city", "")) if str(row.get("city", "")).strip() else "No disponible")

    if bool(row.get("is_played_bool", False)):
        st.markdown("### Resultado real/base")
        st.info(
            f"{team_name(row['home_team'])} {int(row['current_home_goals'])} - "
            f"{int(row['current_away_goals'])} {team_name(row['away_team'])}"
        )

    st.markdown("### Agregar marcador al escenario")

    g1, g2, g3 = st.columns([2, 1, 1])

    with g1:
        st.write(f"**{team_name(row['home_team'])} vs {team_name(row['away_team'])}**")
        if bool(row.get("is_played_bool", False)) and not allow_overwrite_played:
            st.warning("Este partido ya se jugó. Activa el modo contrafactual en la barra lateral para modificarlo.")

    with g2:
        default_h = int(row["current_home_goals"]) if pd.notna(row.get("current_home_goals", None)) else 1
        manual_h = st.number_input("Goles local", min_value=0, max_value=15, value=default_h, step=1, key="ind_h")

    with g3:
        default_a = int(row["current_away_goals"]) if pd.notna(row.get("current_away_goals", None)) else 0
        manual_a = st.number_input("Goles visitante", min_value=0, max_value=15, value=default_a, step=1, key="ind_a")

    can_add = (not bool(row.get("is_played_bool", False))) or allow_overwrite_played

    if st.button("Agregar este marcador", disabled=not can_add, type="primary", key="add_individual"):
        add_scenario_row(row, manual_h, manual_a)
        st.success("Marcador agregado al escenario.")


# ---------------------------------------------------------------------
# Tab 3: Escenario
# ---------------------------------------------------------------------
with tabs[2]:
    st.header("Constructor de escenario manual")

    st.markdown(
        """
Aquí puedes acumular marcadores. Se guardan en la sesión actual de Streamlit.
Al simular, se crea un CSV en `data/scenarios/` y se recalcula el torneo.
"""
    )

    current_df = scenario_df()
    st.dataframe(current_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Simular escenario completo", disabled=current_df.empty, type="primary"):
            with st.spinner("Simulando escenario completo..."):
                code, out, err, input_path = run_scenario(
                    scenario_name=scenario_name,
                    seed=seed,
                    allow_overwrite_played=allow_overwrite_played,
                )

            st.write(f"Entrada guardada en: `{input_path}`")

            if code == 0:
                st.success("Escenario simulado correctamente.")
            else:
                st.error("Falló la simulación.")

            with st.expander("Salida del proceso"):
                st.code(out)
                if err:
                    st.code(err)

    with col2:
        st.info(
            "Los resultados manuales no sustituyen los datos reales. "
            "Sirven para escenarios personales o contrafactuales."
        )

    folder = scenario_folder(scenario_name)
    if folder.exists():
        st.markdown("### Archivos del escenario")
        files = sorted([p.name for p in folder.iterdir()])
        st.write(files)


# ---------------------------------------------------------------------
# Tab 4: Tablas + Round of 32
# ---------------------------------------------------------------------
with tabs[3]:
    st.header("Tablas de grupo y Round of 32")

    source = st.radio(
        "Fuente",
        ["Escenario actual", "Base oficial/fixed"],
        horizontal=True,
    )

    if source == "Escenario actual":
        folder = scenario_folder(scenario_name)
        standings = safe_read_csv(folder / "scenario_group_standings.csv")
        thirds = safe_read_csv(folder / "scenario_best_thirds.csv")
        r32 = safe_read_csv(folder / "scenario_round_of_32.csv")

        if standings is None:
            st.warning("Todavía no hay escenario simulado. Ve a la pestaña Escenario y presiona Simular.")
    else:
        standings = safe_read_csv(OFFICIAL_STANDINGS)
        thirds = safe_read_csv(OFFICIAL_THIRDS)
        r32 = safe_read_csv(OFFICIAL_R32)

    st.subheader("Tablas de grupo")
    show_group_tabs(standings)

    st.subheader("Mejores terceros")
    if thirds is not None:
        st.dataframe(thirds, use_container_width=True, hide_index=True)
    else:
        st.info("No hay tabla de mejores terceros.")

    st.subheader("Round of 32")
    if r32 is not None:
        keep = [
            c for c in [
                "match_number", "slot_a", "team_a", "slot_b", "team_b",
                "group_a", "group_b", "third_groups_key", "same_group_violation"
            ]
            if c in r32.columns
        ]
        st.dataframe(r32[keep], use_container_width=True, hide_index=True)
    else:
        st.info("No hay Round of 32.")


# ---------------------------------------------------------------------
# Tab 5: Bracket
# ---------------------------------------------------------------------
with tabs[4]:
    st.header("Bracket visual")

    source_bracket = st.radio(
        "Fuente del bracket",
        ["Escenario actual", "Base oficial/fixed"],
        horizontal=True,
        key="bracket_source",
    )

    if source_bracket == "Escenario actual":
        folder = scenario_folder(scenario_name)
        r32 = safe_read_csv(folder / "scenario_round_of_32.csv")
        ko = safe_read_csv(folder / "scenario_full_tournament_results.csv")
    else:
        r32 = safe_read_csv(OFFICIAL_R32)
        ko = safe_read_csv(OFFICIAL_KO)

    st.subheader("Round of 32")
    show_r32_cards(r32)

    st.subheader("Eliminatorias simuladas")
    if ko is not None:
        keep = [
            c for c in [
                "round", "match_number", "team_a", "team_b",
                "goals_a_total", "goals_b_total", "decided_by", "winner"
            ]
            if c in ko.columns
        ]
        st.dataframe(ko[keep], use_container_width=True, hide_index=True)
    else:
        st.info("Simula el escenario para ver eliminatorias.")


# ---------------------------------------------------------------------
# Tab 6: Campeón / Monte Carlo
# ---------------------------------------------------------------------
with tabs[5]:
    st.header("¿Quién puede ser campeón del mundo?")

    st.markdown(
        """
El modelo puede simular el torneo completo. Más simulaciones reducen el ruido aleatorio,
pero no hacen que el modelo sea perfecto: solo estabilizan la estimación.
"""
    )

    scale_table = pd.DataFrame([
        {"simulaciones": 10, "uso": "Prueba rápida", "precisión": "Muy baja", "tiempo esperado": "Rápido"},
        {"simulaciones": 100, "uso": "Exploración básica", "precisión": "Baja/media", "tiempo esperado": "Puede tardar varios minutos"},
        {"simulaciones": 1000, "uso": "Estimación razonable", "precisión": "Media", "tiempo esperado": "Pesado en versión fiel"},
        {"simulaciones": 10000, "uso": "Estimación estable", "precisión": "Alta", "tiempo esperado": "Requiere versión rápida/optimizada"},
    ])
    st.dataframe(scale_table, use_container_width=True, hide_index=True)

    source_champ = st.radio(
        "Fuente",
        ["Escenario actual", "Base oficial/fixed"],
        horizontal=True,
        key="champ_source",
    )

    if source_champ == "Escenario actual":
        folder = scenario_folder(scenario_name)
        summary = safe_read_json(folder / "scenario_full_tournament_summary.json")
    else:
        summary = safe_read_json(OFFICIAL_SUMMARY)

    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Campeón", summary.get("champion", "-"))
        c2.metric("Subcampeón", summary.get("runner_up", "-"))
        c3.metric("Tercer lugar", summary.get("third_place", "-"))
        c4.metric("Cuarto lugar", summary.get("fourth_place", "-"))
    else:
        st.info("Todavía no hay resumen de torneo.")

    st.subheader("Monte Carlo")

    n_sims = st.selectbox("Número de simulaciones", [10, 100, 1000, 10000], index=0)
    st.caption("Por ahora, desde la app se recomienda 10 o 100 con el motor fiel. Para 1000+ conviene conectar la versión rápida.")

    if st.button("Correr Monte Carlo", type="primary"):
        if n_sims > 100:
            st.warning("Para 1000+ aún conviene usar el script rápido por consola. Lo conectaremos en el siguiente ajuste.")
        else:
            run_name = f"streamlit_mc_{n_sims}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cmd = [
                sys.executable,
                "scripts/simulate_tournament_montecarlo_original_safe_v2.py",
                "--played-policy",
                "fixed",
                "--n-total",
                str(n_sims),
                "--seed",
                str(int(seed)),
                "--run-name",
                run_name,
                "--progress-every",
                "1",
            ]

            with st.spinner("Corriendo Monte Carlo fiel..."):
                code, out, err = run_command(cmd)

            with st.expander("Salida Monte Carlo"):
                st.code(out)
                if err:
                    st.code(err)

            out_dir = PROJECT_ROOT / "data" / "predictions" / "mc_original_v2_runs" / run_name
            champs = safe_read_csv(out_dir / "champion_probabilities.csv")

            if code == 0 and champs is not None:
                st.success("Monte Carlo terminado.")
                st.dataframe(champs, use_container_width=True, hide_index=True)
            else:
                st.error("No se pudo leer el resultado del Monte Carlo.")


# ---------------------------------------------------------------------
# Tab 7: Modelo vs realidad
# ---------------------------------------------------------------------
with tabs[6]:
    st.header("Comparación del modelo contra resultados reales")

    # Lectura directa para evitar que la comparación se quede atrasada por caché
    raw_compare = pd.read_csv(GROUP_MATCHES)

    played = raw_compare[
        raw_compare["is_played_real_life"].astype(str).str.lower().isin(["true", "1"])
    ].copy()

    played["date"] = played["fecha"].astype(str)
    played["home_team"] = played["home"].astype(str)
    played["away_team"] = played["away"].astype(str)
    played["match"] = played["home_team"] + " vs " + played["away_team"]

    played["current_home_goals"] = played["real_home_goals"].fillna(played["home_goals"])
    played["current_away_goals"] = played["real_away_goals"].fillna(played["away_goals"])

    played["lambda_home"] = pd.to_numeric(played["lambda_home"], errors="coerce")
    played["lambda_away"] = pd.to_numeric(played["lambda_away"], errors="coerce")

    played = played.dropna(subset=["lambda_home", "lambda_away"]).copy()

    if played.empty:
        st.info("No hay partidos jugados con predicción pre-partido disponible.")
    else:
        dates = sorted(played["date"].astype(str).unique())
        d = st.selectbox("Fecha", dates, key="compare_date")
        sub = played[played["date"] == d].copy()

        labels = {}
        for idx, row in sub.iterrows():
            label = (
                f"{row['match']} | Real "
                f"{int(row['current_home_goals'])}-{int(row['current_away_goals'])}"
            )
            labels[label] = idx

        selected = st.selectbox("Partido jugado", list(labels.keys()), key="compare_match")
        row = played.loc[labels[selected]]
        pred = prediction_summary(row)

        real_h = int(row["current_home_goals"])
        real_a = int(row["current_away_goals"])
        real_score = f"{real_h}-{real_a}"

        markets = pred["markets"]
        score_probs = pred["scores"]

        top_1x2 = markets[markets["mercado"].isin(["Local gana", "Empate", "Visitante gana"])].sort_values(
            "probabilidad", ascending=False
        ).iloc[0]["mercado"]

        pred_1x2 = {
            "Local gana": "Local",
            "Empate": "Empate",
            "Visitante gana": "Visitante",
        }.get(top_1x2, "")

        real_1x2 = result_1x2(real_h, real_a)

        st.markdown("### Resultado real")
        st.info(f"{team_name(row['home_team'])} {real_h} - {real_a} {team_name(row['away_team'])}")

        st.markdown("### Pronóstico hecho con IA")
        show_prediction_details(row)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1X2 real", real_1x2)
        c2.metric("1X2 modelo", pred_1x2)
        c3.metric("Acierto 1X2", "Sí" if real_1x2 == pred_1x2 else "No")
        c4.metric("Marcador exacto", "Sí" if real_score == pred["top_score"] else "No")

        st.markdown("### Error de goles")
        e1, e2 = st.columns(2)
        model_h, model_a = map(int, pred["top_score"].split("-"))
        e1.metric("Error goles local", abs(real_h - model_h))
        e2.metric("Error goles visitante", abs(real_a - model_a))
