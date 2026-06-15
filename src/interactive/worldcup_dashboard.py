
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import time
import pandas as pd
from IPython.display import display, HTML, clear_output

try:
    import ipywidgets as widgets
except Exception:
    widgets = None

from src.interactive.group_match_selector import GROUPS


TEAM_FLAGS = {
    "Mexico": "🇲🇽",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Czechia": "🇨🇿",
    "Canada": "🇨🇦",
    "Bosnia and Herzegovina": "🇧🇦",
    "Qatar": "🇶🇦",
    "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷",
    "Morocco": "🇲🇦",
    "Haiti": "🇭🇹",
    "Scotland": "🏴",
    "United States": "🇺🇸",
    "Paraguay": "🇵🇾",
    "Australia": "🇦🇺",
    "Turkey": "🇹🇷",
    "Germany": "🇩🇪",
    "Curacao": "🇨🇼",
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


def flag(team):
    return TEAM_FLAGS.get(str(team), "🏳️")


def team_label(team):
    return f"{flag(team)} {team}"


def whtml(html: str):
    if widgets is not None:
        return widgets.HTML(value=html)
    return HTML(html)


def display_html(html: str):
    display(HTML(html))


def header():
    return whtml(
        "<div style='padding:24px;border-radius:24px;"
        "background:linear-gradient(135deg,#5A002A,#071B3A 58%,#0B5D4A);"
        "color:white;margin-bottom:16px;border:1px solid rgba(255,255,255,.20);"
        "box-shadow:0 12px 32px rgba(16,24,40,.18);'>"
        "<div style='font-size:12px;letter-spacing:2.5px;color:#F5D06F;font-weight:900;'>"
        "FIFA WORLD CUP 2026 · PREDICTIVE SIMULATOR"
        "</div>"
        "<h1 style='margin:8px 0 4px 0;font-size:30px;'>🏆 Dashboard Mundial 2026</h1>"
        "<p style='margin:0;color:#E6E8EC;font-size:14px;'>"
        "Partidos, grupos, modo copa, rutas por equipo y Monte Carlo."
        "</p>"
        "</div>"
    )


def small_note(text):
    return whtml(
        f"<div style='padding:12px;border-radius:14px;background:#F8FAFC;"
        f"border:1px solid #EAECF0;color:#344054;margin:8px 0;'>{text}</div>"
    )


def ok_html(text):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#ECFDF3;"
        f"border:1px solid #ABEFC6;color:#027A48;margin:10px 0;'><b>{text}</b></div>"
    )


def warn_html(text):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#FFFAEB;"
        f"border:1px solid #FEDF89;color:#B54708;margin:10px 0;'><b>{text}</b></div>"
    )


def error_html(e):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#FEF3F2;"
        f"border:1px solid #FECDCA;color:#B42318;margin:10px 0;'><b>Error:</b> {e}</div>"
    )


def dashboard_cards():
    html = """
    <div style='display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:14px;margin:10px 0 16px 0;'>
      <div style='background:white;border:1px solid #EAECF0;border-radius:20px;padding:18px;box-shadow:0 8px 22px rgba(16,24,40,.07);'>
        <div style='font-size:13px;color:#5A002A;font-weight:900;'>🎮 Modo Copa</div>
        <h3 style='margin:6px 0;color:#101828;'>Una historia completa del Mundial</h3>
        <p style='color:#667085;margin:0;'>Simula grupos, mejores terceros, ronda de 32, eliminatorias y campeón.</p>
      </div>
      <div style='background:white;border:1px solid #EAECF0;border-radius:20px;padding:18px;box-shadow:0 8px 22px rgba(16,24,40,.07);'>
        <div style='font-size:13px;color:#5A002A;font-weight:900;'>⭐ Modo Equipo</div>
        <h3 style='margin:6px 0;color:#101828;'>Ruta probable de una selección</h3>
        <p style='color:#667085;margin:0;'>Elige grupo y equipo para ver probabilidad de campeón, rivales y caminos frecuentes.</p>
      </div>
      <div style='background:white;border:1px solid #EAECF0;border-radius:20px;padding:18px;box-shadow:0 8px 22px rgba(16,24,40,.07);'>
        <div style='font-size:13px;color:#5A002A;font-weight:900;'>🎲 MC Grupos</div>
        <h3 style='margin:6px 0;color:#101828;'>Probabilidades de fase de grupos</h3>
        <p style='color:#667085;margin:0;'>Calcula P(1°), P(2°), P(3°), P(4°) y P(clasifica).</p>
      </div>
      <div style='background:white;border:1px solid #EAECF0;border-radius:20px;padding:18px;box-shadow:0 8px 22px rgba(16,24,40,.07);'>
        <div style='font-size:13px;color:#5A002A;font-weight:900;'>⚽ Partidos</div>
        <h3 style='margin:6px 0;color:#101828;'>Predicción individual</h3>
        <p style='color:#667085;margin:0;'>Simula partidos de grupos o eliminatorias hipotéticas.</p>
      </div>
    </div>
    """
    return whtml(html)


def make_team_dropdowns(prefix: str):
    group_dd = widgets.Dropdown(
        options=sorted(GROUPS.keys()),
        value="A",
        description=f"Grupo {prefix}:",
        layout=widgets.Layout(width="190px"),
    )

    team_dd = widgets.Dropdown(
        options=GROUPS["A"],
        value=GROUPS["A"][0],
        description=f"Equipo {prefix}:",
        layout=widgets.Layout(width="330px"),
    )

    def update(change):
        g = change["new"]
        team_dd.options = GROUPS[g]
        team_dd.value = GROUPS[g][0]

    group_dd.observe(update, names="value")

    return group_dd, team_dd


def lanzar_dashboard_mundial_2026(project_root: str | Path):
    if widgets is None:
        raise ImportError("ipywidgets no está disponible.")

    project_root = Path(project_root)

    # ==========================================================
    # INICIO
    # ==========================================================
    tab_inicio = widgets.VBox(
        [
            small_note(
                "Este dashboard concentra los módulos principales. "
                "Para una entrega final, usa Modo Copa y Modo Equipo como vistas principales."
            ),
            dashboard_cards(),
        ]
    )

    # ==========================================================
    # PARTIDOS
    # ==========================================================
    out_partidos = widgets.Output()

    btn_selector_grupos = widgets.Button(
        description="Abrir selector de grupos",
        button_style="success",
        layout=widgets.Layout(width="260px"),
    )

    def abrir_selector_grupos(_):
        with out_partidos:
            clear_output()
            try:
                from src.interactive.group_match_selector_v2 import lanzar_selector_grupos_v2
                display_html(ok_html("Selector de fase de grupos abierto."))
                lanzar_selector_grupos_v2(project_root=project_root, corners_cards_mode="legacy")
            except Exception as e:
                display_html(error_html(e))

    btn_selector_grupos.on_click(abrir_selector_grupos)

    g1, e1 = make_team_dropdowns("1")
    g2, e2 = make_team_dropdowns("2")

    ko_round = widgets.Dropdown(
        options=["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"],
        value="Quarterfinal",
        description="Ronda:",
        layout=widgets.Layout(width="260px"),
    )

    ko_date = widgets.Text(
        value="2026-07-09",
        description="Fecha:",
        layout=widgets.Layout(width="230px"),
    )

    ko_venue = widgets.Dropdown(
        options=["United States", "Mexico", "Canada", "TBD"],
        value="United States",
        description="Sede:",
        layout=widgets.Layout(width="240px"),
    )

    ko_scenario = widgets.Dropdown(
        options=["base", "moderate", "diaspora"],
        value="base",
        description="Escenario:",
        layout=widgets.Layout(width="240px"),
    )

    ko_seed = widgets.IntText(
        value=2026,
        description="Seed:",
        layout=widgets.Layout(width="160px"),
    )

    btn_ko = widgets.Button(
        description="Simular KO libre",
        button_style="warning",
        layout=widgets.Layout(width="220px"),
    )

    def simular_ko(_):
        with out_partidos:
            clear_output()
            try:
                from src.simulation.knockout_match_simulator import simulate_knockout_match

                display_html(
                    warn_html(
                        "KO libre es hipotético. Para la llave oficial simulada, usa Modo Copa."
                    )
                )

                t0 = time.time()

                result = simulate_knockout_match(
                    home_team=e1.value,
                    away_team=e2.value,
                    match_date=ko_date.value,
                    round_name=ko_round.value,
                    venue_country=ko_venue.value,
                    project_root=project_root,
                    scenario=ko_scenario.value,
                    seed=int(ko_seed.value),
                    corners_cards_mode="legacy",
                )

                elapsed = time.time() - t0

                display_html(
                    ok_html(
                        f"Ganador simulado: {team_label(result['winner'])} · Tiempo: {elapsed:.1f} s"
                    )
                )

                cols = [
                    "home_team",
                    "away_team",
                    "round_name",
                    "winner",
                    "method",
                    "home_goals_90",
                    "away_goals_90",
                    "home_goals_et",
                    "away_goals_et",
                    "scenario",
                    "net_host_advantage",
                ]
                cols = [c for c in cols if c in result]
                display(pd.DataFrame([{c: result.get(c) for c in cols}]))

            except Exception as e:
                display_html(error_html(e))

    btn_ko.on_click(simular_ko)

    tab_partidos = widgets.VBox(
        [
            small_note(
                "Partidos individuales. Para fase de grupos usa el selector existente; "
                "para eliminatorias hipotéticas elige equipos desde grupo/equipo."
            ),
            widgets.HBox([btn_selector_grupos]),
            whtml("<hr style='border:none;border-top:1px solid #EAECF0;margin:14px 0;'>"),
            whtml("<h3 style='color:#101828;margin:4px 0;'>Eliminatoria libre</h3>"),
            widgets.HBox([g1, e1]),
            widgets.HBox([g2, e2]),
            widgets.HBox([ko_round, ko_date, ko_venue]),
            widgets.HBox([ko_scenario, ko_seed, btn_ko]),
            out_partidos,
        ]
    )

    # ==========================================================
    # MODO COPA
    # ==========================================================
    out_copa = widgets.Output()

    btn_copa = widgets.Button(
        description="Abrir Modo Copa",
        button_style="danger",
        layout=widgets.Layout(width="240px"),
    )

    def abrir_copa(_):
        with out_copa:
            clear_output()
            try:
                from src.interactive.career_mode_menu import lanzar_career_mode_menu
                lanzar_career_mode_menu(project_root)
            except Exception as e:
                display_html(error_html(e))
                display_html(
                    warn_html(
                        "Si falta este módulo, corre primero los bloques 70.4 y 70.5."
                    )
                )

    btn_copa.on_click(abrir_copa)

    tab_copa = widgets.VBox(
        [
            small_note(
                "Modo carrera de Copa completa. Simula una edición concreta del Mundial, "
                "con grupos, mejores terceros, llave y campeón."
            ),
            btn_copa,
            out_copa,
        ]
    )

    # ==========================================================
    # MODO EQUIPO
    # ==========================================================
    out_equipo = widgets.Output()

    btn_equipo = widgets.Button(
        description="Abrir Modo Equipo",
        button_style="success",
        layout=widgets.Layout(width="240px"),
    )

    def abrir_equipo(_):
        with out_equipo:
            clear_output()
            try:
                from src.interactive.team_roadmap_menu import lanzar_modo_equipo
                lanzar_modo_equipo(project_root)
            except Exception as e:
                display_html(error_html(e))
                display_html(
                    warn_html(
                        "Si falta este módulo, corre primero los bloques 70.9, 70.10 y 70.11."
                    )
                )

    btn_equipo.on_click(abrir_equipo)

    tab_equipo = widgets.VBox(
        [
            small_note(
                "Elige grupo y equipo. El simulador corre Copas completas y reconstruye "
                "los rivales más probables de esa selección desde grupos hasta la final."
            ),
            btn_equipo,
            out_equipo,
        ]
    )

    # ==========================================================
    # MC GRUPOS
    # ==========================================================
    out_mc_grupos = widgets.Output()

    btn_mc_grupos = widgets.Button(
        description="Abrir MC Grupos",
        button_style="info",
        layout=widgets.Layout(width="240px"),
    )

    def abrir_mc_grupos(_):
        with out_mc_grupos:
            clear_output()
            try:
                from src.interactive.group_mc_menu import lanzar_mc_grupos
                lanzar_mc_grupos(project_root)
            except Exception as e:
                display_html(error_html(e))
                display_html(
                    warn_html(
                        "Si falta este módulo, corre primero los bloques 70.6, 70.7 y 70.8."
                    )
                )

    btn_mc_grupos.on_click(abrir_mc_grupos)

    tab_mc_grupos = widgets.VBox(
        [
            small_note(
                "Monte Carlo solo de fase de grupos. Ideal para analizar clasificación, "
                "primer lugar, segundo lugar y mejores terceros."
            ),
            btn_mc_grupos,
            out_mc_grupos,
        ]
    )

    tabs = widgets.Tab(
        children=[
            tab_inicio,
            tab_partidos,
            tab_copa,
            tab_equipo,
            tab_mc_grupos,
        ]
    )

    titles = [
        "Inicio",
        "Partidos",
        "Modo Copa",
        "Modo Equipo",
        "MC Grupos",
    ]

    for i, title in enumerate(titles):
        tabs.set_title(i, title)

    display(header())
    display(tabs)
