
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


def pct(x):
    try:
        return f"{100 * float(x):.1f}%"
    except Exception:
        return "NA"


def whtml(html: str):
    if widgets is not None:
        return widgets.HTML(value=html)
    return HTML(html)


def display_html(html: str):
    display(HTML(html))


def header():
    return whtml(
        "<div style='padding:24px;border-radius:22px;"
        "background:linear-gradient(135deg,#5A002A,#071B3A 62%,#0B5D4A);"
        "color:white;margin-bottom:16px;border:1px solid rgba(255,255,255,.18);'>"
        "<div style='font-size:12px;letter-spacing:2.5px;color:#F5D06F;font-weight:800;'>FIFA WORLD CUP 2026 · TEAM ROADMAP</div>"
        "<h2 style='margin:8px 0 4px 0;'>🏆 Modo Equipo</h2>"
        "<p style='margin:0;color:#E6E8EC;'>Elige una selección y estima sus rutas más probables al título.</p>"
        "</div>"
    )


def card_metric(title, value, subtitle=""):
    return (
        "<div style='background:white;border:1px solid #EAECF0;border-radius:18px;"
        "box-shadow:0 8px 20px rgba(16,24,40,.07);padding:16px;'>"
        f"<div style='font-size:12px;color:#667085;font-weight:700;'>{title}</div>"
        f"<div style='font-size:28px;color:#101828;font-weight:900;margin-top:4px;'>{value}</div>"
        f"<div style='font-size:12px;color:#98A2B3;margin-top:4px;'>{subtitle}</div>"
        "</div>"
    )


def render_key_summary(team, summary):
    k = summary["key_probs"]

    cards = ""
    cards += card_metric("Campeón", pct(k.get("p_champion")), "Probabilidad de ganar el Mundial")
    cards += card_metric("Final", pct(k.get("p_reach_final")), "Probabilidad de llegar a la final")
    cards += card_metric("Semifinal", pct(k.get("p_reach_semifinal")), "Probabilidad de llegar a semifinales")
    cards += card_metric("Cuartos", pct(k.get("p_reach_quarterfinal")), "Probabilidad de llegar a cuartos")
    cards += card_metric("Clasifica de grupo", pct(k.get("p_qualify_group")), "Top 2 o mejor tercero")
    cards += card_metric("Pts grupo prom.", f"{k.get('avg_group_points', 0):.2f}", "Puntos promedio en fase de grupos")

    return (
        "<div style='margin:16px 0;'>"
        f"<h2 style='color:#101828;margin-bottom:8px;'>Resumen de {team_label(team)}</h2>"
        "<div style='display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:12px;'>"
        f"{cards}"
        "</div>"
        "</div>"
    )


def render_stage_probs(stage_probs):
    if stage_probs is None or stage_probs.empty:
        return "<div></div>"

    rows = ""
    for _, r in stage_probs.iterrows():
        rows += (
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;'>{r['stage']}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;font-weight:800;'>{pct(r['probability'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;color:#667085;'>{int(r['count'])}</td>"
            "</tr>"
        )

    return (
        "<div style='background:white;border:1px solid #EAECF0;border-radius:18px;padding:14px;margin:12px 0;'>"
        "<h3 style='margin-top:0;color:#101828;'>Etapa máxima alcanzada</h3>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='color:#667085;text-align:left;'><th>Etapa</th><th>Prob.</th><th>Veces</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


def render_opponent_probs(opponent_probs):
    if opponent_probs is None or opponent_probs.empty:
        return "<div style='color:#667085;'>Sin rivales de eliminación directa en las simulaciones.</div>"

    html = "<div style='background:white;border:1px solid #EAECF0;border-radius:18px;padding:14px;margin:12px 0;'>"
    html += "<h3 style='margin-top:0;color:#101828;'>Rivales más probables por ronda</h3>"

    for round_name, df_round in opponent_probs.groupby("round"):
        html += f"<h4 style='color:#5A002A;margin-bottom:6px;'>{round_name}</h4>"
        top = df_round.sort_values("conditional_probability", ascending=False).head(8)

        chips = ""
        for _, r in top.iterrows():
            chips += (
                "<div style='display:inline-block;margin:4px;padding:8px 10px;border-radius:999px;"
                "background:#F8FAFC;border:1px solid #EAECF0;font-size:13px;'>"
                f"{team_label(r['opponent'])} · <b>{pct(r['conditional_probability'])}</b>"
                "</div>"
            )

        html += chips

    html += "</div>"
    return html


def render_common_paths(paths):
    if paths is None or paths.empty:
        return "<div></div>"

    rows = ""
    for _, r in paths.head(10).iterrows():
        rows += (
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;'>{r['path']}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;font-weight:800;'>{pct(r['probability'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;color:#667085;'>{int(r['count'])}</td>"
            "</tr>"
        )

    return (
        "<div style='background:white;border:1px solid #EAECF0;border-radius:18px;padding:14px;margin:12px 0;'>"
        "<h3 style='margin-top:0;color:#101828;'>Rutas más frecuentes</h3>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='color:#667085;text-align:left;'><th>Ruta</th><th>Prob.</th><th>Veces</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


def render_group_position_probs(group_probs):
    if group_probs is None or group_probs.empty:
        return "<div></div>"

    rows = ""
    for _, r in group_probs.iterrows():
        rows += (
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;'>Posición {int(r['group_position'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;font-weight:800;'>{pct(r['probability'])}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #EAECF0;color:#667085;'>{int(r['count'])}</td>"
            "</tr>"
        )

    return (
        "<div style='background:white;border:1px solid #EAECF0;border-radius:18px;padding:14px;margin:12px 0;'>"
        "<h3 style='margin-top:0;color:#101828;'>Posición en grupo</h3>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='color:#667085;text-align:left;'><th>Resultado</th><th>Prob.</th><th>Veces</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


def lanzar_modo_equipo(project_root: str | Path):
    if widgets is None:
        raise ImportError("ipywidgets no está disponible.")

    project_root = Path(project_root)

    group_dd = widgets.Dropdown(
        options=sorted(GROUPS.keys()),
        value="A",
        description="Grupo:",
        layout=widgets.Layout(width="180px"),
    )

    team_dd = widgets.Dropdown(
        options=GROUPS["A"],
        value=GROUPS["A"][0],
        description="Equipo:",
        layout=widgets.Layout(width="320px"),
    )

    def update_teams(change):
        g = change["new"]
        team_dd.options = GROUPS[g]
        team_dd.value = GROUPS[g][0]

    group_dd.observe(update_teams, names="value")

    fecha_corte = widgets.Text(
        value=str(pd.Timestamp.today().date()),
        description="Fecha corte:",
        layout=widgets.Layout(width="270px"),
    )

    escenario = widgets.Dropdown(
        options=["base", "moderate", "diaspora"],
        value="base",
        description="Escenario:",
        layout=widgets.Layout(width="240px"),
    )

    seed = widgets.IntText(
        value=2026,
        description="Seed:",
        layout=widgets.Layout(width="170px"),
    )

    usar_registro = widgets.Checkbox(
        value=True,
        description="Usar resultados registrados",
        indent=False,
        layout=widgets.Layout(width="260px"),
    )

    preset = widgets.Dropdown(
        options=[
            ("Prueba mínima — 3 Copas", 3),
            ("Prueba — 5 Copas", 5),
            ("Rápido — 10 Copas", 10),
            ("Medio — 50 Copas", 50),
            ("Más serio — 100 Copas", 100),
        ],
        value=5,
        description="Preset:",
        layout=widgets.Layout(width="310px"),
    )

    n_box = widgets.IntText(
        value=5,
        description="N:",
        layout=widgets.Layout(width="150px"),
    )

    def sync_n(change):
        n_box.value = int(change["new"])

    preset.observe(sync_n, names="value")

    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        bar_style="info",
        layout=widgets.Layout(width="75%"),
    )

    status = widgets.HTML(value="<b>Listo.</b> Elige equipo y número de Copas.")

    btn = widgets.Button(
        description="Simular ruta del equipo",
        button_style="success",
        layout=widgets.Layout(width="260px"),
    )

    out = widgets.Output()

    def callback(current, total, message):
        p = int((current / max(total, 1)) * 100)
        progress.value = min(max(p, 0), 100)
        status.value = f"<b>{message}</b> · {current}/{total}"

    def run(_):
        with out:
            clear_output()

            try:
                from src.simulation.team_roadmap_simulator import (
                    run_team_roadmap_monte_carlo,
                    save_team_roadmap_outputs,
                )

                team = team_dd.value
                n = int(n_box.value)

                display_html(
                    "<div style='padding:12px;border-radius:14px;background:#F8FAFC;"
                    "border:1px solid #EAECF0;margin:8px 0;color:#344054;'>"
                    f"Simulando <b>{n}</b> Copas completas para <b>{team_label(team)}</b>. "
                    "El sistema simula también los demás partidos para construir sus posibles llaves."
                    "</div>"
                )

                t0 = time.time()

                result = run_team_roadmap_monte_carlo(
                    project_root=project_root,
                    team=team,
                    n_simulations=n,
                    analysis_date=fecha_corte.value,
                    use_registered_results=bool(usar_registro.value),
                    seed=int(seed.value),
                    scenario=escenario.value,
                    corners_cards_mode="legacy",
                    progress_callback=callback,
                )

                paths = save_team_roadmap_outputs(result, project_root)
                elapsed = time.time() - t0

                progress.value = 100
                status.value = f"<b>Listo.</b> {n} Copas simuladas en {elapsed:.1f} s."

                summary = result["summary"]

                display_html(render_key_summary(team, summary))

                display_html(
                    "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>"
                    + render_group_position_probs(summary["group_position_probs"])
                    + render_stage_probs(summary["stage_probs"])
                    + "</div>"
                )

                display_html(render_opponent_probs(summary["opponent_probs"]))
                display_html(render_common_paths(summary["common_paths"]))

                display_html(
                    f"<div style='font-size:12px;color:#667085;margin-top:12px;'>"
                    f"Archivos guardados:<br>{paths['results_path']}<br>{paths['paths_path']}<br>{paths['opponents_path']}"
                    f"</div>"
                )

            except Exception as e:
                display_html(
                    f"<div style='padding:14px;border-radius:12px;background:#FEF3F2;"
                    f"border:1px solid #FECDCA;color:#B42318;margin:8px 0;'><b>Error:</b> {e}</div>"
                )
                status.value = "<b>Error.</b> Falló la simulación."

    btn.on_click(run)

    display(header())

    display(
        widgets.VBox(
            [
                whtml(
                    "<div style='padding:12px;border-radius:14px;background:#F8FAFC;"
                    "border:1px solid #EAECF0;color:#344054;margin-bottom:8px;'>"
                    "Selecciona un grupo y un equipo. El modelo simula Copas completas para estimar "
                    "sus rutas más probables desde grupos hasta la final."
                    "</div>"
                ),
                widgets.HBox([group_dd, team_dd]),
                widgets.HBox([fecha_corte, escenario, seed, usar_registro]),
                widgets.HBox([preset, n_box]),
                widgets.HBox([progress]),
                status,
                btn,
                out,
            ]
        )
    )
