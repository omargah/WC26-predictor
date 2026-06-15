
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


def whtml(html: str):
    if widgets is not None:
        return widgets.HTML(value=html)
    return HTML(html)


def pct(x):
    try:
        return f"{100 * float(x):.1f}%"
    except Exception:
        return x


def style_summary(df: pd.DataFrame):
    show = df.copy()

    percent_cols = [
        "p_pos_1",
        "p_pos_2",
        "p_pos_3",
        "p_pos_4",
        "p_qualify_top2",
        "p_qualify_best_third",
        "p_qualify_total",
    ]

    for c in percent_cols:
        if c in show.columns:
            show[c] = show[c].map(pct)

    numeric_cols = ["avg_position", "avg_points", "avg_gd", "avg_gf", "avg_ga"]
    for c in numeric_cols:
        if c in show.columns:
            show[c] = show[c].map(lambda x: f"{float(x):.2f}")

    rename = {
        "group": "Grupo",
        "team": "Equipo",
        "simulations": "Sims",
        "p_pos_1": "P(1°)",
        "p_pos_2": "P(2°)",
        "p_pos_3": "P(3°)",
        "p_pos_4": "P(4°)",
        "p_qualify_top2": "P(top 2)",
        "p_qualify_best_third": "P(mejor 3°)",
        "p_qualify_total": "P(clasifica)",
        "avg_position": "Pos prom.",
        "avg_points": "Pts prom.",
        "avg_gd": "DG prom.",
        "avg_gf": "GF prom.",
        "avg_ga": "GC prom.",
    }

    show = show.rename(columns=rename)
    return show


def header():
    return whtml(
        "<div style='padding:20px;border-radius:18px;"
        "background:linear-gradient(135deg,#5A002A,#071B3A 65%,#0B5D4A);"
        "color:white;margin-bottom:14px;'>"
        "<div style='font-size:13px;letter-spacing:2px;color:#F5D06F;font-weight:700;'>MUNDIAL 2026</div>"
        "<h2 style='margin:6px 0 4px 0;'>🎲 Monte Carlo de fase de grupos</h2>"
        "<p style='margin:0;color:#E6E8EC;'>Estima posiciones de grupo y probabilidades de clasificación.</p>"
        "</div>"
    )


def note(text):
    return whtml(
        f"<div style='padding:12px;border-radius:12px;background:#F8FAFC;"
        f"border:1px solid #EAECF0;color:#344054;margin:8px 0;'>{text}</div>"
    )


def ok(text):
    return HTML(
        f"<div style='padding:14px;border-radius:12px;background:#ECFDF3;"
        f"border:1px solid #ABEFC6;color:#027A48;margin:8px 0;'><b>{text}</b></div>"
    )


def err(e):
    return HTML(
        f"<div style='padding:14px;border-radius:12px;background:#FEF3F2;"
        f"border:1px solid #FECDCA;color:#B42318;margin:8px 0;'><b>Error:</b> {e}</div>"
    )


def lanzar_mc_grupos(project_root: str | Path):
    if widgets is None:
        raise ImportError("ipywidgets no está disponible.")

    project_root = Path(project_root)

    fecha_corte = widgets.Text(
        value=str(pd.Timestamp.today().date()),
        description="Fecha corte:",
        layout=widgets.Layout(width="270px"),
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
            ("Prueba mínima — 3", 3),
            ("Prueba — 10", 10),
            ("Rápido — 50", 50),
            ("Medio — 100", 100),
            ("Más estable — 500", 500),
            ("Reporte — 1,000", 1000),
        ],
        value=10,
        description="Preset:",
        layout=widgets.Layout(width="300px"),
    )

    n_box = widgets.IntText(
        value=10,
        description="N:",
        layout=widgets.Layout(width="150px"),
    )

    group_filter = widgets.Dropdown(
        options=["Todos"] + list("ABCDEFGHIJKL"),
        value="Todos",
        description="Grupo:",
        layout=widgets.Layout(width="180px"),
    )

    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        bar_style="info",
        layout=widgets.Layout(width="70%"),
    )

    status = widgets.HTML(
        value="<b>Listo.</b> Elige N y presiona correr.",
        layout=widgets.Layout(width="100%"),
    )

    btn = widgets.Button(
        description="Correr MC de grupos",
        button_style="success",
        layout=widgets.Layout(width="230px"),
    )

    out = widgets.Output()

    def sync(change):
        n_box.value = int(change["new"])

    preset.observe(sync, names="value")

    def callback(current, total, message):
        p = int((current / max(total, 1)) * 100)
        progress.value = min(max(p, 0), 100)
        status.value = f"<b>{message}</b> · {current}/{total}"

    def run(_):
        with out:
            clear_output()

            try:
                from src.simulation.group_stage_monte_carlo import (
                    run_group_stage_monte_carlo,
                    save_group_monte_carlo_outputs,
                )

                n = int(n_box.value)

                display(
                    HTML(
                        f"<div style='padding:12px;border-radius:12px;background:#F8FAFC;"
                        f"border:1px solid #EAECF0;margin:8px 0;'>"
                        f"Corriendo <b>{n:,}</b> simulaciones de fase de grupos. "
                        f"Empieza con 10 o 50 para validar velocidad."
                        f"</div>"
                    )
                )

                t0 = time.time()

                result = run_group_stage_monte_carlo(
                    project_root=project_root,
                    n_simulations=n,
                    analysis_date=fecha_corte.value,
                    use_registered_results=bool(usar_registro.value),
                    seed=int(seed.value),
                    corners_cards_mode="legacy",
                    progress_callback=callback,
                )

                paths = save_group_monte_carlo_outputs(result, project_root)

                elapsed = time.time() - t0

                display(ok(f"MC de grupos terminado: {n:,} simulaciones en {elapsed:.1f} s."))

                summary = result["summary"].copy()

                if group_filter.value != "Todos":
                    summary = summary[summary["group"] == group_filter.value].copy()

                display(style_summary(summary))

                display(
                    HTML(
                        f"<div style='font-size:12px;color:#667085;margin-top:8px;'>"
                        f"Guardado en:<br>{paths['summary_path']}<br>{paths['team_simulations_path']}"
                        f"</div>"
                    )
                )

            except Exception as e:
                display(err(e))
                status.value = "<b>Error.</b> Falló el Monte Carlo de grupos."

    btn.on_click(run)

    display(header())

    display(
        widgets.VBox(
            [
                note(
                    "Este módulo no genera una sola Copa, sino probabilidades agregadas de la fase de grupos. "
                    "El Modo Copa sigue siendo una simulación única."
                ),
                widgets.HBox([fecha_corte, seed, usar_registro]),
                widgets.HBox([preset, n_box, group_filter]),
                widgets.HBox([progress]),
                status,
                btn,
                out,
            ]
        )
    )
