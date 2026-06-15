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


def _html_widget(html: str):
    if widgets is not None:
        return widgets.HTML(value=html)
    return HTML(html)


def _header():
    return _html_widget(
        "<div style='padding:20px;border-radius:18px;background:linear-gradient(135deg,#101828,#344054);color:white;margin-bottom:14px;'>"
        "<h2 style='margin:0;'>🏆 Simulador Mundial 2026</h2>"
        "<p style='margin:6px 0 0 0;color:#D0D5DD;'>Elige una tarea. Cada pestaña muestra solo los controles necesarios.</p>"
        "</div>"
    )


def _note(text):
    return _html_widget(
        f"<div style='padding:12px;border-radius:12px;background:#F2F4F7;color:#344054;margin:8px 0;'>{text}</div>"
    )


def _ok(text):
    return _html_widget(
        f"<div style='padding:14px;border-radius:12px;background:#ECFDF3;color:#027A48;margin:8px 0;'><b>{text}</b></div>"
    )


def _err(e):
    return _html_widget(
        f"<div style='padding:14px;border-radius:12px;background:#FEF3F2;color:#B42318;margin:8px 0;'><b>Error:</b> {e}</div>"
    )


def _display_bracket(bracket: pd.DataFrame):
    cols = ['match_id', 'round', 'home_team', 'away_team', 'home_goals_90', 'away_goals_90', 'winner', 'method']
    cols = [c for c in cols if c in bracket.columns]
    display(HTML('<h3 style="color:#101828;">Llave simulada</h3>'))
    display(bracket[cols])


def lanzar_menu_simulador_v2(project_root: str | Path):
    if widgets is None:
        raise ImportError('ipywidgets no está disponible en este entorno.')

    project_root = Path(project_root)

    # Controles comunes
    fecha_corte = widgets.Text(value=str(pd.Timestamp.today().date()), description='Fecha corte:', layout=widgets.Layout(width='260px'))
    seed = widgets.IntText(value=2026, description='Seed:', layout=widgets.Layout(width='170px'))
    escenario = widgets.Dropdown(options=['base', 'moderate', 'diaspora'], value='base', description='Escenario:', layout=widgets.Layout(width='240px'))

    # ------------------------------------------------------------
    # Pestaña 1: fase de grupos
    # ------------------------------------------------------------
    out_grupos = widgets.Output()
    btn_grupos = widgets.Button(description='Abrir selector de grupos', button_style='success', layout=widgets.Layout(width='260px'))

    def run_grupos(_):
        with out_grupos:
            clear_output()
            try:
                from src.interactive.group_match_selector_v2 import lanzar_selector_grupos_v2
                display(_note('Aquí eliges grupo, jornada y partido. Para jornadas 2 y 3 se usan resultados registrados o simulados.'))
                lanzar_selector_grupos_v2(project_root=project_root, corners_cards_mode='legacy')
            except Exception as e:
                display(_err(e))

    btn_grupos.on_click(run_grupos)
    tab_grupos = widgets.VBox([
        _note('Para pronosticar un partido de fase de grupos, usa este selector.'),
        btn_grupos,
        out_grupos,
    ])

    # ------------------------------------------------------------
    # Pestaña 2: eliminatoria específica
    # ------------------------------------------------------------
    eq1 = widgets.Text(value='Mexico', description='Equipo 1:', layout=widgets.Layout(width='300px'))
    eq2 = widgets.Text(value='France', description='Equipo 2:', layout=widgets.Layout(width='300px'))
    ronda = widgets.Dropdown(options=['Round of 32', 'Round of 16', 'Quarterfinal', 'Semifinal', 'Final'], value='Quarterfinal', description='Ronda:', layout=widgets.Layout(width='260px'))
    sede = widgets.Text(value='United States', description='País sede:', layout=widgets.Layout(width='300px'))
    fecha_partido = widgets.Text(value='2026-07-09', description='Fecha:', layout=widgets.Layout(width='230px'))
    btn_ko = widgets.Button(description='Simular eliminatoria', button_style='success', layout=widgets.Layout(width='230px'))
    out_ko = widgets.Output()

    def run_ko(_):
        with out_ko:
            clear_output()
            try:
                from src.simulation.knockout_match_simulator import simulate_knockout_match
                t0 = time.time()
                result = simulate_knockout_match(
                    home_team=eq1.value,
                    away_team=eq2.value,
                    match_date=fecha_partido.value,
                    round_name=ronda.value,
                    venue_country=sede.value,
                    project_root=project_root,
                    scenario=escenario.value,
                    seed=int(seed.value),
                    corners_cards_mode='legacy',
                )
                display(_ok(f"Ganador simulado: {result.get('winner')} · Tiempo: {time.time() - t0:.1f} s"))
                cols = ['home_team', 'away_team', 'round_name', 'winner', 'method', 'home_goals_90', 'away_goals_90', 'home_goals_et', 'away_goals_et', 'scenario', 'net_host_advantage']
                cols = [c for c in cols if c in result]
                display(pd.DataFrame([{c: result.get(c) for c in cols}]))
            except Exception as e:
                display(_err(e))

    btn_ko.on_click(run_ko)
    tab_ko = widgets.VBox([
        _note('Para probar una llave hipotética, escribe dos equipos y la ronda.'),
        widgets.HBox([eq1, eq2]),
        widgets.HBox([ronda, sede, fecha_partido]),
        widgets.HBox([escenario, seed]),
        btn_ko,
        out_ko,
    ])

    # ------------------------------------------------------------
    # Pestaña 3: un Mundial completo
    # ------------------------------------------------------------
    btn_uno = widgets.Button(description='Simular 1 Mundial completo', button_style='success', layout=widgets.Layout(width='280px'))
    out_uno = widgets.Output()

    def run_uno(_):
        with out_uno:
            clear_output()
            try:
                from src.simulation.tournament_simulator import simulate_tournament_once
                display(_note('Simulando una edición completa. La primera ejecución puede tardar más porque carga modelos.'))
                t0 = time.time()
                out = simulate_tournament_once(
                    project_root=project_root,
                    analysis_date=fecha_corte.value,
                    use_registered_results=True,
                    seed=int(seed.value),
                    scenario=escenario.value,
                    corners_cards_mode='legacy',
                )
                display(_ok(f"Campeón: {out['champion']} · Tiempo: {time.time() - t0:.1f} s"))
                display(HTML(f"<p><b>Subcampeón:</b> {out['runner_up']} · <b>Tercer lugar:</b> {out['third_place']}</p>"))
                _display_bracket(out['bracket'])
            except Exception as e:
                display(_err(e))

    btn_uno.on_click(run_uno)
    tab_uno = widgets.VBox([
        _note('Simula una edición completa: grupos, mejores terceros, ronda de 32 y final.'),
        widgets.HBox([fecha_corte, escenario, seed]),
        btn_uno,
        out_uno,
    ])

    # ------------------------------------------------------------
    # Pestaña 4: probabilidades de campeón
    # ------------------------------------------------------------
    preset = widgets.Dropdown(options=[('Prueba — 10', 10), ('Rápido — 100', 100), ('Medio — 1,000', 1000), ('Reporte — 10,000', 10000)], value=100, description='Preset:', layout=widgets.Layout(width='280px'))
    n = widgets.IntText(value=100, description='N:', layout=widgets.Layout(width='160px'))
    btn_probs = widgets.Button(description='Calcular probabilidades', button_style='success', layout=widgets.Layout(width='260px'))
    out_probs = widgets.Output()

    def sync_preset(change):
        n.value = int(change['new'])

    preset.observe(sync_preset, names='value')

    def run_probs(_):
        with out_probs:
            clear_output()
            try:
                from src.simulation.tournament_simulator import run_tournament_monte_carlo
                display(_note(f'Corriendo {int(n.value):,} torneos. Empieza con 10 o 100 para validar.'))
                t0 = time.time()
                out = run_tournament_monte_carlo(
                    project_root=project_root,
                    n_simulations=int(n.value),
                    analysis_date=fecha_corte.value,
                    use_registered_results=True,
                    seed=int(seed.value),
                    scenario=escenario.value,
                    corners_cards_mode='legacy',
                )
                display(_ok(f"Listo · Tiempo: {time.time() - t0:.1f} s"))
                display(out['champion_probs'].head(25))
            except Exception as e:
                display(_err(e))

    btn_probs.on_click(run_probs)
    tab_probs = widgets.VBox([
        _note('Monte Carlo de torneos completos. Más simulaciones = más estabilidad, pero más tiempo.'),
        widgets.HBox([fecha_corte, escenario, seed]),
        widgets.HBox([preset, n]),
        btn_probs,
        out_probs,
    ])

    # ------------------------------------------------------------
    # Pestaña 5: comparar escenarios
    # ------------------------------------------------------------
    n_scen = widgets.IntText(value=100, description='N por escenario:', layout=widgets.Layout(width='240px'))
    btn_scen = widgets.Button(description='Comparar escenarios', button_style='success', layout=widgets.Layout(width='240px'))
    out_scen = widgets.Output()

    def run_scen(_):
        with out_scen:
            clear_output()
            try:
                from src.simulation.tournament_simulator import run_tournament_monte_carlo
                frames = []
                t0 = time.time()
                for sc in ['base', 'moderate', 'diaspora']:
                    display(_note(f'Corriendo escenario {sc} con {int(n_scen.value):,} torneos...'))
                    out = run_tournament_monte_carlo(
                        project_root=project_root,
                        n_simulations=int(n_scen.value),
                        analysis_date=fecha_corte.value,
                        use_registered_results=True,
                        seed=int(seed.value),
                        scenario=sc,
                        corners_cards_mode='legacy',
                    )
                    tmp = out['champion_probs'].copy()
                    tmp['scenario'] = sc
                    frames.append(tmp)
                df = pd.concat(frames, ignore_index=True)
                pivot = df.pivot_table(index='team', columns='scenario', values='champion_probability', fill_value=0.0)
                pivot['max_prob'] = pivot.max(axis=1)
                pivot = pivot.sort_values('max_prob', ascending=False).drop(columns=['max_prob'])
                display(_ok(f"Comparación lista · Tiempo: {time.time() - t0:.1f} s"))
                display(pivot.head(25))
            except Exception as e:
                display(_err(e))

    btn_scen.on_click(run_scen)
    tab_scen = widgets.VBox([
        _note('Compara base, moderate y diaspora. Empieza con N=10 o N=100 por escenario.'),
        widgets.HBox([fecha_corte, seed, n_scen]),
        btn_scen,
        out_scen,
    ])

    tabs = widgets.Tab(children=[tab_grupos, tab_ko, tab_uno, tab_probs, tab_scen])
    for i, title in enumerate(['Grupos', 'Eliminatoria', '1 Mundial', 'Campeón MC', 'Escenarios']):
        tabs.set_title(i, title)

    display(_header())
    display(tabs)