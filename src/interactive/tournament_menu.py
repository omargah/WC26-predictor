# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import pandas as pd
from IPython.display import display, HTML, clear_output

try:
    import ipywidgets as widgets
except Exception:
    widgets = None


def html_menu_header():
    return HTML("<div style='padding:18px;border-radius:18px;background:#101828;color:white;margin-bottom:14px;'><h2 style='margin:0;'>🏆 Simulador Mundial 2026</h2><p style='margin:6px 0 0 0;color:#D0D5DD;'>Partidos, grupos, eliminatorias, campeón y escenarios de localía.</p></div>")


def display_bracket(bracket: pd.DataFrame):
    show = bracket.copy()
    cols = ['match_id', 'round', 'home_team', 'away_team', 'home_goals_90', 'away_goals_90', 'winner', 'method']
    cols = [c for c in cols if c in show.columns]
    display(HTML('<h3 style="color:#101828;">Llave simulada</h3>'))
    display(show[cols])


def lanzar_menu_simulador(project_root: str | Path):
    if widgets is None:
        raise ImportError('ipywidgets no está disponible.')

    project_root = Path(project_root)

    action = widgets.Dropdown(
        options=[
            ('Partido de fase de grupos', 'group_match'),
            ('Partido eliminatorio específico', 'knockout_match'),
            ('Simular una edición completa', 'one_tournament'),
            ('Probabilidades de campeón', 'champion_probs'),
            ('Comparar escenarios base / moderate / diaspora', 'compare_scenarios'),
        ],
        value='group_match',
        description='Acción:',
        layout=widgets.Layout(width='520px'),
    )

    analysis_date = widgets.Text(value=str(pd.Timestamp.today().date()), description='Fecha corte:')
    seed_box = widgets.IntText(value=2026, description='Seed:')
    n_box = widgets.IntText(value=1000, description='N torneos:')
    scenario_dd = widgets.Dropdown(options=['base', 'moderate', 'diaspora'], value='base', description='Escenario:')
    button = widgets.Button(description='Ejecutar', button_style='success')
    output = widgets.Output()

    # Eliminatoria específica.
    home_text = widgets.Text(value='Mexico', description='Local:')
    away_text = widgets.Text(value='France', description='Visitante:')
    round_dd = widgets.Dropdown(options=['Round of 32', 'Round of 16', 'Quarterfinal', 'Semifinal', 'Final'], value='Quarterfinal', description='Ronda:')
    venue_text = widgets.Text(value='United States', description='País sede:')
    date_text = widgets.Text(value='2026-07-09', description='Fecha:')

    def on_click(_):
        with output:
            clear_output()
            display(html_menu_header())
            try:
                if action.value == 'group_match':
                    from src.interactive.group_match_selector_v2 import lanzar_selector_grupos_v2
                    lanzar_selector_grupos_v2(project_root=project_root, corners_cards_mode='legacy')

                elif action.value == 'knockout_match':
                    from src.simulation.knockout_match_simulator import simulate_knockout_match
                    result = simulate_knockout_match(
                        home_team=home_text.value,
                        away_team=away_text.value,
                        match_date=date_text.value,
                        round_name=round_dd.value,
                        venue_country=venue_text.value,
                        project_root=project_root,
                        scenario=scenario_dd.value,
                        seed=int(seed_box.value),
                    )
                    display(pd.DataFrame([result]))

                elif action.value == 'one_tournament':
                    from src.simulation.tournament_simulator import simulate_tournament_once
                    out = simulate_tournament_once(
                        project_root=project_root,
                        analysis_date=analysis_date.value,
                        use_registered_results=True,
                        seed=int(seed_box.value),
                        scenario=scenario_dd.value,
                    )
                    display(HTML(f"<h2>Campeón simulado: {out['champion']}</h2><p>Subcampeón: {out['runner_up']} · Tercer lugar: {out['third_place']}</p>"))
                    display_bracket(out['bracket'])

                elif action.value == 'champion_probs':
                    from src.simulation.tournament_simulator import run_tournament_monte_carlo
                    out = run_tournament_monte_carlo(
                        project_root=project_root,
                        n_simulations=int(n_box.value),
                        analysis_date=analysis_date.value,
                        use_registered_results=True,
                        seed=int(seed_box.value),
                        scenario=scenario_dd.value,
                    )
                    display(HTML(f"<h3>Probabilidades de campeón — escenario {scenario_dd.value}</h3>"))
                    display(out['champion_probs'].head(20))

                elif action.value == 'compare_scenarios':
                    from src.simulation.tournament_simulator import run_tournament_monte_carlo
                    frames = []
                    for sc in ['base', 'moderate', 'diaspora']:
                        out = run_tournament_monte_carlo(
                            project_root=project_root,
                            n_simulations=int(n_box.value),
                            analysis_date=analysis_date.value,
                            use_registered_results=True,
                            seed=int(seed_box.value),
                            scenario=sc,
                        )
                        tmp = out['champion_probs'].copy()
                        tmp['scenario'] = sc
                        frames.append(tmp)
                    df = pd.concat(frames, ignore_index=True)
                    pivot = df.pivot_table(index='team', columns='scenario', values='champion_probability', fill_value=0.0)
                    pivot['max_prob'] = pivot.max(axis=1)
                    pivot = pivot.sort_values('max_prob', ascending=False).drop(columns=['max_prob'])
                    display(HTML('<h3>Comparación de escenarios</h3>'))
                    display(pivot.head(25))

            except Exception as e:
                display(HTML(f"<div style='padding:14px;border-radius:12px;background:#FEF3F2;color:#B42318;'><b>Error:</b> {e}</div>"))

    button.on_click(on_click)

    display(html_menu_header())
    display(widgets.VBox([
        action,
        widgets.HBox([analysis_date, seed_box, n_box, scenario_dd]),
        widgets.HTML('<b>Opciones para eliminatoria específica</b>'),
        widgets.HBox([home_text, away_text]),
        widgets.HBox([round_dd, venue_text, date_text]),
        button,
        output,
    ]))
