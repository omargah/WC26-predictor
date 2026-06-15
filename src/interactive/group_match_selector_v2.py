# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from IPython.display import display, HTML, clear_output

try:
    import ipywidgets as widgets
except Exception:
    widgets = None

from src.interactive.group_match_selector import (
    GROUPS,
    get_group_fixtures,
    empty_table,
    apply_result_to_table,
    parse_score_text,
    simulate_score_from_prediction,
    html_header,
    html_match_card,
    display_standings,
)
from src.utils.team_names import canonical_team, team_aliases


def predict_match_textual(project_root: Path, fixture: dict, verbose: bool = True, corners_cards_mode: str = 'legacy'):
    from src.prediction.flexible_full_predictor import predecir_partido_completo_flexible
    candidate_dates = [str(pd.to_datetime(fixture['date']).date())]
    return predecir_partido_completo_flexible(
        equipo_local=fixture['home_team'],
        equipo_visitante=fixture['away_team'],
        fecha_partido=str(pd.to_datetime(fixture['date']).date()),
        torneo='FIFA World Cup',
        fase='Group Stage',
        ciudad=fixture.get('city', 'TBD'),
        estadio=fixture.get('stadium', 'TBD'),
        pais_sede=fixture.get('venue_country', 'TBD'),
        neutral=int(fixture.get('neutral', 1)),
        project_root=project_root,
        verbose=verbose,
        save=True,
        candidate_dates=candidate_dates,
        corners_cards_mode=corners_cards_mode,
    )


def normalize_date(x):
    value = pd.to_datetime(x, errors='coerce')
    if pd.isna(value):
        return None
    return value.date()


def load_results_registry(project_root: str | Path) -> pd.DataFrame:
    path = Path(project_root) / 'data' / 'manual' / 'wc2026_group_results_registry.csv'
    if not path.exists():
        return pd.DataFrame(columns=['date','group','home_team','away_team','home_goals','away_goals','status'])
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df


def find_registered_result(project_root: str | Path, fixture: dict, analysis_date: str | None = None):
    df = load_results_registry(project_root)
    if len(df) == 0:
        return None

    fixture_date = normalize_date(fixture['date'])
    cutoff = normalize_date(analysis_date) if analysis_date is not None else None

    if cutoff is not None and fixture_date is not None and fixture_date > cutoff:
        return None

    home_aliases = set(team_aliases(fixture['home_team']))
    away_aliases = set(team_aliases(fixture['away_team']))

    candidates = df[df['status'].astype(str).str.lower().eq('complete')].copy()
    candidates = candidates[candidates['date'].dt.date == fixture_date]

    for _, row in candidates.iterrows():
        rh = str(row['home_team']).strip()
        ra = str(row['away_team']).strip()

        direct = rh in home_aliases and ra in away_aliases
        reverse = rh in away_aliases and ra in home_aliases

        if direct:
            return int(row['home_goals']), int(row['away_goals']), 'registrado'

        if reverse:
            return int(row['away_goals']), int(row['home_goals']), 'registrado_invertido'

    return None


def date_label(fixture: dict, analysis_date: str | None):
    fixture_date = normalize_date(fixture['date'])
    cutoff = normalize_date(analysis_date)

    if fixture_date is None or cutoff is None:
        return 'fecha_sin_clasificar'

    if fixture_date < cutoff:
        return 'pasado'
    if fixture_date == cutoff:
        return 'hoy_fecha_corte'
    return 'futuro'


def get_prior_fixtures(group: str, matchday: int) -> pd.DataFrame:
    df = get_group_fixtures()
    prior = df[(df['group'] == group) & (df['matchday'] < matchday)].copy()
    return prior.sort_values(['matchday', 'fixture_id']).reset_index(drop=True)


def resolve_prior_results_auto(project_root: Path, group: str, matchday: int, score_texts: dict, analysis_date: str, seed: int = 2026):
    prior = get_prior_fixtures(group, matchday)
    table = empty_table(group)
    rows = []

    for i, fixture_row in prior.iterrows():
        fixture = fixture_row.to_dict()
        key = f'prior_{i}'
        typed = parse_score_text(score_texts.get(key, ''))
        label = date_label(fixture, analysis_date)

        if typed is not None:
            gh, ga = typed
            source = 'manual_usuario'
        else:
            registered = find_registered_result(project_root, fixture, analysis_date=analysis_date)
            if registered is not None:
                gh, ga, source = registered
            else:
                pred_prev = predict_match_textual(project_root, fixture, verbose=False, corners_cards_mode='legacy')
                gh, ga = simulate_score_from_prediction(pred_prev, seed=seed + i)
                if label == 'futuro':
                    source = 'simulado_futuro'
                else:
                    source = 'simulado_sin_marcador_registrado'

        table = apply_result_to_table(table, fixture['home_team'], fixture['away_team'], gh, ga)
        rows.append({
            'Jornada': fixture['matchday'],
            'Fecha': str(pd.to_datetime(fixture['date']).date()),
            'Partido': f"{fixture['home_team']} vs {fixture['away_team']}",
            'Resultado': f'{gh}-{ga}',
            'Etiqueta_fecha': label,
            'Fuente': source,
        })

    return table, pd.DataFrame(rows)


def html_date_badge(fixture: dict, analysis_date: str | None):
    label = date_label(fixture, analysis_date)
    if label == 'futuro':
        color = '#027A48'
        bg = '#ECFDF3'
        text = 'Pronóstico futuro'
    elif label == 'hoy_fecha_corte':
        color = '#175CD3'
        bg = '#EFF8FF'
        text = 'Partido en fecha de corte'
    else:
        color = '#B54708'
        bg = '#FFFAEB'
        text = 'Demostración histórica / partido pasado'

    return HTML(f"<div style='display:inline-block;background:{bg};color:{color};padding:8px 12px;border-radius:999px;font-weight:700;margin:8px 0;'>{text}</div>")


def build_static_prediction_v2(project_root: str | Path, group: str, matchday: int, match_index: int, prior_scores: dict | None = None, analysis_date: str | None = None, seed: int = 2026, corners_cards_mode: str = 'legacy', monte_carlo: bool = True, n_simulations: int = 20000):
    project_root = Path(project_root)
    analysis_date = analysis_date or str(pd.Timestamp.today().date())
    df = get_group_fixtures()
    options = df[(df['group'] == group) & (df['matchday'] == matchday)].sort_values(['fixture_id']).reset_index(drop=True)
    if len(options) == 0:
        raise ValueError('No hay partidos para esa combinación de grupo y jornada.')

    fixture = options.iloc[int(match_index)].to_dict()
    prior_scores = prior_scores or {}
    table, used_results = resolve_prior_results_auto(project_root, group, matchday, prior_scores, analysis_date=analysis_date, seed=seed)

    display(html_header())
    display(html_date_badge(fixture, analysis_date))
    display(html_match_card(fixture, pred=None))

    if len(used_results) > 0:
        display(HTML('<h3 style="color:#101828;">Resultados previos usados</h3>'))
        display(used_results)

    display_standings(table, f'Tabla del Grupo {group} antes del partido seleccionado')

    pred = predict_match_textual(project_root, fixture, verbose=True, corners_cards_mode=corners_cards_mode)
    display(html_match_card(fixture, pred=pred))

    mc_result = None
    if monte_carlo:
        try:
            from src.simulation.match_monte_carlo import simulate_match_monte_carlo_from_prediction
            from src.interactive.monte_carlo_view import display_monte_carlo_result
            mc_result = simulate_match_monte_carlo_from_prediction(
                pred=pred,
                n_simulations=int(n_simulations),
                seed=int(seed),
            )
            display_monte_carlo_result(mc_result, top_n=10)
        except Exception as e:
            display(HTML(f"<div style='padding:14px;border-radius:12px;background:#FFFAEB;color:#B54708;'><b>Monte Carlo no disponible:</b> {e}</div>"))

    return {
        'fixture': fixture,
        'analysis_date': analysis_date,
        'date_label': date_label(fixture, analysis_date),
        'prior_results': used_results,
        'standings_before_match': table,
        'prediction': pred,
        'monte_carlo': mc_result,
    }


def lanzar_selector_grupos_v2(project_root: str | Path, corners_cards_mode: str = 'legacy'):
    if widgets is None:
        raise ImportError('ipywidgets no está disponible. Usa build_static_prediction_v2(...) como alternativa.')

    project_root = Path(project_root)
    df = get_group_fixtures()

    group_dd = widgets.Dropdown(options=sorted(GROUPS.keys()), value='A', description='Grupo:')
    matchday_dd = widgets.Dropdown(options=[1, 2, 3], value=1, description='Jornada:')
    match_dd = widgets.Dropdown(options=[], description='Partido:', layout=widgets.Layout(width='700px'))
    analysis_date_text = widgets.Text(value=str(pd.Timestamp.today().date()), description='Fecha corte:', layout=widgets.Layout(width='260px'))
    seed_box = widgets.IntText(value=2026, description='Seed sim.:')
    mode_dd = widgets.Dropdown(options=['legacy', 'joblib', 'auto'], value=corners_cards_mode, description='C/T modo:')
    mc_check = widgets.Checkbox(value=True, description='Monte Carlo')
    n_sim_box = widgets.IntText(value=20000, description='N sim.:')
    prior_box = widgets.VBox([])
    button = widgets.Button(description='Generar predicción', button_style='success')
    output = widgets.Output()

    def refresh_matches(*args):
        subset = df[(df['group'] == group_dd.value) & (df['matchday'] == matchday_dd.value)].sort_values(['fixture_id']).reset_index(drop=True)
        opts = [(row['match_label'], i) for i, row in subset.iterrows()]
        match_dd.options = opts
        if opts:
            match_dd.value = opts[0][1]
        refresh_prior_widgets()

    def refresh_prior_widgets(*args):
        prior = get_prior_fixtures(group_dd.value, matchday_dd.value)
        children = []
        if len(prior) > 0:
            children.append(widgets.HTML('<b>Resultados anteriores del grupo</b><br><span style="color:#667085;">Si existe marcador registrado antes de la fecha de corte, se usa automático. Si escribes un resultado, lo sobrescribes. Si dejas vacío y no hay registro, se simula.</span>'))
            for i, row in prior.iterrows():
                label = f"J{row['matchday']} · {row['home_team']} vs {row['away_team']}"
                children.append(widgets.Text(value='', placeholder='vacío = automático', description=label, layout=widgets.Layout(width='850px')))
        else:
            children.append(widgets.HTML('<span style="color:#667085;">Jornada 1: no se requieren resultados previos.</span>'))
        prior_box.children = children

    def on_click(_):
        with output:
            clear_output()
            prior_scores = {}
            if matchday_dd.value > 1:
                text_widgets = [w for w in prior_box.children if isinstance(w, widgets.Text)]
                for i, w in enumerate(text_widgets):
                    prior_scores[f'prior_{i}'] = w.value
            try:
                build_static_prediction_v2(
                    project_root=project_root,
                    group=group_dd.value,
                    matchday=matchday_dd.value,
                    match_index=match_dd.value,
                    prior_scores=prior_scores,
                    analysis_date=analysis_date_text.value,
                    seed=int(seed_box.value),
                    corners_cards_mode=mode_dd.value,
                    monte_carlo=bool(mc_check.value),
                    n_simulations=int(n_sim_box.value),
                )
            except Exception as e:
                display(HTML(f"<div style='padding:14px;border-radius:12px;background:#FEF3F2;color:#B42318;'><b>Error:</b> {e}</div>"))

    group_dd.observe(refresh_matches, names='value')
    matchday_dd.observe(refresh_matches, names='value')
    button.on_click(on_click)
    refresh_matches()

    display(html_header())
    display(widgets.VBox([
        widgets.HBox([group_dd, matchday_dd, analysis_date_text]),
        widgets.HBox([seed_box, mode_dd, mc_check, n_sim_box]),
        match_dd,
        prior_box,
        button,
        output,
    ]))