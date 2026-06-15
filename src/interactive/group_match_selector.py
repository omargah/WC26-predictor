# -*- coding: utf-8 -*-

# ============================================================
# Selector semi-automático de partidos de fase de grupos
# Mundial 2026
# ============================================================

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

from IPython.display import display, HTML, clear_output

try:
    import ipywidgets as widgets
except Exception:
    widgets = None


GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}


GROUP_FIXTURES = [
    {'group':'A','matchday':1,'date':'2026-06-11','home_team':'Mexico','away_team':'South Africa','city':'Mexico City','stadium':'Estadio Azteca','venue_country':'Mexico','neutral':0},
    {'group':'A','matchday':1,'date':'2026-06-11','home_team':'South Korea','away_team':'Czechia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'A','matchday':2,'date':'2026-06-18','home_team':'Czechia','away_team':'South Africa','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'A','matchday':2,'date':'2026-06-18','home_team':'Mexico','away_team':'South Korea','city':'TBD','stadium':'TBD','venue_country':'Mexico','neutral':0},
    {'group':'A','matchday':3,'date':'2026-06-24','home_team':'Czechia','away_team':'Mexico','city':'TBD','stadium':'TBD','venue_country':'Mexico','neutral':0},
    {'group':'A','matchday':3,'date':'2026-06-24','home_team':'South Africa','away_team':'South Korea','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'B','matchday':1,'date':'2026-06-12','home_team':'Canada','away_team':'Bosnia and Herzegovina','city':'Toronto','stadium':'Toronto Stadium','venue_country':'Canada','neutral':0},
    {'group':'B','matchday':1,'date':'2026-06-13','home_team':'Qatar','away_team':'Switzerland','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'B','matchday':2,'date':'2026-06-18','home_team':'Switzerland','away_team':'Bosnia and Herzegovina','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'B','matchday':2,'date':'2026-06-18','home_team':'Canada','away_team':'Qatar','city':'TBD','stadium':'TBD','venue_country':'Canada','neutral':0},
    {'group':'B','matchday':3,'date':'2026-06-24','home_team':'Switzerland','away_team':'Canada','city':'TBD','stadium':'TBD','venue_country':'Canada','neutral':0},
    {'group':'B','matchday':3,'date':'2026-06-24','home_team':'Bosnia and Herzegovina','away_team':'Qatar','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'C','matchday':1,'date':'2026-06-13','home_team':'Brazil','away_team':'Morocco','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'C','matchday':1,'date':'2026-06-13','home_team':'Haiti','away_team':'Scotland','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'C','matchday':2,'date':'2026-06-19','home_team':'Scotland','away_team':'Morocco','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'C','matchday':2,'date':'2026-06-19','home_team':'Brazil','away_team':'Haiti','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'C','matchday':3,'date':'2026-06-24','home_team':'Scotland','away_team':'Brazil','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'C','matchday':3,'date':'2026-06-24','home_team':'Morocco','away_team':'Haiti','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'D','matchday':1,'date':'2026-06-12','home_team':'United States','away_team':'Paraguay','city':'Los Angeles','stadium':'Los Angeles Stadium','venue_country':'United States','neutral':0},
    {'group':'D','matchday':1,'date':'2026-06-14','home_team':'Australia','away_team':'Turkey','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'D','matchday':2,'date':'2026-06-19','home_team':'United States','away_team':'Australia','city':'TBD','stadium':'TBD','venue_country':'United States','neutral':0},
    {'group':'D','matchday':2,'date':'2026-06-19','home_team':'Turkey','away_team':'Paraguay','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'D','matchday':3,'date':'2026-06-25','home_team':'Turkey','away_team':'United States','city':'TBD','stadium':'TBD','venue_country':'United States','neutral':0},
    {'group':'D','matchday':3,'date':'2026-06-25','home_team':'Paraguay','away_team':'Australia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'E','matchday':1,'date':'2026-06-14','home_team':'Germany','away_team':'Curacao','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'E','matchday':1,'date':'2026-06-14','home_team':'Ivory Coast','away_team':'Ecuador','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'E','matchday':2,'date':'2026-06-20','home_team':'Germany','away_team':'Ivory Coast','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'E','matchday':2,'date':'2026-06-20','home_team':'Ecuador','away_team':'Curacao','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'E','matchday':3,'date':'2026-06-25','home_team':'Ecuador','away_team':'Germany','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'E','matchday':3,'date':'2026-06-25','home_team':'Curacao','away_team':'Ivory Coast','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'F','matchday':1,'date':'2026-06-14','home_team':'Netherlands','away_team':'Japan','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'F','matchday':1,'date':'2026-06-14','home_team':'Sweden','away_team':'Tunisia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'F','matchday':2,'date':'2026-06-20','home_team':'Netherlands','away_team':'Sweden','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'F','matchday':2,'date':'2026-06-21','home_team':'Tunisia','away_team':'Japan','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'F','matchday':3,'date':'2026-06-25','home_team':'Tunisia','away_team':'Netherlands','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'F','matchday':3,'date':'2026-06-25','home_team':'Japan','away_team':'Sweden','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'G','matchday':1,'date':'2026-06-15','home_team':'Belgium','away_team':'Egypt','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'G','matchday':1,'date':'2026-06-15','home_team':'Iran','away_team':'New Zealand','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'G','matchday':2,'date':'2026-06-21','home_team':'Belgium','away_team':'Iran','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'G','matchday':2,'date':'2026-06-21','home_team':'New Zealand','away_team':'Egypt','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'G','matchday':3,'date':'2026-06-26','home_team':'New Zealand','away_team':'Belgium','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'G','matchday':3,'date':'2026-06-26','home_team':'Egypt','away_team':'Iran','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'H','matchday':1,'date':'2026-06-15','home_team':'Spain','away_team':'Cape Verde','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'H','matchday':1,'date':'2026-06-15','home_team':'Saudi Arabia','away_team':'Uruguay','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'H','matchday':2,'date':'2026-06-21','home_team':'Spain','away_team':'Saudi Arabia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'H','matchday':2,'date':'2026-06-21','home_team':'Uruguay','away_team':'Cape Verde','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'H','matchday':3,'date':'2026-06-26','home_team':'Uruguay','away_team':'Spain','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'H','matchday':3,'date':'2026-06-26','home_team':'Cape Verde','away_team':'Saudi Arabia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'I','matchday':1,'date':'2026-06-16','home_team':'France','away_team':'Senegal','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'I','matchday':1,'date':'2026-06-16','home_team':'Iraq','away_team':'Norway','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'I','matchday':2,'date':'2026-06-22','home_team':'France','away_team':'Iraq','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'I','matchday':2,'date':'2026-06-22','home_team':'Norway','away_team':'Senegal','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'I','matchday':3,'date':'2026-06-26','home_team':'Norway','away_team':'France','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'I','matchday':3,'date':'2026-06-26','home_team':'Senegal','away_team':'Iraq','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'J','matchday':1,'date':'2026-06-16','home_team':'Argentina','away_team':'Algeria','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'J','matchday':1,'date':'2026-06-17','home_team':'Austria','away_team':'Jordan','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'J','matchday':2,'date':'2026-06-22','home_team':'Argentina','away_team':'Austria','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'J','matchday':2,'date':'2026-06-22','home_team':'Jordan','away_team':'Algeria','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'J','matchday':3,'date':'2026-06-27','home_team':'Jordan','away_team':'Argentina','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'J','matchday':3,'date':'2026-06-27','home_team':'Algeria','away_team':'Austria','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'K','matchday':1,'date':'2026-06-17','home_team':'Portugal','away_team':'DR Congo','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'K','matchday':1,'date':'2026-06-17','home_team':'Uzbekistan','away_team':'Colombia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'K','matchday':2,'date':'2026-06-23','home_team':'Portugal','away_team':'Uzbekistan','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'K','matchday':2,'date':'2026-06-23','home_team':'Colombia','away_team':'DR Congo','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'K','matchday':3,'date':'2026-06-27','home_team':'Colombia','away_team':'Portugal','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'K','matchday':3,'date':'2026-06-27','home_team':'DR Congo','away_team':'Uzbekistan','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},

    {'group':'L','matchday':1,'date':'2026-06-17','home_team':'England','away_team':'Croatia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'L','matchday':1,'date':'2026-06-17','home_team':'Ghana','away_team':'Panama','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'L','matchday':2,'date':'2026-06-23','home_team':'England','away_team':'Ghana','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'L','matchday':2,'date':'2026-06-23','home_team':'Panama','away_team':'Croatia','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'L','matchday':3,'date':'2026-06-27','home_team':'Panama','away_team':'England','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
    {'group':'L','matchday':3,'date':'2026-06-27','home_team':'Croatia','away_team':'Ghana','city':'TBD','stadium':'TBD','venue_country':'TBD','neutral':1},
]


def get_group_fixtures() -> pd.DataFrame:
    df = pd.DataFrame(GROUP_FIXTURES).copy()
    df['fixture_id'] = range(len(df))
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['match_label'] = df.apply(
        lambda r: f"{r['home_team']} vs {r['away_team']} — {r['date'].date()}",
        axis=1,
    )
    return df


def empty_table(group: str) -> pd.DataFrame:
    teams = GROUPS[group]
    return pd.DataFrame({
        'Equipo': teams,
        'PJ': 0,
        'G': 0,
        'E': 0,
        'P': 0,
        'GF': 0,
        'GC': 0,
        'DG': 0,
        'Pts': 0,
    })


def apply_result_to_table(table: pd.DataFrame, home: str, away: str, gh: int, ga: int) -> pd.DataFrame:
    table = table.copy()
    for team, gf, gc in [(home, gh, ga), (away, ga, gh)]:
        idx = table.index[table['Equipo'] == team]
        if len(idx) == 0:
            continue
        i = idx[0]
        table.loc[i, 'PJ'] += 1
        table.loc[i, 'GF'] += int(gf)
        table.loc[i, 'GC'] += int(gc)
        table.loc[i, 'DG'] = table.loc[i, 'GF'] - table.loc[i, 'GC']
    if gh > ga:
        table.loc[table['Equipo'] == home, 'G'] += 1
        table.loc[table['Equipo'] == away, 'P'] += 1
        table.loc[table['Equipo'] == home, 'Pts'] += 3
    elif gh < ga:
        table.loc[table['Equipo'] == away, 'G'] += 1
        table.loc[table['Equipo'] == home, 'P'] += 1
        table.loc[table['Equipo'] == away, 'Pts'] += 3
    else:
        table.loc[table['Equipo'] == home, 'E'] += 1
        table.loc[table['Equipo'] == away, 'E'] += 1
        table.loc[table['Equipo'] == home, 'Pts'] += 1
        table.loc[table['Equipo'] == away, 'Pts'] += 1
    table = table.sort_values(['Pts', 'DG', 'GF', 'Equipo'], ascending=[False, False, False, True]).reset_index(drop=True)
    return table


def parse_score_text(text: str):
    value = str(text).strip()
    if value == '':
        return None
    value = value.replace(' ', '')
    for sep in ['-', ':', ',']:
        if sep in value:
            parts = value.split(sep)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                return int(parts[0]), int(parts[1])
    return None


def simulate_score_from_prediction(pred: dict, seed: int | None = None):
    rng = np.random.default_rng(seed)
    lh = float(pred.get('lambda_home', 1.1))
    la = float(pred.get('lambda_away', 1.1))
    gh = int(min(rng.poisson(max(lh, 0.05)), 9))
    ga = int(min(rng.poisson(max(la, 0.05)), 9))
    return gh, ga


def html_header():
    return HTML("""
    <div style='padding:18px;border-radius:18px;background:linear-gradient(135deg,#101828,#344054);color:white;margin-bottom:14px;'>
      <h2 style='margin:0;'>🌎 Mundial 2026 — Selector semi-automático</h2>
      <p style='margin:6px 0 0 0;color:#D0D5DD;'>Elige grupo, jornada y partido. Para jornadas 2 y 3 puedes capturar resultados previos o dejar vacío para simularlos.</p>
    </div>
    """)


def html_match_card(fixture: dict, pred: dict | None = None):
    home = fixture['home_team']
    away = fixture['away_team']
    group = fixture['group']
    md = fixture['matchday']
    date = fixture['date']
    city = fixture.get('city', 'TBD')
    stadium = fixture.get('stadium', 'TBD')
    country = fixture.get('venue_country', 'TBD')
    if pred is None:
        center = '<div style="font-size:18px;color:#667085;">Predicción pendiente</div>'
    else:
        center = f"""
        <div style='display:flex;gap:10px;justify-content:center;margin-top:12px;'>
          <div style='background:#ECFDF3;color:#027A48;padding:10px 14px;border-radius:12px;'>Gana {home}<br><b>{pred.get('prob_home',0):.1%}</b></div>
          <div style='background:#FFFAEB;color:#B54708;padding:10px 14px;border-radius:12px;'>Empate<br><b>{pred.get('prob_draw',0):.1%}</b></div>
          <div style='background:#EFF8FF;color:#175CD3;padding:10px 14px;border-radius:12px;'>Gana {away}<br><b>{pred.get('prob_away',0):.1%}</b></div>
        </div>
        <div style='margin-top:10px;color:#475467;'>λ: {pred.get('lambda_home',0):.2f} - {pred.get('lambda_away',0):.2f} · Marcador modal: <b>{pred.get('top_score','NA')}</b></div>
        """
    html = f"""
    <div style='border:1px solid #EAECF0;border-radius:18px;padding:18px;background:#FFFFFF;box-shadow:0 6px 18px rgba(16,24,40,.08);margin:12px 0;'>
      <div style='font-size:13px;color:#667085;'>Grupo {group} · Jornada {md} · {date.date()} · {city} · {stadium} · {country}</div>
      <div style='display:flex;align-items:center;justify-content:space-between;margin-top:12px;'>
        <div style='font-size:24px;font-weight:700;color:#101828;width:40%;text-align:right;'>{home}</div>
        <div style='font-size:20px;font-weight:700;color:#667085;width:20%;text-align:center;'>vs</div>
        <div style='font-size:24px;font-weight:700;color:#101828;width:40%;text-align:left;'>{away}</div>
      </div>
      {center}
    </div>
    """
    return HTML(html)


def display_standings(table: pd.DataFrame, title: str):
    styled = table.copy()
    html = styled.to_html(index=False)
    html = html.replace('<table border="1" class="dataframe">', '<table style="border-collapse:collapse;width:100%;font-size:14px;">')
    html = html.replace('<th>', '<th style="background:#101828;color:white;padding:8px;border:1px solid #EAECF0;text-align:center;">')
    html = html.replace('<td>', '<td style="padding:8px;border:1px solid #EAECF0;text-align:center;">')
    display(HTML(f"<h3 style='color:#101828;'>{title}</h3>" + html))


def get_prior_fixtures(group: str, matchday: int, selected_idx: int | None = None) -> pd.DataFrame:
    df = get_group_fixtures()
    prior = df[(df['group'] == group) & (df['matchday'] < matchday)].copy()
    return prior.sort_values(['matchday', 'date']).reset_index(drop=True)


def predict_match_textual(project_root: Path, fixture: dict, verbose: bool = True, corners_cards_mode: str = 'legacy'):
    from src.prediction.phase048_full_match_predictor import predecir_partido_completo
    candidate_dates = [str(pd.to_datetime(fixture['date']).date())]
    try:
        return predecir_partido_completo(
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
    except TypeError:
        from src.prediction.predict_recommended import predecir_partido_completo_recomendado
        return predecir_partido_completo_recomendado(
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
        )


def resolve_prior_results(project_root: Path, group: str, matchday: int, score_texts: dict, seed: int = 2026):
    prior = get_prior_fixtures(group, matchday)
    table = empty_table(group)
    rows = []
    for i, fixture_row in prior.iterrows():
        fixture = fixture_row.to_dict()
        key = f'prior_{i}'
        typed = parse_score_text(score_texts.get(key, ''))
        source = 'manual'
        if typed is None:
            pred_prev = predict_match_textual(project_root, fixture, verbose=False, corners_cards_mode='legacy')
            gh, ga = simulate_score_from_prediction(pred_prev, seed=seed + i)
            source = 'simulado'
        else:
            gh, ga = typed
        table = apply_result_to_table(table, fixture['home_team'], fixture['away_team'], gh, ga)
        rows.append({
            'Jornada': fixture['matchday'],
            'Partido': f"{fixture['home_team']} vs {fixture['away_team']}",
            'Resultado': f'{gh}-{ga}',
            'Fuente': source,
        })
    return table, pd.DataFrame(rows)


def build_static_prediction(project_root: str | Path, group: str, matchday: int, match_index: int, prior_scores: dict | None = None, seed: int = 2026, corners_cards_mode: str = 'legacy'):
    project_root = Path(project_root)
    df = get_group_fixtures()
    options = df[(df['group'] == group) & (df['matchday'] == matchday)].sort_values(['date', 'home_team']).reset_index(drop=True)
    if len(options) == 0:
        raise ValueError('No hay partidos para esa combinación de grupo y jornada.')
    fixture = options.iloc[int(match_index)].to_dict()
    prior_scores = prior_scores or {}
    table, used_results = resolve_prior_results(project_root, group, matchday, prior_scores, seed=seed)
    display(html_header())
    display(html_match_card(fixture, pred=None))
    if len(used_results) > 0:
        display(HTML('<h3 style="color:#101828;">Resultados previos usados</h3>'))
        display(used_results)
    display_standings(table, f'Tabla del Grupo {group} antes del partido seleccionado')
    pred = predict_match_textual(project_root, fixture, verbose=True, corners_cards_mode=corners_cards_mode)
    display(html_match_card(fixture, pred=pred))
    return {'fixture': fixture, 'prior_results': used_results, 'standings_before_match': table, 'prediction': pred}


def lanzar_selector_grupos(project_root: str | Path, corners_cards_mode: str = 'legacy'):
    if widgets is None:
        raise ImportError('ipywidgets no está disponible. Puedes usar build_static_prediction(...) como alternativa.')
    project_root = Path(project_root)
    df = get_group_fixtures()

    group_dd = widgets.Dropdown(options=sorted(GROUPS.keys()), value='A', description='Grupo:')
    matchday_dd = widgets.Dropdown(options=[1, 2, 3], value=1, description='Jornada:')
    match_dd = widgets.Dropdown(options=[], description='Partido:', layout=widgets.Layout(width='620px'))
    seed_box = widgets.IntText(value=2026, description='Seed sim.:')
    mode_dd = widgets.Dropdown(options=['legacy', 'joblib', 'auto'], value=corners_cards_mode, description='C/T modo:')
    prior_box = widgets.VBox([])
    button = widgets.Button(description='Generar predicción', button_style='success', icon='soccer-ball-o')
    output = widgets.Output()

    def refresh_matches(*args):
        subset = df[(df['group'] == group_dd.value) & (df['matchday'] == matchday_dd.value)].sort_values(['date', 'home_team']).reset_index(drop=True)
        opts = [(row['match_label'], i) for i, row in subset.iterrows()]
        match_dd.options = opts
        if opts:
            match_dd.value = opts[0][1]
        refresh_prior_widgets()

    def refresh_prior_widgets(*args):
        prior = get_prior_fixtures(group_dd.value, matchday_dd.value)
        children = []
        if len(prior) > 0:
            children.append(widgets.HTML('<b>Resultados anteriores del grupo</b><br><span style="color:#667085;">Formato: 2-1. Si dejas vacío, el programa simula ese resultado.</span>'))
            for i, row in prior.iterrows():
                label = f"J{row['matchday']} · {row['home_team']} vs {row['away_team']}"
                children.append(widgets.Text(value='', placeholder='vacío = simular', description=label, layout=widgets.Layout(width='760px')))
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
                build_static_prediction(
                    project_root=project_root,
                    group=group_dd.value,
                    matchday=matchday_dd.value,
                    match_index=match_dd.value,
                    prior_scores=prior_scores,
                    seed=int(seed_box.value),
                    corners_cards_mode=mode_dd.value,
                )
            except Exception as e:
                display(HTML(f"<div style='padding:14px;border-radius:12px;background:#FEF3F2;color:#B42318;'><b>Error:</b> {e}</div>"))

    group_dd.observe(refresh_matches, names='value')
    matchday_dd.observe(refresh_matches, names='value')
    button.on_click(on_click)
    refresh_matches()

    display(html_header())
    display(widgets.VBox([
        widgets.HBox([group_dd, matchday_dd, seed_box, mode_dd]),
        match_dd,
        prior_box,
        button,
        output,
    ]))
