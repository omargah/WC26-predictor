# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd
from IPython.display import display, HTML


def pct(x):
    try:
        return f'{float(x):.2%}'
    except Exception:
        return 'NA'


def html_monte_carlo_summary(mc_result: dict):
    m = mc_result['metrics']
    home = m.get('home_team', 'Local')
    away = m.get('away_team', 'Visitante')
    n = int(m.get('n_simulations', 0))
    seed = int(m.get('seed', 0))

    cards = [
        ('Gana ' + home, pct(m.get('prob_home_mc')), pct(m.get('prob_home_mc_ci_low')) + ' – ' + pct(m.get('prob_home_mc_ci_high'))),
        ('Empate', pct(m.get('prob_draw_mc')), pct(m.get('prob_draw_mc_ci_low')) + ' – ' + pct(m.get('prob_draw_mc_ci_high'))),
        ('Gana ' + away, pct(m.get('prob_away_mc')), pct(m.get('prob_away_mc_ci_low')) + ' – ' + pct(m.get('prob_away_mc_ci_high'))),
        ('Over 2.5', pct(m.get('over_2_5_mc')), pct(m.get('over_2_5_mc_ci_low')) + ' – ' + pct(m.get('over_2_5_mc_ci_high'))),
        ('BTTS Sí', pct(m.get('btts_yes_mc')), pct(m.get('btts_yes_mc_ci_low')) + ' – ' + pct(m.get('btts_yes_mc_ci_high'))),
    ]

    card_html = ''
    for title, value, ci in cards:
        card_html += f"""
        <div style='background:white;border:1px solid #EAECF0;border-radius:14px;padding:14px;box-shadow:0 4px 12px rgba(16,24,40,.06);'>
          <div style='font-size:13px;color:#667085;'>{title}</div>
          <div style='font-size:24px;font-weight:800;color:#101828;margin-top:4px;'>{value}</div>
          <div style='font-size:12px;color:#98A2B3;margin-top:4px;'>IC aprox. 95%: {ci}</div>
        </div>
        """

    html = f"""
    <div style='margin:18px 0;padding:18px;border-radius:18px;background:#F8FAFC;border:1px solid #EAECF0;'>
      <h3 style='margin:0 0 8px 0;color:#101828;'>🎲 Simulación Monte Carlo</h3>
      <div style='color:#667085;margin-bottom:14px;'>
        {n:,} simulaciones · seed = {seed}.
        Las probabilidades analíticas siguen siendo la salida principal; Monte Carlo sirve como validación simulada y demostración probabilística.
      </div>
      <div style='display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px;'>
        {card_html}
      </div>
    </div>
    """
    return HTML(html)


def display_monte_carlo_result(mc_result: dict, top_n: int = 10):
    display(html_monte_carlo_summary(mc_result))

    score_counts = mc_result['score_counts'].head(top_n).copy()
    score_counts['prob_mc'] = score_counts['prob_mc'].map(lambda x: f'{x:.2%}')
    score_counts = score_counts.rename(columns={
        'score': 'Marcador',
        'count': 'Frecuencia',
        'prob_mc': 'Prob. MC',
    })

    display(HTML('<h3 style="color:#101828;">Marcadores más frecuentes por Monte Carlo</h3>'))
    display(score_counts)
