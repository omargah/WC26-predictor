# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.prediction.flexible_match_predictor import predecir_goles_flexible
from src.models.phase047_corners_cards_model import predecir_corners_tarjetas_experimental, imprimir_corners_tarjetas


def predecir_partido_completo_flexible(
    equipo_local: str,
    equipo_visitante: str,
    fecha_partido: str,
    torneo: str = 'FIFA World Cup',
    fase: str = 'Group Stage',
    ciudad: str = 'TBD',
    estadio: str = 'TBD',
    pais_sede: str = 'TBD',
    neutral: int = 1,
    project_root: str | Path = '.',
    verbose: bool = True,
    save: bool = True,
    candidate_dates: list[str] | None = None,
    corners_cards_mode: str = 'legacy',
):
    project_root = Path(project_root)

    goles = predecir_goles_flexible(
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        torneo=torneo,
        fase=fase,
        ciudad=ciudad,
        estadio=estadio,
        pais_sede=pais_sede,
        neutral=neutral,
        project_root=project_root,
        verbose=False,
        candidate_dates=candidate_dates,
    )

    prefer_joblib = corners_cards_mode in ['joblib', 'auto']

    corners_cards = predecir_corners_tarjetas_experimental(
        equipo_local=equipo_local,
        equipo_visitante=equipo_visitante,
        fecha_partido=fecha_partido,
        project_root=project_root,
        candidate_dates=candidate_dates,
        prefer_joblib=prefer_joblib,
    )

    if corners_cards_mode == 'joblib' and corners_cards.get('source') == 'legacy_benchmark_compatible':
        corners_cards = {
            'available': False,
            'source': 'joblib_requested_but_fallback_returned',
            'reason': 'Se pidió modo joblib, pero el módulo cayó a benchmark legacy.',
        }

    resultado = {
        **goles,
        'corners_cards_mode': corners_cards_mode,
        'corners_cards_available': bool(corners_cards.get('available', False)),
        'corners_cards_source': corners_cards.get('source'),
        'corners_cards_reason': corners_cards.get('reason'),
    }

    if corners_cards.get('available', False):
        resultado.update({
            'corners_home': corners_cards['corners_home'],
            'corners_away': corners_cards['corners_away'],
            'corners_total': corners_cards['corners_total'],
            'cards_home': corners_cards['cards_home'],
            'cards_away': corners_cards['cards_away'],
            'cards_total': corners_cards['cards_total'],
            'corners_markets': corners_cards['corners_markets'],
            'cards_markets': corners_cards['cards_markets'],
        })

    if verbose:
        print('=' * 90)
        print('PREDICCIÓN COMPLETA FLEXIBLE — GOLES + CÓRNERS + TARJETAS')
        print('=' * 90)
        print(f'{equipo_local} vs {equipo_visitante}')
        print(f'Fecha: {fecha_partido}')
        print(f'Torneo: {torneo}')
        print(f'Fase: {fase}')
        print(f'Ciudad: {ciudad}')
        print(f'Estadio: {estadio}')
        print(f'País sede: {pais_sede}')
        print(f'Neutral: {neutral}')
        print(f'Modo córners/tarjetas: {corners_cards_mode}')
        print(f'Fuente features goles: {resultado.get("feature_source")}')

        if resultado.get('feature_source') == 'future_feature_builder':
            print(f'Último historial local: {resultado.get("home_latest_date")}')
            print(f'Último historial visitante: {resultado.get("away_latest_date")}')

        print()
        print('-' * 90)
        print('GOLES')
        print('-' * 90)
        print(f'Goles esperados {equipo_local}: {resultado["lambda_home"]:.3f}')
        print(f'Goles esperados {equipo_visitante}: {resultado["lambda_away"]:.3f}')
        print(f'Total goles esperado: {resultado["lambda_total"]:.3f}')

        print()
        print('Probabilidades 1X2:')
        print(f'  Gana {equipo_local}: {resultado["prob_home"]:.2%}')
        print(f'  Empate: {resultado["prob_draw"]:.2%}')
        print(f'  Gana {equipo_visitante}: {resultado["prob_away"]:.2%}')

        print()
        print('Mercados goles:')
        print(f'  Over 1.5: {resultado["over_1_5"]:.2%}')
        print(f'  Under 1.5: {resultado["under_1_5"]:.2%}')
        print(f'  Over 2.5: {resultado["over_2_5"]:.2%}')
        print(f'  Under 2.5: {resultado["under_2_5"]:.2%}')
        print(f'  Over 3.5: {resultado["over_3_5"]:.2%}')
        print(f'  Under 3.5: {resultado["under_3_5"]:.2%}')

        print()
        print('Ambos anotan:')
        print(f'  BTTS Sí: {resultado["btts_yes"]:.2%}')
        print(f'  BTTS No: {resultado["btts_no"]:.2%}')

        print()
        print('Marcador más probable:')
        print(f'  {resultado["top_score"]} → {resultado["top_score_prob"]:.2%}')

        print()
        print('Top marcadores:')
        for score in resultado['top_10_scores']:
            print(f'  {score["score"]}: {score["prob"]:.2%}')

        imprimir_corners_tarjetas(equipo_local, equipo_visitante, corners_cards)

        print()
        print('-' * 90)
        print('NOTA')
        print('-' * 90)
        print('Si Fuente features goles = future_feature_builder, el partido no existía como fila exacta y se construyó desde el último historial disponible.')

    if save:
        out_path = project_root / 'data' / 'predictions' / 'manual_full_match_predictions_updated.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_row = resultado.copy()
        for col in ['top_10_scores', 'corners_markets', 'cards_markets']:
            if col in save_row:
                save_row[col] = json.dumps(save_row[col], ensure_ascii=False)
        df_save = pd.DataFrame([save_row])
        if out_path.exists():
            old = pd.read_csv(out_path)
            df_save = pd.concat([old, df_save], ignore_index=True)
        df_save.to_csv(out_path, index=False, encoding='utf-8')

    return resultado
