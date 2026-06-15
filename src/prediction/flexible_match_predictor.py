# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import math
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.utils.team_names import team_aliases, canonical_team


def _load_model_package(project_root: str | Path):
    path = Path(project_root) / 'models' / 'poisson_dc_base.joblib'
    if not path.exists():
        raise FileNotFoundError(f'No existe {path}')
    return joblib.load(path)


def _walk_objects(obj, path='root', max_depth=5):
    rows = [(path, obj)]
    if max_depth <= 0:
        return rows
    if isinstance(obj, dict):
        for k, v in obj.items():
            rows.extend(_walk_objects(v, f'{path}.{k}', max_depth - 1))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            rows.extend(_walk_objects(v, f'{path}[{i}]', max_depth - 1))
    return rows


def _find_feature_cols(package):
    names = ['feature_cols', 'features', 'selected_features', 'model_features', 'features_used', 'feature_names']
    if isinstance(package, dict):
        for name in names:
            if name in package and isinstance(package[name], (list, tuple)):
                if all(isinstance(x, str) for x in package[name]):
                    return list(package[name])
    for _, obj in _walk_objects(package):
        if hasattr(obj, 'feature_names_in_'):
            try:
                return list(obj.feature_names_in_)
            except Exception:
                pass
    raise ValueError('No pude identificar feature_cols dentro de poisson_dc_base.joblib')


def _find_scaler(package):
    if isinstance(package, dict):
        for key in ['scaler', 'x_scaler', 'standard_scaler', 'preprocessor']:
            if key in package and hasattr(package[key], 'transform'):
                return package[key]
    for path, obj in _walk_objects(package):
        p = path.lower()
        if hasattr(obj, 'transform') and any(t in p for t in ['scaler', 'preprocess', 'standard']):
            return obj
    return None


def _find_models(package):
    estimators = {}
    for path, obj in _walk_objects(package):
        if hasattr(obj, 'predict'):
            estimators[path] = obj

    home_model = None
    away_model = None
    home_path = None
    away_path = None

    for path, model in estimators.items():
        p = path.lower()
        if home_model is None and any(t in p for t in ['home', 'local', 'model_home', 'modelo_local']):
            if not any(t in p for t in ['away', 'visitante']):
                home_model = model
                home_path = path
        if away_model is None and any(t in p for t in ['away', 'visitante', 'model_away', 'modelo_visitante']):
            away_model = model
            away_path = path

    if home_model is None or away_model is None:
        keys = list(estimators.keys())
        if len(keys) >= 2:
            home_path = keys[0]
            away_path = keys[1]
            home_model = estimators[home_path]
            away_model = estimators[away_path]

    if home_model is None or away_model is None:
        raise ValueError(f'No pude identificar modelos local/visitante. Estimadores: {list(estimators.keys())}')

    return home_model, away_model, home_path, away_path


def _load_dataset(project_root: str | Path):
    path = Path(project_root) / 'data' / 'features' / 'modeling_dataset.parquet'
    if not path.exists():
        raise FileNotFoundError(f'No existe {path}')
    df = pd.read_parquet(path)
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    return df


def _find_team_latest_row(df: pd.DataFrame, team: str, before_date: str | None = None):
    aliases = team_aliases(team)
    mask = df['equipo_local'].isin(aliases) | df['equipo_visitante'].isin(aliases)
    sub = df[mask].copy()
    if before_date is not None:
        date = pd.to_datetime(before_date, errors='coerce')
        if not pd.isna(date):
            sub = sub[sub['fecha'] <= date]
    if len(sub) == 0:
        raise ValueError(f'No encontré historial para {team}. aliases={aliases}')
    sub = sub.sort_values('fecha')
    row = sub.iloc[-1].copy()
    side = 'L' if row['equipo_local'] in aliases else 'V'
    return row, side


def _team_snapshot(row: pd.Series, side: str):
    snap = {}
    suffix = '_' + side
    for col in row.index:
        if col.endswith(suffix):
            snap[col[:-2]] = row[col]

    if side == 'L':
        if 'elo_local_pre' in row.index:
            snap['elo_pre'] = row['elo_local_pre']
    else:
        if 'elo_visitante_pre' in row.index:
            snap['elo_pre'] = row['elo_visitante_pre']

    return snap


def _build_future_feature_row(df: pd.DataFrame, feature_cols: list[str], home_team: str, away_team: str, fecha_partido: str, neutral: int = 1):
    home_latest, home_side = _find_team_latest_row(df, home_team, before_date=fecha_partido)
    away_latest, away_side = _find_team_latest_row(df, away_team, before_date=fecha_partido)

    home_snap = _team_snapshot(home_latest, home_side)
    away_snap = _team_snapshot(away_latest, away_side)

    medians = df.select_dtypes(include=[np.number]).median(numeric_only=True)
    data = {}

    for f in feature_cols:
        value = np.nan

        if f == 'neutral':
            value = int(neutral)
        elif f in ['year', 'anio']:
            value = pd.to_datetime(fecha_partido).year
        elif f in ['month', 'mes']:
            value = pd.to_datetime(fecha_partido).month
        elif f.endswith('_L'):
            base = f[:-2]
            value = home_snap.get(base, np.nan)
        elif f.endswith('_V'):
            base = f[:-2]
            value = away_snap.get(base, np.nan)
        elif f == 'elo_local_pre':
            value = home_snap.get('elo_pre', np.nan)
        elif f == 'elo_visitante_pre':
            value = away_snap.get('elo_pre', np.nan)
        elif f == 'diff_elo_pre':
            value = home_snap.get('elo_pre', np.nan) - away_snap.get('elo_pre', np.nan)
        elif f.startswith('diff_'):
            base = f.replace('diff_', '', 1)
            hv = home_snap.get(base, np.nan)
            av = away_snap.get(base, np.nan)
            if pd.notna(hv) and pd.notna(av):
                value = hv - av

        if pd.isna(value):
            if f in medians.index:
                value = medians[f]
            else:
                value = 0.0

        data[f] = value

    X = pd.DataFrame([data])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    diagnostics = {
        'home_latest_date': str(home_latest['fecha'].date()),
        'away_latest_date': str(away_latest['fecha'].date()),
        'home_latest_side': home_side,
        'away_latest_side': away_side,
        'features_count': len(feature_cols),
    }

    return X, diagnostics


def _apply_scaler(scaler, X):
    if scaler is None:
        return X
    try:
        return scaler.transform(X)
    except Exception:
        return scaler.transform(X.values)


def _dc_factor(i, j, lh, la, rho):
    if i == 0 and j == 0:
        return max(0.0, 1.0 - lh * la * rho)
    if i == 0 and j == 1:
        return max(0.0, 1.0 + lh * rho)
    if i == 1 and j == 0:
        return max(0.0, 1.0 + la * rho)
    if i == 1 and j == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def _score_matrix(lh, la, rho=-0.075, max_goals=10):
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la) * _dc_factor(i, j, lh, la, rho)
            mat[i, j] = p
    s = mat.sum()
    if s > 0:
        mat = mat / s
    return mat


def _markets_from_matrix(mat):
    max_goals = mat.shape[0] - 1
    prob_home = float(np.tril(mat, -1).sum())
    prob_draw = float(np.trace(mat))
    prob_away = float(np.triu(mat, 1).sum())

    def over(line):
        total = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i + j > line:
                    total += mat[i, j]
        return float(total)

    btts_yes = float(mat[1:, 1:].sum())

    scores = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            scores.append({'score': f'{i}-{j}', 'prob': float(mat[i, j])})
    scores = sorted(scores, key=lambda x: x['prob'], reverse=True)

    return {
        'prob_home': prob_home,
        'prob_draw': prob_draw,
        'prob_away': prob_away,
        'over_1_5': over(1.5),
        'under_1_5': 1.0 - over(1.5),
        'over_2_5': over(2.5),
        'under_2_5': 1.0 - over(2.5),
        'over_3_5': over(3.5),
        'under_3_5': 1.0 - over(3.5),
        'btts_yes': btts_yes,
        'btts_no': 1.0 - btts_yes,
        'top_score': scores[0]['score'],
        'top_score_prob': scores[0]['prob'],
        'top_10_scores': scores[:10],
    }


def predecir_goles_futuro(
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
):
    project_root = Path(project_root)
    package = _load_model_package(project_root)
    df = _load_dataset(project_root)

    feature_cols = _find_feature_cols(package)
    scaler = _find_scaler(package)
    model_home, model_away, model_home_path, model_away_path = _find_models(package)

    X, diagnostics = _build_future_feature_row(
        df=df,
        feature_cols=feature_cols,
        home_team=equipo_local,
        away_team=equipo_visitante,
        fecha_partido=fecha_partido,
        neutral=neutral,
    )

    X_model = _apply_scaler(scaler, X)

    lh = float(model_home.predict(X_model)[0])
    la = float(model_away.predict(X_model)[0])

    lh = float(np.clip(lh, 0.05, 8.0))
    la = float(np.clip(la, 0.05, 8.0))

    rho = -0.075
    if isinstance(package, dict):
        rho = float(package.get('rho', package.get('dc_rho', rho)))

    mat = _score_matrix(lh, la, rho=rho, max_goals=10)
    markets = _markets_from_matrix(mat)

    result = {
        'home_team': equipo_local,
        'away_team': equipo_visitante,
        'fecha_partido': fecha_partido,
        'torneo': torneo,
        'fase': fase,
        'ciudad': ciudad,
        'estadio': estadio,
        'pais_sede': pais_sede,
        'neutral': neutral,
        'lambda_home': lh,
        'lambda_away': la,
        'lambda_total': lh + la,
        'feature_source': 'future_feature_builder',
        'model_home_path': model_home_path,
        'model_away_path': model_away_path,
        **diagnostics,
        **markets,
    }

    if verbose:
        print('=' * 90)
        print('PREDICCIÓN FLEXIBLE — FEATURES FUTURAS')
        print('=' * 90)
        print(f'{equipo_local} vs {equipo_visitante}')
        print(f'Fecha: {fecha_partido}')
        print(f'Fuente features: future_feature_builder')
        print(f'Último partido base {equipo_local}: {diagnostics.get("home_latest_date")}')
        print(f'Último partido base {equipo_visitante}: {diagnostics.get("away_latest_date")}')
        print()
        print('GOLES')
        print('-' * 90)
        print(f'Goles esperados {equipo_local}: {lh:.3f}')
        print(f'Goles esperados {equipo_visitante}: {la:.3f}')
        print(f'Total goles esperado: {lh + la:.3f}')
        print()
        print('Probabilidades 1X2:')
        print(f'  Gana {equipo_local}: {result["prob_home"]:.2%}')
        print(f'  Empate: {result["prob_draw"]:.2%}')
        print(f'  Gana {equipo_visitante}: {result["prob_away"]:.2%}')

    return result


def predecir_goles_flexible(
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
    candidate_dates: list[str] | None = None,
):
    try:
        from src.prediction.phase04_predict_match import predecir_partido_manual
        out = predecir_partido_manual(
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
            verbose=verbose,
            save=False,
            candidate_dates=candidate_dates,
        )
        out['feature_source'] = out.get('search_mode', 'exact_fixture_row')
        return out
    except Exception as e:
        msg = str(e)
        if 'No encontré fila de features' not in msg and 'No encontre fila de features' not in msg:
            raise
        return predecir_goles_futuro(
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
            verbose=verbose,
        )
