# Estado actual del modelo Mundial 2026

Generado: `2026-06-26T01:43:50`

## 1. Resumen ejecutivo

El proyecto ya cuenta con un pipeline funcional para datos históricos, features temporales sin leakage, modelo de goles Poisson/Dixon-Coles, predicción de partidos individuales, simulación de grupos y eliminatorias del Mundial 2026, y Monte Carlo técnico con checkpoint.

## 2. Estado por fase

| Fase | Estado | Objetivo |
|---|---:|---|
| Fase 1 — Datos | TERMINADA | Construir dataset limpio histórico + fixture Mundial 2026. |
| Fase 2 — Features | TERMINADA | Generar variables temporales sin leakage. |
| Fase 3 — Modelo de goles | TERMINADA | Entrenar Poisson/Dixon-Coles y validar 1X2/goles. |
| Fase 4 — Predictor de partidos | TERMINADA | Calcular lambdas, 1X2, marcadores, over/BTTS para partidos. |
| Fase 5 — Simulación torneo | TERMINADA | Simular grupos, mejores terceros, Annexe C y KO. |
| Fase 5B — Monte Carlo técnico | INCOMPLETA | Estimar probabilidades de campeón y rondas con checkpoint. |
| Fase 6 — Consolidación final | EN PROCESO | Documentar, limpiar, empaquetar y dejar repo listo para portafolio. |

## 3. Inventario de archivos clave

| Archivo | Estado | Descripción |
|---|---:|---|
| `/home/runner/work/WC26-predictor/WC26-predictor/data/processed/matches_clean.parquet` | OK | Dataset histórico limpio de partidos internacionales. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/processed/worldcup_2026_fixture_clean.csv` | OK | Fixture limpio del Mundial 2026. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/features/modeling_dataset_all.parquet` | OK | Dataset de modelado con partidos jugados y pendientes. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/features/modeling_dataset_train.parquet` | OK | Dataset de entrenamiento sin fixtures futuros. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/features/modeling_dataset_pending.parquet` | OK | Dataset de partidos pendientes para predicción. |
| `/home/runner/work/WC26-predictor/WC26-predictor/models/poisson_dc_base.joblib` | OK | Modelo Poisson/Dixon-Coles entrenado. |
| `/home/runner/work/WC26-predictor/WC26-predictor/reports/phase03_metrics.json` | OK | Métricas de validación del modelo de goles. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/phase03_pending_predictions.csv` | OK | Predicciones de partidos pendientes. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/phase05_v2_group_matches_once_fixed.csv` | OK | Partidos de grupo simulados con resultados reales fijos. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/phase05_v2_round_of_32_fixed.csv` | OK | Cruces de Round of 32 del escenario fixed. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/phase05_v2_full_tournament_results_fixed.csv` | OK | Una simulación completa del torneo fixed. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/match_results.csv` | FALTA | Resultados KO del Monte Carlo original/técnico fixed. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/champion_probabilities.csv` | FALTA | Probabilidades de campeón del Monte Carlo original/técnico fixed. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/round_probabilities.csv` | FALTA | Probabilidades de avance por ronda del Monte Carlo original/técnico fixed. |
| `/home/runner/work/WC26-predictor/WC26-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/metadata.json` | FALTA | Metadata de corrida Monte Carlo original/técnico fixed. |

## 4. Métricas del modelo de goles

- `n_validation`: `3170`
- `mae_goles_local`: `1.0209273089969804`
- `mae_goles_visitante`: `0.8494612569069722`
- `mae_total`: `1.8703885659039525`
- `rmse_goles_local`: `1.3504281583991347`
- `rmse_goles_visitante`: `1.1231039057601024`
- `accuracy_1x2`: `0.6091482649842271`
- `log_loss_1x2`: `0.8591363305861949`
- `lambda_local_promedio`: `1.628923649523608`
- `lambda_visitante_promedio`: `1.1148662050120495`
- `goles_local_promedio_real`: `1.6425867507886436`
- `goles_visitante_promedio_real`: `1.1425867507886436`

## 5. Predicciones pendientes

- Partidos pendientes predichos: `18`
- Fecha mínima: `2026-06-25`
- Fecha máxima: `2026-06-27`

## 6. Monte Carlo original/técnico fixed


## 7. Recomendaciones de cierre

- No integrar córners/tarjetas en esta versión base hasta tener datos confiables.
- Mantener Poisson/Dixon-Coles como núcleo oficial del modelo actual.
- Usar Monte Carlo original/técnico como referencia principal.
- Usar Fast Monte Carlo solo como exploración, no como resultado oficial.
- Construir un README formal y un reporte técnico antes del commit final.
- Crear un script final único para predicción de partidos individuales.
- Crear un script final único para simulación de torneo.

## 8. Próximo paso

Construir los scripts finales de uso: uno para predicción de partidos individuales y otro para simulación consolidada del torneo.