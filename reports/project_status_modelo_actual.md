# Estado actual del modelo Mundial 2026

Generado: `2026-06-25T11:26:50`

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
| Fase 5B — Monte Carlo técnico | TERMINADA | Estimar probabilidades de campeón y rondas con checkpoint. |
| Fase 6 — Consolidación final | EN PROCESO | Documentar, limpiar, empaquetar y dejar repo listo para portafolio. |

## 3. Inventario de archivos clave

| Archivo | Estado | Descripción |
|---|---:|---|
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/processed/matches_clean.parquet` | OK | Dataset histórico limpio de partidos internacionales. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/processed/worldcup_2026_fixture_clean.csv` | OK | Fixture limpio del Mundial 2026. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/features/modeling_dataset_all.parquet` | OK | Dataset de modelado con partidos jugados y pendientes. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/features/modeling_dataset_train.parquet` | OK | Dataset de entrenamiento sin fixtures futuros. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/features/modeling_dataset_pending.parquet` | OK | Dataset de partidos pendientes para predicción. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/models/poisson_dc_base.joblib` | OK | Modelo Poisson/Dixon-Coles entrenado. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/reports/phase03_metrics.json` | OK | Métricas de validación del modelo de goles. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/phase03_pending_predictions.csv` | OK | Predicciones de partidos pendientes. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/phase05_v2_group_matches_once_fixed.csv` | OK | Partidos de grupo simulados con resultados reales fijos. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/phase05_v2_round_of_32_fixed.csv` | OK | Cruces de Round of 32 del escenario fixed. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/phase05_v2_full_tournament_results_fixed.csv` | OK | Una simulación completa del torneo fixed. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/match_results.csv` | OK | Resultados KO del Monte Carlo original/técnico fixed. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/champion_probabilities.csv` | OK | Probabilidades de campeón del Monte Carlo original/técnico fixed. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/round_probabilities.csv` | OK | Probabilidades de avance por ronda del Monte Carlo original/técnico fixed. |
| `/Users/omargah/Documents/Workspace/mundial-2026-predictor/data/predictions/mc_original_v2_runs/fixed_original_seed42/metadata.json` | OK | Metadata de corrida Monte Carlo original/técnico fixed. |

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

- Simulaciones completadas: `100`

### Top campeones

| Equipo | Campeonatos | Probabilidad |
|---|---:|---:|
| Spain | 22 | 22.00% |
| Argentina | 11 | 11.00% |
| Brazil | 9 | 9.00% |
| Norway | 8 | 8.00% |
| France | 7 | 7.00% |
| Colombia | 6 | 6.00% |
| Germany | 5 | 5.00% |
| Switzerland | 5 | 5.00% |
| Portugal | 5 | 5.00% |
| Belgium | 4 | 4.00% |
| England | 4 | 4.00% |
| United States | 2 | 2.00% |
| Japan | 2 | 2.00% |
| Morocco | 2 | 2.00% |
| Ecuador | 2 | 2.00% |
| Mexico | 2 | 2.00% |
| Netherlands | 1 | 1.00% |
| Ivory Coast | 1 | 1.00% |
| Paraguay | 1 | 1.00% |
| Uruguay | 1 | 1.00% |

### Equipos clave

| Equipo | R16 | QF | SF | Final | Campeón |
|---|---:|---:|---:|---:|---:|
| Mexico | 61.00% | 36.00% | 16.00% | 7.00% | 2.00% |
| United States | 83.00% | 23.00% | 7.00% | 2.00% | 2.00% |
| Canada | 46.00% | 12.00% | 7.00% | 2.00% | 0.00% |
| Spain | 84.00% | 60.00% | 44.00% | 33.00% | 22.00% |
| Argentina | 76.00% | 53.00% | 38.00% | 18.00% | 11.00% |
| Brazil | 65.00% | 43.00% | 28.00% | 18.00% | 9.00% |
| France | 49.00% | 35.00% | 15.00% | 9.00% | 7.00% |
| Belgium | 92.00% | 73.00% | 24.00% | 12.00% | 4.00% |
| England | 77.00% | 47.00% | 21.00% | 7.00% | 4.00% |
| Colombia | 92.00% | 66.00% | 28.00% | 13.00% | 6.00% |

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