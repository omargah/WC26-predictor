# Simulador interactivo de escenarios

Notebook:

notebooks/Mundial_2026_Scenario_Simulator_Colab.ipynb

## Objetivo

Permite construir escenarios manuales de fase de grupos y recalcular automáticamente:

- tablas de grupo;
- mejores terceros;
- cruces de Round of 32;
- partidos de eliminación directa;
- campeón proyectado.

## Flujo de uso

1. Actualizar datos con update_worldcup_state.py --scope full.
2. Elegir una fecha pendiente.
3. Elegir un partido.
4. Definir marcador local y visitante.
5. Agregar el marcador al escenario.
6. Repetir para tantos partidos como se desee.
7. Ejecutar el escenario.
8. Revisar mejores terceros, Round of 32 y campeón proyectado.

## Archivos generados

Cada escenario se guarda en:

data/scenarios/<nombre_del_escenario>/

Archivos principales:

- manual_results_applied.csv
- scenario_group_standings.csv
- scenario_best_thirds.csv
- scenario_qualified.csv
- scenario_round_of_32.csv
- scenario_full_tournament_results.csv
- scenario_full_tournament_predictions.csv
- scenario_full_tournament_summary.json
- scenario_report.md

## Nota técnica

El simulador no modifica permanentemente los archivos oficiales del proyecto. Durante la ejecución reutiliza los scripts oficiales de tablas, cruces y torneo, pero restaura los archivos base al terminar.
