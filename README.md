# Mundial 2026 Predictor

Proyecto de modelación predictiva y simulación del Mundial 2026.

El sistema permite predecir partidos individuales, simular fase de grupos, calcular mejores terceros, construir cruces de Round of 32, simular eliminatorias y proyectar campeón.

## Funciones principales

- Pronóstico hecho con IA para partidos individuales.
- Goles esperados mediante modelo Poisson/Dixon-Coles.
- Probabilidades 1X2.
- Over/Under.
- BTTS.
- Marcadores exactos más probables.
- Simulación de fase de grupos.
- Cálculo de mejores terceros.
- Round of 32.
- Simulación de eliminatorias.
- Campeón, subcampeón, tercer y cuarto lugar proyectados.
- Escenarios manuales.
- Modo contrafactual para modificar partidos ya jugados.
- Comparación modelo vs realidad.
- Interfaz web con Streamlit.

## Ejecutar la app

Instalar dependencias:

    pip install -r requirements.txt

Ejecutar Streamlit:

    streamlit run app.py

## Actualizar datos y torneo

    python scripts/update_worldcup_state.py --scope full

## Simular escenario manual

Crear plantilla:

    python scripts/simulate_scenario.py --create-template data/scenarios/manual_results.csv

Ejecutar escenario:

    python scripts/simulate_scenario.py --input data/scenarios/manual_results.csv --scenario-name prueba --seed 42

Ejecutar escenario permitiendo modificar partidos ya jugados:

    python scripts/simulate_scenario.py --input data/scenarios/manual_results.csv --scenario-name contrafactual --seed 42 --allow-overwrite-played

## Notebooks

- notebooks/Mundial_2026_Predictor_Colab.ipynb
- notebooks/Mundial_2026_Scenario_Simulator_Colab.ipynb

## Documentación

- docs/STREAMLIT_APP.md
- docs/SCENARIO_SIMULATOR.md
- docs/COLAB_USAGE.md
- docs/DEPLOYMENT.md

## Despliegue online

La app está preparada para publicarse con GitHub y Streamlit Community Cloud.

Archivo principal:

    app.py

## Limitaciones

Este proyecto es una herramienta estadística y de simulación. No garantiza resultados deportivos reales.

Actualmente no integra oficialmente córners, tarjetas, tiros, posesión ni datos tácticos avanzados.

## Nota visual y legal

La app usa una identidad visual propia inspirada en el ambiente del Mundial 2026. No usa logos oficiales de FIFA ni pretende representar una app oficial.

## Autor

Omar Garcia.
