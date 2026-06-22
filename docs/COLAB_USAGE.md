# Uso del predictor desde Google Colab

Este proyecto incluye un notebook para ejecutar el predictor del Mundial 2026 desde la web, sin instalar nada localmente.

## Notebook

Archivo: notebooks/Mundial_2026_Predictor_Colab.ipynb

## Flujo de uso

1. Abrir el notebook en Google Colab.
2. Ejecutar la celda de clonación del repositorio.
3. Instalar dependencias.
4. Generar opciones de fechas y partidos.
5. Seleccionar una fecha o partido desde menús desplegables.
6. Ejecutar la predicción.
7. Revisar las salidas generadas.

## Salidas principales

Cada predicción genera una carpeta analítica con:

- match_summary.csv
- markets_long.csv
- scorelines_long.csv
- report.md
- raw_full.csv
- metadata.json

## Evitar errores de nombres

El notebook no exige escribir manualmente los países. Las opciones se leen desde archivos de referencia generados por el proyecto:

- data/reference/available_dates.csv
- data/reference/available_matches.csv
- data/reference/team_aliases.csv

Esto evita problemas como España / Spain / Espana, Cabo Verde / Cape Verde o Estados Unidos / USA / United States.

## Escenarios de torneo

- fixed: usa como fijos los partidos ya jugados que existen en los datos locales y simula los pendientes.
- resimulate: vuelve a simular también partidos ya jugados. Sirve como análisis contrafactual.

## Nota sobre actualización

El modelo no se actualiza en tiempo real automáticamente. Usa los datos locales disponibles al momento de correr el pipeline.

Para actualizar resultados reales nuevos, debe ejecutarse el pipeline de actualización del proyecto.
