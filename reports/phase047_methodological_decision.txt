# Decisión metodológica sobre córners y tarjetas

## Contexto

El proyecto recuperó el artefacto legacy:

models/corners_cards_poisson_stable.joblib

Este modelo sí puede cargarse y sí produce predicciones. Sin embargo, al compararlo contra las salidas del primer Colab se observó que no reproduce exactamente los valores legacy de córners/tarjetas.

## Evidencia técnica

El modelo de córners/tarjetas espera 183 features.

En el dataset básico actual faltan 71 features.

Muchas de las features faltantes corresponden precisamente a variables avanzadas de córners, tarjetas y xG, por ejemplo:

- corners_for_avg_5_L
- corners_against_avg_5_L
- yellow_cards_for_avg_5_L
- yellow_cards_against_avg_5_L
- xg_for_avg_10
- xg_against_avg_10

Además, en la prueba robusta los partidos benchmark fueron encontrados con modos de búsqueda basic_*, no advanced_*. Eso indica que el joblib está trabajando con una fila básica y que varias features avanzadas se están rellenando.

## Mayor diferencia observada

Caso con mayor diferencia de córners totales: USA_PAR_2026_GROUP_FULL

Diferencia observada: 2.383604

## Decisión

Para reproducir las salidas del primer Colab:

corners_cards_mode = "legacy"

Para explorar el modelo real recuperado:

corners_cards_mode = "joblib"

Para uso normal mientras no se reconstruya modeling_dataset_advanced.parquet completo:

corners_cards_mode = "legacy"

## Justificación

El modo legacy conserva la comparabilidad con el primer Colab y evita introducir cambios no explicados en córners/tarjetas.

El modo joblib queda disponible como experimental, pero no debe usarse como salida principal hasta reconstruir correctamente las features avanzadas.
