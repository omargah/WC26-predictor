
# Benchmarks del primer Colab

Este archivo documenta los partidos que se usarán como pruebas de regresión
del modelo predictivo del Mundial 2026.

## Por qué existen estos benchmarks

En el primer Colab se obtuvieron predicciones razonables para varios partidos
del Mundial 2026. Al reconstruir el proyecto desde cero, no queremos perder
ese comportamiento.

Por eso guardamos las predicciones antiguas y luego las compararemos contra
la nueva versión del modelo.

## Partidos guardados

1. Mexico vs South Africa
2. Czechia vs South Korea
3. Canada vs Bosnia and Herzegovina
4. United States vs Paraguay

## Qué se compara

Para cada partido se compararán:

- goles esperados local;
- goles esperados visitante;
- probabilidades 1X2;
- Over/Under 1.5, 2.5 y 3.5;
- BTTS;
- marcador más probable;
- córners y tarjetas cuando existan.

## Uso

Estos benchmarks no se usan para entrenar el modelo.

Se usan para auditar si una nueva versión cambia demasiado respecto a una
versión anterior que ya había dado resultados plausibles.

## Regla práctica

Si una nueva versión produce probabilidades muy diferentes, debe explicarse
por qué cambió:

- cambio de datos;
- cambio de features;
- cambio de tratamiento de localía;
- cambio de filtro temporal;
- cambio del modelo;
- corrección de leakage;
- corrección de sesgo de anfitrión.

Si no hay explicación clara, el cambio debe revisarse antes de aceptarse.
