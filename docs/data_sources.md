
# Fuentes de datos

## 1. Resultados históricos internacionales

Fuente base usada:

- KaggleHub dataset: martj42/international-football-results-from-1872-to-2017
- Archivo principal: results.csv

Este archivo contiene partidos internacionales históricos con columnas como:

- fecha;
- equipo local;
- equipo visitante;
- goles local;
- goles visitante;
- torneo;
- ciudad;
- país;
- neutralidad.

## Por qué usamos esta fuente como base

Se usa como punto de partida porque permite reconstruir el pipeline desde cero,
sin depender de archivos procesados de versiones anteriores.

La idea es que cualquier persona que clone el repo pueda entender el flujo:

1. descargar datos crudos;
2. limpiarlos;
3. generar features;
4. entrenar modelos;
5. simular el Mundial.

## Limitaciones

Esta fuente no contiene por sí sola:

- xG;
- tiros;
- posesión;
- córners;
- tarjetas;
- valor de plantilla;
- ranking FIFA actualizado;
- fixture definitivo del Mundial 2026.

Esas variables se integrarán después como capas adicionales. El modelo base
debe funcionar primero con resultados históricos limpios.
