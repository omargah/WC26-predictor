
# FIFA World Cup 2026 Predictor

Modelo predictivo y simulador de torneo para la Copa Mundial FIFA 2026.

## Objetivo

Construir un sistema reproducible, explicable y auditable para:

- predecir partidos individuales;
- estimar probabilidades de mercados como 1X2, Over/Under y BTTS;
- simular el Mundial 2026 completo;
- auditar sesgos de sede, localía y bracket;
- generar reportes exportables en CSV, Excel y HTML.

## Filosofía del proyecto

Este proyecto prioriza tres principios:

1. Reproducibilidad: cada resultado debe poder regenerarse.
2. Explicabilidad: cada bloque debe explicar qué hace y por qué.
3. Auditoría: las probabilidades anómalas deben detectarse y justificarse.

## Arquitectura

Flujo general del proyecto:

data -> features -> models -> prediction -> simulation -> audits -> reports

## Componentes principales

- ETL de datos históricos de selecciones.
- Features temporales sin leakage.
- Features contextuales de Mundial 2026.
- Modelo Poisson/Dixon-Coles para goles.
- Capa de Machine Learning calibrada.
- Simulación Monte Carlo de partidos y torneo.
- Auditoría de ventaja de anfitrión.

## Caso especial: ventaja de anfitrión

En simulaciones previas, México apareció sobreestimado en fases finales.

Por eso este proyecto incluye una auditoría específica para revisar:

- si la variable es_sede_2026 se aplica correctamente;
- si los cruces eliminatorios son tratados como neutrales;
- si la altitud se aplica solo cuando corresponde;
- si la ventaja de anfitrión se acumula artificialmente.

## Estado

Proyecto en reconstrucción desde cero.
