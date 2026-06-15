# Metodología del modelo

## 1. Idea general

El proyecto combina tres niveles de modelado:

1. Un modelo estadístico de goles basado en Poisson/Dixon-Coles.
2. Una capa de Machine Learning para capturar relaciones no lineales.
3. Una simulación Monte Carlo para estimar probabilidades agregadas de partidos y torneo.

## 2. Principio anti-leakage

Todas las variables históricas deben calcularse usando únicamente información disponible antes del partido que se quiere predecir.

Por ejemplo, para calcular la forma reciente de México antes de un partido del 2026, no se pueden usar partidos posteriores a esa fecha.

## 3. Features principales

Las features se agrupan en:

- forma reciente;
- fuerza ofensiva y defensiva;
- ranking/ELO;
- historial directo H2H;
- contexto del partido;
- sede, altitud y neutralidad;
- fase del torneo.

## 4. Ventaja de anfitrión

La ventaja de anfitrión se trata con cuidado.

En fases de grupos puede existir una ventaja real si la selección juega en su país.
En fases eliminatorias, especialmente después de octavos, se reduce o elimina para evitar inflar artificialmente a los anfitriones.

## 5. Auditoría de plausibilidad

Cada simulación completa del Mundial debe pasar por una auditoría:

- top de campeones probables;
- comparación con ranking/ELO;
- detección de outliers;
- sensibilidad a la ventaja de sede;
- sensibilidad a la altitud;
- sensibilidad al bracket.

El caso México será auditado explícitamente porque en simulaciones anteriores apareció sobreestimado.
