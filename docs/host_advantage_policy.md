# Política de localía para simulación del Mundial 2026

## Objetivo

Evitar que México, Canadá o Estados Unidos queden inflados artificialmente por una localía completa aplicada durante todo el torneo.

## Principio central

La localía se asigna por sede territorial real, no solamente por condición de anfitrión.

## Escenario base

| Ronda | Condición | Multiplicador |
|---|---|---:|
| Grupos | Anfitrión jugando en su país | 1.00 |
| 16avos | Anfitrión jugando en su país | 0.60 |
| Octavos | Anfitrión jugando en su país | 0.35 |
| Cuartos o posterior | Cualquier caso | 0.00 |

## México en Estados Unidos

En el escenario base, México jugando en Estados Unidos no recibe localía completa.

Puede existir un escenario de sensibilidad llamado `diaspora`, donde México recibe un apoyo pequeño en Estados Unidos, pero capado a 0.15.

## Estados Unidos en Estados Unidos

Estados Unidos recibe localía completa en grupos si juega en Estados Unidos.

En el escenario base, desde cuartos en adelante la simulación se neutraliza para evitar acumulación excesiva de ventaja.

## Canadá

Canadá recibe localía completa en grupos si juega en Canadá.

Si Canadá juega fuera de Canadá, no recibe localía completa.

## Escenarios

### base

Escenario recomendado para simulación principal.

### moderate

Escenario más generoso con anfitriones.

### diaspora

Escenario de sensibilidad donde se permite un apoyo pequeño por afición fuera del país, especialmente México en Estados Unidos.

## Decisión metodológica

La simulación principal usará `scenario='base'`.

Los escenarios `moderate` y `diaspora` se usarán solo para auditoría de sensibilidad.
