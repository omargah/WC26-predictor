
# -*- coding: utf-8 -*-
"""
Configuración central del proyecto Mundial 2026.

Este archivo concentra rutas, semillas aleatorias, parámetros del modelo
y supuestos metodológicos. La idea es que el resto del código importe
esta configuración en lugar de repetir valores manualmente.

Una buena configuración central evita errores silenciosos y facilita
auditorías futuras.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class ProjectConfig:
    """
    Configuración general del proyecto.

    Attributes
    ----------
    project_root:
        Carpeta raíz del repositorio.

    random_seed:
        Semilla aleatoria para que las simulaciones y los modelos sean
        reproducibles.

    n_match_simulations:
        Número de simulaciones Monte Carlo por partido.

    n_tournament_simulations:
        Número de simulaciones del torneo completo.

    max_goals_matrix:
        Máximo número de goles considerado al construir la matriz de
        marcador exacto.
    """

    project_root: Path
    random_seed: int = 42

    # --------------------------------------------------------
    # Parámetros de simulación
    # --------------------------------------------------------
    # Usamos 100,000 simulaciones por partido porque da estimaciones
    # suficientemente estables sin volver demasiado lento el Colab.
    n_match_simulations: int = 100_000

    # Para el torneo completo empezamos con 20,000 simulaciones.
    # Más adelante se puede subir si el tiempo de ejecución lo permite.
    n_tournament_simulations: int = 20_000

    # La matriz de marcador exacto considerará resultados de 0 a 10 goles.
    # Esto cubre prácticamente toda la masa de probabilidad en fútbol.
    max_goals_matrix: int = 10

    # --------------------------------------------------------
    # Parámetros Dixon-Coles
    # --------------------------------------------------------
    # rho negativo corrige la dependencia en marcadores bajos:
    # 0-0, 1-0, 0-1 y 1-1.
    dixon_coles_rho: float = -0.075

    # --------------------------------------------------------
    # Corrección mundialista de goles
    # --------------------------------------------------------
    # En torneos cortos de selecciones suele haber menos goles que en
    # ligas de clubes. Por eso aplicamos un factor moderado a las lambdas.
    worldcup_goal_factor: float = 0.92

    # --------------------------------------------------------
    # Parámetros de historial directo H2H
    # --------------------------------------------------------
    # El H2H puede aportar información, pero no debe dominar el modelo:
    # pocos partidos antiguos no pueden pesar más que la fuerza global
    # actual de las selecciones.
    max_h2h_weight: float = 0.20

    # Decaimiento temporal para H2H:
    # un partido reciente pesa más que uno de hace muchos años.
    h2h_decay: float = 0.35

    # --------------------------------------------------------
    # Tratamiento de localía/anfitrión
    # --------------------------------------------------------
    # Esta sección existe porque en simulaciones previas México apareció
    # sobreestimado en fases finales.
    #
    # La solución no será "bajar a México" manualmente, sino hacer que la
    # ventaja de anfitrión dependa correctamente de la fase y de la sede.
    host_advantage_group: float = 1.00
    host_advantage_round32: float = 0.60
    host_advantage_round16: float = 0.35
    host_advantage_after_round16: float = 0.00

    # --------------------------------------------------------
    # Tratamiento de altitud
    # --------------------------------------------------------
    # La altitud solo debe aplicarse si el partido se juega realmente
    # en una sede de altitud relevante. No debe convertirse en una
    # ventaja global del país durante todo el torneo.
    altitude_effect_cap: float = 0.06


def get_config(project_root: Union[str, Path]) -> ProjectConfig:
    """
    Construye la configuración del proyecto.

    Parameters
    ----------
    project_root:
        Ruta raíz del repositorio.

    Returns
    -------
    ProjectConfig
        Objeto inmutable con parámetros globales.
    """

    return ProjectConfig(project_root=Path(project_root))


def get_paths(config: ProjectConfig) -> dict:
    """
    Devuelve las rutas principales del proyecto.

    Mantener las rutas en una función evita escribir rutas manualmente
    dentro de los módulos de datos, features, modelos o simulación.
    """

    root = config.project_root

    return {
        "root": root,
        "src": root / "src",
        "data": root / "data",
        "manual": root / "data" / "manual",
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "features": root / "data" / "features",
        "predictions": root / "data" / "predictions",
        "models": root / "models",
        "reports": root / "reports",
        "docs": root / "docs",
    }
