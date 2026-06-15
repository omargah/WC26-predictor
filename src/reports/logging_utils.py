
# -*- coding: utf-8 -*-
"""
Utilidades de logging para notebooks y scripts.

Estas funciones no hacen cálculos del modelo. Su objetivo es mejorar
la legibilidad de la ejecución en Colab y en consola.
"""


def print_section(title: str) -> None:
    """
    Imprime un encabezado grande para separar secciones principales.
    """

    line = "=" * 80
    print(f"\n{line}")
    print(title.upper())
    print(line)


def print_subsection(title: str) -> None:
    """
    Imprime un encabezado intermedio.
    """

    line = "-" * 80
    print(f"\n{line}")
    print(title)
    print(line)


def print_step(message: str) -> None:
    """
    Imprime un paso puntual dentro de una sección.
    """

    print(f"  → {message}")


def print_success(message: str) -> None:
    """
    Imprime un mensaje de éxito.
    """

    print(f"  ✅ {message}")


def print_warning(message: str) -> None:
    """
    Imprime una advertencia metodológica o de datos.
    """

    print(f"  ⚠️  {message}")


def print_error(message: str) -> None:
    """
    Imprime un mensaje de error.
    """

    print(f"  ❌ {message}")
