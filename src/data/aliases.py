
# -*- coding: utf-8 -*-
"""
Normalización de nombres de selecciones.

Este módulo centraliza la lógica de alias. Su objetivo es que todas las
fuentes de datos usen un mismo nombre canónico para cada selección.

Ejemplo:
    "USA", "USMNT" y "Estados Unidos" deben convertirse en "United States".
"""

from __future__ import annotations

import pandas as pd


# Diccionario base de alias.
# La llave es una forma alternativa del nombre.
# El valor es el nombre canónico que usará todo el proyecto.
TEAM_ALIASES = {
    # Norteamérica
    "USA": "United States",
    "U.S.A.": "United States",
    "United States of America": "United States",
    "Estados Unidos": "United States",
    "USMNT": "United States",
    "Mexico": "Mexico",
    "México": "Mexico",
    "Canada": "Canada",
    "Canadá": "Canada",

    # Sudamérica
    "Argentina": "Argentina",
    "Brazil": "Brazil",
    "Brasil": "Brazil",
    "Uruguay": "Uruguay",
    "Colombia": "Colombia",
    "Ecuador": "Ecuador",
    "Chile": "Chile",
    "Peru": "Peru",
    "Perú": "Peru",
    "Paraguay": "Paraguay",
    "Venezuela": "Venezuela",
    "Bolivia": "Bolivia",

    # Europa
    "Germany": "Germany",
    "Alemania": "Germany",
    "Spain": "Spain",
    "España": "Spain",
    "France": "France",
    "Francia": "France",
    "England": "England",
    "Inglaterra": "England",
    "Portugal": "Portugal",
    "Netherlands": "Netherlands",
    "Países Bajos": "Netherlands",
    "Holland": "Netherlands",
    "Belgium": "Belgium",
    "Bélgica": "Belgium",
    "Italy": "Italy",
    "Italia": "Italy",
    "Croatia": "Croatia",
    "Croacia": "Croatia",
    "Switzerland": "Switzerland",
    "Suiza": "Switzerland",
    "Denmark": "Denmark",
    "Dinamarca": "Denmark",
    "Austria": "Austria",
    "Turkey": "Turkey",
    "Turquía": "Turkey",
    "Scotland": "Scotland",
    "Escocia": "Scotland",
    "Wales": "Wales",
    "Gales": "Wales",
    "Poland": "Poland",
    "Polonia": "Poland",
    "Serbia": "Serbia",
    "Ukraine": "Ukraine",
    "Ucrania": "Ukraine",

    # África
    "Morocco": "Morocco",
    "Marruecos": "Morocco",
    "Nigeria": "Nigeria",
    "Senegal": "Senegal",
    "Cameroon": "Cameroon",
    "Camerún": "Cameroon",
    "Egypt": "Egypt",
    "Egipto": "Egypt",
    "Ghana": "Ghana",
    "Ivory Coast": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Costa de Marfil": "Ivory Coast",
    "Tunisia": "Tunisia",
    "Túnez": "Tunisia",
    "Algeria": "Algeria",
    "Argelia": "Algeria",

    # Asia / Oceanía
    "Japan": "Japan",
    "Japón": "Japan",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Corea del Sur": "South Korea",
    "Australia": "Australia",
    "Iran": "Iran",
    "Irán": "Iran",
    "Saudi Arabia": "Saudi Arabia",
    "Arabia Saudita": "Saudi Arabia",
    "Qatar": "Qatar",
    "Iraq": "Iraq",
    "Irak": "Iraq",
    "New Zealand": "New Zealand",
    "Nueva Zelanda": "New Zealand",
}


def normalize_team_name(team_name: str) -> str:
    """
    Convierte un nombre de selección a su versión canónica.

    Parameters
    ----------
    team_name:
        Nombre original del equipo como aparece en una fuente de datos.

    Returns
    -------
    str
        Nombre canónico usado por el proyecto.
    """

    if pd.isna(team_name):
        return team_name

    clean_name = str(team_name).strip()

    # Primero buscamos coincidencia exacta.
    if clean_name in TEAM_ALIASES:
        return TEAM_ALIASES[clean_name]

    # Si no hay coincidencia exacta, intentamos una búsqueda sin distinguir
    # mayúsculas/minúsculas. Esto ayuda cuando una fuente usa "mexico"
    # y otra usa "Mexico".
    lower_lookup = {k.lower(): v for k, v in TEAM_ALIASES.items()}
    if clean_name.lower() in lower_lookup:
        return lower_lookup[clean_name.lower()]

    # Si no lo conocemos, lo devolvemos sin modificar.
    # Más adelante una validación nos dirá qué nombres faltan por mapear.
    return clean_name


def apply_team_aliases(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """
    Aplica normalización de nombres a las columnas de local y visitante.

    La función devuelve una copia para no modificar silenciosamente el
    DataFrame original.
    """

    df_out = df.copy()

    if home_col in df_out.columns:
        df_out[home_col] = df_out[home_col].apply(normalize_team_name)

    if away_col in df_out.columns:
        df_out[away_col] = df_out[away_col].apply(normalize_team_name)

    return df_out


def find_unknown_teams(
    df: pd.DataFrame,
    accepted_teams: set[str],
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> list[str]:
    """
    Detecta equipos que aparecen en los datos pero no están en la lista aceptada.

    Esto es una validación importante antes de entrenar: si aparece "USA"
    después de normalizar, significa que falta un alias o hay un error de datos.
    """

    teams = set()

    if home_col in df.columns:
        teams.update(df[home_col].dropna().unique())

    if away_col in df.columns:
        teams.update(df[away_col].dropna().unique())

    unknown = sorted(team for team in teams if team not in accepted_teams)

    return unknown
