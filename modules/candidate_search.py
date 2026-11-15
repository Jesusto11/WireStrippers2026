"""Capa de búsqueda de candidatos sobre los filtros topográficos del DEM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from modules.terrain import generar_candidatos_dem


def buscar_candidatos_topograficos(
    dem_path: str | Path,
    center_lat: float,
    center_lon: float,
    search_radius_px: int = 150,
    max_slope_deg: float = 5.0,
    max_roughness_m: float = 3.0,
    min_component_pixels: int = 9,
    min_spacing_m: float = 300.0,
    max_candidates: int = 20,
    elevation_to_m_factor: float = 1.0,
) -> pd.DataFrame:
    """Devuelve una tabla de parches candidatos que pasan el filtro topográfico.

    Esto es una preselección topográfica. No aplica todavía iluminación,
    comunicaciones, huella del módulo ni restricciones propias de un lander.
    """
    candidates: list[dict[str, Any]] = generar_candidatos_dem(
        dem_path=dem_path,
        lat=center_lat,
        lon=center_lon,
        radius_px=search_radius_px,
        max_slope_deg=max_slope_deg,
        max_roughness_m=max_roughness_m,
        min_component_pixels=min_component_pixels,
        min_spacing_m=min_spacing_m,
        max_candidates=max_candidates,
        elevation_to_m_factor=elevation_to_m_factor,
    )
    return pd.DataFrame(candidates)
