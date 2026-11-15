"""Altimetría lunar mediante el servicio ImageServer de NASA Moon Trek.

El módulo consulta el producto LRO LOLA DEM Global 128 ppd a través de su
operación ArcGIS ImageServer/identify. No requiere un GeoTIFF local.

La micromalla 5x5 es un muestreo puntual del servicio, no un DEM continuo ni
un análisis certificado de obstáculos o seguridad de aterrizaje.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np
import requests

LOLA_IDENTIFY_URL = (
    "https://trek.nasa.gov/moon/trekarcgis/rest/services/"
    "LRO_LOLA_DEM_Global_128ppd_v04/ImageServer/identify"
)
LUNAR_RADIUS_M = 1_737_400.0
LOLA_PIXELS_PER_DEGREE = 128
LOLA_STEP_DEG = 1.0 / LOLA_PIXELS_PER_DEGREE


@lru_cache(maxsize=10000)
def consultar_pixel_lola(
    latitud: float,
    longitud: float,
    identify_url: str = LOLA_IDENTIFY_URL,
) -> float | None:
    """Consulta una elevación puntual en el ImageServer de NASA Moon Trek.

    El valor del servicio se interpreta en metros, como en el ejemplo de
    consulta del producto LOLA 128 ppd. La respuesta se conserva en caché
    en memoria durante la vida del proceso para evitar consultas repetidas.
    Devuelve None cuando el servicio responde explícitamente NoData.
    """
    lat = float(latitud)
    lon = float(longitud)
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError("La latitud y longitud deben ser valores numéricos finitos.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError("La latitud lunar debe estar entre -90° y 90°.")

    params = {
        "geometry": f"{lon:.8f},{lat:.8f}",
        "geometryType": "esriGeometryPoint",
        "f": "json",
    }
    try:
        response = requests.get(
            identify_url,
            params=params,
            timeout=(8, 20),
            headers={"User-Agent": "ECLIPSE-Lunar-Site-Evaluator/1.0"},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"No fue posible consultar NASA Moon Trek: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("NASA Moon Trek devolvió una respuesta que no es JSON válido.") from exc

    if payload.get("error"):
        error = payload["error"]
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise RuntimeError(f"NASA Moon Trek respondió con un error: {message}")

    value = payload.get("value")
    if value is None or str(value).strip().lower() in {"", "nodata", "no data", "nan", "none"}:
        return None

    try:
        elevation_m = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"El servicio devolvió un valor de elevación no interpretable: {value!r}") from exc

    if not np.isfinite(elevation_m):
        return None
    return elevation_m


def consultar_altitud_lola(
    latitud: float,
    longitud: float,
    identify_url: str = LOLA_IDENTIFY_URL,
) -> float:
    """Obtiene la altitud de un punto, en metros, o genera un error explícito."""
    elevation = consultar_pixel_lola(latitud, longitud, identify_url)
    if elevation is None:
        raise ValueError(
            f"NASA LOLA no tiene un valor válido para latitud {latitud:.6f}°, "
            f"longitud {longitud:.6f}° (NoData)."
        )
    return float(elevation)


def analizar_dem_via_api(
    latitud_centro: float,
    longitud_centro: float,
    n: int = 5,
    identify_url: str = LOLA_IDENTIFY_URL,
    max_workers: int = 8,
    lunar_radius_m: float = LUNAR_RADIUS_M,
) -> dict[str, Any]:
    """Muestrea una micromalla impar NxN y estima pendiente local.

    El espaciamiento angular se toma de la resolución nominal de 128 ppd.
    Las distancias físicas se aproximan sobre una esfera lunar de radio
    1,737.4 km; el paso este-oeste se multiplica por cos(latitud).
    """
    lat0 = float(latitud_centro)
    lon0 = float(longitud_centro)
    if not -90.0 <= lat0 <= 90.0:
        raise ValueError("La latitud del centro debe estar entre -90° y 90°.")
    if not np.isfinite(lon0):
        raise ValueError("La longitud del centro debe ser un valor finito.")
    if n < 3 or n > 9 or n % 2 == 0:
        raise ValueError("El tamaño de la micromalla debe ser impar y estar entre 3 y 9.")
    if max_workers < 1:
        raise ValueError("El número de solicitudes paralelas debe ser al menos 1.")

    half = n // 2
    lats = [lat0 + (half - row) * LOLA_STEP_DEG for row in range(n)]
    lons = [lon0 + (col - half) * LOLA_STEP_DEG for col in range(n)]
    coordinates = [(lat, lon) for lat in lats for lon in lons]

    def _query(point: tuple[float, float]) -> float:
        lat, lon = point
        value = consultar_pixel_lola(lat, lon, identify_url)
        return np.nan if value is None else float(value)

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(coordinates))) as executor:
        values = list(executor.map(_query, coordinates))

    elevation = np.asarray(values, dtype=float).reshape((n, n))
    if not np.isfinite(elevation[half, half]):
        raise ValueError(
            "NASA LOLA no devolvió una elevación válida para el punto central. "
            "No se puede calcular ni representar la altitud de este sitio."
        )

    valid_values = elevation[np.isfinite(elevation)]
    if valid_values.size < max(5, int(n * n * 0.6)):
        raise ValueError(
            f"Cobertura insuficiente de NASA LOLA en la micromalla: "
            f"{valid_values.size} de {n*n} puntos válidos."
        )

    # Aproximación de distancias sobre una esfera lunar; filas de norte a sur.
    radius_m = float(lunar_radius_m)
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("El radio lunar de referencia debe ser positivo y finito.")
    dy_m = LOLA_STEP_DEG * np.pi / 180.0 * radius_m
    dx_m = dy_m * abs(np.cos(np.radians(lat0)))
    if dx_m < 0.1:
        raise ValueError(
            "La separación este-oeste de la micromalla es demasiado pequeña "
            "cerca del polo para estimar una pendiente estable con esta cuadrícula."
        )

    # np.gradient conserva NaN alrededor de celdas sin dato; no se interpolan.
    safe_elevation = elevation.copy()
    grad_y, grad_x = np.gradient(safe_elevation, dy_m, dx_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))

    center_slope = float(slope_deg[half, half])
    if not np.isfinite(center_slope):
        raise ValueError(
            "La elevación central existe, pero faltan muestras vecinas para "
            "estimar su pendiente local."
        )

    valid_slopes = slope_deg[np.isfinite(slope_deg)]
    min_elev = float(np.nanmin(elevation))
    max_elev = float(np.nanmax(elevation))
    return {
        "elevation": elevation,
        "slope_deg": slope_deg,
        "point_elevation": float(elevation[half, half]),
        "point_slope_deg": center_slope,
        "mean_slope_deg": float(np.nanmean(valid_slopes)),
        "max_slope_deg": float(np.nanmax(valid_slopes)),
        "relief_m": max_elev - min_elev,
        "elevation_std_m": float(np.nanstd(valid_values)),
        "fraction_slope_below_5deg_pct": float(np.mean(valid_slopes < 5.0) * 100.0),
        "valid_samples": int(valid_values.size),
        "total_samples": int(n * n),
        "grid_size": int(n),
        "sample_step_deg": float(LOLA_STEP_DEG),
        "resolution_y_m": float(dy_m),
        "resolution_x_m": float(dx_m),
        "latitudes": np.asarray(lats, dtype=float),
        "longitudes": np.asarray(lons, dtype=float),
        "extension_lons": [float(lons[0]), float(lons[-1])],
        "extension_lats": [float(lats[-1]), float(lats[0])],
        "source": "NASA Moon Trek · LRO LOLA DEM Global 128 ppd · ImageServer identify",
        "reference_radius_m": radius_m,
    }
