"""Herramientas de DEM lunar: recorte local, pendiente, rugosidad y candidatos.

IMPORTANTE: verificar el CRS, unidades, resolución y datum del DEM antes de
usar estos resultados en una evaluación de ingeniería. Las métricas de este
módulo son de cribado y no sustituyen un análisis de terreno certificado.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform
from scipy import ndimage

LUNAR_RADIUS_M = 1_737_400.0


def _lunar_geographic_crs() -> CRS:
    return CRS.from_proj4(f"+proj=longlat +R={LUNAR_RADIUS_M} +no_defs")


def _site_to_dataset_xy(dataset: rasterio.io.DatasetReader, lat: float, lon: float) -> tuple[float, float]:
    if dataset.crs is None:
        raise ValueError("El DEM no declara CRS. No es seguro inferir cómo transformar las coordenadas lunares.")
    if dataset.crs.is_geographic:
        return float(lon), float(lat)
    xs, ys = rio_transform(_lunar_geographic_crs(), dataset.crs, [float(lon)], [float(lat)])
    return float(xs[0]), float(ys[0])


def _dataset_xy_to_lunar(dataset: rasterio.io.DatasetReader, x: float, y: float) -> tuple[float, float]:
    if dataset.crs is None:
        raise ValueError("El DEM no declara CRS.")
    if dataset.crs.is_geographic:
        return float(y), float(x)
    lons, lats = rio_transform(dataset.crs, _lunar_geographic_crs(), [float(x)], [float(y)])
    return float(lats[0]), float(lons[0])


def _pixel_size_m(dataset: rasterio.io.DatasetReader, lat: float) -> tuple[float, float]:
    """Devuelve (resolución_y, resolución_x) aproximada en metros/píxel."""
    res_x, res_y = dataset.res
    if dataset.crs is None:
        raise ValueError("El DEM no declara CRS; no se pueden interpretar sus unidades.")
    if dataset.crs.is_geographic:
        radians_per_degree = np.pi / 180.0
        meters_per_degree = LUNAR_RADIUS_M * radians_per_degree
        dy = abs(res_y) * meters_per_degree
        dx = abs(res_x) * meters_per_degree * max(abs(np.cos(np.radians(lat))), 1e-6)
        return dy, dx

    try:
        _, meters_per_unit = dataset.crs.linear_units_factor
    except Exception:
        meters_per_unit = 1.0
    return abs(res_y) * meters_per_unit, abs(res_x) * meters_per_unit


def _terrain_derivatives(elevation: np.ndarray, dy_m: float, dx_m: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(elevation)
    safe = np.where(valid, elevation, np.nan)
    # np.gradient propaga NaN alrededor de nodata, lo cual es preferible a
    # inventar pendientes en zonas sin elevación válida.
    gy, gx = np.gradient(safe, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

    values = np.where(valid, elevation, 0.0)
    weights = valid.astype(float)
    count = ndimage.uniform_filter(weights, size=5, mode="nearest")
    mean = ndimage.uniform_filter(values, size=5, mode="nearest") / np.maximum(count, 1e-9)
    mean2 = ndimage.uniform_filter(values**2, size=5, mode="nearest") / np.maximum(count, 1e-9)
    roughness = np.sqrt(np.maximum(mean2 - mean**2, 0.0))
    roughness[count < 0.5] = np.nan
    return slope, roughness


def leer_altitud_punto_dem(
    dem_path: str | Path,
    lat: float,
    lon: float,
    elevation_to_m_factor: float = 1.0,
) -> float:
    """Lee la elevación de un único píxel del DEM en metros.

    Aplica la escala/offset declarados en el GeoTIFF y el factor vertical de
    dem_config.json. No carga el ráster global completo en memoria.
    """
    path = Path(dem_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el DEM: {path}")

    with rasterio.open(path) as dataset:
        x, y = _site_to_dataset_xy(dataset, lat, lon)
        row, col = dataset.index(x, y)
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            raise ValueError(
                f"La coordenada ({lat:.6f}, {lon:.6f}) queda fuera de la cobertura del DEM."
            )

        pixel = dataset.read(
            1, window=Window(col, row, 1, 1), masked=True
        )
        if pixel.size == 0 or np.ma.is_masked(pixel[0, 0]):
            raise ValueError(
                f"El DEM no contiene un dato válido para ({lat:.6f}, {lon:.6f})."
            )

        raw_value = float(pixel[0, 0])
        band_scale = float(dataset.scales[0]) if dataset.scales else 1.0
        band_offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
        elevation_m = (raw_value * band_scale + band_offset) * float(elevation_to_m_factor)

        if not np.isfinite(elevation_m):
            raise ValueError("El DEM devolvió una elevación no finita para el punto seleccionado.")
        return float(elevation_m)


def leer_dem_local(
    dem_path: str | Path,
    lat: float,
    lon: float,
    radius_px: int = 100,
    elevation_to_m_factor: float = 1.0,
) -> dict[str, Any]:
    """Lee solo una ventana local alrededor de una coordenada lunar."""
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el DEM: {path}")
    if radius_px < 8:
        raise ValueError("El radio de búsqueda en píxeles debe ser de al menos 8 para evaluar el entorno local.")

    with rasterio.open(path) as dataset:
        x, y = _site_to_dataset_xy(dataset, lat, lon)
        row, col = dataset.index(x, y)
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            raise ValueError("La coordenada seleccionada queda fuera de la cobertura del DEM.")

        requested = Window(col - radius_px, row - radius_px, 2 * radius_px + 1, 2 * radius_px + 1)
        full = Window(0, 0, dataset.width, dataset.height)
        window = requested.intersection(full)
        matrix = dataset.read(1, window=window, masked=True).astype("float64").filled(np.nan)
        band_scale = float(dataset.scales[0]) if dataset.scales else 1.0
        band_offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
        matrix = (matrix * band_scale + band_offset) * float(elevation_to_m_factor)
        transform = dataset.window_transform(window)
        local_row = row - int(window.row_off)
        local_col = col - int(window.col_off)
        dy_m, dx_m = _pixel_size_m(dataset, lat)
        slope, roughness = _terrain_derivatives(matrix, dy_m, dx_m)
        point_elevation = float(matrix[local_row, local_col])
        point_slope = float(slope[local_row, local_col])
        point_roughness = float(roughness[local_row, local_col])

        return {
            "elevation": matrix,
            "slope_deg": slope,
            "roughness_m": roughness,
            "point_elevation": point_elevation,
            "point_slope_deg": point_slope,
            "point_roughness_m": point_roughness,
            "mean_slope_deg": float(np.nanmean(slope)),
            "fraction_slope_below_5deg_pct": float(np.nanmean(slope < 5.0) * 100),
            "transform": transform,
            "window": window,
            "local_row": local_row,
            "local_col": local_col,
            "resolution_y_m": dy_m,
            "resolution_x_m": dx_m,
            "crs": str(dataset.crs),
            "bounds": tuple(dataset.bounds),
        }


def generar_candidatos_dem(
    dem_path: str | Path,
    lat: float,
    lon: float,
    radius_px: int = 150,
    max_slope_deg: float = 5.0,
    max_roughness_m: float = 3.0,
    min_component_pixels: int = 9,
    min_spacing_m: float = 300.0,
    max_candidates: int = 20,
    elevation_to_m_factor: float = 1.0,
) -> list[dict[str, float | int]]:
    """Busca parches conectados que cumplen filtros topográficos iniciales.

    Los candidatos son centros de parches de cribado topográfico. No se
    consideran todavía iluminación, comunicaciones, obstáculos subpíxel,
    dispersión real del vehículo, huella del tren de aterrizaje ni restricciones
    específicas de un módulo. Por tanto, no deben presentarse como sitios seguros.
    """
    path = Path(dem_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el DEM: {path}")

    with rasterio.open(path) as dataset:
        x0, y0 = _site_to_dataset_xy(dataset, lat, lon)
        row0, col0 = dataset.index(x0, y0)
        if not (0 <= row0 < dataset.height and 0 <= col0 < dataset.width):
            raise ValueError("El centro de búsqueda queda fuera de la cobertura del DEM.")
        req = Window(col0 - radius_px, row0 - radius_px, 2 * radius_px + 1, 2 * radius_px + 1)
        win = req.intersection(Window(0, 0, dataset.width, dataset.height))
        elevation = dataset.read(1, window=win, masked=True).astype("float64").filled(np.nan)
        band_scale = float(dataset.scales[0]) if dataset.scales else 1.0
        band_offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
        elevation = (elevation * band_scale + band_offset) * float(elevation_to_m_factor)
        win_transform = dataset.window_transform(win)
        dy_m, dx_m = _pixel_size_m(dataset, lat)
        slope, rough = _terrain_derivatives(elevation, dy_m, dx_m)

        suitable = (
            np.isfinite(elevation)
            & np.isfinite(slope)
            & np.isfinite(rough)
            & (slope <= float(max_slope_deg))
            & (rough <= float(max_roughness_m))
        )
        labels, count = ndimage.label(suitable)
        components: list[dict[str, float | int]] = []
        for label_id in range(1, count + 1):
            region = labels == label_id
            pixels = int(region.sum())
            if pixels < int(min_component_pixels):
                continue
            rows, cols = np.where(region)
            # Elegir el píxel más representativo cerca del centroide de cada parche,
            # ponderando pendiente y rugosidad bajas.
            centroid_r, centroid_c = float(rows.mean()), float(cols.mean())
            score = (slope[region] / max(max_slope_deg, 1e-6)) + (rough[region] / max(max_roughness_m, 1e-6))
            candidate_index = int(np.argmin(score))
            # Priorizar calidad topográfica; desempatar con cercanía al centroide.
            best_score = np.min(score)
            best_positions = np.where(score <= best_score + 1e-9)[0]
            if len(best_positions) > 1:
                chosen = min(best_positions, key=lambda k: (rows[k] - centroid_r) ** 2 + (cols[k] - centroid_c) ** 2)
            else:
                chosen = candidate_index
            r, c = int(rows[chosen]), int(cols[chosen])
            px, py = rasterio.transform.xy(win_transform, r, c, offset="center")
            cand_lat, cand_lon = _dataset_xy_to_lunar(dataset, px, py)
            components.append({
                "lat": float(cand_lat),
                "lon": float(cand_lon),
                "slope_deg": float(slope[r, c]),
                "roughness_m": float(rough[r, c]),
                "component_pixels": pixels,
                "area_km2_approx": float(pixels * dx_m * dy_m / 1e6),
            })

    # Ordenar por pendiente, rugosidad y tamaño del parche.
    components.sort(key=lambda item: (
        float(item["slope_deg"]),
        float(item["roughness_m"]),
        -int(item["component_pixels"]),
    ))
    accepted: list[dict[str, float | int]] = []
    for candidate in components:
        spaced = True
        for other in accepted:
            mean_lat = np.radians((float(candidate["lat"]) + float(other["lat"])) / 2)
            north_m = np.radians(float(candidate["lat"]) - float(other["lat"])) * LUNAR_RADIUS_M
            east_m = np.radians(float(candidate["lon"]) - float(other["lon"])) * LUNAR_RADIUS_M * np.cos(mean_lat)
            if np.hypot(north_m, east_m) < min_spacing_m:
                spaced = False
                break
        if spaced:
            candidate["candidate_id"] = len(accepted) + 1
            accepted.append(candidate)
        if len(accepted) >= max_candidates:
            break
    return accepted
