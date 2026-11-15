"""Métricas temporales y etiquetas exploratorias para análisis de misión."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def agregar_duracion_muestras(df: pd.DataFrame, fecha_fin: datetime | str) -> pd.DataFrame:
    """Asigna a cada muestra el tiempo hasta la siguiente muestra.

    La última muestra se extiende hasta el fin solicitado. Esto permite
    integrar horas aunque el último timestamp de Horizons no coincida con
    exactamente la hora final solicitada.
    """
    if df.empty:
        result = df.copy()
        result["duration_h"] = pd.Series(dtype=float)
        return result

    result = df.copy().sort_values("datetime").reset_index(drop=True)
    end = pd.Timestamp(fecha_fin)
    next_time = result["datetime"].shift(-1)
    next_time.iloc[-1] = end
    result["duration_h"] = (
        (next_time - result["datetime"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    # No contar muestras posteriores al final solicitado.
    before_end = result["datetime"] < end
    result.loc[~before_end, "duration_h"] = 0.0
    return result


def calcular_metricas_sitio(
    resultado: dict[str, Any],
    fecha_inicio: datetime | str,
    fecha_fin: datetime | str,
    margen_tierra_deg: float = 10.0,
    banda_solar_deg: tuple[float, float] = (5.0, 20.0),
    umbral_exposicion_solar_deg: float = 70.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calcula duraciones geométricas y devuelve el dataframe con duración."""
    start = pd.Timestamp(fecha_inicio)
    end = pd.Timestamp(fecha_fin)
    total_hours = max((end - start).total_seconds() / 3600, 0.0)
    df = agregar_duracion_muestras(resultado["datos"], end)

    def duration(mask: pd.Series) -> float:
        valid = mask.fillna(False)
        return float(df.loc[valid, "duration_h"].sum())

    sun_elev = df.get("sun_elevation_deg", pd.Series(index=df.index, dtype=float))
    earth_elev = df.get("earth_elevation_deg", pd.Series(index=df.index, dtype=float))
    sun_visible_h = duration(sun_elev > 0)
    earth_visible_h = duration(earth_elev > 0)
    earth_margin_h = duration(earth_elev >= margen_tierra_deg)
    sun_band_h = duration(sun_elev.between(*banda_solar_deg, inclusive="both"))
    sun_high_h = duration(sun_elev > umbral_exposicion_solar_deg)

    def pct(hours: float) -> float:
        return 100.0 * hours / total_hours if total_hours else np.nan

    summary = {
        "Site": resultado["sitio"],
        "Latitude (°)": float(resultado["lat"]),
        "Longitude (°)": float(resultado["lon"]),
        "Sun geometrically visible (h)": sun_visible_h,
        "Sun visible (%)": pct(sun_visible_h),
        "Earth geometrically visible (h)": earth_visible_h,
        f"Earth ≥ {margen_tierra_deg:g}° (h)": earth_margin_h,
        f"Earth ≥ {margen_tierra_deg:g}° (%)": pct(earth_margin_h),
        f"Sun {banda_solar_deg[0]:g}–{banda_solar_deg[1]:g}° (h)": sun_band_h,
        f"Sun > {umbral_exposicion_solar_deg:g}° (h)": sun_high_h,
    }
    return summary, df


def clasificar_condiciones_exploratorias(
    summary: dict[str, Any],
    earth_margin_key: str,
    min_sun_percent: float = 50.0,
    min_earth_margin_percent: float = 50.0,
) -> tuple[str, str]:
    """Devuelve una etiqueta de cribado, no una certificación de seguridad.

    Los porcentajes son parámetros exploratorios configurables por el equipo.
    No representan criterios de aceptación oficiales de NASA/HLS.
    """
    sun_pct = summary.get("Sun visible (%)", np.nan)
    earth_margin_h = summary.get(earth_margin_key, np.nan)
    total_sun_h = summary.get("Sun geometrically visible (h)", 0.0)
    # La duración total se reconstruye desde el porcentaje y las horas solares.
    total_hours = (total_sun_h * 100 / sun_pct) if pd.notna(sun_pct) and sun_pct > 0 else 0.0
    earth_pct = 100 * earth_margin_h / total_hours if total_hours > 0 else 0.0

    if total_sun_h <= 0:
        return "Priority review", "No direct solar illumination was observed in the selected time window."
    if sun_pct >= min_sun_percent and earth_pct >= min_earth_margin_percent:
        return "Favorable geometric conditions", "Meets the configured exploratory percentages for solar visibility and Earth elevation margin."
    return "Requires review", "One or both metrics are below the configured exploratory percentages."


def construir_tabla_comparativa(
    resultados: list[dict[str, Any]],
    fecha_inicio: datetime | str,
    fecha_fin: datetime | str,
    margen_tierra_deg: float = 10.0,
    min_sun_percent: float = 50.0,
    min_earth_margin_percent: float = 50.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Construye resumen y series con duración para varios sitios."""
    summaries = []
    frames: dict[str, pd.DataFrame] = {}
    earth_key = f"Earth ≥ {margen_tierra_deg:g}° (h)"
    for result in resultados:
        summary, frame = calcular_metricas_sitio(
            result,
            fecha_inicio,
            fecha_fin,
            margen_tierra_deg=margen_tierra_deg,
        )
        label, explanation = clasificar_condiciones_exploratorias(
            summary,
            earth_key,
            min_sun_percent=min_sun_percent,
            min_earth_margin_percent=min_earth_margin_percent,
        )
        summary["Exploratory screening"] = label
        summary["Note"] = explanation
        summaries.append(summary)
        frames[result["sitio"]] = frame
    return pd.DataFrame(summaries), frames
