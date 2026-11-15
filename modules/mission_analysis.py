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


def _maximo_intervalo_no_disponible(
    timed_frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> tuple[float, str, str]:
    """Estima el tramo continuo más largo con una magnitud <= threshold.

    La muestra se considera representativa hasta la siguiente marca temporal.
    Saltos de datos superiores a 1.5 veces el paso mediano se tratan como
    intervalos desconocidos, no como pérdida demostrada.
    """
    if timed_frame.empty or column not in timed_frame.columns:
        return 0.0, "n/a", "n/a"

    frame = timed_frame.sort_values("datetime").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(frame["datetime"])
    gaps_h = timestamps.diff().dt.total_seconds().dropna() / 3600.0
    positive_gaps = gaps_h[gaps_h > 0]
    median_step_h = float(positive_gaps.median()) if not positive_gaps.empty else None

    best_duration = 0.0
    best_start = None
    best_end = None
    run_duration = 0.0
    run_start = None
    run_end = None

    for _, row in frame.iterrows():
        sample_time = pd.Timestamp(row["datetime"])
        duration_raw = row.get("duration_h", 0.0)
        duration_h = float(duration_raw) if pd.notna(duration_raw) else 0.0
        value = row.get(column, np.nan)
        if pd.isna(value) or not np.isfinite(duration_h) or duration_h <= 0:
            if run_duration > best_duration:
                best_duration, best_start, best_end = run_duration, run_start, run_end
            run_duration, run_start, run_end = 0.0, None, None
            continue

        # Si falta una muestra intermedia, no afirmar que todo el intervalo
        # permaneció en estado de pérdida.
        if median_step_h and duration_h > median_step_h * 1.5:
            if run_duration > best_duration:
                best_duration, best_start, best_end = run_duration, run_start, run_end
            run_duration, run_start, run_end = 0.0, None, None
            continue

        unavailable = float(value) <= float(threshold)
        if unavailable:
            if run_start is None:
                run_start = sample_time
            run_duration += duration_h
            run_end = sample_time + pd.to_timedelta(duration_h, unit="h")
        else:
            if run_duration > best_duration:
                best_duration, best_start, best_end = run_duration, run_start, run_end
            run_duration, run_start, run_end = 0.0, None, None

    if run_duration > best_duration:
        best_duration, best_start, best_end = run_duration, run_start, run_end

    if best_start is None or best_end is None:
        return 0.0, "n/a", "n/a"
    return (
        float(best_duration),
        best_start.strftime("%Y-%m-%d %H:%M UT"),
        best_end.strftime("%Y-%m-%d %H:%M UT"),
    )


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
    longest_dark_h, dark_start, dark_end = _maximo_intervalo_no_disponible(
        df, "sun_elevation_deg", 0.0
    )
    longest_earth_gap_h, earth_gap_start, earth_gap_end = _maximo_intervalo_no_disponible(
        df, "earth_elevation_deg", float(margen_tierra_deg)
    )

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
        "Longest solar darkness (h)": longest_dark_h,
        "Longest solar darkness interval (UT)": (f"{dark_start} → {dark_end}" if dark_start != "n/a" else "n/a"),
        "Longest Earth below margin (h)": longest_earth_gap_h,
        "Longest Earth gap interval (UT)": (f"{earth_gap_start} → {earth_gap_end}" if earth_gap_start != "n/a" else "n/a"),
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
