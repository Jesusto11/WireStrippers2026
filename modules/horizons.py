"""Consultas y procesamiento de efemérides de NASA/JPL Horizons."""
from __future__ import annotations

import csv
from datetime import datetime
from math import isfinite
from typing import Any

import pandas as pd
import requests
import streamlit as st

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
SUN_COMMAND = "10"
EARTH_COMMAND = "399"


def _normalizar_nombre_columna(value: str) -> str:
    return " ".join(str(value).strip().strip('"').lower().replace("_", " ").split())


def parsear_tabla_horizons_csv(texto_resultado: str) -> pd.DataFrame:
    """Convierte la tabla delimitada por $$SOE/$$EOE en datetime, azimut y elevación.

    Las fechas devueltas por Horizons son UT. Se guardan como timestamps
    sin zona horaria explícita (UTC-naive) para conservar la compatibilidad
    con pandas/Plotly; la interfaz siempre las etiqueta como UT.
    """
    lineas = texto_resultado.splitlines()
    try:
        idx_soe = next(i for i, line in enumerate(lineas) if "$$SOE" in line)
        idx_eoe = next(i for i, line in enumerate(lineas) if "$$EOE" in line)
    except StopIteration as exc:
        raise ValueError(
            "Horizons no incluyó los marcadores $$SOE/$$EOE. "
            "Revisa los parámetros o el mensaje de error de la API."
        ) from exc

    encabezado = None
    for line in reversed(lineas[:idx_soe]):
        low = line.lower()
        if "date" in low and ("azi" in low or "azimuth" in low) and ("elev" in low or "elevation" in low):
            encabezado = line.strip()
            break
    if encabezado is None:
        raise ValueError("No se encontró el encabezado CSV con fecha, azimut y elevación.")

    nombres = next(csv.reader([encabezado], skipinitialspace=True))
    nombres = [name.strip().strip('"') for name in nombres]
    filas: list[list[str]] = []
    for line in lineas[idx_soe + 1:idx_eoe]:
        if not line.strip():
            continue
        row = next(csv.reader([line], skipinitialspace=True))
        if len(row) < len(nombres):
            row.extend([""] * (len(nombres) - len(row)))
        filas.append(row[:len(nombres)])

    if not filas:
        raise ValueError("Horizons no devolvió muestras en el intervalo solicitado.")

    norm_names = [_normalizar_nombre_columna(name) for name in nombres]

    def locate(predicate, label: str) -> int:
        for index, name in enumerate(norm_names):
            if predicate(name):
                return index
        raise ValueError(f"No se localizó la columna {label} en la respuesta CSV de Horizons.")

    i_date = locate(lambda n: "date" in n, "fecha UT")
    i_az = locate(lambda n: "azimuth" in n or " azi" in f" {n}" or n.startswith("azi"), "azimut aparente")
    i_el = locate(lambda n: "elevation" in n or " elev" in f" {n}" or n.startswith("elev"), "elevación aparente")

    raw = pd.DataFrame(filas, columns=nombres)
    date_values = (
        raw.iloc[:, i_date].astype(str)
        .str.replace(r"^A\.D\.\s*", "", regex=True)
        .str.strip()
    )
    # Horizons puede añadir marcas como ' *' a valores angulares.
    az_values = raw.iloc[:, i_az].astype(str).str.replace("*", "", regex=False).str.strip()
    el_values = raw.iloc[:, i_el].astype(str).str.replace("*", "", regex=False).str.strip()

    output = pd.DataFrame({
        "datetime": pd.to_datetime(date_values, errors="coerce", utc=False),
        "azimuth_deg": pd.to_numeric(az_values, errors="coerce"),
        "elevation_deg": pd.to_numeric(el_values, errors="coerce"),
    })
    output = (
        output.dropna(subset=["datetime"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )
    if output.empty:
        raise ValueError("No fue posible interpretar las fechas de la tabla de Horizons.")
    return output


@st.cache_data(ttl=3600, show_spinner=False)
def consultar_objeto_horizons(
    comando: str,
    latitud: float,
    longitud: float,
    fecha_inicio: str,
    fecha_fin: str,
    paso: str,
    altitud_metros: float,
) -> pd.DataFrame:
    """Consulta un objeto visto desde una ubicación lunar.

    altitud_metros es la elevación puntual obtenida desde NASA Moon Trek/LOLA.
    Se convierte a kilómetros para SITE_COORD. Horizons interpreta la altura
    geodética respecto a la superficie de referencia lunar de su modelo.
    La compatibilidad exacta entre datums verticales debe verificarse para uso
    de ingeniería de misión.
    """
    altitude = float(altitud_metros)
    if not isfinite(altitude):
        raise ValueError("Se requiere una elevación LOLA válida y finita para consultar Horizons.")

    params = {
        "format": "json",
        "COMMAND": str(comando),
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "coord@301",
        "COORD_TYPE": "GEODETIC",
        "SITE_COORD": (
            f"{float(longitud):.8f},{float(latitud):.8f},"
            f"{altitude / 1000.0:.6f}"
        ),
        "START_TIME": str(fecha_inicio),
        "STOP_TIME": str(fecha_fin),
        "STEP_SIZE": str(paso),
        "TIME_TYPE": "UT",
        "TIME_DIGITS": "SECONDS",
        "CAL_FORMAT": "CAL",
        "QUANTITIES": "4",
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
        "EXTRA_PREC": "YES",
    }
    response = requests.get(HORIZONS_API, params=params, timeout=(10, 45))
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if payload.get("error"):
        raise RuntimeError(f"JPL Horizons informó el siguiente error: {payload['error']}")
    text = payload.get("result")
    if not text:
        raise RuntimeError("JPL Horizons devolvió una respuesta sin tabla de resultados.")
    return parsear_tabla_horizons_csv(text)


@st.cache_data(ttl=3600, show_spinner=False)
def evaluar_sitio_lunar(
    nombre: str,
    latitud: float,
    longitud: float,
    fecha_inicio: str,
    fecha_fin: str,
    paso: str,
    altitud_metros: float,
) -> dict[str, Any]:
    """Une las series del Sol y de la Tierra para un sitio y ventana temporal."""
    sun = consultar_objeto_horizons(
        SUN_COMMAND, latitud, longitud, fecha_inicio, fecha_fin, paso, altitud_metros
    ).rename(columns={
        "azimuth_deg": "sun_azimuth_deg",
        "elevation_deg": "sun_elevation_deg",
    })
    earth = consultar_objeto_horizons(
        EARTH_COMMAND, latitud, longitud, fecha_inicio, fecha_fin, paso, altitud_metros
    ).rename(columns={
        "azimuth_deg": "earth_azimuth_deg",
        "elevation_deg": "earth_elevation_deg",
    })

    # Normalmente ambos objetos devuelven las mismas marcas. La unión exterior
    # conserva cualquier marca temporal discrepante y evita borrar datos.
    frame = pd.merge(sun, earth, on="datetime", how="outer")
    frame = frame.sort_values("datetime").reset_index(drop=True)
    frame["sun_visible_geometric"] = frame["sun_elevation_deg"] > 0
    frame["earth_visible_geometric"] = frame["earth_elevation_deg"] > 0
    frame["earth_above_10deg"] = frame["earth_elevation_deg"] >= 10

    return {
        "sitio": nombre,
        "lat": float(latitud),
        "lon": float(longitud),
        "altitud_metros": float(altitud_metros),
        "datos": frame,
    }


def estimar_numero_muestras(inicio: datetime, fin: datetime, intervalo_minutos: int) -> int:
    """Estimación conservadora de cantidad de muestras por objeto."""
    if intervalo_minutos <= 0:
        raise ValueError("El intervalo de muestreo debe ser positivo.")
    minutes = (fin - inicio).total_seconds() / 60
    if minutes <= 0:
        return 0
    return int(minutes // intervalo_minutos) + 2
