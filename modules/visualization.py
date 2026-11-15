"""Mapas Folium, series temporales Plotly y visualización 3D."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from folium.plugins import MousePosition
from plotly.subplots import make_subplots

ROOT_DIR = Path(__file__).resolve().parents[1]
SITES_CONFIG = ROOT_DIR / "data" / "sites.json"


def load_sites_config() -> dict[str, Any]:
    with SITES_CONFIG.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(ttl=86400, show_spinner=False)
def cargar_waypoints_apollo(misiones: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Descarga y almacena en caché los waypoints de las misiones solicitadas."""
    config = load_sites_config()["apollo_missions"]
    bank: dict[str, pd.DataFrame] = {}
    for mission in misiones:
        if mission not in config:
            continue
        try:
            response = requests.get(config[mission]["waypoints_url"], timeout=(10, 30))
            response.raise_for_status()
            payload = response.json()
            rows = [
                {**feature.get("attributes", {}), **feature.get("geometry", {})}
                for feature in payload.get("features", [])
            ]
            frame = pd.DataFrame(rows)
            if not frame.empty and {"x", "y"}.issubset(frame.columns):
                frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
                frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
                frame = frame.dropna(subset=["x", "y"])
            bank[mission] = frame
        except Exception:
            bank[mission] = pd.DataFrame()
    return bank


def construir_mapa_lunar(
    resultados: list[dict[str, Any]],
    banco_waypoints: dict[str, pd.DataFrame] | None = None,
    zoom: int = 7,
) -> folium.Map:
    """Crea un mapa NASA Moon Trek con sitios, waypoints y capas de alta resolución."""
    if not resultados:
        raise ValueError("Se requiere al menos un sitio para construir el mapa.")
    banco_waypoints = banco_waypoints or {}
    config = load_sites_config()

    latitudes = [float(item["lat"]) for item in resultados]
    longitudes = [float(item["lon"]) for item in resultados]
    center_lat = float(np.mean(latitudes))
    lon_rad = np.radians(longitudes)
    center_lon = float(np.degrees(np.arctan2(np.sin(lon_rad).mean(), np.cos(lon_rad).mean())))

    lunar_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        crs="EPSG4326",
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )
    global_layer = config["global_lunar_layer"]
    folium.TileLayer(
        tiles=global_layer["url"],
        attr="NASA / LRO WAC Global",
        name=global_layer["name"],
        max_zoom=global_layer.get("max_zoom", 20),
        max_native_zoom=global_layer.get("max_native_zoom", 7),
        overlay=False,
    ).add_to(lunar_map)

    if len(resultados) > 1:
        # Ajustar longitud de forma local alrededor del centro para evitar
        # que el encuadre trate 179° y -179° como puntos separados por 358°.
        adjusted_lons = [center_lon + ((lon - center_lon + 180) % 360 - 180) for lon in longitudes]
        bounds = [
            [min(latitudes), min(adjusted_lons)],
            [max(latitudes), max(adjusted_lons)],
        ]
        lunar_map.fit_bounds(bounds, padding=(25, 25))

    colors = ["#49B9D1", "#F28E5B", "#79C98A", "#AA8BE8", "#E3C45B"]
    for index, result in enumerate(resultados):
        name = html.escape(str(result["sitio"]))
        color = colors[index % len(colors)]
        lat = float(result["lat"])
        lon = float(result["lon"])
        popup_html = (
            f"<div style='font-family:Arial;min-width:220px;color:#18202a'>"
            f"<b>{name}</b><hr style='margin:5px 0'>"
            f"Latitude: {lat:.5f}°<br>Longitude: {lon:.5f}°<br>"
            f"Origen: {html.escape(str(result.get('source', 'Custom / catalog')))}</div>"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=str(result["sitio"]),
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(lunar_map)

        mission = result.get("mision")
        if mission and mission in config["apollo_missions"]:
            layer = config["apollo_missions"][mission].get("high_res_layer")
            if layer:
                folium.TileLayer(
                    tiles=layer["url"],
                    attr=f"NASA / LROC / {mission}",
                    name=layer["name"],
                    max_zoom=20,
                    max_native_zoom=layer.get("max_native_zoom", 14),
                    overlay=True,
                    control=True,
                ).add_to(lunar_map)

            waypoints = banco_waypoints.get(mission, pd.DataFrame())
            if not waypoints.empty and {"x", "y"}.issubset(waypoints.columns):
                # Limitar cantidad de puntos dibujados si el servicio devuelve
                # cientos o miles de waypoints, para que el navegador siga fluido.
                stride = max(1, len(waypoints) // 500)
                for _, point in waypoints.iloc[::stride].iterrows():
                    try:
                        wp_lat = float(point["y"])
                        wp_lon = float(point["x"])
                    except (TypeError, ValueError):
                        continue
                    waypoint_name = html.escape(str(point.get("name", "Waypoint")))
                    folium.CircleMarker(
                        location=[wp_lat, wp_lon],
                        radius=2.5,
                        color=color,
                        fill=True,
                        fill_opacity=0.65,
                        tooltip=f"{mission}: {waypoint_name}",
                    ).add_to(lunar_map)

    folium.LayerControl(collapsed=True).add_to(lunar_map)
    MousePosition(
        position="bottomleft",
        separator=" | Lon: ",
        prefix="Lat: ",
        num_digits=5,
    ).add_to(lunar_map)
    return lunar_map


def crear_timeline_plotly(
    frames_by_site: dict[str, pd.DataFrame],
    show_sun_band: bool = True,
    earth_reference_deg: float = 10.0,
) -> go.Figure:
    """Crea dos paneles temporales: elevación solar y terrestre."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("SOL · Elevación aparente", "TIERRA · Elevación aparente"),
    )
    palette = ["#53D5E8", "#FF8D83", "#F4BD62", "#91A8FF", "#79D9B5"]
    for index, (site, frame) in enumerate(frames_by_site.items()):
        color = palette[index % len(palette)]
        if "sun_elevation_deg" in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame["datetime"],
                    y=frame["sun_elevation_deg"],
                    mode="lines",
                    name=f"{site} · Sol",
                    legendgroup=site,
                    line=dict(color=color, width=2),
                    connectgaps=False,
                    customdata=np.column_stack([
                        frame.get("sun_azimuth_deg", pd.Series(np.nan, index=frame.index)),
                    ]),
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M:%S} UT<br>"
                        "Elevación: %{y:.2f}°<br>Azimut: %{customdata[0]:.2f}°"
                        f"<extra>{site} · Sol</extra>"
                    ),
                ), row=1, col=1,
            )
        if "earth_elevation_deg" in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame["datetime"],
                    y=frame["earth_elevation_deg"],
                    mode="lines",
                    name=f"{site} · Tierra",
                    legendgroup=site,
                    line=dict(color=color, width=2, dash="dash"),
                    connectgaps=False,
                    customdata=np.column_stack([
                        frame.get("earth_azimuth_deg", pd.Series(np.nan, index=frame.index)),
                    ]),
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M:%S} UT<br>"
                        "Elevación: %{y:.2f}°<br>Azimut: %{customdata[0]:.2f}°"
                        f"<extra>{site} · Tierra</extra>"
                    ),
                ), row=2, col=1,
            )

    fig.add_hline(y=0, line_dash="dot", line_color="#9AA9B8", row=1, col=1)
    if show_sun_band:
        fig.add_hrect(
            y0=5, y1=20, fillcolor="#68D5C2", opacity=0.09,
            line_width=0, row=1, col=1,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#9AA9B8", row=2, col=1)
    fig.add_hline(y=earth_reference_deg, line_dash="dash", line_color="#F0B35D", row=2, col=1)
    fig.update_yaxes(title_text="Elevación (°)", row=1, col=1, gridcolor="rgba(150,170,190,.15)")
    fig.update_yaxes(title_text="Elevación (°)", row=2, col=1, gridcolor="rgba(150,170,190,.15)")
    fig.update_xaxes(title_text="Fecha y hora UT", row=2, col=1, rangeslider_visible=True)
    fig.update_layout(
        height=650,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#DCE7F1"),
        legend_title_text="Site / object",
        margin=dict(l=15, r=15, t=65, b=35),
    )
    return fig


def crear_grafica_comparacion(df_summary: pd.DataFrame) -> go.Figure:
    """Comparación visual de porcentaje de Sol y tiempo de Tierra sobre margen."""
    fig = go.Figure()
    if df_summary.empty:
        return fig
    fig.add_trace(go.Bar(
        x=df_summary["Site"],
        y=df_summary["Sun visible (%)"],
        name="Sun above horizon (%)",
        marker_color="#F4BD62",
    ))
    earth_percent_col = next(
        (column for column in df_summary.columns if column.startswith("Earth ≥") and column.endswith("(%)")),
        None,
    )
    if earth_percent_col:
        fig.add_trace(go.Bar(
            x=df_summary["Site"],
            y=df_summary[earth_percent_col],
            name=earth_percent_col,
            marker_color="#53D5E8",
        ))
    fig.update_layout(
        template="plotly_dark",
        title="Geometric Sun visibility and Earth elevation margin",
        yaxis_title="Window availability (%)",
        xaxis_title="Site",
        yaxis=dict(range=[0, 100]),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=20),
    )
    return fig


def crear_superficie_3d(elevation: np.ndarray, title: str = "Modelo de elevación local") -> go.Figure:
    """Genera una superficie 3D sencilla a partir de una matriz DEM recortada."""
    z = np.asarray(elevation, dtype=float)
    if z.ndim != 2:
        raise ValueError("La matriz de elevación debe ser bidimensional.")
    stride = max(1, int(max(z.shape) / 250))
    z = z[::stride, ::stride]
    y_grid, x_grid = np.indices(z.shape)
    fig = go.Figure(data=[go.Surface(x=x_grid, y=y_grid, z=z, colorscale="Earth", connectgaps=False)])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        scene=dict(xaxis_title="Columna de píxel", yaxis_title="Fila de píxel", zaxis_title="Elevación (unidades DEM)"),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=50, b=0),
        height=560,
    )
    return fig



def crear_grafica_pareto_topografica(candidates: pd.DataFrame) -> go.Figure:
    """Grafica la frontera Pareto actual basada solo en atributos topográficos.

    Eje X: pendiente (minimizar). Eje Y: rugosidad (minimizar).
    Tamaño del marcador: área aproximada del parche (maximizar).
    El color distingue candidatos no dominados del resto.
    """
    fig = go.Figure()
    if candidates is None or candidates.empty:
        fig.update_layout(
            template="plotly_dark",
            title="Topographic Pareto screening",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    frame = candidates.copy()
    for column in ("slope_deg", "roughness_m", "area_km2_approx", "lat", "lon"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "Pareto" not in frame:
        frame["Pareto"] = False
    frame["Pareto"] = frame["Pareto"].fillna(False).astype(bool)

    area = frame.get("area_km2_approx", pd.Series(0.0, index=frame.index)).fillna(0).clip(lower=0)
    max_area = float(area.max()) if len(area) else 0.0
    marker_sizes = 10 + (18 * np.sqrt(area / max_area)) if max_area > 0 else pd.Series(12, index=frame.index)
    custom = np.empty((len(frame), 4), dtype=object)
    custom[:, 0] = frame.get("candidate_id", pd.Series(np.arange(1, len(frame) + 1), index=frame.index)).astype(str).to_numpy()
    custom[:, 1] = frame.get("lat", pd.Series(np.nan, index=frame.index)).to_numpy()
    custom[:, 2] = frame.get("lon", pd.Series(np.nan, index=frame.index)).to_numpy()
    custom[:, 3] = area.to_numpy(dtype=float)

    for is_pareto, label, color in [
        (False, "Dominated candidate", "#FF8D83"),
        (True, "Non-dominated / Pareto", "#53D5E8"),
    ]:
        subset = frame[frame["Pareto"] == is_pareto]
        if subset.empty:
            continue
        subset_indices = subset.index
        sizes = marker_sizes.loc[subset_indices] if hasattr(marker_sizes, "loc") else marker_sizes
        subset_custom = custom[[frame.index.get_loc(i) for i in subset_indices]]
        fig.add_trace(go.Scatter(
            x=subset["slope_deg"],
            y=subset["roughness_m"],
            mode="markers",
            name=label,
            customdata=subset_custom,
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.88,
                line=dict(color="#E8F4FA", width=1),
            ),
            hovertemplate=(
                "Candidate %{customdata[0]}<br>"
                "Slope: %{x:.2f}°<br>"
                "Roughness: %{y:.2f} m<br>"
                "Latitude: %{customdata[1]:.5f}°<br>"
                "Longitude: %{customdata[2]:.5f}°<br>"
                "Patch area: %{customdata[3]:.5f} km²"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Topographic trade-off / Pareto frontier",
        xaxis_title="Local slope (°) · lower is better",
        yaxis_title="Local roughness (m) · lower is better",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,18,32,.45)",
        font=dict(family="Manrope, sans-serif", color="#DCE7F1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=18, r=15, t=65, b=45),
        height=430,
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor="rgba(130,170,195,.14)")
    fig.update_yaxes(gridcolor="rgba(130,170,195,.14)")
    return fig
