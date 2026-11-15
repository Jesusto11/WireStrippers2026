"""ECLIPSE — interfaz principal de Streamlit.

Ejecutar desde la raíz del repositorio:
    streamlit run app.py
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules.horizons import evaluar_sitio_lunar, estimar_numero_muestras
from modules.mission_analysis import construir_tabla_comparativa
from modules.terrain import LOLA_IDENTIFY_URL, analizar_dem_via_api, consultar_altitud_lola
from modules.visualization import (
    cargar_waypoints_apollo,
    crear_grafica_comparacion,
    crear_mapa_pendiente,
    crear_superficie_3d,
    crear_timeline_plotly,
    construir_mapa_lunar,
)

ROOT = Path(__file__).resolve().parent
SITES_PATH = ROOT / "data" / "sites.json"
LOLA_API_CONFIG_PATH = ROOT / "data" / "lola_api_config.json"

st.set_page_config(
    page_title="ECLIPSE | Evaluación de sitios polares lunares",
    page_icon="🌘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# ESTILO VISUAL
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --space: #000000;
        --panel: #0B1D39;
        --panel-2: #123A63;
        --panel-edge: rgba(47, 93, 138, .46);
        --cyan: #69C9E8;
        --blue: #2F5D8A;
        --amber: #E8B45B;
        --coral: #E98279;
        --text: #F0F4FA;
        --muted: #AAB9CA;
      }
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .stApp {
        background:
          radial-gradient(ellipse at 50% -10%, rgba(53,154,201,.22), transparent 40%),
          radial-gradient(ellipse at 2% 30%, rgba(220,104,145,.09), transparent 26%),
          radial-gradient(ellipse at 96% 5%, rgba(244,189,98,.08), transparent 24%),
          linear-gradient(180deg, #000000 0%, #0B1D39 48%, #02050A 100%);
        color: var(--text);
      }
      [data-testid="stHeader"] { background: rgba(5, 16, 30, .82); }
      [data-testid="stAppViewContainer"] > .main { background: transparent; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1D39 0%, #000000 100%);
        border-right: 1px solid rgba(83,213,232,.22);
        box-shadow: 10px 0 35px rgba(0,0,0,.18);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(18,58,99,.92), rgba(7,15,27,.98));
        border: 1px solid rgba(83,213,232,.23);
        border-top: 2px solid rgba(83,213,232,.62);
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: inset 0 1px rgba(255,255,255,.035), 0 9px 25px rgba(0,0,0,.13);
      }
      [data-testid="stMetricLabel"] { color: #9FBBD0; font: 500 10px 'DM Mono', monospace; letter-spacing: .10em; text-transform: uppercase; }
      [data-testid="stMetricValue"] { color: #F2F8FC; font-family: 'Barlow Condensed', sans-serif; font-size: 30px; letter-spacing: .02em; }
      [data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace; }
      .hero {
        position: relative; overflow: hidden;
        padding: 25px 29px 22px;
        border: 1px solid rgba(83,213,232,.38);
        border-top: 3px solid #2F5D8A;
        border-radius: 17px;
        background:
          linear-gradient(110deg, rgba(5,8,13,.99) 0%, rgba(11,29,57,.98) 48%, rgba(18,58,99,.96) 100%);
        box-shadow: 0 18px 55px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.08);
        margin-bottom: 18px;
      }
      .hero:before {
        content: ''; position: absolute; inset: 0 0 auto 0; height: 5px;
        background: repeating-linear-gradient(90deg, #3DB7D2 0 46px, transparent 46px 54px, #E8B45B 54px 61px, transparent 61px 78px);
        opacity: .8;
      }
      .hero:after {
        content: ''; position: absolute; right: -35px; top: -75px; width: 360px; height: 260px;
        border: 1px solid rgba(83,213,232,.14); border-radius: 50%;
        box-shadow: 0 0 0 18px rgba(83,213,232,.025), 0 0 0 38px rgba(83,213,232,.02), 0 0 55px rgba(83,213,232,.08);
        pointer-events: none;
      }
      .cockpit-status { display:flex; flex-wrap:wrap; gap:17px; align-items:center; margin-bottom:13px; color:#9BB8CC; font:500 10px 'DM Mono',monospace; letter-spacing:.13em; }
      .status-led { display:inline-block; width:7px; height:7px; margin-right:7px; border-radius:50%; background:#63E0B3; box-shadow:0 0 9px rgba(99,224,179,.85); }
      .status-led.amber { background:var(--amber); box-shadow:0 0 9px rgba(244,189,98,.75); }
      .hero-grid { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:end; gap:22px; position:relative; z-index:1; }
      .hero-kicker { color: #69D7E7; font: 500 10px 'DM Mono', monospace; letter-spacing: .24em; }
      .hero-title { color: #F5F9FC; font: 800 clamp(30px, 3.8vw, 47px) 'Barlow Condensed', sans-serif; letter-spacing: .045em; margin: 4px 0 2px; text-shadow: 0 3px 18px rgba(83,213,232,.12); }
      .hero-sub { color: #B7C9D9; font-size: 13px; max-width: 820px; line-height: 1.65; }
      .cockpit-readouts { display:flex; gap:9px; align-items:stretch; }
      .readout { min-width:100px; padding:10px 12px; border:1px solid rgba(83,213,232,.24); border-radius:9px; background:rgba(3,6,11,.72); box-shadow:inset 0 0 18px rgba(83,213,232,.035); }
      .readout-label { color:#82A9C3; font:500 9px 'DM Mono',monospace; letter-spacing:.12em; }
      .readout-value { color:#E8B45B; font:700 18px 'Barlow Condensed',sans-serif; letter-spacing:.08em; margin-top:3px; }
      .readout-small { color:#A7BECE; font:400 8px 'DM Mono',monospace; margin-top:2px; }
      .section-kicker { color: #65D5E5; font: 500 10px 'DM Mono', monospace; letter-spacing: .18em; margin: 3px 0 5px; }
      .panel {
        background: linear-gradient(145deg, rgba(18,58,99,.48), rgba(5,8,13,.94));
        border: 1px solid rgba(83,213,232,.22); border-left: 3px solid rgba(83,213,232,.58);
        padding: 17px 19px; border-radius: 12px; box-shadow:inset 0 1px rgba(255,255,255,.035),0 10px 25px rgba(0,0,0,.12);
      }
      .soft-note { color: #A6BDCF; font-size: 11px; line-height: 1.6; }
      .tag { display: inline-block; padding: 4px 9px; border: 1px solid rgba(83,213,232,.32); color: #83E5F0; background: rgba(43,159,192,.10); border-radius: 4px; font: 500 9px 'DM Mono', monospace; letter-spacing: .10em; }
      .mono { font-family: 'DM Mono', monospace; }
      div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
        background: linear-gradient(100deg,#49C8DE 0%,#378FC6 100%); color:#05080D;
        border:1px solid rgba(160,238,247,.4); font-weight:800; letter-spacing:.08em;
        border-radius:7px; min-height:42px; box-shadow:0 5px 15px rgba(39,145,188,.18);
      }
      div.stButton > button { border:1px solid rgba(83,213,232,.24); border-radius:7px; background:rgba(18,58,99,.42); color:#DDEAF4; }
      div.stButton > button:hover, div.stFormSubmitButton > button:hover { border-color:#E8B45B; color:#FFF; }
      div[data-testid="stTabs"] [role="tablist"] { gap:7px; border-bottom:1px solid rgba(83,213,232,.2); }
      div[data-testid="stTabs"] button[role="tab"] { background:rgba(5,8,13,.82); border:1px solid rgba(83,213,232,.14); border-bottom:0; border-radius:8px 8px 0 0; color:#9DB7CB; padding:9px 15px; }
      div[data-testid="stTabs"] button[aria-selected="true"] { background:linear-gradient(180deg,rgba(38,108,145,.34),rgba(11,29,57,.98)); color:#75E0EC; border-color:rgba(83,213,232,.42); box-shadow:inset 0 2px #69C9E8; }
      [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stFoliumChart"] { border:1px solid rgba(83,213,232,.18); border-radius:10px; overflow:hidden; background:rgba(8,24,42,.42); }
      [data-testid="stForm"] { border:1px solid rgba(83,213,232,.18); border-radius:11px; background:rgba(6,22,39,.35); padding:12px; }
      [data-testid="stExpander"] { border:1px solid rgba(83,213,232,.18); border-radius:9px; background:rgba(10,29,49,.42); }
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input { background:#0B1D39; color:#EAF5FC; border-color:rgba(83,213,232,.25); }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div { background:#0B1D39; border-color:rgba(83,213,232,.25); }
      [data-testid="stSlider"] [data-testid="stThumbValue"] { color:#E8B45B; }
      hr { border-color:rgba(83,213,232,.18); }
      .instrument-caption { color:#7FA2BB; font:500 9px 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }
      footer { visibility: hidden; }
      @media (max-width: 850px) { .hero-grid { grid-template-columns:1fr; } .cockpit-readouts { display:none; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _unique_site_names(sites: list[dict]) -> list[dict]:
    used: set[str] = set()
    for site in sites:
        base = str(site["nombre"])
        name = base
        suffix = 2
        while name in used:
            name = f"{base} ({suffix})"
            suffix += 1
        site["nombre"] = name
        used.add(name)
    return sites


def _build_selected_sites(missions: list[str], bank: dict, custom: dict | None) -> list[dict]:
    sites: list[dict] = []
    for mission in missions:
        frame = bank.get(mission, pd.DataFrame())
        if frame.empty or not {"x", "y"}.issubset(frame.columns):
            continue
        coords = frame[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        if coords.empty:
            continue
        sites.append({
            "nombre": f"{mission} · centro de los puntos de recorrido",
            "mision": mission,
            "source": "Promedio de puntos de recorrido de NASA Moon Trek",
            "lat": float(coords["y"].mean()),
            "lon": float(coords["x"].mean()),
        })
    if custom:
        sites.append({
            "nombre": custom["name"],
            "mision": None,
            "source": "Coordenadas personalizadas",
            "lat": float(custom["lat"]),
            "lon": float(custom["lon"]),
        })
    return _unique_site_names(sites)


def _format_value(value: float, suffix: str = "°") -> str:
    return "N/D" if pd.isna(value) else f"{float(value):.2f}{suffix}"


def _run_site_analysis(
    sites: list[dict],
    start: datetime,
    end: datetime,
    step: str,
    interval_minutes: int,
    identify_url: str = LOLA_IDENTIFY_URL,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Consulta primero la altitud LOLA vía API y después las efemérides Horizons."""
    outputs: list[dict] = []
    failures: list[tuple[str, str]] = []
    progress = st.progress(0, text="Preparando consultas a NASA Moon Trek y JPL Horizons…")
    for index, site in enumerate(sites):
        progress.progress(
            int(index / max(len(sites), 1) * 100),
            text=f"Consultando altitud LOLA y efemérides · {site['nombre']} · {index + 1}/{len(sites)}",
        )
        try:
            # La altitud se consulta al servicio NASA Moon Trek. Si no hay respuesta
            # o el punto es NoData, no se usa una altura de 0 m como sustituto.
            site_altitude_m = consultar_altitud_lola(
                site["lat"],
                site["lon"],
                identify_url=identify_url,
            )
            result = evaluar_sitio_lunar(
                site["nombre"],
                site["lat"],
                site["lon"],
                start.strftime("%Y-%m-%d %H:%M"),
                end.strftime("%Y-%m-%d %H:%M"),
                step,
                altitud_metros=site_altitude_m,
            )
            result["mision"] = site.get("mision")
            result["source"] = site.get("source")
            result["interval_minutes"] = interval_minutes
            result["altitude_source"] = "NASA Moon Trek · LRO LOLA DEM 128 ppd (API)"
            result["altitude_api"] = identify_url
            outputs.append(result)
        except Exception as exc:  # mostrar el error asociado al sitio
            failures.append((site["nombre"], str(exc)))
    progress.progress(100, text="Consultas finalizadas")
    progress.empty()
    return outputs, failures


def _render_site_inspector(frames_by_site: dict[str, pd.DataFrame]) -> None:
    if not frames_by_site:
        return
    st.markdown('<div class="section-kicker">CURSOR TEMPORAL / INSPECCIÓN DE MUESTRAS</div>', unsafe_allow_html=True)
    selected_site = st.selectbox("Sitio para inspeccionar", list(frames_by_site), key="time_inspector_site")
    frame = frames_by_site[selected_site].dropna(subset=["datetime"]).copy()
    if len(frame) < 1:
        st.info("No hay muestras temporales disponibles para consultar.")
        return
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    timestamps = frame["datetime"].sort_values()
    min_time = timestamps.iloc[0].to_pydatetime()
    max_time = timestamps.iloc[-1].to_pydatetime()
    if min_time == max_time:
        selected_time = min_time
    else:
        sample_step = int(frame["datetime"].diff().dropna().dt.total_seconds().median() / 60) if len(frame) > 1 else 60
        selected_time = st.slider(
            "Desplaza el tiempo de misión (UT)",
            min_value=min_time,
            max_value=max_time,
            value=min_time,
            step=timedelta(minutes=max(1, sample_step)),
            format="YYYY-MM-DD HH:mm",
            key="time_inspector_slider",
        )
    idx = (frame["datetime"] - pd.Timestamp(selected_time)).abs().idxmin()
    sample = frame.loc[idx]
    metrics = st.columns(4)
    metrics[0].metric("Azimut del Sol", _format_value(sample.get("sun_azimuth_deg", np.nan)))
    metrics[1].metric("Elevación del Sol", _format_value(sample.get("sun_elevation_deg", np.nan)))
    metrics[2].metric("Azimut de la Tierra", _format_value(sample.get("earth_azimuth_deg", np.nan)))
    metrics[3].metric("Elevación de la Tierra", _format_value(sample.get("earth_elevation_deg", np.nan)))
    st.caption(f"Muestra más cercana: {sample['datetime']:%Y-%m-%d %H:%M:%S} UT")


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="cockpit-status">
        <span><i class="status-led"></i>SISTEMAS LISTOS</span>
        <span><i class="status-led amber"></i>EFEMÉRIDES A SOLICITUD</span>
        <span>CENTRO DE CONTROL / OPERACIONES EN LA SUPERFICIE LUNAR</span>
      </div>
      <div class="hero-grid">
        <div>
          <div class="hero-kicker">EVALUADOR DE COMUNICACIONES, ATERRIZAJE E ILUMINACIÓN PARA LA EXPLORACIÓN DE SITIOS POLARES</div>
          <div class="hero-title">ECLIPSE <span style="color:#69C9E8">/</span> EXPLORACIÓN POLAR</div>
          <div class="hero-sub">Evaluador de comunicaciones, aterrizaje e iluminación para la exploración de sitios polares. Compara ubicaciones lunares, analiza la geometría entre el Sol y la Tierra y examina el terreno con datos de NASA y JPL.</div>
          <div style="margin-top:14px"><span class="tag">JPL HORIZONS</span>&nbsp; <span class="tag">LRO / MOON TREK</span>&nbsp; <span class="tag">SERIES TEMPORALES</span>&nbsp; <span class="tag">EVALUACIÓN DEL TERRENO</span></div>
        </div>
        <div class="cockpit-readouts">
          <div class="readout"><div class="readout-label">CONEXIÓN DE DATOS</div><div class="readout-value">NOMINAL</div><div class="readout-small">NASA / JPL</div></div>
          <div class="readout"><div class="readout-label">MODO DE SUPERFICIE</div><div class="readout-value" style="color:#69C9E8">ANÁLISIS</div><div class="readout-small">OPERACIONES EN LA LUNA</div></div>
          <div class="readout"><div class="readout-label">SISTEMA</div><div class="readout-value" style="color:#7FE0B5">LISTO</div><div class="readout-small">STREAMLIT</div></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

config = _load_json(SITES_PATH)
mission_options = list(config["apollo_missions"].keys())
interval_options = {
    "5 minutos": ("5 min", 5),
    "15 minutos": ("15 min", 15),
    "30 minutos": ("30 min", 30),
    "1 hora": ("60 min", 60),
    "2 horas": ("120 min", 120),
    "6 horas": ("360 min", 360),
}

# -----------------------------------------------------------------------------
# SIDEBAR / CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.sidebar.markdown("<div class='section-kicker'>CONFIGURACIÓN DE MISIÓN</div>", unsafe_allow_html=True)
st.sidebar.markdown("## Parámetros de misión")
with st.sidebar.form("mission_setup_form"):
    selected_missions = st.multiselect(
        "Sitios de referencia Apollo",
        options=mission_options,
        default=["Apollo 11"],
        max_selections=5,
        help="La coordenada de referencia se calcula como promedio de los puntos de recorrido disponibles; no necesariamente coincide con el punto oficial de aterrizaje.",
    )
    add_custom = st.checkbox("Agregar coordenadas personalizadas", value=False)
    custom_name = "Sitio personalizado"
    custom_lat = -89.0
    custom_lon = 0.0
    if add_custom:
        custom_name = st.text_input("Nombre del sitio", value="Candidato del polo sur")
        custom_lat = st.number_input("Latitud (°)", min_value=-90.0, max_value=90.0, value=-89.0, step=0.01, format="%.5f")
        custom_lon = st.number_input("Longitud (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, format="%.5f")

    st.markdown("---")
    st.caption("VENTANA DE MISIÓN / UTC")
    today = date.today()
    window_mode = st.radio(
        "Definir ventana de misión",
        ["Llegada + estancia en superficie", "Inicio / fin personalizados"],
        horizontal=False,
        help="El modo de llegada calcula el final del análisis a partir de la duración prevista de la estancia en superficie.",
    )
    start_date = st.date_input("Fecha de llegada / inicio", value=today)
    start_time = st.time_input("Hora de llegada / inicio (UT)", value=time(0, 0))
    if window_mode == "Llegada + estancia en superficie":
        stay_hours = st.number_input(
            "Duración prevista en superficie (horas)", min_value=1.0, max_value=720.0,
            value=144.0, step=6.0,
            help="Por ejemplo, 144 horas equivalen a seis días de estancia en la superficie.",
        )
        st.caption(f"Fin calculado:  {(datetime.combine(start_date, start_time) + timedelta(hours=float(stay_hours))):%Y-%m-%d %H:%M} UT")
        end_date = None
        end_time = None
    else:
        st.caption("Define una fecha y hora final para un intervalo de análisis personalizado.")
        end_date = st.date_input("Fecha de fin", value=today + timedelta(days=1))
        end_time = st.time_input("Hora de fin (UT)", value=time(0, 0))
        stay_hours = None
    interval_label = st.selectbox("Intervalo de muestreo de efemérides", list(interval_options), index=3)
    margin_earth = st.slider("Margen de elevación terrestre (°)", min_value=0, max_value=30, value=10,
                             help="Margen angular exploratorio; no certifica un enlace de radiofrecuencia.")
    min_sun_percent = st.slider("Disponibilidad solar mínima para el cribado (%)", 0, 100, 50, 5)
    min_earth_percent = st.slider("Disponibilidad terrestre mínima para el cribado (%)", 0, 100, 50, 5)
    run_button = st.form_submit_button("EJECUTAR ANÁLISIS DE MISIÓN", type="primary", width="stretch")

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='section-kicker'>DATOS TOPOGRÁFICOS</div>", unsafe_allow_html=True)
lola_api_config = _load_json(LOLA_API_CONFIG_PATH)
lola_identify_url = str(lola_api_config.get("identify_url", LOLA_IDENTIFY_URL))
with st.sidebar.expander("Altimetría lunar · NASA Moon Trek", expanded=False):
    st.success("Fuente configurada · LRO LOLA DEM Global 128 ppd")
    st.caption("La altitud se consulta en línea mediante la operación ImageServer/identify. No requiere descargar ni almacenar el GeoTIFF global.")
    st.caption("El resultado puntual se incorpora automáticamente a las coordenadas de observador de JPL Horizons.")
    st.link_button("Abrir servicio de NASA Moon Trek", lola_identify_url, width="stretch")
st.sidebar.caption("La consulta necesita conexión a internet y disponibilidad del servicio NASA. Si el punto no tiene dato o el servicio falla, el sitio se reporta como error; ECLIPSE no sustituye la elevación por 0 m.")
st.sidebar.warning("La elevación LOLA se transfiere directamente a Horizons. La equivalencia exacta entre el datum de LOLA y la superficie de referencia IAU de Horizons todavía debe verificarse para uso de ingeniería de misión.")
with st.sidebar.expander("Regiones históricas de estudio del polo sur (Artemis)"):
    artemis_metadata = _load_json(ROOT / "data" / "artemis_study_regions.json")
    st.caption(artemis_metadata["description"])
    for region_name in artemis_metadata["regions"]:
        st.write(f"• {region_name}")
    st.caption("Estas regiones históricas no son coordenadas seleccionables. Añade coordenadas verificadas y su procedencia antes de evaluarlas.")
st.sidebar.markdown("---")
st.sidebar.caption("ECLIPSE · Prototipo de apoyo a decisiones. Los resultados son exploratorios y no constituyen una certificación de aterrizaje.")

start_ut = datetime.combine(start_date, start_time)
if window_mode == "Llegada + estancia en superficie":
    end_ut = start_ut + timedelta(hours=float(stay_hours))
else:
    end_ut = datetime.combine(end_date, end_time)
step_size, interval_minutes = interval_options[interval_label]

if run_button:
    st.session_state.pop("analysis_results", None)
    st.session_state.pop("analysis_summary", None)
    st.session_state.pop("analysis_frames", None)
    if end_ut <= start_ut:
        st.sidebar.error("La fecha y hora de fin deben ser posteriores a las de inicio.")
    elif not selected_missions and not add_custom:
        st.sidebar.error("Selecciona al menos un sitio Apollo o agrega coordenadas personalizadas.")
    else:
        with st.spinner("Cargando los puntos de referencia Apollo…"):
            waypoints = cargar_waypoints_apollo(tuple(selected_missions))
        custom = ({"name": custom_name.strip() or "Sitio personalizado", "lat": custom_lat, "lon": custom_lon} if add_custom else None)
        sites = _build_selected_sites(selected_missions, waypoints, custom)
        if len(sites) > 5:
            st.sidebar.error("Se pueden analizar como máximo cinco sitios a la vez.")
        elif not sites:
            st.sidebar.error("No fue posible obtener las coordenadas de los sitios seleccionados.")
        else:
            estimated = estimar_numero_muestras(start_ut, end_ut, interval_minutes)
            if estimated > 2500:
                st.sidebar.warning(f"Se estiman aproximadamente {estimated:,} muestras por objeto y sitio. Considera utilizar un intervalo de muestreo mayor.")
            if estimated > 5000:
                st.sidebar.error("Esta ventana generaría demasiadas muestras para esta versión. Aumenta el intervalo de muestreo.")
            else:
                outputs, failures = _run_site_analysis(
                    sites,
                    start_ut,
                    end_ut,
                    step_size,
                    interval_minutes,
                    identify_url=lola_identify_url,
                )
                if outputs:
                    summary, frame_map = construir_tabla_comparativa(
                        outputs,
                        start_ut,
                        end_ut,
                        margen_tierra_deg=float(margin_earth),
                        min_sun_percent=float(min_sun_percent),
                        min_earth_margin_percent=float(min_earth_percent),
                    )
                    st.session_state["analysis_results"] = outputs
                    st.session_state["analysis_summary"] = summary
                    st.session_state["analysis_frames"] = frame_map
                    st.session_state["analysis_waypoints"] = waypoints
                    st.session_state["analysis_start"] = start_ut
                    st.session_state["analysis_end"] = end_ut
                    st.session_state["analysis_window_mode"] = window_mode
                    st.session_state["analysis_stay_hours"] = (end_ut - start_ut).total_seconds() / 3600.0
                    st.session_state["analysis_step"] = step_size
                    st.session_state["analysis_interval_minutes"] = interval_minutes
                    st.session_state["analysis_earth_margin"] = margin_earth
                    st.session_state["analysis_min_sun_percent"] = min_sun_percent
                    st.session_state["analysis_min_earth_percent"] = min_earth_percent
                    st.session_state["analysis_failures"] = failures
                    st.session_state.pop("terrain_output", None)
                    st.session_state.pop("terrain_api_output", None)
                    st.session_state.pop("terrain_api_site_name", None)
                    st.sidebar.success(f"Análisis completado: {len(outputs)} sitio(s).")
                else:
                    st.sidebar.error("No se obtuvieron resultados. Comprueba la conexión y los parámetros de Horizons.")
                for name, message in failures:
                    st.warning(f"{name}: {message}")


# -----------------------------------------------------------------------------
# RESULTADOS
# -----------------------------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.markdown(
        """
        <div class="panel">
          <div class="section-kicker">LISTO PARA EL ANÁLISIS</div>
          <h3 style="margin:5px 0 8px;color:#EAF5F6">Configura un escenario de misión lunar</h3>
          <p class="soft-note">Selecciona ubicaciones de referencia Apollo o introduce coordenadas lunares personalizadas. Define una ventana temporal común en UT y el intervalo de muestreo. ECLIPSE consultará el azimut y la elevación aparentes del Sol y la Tierra para cada sitio.</p>
          <p class="soft-note"><b>Primera prueba sugerida:</b> Apollo 11 · ventana de 24 horas · muestreo de 1 hora. Después, compara dos o más sitios y reduce el intervalo de muestreo a 15 minutos.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

results = st.session_state["analysis_results"]
summary = st.session_state["analysis_summary"]
frames_by_site = st.session_state["analysis_frames"]
waypoint_bank = st.session_state.get("analysis_waypoints", {})
analysis_start = st.session_state["analysis_start"]
analysis_end = st.session_state["analysis_end"]
analysis_window_mode = st.session_state.get("analysis_window_mode", "Inicio / fin personalizados")
analysis_step = st.session_state["analysis_step"]
earth_margin = st.session_state["analysis_earth_margin"]

st.markdown(
    f"<div class='panel' style='margin-bottom:14px'><span class='tag'>ANÁLISIS COMPLETADO</span> &nbsp; "
    f"<span class='mono' style='color:#B5CAD8'>{analysis_start:%Y-%m-%d %H:%M} → {analysis_end:%Y-%m-%d %H:%M} UT</span>"
    f"<br><span class='soft-note'>Sitios: {len(results)} &nbsp;·&nbsp; Muestreo: {analysis_step} &nbsp;·&nbsp; Margen terrestre: {earth_margin}°</span></div>",
    unsafe_allow_html=True,
)

# Top-line status metrics
number_sites = len(results)
total_samples = sum(len(item["datos"]) for item in results)
mean_sun = float(summary["Sun visible (%)"].mean()) if "Sun visible (%)" in summary else np.nan
mean_earth = float(np.nanmean([
    row.get(f"Earth ≥ {float(earth_margin):g}° (h)", np.nan) for row in summary.to_dict("records")
])) if number_sites else np.nan
kpis = st.columns(4)
kpis[0].metric("SITIOS ANALIZADOS", number_sites)
kpis[1].metric("MUESTRAS DE EFEMÉRIDES", f"{total_samples:,}")
kpis[2].metric("VISIBILIDAD SOLAR MEDIA", "N/D" if pd.isna(mean_sun) else f"{mean_sun:.1f}%")
kpis[3].metric("TIERRA SOBRE EL MARGEN (MEDIA)", "N/D" if pd.isna(mean_earth) else f"{mean_earth:.2f} h")

# Main navigation tabs
tab_overview, tab_map, tab_timeline, tab_terrain = st.tabs([
    "◈  Resumen de misión",
    "⌖  Mapa lunar",
    "◷  Línea de tiempo",
    "△  Terreno",
])

with tab_overview:
    st.markdown('<div class="section-kicker">ALTITUD DEL PUNTO DE OBSERVACIÓN</div>', unsafe_allow_html=True)
    altitude_rows = [{
        "Sitio": item["sitio"],
        "Altitud utilizada (m)": item.get("altitud_metros", np.nan),
        "Origen": item.get("altitude_source", "NASA LOLA API"),
    } for item in results]
    st.dataframe(
        pd.DataFrame(altitude_rows),
        width="stretch",
        hide_index=True,
        column_config={"Altitud utilizada (m)": st.column_config.NumberColumn(format="%.2f m")},
    )
    st.caption("Las elevaciones se consultaron en NASA Moon Trek (LRO LOLA DEM Global 128 ppd) y se pasaron a JPL Horizons como altura del observador, convertidas de metros a kilómetros. La disponibilidad depende del servicio y de la cobertura de datos del punto consultado.")

    st.markdown('<div class="section-kicker">ANALIZADOR DE VENTANA DE MISIÓN / INTERRUPCIONES CONTINUAS</div>', unsafe_allow_html=True)
    st.subheader("Ventana operativa de estancia en superficie")
    if analysis_window_mode == "Llegada + estancia en superficie":
        st.caption("La fecha de llegada y la duración prevista en superficie definen toda la ventana de misión. Las duraciones de interrupción se estiman a partir del intervalo de muestreo de Horizons.")
    else:
        st.caption("El intervalo personalizado de inicio y fin define la ventana de análisis. Las duraciones de interrupción se estiman a partir del intervalo de muestreo de Horizons.")
    window_site = st.selectbox("Sitio para consultar el detalle de la ventana", list(frames_by_site), key="mission_window_site")
    window_row = summary.loc[summary["Site"] == window_site].iloc[0]
    earth_gap_col = "Longest Earth below margin (h)"
    sun_gap_col = "Longest solar darkness (h)"
    stay_hours_display = (analysis_end - analysis_start).total_seconds() / 3600.0
    win_metrics = st.columns(4)
    window_metric_label = "ESTANCIA PREVISTA EN SUPERFICIE" if analysis_window_mode == "Llegada + estancia en superficie" else "VENTANA DE ANÁLISIS"
    win_metrics[0].metric(window_metric_label, f"{stay_hours_display:.1f} h")
    win_metrics[1].metric("SOL SOBRE EL HORIZONTE", f"{window_row.get('Sun geometrically visible (h)', np.nan):.1f} h")
    win_metrics[2].metric("MAYOR PERIODO SIN SOL", f"{window_row.get(sun_gap_col, np.nan):.2f} h")
    win_metrics[3].metric("MAYOR PERIODO BAJO EL MARGEN TERRESTRE", f"{window_row.get(earth_gap_col, np.nan):.2f} h")
    sun_gap_text = window_row.get("Longest solar darkness interval (UT)", "N/D")
    earth_gap_text = window_row.get("Longest Earth gap interval (UT)", "N/D")
    st.markdown(
        f"<div class='panel'><div class='instrument-caption'>REGISTRO DE INTERRUPCIONES OPERATIVAS / {window_site}</div>"
        f"<div class='soft-note' style='margin-top:7px'><b style='color:#E8B45B'>Sin iluminación solar:</b> {sun_gap_text} &nbsp; · &nbsp; "
        f"<b style='color:#65D5E5'>Tierra por debajo de {earth_margin:g}°:</b> {earth_gap_text}</div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="section-kicker">COMPARACIÓN DE SITIOS</div>', unsafe_allow_html=True)
        st.subheader("Comparación ambiental entre sitios")
        nombres_columnas = {
            "Site": "Sitio",
            "Latitude (°)": "Latitud (°)",
            "Longitude (°)": "Longitud (°)",
            "Sun geometrically visible (h)": "Sol visible geométricamente (h)",
            "Sun visible (%)": "Sol visible (%)",
            "Earth geometrically visible (h)": "Tierra visible geométricamente (h)",
            f"Earth ≥ {earth_margin:g}° (h)": f"Tierra ≥ {earth_margin:g}° (h)",
            f"Earth ≥ {earth_margin:g}° (%)": f"Tierra ≥ {earth_margin:g}° (%)",
            "Sun 5–20° (h)": "Sol entre 5° y 20° (h)",
            "Sun > 70° (h)": "Sol > 70° (h)",
            "Longest solar darkness (h)": "Mayor periodo continuo sin Sol (h)",
            "Longest solar darkness interval (UT)": "Intervalo continuo más largo sin Sol (UT)",
            "Longest Earth below margin (h)": "Mayor periodo continuo bajo el margen terrestre (h)",
            "Longest Earth gap interval (UT)": "Intervalo continuo más largo bajo el margen terrestre (UT)",
            "Exploratory screening": "Evaluación exploratoria",
            "Note": "Nota",
        }
        resumen_es = summary.rename(columns=nombres_columnas)
        st.dataframe(
            resumen_es,
            width="stretch",
            hide_index=True,
            column_config={
                "Latitud (°)": st.column_config.NumberColumn(format="%.5f°"),
                "Longitud (°)": st.column_config.NumberColumn(format="%.5f°"),
                "Sol visible (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Sol visible geométricamente (h)": st.column_config.NumberColumn(format="%.2f h"),
                "Tierra visible geométricamente (h)": st.column_config.NumberColumn(format="%.2f h"),
            },
        )
        st.caption("La visibilidad geométrica corresponde a una elevación aparente > 0°. El margen de elevación terrestre es un filtro angular exploratorio; no modela enlaces de radiofrecuencia ni obstrucciones del terreno.")
    with right:
        st.markdown('<div class="section-kicker">COMPARACIÓN ENTRE SITIOS</div>', unsafe_allow_html=True)
        st.plotly_chart(crear_grafica_comparacion(summary), width="stretch", config={"displaylogo": False})
        st.download_button(
            "Descargar comparación en CSV",
            data=resumen_es.to_csv(index=False).encode("utf-8-sig"),
            file_name="eclipse_comparacion_sitios.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Metodología e interpretación", expanded=False):
        st.markdown(
            "- **Sol sobre el horizonte:** elevación solar aparente > 0° en el instante muestreado.\n"
            "- **Tierra geométricamente visible:** elevación aparente de la Tierra > 0°. Esto no garantiza un enlace de radio.\n"
            f"- **Margen de elevación terrestre:** elevación aparente de la Tierra ≥ {earth_margin}°. El valor es configurable y exploratorio.\n"
            "- **Sol entre 5° y 20°:** se conserva como banda de referencia del prototipo anterior; no es un requisito universal de aterrizaje.\n"
            "- **Coordenadas de referencia Apollo:** promedios de puntos de recorrido de Moon Trek; no necesariamente corresponden a coordenadas oficiales de contacto."
        )

with tab_map:
    st.markdown('<div class="section-kicker">CONTEXTO GEOESPACIAL LUNAR</div>', unsafe_allow_html=True)
    st.subheader("Mapa de la superficie lunar")
    st.caption("Mapa global LRO WAC con capas y puntos de recorrido de misiones disponibles. La disponibilidad de mosaicos depende de los servicios de NASA Moon Trek.")
    lunar_map = construir_mapa_lunar(results, waypoint_bank, zoom=7)
    map_state = st_folium(lunar_map, height=600, use_container_width=True, key="eclipse_comparison_map")
    if map_state and map_state.get("last_clicked"):
        clicked = map_state["last_clicked"]
        st.info(f"Coordenadas seleccionadas en el mapa: latitud {clicked['lat']:.5f}° · longitud {clicked['lng']:.5f}°. Copia estas coordenadas en el apartado de sitio personalizado de la barra lateral.")

with tab_timeline:
    st.markdown('<div class="section-kicker">GEOMETRÍA TEMPORAL / UTC</div>', unsafe_allow_html=True)
    st.subheader("Evolución temporal del Sol y la Tierra")
    st.caption("Las líneas continuas representan al Sol y las discontinuas a la Tierra. Coloca el cursor sobre la gráfica para consultar el azimut y la elevación en cada instante calculado por Horizons.")
    timeline = crear_timeline_plotly(
        frames_by_site,
        show_sun_band=True,
        earth_reference_deg=float(earth_margin),
    )
    st.plotly_chart(timeline, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    _render_site_inspector(frames_by_site)
    with st.expander("Exportar series temporales", expanded=False):
        available_site = st.selectbox("Selecciona el sitio", list(frames_by_site), key="download_timeseries_site")
        serie_es = frames_by_site[available_site].rename(columns={
            "datetime": "Fecha y hora (UT)",
            "sun_azimuth_deg": "Azimut del Sol (°)",
            "sun_elevation_deg": "Elevación del Sol (°)",
            "earth_azimuth_deg": "Azimut de la Tierra (°)",
            "earth_elevation_deg": "Elevación de la Tierra (°)",
            "sun_visible_geometric": "Sol visible geométricamente",
            "earth_visible_geometric": "Tierra visible geométricamente",
            "earth_above_10deg": "Tierra por encima de 10°",
            "duration_h": "Duración de muestra (h)",
        })
        csv_bytes = serie_es.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Descargar efemérides del sitio en CSV",
            data=csv_bytes,
            file_name=f"eclipse_{available_site.lower().replace(' ', '_')}_efemerides.csv",
            mime="text/csv",
        )

with tab_terrain:
    st.markdown('<div class="section-kicker">NASA MOON TREK / LRO LOLA 128 PPd</div>', unsafe_allow_html=True)
    st.subheader("Altimetría y relieve local")
    st.caption(
        "ECLIPSE consulta el ImageServer de NASA para obtener la elevación en el punto y "
        "una micromalla alrededor de la coordenada seleccionada. No se necesita un TIFF local."
    )
    site_names = list(frames_by_site)
    selected_terrain_site = st.selectbox("Ubicación de referencia", site_names, key="terrain_site_select")
    terrain_site = next(item for item in results if item["sitio"] == selected_terrain_site)
    terrain_config = lola_api_config.get("terrain_sampling", {})
    grid_size = st.select_slider(
        "Tamaño de la micromalla de altitud",
        options=list(terrain_config.get("allowed_grid_sizes", [3, 5, 7])),
        value=int(terrain_config.get("default_grid_size", 5)),
        help="Cada nodo se consulta individualmente al servicio LOLA. Una malla mayor genera más solicitudes y tarda más.",
    )
    st.markdown(
        "<div class='panel'><b>Alcance del muestreo</b>"
        "<p class='soft-note'>La malla usa el paso nominal de 1/128° del producto. La distancia norte-sur es aproximadamente "
        "237 m por muestra; la distancia este-oeste disminuye con cos(latitud). La pendiente se estima numéricamente "
        "a partir de las muestras válidas. Esto no equivale a una superficie continua de alta resolución ni a un análisis de línea de visión.</p></div>",
        unsafe_allow_html=True,
    )

    if st.button("CONSULTAR ALTITUD Y ANALIZAR RELIEVE", type="primary", width="content"):
        try:
            with st.spinner(f"Consultando {grid_size * grid_size} puntos en NASA Moon Trek…"):
                terrain_output = analizar_dem_via_api(
                    terrain_site["lat"],
                    terrain_site["lon"],
                    n=int(grid_size),
                    identify_url=lola_identify_url,
                    max_workers=int(lola_api_config.get("max_parallel_requests", 8)),
                    lunar_radius_m=float(lola_api_config.get("lunar_reference_radius_m", 1_737_400.0)),
                )
            st.session_state["terrain_api_output"] = terrain_output
            st.session_state["terrain_api_site_name"] = selected_terrain_site
        except Exception as exc:
            st.session_state.pop("terrain_api_output", None)
            st.error(f"No se pudo completar el análisis topográfico: {exc}")

    terrain_output = st.session_state.get("terrain_api_output")
    if terrain_output and st.session_state.get("terrain_api_site_name") == selected_terrain_site:
        st.markdown("### Lecturas del entorno")
        tm = st.columns(4)
        tm[0].metric("Altitud del punto", f"{terrain_output['point_elevation']:.2f} m")
        tm[1].metric("Pendiente central", f"{terrain_output['point_slope_deg']:.2f}°")
        tm[2].metric("Desnivel de la malla", f"{terrain_output['relief_m']:.2f} m")
        tm[3].metric("Muestras válidas", f"{terrain_output['valid_samples']}/{terrain_output['total_samples']}")

        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(
                crear_superficie_3d(
                    terrain_output["elevation"],
                    f"Superficie LOLA local · {selected_terrain_site}",
                    resolution_x_m=terrain_output["resolution_x_m"],
                    resolution_y_m=terrain_output["resolution_y_m"],
                ),
                width="stretch",
                config={"displaylogo": False},
            )
        with right:
            st.plotly_chart(
                crear_mapa_pendiente(
                    terrain_output["slope_deg"],
                    "Pendiente estimada de la micromalla",
                ),
                width="stretch",
                config={"displaylogo": False},
            )

        st.markdown("#### Matriz de elevaciones consultadas")
        matrix_frame = pd.DataFrame(
            terrain_output["elevation"],
            index=[f"Fila {i + 1}" for i in range(terrain_output["grid_size"])],
            columns=[f"Columna {i + 1}" for i in range(terrain_output["grid_size"])],
        )
        st.dataframe(
            matrix_frame.style.format("{:.2f} m", na_rep="NoData"),
            width="stretch",
        )
        details_col1, details_col2, details_col3 = st.columns(3)
        details_col1.metric("Pendiente media muestreada", f"{terrain_output['mean_slope_deg']:.2f}°")
        details_col2.metric("Desviación estándar de elevación", f"{terrain_output['elevation_std_m']:.2f} m")
        details_col3.metric("Muestras con pendiente menor a 5°", f"{terrain_output['fraction_slope_below_5deg_pct']:.1f}%")
        st.caption(
            f"Fuente: {terrain_output['source']} · Separación aproximada: "
            f"{terrain_output['resolution_y_m']:.1f} m N-S × {terrain_output['resolution_x_m']:.1f} m E-O "
            f"· Radio de referencia lunar: {terrain_output['reference_radius_m']:,.0f} m. "
            "La rugosidad se resume aquí mediante el desnivel máximo y la desviación estándar de las muestras; "
            "no es una métrica oficial de aceptación de un vehículo."
        )
        st.download_button(
            "Descargar micromalla LOLA en CSV",
            data=pd.DataFrame(terrain_output["elevation"]).to_csv(index=False, header=False).encode("utf-8-sig"),
            file_name=f"eclipse_lola_micromalla_{selected_terrain_site.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### Búsqueda de zonas candidatas")
    st.info(
        "La búsqueda automática de parches candidatos se ha retirado de esta versión. "
        "El servicio identify devuelve elevaciones puntuales; la micromalla de 3×3 a 7×7 "
        "muestras sirve para caracterizar el entorno inmediato, pero no proporciona la "
        "cobertura continua necesaria para detectar y delimitar zonas de aterrizaje extensas. "
        "Para recuperar esa función se necesitaría consumir un servicio de subsetting/áreas "
        "o un DEM regional alojado en la nube."
    )

st.divider()
st.markdown(
    "<div class='soft-note'>ECLIPSE es un prototipo de análisis exploratorio. La visibilidad geométrica no equivale a un enlace de radio verificado; la micromalla de altitud y pendiente no certifica un sitio de aterrizaje ni sustituye un análisis de sombras por azimut; y los promedios de puntos de recorrido no son coordenadas oficiales de contacto. Verifica la procedencia de los datos, la compatibilidad de datums verticales y los requisitos de cada misión antes de utilizar los resultados con fines de ingeniería.</div>",
    unsafe_allow_html=True,
)
