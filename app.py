"""LUNASCOUT — interfaz principal Streamlit.

Ejecutar desde la raíz del repositorio:
    streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules.candidate_search import buscar_candidatos_topograficos
from modules.horizons import evaluar_sitio_lunar, estimar_numero_muestras
from modules.mission_analysis import construir_tabla_comparativa
from modules.optimization import filtrar_frontera_pareto
from modules.terrain import leer_dem_local
from modules.visualization import (
    cargar_waypoints_apollo,
    crear_grafica_comparacion,
    crear_grafica_pareto_topografica,
    crear_superficie_3d,
    crear_timeline_plotly,
    construir_mapa_lunar,
)

ROOT = Path(__file__).resolve().parent
SITES_PATH = ROOT / "data" / "sites.json"
DEM_CONFIG_PATH = ROOT / "data" / "dem_config.json"

st.set_page_config(
    page_title="LUNASCOUT | Lunar Site Intelligence",
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
        --space: #071426;
        --panel: #0D2038;
        --panel-2: #122B49;
        --panel-edge: rgba(93, 190, 220, .30);
        --cyan: #53D5E8;
        --blue: #267CB8;
        --amber: #F4BD62;
        --coral: #FF8D83;
        --text: #F2F7FC;
        --muted: #A9BED2;
      }
      html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
      .stApp {
        background:
          radial-gradient(ellipse at 50% -10%, rgba(53,154,201,.22), transparent 40%),
          radial-gradient(ellipse at 2% 30%, rgba(220,104,145,.09), transparent 26%),
          radial-gradient(ellipse at 96% 5%, rgba(244,189,98,.08), transparent 24%),
          linear-gradient(180deg, #071426 0%, #08182B 48%, #06111F 100%);
        color: var(--text);
      }
      [data-testid="stHeader"] { background: rgba(5, 16, 30, .82); }
      [data-testid="stAppViewContainer"] > .main { background: transparent; }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1D34 0%, #081629 100%);
        border-right: 1px solid rgba(83,213,232,.22);
        box-shadow: 10px 0 35px rgba(0,0,0,.18);
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(19,47,75,.96), rgba(10,29,50,.96));
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
        border-top: 3px solid #348DBB;
        border-radius: 17px;
        background:
          linear-gradient(110deg, rgba(10,29,52,.98) 0%, rgba(17,51,79,.96) 55%, rgba(18,39,69,.96) 100%);
        box-shadow: 0 18px 55px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.08);
        margin-bottom: 18px;
      }
      .hero:before {
        content: ''; position: absolute; inset: 0 0 auto 0; height: 5px;
        background: repeating-linear-gradient(90deg, #3DB7D2 0 46px, transparent 46px 54px, #F4BD62 54px 61px, transparent 61px 78px);
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
      .readout { min-width:100px; padding:10px 12px; border:1px solid rgba(83,213,232,.24); border-radius:9px; background:rgba(4,17,32,.47); box-shadow:inset 0 0 18px rgba(83,213,232,.035); }
      .readout-label { color:#82A9C3; font:500 9px 'DM Mono',monospace; letter-spacing:.12em; }
      .readout-value { color:#F4BD62; font:700 18px 'Barlow Condensed',sans-serif; letter-spacing:.08em; margin-top:3px; }
      .readout-small { color:#A7BECE; font:400 8px 'DM Mono',monospace; margin-top:2px; }
      .section-kicker { color: #65D5E5; font: 500 10px 'DM Mono', monospace; letter-spacing: .18em; margin: 3px 0 5px; }
      .panel {
        background: linear-gradient(145deg, rgba(15,37,61,.90), rgba(9,26,46,.92));
        border: 1px solid rgba(83,213,232,.22); border-left: 3px solid rgba(83,213,232,.58);
        padding: 17px 19px; border-radius: 12px; box-shadow:inset 0 1px rgba(255,255,255,.035),0 10px 25px rgba(0,0,0,.12);
      }
      .soft-note { color: #A6BDCF; font-size: 11px; line-height: 1.6; }
      .tag { display: inline-block; padding: 4px 9px; border: 1px solid rgba(83,213,232,.32); color: #83E5F0; background: rgba(43,159,192,.10); border-radius: 4px; font: 500 9px 'DM Mono', monospace; letter-spacing: .10em; }
      .mono { font-family: 'DM Mono', monospace; }
      div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
        background: linear-gradient(100deg,#49C8DE 0%,#378FC6 100%); color:#071626;
        border:1px solid rgba(160,238,247,.4); font-weight:800; letter-spacing:.08em;
        border-radius:7px; min-height:42px; box-shadow:0 5px 15px rgba(39,145,188,.18);
      }
      div.stButton > button { border:1px solid rgba(83,213,232,.24); border-radius:7px; background:rgba(16,44,70,.76); color:#DDEAF4; }
      div.stButton > button:hover, div.stFormSubmitButton > button:hover { border-color:#F4BD62; color:#FFF; }
      div[data-testid="stTabs"] [role="tablist"] { gap:7px; border-bottom:1px solid rgba(83,213,232,.2); }
      div[data-testid="stTabs"] button[role="tab"] { background:rgba(11,31,52,.72); border:1px solid rgba(83,213,232,.14); border-bottom:0; border-radius:8px 8px 0 0; color:#9DB7CB; padding:9px 15px; }
      div[data-testid="stTabs"] button[aria-selected="true"] { background:linear-gradient(180deg,rgba(38,108,145,.34),rgba(12,36,59,.95)); color:#75E0EC; border-color:rgba(83,213,232,.42); box-shadow:inset 0 2px #53D5E8; }
      [data-testid="stDataFrame"], [data-testid="stPlotlyChart"], [data-testid="stFoliumChart"] { border:1px solid rgba(83,213,232,.18); border-radius:10px; overflow:hidden; background:rgba(8,24,42,.42); }
      [data-testid="stForm"] { border:1px solid rgba(83,213,232,.18); border-radius:11px; background:rgba(6,22,39,.35); padding:12px; }
      [data-testid="stExpander"] { border:1px solid rgba(83,213,232,.18); border-radius:9px; background:rgba(10,29,49,.42); }
      [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input { background:#0B2036; color:#EAF5FC; border-color:rgba(83,213,232,.25); }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div { background:#0B2036; border-color:rgba(83,213,232,.25); }
      [data-testid="stSlider"] [data-testid="stThumbValue"] { color:#F4BD62; }
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
            "nombre": f"{mission} · waypoint centroid",
            "mision": mission,
            "source": "Mean of NASA Moon Trek waypoints",
            "lat": float(coords["y"].mean()),
            "lon": float(coords["x"].mean()),
        })
    if custom:
        sites.append({
            "nombre": custom["name"],
            "mision": None,
            "source": "Custom coordinates",
            "lat": float(custom["lat"]),
            "lon": float(custom["lon"]),
        })
    return _unique_site_names(sites)


def _format_value(value: float, suffix: str = "°") -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}{suffix}"


def _run_site_analysis(
    sites: list[dict], start: datetime, end: datetime, step: str, interval_minutes: int
) -> tuple[list[dict], list[tuple[str, str]]]:
    outputs: list[dict] = []
    failures: list[tuple[str, str]] = []
    progress = st.progress(0, text="Preparando consultas a NASA/JPL Horizons…")
    for index, site in enumerate(sites):
        progress.progress(
            int(index / max(len(sites), 1) * 100),
            text=f"Consultando {site['nombre']} · {index + 1}/{len(sites)}",
        )
        try:
            result = evaluar_sitio_lunar(
                site["nombre"],
                site["lat"],
                site["lon"],
                start.strftime("%Y-%m-%d %H:%M"),
                end.strftime("%Y-%m-%d %H:%M"),
                step,
            )
            result["mision"] = site.get("mision")
            result["source"] = site.get("source")
            result["interval_minutes"] = interval_minutes
            outputs.append(result)
        except Exception as exc:  # mostrar el error asociado al sitio
            failures.append((site["nombre"], str(exc)))
    progress.progress(100, text="Consultas finalizadas")
    progress.empty()
    return outputs, failures


def _render_site_inspector(frames_by_site: dict[str, pd.DataFrame]) -> None:
    if not frames_by_site:
        return
    st.markdown('<div class="section-kicker">TIME CURSOR / SAMPLE INSPECTOR</div>', unsafe_allow_html=True)
    selected_site = st.selectbox("Site to inspect", list(frames_by_site), key="time_inspector_site")
    frame = frames_by_site[selected_site].dropna(subset=["datetime"]).copy()
    if len(frame) < 1:
        st.info("No time samples are available for inspection.")
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
    metrics[0].metric("Sun azimuth", _format_value(sample.get("sun_azimuth_deg", np.nan)))
    metrics[1].metric("Sun elevation", _format_value(sample.get("sun_elevation_deg", np.nan)))
    metrics[2].metric("Earth azimuth", _format_value(sample.get("earth_azimuth_deg", np.nan)))
    metrics[3].metric("Earth elevation", _format_value(sample.get("earth_elevation_deg", np.nan)))
    st.caption(f"Nearest sample: {sample['datetime']:%Y-%m-%d %H:%M:%S} UT")


def _write_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".tif"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getvalue())
    temp.close()
    return temp.name


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="cockpit-status">
        <span><i class="status-led"></i>FLIGHT SYSTEMS READY</span>
        <span><i class="status-led amber"></i>EPHEMERIS ON DEMAND</span>
        <span>CONTROL DECK / LUNAR SURFACE OPERATIONS</span>
      </div>
      <div class="hero-grid">
        <div>
          <div class="hero-kicker">LUNAR LANDING SITE INTELLIGENCE / MISSION SUPPORT</div>
          <div class="hero-title">LUNASCOUT <span style="color:#53D5E8">/</span> LUNAR OPERATIONS</div>
          <div class="hero-sub">A cockpit-inspired mission console to compare lunar sites, inspect Sun–Earth geometry and screen local terrain using NASA/JPL data.</div>
          <div style="margin-top:14px"><span class="tag">JPL HORIZONS</span>&nbsp; <span class="tag">LRO / MOON TREK</span>&nbsp; <span class="tag">MISSION TIMELINE</span>&nbsp; <span class="tag">TERRAIN LAB</span></div>
        </div>
        <div class="cockpit-readouts">
          <div class="readout"><div class="readout-label">DATA BUS</div><div class="readout-value">NOMINAL</div><div class="readout-small">NASA / JPL</div></div>
          <div class="readout"><div class="readout-label">SURFACE MODE</div><div class="readout-value" style="color:#65D5E5">ANALYSIS</div><div class="readout-small">LUNAR OPS</div></div>
          <div class="readout"><div class="readout-label">SYSTEM</div><div class="readout-value" style="color:#7FE0B5">READY</div><div class="readout-small">STREAMLIT</div></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

config = _load_json(SITES_PATH)
mission_options = list(config["apollo_missions"].keys())
interval_options = {
    "5 minutes": ("5 min", 5),
    "15 minutes": ("15 min", 15),
    "30 minutes": ("30 min", 30),
    "1 hour": ("60 min", 60),
    "2 hours": ("120 min", 120),
    "6 hours": ("360 min", 360),
}

# -----------------------------------------------------------------------------
# SIDEBAR / CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.sidebar.markdown("<div class='section-kicker'>MISSION CONFIGURATION</div>", unsafe_allow_html=True)
st.sidebar.markdown("## Mission parameters")
with st.sidebar.form("mission_setup_form"):
    selected_missions = st.multiselect(
        "Apollo reference sites",
        options=mission_options,
        default=["Apollo 11"],
        max_selections=5,
        help="La coordenada de referencia se calcula como promedio de waypoints disponibles; no necesariamente es el touchdown oficial.",
    )
    add_custom = st.checkbox("Add custom coordinates", value=False)
    custom_name = "Custom site"
    custom_lat = -89.0
    custom_lon = 0.0
    if add_custom:
        custom_name = st.text_input("Site label", value="South Pole candidate")
        custom_lat = st.number_input("Latitude (°)", min_value=-90.0, max_value=90.0, value=-89.0, step=0.01, format="%.5f")
        custom_lon = st.number_input("Longitude (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.01, format="%.5f")

    st.markdown("---")
    st.caption("MISSION WINDOW / UTC")
    today = date.today()
    window_mode = st.radio(
        "Define mission window",
        ["Arrival + surface stay", "Custom start / end"],
        horizontal=False,
        help="Arrival mode computes the end of the analysis from the planned surface-stay duration.",
    )
    start_date = st.date_input("Landing / start date", value=today)
    start_time = st.time_input("Landing / start time (UT)", value=time(0, 0))
    if window_mode == "Arrival + surface stay":
        stay_hours = st.number_input(
            "Planned surface stay (hours)", min_value=1.0, max_value=720.0,
            value=144.0, step=6.0,
            help="For example, 144 hours corresponds to a six-day surface stay.",
        )
        st.caption(f"Calculated end: {(datetime.combine(start_date, start_time) + timedelta(hours=float(stay_hours))):%Y-%m-%d %H:%M} UT")
        end_date = None
        end_time = None
    else:
        st.caption("Set an explicit end timestamp for a custom analysis range.")
        end_date = st.date_input("End date", value=today + timedelta(days=1))
        end_time = st.time_input("End time (UT)", value=time(0, 0))
        stay_hours = None
    interval_label = st.selectbox("Ephemeris sampling interval", list(interval_options), index=3)
    margin_earth = st.slider("Earth elevation reference (°)", min_value=0, max_value=30, value=10,
                             help="Exploratory angular margin. It does not certify an RF link.")
    min_sun_percent = st.slider("Minimum Sun-visible window for screening (%)", 0, 100, 50, 5)
    min_earth_percent = st.slider("Minimum Earth-margin window for screening (%)", 0, 100, 50, 5)
    run_button = st.form_submit_button("RUN MISSION ANALYSIS", type="primary", width="stretch")

st.sidebar.markdown("---")
st.sidebar.markdown("<div class='section-kicker'>TERRAIN DATA</div>", unsafe_allow_html=True)
dem_upload = st.sidebar.file_uploader("Upload lunar DEM (.tif)", type=["tif", "tiff"])
dem_config = _load_json(DEM_CONFIG_PATH)
dem_local_path = st.sidebar.text_input("Or DEM path in repository", value=dem_config.get("default_path", ""))
st.sidebar.caption("DEM is optional. Verify the CRS, resolution and units before interpreting slope or roughness.")
with st.sidebar.expander("Artemis south-pole study-region metadata"):
    artemis_metadata = _load_json(ROOT / "data" / "artemis_study_regions.json")
    st.caption(artemis_metadata["description"])
    for region_name in artemis_metadata["regions"]:
        st.write(f"• {region_name}")
    st.caption("These historical regions are not selectable coordinates. Add verified coordinates and provenance before evaluating them.")
st.sidebar.markdown("---")
st.sidebar.caption("LUNASCOUT prototype · Results are exploratory decision-support outputs, not landing certification.")

start_ut = datetime.combine(start_date, start_time)
if window_mode == "Arrival + surface stay":
    end_ut = start_ut + timedelta(hours=float(stay_hours))
else:
    end_ut = datetime.combine(end_date, end_time)
step_size, interval_minutes = interval_options[interval_label]

if run_button:
    st.session_state.pop("analysis_results", None)
    st.session_state.pop("analysis_summary", None)
    st.session_state.pop("analysis_frames", None)
    if end_ut <= start_ut:
        st.sidebar.error("The end date/time must be later than the start date/time.")
    elif not selected_missions and not add_custom:
        st.sidebar.error("Select at least one Apollo site or add custom coordinates.")
    else:
        with st.spinner("Loading Apollo waypoint reference data…"):
            waypoints = cargar_waypoints_apollo(tuple(selected_missions))
        custom = ({"name": custom_name.strip() or "Custom site", "lat": custom_lat, "lon": custom_lon} if add_custom else None)
        sites = _build_selected_sites(selected_missions, waypoints, custom)
        if len(sites) > 5:
            st.sidebar.error("A maximum of five sites can be analyzed at once.")
        elif not sites:
            st.sidebar.error("Coordinates could not be resolved for the selected sites.")
        else:
            estimated = estimar_numero_muestras(start_ut, end_ut, interval_minutes)
            if estimated > 2500:
                st.sidebar.warning(f"About {estimated:,} samples per object and site are estimated. Consider a longer sampling interval.")
            if estimated > 5000:
                st.sidebar.error("This window would generate too many samples for this version. Increase the sampling interval.")
            else:
                outputs, failures = _run_site_analysis(sites, start_ut, end_ut, step_size, interval_minutes)
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
                    st.session_state.pop("candidate_output", None)
                    st.sidebar.success(f"Analysis completed: {len(outputs)} site(s).")
                else:
                    st.sidebar.error("No results were returned. Check connectivity and Horizons parameters.")
                for name, message in failures:
                    st.warning(f"{name}: {message}")


# -----------------------------------------------------------------------------
# RESULTADOS
# -----------------------------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.markdown(
        """
        <div class="panel">
          <div class="section-kicker">READY FOR ANALYSIS</div>
          <h3 style="margin:5px 0 8px;color:#EAF5F6">Configure a lunar mission scenario</h3>
          <p class="soft-note">Select Apollo reference locations or enter custom lunar coordinates, define a common UT time window and choose the ephemeris sampling interval. LUNASCOUT will query the apparent azimuth and elevation of the Sun and Earth for every selected site.</p>
          <p class="soft-note"><b>Suggested first test:</b> Apollo 11 · 24-hour window · 1-hour sampling. Then compare two or more sites and reduce the sampling interval to 15 minutes.</p>
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
analysis_window_mode = st.session_state.get("analysis_window_mode", "Custom start / end")
analysis_step = st.session_state["analysis_step"]
earth_margin = st.session_state["analysis_earth_margin"]

st.markdown(
    f"<div class='panel' style='margin-bottom:14px'><span class='tag'>ANALYSIS COMPLETE</span> &nbsp; "
    f"<span class='mono' style='color:#B5CAD8'>{analysis_start:%Y-%m-%d %H:%M} → {analysis_end:%Y-%m-%d %H:%M} UT</span>"
    f"<br><span class='soft-note'>Sites: {len(results)} &nbsp;·&nbsp; Sampling: {analysis_step} &nbsp;·&nbsp; Earth reference: {earth_margin}°</span></div>",
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
kpis[0].metric("SITES ANALYZED", number_sites)
kpis[1].metric("EPHEMERIS SAMPLES", f"{total_samples:,}")
kpis[2].metric("MEAN SUN VISIBILITY", "n/a" if pd.isna(mean_sun) else f"{mean_sun:.1f}%")
kpis[3].metric("MEAN EARTH ≥ REF.", "n/a" if pd.isna(mean_earth) else f"{mean_earth:.2f} h")

# Main navigation tabs
tab_overview, tab_map, tab_timeline, tab_terrain = st.tabs([
    "◈  Mission deck",
    "⌖  Lunar radar",
    "◷  Flight timeline",
    "△  Terrain systems",
])

with tab_overview:
    st.markdown('<div class="section-kicker">MISSION WINDOW ANALYZER / CONTINUOUS OUTAGE SCREENING</div>', unsafe_allow_html=True)
    st.subheader("Surface-stay operational window")
    if analysis_window_mode == "Arrival + surface stay":
        st.caption("The selected arrival timestamp and planned surface-stay duration define the complete mission window. Outage durations are approximated from the selected Horizons sampling interval.")
    else:
        st.caption("The selected custom start/end range defines the analysis window. Outage durations are approximated from the selected Horizons sampling interval.")
    window_site = st.selectbox("Site for detailed window readout", list(frames_by_site), key="mission_window_site")
    window_row = summary.loc[summary["Site"] == window_site].iloc[0]
    earth_gap_col = "Longest Earth below margin (h)"
    sun_gap_col = "Longest solar darkness (h)"
    stay_hours_display = (analysis_end - analysis_start).total_seconds() / 3600.0
    win_metrics = st.columns(4)
    window_metric_label = "PLANNED SURFACE STAY" if analysis_window_mode == "Arrival + surface stay" else "ANALYSIS WINDOW"
    win_metrics[0].metric(window_metric_label, f"{stay_hours_display:.1f} h")
    win_metrics[1].metric("SUN ABOVE HORIZON", f"{window_row.get('Sun geometrically visible (h)', np.nan):.1f} h")
    win_metrics[2].metric("LONGEST SOLAR DARKNESS", f"{window_row.get(sun_gap_col, np.nan):.2f} h")
    win_metrics[3].metric("LONGEST EARTH-MARGIN GAP", f"{window_row.get(earth_gap_col, np.nan):.2f} h")
    sun_gap_text = window_row.get("Longest solar darkness interval (UT)", "n/a")
    earth_gap_text = window_row.get("Longest Earth gap interval (UT)", "n/a")
    st.markdown(
        f"<div class='panel'><div class='instrument-caption'>OPERATIONAL GAP LOG / {window_site}</div>"
        f"<div class='soft-note' style='margin-top:7px'><b style='color:#F4BD62'>Solar darkness:</b> {sun_gap_text} &nbsp; · &nbsp; "
        f"<b style='color:#65D5E5'>Earth below {earth_margin:g}°:</b> {earth_gap_text}</div></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.markdown('<div class="section-kicker">SITE COMPARISON</div>', unsafe_allow_html=True)
        st.subheader("Cross-site environmental comparison")
        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
            column_config={
                "Latitude (°)": st.column_config.NumberColumn(format="%.5f°"),
                "Longitude (°)": st.column_config.NumberColumn(format="%.5f°"),
                "Sun visible (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Sun geometrically visible (h)": st.column_config.NumberColumn(format="%.2f h"),
                "Earth geometrically visible (h)": st.column_config.NumberColumn(format="%.2f h"),
            },
        )
        st.caption("Geometric visibility means apparent elevation > 0°. The Earth elevation margin is an exploratory angular filter; it does not model RF links or terrain obstruction.")
    with right:
        st.markdown('<div class="section-kicker">CROSS-SITE SIGNAL</div>', unsafe_allow_html=True)
        st.plotly_chart(crear_grafica_comparacion(summary), width="stretch", config={"displaylogo": False})
        st.download_button(
            "Download comparison CSV",
            data=summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="lunascout_site_comparison.csv",
            mime="text/csv",
            width="stretch",
        )

    with st.expander("Methodology and interpretation", expanded=False):
        st.markdown(
            "- **Sun above horizon:** apparent solar elevation > 0° at the sampled time.\n"
            "- **Earth geometrically visible:** apparent Earth elevation > 0°. This does not guarantee a radio link.\n"
            f"- **Earth elevation margin:** apparent Earth elevation ≥ {earth_margin}°. The value is configurable and exploratory.\n"
            "- **Solar 5°–20°:** retained as a reference band from the previous prototype, not as a universal landing requirement.\n"
            "- **Apollo reference coordinates:** means of Moon Trek waypoints; not necessarily official landing touchdown coordinates."
        )

with tab_map:
    st.markdown('<div class="section-kicker">LUNAR GEOSPATIAL CONTEXT</div>', unsafe_allow_html=True)
    st.subheader("Lunar navigation display")
    st.caption("Global LRO WAC basemap with available mission-specific layers and waypoints. Tile availability depends on NASA Moon Trek services.")
    lunar_map = construir_mapa_lunar(results, waypoint_bank, zoom=7)
    map_state = st_folium(lunar_map, height=600, use_container_width=True, key="lunascout_comparison_map")
    if map_state and map_state.get("last_clicked"):
        clicked = map_state["last_clicked"]
        st.info(f"Map cursor: latitude {clicked['lat']:.5f}° · longitude {clicked['lng']:.5f}°. Copy these coordinates into a custom-site run in the sidebar.")

with tab_timeline:
    st.markdown('<div class="section-kicker">TEMPORAL GEOMETRY / UTC</div>', unsafe_allow_html=True)
    st.subheader("Sun / Earth temporal telemetry")
    st.caption("Solid traces show the Sun; dashed traces show the Earth. Hover to inspect azimuth and elevation at each Horizons timestamp.")
    timeline = crear_timeline_plotly(
        frames_by_site,
        show_sun_band=True,
        earth_reference_deg=float(earth_margin),
    )
    st.plotly_chart(timeline, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    _render_site_inspector(frames_by_site)
    with st.expander("Download time series", expanded=False):
        available_site = st.selectbox("Select time series", list(frames_by_site), key="download_timeseries_site")
        csv_bytes = frames_by_site[available_site].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download selected site's ephemeris CSV",
            data=csv_bytes,
            file_name=f"lunascout_{available_site.lower().replace(' ', '_')}_ephemeris.csv",
            mime="text/csv",
        )

with tab_terrain:
    st.markdown('<div class="section-kicker">DIGITAL ELEVATION MODEL / LOCAL SCREENING</div>', unsafe_allow_html=True)
    st.subheader("Terrain systems / candidate scouting")
    st.caption("Upload a LOLA/LROC DEM or specify a DEM file already committed to the repository. Local processing uses a window around the selected site rather than loading the complete global raster.")
    site_names = list(frames_by_site)
    selected_terrain_site = st.selectbox("Reference location", site_names, key="terrain_site_select")
    terrain_site = next(item for item in results if item["sitio"] == selected_terrain_site)
    dem_source_available = bool(dem_upload is not None or (dem_local_path and Path(dem_local_path).exists()))
    if not dem_source_available:
        st.info("No DEM source detected. Upload a GeoTIFF in the sidebar or set a valid repository path.")
    else:
        terrain_col1, terrain_col2 = st.columns([1, 1])
        with terrain_col1:
            radius_px = st.slider("Local DEM window radius (pixels)", 30, 250, int(dem_config.get("default_window_radius_px", 100)), 10)
        with terrain_col2:
            st.markdown("<div class='panel'><b>Screening note</b><p class='soft-note'>Slope and local roughness are topographic indicators only. This prototype does not certify a landing ellipse or evaluate all vehicle-specific hazards.</p></div>", unsafe_allow_html=True)

        if st.button("ANALYZE SELECTED TERRAIN", type="primary", width="content"):
            temp_path = None
            try:
                if dem_upload is not None:
                    temp_path = _write_upload_to_temp(dem_upload)
                    source_path = temp_path
                else:
                    source_path = dem_local_path
                with st.spinner("Reading local DEM window and calculating terrain derivatives…"):
                    terrain_output = leer_dem_local(
                        source_path,
                        terrain_site["lat"],
                        terrain_site["lon"],
                        radius_px=radius_px,
                        elevation_to_m_factor=float(dem_config.get("elevation_to_m_factor", 1.0)),
                    )
                st.session_state["terrain_output"] = terrain_output
                st.session_state["terrain_site_name"] = selected_terrain_site
            except Exception as exc:
                st.error(f"DEM analysis failed: {exc}")
            finally:
                if temp_path and Path(temp_path).exists():
                    Path(temp_path).unlink(missing_ok=True)

        terrain_output = st.session_state.get("terrain_output")
        if terrain_output and st.session_state.get("terrain_site_name") == selected_terrain_site:
            tm = st.columns(3)
            tm[0].metric("Elevation at point", "n/a" if not np.isfinite(terrain_output["point_elevation"]) else f"{terrain_output['point_elevation']:.2f} m")
            tm[1].metric("Local slope", "n/a" if not np.isfinite(terrain_output["point_slope_deg"]) else f"{terrain_output['point_slope_deg']:.2f}°")
            tm[2].metric("Mean local slope", f"{terrain_output['mean_slope_deg']:.2f}°")
            st.plotly_chart(crear_superficie_3d(terrain_output["elevation"], f"Local terrain · {selected_terrain_site}"), width="stretch")
            st.caption(f"DEM CRS: {terrain_output['crs']} · Approx. pixel size: {terrain_output['resolution_x_m']:.2f} m × {terrain_output['resolution_y_m']:.2f} m · Vertical scale factor: {dem_config.get('elevation_to_m_factor', 1.0)} (verify against source DEM metadata).")

            st.markdown("---")
            st.markdown('<div class="section-kicker">AUTOMATIC LANDING ZONE DETECTION / TOPOGRAPHIC SCREEN</div>', unsafe_allow_html=True)
            st.write("Search for connected candidate patches around the selected reference coordinate using slope and local roughness constraints.")
            st.caption("For Haworth, Nobile or Malapert, use a verified regional reference coordinate as a Custom site in the sidebar. The bundled Artemis list contains names only, not authoritative region-center coordinates.")
            c1, c2, c3 = st.columns(3)
            max_slope = c1.slider("Maximum slope (°)", 1.0, 15.0, float(dem_config["candidate_search"]["max_slope_deg"]), 0.5)
            max_rough = c2.slider("Maximum roughness (m)", 0.5, 10.0, float(dem_config["candidate_search"]["max_roughness_m"]), 0.5)
            max_count = c3.slider("Maximum candidates", 3, 30, 12)
            if st.button("SEARCH TOPOGRAPHIC CANDIDATES", width="stretch"):
                candidate_temp = None
                try:
                    if dem_upload is not None:
                        candidate_temp = _write_upload_to_temp(dem_upload)
                        source_path = candidate_temp
                    else:
                        source_path = dem_local_path
                    with st.spinner("Screening DEM patches…"):
                        candidates = buscar_candidatos_topograficos(
                            source_path,
                            terrain_site["lat"],
                            terrain_site["lon"],
                            search_radius_px=max(60, radius_px),
                            max_slope_deg=max_slope,
                            max_roughness_m=max_rough,
                            min_component_pixels=int(dem_config["candidate_search"]["minimum_component_pixels"]),
                            min_spacing_m=float(dem_config["candidate_search"]["minimum_spacing_m"]),
                            max_candidates=max_count,
                            elevation_to_m_factor=float(dem_config.get("elevation_to_m_factor", 1.0)),
                        )
                    if candidates.empty:
                        st.warning("No patches met the selected filters. Increase the search radius or review the thresholds.")
                        st.session_state.pop("candidate_output", None)
                    else:
                        candidates = filtrar_frontera_pareto(
                            candidates,
                            objective_columns=["slope_deg", "roughness_m", "area_km2_approx"],
                            maximize=["area_km2_approx"],
                        )
                        st.session_state["candidate_output"] = candidates
                        st.session_state["candidate_site_name"] = selected_terrain_site
                except Exception as exc:
                    st.error(f"Candidate search failed: {exc}")
                finally:
                    if candidate_temp and Path(candidate_temp).exists():
                        Path(candidate_temp).unlink(missing_ok=True)

            candidate_output = st.session_state.get("candidate_output")
            if candidate_output is not None and st.session_state.get("candidate_site_name") == selected_terrain_site:
                st.markdown("#### Topographic candidate patches")
                st.dataframe(candidate_output, width="stretch", hide_index=True)
                pareto_count = int(candidate_output["Pareto"].sum())
                st.info(f"{pareto_count} candidate(s) are non-dominated across slope, roughness and approximate patch area. This is a topography-only shortlist, not a landing-site safety result.")
                st.caption("The chart minimizes slope and roughness; marker size represents approximate patch area. Cyan points are non-dominated when all three terrain objectives are considered.")
                st.plotly_chart(
                    crear_grafica_pareto_topografica(candidate_output),
                    width="stretch",
                    config={"displaylogo": False},
                )
                st.markdown("#### Candidate coordinates on the lunar map")
                candidate_map_records = [{
                    "sitio": f"Reference · {selected_terrain_site}",
                    "lat": float(terrain_site["lat"]),
                    "lon": float(terrain_site["lon"]),
                    "source": "Selected reference coordinate",
                }]
                for _, candidate_row in candidate_output.iterrows():
                    candidate_map_records.append({
                        "sitio": f"Candidate {int(candidate_row['candidate_id'])}",
                        "lat": float(candidate_row["lat"]),
                        "lon": float(candidate_row["lon"]),
                        "source": "DEM topographic screening",
                    })
                candidate_lunar_map = construir_mapa_lunar(candidate_map_records, {}, zoom=11)
                st_folium(
                    candidate_lunar_map,
                    height=440,
                    use_container_width=True,
                    key=f"candidate_map_{selected_terrain_site}",
                )
                st.download_button(
                    "Download candidate coordinates CSV",
                    data=candidate_output.to_csv(index=False).encode("utf-8-sig"),
                    file_name="lunascout_topographic_candidates.csv",
                    mime="text/csv",
                )

st.divider()
st.markdown(
    "<div class='soft-note'>LUNASCOUT is an exploratory analysis prototype. Geometric visibility is not a verified RF link; a flat/smooth DEM screen is not a certified landing assessment; and waypoint averages are not official touchdown coordinates. Confirm source provenance, coordinate reference systems, DEM vertical units and mission-specific requirements before engineering use.</div>",
    unsafe_allow_html=True,
)
