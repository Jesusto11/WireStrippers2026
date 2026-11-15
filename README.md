# LUNASCOUT — Lunar Landing Site Intelligence

Streamlit application for exploratory comparison of lunar reference sites, apparent Sun/Earth geometry using NASA/JPL Horizons, and local lunar DEM screening.

> **Engineering scope:** LUNASCOUT is a research/educational decision-support prototype. It does not certify a landing site, predict RF link availability, model lander thermal behavior, or replace vehicle-specific landing hazard assessment.

## 1. Run locally

Recommended runtime: **Python 3.12**. The `.python-version` file records this preference for local tools.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The application normally opens at `http://localhost:8501`.

## 2. GitHub / Streamlit Community Cloud deployment

1. Push the complete folder structure to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. In Advanced settings, select Python 3.12 (the `.python-version` file is a local-tool hint; select the cloud runtime in the deployment settings).
5. Deploy; Streamlit installs dependencies from `requirements.txt`.
6. Avoid committing large DEM GeoTIFFs unless repository and hosting limits are appropriate. Use the sidebar uploader or separately managed data storage.

## 3. Project structure

```text
LUNASCOUT/
├── app.py
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── sites.json
│   ├── artemis_study_regions.json
│   └── dem_config.json
└── modules/
    ├── __init__.py
    ├── horizons.py
    ├── terrain.py
    ├── candidate_search.py
    ├── optimization.py
    ├── mission_analysis.py
    └── visualization.py
```

## 4. Module responsibilities

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, mission configuration, tabs, session state and module orchestration |
| `modules/horizons.py` | NASA/JPL Horizons requests, CSV parsing and unified Sun/Earth series |
| `modules/terrain.py` | Local DEM windows, slope, roughness and preliminary topographic candidate patches |
| `modules/candidate_search.py` | Application-facing interface for DEM candidate screening |
| `modules/optimization.py` | Pareto filtering, Monte Carlo position dispersion and optional NSGA-II wrapper |
| `modules/mission_analysis.py` | Duration integration, comparison metrics and exploratory screening labels |
| `modules/visualization.py` | NASA Moon Trek waypoints, Folium map, Plotly timeline, comparison chart and 3D DEM surface |
| `data/sites.json` | Apollo waypoint endpoints, LRO WAC basemap and high-resolution tile metadata |
| `data/artemis_study_regions.json` | Historical Artemis III study-region names, deliberately without unverified coordinates |
| `data/dem_config.json` | DEM defaults and configurable terrain-screening assumptions |

## 5. Current interface functionality

- Select up to five Apollo reference missions and/or one custom coordinate.
- Set common start/end dates and UT times for every selected site.
- Select Horizons sampling interval (5, 15, 30, 60, 120 or 360 minutes).
- Compare integrated geometric Sun visibility, Earth visibility and configurable Earth elevation margin.
- Explore apparent Sun/Earth elevation; hover to inspect azimuth and elevation for each Horizons timestamp.
- Inspect the nearest sample to a selected mission time.
- View selected sites and Apollo waypoints on a NASA Moon Trek basemap.
- Optionally upload a GeoTIFF DEM, inspect a local window in 3D and search preliminary topographic candidate patches.
- Export comparison metrics, one site's Horizons time series and candidate coordinates as CSV.

## 6. Important definitions and assumptions

### Horizons

- Observer center: `coord@301` (Moon center).
- Lunar site coordinates are passed as longitude, latitude, altitude.
- Quantity 4 is used for apparent azimuth/elevation of the Sun (`10`) and Earth (`399`).
- Returned timestamps are interpreted and labeled as UT.
- Durations are integrated between adjacent timestamps; the final sample is extended to the requested stop time when needed.

### Apollo coordinates

Each Apollo reference point is the mean of available Moon Trek waypoint coordinates. It is a convenient reference for visualization/analysis, not necessarily an official touchdown coordinate.

### Earth visibility

- Elevation `> 0°` means geometrically above the local astronomical horizon.
- The configurable Earth elevation margin (default 10°) is an angular screening reference only.
- Neither test confirms RF link margin, antenna pointing, terrain obstruction, multipath, relay availability or ground-network availability.

### Solar metrics

- Solar elevation `> 0°` is geometric direct-Sun visibility before terrain-horizon occlusion.
- The 5°–20° band is retained as an exploratory reference from the earlier prototype, not a universal landing requirement.
- High solar elevation is not a temperature prediction. Thermal evaluation requires a separate thermal model and material properties.

### DEM and terrain candidate screening

- The DEM must have a valid CRS and known horizontal/vertical units. Verify these from the product metadata.
- Set `elevation_units` and `elevation_to_m_factor` in `data/dem_config.json`; elevations are scaled to meters before slope and roughness are calculated. The default factor is 1.0 and assumes that the DEM vertical values are already meters.
- For geographic rasters, local pixel size is approximated on a lunar sphere using the configured lunar radius.
- For projected rasters, CRS linear units are used. Confirm the projection is appropriate for the Moon and its units are suitable.
- Slope and 5×5-neighborhood elevation standard deviation are screening indicators only.
- Candidate patches are not landing-safe declarations. The current screening does not include vehicle footprint, leg geometry, boulder detection, sub-pixel hazards, navigation/landing dispersion, Sun/Earth temporal conditions or mission-specific constraints.

### Artemis site metadata

`data/artemis_study_regions.json` contains names from a historical Artemis III study cycle for reference only. It intentionally does not assign coordinates or imply that those regions are approved for Artemis IV. Add a region to the selectable catalog only after attaching verified coordinates, coordinate reference information and a source.

## 7. Optimization module status

`modules/optimization.py` provides reusable functions:

- `pareto_mask` / `filtrar_frontera_pareto`: non-dominated alternatives for selected objective columns.
- `generar_dispersion_lunar`: local east/north Gaussian perturbations expressed in meters.
- `estimar_probabilidad_factibilidad`: fraction of simulated points passing a caller-supplied boolean evaluator.
- `ejecutar_nsga2`: optional NSGA-II wrapper for a caller-supplied objective function.

The app currently applies Pareto filtering to topographic candidate patches. It does **not** automatically run NSGA-II or Monte Carlo because defensible objective functions and vehicle-specific dispersion/feasibility models must first be defined.

## 8. Suggested validation sequence

1. Run Apollo 11 for 24 hours with 60-minute sampling.
2. Confirm the time-series CSV contains `datetime`, `sun_azimuth_deg`, `sun_elevation_deg`, `earth_azimuth_deg` and `earth_elevation_deg`.
3. Repeat with 15-minute sampling and check that the number of samples increases.
4. Compare Apollo 11 and Apollo 12 over the same UT window.
5. Only then upload a verified LOLA/LROC DEM and test coordinate conversion, slope and roughness against known product metadata or reference calculations.

## 9. Troubleshooting

- **Horizons returns an error:** shorten the interval/date range, verify API availability and inspect the displayed error.
- **No Apollo coordinates:** the Moon Trek waypoint endpoint may be unavailable; retry later or use a custom coordinate with known provenance.
- **DEM coordinate outside coverage:** verify coordinate convention, longitude range, DEM CRS and Moon-specific projection.
- **Unexpected slope/roughness:** verify elevation units, horizontal units, scale factor and nodata encoding.
- **Slow app:** increase the sampling interval, reduce the number of sites and upload a regional DEM instead of a global raster.
