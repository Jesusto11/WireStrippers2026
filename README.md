# ECLIPSE

**ECLIPSE** corresponde al nombre *Evaluator of Communications, Landing, and Illumination for Polar Site Exploration*.  
Su significado en español es: *Evaluador de comunicaciones, aterrizaje e iluminación para la exploración de sitios polares*.

ECLIPSE es un prototipo de apoyo para el análisis preliminar de ubicaciones lunares. Integra efemérides de NASA/JPL Horizons, capas cartográficas de NASA Moon Trek y procesamiento local de modelos digitales de elevación LOLA.

> Los resultados son exploratorios. La visibilidad geométrica no certifica un enlace de radio; los filtros de pendiente y rugosidad no certifican la seguridad de un sitio; y los promedios de waypoints Apollo no son necesariamente coordenadas oficiales de contacto.

## 1. Estructura del proyecto

```text
ECLIPSE/
├── app.py
├── requirements.txt
├── .python-version
├── .gitattributes
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── sites.json
│   ├── artemis_study_regions.json
│   ├── dem_config.json
│   └── LRO_LOLA_DEM_Global_128ppd_v04.tif  # archivo local grande; no incluido en este ZIP
└── modules/
    ├── horizons.py
    ├── terrain.py
    ├── candidate_search.py
    ├── optimization.py
    ├── mission_analysis.py
    └── visualization.py
```

## 2. Instalación local

Requiere Python 3.12.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 3. Instalar el DEM LOLA

ECLIPSE ya está configurado para buscar automáticamente este archivo:

```text
data/LRO_LOLA_DEM_Global_128ppd_v04.tif
```

Copia el GeoTIFF completo dentro de la carpeta `data/` con ese nombre exacto. No necesitas cargarlo desde la interfaz ni modificar la ruta en cada ejecución. La aplicación mostrará si el archivo fue detectado.

El código de terreno lee ventanas locales del GeoTIFF con Rasterio; no carga intencionalmente todo el DEM global en una matriz. Aun así, el tamaño, la compresión, la organización interna de bloques y los recursos del servidor afectan el rendimiento. Verifica CRS, resolución, datum y unidades verticales del producto antes de interpretar resultados.

### ¿Se puede subir el GeoTIFF de ~2 GB a GitHub?

No lo subas como archivo Git normal. GitHub aplica un límite de 100 MB por objeto en repositorios Git ordinarios. El proyecto incluye `.gitattributes` para rastrear exclusivamente este DEM mediante Git Large File Storage (Git LFS). La documentación de GitHub indica un máximo de 2 GB por archivo para cuentas Free y Pro; comprueba el tamaño exacto antes de intentar subirlo. Si supera ese límite, no podrá almacenarse con Git LFS en esos planes.

Para usar Git LFS desde tu computadora:

1. Instala Git LFS: https://git-lfs.com/.
2. Abre una terminal en la carpeta raíz de ECLIPSE.
3. Ejecuta:

```bash
git lfs install
git lfs track "data/LRO_LOLA_DEM_Global_128ppd_v04.tif"
git add .gitattributes
git add data/LRO_LOLA_DEM_Global_128ppd_v04.tif
git commit -m "Agregar DEM LOLA mediante Git LFS"
git push
```

Si Git muestra que el archivo está ignorado, revisa que el patrón de excepción del `.gitignore` esté presente. El archivo debe quedar dentro de `data/`, no en la raíz.

GitHub Free/Pro incluye actualmente 10 GiB de almacenamiento LFS y 10 GiB de ancho de banda LFS para descargas. Un DEM cercano a 2 GB puede consumir buena parte de esa cuota con varias descargas/despliegues. Revisa el uso en la configuración de facturación de GitHub.

Streamlit Community Cloud admite repositorios que utilizan Git LFS, pero el archivo debe descargarse al entorno de ejecución. Un archivo de este tamaño puede aumentar el tiempo de despliegue y acercar la aplicación a sus límites de almacenamiento/memoria. Si el archivo supera 2 GB, la cuota LFS no alcanza o el despliegue resulta pesado, la alternativa es alojar el DEM en almacenamiento de objetos externo y descargar solo recortes/regiones, o ejecutar ECLIPSE en un servidor propio con almacenamiento suficiente.

### Comprobar el tamaño exacto en Windows

En PowerShell:

```powershell
(Get-Item .\data\LRO_LOLA_DEM_Global_128ppd_v04.tif).Length / 1GB
```

El resultado se expresa en GiB aproximados. No cambies el nombre del archivo sin actualizar `data/dem_config.json`.

## 4. Despliegue en Streamlit Community Cloud

- Repositorio GitHub: sube la carpeta del proyecto conservando su estructura.
- Archivo principal: `app.py`.
- Dependencias: `requirements.txt`.
- Versión de Python: 3.12, declarada en `.python-version`.
- Configuración visual: `.streamlit/config.toml`.
- DEM: disponible en la ruta configurada mediante Git LFS o en el almacenamiento definido por el equipo.

## 5. Módulos

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Interfaz principal en Streamlit y flujo de análisis |
| `modules/horizons.py` | Consulta y procesamiento de efemérides |
| `modules/terrain.py` | Lectura local del DEM, pendiente y rugosidad |
| `modules/candidate_search.py` | Búsqueda de parches topográficos candidatos |
| `modules/optimization.py` | Frontera de Pareto, funciones NSGA-II y Monte Carlo |
| `modules/mission_analysis.py` | Métricas temporales y evaluación exploratoria |
| `modules/visualization.py` | Mapa Folium, gráficas Plotly y superficie 3D |

## 6. Alcance y advertencias

- La elevación aparente del Sol/Tierra se calcula con Horizons para la ubicación lunar introducida.
- La elevación terrestre sobre el horizonte no demuestra por sí sola que exista un enlace RF utilizable.
- La selección automática actual es un filtro topográfico local, no una certificación de sitio.
- Los umbrales de pendiente/rugosidad son configurables y deben justificarse respecto al vehículo y la misión.
- Las regiones Artemis del catálogo son nombres históricos de estudio; no se les asignan coordenadas inventadas.
- La estimación de disponibilidad solar no equivale a predicción de potencia eléctrica o temperatura.

## 7. Fuentes de datos

- NASA/JPL Horizons: https://ssd.jpl.nasa.gov/horizons/
- NASA Moon Trek: https://trek.nasa.gov/moon/
- LOLA PDS: https://imbrium.mit.edu/


## 8. Documentación de almacenamiento y despliegue

- GitHub: límites y uso de Git LFS: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage
- Cuotas de almacenamiento y ancho de banda de Git LFS: https://docs.github.com/es/billing/concepts/product-billing/git-lfs
- Streamlit Community Cloud: organización de archivos y compatibilidad con Git LFS: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- Streamlit Community Cloud: recomendaciones ante límites de recursos: https://docs.streamlit.io/knowledge-base/deploy/resource-limits

**Nota sobre el paquete:** el GeoTIFF de aproximadamente 2 GB no se incluye en este ZIP. Debes copiar tu archivo original a `data/LRO_LOLA_DEM_Global_128ppd_v04.tif`. La aplicación detectará automáticamente esa ruta. Para desplegarlo desde GitHub, utiliza Git LFS y verifica el tamaño exacto del archivo y la cuota disponible de tu cuenta.
