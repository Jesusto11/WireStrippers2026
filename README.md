# ECLIPSE

**ECLIPSE** significa *Evaluator of Communications, Landing, and Illumination for Polar Site Exploration*. En español: *Evaluador de comunicaciones, aterrizaje e iluminación para la exploración de sitios polares*.

ECLIPSE es un prototipo de apoyo al análisis preliminar de ubicaciones lunares. Integra efemérides de NASA/JPL Horizons, capas cartográficas de NASA Moon Trek y consultas de altimetría al servicio LRO LOLA DEM Global 128 ppd.

> Los resultados son exploratorios. La visibilidad geométrica no certifica un enlace de radio; la micromalla de elevación no certifica la seguridad de un sitio; y los promedios de waypoints Apollo no son necesariamente coordenadas oficiales de contacto.

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
│   └── lola_api_config.json
└── modules/
    ├── horizons.py
    ├── terrain.py
    ├── optimization.py
    ├── mission_analysis.py
    └── visualization.py
```

**No se requiere descargar, guardar ni subir el archivo `LRO_LOLA_DEM_Global_128ppd_v04.tif`.** La elevación se consulta directamente al ImageServer de NASA Moon Trek.

## 2. Instalación local

Requiere Python 3.12.

```bash
python -m venv .venv
```

En Windows CMD:

```cmd
.venv\Scripts\activate
```

Instala las dependencias y ejecuta la aplicación:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Fuente de altimetría: NASA Moon Trek / LRO LOLA

ECLIPSE consulta el endpoint:

```text
https://trek.nasa.gov/moon/trekarcgis/rest/services/LRO_LOLA_DEM_Global_128ppd_v04/ImageServer/identify
```

La consulta puntual se realiza con `geometry`, `geometryType=esriGeometryPoint` y `f=json`. El valor `value` devuelto por el ImageServer se interpreta en metros, siguiendo el ejemplo de integración utilizado para el producto LOLA 128 ppd.

La función de consulta puntual se almacena en caché en memoria durante la vida del proceso de la aplicación para evitar repetir solicitudes idénticas; al reiniciar Streamlit se vacía esa caché. Si el servicio devuelve `NoData`, un error o no responde, el sitio se reporta como fallido: ECLIPSE no sustituye silenciosamente la altitud por cero.

### Altitud utilizada en Horizons

Para cada sitio, ECLIPSE consulta una elevación puntual y pasa el resultado a JPL Horizons mediante `SITE_COORD` en el orden:

```text
longitud este, latitud, altura en kilómetros
```

La conversión se hace mediante `altitud_metros / 1000`. La documentación de Horizons define la altura geodética como la altura sobre la superficie de referencia del cuerpo, expresada en kilómetros.

**Advertencia sobre el datum vertical:** los datos de LOLA y la superficie de referencia interna de Horizons deben estar en el mismo marco vertical para obtener una posición topocéntrica físicamente rigurosa. La conexión actual transfiere directamente la elevación devuelta por Moon Trek; antes de utilizar resultados en ingeniería de misión se debe verificar la correspondencia entre el datum de la capa LOLA y la superficie de referencia IAU/ME de Horizons.

## 4. Análisis del relieve local

La pestaña **Terreno** consulta una micromalla de 3×3, 5×5 o 7×7 puntos alrededor de la coordenada seleccionada. El tamaño predeterminado es 5×5. Cada punto se consulta al ImageServer; las respuestas puntuales se almacenan en caché en memoria mientras se ejecuta el proceso de la aplicación.

Con el espaciamiento angular nominal de 1/128° y el radio lunar configurado (1,737,400 m), ECLIPSE aproxima las separaciones físicas norte-sur y este-oeste. La dimensión este-oeste se corrige con el coseno de la latitud. A partir de la micromalla calcula:

- Altitud en el punto central.
- Pendiente estimada en el centro y pendiente media de las muestras válidas.
- Desnivel máximo de la micromalla y desviación estándar de elevación.
- Porcentaje de muestras con pendiente estimada inferior a 5°.
- Superficie 3D y mapa de calor de pendientes.

La micromalla es una muestra de puntos de la capa global, no un recorte rasterizado de alta resolución ni un modelo continuo de obstáculos. Los resultados de pendiente son aproximaciones numéricas sensibles al espaciamiento, la latitud, la cobertura y la resolución del producto.

### ¿Por qué se retiró la búsqueda de candidatos?

La versión anterior identificaba parches conectados y calculaba una frontera de Pareto leyendo cientos de píxeles del GeoTIFF local. El endpoint puntual `identify` no entrega, por sí solo, una ventana raster continua. Una micromalla de 5×5 o 7×7 no ofrece cobertura suficiente para delimitar zonas de aterrizaje amplias. Por eso se retiró el buscador de candidatos, en lugar de conservar una función que aparentara tener una resolución o cobertura que la API utilizada no proporciona.

Para recuperar esa función se necesitaría integrar un servicio que permita descargar/subconjuntar áreas del ImageServer o utilizar un DEM regional accesible en la nube.

## 5. Despliegue en Streamlit Community Cloud

- Repositorio GitHub: publica el código y los archivos de configuración.
- Archivo principal: `app.py`.
- Dependencias: `requirements.txt`.
- Versión de Python: 3.12, declarada en `.python-version`.
- No es necesario Git LFS ni incluir el GeoTIFF de aproximadamente 2.2 GB.

El entorno desplegado necesita salida a internet para consultar NASA Moon Trek y JPL Horizons. Si alguno de esos servicios no está disponible, la operación que depende de ese servicio no podrá completarse.

## 6. Módulos

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Interfaz Streamlit y flujo principal de análisis |
| `modules/horizons.py` | Consulta y procesamiento de efemérides de Sol y Tierra |
| `modules/terrain.py` | Consulta de altitud LOLA por API y análisis de micromalla |
| `modules/optimization.py` | Funciones de frontera de Pareto, NSGA-II y Monte Carlo reutilizables |
| `modules/mission_analysis.py` | Métricas temporales y evaluación exploratoria |
| `modules/visualization.py` | Mapa lunar, gráficas Plotly y superficie 3D |

## 7. Fuentes

- NASA Moon Trek: https://trek.nasa.gov/moon/
- NASA Moon Trek ImageServer (LRO LOLA DEM Global 128 ppd): https://trek.nasa.gov/moon/trekarcgis/rest/services/LRO_LOLA_DEM_Global_128ppd_v04/ImageServer
- JPL Horizons API: https://ssd-api.jpl.nasa.gov/doc/horizons.html
- JPL Horizons Manual: https://ssd.jpl.nasa.gov/horizons/manual.html
- NASA LOLA / Lunar Reconnaissance Orbiter: https://science.nasa.gov/mission/lunar-reconnaissance-orbiter/

## 8. Limitaciones que deben conservarse en la interpretación

- La elevación terrestre sobre el horizonte no demuestra por sí sola que exista un enlace RF utilizable.
- La micromalla de altitud no modela el perfil de horizonte por azimut, sombras topográficas completas, obstáculos subpíxel ni la dispersión del vehículo.
- La pendiente local estimada no certifica la estabilidad del módulo de aterrizaje.
- Los waypoints Apollo sirven como referencias de recorrido; sus promedios no deben presentarse como coordenadas oficiales de contacto.
- Las regiones Artemis del catálogo son nombres históricos de estudio; no se les asignan coordenadas inventadas.
- La disponibilidad solar no equivale a una predicción completa de potencia eléctrica o temperatura.
