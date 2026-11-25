# VIALNEX INTELLIGENCE 2.0 — Base funcional

**Lema:** El futuro de las vías comienza con datos.

Plataforma académica para registrar inspecciones viales, consultar indicadores PCI,
visualizar tendencias, explorar un mapa base y generar predicciones experimentales.

## Requisitos
- Python 3.10 o superior
- pip

## Instalación en Windows (PowerShell)
```powershell
cd VIALNEX_INTELLIGENCE_2_0
py -m venv entorno
.\entorno\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Abrir: http://127.0.0.1:5000

## Funciones disponibles en esta entrega
- Dashboard con indicadores, gráficos y distribución por condición PCI.
- Registro, consulta, edición mediante API y eliminación de inspecciones.
- Base de datos SQLite.
- Predicción PCI con regresión lineal.
- Comparación exploratoria de regresión lineal y Random Forest (`GET /api/model-comparison`).
- Exportación CSV (`GET /api/export`).
- Mapa base Leaflet con capas satelital y calles.
- Endpoint de estado (`GET /api/health`).

## Endpoints principales
| Método | Ruta | Función |
|---|---|---|
| GET | `/api/dashboard` | Indicadores e historial reciente |
| POST | `/api/inspections` | Registrar inspección |
| GET | `/api/inspections/<id>` | Consultar inspección |
| PUT | `/api/inspections/<id>` | Actualizar inspección |
| DELETE | `/api/inspections/<id>` | Eliminar inspección |
| POST | `/api/predict` | Estimar PCI |
| GET | `/api/model-comparison` | Comparar modelos |
| GET | `/api/export` | Descargar CSV |
| GET | `/api/health` | Verificar servidor y base de datos |

## Consideraciones técnicas importantes
- El PCI ingresado en una inspección es el valor registrado por el evaluador; el PCI predicho es una estimación del modelo. No deben confundirse.
- La base de datos inicial puede contener registros demostrativos. Identifícalos como datos de prueba y reemplázalos por inspecciones verificadas antes de presentar resultados reales.
- Las métricas de ML dependen de la calidad, cantidad y representatividad de los datos. La comparación actual usa una partición train/test y es exploratoria.
- El mapa satelital es una capa cartográfica, no una imagen en tiempo real. En esta versión los registros no están georreferenciados desde el formulario.
- El sistema no incorpora todavía autenticación, roles, imágenes de daños, planificación automatizada de mantenimiento, exportación PDF/Excel ni PWA. Son módulos de las siguientes iteraciones; no se presentan como funciones terminadas.

## Ruta recomendada para completar la versión 2.0
1. Completar edición desde la interfaz y filtros avanzados.
2. Incorporar latitud/longitud en inspecciones y mostrar marcadores en el mapa.
3. Añadir reportes PDF y Excel.
4. Incorporar autenticación segura y permisos por rol.
5. Desarrollar planificación de mantenimiento basada en PCI, criticidad y presupuesto.
6. Entrenar y validar modelos con datos reales; después evaluar visión artificial para grietas y baches.
7. Convertir la interfaz en PWA y añadir un asistente que responda con base en los datos del sistema.

## Ejecución
`python app.py`

Para activar el modo debug únicamente en desarrollo:
```powershell
$env:VIALNEX_DEBUG="1"
python app.py
```
