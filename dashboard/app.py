# ==============================================================================
# dashboard/app.py
# Dashboard interactivo para visualización de anomalías en consumo de gas.
# Incluye mejoras en gráficos y documentación.
# ==============================================================================
import os
import yaml
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

# Configuración básica de logging para el dashboard
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Paleta de Colores Contugas (Ejemplo, ajustar según identidad de marca) ---
CONTUGAS_COLORS = {
    'primary_green': '#05a542', 
    'light_green': '#86d03a',   
    'primary_blue': '#3d77dc',  
    'light_blue': '#29a5df',    
    'dark_text': '#231942',     
    'background_main': '#f8f9fa', 
    'background_sidebar': '#e9f5f1', 
    'header_bg': '#004d40', # Un verde oscuro para el encabezado
    'header_text': '#ffffff',
    'card_border': '#d1e7dd', # Borde suave para tarjetas
    'anomaly_alta': '#d9534f', # Rojo para Alta criticidad
    'anomaly_media': '#f0ad4e', # Naranja para Media
    'anomaly_baja': '#5bc0de',  # Azul claro/info para Baja
    'anomaly_muy_baja': '#86d03a', # Verde claro para Muy Baja (si se usa)
    'banda_normal_fill': 'rgba(61,119,220,0.15)' # Relleno para la banda de normalidad (azul claro transparente)
}

# --- 1) CARGA DE CONFIGURACIÓN Y RUTAS DE ARCHIVOS ---
BASE_DIR_APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Directorio raíz del proyecto
CONFIG_PATH_APP = os.path.join(BASE_DIR_APP, 'config.yaml')
CONFIG_APP = {}
try:
    with open(CONFIG_PATH_APP, 'r', encoding='utf-8') as f_cfg_app: 
        CONFIG_APP = yaml.safe_load(f_cfg_app)
    logger.info(f"Configuración cargada desde {CONFIG_PATH_APP} para el dashboard.")
except Exception as e_cfg_app:
    logger.error(f"Error crítico cargando config.yaml para el dashboard: {e_cfg_app}", exc_info=True)
    # Fallback a una configuración mínima si falla la carga
    CONFIG_APP = { 
        'rutas': {'preprocesado': 'data/preprocessed.csv', 'anomalias': 'data/anomalies_detected.csv', 
                  'cluster_assignments': 'data/client_clusters.csv', 'logs': 'logs/anomaly_detection.log', 
                  'resultados_evaluacion': 'evaluation/performance_metrics.csv', 'execution_times': 'logs/execution_times.yaml'},
        'etl': {'columnas_numericas': ['Presion', 'Temperatura', 'Volumen']},
        'feature_engineering': {'gas_law': {'activo': False, 'output_feature_name': 'cantidad_gas_calculada'}},
        'dashboard': {'meses_historia_default': 6, 'metricas_estadisticas': ['mean', 'std', 'min', 'max', 'median'], 'banda_normal_std_factor': 1.5},
        'evaluacion_rendimiento': { # Fallback para criterios de evaluación
            'tasa_falsos_positivos_max': 0.15,
            'deteccion_incidentes_reales_min': 0.80,
            'precision_clasificacion_criticidad_min': 0.90,
            'sensibilidad_min_por_variable': {'Presion': 0.80, 'Temperatura': 0.80, 'Volumen': 0.80, 'cantidad_gas_calculada': 0.80},
            'latencia_procesamiento_max_minutos': 10
        }
    }

RUTAS_CFG_APP = CONFIG_APP.get('rutas', {})
ETL_CFG_APP = CONFIG_APP.get('etl', {})
DASHBOARD_CFG_APP = CONFIG_APP.get('dashboard', {})
FEATURE_ENG_CFG_APP = CONFIG_APP.get('feature_engineering', {})
EVAL_CFG_APP = CONFIG_APP.get('evaluacion_rendimiento', {})


# Lista de features base para mostrar en el dropdown de variables y para el melt
FEATURES_FOR_DISPLAY_APP = ETL_CFG_APP.get('columnas_numericas', ['Presion', 'Temperatura', 'Volumen'])[:]
GAS_LAW_CFG_APP = FEATURE_ENG_CFG_APP.get('gas_law', {})
if GAS_LAW_CFG_APP.get('activo', False):
    gas_feature_name_app = GAS_LAW_CFG_APP.get('output_feature_name', 'cantidad_gas_calculada')
    if gas_feature_name_app not in FEATURES_FOR_DISPLAY_APP:
        FEATURES_FOR_DISPLAY_APP.append(gas_feature_name_app)

# Rutas a los archivos de datos
PREPROCESSED_CSV_PATH_APP = os.path.join(BASE_DIR_APP, RUTAS_CFG_APP.get('preprocesado', 'data/preprocessed.csv'))
ANOMALIES_CSV_PATH_APP = os.path.join(BASE_DIR_APP, RUTAS_CFG_APP.get('anomalias', 'data/anomalies_detected.csv')) 
CLIENT_CLUSTERS_CSV_PATH_APP = os.path.join(BASE_DIR_APP, RUTAS_CFG_APP.get('cluster_assignments', 'data/client_clusters.csv'))
EXEC_TIMES_YAML_PATH_APP = os.path.join(BASE_DIR_APP, RUTAS_CFG_APP.get('execution_times', 'logs/execution_times.yaml')) 
PERF_METRICS_CSV_PATH_APP = os.path.join(BASE_DIR_APP, RUTAS_CFG_APP.get('resultados_evaluacion', 'evaluation/performance_metrics.csv'))

def load_data_app(file_path, is_yaml_file=False, parse_dates_cols_list=None):
    """Función auxiliar para cargar datos desde CSV o YAML con manejo de errores."""
    if not os.path.exists(file_path):
        logger.warning(f"Archivo no encontrado, no se cargará: {file_path}")
        return pd.DataFrame() if not is_yaml_file else {}
    try:
        if is_yaml_file:
            with open(file_path, 'r', encoding='utf-8') as f_yaml: return yaml.safe_load(f_yaml)
        else:
            return pd.read_csv(file_path, parse_dates=parse_dates_cols_list)
    except Exception as e_load:
        logger.error(f"Error cargando archivo {file_path}: {e_load}", exc_info=True)
        return pd.DataFrame() if not is_yaml_file else {}

# --- 2) CARGA Y PREPARACIÓN INICIAL DE DATOS ---
df_preprocessed_wide_app = load_data_app(PREPROCESSED_CSV_PATH_APP, parse_dates_cols_list=['Fecha'])
if not df_preprocessed_wide_app.empty:
    if 'ID_cliente' in df_preprocessed_wide_app.columns and 'cliente_id' not in df_preprocessed_wide_app.columns:
        df_preprocessed_wide_app.rename(columns={'ID_cliente': 'cliente_id'}, inplace=True)
    if 'Fecha' in df_preprocessed_wide_app.columns and 'datetime' not in df_preprocessed_wide_app.columns:
        df_preprocessed_wide_app.rename(columns={'Fecha': 'datetime'}, inplace=True)
    
    df_client_clusters_temp_app = load_data_app(CLIENT_CLUSTERS_CSV_PATH_APP)
    if 'cluster_id' not in df_preprocessed_wide_app.columns and \
       not df_client_clusters_temp_app.empty and \
       'cliente_id' in df_client_clusters_temp_app.columns and \
       'cliente_id' in df_preprocessed_wide_app.columns:
        df_preprocessed_wide_app = pd.merge(df_preprocessed_wide_app, df_client_clusters_temp_app[['cliente_id', 'cluster_id']], on='cliente_id', how='left')
    
    if 'cluster_id' in df_preprocessed_wide_app.columns:
        df_preprocessed_wide_app['cluster_id'] = df_preprocessed_wide_app['cluster_id'].fillna(-1).astype(int)
    else:
        df_preprocessed_wide_app['cluster_id'] = 0 

df_anomalies_app = load_data_app(ANOMALIES_CSV_PATH_APP, parse_dates_cols_list=['Fecha', 'timestamp_deteccion']) 
if not df_anomalies_app.empty:
    if 'cliente' in df_anomalies_app.columns and 'cliente_id' not in df_anomalies_app.columns: 
        df_anomalies_app.rename(columns={'cliente': 'cliente_id'}, inplace=True)
    if 'Fecha' in df_anomalies_app.columns and 'datetime' not in df_anomalies_app.columns:
        df_anomalies_app.rename(columns={'Fecha': 'datetime'}, inplace=True)
    if 'severidad' in df_anomalies_app.columns and 'criticidad_predicha' not in df_anomalies_app.columns: 
        df_anomalies_app.rename(columns={'severidad': 'criticidad_predicha'}, inplace=True)
    if 'anomalia_id' not in df_anomalies_app.columns:
        df_anomalies_app = df_anomalies_app.reset_index().rename(columns={'index':'anomalia_id'})
    
    if 'cluster_id' not in df_anomalies_app.columns and not df_preprocessed_wide_app.empty :
        if 'cliente_id' in df_anomalies_app.columns and 'cliente_id' in df_preprocessed_wide_app.columns and 'cluster_id' in df_preprocessed_wide_app.columns:
            df_clusters_map_app = df_preprocessed_wide_app[['cliente_id', 'cluster_id']].drop_duplicates()
            df_anomalies_app = pd.merge(df_anomalies_app, df_clusters_map_app, on='cliente_id', how='left', suffixes=('_anom', ''))
            if 'cluster_id' in df_anomalies_app.columns:
                 df_anomalies_app['cluster_id'] = df_anomalies_app['cluster_id'].fillna(-1).astype(int)
            else: df_anomalies_app['cluster_id'] = 0 
        else: df_anomalies_app['cluster_id'] = 0 
    elif 'cluster_id' not in df_anomalies_app.columns: 
        df_anomalies_app['cluster_id'] = 0

df_long_display_app_style = pd.DataFrame()
if not df_preprocessed_wide_app.empty and 'datetime' in df_preprocessed_wide_app.columns and 'cliente_id' in df_preprocessed_wide_app.columns:
    value_vars_for_melt_app = [f for f in FEATURES_FOR_DISPLAY_APP if f in df_preprocessed_wide_app.columns]
    id_vars_for_melt_app = ['datetime', 'cliente_id']
    if 'cluster_id' in df_preprocessed_wide_app.columns: id_vars_for_melt_app.append('cluster_id')

    if value_vars_for_melt_app: 
        df_long_display_app_style = df_preprocessed_wide_app.melt(
            id_vars=id_vars_for_melt_app, value_vars=value_vars_for_melt_app,
            var_name='variable', value_name='valor'
        )
    else: logger.warning("No hay variables de valor para transformar a formato largo en df_preprocessed_wide_app.")
else: logger.warning("df_preprocessed_wide_app está vacío o sin columnas clave para transformación a formato largo.")

execution_times_app = load_data_app(EXEC_TIMES_YAML_PATH_APP, is_yaml_file=True) 
df_performance_metrics_app = load_data_app(PERF_METRICS_CSV_PATH_APP) 

available_clients_app = sorted(df_long_display_app_style['cliente_id'].unique()) if not df_long_display_app_style.empty and 'cliente_id' in df_long_display_app_style.columns else []
available_vars_app = sorted(df_long_display_app_style['variable'].unique()) if not df_long_display_app_style.empty and 'variable' in df_long_display_app_style.columns else FEATURES_FOR_DISPLAY_APP
available_criticality_app = ['Todas'] + sorted(df_anomalies_app['criticidad_predicha'].dropna().unique()) if not df_anomalies_app.empty and 'criticidad_predicha' in df_anomalies_app.columns else ['Todas']
available_clusters_app = sorted(df_long_display_app_style['cluster_id'].unique()) if not df_long_display_app_style.empty and 'cluster_id' in df_long_display_app_style.columns else []


# --- 3) INICIALIZACIÓN DE LA APLICACIÓN DASH ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "Contugas - Detección de Anomalías en Consumo de Gas"
server = app.server

# --- 4) DEFINICIÓN DEL LAYOUT DEL DASHBOARD ---
app.layout = html.Div(style={"margin":"0 auto", "maxWidth":"1300px", "backgroundColor": CONTUGAS_COLORS['background_main'], "fontFamily":"'Segoe UI', Arial, sans-serif", "padding":"1rem"}, children=[
    html.Div(
        "Sistema de Detección de Anomalías en Consumo de Gas - Contugas",
        style={"backgroundColor": CONTUGAS_COLORS['header_bg'], "color": CONTUGAS_COLORS['header_text'], "padding":"1rem", "fontSize":"24px", "fontWeight":"bold", "textAlign": "center", "borderRadius":"5px", "marginBottom":"1.5rem"}
    ),
    html.Div(style={"display":"flex", "gap":"1.5rem"}, children=[
        html.Div(style={"width":"280px", "backgroundColor":CONTUGAS_COLORS['background_sidebar'], "padding":"1.5rem", "borderRadius":"8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
            html.H4("Filtros de Visualización", style={"marginBottom":"1.5rem", "color": CONTUGAS_COLORS['dark_text'], "borderBottom": f"2px solid {CONTUGAS_COLORS['primary_green']}", "paddingBottom":"0.5rem"}),
            html.Label("Cliente:", style={"fontWeight":"bold", "color": CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"5px"}),
            dcc.Dropdown(id="filtro-cliente-app", options=[{"label":c, "value":c} for c in available_clients_app], value=available_clients_app[0] if available_clients_app else None, clearable=True, style={"marginBottom":"1rem"}),
            html.Label("Cluster ID:", style={"fontWeight":"bold", "color": CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"5px"}),
            dcc.Dropdown(id="filtro-cluster-app", options=[{"label":f"Cluster {c}", "value":c} for c in available_clusters_app], value=None, clearable=True, placeholder="Todos los Clusters", style={"marginBottom":"1rem"}),
            html.Label("Periodo de Fechas:", style={"fontWeight":"bold", "color": CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"5px"}),
            dcc.DatePickerRange(
                id="filtro-fechas-app",
                min_date_allowed=df_long_display_app_style['datetime'].min().date() if not df_long_display_app_style.empty and 'datetime' in df_long_display_app_style.columns and not df_long_display_app_style['datetime'].dropna().empty else (datetime.now() - timedelta(days=365*2)).date(),
                max_date_allowed=df_long_display_app_style['datetime'].max().date() if not df_long_display_app_style.empty and 'datetime' in df_long_display_app_style.columns and not df_long_display_app_style['datetime'].dropna().empty else datetime.now().date(),
                start_date=(df_long_display_app_style['datetime'].max() - timedelta(days=DASHBOARD_CFG_APP.get('meses_historia_default', 6)*30)).date() if not df_long_display_app_style.empty and 'datetime' in df_long_display_app_style.columns and pd.notna(df_long_display_app_style['datetime'].max()) else (datetime.now() - timedelta(days=180)).date(),
                end_date=df_long_display_app_style['datetime'].max().date() if not df_long_display_app_style.empty and 'datetime' in df_long_display_app_style.columns and pd.notna(df_long_display_app_style['datetime'].max()) else datetime.now().date(),
                display_format='YYYY-MM-DD', style={"marginBottom":"1rem", "width":"100%"}
            ),
            html.Label("Variable Operativa:", style={"fontWeight":"bold", "color": CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"5px"}),
            dcc.Dropdown(id="filtro-variable-app", options=[{"label":v.replace("_"," ").title(),"value":v} for v in available_vars_app], value=available_vars_app[0] if available_vars_app else None, clearable=False, style={"marginBottom":"1rem"}),
            html.Label("Severidad de Anomalía:", style={"fontWeight":"bold", "color": CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"5px"}), 
            dcc.Dropdown(id="filtro-severidad-app", options=[{"label":s,"value":s} for s in available_criticality_app], value="Todas", clearable=False, style={"marginBottom":"1.5rem"}),
            html.Button("Aplicar Filtros", id="btn-aplicar-filtros-app", n_clicks=0, style={"backgroundColor":CONTUGAS_COLORS['primary_green'],"color":"white","border":"none","width":"100%","padding":"0.75rem","borderRadius":"5px","cursor":"pointer","fontWeight":"bold", "fontSize":"16px"}),
            html.Hr(style={"marginTop":"2rem", "marginBottom":"1.5rem", "borderColor": CONTUGAS_COLORS['card_border']}),
            html.H5("Clientes con Anomalías (Periodo)", style={"color": CONTUGAS_COLORS['dark_text'], "fontSize":"1em"}),
            dcc.Loading(html.Div(id="lista-clientes-con-anomalias-app", style={"backgroundColor":"white","border":f"1px solid {CONTUGAS_COLORS['card_border']}","maxHeight":"200px","overflowY":"auto","padding":"0.75rem", "borderRadius":"5px"}))
        ]),
        html.Div(style={"flex":"1","backgroundColor":"white","padding":"1.5rem","borderRadius":"8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
            dcc.Loading(html.Div(id="kpi-cards-container-app", style={"display":"flex","gap":"1rem","marginBottom":"1.5rem", "flexWrap":"wrap"})),
            html.Div(id="barra-info-seleccion-app", style={"backgroundColor":CONTUGAS_COLORS['primary_blue'],"color":"white","padding":"0.75rem","fontWeight":"bold","marginBottom":"1.5rem","borderRadius":"5px", "textAlign":"center", "fontSize":"18px"}),
            dcc.Tabs(id="tabs-main-content-app", value='tab-serie-temporal-anomalias', children=[
                dcc.Tab(label='Serie Temporal y Anomalías', value='tab-serie-temporal-anomalias', children=[ 
                    dcc.Loading(dcc.Graph(id="grafico-serie-temporal-anomalias-app", style={"height":"400px", "marginBottom":"1.5rem"})), 
                    html.Div(style={"display":"flex","gap":"1.5rem","marginTop":"1rem"}, children=[
                        dcc.Loading(html.Div(id="stats-clave-variable-app-container", style={"flex":"1","backgroundColor":CONTUGAS_COLORS['background_sidebar'],"padding":"1rem","border":f"1px solid {CONTUGAS_COLORS['card_border']}", "borderRadius":"5px"})), 
                        html.Div(children=[ 
                            html.Strong("Acciones Recomendadas (Ejemplo):", style={"color":CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"5px"}),
                            html.Ul([html.Li("Verificar estado del medidor."), html.Li("Inspeccionar posibles fugas."), html.Li("Contactar al cliente para clarificación.")], style={"paddingLeft":"20px", "fontSize":"0.9em"})
                        ], style={"flex":"1","backgroundColor":CONTUGAS_COLORS['background_sidebar'],"padding":"1rem","border":f"1px solid {CONTUGAS_COLORS['card_border']}", "borderRadius":"5px"}),
                        html.Div(children=[ 
                            html.Button("Generar Reporte Incidente", id="btn-generar-reporte-incidente-app", style={"backgroundColor":CONTUGAS_COLORS['primary_green'], "color":"white","width":"100%","marginBottom":"0.5rem", "padding":"0.5rem", "border":"none", "borderRadius":"5px", "cursor":"pointer"}),
                            html.Button("Ignorar Alerta", id="btn-ignorar-alerta-app", style={"backgroundColor":"white", "border":f"2px solid {CONTUGAS_COLORS['primary_blue']}", "color":CONTUGAS_COLORS['primary_blue'],"width":"100%","marginBottom":"0.5rem", "padding":"0.5rem", "borderRadius":"5px", "cursor":"pointer"}),
                            html.Button("Posponer Revisión", id="btn-posponer-revision-app", style={"backgroundColor":"white", "border":f"2px solid {CONTUGAS_COLORS['light_blue']}", "color":CONTUGAS_COLORS['light_blue'],"width":"100%", "padding":"0.5rem", "borderRadius":"5px", "cursor":"pointer"})
                        ], style={"width":"230px","display":"flex", "flexDirection":"column", "gap":"0.5rem"})
                    ])
                ]),
                dcc.Tab(label='Análisis de Correlaciones', value='tab-correlaciones', children=[
                    dcc.Loading(dcc.Graph(id='graph-correlation-heatmap-app', style={"height":"450px"}))
                ]),
                dcc.Tab(label='Patrones Temporales Agregados', value='tab-patrones-temporales', children=[
                    dcc.Dropdown(id='dropdown-agregacion-temporal-app',
                                 options=[ 
                                     {'label': 'Promedio por Hora del Día', 'value': 'hora_dia'},
                                     {'label': 'Promedio por Día de la Semana', 'value': 'dia_semana'},
                                     {'label': 'Promedio por Mes', 'value': 'mes'}
                                 ], value='hora_dia', clearable=False, style={'width':'60%', 'margin':'10px auto'}),
                    dcc.Loading(dcc.Graph(id='graph-patrones-temporales-app', style={"height":"450px"}))
                ]),
                dcc.Tab(label='Métricas de Desempeño y Cumplimiento', value='tab-metricas-desempeno', children=[ # Título de pestaña actualizado
                    html.Div(id='metricas-desempeno-container-app', style={'padding':'20px', 'overflowX':'auto'})
                ])
            ]), 
        ]) 
    ]), 
    html.Div(style={"marginTop":"2rem", "padding":"1.5rem", "backgroundColor":"white", "borderRadius":"8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
        html.H4("Tabla Detallada de Anomalías Detectadas", style={"color":CONTUGAS_COLORS['dark_text'], "borderBottom": f"2px solid {CONTUGAS_COLORS['primary_green']}", "paddingBottom":"0.5rem", "marginBottom":"1rem"}), 
        dcc.Loading(dash_table.DataTable(
            id="tabla-detallada-anomalias-app", page_size=10, 
            style_table={"overflowX": "auto"},
            style_cell={'textAlign': 'left', 'padding': '8px', 'minWidth': '100px', 'whiteSpace': 'normal', 'height': 'auto', 'border': '1px solid #eee', 'fontFamily':"'Segoe UI', Arial, sans-serif"},
            style_header={'backgroundColor': CONTUGAS_COLORS['background_sidebar'], 'fontWeight': 'bold', 'color': CONTUGAS_COLORS['dark_text'], 'border': '1px solid #ddd'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}],
            filter_action="native", sort_action="native", sort_mode="multi",
        )),
        html.Button("Descargar Tabla CSV", id="btn-descargar-tabla-csv-app", style={"backgroundColor":CONTUGAS_COLORS['primary_blue'],"color":"white","border":"none","padding":"0.75rem","borderRadius":"5px","marginTop":"1rem","cursor":"pointer"}), 
        dcc.Download(id="download-csv-tabla-app") 
    ])
])

# --- 5) DEFINICIÓN DE CALLBACKS ---

def safe_aggregator_app(series_data, func_or_name_agg):
    """Agregador seguro que maneja errores y nombres de función especiales."""
    if series_data.empty: return np.nan
    try:
        if callable(func_or_name_agg):
            return func_or_name_agg(series_data)
        elif isinstance(func_or_name_agg, str):
            if hasattr(series_data, func_or_name_agg): 
                return getattr(series_data, func_or_name_agg)()
            elif func_or_name_agg == 'quantile_25': return series_data.quantile(0.25)
            elif func_or_name_agg == 'quantile_75': return series_data.quantile(0.75)
        logger.warning(f"Función de agregación '{func_or_name_agg}' no reconocida o no aplicable en safe_aggregator_app.")
    except Exception as e_agg:
        logger.error(f"Error en safe_aggregator_app con función '{func_or_name_agg}': {e_agg}")
    return np.nan

def generate_compliance_report_md(df_metrics, exec_times, eval_config):
    """Genera un reporte de cumplimiento en formato Markdown."""
    if df_metrics is None or df_metrics.empty:
        return "No hay datos de métricas de rendimiento para generar el reporte de cumplimiento."

    metric_data = df_metrics.iloc[0].to_dict() # Asumir que es una fila
    
    report_lines = ["### Reporte de Cumplimiento de Requerimientos Medibles",
                    "| ID Req. | Requerimiento                                  | Métrica/KPI Actual                                 | Criterio Aceptación | Cumple |",
                    "|:--------|:-----------------------------------------------|:---------------------------------------------------|:--------------------|:-------|"]

    # N2: Minimizar pérdidas no técnicas (detección de incidentes)
    n2_actual = metric_data.get('recall_anomalia_sensibilidad', None)
    n2_req = eval_config.get('deteccion_incidentes_reales_min', 0.80)
    n2_cumple = "Sí" if n2_actual is not None and n2_actual >= n2_req else ("No" if n2_actual is not None else "N/A")
    n2_actual_str = f"{n2_actual*100:.1f}%" if n2_actual is not None else "N/A"
    report_lines.append(f"| N2      | Minimizar pérdidas no técnicas (detección) | Sensibilidad General: {n2_actual_str}                | ≥ {n2_req*100:.0f}%             | {n2_cumple}  |")

    # N3: Priorización de anomalías críticas
    n3_actual = metric_data.get('precision_clasificacion_criticidad', None)
    n3_req = eval_config.get('precision_clasificacion_criticidad_min', 0.90)
    n3_cumple = "Sí" if n3_actual is not None and n3_actual >= n3_req else ("No" if n3_actual is not None else "N/A")
    n3_actual_str = f"{n3_actual*100:.1f}%" if n3_actual is not None else "N/A"
    report_lines.append(f"| N3      | Precisión en clasificación de criticidad     | Precisión Criticidad: {n3_actual_str}              | ≥ {n3_req*100:.0f}%             | {n3_cumple}  |")

    # D1: Baja tasa de falsos positivos
    d1_actual = metric_data.get('tasa_falsos_positivos', None)
    d1_req = eval_config.get('tasa_falsos_positivos_max', 0.15)
    d1_cumple = "Sí" if d1_actual is not None and d1_actual <= d1_req else ("No" if d1_actual is not None else "N/A")
    d1_actual_str = f"{d1_actual*100:.1f}%" if d1_actual is not None else "N/A"
    report_lines.append(f"| D1      | Baja tasa de falsos positivos                | Tasa Falsos Positivos: {d1_actual_str}             | < {d1_req*100:.0f}%             | {d1_cumple}  |")

    # D2: Sensibilidad por variable (General como proxy)
    # Este es un placeholder ya que la sensibilidad por variable específica no se calcula de forma granular.
    # Se usa la sensibilidad general como un indicador.
    d2_actual = metric_data.get('recall_anomalia_sensibilidad', None) # Usando sensibilidad general
    d2_req = eval_config.get('sensibilidad_min_por_variable', {}).get('Volumen', 0.80) # Ejemplo con Volumen
    d2_cumple = "Sí" if d2_actual is not None and d2_actual >= d2_req else ("No" if d2_actual is not None else "N/A")
    d2_actual_str = f"{d2_actual*100:.1f}%" if d2_actual is not None else "N/A"
    report_lines.append(f"| D2      | Detectar anomalías en variables (General)  | Sensibilidad General: {d2_actual_str}                | ≥ {d2_req*100:.0f}% (ej. Vol) | {d2_cumple}  |")
    
    # D3: Latencia de procesamiento
    d3_actual_seconds = exec_times.get('anomaly_detection_seconds', None) if exec_times else None
    d3_req_min = eval_config.get('latencia_procesamiento_max_minutos', 10)
    d3_cumple = "N/A"
    d3_actual_str = "N/A"
    if d3_actual_seconds is not None:
        d3_actual_min = d3_actual_seconds / 60
        d3_actual_str = f"{d3_actual_min:.2f} min"
        d3_cumple = "Sí" if d3_actual_min <= d3_req_min else "No"
    report_lines.append(f"| D3      | Latencia de procesamiento de alertas         | Latencia Detección: {d3_actual_str}                | ≤ {d3_req_min} min          | {d3_cumple}  |")

    return "\n".join(report_lines)


@app.callback(
    [Output("grafico-serie-temporal-anomalias-app", "figure"),
     Output("tabla-detallada-anomalias-app", "columns"), 
     Output("tabla-detallada-anomalias-app", "data"),
     Output("kpi-cards-container-app","children"),
     Output("barra-info-seleccion-app", "children"),
     Output("lista-clientes-con-anomalias-app","children"),
     Output("stats-clave-variable-app-container","children"),
     Output('filtro-cliente-app', 'options'), 
     Output('filtro-cliente-app', 'value'),   
     Output('graph-correlation-heatmap-app', 'figure'),
     Output('graph-patrones-temporales-app', 'figure'),
     Output('metricas-desempeno-container-app', 'children')
     ],
    [Input("btn-aplicar-filtros-app","n_clicks"), 
     Input("dropdown-agregacion-temporal-app", "value")
    ], 
    [State("filtro-cliente-app","value"), State("filtro-cluster-app","value"),
     State("filtro-fechas-app","start_date"), State("filtro-fechas-app","end_date"),
     State("filtro-variable-app","value"), State("filtro-severidad-app","value")]
)
def actualizar_dashboard_completo_app(n_clicks_btn, agg_temp_col_selected, 
                                   cliente_seleccionado_state, cluster_seleccionado_state, 
                                   fecha_inicio_str_state, fecha_fin_str_state, 
                                   variable_seleccionada_state, severidad_seleccionada_state):
    empty_figure_default = go.Figure().update_layout(title_text='No hay datos para los filtros seleccionados.', paper_bgcolor='white', plot_bgcolor='white')
    no_data_message_html = html.Div("No hay datos disponibles para mostrar con los filtros actuales.", style={'textAlign':'center', 'padding':'20px', 'fontSize':'18px', 'color':CONTUGAS_COLORS['dark_text']})
    default_cliente_options_app = [{'label':c, 'value':c} for c in available_clients_app] if available_clients_app else []
    default_cliente_value_app = cliente_seleccionado_state 
    kpi_placeholders_list = [
        html.Div([html.H5(kpi_name, style={'color':CONTUGAS_COLORS['dark_text'], 'margin':'0 0 5px 0', 'fontSize':'0.9em'}), html.P("N/A", style={'fontSize':'22px', 'fontWeight':'bold', 'color':CONTUGAS_COLORS['primary_green']})], className="kpi-card", style={'border': f"1px solid {CONTUGAS_COLORS['card_border']}", 'padding': '15px', 'textAlign': 'center', 'flex':'1', 'minWidth':'120px', 'backgroundColor':'#fff', 'borderRadius':'5px'}) 
        for kpi_name in ["Total Clientes", "Clientes c/ Anom.", "Alertas Activas", "Alta Criticidad"]
    ]

    if df_long_display_app_style.empty and df_anomalies_app.empty:
        return (empty_figure_default, [], [], kpi_placeholders_list, "N/A", 
                [no_data_message_html], no_data_message_html, 
                default_cliente_options_app, default_cliente_value_app, 
                empty_figure_default, empty_figure_default, no_data_message_html)

    min_fecha_datos_app = df_long_display_app_style['datetime'].min() if not df_long_display_app_style.empty and 'datetime' in df_long_display_app_style.columns and not df_long_display_app_style['datetime'].dropna().empty else None
    max_fecha_datos_app = df_long_display_app_style['datetime'].max() if not df_long_display_app_style.empty and 'datetime' in df_long_display_app_style.columns and not df_long_display_app_style['datetime'].dropna().empty else None
    start_date_dt = pd.to_datetime(fecha_inicio_str_state).normalize() if fecha_inicio_str_state and pd.notna(fecha_inicio_str_state) else (min_fecha_datos_app.normalize() if pd.notna(min_fecha_datos_app) else (datetime.now() - timedelta(days=180)).normalize())
    end_date_dt = (pd.to_datetime(fecha_fin_str_state).normalize() + timedelta(days=1) - timedelta(seconds=1)) if fecha_fin_str_state and pd.notna(fecha_fin_str_state) else ((max_fecha_datos_app.normalize() + timedelta(days=1) - timedelta(seconds=1)) if pd.notna(max_fecha_datos_app) else (datetime.now() + timedelta(days=1) - timedelta(seconds=1)).normalize())

    dff_series_temporales_filtrado = df_long_display_app_style.copy()
    if not dff_series_temporales_filtrado.empty and 'datetime' in dff_series_temporales_filtrado.columns:
        dff_series_temporales_filtrado = dff_series_temporales_filtrado[
            (dff_series_temporales_filtrado['datetime'] >= start_date_dt) & 
            (dff_series_temporales_filtrado['datetime'] <= end_date_dt)
        ]
    
    current_cliente_options_dinamico = default_cliente_options_app
    current_cliente_value_dinamico = cliente_seleccionado_state

    if cluster_seleccionado_state is not None and 'cluster_id' in dff_series_temporales_filtrado.columns:
        dff_series_temporales_filtrado = dff_series_temporales_filtrado[dff_series_temporales_filtrado['cluster_id'] == cluster_seleccionado_state]
        clientes_en_cluster_filtrado = sorted(dff_series_temporales_filtrado['cliente_id'].unique()) if 'cliente_id' in dff_series_temporales_filtrado.columns else []
        current_cliente_options_dinamico = [{'label': c, 'value': c} for c in clientes_en_cluster_filtrado]
        if cliente_seleccionado_state not in clientes_en_cluster_filtrado:
            current_cliente_value_dinamico = clientes_en_cluster_filtrado[0] if clientes_en_cluster_filtrado else None
            
    dff_series_temporales_cliente_actual_aggs = dff_series_temporales_filtrado.copy() 
    if current_cliente_value_dinamico and 'cliente_id' in dff_series_temporales_filtrado.columns:
        dff_series_temporales_filtrado = dff_series_temporales_filtrado[dff_series_temporales_filtrado['cliente_id'] == current_cliente_value_dinamico]
    
    dff_series_temporales_variable_actual = pd.DataFrame() 
    if variable_seleccionada_state and 'variable' in dff_series_temporales_filtrado.columns:
        dff_series_temporales_variable_actual = dff_series_temporales_filtrado[dff_series_temporales_filtrado['variable'] == variable_seleccionada_state] 

    dff_anomalias_filtrado = df_anomalies_app.copy()
    if not dff_anomalias_filtrado.empty:
        if 'datetime' in dff_anomalias_filtrado.columns: 
            dff_anomalias_filtrado = dff_anomalias_filtrado[
                (dff_anomalias_filtrado['datetime'] >= start_date_dt) & 
                (dff_anomalias_filtrado['datetime'] <= end_date_dt)
            ]
        if cluster_seleccionado_state is not None and 'cluster_id' in dff_anomalias_filtrado.columns:
            dff_anomalias_filtrado = dff_anomalias_filtrado[dff_anomalias_filtrado['cluster_id'] == cluster_seleccionado_state]
        if current_cliente_value_dinamico and 'cliente_id' in dff_anomalias_filtrado.columns:
            dff_anomalias_filtrado = dff_anomalias_filtrado[dff_anomalias_filtrado['cliente_id'] == current_cliente_value_dinamico]
        if severidad_seleccionada_state != 'Todas' and 'criticidad_predicha' in dff_anomalias_filtrado.columns:
            dff_anomalias_filtrado = dff_anomalias_filtrado[dff_anomalias_filtrado['criticidad_predicha'] == severidad_seleccionada_state]
    
    kpis_generados_app = []
    kpis_generados_app.append(html.Div([html.Div("Total Clientes", style={"fontWeight":"bold", "color":CONTUGAS_COLORS['dark_text'], "fontSize":"0.9em"}), html.Div(f"{len(available_clients_app)}", style={"fontSize":"22px","fontWeight":"bold", "color":CONTUGAS_COLORS['primary_green']})], style={"width":"23%","minWidth":"120px","height":"80px","backgroundColor":"#fff","border":f"1px solid {CONTUGAS_COLORS['card_border']}","borderRadius":"5px","padding":"0.5rem","textAlign":"center", "boxSizing":"border-box"}))
    num_clientes_con_anomalias_app = dff_anomalias_filtrado['cliente_id'].nunique() if not dff_anomalias_filtrado.empty and 'cliente_id' in dff_anomalias_filtrado else 0
    kpis_generados_app.append(html.Div([html.Div("Con Anomalías", style={"fontWeight":"bold", "color":CONTUGAS_COLORS['dark_text'], "fontSize":"0.9em"}), html.Div(f"{num_clientes_con_anomalias_app}", style={"fontSize":"22px","fontWeight":"bold", "color":CONTUGAS_COLORS['primary_blue']})], style={"width":"23%","minWidth":"120px","height":"80px","backgroundColor":"#fff","border":f"1px solid {CONTUGAS_COLORS['card_border']}","borderRadius":"5px","padding":"0.5rem","textAlign":"center", "boxSizing":"border-box"}))
    counts_criticidad_app = dff_anomalias_filtrado['criticidad_predicha'].value_counts() if not dff_anomalias_filtrado.empty and 'criticidad_predicha' in dff_anomalias_filtrado else pd.Series(dtype='int')
    kpis_generados_app.append(html.Div([html.Div("Alta Severidad", style={"fontWeight":"bold", "color":CONTUGAS_COLORS['anomaly_alta'], "fontSize":"0.9em"}), html.Div(f"{counts_criticidad_app.get('Alta',0)}", style={"fontSize":"22px","fontWeight":"bold", "color":CONTUGAS_COLORS['anomaly_alta']})], style={"width":"23%","minWidth":"120px","height":"80px","backgroundColor":"#fff","border":f"1px solid {CONTUGAS_COLORS['card_border']}","borderRadius":"5px","padding":"0.5rem","textAlign":"center", "boxSizing":"border-box"}))
    total_alertas_activas_app = len(dff_anomalias_filtrado)
    kpis_generados_app.append(html.Div([html.Div("Alertas Activas", style={"fontWeight":"bold", "color":CONTUGAS_COLORS['dark_text'], "fontSize":"0.9em"}), html.Div(f"{total_alertas_activas_app}", style={"fontSize":"22px","fontWeight":"bold", "color":CONTUGAS_COLORS['light_blue']})], style={"width":"23%","minWidth":"120px","height":"80px","backgroundColor":"#fff","border":f"1px solid {CONTUGAS_COLORS['card_border']}","borderRadius":"5px","padding":"0.5rem","textAlign":"center", "boxSizing":"border-box"}))

    info_cliente_actual_txt_app = current_cliente_value_dinamico if current_cliente_value_dinamico else "Todos los Clientes"
    info_cluster_actual_txt_app = f"Cluster {cluster_seleccionado_state}" if cluster_seleccionado_state is not None else "Todos los Clusters"
    info_variable_actual_txt_app = variable_seleccionada_state.replace('_',' ').title() if variable_seleccionada_state else "N/A"
    barra_info_texto_app = f"Mostrando: {info_variable_actual_txt_app} para {info_cliente_actual_txt_app} ({info_cluster_actual_txt_app})"
    if severidad_seleccionada_state != 'Todas': barra_info_texto_app += f" | Severidad Filtrada: {severidad_seleccionada_state}"

    fig_serie_temporal_anom_app = go.Figure()
    if not dff_series_temporales_variable_actual.empty and 'datetime' in dff_series_temporales_variable_actual.columns and 'valor' in dff_series_temporales_variable_actual.columns:
        df_plot_serie_app = dff_series_temporales_variable_actual.sort_values(by='datetime')
        fig_serie_temporal_anom_app.add_trace(go.Scatter(
            x=df_plot_serie_app['datetime'], y=df_plot_serie_app['valor'], mode='lines', 
            name=variable_seleccionada_state.replace("_"," ").title(), 
            line=dict(color=CONTUGAS_COLORS['primary_blue'], width=2)
        ))
        if not df_plot_serie_app.empty and 'valor' in df_plot_serie_app.columns:
            mean_val_banda = df_plot_serie_app['valor'].mean()
            std_val_banda = df_plot_serie_app['valor'].std()
            banda_std_factor = DASHBOARD_CFG_APP.get('banda_normal_std_factor', 1.5) 
            if pd.notna(mean_val_banda) and pd.notna(std_val_banda) and std_val_banda > 0: 
                upper_bound_banda = mean_val_banda + banda_std_factor * std_val_banda
                lower_bound_banda = mean_val_banda - banda_std_factor * std_val_banda
                fig_serie_temporal_anom_app.add_trace(go.Scatter(
                    x=df_plot_serie_app['datetime'], y=[upper_bound_banda] * len(df_plot_serie_app),
                    mode='lines', line=dict(width=0), showlegend=False, name='Límite Superior Normal'
                ))
                fig_serie_temporal_anom_app.add_trace(go.Scatter(
                    x=df_plot_serie_app['datetime'], y=[lower_bound_banda] * len(df_plot_serie_app),
                    mode='lines', line=dict(width=0), fill='tonexty', 
                    fillcolor=CONTUGAS_COLORS.get('banda_normal_fill', 'rgba(0,100,80,0.2)'), 
                    showlegend=False, name='Límite Inferior Normal'
                ))
        anomalias_para_grafico_app = dff_anomalias_filtrado.copy() 
        if not anomalias_para_grafico_app.empty and variable_seleccionada_state and variable_seleccionada_state in anomalias_para_grafico_app.columns:
            anom_plot_data_app = anomalias_para_grafico_app.dropna(subset=['datetime', variable_seleccionada_state, 'criticidad_predicha'])
            if not anom_plot_data_app.empty:
                color_map_criticidad_app = {'Alta': CONTUGAS_COLORS['anomaly_alta'], 'Media': CONTUGAS_COLORS['anomaly_media'], 'Baja': CONTUGAS_COLORS['anomaly_baja'], 'Muy Baja': CONTUGAS_COLORS['anomaly_muy_baja'], 'Indeterminada': 'grey'}
                customdata_list_app = []
                for _, row_anom in anom_plot_data_app.iterrows():
                    data_tuple_anom = (
                        row_anom.get('criticidad_predicha', 'N/A'),
                        row_anom.get('score_anomalia_final', np.nan),
                        row_anom.get('modelo_deteccion', 'N/A')
                    )
                    customdata_list_app.append(data_tuple_anom)
                fig_serie_temporal_anom_app.add_trace(go.Scatter(
                    x=anom_plot_data_app['datetime'], y=anom_plot_data_app[variable_seleccionada_state], mode='markers',
                    marker=dict(size=10, symbol='x', line=dict(width=2),
                                color=[color_map_criticidad_app.get(c, 'grey') for c in anom_plot_data_app['criticidad_predicha']]),
                    name='Anomalía Detectada',
                    customdata=np.array(customdata_list_app, dtype=object),
                    hovertemplate=(
                        "<b>Anomalía Detectada</b><br>" +
                        "Fecha: %{x|%Y-%m-%d %H:%M}<br>" +
                        "Valor: %{y:.2f}<br>" +
                        "Criticidad: %{customdata[0]}<br>" +
                        "Score: %{customdata[1]:.3f}<br>" +
                        "Modelo: %{customdata[2]}" +
                        "<extra></extra>" 
                    )
                ))
        fig_serie_temporal_anom_app.update_layout(
            title_text=f"Serie Temporal de {variable_seleccionada_state.replace('_',' ').title()} para Cliente: {current_cliente_value_dinamico or 'Todos'}",
            xaxis_title="Fecha y Hora", 
            yaxis_title=variable_seleccionada_state.replace("_"," ").title(),
            margin=dict(l=40,r=20,t=50,b=40), 
            paper_bgcolor='white', plot_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else: 
        fig_serie_temporal_anom_app.update_layout(
            title_text=f"Seleccione cliente y variable para mostrar datos.", 
            xaxis_title='Fecha y Hora', 
            yaxis_title=variable_seleccionada_state.replace("_"," ").title() if variable_seleccionada_state else "Valor",
            paper_bgcolor='white', plot_bgcolor='white'
        )

    stats_clave_html_children = [html.Strong(f"Estadísticas Clave para {variable_seleccionada_state.replace('_',' ').title()}:", style={"color":CONTUGAS_COLORS['dark_text'], "display":"block", "marginBottom":"8px"})]
    if not dff_series_temporales_variable_actual.empty and 'valor' in dff_series_temporales_variable_actual.columns: 
        data_series_para_stats_app = dff_series_temporales_variable_actual['valor'].dropna()
        if not data_series_para_stats_app.empty:
            metric_lambdas_app = {'mean': 'mean', 'std': 'std', 'min': 'min', 'max': 'max', 'median': 'median', 'quantile_25': lambda x: x.quantile(0.25), 'quantile_75': lambda x: x.quantile(0.75)}
            metric_display_names_config = DASHBOARD_CFG_APP.get('metricas_estadisticas', ['mean', 'std', 'min', 'max', 'median'])
            stats_list_items_html = []
            for name_key_metric in metric_display_names_config:
                func_to_apply_metric = metric_lambdas_app.get(name_key_metric, name_key_metric)
                try:
                    value_metric = safe_aggregator_app(data_series_para_stats_app, func_to_apply_metric)
                    display_name_metric = name_key_metric.replace('_',' ').title()
                    stats_list_items_html.append(html.Li(f"{display_name_metric}: {value_metric:.2f}" if isinstance(value_metric, (float, int)) else f"{display_name_metric}: {value_metric}", style={'padding':'3px 0', 'fontSize':'0.9em'}))
                except: stats_list_items_html.append(html.Li(f"{name_key_metric.replace('_',' ').title()}: N/A", style={'padding':'3px 0', 'fontSize':'0.9em'}))
            stats_clave_html_children.append(html.Ul(stats_list_items_html, style={"paddingLeft":"0px", "listStyleType":"none", "marginTop":"5px"}))
        else: stats_clave_html_children.append(html.P("No hay datos para calcular estadísticas.", style={'fontSize':'0.9em'}))
    else: stats_clave_html_children.append(html.P("Seleccione cliente y variable para ver estadísticas.", style={'fontSize':'0.9em'}))
    stats_clave_container_html = html.Div(stats_clave_html_children)

    lista_clientes_con_anomalias_html = [html.P("No hay clientes con anomalías en el periodo y filtros seleccionados.", style={'textAlign':'center', 'padding':'10px', 'fontSize':'0.9em'})]
    if not dff_anomalias_filtrado.empty and 'cliente_id' in dff_anomalias_filtrado.columns:
        clientes_con_anom_sidebar = dff_anomalias_filtrado.groupby('cliente_id')['criticidad_predicha'].agg(
            lambda s: s.value_counts().index[0] if not s.empty else 'N/A' 
        ).reset_index()
        if not clientes_con_anom_sidebar.empty:
            lista_clientes_con_anomalias_html = [
                html.Div(f"• {row_cli['cliente_id']} (Crit. Frec: {row_cli['criticidad_predicha']})", style={'padding': '4px 0', 'borderBottom': '1px solid #eee', 'color': CONTUGAS_COLORS['dark_text'], 'fontSize':'14px'}) 
                for _, row_cli in clientes_con_anom_sidebar.iterrows()]

    tabla_anomalias_cols_app, tabla_anomalias_data_app = [], []
    if not dff_anomalias_filtrado.empty:
        df_tabla_anom_display = dff_anomalias_filtrado.copy()
        cols_para_tabla = ['anomalia_id', 'datetime', 'cliente_id', 'criticidad_predicha', 'modelo_deteccion', 'score_anomalia_final']
        for var_op in FEATURES_FOR_DISPLAY_APP:
            if var_op in df_tabla_anom_display.columns and var_op not in cols_para_tabla:
                cols_para_tabla.append(var_op)
        df_tabla_anom_display = df_tabla_anom_display[[col for col in cols_para_tabla if col in df_tabla_anom_display.columns]] 
        if 'datetime' in df_tabla_anom_display.columns: 
            df_tabla_anom_display['datetime'] = pd.to_datetime(df_tabla_anom_display['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        for col_float_format in df_tabla_anom_display.select_dtypes(include='float64').columns:
            df_tabla_anom_display[col_float_format] = df_tabla_anom_display[col_float_format].round(3)
        tabla_anomalias_cols_app = [{"name": col.replace("_"," ").title(), "id": col} for col in df_tabla_anom_display.columns]
        tabla_anomalias_data_app = df_tabla_anom_display.to_dict('records')

    # --- Pestaña: Análisis de Correlaciones ---
    fig_corr_heatmap_app = go.Figure()
    if not df_preprocessed_wide_app.empty: # Usar datos wide para correlaciones
        df_corr_source_app = df_preprocessed_wide_app.copy()
        if 'datetime' in df_corr_source_app.columns: 
            df_corr_source_app = df_corr_source_app[
                (df_corr_source_app['datetime'] >= start_date_dt) & 
                (df_corr_source_app['datetime'] <= end_date_dt) # Completar la condición de fecha
            ]
        if cluster_seleccionado_state is not None and 'cluster_id' in df_corr_source_app.columns:
            df_corr_source_app = df_corr_source_app[df_corr_source_app['cluster_id'] == cluster_seleccionado_state]
        if current_cliente_value_dinamico and 'cliente_id' in df_corr_source_app.columns:
            df_corr_source_app = df_corr_source_app[df_corr_source_app['cliente_id'] == current_cliente_value_dinamico]
        
        cols_corr_app = [v for v in FEATURES_FOR_DISPLAY_APP if v in df_corr_source_app.columns] # Solo columnas numéricas base
        if len(cols_corr_app) > 1 and not df_corr_source_app[cols_corr_app].empty:
            corr_matrix_app = df_corr_source_app[cols_corr_app].corr()
            fig_corr_heatmap_app = px.imshow(corr_matrix_app, text_auto=".2f", aspect="auto", 
                                         title=f"Heatmap de Correlación de Variables ({current_cliente_value_dinamico or 'Agregado'})",
                                         color_continuous_scale=px.colors.diverging.RdBu_r, # Paleta invertida para +1 rojo
                                         color_continuous_midpoint=0, 
                                         labels=dict(color="Correlación"))
            fig_corr_heatmap_app.update_layout(margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor='white', plot_bgcolor='white')
        else: fig_corr_heatmap_app.update_layout(title_text='No suficientes variables/datos para calcular correlación.')
    else: fig_corr_heatmap_app.update_layout(title_text='No hay datos para el heatmap de correlación.')

    # --- Pestaña: Patrones Temporales Agregados ---
    fig_patrones_temporales_app = go.Figure()
    # Usar df_preprocessed_wide_app filtrado por fecha, cluster, cliente
    df_temp_pattern_source_app = df_preprocessed_wide_app.copy()
    if 'datetime' in df_temp_pattern_source_app.columns:
        df_temp_pattern_source_app = df_temp_pattern_source_app[
            (df_temp_pattern_source_app['datetime'] >= start_date_dt) &
            (df_temp_pattern_source_app['datetime'] <= end_date_dt)
        ]
    if cluster_seleccionado_state is not None and 'cluster_id' in df_temp_pattern_source_app.columns:
        df_temp_pattern_source_app = df_temp_pattern_source_app[df_temp_pattern_source_app['cluster_id'] == cluster_seleccionado_state]
    if current_cliente_value_dinamico and 'cliente_id' in df_temp_pattern_source_app.columns:
        df_temp_pattern_source_app = df_temp_pattern_source_app[df_temp_pattern_source_app['cliente_id'] == current_cliente_value_dinamico]

    if not df_temp_pattern_source_app.empty and variable_seleccionada_state and agg_temp_col_selected:
        if variable_seleccionada_state in df_temp_pattern_source_app.columns and agg_temp_col_selected in df_temp_pattern_source_app.columns:
            # Asegurar que la columna de agregación no tenga NaNs para el groupby
            df_temp_pattern_source_app_agg = df_temp_pattern_source_app.dropna(subset=[agg_temp_col_selected, variable_seleccionada_state])
            if not df_temp_pattern_source_app_agg.empty:
                agg_data_app = df_temp_pattern_source_app_agg.groupby(agg_temp_col_selected)[variable_seleccionada_state].agg(['mean', 'std']).reset_index()
                agg_data_app['std'] = agg_data_app['std'].fillna(0) # std puede ser NaN si hay un solo punto por grupo
                agg_data_app['upper_bound'] = agg_data_app['mean'] + agg_data_app['std']
                agg_data_app['lower_bound'] = agg_data_app['mean'] - agg_data_app['std']
                
                fig_patrones_temporales_app.add_trace(go.Scatter(x=agg_data_app[agg_temp_col_selected], y=agg_data_app['mean'], mode='lines+markers', name=f'Promedio {variable_seleccionada_state.replace("_"," ").title()}', line=dict(color=CONTUGAS_COLORS['primary_green'])))
                fig_patrones_temporales_app.add_trace(go.Scatter(x=agg_data_app[agg_temp_col_selected], y=agg_data_app['upper_bound'], mode='lines', line=dict(width=0), showlegend=False))
                fig_patrones_temporales_app.add_trace(go.Scatter(x=agg_data_app[agg_temp_col_selected], y=agg_data_app['lower_bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(5,165,66,0.2)', showlegend=False)) # Verde transparente
                
                title_agg_temp_app = f'Patrón de {variable_seleccionada_state.replace("_"," ").title()} por {agg_temp_col_selected.replace("_", " ").title()}'
                fig_patrones_temporales_app.update_layout(title_text=title_agg_temp_app, xaxis_title=agg_temp_col_selected.replace("_", " ").title(), yaxis_title=f'Promedio {variable_seleccionada_state.replace("_"," ").title()}', margin=dict(l=40, r=40, t=60, b=40), paper_bgcolor='white', plot_bgcolor='white')
            else: fig_patrones_temporales_app.update_layout(title_text=f"No hay datos para agregación después de quitar NaNs en '{agg_temp_col_selected}' o '{variable_seleccionada_state}'.")
        else: fig_patrones_temporales_app.update_layout(title_text=f"Variable '{variable_seleccionada_state}' o columna de agregación '{agg_temp_col_selected}' no encontrada para patrones temporales.")
    else: fig_patrones_temporales_app.update_layout(title_text='No hay datos o variable/agregación no válida para mostrar patrones temporales.')

    # --- Pestaña: Métricas de Desempeño y Cumplimiento ---
    metricas_desempeno_children = [html.H5("Métricas de Desempeño del Sistema", style={'color':CONTUGAS_COLORS['dark_text'], "marginBottom":"10px"})]
    if not df_performance_metrics_app.empty:
        # Formatear floats en el DataFrame de métricas
        df_metrics_display = df_performance_metrics_app.copy()
        for col_metric_float in df_metrics_display.select_dtypes(include='float64').columns:
            df_metrics_display[col_metric_float] = df_metrics_display[col_metric_float].round(3)
        
        metricas_desempeno_children.append(dash_table.DataTable(
            columns=[{"name": i.replace("_"," ").title(), "id": i} for i in df_metrics_display.columns],
            data=df_metrics_display.to_dict('records'),
            style_cell={'textAlign': 'left', 'padding':'5px'}, 
            style_header={'backgroundColor': CONTUGAS_COLORS['background_sidebar'], 'fontWeight':'bold', 'color':CONTUGAS_COLORS['dark_text']}
        ))
    else: metricas_desempeno_children.append(html.P("Archivo de métricas de desempeño (evaluation/performance_metrics.csv) no encontrado o vacío.", style={'fontSize':'0.9em'}))
    
    metricas_desempeno_children.append(html.H5("Tiempos de Ejecución del Pipeline", style={'marginTop':'20px', 'color':CONTUGAS_COLORS['dark_text'], "marginBottom":"10px"}))
    if execution_times_app and isinstance(execution_times_app, dict): 
        exec_times_list_items = [html.Li(f"{str(key).replace('_',' ').title()}: {val:.2f} segundos" if isinstance(val, float) else f"{str(key).replace('_',' ').title()}: {val}") for key, val in execution_times_app.items()]
        metricas_desempeno_children.append(html.Ul(exec_times_list_items, style={'listStyleType':'none', 'paddingLeft':'0', 'fontSize':'0.9em'}))
    else: metricas_desempeno_children.append(html.P("Archivo de tiempos de ejecución (logs/execution_times.yaml) no encontrado, vacío o con formato incorrecto.", style={'fontSize':'0.9em'}))

    # Generar y añadir reporte de cumplimiento
    compliance_report_md_str = generate_compliance_report_md(df_performance_metrics_app, execution_times_app, EVAL_CFG_APP)
    metricas_desempeno_children.append(dcc.Markdown(compliance_report_md_str, style={'marginTop':'20px', 'padding':'10px', 'border':f'1px solid {CONTUGAS_COLORS["card_border"]}', 'borderRadius':'5px', 'backgroundColor':'#fdfdfd'}))

    return (fig_serie_temporal_anom_app, tabla_anomalias_cols_app, tabla_anomalias_data_app,
            kpis_generados_app, barra_info_texto_app, lista_clientes_con_anomalias_html, stats_clave_container_html,
            current_cliente_options_dinamico, current_cliente_value_dinamico,
            fig_corr_heatmap_app, fig_patrones_temporales_app, metricas_desempeno_children)

@app.callback(
    Output("download-csv-tabla-app", "data"), # ID corregido
    Input("btn-descargar-tabla-csv-app", "n_clicks"), # ID corregido
    State("tabla-detallada-anomalias-app", "data"), # ID corregido
    prevent_initial_call=True,
)
def descargar_tabla_anomalias_csv_app(n_clicks_download, table_data_download):
    """Callback para descargar los datos de la tabla de anomalías filtrada como CSV."""
    if not table_data_download: # Si no hay datos en la tabla
        return dash.no_update
    df_to_download = pd.DataFrame(table_data_download)
    # Generar nombre de archivo con timestamp
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_data_frame(df_to_download.to_csv, f"anomalias_detectadas_contugas_{timestamp_str}.csv", index=False)

# --- 6) EJECUCIÓN DE LA APLICACIÓN DASHBOARD ---
if __name__ == '__main__':
    logger.info("Iniciando Dashboard de Detección de Anomalías Contugas...")
    # Re-cargar configuración si es necesario (aunque ya se hace al inicio del script)
    # Esto es más por si se ejecuta el script directamente y las variables globales no se poblaron.
    if not CONFIG_APP or not RUTAS_CFG_APP: 
        try:
            with open(CONFIG_PATH_APP, 'r', encoding='utf-8') as f_cfg_main: 
                CONFIG_APP = yaml.safe_load(f_cfg_main)
            RUTAS_CFG_APP = CONFIG_APP.get('rutas', {}) 
            DASHBOARD_CFG_APP = CONFIG_APP.get('dashboard', {}) 
            ETL_CFG_APP = CONFIG_APP.get('etl', {})
            FEATURE_ENG_CFG_APP = CONFIG_APP.get('feature_engineering', {})
            EVAL_CFG_APP = CONFIG_APP.get('evaluacion_rendimiento', {})
            
            FEATURES_FOR_DISPLAY_APP = ETL_CFG_APP.get('columnas_numericas', ['Presion', 'Temperatura', 'Volumen'])[:]
            GAS_LAW_CFG_APP = FEATURE_ENG_CFG_APP.get('gas_law', {})
            if GAS_LAW_CFG_APP.get('activo', False):
                gas_feature_name_app_main = GAS_LAW_CFG_APP.get('output_feature_name', 'cantidad_gas_calculada')
                if gas_feature_name_app_main not in FEATURES_FOR_DISPLAY_APP:
                    FEATURES_FOR_DISPLAY_APP.append(gas_feature_name_app_main)
            logger.info(f"Configuración (re)cargada para ejecución directa de app.py desde {CONFIG_PATH_APP}")
        except Exception as e_app_main_cfg:
            logger.error(f"Error crítico cargando config.yaml en ejecución directa de app.py: {e_app_main_cfg}", exc_info=True)
            # Usar defaults si la carga falla aquí también
            DASHBOARD_CFG_APP = DASHBOARD_CFG_APP or {'debug': True, 'puerto': 8050} 
    
    app.run_server(
        debug=DASHBOARD_CFG_APP.get('debug', True), 
        port=DASHBOARD_CFG_APP.get('puerto', 8050)
    )
    logger.info("Dashboard de Detección de Anomalías Contugas iniciado.")
    logger.info("Ejecutando app.py como script principal.")