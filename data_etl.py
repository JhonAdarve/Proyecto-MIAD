# ==============================================================================
# data_etl.py
# Módulo para la extracción, transformación y carga (ETL) de datos de consumo de gas.
# Incluye cálculo de cantidad de gas según la ley de los gases ideales/reales.
# ==============================================================================

import os
import glob
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from scipy.stats import median_abs_deviation # Para Robust Z-score

logger = logging.getLogger(__name__)

# Variables globales para configuración (se cargarán desde main.py)
CONFIG = {}
NUMERIC_COLS_BASE_ETL = [] # Columnas numéricas originales (ej. Presion, Temperatura, Volumen)
DATE_COL_ORIGINAL_ETL = 'Fecha' # Nombre de la columna de fecha en los archivos raw

def cargar_configuracion_etl(cfg: dict):
    """
    Carga la configuración ETL globalmente para este módulo.
    Esto permite que las funciones del módulo accedan a los parámetros de configuración.

    Args:
        cfg (dict): Diccionario de configuración global del proyecto.
    """
    global CONFIG, NUMERIC_COLS_BASE_ETL, DATE_COL_ORIGINAL_ETL
    CONFIG = cfg
    etl_specific_config = cfg.get('etl', {})
    NUMERIC_COLS_BASE_ETL = etl_specific_config.get('columnas_numericas', ['Presion', 'Temperatura', 'Volumen'])
    DATE_COL_ORIGINAL_ETL = etl_specific_config.get('columna_fecha_original', 'Fecha')
    logger.info("Configuración ETL cargada y aplicada en data_etl.py.")

def calcular_cantidad_gas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la cantidad de gas (en moles) utilizando la ecuación de estado de los gases ideales
    o una aproximación para gases reales (PV = ZnRT).
    Las unidades se asumen y convierten según la configuración:
    - Presión (P): Se espera en kPa, se convierte a Pascal (Pa).
    - Volumen (V): Se espera en metros cúbicos (m³).
    - Temperatura (T): Se espera en grados Celsius (°C), se convierte a Kelvin (K).
    - Z (Factor de compresibilidad): Tomado de la configuración (default 1.0 para gas ideal).
    - R (Constante universal de los gases): Tomada de la configuración.

    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas 'Presion', 'Temperatura', 'Volumen'.

    Returns:
        pd.DataFrame: El DataFrame original con una nueva columna (nombre definido en config,
                      ej. 'cantidad_gas_calculada') con la cantidad de gas en moles.
                      Si el cálculo no está activo o falla, la columna puede contener NaNs.
    """
    df_calculo = df.copy()
    gas_law_cfg = CONFIG.get('feature_engineering', {}).get('gas_law', {})
    
    if not gas_law_cfg.get('activo', False):
        logger.info("Cálculo de cantidad de gas (feature 'gas_law') no está activo en la configuración. Omitiendo.")
        # Opcional: añadir una columna de NaNs si se espera que exista más adelante
        # df_calculo[gas_law_cfg.get('output_feature_name', 'cantidad_gas_calculada')] = np.nan
        return df_calculo

    R_const = gas_law_cfg.get('R_gas_constant', 8.314)  # J/(mol·K)
    Z_factor_compresibilidad = gas_law_cfg.get('default_Z_factor', 1.0) # Adimensional
    output_col_name_gas = gas_law_cfg.get('output_feature_name', 'cantidad_gas_calculada')

    logger.info(f"Calculando feature '{output_col_name_gas}' usando ley de gases (Z={Z_factor_compresibilidad}, R={R_const}).")

    # Asegurar que las columnas necesarias para el cálculo existen
    required_cols_for_gas_law = ['Presion', 'Volumen', 'Temperatura'] # Nombres base esperados
    for col in required_cols_for_gas_law:
        if col not in df_calculo.columns:
            logger.error(f"Columna '{col}' requerida para cálculo de cantidad de gas no encontrada en el DataFrame. Abortando cálculo de esta feature.")
            df_calculo[output_col_name_gas] = np.nan # Añadir columna con NaNs si no se puede calcular
            return df_calculo

    # Conversión de unidades según supuestos (ajustar si las unidades de entrada son diferentes)
    # Presión: de kPa a Pa (1 kPa = 1000 Pa)
    P_Pa = df_calculo['Presion'] * 1000  
    
    # Temperatura: de °C a K (T_K = T_°C + 273.15)
    T_K = df_calculo['Temperatura'] + 273.15
    
    # Volumen: se asume en m³ (según configuración o estándar)
    V_m3 = df_calculo['Volumen']

    # Cálculo: n = (P * V) / (Z * R * T)
    # Denominador: Z * R * T_K
    denominator_calc = Z_factor_compresibilidad * R_const * T_K
    
    # Inicializar la columna de salida con NaN para manejar casos problemáticos
    df_calculo[output_col_name_gas] = np.nan
    
    # Calcular 'n' solo donde el denominador no es cero, inf o NaN para evitar errores/warnings
    # y donde las entradas P, V, T no son NaN (implícito, ya que NaN en cálculo da NaN)
    valid_denominator_mask = (denominator_calc != 0) & (~np.isinf(denominator_calc)) & (~np.isnan(denominator_calc))
    
    df_calculo.loc[valid_denominator_mask, output_col_name_gas] = \
        (P_Pa[valid_denominator_mask] * V_m3[valid_denominator_mask]) / denominator_calc[valid_denominator_mask]

    num_invalid_denominators = (~valid_denominator_mask).sum()
    if num_invalid_denominators > 0:
        logger.warning(f"Se encontraron {num_invalid_denominators} puntos donde el denominador (ZRT) para el cálculo de cantidad de gas era cero, inf o NaN. Esos puntos tendrán NaN como '{output_col_name_gas}'.")

    # Manejar NaNs o Infinitos que puedan resultar del cálculo si P, V, o T eran NaN/Inf
    # (aunque el producto/división con NaN ya da NaN, Inf/algo puede dar Inf)
    df_calculo[output_col_name_gas].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    nan_count_after_calc = df_calculo[output_col_name_gas].isnull().sum()
    logger.info(f"Feature '{output_col_name_gas}' calculada. Min: {df_calculo[output_col_name_gas].min():.2f}, Max: {df_calculo[output_col_name_gas].max():.2f}, Mean: {df_calculo[output_col_name_gas].mean():.2f}, NaNs: {nan_count_after_calc}")
    return df_calculo


def cargar_datos_excel(ruta_carpeta_raw: str) -> pd.DataFrame:
    """
    Carga datos de consumo desde múltiples archivos Excel ubicados en una carpeta.
    Cada archivo Excel puede contener múltiples hojas, donde cada hoja representa
    los datos de un cliente.

    Args:
        ruta_carpeta_raw (str): Ruta a la carpeta que contiene los archivos Excel.

    Returns:
        pd.DataFrame: DataFrame consolidado con todos los datos de los clientes,
                      o un DataFrame vacío si no se encuentran archivos o datos.
    """
    logger.info(f"Iniciando carga de datos de consumo desde la carpeta: {ruta_carpeta_raw}")
    archivos_excel = glob.glob(os.path.join(ruta_carpeta_raw, "*.xlsx")) # Buscar archivos .xlsx
    
    if not archivos_excel:
        logger.warning(f"No se encontraron archivos .xlsx en la carpeta especificada: {ruta_carpeta_raw}")
        return pd.DataFrame()

    dfs_por_cliente_dict = {} # Usar un diccionario para agrupar DFs por cliente_id antes de concatenar
    
    for archivo_path in archivos_excel:
        nombre_base_archivo_log = os.path.splitext(os.path.basename(archivo_path))[0]
        logger.debug(f"Procesando archivo Excel: {archivo_path}")
        try:
            excel_file_handler = pd.ExcelFile(archivo_path)
            for nombre_hoja_cliente in excel_file_handler.sheet_names:
                try:
                    df_hoja_actual = pd.read_excel(excel_file_handler, sheet_name=nombre_hoja_cliente)
                    logger.debug(f"  Leyendo hoja: '{nombre_hoja_cliente}' del archivo '{nombre_base_archivo_log}'")
                    
                    # Verificar que las columnas esperadas (fecha y numéricas base) existan
                    columnas_esperadas_en_hoja = [DATE_COL_ORIGINAL_ETL] + NUMERIC_COLS_BASE_ETL
                    if not all(col in df_hoja_actual.columns for col in columnas_esperadas_en_hoja):
                        logger.warning(f"  Hoja '{nombre_hoja_cliente}' del archivo '{nombre_base_archivo_log}' no contiene todas las columnas esperadas ({columnas_esperadas_en_hoja}). Omitiendo esta hoja.")
                        continue
                    
                    # Renombrar columna de fecha original para estandarizar y añadir ID de cliente
                    df_hoja_actual = df_hoja_actual.rename(columns={DATE_COL_ORIGINAL_ETL: 'Fecha_original_etl'})
                    id_cliente_hoja = str(nombre_hoja_cliente).strip() # Usar nombre de hoja como ID de cliente
                    df_hoja_actual['cliente_id'] = id_cliente_hoja
                    
                    # Seleccionar solo las columnas necesarias para el DataFrame consolidado
                    columnas_a_mantener_etl = ['cliente_id', 'Fecha_original_etl'] + NUMERIC_COLS_BASE_ETL
                    df_hoja_actual = df_hoja_actual[columnas_a_mantener_etl]
                    
                    # Acumular DataFrames por cliente_id
                    if id_cliente_hoja not in dfs_por_cliente_dict:
                        dfs_por_cliente_dict[id_cliente_hoja] = []
                    dfs_por_cliente_dict[id_cliente_hoja].append(df_hoja_actual)
                    
                    logger.debug(f"    Hoja '{nombre_hoja_cliente}' (cliente '{id_cliente_hoja}') cargada con {len(df_hoja_actual)} filas.")
                except Exception as e_hoja_proc:
                    logger.error(f"  Error al procesar la hoja '{nombre_hoja_cliente}' del archivo '{archivo_path}': {e_hoja_proc}", exc_info=True)
        except Exception as e_archivo_proc:
            logger.error(f"Error al abrir o procesar el archivo Excel '{archivo_path}': {e_archivo_proc}", exc_info=True)
            
    if not dfs_por_cliente_dict: # Si no se cargó ningún dato de ninguna hoja/archivo
        logger.warning("No se cargaron datos de ningún cliente desde los archivos Excel.")
        return pd.DataFrame()

    # Concatenar todos los DataFrames de cada cliente y luego todos los clientes
    lista_final_dfs_consolidados = []
    for cliente_id_key, lista_df_cliente_val in dfs_por_cliente_dict.items():
        if lista_df_cliente_val: # Si hay DataFrames para este cliente
            df_cliente_consolidado = pd.concat(lista_df_cliente_val, ignore_index=True)
            lista_final_dfs_consolidados.append(df_cliente_consolidado)
            
    if not lista_final_dfs_consolidados:
        logger.warning("No hay DataFrames consolidados para ningún cliente después del procesamiento de archivos.")
        return pd.DataFrame()
        
    df_consolidado_final = pd.concat(lista_final_dfs_consolidados, ignore_index=True)
    logger.info(f"Carga de datos completada. Total de datos consolidados: {df_consolidado_final.shape[0]} filas, {df_consolidado_final.shape[1]} columnas, de {df_consolidado_final['cliente_id'].nunique()} clientes.")
    return df_consolidado_final

def limpiar_y_transformar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y transforma la columna de fecha/hora.
    Convierte la columna 'Fecha_original_etl' a datetime, la renombra a 'Fecha',
    elimina filas con fechas inválidas y elimina duplicados por (cliente_id, Fecha).

    Args:
        df (pd.DataFrame): DataFrame con los datos cargados.

    Returns:
        pd.DataFrame: DataFrame con la columna 'Fecha' limpia y transformada,
                      y sin duplicados.
    """
    if df.empty: return df
    logger.info("Limpiando datos y transformando columna de fecha...")
    
    # Convertir a datetime, los errores se coercen a NaT (Not a Time)
    df['Fecha'] = pd.to_datetime(df['Fecha_original_etl'], errors='coerce')
    df = df.drop(columns=['Fecha_original_etl']) # Eliminar la columna original
    
    filas_con_fecha_invalida_count = df['Fecha'].isnull().sum()
    if filas_con_fecha_invalida_count > 0:
        logger.warning(f"Se encontraron {filas_con_fecha_invalida_count} filas con fechas inválidas (NaT) que serán eliminadas.")
        df = df.dropna(subset=['Fecha']) # Eliminar filas donde 'Fecha' es NaT
        
    if df.empty: # Si todas las fechas eran inválidas
        logger.error("DataFrame vacío después de eliminar fechas inválidas.")
        return df

    num_filas_antes_duplicados_check = len(df)
    # Eliminar duplicados basados en cliente y fecha/hora exacta, manteniendo la primera ocurrencia
    df = df.drop_duplicates(subset=['cliente_id', 'Fecha'], keep='first')
    num_duplicados_eliminados_check = num_filas_antes_duplicados_check - len(df)
    if num_duplicados_eliminados_check > 0:
        logger.info(f"Se eliminaron {num_duplicados_eliminados_check} filas duplicadas (basadas en cliente_id y Fecha).")
        
    # Ordenar los datos cronológicamente por cliente
    df = df.sort_values(['cliente_id', 'Fecha']).reset_index(drop=True)
    logger.info(f"Columna de fecha procesada. Datos limpios y ordenados: {len(df)} filas.")
    return df

def tratar_valores_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes (NaN) en las columnas numéricas.
    El método de imputación se define en la configuración (ej. 'interpolacion_temporal').
    La imputación se realiza por grupo de 'cliente_id'.

    Args:
        df (pd.DataFrame): DataFrame con valores potencialmente faltantes.

    Returns:
        pd.DataFrame: DataFrame con valores faltantes imputados.
    """
    if df.empty: return df
    
    etl_config_faltantes = CONFIG.get('etl', {})
    metodo_imputacion_faltantes = etl_config_faltantes.get('metodo_faltantes', 'interpolacion_temporal')
    logger.info(f"Tratando valores faltantes en columnas numéricas con el método: '{metodo_imputacion_faltantes}' por cliente.")
    
    df_imputado_local = df.copy()
    
    # Columnas a imputar: numéricas base + la de cantidad de gas si existe y está activa
    columnas_para_imputar = NUMERIC_COLS_BASE_ETL[:] # Copia
    gas_law_feat_name = CONFIG.get('feature_engineering', {}).get('gas_law', {}).get('output_feature_name', 'cantidad_gas_calculada')
    if CONFIG.get('feature_engineering', {}).get('gas_law', {}).get('activo', False) and gas_law_feat_name in df_imputado_local.columns:
        if gas_law_feat_name not in columnas_para_imputar:
            columnas_para_imputar.append(gas_law_feat_name)
    
    # Asegurar que solo se intenten imputar columnas que realmente existen en el DataFrame
    columnas_para_imputar_existentes = [col for col in columnas_para_imputar if col in df_imputado_local.columns]

    for col_imputar in columnas_para_imputar_existentes:
        num_faltantes_antes_imputacion = df_imputado_local[col_imputar].isnull().sum()
        if num_faltantes_antes_imputacion == 0: # Si no hay faltantes en esta columna, saltar
            logger.debug(f"Columna '{col_imputar}' no tiene valores faltantes. Omitiendo imputación.")
            continue
            
        logger.debug(f"Procesando columna '{col_imputar}' para imputación de faltantes (método: {metodo_imputacion_faltantes}). Faltantes antes: {num_faltantes_antes_imputacion}")
        
        if metodo_imputacion_faltantes == 'interpolacion_temporal':
            # Interpolar linealmente dentro de cada grupo de cliente_id
            # limit_direction='both' para rellenar NaNs al principio/final si es posible
            df_imputado_local[col_imputar] = df_imputado_local.groupby('cliente_id', group_keys=False)[col_imputar].apply(
                lambda x: x.interpolate(method='linear', limit_direction='both', limit_area=None) 
            )
        elif metodo_imputacion_faltantes == 'ffill': # Forward fill
            df_imputado_local[col_imputar] = df_imputado_local.groupby('cliente_id', group_keys=False)[col_imputar].ffill()
        elif metodo_imputacion_faltantes == 'bfill': # Backward fill
            df_imputado_local[col_imputar] = df_imputado_local.groupby('cliente_id', group_keys=False)[col_imputar].bfill()
        elif metodo_imputacion_faltantes == 'media_cliente': # Imputar con la media del cliente
            df_imputado_local[col_imputar] = df_imputado_local.groupby('cliente_id', group_keys=False)[col_imputar].apply(
                lambda x: x.fillna(x.mean())
            )
        else: # Fallback: imputar con 0 si el método no es reconocido
            logger.warning(f"Método de imputación '{metodo_imputacion_faltantes}' no reconocido para columna '{col_imputar}'. Imputando con 0 como fallback.")
            df_imputado_local[col_imputar] = df_imputado_local[col_imputar].fillna(0)
            
        # Fallback final: si después de la imputación por grupo aún quedan NaNs (ej. un cliente solo tiene NaNs), imputar con 0 globalmente.
        if df_imputado_local[col_imputar].isnull().any():
            logger.warning(f"Aún existen NaNs en '{col_imputar}' después de imputación por grupo. Imputando restantes con 0.")
            df_imputado_local[col_imputar] = df_imputado_local[col_imputar].fillna(0)
            
        num_faltantes_despues_imputacion = df_imputado_local[col_imputar].isnull().sum()
        logger.info(f"Columna '{col_imputar}': {num_faltantes_antes_imputacion - num_faltantes_despues_imputacion} valores faltantes imputados. Faltantes restantes: {num_faltantes_despues_imputacion}.")
        
    return df_imputado_local

def tratar_outliers_columna_individual(serie_datos: pd.Series, metodo_outlier: str, factor_iqr_out: float, factor_z_out: float) -> pd.Series:
    """
    Trata outliers en una única serie de datos (columna) usando el método especificado (IQR o Robust Z-score).
    Los outliers se reemplazan por los límites calculados (capping/winsorizing).

    Args:
        serie_datos (pd.Series): Serie de Pandas con los datos de una columna.
        metodo_outlier (str): Método para detectar outliers ('iqr' o 'robust_zscore').
        factor_iqr_out (float): Factor multiplicador para el Rango Intercuartílico (IQR).
        factor_z_out (float): Factor multiplicador para el Robust Z-score (basado en MAD).

    Returns:
        pd.Series: Serie de Pandas con outliers tratados.
    """
    serie_tratada_out = serie_datos.copy()
    lim_inf_out, lim_sup_out = -np.inf, np.inf # Inicializar límites

    if metodo_outlier == 'iqr':
        Q1 = serie_tratada_out.quantile(0.25)
        Q3 = serie_tratada_out.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0: # Si IQR es 0, no se pueden definir límites basados en él de forma útil
            logger.debug(f"IQR es 0 para la serie. No se aplicará capping por IQR.")
            return serie_tratada_out # Retornar sin cambios si IQR es 0
        lim_inf_out = Q1 - factor_iqr_out * IQR
        lim_sup_out = Q3 + factor_iqr_out * IQR
    elif metodo_outlier == 'robust_zscore': # Basado en Median Absolute Deviation (MAD)
        median_val = serie_tratada_out.median()
        # dropna() es importante antes de calcular MAD para evitar errores si hay NaNs
        mad_val = median_abs_deviation(serie_tratada_out.dropna(), nan_policy='omit') 
        if mad_val == 0: # Si MAD es 0, similar a IQR=0, fallback o no tratar
            logger.debug(f"MAD es 0 para la serie. No se aplicará capping por Robust Z-score.")
            return serie_tratada_out
        # El factor 0.6745 convierte MAD a una estimación de la desviación estándar para datos normales
        lim_inf_out = median_val - factor_z_out * mad_val / 0.6745
        lim_sup_out = median_val + factor_z_out * mad_val / 0.6745
    else: # Si el método no es reconocido, no tratar outliers
        logger.warning(f"Método de tratamiento de outliers '{metodo_outlier}' no reconocido. No se tratarán outliers para esta serie.")
        return serie_tratada_out

    # Aplicar capping (reemplazar outliers por los límites)
    serie_tratada_out[serie_tratada_out < lim_inf_out] = lim_inf_out
    serie_tratada_out[serie_tratada_out > lim_sup_out] = lim_sup_out
    
    num_outliers_tratados = (serie_datos < lim_inf_out).sum() + (serie_datos > lim_sup_out).sum()
    if num_outliers_tratados > 0:
        logger.debug(f"Tratados {num_outliers_tratados} outliers en la serie usando método '{metodo_outlier}'. Lim Inf: {lim_inf_out:.2f}, Lim Sup: {lim_sup_out:.2f}")
        
    return serie_tratada_out

def identificar_y_tratar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y trata outliers en las columnas numéricas del DataFrame.
    El tratamiento se realiza por grupo de 'cliente_id'.

    Args:
        df (pd.DataFrame): DataFrame con datos preprocesados.

    Returns:
        pd.DataFrame: DataFrame con outliers tratados.
    """
    if df.empty: return df
    
    etl_config_outliers = CONFIG.get('etl', {})
    metodo_outliers_global = etl_config_outliers.get('metodo_outliers_prepro', 'iqr') # 'iqr' o 'robust_zscore'
    factor_iqr_global = etl_config_outliers.get('factor_outliers_iqr', 1.5)
    factor_zscore_global = etl_config_outliers.get('factor_outliers_zscore', 3.5) # Usado si metodo es 'robust_zscore'
    
    logger.info(f"Identificando y tratando outliers en columnas numéricas con método: '{metodo_outliers_global}' por cliente.")
    df_tratado_outliers = df.copy()
    
    # Columnas a tratar: numéricas base + la de cantidad de gas si existe y está activa
    columnas_para_tratar_outliers = NUMERIC_COLS_BASE_ETL[:]
    gas_law_feat_name_out = CONFIG.get('feature_engineering', {}).get('gas_law', {}).get('output_feature_name', 'cantidad_gas_calculada')
    if CONFIG.get('feature_engineering', {}).get('gas_law', {}).get('activo', False) and gas_law_feat_name_out in df_tratado_outliers.columns:
        if gas_law_feat_name_out not in columnas_para_tratar_outliers:
            columnas_para_tratar_outliers.append(gas_law_feat_name_out)
            
    columnas_para_tratar_outliers_existentes = [col for col in columnas_para_tratar_outliers if col in df_tratado_outliers.columns]

    for col_tratar_out in columnas_para_tratar_outliers_existentes:
        logger.debug(f"Tratando outliers para la columna '{col_tratar_out}' por cliente.")
        # Aplicar la función de tratamiento de outliers a cada grupo de cliente
        df_tratado_outliers[col_tratar_out] = df_tratado_outliers.groupby('cliente_id', group_keys=False)[col_tratar_out].apply(
            lambda x: tratar_outliers_columna_individual(x, metodo_outliers_global, factor_iqr_global, factor_zscore_global)
        )
    return df_tratado_outliers

def extraer_caracteristicas_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características temporales a partir de la columna 'Fecha' (datetime).
    Incluye componentes cíclicos (seno/coseno) para hora, día de la semana, mes.

    Args:
        df (pd.DataFrame): DataFrame con una columna 'Fecha' de tipo datetime.

    Returns:
        pd.DataFrame: DataFrame con nuevas columnas de características temporales.
    """
    if df.empty or 'Fecha' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        logger.warning("DataFrame vacío, sin columna 'Fecha' o 'Fecha' no es datetime. No se extraerán características temporales.")
        return df
        
    logger.info("Extrayendo características temporales (hora, día, mes, cíclicas)...")
    df_feat_temp = df.copy()
    fecha_dt_series = df_feat_temp['Fecha'] # Acceder a la serie una vez
    
    # Componentes básicos del tiempo
    df_feat_temp['hora_dia'] = fecha_dt_series.dt.hour
    df_feat_temp['dia_semana'] = fecha_dt_series.dt.dayofweek # Lunes=0, Domingo=6
    df_feat_temp['dia_mes'] = fecha_dt_series.dt.day
    df_feat_temp['dia_anio'] = fecha_dt_series.dt.dayofyear
    df_feat_temp['semana_anio'] = fecha_dt_series.dt.isocalendar().week.astype(int)
    df_feat_temp['mes'] = fecha_dt_series.dt.month
    df_feat_temp['trimestre'] = fecha_dt_series.dt.quarter
    df_feat_temp['anio'] = fecha_dt_series.dt.year
    df_feat_temp['es_findesemana'] = df_feat_temp['dia_semana'].isin([5, 6]).astype(int) # Sábado=5, Domingo=6
    
    # Componentes cíclicos para capturar patrones periódicos
    # Estos ayudan a los modelos a entender la naturaleza cíclica del tiempo.
    feature_cfg_temporal = CONFIG.get('etl', {}).get('features_temporales_adicionales', {})
    
    if feature_cfg_temporal.get('ciclo_diario', True): # Patrón diario (horas)
        df_feat_temp['hora_sin'] = np.sin(2 * np.pi * df_feat_temp['hora_dia'] / 24.0)
        df_feat_temp['hora_cos'] = np.cos(2 * np.pi * df_feat_temp['hora_dia'] / 24.0)
        
    if feature_cfg_temporal.get('ciclo_semanal', True): # Patrón semanal (días de la semana)
        df_feat_temp['dia_semana_sin'] = np.sin(2 * np.pi * df_feat_temp['dia_semana'] / 7.0)
        df_feat_temp['dia_semana_cos'] = np.cos(2 * np.pi * df_feat_temp['dia_semana'] / 7.0)
        
    if feature_cfg_temporal.get('ciclo_mensual', True): # Patrón anual (meses)
        # Usar 'mes' (1-12) para el ciclo mensual/anual
        df_feat_temp['mes_sin'] = np.sin(2 * np.pi * df_feat_temp['mes'] / 12.0)
        df_feat_temp['mes_cos'] = np.cos(2 * np.pi * df_feat_temp['mes'] / 12.0)
        
    logger.info(f"Características temporales extraídas. Nuevas columnas: {list(set(df_feat_temp.columns) - set(df.columns))}")
    return df_feat_temp

def validar_calidad_datos(df: pd.DataFrame, etapa_validacion: str):
    """
    Realiza una validación básica de la calidad de los datos en una etapa específica del ETL.
    Imprime información sobre dimensiones, NaNs, y estadísticas descriptivas.

    Args:
        df (pd.DataFrame): DataFrame a validar.
        etapa_validacion (str): Nombre de la etapa del ETL para logging (ej. "Después de Carga").
    """
    logger.info(f"--- Iniciando Validación de Calidad de Datos (Etapa: {etapa_validacion}) ---")
    if df.empty:
        logger.warning(f"DataFrame vacío en la etapa '{etapa_validacion}'. No hay datos para validar.")
        logger.info(f"--- Fin Validación de Calidad de Datos ({etapa_validacion}) ---")
        return

    logger.info(f"Dimensiones del DataFrame: {df.shape}")
    
    # Completitud general (porcentaje de celdas no NaN)
    total_celdas = np.prod(df.shape)
    celdas_no_nan = df.notnull().sum().sum()
    completitud_porc = (celdas_no_nan / total_celdas) * 100 if total_celdas > 0 else 0
    logger.info(f"Completitud general: {completitud_porc:.2f}% de celdas no son NaN.")

    # Conteo de NaNs por columna
    nans_por_columna = df.isnull().sum()
    columnas_con_nans = nans_por_columna[nans_por_columna > 0]
    if not columnas_con_nans.empty:
        logger.info("Conteo de NaNs por columna (solo columnas con NaNs):")
        for col_nan, count_nan in columnas_con_nans.items():
            logger.info(f"  - Columna '{col_nan}': {count_nan} NaNs ({count_nan*100/len(df):.2f}%)")
    else:
        logger.info("No se encontraron NaNs en ninguna columna.")
        
    # Estadísticas descriptivas para columnas numéricas
    columnas_numericas_df = df.select_dtypes(include=np.number).columns.tolist()
    if columnas_numericas_df:
        logger.info("Estadísticas descriptivas para columnas numéricas:")
        # Usar .describe().T para transponer y que sea más legible en logs
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000): # Para que se imprima completo en log
             logger.info(f"\n{df[columnas_numericas_df].describe().T}")
    else:
        logger.info("No hay columnas numéricas para mostrar estadísticas descriptivas.")
        
    logger.info(f"--- Fin Validación de Calidad de Datos ({etapa_validacion}) ---")

def normalizar_variables_por_grupo(df: pd.DataFrame, columnas_a_escalar_norm: list, metodo_escalado_norm: str) -> tuple[pd.DataFrame, dict]:
    """
    Normaliza o estandariza las columnas especificadas.
    El escalado se realiza por grupo (ej. 'cluster_id' o 'cliente_id') o globalmente
    si no se encuentra una columna de grupo adecuada.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columnas_a_escalar_norm (list): Lista de nombres de columnas a escalar.
        metodo_escalado_norm (str): Método de escalado ('standard_scaler' o 'min_max_scaler').

    Returns:
        tuple[pd.DataFrame, dict]:
            - pd.DataFrame: DataFrame con las columnas escaladas (añadidas como nuevas columnas con sufijo '_scaled').
            - dict: Diccionario que almacena los objetos scaler ajustados para cada grupo y columna,
                    para poder invertir la transformación si es necesario.
    """
    if df.empty:
        logger.warning("DataFrame vacío en normalizar_variables_por_grupo. Retornando vacío.")
        return df, {}
        
    logger.info(f"Normalizando/Estandarizando columnas: {columnas_a_escalar_norm} con el método '{metodo_escalado_norm}'.")
    df_escalado_result = df.copy()
    scalers_dict_fitted = {} 
    
    # Determinar la columna de agrupación: priorizar 'cluster_id', luego 'cliente_id'.
    # Si ninguna existe, el escalado se hará globalmente.
    columna_grupo_escalado = None
    if 'cluster_id' in df.columns:
        columna_grupo_escalado = 'cluster_id'
    elif 'cliente_id' in df.columns:
        columna_grupo_escalado = 'cliente_id'
        logger.info(f"Columna 'cluster_id' no encontrada. Escalado se realizará por '{columna_grupo_escalado}'.")
    else:
        logger.info("Columnas 'cluster_id' y 'cliente_id' no encontradas. Escalado se realizará globalmente para todas las muestras.")

    # Filtrar columnas a escalar para que solo incluya las que existen en el DataFrame
    columnas_a_escalar_existentes_norm = [col for col in columnas_a_escalar_norm if col in df_escalado_result.columns]
    if not columnas_a_escalar_existentes_norm:
        logger.warning("Ninguna de las columnas especificadas para escalar existe en el DataFrame. No se realizará escalado.")
        return df_escalado_result, scalers_dict_fitted

    if columna_grupo_escalado: # Escalar por grupo
        for col_esc in columnas_a_escalar_existentes_norm:
            scaled_col_name_esc = f"{col_esc}_scaled"
            df_escalado_result[scaled_col_name_esc] = np.nan # Inicializar columna escalada
            
            for group_id, group_df_data in df_escalado_result.groupby(columna_grupo_escalado):
                if group_id not in scalers_dict_fitted: scalers_dict_fitted[group_id] = {}
                
                scaler_obj_group = StandardScaler() if metodo_escalado_norm == 'standard_scaler' else MinMaxScaler()
                col_data_grupo_esc = group_df_data[[col_esc]].copy() # Obtener datos de la columna para el grupo
                
                # Imputar NaNs con 0 antes de escalar si aún existen (aunque deberían estar tratados)
                if col_data_grupo_esc[col_esc].isnull().any():
                    logger.warning(f"NaNs encontrados en columna '{col_esc}' para grupo '{group_id}' antes de escalar. Imputando con 0.")
                    col_data_grupo_esc[col_esc] = col_data_grupo_esc[col_esc].fillna(0)
                
                if len(col_data_grupo_esc) > 0: # Asegurar que haya datos en el grupo
                    try:
                        scaled_values_group = scaler_obj_group.fit_transform(col_data_grupo_esc[[col_esc]])
                        df_escalado_result.loc[group_df_data.index, scaled_col_name_esc] = scaled_values_group
                        scalers_dict_fitted[group_id][col_esc] = scaler_obj_group
                    except ValueError as e_scaler_group: # Puede ocurrir si todos los valores son iguales (std=0 para StandardScaler)
                        logger.error(f"Error al escalar columna '{col_esc}' para grupo '{group_id}': {e_scaler_group}. Los valores podrían ser constantes. Se usará la columna original o ceros.")
                        # Fallback: usar valores originales o ceros si el escalado falla
                        df_escalado_result.loc[group_df_data.index, scaled_col_name_esc] = col_data_grupo_esc[col_esc].values 
                else:
                    logger.warning(f"Grupo '{group_id}' no tiene datos para la columna '{col_esc}'. No se escalará para este grupo.")
    else: # Escalar globalmente
        scalers_dict_fitted['global'] = {}
        for col_esc in columnas_a_escalar_existentes_norm:
            scaled_col_name_esc = f"{col_esc}_scaled"
            scaler_obj_global = StandardScaler() if metodo_escalado_norm == 'standard_scaler' else MinMaxScaler()
            col_data_global_esc = df_escalado_result[[col_esc]].copy()
            
            if col_data_global_esc[col_esc].isnull().any():
                logger.warning(f"NaNs encontrados en columna '{col_esc}' antes de escalado global. Imputando con 0.")
                col_data_global_esc[col_esc] = col_data_global_esc[col_esc].fillna(0)
            
            try:
                df_escalado_result[scaled_col_name_esc] = scaler_obj_global.fit_transform(col_data_global_esc[[col_esc]])
                scalers_dict_fitted['global'][col_esc] = scaler_obj_global
            except ValueError as e_scaler_global:
                logger.error(f"Error al escalar columna '{col_esc}' globalmente: {e_scaler_global}. Se usará la columna original o ceros.")
                df_escalado_result[scaled_col_name_esc] = col_data_global_esc[col_esc].values

    return df_escalado_result, scalers_dict_fitted

def dividir_datos_chrono_por_grupo(df: pd.DataFrame, group_col_split: str, test_size_split: float, val_size_split: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos cronológicamente en conjuntos de entrenamiento, validación y prueba
    para cada grupo especificado por 'group_col_split' (ej. 'cliente_id' o 'cluster_id').
    Asegura que los datos dentro de cada grupo se mantengan ordenados por fecha.

    Args:
        df (pd.DataFrame): DataFrame completo, ordenado por 'Fecha'.
        group_col_split (str): Nombre de la columna para agrupar antes de dividir (ej. 'cliente_id').
        test_size_split (float): Proporción del conjunto de prueba (ej. 0.2 para 20%).
        val_size_split (float): Proporción del conjunto de validación, tomada del conjunto
                                que NO es de prueba (ej. 0.15 del 80% restante).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_train: DataFrame de entrenamiento.
            - df_val: DataFrame de validación.
            - df_test: DataFrame de prueba.
            Cualquiera puede ser vacío si no hay suficientes datos.
    """
    if df.empty:
        logger.warning("DataFrame vacío en dividir_datos_chrono_por_grupo. Retornando DataFrames vacíos.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if group_col_split not in df.columns:
        logger.error(f"Columna de grupo '{group_col_split}' no encontrada para división de datos. No se puede dividir.")
        # Podría dividir globalmente como fallback, pero es mejor fallar si se espera por grupo.
        return df, pd.DataFrame(), pd.DataFrame() # Retornar todo como train, o vacíos

    train_dfs_list, val_dfs_list, test_dfs_list = [], [], []
    
    for _, group_data_df in df.groupby(group_col_split):
        n_group = len(group_data_df)
        if n_group < 3: # No suficientes datos para dividir en 3 conjuntos
            logger.warning(f"Grupo '{_}' tiene {n_group} muestras, insuficientes para dividir. Se usará todo para entrenamiento si es posible.")
            if n_group > 0: train_dfs_list.append(group_data_df)
            continue

        # Calcular índices para la división cronológica
        n_test_group = int(n_group * test_size_split)
        n_train_val_group = n_group - n_test_group
        
        if n_train_val_group == 0: # Si todo es test (test_size_split cercano a 1)
            test_dfs_list.append(group_data_df)
            continue
            
        # val_size_split es sobre el conjunto (train+val), no sobre el total original
        n_val_group = int(n_train_val_group * val_size_split) 
        n_train_group = n_train_val_group - n_val_group

        if n_train_group <= 0 : # Si no hay suficientes datos para train después de sacar val y test
            logger.warning(f"Grupo '{_}' no tiene suficientes datos para un conjunto de entrenamiento ({n_train_group} muestras). Ajustando división.")
            # Asignar a validación y prueba si es posible, o todo a prueba.
            if n_val_group > 0: val_dfs_list.append(group_data_df.iloc[:n_val_group])
            if n_test_group > 0: test_dfs_list.append(group_data_df.iloc[n_val_group if n_val_group > 0 else 0:])
            continue
            
        # Realizar la división
        test_set_group = group_data_df.iloc[n_train_val_group:]
        train_val_set_group = group_data_df.iloc[:n_train_val_group]
        
        val_set_group = train_val_set_group.iloc[n_train_group:]
        train_set_group = train_val_set_group.iloc[:n_train_group]
        
        if not train_set_group.empty: train_dfs_list.append(train_set_group)
        if not val_set_group.empty: val_dfs_list.append(val_set_group)
        if not test_set_group.empty: test_dfs_list.append(test_set_group)

    # Concatenar los DataFrames de todos los grupos
    df_train_final = pd.concat(train_dfs_list).sort_values([group_col_split, 'Fecha']).reset_index(drop=True) if train_dfs_list else pd.DataFrame()
    df_val_final = pd.concat(val_dfs_list).sort_values([group_col_split, 'Fecha']).reset_index(drop=True) if val_dfs_list else pd.DataFrame()
    df_test_final = pd.concat(test_dfs_list).sort_values([group_col_split, 'Fecha']).reset_index(drop=True) if test_dfs_list else pd.DataFrame()
    
    logger.info(f"División de datos por '{group_col_split}' completada: "
                f"Train={len(df_train_final)} ({df_train_final[group_col_split].nunique() if group_col_split in df_train_final else 0} grupos), "
                f"Val={len(df_val_final)} ({df_val_final[group_col_split].nunique() if group_col_split in df_val_final else 0} grupos), "
                f"Test={len(df_test_final)} ({df_test_final[group_col_split].nunique() if group_col_split in df_test_final else 0} grupos).")
    return df_train_final, df_val_final, df_test_final

# --- Pipeline Principal de ETL ---
def pipeline_etl_completo(config_global_etl: dict, ruta_salida_csv_etl: str) -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo: carga, limpieza, transformación,
    ingeniería de características y validación.

    Args:
        config_global_etl (dict): Diccionario de configuración global.
        ruta_salida_csv_etl (str): Ruta donde se guardará el DataFrame preprocesado.

    Returns:
        pd.DataFrame: DataFrame preprocesado y listo para las siguientes etapas (clustering/modelado).
                      Retorna DataFrame vacío si alguna etapa crítica falla.
    """
    cargar_configuracion_etl(config_global_etl) # Aplicar config a este módulo
    rutas_cfg_etl = CONFIG.get('rutas', {})
    
    # 1. Cargar datos brutos
    df_bruto_etl = cargar_datos_excel(rutas_cfg_etl.get('raw', 'data/raw/'))
    if df_bruto_etl.empty: 
        logger.error("ETL Abortado: No se pudieron cargar datos brutos.")
        return pd.DataFrame()
    validar_calidad_datos(df_bruto_etl, "Después de Carga Bruta")

    # 2. Limpiar y transformar columna de fecha
    df_limpio_fechas_etl = limpiar_y_transformar_fecha(df_bruto_etl)
    if df_limpio_fechas_etl.empty:
        logger.error("ETL Abortado: Falló la limpieza de fechas.")
        return pd.DataFrame()
    validar_calidad_datos(df_limpio_fechas_etl, "Después de Limpieza de Fechas")

    # 3. Calcular cantidad de gas (si está activo en config)
    # Se calcula ANTES de la imputación de faltantes para P,V,T,
    # ya que la cantidad de gas depende de estos valores.
    # Si P,V,T tienen NaNs, cantidad_gas también tendrá NaNs y se imputará después.
    df_con_gas_qty_etl = calcular_cantidad_gas(df_limpio_fechas_etl)
    validar_calidad_datos(df_con_gas_qty_etl, "Después de Calcular Cantidad de Gas")

    # 4. Tratar valores faltantes (imputación)
    # Ahora imputará P,V,T y la nueva feature de gas si tiene NaNs
    df_imputado_etl = tratar_valores_faltantes(df_con_gas_qty_etl) 
    validar_calidad_datos(df_imputado_etl, "Después de Imputación de Faltantes")

    # 5. Identificar y tratar outliers
    # Tratará outliers en P,V,T y la nueva feature de gas (si existe)
    df_sin_outliers_etl = identificar_y_tratar_outliers(df_imputado_etl) 
    validar_calidad_datos(df_sin_outliers_etl, "Después de Tratamiento de Outliers")

    # 6. Extraer características temporales
    df_con_features_temporales_etl = extraer_caracteristicas_temporales(df_sin_outliers_etl)
    validar_calidad_datos(df_con_features_temporales_etl, "Después de Ingeniería de Características Temporales")
    
    # 7. Guardar datos preprocesados
    try:
        directorio_salida_etl = os.path.dirname(ruta_salida_csv_etl)
        if directorio_salida_etl: os.makedirs(directorio_salida_etl, exist_ok=True)
        df_con_features_temporales_etl.to_csv(ruta_salida_csv_etl, index=False)
        logger.info(f"Datos preprocesados guardados exitosamente en: {ruta_salida_csv_etl}")
    except Exception as e_save_etl:
        logger.error(f"Error al guardar datos preprocesados en '{ruta_salida_csv_etl}': {e_save_etl}", exc_info=True)

    logger.info("Pipeline ETL (pre-clustering y pre-escalado de modelo) completado.")
    return df_con_features_temporales_etl