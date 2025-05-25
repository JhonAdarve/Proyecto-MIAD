# ==============================================================================
# alerts/alert_system.py
# Funciones para clasificación de severidad de anomalías y generación de reportes.
# ==============================================================================

import os
import yaml
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
CONFIG = {} 

def cargar_configuracion_alertas(cfg: dict):
    """
    Carga la configuración de alertas globalmente para este módulo.

    Args:
        cfg (dict): Diccionario de configuración global.
    """
    global CONFIG
    CONFIG = cfg
    logger.info("Configuración de alertas cargada en alert_system.py.")


def calcular_estadisticas_scores_normales(scores_normales: np.ndarray) -> tuple[float, float]:
    """
    Calcula la media y desviación estándar de los scores de anomalía (errores de reconstrucción)
    provenientes de datos considerados normales.

    Args:
        scores_normales (np.ndarray): Array de scores de error de datos normales.

    Returns:
        tuple[float, float]: Tupla conteniendo (media_score, std_score).
                             Retorna (0.0, 1.0) si no hay scores o si std es 0.
    """
    if scores_normales is None or len(scores_normales) == 0:
        logger.warning("Scores normales vacíos o None. Retornando media=0, std=1.")
        return 0.0, 1.0
    
    scores_normales_validos = scores_normales[~np.isnan(scores_normales)]
    if len(scores_normales_validos) == 0:
        logger.warning("Todos los scores normales eran NaN. Retornando media=0, std=1.")
        return 0.0, 1.0

    mean_score = np.mean(scores_normales_validos)
    std_score = np.std(scores_normales_validos)
    
    if std_score < 1e-6: 
        logger.warning(f"La desviación estándar de los scores normales válidos es muy baja ({std_score:.6f}). Se usará std=1.0 para la clasificación para evitar sigmas infladas.")
        std_score = 1.0 
        
    logger.debug(f"Estadísticas de scores normales (errores LSTM): media={mean_score:.6f}, std={std_score:.6f} (calculado sobre {len(scores_normales_validos)} valores válidos)")
    return mean_score, std_score


def clasificar_criticidad_anomalia(score_anomalia: float, 
                                   media_scores_normales: float, 
                                   std_scores_normales: float
                                  ) -> str:
    """
    Clasifica la criticidad de una anomalía LSTM según su desviación (en sigmas)
    del error de reconstrucción medio en datos normales.

    Args:
        score_anomalia (float): El score de error de reconstrucción para el punto de datos anómalo.
        media_scores_normales (float): La media de los scores de error de datos normales.
        std_scores_normales (float): La desviación estándar de los scores de error de datos normales.

    Returns:
        str: Cadena indicando la criticidad ('Alta', 'Media', 'Baja', 'Muy Baja', 'Indeterminada').
    """
    alert_cfg = CONFIG.get('alertas', {}).get('niveles_criticidad', {})
    crit_alta_cfg = alert_cfg.get('Alta', {'sigma_min': 3.0})
    crit_media_cfg = alert_cfg.get('Media', {'sigma_min': 2.0, 'sigma_max': 3.0})
    crit_baja_cfg = alert_cfg.get('Baja', {'sigma_min': 1.5, 'sigma_max': 2.0})

    if pd.isna(score_anomalia) or pd.isna(media_scores_normales) or pd.isna(std_scores_normales):
        logger.warning(f"Clasificación de criticidad: Input NaN (Score: {score_anomalia}, Media: {media_scores_normales}, Std: {std_scores_normales}). Criticidad: Indeterminada.")
        return "Indeterminada"

    if std_scores_normales < 1e-6: 
        logger.warning(f"Clasificación de criticidad: std_scores_normales ({std_scores_normales:.6f}) es efectivamente cero. Score: {score_anomalia:.6f}, Media: {media_scores_normales:.6f}. Si score > media, se marcará como 'Alta', sino 'Muy Baja'.")
        return 'Alta' if score_anomalia > (media_scores_normales + 1e-6) else 'Muy Baja' 

    desviacion_sigma = (score_anomalia - media_scores_normales) / std_scores_normales
    
    criticidad = 'Muy Baja' 
    
    sigma_max_media = crit_media_cfg.get('sigma_max', crit_alta_cfg['sigma_min'])
    sigma_max_baja = crit_baja_cfg.get('sigma_max', crit_media_cfg['sigma_min'])

    if desviacion_sigma >= crit_alta_cfg['sigma_min']:
        criticidad = 'Alta'
    elif crit_media_cfg['sigma_min'] <= desviacion_sigma < sigma_max_media:
        criticidad = 'Media'
    elif crit_baja_cfg['sigma_min'] <= desviacion_sigma < sigma_max_baja:
        criticidad = 'Baja'
    
    # Cambiado de logger.info a logger.debug para reducir la verbosidad en logs de nivel INFO
    logger.debug(
        f"Clasificación Criticidad Detalle: "
        f"ScoreAnom={score_anomalia:.4f}, "
        f"MediaNormal={media_scores_normales:.4f}, "
        f"StdNormal={std_scores_normales:.4f} => "
        f"DesvSigma={desviacion_sigma:.2f}. "
        f"Umbrales (Baja:[{crit_baja_cfg['sigma_min']},{sigma_max_baja}), Media:[{crit_media_cfg['sigma_min']},{sigma_max_media}), Alta:[{crit_alta_cfg['sigma_min']},Inf)). "
        f"Criticidad Asignada: {criticidad}"
    )
    return criticidad


def generar_reporte_anomalias_final(df_anomalias_completas: pd.DataFrame, ruta_salida: str):
    """
    Genera y guarda un reporte en formato CSV con la información detallada
    de todas las anomalías detectadas por el sistema.
    """
    if not isinstance(df_anomalias_completas, pd.DataFrame):
        logger.error("El argumento 'df_anomalias_completas' debe ser un pandas DataFrame.")
        return

    if df_anomalias_completas.empty:
        logger.info("No se detectaron anomalías para generar el reporte. No se creará archivo.")
        return

    directorio_salida = os.path.dirname(ruta_salida)
    if directorio_salida: 
        os.makedirs(directorio_salida, exist_ok=True)

    try:
        columnas_base_etl = CONFIG.get('etl', {}).get('columnas_numericas', ['Presion', 'Temperatura', 'Volumen'])
        gas_law_feat_cfg = CONFIG.get('feature_engineering', {}).get('gas_law', {})
        gas_law_col_name = gas_law_feat_cfg.get('output_feature_name', 'cantidad_gas_calculada')
        
        columnas_valores_originales = columnas_base_etl[:]
        if gas_law_feat_cfg.get('activo', False) and gas_law_col_name not in columnas_valores_originales:
            columnas_valores_originales.append(gas_law_col_name)
            
        columnas_identificativas = ['timestamp_deteccion', 'cliente_id', 'cluster_id', 'Fecha']
        columnas_info_anomalia = ['modelo_deteccion', 'score_anomalia_final', 'criticidad_predicha']
        
        columnas_reporte_deseadas = columnas_identificativas + columnas_valores_originales + columnas_info_anomalia
        columnas_a_guardar = [col for col in columnas_reporte_deseadas if col in df_anomalias_completas.columns]
        
        df_reporte = df_anomalias_completas[columnas_a_guardar].copy()
        
        for col_fecha in ['timestamp_deteccion', 'Fecha']:
            if col_fecha in df_reporte.columns and pd.api.types.is_datetime64_any_dtype(df_reporte[col_fecha]):
                 df_reporte[col_fecha] = pd.to_datetime(df_reporte[col_fecha]).dt.strftime('%Y-%m-%d %H:%M:%S')

        df_reporte = df_reporte.sort_values(by=['timestamp_deteccion', 'cliente_id', 'Fecha'], ascending=[False, True, True])
        df_reporte.to_csv(ruta_salida, index=False, date_format='%Y-%m-%d %H:%M:%S')
        logger.info(f"Reporte de anomalías guardado exitosamente en: {ruta_salida}")
    except Exception as e:
        logger.error(f"Error generando el reporte de anomalías final en '{ruta_salida}': {e}", exc_info=True)






