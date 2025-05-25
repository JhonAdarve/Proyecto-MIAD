# ==============================================================================
# evaluation/experiments.py
# Funciones para evaluar el rendimiento del sistema de detección de anomalías
# comparando las predicciones con un conjunto de datos de ground truth.
# ==============================================================================

import os
import yaml
import time 
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import logging

logger = logging.getLogger(__name__)
CONFIG = {} 

def cargar_configuracion_evaluacion(cfg: dict):
    """
    Carga la configuración de evaluación globalmente para este módulo.
    """
    global CONFIG
    CONFIG = cfg
    logger.info("Configuración de evaluación cargada en experiments.py.")

def calcular_metricas_rendimiento(
    y_true_flags_eval: np.ndarray, 
    y_pred_flags_eval: np.ndarray, 
    y_pred_criticidad_eval: pd.Series | None = None,
    y_true_criticidad_eval: pd.Series | None = None
) -> dict:
    """
    Calcula un conjunto de métricas de rendimiento para la detección y clasificación de anomalías.
    """
    if len(y_true_flags_eval) != len(y_pred_flags_eval):
        logger.error("Las longitudes de y_true_flags y y_pred_flags no coinciden. No se pueden calcular métricas.")
        return {} 
    
    if len(y_true_flags_eval) == 0: 
        logger.warning("No hay datos (y_true_flags_eval está vacío) para calcular métricas de rendimiento.")
        return {
            'total_muestras': 0, 'positivos_reales': 0, 'negativos_reales': 0,
            'positivos_predichos': 0, 'verdaderos_positivos_tp': 0, 'falsos_positivos_fp': 0,
            'verdaderos_negativos_tn': 0, 'falsos_negativos_fn': 0, 'precision_anomalia': None,
            'recall_anomalia_sensibilidad': None, 'f1_score_anomalia': None,
            'tasa_falsos_positivos': None, 'especificidad': None, 'accuracy_general': None,
            'precision_clasificacion_criticidad': None
        }

    cm = confusion_matrix(y_true_flags_eval.astype(int), y_pred_flags_eval.astype(int), labels=[0, 1]) 
    
    if cm.shape == (2,2): 
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1,1): 
        if np.unique(y_true_flags_eval)[0] == 0: 
            tn = cm[0,0]; fp = 0; fn = 0; tp = 0;
        else: 
            tn = 0; fp = 0; fn = 0; tp = cm[0,0];
    else: 
        logger.warning(f"Matriz de confusión con forma inesperada: {cm.shape}. Recalculando TP/TN/FP/FN.")
        tp = np.sum((y_true_flags_eval == 1) & (y_pred_flags_eval == 1))
        tn = np.sum((y_true_flags_eval == 0) & (y_pred_flags_eval == 0))
        fp = np.sum((y_true_flags_eval == 0) & (y_pred_flags_eval == 1))
        fn = np.sum((y_true_flags_eval == 1) & (y_pred_flags_eval == 0))

    metricas_dict = {
        'total_muestras': len(y_true_flags_eval),
        'positivos_reales': int(np.sum(y_true_flags_eval)),      
        'negativos_reales': int(len(y_true_flags_eval) - np.sum(y_true_flags_eval)), 
        'positivos_predichos': int(np.sum(y_pred_flags_eval)),  
        'verdaderos_positivos_tp': int(tp),
        'falsos_positivos_fp': int(fp),
        'verdaderos_negativos_tn': int(tn),
        'falsos_negativos_fn': int(fn),
    }

    metricas_dict['precision_anomalia'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metricas_dict['recall_anomalia_sensibilidad'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metricas_dict['f1_score_anomalia'] = 2 * (metricas_dict['precision_anomalia'] * metricas_dict['recall_anomalia_sensibilidad']) / \
                                   (metricas_dict['precision_anomalia'] + metricas_dict['recall_anomalia_sensibilidad']) if \
                                   (metricas_dict['precision_anomalia'] + metricas_dict['recall_anomalia_sensibilidad']) > 0 else 0.0
    metricas_dict['tasa_falsos_positivos'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metricas_dict['especificidad'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metricas_dict['accuracy_general'] = (tp + tn) / len(y_true_flags_eval) if len(y_true_flags_eval) > 0 else 0.0

    if y_pred_criticidad_eval is not None and y_true_criticidad_eval is not None:
        idx_tp_eval = np.where((y_true_flags_eval == 1) & (y_pred_flags_eval == 1))[0] 
        if len(idx_tp_eval) > 0:
            pred_crit_tp_eval = y_pred_criticidad_eval.iloc[idx_tp_eval] if isinstance(y_pred_criticidad_eval, pd.Series) else y_pred_criticidad_eval[idx_tp_eval]
            true_crit_tp_eval = y_true_criticidad_eval.iloc[idx_tp_eval] if isinstance(y_true_criticidad_eval, pd.Series) else y_true_criticidad_eval[idx_tp_eval]
            metricas_dict['precision_clasificacion_criticidad'] = np.mean(pred_crit_tp_eval.astype(str) == true_crit_tp_eval.astype(str)) if len(pred_crit_tp_eval) > 0 else 0.0
        else:
            metricas_dict['precision_clasificacion_criticidad'] = None 
    else:
        metricas_dict['precision_clasificacion_criticidad'] = None
    
    logger.info(f"Métricas de rendimiento calculadas:\n{pd.Series(metricas_dict)}")
    return metricas_dict


def evaluar_sistema_completo(df_resultados_deteccion_eval: pd.DataFrame, 
                             df_ground_truth_eval: pd.DataFrame,
                             ruta_salida_metricas_csv: str) -> pd.DataFrame: 
    """
    Evalúa el sistema completo comparando los resultados de detección del pipeline
    contra un DataFrame de ground truth y guarda las métricas.
    """
    logger.info("Iniciando evaluación del sistema completo contra ground truth.")
    if df_resultados_deteccion_eval.empty or df_ground_truth_eval.empty:
        logger.error("Se requieren DataFrames de resultados y ground truth no vacíos para la evaluación.")
        return pd.DataFrame()

    try:
        df_resultados_deteccion_eval['Fecha'] = pd.to_datetime(df_resultados_deteccion_eval['Fecha'])
        df_ground_truth_eval['Fecha'] = pd.to_datetime(df_ground_truth_eval['Fecha'])
    except Exception as e_date_conv:
        logger.error(f"Error convirtiendo 'Fecha' a datetime: {e_date_conv}.", exc_info=True)
        return pd.DataFrame()

    df_evaluacion_merged = pd.merge(
        df_resultados_deteccion_eval, 
        df_ground_truth_eval, 
        on=['cliente_id', 'Fecha'], 
        how='left',
        suffixes=('_pred', '_real') 
    )
    
    if 'anomalia_real' not in df_evaluacion_merged.columns: 
        logger.error("Columna 'anomalia_real' no encontrada en ground truth o después del merge.")
        return pd.DataFrame()
        
    df_evaluacion_merged['anomalia_real'] = df_evaluacion_merged['anomalia_real'].fillna(False).astype(bool)
    if 'criticidad_real' in df_evaluacion_merged.columns:
        df_evaluacion_merged['criticidad_real'] = df_evaluacion_merged['criticidad_real'].fillna('Normal') 
    
    y_true_flags_final = df_evaluacion_merged['anomalia_real'].values
    if 'anomalia_predicha' not in df_evaluacion_merged.columns:
        logger.error("Columna 'anomalia_predicha' no encontrada en resultados de detección.")
        return pd.DataFrame()
    y_pred_flags_final = df_evaluacion_merged['anomalia_predicha'].astype(bool).values 
    
    y_pred_criticidad_final = df_evaluacion_merged['criticidad_predicha'] if 'criticidad_predicha' in df_evaluacion_merged.columns else None
    y_true_criticidad_final = df_evaluacion_merged['criticidad_real'] if 'criticidad_real' in df_evaluacion_merged.columns else None

    metricas_globales_calculadas = calcular_metricas_rendimiento(
        y_true_flags_final, y_pred_flags_final, 
        y_pred_criticidad_eval=y_pred_criticidad_final,
        y_true_criticidad_eval=y_true_criticidad_final
    )
    
    if not metricas_globales_calculadas: 
        logger.error("No se pudieron calcular las métricas de rendimiento globales.")
        return pd.DataFrame()
        
    df_metricas_resumen_final = pd.DataFrame([metricas_globales_calculadas]) 
    
    try:
        # Asegurar que el directorio de salida exista
        directorio_salida_metricas = os.path.dirname(ruta_salida_metricas_csv)
        if directorio_salida_metricas: # Si no es el directorio actual
             os.makedirs(directorio_salida_metricas, exist_ok=True)
        df_metricas_resumen_final.to_csv(ruta_salida_metricas_csv, index=False)
        logger.info(f"Métricas de rendimiento del sistema guardadas en: {ruta_salida_metricas_csv}")
    except Exception as e_save_metrics:
        logger.error(f"Error al guardar métricas de rendimiento en '{ruta_salida_metricas_csv}': {e_save_metrics}", exc_info=True)
    
    eval_cfg_criterios = CONFIG.get('evaluacion_rendimiento', {})
    if eval_cfg_criterios:
        logger.info("\n--- Verificación de Criterios de Aceptación del Rendimiento ---")
        fpr_max_req = eval_cfg_criterios.get('tasa_falsos_positivos_max', 0.15)
        fpr_actual = metricas_globales_calculadas.get('tasa_falsos_positivos')
        if fpr_actual is not None:
            fpr_cumple = fpr_actual <= fpr_max_req
            logger.info(f"Tasa de Falsos Positivos: {fpr_actual:.3f} (Requerido <= {fpr_max_req}) - Cumple: {fpr_cumple}")
        
        sens_min_req_general = eval_cfg_criterios.get('deteccion_incidentes_reales_min', 0.80) 
        sens_actual_general = metricas_globales_calculadas.get('recall_anomalia_sensibilidad')
        if sens_actual_general is not None:
            sens_gral_cumple = sens_actual_general >= sens_min_req_general
            logger.info(f"Sensibilidad General (Detección Incidentes Reales): {sens_actual_general:.3f} (Requerido >= {sens_min_req_general}) - Cumple: {sens_gral_cumple}")

        prec_crit_min_req = eval_cfg_criterios.get('precision_clasificacion_criticidad_min', 0.90) 
        prec_crit_actual = metricas_globales_calculadas.get('precision_clasificacion_criticidad')
        if prec_crit_actual is not None:
            prec_crit_cumple = prec_crit_actual >= prec_crit_min_req
            logger.info(f"Precisión Clasificación Criticidad (sobre TPs): {prec_crit_actual:.3f} (Requerido >= {prec_crit_min_req}) - Cumple: {prec_crit_cumple}")
    return df_metricas_resumen_final
