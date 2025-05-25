# ==============================================================================
# main.py
# Script principal para orquestar el pipeline de detección de anomalías:
# ETL, clustering, entrenamiento de modelos por cluster, detección y evaluación.
# ==============================================================================
import argparse
import os
import yaml
import pandas as pd
import numpy as np
import time
import logging
import logging.config # Importante para configurar logging desde diccionario
from datetime import datetime
import multiprocessing # Para paralelizar el entrenamiento de modelos

# Módulos del proyecto
from data_etl import (
    pipeline_etl_completo as run_etl_pipeline,
    normalizar_variables_por_grupo, 
    dividir_datos_chrono_por_grupo, 
    cargar_configuracion_etl
)
from clustering import (
    pipeline_cliente_clustering, 
    cargar_configuracion_clustering
)
from models.lstm_model import (
    crear_lstm_autoencoder, 
    preparar_datos_secuencia, 
    entrenar_lstm_ae,
    determinar_umbral_anomalia_lstm, 
    detectar_anomalias_lstm,
    guardar_modelo_lstm, 
    cargar_modelo_lstm, 
    cargar_configuracion_lstm,
    calcular_errores_reconstruccion 
)
from alerts.alert_system import (
    clasificar_criticidad_anomalia, 
    generar_reporte_anomalias_final,
    calcular_estadisticas_scores_normales, 
    cargar_configuracion_alertas
)
from evaluation.experiments import (
    evaluar_sistema_completo, 
    cargar_configuracion_evaluacion
)

import tensorflow as tf

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()] 
)
logger = logging.getLogger(__name__) 
CONFIG = {} 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

def setup_logging_from_config(config_dict: dict):
    global logger 
    try:
        if 'logging' in config_dict:
            log_config_section = config_dict['logging']
            if 'version' not in log_config_section:
                log_config_section['version'] = 1
                logging.warning("Se añadió 'version: 1' a la configuración de logging ya que faltaba.")
            if 'handlers' in log_config_section and 'file' in log_config_section['handlers']:
                rutas_cfg_log = config_dict.get('rutas', {})
                log_file_from_rutas_cfg = rutas_cfg_log.get('logs', 'logs/default_anomaly_detection.log') 
                
                if not os.path.isabs(log_file_from_rutas_cfg):
                    log_file_from_rutas_cfg = os.path.join(BASE_DIR, log_file_from_rutas_cfg)

                log_dir_path = os.path.dirname(log_file_from_rutas_cfg)
                if log_dir_path and not os.path.exists(log_dir_path): 
                    os.makedirs(log_dir_path, exist_ok=True)
                    logging.info(f"Directorio de logs creado en: {log_dir_path}")
                log_config_section['handlers']['file']['filename'] = log_file_from_rutas_cfg
            
            logging.config.dictConfig(log_config_section)
            logger = logging.getLogger(__name__) 
            logger.info("Sistema de logging configurado exitosamente desde la estructura YAML.")
        else:
            logger.info("Sección 'logging' no encontrada en config.yaml. Se mantiene configuración de logging básica.")
    except Exception as e_log_setup:
        logging.getLogger(__name__).error(f"Error CRÍTICO configurando logging desde config.yaml: {e_log_setup}", exc_info=True)

def cargar_configuracion_global(config_path: str) -> dict | None:
    global CONFIG 
    try:
        if not os.path.isabs(config_path):
            config_path = os.path.join(BASE_DIR, config_path)

        with open(config_path, 'r', encoding='utf-8') as f_config: 
            loaded_config_dict = yaml.safe_load(f_config)
        
        if not loaded_config_dict: 
            logger.critical(f"El archivo de configuración '{config_path}' está vacío o no es un YAML válido.")
            return None
        
        CONFIG = loaded_config_dict 
        setup_logging_from_config(CONFIG) 
        logger.info(f"Configuración global cargada exitosamente desde: {config_path}")
        
        try:
            cargar_configuracion_etl(CONFIG)
            cargar_configuracion_clustering(CONFIG)
            cargar_configuracion_lstm(CONFIG)
            cargar_configuracion_alertas(CONFIG)
            cargar_configuracion_evaluacion(CONFIG)
            logger.info("Configuración propagada a todos los módulos.")
        except NameError as ne_propagate: 
            logger.error(f"Error al propagar configuración a un módulo (NameError): {ne_propagate}.", exc_info=True)
        except AttributeError as ae_propagate: 
            logger.error(f"Error al propagar configuración (AttributeError): {ae_propagate}.", exc_info=True)
        return CONFIG
    except FileNotFoundError:
        logger.critical(f"Archivo de configuración '{config_path}' no encontrado.", exc_info=True)
        return None 
    except yaml.YAMLError as ye_parse:
        logger.critical(f"Error al parsear el archivo YAML de configuración '{config_path}': {ye_parse}", exc_info=True)
        return None
    except Exception as e_load_config: 
        logger.critical(f"Error general inesperado al cargar la configuración desde '{config_path}': {e_load_config}", exc_info=True)
        return None

def run_hyperparameter_optimization_lstm(
    X_train_seq_hpo: np.ndarray, 
    X_val_seq_hpo: np.ndarray | None, 
    param_grid_lstm_hpo: dict, 
    seq_len_hpo: int, 
    n_features_hpo: int, 
    cluster_id_hpo: str | int, 
    global_config_hpo: dict
) -> tuple[tf.keras.Model | None, dict | None, tuple[float, float] | None]:
    hpo_logger_instance = logging.getLogger(f"LSTM_HPO_Cluster_{cluster_id_hpo}") 
    hpo_logger_instance.info(f"Iniciando optimización de hiperparámetros para LSTM AE (Cluster {cluster_id_hpo}).")
    best_model_found_hpo = None 
    best_params_found = None
    best_val_loss_hpo = float('inf') 
    best_model_val_error_stats_hpo = None 
    from itertools import product 
    param_names_hpo = list(param_grid_lstm_hpo.keys())
    param_value_combinations_list = list(product(*param_grid_lstm_hpo.values()))
    total_combinations_to_try = len(param_value_combinations_list)
    hpo_logger_instance.info(f"Se probarán {total_combinations_to_try} combinaciones de hiperparámetros.")
    lstm_training_config = global_config_hpo.get('modelos',{}).get('lstm_autoencoder',{})
    patience_es_hpo = lstm_training_config.get('patience_es', 10) 
    patience_lr_hpo = lstm_training_config.get('patience_lr', 5)   
    hpo_logger_instance.info(f"Parámetros de entrenamiento para HPO: patience_early_stopping={patience_es_hpo}, patience_reduce_lr={patience_lr_hpo}")

    for i, params_tuple_values in enumerate(param_value_combinations_list):
        current_params_dict = dict(zip(param_names_hpo, params_tuple_values))
        try:
            current_params_dict['encoding_dim'] = int(current_params_dict['encoding_dim'])
            current_params_dict['learning_rate'] = float(current_params_dict['learning_rate'])
            current_params_dict['epochs'] = int(current_params_dict['epochs'])
            current_params_dict['batch_size'] = int(current_params_dict['batch_size'])
        except KeyError as ke:
            hpo_logger_instance.error(f"Error de clave al convertir parámetros HPO: {ke}. Parámetros: {current_params_dict}. Grid: {param_grid_lstm_hpo}. Saltando.", exc_info=True)
            continue
        except Exception as e_param_conv:
            hpo_logger_instance.error(f"Error al convertir tipos de parámetros HPO: {e_param_conv}. Parámetros: {current_params_dict}. Saltando.", exc_info=True)
            continue

        hpo_logger_instance.info(f"Prueba HPO ({i+1}/{total_combinations_to_try}): Probando parámetros: {current_params_dict}")
        try:
            tf.keras.backend.clear_session() 
            model_hpo_iter = crear_lstm_autoencoder(
                seq_len=seq_len_hpo, n_features=n_features_hpo,
                encoding_dim=current_params_dict['encoding_dim'], 
                learning_rate=current_params_dict['learning_rate']
            )
            history_hpo_iter = entrenar_lstm_ae(
                model_hpo_iter, X_train_seq_hpo, X_val_seq_hpo, 
                epochs=current_params_dict['epochs'], 
                batch_size=current_params_dict['batch_size'],
                patience_early_stopping=patience_es_hpo, 
                patience_reduce_lr=patience_lr_hpo      
            )
            loss_to_check = 'val_loss' if X_val_seq_hpo is not None and len(X_val_seq_hpo) > 0 else 'loss'
            current_loss_from_history = min(history_hpo_iter.history.get(loss_to_check, [float('inf')]))
            
            if current_loss_from_history < best_val_loss_hpo:
                best_val_loss_hpo = current_loss_from_history
                best_params_found = current_params_dict
                best_model_found_hpo = model_hpo_iter 
                hpo_logger_instance.info(f"Nueva mejor {loss_to_check} = {best_val_loss_hpo:.6f} encontrada con parámetros: {best_params_found}")
                if X_val_seq_hpo is not None and X_val_seq_hpo.shape[0] > 0:
                    val_errors_best_model_iter = calcular_errores_reconstruccion(best_model_found_hpo, X_val_seq_hpo)
                    if val_errors_best_model_iter.size > 0:
                        best_model_val_error_stats_hpo = calcular_estadisticas_scores_normales(val_errors_best_model_iter)
                    else: 
                        best_model_val_error_stats_hpo = None 
                else: 
                    best_model_val_error_stats_hpo = None
        except Exception as e_hpo_iter:
            hpo_logger_instance.error(f"Error durante prueba HPO con parámetros {current_params_dict}: {e_hpo_iter}", exc_info=True)
            continue             
    if best_model_found_hpo:
        loss_monitor_final = 'val_loss' if X_val_seq_hpo is not None and len(X_val_seq_hpo) > 0 else 'loss'
        hpo_logger_instance.info(f"Optimización HPO completada. Mejor modelo con {loss_monitor_final}={best_val_loss_hpo:.6f}, params: {best_params_found}")
        if best_model_val_error_stats_hpo:
            hpo_logger_instance.info(f"Stats error validación mejor modelo (media, std): {best_model_val_error_stats_hpo}")
    else:
        hpo_logger_instance.warning(f"No se encontró modelo LSTM óptimo tras {total_combinations_to_try} pruebas HPO.")
    return best_model_found_hpo, best_params_found, best_model_val_error_stats_hpo

def train_lstm_for_cluster_worker(args_tuple_worker):
    cluster_id_worker, df_cluster_actual_worker, columnas_a_escalar_worker, metodo_escalado_worker, \
    features_para_lstm_worker, seq_len_lstm_worker, param_grid_lstm_worker, \
    rutas_modelos_entrenados_worker, global_config_dict_worker = args_tuple_worker

    worker_logger_instance = logging.getLogger(f"Worker_LSTM_Cluster_{cluster_id_worker}")
    worker_logger_instance.info(f"Proceso worker iniciado para cluster {cluster_id_worker}.")
    
    if 'data_etl' in globals() and hasattr(globals()['data_etl'], 'cargar_configuracion_etl'):
        cargar_configuracion_etl(global_config_dict_worker)
    if 'models.lstm_model' in globals() and hasattr(globals()['models.lstm_model'], 'cargar_configuracion_lstm'):
        cargar_configuracion_lstm(global_config_dict_worker)
    if 'alerts.alert_system' in globals() and hasattr(globals()['alerts.alert_system'], 'cargar_configuracion_alertas'):
        cargar_configuracion_alertas(global_config_dict_worker)

    worker_logger_instance.info(f"Normalizando datos para cluster {cluster_id_worker}...")
    df_cluster_norm_worker, scalers_cluster_col_worker = normalizar_variables_por_grupo(
        df_cluster_actual_worker, columnas_a_escalar_worker, metodo_escalado_worker
    )
    scalers_para_este_cluster_worker = {}
    if 'global' in scalers_cluster_col_worker: 
        scalers_para_este_cluster_worker = scalers_cluster_col_worker['global']
    elif cluster_id_worker in scalers_cluster_col_worker and isinstance(scalers_cluster_col_worker[cluster_id_worker], dict):
        scalers_para_este_cluster_worker = scalers_cluster_col_worker[cluster_id_worker]
    elif isinstance(scalers_cluster_col_worker, dict) and not any(isinstance(v, dict) for v in scalers_cluster_col_worker.values()):
        scalers_para_este_cluster_worker = scalers_cluster_col_worker
    else:
        worker_logger_instance.warning(f"Estructura de scalers no reconocida para cluster {cluster_id_worker}.")

    worker_logger_instance.info(f"Dividiendo datos para cluster {cluster_id_worker}...")
    etl_cfg_worker_local = global_config_dict_worker.get('etl', {})
    df_train_c_worker, df_val_c_worker, _ = dividir_datos_chrono_por_grupo( 
        df_cluster_norm_worker, 'cliente_id', 
        etl_cfg_worker_local.get('test_size', 0.2), 
        etl_cfg_worker_local.get('validation_size', 0.15) 
    )
    if df_train_c_worker.empty: 
        worker_logger_instance.warning(f"Sin datos de entrenamiento para cluster {cluster_id_worker}. Omitiendo.")
        return cluster_id_worker, None, None, None, scalers_para_este_cluster_worker

    worker_logger_instance.info(f"Preparando secuencias y HPO para LSTM en cluster {cluster_id_worker}...")
    X_train_seq_worker = preparar_datos_secuencia(df_train_c_worker, features_para_lstm_worker, seq_len_lstm_worker)
    X_val_seq_worker = preparar_datos_secuencia(df_val_c_worker, features_para_lstm_worker, seq_len_lstm_worker) if not df_val_c_worker.empty else None

    if X_train_seq_worker.shape[0] == 0: 
        worker_logger_instance.warning(f"No se generaron secuencias de entrenamiento para cluster {cluster_id_worker}. Omitiendo.")
        return cluster_id_worker, None, None, None, scalers_para_este_cluster_worker

    n_features_lstm_worker = X_train_seq_worker.shape[2] 
    
    modelo_lstm_optimo_worker, params_lstm_optimos_worker, stats_err_val_lstm_optimo_worker = run_hyperparameter_optimization_lstm(
        X_train_seq_worker, X_val_seq_worker, param_grid_lstm_worker, 
        seq_len_lstm_worker, n_features_lstm_worker, cluster_id_worker, global_config_dict_worker
    )

    if not modelo_lstm_optimo_worker: 
        worker_logger_instance.warning(f"No se pudo entrenar/optimizar un modelo LSTM para cluster {cluster_id_worker}.")
        return cluster_id_worker, None, None, None, scalers_para_este_cluster_worker

    ruta_directorio_modelo_cluster = os.path.join(rutas_modelos_entrenados_worker, f"cluster_{cluster_id_worker}")
    os.makedirs(ruta_directorio_modelo_cluster, exist_ok=True)
    ruta_modelo_lstm_guardado = os.path.join(ruta_directorio_modelo_cluster, "lstm_ae_model.keras") 
    
    guardar_modelo_lstm(modelo_lstm_optimo_worker, ruta_modelo_lstm_guardado)
    
    umbral_lstm_final_worker = np.inf 
    umbral_calculado_sobre_info = "ninguno (falló cálculo o no hay datos)"
    stats_errores_normales_final_worker = stats_err_val_lstm_optimo_worker 
    
    # Usar el nuevo sigma_factor_deteccion_inicial para determinar el umbral de detección
    sigma_factor_deteccion = global_config_dict_worker.get('alertas',{}).get('sigma_factor_deteccion_inicial', 2.5) # Fallback a 2.5 si no está en config
    worker_logger_instance.info(f"Usando sigma_factor_deteccion_inicial = {sigma_factor_deteccion} para determinar umbral de anomalía.")


    if X_val_seq_worker is not None and X_val_seq_worker.shape[0] > 0:
        umbral_lstm_final_worker = determinar_umbral_anomalia_lstm(
            modelo_lstm_optimo_worker, X_val_seq_worker, sigma_factor_deteccion # Usar el nuevo sigma_factor
        )
        umbral_calculado_sobre_info = "validación"
        if stats_errores_normales_final_worker is None: 
            val_errors_final_check = calcular_errores_reconstruccion(modelo_lstm_optimo_worker, X_val_seq_worker)
            if val_errors_final_check.size > 0:
                stats_errores_normales_final_worker = calcular_estadisticas_scores_normales(val_errors_final_check)
    elif X_train_seq_worker.shape[0] > 0: 
        worker_logger_instance.warning(f"No hay datos de validación para cluster {cluster_id_worker}. Umbral y stats de error se basarán en entrenamiento.")
        train_errors_for_fallback = calcular_errores_reconstruccion(modelo_lstm_optimo_worker, X_train_seq_worker)
        if train_errors_for_fallback.size > 0:
            media_err_train, std_err_train = calcular_estadisticas_scores_normales(train_errors_for_fallback)
            umbral_lstm_final_worker = media_err_train + sigma_factor_deteccion * std_err_train # Usar el nuevo sigma_factor
            umbral_calculado_sobre_info = "entrenamiento (fallback)"
            if stats_errores_normales_final_worker is None: 
                stats_errores_normales_final_worker = (media_err_train, std_err_train)
    
    worker_logger_instance.info(f"Umbral LSTM para cluster {cluster_id_worker} (calculado sobre {umbral_calculado_sobre_info}): {umbral_lstm_final_worker:.6f}")
    if stats_errores_normales_final_worker:
        worker_logger_instance.info(f"Stats error (media, std) para datos normales (cluster {cluster_id_worker}): {stats_errores_normales_final_worker}")
    else:
        worker_logger_instance.warning(f"No se pudieron determinar stats de error para datos normales (cluster {cluster_id_worker}).")

    return cluster_id_worker, ruta_modelo_lstm_guardado, umbral_lstm_final_worker, stats_errores_normales_final_worker, scalers_para_este_cluster_worker


def run_full_pipeline(args_cli):
    """
    Ejecuta el pipeline completo de detección de anomalías.
    """
    start_time_total_pipeline = time.time()
    logger.info("====== INICIANDO PIPELINE DE DETECCIÓN DE ANOMALÍAS ======")
    if not CONFIG: 
        logger.critical("Configuración global (CONFIG) no disponible. Abortando pipeline.")
        return

    rutas_cfg_pipeline = CONFIG.get('rutas', {})
    etl_cfg_pipeline = CONFIG.get('etl', {})
    modelos_cfg_pipeline = CONFIG.get('modelos', {})
    
    logger.info("\n--- Etapa 1: Ejecutando Pipeline ETL ---")
    start_time_etl_pipeline = time.time()
    ruta_preprocesado_cfg = rutas_cfg_pipeline.get('preprocesado', 'data/preprocessed.csv')
    ruta_preprocesado_abs = os.path.join(BASE_DIR, ruta_preprocesado_cfg) if not os.path.isabs(ruta_preprocesado_cfg) else ruta_preprocesado_cfg
    df_preprocesado_etl = run_etl_pipeline(CONFIG, ruta_preprocesado_abs) 
    if df_preprocesado_etl.empty:
        logger.error("Pipeline abortado: Falló la etapa de ETL.")
        return
    tiempo_etl_pipeline = time.time() - start_time_etl_pipeline
    logger.info(f"Pipeline ETL completado en {tiempo_etl_pipeline:.2f} segundos. Shape: {df_preprocesado_etl.shape}")

    logger.info("\n--- Etapa 2: Ejecutando Clustering de Clientes ---")
    start_time_clustering_pipeline = time.time()
    df_asignaciones_cluster_pipeline = pipeline_cliente_clustering(CONFIG, df_preprocesado_etl)
    if df_asignaciones_cluster_pipeline.empty or 'cluster_id' not in df_asignaciones_cluster_pipeline.columns:
        logger.warning("No se generaron clusters. Se procederá con un único cluster global (cluster_id=0).")
        df_asignaciones_cluster_pipeline = pd.DataFrame({
            'cliente_id': df_preprocesado_etl['cliente_id'].unique(), 
            'cluster_id': 0 
        })
    df_trabajo_pipeline = pd.merge(df_preprocesado_etl, df_asignaciones_cluster_pipeline, on='cliente_id', how='left')
    df_trabajo_pipeline['cluster_id'] = df_trabajo_pipeline['cluster_id'].fillna(-1).astype(int) 
    tiempo_clustering_pipeline = time.time() - start_time_clustering_pipeline
    logger.info(f"Clustering de clientes completado en {tiempo_clustering_pipeline:.2f} segundos. Clusters: {df_trabajo_pipeline['cluster_id'].nunique()}.")
    logger.info(f"Distribución de clientes por cluster:\n{df_trabajo_pipeline['cluster_id'].value_counts().sort_index()}")

    logger.info("\n--- Etapa 3: Entrenamiento de Modelos LSTM por Cluster (Paralelizado) ---")
    start_time_entrenamiento_total_pipeline = time.time()
    gas_law_output_feat_name_train = CONFIG.get('feature_engineering', {}).get('gas_law', {}).get('output_feature_name', 'cantidad_gas_calculada')
    columnas_a_escalar_base_train = etl_cfg_pipeline.get('columnas_numericas', [])[:] 
    if CONFIG.get('feature_engineering', {}).get('gas_law', {}).get('activo', False) and gas_law_output_feat_name_train in df_trabajo_pipeline.columns:
        if gas_law_output_feat_name_train not in columnas_a_escalar_base_train: 
             columnas_a_escalar_base_train.append(gas_law_output_feat_name_train)
    features_para_lstm_train = modelos_cfg_pipeline.get('features_entrenamiento', []) 
    metodo_escalado_train = etl_cfg_pipeline.get('normalizacion', 'standard_scaler')
    seq_len_lstm_train = modelos_cfg_pipeline.get('lstm_autoencoder', {}).get('seq_len', 24)
    param_grid_lstm_train = modelos_cfg_pipeline.get('lstm_autoencoder', {}).get('param_grid', {}) 
    
    rutas_modelos_entrenados_train_cfg = rutas_cfg_pipeline.get('modelos_entrenados', 'models/trained/')
    if not os.path.isabs(rutas_modelos_entrenados_train_cfg):
        rutas_modelos_entrenados_train_cfg = os.path.join(BASE_DIR, rutas_modelos_entrenados_train_cfg)

    scalers_por_cluster_col_train = {} 
    modelos_lstm_por_cluster_train = {} 
    umbrales_lstm_por_cluster_train = {}
    stats_errores_val_lstm_por_cluster_train = {} 

    tasks_para_pool_entrenamiento = []
    cluster_ids_validos_entrenamiento = [cid for cid in df_trabajo_pipeline['cluster_id'].unique() if cid != -1]

    for cluster_id_iter_train in cluster_ids_validos_entrenamiento:
        df_cluster_actual_para_train = df_trabajo_pipeline[df_trabajo_pipeline['cluster_id'] == cluster_id_iter_train].copy()
        if df_cluster_actual_para_train.empty:
            logger.warning(f"Cluster {cluster_id_iter_train} sin datos. Omitiendo.")
            continue
        tasks_para_pool_entrenamiento.append(
            (cluster_id_iter_train, df_cluster_actual_para_train, columnas_a_escalar_base_train, metodo_escalado_train,
             features_para_lstm_train, seq_len_lstm_train, param_grid_lstm_train,
             rutas_modelos_entrenados_train_cfg, CONFIG) 
        )
    
    num_procesos_pool = min(len(tasks_para_pool_entrenamiento), os.cpu_count() if os.cpu_count() else 1, 4) 
    logger.info(f"Iniciando entrenamiento paralelo para {len(tasks_para_pool_entrenamiento)} clusters usando {num_procesos_pool} procesos.")

    results_entrenamiento = [] 
    if tasks_para_pool_entrenamiento and num_procesos_pool > 0:
        pool = None
        try:
            pool = multiprocessing.Pool(processes=num_procesos_pool)
            results_entrenamiento = pool.map(train_lstm_for_cluster_worker, tasks_para_pool_entrenamiento)
        except Exception as e_pool:
            logger.error(f"Error durante ejecución del pool de multiprocessing para entrenamiento: {e_pool}", exc_info=True)
        finally:
            if pool:
                pool.close() 
                pool.join()  
        
        for result_item_train in results_entrenamiento:
            if result_item_train: 
                c_id_res, ruta_modelo_res, umbral_res, stats_err_res, scalers_c_res = result_item_train
                if ruta_modelo_res and os.path.exists(ruta_modelo_res): 
                    modelos_lstm_por_cluster_train[c_id_res] = cargar_modelo_lstm(ruta_modelo_res) 
                    umbrales_lstm_por_cluster_train[c_id_res] = umbral_res
                    stats_errores_val_lstm_por_cluster_train[c_id_res] = stats_err_res 
                    scalers_por_cluster_col_train[c_id_res] = scalers_c_res
                elif scalers_c_res: 
                    logger.warning(f"Modelo para cluster {c_id_res} no guardado/cargado, pero se recuperaron scalers.")
                    scalers_por_cluster_col_train[c_id_res] = scalers_c_res
    else:
        logger.info("No hay tareas de entrenamiento para ejecutar en paralelo o num_procesos_pool es 0.")

    tiempo_entrenamiento_total_pipeline = time.time() - start_time_entrenamiento_total_pipeline
    logger.info(f"Entrenamiento de modelos LSTM por cluster (paralelo) completado en {tiempo_entrenamiento_total_pipeline:.2f} segundos.")

    logger.info("\n--- Etapa 4: Detección de Anomalías con Modelos LSTM por Cluster ---")
    start_time_deteccion_pipeline = time.time()
    df_resultados_finales_list_pipeline = []

    for cluster_id_val_det in df_trabajo_pipeline['cluster_id'].unique():
        cluster_id_det = int(cluster_id_val_det) 
        if cluster_id_det == -1: 
            logger.info(f"Omitiendo detección para cluster_id {cluster_id_det}.")
            continue 

        logger.info(f"\n--- Detectando anomalías para Cluster ID: {cluster_id_det} (Usando LSTM) ---")
        df_datos_cluster_para_deteccion = df_trabajo_pipeline[df_trabajo_pipeline['cluster_id'] == cluster_id_det].copy()
        if df_datos_cluster_para_deteccion.empty:
            logger.warning(f"No hay datos para cluster {cluster_id_det} en detección.")
            continue
        
        df_scaled_para_deteccion_det = df_datos_cluster_para_deteccion.copy()
        scalers_para_este_cluster_det = scalers_por_cluster_col_train.get(cluster_id_det, {})

        if not scalers_para_este_cluster_det:
            logger.warning(f"[Cluster {cluster_id_det} Detección] No scalers. Usando columnas '_scaled' o originales.")
        
        for col_base_det in columnas_a_escalar_base_train: 
            scaled_col_name_det = f"{col_base_det}_scaled"
            if col_base_det in df_scaled_para_deteccion_det.columns: 
                scaler_obj_det = scalers_para_este_cluster_det.get(col_base_det)
                if scaler_obj_det: 
                    col_data_to_scale_det = df_scaled_para_deteccion_det[[col_base_det]].copy()
                    if col_data_to_scale_det[col_base_det].isnull().any():
                        col_data_to_scale_det[col_base_det] = col_data_to_scale_det[col_base_det].fillna(0) 
                    try:
                        df_scaled_para_deteccion_det[scaled_col_name_det] = scaler_obj_det.transform(col_data_to_scale_det[[col_base_det]])
                    except Exception as e_scale_det:
                        logger.error(f"Error aplicando scaler para '{col_base_det}' en cluster {cluster_id_det} (detección): {e_scale_det}.")
                        if scaled_col_name_det not in df_scaled_para_deteccion_det.columns:
                             df_scaled_para_deteccion_det[scaled_col_name_det] = df_scaled_para_deteccion_det[col_base_det]
                elif scaled_col_name_det not in df_scaled_para_deteccion_det.columns: 
                     logger.warning(f"[Cluster {cluster_id_det} Detección] No scaler para '{col_base_det}' y '{scaled_col_name_det}' no existe. Usando original.")
                     df_scaled_para_deteccion_det[scaled_col_name_det] = df_scaled_para_deteccion_det[col_base_det]
            elif scaled_col_name_det in df_scaled_para_deteccion_det.columns: 
                logger.debug(f"[Cluster {cluster_id_det} Detección] Usando '{scaled_col_name_det}' preexistente.")
            else: 
                 logger.warning(f"[Cluster {cluster_id_det} Detección] Columna '{col_base_det}' ni '{scaled_col_name_det}' encontradas.")

        missing_lstm_features_for_detection = [f for f in features_para_lstm_train if f not in df_scaled_para_deteccion_det.columns]
        if missing_lstm_features_for_detection:
            logger.error(f"[Cluster {cluster_id_det} Detección] Faltan features LSTM: {missing_lstm_features_for_detection}. Omitiendo.")
            continue
            
        anomalias_lstm_flags_det = np.zeros(len(df_scaled_para_deteccion_det), dtype=bool)
        scores_lstm_det = np.full(len(df_scaled_para_deteccion_det), np.nan)
        
        modelo_lstm_para_deteccion = modelos_lstm_por_cluster_train.get(cluster_id_det)
        umbral_lstm_para_deteccion = umbrales_lstm_por_cluster_train.get(cluster_id_det)
        stats_errores_normales_para_criticidad = stats_errores_val_lstm_por_cluster_train.get(cluster_id_det) 

        if stats_errores_normales_para_criticidad:
            logger.info(f"[Cluster {cluster_id_det} Criticidad PREP] Stats error normal (media, std): {stats_errores_normales_para_criticidad}")
        else:
            logger.warning(f"[Cluster {cluster_id_det} Criticidad PREP] No se encontraron stats de error normal para este cluster.")

        if modelo_lstm_para_deteccion and umbral_lstm_para_deteccion is not None and umbral_lstm_para_deteccion != np.inf:
            X_det_lstm_seq_cluster = preparar_datos_secuencia(df_scaled_para_deteccion_det, features_para_lstm_train, seq_len_lstm_train)
            if X_det_lstm_seq_cluster.shape[0] > 0: 
                flags_seq_det, errors_seq_det = detectar_anomalias_lstm(
                    modelo_lstm_para_deteccion, X_det_lstm_seq_cluster, umbral_lstm_para_deteccion
                )
                for i in range(len(flags_seq_det)):
                    original_df_index_pos_det = i + seq_len_lstm_train - 1 
                    if original_df_index_pos_det < len(df_scaled_para_deteccion_det):
                        if flags_seq_det[i]: 
                            anomalias_lstm_flags_det[original_df_index_pos_det] = True
                        scores_lstm_det[original_df_index_pos_det] = errors_seq_det[i]
            else:
                logger.warning(f"[Cluster {cluster_id_det} Detección] No secuencias LSTM para detección.")
        else:
            logger.warning(f"[Cluster {cluster_id_det} Detección] Modelo LSTM o umbral no disponible.")

        df_scaled_para_deteccion_det['anomalia_predicha'] = anomalias_lstm_flags_det
        df_scaled_para_deteccion_det['score_anomalia_final'] = scores_lstm_det
        df_scaled_para_deteccion_det['modelo_deteccion'] = "LSTM_AE_Cluster" 
        df_scaled_para_deteccion_det['criticidad_predicha'] = "Normal" 

        if stats_errores_normales_para_criticidad: 
            media_err_norm_crit, std_err_norm_crit = stats_errores_normales_para_criticidad
            logger.info(f"[Cluster {cluster_id_det} Criticidad CALL] Usando media_error_normal={media_err_norm_crit:.6f}, std_error_normal={std_err_norm_crit:.6f} para clasificación.")
            for idx_row_label_anom in df_scaled_para_deteccion_det[df_scaled_para_deteccion_det['anomalia_predicha']].index:
                score_actual_anom = df_scaled_para_deteccion_det.loc[idx_row_label_anom, 'score_anomalia_final']
                if not pd.isna(score_actual_anom): 
                    df_scaled_para_deteccion_det.loc[idx_row_label_anom, 'criticidad_predicha'] = clasificar_criticidad_anomalia(
                        score_actual_anom, media_err_norm_crit, std_err_norm_crit
                    )
        else: 
            logger.warning(f"[Cluster {cluster_id_det} Criticidad CALL] No stats errores normales. Criticidad será 'Indeterminada'.")
            df_scaled_para_deteccion_det.loc[df_scaled_para_deteccion_det['anomalia_predicha'], 'criticidad_predicha'] = "Indeterminada"
        
        df_resultados_finales_list_pipeline.append(df_scaled_para_deteccion_det)

    df_resultados_completos_pipeline = pd.concat(df_resultados_finales_list_pipeline).sort_values(['cliente_id', 'Fecha']) if df_resultados_finales_list_pipeline else pd.DataFrame()
    tiempo_deteccion_pipeline = time.time() - start_time_deteccion_pipeline
    logger.info(f"Detección de anomalías (LSTM por Cluster) completada en {tiempo_deteccion_pipeline:.2f} segundos.")
    
    df_anomalias_detectadas_para_reporte = df_resultados_completos_pipeline[df_resultados_completos_pipeline['anomalia_predicha']].copy() if not df_resultados_completos_pipeline.empty else pd.DataFrame()
    if not df_anomalias_detectadas_para_reporte.empty:
        df_anomalias_detectadas_para_reporte['timestamp_deteccion'] = datetime.now() 
    
    ruta_anomalias_cfg = rutas_cfg_pipeline.get('anomalias', 'data/anomalies_detected.csv')
    ruta_anomalias_abs = os.path.join(BASE_DIR, ruta_anomalias_cfg) if not os.path.isabs(ruta_anomalias_cfg) else ruta_anomalias_cfg
    generar_reporte_anomalias_final(df_anomalias_detectadas_para_reporte, ruta_anomalias_abs)

    # --- Etapa 5: Evaluación del Sistema ---
    ruta_ground_truth_config = rutas_cfg_pipeline.get('ground_truth_data') 
    ruta_ground_truth_eval_abs = None
    if ruta_ground_truth_config:
        ruta_ground_truth_eval_abs = os.path.join(BASE_DIR, ruta_ground_truth_config) if not os.path.isabs(ruta_ground_truth_config) else ruta_ground_truth_config
            
    tiempo_evaluacion_pipeline = None 
    if ruta_ground_truth_eval_abs and os.path.exists(ruta_ground_truth_eval_abs):
        logger.info(f"\n--- Etapa 5: Evaluación del Sistema contra Ground Truth ({ruta_ground_truth_eval_abs}) ---")
        start_time_eval_pipeline = time.time()
        try:
            df_ground_truth_data = pd.read_csv(ruta_ground_truth_eval_abs) 
            if not df_ground_truth_data.empty:
                ruta_metricas_eval_cfg = rutas_cfg_pipeline.get('resultados_evaluacion', 'evaluation/performance_metrics.csv')
                ruta_metricas_eval_abs = os.path.join(BASE_DIR, ruta_metricas_eval_cfg) if not os.path.isabs(ruta_metricas_eval_cfg) else ruta_metricas_eval_cfg
                
                evaluar_sistema_completo(df_resultados_completos_pipeline, df_ground_truth_data, ruta_metricas_eval_abs) 
                tiempo_evaluacion_pipeline = time.time() - start_time_eval_pipeline
                logger.info(f"Evaluación del sistema completada en {tiempo_evaluacion_pipeline:.2f} segundos. Métricas guardadas en {ruta_metricas_eval_abs}")
            else:
                logger.warning(f"Archivo de ground truth '{ruta_ground_truth_eval_abs}' está vacío. Omitiendo evaluación.")
        except Exception as e_eval:
            logger.error(f"Error durante la etapa de evaluación con archivo '{ruta_ground_truth_eval_abs}': {e_eval}", exc_info=True)
    else:
        logger.info(f"Archivo de ground truth no especificado ('{ruta_ground_truth_config}') o no encontrado en ruta '{ruta_ground_truth_eval_abs}'. Omitiendo etapa de evaluación del sistema.")

    execution_times_log = {
        'etl_pipeline_seconds': round(tiempo_etl_pipeline, 2),
        'clustering_pipeline_seconds': round(tiempo_clustering_pipeline, 2),
        'training_lstm_parallel_seconds': round(tiempo_entrenamiento_total_pipeline, 2),
        'anomaly_detection_seconds': round(tiempo_deteccion_pipeline, 2),
    }
    if tiempo_evaluacion_pipeline is not None:
        execution_times_log['system_evaluation_seconds'] = round(tiempo_evaluacion_pipeline, 2)
    
    ruta_exec_times_yaml_cfg = rutas_cfg_pipeline.get('execution_times', 'logs/execution_times.yaml')
    ruta_exec_times_yaml_abs = os.path.join(BASE_DIR, ruta_exec_times_yaml_cfg) if not os.path.isabs(ruta_exec_times_yaml_cfg) else ruta_exec_times_yaml_cfg
    try:
        os.makedirs(os.path.dirname(ruta_exec_times_yaml_abs), exist_ok=True)
        with open(ruta_exec_times_yaml_abs, 'w', encoding='utf-8') as f_exec_times:
            yaml.dump(execution_times_log, f_exec_times, default_flow_style=False)
        logger.info(f"Tiempos de ejecución del pipeline guardados en: {ruta_exec_times_yaml_abs}")
    except Exception as e_save_times:
        logger.error(f"Error al guardar los tiempos de ejecución: {e_save_times}", exc_info=True)

    tiempo_total_pipeline_final = time.time() - start_time_total_pipeline
    logger.info(f"\n====== PIPELINE COMPLETO FINALIZADO EN {tiempo_total_pipeline_final:.2f} SEGUNDOS ======")
    

def main():
    if os.name == 'nt': 
        os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Pipeline de Detección de Anomalías en Consumo de Gas.")
    parser.add_argument(
        '--config', 
        default='config.yaml', 
        help='Ruta al archivo de configuración YAML (default: config.yaml)'
    )
    args_cli_parsed = parser.parse_args()
    
    if not cargar_configuracion_global(args_cli_parsed.config):
        print(f"Fallo crítico durante la carga de configuración desde '{args_cli_parsed.config}'. Revise los logs. Abortando ejecución.")
        return 
        
    run_full_pipeline(args_cli_parsed)

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main()


