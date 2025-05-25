# ==============================================================================
# models/lstm_model.py
# Implementación de Autoencoder LSTM para detección de anomalías.
# ==============================================================================

import os
import yaml
# Desactivar mensajes informativos de oneDNN si no se usa MKL para evitar ruido en logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suprimir logs de TensorFlow (0 = todos, 1 = INFO, 2 = WARNING, 3 = ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Mostrar solo errores de TF


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)
CONFIG = {} # Variable global para la configuración, se cargará desde main.py

def cargar_configuracion_lstm(cfg: dict):
    """
    Carga la configuración globalmente para este módulo.

    Args:
        cfg (dict): Diccionario de configuración global.
    """
    global CONFIG
    CONFIG = cfg
    logger.info("Configuración LSTM cargada en lstm_model.py.")

def preparar_datos_secuencia(data_df: pd.DataFrame, feature_cols: list, seq_len: int) -> np.ndarray:
    """
    Convierte un DataFrame de series temporales (ya escalado y con features) 
    en secuencias de entrada adecuadas para un modelo LSTM.

    Args:
        data_df (pd.DataFrame): DataFrame que contiene los datos. Debe tener las 'feature_cols'.
        feature_cols (list): Lista de nombres de columnas a incluir en las secuencias.
        seq_len (int): Longitud de cada secuencia (número de pasos de tiempo).

    Returns:
        np.ndarray: Array de NumPy con forma (num_sequences, seq_len, num_features),
                    o un array vacío si no se pueden generar secuencias.
    """
    if data_df.empty or not feature_cols:
        logger.warning("DataFrame vacío o sin columnas de features especificadas para preparar secuencias LSTM.")
        return np.array([])
    
    # Verificar que todas las feature_cols existan en el DataFrame
    missing_cols = [col for col in feature_cols if col not in data_df.columns]
    if missing_cols:
        logger.error(f"Columnas de features faltantes en el DataFrame para LSTM: {missing_cols}. No se pueden crear secuencias.")
        return np.array([])

    # Convertir las columnas seleccionadas a un array NumPy de tipo float32
    arr_data = data_df[feature_cols].values.astype(np.float32)
    
    if len(arr_data) < seq_len:
        logger.warning(f"No hay suficientes datos (filas: {len(arr_data)}) para crear secuencias de longitud {seq_len}.")
        return np.array([])
        
    sequences_list = []
    # Deslizar una ventana de tamaño seq_len sobre los datos
    for i in range(len(arr_data) - seq_len + 1):
        sequences_list.append(arr_data[i:i + seq_len])
    
    if not sequences_list: # Si, por alguna razón, la lista está vacía
        logger.warning("No se generaron secuencias a pesar de haber suficientes datos iniciales.")
        return np.array([])
        
    return np.array(sequences_list, dtype=np.float32)

def crear_lstm_autoencoder(seq_len: int, n_features: int, encoding_dim: int = 32, learning_rate: float = 0.001) -> Model:
    """
    Crea y compila un modelo Autoencoder LSTM.

    La arquitectura consiste en un encoder LSTM que comprime la secuencia de entrada
    a un vector de menor dimensionalidad (encoding_dim), y un decoder LSTM
    que intenta reconstruir la secuencia original a partir de este vector codificado.

    Args:
        seq_len (int): Longitud de las secuencias de entrada (número de pasos de tiempo).
        n_features (int): Número de características en cada paso de tiempo de la secuencia.
        encoding_dim (int, optional): Dimensión del espacio latente (cuello de botella). Defaults to 32.
        learning_rate (float, optional): Tasa de aprendizaje para el optimizador Adam. Defaults to 0.001.

    Returns:
        tf.keras.models.Model: El modelo Autoencoder LSTM compilado.
    
    Raises:
        ValueError: Si los parámetros de dimensión o secuencia no son positivos.
    """
    if seq_len <= 0 or n_features <= 0 or encoding_dim <= 0:
        logger.error(f"Parámetros inválidos para crear LSTM AE: seq_len={seq_len}, n_features={n_features}, encoding_dim={encoding_dim}")
        raise ValueError("Los parámetros de dimensión y secuencia deben ser positivos.")

    # Definición del modelo usando la API Funcional de Keras para mayor claridad
    input_seq = Input(shape=(seq_len, n_features))

    # Encoder
    # Capas LSTM apiladas para capturar dependencias temporales complejas.
    # 'return_sequences=True' es necesario si la siguiente capa LSTM espera secuencias.
    encoder = LSTM(128, activation='relu', return_sequences=True)(input_seq)
    encoder = LSTM(64, activation='relu', return_sequences=False)(encoder) # La última LSTM del encoder no retorna secuencia completa
    
    # Capa densa como cuello de botella para la representación codificada.
    bottleneck = Dense(encoding_dim, activation='relu')(encoder)

    # Decoder
    # RepeatVector duplica el vector del cuello de botella 'seq_len' veces para alimentar al decoder LSTM.
    decoder_input = RepeatVector(seq_len)(bottleneck)
    
    # Capa densa opcional antes de las LSTM del decoder
    decoder = Dense(encoding_dim, activation='relu')(decoder_input) 
    decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
    decoder = LSTM(128, activation='relu', return_sequences=True)(decoder)
    
    # TimeDistributed aplica una capa Dense a cada paso de tiempo de la salida del LSTM decoder,
    # para reconstruir las 'n_features' originales en cada paso.
    output_seq = TimeDistributed(Dense(n_features))(decoder)

    model = Model(inputs=input_seq, outputs=output_seq)
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error es una función de pérdida común para autoencoders.
    
    logger.debug(f"Modelo LSTM Autoencoder creado con: seq_len={seq_len}, n_features={n_features}, encoding_dim={encoding_dim}, lr={learning_rate}")
    # model.summary(print_fn=logger.debug) # Descomentar para ver el resumen del modelo en logs
    return model

def entrenar_lstm_ae(model: Model, 
                     X_train_seq: np.ndarray, 
                     X_val_seq: np.ndarray | None, # Hacer X_val_seq opcional
                     epochs: int = 50, 
                     batch_size: int = 32,
                     patience_early_stopping: int = 10,
                     patience_reduce_lr: int = 5) -> tf.keras.callbacks.History:
    """
    Entrena el modelo Autoencoder LSTM.

    Utiliza EarlyStopping para detener el entrenamiento si la pérdida de validación no mejora,
    y ReduceLROnPlateau para disminuir la tasa de aprendizaje si el entrenamiento se estanca.

    Args:
        model (tf.keras.models.Model): El modelo Autoencoder LSTM a entrenar.
        X_train_seq (np.ndarray): Secuencias de datos de entrenamiento.
        X_val_seq (np.ndarray | None): Secuencias de datos de validación. Si es None,
                                       EarlyStopping y ReduceLROnPlateau monitorizarán 'loss'.
        epochs (int, optional): Número máximo de épocas de entrenamiento. Defaults to 50.
        batch_size (int, optional): Tamaño del lote para el entrenamiento. Defaults to 32.
        patience_early_stopping (int, optional): Paciencia para EarlyStopping. Defaults to 10.
        patience_reduce_lr (int, optional): Paciencia para ReduceLROnPlateau. Defaults to 5.

    Returns:
        tf.keras.callbacks.History: Objeto History que contiene el historial de entrenamiento.
    
    Raises:
        ValueError: Si los datos de secuencia no tienen 3 dimensiones.
    """
    if X_train_seq.ndim != 3 or (X_val_seq is not None and X_val_seq.ndim != 3 and len(X_val_seq) > 0) :
        val_shape_msg = X_val_seq.shape if X_val_seq is not None else "None"
        logger.error(f"Dimensiones de datos incorrectas para entrenamiento LSTM. X_train_seq: {X_train_seq.shape}, X_val_seq: {val_shape_msg}")
        raise ValueError("Los datos de secuencia (train y val si existe) deben tener 3 dimensiones (samples, timesteps, features).")

    callbacks_list = []
    monitor_metric = 'val_loss'
    validation_data_for_fit = None

    if X_val_seq is not None and len(X_val_seq) > 0:
        validation_data_for_fit = (X_val_seq, X_val_seq) # El autoencoder se valida reconstruyendo su propia entrada de validación
        logger.info(f"Usando datos de validación para EarlyStopping y ReduceLROnPlateau (monitor: {monitor_metric}).")
    else:
        monitor_metric = 'loss' # Si no hay datos de validación, monitorizar la pérdida de entrenamiento
        logger.warning(f"No se proporcionaron datos de validación. EarlyStopping y ReduceLROnPlateau monitorizarán '{monitor_metric}'.")
        # Aumentar la paciencia si se monitoriza 'loss' ya que puede fluctuar más.
        patience_early_stopping *= 2 
        patience_reduce_lr = max(3, patience_reduce_lr) # Asegurar una paciencia mínima para LR


    callbacks_list.append(
        EarlyStopping(monitor=monitor_metric, patience=patience_early_stopping, restore_best_weights=True, verbose=1)
    )
    callbacks_list.append(
        ReduceLROnPlateau(monitor=monitor_metric, factor=0.2, patience=patience_reduce_lr, min_lr=1e-6, verbose=1)
    )

    num_train_samples = len(X_train_seq) if X_train_seq is not None else 0
    num_val_samples = len(X_val_seq) if X_val_seq is not None else 0

    logger.info(f"Iniciando entrenamiento LSTM AE: epochs={epochs}, batch_size={batch_size}, train_samples={num_train_samples}, val_samples={num_val_samples}")
    
    history = model.fit(
        X_train_seq, X_train_seq, # El autoencoder aprende a reconstruir su entrada
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data_for_fit,
        callbacks=callbacks_list,
        verbose=1 # 0 = silent, 1 = progress bar, 2 = one line per epoch.
    )
    logger.info("Entrenamiento LSTM AE completado.")
    return history

def calcular_errores_reconstruccion(model: Model, X_seq: np.ndarray) -> np.ndarray:
    """
    Calcula el error de reconstrucción (ej. Mean Absolute Error - MAE) para cada secuencia de entrada.

    Args:
        model (tf.keras.models.Model): El modelo Autoencoder LSTM entrenado.
        X_seq (np.ndarray): Secuencias de datos para las cuales calcular el error.

    Returns:
        np.ndarray: Array de errores de reconstrucción, uno por secuencia.
                    Array vacío si la entrada X_seq está vacía o tiene dimensiones incorrectas.
    """
    if X_seq.ndim != 3:
        logger.error(f"Dimensiones de datos incorrectas para calcular error de reconstrucción. Shape: {X_seq.shape}")
        return np.array([])
    if len(X_seq) == 0:
        logger.debug("X_seq está vacío, no se calculan errores de reconstrucción.")
        return np.array([])

    logger.debug(f"Calculando errores de reconstrucción para {len(X_seq)} secuencias.")
    # Predecir (reconstruir) las secuencias de entrada
    X_pred_seq = model.predict(X_seq, verbose=0) # verbose=0 para suprimir logs de predicción
    
    # Calcular el Mean Absolute Error (MAE) para cada secuencia.
    # MAE = mean(|X_seq - X_pred_seq|) a lo largo de los pasos de tiempo y características.
    # axis=(1, 2) promedia sobre la dimensión de pasos de tiempo (1) y la dimensión de features (2).
    mae_per_sequence = np.mean(np.abs(X_seq - X_pred_seq), axis=(1,2))
    
    # Alternativamente, se podría usar MSE:
    # mse_per_sequence = np.mean(np.power(X_seq - X_pred_seq, 2), axis=(1, 2))

    return mae_per_sequence

def determinar_umbral_anomalia_lstm(model: Model, X_val_seq: np.ndarray | None, sigma_factor: float = 3.0) -> float:
    """
    Determina un umbral para la detección de anomalías basado en los errores de reconstrucción
    de un conjunto de datos de validación (que se asume representa comportamiento normal).
    El umbral se establece típicamente como: media_errores_normales + sigma_factor * std_dev_errores_normales.

    Args:
        model (tf.keras.models.Model): El modelo Autoencoder LSTM entrenado.
        X_val_seq (np.ndarray | None): Secuencias de datos de validación (normales).
                                     Si es None o vacío, se retorna un umbral infinito (no se detectarán anomalías).
        sigma_factor (float, optional): Factor multiplicador para la desviación estándar. Defaults to 3.0.

    Returns:
        float: El umbral de error de reconstrucción para identificar anomalías.
               Retorna np.inf si no se pueden calcular errores de validación.
    """
    if X_val_seq is None or len(X_val_seq) == 0:
        logger.warning("No hay datos de validación (X_val_seq) para determinar el umbral LSTM. Se devolverá un umbral infinito (np.inf), lo que significa que no se detectarán anomalías.")
        return np.inf 
        
    logger.info(f"Determinando umbral de anomalía LSTM usando {len(X_val_seq)} secuencias de validación y sigma_factor={sigma_factor}.")
    val_errors = calcular_errores_reconstruccion(model, X_val_seq)
    
    if len(val_errors) == 0: # Si calcular_errores_reconstruccion retornó array vacío
        logger.warning("No se pudieron calcular errores de validación para el umbral LSTM. Retornando np.inf.")
        return np.inf

    mean_error_val = np.mean(val_errors)
    std_error_val = np.std(val_errors)
    
    # Si la std es 0 (todos los errores de validación son idénticos),
    # la adición de sigma_factor * std_error no tendrá efecto.
    # Se podría añadir un pequeño épsilon o manejarlo de otra forma si esto es un problema.
    if std_error_val == 0:
        logger.warning("La desviación estándar de los errores de validación es 0. El umbral será igual a la media de los errores.")

    umbral = mean_error_val + sigma_factor * std_error_val
    
    logger.info(f"Umbral LSTM determinado: {umbral:.6f} (media_error_val={mean_error_val:.6f}, std_error_val={std_error_val:.6f})")
    return umbral

def detectar_anomalias_lstm(model: Model, X_data_seq: np.ndarray, umbral: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta anomalías en un conjunto de secuencias de datos comparando su error de reconstrucción
    con un umbral predefinido.

    Args:
        model (tf.keras.models.Model): El modelo Autoencoder LSTM entrenado.
        X_data_seq (np.ndarray): Secuencias de datos a evaluar.
        umbral (float): El umbral de error de reconstrucción. Errores por encima de este umbral
                        se consideran anomalías.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - anomalias_flags (np.ndarray): Array booleano, True si la secuencia es anómala.
            - errores (np.ndarray): Array de los errores de reconstrucción para cada secuencia.
            Ambos arrays son vacíos si X_data_seq está vacío.
    """
    if len(X_data_seq) == 0:
        logger.debug("X_data_seq vacío, no se realiza detección de anomalías.")
        return np.array([], dtype=bool), np.array([])

    errores = calcular_errores_reconstruccion(model, X_data_seq)
    if len(errores) == 0: # Si calcular_errores_reconstruccion falló
        return np.array([], dtype=bool), np.array([])
        
    anomalias_flags = errores > umbral
    
    num_anomalias = anomalias_flags.sum()
    logger.debug(f"Detección LSTM: {num_anomalias} anomalías encontradas sobre {len(X_data_seq)} secuencias usando umbral {umbral:.6f}.")
    return anomalias_flags, errores

def guardar_modelo_lstm(model: Model, ruta_archivo: str):
    """
    Guarda el modelo Keras entrenado en el disco.
    Se recomienda el formato ".keras" para modelos más recientes.

    Args:
        model (tf.keras.models.Model): El modelo Keras a guardar.
        ruta_archivo (str): Ruta completa (incluyendo nombre de archivo y extensión, ej. "modelo.keras")
                            donde se guardará el modelo.
    """
    try:
        directorio = os.path.dirname(ruta_archivo)
        if directorio: # Crear directorio si no existe
            os.makedirs(directorio, exist_ok=True)
        model.save(ruta_archivo) 
        logger.info(f"Modelo LSTM guardado exitosamente en: {ruta_archivo}")
    except Exception as e:
        logger.error(f"Error al guardar el modelo LSTM en '{ruta_archivo}': {e}", exc_info=True)

def cargar_modelo_lstm(ruta_archivo: str) -> Model | None:
    """
    Carga un modelo Keras previamente guardado desde el disco.

    Args:
        ruta_archivo (str): Ruta completa al archivo del modelo (ej. "modelo.keras").

    Returns:
        tf.keras.models.Model | None: El modelo Keras cargado, o None si ocurre un error.
    """
    if not os.path.exists(ruta_archivo):
        logger.error(f"No se encontró el archivo del modelo LSTM en la ruta especificada: {ruta_archivo}")
        return None
    try:
        # No se necesitan custom_objects si se usan capas estándar de Keras.
        # Si se usaron capas personalizadas, se deben pasar aquí.
        model = load_model(ruta_archivo)
        logger.info(f"Modelo LSTM cargado exitosamente desde: {ruta_archivo}")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo LSTM desde '{ruta_archivo}': {e}", exc_info=True)
        return None
    
def guardar_historial_entrenamiento(historial: tf.keras.callbacks.History, ruta_archivo: str):
    """
    Guarda el historial de entrenamiento del modelo en un archivo YAML.

    Args:
        historial (tf.keras.callbacks.History): Objeto History que contiene el historial de entrenamiento.
        ruta_archivo (str): Ruta completa (incluyendo nombre de archivo y extensión, ej. "historial_entrenamiento.yaml")
                            donde se guardará el historial.
    """
    try:
        # Convertir el historial a un diccionario
        historial_dict = {k: v for k, v in historial.history.items()}
        with open(ruta_archivo, 'w') as file:
            yaml.dump(historial_dict, file)
        logger.info(f"Historial de entrenamiento guardado exitosamente en: {ruta_archivo}")
    except Exception as e:
        logger.error(f"Error al guardar el historial de entrenamiento en '{ruta_archivo}': {e}", exc_info=True)
        # Manejo de errores adicional (si es necesario)

def cargar_historial_entrenamiento(ruta_archivo: str) -> dict | None:
    """
    Carga el historial de entrenamiento desde un archivo YAML.

    Args:
        ruta_archivo (str): Ruta completa al archivo del historial (ej. "historial_entrenamiento.yaml").

    Returns:
        dict | None: Diccionario con el historial de entrenamiento, o None si ocurre un error.
    """
    if not os.path.exists(ruta_archivo):
        logger.error(f"No se encontró el archivo del historial de entrenamiento en la ruta especificada: {ruta_archivo}")
        return None
    try:
        with open(ruta_archivo, 'r') as file:
            historial_dict = yaml.safe_load(file)
        logger.info(f"Historial de entrenamiento cargado exitosamente desde: {ruta_archivo}")
        return historial_dict
    except Exception as e:
        logger.error(f"Error al cargar el historial de entrenamiento desde '{ruta_archivo}': {e}", exc_info=True)
        return None
    
def resetear_grafico_tensorflow():
    """
    Resetea el gráfico de TensorFlow para evitar problemas de memoria y conflictos
    al crear nuevos modelos o capas.
    """
    tf.keras.backend.clear_session()
    logger.debug("Gráfico de TensorFlow reseteado.")

def liberar_memoria_tensorflow():
    """
    Libera memoria de GPU y CPU utilizada por TensorFlow.
    Útil para evitar problemas de memoria al cargar o entrenar nuevos modelos.
    """
    tf.keras.backend.clear_session()
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
            logger.debug("Memoria de GPU liberada y configurada para crecimiento dinámico.")
        except Exception as e:
            logger.error(f"Error al liberar memoria de GPU: {e}", exc_info=True)
    logger.debug("Memoria de TensorFlow liberada.")

def resetear_entrenamiento_lstm():
    """
    Resetea el estado del entrenamiento LSTM, liberando memoria y reseteando el gráfico de TensorFlow.
    Útil para reiniciar el proceso de entrenamiento sin conflictos de memoria.
    """
    resetear_grafico_tensorflow()
    liberar_memoria_tensorflow()
    logger.info("Estado de entrenamiento LSTM reseteado.")

