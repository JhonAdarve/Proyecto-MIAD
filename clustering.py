# ==============================================================================
# clustering.py
# Funciones para la segmentación de clientes usando K-Means.
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)
CONFIG = {} # Variable global para la configuración, se cargará desde main.py

def cargar_configuracion_clustering(cfg: dict):
    """
    Carga la configuración de clustering globalmente para este módulo.

    Args:
        cfg (dict): Diccionario de configuración global.
    """
    global CONFIG
    CONFIG = cfg
    logger.info("Configuración de clustering cargada en clustering.py.")

def preparar_features_cliente_para_clustering(df_preprocesado: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula características (features) agregadas por cliente a partir de los datos preprocesados.
    Estas características se utilizan para realizar el clustering de clientes.
    Las features a calcular y las funciones de agregación se definen en el archivo
    de configuración (`config.yaml` bajo `clustering.features_cliente_para_clustering`).

    Args:
        df_preprocesado (pd.DataFrame): DataFrame con los datos de series temporales preprocesados
                                        para todos los clientes. Debe contener 'cliente_id' y las
                                        columnas numéricas base.

    Returns:
        pd.DataFrame: DataFrame donde cada fila representa un cliente y las columnas son
                      las features agregadas calculadas. El índice es 'cliente_id'.
                      Retorna un DataFrame vacío si no se pueden generar features.
    """
    if df_preprocesado.empty:
        logger.error("DataFrame preprocesado vacío. No se pueden preparar features para clustering.")
        return pd.DataFrame()

    clustering_cfg = CONFIG.get('clustering', {})
    feature_list_config = clustering_cfg.get('features_cliente_para_clustering', [])
    
    if not feature_list_config:
        logger.error("No se han definido 'features_cliente_para_clustering' en la configuración. No se puede proceder con la preparación de features para clustering.")
        return pd.DataFrame()

    logger.info(f"Preparando características de cliente para clustering usando las siguientes definiciones: {feature_list_config}")
    
    # Determinar las columnas numéricas base disponibles para agregación
    numeric_cols_base_etl = CONFIG.get('etl', {}).get('columnas_numericas', ['Presion', 'Temperatura', 'Volumen'])
    gas_law_feature_cfg = CONFIG.get('feature_engineering', {}).get('gas_law', {})
    gas_law_active = gas_law_feature_cfg.get('activo', False)
    gas_law_output_name = gas_law_feature_cfg.get('output_feature_name', 'cantidad_gas_calculada')
    
    all_available_base_features_for_agg = numeric_cols_base_etl[:] # Crear una copia
    if gas_law_active and gas_law_output_name not in all_available_base_features_for_agg:
        all_available_base_features_for_agg.append(gas_law_output_name)

    grouped_by_client = df_preprocesado.groupby('cliente_id')
    client_profiles_list = []

    for client_id, data_cliente in grouped_by_client:
        profile = {'cliente_id': client_id}
        for feature_desc_config in feature_list_config: # Ej: 'Volumen_mean', 'Presion_std'
            try:
                # Parsear el nombre de la columna base y la función de agregación
                # Ej: 'cantidad_gas_calculada_mean' -> col_name_base='cantidad_gas_calculada', agg_func_name='mean'
                parts = feature_desc_config.split('_')
                if len(parts) < 2:
                    logger.warning(f"Formato de feature de clustering '{feature_desc_config}' no reconocido (esperado 'COLUMNA_FUNCION'). Omitiendo.")
                    profile[feature_desc_config] = np.nan
                    continue
                
                agg_func_name = parts[-1] # La función de agregación es la última parte
                col_name_base = "_".join(parts[:-1]) # El nombre base es todo lo anterior

                if col_name_base not in all_available_base_features_for_agg:
                    logger.warning(f"Columna base '{col_name_base}' (de '{feature_desc_config}') no es una columna numérica reconocida ({all_available_base_features_for_agg}). Omitiendo feature.")
                    profile[feature_desc_config] = np.nan
                    continue
                
                if col_name_base not in data_cliente.columns:
                    logger.warning(f"Columna base '{col_name_base}' no encontrada en los datos del cliente '{client_id}'. Omitiendo feature '{feature_desc_config}'.")
                    profile[feature_desc_config] = np.nan
                    continue

                serie_cliente_col = data_cliente[col_name_base].dropna() # Trabajar con la serie sin NaNs para esta columna
                if serie_cliente_col.empty:
                    profile[feature_desc_config] = np.nan
                    logger.debug(f"Serie vacía para '{col_name_base}' en cliente '{client_id}' después de dropna. Feature '{feature_desc_config}' será NaN.")
                    continue

                # Aplicar la función de agregación
                if agg_func_name == 'mean':
                    profile[feature_desc_config] = serie_cliente_col.mean()
                elif agg_func_name == 'std':
                    profile[feature_desc_config] = serie_cliente_col.std(ddof=0) # ddof=0 para consistencia si hay pocas muestras (N en denominador)
                elif agg_func_name == 'median':
                    profile[feature_desc_config] = serie_cliente_col.median()
                elif agg_func_name == 'sum':
                    profile[feature_desc_config] = serie_cliente_col.sum()
                elif agg_func_name == 'min':
                    profile[feature_desc_config] = serie_cliente_col.min()
                elif agg_func_name == 'max':
                    profile[feature_desc_config] = serie_cliente_col.max()
                elif agg_func_name == 'range': # Rango (max - min)
                    profile[feature_desc_config] = serie_cliente_col.max() - serie_cliente_col.min()
                elif agg_func_name == 'cv': # Coeficiente de variación (std / mean)
                    mean_val = serie_cliente_col.mean()
                    std_val = serie_cliente_col.std(ddof=0)
                    # Evitar división por cero o NaN/NaN
                    if mean_val != 0 and not np.isnan(mean_val) and not np.isnan(std_val):
                        profile[feature_desc_config] = std_val / mean_val
                    else:
                        profile[feature_desc_config] = 0 # O np.nan, dependiendo de cómo se quiera tratar
                # Ejemplo de feature más compleja (si se define en config)
                elif feature_desc_config == 'consumo_finde_ratio' and 'Volumen' in data_cliente.columns and 'es_findesemana' in data_cliente.columns:
                    consumo_total_cliente = data_cliente['Volumen'].sum()
                    consumo_finde_cliente = data_cliente[data_cliente['es_findesemana'] == 1]['Volumen'].sum()
                    if consumo_total_cliente > 0 and not np.isnan(consumo_total_cliente) and not np.isnan(consumo_finde_cliente):
                        profile[feature_desc_config] = consumo_finde_cliente / consumo_total_cliente
                    else:
                        profile[feature_desc_config] = 0 # O np.nan
                else:
                    logger.warning(f"Función de agregación '{agg_func_name}' (derivada de '{feature_desc_config}') no reconocida o la columna base '{col_name_base}' no tiene un manejo específico. Feature omitida.")
                    profile[feature_desc_config] = np.nan
            except Exception as e:
                logger.error(f"Error calculando feature de clustering '{feature_desc_config}' para cliente '{client_id}': {e}", exc_info=True)
                profile[feature_desc_config] = np.nan
        client_profiles_list.append(profile)
    
    client_features_df = pd.DataFrame(client_profiles_list)
    if client_features_df.empty:
        logger.error("No se pudieron generar perfiles de cliente para clustering (DataFrame vacío).")
        return pd.DataFrame()
        
    client_features_df = client_features_df.set_index('cliente_id')
    
    # Imputar NaNs que puedan surgir (e.g., std de un solo punto, división por cero en CV).
    # Usar la media de la columna de features para imputar es una opción simple.
    # Si una columna entera es NaN (porque todos los clientes tuvieron problemas calculándola), la media será NaN.
    # En ese caso, se imputa con 0 para evitar problemas en K-Means.
    logger.info("Revisando NaNs en features de cliente antes de la imputación final:")
    for col in client_features_df.columns:
        nan_count = client_features_df[col].isnull().sum()
        if nan_count > 0:
            logger.warning(f"Feature de clustering '{col}' tiene {nan_count} NaNs de {len(client_features_df)} clientes antes de imputación final.")

    for col in client_features_df.columns:
        if client_features_df[col].isnull().any():
            col_mean = client_features_df[col].mean()
            if np.isnan(col_mean): # Si la media es NaN (todos los valores de la columna eran NaN)
                logger.warning(f"La media de la feature de clustering '{col}' es NaN (probablemente todos los valores eran NaN). Imputando NaNs restantes en esta feature con 0.")
                client_features_df[col] = client_features_df[col].fillna(0)
            else:
                client_features_df[col] = client_features_df[col].fillna(col_mean)
                logger.debug(f"NaNs en feature de clustering '{col}' imputados con la media de la columna ({col_mean:.2f}).")
            
    logger.info(f"Perfiles de cliente para clustering generados. Shape: {client_features_df.shape}. NaNs imputados.")
    
    # Verificación final de NaNs
    if client_features_df.isnull().sum().sum() > 0:
        logger.error("¡ALERTA CRÍTICA! Todavía hay NaNs en las features de cliente después de la imputación. Esto causará errores en KMeans.")
        logger.error(f"Conteos de NaN por columna después de imputación:\n{client_features_df.isnull().sum()[client_features_df.isnull().sum() > 0]}")
    
    return client_features_df

def encontrar_k_optimo(data_scaled: pd.DataFrame, metodo: str, k_range: list, output_dir: str) -> int:
    """
    Determina el número óptimo de clusters (k) para K-Means utilizando el método
    especificado (Silhouette Score o Elbow Method).
    Guarda un gráfico de evaluación del método utilizado.

    Args:
        data_scaled (pd.DataFrame): DataFrame con los datos escalados sobre los cuales
                                    se determinará k.
        metodo (str): Método a utilizar ('silhouette' o 'elbow').
        k_range (list): Lista o rango de valores de k a probar (ej. range(2, 10)).
        output_dir (str): Directorio donde se guardará el gráfico de evaluación.

    Returns:
        int: El número óptimo de clusters (k) determinado.
             Retorna un valor por defecto (ej. 3) si el método falla.
    """
    logger.info(f"Determinando k óptimo para K-Means usando el método '{metodo}' en el rango {k_range}.")
    
    # Verificar si data_scaled contiene NaNs ANTES de KMeans
    if data_scaled.isnull().any().any(): # .any().any() para verificar en todo el DataFrame
        logger.error("Error Crítico: `data_scaled` contiene NaNs ANTES de pasarlo a KMeans en `encontrar_k_optimo`. Abortando búsqueda de k.")
        logger.error(f"Columnas con NaNs: {data_scaled.columns[data_scaled.isnull().any()].tolist()}")
        # Fallback a un k fijo o un valor por defecto si hay NaNs.
        return CONFIG.get('clustering', {}).get('n_clusters_fijo', 3) 

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{metodo}_plot_evaluacion_k.png")

    scores = {}
    optimal_k_determined = 3 # Default k en caso de fallo

    if metodo == 'silhouette':
        # Silhouette score requiere al menos 2 clusters y menos clusters que muestras.
        valid_k_for_silhouette = [k for k in k_range if 2 <= k < len(data_scaled)]
        if not valid_k_for_silhouette:
            logger.warning(f"Rango k {k_range} no es válido para Silhouette Score con {len(data_scaled)} muestras. Se usará k=2 si es posible, o default.")
            optimal_k_determined = min(2, len(data_scaled)-1) if len(data_scaled) > 2 else 2 # Ajuste simple
            if optimal_k_determined < 2: optimal_k_determined = CONFIG.get('clustering', {}).get('n_clusters_fijo', 2) # Fallback a config o 2
            return max(2, optimal_k_determined) # Asegurar al menos 2

        for k_clusters in valid_k_for_silhouette:
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto', algorithm='lloyd')
            try:
                cluster_labels_temp = kmeans.fit_predict(data_scaled)
                # Asegurar que haya más de 1 cluster único para silhouette_score
                if len(np.unique(cluster_labels_temp)) > 1:
                    silhouette_avg = silhouette_score(data_scaled, cluster_labels_temp)
                    scores[k_clusters] = silhouette_avg
                    logger.info(f"Para n_clusters = {k_clusters}, Silhouette Score: {silhouette_avg:.4f}")
                else:
                    logger.warning(f"KMeans con k={k_clusters} resultó en un solo cluster. No se puede calcular Silhouette Score.")
                    scores[k_clusters] = -1 # Penalizar este k
            except ValueError as e:
                logger.error(f"Error al ajustar KMeans o calcular Silhouette Score para k={k_clusters}: {e}. Puede ser por NaNs persistentes o datos insuficientes.")
                continue 
        
        if scores: # Si se calcularon algunos scores
            optimal_k_determined = max(scores, key=scores.get)
        else: # Si no se pudo calcular ningún score (ej. todos los k fallaron)
            logger.error("No se pudieron calcular Silhouette Scores para ningún k en el rango. Usando k por defecto o configurado.")
            optimal_k_determined = CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
        
        if scores: # Solo graficar si hay scores válidos
            plt.figure(figsize=(10, 6))
            plt.plot(list(scores.keys()), list(scores.values()), marker='o', color=CONFIG.get('dashboard',{}).get('CONTUGAS_COLORS',{}).get('primary_blue','#3d77dc'))
            plt.title('Evaluación de k mediante Silhouette Score')
            plt.xlabel('Número de Clusters (k)')
            plt.ylabel('Silhouette Score Promedio')
            plt.xticks(list(scores.keys())) 
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Gráfico de Silhouette Score guardado en: {plot_path}")

    elif metodo == 'elbow':
        wcss = [] # Within-cluster sum of squares (Inertia)
        valid_k_for_elbow = [k for k in k_range if 1 <= k <= len(data_scaled)] # K-Means n_clusters debe ser <= n_samples
        if not valid_k_for_elbow:
            logger.warning(f"Rango k {k_range} no es válido para Elbow Method con {len(data_scaled)} muestras. Usando k por defecto.")
            return CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)

        for k_clusters in valid_k_for_elbow:
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto', algorithm='lloyd')
            try:
                kmeans.fit(data_scaled)
                wcss.append(kmeans.inertia_) 
                logger.info(f"Para n_clusters = {k_clusters}, WCSS (Inertia): {kmeans.inertia_:.2f}")
            except ValueError as e:
                logger.error(f"Error al ajustar KMeans para Elbow Method con k={k_clusters}: {e}. Puede ser por NaNs.")
                # No añadir a wcss si falla, y quitar de valid_k_for_elbow si estaba
                if k_clusters in valid_k_for_elbow: valid_k_for_elbow.remove(k_clusters)
                continue

        if not wcss or not valid_k_for_elbow:
            logger.error("No se pudieron calcular WCSS para el método del codo. Usando k por defecto.")
            optimal_k_determined = CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(valid_k_for_elbow, wcss, marker='o', color=CONFIG.get('dashboard',{}).get('CONTUGAS_COLORS',{}).get('primary_green','#05a542'))
            plt.title('Evaluación de k mediante Método del Codo (Elbow Method)')
            plt.xlabel('Número de Clusters (k)')
            plt.ylabel('WCSS (Inertia)')
            plt.xticks(valid_k_for_elbow)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Gráfico del Método del Codo guardado en: {plot_path}")
            
            # Intentar encontrar el codo automáticamente usando kneed si está disponible
            try:
                from kneed import KneeLocator
                if len(valid_k_for_elbow) >=2 and len(wcss) >=2 : # kneed necesita al menos 2 puntos
                    # S=1 para sensibilidad, puede ajustarse. curve='convex', direction='decreasing' es típico para WCSS.
                    kl = KneeLocator(valid_k_for_elbow, wcss, curve='convex', direction='decreasing', S=1.0)
                    if kl.elbow is not None:
                        optimal_k_determined = kl.elbow
                        logger.info(f"KneeLocator encontró codo en k={optimal_k_determined}.")
                    else: # Fallback si kneed no encuentra un codo claro
                        logger.warning("KneeLocator no pudo encontrar un codo. Se usará un k intermedio o el configurado.")
                        optimal_k_determined = valid_k_for_elbow[len(valid_k_for_elbow)//2] if valid_k_for_elbow else CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
                else:
                    logger.warning("No hay suficientes puntos (k o WCSS) para usar KneeLocator. Se usará k intermedio o configurado.")
                    optimal_k_determined = valid_k_for_elbow[len(valid_k_for_elbow)//2] if valid_k_for_elbow else CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
            except ImportError:
                logger.warning("Librería 'kneed' no instalada. La determinación automática del codo puede ser menos precisa. Se usará k intermedio o configurado.")
                optimal_k_determined = valid_k_for_elbow[len(valid_k_for_elbow)//2] if valid_k_for_elbow else CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
            except Exception as e_kneed:
                logger.error(f"Error usando KneeLocator: {e_kneed}. Se usará k intermedio o configurado.")
                optimal_k_determined = valid_k_for_elbow[len(valid_k_for_elbow)//2] if valid_k_for_elbow else CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
    else:
        logger.error(f"Método de determinación de k óptimo '{metodo}' no soportado. Usando k por defecto o configurado.")
        optimal_k_determined = CONFIG.get('clustering', {}).get('n_clusters_fijo', 3)
        
    logger.info(f"k óptimo determinado/seleccionado por método '{metodo}': {optimal_k_determined}")
    return int(optimal_k_determined) # Asegurar que es un entero

def aplicar_kmeans_y_asignar_clusters(df_features_cliente_scaled: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Aplica el algoritmo K-Means a las características de cliente (escaladas) para
    asignar cada cliente a un cluster.

    Args:
        df_features_cliente_scaled (pd.DataFrame): DataFrame con las características de cliente
                                                   escaladas. El índice debe ser 'cliente_id'.
        n_clusters (int): Número de clusters a formar.

    Returns:
        pd.DataFrame: DataFrame con dos columnas: 'cliente_id' y 'cluster_id' (la etiqueta
                      del cluster asignado). Retorna DataFrame vacío si K-Means falla.
    """
    if df_features_cliente_scaled.empty:
        logger.error("DataFrame de características de cliente escalado vacío. No se puede aplicar K-Means.")
        return pd.DataFrame()
    
    # Doble verificación de NaNs (aunque deberían haber sido tratados antes)
    if df_features_cliente_scaled.isnull().sum().sum() > 0:
        logger.error("¡ERROR CRÍTICO! NaNs detectados en `df_features_cliente_scaled` justo antes de KMeans en `aplicar_kmeans_y_asignar_clusters`.")
        logger.error(f"Columnas con NaNs: {df_features_cliente_scaled.columns[df_features_cliente_scaled.isnull().any()].tolist()}")
        # Imputar con 0 como último recurso para evitar fallo de KMeans, aunque esto es un parche.
        df_features_cliente_imputed_final = df_features_cliente_scaled.fillna(0)
        logger.warning("NaNs fueron imputados con 0 como último recurso antes de KMeans.")
        data_for_kmeans_values = df_features_cliente_imputed_final.values
    else:
        data_for_kmeans_values = df_features_cliente_scaled.values # K-Means espera un array numpy
    
    if n_clusters <= 0:
        logger.error(f"Número de clusters inválido: {n_clusters}. No se puede aplicar K-Means. Asignando todos al cluster 0 por defecto.")
        # Crear un DataFrame con todos los clientes en un cluster 0
        return pd.DataFrame({'cliente_id': df_features_cliente_scaled.index, 'cluster_id': 0})

    if len(data_for_kmeans_values) < n_clusters:
        logger.warning(f"El número de muestras ({len(data_for_kmeans_values)}) es menor que n_clusters ({n_clusters}). Ajustando n_clusters a {len(data_for_kmeans_values)}.")
        n_clusters = len(data_for_kmeans_values)
        if n_clusters == 0: # Si no hay datos después de todo
            logger.error("No hay datos para K-Means después de ajustes. Retornando DataFrame vacío.")
            return pd.DataFrame()

    logger.info(f"Aplicando K-Means con n_clusters = {n_clusters}")
    
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', algorithm='lloyd')
    try:
        cluster_labels_assigned = kmeans_model.fit_predict(data_for_kmeans_values)
    except ValueError as e:
        logger.error(f"Error durante K-Means fit_predict: {e}. Esto usualmente indica NaNs o Infs persistentes en los datos.")
        logger.error(f"Datos que podrían haber causado el error (primeras 5 filas con problemas si es posible):\n{pd.DataFrame(data_for_kmeans_values)[np.any(np.isnan(data_for_kmeans_values) | np.isinf(data_for_kmeans_values), axis=1)].head()}")
        # Fallback: asignar todos a un solo cluster si K-Means falla catastróficamente
        cluster_labels_assigned = np.zeros(len(data_for_kmeans_values), dtype=int)
        logger.warning("K-Means falló. Todos los clientes asignados al cluster 0 como fallback.")

    df_cluster_asignaciones = pd.DataFrame({'cliente_id': df_features_cliente_scaled.index, 'cluster_id': cluster_labels_assigned})
    logger.info(f"Asignaciones de cluster generadas. Distribución de clientes por cluster:\n{df_cluster_asignaciones['cluster_id'].value_counts().sort_index()}")
    
    # Calcular métricas de calidad del clustering si hay más de un cluster y suficientes muestras
    try:
        num_unique_labels = len(np.unique(cluster_labels_assigned))
        if num_unique_labels > 1 and num_unique_labels < len(data_for_kmeans_values):
            sil_score = silhouette_score(data_for_kmeans_values, cluster_labels_assigned)
            # cal_har_score = calinski_harabasz_score(data_for_kmeans_values, cluster_labels_assigned) # Puede ser sensible a la forma del cluster
            dav_bou_score = davies_bouldin_score(data_for_kmeans_values, cluster_labels_assigned)
            logger.info(f"Métricas de calidad para {n_clusters} clusters: Silhouette={sil_score:.3f}, Davies-Bouldin={dav_bou_score:.3f}")
    except Exception as e_metrics:
        logger.warning(f"No se pudieron calcular métricas de calidad del clustering después de la asignación: {e_metrics}")

    return df_cluster_asignaciones

def visualizar_clusters(df_features_cliente_scaled: pd.DataFrame, cluster_labels: np.ndarray, output_dir: str):
    """
    Genera visualizaciones de los clusters formados.
    Actualmente, utiliza PCA para reducir a 2D y graficar, o un pairplot si hay pocas features.

    Args:
        df_features_cliente_scaled (pd.DataFrame): DataFrame con las características de cliente escaladas.
        cluster_labels (np.ndarray): Array con las etiquetas de cluster asignadas a cada cliente.
        output_dir (str): Directorio donde se guardarán los gráficos.
    """
    if df_features_cliente_scaled.empty or df_features_cliente_scaled.shape[1] < 1:
        logger.warning("DataFrame de features escalado vacío o sin suficientes features para visualización de clusters. Omitiendo.")
        return
    if len(np.unique(cluster_labels)) < 1: 
        logger.warning("No hay clusters válidos (o solo uno) para visualizar. Omitiendo visualización.")
        return

    logger.info("Generando visualización de clusters...")
    os.makedirs(output_dir, exist_ok=True)
    
    num_unique_clusters_viz = len(np.unique(cluster_labels))
    palette_colors = sns.color_palette("viridis", n_colors=num_unique_clusters_viz) \
                     if num_unique_clusters_viz > 0 else ["#3d77dc"]


    # Visualización con PCA si hay 2 o más features
    if df_features_cliente_scaled.shape[1] >= 2:
        try:
            from sklearn.decomposition import PCA
            if df_features_cliente_scaled.isnull().sum().sum() > 0: # Doble chequeo de NaNs
                logger.error("NaNs encontrados en `df_features_cliente_scaled` antes de PCA. Visualización PCA omitida.")
            else:
                pca = PCA(n_components=2, random_state=42)
                principal_components = pca.fit_transform(df_features_cliente_scaled)
                df_pca_viz = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=df_features_cliente_scaled.index)
                df_pca_viz['cluster_id'] = cluster_labels

                plt.figure(figsize=(12, 8))
                sns.scatterplot(x='PC1', y='PC2', hue='cluster_id', 
                                palette=palette_colors, 
                                data=df_pca_viz, legend="full", alpha=0.7, s=70) # s para tamaño de puntos
                plt.title('Visualización de Clusters de Clientes (PCA 2D)', fontsize=16)
                plt.xlabel('Componente Principal 1', fontsize=12)
                plt.ylabel('Componente Principal 2', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plot_path_pca_viz = os.path.join(output_dir, "cluster_visualizacion_pca.png")
                plt.savefig(plot_path_pca_viz)
                plt.close()
                logger.info(f"Visualización de clusters (PCA) guardada en: {plot_path_pca_viz}")
        except Exception as e_pca:
            logger.error(f"Error generando visualización PCA de clusters: {e_pca}", exc_info=True)

    # Pairplot si hay pocas features (ej. <= 5) y más de una feature
    if 1 < df_features_cliente_scaled.shape[1] <= 5 and num_unique_clusters_viz > 0:
        try:
            df_plot_pair = df_features_cliente_scaled.copy()
            df_plot_pair['cluster_id'] = cluster_labels
            
            if df_plot_pair.isnull().sum().sum() > 0: # Chequeo de NaNs
                 logger.error("NaNs encontrados en datos para pairplot. Pairplot omitido.")
            else:
                plt.figure() # Crear nueva figura para sns.pairplot
                pair_plot_fig = sns.pairplot(df_plot_pair, hue='cluster_id', 
                                         palette=palette_colors, 
                                         diag_kind='kde', # Kernel Density Estimate en la diagonal
                                         plot_kws={'alpha': 0.6, 's': 50}, # Estilo para scatter plots
                                         diag_kws={'fill': True, 'alpha': 0.5}) # Estilo para KDE plots
                pair_plot_fig.fig.suptitle('Pairplot de Features por Cluster', y=1.02, fontsize=16) # Título ajustado
                plot_path_pairplot_viz = os.path.join(output_dir, "cluster_visualizacion_pairplot.png")
                pair_plot_fig.savefig(plot_path_pairplot_viz)
                plt.close() # Cerrar la figura de pairplot
                logger.info(f"Visualización de clusters (Pairplot) guardada en: {plot_path_pairplot_viz}")
        except Exception as e_pair:
            logger.error(f"Error generando pairplot de clusters: {e_pair}", exc_info=True)


# --- Pipeline Principal de Clustering ---
def pipeline_cliente_clustering(config_global: dict, df_preprocesado: pd.DataFrame) -> pd.DataFrame:
    """
    Orquesta el pipeline completo de clustering de clientes:
    1. Carga configuración.
    2. Prepara features agregadas por cliente.
    3. Escala las features.
    4. Determina el número óptimo de clusters (k) o usa un k fijo.
    5. Aplica K-Means y asigna clusters.
    6. Genera visualizaciones de los clusters.
    7. Guarda las asignaciones de cluster.

    Args:
        config_global (dict): Diccionario de configuración global del proyecto.
        df_preprocesado (pd.DataFrame): DataFrame con los datos preprocesados para todos los clientes.

    Returns:
        pd.DataFrame: DataFrame con las asignaciones de 'cliente_id' a 'cluster_id'.
                      Si el clustering no está activo o falla, puede asignar todos al cluster 0.
    """
    cargar_configuracion_clustering(config_global) # Carga CONFIG para este módulo
    
    clustering_cfg = CONFIG.get('clustering', {})
    rutas_cfg = CONFIG.get('rutas', {})
    
    # Crear directorio para diagnósticos de clustering si no existe
    # Base path para logs, y luego subdirectorio 'clustering_diagnostics'
    log_file_path_base = rutas_cfg.get('logs', 'logs/anomaly_detection.log')
    output_dir_clustering_diagnostics = os.path.join(os.path.dirname(log_file_path_base), "clustering_diagnostics")
    os.makedirs(output_dir_clustering_diagnostics, exist_ok=True)

    if not clustering_cfg.get('activo', False):
        logger.info("Clustering no está activo en la configuración. Se asignarán todos los clientes al cluster 0 por defecto.")
        # Crear un DataFrame con todos los clientes únicos y asignarles cluster_id = 0
        if 'cliente_id' in df_preprocesado.columns:
            df_asignaciones_default = pd.DataFrame({'cliente_id': df_preprocesado['cliente_id'].unique(), 'cluster_id': 0})
        else:
            logger.error("Columna 'cliente_id' no encontrada en df_preprocesado. No se pueden generar asignaciones de cluster por defecto.")
            df_asignaciones_default = pd.DataFrame()
        return df_asignaciones_default

    # 1. Preparar features de cliente
    df_features_cliente = preparar_features_cliente_para_clustering(df_preprocesado)
    if df_features_cliente.empty or df_features_cliente.isnull().sum().sum() > 0: 
        logger.error("Clustering Abortado: No se pudieron generar features de cliente válidas (sin NaNs). Asignando todos al cluster 0.")
        # Fallback similar al de clustering inactivo
        if 'cliente_id' in df_preprocesado.columns:
            return pd.DataFrame({'cliente_id': df_preprocesado['cliente_id'].unique(), 'cluster_id': 0})
        else:
            return pd.DataFrame()

    # 2. Escalar features
    scaler = StandardScaler()
    try:
        # K-Means espera un array NumPy para fit_transform
        data_scaled_np_array = scaler.fit_transform(df_features_cliente.values) 
    except ValueError as e_scale: # Común si hay NaNs/Infs que no se trataron
        logger.error(f"Error durante el escalado de features de cliente (posiblemente por NaNs/Infs no tratados): {e_scale}")
        logger.error(f"Features que podrían haber causado el error (si hay NaNs/Infs):\n{df_features_cliente[df_features_cliente.isnull().any(axis=1) | np.isinf(df_features_cliente.values).any(axis=1)]}")
        if 'cliente_id' in df_preprocesado.columns: # Fallback
            return pd.DataFrame({'cliente_id': df_preprocesado['cliente_id'].unique(), 'cluster_id': 0})
        else: return pd.DataFrame()

    df_features_cliente_scaled = pd.DataFrame(data_scaled_np_array, columns=df_features_cliente.columns, index=df_features_cliente.index)

    # 3. Determinar k (número de clusters)
    n_clusters_final = clustering_cfg.get('n_clusters_fijo') # Intentar obtener k fijo desde config
    if n_clusters_final is None: # Si no hay k fijo, determinarlo dinámicamente
        metodo_k_optimo = clustering_cfg.get('metodo_optimal_k', 'silhouette') # 'silhouette' o 'elbow'
        rango_k_config = clustering_cfg.get('rango_k', [2, 8]) # Rango de k a probar
        
        # Asegurar que el rango_k sea válido para el número de muestras
        max_k_possible_silhouette = len(df_features_cliente_scaled) -1 if len(df_features_cliente_scaled) > 1 else 1
        max_k_possible_elbow = len(df_features_cliente_scaled)

        rango_k_valido_metodo = [k for k in rango_k_config 
                                 if isinstance(k,int) and k >= (2 if metodo_k_optimo == 'silhouette' else 1) 
                                 and k <= (max_k_possible_silhouette if metodo_k_optimo == 'silhouette' else max_k_possible_elbow)]
        
        if not rango_k_valido_metodo:
            logger.warning(f"Rango k {rango_k_config} es inválido o insuficiente para {len(df_features_cliente_scaled)} clientes y método '{metodo_k_optimo}'. Se usará k=2 (o n_clientes si es menor) o el k fijo si está configurado.")
            # Fallback a un k pequeño si el rango no es válido
            n_clusters_final = min(2, len(df_features_cliente_scaled)) if len(df_features_cliente_scaled) > 0 else 1
            if n_clusters_final == 0 : n_clusters_final = 1 # Evitar k=0
        else:
             n_clusters_final = encontrar_k_optimo(df_features_cliente_scaled, metodo_k_optimo, rango_k_valido_metodo, output_dir_clustering_diagnostics)
    else: # Si se usa k fijo
        n_clusters_final = int(n_clusters_final)
        logger.info(f"Usando número de clusters fijo de la configuración: {n_clusters_final}")
    
    # Asegurar que n_clusters_final sea al menos 1 y no mayor que el número de clientes
    if n_clusters_final <= 0: n_clusters_final = 1
    if len(df_features_cliente_scaled) == 0 and n_clusters_final > 0 :
        logger.error("No hay datos de clientes para aplicar K-Means, aunque se determinó un n_clusters > 0. Retornando DataFrame vacío.")
        return pd.DataFrame()
    if len(df_features_cliente_scaled) > 0 and n_clusters_final > len(df_features_cliente_scaled):
        logger.warning(f"n_clusters_final ({n_clusters_final}) es mayor que el número de clientes ({len(df_features_cliente_scaled)}). Se ajustará a {len(df_features_cliente_scaled)}.")
        n_clusters_final = len(df_features_cliente_scaled)

    # 4. Aplicar K-Means y asignar clusters
    df_asignaciones_cluster = aplicar_kmeans_y_asignar_clusters(df_features_cliente_scaled, n_clusters_final)
    
    # 5. Visualizar (si se generaron asignaciones y hay más de un cluster)
    if not df_asignaciones_cluster.empty and df_asignaciones_cluster['cluster_id'].nunique() > 0 :
        visualizar_clusters(df_features_cliente_scaled, df_asignaciones_cluster['cluster_id'].values, output_dir_clustering_diagnostics)
    
    # 6. Guardar asignaciones
    ruta_archivo_asignaciones = rutas_cfg.get('cluster_assignments', 'data/client_clusters.csv')
    try:
        os.makedirs(os.path.dirname(ruta_archivo_asignaciones), exist_ok=True)
        df_asignaciones_cluster.to_csv(ruta_archivo_asignaciones, index=False) # Guardar sin el índice del DataFrame
        logger.info(f"Asignaciones de cluster guardadas en: {ruta_archivo_asignaciones}")
    except Exception as e_save:
        logger.error(f"Error al guardar las asignaciones de cluster en '{ruta_archivo_asignaciones}': {e_save}", exc_info=True)
        
    return df_asignaciones_cluster
# --- Fin del módulo de clustering ---
# --- Ejecución del pipeline de clustering --- 