# Sistema de Detección de Anomalías en Consumo de Gas para Contugas

## Descripción General

Este proyecto implementa un sistema avanzado para la **detección proactiva y clasificación de anomalías** en el consumo horario de gas natural (considerando presión, temperatura y volumen) de clientes industriales de Contugas. El objetivo principal es identificar patrones de consumo inusuales que puedan indicar fugas, fallas en equipos de medición, usos no contractuales u otras ineficiencias operativas, permitiendo una intervención oportuna y la mitigación de riesgos y pérdidas.

La solución se basa en un pipeline de ciencia de datos escrito en **Python**, utilizando librerías como Pandas, NumPy, Scikit-learn y TensorFlow/Keras. El enfoque central es el uso de modelos **Autoencoder LSTM (Long Short-Term Memory)** entrenados específicamente para diferentes segmentos de clientes (clusters), identificados mediante **K-Means clustering**. Una característica distintiva es la ingeniería de una nueva variable, la **"cantidad de gas calculada"** basada en la ley de los gases, para enriquecer el análisis. Los resultados y alertas se presentan en un **dashboard interactivo** desarrollado con Dash.

## Características Principales

* **Pipeline ETL Robusto:** Carga de datos desde múltiples archivos Excel, limpieza exhaustiva, tratamiento inteligente de valores faltantes y outliers, y una rica ingeniería de características que incluye variables temporales cíclicas y la "cantidad de gas calculada".
* **Segmentación Inteligente de Clientes:** Aplicación de K-Means clustering (actualmente configurado para 2 clusters) para agrupar clientes con perfiles de consumo similares, permitiendo una detección más precisa y adaptada.
* **Modelado LSTM Avanzado por Cluster:** Entrenamiento de modelos Autoencoder LSTM específicos para cada segmento de clientes. El sistema está preparado para optimización de hiperparámetros (actualmente con un grid simplificado para agilizar el entrenamiento del prototipo). El entrenamiento de los modelos para diferentes clusters se realiza en paralelo para mejorar la eficiencia en máquinas multinúcleo.
* **Detección de Anomalías Basada en Reconstrucción:** Las anomalías se identifican comparando el consumo real con la reconstrucción generada por el modelo LSTM; errores de reconstrucción elevados indican un comportamiento anómalo.
* **Clasificación de Criticidad de Alertas:** Las anomalías detectadas se clasifican automáticamente en tres niveles de severidad (Alta, Media, Baja) basándose en la magnitud de la desviación del error de reconstrucción respecto a la norma del cluster.
* **Dashboard Interactivo y Multifacético:** Una aplicación Dash permite:
    * Visualizar series temporales de consumo (Presión, Temperatura, Volumen, Cantidad de Gas Calculada) con anomalías claramente marcadas y coloreadas por criticidad.
    * Filtrar datos por cliente, cluster, rango de fechas, variable y nivel de severidad.
    * Consultar KPIs resumen.
    * Explorar una tabla detallada de las anomalías detectadas.
    * Analizar correlaciones entre variables.
    * Visualizar patrones temporales agregados (ej. consumo promedio por hora del día).
    * Revisar métricas de desempeño del sistema y tiempos de ejecución (si los archivos correspondientes son generados).
* **Orquestación Centralizada y Configurable:** Un script principal (`main.py`) maneja todo el flujo del pipeline (ETL, clustering, entrenamiento, detección y evaluación opcional), controlado por un archivo de configuración centralizado (`config.yaml`).

## Estructura del Proyecto
contugas_anomaly_detection/
├── config.yaml                 # Archivo de configuración central del pipeline
├── data/
│   ├── anomalies_detected.csv  # Salida con anomalías detectadas y su criticidad
│   ├── client_clusters.csv     # Asignación de clientes a clusters
│   ├── preprocessed.csv        # Datos limpios y transformados listos para modelado
│   ├── raw/                    # Datos crudos de consumo (archivos .xlsx por cliente)
│   └── (opcional) ground_truth.csv # Datos etiquetados para evaluación del modelo
├── dashboard/
│   └── app.py                  # Aplicación Dash para visualización interactiva
├── data_etl.py                 # Scripts para extracción, transformación y carga de datos
├── clustering.py               # Scripts para la segmentación de clientes (K-Means)
├── models/
│   └── lstm_model.py           # Implementación del Autoencoder LSTM
├── alerts/
│   └── alert_system.py         # Lógica para clasificación de criticidad de alertas
├── evaluation/
│   ├── experiments.py          # Funciones para evaluar el rendimiento del sistema
│   └── (opcional) performance_metrics.csv # Métricas de desempeño si se usa ground_truth
├── logs/
│   ├── anomaly_detection.log   # Log general del pipeline
│   ├── (opcional) execution_times.yaml # Tiempos de ejecución de las etapas del pipeline
│   └── clustering_diagnostics/ # Gráficos de evaluación del clustering (Elbow, Silhouette)
├── main.py                     # Script principal para orquestar todo el pipeline
├── requirements.txt            # Dependencias de Python del proyecto
└── README.md                   # Este archivo

## Requisitos Previos

* Python 3.9 o superior.
* Se recomienda el uso de un entorno virtual (ej. `venv`, `conda`).
* Las dependencias de Python listadas en `requirements.txt`. Estas incluyen, pero no se limitan a:
    * `pandas`, `numpy`
    * `scikit-learn`
    * `tensorflow` (o `tensorflow-cpu`)
    * `pyyaml`
    * `dash`, `plotly`
    * `kneed` (opcional, para la determinación automática del "codo" en K-Means)

## Instalación

1.  **Clonar el Repositorio:**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd contugas_anomaly_detection
    ```
2.  **Crear y Activar Entorno Virtual** (Recomendado):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\Activate.ps1
    # macOS/Linux
    source venv/bin/activate
    ```
3.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuración

1.  **Archivo de Configuración:** El comportamiento del pipeline se controla a través del archivo `config.yaml`. Asegúrate de que este archivo esté presente en la raíz del proyecto y configúralo según tus necesidades:
    * **`rutas`**: Define las rutas a los directorios de datos, modelos guardados, logs, etc. Es crucial que `rutas.raw` apunte al directorio que contiene los archivos Excel de consumo de los clientes.
    * **`etl`**: Parámetros para el preprocesamiento de datos (nombres de columnas, métodos de imputación, tratamiento de outliers, normalización).
    * **`feature_engineering.gas_law`**: Configuración para el cálculo de la feature "cantidad de gas calculada" (activación, constante R, factor Z).
    * **`clustering`**: Parámetros para la segmentación de clientes (activación, features a usar para perfilar clientes, método para k óptimo, o `n_clusters_fijo` – actualmente 2).
    * **`modelos.lstm_autoencoder`**: Configuración del modelo LSTM (longitud de secuencia, grilla de hiperparámetros para optimización – actualmente simplificada, parámetros de paciencia para Early Stopping y reducción de Learning Rate).
    * **`alertas`**: Umbrales para la clasificación de criticidad de anomalías.
    * **`dashboard`**: Puerto y modo debug para la aplicación Dash.
    * **`logging`**: Configuración del sistema de logging (nivel, formato, archivo de salida).

2.  **Datos Crudos:** Coloca los archivos de datos de consumo de gas de los clientes (en formato `.xlsx`, con una hoja por cliente o según la lógica de `data_etl.py`) en el directorio especificado en `rutas.raw` dentro de `config.yaml` (por defecto `data/raw/`).

3.  **(Opcional) Datos de Ground Truth:** Si deseas realizar una evaluación cuantitativa del rendimiento del sistema de detección, crea un archivo `ground_truth.csv` con anomalías reales previamente identificadas y etiquetadas. Asegúrate de que la ruta a este archivo esté especificada en `rutas.ground_truth_data` en `config.yaml`.

## Uso del Sistema

El pipeline se ejecuta principalmente a través del script `main.py`.

1.  **Ejecutar el Pipeline Completo:**
    Este comando realiza todas las etapas: ETL, clustering de clientes, entrenamiento paralelo de modelos LSTM por cluster (con optimización de hiperparámetros), detección de anomalías, y generación de reportes. Si se proporciona un archivo de ground truth, también realizará la evaluación del sistema.
    ```powershell
    # Desde la raíz del proyecto, con el entorno virtual activado:
    python main.py --config config.yaml
    ```
    Los principales archivos de salida generados en el directorio `data/` serán:
    * `preprocessed.csv`: Datos limpios, transformados y con features de ingeniería.
    * `client_clusters.csv`: Asignación de cada cliente a un cluster.
    * `anomalies_detected.csv`: Lista de anomalías detectadas con su criticidad y detalles.
    Archivos adicionales se generan en `models/trained/`, `logs/`, y opcionalmente en `evaluation/`.

2.  **Visualizar Resultados en el Dashboard:**
    Una vez que el pipeline `main.py` ha finalizado, puedes lanzar el dashboard interactivo para explorar los resultados:
    ```powershell
    # Desde la raíz del proyecto, con el entorno virtual activado:
    python dashboard/app.py
    ```
    Luego, abre tu navegador web y dirígete a la dirección que se muestra en la terminal (usualmente `http://127.0.0.1:PUERTO/` o `http://localhost:PUERTO/`, donde `PUERTO` es el especificado en `config.yaml`, por defecto 8050).

## Detalle de Módulos Clave

* **`main.py`**: Script orquestador que ejecuta secuencialmente todas las etapas del pipeline de detección de anomalías. Maneja la configuración global y la ejecución paralela del entrenamiento de modelos.
* **`config.yaml`**: Archivo central que parametriza todo el comportamiento del sistema, desde rutas de datos hasta hiperparámetros de modelos y umbrales de alerta.
* **`data_etl.py`**: Contiene todas las funciones para la extracción de datos de archivos Excel, limpieza, tratamiento de valores faltantes y outliers, ingeniería de características (incluyendo variables temporales y la "cantidad de gas calculada" basada en la ley de los gases), y validación de calidad de datos.
* **`clustering.py`**: Implementa la lógica para la segmentación de clientes. Prepara features agregadas por cliente y aplica K-Means para agruparlos, determinando el número óptimo de clusters (o usando un valor fijo de la configuración).
* **`models/lstm_model.py`**: Define la arquitectura del autoencoder LSTM, las funciones para preparar secuencias de datos, entrenar el modelo (con Early Stopping y reducción de Learning Rate), calcular errores de reconstrucción, y detectar anomalías basadas en un umbral dinámico.
* **`alerts/alert_system.py`**: Contiene la lógica para clasificar la severidad (criticidad) de las anomalías detectadas basándose en la desviación de los scores/errores respecto a la norma, y para generar el reporte final de anomalías.
* **`evaluation/experiments.py`**: Provee funciones para calcular métricas de rendimiento del sistema de detección (como precisión, recall, F1-score, tasa de falsos positivos) cuando se dispone de un conjunto de datos de ground truth.
* **`dashboard/app.py`**: Aplicación web interactiva construida con Dash y Plotly para la visualización de las series temporales de consumo, las anomalías detectadas marcadas por criticidad, KPIs resumen, tablas de detalle, y análisis exploratorios adicionales (correlaciones, patrones temporales).

## Próximos Pasos y Mejoras Potenciales

* Validación exhaustiva con un conjunto de datos etiquetado (`ground_truth.csv`) para cuantificar con precisión las métricas de desempeño.
* Implementación de un cálculo dinámico del factor de compresibilidad (Z-factor) para la feature "cantidad de gas calculada", mejorando su precisión para gases reales.
* Desarrollo de una estrategia robusta para el reentrenamiento periódico de los modelos LSTM para adaptarse a la evolución de los patrones de consumo de los clientes.
* Integración con los sistemas de alerta y ticketing operacionales de Contugas.
* Expansión de las capacidades analíticas del dashboard.





