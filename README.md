# Optimización de la Ocupación y Estrategias de Precios en Airbnb
## Autores: Pablo Ezequiel Brahim - José Roberto Castro -  Abril Noguera - Kevin Nelson Pennington 

## Problema de Negocio
Los anfitriones de Airbnb en Buenos Aires buscan continuamente optimizar la ocupación de sus propiedades para maximizar ingresos y mejorar la satisfacción del cliente. El desafío consiste en adaptar las estrategias de precios y mejorar las características de las propiedades de manera que respondan a las dinámicas del mercado y las preferencias cambiantes de los huéspedes en diferentes culturas y condiciones económicas.

## Objetivo
Desarrollar un modelo predictivo de machine learning que permita a los anfitriones de Airbnb en estas ciudades predecir si tendran alta o baja opcipación en sus propiedades basándose en datos históricos y tendencias de mercado. El objetivo final es utilizar estos insights para recomendar estrategias de precios que maximicen la ocupación y, consecuentemente, los ingresos.

Este proyecto implementa un flujo de trabajo completo de Machine Learning enfocado en la predicción de la ocupación de las propiedades publicadas en Airbnb. La solución automatiza desde el scrapping de datos hasta el despliegue del mejor modelo en producción utilizando Apache Airflow como orquestador y MLflow para el tracking y versionado de modelos.

### Pipeline del Proyecto

1. **Scrapear** datos de autos usados desde una plataforma online.
2. **Limpiar** los datos descargados de forma incremental.
3. **Preprocesar** concatenar descargas anteriores, procesar variables categóricas, valores faltantes y construir los datasets `X_train`, `y_train`.
4. **Entrenar** tres modelos de regresión y promover el mejor a producción.
5. **Realizar predicciones** con el modelo en producción, desplegado en FastApi y Streamlit.

## Tecnologías utilizadas

| Herramienta         | Descripción                                           |
|---------------------|-------------------------------------------------------|
| **Apache Airflow**  | Orquestador de tareas para flujos de trabajo diarios |
| **MLflow**          | Seguimiento, versionado y registro de modelos ML     |
| **Docker Compose**  | Entorno reproducible con contenedores                |
| **Pandas & Scikit-learn** | Procesamiento de datos y entrenamiento de modelos |
| **CatBoost / SVR / Random Forest**  | Modelos adicionales para comparación de desempeño    |
| **FastApi**           | Desplegado de la API de prediccion en produccion    |
| **Streamlit**  | UI para uso del servicio de prediccion on demand    |


## ⚙️ Estructura del Proyecto
- ├── airflow/
- │ ├── dags/
- │ │ ├── etl_process.py
- │ │ ├── retrain_the_model.py
- │ │ └── make_predictions.py
- │ ├──── utils/
- │ │ └── utils_etl.py
- │ ├── plugins/
- │ ├── config/
- │ ├── logs/
- │ └── secrets/
- ├── dockerfiles/
- │ ├── airflow/ # Imagen extendida con dependencias
- │ ├── fastapi/
- │ ├── mlflow/
- │ └── postgres/
- ├── mlruns/ # Entrenamiento y Tuning
- ├── .env
- ├── docker-compose.yaml
- └── README.md

### Flujo completo de DAGs

1. ### `etl_process` (Ejecutado manual o programado)
   - **get_data** → Descarga los datos crudos de [Inside Airbnb](https://insideairbnb.com/get-the-data/). y los guarda en S3.  
   - **preprocess_data** → Limpia, imputa y transforma datos (precios, nulos, dummies, target).  
   - **split_dataset** → Separa en train/test (70/30 estratificado) y guarda en S3.  
   - **encode_normalize_features** → Escala numéricos, aplica one-hot encoding y registra en MLflow. 
   - Al finalizar, **dispara automáticamente** el DAG de re-entrenamiento. 

2. ### `retrain_the_model` (Ejecutado automáticamente tras scraping o de forma diaria)
   
   - **train_the_challenger_model** → Clona el modelo actual en producción (champion), lo re-entrena con los datos más recientes y lo registra como modelo candidato (challenger).  
   - **evaluate_champion_challenger** → Compara el modelo champion y el challenger usando métricas (F1, accuracy, precision, recall, AUC).  
      - Si el challenger supera al champion en F1, se promueve a producción.  
      - Si no, se descarta y se mantiene el modelo actual.  

   - **Resultado final:** siempre queda en producción el mejor modelo disponible según el desempeño en test.  
   - Al finalizar, **dispara automáticamente** el DAG de predicción.

3. ### `make_predictions` (Ejecutado automáticamente tras re-entrenamiento o de forma diaria)
   - Carga el modelo en stage `"Production"` desde **MLflow**.
   - Prepara nuevas muestras o clientes para predicción.
   - Genera predicciones y las guarda en el directorio `data/predicciones/`.

## Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/abrilnoguera/TP-MLOps1.git
```

### 2. Moverse al directorio
```bash
cd CEIA_TP_MLOps_1
```
### 3. Definir AIRFLOW_UID
Si estás en Linux o MacOS, en el archivo .env, reemplaza AIRFLOW_UID por el de tu usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando id -u <username>). De lo contrario, Airflow dejará sus carpetas internas como root y no podrás subir DAGs (en airflow/dags) o plugins, etc.

### 4. Levantar el entorno completo
```bash
docker-compose up --build -d
```

Para detenerlo:
```Bash
docker compose down
```

Para detenerlo y eliminar todo:
```Bash
docker compose down --rmi all --volumes
```

### 5. Acceder a interfaces
`Airflow UI`: http://localhost:8080
Para correr por primera vez, hay que activar el dag inicializador del workflow `etl_process` con eso se descargan los datos, preprocesan, entrenan y promueven a produccion el mejor modelo, una vez hecho esto, se pueden usar los servicios de FastApi y Streamlit.

`MLflow UI`: http://localhost:5000

`FastAPI`: http://localhost:8000 - A tomar en cuenta, ya que al inicializar el proyecto, no hay datos descargados -> no existe artefacto de prediccion. Entonces
                                 La API no retorna prediccion, pero si un log que retorna un status code 503 y una indicacion: `Modelo no disponible. Aún no ha sido entrenado o registrado en MLflow, por favor espera a completar el ciclo de descarga-entrenamiento.`

`Streamlit`: UI para prediccion: http://localhost:8501
