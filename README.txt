Que contiene la implementacion: 

- En Apache Airflow, un DAG para obtener datos del repositorio Inside Airbnb Repository. El DAG contiene varias tareas para preprocesar los datos y hacer el split en train y test. Los datasets son guardados en un bucket S3.

- El seguimiento del proceso se hace con MLflow.

Requerimientos:

- OS: Linux, macOS
- Paquetes: Docker compose
- ...

Instalacion:

- Clonar este repositorio
- En el archivo .env setear la variable de entorno AIRFLOW_UID con la uuid del usuario actual (id -u <username>)
- Levantar los servicios: docker compose --profile all up

Para acceder a las apps de Airflow y MLflow, se comunican con los puertos 8080 y 5001 respectivamente.

localhost:8080 y localhost:5001

Testeo:

Una vez que los servicios estan levantados, se pueden correr las tareas del DAG process_etl_airbnb_data.

Las 4 tareas (get_data, preprocess_data, split_dataset y encode_normalize_features) deberian terminar con state=success 
