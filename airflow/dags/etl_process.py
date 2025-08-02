import datetime
import pandas as pd
import yaml
import os

from airflow.decorators import dag, task

def get_variables_from_yaml():
    """
    Helper function to read variables directly from YAML file
    This avoids issues with Airflow LocalFilesystemBackend and list parsing
    """
    yaml_path = "/opt/secrets/variables.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        # Fallback to local path for development
        yaml_path = "./airflow/secrets/variables.yaml"
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

markdown_text = """
### ETL Process for AirBnb Dataset

This DAG extracts information from the original CSV file stored in the Inside Airbnb Repository of the 
[AirBnb dataset](https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz). 
It preprocesses the data by creating dummy variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Abril Noguera - José Roberto Castro - Kevin Nelson Pennington - Pablo Ezequiel Brahim",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_airbnb_data",
    description="ETL process for Airbnb Buenos Aires, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Airbnb"],
    default_args=default_args,
    catchup=False,
)
def process_etl_airbnb_data():

    @task(
        task_id="get_data"
    )
    def get_data():
        """
        Load the raw data from Inside Airbnb repository
        """
        import awswrangler as wr
        from airflow.models import Variable
        from utils_etl import load_and_get_df

        # fetch dataset
        urls = {
            "Buenos Aires": "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz",
        }

        # Cargar y unir todos los dataframes
        dfs = [load_and_get_df(url, city) for city, url in urls.items()]
        dataframe = pd.concat(dfs, ignore_index=True)

        data_path = "s3://data/raw/airbnb.csv"

        wr.s3.to_csv(df=dataframe,
                     path=data_path,
                     index=False)


    @task(
        task_id="preprocess_data"
    )
    def preprocess_data():
        """
        Convert categorical variables into one-hot encoding.
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import pandas as pd
        import numpy as np

        from airflow.models import Variable

        from utils_etl import eliminar_columnas_con_muchos_nulos, eliminar_filas_con_nulos, remove_html_tags, obtener_tasas_cambio, convert_currency

        data_original_path = "s3://data/raw/airbnb.csv"
        data_end_path = "s3://data/raw/airbnb_preprocessed.csv"
        data = wr.s3.read_csv(data_original_path)
        df = data.copy()

        # Obtengo scrapping date
        date = data['last_scraped'][0]

        # Clean duplicates
        df.drop_duplicates(inplace=True, ignore_index=True)
        
        # Convertir host_response_rate y host_acceptance_rate de porcentaje a número
        df["host_response_rate"] = df["host_response_rate"].str.replace("%", "").astype(float) / 100
        df["host_acceptance_rate"] = df["host_acceptance_rate"].str.replace("%", "").astype(float) / 100

        # Convertir precio de string con $ y , a float
        df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)

        # Columnas categóricas (tipo object o string)
        categoric_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        print("Las columnas de tipo categóricas son: ", categoric_cols, '\n')

        # Columnas numéricas (int o float)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        print("Las columnas de tipo numéricas son: ", numeric_cols)

        # Reemplazar '.' o '-' por NaN si son los únicos caracteres en la celda
        for column_name in categoric_cols:
            df[column_name] = df[column_name].replace(r'^[\.\-]$', np.nan, regex=True)

        # Reemplazar "N/A" por NaN en todas las columnas
        df.replace("N/A", np.nan, inplace=True)

        # Eliminar columnas con más del 90% de nulos
        df = eliminar_columnas_con_muchos_nulos(df, umbral=0.9)

        # Se mantienen como vacios las descripciones, dado que puede ser un indicador importante en la performance del alquiler.
        df[["description", "neighborhood_overview", "host_about"]] = df[["description", "neighborhood_overview", "host_about"]].fillna("")

        # La columna neighbourhood no es muy representativa al tratarse de información de CABA, por lo que se mantiene unicamente neighbourhood_cleansed.
        df.drop(columns=["neighbourhood"], inplace=True)

        # Extraer número de bathrooms desde bathrooms_text (si falta bathrooms)
        df["extracted_bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+)").astype(float)
        df["bathrooms"] = df["bathrooms"].fillna(df["extracted_bathrooms"])
        df.drop(columns=["extracted_bathrooms"], inplace=True)

        # Imputar 'first_review' y 'last_review' con 'host_since' si están vacíos
        df["first_review"] = df["first_review"].fillna(df["host_since"])
        df["last_review"] = df["last_review"].fillna(df["host_since"])

        # Imputar columnas con mediana
        cols_mediana = ['bathrooms', 'bedrooms', 'beds']
        for col_name in cols_mediana:
            mediana = df[col_name].median()
            df[col_name] = df[col_name].fillna(mediana)

        # Imputar columnas con media
        cols_media = [
            'host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 
            'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
            'review_scores_communication', 'review_scores_location', 'review_scores_value', 
            'reviews_per_month'
        ]

        for col_name in cols_media:
            media = df[col_name].mean()
            df[col_name] = df[col_name].fillna(media)

        # Imputar columnas con moda
        cols_moda = ["host_location", "host_neighbourhood", "host_response_time", "host_is_superhost", "bathrooms_text"]
        for col_name in cols_moda:
            moda = df[col_name].mode().iloc[0]
            df[col_name] = df[col_name].fillna(moda)

        # Definir columna has_availability si todas las availability son 0 y está vacía
        cond = (
            (df['availability_30'] == 0) &
            (df['availability_60'] == 0) &
            (df['availability_90'] == 0) &
            (df['availability_365'] == 0) &
            (df['has_availability'].isna())
        )
        df.loc[cond, 'has_availability'] = 'f'
        df["has_availability"] = df["has_availability"].fillna("t")

        # Eliminar filas con nulos en columnas críticas
        df = eliminar_filas_con_nulos(df, ["price", "host_name", "picture_url"])

        # Eliminación de etiquetas de HTML
        df = remove_html_tags(df)

        # Limpiar y convertir la columna en listas
        df["host_verifications"] = (
            df["host_verifications"]
            .astype(str)
            .str.replace(r"[\[\]'\"]", "", regex=True)  # eliminar comillas y corchetes
            .str.strip()
            .str.split(",")                             # convertir a lista
            .apply(lambda lst: [item.strip() for item in lst if item])
        )

        # Obtener todos los tipos únicos
        unique_types = set()
        df["host_verifications"].dropna().apply(unique_types.update)
        unique_types = sorted(filter(None, unique_types))  # eliminar strings vacíos
        
        # Crear columnas dummy (una por cada tipo de verificación)
        for item in unique_types:
            col_name = "verif_" + item.replace(" ", "_")
            df[col_name] = df["host_verifications"].apply(lambda x: int(item in x) if isinstance(x, list) else 0)

        # # Eliminar la columna original
        df.drop(columns=["host_verifications"], inplace=True)

        # Eliminar caracteres no alfabéticos ni comas/espacios
        df["amenities"] = df["amenities"].astype(str).str.replace(r"[^a-zA-Z ,]", "", regex=True)

        # Reemplazar múltiples espacios por uno solo
        df["amenities"] = df["amenities"].str.replace(r"\s+", " ", regex=True).str.strip()

        # Separar por coma + espacio → lista limpia de strings
        df["amenities"] = df["amenities"].str.split(", ").apply(lambda lst: [item.strip() for item in lst if item])

        # Tasas de cambio aproximadas en la fecha de scrapping
        tasas_cambio = obtener_tasas_cambio(date)

        #  Aplicar la conversión a toda la columna
        df["price"] = df.apply(lambda row: convert_currency(tasas_cambio, row["price"], row["city"]), axis=1)

        # Crear nueva columna de precio logarítmico
        df["log_price"] = np.log1p(df["price"])

        # Conversión de 't'/'f' a 1/0
        cols_binarias = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]

        for col in cols_binarias:
            df[col] = df[col].map({'t': 1, 'f': 0})

        # Calcular la tasa de ocupación en los próximos 60 días
        df["occupation_rate"] = 1 - (df["availability_60"] / 60)

        # Definir una columna binaria de ocupación (target)
        df["high_occupancy"] = (df["occupation_rate"] >= 0.8).astype(int)

        # Eliminar columna que define al target
        df.drop(columns=["occupation_rate"], inplace=True)
        
        # Eliminar la columna amenities ya que contiene listas y no está en feature_cols
        df.drop(columns=["amenities"], inplace=True, errors='ignore')

        wr.s3.to_csv(df=df,
                     path=data_end_path,
                     index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        target_col = Variable.get("target_col")
        dataset_log = df.drop(columns=target_col)

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = df.select_dtypes(include=["object", "string"]).columns.tolist()
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}

        category_dummies_dict = {}
        for category in data_dict['categorical_columns']:
            # Skip columns that contain lists (like amenities)
            if category in dataset_log.columns:
                try:
                    # Check if the column contains hashable values
                    unique_values = dataset_log[category].unique()
                    category_dummies_dict[category] = np.sort(unique_values).tolist()
                except TypeError:
                    # Skip columns with unhashable types (like lists)
                    print(f"Skipping column '{category}' as it contains unhashable types (likely lists)")
                    continue

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Airbnb Buenos Aires")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Airbnb Buenos Aires"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(data,
                                                 source="https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz",
                                                 name="airbnb_data_complete")
        mlflow_dataset_dummies = mlflow.data.from_pandas(df,
                                                         source="https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz",
                                                         targets=target_col,
                                                         name="airbnb_data_preprocessed")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")

    @task(
        task_id="split_dataset"
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        import json
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        data_original_path = "s3://data/raw/airbnb_preprocessed.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # Use our helper function to get variables from YAML
        variables = get_variables_from_yaml()
        test_size = variables["test_size"]
        target_col = variables["target_col"]
        feature_cols = variables["feature_cols"]

        X = dataset[feature_cols]
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        save_to_csv(X_train, "s3://data/final/train/airbnb_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/airbnb_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/airbnb_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/airbnb_y_test.csv")

    @task(
        task_id="encode_normalize_features"
    )
    def encode_normalize_features():
        """
        Encoding of categorical features
        Normalization of numerical features
        """

        from airflow.models import Variable

        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        import mlflow.sklearn

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/airbnb_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/airbnb_X_test.csv")

        # Use our helper function to get variables from YAML
        variables = get_variables_from_yaml()
        cat_cols = variables["cat_cols"]
        num_cols = [col for col in X_train.columns if col not in cat_cols]

        # Preprocesador
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ])


        X_train_arr = preprocessor.fit_transform(X_train)
        X_test_arr = preprocessor.transform(X_test)

        # Get feature names after transformation
        num_feature_names = num_cols
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        all_feature_names = list(num_feature_names) + list(cat_feature_names)

        X_train = pd.DataFrame(X_train_arr, columns=all_feature_names)
        X_test = pd.DataFrame(X_test_arr, columns=all_feature_names)

        save_to_csv(X_train, "s3://data/final/train/airbnb_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/airbnb_X_test.csv")

        # Save information of the dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = preprocessor.named_transformers_['num'].mean_.tolist()
        data_dict['standard_scaler_std'] = preprocessor.named_transformers_['num'].scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Airbnb Buenos Aires")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", preprocessor.named_transformers_['num'].feature_names_in_)
            mlflow.log_param("One Hot Encoder feature names", preprocessor.named_transformers_['cat'].get_feature_names_out())
            mlflow.log_param("Standard Scaler mean values", preprocessor.named_transformers_['num'].mean_)
            mlflow.log_param("Standard Scaler scale values", preprocessor.named_transformers_['num'].scale_)
            mlflow.log_param("One Hot Encoder categories", preprocessor.named_transformers_['cat'].categories_)

    raw = get_data()
    preprocessed = preprocess_data()
    splitted = split_dataset()
    encoded = encode_normalize_features()

    raw >> preprocessed >> splitted >> encoded


dag = process_etl_airbnb_data()