import datetime
import pandas as pd
import yaml
import os
import pendulum

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator 
BA = pendulum.timezone("America/Argentina/Buenos_Aires")
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
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
}


@dag(
    dag_id="process_etl_airbnb_data",
    dagrun_timeout = datetime.timedelta(minutes=15),
    description="ETL process for Airbnb Buenos Aires, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Airbnb"],
    default_args=default_args,
    schedule='0 2 * * *',
    start_date=pendulum.datetime(2025, 8, 25, tz=BA),
    max_active_runs=1,
    catchup=False,
)
def process_etl_airbnb_data():

    @task(
        task_id="get_data"
    )
    def get_data():
        """
        Load the raw data from Inside Airbnb repository.
        """
        import awswrangler as wr
        from airflow.models import Variable
        from utils.utils_etl import load_and_get_df

        # Fetch dataset
        urls = {
            "Buenos Aires": "https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz",
        }

        # Load and merge all dataframes
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
        Clean and preprocess data:
        - Handle missing values
        - Convert prices, percentages and text fields
        - Create features and target
        - Save processed dataset to S3
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

        from utils.utils_etl import get_variables_from_yaml, eliminar_columnas_con_muchos_nulos, eliminar_filas_con_nulos, remove_html_tags, obtener_tasas_cambio, convert_currency

        data_original_path = "s3://data/raw/airbnb.csv"
        data_end_path = "s3://data/raw/airbnb_preprocessed.csv"
        data = wr.s3.read_csv(data_original_path)
        df = data.copy()

        # keep id as listing_id for inference/UX
        if "id" in df.columns and "listing_id" not in df.columns:
            df = df.rename(columns={"id": "listing_id"})

        # Get scraping date
        date = data['last_scraped'][0]

        # Drop duplicates
        df.drop_duplicates(inplace=True, ignore_index=True)
        
        # Convert host_response_rate and host_acceptance_rate from percentage to float
        df["host_response_rate"] = df["host_response_rate"].str.replace("%", "").astype(float) / 100
        df["host_acceptance_rate"] = df["host_acceptance_rate"].str.replace("%", "").astype(float) / 100

        # Convert price string with $ and , to float
        df["price"] = df["price"].replace('[\$,]', '', regex=True).astype(float)

        # Categorical columns (object or string type)
        categoric_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        print("Categorical columns: ", categoric_cols, '\n')

        # Numeric columns (int or float type)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        print("Numeric columns: ", numeric_cols)

        # Replace '.' or '-' by NaN if they are the only characters
        for column_name in categoric_cols:
            df[column_name] = df[column_name].replace(r'^[\.\-]$', np.nan, regex=True)

        # Replace "N/A" with NaN
        df.replace("N/A", np.nan, inplace=True)

        # Drop columns with more than 90% nulls
        df = eliminar_columnas_con_muchos_nulos(df, umbral=0.9)

        # Keep empty text for descriptions (may hold relevant signals)
        df[["description", "neighborhood_overview", "host_about"]] = df[["description", "neighborhood_overview", "host_about"]].fillna("")

        # Drop neighbourhood column, keep neighbourhood_cleansed
        df.drop(columns=["neighbourhood"], inplace=True)

        # Extract bathrooms number from bathrooms_text if missing
        df["extracted_bathrooms"] = df["bathrooms_text"].str.extract(r"(\d+)").astype(float)
        df["bathrooms"] = df["bathrooms"].fillna(df["extracted_bathrooms"])
        df.drop(columns=["extracted_bathrooms"], inplace=True)

        # Impute first_review and last_review with host_since if missing
        df["first_review"] = df["first_review"].fillna(df["host_since"])
        df["last_review"] = df["last_review"].fillna(df["host_since"])

        # Impute with median
        cols_mediana = ['bathrooms', 'bedrooms', 'beds']
        for col_name in cols_mediana:
            mediana = df[col_name].median()
            df[col_name] = df[col_name].fillna(mediana)

        # Impute with mean
        cols_media = [
            'host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 
            'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 
            'review_scores_communication', 'review_scores_location', 'review_scores_value', 
            'reviews_per_month'
        ]
        for col_name in cols_media:
            media = df[col_name].mean()
            df[col_name] = df[col_name].fillna(media)

        # Impute with mode
        cols_moda = ["host_location", "host_neighbourhood", "host_response_time", "host_is_superhost", "bathrooms_text"]
        for col_name in cols_moda:
            moda = df[col_name].mode().iloc[0]
            df[col_name] = df[col_name].fillna(moda)

        # Define has_availability if all availability columns are 0 and value is missing
        cond = (
            (df['availability_30'] == 0) &
            (df['availability_60'] == 0) &
            (df['availability_90'] == 0) &
            (df['availability_365'] == 0) &
            (df['has_availability'].isna())
        )
        df.loc[cond, 'has_availability'] = 'f'
        df["has_availability"] = df["has_availability"].fillna("t")

        # Drop rows with nulls in critical columns
        df = eliminar_filas_con_nulos(df, ["price", "host_name", "picture_url"])

        # Remove HTML tags
        df = remove_html_tags(df)

        # Clean and convert host_verifications into dummy variables
        df["host_verifications"] = (
            df["host_verifications"]
            .astype(str)
            .str.replace(r"[\[\]'\"]", "", regex=True)
            .str.strip()
            .str.split(",")
            .apply(lambda lst: [item.strip() for item in lst if item])
        )
        unique_types = set()
        df["host_verifications"].dropna().apply(unique_types.update)
        unique_types = sorted(filter(None, unique_types))  
        
        for item in unique_types:
            col_name = "verif_" + item.replace(" ", "_")
            df[col_name] = df["host_verifications"].apply(lambda x: int(item in x) if isinstance(x, list) else 0)

        df.drop(columns=["host_verifications"], inplace=True)

        # Clean amenities and drop afterwards
        df["amenities"] = df["amenities"].astype(str).str.replace(r"[^a-zA-Z ,]", "", regex=True)
        df["amenities"] = df["amenities"].str.replace(r"\s+", " ", regex=True).str.strip()
        df["amenities"] = df["amenities"].str.split(", ").apply(lambda lst: [item.strip() for item in lst if item])

        # Convert price to same currency using exchange rates
        tasas_cambio = obtener_tasas_cambio(date)
        df["price"] = df.apply(lambda row: convert_currency(tasas_cambio, row["price"], row["city"]), axis=1)

        # Create log_price feature
        df["log_price"] = np.log1p(df["price"])

        # Convert t/f to 1/0
        cols_binarias = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]
        for col in cols_binarias:
            df[col] = df[col].map({'t': 1, 'f': 0})

        # Calculate occupation rate and create target
        df["occupation_rate"] = 1 - (df["availability_60"] / 60)
        df["high_occupancy"] = (df["occupation_rate"] >= 0.8).astype(int)
        df.drop(columns=["occupation_rate"], inplace=True)
        
        # Drop amenities (list type)
        df.drop(columns=["amenities"], inplace=True, errors='ignore')

        wr.s3.to_csv(df=df,
                     path=data_end_path,
                     index=False)

        # Save dataset info to S3
        client = boto3.client("s3")
        try:
            client.head_object(Bucket="data", Key="data_info/data.json")
            result = client.get_object(Bucket="data", Key="data_info/data.json")
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                data_dict = {}
            else:
                raise e

        variables = get_variables_from_yaml()
        target_col = variables["target_col"]
        feature_cols = variables["feature_cols"]

        # For logging purposes (everything except the target column)
        dataset_log = df.drop(columns=[target_col], errors="ignore")

        data_dict["columns"] = dataset_log.columns.to_list()
        data_dict["target_col"] = target_col
        data_dict["categorical_columns"] = df.select_dtypes(include=["object", "string"]).columns.tolist()
        data_dict["columns_dtypes"] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}

        # keys for inference/UX
        data_dict["id_column"] = "listing_id"
        data_dict["lat_column"] = "latitude"
        data_dict["lon_column"] = "longitude"
        data_dict["feature_cols_raw"] = feature_cols  # before encoding

        # Unique values of categorical features (for input validation if needed)
        category_dummies_dict = {}
        for category in data_dict["categorical_columns"]:
            if category in dataset_log.columns:
                try:
                    unique_values = dataset_log[category].dropna().unique()
                    category_dummies_dict[category] = np.sort(unique_values).tolist()
                except TypeError:
                    # Skip columns with unhashable types (likely lists)
                    continue
        data_dict["categories_values_per_categorical"] = category_dummies_dict

        data_dict["date"] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)
        client.put_object(Bucket="data", Key="data_info/data.json", Body=data_string)

        # ------- MLflow dataset tracking -------
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("Airbnb Buenos Aires")
        mlflow.start_run(
            run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
            experiment_id=experiment.experiment_id,
            tags={"experiment": "etl", "dataset": "Airbnb Buenos Aires"},
            log_system_metrics=True,
        )
        mlflow_dataset = mlflow.data.from_pandas(
            data,
            source="https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz",
            name="airbnb_data_complete",
        )
        mlflow_dataset_prep = mlflow.data.from_pandas(
            df,
            source="https://data.insideairbnb.com/argentina/ciudad-aut%C3%B3noma-de-buenos-aires/buenos-aires/2025-01-29/data/listings.csv.gz",
            targets=target_col,
            name="airbnb_data_preprocessed",
        )
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_prep, context="Dataset")

    @task(
        task_id="split_dataset"
    )
    def split_dataset():
        """
        Split dataset into train and test sets.
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from utils.utils_etl import get_variables_from_yaml
        import pandas as pd

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        # Load preprocessed data
        data_original_path = "s3://data/raw/airbnb_preprocessed.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # Ensure listing_id exists (raw column was 'id')
        if "listing_id" not in dataset.columns and "id" in dataset.columns:
            dataset = dataset.rename(columns={"id": "listing_id"})

        # Load variables
        variables   = get_variables_from_yaml()
        test_size   = variables["test_size"]
        target_col  = variables["target_col"]
        feature_cols= variables["feature_cols"]

        # Drop duplicates BEFORE the split (optional: by listing_id if present)
        if "listing_id" in dataset.columns:
            dataset = dataset.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)
        else:
            dataset = dataset.drop_duplicates().reset_index(drop=True)

        # Build X with listing_id as first column
        cols_for_X = (["listing_id"] if "listing_id" in dataset.columns else []) + feature_cols
        X = dataset[cols_for_X].copy()
        X['lat'] = X['latitude']
        X['lon'] = X['longitude']

        # y as a Series (not DataFrame) for stratify
        y = dataset[target_col].astype(int)

        # Split (stratified & reproducible)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42, shuffle=True
        )

        # Save
        save_to_csv(X_train, "s3://data/final/train/airbnb_X_train.csv")
        save_to_csv(X_test,  "s3://data/final/test/airbnb_X_test.csv")
        save_to_csv(pd.DataFrame({target_col: y_train}), "s3://data/final/train/airbnb_y_train.csv")
        save_to_csv(pd.DataFrame({target_col: y_test}),  "s3://data/final/test/airbnb_y_test.csv")

    @task(
        task_id="encode_normalize_features"
    )
    def encode_normalize_features():
        """
        Encode categorical features and normalize numeric features.
        Keep `listing_id` column for traceability.
        Save preprocessor and transformed data.
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions
        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        import mlflow.sklearn

        from utils.utils_etl import get_variables_from_yaml
        from utils.plots import plot_correlation_with_target, plot_information_gain_with_target, plot_class_balance

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)
        
        def create_plots(X, y):
            target_column = y.columns[0]
            correlation_plot = plot_correlation_with_target(X, y, target_col=target_column)
            information_gain_plot = plot_information_gain_with_target(X, y, target_col=target_column)
            class_balance_plot = plot_class_balance(y, target_col=target_column)
            return {
                "correlation_plot": correlation_plot, 
                "information_gain_plot": information_gain_plot, 
                "class_balance_plot": class_balance_plot
            }

        # === Load datasets (with listing_id preserved) ===
        X_train = wr.s3.read_csv("s3://data/final/train/airbnb_X_train.csv")
        y_train = wr.s3.read_csv("s3://data/final/train/airbnb_y_train.csv")
        X_test  = wr.s3.read_csv("s3://data/final/test/airbnb_X_test.csv")

        # Keep id + raw coordinates in dataframes
        have_id = "listing_id" in X_train.columns
        have_lat = "lat" in X_train.columns
        have_long = "lon" in X_train.columns 

        if have_id and have_lat and have_long:
            id_train = X_train[["listing_id", "lat", "lon"]].reset_index(drop=True)
            id_test  = X_test[["listing_id", "lat", "lon"]].reset_index(drop=True)
        else:
            id_train = None
            id_test = None

        # Drop id + coords from the features used for training
        drop_cols = []
        if have_id:   drop_cols.append("listing_id")
        if have_lat:  drop_cols.append("lat")
        if have_long: drop_cols.append("long")

        if drop_cols:
            X_train = X_train.drop(columns=drop_cols, errors="ignore")
            X_test  = X_test.drop(columns=drop_cols, errors="ignore")

        # === Preprocess ===
        variables = get_variables_from_yaml()
        cat_cols = variables["cat_cols"]
        num_cols = [c for c in X_train.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ]
        )

        X_train_arr = preprocessor.fit_transform(X_train)
        X_test_arr  = preprocessor.transform(X_test)

        num_feature_names = num_cols
        cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        all_feature_names = list(num_feature_names) + list(cat_feature_names)

        X_train_proc = pd.DataFrame(X_train_arr, columns=all_feature_names)
        X_test_proc  = pd.DataFrame(X_test_arr,  columns=all_feature_names)

        # Re-attach id + raw lat/lon as columnas separadas (al frente), sin tocarlas
        if id_train is not None:
            X_train_proc = pd.concat([id_train, X_train_proc.reset_index(drop=True)], axis=1)
            X_test_proc  = pd.concat([id_test,  X_test_proc.reset_index(drop=True)],  axis=1)

        # === Plots ===
        plots = create_plots(X_train_proc.drop(columns=["listing_id", "lat", "long"], errors="ignore"), y_train)

        # === Save transformed datasets ===
        save_to_csv(X_train_proc, "s3://data/final/train/airbnb_X_train.csv")
        save_to_csv(y_train, "s3://data/final/train/airbnb_y_train.csv")
        save_to_csv(X_test_proc,  "s3://data/final/test/airbnb_X_test.csv")

        # === Save dataset metadata ===
        client = boto3.client('s3')
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            raise e

        data_dict['standard_scaler_mean'] = preprocessor.named_transformers_['num'].mean_.tolist()
        data_dict['standard_scaler_std']  = preprocessor.named_transformers_['num'].scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(Bucket='data', Key='data_info/data.json', Body=data_string)

        # === Track in MLflow ===
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Airbnb Buenos Aires")

        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")
        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")
            mlflow.log_param("Train observations", X_train_proc.shape[0])
            mlflow.log_param("Test observations", X_test_proc.shape[0])
            mlflow.log_param("Standard Scaler feature names", preprocessor.named_transformers_['num'].feature_names_in_)
            mlflow.log_param("One Hot Encoder feature names", preprocessor.named_transformers_['cat'].get_feature_names_out())
            mlflow.log_param("Standard Scaler mean values", preprocessor.named_transformers_['num'].scale_)
            mlflow.log_param("One Hot Encoder categories", preprocessor.named_transformers_['cat'].categories_)

            for plot_name, plot_fig in plots.items():
                mlflow.log_figure(plot_fig, f"feature_evaluation_plots/{plot_name}.png")

    raw = get_data()
    preprocessed = preprocess_data()
    splitted = split_dataset()
    encoded = encode_normalize_features()

    # Trigger retrain DAG after ETL finishes
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain_the_model",
        trigger_dag_id="retrain_the_model",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule="none_failed_min_one_success",
        retries=0,
    )
    # Task dependencies
    raw >> preprocessed >> splitted >> encoded >> trigger_retrain


dag = process_etl_airbnb_data()
