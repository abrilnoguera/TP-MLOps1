import os
import datetime as dt
import pandas as pd
import yaml

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException

markdown_text = """
### Make Predictions with Champion Model

This DAG:
1) Loads the champion model from MLflow Model Registry (alias defined in YAML),
2) Reads the full features dataset from S3/MinIO,
3) Computes predictions (and probability score when available),
4) Saves results as a partitioned Parquet dataset in S3 with a timestamp partition,
5) Logs a lightweight prediction run in MLflow for traceability.
"""

default_args = {
    "owner": "Abril Noguera - José Roberto Castro - Kevin Nelson Pennington - Pablo Ezequiel Brahim",
    "depends_on_past": False,
    "retries": 1,
}

@dag(
    dag_id="make_predictions",
    description="Predict on full dataset using the current champion model and store results in S3/MinIO.",
    doc_md=markdown_text,
    tags=["predictions", "Airbnb"],
    default_args=default_args,
    catchup=False,
)
def predictions_dag():
    @task(task_id="load_config")
    def load_config():
        """
        Read config (YAML + env) and return resolved constants.
        We intentionally hardcode the tracking URI to the in-cluster MLflow service.
        """
        from utils.utils_etl import get_variables_from_yaml

        cfg = get_variables_from_yaml()

        # Registry name and alias can be overridden via env
        model_reg_name = os.getenv("MODEL_REG_NAME", "airbnb_model_prod")
        model_alias = os.getenv("MODEL_ALIAS", "champion")

        # Feature source and output root in MinIO
        features_uri = os.getenv("FEATURES_URI", "s3://data/final/test/airbnb_X_test.csv")
        output_base_uri = os.getenv("OUTPUT_BASE_URI", "s3://data/predictions/")
        id_column = cfg.get("id_col", "listing_id")

        # MinIO endpoint (used by awswrangler)
        s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")

        return {
            "mlflow_tracking_uri": "http://mlflow:5000",
            "model_reg_name": model_reg_name,
            "model_alias": model_alias,
            "features_uri": features_uri,
            "output_base_uri": output_base_uri.rstrip("/"),
            "id_column": id_column,
            "s3_endpoint": s3_endpoint,
        }

    @task(task_id="load_champion_metadata")
    def load_champion_metadata(cfg: dict):
        """
        Resolve the current champion version in the Model Registry.
        If the alias does not exist yet (no trained/promoted model), skip the DAG cleanly.
        """
        import mlflow
        from mlflow.exceptions import RestException

        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        client = mlflow.MlflowClient()

        try:
            mv = client.get_model_version_by_alias(cfg["model_reg_name"], cfg["model_alias"])
        except RestException as e:
            # Friendly skip when no alias found yet
            raise AirflowSkipException(
                f"No model published yet for '{cfg['model_reg_name']}' alias "
                f"'{cfg['model_alias']}'. Train & promote a model before running predictions."
            ) from e

        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "source": mv.source,  # informative only
        }

    @task(task_id="predict_and_write")
    def predict_and_write(cfg: dict, meta: dict):
        """
        Score features with the champion model and write partitioned outputs to S3/MinIO.
        Model is loaded via the canonical registry URI: models:/NAME@ALIAS.
        """
        import datetime as dt
        import numpy as np
        import pandas as pd
        import awswrangler as wr
        import mlflow
        import json

        # Configure MinIO endpoint for awswrangler
        wr.config.s3_endpoint_url = cfg["s3_endpoint"]

        # Connect MLflow and load model from registry alias (source-agnostic)
        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        client = mlflow.MlflowClient()
        
        mv = client.get_model_version_by_alias(cfg["model_reg_name"], cfg["model_alias"])
        model_uri = mv.source.rstrip("/")
        if not model_uri.endswith("/model"):
            model_uri = model_uri + "/model"

        print(f"[Pred] Loading {cfg['model_reg_name']}@{cfg['model_alias']} -> v{mv.version} (run {mv.run_id})")

        # Prefer sklearn flavor to avoid strict pyfunc schema enforcement
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(model_uri)

        # Determine expected feature names (strict alignment for sklearn)
        expected_cols = None
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", None)
            if clf is not None and hasattr(clf, "feature_names_in_"):
                expected_cols = list(clf.feature_names_in_)

        # Destination partition (ts=YYYYmmdd_HHMMSS)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest_prefix = f'{cfg["output_base_uri"]}/ts={ts}'

        chunksize = 200_000
        wrote_any = False

        def _prepare_features(df: pd.DataFrame):
            """Drop ID col and align columns to the model's expected schema."""
            id_col = cfg["id_column"]

            if id_col in df.columns:
                ids = df[id_col].copy()
                X = df.drop(columns=[id_col])
            else:
                # Create a synthetic id if missing so outputs are traceable
                ids = pd.Series(df.index.astype("int64"), name=id_col)
                X = df.copy()

            if expected_cols is not None:
                # Add missing columns with 0.0, drop extras, and reorder
                missing = [c for c in expected_cols if c not in X.columns]
                for c in missing:
                    X[c] = 0.0
                extra = [c for c in X.columns if c not in expected_cols]
                if extra:
                    X = X.drop(columns=extra)
                X = X[expected_cols]

            return X, ids

        def _write_chunk(df: pd.DataFrame, part_idx: int):
            """Predict for one chunk and append to the parquet dataset."""
            X, ids = _prepare_features(df)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]
                pred = (proba >= 0.5).astype(int)
                out = pd.DataFrame({cfg["id_column"]: ids, "prediction": pred, "score": proba})
            else:
                pred = model.predict(X)
                out = pd.DataFrame({cfg["id_column"]: ids, "prediction": pred})

            wr.s3.to_parquet(
                df=out,
                path=dest_prefix,
                dataset=True,
                mode="append",
                index=False,
            )

        # Stream CSV by chunks when possible (fallback to single read)
        try:
            for i, df in enumerate(wr.s3.read_csv(cfg["features_uri"], chunksize=chunksize)):
                _write_chunk(df, i)
                wrote_any = True
        except Exception:
            pass

        if not wrote_any:
            if cfg["features_uri"].lower().endswith(".parquet"):
                full_df = wr.s3.read_parquet(cfg["features_uri"])
            else:
                full_df = wr.s3.read_csv(cfg["features_uri"])
            _write_chunk(full_df, 0)

        return dest_prefix

    @task(task_id="log_prediction_run")
    def log_prediction_run(cfg: dict, meta: dict, output_path: str):
        """
        Create a lightweight MLflow run for traceability of the prediction job.
        """
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        mlflow.set_experiment("Airbnb Buenos Aires")
        with mlflow.start_run(run_name=f'predictions_{dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")}'):
            mlflow.log_param("model_name", cfg["model_reg_name"])
            mlflow.log_param("model_alias", cfg["model_alias"])
            mlflow.log_param("model_version", meta["version"])
            mlflow.log_param("model_run_id", meta["run_id"])
            mlflow.log_param("features_uri", cfg["features_uri"])
            mlflow.log_param("output_path", output_path)

    cfg = load_config()
    meta = load_champion_metadata(cfg)
    out = predict_and_write(cfg, meta)
    log_prediction_run(cfg, meta, out)

predictions_dag = predictions_dag()