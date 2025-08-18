import os
import datetime as dt
import pandas as pd
import yaml

from airflow.decorators import dag, task

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
        Load configuration from YAML and build resolved constants.
        Expected YAML keys:
          - MLFLOW_HOST, MLFLOW_PORT
          - MODEL_REG_NAME, MODEL_ALIAS
          - FEATURES_URI, OUTPUT_BASE_URI, ID_COLUMN
        """
        from utils.utils_etl import get_variables_from_yaml
        cfg = get_variables_from_yaml()

        # Resolve values with minimal defaults (host/port often come from YAML)
        model_reg_name = cfg.get("MODEL_REG_NAME", "airbnb_model_prod")
        model_alias = cfg.get("MODEL_ALIAS", "champion")
        features_uri = cfg.get("FEATURES_URI", "s3://data/final/test/airbnb_X_test.csv")
        output_base_uri = cfg.get("OUTPUT_BASE_URI", "s3://data/predictions/")
        id_column = cfg.get("ID_COLUMN", "listing_id")

        # S3 endpoint is typically provided via env (works with MinIO). Keep it optional.
        s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")

        return {
            "mlflow_tracking_uri": 'http://mlflow:5000',
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
        Load the champion's metadata from MLflow (version, run_id, source).
        We return only metadata here; the model is loaded again inside the
        prediction task (separate process).
        """
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        client = mlflow.MlflowClient()

        mv = client.get_model_version_by_alias(cfg["model_reg_name"], cfg["model_alias"])
        return {
            "version": mv.version,
            "run_id": mv.run_id,
            "source": mv.source,
        }

    @task(task_id="predict_and_write")
    def predict_and_write(cfg: dict, meta: dict):
        """
        Read features from S3, score them with the champion, and write results to S3.
        - Writes a Parquet dataset partitioned by ts=YYYYmmdd_HHMMSS
        - Saves a small JSON manifest with model metadata next to the outputs
        Returns the final output prefix (partition path).
        """
        import datetime as dt
        import numpy as np
        import pandas as pd
        import awswrangler as wr
        import mlflow

        # Configure S3 endpoint (MinIO)
        wr.config.s3_endpoint_url = cfg["s3_endpoint"]

        # Connect MLflow
        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        client = mlflow.MlflowClient()

        # Get model by alias (champion/challenger)
        mv = client.get_model_version_by_alias(cfg["model_reg_name"], cfg["model_alias"])

        run_id = mv.run_id
        print(f"Champion run_id: {run_id}")

        # Load the model (sklearn flavor)
        model = mlflow.sklearn.load_model(mv.source)

        # Figure out expected feature names (strict match for sklearn models)
        expected_cols = None
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf", None)
            if clf is not None and hasattr(clf, "feature_names_in_"):
                expected_cols = list(clf.feature_names_in_)
        # Si no se puede inferir, dejamos expected_cols=None y solo quitamos el ID

        # Destination prefix with timestamp partition
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dest_prefix = f'{cfg["output_base_uri"]}/ts={ts}'

        chunksize = 200_000
        wrote_any = False

        def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
            """Drop ID col, align to expected_cols (add missing=0, drop extra, reorder)."""
            id_series = None
            id_col = cfg["id_column"]

            if id_col in df.columns:
                id_series = df[id_col].copy()
                X = df.drop(columns=[id_col])
            else:
                # si no existe, generamos uno para salida
                id_series = pd.Series(df.index.astype("int64"), name=id_col)
                X = df.copy()

            if expected_cols is not None:
                # agregar faltantes
                missing = [c for c in expected_cols if c not in X.columns]
                for c in missing:
                    X[c] = 0.0
                # eliminar extras (incluye 'listing_id' u otros no vistos)
                extra = [c for c in X.columns if c not in expected_cols]
                if extra:
                    X = X.drop(columns=extra)
                # reordenar
                X = X[expected_cols]

            return X, id_series

        def _write_chunk(df: pd.DataFrame, part_idx: int):
            """Compute predictions for a chunk and append to dataset."""
            X, ids = _prepare_features(df)

            # Predict
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

        # Try chunked CSV first
        try:
            for i, df in enumerate(wr.s3.read_csv(cfg["features_uri"], chunksize=chunksize)):
                _write_chunk(df, i)
                wrote_any = True
        except Exception:
            pass

        if not wrote_any:
            # Single shot load (CSV o Parquet)
            if cfg["features_uri"].lower().endswith(".parquet"):
                full_df = wr.s3.read_parquet(cfg["features_uri"])
            else:
                full_df = wr.s3.read_csv(cfg["features_uri"])
            _write_chunk(full_df, 0)

        # Write manifest with model and data lineage
        manifest = {
            "model_name": cfg["model_reg_name"],
            "alias": cfg["model_alias"],
            "version": mv.version,
            "run_id": run_id,
            "source": mv.source,
            "features_uri": cfg["features_uri"],
            "output_path": dest_prefix,
            "created_utc": ts,
        }
        wr.s3.to_json(manifest, path=f"{dest_prefix}/_model_manifest.json", orient="records", lines=False)

        return dest_prefix

    @task(task_id="log_prediction_run")
    def log_prediction_run(cfg: dict, meta: dict, output_path: str):
        """
        Create a lightweight MLflow run for traceability of the prediction job.
        """
        import mlflow

        mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
        exp = mlflow.set_experiment("Airbnb Buenos Aires - Predictions")
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