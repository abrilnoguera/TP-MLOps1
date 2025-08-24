import os
import json
import requests
import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ---- ENV ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "airbnb_model_prod")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# ---- GLOBAL STATE ----
model = None
model_version: Optional[int] = None
expected_cols: Optional[list[str]] = None
model_uri_cached: Optional[str] = None
last_error: Optional[str] = None

# ---- FASTAPI ----
app = FastAPI(title="Airbnb Predictor", version="1.0.0")

# -------------------------------
# Helpers
# -------------------------------
def _infer_expected_cols(loaded_model) -> Optional[list[str]]:
    """
    - If model has feature_names_in_, use that
    - Else if it's a pipeline, try named_steps['clf'].feature_names_in_
    - Else unknown (None)
    """
    if hasattr(loaded_model, "feature_names_in_"):
        return list(loaded_model.feature_names_in_)
    if hasattr(loaded_model, "named_steps"):
        clf = loaded_model.named_steps.get("clf", None)
        if clf is not None and hasattr(clf, "feature_names_in_"):
            return list(clf.feature_names_in_)
    return None


def _prepare_features(df: pd.DataFrame, id_col: str = "listing_id") -> tuple[pd.DataFrame, pd.Series]:
    """
    - keep an id column if present (or synthesize one)
    - align to expected_cols (add missing=0.0, drop extras, reorder)
    """
    if id_col in df.columns:
        ids = df[id_col].copy()
        X = df.drop(columns=[id_col])
    else:
        ids = pd.Series(df.index.astype("int64"), name=id_col)
        X = df.copy()

    global expected_cols
    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        extra = [c for c in X.columns if c not in expected_cols]
        if extra:
            X = X.drop(columns=extra)
        X = X[expected_cols]

    return X, ids

def load_model() -> None:
    """
    Robust loader:
      - Resolve alias -> ModelVersion
      - Try storage_location, then source, then runs:/<run_id>/model (if exists), then models:/NAME@ALIAS
      - Prefer sklearn flavor; fallback to pyfunc
      - Infer expected_cols
    """
    import boto3
    from urllib.parse import urlparse

    global model, model_version, expected_cols, model_uri_cached, last_error

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.MlflowClient()

    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model_version = int(mv.version)
    except Exception as e:
        model = None
        model_version = None
        model_uri_cached = None
        expected_cols = None
        last_error = f"alias resolve failed: {e}"
        print(f"[Model] alias resolve failed: {e}")
        return

    # Helpers
    def _try_load(uri: str):
        try:
            try:
                m = mlflow.sklearn.load_model(uri)
            except Exception:
                m = mlflow.pyfunc.load_model(uri)
            return m, None
        except Exception as e:
            return None, str(e)

    def _s3_list(prefix: str):
        """List keys in MinIO for debugging."""
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
                config=boto3.session.Config(s3={'addressing_style': 'path'})
            )
            # Bucket is usually 'mlflow'; prefix like '1/<run_id>/artifacts/model/'
            resp = s3.list_objects_v2(Bucket="mlflow", Prefix=prefix, MaxKeys=5)
            return [c["Key"] for c in resp.get("Contents", [])]
        except Exception as e:
            return [f"(s3 list failed: {e})"]

    # Candidate URIs to try
    candidates: list[tuple[str, str]] = []

    # 1) storage_location (if present)
    storage_loc = getattr(mv, "storage_location", None)
    if storage_loc:
        candidates.append(("storage_location", storage_loc))

    # 2) source
    if mv.source:
        candidates.append(("source", mv.source))

    # 3) runs:/<run_id>/model  (only if the prefix actually has files)
    run_prefix = f"1/{mv.run_id}/artifacts/model/"
    keys = _s3_list(run_prefix)
    if keys and not str(keys[0]).startswith("(s3 list failed"):
        candidates.append(("runs_model", f"runs:/{mv.run_id}/model"))
    else:
        # dejar log del listado vacío para diagnóstico
        print(f"[Model] S3 prefix empty for run: bucket=mlflow, prefix={run_prefix}, keys={keys}")

    # 4) models:/NAME@ALIAS (fallback)
    candidates.append(("models_alias", f"models:/{MODEL_NAME}@{MODEL_ALIAS}"))

    last_errs: list[str] = []
    for label, uri in candidates:
        loaded, err = _try_load(uri)
        if loaded is not None:
            model = loaded
            expected_cols = _infer_expected_cols(loaded)
            model_uri_cached = uri
            last_error = None
            print(f"[Model] Loaded v{model_version} from {label}: {uri} "
                  f"(expected_cols={'known' if expected_cols else 'unknown'})")
            return
        else:
            last_errs.append(f"{label}: {err}")

    # If we got here, all attempts failed
    model = None
    model_version = None
    model_uri_cached = None
    expected_cols = None
    last_error = " | ".join(last_errs)
    print(f"[Model] Registry load failed or artifacts unreachable: {last_error}")


# ---------------------------------
# Startup: try to load the model
# ---------------------------------
print("Starting model loading...")
load_model()

# -------------------------------
# Schemas & endpoints
# -------------------------------
class PredictRecords(BaseModel):
    """Accept a list of records (dicts) to predict."""
    records: List[Dict[str, Any]] = Field(..., description="Array of feature dicts")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": model is not None,
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_version": model_version,
    }


@app.get("/model/status")
def model_status():
    if model is None:
        msg = "No trained model published yet or artifacts not reachable."
        if last_error:
            msg += f" Last error: {last_error}"
        return {"available": False, "message": msg}
    return {
        "available": True,
        "name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": model_version,
        "expected_cols_known": expected_cols is not None,
    }


@app.post("/model/reload")
def model_reload():
    load_model()
    if model is None:
        raise HTTPException(status_code=503, detail=f"reload failed: {last_error or 'unknown error'}")
    return {"reloaded": True, "version": model_version}


@app.post("/predict")
def predict(payload: PredictRecords):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available yet. Run training to publish a 'champion' in MLflow.",
        )

    df = pd.DataFrame(payload.records)
    X, ids = _prepare_features(df, id_col="listing_id")

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)
            out = pd.DataFrame({"listing_id": ids, "prediction": pred, "score": proba})
        else:
            pred = model.predict(X)
            out = pd.DataFrame({"listing_id": ids, "prediction": pred})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"prediction failed: {e}")

    return {"count": len(out), "results": json.loads(out.to_json(orient="records"))}