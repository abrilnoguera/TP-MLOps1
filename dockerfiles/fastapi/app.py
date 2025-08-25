import os
import json
import requests
import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import awswrangler as wr

# ---- ENV ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "airbnb_model_prod")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
FEATURES_TRAIN_URI = os.getenv("FEATURES_TRAIN_URI", "s3://data/final/train/airbnb_X_train.csv")
FEATURES_TEST_URI  = os.getenv("FEATURES_TEST_URI",  "s3://data/final/test/airbnb_X_test.csv")
ID_COLUMN          = os.getenv("ID_COLUMN", "listing_id")
WR_S3_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
_FEATURE_ID_CACHE: dict = {"ts": 0.0, "ids": []}
_FEATURE_ID_TTL = int(os.getenv("FEATURE_IDS_TTL_SEC", "300"))  # 5 minutes

# cache de dataset unido en memoria (opcional, simple)
_features_df_cache: Optional[pd.DataFrame] = None

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

def _load_features_union() -> pd.DataFrame:
    """
    Read train & test feature CSVs from MinIO (awswrangler) and return their union.
    Keeps the ID column (ID_COLUMN). No target columns aquí.
    """
    import awswrangler as wr

    wr.config.s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")

    df_train = wr.s3.read_csv(FEATURES_TRAIN_URI)
    df_test  = wr.s3.read_csv(FEATURES_TEST_URI)

    for df in (df_train, df_test):
        if ID_COLUMN in df.columns:
            try:
                df[ID_COLUMN] = pd.to_numeric(df[ID_COLUMN], errors="ignore")
            except Exception:
                pass

    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    if ID_COLUMN not in df.columns:
        raise RuntimeError(f"ID column '{ID_COLUMN}' not found in features CSVs.")

    return df

def _get_features_for_ids(id_list: list) -> pd.DataFrame:
    """
    Return rows for the given IDs from the cached (or freshly loaded) features union.
    """
    global _features_df_cache
    if _features_df_cache is None:
        _features_df_cache = _load_features_union()

    series = _features_df_cache[ID_COLUMN]
    if pd.api.types.is_integer_dtype(series):
        try:
            id_list = [int(x) for x in id_list]
        except Exception:
            pass

    sub = _features_df_cache[_features_df_cache[ID_COLUMN].isin(id_list)].copy()
    return sub

def _load_feature_ids() -> List[int]:
    """
    Read only the 'listing_id' column from train + test features in S3/MinIO,
    concatenate, drop duplicates, and return as a python list[int].
    """
    wr.config.s3_endpoint_url = WR_S3_ENDPOINT

    # Read ONLY the id column – faster and cheaper
    cols = ["listing_id"]
    dfs = []
    for uri in (FEATURES_TRAIN_URI, FEATURES_TEST_URI):
        try:
            if uri.lower().endswith(".parquet"):
                df = wr.s3.read_parquet(uri, columns=cols)
            else:
                df = wr.s3.read_csv(uri, usecols=cols)
            dfs.append(df)
        except Exception:
            # If one side is missing, keep going with what we have
            continue

    if not dfs:
        return []

    all_ids = (
        pd.concat(dfs, ignore_index=True)
        .dropna(subset=["listing_id"])
        .listing_id.astype("int64", errors="ignore")
        .drop_duplicates()
        .tolist()
    )

    # Normalize to regular python ints for JSON
    return [int(x) for x in all_ids]

def _get_ids_cached() -> List[int]:
    """
    Return cached ids if still fresh; otherwise reload and refresh the cache.
    """
    now = time.time()
    if (now - _FEATURE_ID_CACHE["ts"]) < _FEATURE_ID_TTL and _FEATURE_ID_CACHE["ids"]:
        return _FEATURE_ID_CACHE["ids"]

    ids = _load_feature_ids()
    _FEATURE_ID_CACHE["ids"] = ids
    _FEATURE_ID_CACHE["ts"] = now
    return ids

from typing import Union

class PredictById(BaseModel):
    ids: List[Union[int, str]] = Field(..., description="Listing IDs to score")

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
    Load the model from the MLflow Model Registry:
    - Resolve the alias (e.g., champion) in the Registry,
    - Use mv.source (Registry path, not run path),
    - Normalize path to end with /model,
    - Try sklearn flavor first (lighter), fallback to pyfunc,
    - Store expected feature columns if available,
    - Keep API alive even if loading fails (model=None).
    """
    global model, model_version, expected_cols, model_uri_cached, last_error

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = mlflow.MlflowClient()
        # 1) Resolve alias in the Registry
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        # 2) Use the source returned by the Registry (models/m-.../artifacts[/model])
        model_uri = mv.source.rstrip("/")
        if not model_uri.endswith("/model"):
            model_uri = model_uri + "/model"

        # 3) Load the model
        try:
            loaded = mlflow.sklearn.load_model(model_uri)
        except Exception:
            loaded = mlflow.pyfunc.load_model(model_uri)

        # 4) Update global state for /health and /status
        model = loaded
        model_version = int(mv.version)
        expected_cols = _infer_expected_cols(loaded)
        model_uri_cached = model_uri
        last_error = None
        print(f"[Model] Loaded from Registry: {MODEL_NAME}@{MODEL_ALIAS} "
              f"(v{model_version}) -> {model_uri}; expected_cols={'known' if expected_cols else 'unknown'}")

    except Exception as e:
        # If anything fails, keep API alive but mark model unavailable
        model = None
        model_version = None
        expected_cols = None
        model_uri_cached = None
        last_error = str(e)
        print(f"[Model] Registry load failed or artifacts unreachable: {e}")



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

@app.post("/predict/by_id")
def predict_by_id(payload: PredictById):
    if model is None:
        raise HTTPException(status_code=503, detail="No trained model available yet.")

    ids_req = list(payload.ids)
    feats = _get_features_for_ids(ids_req)

    if feats.empty:
        return {"count": 0, "missing_ids": ids_req, "results": []}

    # Extraer coordenadas si existen
    coords = {}
    if "lat" in feats.columns and "lon" in feats.columns:
        coords = feats[[ID_COLUMN, "lat", "lon"]].set_index(ID_COLUMN).to_dict("index")

    X, ids = _prepare_features(feats, id_col=ID_COLUMN)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        out = pd.DataFrame({
            ID_COLUMN: ids,
            "prediction": pred,
            "score": proba
        })
    else:
        pred = model.predict(X)
        out = pd.DataFrame({ID_COLUMN: ids, "prediction": pred})

    # Merge coords into results
    results = []
    for rec in out.to_dict(orient="records"):
        lid = rec[ID_COLUMN]
        if lid in coords:
            rec["lat"] = coords[lid]["lat"]
            rec["lon"] = coords[lid]["lon"]
        rec["model_version"] = model_version
        results.append(rec)

    return {
        "count": len(results),
        "missing_ids": [i for i in ids_req if i not in out[ID_COLUMN].tolist()],
        "results": results
    }

@app.get("/features/ids")
def features_ids(
    limit: int = Query(500, ge=1, le=10000, description="Maximum number of IDs to return"),
    shuffle: bool = Query(True, description="Shuffle before truncating")
):
    """
    Return available listing_ids from the preprocessed features (train + test).
    - Reads only the 'listing_id' column from both sources.
    - Dedupe + (optionally) shuffle + limit.
    Example: GET /features/ids?limit=1000&shuffle=false
    Response: {"ids": [11508, 12345, ...]}
    """
    ids = _get_ids_cached().copy()
    if shuffle:
        import random
        random.shuffle(ids)
    if limit:
        ids = ids[:limit]
    return {"ids": ids}