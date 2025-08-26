from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np
import mlflow
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but .* was fitted with feature names",
    category=UserWarning,
)

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def objective(trial, X_train, y_train, experiment_id):
    """
    Optimize RF hyperparams with proper CV where SMOTE is applied ONLY on the train fold.
    Target metric for Optuna: mean CV AUC (maximize).
    """
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"RF Trial: {trial.number}",
                          nested=True):

        # ---- search space ----
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            # IMPORTANT: avoid loky warning when running under multiprocessing
            "n_jobs": 1,
            "random_state": 42,
        }

        # ---- pipeline: SMOTE inside CV ----
        pipe = Pipeline(
            steps=[
                ("smote", SMOTE(random_state=42, k_neighbors=5)),
                ("clf", RandomForestClassifier(**params)),
            ]
        )

        # ---- stratified CV on the original (imbalanced) training data ----
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Ensure y is a 1D pandas Series (keep DataFrame for X to preserve feature names)
        if hasattr(y_train, "squeeze"):
            y_vec = y_train.squeeze()
        else:
            # fallback, but keep shape (n,)
            y_vec = np.asarray(y_train).reshape(-1)

        # AUC is robust under imbalance; cross_val_score uses predict_proba under scoring="roc_auc"
        cv_auc = cross_val_score(
            pipe,
            X_train,          # keep as DataFrame to retain feature names
            y_vec,            # 1D series/array
            scoring="roc_auc",
            cv=skf,
            n_jobs=1,         # <--- clave para eliminar el warning de loky
            error_score="raise",
        )

        auc_mean = float(cv_auc.mean())

        # log to mlflow
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc_mean)

        return auc_mean