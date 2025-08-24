import datetime
import os

from airflow.decorators import dag, task

from airflow.operators.trigger_dagrun import TriggerDagRunOperator 

markdown_text = """
### Re-Train the Model for Airbnb Data

This DAG has two main steps: data loading/validation and training.
- If a champion model exists, it trains a challenger and registers it.
- If not, it performs the initial training and registers a first champion.
Evaluation promotes/demotes based on auc (if both models exist).
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
    dag_id="retrain_the_model",
    description="Load data, then train challenger if champion exists; otherwise perform initial training.",
    doc_md=markdown_text,
    tags=["Re-Train", "Airbnb"],
    default_args=default_args,
    catchup=False,
)
def processing_dag():

    @task
    def train_or_init_model():
        """
        If a champion exists: clone it, train a challenger, log to MLflow, register alias 'challenger'.
        If not: train an initial model and register alias 'champion'.
        """
        import datetime
        import mlflow
        import awswrangler as wr
        import os
        import datetime
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        from sklearn.svm import SVC 
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        from sklearn.base import clone
        from mlflow.models import infer_signature
        from mlflow.exceptions import MlflowException

        from utils.plots import plot_feature_importance, plot_calibration_curve
        from utils.metrics import performance_report
        from utils.optuna_aux import champion_callback, objective
        from utils.utils_etl import get_variables_from_yaml

        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE

        mlflow.set_tracking_uri(f'http://mlflow:5000')

        MODEL_REG_NAME = "airbnb_model_prod"

        def load_train_data():
            import awswrangler as wr, os
            wr.config.s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
            X_train = wr.s3.read_csv("s3://data/final/train/airbnb_X_train.csv").drop(columns=["listing_id"])
            y_train = wr.s3.read_csv("s3://data/final/train/airbnb_y_train.csv")
            return X_train, y_train

        def load_test_data():
            import awswrangler as wr, os
            wr.config.s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
            X_test = wr.s3.read_csv("s3://data/final/test/airbnb_X_test.csv").drop(columns=["listing_id"])
            y_test = wr.s3.read_csv("s3://data/final/test/airbnb_y_test.csv")
            return X_test, y_test

        def champion_exists(client, name):
            try:
                client.get_model_version_by_alias(name, "champion")
                return True
            except MlflowException:
                return False
        
        def load_the_champion_model():
            model_name = "airbnb_model_prod"
            alias = "champion"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            champion_version = mlflow.sklearn.load_model(model_data.source)

            return champion_version

        def load_mlflow(model, X_train, y_train, X_test, y_test):
            # Train metrics
            y_train_proba = model.predict_proba(X_train)[:, 1]
            out_deciles_train, summary_train = performance_report(
                y_true=y_train.values.ravel(),
                y_score=y_train_proba,
                threshold=0.5
            )

            # Test metrics
            y_test_proba = model.predict_proba(X_test)[:, 1]
            out_deciles_test, summary_test = performance_report(
                y_true=y_test.values.ravel(),
                y_score=y_test_proba,
                threshold=0.5
            )

            train_metrics = {f"train_{k}": float(v) for k, v in summary_train.iloc[0].items()}
            test_metrics  = {f"test_{k}":  float(v) for k, v in summary_test.iloc[0].items()}

            for metric, value in train_metrics.items():
                mlflow.log_metric(metric, value)
            for metric, value in test_metrics.items():
                mlflow.log_metric(metric, value)

            os.makedirs('./artifacts', exist_ok=True)
            out_deciles_test.to_html('./artifacts/performance_table.html')
            mlflow.log_artifact('./artifacts/performance_table.html')

            fig, imp = plot_feature_importance(model, X_train.columns, X=X_train, y=y_train, kind="auto", n_top=20, save_path='./artifacts/feature_importance.png')
            mlflow.log_artifact('./artifacts/feature_importance.png')

            calibration_fig = plot_calibration_curve(y_train.values.ravel(), y_train_proba, save_path='./artifacts/calibration_curve_train.png')
            mlflow.log_artifact('./artifacts/calibration_curve_train.png')

            calibration_fig = plot_calibration_curve(y_test.values.ravel(), y_test_proba, save_path='./artifacts/calibration_curve_test.png')
            mlflow.log_artifact('./artifacts/calibration_curve_test.png')

            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                serialization_format="cloudpickle",
                metadata={"model_data_version": 1},
            )

            return float(summary_test.iloc[0]["auc"])

        def mlflow_track_experiment(model, X_train, y_train, X_test, y_test):
            experiment = mlflow.set_experiment("Airbnb Buenos Aires")

            with mlflow.start_run(
                run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": "Airbnb"},
                log_system_metrics=True
            ) as run:
                params = model.get_params()
                params["model"] = type(model).__name__
                mlflow.log_params(params)

                auc = load_mlflow(model, X_train, y_train, X_test, y_test)

                run_id = run.info.run_id      

            return run_id, auc
        
        def create_initial_model():
            experiment = mlflow.set_experiment("Airbnb Buenos Aires")

            os.makedirs('./artifacts', exist_ok=True)

            # Run Optuna search
            with mlflow.start_run(
                run_name="best_hyperparam_" + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "optuna tuning", "dataset": "Airbnb"},
                log_system_metrics=True
            ) as run:

                study = optuna.create_study(direction="maximize")
                study.set_user_attr("winner", None)

                study.optimize(
                    lambda trial: objective(trial, X_train, y_train, experiment.experiment_id),
                    n_trials=10,
                    callbacks=[champion_callback]
                )

                # Log best params
                mlflow.log_params(study.best_params)

                # Train model with best params
                model = RandomForestClassifier(**study.best_params, n_jobs=-1, random_state=42)
                model.fit(X_train, y_train.to_numpy().ravel())

                auc = load_mlflow(model, X_train, y_train, X_test, y_test)

                run_id = run.info.run_id

            return model, run_id, auc
        
        def register(run_id: str,
                    alias: str,
                    name: str = "airbnb_model_prod",
                    extra_tags: dict | None = None,
                    create_if_missing: bool = False) -> int:
            """
            Set an alias on an existing ModelVersion produced by `run_id` WITHOUT creating
            a new version if one already exists.

            Workflow:
            - Look for ModelVersions in the registry `name` that were logged with the given run_id.
            - If found:
                - Use the most recent version and only update the alias.
            - If not found:
                - If create_if_missing=True -> create a new ModelVersion from runs:/{run_id}/model
                - If create_if_missing=False -> raise ValueError
            Returns the version number used.
            """
            import mlflow
            from mlflow.exceptions import MlflowException

            mlflow.set_tracking_uri("http://mlflow:5000")
            c = mlflow.MlflowClient()

            # 1) Look up if a ModelVersion already exists for this run_id
            existing = [
                mv for mv in c.search_model_versions(f"name = '{name}'")
                if mv.run_id == run_id
            ]
            if existing:
                # If there are multiple, take the most recent one
                existing.sort(key=lambda mv: int(mv.version), reverse=True)
                version_to_use = int(existing[0].version)
            else:
                if not create_if_missing:
                    raise ValueError(
                        f"No ModelVersion found for run_id={run_id} in '{name}'. "
                        f"Log the model with registered_model_name or call with create_if_missing=True."
                    )
                # 2) Create a new ModelVersion only if explicitly allowed
                source_uri = f"runs:/{run_id}/model"
                tags = {}
                if extra_tags:
                    tags.update({k: str(v) for k, v in extra_tags.items()})
                # Ensure the RegisteredModel exists
                try:
                    c.get_registered_model(name)
                except MlflowException:
                    c.create_registered_model(name)
                mv = c.create_model_version(name=name, source=source_uri, run_id=run_id, tags=tags)
                version_to_use = int(mv.version)

            # 3) Set the alias (delete it first if it already exists, then reassign)
            try:
                c.set_registered_model_alias(name, alias, version_to_use)
            except MlflowException:
                try:
                    c.delete_registered_model_alias(name, alias)
                except MlflowException:
                    pass
                c.set_registered_model_alias(name, alias, version_to_use)

            return version_to_use

        # Load data
        X_train, y_train = load_train_data()
        X_test, y_test = load_test_data()

        client = mlflow.MlflowClient()

        if champion_exists(client, MODEL_REG_NAME):
            print("Champion model exists. Training challenger...")

            champion_model = load_the_champion_model()
            challenger_model = clone(champion_model)

            pipe = Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=42, k_neighbors=5)),
                    ("clf", challenger_model),
                ]
            )
            pipe.fit(X_train, y_train.to_numpy().ravel())

            run_id, auc = mlflow_track_experiment(pipe, X_train, y_train, X_test, y_test)

            # Optional extra tags
            extra = {"model": type(challenger_model).__name__}
            _ = register(
                    run_id=run_id,
                    alias="challenger",
                    name=MODEL_REG_NAME,
                    extra_tags={**(extra or {}), "auc": auc},
                    create_if_missing=True
                )

        else:
            print("No champion model found. Training initial model...")
            model, run_id, auc = create_initial_model()
            extra = {"model": type(model).__name__}
            _ = register(
                    run_id=run_id,
                    alias="champion",
                    name=MODEL_REG_NAME,
                    extra_tags={**(extra or {}), "auc": auc},
                    create_if_missing=True
                )

        # end run safety (no-op if ya cerramos con context manager)
        mlflow.end_run()

    @task
    def evaluate_champion_challenger():
        import mlflow
        from mlflow.exceptions import MlflowException

        # Same tracking URI you used for training
        mlflow.set_tracking_uri(f'http://mlflow:5000')

        MODEL_REG_NAME = "airbnb_model_prod"
        DECISION_METRIC_KEY = "test_auc"  

        client = mlflow.MlflowClient()

        def get_version_and_metrics(alias: str):
            """
            Returns (model_version, metrics_dict) for a given alias.
            Reads metrics from the run associated with that model version.
            """
            mv = client.get_model_version_by_alias(MODEL_REG_NAME, alias)  # -> ModelVersion
            run = client.get_run(mv.run_id)                                # -> Run
            metrics = dict(run.data.metrics)                               # last metric values
            return mv, metrics

        def metric_value(metrics: dict, key: str, fallback_keys=("auc",)):
            """
            Returns the decision metric; if it does not exist, tries fallbacks.
            If none are found, returns None.
            """
            if key in metrics:
                return metrics[key]
            for fk in fallback_keys:
                if fk in metrics:
                    return metrics[fk]
            return None

        # Try to load both aliases; if one is missing, exit gracefully
        try:
            champion_mv, champion_metrics = get_version_and_metrics("champion")
        except MlflowException:
            print("No champion alias found — nothing to compare/promote.")
            return

        try:
            challenger_mv, challenger_metrics = get_version_and_metrics("challenger")
        except MlflowException:
            print("No challenger alias found — nothing to compare/promote.")
            return

        # Extract decision metric
        champion_val = metric_value(champion_metrics, DECISION_METRIC_KEY)
        challenger_val = metric_value(challenger_metrics, DECISION_METRIC_KEY)

        if champion_val is None or challenger_val is None:
            print(f"Decision metric not found (champion={champion_val}, challenger={challenger_val}).")
            return

        print(f"Champion {DECISION_METRIC_KEY}: {champion_val}")
        print(f"Challenger {DECISION_METRIC_KEY}: {challenger_val}")

        # Promotion rule: if challenger is better, promote it
        if challenger_val > champion_val:
            # Remove current champion alias
            client.delete_registered_model_alias(MODEL_REG_NAME, "champion")
            # Remove challenger alias from the winner and set it as champion
            client.delete_registered_model_alias(MODEL_REG_NAME, "challenger")
            client.set_registered_model_alias(MODEL_REG_NAME, "champion", challenger_mv.version)
            print(f"Promoted version {challenger_mv.version} to 'champion'.")
        else:
            # Challenger does not surpass champion: just remove its alias
            client.delete_registered_model_alias(MODEL_REG_NAME, "challenger")
            print(f"Kept champion (v{champion_mv.version}). Challenger (v{challenger_mv.version}) demoted (alias removed).")

    # Trigger predictions DAG after ETL finishes
    trigger_predictions = TriggerDagRunOperator(
        task_id="trigger_predictions",
        trigger_dag_id="make_predictions",
        wait_for_completion=False,
        reset_dag_run=True,
        retries=0,
    )
    
    train_or_init_model() >> evaluate_champion_challenger() >> trigger_predictions


my_dag = processing_dag()
