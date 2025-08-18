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

        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE

        mlflow.set_tracking_uri(f'http://mlflow:5000')

        MODEL_REG_NAME = "airbnb_model_prod"

        def load_train_data():
            X_train = wr.s3.read_csv("s3://data/final/train/airbnb_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/airbnb_y_train.csv")

            return X_train, y_train
        
        def load_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/airbnb_X_test.csv")
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

            # Log metrics
            for metric, value in train_metrics.items():
                mlflow.log_metric(metric, value)

            for metric, value in test_metrics.items():
                mlflow.log_metric(metric, value)

            out_deciles_test.to_html('./artifacts/performance_table.html')
            mlflow.log_artifact('./artifacts/performance_table.html')

            # Feature importance
            fig, imp = plot_feature_importance(model, X_train.columns, kind="auto", n_top=20, save_path='./artifacts/feature_importance.png')
            mlflow.log_artifact('./artifacts/feature_importance.png')

            # Calibration plot
            calibration_fig = plot_calibration_curve(y_train.values.ravel(), y_train_proba, save_path='./artifacts/calibration_curve_train.png')
            mlflow.log_artifact('./artifacts/calibration_curve_train.png')

            calibration_fig = plot_calibration_curve(y_test.values.ravel(), y_test_proba, save_path='./artifacts/calibration_curve_test.png')
            mlflow.log_artifact('./artifacts/calibration_curve_test.png')

            # Save the artifact of the challenger model
            artifact_path = "model"

            signature = infer_signature(X_test, model.predict(X_test))

            mlflow.sklearn.log_model(
                sk_model=model,
                name=artifact_path,
                signature=signature,
                serialization_format='cloudpickle',
                registered_model_name="airbnb_model_prod",
                metadata={"model_data_version": 1}
            )

            return summary_test['auc']

        def mlflow_track_experiment(model, X_train, y_train, X_test, y_test):
            # Track the experiment
            experiment = mlflow.set_experiment("Airbnb Buenos Aires")

            # Create the directory if it doesn't exist
            os.makedirs('./artifacts', exist_ok=True)

            mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                             experiment_id=experiment.experiment_id,
                             tags={"experiment": "challenger models", "dataset": "Airbnb"},
                             log_system_metrics=True)

            # Log model parameters
            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            # Load MLflow metrics
            roc_auc_score_value = load_mlflow(model, X_train, y_train, X_test, y_test)

            run = mlflow.active_run()
            run_id = run.info.run_id

            return run_id, roc_auc_score_value
        
        def create_initial_model():

            # Track the experiment
            experiment = mlflow.set_experiment("Airbnb Buenos Aires")

            # Create the directory if it doesn't exist
            os.makedirs('./artifacts', exist_ok=True)

            mlflow.start_run(run_name="best_hyperparam_"  + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                             experiment_id=experiment.experiment_id,
                             nested=True,
                             tags={"experiment": "optuna tuning", "dataset": "Airbnb"},
                             log_system_metrics=True)
        
            # Define the objective function for Optuna
            study = optuna.create_study(direction="maximize")
            study.set_user_attr("winner", None)
            study.optimize(lambda trial: objective(trial, X_train, y_train, experiment.experiment_id), n_trials=10, callbacks=[champion_callback])

            # Save the best parameters
            mlflow.log_params(study.best_params)

            # Define the model based on the best parameters
            model = RandomForestClassifier(**study.best_params, n_jobs=-1, random_state=42)

            model = model.fit(X_train, y_train.to_numpy().ravel())

            # Load MLflow metrics
            roc_auc_score_value = load_mlflow(model, X_train, y_train, X_test, y_test)

            run = mlflow.active_run()
            run_id = run.info.run_id

            return model, run_id, roc_auc_score_value
        
        def register(model, roc_auc_score_value, run_id, alias = "challenger"):
            client = mlflow.MlflowClient()
            name = "airbnb_model_prod"

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["auc"] = str(roc_auc_score_value)

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=f"runs:/{run_id}/model",
                run_id=run_id,
                tags=tags
            )

            # Save the alias
            client.set_registered_model_alias(name, alias, result.version)

        # Load the dataset
        X_train, y_train = load_train_data()
        X_test, y_test = load_test_data()

        # Initialize MLflow client
        client = mlflow.MlflowClient()

        if champion_exists(client, MODEL_REG_NAME):
            print("Champion model exists. Training challenger...")

            # Load the champion model
            champion_model = load_the_champion_model()
            
            # Clone the champion model to create a challenger
            challenger_model = clone(champion_model)

            pipe = Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=42, k_neighbors=5)),
                    ("clf", challenger_model),
                ]
            )

            # Fit the training model
            pipe.fit(X_train, y_train.to_numpy().ravel())

            # MLFlow track experiment
            run_id, roc_auc_score_value = mlflow_track_experiment(pipe, X_train, y_train, X_test, y_test)

            # Register the challenger model
            register(challenger_model, roc_auc_score_value, run_id, alias="challenger")

        else:
            print("No champion model found. Training initial model...")
            # Create the initial model
            model, run_id, roc_auc_score_value = create_initial_model()

            # Register the champion model
            register(model, roc_auc_score_value, run_id, alias="champion")

        # End the MLflow run
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
