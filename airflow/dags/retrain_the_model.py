import datetime
import os

from airflow.decorators import dag, task

markdown_text = """
### Re-Train the Model for Airbnb Data

This DAG re-trains the model based on new data, tests the previous model, and put in production the new one 
if it performs  better than the old one. It uses the F1 score to evaluate the model with the test data.

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
    description="Re-train the model based on new data, tests the previous model, and put in production the new one if "
                "it performs better than the old one",
    doc_md=markdown_text,
    tags=["Re-Train", "Airbnb"],
    default_args=default_args,
    catchup=False,
)
def processing_dag():

    @task
    def train_the_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr
        import os

        from sklearn.base import clone
        from sklearn.metrics import f1_score
        from mlflow.models import infer_signature

        # Get MLflow tracking URI from environment variable or use default
        mlflow_port =str(os.getenv('MLFLOW_PORT', 5001))
        mlflow.set_tracking_uri(f'http://mlflow:{mlflow_port}')

        def load_the_champion_model():
            model_name = "airbnb_model_prod"
            alias = "champion"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            champion_version = mlflow.sklearn.load_model(model_data.source)

            return champion_version

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/airbnb_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/airbnb_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/airbnb_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/airbnb_y_test.csv")

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            # Track the experiment
            experiment = mlflow.set_experiment("Airbnb Buenos Aires")

            mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                             experiment_id=experiment.experiment_id,
                             tags={"experiment": "challenger models", "dataset": "Airbnb"},
                             log_system_metrics=True)

            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            # Save the artifact of the challenger model
            artifact_path = "model"

            signature = infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format='cloudpickle',
                registered_model_name="airbnb_model_dev",
                metadata={"model_data_version": 1}
            )

            # Obtain the model URI
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, f1_score_value, model_uri):
            client = mlflow.MlflowClient()
            name = "airbnb_model_prod"

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["f1-score"] = str(f1_score_value)

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(name, "challenger", result.version)

        # Load the champion model
        champion_model = load_the_champion_model()

        # Clone the model
        challenger_model = clone(champion_model)

        # Load the dataset
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        # Fit the training model
        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # Obtain the metric of the model
        y_pred = challenger_model.predict(X_test)
        f1_score_value = f1_score(y_test.to_numpy().ravel(), y_pred)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)

        # Record the model
        register_challenger(challenger_model, f1_score_value, artifact_uri)
        
        # End the MLflow run
        mlflow.end_run()

    @task
    def evaluate_champion_challenger():
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

        mlflow.set_tracking_uri('http://mlflow:5000')

        def load_the_model(alias):
            model_name = "airbnb_model_prod"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            model = mlflow.sklearn.load_model(model_data.source)

            return model

        def load_the_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/airbnb_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/airbnb_y_test.csv")

            return X_test, y_test

        def promote_challenger(name):

            client = mlflow.MlflowClient()

            # Demote the champion
            client.delete_registered_model_alias(name, "champion")

            # Load the challenger from registry
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

            # Transform in champion
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):

            client = mlflow.MlflowClient()

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

        # Load the champion model
        champion_model = load_the_model("champion")

        # Load the challenger model
        challenger_model = load_the_model("challenger")

        # Load the dataset
        X_test, y_test = load_the_test_data()

        # Obtain the metric of the models
        y_pred_champion = champion_model.predict(X_test)
        f1_score_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion)

        y_pred_challenger = challenger_model.predict(X_test)
        f1_score_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger)

        metrics_champion = {
            'f1': f1_score_champion,
            'accuracy': accuracy_score(y_test.to_numpy().ravel(), y_pred_champion),
            'precision': precision_score(y_test.to_numpy().ravel(), y_pred_champion),
            'recall': recall_score(y_test.to_numpy().ravel(), y_pred_champion),
            'roc_auc': roc_auc_score(y_test.to_numpy().ravel(), champion_model.predict_proba(X_test)[:, 1])
        }

        metrics_challenger = {
            'f1': f1_score_challenger,
            'accuracy': accuracy_score(y_test.to_numpy().ravel(), y_pred_challenger),
            'precision': precision_score(y_test.to_numpy().ravel(), y_pred_challenger),
            'recall': recall_score(y_test.to_numpy().ravel(), y_pred_challenger),
            'roc_auc': roc_auc_score(y_test.to_numpy().ravel(), challenger_model.predict_proba(X_test)[:, 1])
        }

        experiment = mlflow.set_experiment("Airbnb Buenos Aires")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            # Log all metrics
            for metric, value in metrics_champion.items():
                mlflow.log_metric(f"test_{metric}_champion", value)

            for metric, value in metrics_challenger.items():
                mlflow.log_metric(f"test_{metric}_challenger", value)

            # Decision based on F1 (keep current) or use composite score
            decision_metric = 'f1'  # or create a weighted score

            if metrics_challenger[decision_metric] > metrics_champion[decision_metric]:
                mlflow.log_param("Winner", 'Challenger')
            else:
                mlflow.log_param("Winner", 'Champion')

        name = "airbnb_model_prod"
        if metrics_challenger[decision_metric] > metrics_champion[decision_metric]:
            promote_challenger(name)
        else:
            demote_challenger(name)

    train_the_challenger_model() >> evaluate_champion_challenger()


my_dag = processing_dag()
