from config.model_config import MODEL_NAME, MODEL_PARAMS
from config.mlflow_config import EXPERIMENT_NAME, RUN_NAME
import mlflow
import mlflow.sklearn
def log_with_mlflow(model, metrics, plots):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):

        # ---- Model Info ----
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_params(MODEL_PARAMS)

        # ---- Metrics ----
        mlflow.log_metrics(metrics)

        # ---- Plots ----
        if plots:
            for name, fig in plots.items():
                mlflow.log_figure(fig, f"{name}.png")

        # ---- Model ----
        mlflow.sklearn.log_model(model, "model")

        # ---- Report ----
        mlflow.log_artifact("evaluation_report.html")