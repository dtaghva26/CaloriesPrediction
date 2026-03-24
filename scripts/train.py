import logging
import pipeline

from config.mlflow_config import ENABLE_MLFLOW, EXPERIMENT_NAME, RUN_NAME
from config.model_config import MODEL_NAME, MODEL_PARAMS
from config.dataset_config import FILE_PATH, TARGET_NAME
from report.generate_html_report import generate_html_report
from utils.save_model import save_model
from tracking.mlflow_utils import mlflow_run
import mlflow
logger = logging.getLogger(__name__)


def _train_pipeline():
    X, y = pipeline.load_and_prepare_data()
    X_train, X_test, y_train, y_test = pipeline.split_data(X, y)
    model = pipeline.build_model(MODEL_NAME, MODEL_PARAMS)
    model = pipeline.train_model(model, X_train, y_train)
    metrics, _, plots = pipeline.evaluate_model(model, X_test, y_test)
    generate_html_report(metrics, plots)
    if ENABLE_MLFLOW:
        pipeline.log_with_mlflow(model, metrics, plots, MODEL_NAME, MODEL_PARAMS)
    save_model(model)


@mlflow_run(EXPERIMENT_NAME, RUN_NAME, ENABLE_MLFLOW)
def train():
    _train_pipeline()


if __name__ == "__main__":
    train()