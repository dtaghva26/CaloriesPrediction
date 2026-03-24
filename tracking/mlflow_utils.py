from functools import wraps

def mlflow_run(experiment_name: str, run_name: str, enable: bool = True):
    """
    Decorator to wrap a function inside an MLflow run.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not enable:
                return fn(*args, **kwargs)

            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment(experiment_name)
            mlflow.sklearn.autolog()

            with mlflow.start_run(run_name=run_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator