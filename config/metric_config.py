from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
    median_absolute_error,
    max_error,
    explained_variance_score,
)
import numpy as np
import matplotlib.pyplot as plt
METRICS = {
    "rmse": lambda y, y_pred: root_mean_squared_error(y, y_pred),
    "mae": lambda y, y_pred: mean_absolute_error(y, y_pred),
    "r2": lambda y, y_pred: r2_score(y, y_pred),
    "mse": lambda y, y_pred: mean_squared_error(y, y_pred),
    "median_ae": lambda y, y_pred: median_absolute_error(y, y_pred),
    "max_error": lambda y, y_pred: max_error(y, y_pred),
    "explained_variance": lambda y, y_pred: explained_variance_score(y, y_pred),

    # custom metrics 👇
    "bias": lambda y, y_pred: np.mean(y_pred - y),
    "error_std": lambda y, y_pred: np.std(y_pred - y),
    "p95_error": lambda y, y_pred: np.percentile(np.abs(y_pred - y), 95),

    # ⚠️ safe MAPE
    "mape": lambda y, y_pred: np.mean(
        np.abs((y - y_pred) / np.clip(np.abs(y), 1e-8, None))
    ) * 100,
}
def _generate_plots(y_test, y_pred):
    plots = {}

    # 1. Predicted vs Actual
    fig1 = plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plots["pred_vs_actual"] = fig1

    # 2. Residuals distribution
    residuals = y_test - y_pred
    fig2 = plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plots["residuals_hist"] = fig2

    # 3. Residuals vs Predictions
    fig3 = plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predictions")
    plots["residuals_vs_pred"] = fig3

    return plots
