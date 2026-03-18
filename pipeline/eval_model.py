from config.metric_config import METRICS, _generate_plots
def evaluate_model(model, X_test, y_test, generate_plots=True):
    """
    Evaluate a trained model on test data.

    Returns:
        metrics (dict): computed evaluation metrics
        y_pred (array): predictions
        plots (dict | None): matplotlib figures (if enabled)
    """

    print("📊 Evaluating model...")

    # ---- Predictions ----
    y_pred = model.predict(X_test)

    # ---- Metrics ----
    results = {}
    for name, func in METRICS.items():
        try:
            results[name] = func(y_test, y_pred)
        except Exception as e:
            print(f"⚠️ Skipping metric '{name}': {e}")

    # ---- Pretty Print ----
    print("\n📈 Evaluation Results:")
    for k, v in results.items():
        print(f"   {k.upper():<20} {v:.5f}" if isinstance(v, (int, float)) else f"   {k.upper():<20} {v}")

    # ---- Plots (optional) ----
    plots = None
    if generate_plots:
        try:
            plots = _generate_plots(y_test, y_pred)
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")

    return results, y_pred, plots