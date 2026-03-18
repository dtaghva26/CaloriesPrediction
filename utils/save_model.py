import joblib
def save_model(model, path="model/model.pkl"):
    print("💾 Saving model...")
    joblib.dump(model, path)
