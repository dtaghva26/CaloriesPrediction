from config.model_config import MODEL_NAME, MODEL_PARAMS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# ----------------------------
# Model
# ----------------------------
def build_model():
    if MODEL_NAME == "RandomForest":
        return RandomForestRegressor(**MODEL_PARAMS)
    elif MODEL_NAME == "Linear Regression":
        return LinearRegression(**MODEL_PARAMS)
    raise ValueError(f"Unsupported model: {MODEL_NAME}")


def train_model(model, X_train, y_train):
    print("🤖 Training model...")
    model.fit(X_train, y_train)
    return model