import mlflow
import mlflow.pyfunc
import pickle
import pandas as pd

# 1. Connect to MLflow inside Docker
mlflow.set_tracking_uri("http://localhost:5000")

# 2. Path to your trained pickle model
MODEL_PATH = "models/gbm_volatility.pkl"

# 3. Load the pickle model into memory
with open(MODEL_PATH, "rb") as f:
    model_obj = pickle.load(f)

# 4. Wrap it into an MLflow PythonFunctionModel
class VolatilityModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

wrapped_model = VolatilityModel(model_obj)

# 5. Log + Register the model in MLflow
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="crypto-vol-ml",
        python_model=wrapped_model,
        registered_model_name="crypto-vol-ml"
    )

print("✅ Model registered successfully to MLflow!")
