import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "model.pkl"
DEFAULT_TRACKING_URI = (PROJECT_DIR / "mlruns").as_uri()


def train() -> None:
    """Train the Iris classifier and record the run in MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classifier")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    features, target = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    parameters = {"n_estimators": 100, "random_state": 42}
    model = RandomForestClassifier(**parameters)

    with mlflow.start_run():
        model.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(x_test))

        mlflow.log_params(parameters)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.sklearn.log_model(model, name="model")

        MODEL_PATH.parent.mkdir(exist_ok=True)
        with MODEL_PATH.open("wb") as model_file:
            pickle.dump(model, model_file)

    print(f"Model trained and saved to {MODEL_PATH}")
    print(f"MLflow tracking URI: {tracking_uri}")


if __name__ == "__main__":
    train()

