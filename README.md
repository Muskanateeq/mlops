# Iris Prediction API

## MLflow

Training runs are recorded in MLflow. By default, MLflow stores its local tracking data in `mlruns/` and uses the `iris-classifier` experiment.

```powershell
.\venv\Scripts\python.exe train_model.py
.\venv\Scripts\mlflow.exe ui --backend-store-uri .\mlruns
```

Open `http://127.0.0.1:5000` to inspect the run's parameters, test accuracy, and logged scikit-learn model.

Set `MLFLOW_TRACKING_URI` and optionally `MLFLOW_EXPERIMENT_NAME` before training to use a remote tracking server or a different experiment.

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "iris-classifier"
.\venv\Scripts\python.exe train_model.py
```
