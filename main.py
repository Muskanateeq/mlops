import pickle
from fastapi import FastAPI
from schemas import IrisInput, IrisOutput

app = FastAPI(title = "Iris prediction api")
with open("model/model.pkl", 'rb') as f:
    model = pickle.load(f)

CLASSES = ["setosa", "versicolor", "virginica"]

@app.get("/")
def home():
    return("Status: your api is live")

@app.post("/predict", response_model=IrisOutput)
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    predict = model.predict(features)[0]
    conf = model.predict_proba(features).max()
    return IrisOutput(predicted_class=CLASSES[predict], confidence=conf)
