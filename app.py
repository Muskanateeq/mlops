"""Gradio interface deployed as a Hugging Face Space."""

import pickle
from pathlib import Path

import gradio as gr


MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.pkl"
CLASSES = ["setosa", "versicolor", "virginica"]

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)


def predict(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    predicted_index = model.predict(features)[0]
    confidence = model.predict_proba(features).max()
    return CLASSES[predicted_index], f"{confidence:.2%}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal length", value=5.1),
        gr.Number(label="Sepal width", value=3.5),
        gr.Number(label="Petal length", value=1.4),
        gr.Number(label="Petal width", value=0.2),
    ],
    outputs=[
        gr.Textbox(label="Predicted class"),
        gr.Textbox(label="Confidence"),
    ],
    title="Iris Flower Prediction",
    description="Enter the flower measurements to predict its Iris species.",
)


if __name__ == "__main__":
    demo.launch()
