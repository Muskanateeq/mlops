FROM python:3.11-slim

# This is a folder inside the container. It contains the project runtime files.
WORKDIR /project

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The API needs these Python files and the saved machine-learning model.
COPY main.py schemas.py ./
COPY model ./model

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
