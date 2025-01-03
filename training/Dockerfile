FROM python:3.9-slim
WORKDIR /app
COPY requirements_training.txt .
RUN pip install -r requirements_training.txt
COPY . .
CMD ["python", "train_model.py"]

