FROM python:3.9-slim
WORKDIR /app
COPY requirements_inference.txt .
RUN pip install -r requirements_inference.txt
COPY . .
CMD ["python", "inference.py"]

