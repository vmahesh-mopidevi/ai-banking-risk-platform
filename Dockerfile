# ---- Base image -------------------------------------------------------------
# Slim Python image keeps the container small while still having pip available.
FROM python:3.11-slim

# Do not write .pyc files and make stdout/stderr unbuffered (better logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

# ---- Dependencies -----------------------------------------------------------
# Copy requirements first so Docker can cache the pip install layer and only
# reinstall when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Project files ----------------------------------------------------------
COPY . .

# Train the model at build time so the image ships ready to serve predictions.
# (In production this step would be replaced by pulling a versioned model from
# the MLflow Model Registry - see the README roadmap.)
RUN python -m src.train_model

# ---- Runtime ----------------------------------------------------------------
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
