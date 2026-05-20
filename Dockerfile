FROM python:3.11-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python (sin FastAPI extras si solo se usa como CLI)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente
COPY src/ ./src/
COPY configs/ ./configs/
COPY run_pipeline.py .

# Directorios de datos (se montan como volúmenes en producción)
RUN mkdir -p data/raw data/outputs data/ground_truth

EXPOSE 8000

# Por defecto corre como CLI; para FastAPI cambiar CMD en docker-compose
CMD ["python", "run_pipeline.py", "--help"]
