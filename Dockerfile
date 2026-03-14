FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/     ./api/
COPY models/  ./models/
COPY src/     ./src/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
