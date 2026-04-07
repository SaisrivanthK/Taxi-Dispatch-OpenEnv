FROM python:3.11-slim

ENV PORT=7860         PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends         build-essential curl         && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s         CMD curl -f http://localhost:${PORT}/ || exit 1

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
