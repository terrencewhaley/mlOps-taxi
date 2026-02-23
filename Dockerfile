FROM python:3.12-slim

WORKDIR /app

COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

COPY training/ ./training/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app/training

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]