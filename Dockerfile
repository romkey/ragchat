FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY rag_chat.py /app/rag_chat.py
COPY .env.example /app/.env.example

ENTRYPOINT ["python", "/app/rag_chat.py"]
