FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd -r nonroot && useradd -r -g nonroot nonroot

COPY --chown=nonroot:nonroot . .
# AI downloads the model weights from the internet and saving them to this directory
RUN mkdir -p /home/nonroot/.cache && chown -R nonroot:nonroot /home/nonroot

USER nonroot
CMD ["python", "ai_service_socketio.py"]
