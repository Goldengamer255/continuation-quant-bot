# Live execution image. Research (training/backtesting) is run locally; the
# container only needs to load a trained model and trade.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core/ core/
COPY data/ data/
COPY research/ research/
COPY execution/ execution/
COPY config/settings.example.yaml config/settings.example.yaml

# Never run a trading process as root.
RUN useradd --create-home --uid 10001 bot && chown -R bot:bot /app
USER bot

# Secrets, live config and model weights are NOT baked into the image.
# Supply them at runtime:
#   docker run --env-file .env \
#     -v "$PWD/config/settings.yaml:/app/config/settings.yaml:ro" \
#     -v "$PWD/models:/app/models:ro" continuation-quant-bot
#
# SIGTERM (docker stop) triggers the engine's graceful shutdown path.
STOPSIGNAL SIGTERM
CMD ["python", "-m", "execution.engine"]
