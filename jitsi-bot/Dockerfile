# Build from the repository root:
#   docker build -f jitsi-bot/Dockerfile -t parascribe-jitsi-bot:local .
FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

WORKDIR /app
COPY jitsi-bot/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY jitsi-bot/*.py jitsi-bot/capture.js /app/
COPY jitsi-bot/tests /app/tests

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "/app/bot.py"]
