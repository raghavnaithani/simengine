FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/simengine

COPY requirements.txt /tmp/requirements.txt
RUN PIP_REQUIRE_HASHES=0 PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10 pip install --no-cache-dir -r /tmp/requirements.txt

# Keep backend as a top-level package so imports like
# `from backend.app...` work consistently in Docker.
COPY . /opt/simengine/backend

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
