FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for numpy/rosbags
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libsqlite3-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENTRYPOINT ["bash"]
