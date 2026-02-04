# syntax=docker/dockerfile:1.5
# Python בסיסי, ללא CUDA/GPU
FROM python:3.10-slim

# ספריות מערכת ש-OpenCV/MediaPipe צריכים + ffmpeg
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# רצוי לשדרג pip וכלי build
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel

# הגדרות להתמודדות עם בעיות רשת PyPI
ENV PIP_DEFAULT_TIMEOUT=30
ENV PIP_RETRIES=5
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_PROGRESS_BAR=off
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"

# התקנת דרישות
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --timeout 30 --retries 5 --prefer-binary \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# קוד האפליקציה
COPY . .

# ה-API מאזין על 8080
EXPOSE 8080

CMD ["python", "app.py"]
