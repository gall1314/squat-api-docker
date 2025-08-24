# CUDA 12.x + Ubuntu 22.04 (כולל ספריות ריצה של NVIDIA + cuDNN)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# חשוב: ב-Fly GPU המכונה היא VM עם דרייבר. לא מתקינים nvidia-driver בתוך הקונטיינר.
# מתקינים ffmpeg וספריות מערכת נפוצות ש-OpenCV/mediapipe צריכים.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgl1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ENV בסיסיים: חשיפת יכולות GPU (כולל "video" עבור NVENC),
# פייתון בלי .pyc ו-stdout לא מאוכסן.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# סביבת venv נקייה
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# שדרוג pip וכלי build
RUN python -m pip install --upgrade pip setuptools wheel

# (לא חובה אצלך, רק אם תצרף בעתיד Torch עם CUDA)
ARG TORCH_CUDA=cu121
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

# דרישות פייתון
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# קוד האפליקציה
COPY . .

# ודא שהפונט קיים (אתה משתמש ב-Roboto בקוד). אם אין בקוד, תוכל להוסיף חבילה:
# RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu && rm -rf /var/lib/apt/lists/*
# או להשאיר את קובץ ה-TTF שלך בפרויקט (כבר יש לך FONT_PATH בקוד)

# האפליקציה מאזינה על 8080 (ודא שב-app.py אתה משתמש ב-PORT מהסביבה)
EXPOSE 8080

# אפשר להדליק DEBUG/לכבות NVENC עם ENV בזמן ריצה (אופציונלי):
# ENV DEBUG=1 FORCE_CPU_ENCODE=0 MAX_WALL_SEC=120

CMD ["python", "app.py"]


