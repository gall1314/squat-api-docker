# CUDA 12.x + Ubuntu 22.04 (כולל ספריות ריצה של NVIDIA)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# חשוב: לא צריך libnvidia-encode1 בתוך הקונטיינר (מגיע מה-host של Fly)
# מתקינים ffmpeg וספריות X ל-OpenCV + פייתון
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# לאפשר לקונטיינר להשתמש ביכולות GPU כולל video (NVENC)
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# סביבת venv נקייה
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pip עדכני
RUN python -m pip install --upgrade pip

# אם תרצה להשתמש ב-PyTorch עם CUDA (לא חובה לפרויקט הנוכחי)
ARG TORCH_CUDA=cu121
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

# דרישות פייתון
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# קוד האפליקציה
COPY . .

# האפליקציה מאזינה על 8080 (ודא שבקוד אתה משתמש ב-PORT מהסביבה)
EXPOSE 8080

# הפעלה
CMD ["python", "app.py"]

