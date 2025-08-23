# CUDA 12.x + Ubuntu 22.04 (כולל דרייברים בתוך ה־runtime libs)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ffmpeg וספריות X הדרושות ל־opencv, ועוד בסיס לפייתון
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# משתני סביבה מומלצים ל־NVIDIA בתוך קונטיינר
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /app

# התקנת תלויות פייתון בסביבת venv כדי לשמור על אימג’ נקי
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# אופציונלי: נוודא pip עדכני
RUN python -m pip install --upgrade pip

# אם אתה משתמש ב‑PyTorch עם CUDA, עדיף לקבע אינדקס גלגלים מתאים ל‑CUDA
# ניתן לשלוט בגרסה דרך build arg (ברירת מחדל cu121 תואם CUDA 12.1/12.2/12.4 wheels)
ARG TORCH_CUDA=cu121
# אפשר להשאיר את זה כאן כדי ש-requirements.txt יוכל לכלול "torch==<ver>+cu121"
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

# העתקת דרישות והתקנה
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# העתקת הקוד
COPY . .

# אקספוז (התאם לפי האפליקציה שלך)
EXPOSE 8080

# הפעלה
CMD ["python", "app.py"]
