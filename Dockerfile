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
RUN python -m pip install --upgrade pip setuptools wheel

# התקנת דרישות
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# קוד האפליקציה
COPY . .

# אם אתה משתמש בפונט מקומי בקוד – ודא שקובץ ה‑TTF נמצא ברפו. (או התקן חבילת פונטים מערכתית)

# ה־API מאזין על 8080 (ודא שב-app.py אתה קורא PORT מהסביבה או ברירת מחדל 8080)
EXPOSE 8080

CMD ["python", "app.py"]


