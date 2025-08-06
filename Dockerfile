FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ העתק את תיקיית הפונטים
COPY fonts/ /app/fonts/

CMD ["python", "app.py"]
