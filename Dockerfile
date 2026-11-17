FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
# CPU-only torch keeps the image ~2 GB instead of ~6 GB
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt ultralytics easyocr
COPY . .

ENV CERTISCAN_WEIGHTS=/app/models/layout_best.pt \
    CERTISCAN_OCR=easyocr \
    DATABASE_URL=sqlite:////data/certiscan.db \
    CERTISCAN_UPLOADS=/data/uploads
VOLUME /data
EXPOSE 8000
CMD ["gunicorn", "-w", "1", "-t", "180", "-b", "0.0.0.0:8000", "app.server:create_app()"]
