# syntax=docker/dockerfile:1

# CPU-only image: torch/torchvision lấy từ index `+cpu` của PyTorch để tránh
# kéo theo CUDA (~2.5 GB thừa) — container không có GPU nên bản CPU là đủ.
FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # torch là wheel lớn (159 MB); mặc định pip chỉ thử 5 lần và dễ đứt giữa đường.
    PIP_RETRIES=10 \
    PIP_TIMEOUT=120 \
    # Ultralytics ghi settings/cache vào đây; trỏ vào thư mục của user app.
    YOLO_CONFIG_DIR=/home/app/.config/Ultralytics \
    # Không kiểm tra cập nhật/tải font từ mạng khi chạy.
    YOLO_OFFLINE=true

WORKDIR /app

# - libgl1 / libglib2.0-0t64: opencv-python (phụ thuộc của ultralytics).
#   Trixie đổi tên gói glib sang hậu tố `t64` — `libglib2.0-0` không còn tồn tại.
# - libgomp1: OpenMP cho torch
# - libpq5: psycopg2-binary
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0t64 \
        libgomp1 \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 1) torch/torchvision bản CPU trước, để bước sau không kéo bản CUDA.
# 2) Phần còn lại từ PyPI. `==2.14.1` khớp `2.14.1+cpu` theo PEP 440 nên không cài lại.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 \
        torchvision==0.29.1 \
    && pip install -r requirements.txt

# Chỉ copy những gì runtime cần (test/dev nằm trong .dockerignore).
COPY app ./app
COPY scripts ./scripts
COPY artifacts ./artifacts

# Chạy bằng user thường; artifacts/data là chỉ-đọc nên không cần quyền ghi.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p "$YOLO_CONFIG_DIR" \
    && chown -R app:app "$YOLO_CONFIG_DIR"
USER app

EXPOSE 8000

# Dùng /health/ready để orchestrator biết model đã nạp xong (không chỉ tiến trình sống).
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:get_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
