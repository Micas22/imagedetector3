# ── Base image ──────────────────────────────────────────────────────────────
# Use slim Debian so apt packages are available for Playwright / OpenCV deps.
FROM python:3.11-slim

# Prevents Python from writing .pyc files and ensures stdout/stderr are unbuffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HuggingFace model cache inside the container (mapped to a volume).
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Suppress PaddleOCR MKLDNN on Linux/CPU to avoid crashes.
    FLAGS_use_mkldnn=0

WORKDIR /app

# ── System packages ──────────────────────────────────────────────────────────
# libgl1 / libglib2.0 needed by OpenCV (headless build still requires GLib).
# Other libs are required by Playwright Chromium.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libglib2.0-dev \
        libgomp1 \
        # Playwright Chromium system deps
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libasound2 \
        libpango-1.0-0 \
        libcairo2 \
        # General utilities
        curl \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .

# Install PyTorch CPU-only wheel first (much smaller than GPU), then the rest.
# If you want GPU support, remove the --index-url line and ensure the base image has CUDA.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Playwright Chromium ──────────────────────────────────────────────────────
RUN playwright install chromium && playwright install-deps chromium

# ── Application source ───────────────────────────────────────────────────────
COPY *.py ./

# Create the data directory (mapped to a volume at runtime).
RUN mkdir -p /app/data /app/.cache/huggingface

# ── Ports ────────────────────────────────────────────────────────────────────
# 8501 = Streamlit   |   8000 = FastAPI
EXPOSE 8501 8000

# ── Start script ─────────────────────────────────────────────────────────────
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
