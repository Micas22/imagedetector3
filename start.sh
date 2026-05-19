#!/bin/bash
set -e

# The SQLite DB lives in /app/data so it survives container restarts via a volume.
# Symlink it from the working directory so the app code finds it at its default path.
if [ ! -L /app/.crawler.db ]; then
    ln -sf /app/data/.crawler.db /app/.crawler.db
fi

echo "==> Starting FastAPI on :8000"
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "==> Starting Streamlit on :8501"
streamlit run webapp.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

echo "==> Both services started (API PID=$API_PID, Streamlit PID=$STREAMLIT_PID)"

# Exit if either process dies
wait -n
echo "==> A service exited — shutting down"
kill $API_PID $STREAMLIT_PID 2>/dev/null || true
