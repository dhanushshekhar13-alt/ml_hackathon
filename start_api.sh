#!/bin/bash
# Start the PathSnap API on Laptop 1 so Laptop 2 can reach it over the local network.
# Usage:  bash start_api.sh            (serves the student_test trips)
#         PATHSNAP_EVAL_DIR="/path/to/evaluator_trips" bash start_api.sh
cd "$(dirname "$0")"
source .venv/bin/activate
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
echo "=============================================================="
echo " Laptop 2 API URL:  http://${IP}:8000"
echo "=============================================================="
uvicorn api.main:app --host 0.0.0.0 --port 8000
