#!/usr/bin/env bash
# Unified Full-Stack Startup Script for Fake Review Detection System
# Starts FastAPI backend (port 8000) and Vite React frontend (port 5173)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "  FAKE REVIEW DETECTION SYSTEM — FULL-STACK LAUNCHER      "
echo "  FastAPI Backend (8000) + React/Vite Dashboard (5173)    "
echo "=========================================================="

# 1. Activate Python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Activated Python virtual environment (venv)"
else
    echo "⚠️ venv not found; using global python3"
fi

# 2. Start FastAPI Backend in background
echo "Starting FastAPI Backend on http://localhost:8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Trap signals to ensure clean shutdown of both processes
cleanup() {
    echo ""
    echo "Shutting down Full-Stack Application..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Done. Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Give backend a moment to initialize
sleep 2

# 3. Start Frontend
echo "Starting React / Vite Frontend on http://localhost:5173..."
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo "  APPLICATION READY:                                      "
echo "  - Web Dashboard : http://localhost:5173                 "
echo "  - FastAPI Docs  : http://localhost:8000/docs            "
echo "  Press Ctrl+C to terminate both servers.                 "
echo "=========================================================="

wait $FRONTEND_PID $BACKEND_PID

