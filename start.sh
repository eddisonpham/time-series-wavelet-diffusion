#!/bin/bash
# start.sh — Launch WaveDiff backend + frontend

set -e

# ── Colors ──────────────────────────────
AMBER='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${AMBER}▶ WaveDiff Startup${NC}"
echo "─────────────────────────────────"

# ── Backend ─────────────────────────────
echo -e "${CYAN}[1/2] Starting FastAPI backend on :8000${NC}"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo -e "${GREEN}  Backend PID: $BACKEND_PID${NC}"

# ── Frontend ────────────────────────────
echo -e "${CYAN}[2/2] Starting Vite frontend on :5173${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}  Frontend PID: $FRONTEND_PID${NC}"

echo ""
echo -e "${AMBER}✓ WaveDiff running${NC}"
echo -e "  Frontend → ${CYAN}http://localhost:5173${NC}"
echo -e "  API      → ${CYAN}http://localhost:8000${NC}"
echo -e "  API Docs → ${CYAN}http://localhost:8000/docs${NC}"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait and cleanup
trap "echo ''; echo 'Shutting down…'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait