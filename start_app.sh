#!/bin/bash

echo "🚀 Starting Quantum Portfolio Coach..."

# Start Backend
echo "Starting FastAPI Backend on port 8000..."
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to be ready (naive check)
sleep 3

# Start Frontend
echo "Starting Vite + React Frontend on port 3000..."
cd web

# Check if node_modules exists, if not install
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies (this may take a minute)..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!

# Cleanup function
cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    exit
}

trap cleanup SIGINT

# Keep script running
wait
