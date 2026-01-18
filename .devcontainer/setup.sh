#!/bin/bash
# =============================================================================
# Codespace/DevContainer Setup Script for BrandPulse AI
# =============================================================================
# This script runs automatically when the container is created.
# It installs FFmpeg dependencies needed for PyAV and other video processing.
# =============================================================================

set -e  # Exit on error

echo ""
echo "=============================================="
echo "  BrandPulse AI - Codespace Setup"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install FFmpeg and development libraries for PyAV
# -----------------------------------------------------------------------------
echo "[1/4] Installing FFmpeg dependencies for video processing..."
sudo apt-get update -qq

# Core FFmpeg libraries required by PyAV
sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    pkg-config

echo "     FFmpeg version: $(ffmpeg -version | head -n 1)"

# -----------------------------------------------------------------------------
# Step 2: Install Python dependencies (backend)
# -----------------------------------------------------------------------------
echo ""
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip

# Install requirements, with retry for PyAV if needed
if pip install -r backend/requirements.txt; then
    echo "     Python dependencies installed successfully"
else
    echo "     Retrying PyAV installation..."
    pip install av==10.0.0 --no-binary av
    pip install -r backend/requirements.txt
fi

# -----------------------------------------------------------------------------
# Step 3: Install Node.js dependencies (frontend)
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] Installing Node.js dependencies..."
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    npm install --prefix frontend
    echo "     Node.js dependencies installed"
else
    echo "     Skipping frontend (no package.json found)"
fi

# -----------------------------------------------------------------------------
# Step 4: Verify installation
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] Verifying installation..."

# Check Python packages
python -c "import av; print(f'     PyAV: {av.__version__}')" 2>/dev/null || echo "     PyAV: NOT INSTALLED (optional)"
python -c "import cv2; print(f'     OpenCV: {cv2.__version__}')" 2>/dev/null || echo "     OpenCV: NOT INSTALLED"
python -c "import fastapi; print(f'     FastAPI: {fastapi.__version__}')" 2>/dev/null || echo "     FastAPI: NOT INSTALLED"

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start the backend:"
echo "  cd backend && uvicorn app.main:app --reload"
echo ""
echo "To start the frontend:"
echo "  cd frontend && npm run dev"
echo ""
