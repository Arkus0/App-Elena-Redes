#!/bin/bash
# =============================================================================
# PyAV (av) Dependencies Installation Script for GitHub Codespaces
# =============================================================================
#
# This script installs all required system dependencies to build PyAV from
# source on Ubuntu/Debian-based systems (including GitHub Codespaces).
#
# Usage:
#   chmod +x install_pyav_deps.sh
#   ./install_pyav_deps.sh
#
# Then run:
#   pip install av
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "PyAV Dependencies Installer"
echo "=============================================="
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VERSION=$VERSION_ID
    echo "Detected OS: $OS $VERSION"
else
    echo "Warning: Cannot detect OS, assuming Ubuntu/Debian"
    OS="Ubuntu"
fi

echo ""
echo "[1/4] Updating package lists..."
sudo apt-get update -qq

echo ""
echo "[2/4] Installing FFmpeg development libraries..."
# Core FFmpeg libraries required by PyAV
sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libavdevice-dev

echo ""
echo "[3/4] Installing build tools and Python development headers..."
sudo apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3-dev \
    python3-pip

echo ""
echo "[4/4] Installing additional codec libraries (optional but recommended)..."
sudo apt-get install -y --no-install-recommends \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libopus-dev \
    libmp3lame-dev \
    libfdk-aac-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libtheora-dev \
    libvorbis-dev \
    || echo "Some optional codecs not available, continuing..."

echo ""
echo "=============================================="
echo "Dependencies installed successfully!"
echo "=============================================="
echo ""

# Verify FFmpeg installation
echo "Verifying FFmpeg installation..."
ffmpeg -version | head -n 1
echo ""

# Check pkg-config can find libav*
echo "Checking pkg-config for libavcodec..."
if pkg-config --exists libavcodec; then
    echo "  libavcodec: $(pkg-config --modversion libavcodec)"
else
    echo "  WARNING: pkg-config cannot find libavcodec"
fi

if pkg-config --exists libavformat; then
    echo "  libavformat: $(pkg-config --modversion libavformat)"
fi

if pkg-config --exists libavutil; then
    echo "  libavutil: $(pkg-config --modversion libavutil)"
fi

echo ""
echo "=============================================="
echo "Ready to install PyAV!"
echo "=============================================="
echo ""
echo "Run one of the following commands:"
echo ""
echo "  Option 1 (recommended): Install pre-built wheel if available"
echo "    pip install av --only-binary=:all:"
echo ""
echo "  Option 2: Build from source (uses installed FFmpeg)"
echo "    pip install av --no-binary av"
echo ""
echo "  Option 3: Install specific version with known compatibility"
echo "    pip install av==10.0.0"
echo ""
