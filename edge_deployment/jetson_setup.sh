#!/usr/bin/env bash
# =============================================================================
# Sanash Jetson Nano Setup Script
# Tested on JetPack 5.1 (Ubuntu 20.04, L4T 35.x)
# =============================================================================
set -euo pipefail

echo "============================================"
echo " Sanash Edge Deployment — Jetson Nano Setup"
echo "============================================"

# ----- System update ---------------------------------------------------------
echo "[1/7] Updating system packages..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq

# ----- System dependencies ---------------------------------------------------
echo "[2/7] Installing system dependencies..."
sudo apt-get install -y -qq \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    libpng-dev \
    liblapack-dev \
    gfortran \
    curl \
    git \
    sqlite3 \
    libsqlite3-dev

# ----- Python packages -------------------------------------------------------
echo "[3/7] Installing Python packages..."
pip3 install --upgrade pip --quiet

# PyTorch for Jetson (JetPack 5.1 compatible)
echo "  Installing PyTorch for JetPack 5.1..."
pip3 install --quiet \
    "https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl" \
    || echo "  WARNING: JetPack PyTorch wheel not found. Install manually from developer.nvidia.com"

pip3 install --quiet \
    numpy>=1.24 \
    opencv-python-headless>=4.7 \
    requests>=2.31 \
    pyyaml>=6.0 \
    scipy>=1.10 \
    pillow>=9.5

# ----- TensorRT (comes with JetPack, just install Python bindings) -----------
echo "[4/7] Verifying TensorRT installation..."
python3 -c "import tensorrt; print('  TensorRT version:', tensorrt.__version__)" \
    || echo "  WARNING: TensorRT Python bindings not found. Ensure JetPack is fully installed."

# ----- Create directories ----------------------------------------------------
echo "[5/7] Creating Sanash directories..."
sudo mkdir -p /opt/sanash/models
sudo mkdir -p /var/log/sanash
sudo chown -R "$USER":"$USER" /opt/sanash /var/log/sanash
chmod 755 /opt/sanash /var/log/sanash

# Copy project files if running from project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/inference_pipeline.py" ]; then
    cp "$SCRIPT_DIR/inference_pipeline.py" /opt/sanash/
    cp "$SCRIPT_DIR/config.yaml" /opt/sanash/
    echo "  Copied inference_pipeline.py and config.yaml to /opt/sanash/"
fi

# ----- Systemd service -------------------------------------------------------
echo "[6/7] Installing systemd service..."
sudo tee /etc/systemd/system/sanash-inference.service > /dev/null << 'SERVICE'
[Unit]
Description=Sanash Bus Occupancy Inference Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=/opt/sanash
ExecStart=/usr/bin/python3 /opt/sanash/inference_pipeline.py --config /opt/sanash/config.yaml
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
Environment=SANASH_API_TOKEN=REPLACE_WITH_REAL_TOKEN
Environment=BUS_ID=BUS_001

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable sanash-inference.service
echo "  Service registered (not started — configure token first)"

# ----- Completion message ----------------------------------------------------
echo ""
echo "[7/7] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Convert your model:  python3 /opt/sanash/convert_to_tensorrt.py --model <path>"
echo "  2. Edit token:          sudo nano /etc/systemd/system/sanash-inference.service"
echo "     Replace REPLACE_WITH_REAL_TOKEN with your actual API token"
echo "  3. Edit config:         nano /opt/sanash/config.yaml"
echo "     Set bus_id, route_id, and api.endpoint"
echo "  4. Start service:       sudo systemctl start sanash-inference"
echo "  5. Check logs:          journalctl -u sanash-inference -f"
echo ""
echo "Model directory:   /opt/sanash/models/"
echo "Log directory:     /var/log/sanash/"
echo "============================================"
