# Sanash Edge Deployment

Deploys the Sanash inference pipeline on an NVIDIA Jetson Nano mounted inside an Almaty bus.

---

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| SBC | NVIDIA Jetson Nano 4GB (B01) |
| Camera | USB wide-angle or Raspberry Pi v2 CSI camera |
| Connectivity | 4G/LTE USB modem or built-in WiFi |
| Power | 5V 4A barrel jack (recommended over USB) |
| Storage | 32 GB microSD (Class 10 or faster) |

---

## Software Stack

| Layer | Version |
|-------|---------|
| JetPack | 5.1 (Ubuntu 20.04, L4T 35.x) |
| TensorRT | 8.5+ (included with JetPack) |
| Python | 3.8 (system) |
| PyTorch | 2.0 for JetPack 5.1 |

---

## Setup Steps

### 1. Flash JetPack
Download and flash JetPack 5.1 from [developer.nvidia.com](https://developer.nvidia.com/jetson-nano-sd-card-image).

### 2. Run Setup Script
```bash
git clone <repo-url> sanash
cd sanash
bash edge_deployment/jetson_setup.sh
```

### 3. Convert Model
```bash
python3 edge_deployment/convert_to_tensorrt.py \
    --model models/csrnet/checkpoint.pth \
    --output /opt/sanash/models/csrnet_fp16.trt \
    --precision fp16
```

### 4. Configure
```bash
cp edge_deployment/config.yaml /opt/sanash/config.yaml
nano /opt/sanash/config.yaml
# Set: api.endpoint, inference.bus_id, inference.route_id
```

### 5. Set API Token
```bash
sudo nano /etc/systemd/system/sanash-inference.service
# Replace REPLACE_WITH_REAL_TOKEN with actual token
sudo systemctl daemon-reload
```

### 6. Start Service
```bash
sudo systemctl start sanash-inference
sudo systemctl status sanash-inference
```

---

## How It Works

```
Every 30 seconds:
  1. Capture frame from camera (1280×720)
  2. Preprocess: resize to 512×384, normalize (ImageNet stats)
  3. CSRNet TensorRT inference: ~15 ms (FP16 on Jetson GPU)
  4. Sum density map → integer passenger count
  5. POST to FastAPI: {"count": 42, "bus_id": "BUS_001", ...}
  6. If network unavailable → buffer in local SQLite → sync on reconnect
```

---

## File Reference

| File | Purpose |
|------|---------|
| `config.yaml` | All runtime configuration |
| `jetson_setup.sh` | One-time setup script (JetPack deps, systemd service) |
| `convert_to_tensorrt.py` | Convert PyTorch/ONNX model → TensorRT engine |
| `inference_pipeline.py` | Main loop: capture → infer → upload |

---

## Performance

| Metric | Value |
|--------|-------|
| Inference latency (FP16 TRT) | ~15 ms |
| Upload latency (4G LTE) | ~200–500 ms |
| Cycle interval | 30 seconds (configurable) |
| Power draw | ~8 W (camera + GPU inference) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `TensorRT not available` | Run on Jetson Nano with JetPack installed |
| Camera not opening | Check `ls /dev/video*`, set correct `device_id` in config.yaml |
| API upload fails | Check `api.endpoint` and `SANASH_API_TOKEN` env var |
| High memory usage | Reduce `input_height`/`input_width` in config.yaml |
| Service crashes on start | `journalctl -u sanash-inference -n 50` for error details |

---

## Monitoring

```bash
# Live logs
journalctl -u sanash-inference -f

# Check if uploading
tail -f /var/log/sanash/inference.log

# Check offline buffer
sqlite3 /opt/sanash/offline_buffer.db "SELECT count(*) FROM pending_counts WHERE synced=0;"
```
