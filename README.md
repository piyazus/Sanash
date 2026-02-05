# Sanash Crowd Counting Project

This project implements and compares three crowd counting models: YOLOv8, CSRNet, and P2PNet.

## Project Structure
- `data/`: Datasets (ShanghaiTech)
- `models/`: Model implementations
  - `yolov8/`: Ultralytics YOLOv8 wrapper
  - `csrnet/`: Density map estimation
  - `p2pnet/`: Point-based regression
- `comparison/`: Analysis results
- `inference.py`: Unified inference script

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (See individual model folders for specific needs)

2. Dataset:
   The ShanghaiTech dataset is downloaded to `data/shanghaitech`.

## Training

### YOLOv8
```bash
cd models/yolov8
python train.py
```

### CSRNet
First generate density maps:
```bash
python prepare_csrnet.py
```
Then train:
```bash
cd models/csrnet
python train.py
```

### P2PNet
```bash
cd models/p2pnet
python train.py
```

## Inference
Run inference on a video or image:
```bash
python inference.py --model yolov8 --source test_video.mp4
python inference.py --model csrnet --source test_video.mp4
python inference.py --model p2pnet --source test_video.mp4
```

## Results
See `comparison/` for performance metrics.
