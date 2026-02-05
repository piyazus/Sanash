# Comparative Analysis: YOLOv8 vs CSRNet vs P2PNet for Crowd Counting

## Model Overview
- **YOLOv8**: Object detection model (Anchor-free). Adapted by generating pseudo-bounding boxes.
- **CSRNet**: Density regulation model. Uses VGG16 frontend and dilated convolution backend.
- **P2PNet**: Point-based prediction using Hungarian matching.

## Performance Comparison (Preliminary)

| Model | MAE (Validation) | MSE (Validation) | Inference FPS (GPU) | Model Size (MB) |
|-------|------------------|------------------|---------------------|-----------------|
| YOLOv8 | TBD | TBD | TBD | ~6.2 (Nano) |
| CSRNet | TBD | TBD | TBD | 64 |
| P2PNet | TBD | TBD | TBD | 150+ |

## Observations
### YOLOv8
- **Strength**: Fast inference, widely supported.
- **Weakness**: Struggles with high density where bounding boxes overlap significantly. Requires box annotations (simulated here).

### CSRNet
- **Strength**: Excellent for high density (Part A). Handles occlusion well via density map.
- **Weakness**: Spatial resolution of output is 1/8th of input. Exact localization is harder.

### P2PNet
- **Strength**: Precise localization (1:1 points). Robust to scale variation.
- **Weakness**: Training stability can be tricky. Heavier than YOLO.

## Recommendations for Almaty Bus Project
For bus interiors:
- **Scenario**: Moderate density (10-50 people), potential severe occlusion, camera angle from top.
- **Recommendation**: 
  - **YOLOv8** if real-time edge processing is critical (<30ms) and density is low.
  - **P2PNet** for best balance of accuracy and localization if hardware allows.
  - **CSRNet** if exact count is more important than location, but bus interiors might not be "dense" enough to justify it over detection.

**Selected Baseline**: Start with **YOLOv8** for the pilot due to speed and ease of validaton.
