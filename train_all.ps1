
# Master training script
Write-Host "Starting Overnight Training Sequence..." -ForegroundColor Green

# 1. YOLOv8
Write-Host "--- Training YOLOv8 (1/3) ---" -ForegroundColor Cyan
Set-Location "c:\Users\User\OneDrive\Desktop\sana\models\yolov8"
python train.py
if ($LASTEXITCODE -ne 0) { Write-Host "YOLOv8 failed!" -ForegroundColor Red }

# 2. CSRNet
Write-Host "--- Training CSRNet (2/3) ---" -ForegroundColor Cyan
Set-Location "c:\Users\User\OneDrive\Desktop\sana\models\csrnet"
# Ensure density maps are ready (should be done already but safe to check/skip)
if (-not (Test-Path "../../data/shanghaitech/part_B_final/train_data/ground_truth_h5")) {
    Write-Host "Generating density maps..."
    Set-Location "c:\Users\User\OneDrive\Desktop\sana"
    python prepare_csrnet.py
    Set-Location "c:\Users\User\OneDrive\Desktop\sana\models\csrnet"
}
python train.py
if ($LASTEXITCODE -ne 0) { Write-Host "CSRNet failed!" -ForegroundColor Red }

# 3. P2PNet
Write-Host "--- Training P2PNet (3/3) ---" -ForegroundColor Cyan
Set-Location "c:\Users\User\OneDrive\Desktop\sana\models\p2pnet"
python train.py
if ($LASTEXITCODE -ne 0) { Write-Host "P2PNet failed!" -ForegroundColor Red }

Write-Host "All training jobs completed." -ForegroundColor Green
