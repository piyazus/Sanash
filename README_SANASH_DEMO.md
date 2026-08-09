# Sanash Passenger Counting Demo

Local Windows desktop demo for the Sanash P2PNet passenger counting model.

The app runs fully on the local machine. It does not train a model, call cloud APIs, or upload images.

## What It Does

- Loads a Sanash P2PNet PyTorch checkpoint from `sanash_p2pnet_artifacts.zip` or a direct `.pth` file.
- Also supports ONNX models when `onnxruntime` is installed.
- Opens one image or a folder of images.
- Runs passenger-count inference.
- Shows the original image, annotated predicted passenger points, count, inference time, threshold, and backend.
- Saves the annotated image.
- Includes a CLI model inspection tool for input/output shape diagnostics.

## Files Added

- `run_demo.py` - desktop app entry point.
- `sanash_demo/` - local inference and Tkinter UI package.
- `requirements-demo.txt` - runtime and packaging dependencies.

The current project already contains:

- `sanash_p2pnet_artifacts.zip` - trained checkpoint archive.
- `p2pnet_almaty_dataset_block_stratified/images/val` - sample validation images.

## Setup On Windows

From PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-demo.txt
```

If PyTorch installation needs a machine-specific CUDA build, install PyTorch from the official PyTorch selector first, then run:

```powershell
pip install numpy Pillow onnxruntime pyinstaller
```

## Run The App

```powershell
python run_demo.py
```

Workflow:

1. Confirm or browse to the model file. The default should be `sanash_p2pnet_artifacts.zip`.
2. Click `Load` or just click `Run Inference`; the app will load the model first.
3. Open one image or a folder of images.
4. Click `Run Inference`.
5. Use `Previous` and `Next` for folder browsing.
6. Click `Save Annotated` to export the result image.

## Inspect Model Shapes

Print model input/output information:

```powershell
python -m sanash_demo.inspect_model --model sanash_p2pnet_artifacts.zip
```

Run inspection plus a sample inference:

```powershell
python -m sanash_demo.inspect_model `
  --model sanash_p2pnet_artifacts.zip `
  --image p2pnet_almaty_dataset_block_stratified\images\val\D02_20260420073459_001231.jpg
```

For the bundled PyTorch checkpoint, expected shape behavior is:

- Input: RGB tensor `[1, 3, H, W]`, ImageNet normalized.
- Output: `pred_logits [1, N, 2]` and `pred_points [1, N, 2]`.
- Count: number of predicted points whose passenger confidence is above the selected threshold and inside image bounds.

If a selected model has an ambiguous output format, the app and CLI print output names/shapes to help diagnose the export.

## Measure Accuracy / MAE

If ground-truth `.npy` point files are available, evaluate the model and tune the confidence threshold:

```powershell
python -m sanash_demo.evaluate_model `
  --model sanash_p2pnet_artifacts.zip `
  --root p2pnet_almaty_dataset_block_stratified `
  --split val `
  --limit 20
```

The app default threshold is `0.40`, a practical middle point from quick validation checks. Use the evaluator on the full validation set before presenting final accuracy numbers.

## Package As A Windows EXE

From an activated virtual environment:

```powershell
.\package_windows.ps1 -InstallDeps
```

This creates:

```powershell
dist\SanashPassengerCounter-Windows.zip
```

Send that ZIP to another Windows user. They should unzip it and run:

```powershell
SanashPassengerCounter.exe
```

They do not need to install Python or the Python dependencies. The ZIP includes the app, the trained model archive, and a few demo sample images.

If you need exactly one downloadable file that runs by itself, build the one-file executable:

```powershell
.\package_windows.ps1 -OneFile
```

This creates:

```powershell
dist\SanashPassengerCounter-OneFile.exe
```

This single `.exe` includes the app, model archive, and demo samples. It is large because PyTorch is bundled inside it, and the first launch may take longer while Windows extracts the bundled libraries to a temporary folder.

Manual PyInstaller command:

```powershell
pyinstaller --noconfirm --clean --onedir --windowed `
  --name SanashPassengerCounter `
  --collect-all torch `
  --collect-all PIL `
  run_demo.py
```

The executable will be created at:

```powershell
dist\SanashPassengerCounter\SanashPassengerCounter.exe
```

Keep the model archive or checkpoint available next to the project/demo files, or browse to it from the app.

## Notes

- The PyTorch checkpoint inside `sanash_p2pnet_artifacts.zip` is extracted once to `.sanash_demo_cache\models`.
- CPU inference is supported. CUDA is used automatically when PyTorch detects a compatible GPU.
- This is an inference/demo application only; no training code is run.
