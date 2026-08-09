from __future__ import annotations

import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .p2pnet_model import build_p2pnet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_EXTENSIONS = {".onnx", ".pth", ".pt", ".ckpt", ".zip"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def runtime_roots(start_dir: Path) -> list[Path]:
    roots = [start_dir]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        roots.append(Path(bundle_dir))
    executable_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None
    if executable_dir is not None:
        roots.append(executable_dir)
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique_roots and resolved.exists():
            unique_roots.append(resolved)
    return unique_roots


@dataclass
class PredictionResult:
    image_path: Path
    original_image: Image.Image
    annotated_image: Image.Image
    points: np.ndarray
    scores: np.ndarray | None
    count: int
    inference_ms: float
    threshold: float
    backend: str
    diagnostics: str


@dataclass
class ModelSummary:
    path: Path
    resolved_path: Path
    backend: str
    device: str
    details: str


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def list_images(folder: Path) -> list[Path]:
    images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if images:
        return images
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def find_default_model(start_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for root in runtime_roots(start_dir):
        for pattern in (
            "sanash_p2pnet_artifacts.zip",
            "*p2pnet*.onnx",
            "*p2pnet*.pth",
            "*best_mae*.pth",
            "*.onnx",
            "*.pth",
        ):
            candidates.extend(root.glob(pattern))
    if not candidates:
        for root in runtime_roots(start_dir):
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                    candidates.append(path)
    candidates = [path for path in candidates if path.name.lower() != "yolov8n.pt"]
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, str]:
        name = path.name.lower()
        value = 0
        if "sanash" in name:
            value -= 50
        if "p2pnet" in name:
            value -= 40
        if "best" in name or "best_mae" in name:
            value -= 30
        if path.suffix.lower() == ".zip":
            value -= 20
        if path.suffix.lower() == ".onnx":
            value -= 10
        return value, str(path)

    return sorted(set(candidates), key=score)[0]


def find_default_image_source(start_dir: Path) -> Path | None:
    preferred = []
    for root in runtime_roots(start_dir):
        preferred.extend(
            [
                root / "demo_samples",
                root / "p2pnet_almaty_dataset_block_stratified" / "images" / "val",
                root / "prepared_cvat_subset" / "images" / "nonempty",
                root / "frames_with_people",
            ]
        )
    for folder in preferred:
        if folder.exists() and folder.is_dir() and list_images(folder):
            return folder
    for root in runtime_roots(start_dir):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                return path.parent
    return None


def resolve_model_file(model_path: Path, cache_dir: Path | None = None) -> Path:
    path = model_path.expanduser().resolve()
    if path.suffix.lower() != ".zip":
        return path
    if cache_dir is None:
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            cache_dir = Path(local_app_data) / "SanashPassengerCounter" / "models"
        else:
            cache_dir = Path.cwd() / ".sanash_demo_cache" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).suffix.lower() in {".onnx", ".pth", ".pt", ".ckpt"} and not name.endswith("/")
        ]
        if not members:
            raise ValueError(f"No model file (.onnx/.pth/.pt/.ckpt) was found inside {path.name}.")

        def member_score(name: str) -> tuple[int, str]:
            lower = name.lower()
            value = 0
            if "best_mae" in lower or "best" in lower:
                value -= 50
            if lower.endswith(".onnx"):
                value -= 20
            if "latest" in lower:
                value -= 10
            return value, lower

        member = sorted(members, key=member_score)[0]
        output_dir = cache_dir / path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(member).name
        info = archive.getinfo(member)
        if not output_path.exists() or output_path.stat().st_size != info.file_size:
            with archive.open(member) as src, output_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        return output_path.resolve()


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _clean_state_dict(value)
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return _clean_state_dict(checkpoint)
    raise ValueError("Could not find a PyTorch state dict in the selected model file.")


def _infer_row_line(state_dict: dict[str, torch.Tensor]) -> tuple[int, int]:
    weight = state_dict.get("classification.output.weight")
    if weight is None:
        return 2, 2
    num_classes = 2
    num_anchor_points = int(weight.shape[0]) // num_classes
    side = int(math.sqrt(num_anchor_points))
    if side * side == num_anchor_points:
        return side, side
    return 1, num_anchor_points


def _is_p2pnet_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    required = {
        "backbone.body1.0.weight",
        "classification.output.weight",
        "regression.output.weight",
        "fpn.P4_1.weight",
    }
    return required.issubset(state_dict.keys())


def _preprocess_p2pnet(image: Image.Image, stride_multiple: int = 16) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    height, width = array.shape[:2]
    pad_height = (stride_multiple - height % stride_multiple) % stride_multiple
    pad_width = (stride_multiple - width % stride_multiple) % stride_multiple
    if pad_height or pad_width:
        array = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def _softmax_class_one(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if logits.ndim == 1:
        return logits.sigmoid()
    if logits.shape[-1] == 1:
        return logits[..., 0].sigmoid()
    return logits.softmax(dim=-1)[..., 1]


def _filter_points(
    points: np.ndarray,
    scores: np.ndarray | None,
    width: int,
    height: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    if points.size == 0:
        return points.reshape(0, 2).astype(np.float32), scores
    points = points.reshape(-1, 2).astype(np.float32)
    keep = np.ones((points.shape[0],), dtype=bool)
    if scores is not None:
        scores = scores.reshape(-1).astype(np.float32)
        if scores.shape[0] == points.shape[0]:
            keep &= scores >= threshold
    keep &= points[:, 0] >= 0
    keep &= points[:, 1] >= 0
    keep &= points[:, 0] < float(width)
    keep &= points[:, 1] < float(height)
    filtered_scores = scores[keep] if scores is not None and scores.shape[0] == points.shape[0] else None
    return points[keep], filtered_scores


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "arialbd.ttf",
        "Arial Bold.ttf",
        "arial.ttf",
        "calibrib.ttf",
        "calibri.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    )
    for name in candidates if bold else candidates[2:]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    trimmed = text
    while trimmed and draw.textbbox((0, 0), f"{trimmed}...", font=font)[2] > max_width:
        trimmed = trimmed[:-1]
    return f"{trimmed.rstrip()}..." if trimmed else "..."


def draw_annotated_image(
    image: Image.Image,
    points: np.ndarray,
    count: int,
    threshold: float,
    inference_ms: float,
    image_name: str,
) -> Image.Image:
    annotated = image.convert("RGBA")
    draw = ImageDraw.Draw(annotated, "RGBA")
    width, height = annotated.size
    radius = max(6, min(width, height) // 115)
    outline_width = max(2, radius // 3)

    for x_float, y_float in points:
        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        shadow = (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2)
        marker = (x - radius, y - radius, x + radius, y + radius)
        center = max(2, radius // 3)
        draw.ellipse(shadow, fill=(0, 0, 0, 92))
        draw.ellipse(marker, fill=(15, 159, 143, 228), outline=(255, 255, 255, 245), width=outline_width)
        draw.ellipse((x - center, y - center, x + center, y + center), fill=(255, 255, 255, 230))

    title_font = _load_font(max(24, min(width, height) // 24), bold=True)
    meta_font = _load_font(max(14, min(width, height) // 55))
    label_font = _load_font(max(12, min(width, height) // 72), bold=True)
    count_text = f"{count}"
    label_text = "PASSENGERS"
    meta_text = f"{image_name} / {inference_ms:.0f} ms / threshold {threshold:.2f}"
    margin = max(14, min(width, height) // 55)
    padding_x = max(16, min(width, height) // 42)
    padding_y = max(12, min(width, height) // 54)
    max_panel_width = max(260, width - margin * 2)
    meta_text = _fit_text(draw, meta_text, meta_font, max_panel_width - padding_x * 2)
    count_box = draw.textbbox((0, 0), count_text, font=title_font)
    label_box = draw.textbbox((0, 0), label_text, font=label_font)
    meta_box = draw.textbbox((0, 0), meta_text, font=meta_font)
    panel_width = max(
        count_box[2] - count_box[0] + label_box[2] - label_box[0] + padding_x * 3,
        meta_box[2] - meta_box[0] + padding_x * 2,
    )
    panel_width = min(panel_width, width - margin * 2)
    panel_height = count_box[3] - count_box[1] + meta_box[3] - meta_box[1] + padding_y * 3
    panel = (margin, margin, margin + panel_width, margin + panel_height)
    draw.rounded_rectangle(panel, radius=max(10, margin // 2), fill=(11, 24, 22, 206))
    draw.rounded_rectangle(panel, radius=max(10, margin // 2), outline=(255, 255, 255, 78), width=1)
    draw.rectangle((margin, margin, margin + 5, margin + panel_height), fill=(15, 159, 143, 240))
    count_x = margin + padding_x
    count_y = margin + padding_y - 3
    label_x = count_x + count_box[2] - count_box[0] + max(12, padding_x // 2)
    label_y = count_y + max(4, (count_box[3] - count_box[1] - (label_box[3] - label_box[1])) // 2)
    meta_y = margin + padding_y * 2 + count_box[3] - count_box[1]
    draw.text((count_x, count_y), count_text, fill=(255, 255, 255, 255), font=title_font)
    draw.text((label_x, label_y), label_text, fill=(217, 243, 239, 255), font=label_font)
    draw.text((count_x, meta_y), meta_text, fill=(225, 233, 230, 255), font=meta_font)
    return annotated.convert("RGB")


class P2PNetBackend:
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        checkpoint = torch.load(model_path, map_location="cpu")
        self.state_dict = _extract_state_dict(checkpoint)
        if not _is_p2pnet_state_dict(self.state_dict):
            keys = list(self.state_dict.keys())[:20]
            raise ValueError(
                "The selected PyTorch checkpoint does not look like the Sanash P2PNet model. "
                f"First state-dict keys: {keys}"
            )
        row, line = _infer_row_line(self.state_dict)
        self.model = build_p2pnet(row=row, line=line)
        missing, unexpected = self.model.load_state_dict(self.state_dict, strict=False)
        important_missing = [key for key in missing if not key.endswith("num_batches_tracked")]
        if important_missing or unexpected:
            raise ValueError(
                "Could not cleanly load the P2PNet checkpoint. "
                f"Missing keys: {important_missing[:20]}; unexpected keys: {unexpected[:20]}"
            )
        self.model.to(self.device)
        self.model.eval()
        self.row = row
        self.line = line

    def predict(self, image: Image.Image, threshold: float) -> tuple[np.ndarray, np.ndarray, float, str]:
        tensor = _preprocess_p2pnet(image).to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(tensor)
            scores = _softmax_class_one(outputs["pred_logits"][0]).detach().cpu().numpy()
            points = outputs["pred_points"][0].detach().cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        width, height = image.size
        points, scores = _filter_points(points, scores, width, height, threshold)
        diagnostics = (
            f"P2PNet PyTorch checkpoint | device={self.device} | anchors={self.row}x{self.line} | "
            f"pred_logits=[1,N,2] pred_points=[1,N,2]"
        )
        return points, scores if scores is not None else np.array([], dtype=np.float32), elapsed_ms, diagnostics

    def summary(self) -> str:
        params = sum(param.numel() for param in self.model.parameters())
        return (
            f"PyTorch P2PNet\n"
            f"Path: {self.model_path}\n"
            f"Device: {self.device}\n"
            f"Parameters: {params:,}\n"
            f"Backbone: VGG16-BN\n"
            f"Anchor grid: row={self.row}, line={self.line}\n"
            f"Input: RGB tensor [1, 3, H, W], ImageNet normalized\n"
            f"Outputs: pred_logits [1, N, 2], pred_points [1, N, 2]"
        )


class OnnxBackend:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX model selected, but onnxruntime is not installed.") from exc
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()

    def _target_size(self, image: Image.Image) -> tuple[Image.Image, tuple[float, float]]:
        shape = self.input_meta.shape
        if len(shape) != 4:
            raise ValueError(f"Expected ONNX image input rank 4, got input shape {shape}.")
        height_dim, width_dim = shape[2], shape[3]
        if isinstance(height_dim, int) and isinstance(width_dim, int):
            original_width, original_height = image.size
            resized = image.resize((width_dim, height_dim), Image.Resampling.BILINEAR)
            return resized, (original_width / width_dim, original_height / height_dim)
        return image, (1.0, 1.0)

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
        resized, scale = self._target_size(image)
        array = np.asarray(resized).astype(np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD
        array = np.transpose(array, (2, 0, 1))[None, ...]
        return array.astype(np.float32), scale, resized.size

    def _extract(self, arrays: list[np.ndarray], names: list[str], threshold: float, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None, str]:
        points_index = next((idx for idx, name in enumerate(names) if "point" in name.lower() or "coord" in name.lower()), None)
        scores_index = next(
            (idx for idx, name in enumerate(names) if "logit" in name.lower() or "score" in name.lower() or "class" in name.lower()),
            None,
        )
        if points_index is None:
            candidates = [idx for idx, array in enumerate(arrays) if array.ndim >= 2 and array.shape[-1] == 2]
            if len(candidates) == 1:
                points_index = candidates[0]
            elif len(candidates) >= 2:
                spread = [float(np.nanmax(np.abs(arrays[idx]))) for idx in candidates]
                best = int(np.argmax(spread))
                if spread[best] > 2.0:
                    points_index = candidates[best]
                    scores_index = next((idx for idx in candidates if idx != points_index), scores_index)
            if points_index is None:
                diagnostics = "; ".join(f"{name}: shape={array.shape}" for name, array in zip(names, arrays))
                raise ValueError(
                    "Could not identify point coordinates in ONNX outputs. "
                    f"Outputs were: {diagnostics}"
                )

        points = arrays[points_index]
        if points.ndim >= 3:
            points = points[0]
        points = points.reshape(-1, 2).astype(np.float32)

        scores = None
        if scores_index is not None and scores_index != points_index:
            score_array = arrays[scores_index]
            if score_array.ndim >= 3:
                score_array = score_array[0]
            if score_array.shape[-1] == 2:
                logits = score_array.astype(np.float32)
                logits = logits - logits.max(axis=-1, keepdims=True)
                probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
                scores = probs[..., 1].reshape(-1)
            elif score_array.shape[-1] == 1:
                logits = score_array.reshape(-1).astype(np.float32)
                scores = 1.0 / (1.0 + np.exp(-logits))
            else:
                scores = score_array.reshape(-1).astype(np.float32)

        model_width, model_height = image_size
        if points.size and float(np.nanmax(points)) <= 1.5:
            points[:, 0] *= float(model_width)
            points[:, 1] *= float(model_height)

        diagnostics = "; ".join(f"{name}: shape={array.shape}" for name, array in zip(names, arrays))
        return points, scores, diagnostics

    def predict(self, image: Image.Image, threshold: float) -> tuple[np.ndarray, np.ndarray, float, str]:
        model_input, scale, model_size = self._preprocess(image)
        start = time.perf_counter()
        raw_outputs = self.session.run(None, {self.input_meta.name: model_input})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        names = [output.name for output in self.output_meta]
        points, scores, diagnostics = self._extract(raw_outputs, names, threshold, model_size)
        points[:, 0] *= scale[0]
        points[:, 1] *= scale[1]
        points, scores = _filter_points(points, scores, image.width, image.height, threshold)
        return points, scores if scores is not None else np.array([], dtype=np.float32), elapsed_ms, diagnostics

    def summary(self) -> str:
        inputs = "\n".join(f"  {item.name}: {item.shape} {item.type}" for item in self.session.get_inputs())
        outputs = "\n".join(f"  {item.name}: {item.shape} {item.type}" for item in self.session.get_outputs())
        return f"ONNX Runtime\nPath: {self.model_path}\nInputs:\n{inputs}\nOutputs:\n{outputs}"


class PassengerCounter:
    def __init__(self, device: str = "auto") -> None:
        self.device = device
        self.backend: P2PNetBackend | OnnxBackend | None = None
        self.summary: ModelSummary | None = None

    def load_model(self, model_path: Path) -> ModelSummary:
        resolved = resolve_model_file(model_path)
        suffix = resolved.suffix.lower()
        if suffix == ".onnx":
            backend: P2PNetBackend | OnnxBackend = OnnxBackend(resolved)
            backend_name = "onnxruntime"
            device = "cpu"
        elif suffix in {".pth", ".pt", ".ckpt"}:
            backend = P2PNetBackend(resolved, device=self.device)
            backend_name = "pytorch-p2pnet"
            device = backend.device
        else:
            raise ValueError(f"Unsupported model extension: {resolved.suffix}")
        self.backend = backend
        self.summary = ModelSummary(
            path=model_path,
            resolved_path=resolved,
            backend=backend_name,
            device=device,
            details=backend.summary(),
        )
        return self.summary

    def predict_image(self, image_path: Path, threshold: float = 0.5) -> PredictionResult:
        if self.backend is None:
            raise RuntimeError("Load a model before running inference.")
        image = load_rgb_image(image_path)
        points, scores, elapsed_ms, diagnostics = self.backend.predict(image, threshold)
        annotated = draw_annotated_image(image, points, len(points), threshold, elapsed_ms, image_path.name)
        backend_name = self.summary.backend if self.summary else type(self.backend).__name__
        return PredictionResult(
            image_path=image_path,
            original_image=image,
            annotated_image=annotated,
            points=points,
            scores=scores,
            count=len(points),
            inference_ms=elapsed_ms,
            threshold=threshold,
            backend=backend_name,
            diagnostics=diagnostics,
        )
