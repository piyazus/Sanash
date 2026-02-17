import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.factory import create_model, ModelConfig, ModelKind, YOLO11Wrapper
from data_pipeline.shanghaitech import ShanghaiTechDataset


class BenchmarkRunner:
    def __init__(self, device: str | torch.device | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.results: List[Dict[str, Any]] = []
        print(f"Benchmark running on: {self.device}")

    def _get_inference_module(self, model: Any, kind: ModelKind) -> Any:
        """
        Returns the callable used for forward passes during benchmarking.

        For YOLO11Wrapper we use the underlying Ultralytics model object,
        which supports calling directly with tensors and returns Results.
        For CSRNet/P2PNet we use the model itself (nn.Module).
        """
        if kind == ModelKind.YOLO11 and isinstance(model, YOLO11Wrapper):
            return model.model
        return model

    def measure_latency(
        self,
        model: Any,
        sample_input: torch.Tensor,
        model_kind: ModelKind,
        warmup_reps: int = 10,
        reps: int = 50,
    ) -> float:
        """
        Measures pure inference latency.

        Uses CUDA events when running on GPU for high-resolution timing,
        and falls back to wall-clock timing on CPU.
        """
        is_cuda = "cuda" in self.device and torch.cuda.is_available()

        infer_module = self._get_inference_module(model, model_kind)
        if hasattr(infer_module, "eval"):
            infer_module.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_reps):
                _ = infer_module(sample_input)

        # Benchmark
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                for _ in range(reps):
                    _ = infer_module(sample_input)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms = start_event.elapsed_time(end_event)
            avg_latency_ms = total_time_ms / reps
        else:
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(reps):
                    _ = infer_module(sample_input)
            end = time.perf_counter()
            total_time_s = end - start
            avg_latency_ms = (total_time_s * 1000.0) / reps

        return avg_latency_ms

    def evaluate_accuracy(
        self,
        model: Any,
        dataloader: DataLoader,
        model_kind: ModelKind,
    ) -> tuple[float, float]:
        """
        Computes MAE and RMSE over the dataset.

        Ground truth counts are obtained by summing target maps:
        - density maps (CSRNet) or
        - point maps (P2PNet / YOLO11).

        For YOLO11, predictions are obtained by counting detections
        (number of boxes in each `Results` object).
        """
        infer_module = self._get_inference_module(model, model_kind)
        if hasattr(infer_module, "eval"):
            infer_module.eval()

        absolute_errors: list[float] = []
        squared_errors: list[float] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval {model_kind.name}"):
                images = batch["image"].to(self.device)
                # Scalar ground truth counts from maps
                gt_counts = batch["target"].sum(dim=(1, 2, 3)).cpu().numpy()

                preds = infer_module(images)

                if model_kind == ModelKind.YOLO11:
                    # Ultralytics YOLO returns a list of Results objects
                    pred_counts = [len(p.boxes) for p in preds]
                    pred_counts = np.array(pred_counts, dtype=np.float32)
                elif model_kind == ModelKind.CSRNET:
                    # Sum of density map pixels approximates crowd count
                    pred_counts = preds.sum(dim=(1, 2, 3)).cpu().numpy()
                elif model_kind == ModelKind.P2PNET:
                    # Sum over point map (confidence); could be thresholded if desired
                    pred_counts = preds.sum(dim=(1, 2, 3)).cpu().numpy()
                else:
                    raise ValueError(f"Unsupported model kind for evaluation: {model_kind}")

                diffs = pred_counts - gt_counts
                absolute_errors.extend(np.abs(diffs))
                squared_errors.extend(diffs ** 2)

        mae = float(np.mean(absolute_errors))
        rmse = float(np.sqrt(np.mean(squared_errors)))
        return mae, rmse

    def run(self, experiment_specs: list[Dict[str, Any]]) -> None:
        """
        Runs the full benchmark cycle for a list of model specifications.

        Each spec dict should contain:
        - 'name': display name
        - 'config': ModelConfig
        - 'weights': optional path to additional weights (for CSRNet/P2PNet)
        """
        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)

        for spec in experiment_specs:
            name = spec["name"]
            config: ModelConfig = spec["config"]
            weights_path = spec.get("weights")

            print(f"\n--- Benchmarking: {name} ---")

            # 1. Initialize model
            model = create_model(config)
            if weights_path:
                # Optional: implement explicit load_state_dict here if needed
                state = torch.load(weights_path, map_location=self.device)
                # Expect state to be a pure state_dict for CSRNet/P2PNet
                if hasattr(model, "load_state_dict"):
                    model.load_state_dict(state)
            model.to(self.device)

            # 2. Latency
            print("Measuring Latency...")
            latency_ms = self.measure_latency(model, dummy_input, config.kind)
            fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")

            # 3. Accuracy (MAE / RMSE)
            # For CSRNet we use density targets; for others (YOLO, P2PNet) point maps.
            target_type = "density" if config.kind == ModelKind.CSRNET else "points"
            val_ds = ShanghaiTechDataset(
                image_dir="data/ShanghaiTech/part_A/test_data/images",
                mat_dir="data/ShanghaiTech/part_A/test_data/ground_truth",
                target_type=target_type,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=("cuda" in self.device),
            )

            print("Measuring Accuracy...")
            mae, rmse = self.evaluate_accuracy(model, val_loader, config.kind)

            # Parameter count
            if isinstance(model, YOLO11Wrapper):
                params = sum(p.numel() for p in model.model.parameters()) / 1e6
            else:
                params = sum(p.numel() for p in model.parameters()) / 1e6

            self.results.append(
                {
                    "Model": name,
                    "MAE": f"{mae:.2f}",
                    "RMSE": f"{rmse:.2f}",
                    "Latency (ms)": f"{latency_ms:.2f}",
                    "FPS": f"{fps:.1f}",
                    "Params (M)": f"{params:.2f}",
                }
            )

    def print_report(self) -> None:
        df = pd.DataFrame(self.results)
        print("\n=== FINAL BENCHMARK REPORT ===")
        print(df.to_markdown(index=False))
        df.to_csv("benchmark_results.csv", index=False)


if __name__ == "__main__":
    runner = BenchmarkRunner()

    specs: list[Dict[str, Any]] = [
        {
            "name": "YOLOv11-Nano",
            "config": ModelConfig(
                kind=ModelKind.YOLO11,
                weights="yolo11n.pt",
                num_classes=1,
            ),
        },
        {
            "name": "CSRNet-VGG16",
            "config": ModelConfig(
                kind=ModelKind.CSRNET,
                out_channels=1,
            ),
        },
        {
            "name": "P2PNet-ResNet18",
            "config": ModelConfig(
                kind=ModelKind.P2PNET,
                out_channels=1,
            ),
        },
    ]

    runner.run(specs)
    runner.print_report()

