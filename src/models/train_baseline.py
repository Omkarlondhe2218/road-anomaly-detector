import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class RoadAnomalyTrainer:
    """
    YOLOv8 Baseline Model Trainer and Evaluator for Road Anomaly Detection.
    """

    def __init__(
        self,
        config_path: str = "configs/dataset.yaml",
        model_weights: str = "yolov8m.pt",
        project_name: str = "road_anomaly_runs",
        experiment_name: str = "yolov8m_baseline"
    ):
        self.config_path = Path(config_path)
        self.model_weights = model_weights
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.model = None

    def initialize_model(self, custom_weights: Optional[str] = None):
        """Loads pre-trained YOLO model (default YOLOv8m)."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics package is missing. Install with 'pip install ultralytics'")

        weights_to_load = custom_weights or self.model_weights
        print(f"[RoadAnomalyTrainer] Initializing YOLO model with weights: '{weights_to_load}'...")
        self.model = YOLO(weights_to_load)
        return self.model

    def train(
        self,
        epochs: int = 5,
        imgsz: int = 640,
        batch_size: int = 8,
        device: str = "auto",
        lr0: float = 0.01,
        workers: int = 2
    ) -> Dict[str, Any]:
        """
        Executes model training pipeline on the dataset configured in dataset.yaml.
        """
        if self.model is None:
            self.initialize_model()

        if not self.config_path.exists():
            raise FileNotFoundError(f"Dataset configuration file not found at '{self.config_path}'")

        print(f"[RoadAnomalyTrainer] Starting training for {epochs} epochs at resolution {imgsz}x{imgsz}...")

        # Run YOLO training
        results = self.model.train(
            data=str(self.config_path.resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            lr0=lr0,
            workers=workers,
            project=self.project_name,
            name=self.experiment_name,
            exist_ok=True
        )

        print("[RoadAnomalyTrainer] Training complete!")
        return results

    def evaluate(self) -> Dict[str, float]:
        """
        Validates model performance on the validation dataset split.
        Returns precision, recall, and mAP metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or trained yet.")

        print("[RoadAnomalyTrainer] Evaluating model on validation split...")
        metrics = self.model.val()

        results_summary = {
            "precision": float(metrics.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(metrics.results_dict.get("metrics/recall(B)", 0.0)),
            "mAP50": float(metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50-95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
        }

        print(f"[RoadAnomalyTrainer] Evaluation Summary:")
        for k, v in results_summary.items():
            print(f"  • {k}: {v:.4f}")

        return results_summary
