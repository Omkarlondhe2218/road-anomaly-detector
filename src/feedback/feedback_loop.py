import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .failure_queue import FailureQueueManager
from ..preprocessing.augmentor import RoadAugmentor
from ..models.train_baseline import RoadAnomalyTrainer


class AutomatedFeedbackLoop:
    """
    Automated Active Learning Feedback Loop Engine.
    Ingests corrected failure samples from FailureQueueManager, applies augmentations,
    and triggers YOLO fine-tuning iterations to continuously improve model accuracy.
    """

    def __init__(
        self,
        queue_manager: Optional[FailureQueueManager] = None,
        config_path: str = "configs/dataset.yaml",
        base_weights: str = "yolov8n.pt",
        checkpoints_dir: str = "models/checkpoints"
    ):
        self.queue_manager = queue_manager or FailureQueueManager()
        self.config_path = config_path
        self.base_weights = base_weights
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_count = 0

    def run_feedback_iteration(
        self,
        epochs: int = 2,
        version_tag: Optional[str] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """
        Executes a complete feedback loop cycle:
          1. Exports annotated failure cases from queue to training dataset.
          2. Fine-tunes model on updated dataset.
          3. Saves versioned model checkpoint.
        """
        self.iteration_count += 1
        tag = version_tag or f"v{self.iteration_count}_{int(time.time())}"
        print(f"\n[AutomatedFeedbackLoop] Starting Feedback Retraining Cycle '{tag}'...")

        # 1. Export ready items from queue
        exported_count = self.queue_manager.export_ready_items_to_yolo()

        if exported_count == 0:
            print("[AutomatedFeedbackLoop] No new annotated feedback items to train on. Retraining skipped.")
            return {
                "status": "skipped",
                "reason": "no_new_annotations",
                "exported_count": 0
            }

        # 2. Trigger Fine-Tuning
        print(f"[AutomatedFeedbackLoop] Fine-tuning model on updated dataset ({exported_count} new hard examples added)...")
        trainer = RoadAnomalyTrainer(
            config_path=self.config_path,
            model_weights=self.base_weights,
            experiment_name=f"feedback_loop_{tag}"
        )
        trainer.initialize_model()
        trainer.train(epochs=epochs, imgsz=640, batch_size=4, device=device, workers=0)

        # 3. Save Versioned Model Checkpoint
        save_name = f"yolov8_road_feedback_{tag}.pt"
        checkpoint_path = self.checkpoints_dir / save_name

        # Copy best model from runs directory if available
        runs_best = Path("runs") / f"feedback_loop_{tag}" / "weights" / "best.pt"
        if runs_best.exists():
            shutil.copy(str(runs_best), str(checkpoint_path))
            print(f"[AutomatedFeedbackLoop] Saved versioned model checkpoint to '{checkpoint_path}'")
        else:
            checkpoint_path = Path(self.base_weights)

        return {
            "status": "success",
            "iteration": self.iteration_count,
            "version_tag": tag,
            "ingested_samples": exported_count,
            "checkpoint_path": str(checkpoint_path)
        }
