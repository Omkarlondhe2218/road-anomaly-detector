import os
import cv2
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class FailureQueueManager:
    """
    Manages low-confidence, hard example, or user-flagged predictions in a persistent feedback queue.
    Prepares queued samples for human review/annotation and active retraining ingestion.
    """

    def __init__(self, queue_dir: str = "data/feedback_queue"):
        self.queue_dir = Path(queue_dir)
        self.img_dir = self.queue_dir / "images"
        self.meta_dir = self.queue_dir / "metadata"
        self._setup_directories()

    def _setup_directories(self):
        """Ensures queue folders exist."""
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def add_failure_case(
        self,
        image: cv2.typing.MatLike,
        detections: List[Dict[str, Any]],
        reason: str = "low_confidence",
        custom_id: Optional[str] = None
    ) -> str:
        """
        Saves image and prediction metadata into the queue buffer for retraining.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        item_id = custom_id or f"hard_ex_{timestamp}_{int(time.time() * 1000) % 1000}"

        img_filename = f"{item_id}.jpg"
        meta_filename = f"{item_id}.json"

        img_path = self.img_dir / img_filename
        meta_path = self.meta_dir / meta_filename

        cv2.imwrite(str(img_path), image)

        metadata = {
            "item_id": item_id,
            "timestamp": timestamp,
            "reason": reason,
            "image_filename": img_filename,
            "predicted_detections": detections,
            "corrected_annotations": [],  # Filled after review or auto-labeling
            "status": "pending_annotation"
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[FailureQueueManager] Added failure case '{item_id}' to queue (reason: {reason})")
        return item_id

    def annotate_item(
        self,
        item_id: str,
        corrected_boxes_yolo: List[List[float]],
        corrected_classes: List[int]
    ) -> bool:
        """
        Updates queued item with corrected ground-truth YOLO annotations.
        corrected_boxes_yolo: List of normalized [xc, yc, w, h]
        """
        meta_path = self.meta_dir / f"{item_id}.json"
        if not meta_path.exists():
            print(f"[FailureQueueManager] Item '{item_id}' not found in metadata.")
            return False

        with open(meta_path, "r") as f:
            data = json.load(f)

        data["corrected_annotations"] = [
            {"class_id": cid, "bbox_yolo": box}
            for cid, box in zip(corrected_classes, corrected_boxes_yolo)
        ]
        data["status"] = "ready_for_retraining"

        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[FailureQueueManager] Successfully annotated item '{item_id}' (Ready for Retraining)")
        return True

    def list_queue_status(self) -> Dict[str, Any]:
        """Returns overview counts of queued failure cases."""
        total = 0
        status_counts = {"pending_annotation": 0, "ready_for_retraining": 0, "ingested": 0}

        for meta_file in self.meta_dir.glob("*.json"):
            total += 1
            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)
                    st = data.get("status", "pending_annotation")
                    status_counts[st] = status_counts.get(st, 0) + 1
            except Exception:
                pass

        return {
            "total_items": total,
            "status_breakdown": status_counts
        }

    def export_ready_items_to_yolo(self, dataset_processed_dir: str = "data/processed") -> int:
        """
        Exports all 'ready_for_retraining' queued items into the training dataset folder.
        """
        dataset_dir = Path(dataset_processed_dir)
        train_img_dir = dataset_dir / "images" / "train"
        train_lbl_dir = dataset_dir / "labels" / "train"

        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0

        for meta_file in self.meta_dir.glob("*.json"):
            with open(meta_file, "r") as f:
                data = json.load(f)

            if data.get("status") == "ready_for_retraining" and len(data.get("corrected_annotations", [])) > 0:
                item_id = data["item_id"]
                src_img = self.img_dir / data["image_filename"]

                if src_img.exists():
                    dst_img = train_img_dir / f"feedback_{item_id}.jpg"
                    dst_lbl = train_lbl_dir / f"feedback_{item_id}.txt"

                    shutil.copy(str(src_img), str(dst_img))

                    label_lines = []
                    for ann in data["corrected_annotations"]:
                        cid = ann["class_id"]
                        xc, yc, bw, bh = ann["bbox_yolo"]
                        label_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                    with open(dst_lbl, "w") as lbl_f:
                        lbl_f.write("\n".join(label_lines) + "\n")

                    # Mark as ingested
                    data["status"] = "ingested"
                    with open(meta_file, "w") as f_out:
                        json.dump(data, f_out, indent=2)

                    exported_count += 1

        print(f"[FailureQueueManager] Exported {exported_count} annotated feedback items to '{dataset_dir}'")
        return exported_count
