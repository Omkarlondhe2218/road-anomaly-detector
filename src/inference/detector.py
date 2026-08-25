import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class RoadAnomalyDetector:
    """
    Real-Time Road Anomaly Detection Inference Engine powered by YOLOv8.
    Includes uncertainty evaluation to identify hard examples for feedback queueing.
    """

    CLASS_NAMES = {
        0: "pothole",
        1: "crack",
        2: "rutting",
        3: "debris"
    }

    # Color palette for classes (BGR)
    CLASS_COLORS = {
        0: (0, 0, 255),      # Red for pothole
        1: (0, 165, 255),    # Orange for crack
        2: (255, 255, 0),    # Cyan for rutting
        3: (0, 255, 0)       # Green for debris
    }

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        uncertainty_threshold: float = 0.60
    ):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads trained YOLO model."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library is required. Install via 'pip install ultralytics'")
        
        if not self.weights_path.exists() and not str(self.weights_path).endswith(".pt"):
            raise FileNotFoundError(f"Model weights file not found: '{self.weights_path}'")

        print(f"[RoadAnomalyDetector] Loading model weights from '{self.weights_path}'...")
        self.model = YOLO(str(self.weights_path))

    def predict_image(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], bool]:
        """
        Runs inference on an image path or numpy BGR array.
        
        Returns:
            - processed_image: BGR numpy image
            - detections: List of detection dictionaries
            - has_uncertain: True if any detection is below uncertainty_threshold
        """
        if self.model is None:
            self.load_model()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if isinstance(image_input, (str, Path)):
            img_path = str(image_input)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image from path: '{img_path}'")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string, Path object, or numpy ndarray")

        h, w = image.shape[:2]

        # Perform YOLO inference
        results = self.model.predict(source=image, conf=conf, verbose=False)[0]

        detections = []
        has_uncertain = False

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = [float(v) for v in box]
                
                # Normalized YOLO format [xc, yc, w, h]
                xc_norm = float(((x1 + x2) / 2) / w)
                yc_norm = float(((y1 + y2) / 2) / h)
                w_norm = float((x2 - x1) / w)
                h_norm = float((y2 - y1) / h)

                is_uncertain = float(score) < self.uncertainty_threshold
                if is_uncertain:
                    has_uncertain = True

                cls_name = self.CLASS_NAMES.get(cls_id, f"anomaly_{cls_id}")

                detections.append({
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "confidence": float(score),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_yolo": [xc_norm, yc_norm, w_norm, h_norm],
                    "is_uncertain": is_uncertain
                })

        return image, detections, has_uncertain

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        highlight_uncertain: bool = True
    ) -> np.ndarray:
        """
        Renders bounding boxes and prediction labels onto the image.
        Low-confidence/uncertain detections get flagged with distinct warning styling.
        """
        annotated = image.copy()
        h, w = image.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            cls_id = det["class_id"]
            cls_name = det["class_name"]
            conf = det["confidence"]
            is_uncertain = det["is_uncertain"]

            # Color selection
            if highlight_uncertain and is_uncertain:
                color = (0, 255, 255)  # Bright Yellow for uncertain predictions needing review
                label_text = f"⚠️ {cls_name} {conf:.2f}"
                thickness = 3
            else:
                color = self.CLASS_COLORS.get(cls_id, (255, 0, 0))
                label_text = f"{cls_name} {conf:.2f}"
                thickness = 2

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label badge background
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - text_h - 6)),
                (x1 + text_w + 6, max(text_h + 6, y1)),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 3, max(text_h, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return annotated
