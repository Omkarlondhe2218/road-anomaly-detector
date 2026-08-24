import os
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Dict


class SyntheticRoadDatasetGenerator:
    """
    Generates synthetic road surface images with simulated anomalies (potholes, cracks, rutting, debris)
    and formats them into YOLO dataset structure (train/val split).
    """

    CLASSES = {
        0: "pothole",
        1: "crack",
        2: "rutting",
        3: "debris"
    }

    def __init__(self, output_dir: str = "data/processed", img_size: Tuple[int, int] = (640, 640)):
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self._setup_directories()

    def _setup_directories(self):
        """Creates standard YOLO folder layout."""
        for split in ["train", "val"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _create_asphalt_texture(self, h: int, w: int) -> np.ndarray:
        """Renders realistic asphalt texture with grain noise."""
        base_gray = np.random.randint(60, 90)
        texture = np.full((h, w, 3), base_gray, dtype=np.uint8)
        
        # Add high-frequency noise
        noise = np.random.normal(0, 12, (h, w, 3)).astype(np.float32)
        texture = np.clip(texture.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Add subtle lane markings
        if random.random() < 0.6:
            cv2.line(texture, (int(w * 0.1), 0), (int(w * 0.1), h), (200, 200, 200), random.randint(4, 8))
        return texture

    def _draw_anomaly(self, img: np.ndarray, class_id: int) -> Tuple[List[float], bool]:
        """
        Draws a simulated anomaly onto the road image and returns normalized YOLO bbox [xc, yc, w, h].
        """
        h, w = img.shape[:2]
        
        # Random bbox dimensions
        bw_px = random.randint(40, 150)
        bh_px = random.randint(40, 150)
        x_min = random.randint(20, w - bw_px - 20)
        y_min = random.randint(20, h - bh_px - 20)
        x_max = x_min + bw_px
        y_max = y_min + bh_px

        cx = x_min + bw_px // 2
        cy = y_min + bh_px // 2

        if class_id == 0:  # pothole (dark irregular ellipse with shadow depth)
            cv2.ellipse(img, (cx, cy), (bw_px // 2, bh_px // 2), random.randint(0, 180), 0, 360, (20, 20, 20), -1)
            cv2.ellipse(img, (cx + 2, cy + 2), (bw_px // 2, bh_px // 2), random.randint(0, 180), 0, 360, (40, 40, 40), 2)
        
        elif class_id == 1:  # crack (zigzag dark polylines)
            num_pts = random.randint(4, 8)
            pts = []
            for i in range(num_pts):
                px = int(x_min + (i / num_pts) * bw_px + random.randint(-10, 10))
                py = int(y_min + random.randint(0, bh_px))
                pts.append([px, py])
            cv2.polylines(img, [np.array(pts, np.int32)], False, (15, 15, 15), random.randint(2, 4))
        
        elif class_id == 2:  # rutting (long longitudinal depressions)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (45, 45, 45), -1)
            cv2.blur(img[y_min:y_max, x_min:x_max], (15, 15))
        
        elif class_id == 3:  # debris (small high contrast objects)
            color = (random.randint(150, 230), random.randint(150, 230), random.randint(150, 230))
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, -1)

        # Normalize to [0, 1] for YOLO format
        xc_norm = cx / w
        yc_norm = cy / h
        w_norm = bw_px / w
        h_norm = bh_px / h

        return [xc_norm, yc_norm, w_norm, h_norm], True

    def generate_dataset(self, num_train: int = 40, num_val: int = 10) -> Dict[str, int]:
        """
        Generates images and label text files.
        """
        total_counts = {"train": 0, "val": 0}

        for split, count in [("train", num_train), ("val", num_val)]:
            for i in range(count):
                img = self._create_asphalt_texture(self.img_size[1], self.img_size[0])
                
                # Add 1 to 3 anomalies per image
                num_anomalies = random.randint(1, 3)
                label_lines = []
                
                for _ in range(num_anomalies):
                    cid = random.choice(list(self.CLASSES.keys()))
                    bbox, ok = self._draw_anomaly(img, cid)
                    if ok:
                        label_lines.append(f"{cid} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
                
                filename = f"sample_road_{i+1:04d}"
                img_path = self.output_dir / "images" / split / f"{filename}.jpg"
                lbl_path = self.output_dir / "labels" / split / f"{filename}.txt"

                cv2.imwrite(str(img_path), img)
                with open(lbl_path, "w") as f:
                    f.write("\n".join(label_lines) + "\n")
                
                total_counts[split] += 1

        print(f"[DatasetGenerator] Successfully generated {total_counts['train']} train images & {total_counts['val']} val images in '{self.output_dir}'")
        return total_counts
