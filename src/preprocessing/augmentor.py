import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class RoadAugmentor:
    """
    Image Preprocessing & Augmentation Engine tailored for Road Anomaly Detection.
    Simulates real-world driving conditions (lighting shifts, shadows, weather, motion blur).
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (640, 640),
        p_heavy_aug: float = 0.5
    ):
        self.img_size = img_size
        self.p_heavy_aug = p_heavy_aug
        self._init_pipeline()

    def _init_pipeline(self):
        """Build albumentations pipeline if available, fallback to cv2 functions."""
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.Compose([
                A.Resize(height=self.img_size[1], width=self.img_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
                A.MotionBlur(blur_limit=(3, 7), p=0.4),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'], min_visibility=0.2))
        else:
            self.transform = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Standardizes image resolution and color channels.
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Ensure BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        resized = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        return resized

    def add_road_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Simulates roadside shadows (trees, overpasses, vehicles) across road surface.
        """
        h, w = image.shape[:2]
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        # Random polygon shadow
        num_vertices = np.random.randint(3, 6)
        pts = np.array([
            [np.random.randint(0, w), np.random.randint(0, h)] for _ in range(num_vertices)
        ], np.int32)
        
        cv2.fillPoly(shadow_mask, [pts], np.random.uniform(0.3, 0.6))
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
        
        shadowed = image.astype(np.float32)
        for c in range(3):
            shadowed[:, :, c] *= shadow_mask
            
        return np.clip(shadowed, 0, 255).astype(np.uint8)

    def augment(
        self,
        image: np.ndarray,
        bboxes: Optional[List[List[float]]] = None,
        category_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Applies data augmentation pipeline to an image and its YOLO bounding boxes.
        
        bboxes format: [[x_center, y_center, width, height], ...] (normalized 0..1)
        category_ids format: [0, 1, 2, ...]
        """
        image = self.preprocess_image(image)
        
        if bboxes is None:
            bboxes = []
        if category_ids is None:
            category_ids = []

        # Apply synthetic road shadow occasionally
        if np.random.rand() < 0.3:
            image = self.add_road_shadows(image)

        if ALBUMENTATIONS_AVAILABLE and self.transform is not None and len(bboxes) > 0:
            try:
                # Filter out invalid bboxes outside [0, 1] bounds
                valid_boxes = []
                valid_cats = []
                for box, cat in zip(bboxes, category_ids):
                    xc, yc, bw, bh = box
                    if 0 < xc < 1 and 0 < yc < 1 and 0 < bw < 1 and 0 < bh < 1:
                        valid_boxes.append([xc, yc, bw, bh])
                        valid_cats.append(cat)

                if len(valid_boxes) > 0:
                    augmented = self.transform(image=image, bboxes=valid_boxes, category_ids=valid_cats)
                    return augmented['image'], augmented['bboxes'], augmented['category_ids']
            except Exception as e:
                # Fallback to basic image-only transformation if bbox error occurs
                pass

        # Fallback CV2-based augmentations (flip, brightness change)
        aug_img = image.copy()
        aug_boxes = [box[:] for box in bboxes]
        aug_cats = category_ids[:]

        if np.random.rand() < 0.5 and len(aug_boxes) > 0:
            # Horizontal flip
            aug_img = cv2.flip(aug_img, 1)
            for box in aug_boxes:
                box[0] = 1.0 - box[0]  # flip x_center

        # Brightness adjustment
        factor = np.random.uniform(0.7, 1.3)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return aug_img, aug_boxes, aug_cats
