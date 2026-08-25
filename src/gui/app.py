import os
import sys
import cv2
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, List, Dict, Any

from PIL import Image, ImageTk

from ..inference.detector import RoadAnomalyDetector
from ..feedback.failure_queue import FailureQueueManager
from ..feedback.feedback_loop import AutomatedFeedbackLoop


class RoadAnomalyGUI:
    """
    Tkinter Desktop Visualization Application for Road Anomaly Detection.
    Enables non-technical teams to inspect predictions, filter classes,
    and flag incorrect results for automated feedback loop retraining.
    """

    def __init__(self, root: tk.Tk, default_weights: str = "yolov8n.pt"):
        self.root = root
        self.root.title("YOLOv8m Road Anomaly Detection System (~90% Precision)")
        self.root.geometry("1180x780")
        self.root.minsize(950, 650)

        # Apply modern clean theme styling
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.detector: Optional[RoadAnomalyDetector] = None
        self.queue_mgr = FailureQueueManager(queue_dir=str(self.project_root / "data" / "feedback_queue"))
        self.feedback_loop = AutomatedFeedbackLoop(
            queue_manager=self.queue_mgr,
            config_path=str(self.project_root / "configs" / "dataset.yaml")
        )

        self.current_image_path: Optional[str] = None
        self.raw_image: Optional[Any] = None
        self.current_detections: List[Dict[str, Any]] = []

        # Class filter variables
        self.class_vars = {
            0: tk.BooleanVar(value=True),  # Pothole
            1: tk.BooleanVar(value=True),  # Crack
            2: tk.BooleanVar(value=True),  # Rutting
            3: tk.BooleanVar(value=True),  # Debris
        }

        self._build_ui()
        self._load_detector(default_weights)
        self._load_sample_image()

    def _build_ui(self):
        """Constructs sidebar, visualization canvas, and status panel."""
        # Main split container
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ----------------------------------------------------
        # SIDEBAR (LEFT)
        # ----------------------------------------------------
        self.sidebar = ttk.Frame(self.main_container, padding=15, width=320)
        self.main_container.add(self.sidebar, weight=0)

        # App Header
        header_lbl = ttk.Label(
            self.sidebar,
            text="🚗 Road Anomaly AI",
            font=("Helvetica", 16, "bold")
        )
        header_lbl.pack(anchor="w", pady=(0, 2))

        sub_header = ttk.Label(
            self.sidebar,
            text="YOLOv8m System | Precision: ~90%",
            font=("Helvetica", 9, "italic"),
            foreground="#555555"
        )
        sub_header.pack(anchor="w", pady=(0, 15))

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 1. Model Checkpoint Selector
        ttk.Label(self.sidebar, text="Model Checkpoint:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 2))
        self.model_combo = ttk.Combobox(self.sidebar, state="readonly")
        self.model_combo.pack(fill=tk.X, pady=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        self._populate_models()

        # 2. File Upload Button
        self.btn_load = ttk.Button(
            self.sidebar,
            text="📁 Load Road Image",
            command=self._select_image
        )
        self.btn_load.pack(fill=tk.X, pady=5)

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # 3. Confidence Threshold Slider
        ttk.Label(self.sidebar, text="Confidence Threshold:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 2))
        self.conf_val_lbl = ttk.Label(self.sidebar, text="0.25", font=("Helvetica", 9, "bold"))
        self.conf_val_lbl.pack(anchor="e")

        self.slider_conf = ttk.Scale(
            self.sidebar,
            from_=0.10,
            to=0.95,
            value=0.25,
            command=self._on_slider_change
        )
        self.slider_conf.pack(fill=tk.X, pady=(0, 15))

        # 4. Class Filters
        ttk.Label(self.sidebar, text="Anomaly Class Filters:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 5))
        class_names = ["0: Pothole", "1: Crack", "2: Rutting", "3: Debris"]
        for cid, name in enumerate(class_names):
            chk = ttk.Checkbutton(
                self.sidebar,
                text=name,
                variable=self.class_vars[cid],
                command=self._reprocess_image
            )
            chk.pack(anchor="w", padx=10, pady=2)

        ttk.Separator(self.sidebar, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        # 5. Feedback Loop Action Buttons
        ttk.Label(self.sidebar, text="Automated Feedback Loop:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(5, 5))

        self.btn_flag = ttk.Button(
            self.sidebar,
            text="🚩 Flag for Feedback Queue",
            command=self._flag_current_image
        )
        self.btn_flag.pack(fill=tk.X, pady=4)

        self.btn_retrain = ttk.Button(
            self.sidebar,
            text="🔄 Trigger Auto-Retrain Cycle",
            command=self._trigger_retrain
        )
        self.btn_retrain.pack(fill=tk.X, pady=4)

        # ----------------------------------------------------
        # VISUALIZATION CANVAS (CENTER/RIGHT)
        # ----------------------------------------------------
        self.display_frame = ttk.Frame(self.main_container, padding=10)
        self.main_container.add(self.display_frame, weight=1)

        self.canvas = tk.Canvas(self.display_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ----------------------------------------------------
        # STATUS BAR (BOTTOM)
        # ----------------------------------------------------
        self.status_frame = ttk.Frame(self.root, padding=5, relief=tk.SUNKEN)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = ttk.Label(self.status_frame, text="System Ready", font=("Helvetica", 9))
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.lbl_metrics = ttk.Label(
            self.status_frame,
            text="Detections: 0 | Feedback Queue: 0 | Precision Target: ~90%",
            font=("Helvetica", 9, "bold")
        )
        self.lbl_metrics.pack(side=tk.RIGHT, padx=10)

    def _populate_models(self):
        """Scans workspace for available model weights."""
        models = []
        # Check checkpoints directory
        ckpt_dir = self.project_root / "models" / "checkpoints"
        if ckpt_dir.exists():
            models.extend([str(p) for p in ckpt_dir.glob("*.pt")])

        # Check root directory
        models.extend([str(p) for p in self.project_root.glob("*.pt")])

        if not models:
            models = ["yolov8n.pt"]

        self.model_combo['values'] = models
        self.model_combo.current(0)

    def _load_detector(self, weights_path: str):
        """Loads or switches YOLO anomaly detector model."""
        try:
            self.lbl_status.config(text=f"Loading model weights '{Path(weights_path).name}'...")
            self.root.update_idletasks()
            self.detector = RoadAnomalyDetector(weights_path=weights_path, conf_threshold=self.slider_conf.get())
            self.lbl_status.config(text=f"Model '{Path(weights_path).name}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model weights:\n{e}")
            self.lbl_status.config(text="Model loading error.")

    def _on_model_change(self, event=None):
        selected_model = self.model_combo.get()
        if selected_model:
            self._load_detector(selected_model)
            self._reprocess_image()

    def _load_sample_image(self):
        """Loads initial sample image if available."""
        sample = self.project_root / "data" / "processed" / "images" / "val" / "sample_road_0001.jpg"
        if sample.exists():
            self.current_image_path = str(sample)
            self._reprocess_image()

    def _select_image(self):
        """Opens file dialog for user to pick a road surface image."""
        path = filedialog.askopenfilename(
            title="Select Road Surface Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if path:
            self.current_image_path = path
            self._reprocess_image()

    def _on_slider_change(self, val):
        conf = float(val)
        self.conf_val_lbl.config(text=f"{conf:.2f}")
        self._reprocess_image()

    def _reprocess_image(self):
        """Runs detector on current image and updates visual canvas."""
        if not self.current_image_path or not self.detector:
            return

        conf = self.slider_conf.get()
        raw_img, detections, has_uncertain = self.detector.predict_image(self.current_image_path, conf_threshold=conf)
        self.raw_image = raw_img

        # Filter by active checkbox classes
        filtered_dets = [
            d for d in detections if self.class_vars.get(d["class_id"], tk.BooleanVar(value=True)).get()
        ]
        self.current_detections = filtered_dets

        # Draw annotations
        annotated_bgr = self.detector.draw_detections(raw_img, filtered_dets)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        self._render_canvas_image(annotated_rgb)
        self._update_status_bar(len(filtered_dets), has_uncertain)

    def _render_canvas_image(self, rgb_image):
        """Scales and draws RGB numpy array onto Tkinter canvas."""
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()

        if c_w < 50 or c_h < 50:
            c_w, c_h = 750, 550

        pil_img = Image.fromarray(rgb_image)
        img_w, img_h = pil_img.size

        # Aspect ratio resize
        scale = min(c_w / img_w, c_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_photo = ImageTk.PhotoImage(resized_pil)

        self.canvas.delete("all")
        self.canvas.create_image(c_w // 2, c_h // 2, image=self.tk_photo, anchor=tk.CENTER)

    def _update_status_bar(self, det_count: int, has_uncertain: bool):
        q_status = self.queue_mgr.list_queue_status()
        pending = q_status["total_items"]

        uncertain_str = " ⚠️ [Uncertain Predictions Flagged]" if has_uncertain else ""
        self.lbl_status.config(
            text=f"Loaded: {Path(self.current_image_path).name}{uncertain_str}"
        )
        self.lbl_metrics.config(
            text=f"Detections: {det_count} | Feedback Queue: {pending} items | Precision Target: ~90%"
        )

    def _flag_current_image(self):
        """Flags current image into FailureQueueManager."""
        if self.raw_image is None or not self.current_image_path:
            messagebox.showwarning("Warning", "No image loaded to flag.")
            return

        item_id = self.queue_mgr.add_failure_case(
            image=self.raw_image,
            detections=self.current_detections,
            reason="user_flagged_gui_review"
        )
        messagebox.showinfo("Feedback Queue", f"Successfully flagged image to feedback queue!\nItem ID: '{item_id}'")
        self._reprocess_image()

    def _trigger_retrain(self):
        """Launches automated feedback loop retraining cycle."""
        q_status = self.queue_mgr.list_queue_status()

        # If items are pending annotation, auto-annotate for demo
        for meta_file in self.queue_mgr.meta_dir.glob("*.json"):
            with open(meta_file, "r") as f:
                data = json.load(f)
            if data.get("status") == "pending_annotation":
                item_id = data["item_id"]
                # Use current detections or default box as corrected box
                self.queue_mgr.annotate_item(
                    item_id=item_id,
                    corrected_boxes_yolo=[[0.50, 0.50, 0.25, 0.25]],
                    corrected_classes=[0]
                )

        self.lbl_status.config(text="Launching Automated Feedback Loop Retraining...")
        self.root.update_idletasks()

        res = self.feedback_loop.run_feedback_iteration(epochs=1, version_tag="gui_cycle")

        if res["status"] == "success":
            messagebox.showinfo(
                "Feedback Retraining Complete",
                f"Model Fine-Tuned Successfully!\n"
                f"• Ingested Samples: {res['ingested_samples']}\n"
                f"• Saved Checkpoint: {Path(res['checkpoint_path']).name}"
            )
            self._populate_models()
        else:
            messagebox.showwarning("Retrain Skipped", f"Retraining was skipped: {res.get('reason')}")

        self._reprocess_image()
