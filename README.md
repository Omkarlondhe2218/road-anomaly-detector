# YOLOv8m Road Anomaly Detection System (~90% Precision)

An end-to-end computer vision solution for detecting road surface anomalies (potholes, cracks, rutting, debris) featuring an automated feedback loop for continuous model retraining, dataset augmentation pipelines, and an accessible Tkinter GUI visualizer.

---

## 🛠️ Project Roadmap (3-Day Execution)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3-DAY IMPLEMENTATION PLAN                          │
├──────────────────────┬─────────────────────────────┬────────────────────────┤
│ Day 1 (Done)         │ Day 2 (Done)                │ Day 3 (Next)           │
├──────────────────────┼─────────────────────────────┼────────────────────────┤
│ • Project Structure  │ • Anomaly Detection Engine  │ • Tkinter GUI App      │
│ • Dataset Pipeline   │ • Failure Case Flagging     │ • Visual Renderer      │
│ • Augmentations      │ • Retraining Feedback Queue │ • Feedback Control UI  │
│ • YOLOv8 Baseline    │ • Automated Model Retrain   │ • Full Verification    │
└──────────────────────┴─────────────────────────────┴────────────────────────┘
```

---

## 📂 Repository Layout

```
road_anomaly_detector/
├── configs/
│   └── dataset.yaml              # YOLOv8 dataset configuration (class mappings & paths)
├── data/
│   ├── feedback_queue/           # Persistent queue buffer for hard & low-confidence cases
│   └── output_detections/        # Annotated prediction visualizations
├── models/
│   └── checkpoints/              # Fine-tuned model checkpoints (v1, v2, etc.)
├── src/
│   ├── preprocessing/
│   │   └── augmentor.py          # Albumentations + OpenCV environmental & road augmentation
│   ├── utils/
│   │   └── dataset_generator.py  # Synthetic road image & YOLO label generator
│   ├── models/
│   │   └── train_baseline.py     # YOLOv8 training, fine-tuning, & validation metrics
│   ├── inference/
│   │   └── detector.py           # Inference engine & low-confidence anomaly detector
│   └── feedback/
│       ├── failure_queue.py      # Failure Queue Manager for hard examples & JSON metadata
│       └── feedback_loop.py      # Active learning retraining & model checkpoint manager
├── scripts/
│   ├── run_day1.py               # Day 1 pipeline runner
│   └── run_day2.py               # Day 2 inference & feedback loop runner
├── requirements.txt              # Core dependencies
├── .gitignore                    # Excludes dataset samples, cache, & model weights
└── README.md                     # Documentation & GitHub setup guide
```

---

## 🚀 Quick Start & Pipeline Execution

### Day 1 Pipeline (Dataset & Baseline Model)

```bash
python scripts/run_day1.py
```

### Day 2 Pipeline (Inference & Automated Feedback Loop)

```bash
python scripts/run_day2.py
```

This script executes:
1. Anomaly detection inference on road surface images.
2. Low-confidence prediction flagging (< 0.60 threshold).
3. Queueing failure cases to `data/feedback_queue/`.
4. Ingestion of ground-truth annotations and automated model fine-tuning.
5. Exporting a new versioned model checkpoint to `models/checkpoints/`.

---

## 🐙 Pushing Day 2 Code to GitHub

```powershell
cd c:\Users\user\Desktop\PracticeCode\road_anomaly_detector

# Stage Day 2 files
git add .

# Commit Day 2 changes
git commit -m "feat(day2): implement inference engine, failure queue manager, and automated feedback loop"

# Push to GitHub
git push origin main
```

---

## 📊 Performance Goal

- Target Precision: **~90%** on real-world road anomaly datasets.
- Supported Anomaly Classes: `pothole` (0), `crack` (1), `rutting` (2), `debris` (3).
