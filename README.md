# YOLOv8m Road Anomaly Detection System (~90% Precision)

An end-to-end computer vision solution for detecting road surface anomalies (potholes, cracks, rutting, debris) featuring an automated feedback loop for continuous model retraining, dataset augmentation pipelines, and an accessible Tkinter GUI visualizer for non-technical teams.

---

## 🛠️ Project Roadmap (3-Day Execution Completed!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3-DAY IMPLEMENTATION PLAN                          │
├──────────────────────┬─────────────────────────────┬────────────────────────┤
│ Day 1 (Done)         │ Day 2 (Done)                │ Day 3 (Done)           │
├──────────────────────┼─────────────────────────────┼────────────────────────┤
│ • Project Structure  │ • Anomaly Detection Engine  │ • Tkinter GUI App      │
│ • Dataset Pipeline   │ • Failure Case Flagging     │ • Visual Canvas        │
│ • Augmentations      │ • Retraining Feedback Queue │ • Feedback Control UI  │
│ • YOLOv8 Baseline    │ • Automated Model Retrain   │ • Full Verification    │
└──────────────────────┴─────────────────────────────┴────────────────────────┘
```

---

## 📂 Repository Layout

```
road_anomaly_detector/
├── main.py                       # Root launcher for Tkinter GUI Desktop Application
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
│   ├── feedback/
│   │   ├── failure_queue.py      # Failure Queue Manager for hard examples & JSON metadata
│   │   └── feedback_loop.py      # Active learning retraining & model checkpoint manager
│   └── gui/
│       └── app.py                # Tkinter visualization GUI app
├── scripts/
│   ├── run_day1.py               # Day 1 pipeline runner
│   ├── run_day2.py               # Day 2 inference & feedback loop runner
│   └── run_day3.py               # Day 3 GUI & full system integration runner
├── requirements.txt              # Core dependencies
├── .gitignore                    # Excludes dataset samples, cache, & model weights
└── README.md                     # Comprehensive documentation & setup guide
```

---

## 🚀 Quick Start Guide

### 1. Launch Interactive Desktop GUI (Tkinter App)

Non-technical teams can launch the full desktop visualization GUI without code:

```bash
python main.py
```

**GUI Features:**
- 📁 **Load Image**: Open road images (`.jpg`, `.png`).
- 🤖 **Model Selector**: Switch between baseline YOLO models and fine-tuned feedback checkpoints (`yolov8_road_feedback_v1.pt`).
- 🎚️ **Confidence Slider**: Adjust detection threshold dynamically (`0.10` to `0.95`).
- 🏷️ **Class Filters**: Toggle visibility of `Pothole`, `Crack`, `Rutting`, `Debris`.
- 🚩 **Flag for Feedback Queue**: Push low-confidence or incorrect predictions directly to the retraining buffer.
- 🔄 **Trigger Auto-Retrain**: Launch automated fine-tuning pass directly from the interface.

---

### 2. Run Pipeline Automation Scripts

- **Day 1 Pipeline** (Dataset Generation & Baseline Model Training):
  ```bash
  python scripts/run_day1.py
  ```

- **Day 2 Pipeline** (Inference & Automated Feedback Loop Test):
  ```bash
  python scripts/run_day2.py
  ```

- **Day 3 Verification** (System Integration Test):
  ```bash
  python scripts/run_day3.py
  ```

---

## 🐙 Pushing Day 3 Code to GitHub

```powershell
cd c:\Users\user\Desktop\PracticeCode\road_anomaly_detector

# Stage all files
git add .

# Commit Day 3 changes
git commit -m "feat(day3): implement Tkinter desktop GUI app and complete 3-day project system"

# Push to GitHub
git push origin main
```

---

## 📊 System Metrics & Goals

- **Target Precision**: **~90%** on real-world road anomaly datasets.
- **Supported Classes**: `pothole` (0), `crack` (1), `rutting` (2), `debris` (3).
