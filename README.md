# YOLOv8m Road Anomaly Detection System (~90% Precision)

An end-to-end computer vision solution for detecting road surface anomalies (potholes, cracks, rutting, debris) featuring an automated feedback loop for continuous model retraining, dataset augmentation pipelines, and an accessible Tkinter GUI visualizer.

---

## 🛠️ Project Roadmap (3-Day Execution)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          3-DAY IMPLEMENTATION PLAN                          │
├──────────────────────┬─────────────────────────────┬────────────────────────┤
│ Day 1 (Done)         │ Day 2 (Next)                │ Day 3                  │
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
├── src/
│   ├── preprocessing/
│   │   └── augmentor.py          # Albumentations + OpenCV environmental & road augmentation
│   ├── utils/
│   │   └── dataset_generator.py  # Synthetic road image & YOLO label generator
│   └── models/
│       └── train_baseline.py     # YOLOv8 training, fine-tuning, & validation metrics
├── scripts/
│   └── run_day1.py               # Day 1 pipeline runner & integration test
├── requirements.txt              # Core dependencies
├── .gitignore                    # Excludes dataset samples, cache, & model weights
└── README.md                     # Documentation & GitHub setup guide
```

---

## 🚀 Quick Start (Day 1)

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed. Clone/navigate into the project directory and install dependencies:

```bash
cd road_anomaly_detector
pip install -r requirements.txt
```

### 2. Execute Day 1 Pipeline & Verification

Run the automated Day 1 script to generate sample datasets, apply augmentations, and verify YOLO model setup:

```bash
python scripts/run_day1.py
```

This script will:
1. Create train/validation image splits in `data/processed/`.
2. Generate augmented road images simulating lighting and weather in `data/augmented/`.
3. Initialize and run a 1-epoch verification pass with YOLOv8.

---

## 🐙 Pushing Code to GitHub

Follow these steps to initialize git and push your **Day 1** implementation to your GitHub repository:

### Step 1: Initialize Local Git Repository

```bash
# Navigate to project folder
cd road_anomaly_detector

# Initialize git
git init

# Stage all files
git add .

# Create initial commit
git commit -m "feat(day1): implement project setup, augmentation engine, synthetic dataset generator, and YOLOv8 baseline trainer"
```

### Step 2: Connect Remote & Push

```bash
# Rename branch to main
git branch -M main

# Add your GitHub repository URL (Replace with your actual GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/road-anomaly-detector.git

# Push to GitHub
git push -u origin main
```

---

## 📊 Performance Goal

- Target Precision: **~90%** on real-world road anomaly datasets.
- Supported Anomaly Classes: `pothole` (0), `crack` (1), `rutting` (2), `debris` (3).
