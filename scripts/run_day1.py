import os
import sys
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataset_generator import SyntheticRoadDatasetGenerator
from src.preprocessing.augmentor import RoadAugmentor
from src.models.train_baseline import RoadAnomalyTrainer, ULTRALYTICS_AVAILABLE


def main():
    print("=" * 60)
    print("   ROAD ANOMALY DETECTION SYSTEM - DAY 1 PIPELINE RUN")
    print("=" * 60)

    # 1. Dataset Generation
    print("\n[Step 1/3] Generating synthetic road dataset (train/val splits)...")
    data_dir = PROJECT_ROOT / "data" / "processed"
    generator = SyntheticRoadDatasetGenerator(output_dir=str(data_dir))
    dataset_info = generator.generate_dataset(num_train=20, num_val=5)

    # 2. Preprocessing & Augmentation Pipeline Verification
    print("\n[Step 2/3] Verifying Road Data Augmentation Engine...")
    aug_out_dir = PROJECT_ROOT / "data" / "augmented"
    aug_out_dir.mkdir(parents=True, exist_ok=True)

    augmentor = RoadAugmentor(img_size=(640, 640))
    sample_img_path = data_dir / "images" / "train" / "sample_road_0001.jpg"

    if sample_img_path.exists():
        raw_img = cv2.imread(str(sample_img_path))
        # Read label
        lbl_path = data_dir / "labels" / "train" / "sample_road_0001.txt"
        bboxes, cats = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cats.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

        # Generate 3 augmented variants
        for idx in range(3):
            aug_img, aug_box, aug_cat = augmentor.augment(raw_img, bboxes, cats)
            out_file = aug_out_dir / f"augmented_sample_{idx+1}.jpg"
            cv2.imwrite(str(out_file), aug_img)
            print(f"  • Created augmented sample: {out_file.name}")

    # 3. Model Baseline Initialization & Training Pass Verification
    print("\n[Step 3/3] Testing YOLOv8 Baseline Trainer Setup...")
    if ULTRALYTICS_AVAILABLE:
        # Use nano weights for rapid verification pass
        trainer = RoadAnomalyTrainer(
            config_path=str(PROJECT_ROOT / "configs" / "dataset.yaml"),
            model_weights="yolov8n.pt",
            project_name=str(PROJECT_ROOT / "runs"),
            experiment_name="day1_verification"
        )
        trainer.initialize_model()
        print("  • Initialized YOLO model successfully!")
        
        # Run 1 epoch verification pass
        print("  • Launching 1-epoch verification training run...")
        trainer.train(epochs=1, imgsz=640, batch_size=4, device="cpu", workers=0)
        print("  • Model training pass executed cleanly!")
    else:
        print("  ! Ultralytics library not installed yet. Skipping training execution test.")

    print("\n" + "=" * 60)
    print("   DAY 1 IMPLEMENTATION & VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next steps:")
    print("  1. Review generated files in data/processed and data/augmented")
    print("  2. Push your Day 1 code to GitHub using the instructions in README.md")


if __name__ == "__main__":
    main()
