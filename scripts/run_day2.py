import os
import sys
import cv2
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.detector import RoadAnomalyDetector
from src.feedback.failure_queue import FailureQueueManager
from src.feedback.feedback_loop import AutomatedFeedbackLoop


def main():
    print("=" * 65)
    print("   ROAD ANOMALY DETECTION SYSTEM - DAY 2 PIPELINE RUN")
    print("   (Inference Engine & Automated Feedback Loop Verification)")
    print("=" * 65)

    # 1. Initialize Inference Engine
    print("\n[Step 1/4] Initializing Inference Engine...")
    weights_path = PROJECT_ROOT / "yolov8n.pt"
    detector = RoadAnomalyDetector(
        weights_path=str(weights_path),
        conf_threshold=0.15,
        uncertainty_threshold=0.60
    )
    print("  • Inference engine ready!")

    # 2. Run Inference on Test Road Images
    print("\n[Step 2/4] Simulating Road Surface Anomaly Inference...")
    test_img_path = PROJECT_ROOT / "data" / "processed" / "images" / "val" / "sample_road_0001.jpg"
    
    if not test_img_path.exists():
        print("  ! Test sample image not found. Running Day 1 dataset generator first...")
        from src.utils.dataset_generator import SyntheticRoadDatasetGenerator
        gen = SyntheticRoadDatasetGenerator(output_dir=str(PROJECT_ROOT / "data" / "processed"))
        gen.generate_dataset(num_train=10, num_val=3)

    raw_img, detections, has_uncertain = detector.predict_image(str(test_img_path))
    print(f"  • Inferred image: '{test_img_path.name}'")
    print(f"  • Total anomalies detected: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"    - [{i+1}] Class: {d['class_name']} | Conf: {d['confidence']:.2f} | Uncertain: {d['is_uncertain']}")

    # Render & save output detection visualization
    out_vis_dir = PROJECT_ROOT / "data" / "output_detections"
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    vis_img = detector.draw_detections(raw_img, detections)
    vis_path = out_vis_dir / "day2_detection_output.jpg"
    cv2.imwrite(str(vis_path), vis_img)
    print(f"  • Saved visualization to: '{vis_path}'")

    # 3. Low-Confidence & Hard Example Queueing
    print("\n[Step 3/4] Flagging & Queueing Hard Example to Failure Buffer...")
    queue_mgr = FailureQueueManager(queue_dir=str(PROJECT_ROOT / "data" / "feedback_queue"))

    # Push to queue
    item_id = queue_mgr.add_failure_case(
        image=raw_img,
        detections=detections,
        reason="low_confidence_pothole_prediction"
    )

    # Simulate expert annotation of hard example (pothole class 0)
    print(f"  • Annotating item '{item_id}' with ground-truth correction...")
    queue_mgr.annotate_item(
        item_id=item_id,
        corrected_boxes_yolo=[[0.50, 0.50, 0.25, 0.25]],  # [xc, yc, w, h] normalized
        corrected_classes=[0]  # Pothole
    )

    status = queue_mgr.list_queue_status()
    print(f"  • Queue Status Overview: {status['status_breakdown']}")

    # 4. Trigger Automated Feedback Loop Retraining
    print("\n[Step 4/4] Triggering Automated Feedback Loop Retraining Cycle...")
    feedback_loop = AutomatedFeedbackLoop(
        queue_manager=queue_mgr,
        config_path=str(PROJECT_ROOT / "configs" / "dataset.yaml"),
        base_weights=str(weights_path),
        checkpoints_dir=str(PROJECT_ROOT / "models" / "checkpoints")
    )

    res = feedback_loop.run_feedback_iteration(epochs=1, version_tag="v1_day2_demo", device="cpu")
    print(f"  • Feedback Iteration Status: {res['status']}")
    if res["status"] == "success":
        print(f"  • Ingested Hard Samples: {res['ingested_samples']}")
        print(f"  • New Model Checkpoint Saved: '{res['checkpoint_path']}'")

    print("\n" + "=" * 65)
    print("   DAY 2 IMPLEMENTATION & FEEDBACK LOOP COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
