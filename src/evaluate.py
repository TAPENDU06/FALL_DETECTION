
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from ultralytics import YOLO

from features import extract_features
from fall_detector import FallDetector


def process_video(model, video_path, device):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS: {video_path}")

    detector = FallDetector()

    frame_number = 0
    fall_detected = False
    no_person_frames = 0
    possible_fall_frames = 0
    first_detection_timestamp = None

    while True:

        success, frame = cap.read()

        if not success:
            break

        timestamp = frame_number / fps

        # Extract RGB panel from right half.
        height, width = frame.shape[:2]
        rgb_frame = frame[:, width // 2:, :]

        rgb_height, rgb_width = rgb_frame.shape[:2]

        # YOLO pose estimation.
        result = model.predict(
            rgb_frame,
            imgsz=640,
            conf=0.35,
            device=device,
            verbose=False
        )[0]

        valid_pose = False

        if (
            result.keypoints is not None
            and len(result.keypoints) > 0
        ):

            keypoints = (
                result.keypoints.xy[0]
                .cpu()
                .numpy()
            )

            confidence = (
                result.keypoints.conf[0]
                .cpu()
                .numpy()
                if result.keypoints.conf is not None
                else None
            )

            required = [5, 6, 11, 12]

            valid_pose = (
                confidence is not None
                and all(
                    confidence[i] >= 0.35
                    for i in required
                )
            )

            if valid_pose:

                features = extract_features(
                    keypoints,
                    rgb_width,
                    rgb_height
                )

                if features is not None:

                    detection = detector.update(
                        angle=features["torso_angle_deg"],
                        hip=features["normalized_hip"],
                        timestamp=timestamp
                    )

                else:
                    valid_pose = False

        if not valid_pose:

            no_person_frames += 1

            detector.handle_missing(timestamp)

        # Record internal state, including events that
        # transition to COOLDOWN on the next frame.
        if detector.state == "POSSIBLE_FALL":
            possible_fall_frames += 1

        if detector.state == "FALL_DETECTED":

            fall_detected = True

            if first_detection_timestamp is None:
                first_detection_timestamp = timestamp

        frame_number += 1

    cap.release()

    return {
        "frames_processed": frame_number,
        "fps": fps,
        "duration": frame_number / fps,
        "fall_detected": fall_detected,
        "no_person_frames": no_person_frames,
        "possible_fall_frames": possible_fall_frames,
        "first_detection_timestamp": first_detection_timestamp,
        "final_state": detector.state
    }


def calculate_metrics(actual, predicted):

    # Positive class = FALL
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_Score": f1
    }


def save_confusion_matrix(tp, tn, fp, fn, output_path):

    # Rows = actual class
    # Columns = predicted class
    matrix = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(matrix, cmap="Blues")

    ax.set_title("Fall Detection Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["NORMAL", "FALL"])
    ax.set_yticklabels(["NORMAL", "FALL"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="black"
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description="Batch evaluation of URFD fall detection"
    )

    parser.add_argument(
        "--data-root",
        default="data/raw/urfd",
        help="URFD dataset root directory"
    )

    parser.add_argument(
        "--view",
        default="front",
        help="Dataset view folder, e.g. front or top"
    )

    parser.add_argument(
        "--camera",
        choices=["cam0", "cam1"],
        default="cam0",
        help="Camera recording to evaluate"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Evaluation output directory"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)

    output_dir = Path(
        args.output or f"results/evaluation/{args.camera}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------
    # Discover dataset videos
    # -----------------------------------

    videos = []

    for label in ["fall", "normal"]:

        folder = data_root / label / args.view

        if not folder.exists():
            print(f"WARNING: Folder not found: {folder}")
            continue

        for video_path in sorted(
            folder.glob(f"*{args.camera}.mp4")
        ):

            videos.append({
                "path": video_path,
                "label": label,
                "actual": 1 if label == "fall" else 0
            })

    if not videos:
        raise RuntimeError(
            "No videos found. Check dataset paths, view and camera."
        )

    print("\n========== DATASET DISCOVERY ==========")

    print("Camera:", args.camera)
    print("View:", args.view)
    print("Total videos:", len(videos))

    print(
        "Fall videos:",
        sum(v["actual"] == 1 for v in videos)
    )

    print(
        "Normal videos:",
        sum(v["actual"] == 0 for v in videos)
    )

    # -----------------------------------
    # Load YOLO once
    # -----------------------------------

    device = 0 if torch.cuda.is_available() else "cpu"

    print("\nLoading YOLO pose model...")

    model = YOLO("yolo26n-pose.pt")

    print("Device:", device)

    # -----------------------------------
    # Process every video
    # -----------------------------------

    results = []

    for index, video in enumerate(videos, start=1):

        video_path = video["path"]

        print(
            f"\n[{index}/{len(videos)}] Processing: "
            f"{video_path.name}"
        )

        try:

            prediction = process_video(
                model,
                video_path,
                device
            )

            predicted = (
                1 if prediction["fall_detected"] else 0
            )

            correct = predicted == video["actual"]

            results.append({
                "video": video_path.name,
                "path": str(video_path),
                "actual_label": video["label"],
                "predicted_label": (
                    "fall" if predicted == 1 else "normal"
                ),
                "correct": correct,
                **prediction
            })

            print(
                "Actual:",
                video["label"],
                "| Predicted:",
                "fall" if predicted else "normal",
                "| Result:",
                "CORRECT" if correct else "INCORRECT"
            )

        except Exception as error:

            print("ERROR:", error)

    if not results:
        raise RuntimeError("No videos were processed successfully.")

    # -----------------------------------
    # Save individual video results
    # -----------------------------------

    results_path = output_dir / "per_video_results.csv"

    fieldnames = list(results[0].keys())

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------
    # Calculate metrics
    # -----------------------------------

    actual = np.array([
        1 if row["actual_label"] == "fall" else 0
        for row in results
    ])

    predicted = np.array([
        1 if row["predicted_label"] == "fall" else 0
        for row in results
    ])

    metrics = calculate_metrics(
        actual,
        predicted
    )

    # -----------------------------------
    # Save evaluation summary
    # -----------------------------------

    summary_path = output_dir / "metrics_summary.csv"

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])

        for key, value in metrics.items():

            if key in [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1_Score"
            ]:
                writer.writerow([key, round(value, 4)])

            else:
                writer.writerow([key, value])

    # -----------------------------------
    # Save confusion matrix
    # -----------------------------------

    matrix_path = output_dir / "confusion_matrix.png"

    save_confusion_matrix(
        metrics["TP"],
        metrics["TN"],
        metrics["FP"],
        metrics["FN"],
        matrix_path
    )

    # -----------------------------------
    # Final report
    # -----------------------------------

    print("\n========== EVALUATION SUMMARY ==========")

    print("Videos evaluated:", len(results))

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key}: {value * 100:.2f}%")

        else:
            print(f"{key}: {value}")

    print("\n========== OUTPUT FILES ==========")

    print("Individual results:", results_path)
    print("Metrics summary:", summary_path)
    print("Confusion matrix:", matrix_path)


if __name__ == "__main__":
    main()
