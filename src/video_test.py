
import argparse
import csv
import os

import cv2
import torch
from ultralytics import YOLO

from features import extract_features
from fall_detector import FallDetector


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
        help="Path to prerecorded video"
    )

    parser.add_argument(
        "--output",
        default="results/metrics",
        help="Directory for prediction CSV"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display annotated video"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model = YOLO("yolo26n-pose.pt")

    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        raise RuntimeError("Invalid video FPS.")

    detector = FallDetector()

    frame_number = 0
    fall_detected = False

    video_name = os.path.splitext(
        os.path.basename(args.source)
    )[0]

    csv_path = os.path.join(
        args.output,
        f"{video_name}_predictions.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "timestamp",
            "torso_angle",
            "hip_speed",
            "state"
        ])

        while True:

            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_number / fps

            height, width = frame.shape[:2]

            result = model.predict(
                frame,
                imgsz=640,
                conf=0.35,
                device=device,
                verbose=False
            )[0]

            output = result.plot()

            state = "NO_PERSON"
            angle = None
            hip_speed = None

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
                        width,
                        height
                    )

                    if features is not None:

                        detection = detector.update(
                            angle=features["torso_angle_deg"],
                            hip=features["normalized_hip"],
                            timestamp=timestamp
                        )

                        state = detection["state"]

                        angle = features["torso_angle_deg"]
                        hip_speed = detection["hip_speed"]

                        if state == "FALL_DETECTED":
                            fall_detected = True

                else:
                    detector.reset()

            else:
                detector.reset()

            writer.writerow([
                frame_number,
                round(timestamp, 3),
                angle,
                hip_speed,
                state
            ])

            color = (
                (0, 0, 255)
                if state == "FALL_DETECTED"
                else (0, 255, 0)
                if state == "NORMAL"
                else (0, 165, 255)
            )

            cv2.putText(
                output,
                f"Status: {state}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            if args.display:

                cv2.imshow(
                    "Fall Detection - Video Test",
                    output
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\nVIDEO TEST COMPLETED")
    print("Video:", args.source)
    print("Frames processed:", frame_number)
    print("FPS:", fps)
    print("Fall event flagged:", fall_detected)
    print("CSV saved:", csv_path)


if __name__ == "__main__":
    main()
