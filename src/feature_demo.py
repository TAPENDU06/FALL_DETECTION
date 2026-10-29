
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from features import extract_features


def parse_source(value):
    """Convert numeric webcam sources to integers; keep video paths as strings."""
    if value.isdigit():
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(
        description="YOLO pose feature extraction from webcam or video."
    )
    parser.add_argument(
        "--source",
        type=parse_source,
        default=0,
        help='Webcam index or video path. Default: 0 (webcam).'
    )
    args = parser.parse_args()

    model = YOLO("yolo26n-pose.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {args.source}"
        )

    previous_hip = None

    print(f"Feature extraction started. Source: {args.source}")
    print("Press Q to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            height, width = frame.shape[:2]

            result = model.predict(
                frame,
                imgsz=640,
                conf=0.35,
                device=device,
                verbose=False
            )[0]

            output = result.plot()

            if result.keypoints is not None and len(result.keypoints) > 0:
                # Use the first detected person for this baseline.
                keypoints = (
                    result.keypoints.xy[0]
                    .cpu()
                    .numpy()
                )

                confidence = None
                if result.keypoints.conf is not None:
                    confidence = (
                        result.keypoints.conf[0]
                        .cpu()
                        .numpy()
                    )

                # Require reliable shoulder and hip keypoints.
                required = [5, 6, 11, 12]
                valid_pose = (
                    confidence is not None
                    and all(confidence[i] >= 0.35 for i in required)
                )

                features = None
                if valid_pose:
                    features = extract_features(
                        keypoints, width, height
                    )

                if features is not None:
                    current_hip = features["normalized_hip"]

                    if previous_hip is not None:
                        displacement = float(
                            np.linalg.norm(
                                current_hip - previous_hip
                            )
                        )
                    else:
                        displacement = 0.0

                    previous_hip = current_hip.copy()

                    cv2.putText(
                        output,
                        f"Torso angle: {features['torso_angle_deg']:.1f} deg",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        output,
                        f"Hip displacement: {displacement:.4f}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                else:
                    previous_hip = None
                    cv2.putText(
                        output,
                        "Pose keypoints unreliable",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2
                    )
            else:
                previous_hip = None
                cv2.putText(
                    output,
                    "No person detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Pose Features - Fall Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
