

import cv2
import time
import torch
from ultralytics import YOLO

from features import extract_features
from fall_detector import FallDetector


def main():

    model = YOLO("yolo26n-pose.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    detector = FallDetector()

    print("Fall detection started.")
    print("Press Q to exit.")

    while True:

        success, frame = cap.read()

        if not success:
            break

        timestamp = time.monotonic()

        height, width = frame.shape[:2]

        result = model.predict(
            frame,
            imgsz=640,
            conf=0.35,
            device=device,
            verbose=False
        )[0]

        output = result.plot()

        status = "NORMAL"
        angle_text = "N/A"
        hip_speed_text = "N/A"

        if result.keypoints is not None and len(result.keypoints) > 0:

            keypoints = result.keypoints.xy[0].cpu().numpy()

            confidence = (
                result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )

            required = [5, 6, 11, 12]

            valid_pose = (
                confidence is not None
                and all(confidence[i] >= 0.35 for i in required)
            )

            if valid_pose:

                features = extract_features(
                    keypoints, width, height
                )

                if features is not None:

                    detection = detector.update(
                        angle=features["torso_angle_deg"],
                        hip=features["normalized_hip"],
                        timestamp=timestamp
                    )

                    status = detection["state"]

                    angle_text = (
                        f"{features['torso_angle_deg']:.1f} deg"
                    )

                    hip_speed_text = (
                        f"{detection['hip_speed']:.3f}"
                    )

                    if status == "FALL_DETECTED":
                        print(
                            "POSSIBLE FALL DETECTED AT:",
                            time.strftime("%H:%M:%S")
                        )

            else:
                detector.reset()
                status = "UNRELIABLE POSE"

        else:
            detector.reset()
            status = "NO PERSON DETECTED"

        color = (
            (0, 0, 255)
            if status == "FALL_DETECTED"
            else (0, 255, 0)
            if status == "NORMAL"
            else (0, 165, 255)
        )

        cv2.putText(
            output,
            f"Status: {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

        cv2.putText(
            output,
            f"Torso angle: {angle_text}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Hip speed: {hip_speed_text}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Fall Detection System", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
