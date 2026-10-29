
import argparse
import cv2
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()

    # Load a lightweight pretrained human pose model.
    model = YOLO("yolo26n-pose.pt")

    # Verify GPU availability and select the device.
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Inference device:", device)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print("Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.35,
            device=device,
            verbose=False
        )

        result = results[0]
        output = result.plot()

        people = 0
        if result.keypoints is not None:
            people = len(result.keypoints)

        cv2.putText(
            output,
            f"People detected: {people}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Human Pose Estimation", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
