
import cv2
import csv
import os
from ultralytics import YOLO

# CONFIGURATION
VIDEO_PATH = r"D:\Fall_Dataset\Fall\front_cam\fall-30-cam0.mp4"
SEQUENCE_ID = "fall-30"
CAMERA_ID = "cam0"

MODEL_PATH = "yolo26n-pose.pt"
CSV_PATH = "data/annotations/urfd_annotations.csv"

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total frames:", total_frames)
print("FPS:", fps)

selected = {}

print("\nControls:")
print("A = Save current frame as BEFORE FALL")
print("D = Save current frame as AFTER FALL")
print("N = Next frame")
print("P = Previous frame")
print("Q = Finish and save CSV")

frame_no = 0

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        f"Frame: {frame_no}/{total_frames-1}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Fall Annotation", annotated)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        frame_no = min(frame_no + 1, total_frames - 1)

    elif key == ord('p'):
        frame_no = max(frame_no - 1, 0)

    elif key == ord('a'):
        selected["normal"] = frame_no
        print("Before-fall frame:", frame_no)

    elif key == ord('d'):
        selected["post_fall"] = frame_no
        print("After-fall frame:", frame_no)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save annotations
if len(selected) == 2:
    file_exists = os.path.exists(CSV_PATH)

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    with open(CSV_PATH, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(CSV_PATH) == 0:
            writer.writerow([
                "sequence_id",
                "camera_id",
                "start_frame",
                "end_frame",
                "label"
            ])

        for label in ["normal", "post_fall"]:
            frame = selected[label]

            writer.writerow([
                SEQUENCE_ID,
                CAMERA_ID,
                frame,
                frame,
                label
            ])

    print("Annotations saved successfully!")

else:
    print("Both frames were not selected. CSV not updated.")
