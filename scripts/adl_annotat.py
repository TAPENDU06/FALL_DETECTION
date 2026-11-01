
import cv2
import csv
import os
import glob

VIDEO_FOLDER = r"D:\Fall_Dataset\No fall\front_cam"
CSV_PATH = "data/annotations/urfd_annotations.csv"

# Find all ADL videos
videos = sorted(glob.glob(os.path.join(VIDEO_FOLDER, "*.mp4")))

# Read existing annotations
existing = set()

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            existing.add((row["sequence_id"], row["camera_id"], row["label"]))

added = 0
skipped = 0

with open(CSV_PATH, "a", newline="") as file:
    writer = csv.writer(file)

    for video in videos:
        filename = os.path.basename(video)
        sequence_id = filename.replace("-cam0.mp4", "")

        if (sequence_id, "cam0", "normal") in existing:
            print("Already annotated:", sequence_id)
            skipped += 1
            continue

        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            print("Cannot open:", filename)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 0:
            print("Invalid frame count:", filename)
            continue

        writer.writerow([
            sequence_id,
            "cam0",
            0,
            total_frames - 1,
            "normal"
        ])

        existing.add((sequence_id, "cam0", "normal"))
        added += 1

        print(f"Added {sequence_id}: 0-{total_frames-1}")

print("\nADL annotation completed!")
print("New videos annotated:", added)
print("Already present:", skipped)
print("Total videos found:", len(videos))
