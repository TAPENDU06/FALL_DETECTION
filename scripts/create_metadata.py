
import cv2
import csv
import re
from pathlib import Path

DATA_ROOT = Path("data/raw/urfd")
OUTPUT_FILE = Path("data/dataset_metadata.csv")


def get_video_information(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Cannot open:", video_path)
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        frame_count / fps
        if fps > 0 else 0
    )

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration
    }


def identify_video(video_path):

    filename = video_path.name.lower()

    match = re.match(
        r"(fall|adl)-(\d+)-cam(\d+)\.mp4$",
        filename
    )

    if not match:
        return None

    category = match.group(1)
    sequence_number = match.group(2)
    camera_id = "cam" + match.group(3)

    sequence_id = f"{category}-{sequence_number}"

    if camera_id == "cam0":
        camera_position = "floor_parallel"
    elif camera_id == "cam1":
        camera_position = "ceiling"
    else:
        camera_position = "unknown"

    return {
        "sequence_id": sequence_id,
        "category": "fall" if category == "fall" else "normal",
        "camera_id": camera_id,
        "camera_position": camera_position
    }


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    videos = sorted(DATA_ROOT.rglob("*.mp4"))

    print("Videos found:", len(videos))

    for video_path in videos:

        identity = identify_video(video_path)

        if identity is None:
            print("SKIPPED: Unexpected filename:", video_path)
            continue

        information = get_video_information(video_path)

        if information is None:
            continue

        row = {
            **identity,
            "filename": video_path.name,
            "video_path": video_path.as_posix(),
            "container_width": information["width"],
            "container_height": information["height"],
            "rgb_width": information["width"] // 2,
            "rgb_height": information["height"],
            "fps": information["fps"],
            "frame_count": information["frame_count"],
            "duration_seconds": round(
                information["duration"], 3
            ),
            "frame_layout": "depth_left_rgb_right"
        }

        rows.append(row)

        print(
            "Verified:",
            video_path.name,
            information["width"],
            "x",
            information["height"]
        )

    if not rows:
        print("No valid videos found.")
        return

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nMETADATA GENERATION COMPLETED")
    print("Total valid videos:", len(rows))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
