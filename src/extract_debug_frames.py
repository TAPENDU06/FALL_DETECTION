
import cv2
from pathlib import Path


VIDEOS = {
    "fall-04": {
        "path": "data/raw/urfd/fall/front/fall-04-cam0.mp4",
        "times": [0.7, 0.9, 1.1, 1.3, 1.45, 1.8, 2.3, 3.0]
    },

    "adl-04": {
        "path": "data/raw/urfd/normal/front/adl-04-cam0.mp4",
        "times": [2.3, 2.5, 2.7, 2.9, 3.1, 3.23, 3.5, 4.0]
    }
}


OUTPUT = Path("results/debug_frames")


def main():

    for name, info in VIDEOS.items():

        cap = cv2.VideoCapture(info["path"])

        if not cap.isOpened():
            print("Cannot open:", info["path"])
            continue

        folder = OUTPUT / name
        folder.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)

        print("\nExtracting:", name)

        for timestamp in info["times"]:

            frame_number = int(timestamp * fps)

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number
            )

            success, frame = cap.read()

            if not success:
                continue

            # Extract RGB panel (right half).
            height, width = frame.shape[:2]
            rgb_frame = frame[:, width // 2:, :]

            cv2.putText(
                rgb_frame,
                f"Time: {timestamp:.2f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            output_path = folder / f"frame_{timestamp:.2f}.jpg"

            cv2.imwrite(
                str(output_path),
                rgb_frame
            )

            print("Saved:", output_path)

        cap.release()

    print("\nFRAME EXTRACTION COMPLETED")


if __name__ == "__main__":
    main()
