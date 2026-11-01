import cv2

video_path = "data/raw/urfd/fall/front/fall-01-cam0.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Width:", width)
    print("Height:", height)
    print("FPS:", fps)
    print("Total frames:", frames)
    print("Duration:", frames / fps if fps > 0 else 0)

cap.release()