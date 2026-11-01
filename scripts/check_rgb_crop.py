
import cv2

video_path = "data/raw/urfd/fall/front/fall-01-cam0.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

ret, frame = cap.read()

if ret:
    height, width = frame.shape[:2]

    rgb_frame = frame[:, width // 2:, :]

    print("Original dimensions:", width, "x", height)
    print("RGB dimensions:", rgb_frame.shape[1], "x", rgb_frame.shape[0])

    cv2.imshow("Extracted RGB Frame", rgb_frame)
    cv2.imwrite("rgb_test_frame.jpg", rgb_frame)

    cv2.waitKey(0)
else:
    print("Could not read frame")

cap.release()
cv2.destroyAllWindows()
