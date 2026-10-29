
import numpy as np


# COCO keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


def extract_features(keypoints, frame_width, frame_height):
    """
    Extract simple posture features from one person's pose.

    keypoints: NumPy array with shape (17, 2), containing x and y.
    Returns a dictionary, or None if required points are missing.
    """

    if keypoints is None or keypoints.shape[0] < 17:
        return None

    # The keypoints should already be in image pixel coordinates.
    points = np.asarray(keypoints, dtype=float)

    required = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
    ]

    # Check that required coordinates are valid.
    for index in required:
        x, y = points[index]
        if not np.isfinite(x) or not np.isfinite(y):
            return None

    left_shoulder = points[LEFT_SHOULDER]
    right_shoulder = points[RIGHT_SHOULDER]
    left_hip = points[LEFT_HIP]
    right_hip = points[RIGHT_HIP]

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    hip_center = (
        left_hip + right_hip
    ) / 2.0

    dx = shoulder_center[0] - hip_center[0]
    dy = hip_center[1] - shoulder_center[1]

    # Angle measured relative to the image's vertical direction.
    torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy)))

    # Normalize coordinates to make movement more comparable
    # across different image resolutions.
    diagonal = np.hypot(frame_width, frame_height)

    normalized_hip = hip_center / diagonal

    return {
        "torso_angle_deg": float(torso_angle),
        "hip_x": float(hip_center[0]),
        "hip_y": float(hip_center[1]),
        "normalized_hip": normalized_hip,
    }
