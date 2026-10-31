
import numpy as np


class FallDetector:

    def __init__(
        self,
        upright_angle=30.0,
        horizontal_angle=55.0,
        angular_speed_threshold=60.0,
        hip_speed_threshold=0.6,
        upright_memory=1.5,
        confirmation_time=0.8,
        cooldown_time=2.0
    ):
        # All thresholds are initial experimental values.
        # They must be calibrated using validation videos.

        self.upright_angle = upright_angle
        self.horizontal_angle = horizontal_angle
        self.angular_speed_threshold = angular_speed_threshold
        self.hip_speed_threshold = hip_speed_threshold

        self.upright_memory = upright_memory
        self.confirmation_time = confirmation_time
        self.cooldown_time = cooldown_time

        self.previous_angle = None
        self.previous_hip = None
        self.previous_time = None

        self.last_upright_time = None
        self.candidate_start = None
        self.cooldown_start = None

        self.state = "NORMAL"

    def reset(self):
        self.previous_angle = None
        self.previous_hip = None
        self.previous_time = None

        self.last_upright_time = None
        self.candidate_start = None
        self.cooldown_start = None

        self.state = "NORMAL"

    def update(self, angle, hip, timestamp):

        hip = np.asarray(hip, dtype=float)

        angular_speed = 0.0
        hip_speed = 0.0

        if self.previous_time is not None:
            dt = timestamp - self.previous_time

            if dt > 0:
                if self.previous_angle is not None:
                    angular_speed = abs(
                        angle - self.previous_angle
                    ) / dt

                if self.previous_hip is not None:
                    hip_speed = (
                        np.linalg.norm(hip - self.previous_hip) / dt
                    )

        # Remember recent upright posture.
        if angle <= self.upright_angle:
            self.last_upright_time = timestamp

        # Handle cooldown.
        if self.state == "COOLDOWN":
            if timestamp - self.cooldown_start >= self.cooldown_time:
                self.state = "NORMAL"
                self.candidate_start = None

        if self.state == "NORMAL":

            recently_upright = (
                self.last_upright_time is not None
                and timestamp - self.last_upright_time
                <= self.upright_memory
            )

            rapid_change = (
                angular_speed >= self.angular_speed_threshold
                or hip_speed >= self.hip_speed_threshold
            )

            horizontal_posture = angle >= self.horizontal_angle

            if recently_upright and rapid_change and horizontal_posture:
                self.state = "POSSIBLE_FALL"
                self.candidate_start = timestamp

        elif self.state == "POSSIBLE_FALL":

            # Require the low posture to persist.
            if angle < self.horizontal_angle - 10:
                self.state = "NORMAL"
                self.candidate_start = None

            elif timestamp - self.candidate_start >= self.confirmation_time:
                self.state = "FALL_DETECTED"

                # Start cooldown after issuing the event.
                self.cooldown_start = timestamp

        elif self.state == "FALL_DETECTED":
            # The event has been issued; prevent repeated alerts.
            self.state = "COOLDOWN"

        self.previous_angle = angle
        self.previous_hip = hip.copy()
        self.previous_time = timestamp

        return {
            "state": self.state,
            "angular_speed": angular_speed,
            "hip_speed": hip_speed
        }
