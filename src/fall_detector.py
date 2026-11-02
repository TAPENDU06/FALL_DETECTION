
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
        cooldown_time=2.0,
        max_missing_duration=1.0,
        recovery_angle=40.0,
        recovery_time=0.4
    ):

        # Initial experimental thresholds.
        # Calibrate these using validation recordings.

        self.upright_angle = upright_angle
        self.horizontal_angle = horizontal_angle
        self.angular_speed_threshold = angular_speed_threshold
        self.hip_speed_threshold = hip_speed_threshold

        self.upright_memory = upright_memory
        self.confirmation_time = confirmation_time
        self.cooldown_time = cooldown_time

        # New parameters for missing keypoints and recovery.
        self.max_missing_duration = max_missing_duration
        self.recovery_angle = recovery_angle
        self.recovery_time = recovery_time

        self.previous_angle = None
        self.previous_hip = None
        self.previous_time = None

        self.last_upright_time = None
        self.candidate_start = None
        self.cooldown_start = None

        self.missing_start = None
        self.recovery_start = None

        self.state = "NORMAL"

    def reset(self):

        self.previous_angle = None
        self.previous_hip = None
        self.previous_time = None

        self.last_upright_time = None
        self.candidate_start = None
        self.cooldown_start = None

        self.missing_start = None
        self.recovery_start = None

        self.state = "NORMAL"

    def handle_missing(self, timestamp):
        """
        Handle frames where a reliable pose is unavailable.

        A short missing-keypoint gap does not immediately
        destroy an existing POSSIBLE_FALL candidate.
        """

        if self.missing_start is None:
            self.missing_start = timestamp

        # Do not calculate speed across missing frames.
        # The next valid observation starts a fresh measurement.
        self.previous_angle = None
        self.previous_hip = None
        self.previous_time = None

        missing_duration = timestamp - self.missing_start

        # Cancel a candidate only if the missing gap
        # exceeds the allowed duration.
        if (
            self.state == "POSSIBLE_FALL"
            and missing_duration > self.max_missing_duration
        ):
            self.state = "NORMAL"
            self.candidate_start = None
            self.recovery_start = None

        # Handle cooldown even when keypoints are missing.
        if self.state == "COOLDOWN":

            if (
                self.cooldown_start is not None
                and timestamp - self.cooldown_start
                >= self.cooldown_time
            ):
                self.state = "NORMAL"
                self.candidate_start = None

        return {
            "state": self.state,
            "angular_speed": 0.0,
            "hip_speed": 0.0
        }

    def update(self, angle, hip, timestamp):

        hip = np.asarray(hip, dtype=float)

        # A valid pose has been received again.
        self.missing_start = None

        angular_speed = 0.0
        hip_speed = 0.0

        # Calculate movement speed only when consecutive
        # valid observations are available.
        if self.previous_time is not None:

            dt = timestamp - self.previous_time

            if dt > 0:

                if self.previous_angle is not None:
                    angular_speed = abs(
                        angle - self.previous_angle
                    ) / dt

                if self.previous_hip is not None:
                    hip_speed = (
                        np.linalg.norm(
                            hip - self.previous_hip
                        ) / dt
                    )

        # Remember recent upright posture.
        if angle <= self.upright_angle:
            self.last_upright_time = timestamp

        # -----------------------------------
        # COOLDOWN
        # -----------------------------------

        if self.state == "COOLDOWN":

            if (
                self.cooldown_start is not None
                and timestamp - self.cooldown_start
                >= self.cooldown_time
            ):
                self.state = "NORMAL"
                self.candidate_start = None

        # -----------------------------------
        # NORMAL
        # -----------------------------------

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

            horizontal_posture = (
                angle >= self.horizontal_angle
            )

            if (
                recently_upright
                and rapid_change
                and horizontal_posture
            ):

                self.state = "POSSIBLE_FALL"
                self.candidate_start = timestamp
                self.recovery_start = None

        # -----------------------------------
        # POSSIBLE FALL
        # -----------------------------------

        elif self.state == "POSSIBLE_FALL":

            candidate_duration = (
                timestamp - self.candidate_start
            )

            # Confirm only when:
            # 1. The candidate has persisted long enough.
            # 2. The person is sufficiently horizontal.

            if (
                candidate_duration >= self.confirmation_time
                and angle >= self.horizontal_angle
            ):

                self.state = "FALL_DETECTED"
                self.cooldown_start = timestamp
                self.recovery_start = None

            # Do not immediately cancel a candidate
            # because of one temporarily low angle.
            elif angle <= self.recovery_angle:

                if self.recovery_start is None:
                    self.recovery_start = timestamp

                elif (
                    timestamp - self.recovery_start
                    >= self.recovery_time
                ):

                    # Person has remained in a recovered/
                    # non-horizontal posture long enough.
                    self.state = "NORMAL"
                    self.candidate_start = None
                    self.recovery_start = None

            else:
                self.recovery_start = None

        # -----------------------------------
        # FALL DETECTED
        # -----------------------------------

        elif self.state == "FALL_DETECTED":

            # Move into cooldown after issuing the event.
            self.state = "COOLDOWN"

        # Save the current valid observation.
        self.previous_angle = angle
        self.previous_hip = hip.copy()
        self.previous_time = timestamp

        return {
            "state": self.state,
            "angular_speed": angular_speed,
            "hip_speed": hip_speed
        }
