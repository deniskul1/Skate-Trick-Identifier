import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from quiet_mediapipe import create_pose_model

TRICKS_DIR = "tricks"
DATASET_PATH = "trick_dataset.csv"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}

mp_pose = mp.solutions.pose

LANDMARKS_OF_INTEREST = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    "left_foot_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    "right_foot_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}

LANDMARKS_FOR_RAW_SIGNALS = [
    "left_hip", "right_hip",
    "left_ankle", "right_ankle",
    "left_foot_index", "right_foot_index",
]


def extract_raw_landmark_positions(video_path):

    pose_model = create_pose_model(
        mp_pose,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_positions = []

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)

        if results.pose_landmarks is None:
            continue

        frame_height, frame_width, _ = frame.shape
        positions_this_frame = {}
        for name, landmark_id in LANDMARKS_OF_INTEREST.items():
            landmark = results.pose_landmarks.landmark[landmark_id]
            pixel_x = landmark.x * frame_width
            pixel_y = landmark.y * frame_height
            positions_this_frame[name] = (pixel_x, pixel_y)

        frame_positions.append(positions_this_frame)

    video_capture.release()
    pose_model.close()

    return frame_positions


def normalize_positions(frame_positions):

    hip_centers_x = []
    hip_centers_y = []
    torso_lengths = []

    for positions in frame_positions:
        left_hip_x, left_hip_y = positions["left_hip"]
        right_hip_x, right_hip_y = positions["right_hip"]
        hip_center_x = (left_hip_x + right_hip_x) / 2.0
        hip_center_y = (left_hip_y + right_hip_y) / 2.0

        left_shoulder_x, left_shoulder_y = positions["left_shoulder"]
        right_shoulder_x, right_shoulder_y = positions["right_shoulder"]
        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2.0
        shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2.0

        torso_length = math.hypot(
            shoulder_center_x - hip_center_x,
            shoulder_center_y - hip_center_y,
        )

        hip_centers_x.append(hip_center_x)
        hip_centers_y.append(hip_center_y)
        torso_lengths.append(torso_length)

    reference_x = pd.Series(hip_centers_x).median()
    reference_y = pd.Series(hip_centers_y).median()
    reference_scale = pd.Series(torso_lengths).median()

    # Guard against a degenerate video where torso length came out as zero
    # (would cause a divide by zero below).
    if reference_scale == 0:
        reference_scale = 1.0

    normalized_series = {name: [] for name in LANDMARKS_OF_INTEREST}
    for positions in frame_positions:
        for name in LANDMARKS_OF_INTEREST:
            raw_x, raw_y = positions[name]
            normalized_x = (raw_x - reference_x) / reference_scale
            normalized_y = (raw_y - reference_y) / reference_scale
            normalized_series[name].append((normalized_x, normalized_y))

    return normalized_series


def build_signal_series(normalized_series):

    signals = {}

    for name in LANDMARKS_FOR_RAW_SIGNALS:
        coordinate_list = normalized_series[name]
        signals[f"{name}_x"] = [point[0] for point in coordinate_list]
        signals[f"{name}_y"] = [point[1] for point in coordinate_list]

    num_frames = len(normalized_series["left_hip"])

    foot_spread = []
    foot_height_diff = []
    shoulder_angle_rad = []
    foot_angle_rad = []
    left_foot_pitch_rad = []
    right_foot_pitch_rad = []

    for frame_index in range(num_frames):
        left_foot_x, left_foot_y = normalized_series["left_foot_index"][frame_index]
        right_foot_x, right_foot_y = normalized_series["right_foot_index"][frame_index]
        left_ankle_x, left_ankle_y = normalized_series["left_ankle"][frame_index]
        right_ankle_x, right_ankle_y = normalized_series["right_ankle"][frame_index]
        left_shoulder_x, left_shoulder_y = normalized_series["left_shoulder"][frame_index]
        right_shoulder_x, right_shoulder_y = normalized_series["right_shoulder"][frame_index]

        # How far apart the two feet are, horizontally. Tricks with a big
        # flick/kick motion tend to momentarily widen or narrow this.
        foot_spread.append(abs(left_foot_x - right_foot_x))

        # Which foot is higher than the other. Useful for spotting the
        # asymmetric foot motion of a kickflip vs. the more synchronized
        # motion of an ollie.
        foot_height_diff.append(left_ankle_y - right_ankle_y)

        # The angle of the line connecting the shoulders, relative to
        # horizontal. Captures the skater's upper-body rotation, which
        # differs between a shove-it and an ollie (body stays relatively square).
        shoulder_angle_rad.append(math.atan2(
            right_shoulder_y - left_shoulder_y,
            right_shoulder_x - left_shoulder_x,
        ))

        # The angle of the line connecting the two feet, relative to
        # horizontal. Acts as a rough proxy for how the board itself is
        # rotating underneath the skater
        foot_angle_rad.append(math.atan2(
            right_foot_y - left_foot_y,
            right_foot_x - left_foot_x,
        ))

        # The angle of the ankle to toe line for each foot, relative to
        # horizontal. Best proxy for how much a given foot is tilting/flicking
        left_foot_pitch_rad.append(math.atan2(
            left_ankle_y - left_foot_y,
            left_ankle_x - left_foot_x,
        ))
        right_foot_pitch_rad.append(math.atan2(
            right_ankle_y - right_foot_y,
            right_ankle_x - right_foot_x,
        ))

    # atan2 only ever returns a value between -180 and +180 degrees, so if
    # the true angle drifts past that boundary (e.g. from 179 degrees to
    # -179 degrees, a tiny real change), the raw values jump instantly by
    # ~360 degrees. That fake jump would wreck the mean/net change
    # statistics we compute later. np.unwrap detects those jumps and adds
    # back whole rotations so the signal changes smoothly, matching the
    # skater's actual rotation instead of an artifact of the angle format.
    signals["shoulder_angle_deg"] = np.degrees(np.unwrap(shoulder_angle_rad))
    signals["foot_angle_deg"] = np.degrees(np.unwrap(foot_angle_rad))
    signals["left_foot_pitch_deg"] = np.degrees(np.unwrap(left_foot_pitch_rad))
    signals["right_foot_pitch_deg"] = np.degrees(np.unwrap(right_foot_pitch_rad))

    signals["foot_spread"] = foot_spread
    signals["foot_height_diff"] = foot_height_diff

    return signals


def summarize_signals(signals):

    features = {}

    for signal_name, values in signals.items():
        series = pd.Series(values)
        features[f"{signal_name}_mean"] = series.mean()
        features[f"{signal_name}_std"] = series.std()
        features[f"{signal_name}_min"] = series.min()
        features[f"{signal_name}_max"] = series.max()
        # We don't also store a separate "range" (max - min) statistic:
        # since min and max are already both columns, range would just be
        # a fixed linear combination of them, no new information, just
        # one more column for the model to potentially overfit to.
        #
        # "Net change": where the signal ended up relative to where it
        # started. Distinguishes, for example, a trick where the feet end
        # up back where they started vs. one where they end up rotated.
        features[f"{signal_name}_net_change"] = series.iloc[-1] - series.iloc[0]

    return features


def extract_features_from_video(video_path):
    frame_positions = extract_raw_landmark_positions(video_path)

    if len(frame_positions) < 2:
        raise ValueError(
            f"Only detected a person in {len(frame_positions)} frame(s) of "
            f"'{video_path}'. Need at least a couple of frames with a clear "
            f"detection to compute meaningful features."
        )

    normalized_series = normalize_positions(frame_positions)
    signals = build_signal_series(normalized_series)
    features = summarize_signals(signals)
    return features


def find_labeled_videos(tricks_dir):

    tricks_path = Path(tricks_dir)

    if not tricks_path.is_dir():
        raise FileNotFoundError(
            f"Could not find a '{tricks_dir}/' folder. Create it, then add "
            f"one subfolder per trick (e.g. {tricks_dir}/Ollie, "
            f"{tricks_dir}/Kickflip) containing that trick's video clips."
        )

    labeled_videos = []
    for trick_folder in sorted(tricks_path.iterdir()):
        if not trick_folder.is_dir():
            continue

        trick_label = trick_folder.name
        for video_path in sorted(trick_folder.iterdir()):
            if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                labeled_videos.append((video_path, trick_label))

    return labeled_videos


def main():

    labeled_videos = find_labeled_videos(TRICKS_DIR)

    if not labeled_videos:
        print(f"No video files found under '{TRICKS_DIR}/<TrickName>/'.")
        print(f"Add clips like '{TRICKS_DIR}/Ollie/my_clip.mp4' and try again.")
        return False

    trick_names = sorted(set(label for _, label in labeled_videos))
    print(f"Found {len(labeled_videos)} clip(s) across {len(trick_names)} "
          f"trick(s): {', '.join(trick_names)}\n")

    rows = []
    for video_path, trick_label in labeled_videos:
        print(f"  {video_path}  ->  '{trick_label}' ...", end=" ")
        try:
            features = extract_features_from_video(str(video_path))
        except ValueError as error:
            print(f"SKIPPED ({error})")
            continue

        features["trick"] = trick_label
        features["source_video"] = str(video_path)
        rows.append(features)
        print("done")

    dataset = pd.DataFrame(rows)
    dataset.to_csv(DATASET_PATH, index=False)

    print(f"\nSaved {len(dataset)} labeled example(s) to '{DATASET_PATH}':")
    print(dataset["trick"].value_counts().to_string())
    return True


if __name__ == "__main__":
    main()
