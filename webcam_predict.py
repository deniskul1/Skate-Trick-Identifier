import os
import sys
import tempfile

import cv2
import mediapipe as mp

from predict_trick import predict_video, MODEL_PATH
from quiet_mediapipe import create_pose_model

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

MIN_RECORDED_FRAMES = 5


def predict_from_frames(frames, fps):

    frame_height, frame_width = frames[0].shape[:2]

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    try:
        return predict_video(temp_path)
    finally:
        os.remove(temp_path)


def play_recorded_clip(pose_model, frames, fps, label_text):

    wait_ms = max(1, int(1000 / fps))

    for frame in frames:
        frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.rectangle(frame, (0, 0), (630, 40), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Skateboard Trick Classifier - Webcam", frame)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            return False
        if key != 255:
            # Any other key skips the rest of the replay early.
            break

    return True


def run_webcam_session(camera_index=0):
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model yet ('{MODEL_PATH}' doesn't exist).")
        print("Build the dataset and train a model first (build_dataset.py "
              "+ train_classifier.py, or menu.py).")
        return

    pose_model = create_pose_model(
        mp_pose,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        pose_model.close()
        return

    # Webcams often don't report a reliable frame rate through OpenCV, so
    # fall back to a common default rather than writing a video file with
    # a bogus (or zero) fps.
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print("SPACE = start/stop recording a trick attempt.  Q = quit "
          "(works during playback too).")

    is_recording = False
    recorded_frames = []
    last_result_text = None
    quit_requested = False

    while not quit_requested:
        success, frame = video_capture.read()
        if not success:
            print("Could not read from camera.")
            break

        if is_recording:
            recorded_frames.append(frame.copy())
            status_text = f"RECORDING ({len(recorded_frames)} frames) - SPACE to stop"
            status_color = (0, 0, 255)
        else:
            status_text = "Ready - SPACE to record a trick attempt"
            status_color = (0, 255, 0)

        # A filled rectangle behind the text keeps it readable regardless
        # of what's in the background of the shot.
        cv2.rectangle(frame, (0, 0), (630, 70), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        if last_result_text:
            cv2.putText(frame, last_result_text, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Skateboard Trick Classifier - Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            if not is_recording:
                is_recording = True
                recorded_frames = []
                last_result_text = None
            else:
                is_recording = False
                if len(recorded_frames) < MIN_RECORDED_FRAMES:
                    print("Recording too short — hold the attempt a little "
                          "longer next time.")
                    last_result_text = "Too short, try again"
                else:
                    print(f"Analyzing {len(recorded_frames)} recorded frames...")
                    try:
                        predicted_trick, probabilities = predict_from_frames(
                            recorded_frames, fps
                        )
                        confidence = probabilities[predicted_trick]
                        last_result_text = (
                            f"Predicted: {predicted_trick} ({confidence:.0%})"
                        )
                        print(last_result_text)

                        print("Replaying your clip... (Q to quit, any other "
                              "key to skip)")
                        keep_going = play_recorded_clip(
                            pose_model, recorded_frames, fps, last_result_text
                        )
                        if not keep_going:
                            quit_requested = True
                    except ValueError as error:
                        print(f"Couldn't analyze that clip: {error}")
                        last_result_text = "Couldn't get a clear read, try again"
                recorded_frames = []

    video_capture.release()
    cv2.destroyAllWindows()

    for _ in range(4):
        cv2.waitKey(1)
    pose_model.close()
    print("Webcam closed.")


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_webcam_session(camera_index)


if __name__ == "__main__":
    main()
