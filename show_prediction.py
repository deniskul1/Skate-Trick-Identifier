
import sys

import cv2
import mediapipe as mp

# Reusing the exact same prediction logic as predict_trick.py, rather than
# a second copy of it here — one function loading the model and running
# inference means there's only one place that logic can go wrong.
from predict_trick import predict_video
from quiet_mediapipe import create_pose_model

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def play_video_with_prediction(video_path, predicted_trick, confidence):
    pose_model = create_pose_model(
        mp_pose,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file at '{video_path}'.")
        return

    label_text = f"Predicted: {predicted_trick} ({confidence:.0%})"
    print("Press 'q' in the video window to quit early.")

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # A filled rectangle behind the text keeps the prediction readable
        # no matter what's going on in the background of the shot.
        cv2.rectangle(frame, (0, 0), (430, 40), (0, 0, 0), -1)
        cv2.putText(
            frame, label_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("Skateboard Trick Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit key pressed. Stopping early.")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    pose_model.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 show_prediction.py <path_to_video>")
        print("Example: python3 show_prediction.py trick.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print(f"Analyzing '{video_path}'...")
    try:
        predicted_trick, probabilities = predict_video(video_path)
    except FileNotFoundError as error:
        print(f"\n{error}")
        print("(Make sure the model is trained — see train_classifier.py "
              "or menu.py — and the video path is correct.)")
        sys.exit(1)

    confidence = probabilities[predicted_trick]
    print(f"Predicted trick: {predicted_trick} ({confidence:.0%} confidence)")

    play_video_with_prediction(video_path, predicted_trick, confidence)


if __name__ == "__main__":
    main()
