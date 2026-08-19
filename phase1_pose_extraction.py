
import cv2
import mediapipe as mp

from quiet_mediapipe import create_pose_model

mp_drawing = mp.solutions.drawing_utils

pose_model = create_pose_model(
    mp_pose,
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
VIDEO_PATH = "skate_video.mp4"


video_capture = cv2.VideoCapture(VIDEO_PATH)

if not video_capture.isOpened():
    print(f"Error: Could not open video file at '{VIDEO_PATH}'.")
    print("Check that the file exists and the path is correct.")
    exit()

print("Video opened successfully. Press 'q' in the video window to quit early.")

while video_capture.isOpened():

    success, frame = video_capture.read()

    if not success:
        print("Reached the end of the video (or failed to read a frame).")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose_model.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,                          # image to draw on
            results.pose_landmarks,         # the detected landmarks
            mp_pose.POSE_CONNECTIONS,       # which landmarks to connect with lines

        )

    cv2.imshow("Skateboard Trick - Pose Estimation (Phase 1)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit key pressed. Stopping early.")
        break

video_capture.release()
cv2.destroyAllWindows()
pose_model.close()

print("Done.")
