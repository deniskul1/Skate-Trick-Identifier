
import shutil
from pathlib import Path

import build_dataset
import train_classifier
from build_dataset import TRICKS_DIR, VIDEO_EXTENSIONS
from predict_trick import predict_video, MODEL_PATH
from show_prediction import play_video_with_prediction
from webcam_predict import run_webcam_session


def ask(prompt_text):
    return input(prompt_text).strip()


def list_existing_tricks():
    tricks_path = Path(TRICKS_DIR)
    if not tricks_path.is_dir():
        return []
    return sorted(folder.name for folder in tricks_path.iterdir() if folder.is_dir())


def analyze_video():
    print()
    if not Path(MODEL_PATH).exists():
        print(f"No trained model yet ('{MODEL_PATH}' doesn't exist).")
        print("Choose option 4 to build the dataset and train one first.\n")
        return

    video_path = ask("Path to the video you want to analyze: ")
    if not video_path:
        print("No path entered — cancelled.\n")
        return
    if not Path(video_path).is_file():
        print(f"Couldn't find a file at '{video_path}'.\n")
        return

    print(f"\nAnalyzing '{video_path}'...")
    try:
        predicted_trick, probabilities = predict_video(video_path)
    except ValueError as error:
        print(f"Couldn't analyze this clip: {error}\n")
        return

    print(f"\nPredicted trick: {predicted_trick}")
    print("Confidence by trick:")
    for trick_name, probability in probabilities.items():
        print(f"  {trick_name}: {probability:.0%}")

    show_it = ask(
        "\nPlay the video with the skeleton + prediction overlaid? (y/n): "
    )
    if show_it.lower().startswith("y"):
        confidence = probabilities[predicted_trick]
        play_video_with_prediction(video_path, predicted_trick, confidence)
    print()


def use_webcam():
    print()
    if not Path(MODEL_PATH).exists():
        print(f"No trained model yet ('{MODEL_PATH}' doesn't exist).")
        print("Choose option 4 to build the dataset and train one first.\n")
        return

    print("Opening your webcam. SPACE = start/stop recording a trick "
          "attempt. Q = quit back to this menu.")
    run_webcam_session()
    print()


def add_trick_clip():
    print()
    video_path = Path(ask("Path to the new video clip: "))
    if not video_path.is_file():
        print(f"Couldn't find a file at '{video_path}'.\n")
        return
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        print(f"'{video_path.suffix}' isn't a recognized video type "
              f"({', '.join(sorted(VIDEO_EXTENSIONS))}).\n")
        return

    existing_tricks = list_existing_tricks()
    if existing_tricks:
        print(f"Existing tricks: {', '.join(existing_tricks)}")
    trick_name = ask(
        "Which trick is this? (type an existing name above, or a "
        "brand new one to add a new trick type): "
    )
    if not trick_name:
        print("No trick name entered, cancelled.\n")
        return

    destination_folder = Path(TRICKS_DIR) / trick_name
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = destination_folder / video_path.name

    shutil.copy2(video_path, destination_path)
    print(f"Copied to '{destination_path}'.")

    rebuild_now = ask("Rebuild the dataset and retrain the model now? (y/n): ")
    if rebuild_now.lower().startswith("y"):
        rebuild_and_train()
    print()


def rebuild_and_train():
    print("\nRebuilding dataset from tricks/ ...\n")
    dataset_built = build_dataset.main()
    if not dataset_built:
        return

    print("\nRetraining model...\n")
    train_classifier.main()


def print_menu():
    print("=" * 42)
    print(" Skateboard Trick Classifier")
    print("=" * 42)
    print("1) Analyze a video (predict the trick)")
    print("2) Use webcam, record & identify a trick live")
    print("3) Add a new clip (existing trick or brand new one)")
    print("4) Rebuild dataset & retrain the model")
    print("5) Quit")


def main():
    while True:
        print_menu()
        choice = ask("\nChoose an option (1-5): ")

        if choice == "1":
            analyze_video()
        elif choice == "2":
            use_webcam()
        elif choice == "3":
            add_trick_clip()
        elif choice == "4":
            print()
            rebuild_and_train()
            print()
        elif choice == "5":
            print("Bye!")
            break
        else:
            print("Please enter 1, 2, 3, 4, or 5.\n")


if __name__ == "__main__":
    main()
