
import sys

import pandas as pd
import joblib

# We reuse extract_features_from_video from build_dataset.py rather than
# rewriting the feature logic here. It's essential that a new video gets
# processed in exactly the same way (same landmarks, same normalization,
# same signals, same summary statistics) as every video already in the
# training set — any difference would make the model's input look nothing
# like what it was trained on.
from build_dataset import extract_features_from_video

MODEL_PATH = "trick_classifier.joblib"


def predict_video(video_path):

    saved_bundle = joblib.load(MODEL_PATH)
    model = saved_bundle["model"]
    feature_columns = saved_bundle["feature_columns"]

    features = extract_features_from_video(video_path)

    feature_row = pd.DataFrame([features])[feature_columns]

    predicted_trick = model.predict(feature_row)[0]

    class_probabilities = model.predict_proba(feature_row)[0]
    probabilities = pd.Series(
        class_probabilities, index=model.classes_
    ).sort_values(ascending=False)

    return predicted_trick, probabilities


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict_trick.py <path_to_video>")
        print("Example: python3 predict_trick.py new_clip.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print(f"Processing '{video_path}'...")
    try:
        predicted_trick, probabilities = predict_video(video_path)
    except FileNotFoundError as error:
        print(f"\n{error}")
        print("(Make sure the model is trained — see train_classifier.py "
              "or menu.py — and the video path is correct.)")
        sys.exit(1)

    print(f"\nPredicted trick: {predicted_trick}")
    print("\nConfidence by trick:")
    for trick_name, probability in probabilities.items():
        print(f"  {trick_name}: {probability:.0%}")


if __name__ == "__main__":
    main()
