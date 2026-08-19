import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET_PATH = "trick_dataset.csv"
MODEL_PATH = "trick_classifier.joblib"


def main():

    try:
        dataset = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Could not find '{DATASET_PATH}'.")
        print("Run build_dataset.py on at least a few labeled videos first.")
        return False

    label_counts = dataset["trick"].value_counts()
    print("Current dataset:")
    print(label_counts.to_string())
    print()

    if label_counts.shape[0] < 2:
        print(
            "Only one trick label is present in the dataset so far. A "
            "classifier needs at least two different tricks to learn to "
            "tell apart. Add more labeled videos with build_dataset.py "
            "(e.g. some Kickflip and Shove-it clips) and run this again."
        )
        return False

    non_feature_columns = {"trick", "source_video"}
    feature_columns = [
        column for column in dataset.columns if column not in non_feature_columns
    ]
    X = dataset[feature_columns]
    y = dataset["trick"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    smallest_class_size = label_counts.min()
    if smallest_class_size >= 2:
        num_folds = min(5, smallest_class_size)
        cross_validator = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=42
        )

        cv_predictions = cross_val_predict(model, X, y, cv=cross_validator)

        print(f"{num_folds}-fold cross-validated accuracy:",
              accuracy_score(y, cv_predictions))
        print()
        print(classification_report(y, cv_predictions))
        print(
            f"Note: this is based on only {len(dataset)} total clips, so "
            "treat this as a rough signal rather than a reliable benchmark "
            "— accuracy estimates stay noisy until there are considerably "
            "more labeled clips per trick."
        )
    else:
        print(
            "Not enough examples per trick yet for cross-validation "
            "(need at least 2 of each). Training on the full dataset "
            "without any held-out evaluation for now."
        )

    model.fit(X, y)

    saved_bundle = {
        "model": model,
        "feature_columns": feature_columns,
    }
    joblib.dump(saved_bundle, MODEL_PATH)
    print(f"\nSaved trained model to '{MODEL_PATH}'.")
    return True


if __name__ == "__main__":
    main()
