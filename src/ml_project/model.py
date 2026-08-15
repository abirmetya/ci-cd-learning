import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
]

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def train_model(random_state: int = 42) -> tuple[Pipeline, float]:
    """Train and evaluate an Iris classification model."""
    features, target = load_iris(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


def validate_features(features: list[float]) -> np.ndarray:
    """Validate one observation before prediction."""
    values = np.asarray(features, dtype=float)

    if values.shape != (4,):
        raise ValueError("Exactly four feature values are required")

    if not np.isfinite(values).all():
        raise ValueError("Feature values must be finite numbers")

    if (values < 0).any():
        raise ValueError("Feature values cannot be negative")

    return values


def predict_species(model: Pipeline, features: list[float]) -> str:
    """Predict the Iris species for one observation."""
    validated_features = validate_features(features)
    prediction = model.predict([validated_features])[0]

    return TARGET_NAMES[int(prediction)]


def save_model(output_directory: str = "model_artifacts") -> None:
    """Train and save the model with basic metadata."""
    model, accuracy = train_model()

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path / "iris_model.joblib")

    metadata = {
        "model_name": "iris-logistic-regression",
        "model_version": "1.0.0",
        "accuracy": round(accuracy, 4),
        "features": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }

    with (output_path / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Model saved with accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    save_model()
