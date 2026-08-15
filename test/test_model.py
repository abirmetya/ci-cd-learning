import pytest

from src.ml_project.model import predict_species, train_model, validate_features


def test_model_accuracy():
    _, accuracy = train_model()

    assert accuracy >= 0.90


def test_species_prediction():
    model, _ = train_model()

    species = predict_species(model, [5.1, 3.5, 1.4, 0.2])

    assert species == "setosa"


def test_incorrect_feature_count():
    with pytest.raises(
        ValueError,
        match="Exactly four feature values are required",
    ):
        validate_features([5.1, 3.5])


def test_negative_feature_value():
    with pytest.raises(
        ValueError,
        match="Feature values cannot be negative",
    ):
        validate_features([5.1, -3.5, 1.4, 0.2])
