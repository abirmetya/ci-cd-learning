from fastapi.testclient import TestClient

from api.app import app, get_model


class FakeModel:
    def predict(self, features):
        return [0]


def override_get_model():
    return FakeModel()


app.dependency_overrides[get_model] = override_get_model

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"species": "setosa"}


def test_rejects_negative_measurement():
    response = client.post(
        "/predict",
        json={
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422
