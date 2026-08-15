from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from ml_project.model import predict_species

MODEL_PATH = Path("model_artifacts/iris_model.joblib")

app = FastAPI(
    title="Iris Prediction API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


class PredictionResponse(BaseModel):
    species: str


@lru_cache
def get_model() -> Any:
    """Load and cache the trained model."""
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    model: Any = Depends(get_model),
) -> PredictionResponse:
    features = [
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]

    species = predict_species(model, features)

    return PredictionResponse(species=species)
