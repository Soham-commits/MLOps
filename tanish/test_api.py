"""Simple tests for the FastAPI app (Expt 2 & 3).
Uses TestClient to exercise /predict and error handling/logging.
"""
from fastapi.testclient import TestClient

from tanish.app import app
from tanish import train as train_module
from pathlib import Path


def run_tests():
    # ensure model exists and is up-to-date
    model_path = Path(__file__).parent / "model.pkl"
    train_module.train_and_save(model_path)

    # ensure the app has the loaded model available (bypass startup event)
    trained = train_module.train_and_save(model_path)
    # assign directly into the app module so requests work in tests
    import importlib
    app_mod = importlib.import_module('tanish.app')
    app_mod.model = trained
    client = TestClient(app_mod.app)

    # happy path
    resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    print("OK status:", resp.status_code)
    print("OK response:", resp.json())

    # validation error (wrong length)
    resp2 = client.post("/predict", json={"features": [1.0, 2.0]})
    print("Validation status:", resp2.status_code)
    print("Validation response:", resp2.json())


if __name__ == "__main__":
    run_tests()
