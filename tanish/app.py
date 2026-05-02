"""FastAPI app for model inference with Pydantic schemas and structured logging.
Expt 2 & 3: /predict endpoint, schemas, exception handlers, structured logging
"""
from pathlib import Path
import pickle
from typing import List

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, conlist
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.exceptions import RequestValidationError

from .logging_config import setup_logger

logger = setup_logger()

app = FastAPI(title="MLOps Experiment API")

MODEL_PATH = Path(__file__).parent / "model.pkl"
model = None


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
        logger.info("model_loaded", extra={"path": str(MODEL_PATH)})
    except Exception as e:
        logger.exception("model_load_failed", extra={"path": str(MODEL_PATH), "error": str(e)})
        raise


class PredictRequest(BaseModel):
    features: conlist(float, min_length=4, max_length=4)


class PredictResponse(BaseModel):
    prediction: int
    probabilities: List[float]


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        logger.info("request_received", extra={
            "method": request.method,
            "path": request.url.path,
            "body": body.decode("utf-8", errors="ignore")[:200],
        })
        try:
            response = await call_next(request)
            logger.info("response_sent", extra={"status_code": response.status_code})
            return response
        except Exception as e:
            logger.exception("request_error", extra={"error": str(e)})
            raise


app.add_middleware(LoggingMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("validation_error", extra={"errors": exc.errors()})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("internal_error", extra={"error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    import numpy as np

    X = np.array(req.features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else []
    logger.info("prediction_made", extra={"prediction": pred})
    return PredictResponse(prediction=pred, probabilities=probs)
