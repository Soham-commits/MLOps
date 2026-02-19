import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn
import os
import sys
import logging
import time

# --------------------------------------------------------------------------------
# 1. Logging Setup
# --------------------------------------------------------------------------------

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Initialize FastAPI app
# --------------------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML Inference API for detecting fraudulent credit card transactions.",
    version="1.0"
)

# Global model variable
model = None

# --------------------------------------------------------------------------------
# 2. Lifecycle Events
# --------------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Load the model and log startup event."""
    global model
    try:
        model_path = "fraud_model_v2.pkl"
        logger.info(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            
        logger.info("Fraud model loaded successfully")
        logger.info("API server started")
        
    except Exception as e:
        logger.critical(f"Failed to load model: {str(e)}")
        # In a real production app, we might want to shut down if model fails to load
        # raise e 

# --------------------------------------------------------------------------------
# 3. Middleware Logging
# --------------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log details for every incoming request."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (time.time() - start_time) * 1000 # in ms
    formatted_process_time = f"{process_time:.2f}ms"
    
    # Log the request details
    logger.info(
        f"{request.method} {request.url.path} | Status: {response.status_code} | Time: {formatted_process_time}"
    )
    
    return response

# --------------------------------------------------------------------------------
# 5. Error Handling
# --------------------------------------------------------------------------------

# A) Request Validation Errors (Pydantic)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation Error", "details": exc.errors()},
    )

# B) HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )

# C) Generic Exception
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "status_code": 500},
    )

# --------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------

# Define Request Schema
class TransactionFeatures(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Define Response Schema
class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    risk_level: str

# --------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------

# Root endpoint
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    
    # 4) Prediction Logging - Log input payload (sanitized)
    # We'll log a subset or summarized version to avoid massive logs if needed, 
    # but here we log the 'Amount' and 'Time' explicitly as key identifiers.
    logger.info(f"Received prediction request | Amount: {transaction.Amount} | Time: {transaction.Time}")

    if not model:
        logger.error("Model not loaded during prediction request")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        data = transaction.dict()
        df = pd.DataFrame([data])
        
        # Ensure correct column order
        expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = df[expected_columns]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Handle cases where predict_proba might not be available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df)[0][1] # Probability of class 1 (Fraud)
        else:
            probability = float(prediction) # Fallback if proba not available
            
        fraud_prob_percent = round(float(probability) * 100, 2)

        # Log Prediction Result
        logger.info(f"Prediction: {prediction} | Fraud Probability: {fraud_prob_percent}%")
        
        risk = "High"
        if fraud_prob_percent < 1:
            risk = "Low"
        elif fraud_prob_percent < 5:
            risk = "Medium"

        return {
            "prediction": int(prediction),
            "fraud_probability": fraud_prob_percent,
            "risk_level": risk
        }
        
    except Exception as e:
        # Re-raise to be caught by global exception handler
        # But we can log specific context here if needed
        logger.error(f"Error during prediction: {str(e)}")
        raise e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
