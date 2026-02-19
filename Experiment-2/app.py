import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="ML Inference API for detecting fraudulent credit card transactions.",
    version="1.0"
)

# Global model variable
model = None

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    model_path = "fraud_model_v2.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

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

# Root endpoint
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionFeatures):
    if not model:
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
        # Handle cases where predict_proba might not be available or returns different shape
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(df)[0][1] # Probability of class 1 (Fraud)
        else:
            probability = float(prediction) # Fallback if proba not available
            
        fraud_prob_percent = round(float(probability) * 100, 2)
        
        if fraud_prob_percent < 1:
            risk = "Low"
        elif fraud_prob_percent < 5:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "prediction": int(prediction),
            "fraud_probability": fraud_prob_percent,
            "risk_level": risk
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
