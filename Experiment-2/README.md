# Credit Card Fraud Detection API (Experiment-2)

This project provides a FastAPI-based backend for serving predictions from a trained ML model (`fraud_model_v2.pkl`).

## Project Structure

```
Experiment-2/
│
├── app.py              # Main FastAPI application
├── fraud_model_v2.pkl  # Trained ML model (Pickle format)
├── requirements.txt    # Python dependencies
└── README.md           # Instructions
```

## Environment Setup

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Dependencies include:
   - `fastapi`: Web framework
   - `uvicorn`: ASGI server
   - `scikit-learn`: For loading and running the ML model
   - `pandas`: Data manipulation
   - `numpy`: Numerical operations

## Running the Application

Start the API server using `uvicorn`:

```bash
uvicorn app:app --reload
```
The server will start at `http://127.0.0.1:8000`.

## Testing the API

### 1. Root Endpoint
GET request to `/`:
```bash
curl http://127.0.0.1:8000/
```
Response:
```json
{
  "message": "Fraud Detection API is running"
}
```

### 2. Prediction Endpoint
POST request to `/predict`.

**Using Swagger UI**:
- Open your browser and navigate to `http://127.0.0.1:8000/docs`.
- Expand the `POST /predict` endpoint.
- Click "Try it out".
- Enter the JSON body (see example below).
- Click "Execute".

**Example JSON Body**:
```json
{
  "Time": 0.0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

**Response**:
```json
{
  "prediction": 0,
  "fraud_probability": 0.023
}
```

**Note**: The model expects 30 features: `Time`, `V1` to `V28`, and `Amount`. Ensure the order and data types match.

## Deployment Notes
- Assumes `fraud_model_v2.pkl` is present in the root directory.
- Preprocessing (scaling) for `Amount` and `Time` is assumed to be handled before calling the API, or handled by the model if part of a pipeline. This implementation passes raw values directly to the model.
