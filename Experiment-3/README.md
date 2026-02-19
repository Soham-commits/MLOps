# Experiment-3: Logging and Error Handling

This experiment extends `Experiment-2` by adding structured logging and error handling to the API.

## Project Structure

```bash
Experiment-3/
├── app.py              # Main FastAPI application with logging
├── fraud_model_v2.pkl  # Pre-trained Fraud Detection Model
├── logs/               # Directory for log files
│   └── app.log         # Application log file
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Features Implemented

1.  **Logging Setup**:
    *   Logs are saved to `logs/app.log` and printed to the console (stdout).
    *   Format: `timestamp | level | message`.
    *   Log level: `INFO`.

2.  **System Events Logging**:
    *   Startup events ("Fraud model loaded successfully", "API server started").

3.  **Middleware Logging**:
    *   Logs HTTP method, endpoint path, status code, and processing time for every request.
    *   Example: `INFO | POST /predict | Status: 200 | Time: 45.12ms`

4.  **Prediction Logging**:
    *   Logs input payload details (Amount, Time).
    *   Logs prediction result and fraud probability.
    *   Example: `INFO | Prediction: 0 | Fraud Probability: 0.67%`

5.  **Global Error Handling**:
    *   **RequestValidationError**: Returns 422 with structured error details.
    *   **HTTPException**: Returns specific HTTP status code and error message.
    *   **Generic Exception**: Returns 500 Internal Server Error with a generic message, while logging the stack trace.

## Installation

1.  Navigate to the project directory:
    ```bash
    cd Experiment-3
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Start the API server:
    ```bash
    uvicorn app:app --reload
    ```

2.  Access the API documentation at:
    *   Swagger UI: http://127.0.0.1:8000/docs
    *   ReDoc: http://127.0.0.1:8000/redoc

3.  Test the `/predict` endpoint:
    You can use the Swagger UI or `curl`:

    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "Time": 10.0,
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
    }'
    ```

## Logs

Check the logs in `logs/app.log` or the console output to verify logging functionality.

