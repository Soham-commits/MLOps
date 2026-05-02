# MLOps Course - Tanish Experiments

This folder contains three experiments mirroring the course goals:

- Expt 1: Train and pickle a model
- Expt 2: FastAPI backend with a `/predict` endpoint (Pydantic schemas)
- Expt 3: Logging and error handling for the FastAPI app

Quick start:

1. Install dependencies:

```bash
python3 -m pip install -r tanish/requirements.txt
```

2. Train the model and produce `model.pkl`:

```bash
python3 tanish/train.py
```

3. Run the API server:

```bash
uvicorn tanish.app:app --reload --port 8000
```

4. Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"features": [5.1,3.5,1.4,0.2]}'
```

Files:

- `tanish/train.py`: training + pickle save and verify
- `tanish/model.pkl`: produced by `train.py` after running
- `tanish/app.py`: FastAPI app with `/predict` and logging
- `tanish/logging_config.py`: structured JSON logger config
- `tanish/requirements.txt`: dependencies
