"""Simple structured JSON logger configuration for the FastAPI app.
"""
import logging
from pythonjsonlogger import jsonlogger


def setup_logger(name: str = "mlops") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    fmt = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
