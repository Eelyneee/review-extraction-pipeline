# api.py

import time
import json
import logging
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_pipeline

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api")

app = FastAPI(title="Hawker NLP Pipeline")

# Load model info at startup
MODEL_INFO_PATHS = {
    "ner_model": Path("model/ner_model_v1/model_info.json"),
    "rvi_model": Path("model/rvi_model_v1/model_info.json"),
}

def load_model_info():
    info = {}
    for name, path in MODEL_INFO_PATHS.items():
        if path.exists():
            with open(path) as f:
                info[name] = json.load(f)
            logger.info(f"Loaded model info for {name} ({info[name]['version']})")
        else:
            logger.warning(f"model_info.json not found for {name} at {path}")
            info[name] = {"error": "model_info.json not found"}
    return info

model_info = load_model_info()


class ReviewInput(BaseModel):
    review: str


@app.get("/")
def health_check():
    logger.info("Health check called")
    return {"status": "running"}


@app.get("/model-info")
def get_model_info():
    logger.info("Model info endpoint called")
    return model_info


@app.post("/analyze")
def analyze(data: ReviewInput):
    logger.info(f"Received review: '{data.review[:80]}{'...' if len(data.review) > 80 else ''}'")

    start = time.time()
    result = run_pipeline(data.review)
    elapsed = time.time() - start

    logger.info(f"Pipeline completed in {elapsed:.3f}s | "
                f"foods={result['food_entities']} | "
                f"intent={result['revisit_intent']} | "
                f"confidence={result['confidence']:.2f}")

    return result