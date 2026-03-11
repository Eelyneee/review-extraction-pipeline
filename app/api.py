# api.py

import time
import logging
from fastapi import FastAPI, Request
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


class ReviewInput(BaseModel):
    review: str


@app.get("/")
def health_check():
    logger.info("Health check called")
    return {"status": "running"}


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