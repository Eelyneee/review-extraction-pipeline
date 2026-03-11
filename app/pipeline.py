# pipeline.py

import logging
from normalization import normalize_review, load_slang_map, load_food_dict
from ner_model import extract_food_entities
from rvi_model import predict_revisit_intent

logger = logging.getLogger("pipeline")

logger.info("Loading slang map and food dictionary...")
slang_map = load_slang_map()
food_dict = load_food_dict()
logger.info("Slang map and food dictionary loaded successfully")


def run_pipeline(text):
    logger.info("--- Pipeline Start ---")

    # Step 1: Normalize
    logger.info(f"Step 1 | Normalizing text...")
    normalized = normalize_review(text, slang_map=slang_map, food_dict=food_dict)
    logger.info(f"Step 1 | Normalized: '{normalized[:80]}{'...' if len(normalized) > 80 else ''}'")

    # Step 2: NER
    logger.info("Step 2 | Extracting food entities...")
    foods = extract_food_entities(normalized)
    logger.info(f"Step 2 | Entities found: {foods}")

    # Step 3: Revisit intent
    logger.info("Step 3 | Predicting revisit intent...")
    label, confidence = predict_revisit_intent(normalized)
    logger.info(f"Step 3 | Intent: {label} (confidence: {confidence:.2f})")

    logger.info("--- Pipeline End ---")

    return {
        "input_text": text,
        "normalized_text": normalized,
        "food_entities": foods,
        "revisit_intent": label,
        "confidence": confidence
    }