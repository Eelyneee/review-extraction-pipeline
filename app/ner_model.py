# ner_model.py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

MODEL_PATH = "model/ner_model_v1"

label2id = {'B-FOOD':0,'I-FOOD':1,'O':2}
id2label = {0:'B-FOOD',1:'I-FOOD',2:'O'}

device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    id2label=id2label,
    label2id=label2id
)

ner_pipeline = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device
)

def merge_food_entities(preds, text):
    merged = []
    current = None

    for ent in preds:
        label = ent["entity_group"]          # e.g. "B-FOOD", "I-FOOD", "O"
        if label == "O":
            # Close any active span when we hit non-entity
            if current is not None:
                current["word"] = text[current["start"]:current["end"]]
                merged.append(current)
                current = None
            continue

        # Get entity_class: "FOOD"
        if "-" in label:
            _, entity_class = label.split("-", 1)
        else:
            entity_class = label

        start = ent["start"]
        end = ent["end"]

        if current is None:
            # start new span
            current = {
                "entity_class": entity_class,
                "start": start,
                "end": end,
            }
        else:
            # if same class and contiguous (or overlapping), extend
            if entity_class == current["entity_class"] and start <= current["end"] + 1:
                current["end"] = end
            else:
                # close previous and start a new one
                current["word"] = text[current["start"]:current["end"]]
                merged.append(current)
                current = {
                    "entity_class": entity_class,
                    "start": start,
                    "end": end,
                }

    # close last span
    if current is not None:
        current["word"] = text[current["start"]:current["end"]]
        merged.append(current)

    # clean spaces / leftover artifacts
    for m in merged:
        w = m["word"].replace("##", "")
        m["word"] = " ".join(w.split())

    return merged


def extract_food_entities(text):

    preds = ner_pipeline(text)
    merged_entities_user_text = merge_food_entities(preds, text)

    return merged_entities_user_text