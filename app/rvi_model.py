# rvi_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

MODEL_PATH = "model/rvi_model_v1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

id2label = {0: "Negative", 1: "Positive"}


def predict_revisit_intent(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    encoding = {k:v.to(device) for k,v in encoding.items()}

    with torch.no_grad():

        outputs = model(**encoding)

        logits = outputs.logits
        probs = F.softmax(logits,dim=-1)

    pred = torch.argmax(probs).item()

    return id2label[pred], probs[0][pred].item()