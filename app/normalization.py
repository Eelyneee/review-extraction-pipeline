# normalization.py

import re
import emoji
import unicodedata
import pandas as pd
from contractions import fix as expand_contractions

# Load resources
def load_slang_map(path="data/slang_map.csv"):
    df = pd.read_csv(path)
    return dict(zip(df['slang'].str.lower(), df['normalized'].str.lower()))

def load_food_dict(path="data/food_lexicon_augmented.csv"):
    df = pd.read_csv(path)
    food_dict = {}
    for _, row in df.iterrows():
        canonical = row['canonical']
        variants = row['variants'].split('|')
        for variant in variants:
            food_dict[variant.strip().lower()] = canonical
    return food_dict


def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)

def normalize_emoji(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def expand_contractions_text(text):
    return expand_contractions(text)


def normalize_review(text, slang_map=None, food_dict=None):

    text = normalize_unicode(text)
    text = normalize_emoji(text)
    text = expand_contractions_text(text)
    text = normalize_whitespace(text)

    return text