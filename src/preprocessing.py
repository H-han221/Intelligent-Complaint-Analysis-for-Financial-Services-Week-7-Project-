import re
import pandas as pd

TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]

def clean_text(text: str) -> str:
    """
    Clean complaint narrative text for better embedding quality.
    """
    text = text.lower()

    boilerplate_patterns = [
        r"i am writing to file a complaint",
        r"this complaint is regarding",
        r"consumer complaint narrative"
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    # Remove special characters but keep punctuation
    text = re.sub(r"[^a-z0-9\s\.,]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to required products and clean narratives.
    """
    df = df[df["Product"].isin(TARGET_PRODUCTS)].copy()
    df = df[df["Consumer complaint narrative"].notna()]

    df["clean_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    return df
