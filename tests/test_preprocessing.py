import pandas as pd
from src.preprocessing import clean_text, filter_and_clean

def test_clean_text_basic():
    text = "I am Writing to File a Complaint!!! This is BAD."
    cleaned = clean_text(text)

    assert isinstance(cleaned, str)
    assert "writing to file a complaint" not in cleaned
    assert cleaned == cleaned.lower()


def test_filter_and_clean_removes_empty_narratives():
    data = {
        "Product": ["Credit card", "Credit card"],
        "Consumer complaint narrative": ["Valid complaint text", None]
    }
    df = pd.DataFrame(data)

    result = filter_and_clean(df)

    assert len(result) == 1
    assert result.iloc[0]["clean_narrative"] == "valid complaint text"


def test_filter_and_clean_filters_products():
    data = {
        "Product": ["Credit card", "Mortgage"],
        "Consumer complaint narrative": ["text one", "text two"]
    }
    df = pd.DataFrame(data)

    result = filter_and_clean(df)

    assert all(result["Product"] == "Credit card")
