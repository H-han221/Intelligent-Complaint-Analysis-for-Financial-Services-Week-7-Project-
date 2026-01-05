import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.chunking import chunk_text

import pandas as pd
from src.sample_data import stratified_sample
from src.chunking import chunk_text

def test_stratified_sample_size():
    data = {
        "Product": ["Credit card"] * 50 + ["Personal loan"] * 50,
        "clean_narrative": ["text"] * 100
    }
    df = pd.DataFrame(data)

    sample = stratified_sample(df, label_col="Product", total_samples=20)

    assert len(sample) <= 20


def test_stratified_sample_preserves_distribution():
    data = {
        "Product": ["Credit card"] * 80 + ["Personal loan"] * 20,
        "clean_narrative": ["text"] * 100
    }
    df = pd.DataFrame(data)

    sample = stratified_sample(df, label_col="Product", total_samples=50)

    proportions = sample["Product"].value_counts(normalize=True)

    assert proportions["Credit card"] > proportions["Personal loan"]


def test_chunk_text_output():
    text = "This is a test sentence. " * 100
    chunks = chunk_text(text)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
