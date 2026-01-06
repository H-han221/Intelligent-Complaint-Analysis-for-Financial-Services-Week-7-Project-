import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_PATH = Path("data/processed/complaint_embeddings.parquet")
VECTOR_STORE_PATH = Path("models/vector_store.pkl")

def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    # Validate required columns
    required_cols = {"complaint_id", "embedding", "complaint_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Preparing vector store...")
    vector_store = {
        "ids": df["complaint_id"].tolist(),
        "embeddings": np.vstack(df["embedding"].values),
        "texts": df["complaint_text"].tolist()
    }

    VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    print("Vector store saved successfully.")

if __name__ == "__main__":
    main()
