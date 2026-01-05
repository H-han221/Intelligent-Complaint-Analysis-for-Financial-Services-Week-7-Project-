from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)
from src.chunking import chunk_text

sample_df["chunks"] = sample_df["clean_narrative"].apply(chunk_text)
sample_df["num_chunks"] = sample_df["chunks"].apply(len)

sample_df["num_chunks"].describe()
