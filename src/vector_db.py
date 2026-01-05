import chromadb
from chromadb.utils import embedding_functions

def create_chroma_collection(persist_dir: str):
    client = chromadb.Client(
        settings=chromadb.Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
    )

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="complaints",
        embedding_function=embedding_function
    )

    return client, collection
from src.vector_db import create_chroma_collection

client, collection = create_chroma_collection("../vector_store/chroma")

ids = []
documents = []
metadatas = []

for idx, row in sample_df.iterrows():
    for i, chunk in enumerate(row["chunks"]):
        ids.append(f"{row.name}_{i}")
        documents.append(chunk)
        metadatas.append({
            "complaint_id": row.get("Complaint ID", idx),
            "product": row["Product"],
            "issue": row.get("Issue"),
            "company": row.get("Company")
        })

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

client.persist()
