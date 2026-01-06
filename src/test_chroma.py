from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class ComplaintRetriever:
    def __init__(self, persist_dir="vector_store/chroma"):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client(Settings(persist_directory=persist_dir))
        self.collection = self.client.get_collection(name="cfpb_complaints")

    def retrieve(self, query, k=5):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        return documents, metadatas

if __name__ == "__main__":
    retriever = ComplaintRetriever()
    query = "Credit card complaint"
    docs, metas = retriever.retrieve(query)
    for d, m in zip(docs, metas):
        print(m)
        print(d)
        print("---")
