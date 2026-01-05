# Intelligent-Complaint-Analysis-for-Financial-Services-Week-7-Project-

This project builds a Retrieval-Augmented Generation (RAG) system to analyze real-world financial customer complaints and provide actionable insights for product, support, and compliance teams.

# Task 1 – EDA & Preprocessing
- Explored CFPB complaint data
- Analyzed product distribution and narrative lengths
- Filtered complaints to relevant financial products
- Cleaned unstructured text for semantic retrieval
- Saved processed data for downstream RAG tasks
## Task 2 – Text Chunking, Embedding & Vector Store Indexing

### Objective
Convert cleaned complaint narratives into a format suitable for efficient semantic search, enabling the RAG system to retrieve relevant customer complaints.


### 1. Stratified Sampling
- A stratified sample of ~12,000 complaints was created from the filtered dataset to ensure **proportional representation** across the five key product categories: Credit Cards, Personal Loans, Savings Accounts, Money Transfers.
- This prevents high-volume products from dominating embeddings and retrieval results.

 

### 2. Text Chunking
- Complaint narratives were split into **overlapping text chunks** using a recursive splitter:
  - **Chunk size:** 500 characters  
  - **Overlap:** 50 characters
- Chunking preserves semantic context while keeping input sizes manageable for embedding models.
- Each chunk is traceable back to its original complaint via metadata.

 

### 3. Embeddings
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Reason for choice:** 
  - Lightweight and fast
  - Strong semantic encoding for short text
  - Compatible with the full-scale pre-built vector store used in Task 3

 

### 4. Vector Store
- **Database:** ChromaDB
- **Stored items per chunk:**
  - Complaint ID
  - Product category
  - Issue
  - Company
- The persisted vector store is saved under: `vector_store/chroma/`
- This enables **semantic search** by feeding user queries into the RAG pipeline in Task 3.


### 5. Deliverables
- `src/sample_data.py` – Stratified sampling function  
- `src/chunking.py` – Text chunking logic  
- `src/embeddings.py` – Embedding model loader  
- `src/vector_db.py` – Vector store creation and persistence  
- `notebooks/task2_chunking_embedding.ipynb` – End-to-end demonstration  
- `vector_store/` – Persisted ChromaDB embeddings for the sampled dataset  


### Key Learnings
- Stratified sampling ensures **fair representation** in embeddings.  
- Proper chunking preserves context without overloading the LLM.  
- Metadata storage allows **retrieved chunks to be traced to source complaints**, improving explainability and trust in the RAG system.  
- Lightweight embeddings like `all-MiniLM-L6-v2` can efficiently encode thousands of complaints for semantic search.


