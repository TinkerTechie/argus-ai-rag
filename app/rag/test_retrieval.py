import os
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "argus_docs"

if not os.path.exists(QDRANT_PATH):
    raise ValueError(
        f"Qdrant DB not found at {QDRANT_PATH}. Run `python -m app.rag.vectorstore` first."
    )

print(f"Connecting to Qdrant at {QDRANT_PATH}...")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load Qdrant vector store from local path
db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    path=QDRANT_PATH,
    collection_name=COLLECTION_NAME,
)

query = "impact of inflation in India"
print(f"Searching for: '{query}'")

docs = db.similarity_search(query, k=3)

if not docs:
    print("No retrieval results found.")
    raise SystemExit(0)

print(f"\nFound {len(docs)} relevant chunks:")
for i, d in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {d.metadata.get('source', 'Unknown')}")
    print(f"Content: {d.page_content[:200]}...")
