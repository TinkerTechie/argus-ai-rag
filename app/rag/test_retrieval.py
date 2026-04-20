import os

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"

if not os.path.exists(INDEX_PATH):
    raise ValueError(
        "FAISS index not found. Run `python -m app.rag.vectorstore` first."
    )

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

query = "impact of inflation in India"

docs = db.similarity_search(query, k=3)

if not docs:
    print("No retrieval results found.")
    raise SystemExit(0)

for d in docs:
    print("\n---\n")
    print(d.page_content[:200])
