from langchain_qdrant import QdrantVectorStore
from app.rag.ingest import load_documents, split_documents
import os

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vectorstore():
    docs = load_documents()
    chunks = split_documents(docs)

    if not chunks:
        raise ValueError(
            "No text chunks available for indexing. Add a PDF with extractable text and rerun ingest."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Use Qdrant local storage
    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        path="./qdrant_db",
        collection_name="argus_docs",
    )

    print("Qdrant vector store created and saved at ./qdrant_db")

if __name__ == "__main__":
    create_vectorstore()
