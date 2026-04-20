from langchain_community.vectorstores import FAISS
from app.rag.ingest import load_documents, split_documents

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

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("faiss_index")

    print("Vector store created and saved!")

if __name__ == "__main__":
    create_vectorstore()
