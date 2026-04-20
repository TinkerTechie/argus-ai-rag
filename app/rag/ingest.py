from langchain_core.documents import Document
from pypdf import PdfReader
import os

DATA_PATH = "data/raw/"

def load_documents():
    documents = []
    
    if not os.path.exists(DATA_PATH):
        raise ValueError(f"Folder not found: {DATA_PATH}")
    
    files = os.listdir(DATA_PATH)
    pdf_files = []

    for file in files:
        if file.endswith(".pdf"):
            pdf_files.append(file)
            file_path = os.path.join(DATA_PATH, file)
            reader = PdfReader(file_path)

            for page_number, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_number,
                        },
                    )
                )

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {DATA_PATH}. Add one or more .pdf files and rerun ingest."
        )
    
    return documents


def split_documents(documents):
    non_empty_documents = [
        doc for doc in documents if (doc.page_content or "").strip()
    ]

    if not non_empty_documents:
        return []

    chunk_size = 500
    chunk_overlap = 100
    stride = max(1, chunk_size - chunk_overlap)
    chunks = []

    for doc in non_empty_documents:
        text = doc.page_content.strip()
        metadata = dict(getattr(doc, "metadata", {}) or {})
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_start": start,
                    "chunk_end": end,
                }
                chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )
            if end >= len(text):
                break
            start += stride

    return chunks

if __name__ == "__main__":
    try:
        docs = load_documents()
        chunks = split_documents(docs)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)

    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        print(
            "\nNo text chunks were created. The PDF may be blank, image-only, or contain no extractable text."
        )
        raise SystemExit(1)

    print("\nSample chunk:\n")
    print(chunks[0].page_content[:300])
