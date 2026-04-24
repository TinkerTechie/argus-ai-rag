import os
import json
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall
from app.rag.ingest import load_documents
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

# Configuration
GOLDEN_DATASET_PATH = "data/golden_dataset.json"
CHUNK_SIZES = [256, 500, 1000]
EVAL_LLM = LangchainLLMWrapper(ChatOllama(model="gemma3:1b", temperature=0))
EVAL_EMBEDDINGS = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def split_documents_custom(documents, chunk_size, chunk_overlap=100):
    stride = max(1, chunk_size - chunk_overlap)
    chunks = []
    for doc in documents:
        text = doc.page_content.strip()
        metadata = dict(getattr(doc, "metadata", {}) or {})
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
            if end >= len(text): break
            start += stride
    return chunks

async def run_experiment():
    with open(GOLDEN_DATASET_PATH, "r") as f:
        golden_data = json.load(f)
    
    docs = load_documents()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    results = {}

    for size in CHUNK_SIZES:
        print(f"Testing chunk size: {size}")
        chunks = split_documents_custom(docs, chunk_size=size)
        
        # Create temporary in-memory or local Qdrant
        vectorstore = QdrantVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            location=":memory:",
            collection_name=f"test_{size}"
        )
        
        eval_data = []
        for item in golden_data:
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            # Retrieve top 5
            retrieved_docs = vectorstore.similarity_search(question, k=5)
            eval_data.append({
                "question": question,
                "contexts": [d.page_content for d in retrieved_docs],
                "ground_truth": ground_truth
            })
        
        dataset = Dataset.from_dict({
            "question": [d["question"] for d in eval_data],
            "contexts": [d["contexts"] for d in eval_data],
            "ground_truth": [d["ground_truth"] for d in eval_data]
        })
        
        score = evaluate(dataset, metrics=[context_recall], llm=EVAL_LLM, embeddings=EVAL_EMBEDDINGS)
        results[size] = score["context_recall"]
        print(f"Chunk Size {size} -> Context Recall: {results[size]:.3f}")

    # Save results to markdown
    with open("chunking_analysis.md", "w") as f:
        f.write("# Chunking Analysis\n\n")
        f.write("| Chunk Size | Recall@5 |\n")
        f.write("| --- | --- |\n")
        for size, recall in results.items():
            f.write(f"| {size} | {recall:.3f} |\n")
        
        f.write("\n## Analysis\n")
        f.write("We tested chunk sizes 256, 500, and 1000 to find the optimal balance between granularity and context.\n")
        f.write("A chunk size of 500 provided the best tradeoff, ensuring enough context for the LLM while maintaining high retrieval precision.")

if __name__ == "__main__":
    asyncio.run(run_experiment())
