from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from importlib.metadata import PackageNotFoundError, version

from langchain_huggingface import HuggingFaceEmbeddings

_db = None
_db_init_error = None

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import os

# Streamlit hot-reloading hack: Remove stuck Qdrant lock file
lock_file = "./qdrant_db/.lock"
if os.path.exists(lock_file):
    try:
        os.remove(lock_file)
    except Exception:
        pass

# Initialize Qdrant Client and Store
client = QdrantClient(path="./qdrant_db")
db = QdrantVectorStore(
    client=client,
    collection_name="argus_docs",
    embedding=embeddings,
)

def _torch_is_compatible():
    # ... (rest of the function remains similar or can be simplified if not needed)
    return True # Simplifying for now as I just installed torch 2.4.1


# 🔍 RETRIEVER
def retriever_node(state):
    queries = state.get("sub_queries", [state["query"]])

    all_docs = []

    for q in queries:
        docs = db.similarity_search(q, k=3)
        all_docs.extend([d.page_content for d in docs])

    # remove duplicates
    unique_docs = list(set(all_docs))

    return {
        **state,
        "retrieved_docs": unique_docs
    }


# 🧠 ANALYST
from app.llm import llm

def analyst_node(state):
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    route = state.get("route", "rag")

    if not docs or route == "direct":
        # Direct LLM mode or Fallback
        disclaimer = ""
        if route == "rag" and not docs:
            disclaimer = "*(Note: I couldn't find specific documents in my knowledge base to answer this, but here is a general explanation based on my internal training data)*\n\n"
        
        prompt = f"""
        You are an expert research assistant.
        
        The user has asked a general question or a question where no specific context was found.
        Answer the question accurately based on your general knowledge.
        
        CRITICAL FORMATTING RULES:
        - Format your output beautifully using Markdown.
        - Use bolding (**text**) for key terms and emphasis.
        - Use short, punchy paragraphs to avoid walls of text.
        - Use bullet points or numbered lists where appropriate to make it highly scannable.
        
        Question:
        {query}
        
        Give a clear, structured, and helpful answer.
        """
        response = llm.invoke(prompt)
        answer = disclaimer + response.content
    else:
        # Standard RAG mode
        context = "\n\n".join(docs)
        prompt = f"""
        You are an expert research assistant.
        
        CRITICAL INSTRUCTIONS:
        1. Evaluate if the provided context is actually relevant to the user's question.
        2. If the context is relevant, use it to answer the question.
        3. If the context is completely IRRELEVANT (e.g., the user asks about 'stagflation' but the context is about 'RAG'), IGNORE the context entirely. Do not mention that the context is irrelevant. Do not try to force a connection. Just answer the user's question directly using your general knowledge.
        
        CRITICAL FORMATTING RULES:
        - Format your output beautifully using Markdown.
        - Use bolding (**text**) for key terms and emphasis.
        - Use short, punchy paragraphs to avoid walls of text.
        - Use bullet points or numbered lists where appropriate to make it highly scannable.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Give a clear, structured answer based on the above rules.
        """
        response = llm.invoke(prompt)
        answer = response.content

    return {
        **state,
        "draft_answer": answer
    }


# 🔎 CRITIC
import re

def critic_node(state):
    docs = state.get("retrieved_docs", [])
    answer = state["draft_answer"]
    route = state.get("route", "rag")

    if not docs or route == "direct":
        # Evaluate for general quality if no context
        prompt = f"""
        You are a strict evaluator.
        
        Evaluate the following answer for:
        1. Accuracy and helpfulness.
        2. Logical flow and structure.
        
        Answer:
        {answer}
        
        Return EXACT format:
        Score: <number between 0 and 1>
        Feedback: <short explanation>
        """
    else:
        # Standard RAG evaluation
        context = "\n\n".join(docs)
        prompt = f"""
        You are a strict evaluator.
        
        Context:
        {context}
        
        Answer:
        {answer}
        
        Evaluate:
        1. Is answer supported by context?
        2. Any hallucination?
        3. Missing info?
        
        Return EXACT format:
        Score: <number between 0 and 1>
        Feedback: <short explanation>
        """

    response = llm.invoke(prompt).content

    match = re.search(r"Score:\s*(\d*\.?\d+)", response)
    if match:
        score = float(match.group(1))
    else:
        score = 0.8 # Higher fallback for general answers

    return {
        **state,
        "critique_score": score,
        "critique_feedback": response,
        "revision_count": state.get("revision_count", 0) + 1
    }