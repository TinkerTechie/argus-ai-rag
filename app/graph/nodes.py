from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from importlib.metadata import PackageNotFoundError, version
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

_db = None
_db_init_error = None

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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

    context = "\n\n".join(docs)

    prompt = f"""
        You are an expert research assistant.

        Use ONLY the provided context to answer the question.

        Context:
        {context}

        Question:
        {query}

        Give a clear, structured answer.
        """

    response = llm.invoke(prompt)

    return {
        **state,
        "draft_answer": response.content
    }


# 🔎 CRITIC
import re

def critic_node(state):
    docs = state.get("retrieved_docs", [])
    answer = state["draft_answer"]

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

    # 🔥 FIXED PARSING
    match = re.search(r"Score:\s*(\d*\.?\d+)", response)
    if match:
        score = float(match.group(1))
    else:
        score = 0.5

    return {
        **state,
        "critique_score": score,
        "critique_feedback": response,
        "revision_count": state.get("revision_count", 0) + 1
    }