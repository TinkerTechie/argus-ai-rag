import re
from app.llm import llm

def reranker_node(state):
    query = state["query"]
    docs = state["retrieved_docs"]

    docs_text = "\n\n".join([f"{i+1}. {doc}" for i, doc in enumerate(docs)])

    prompt = f"""
    You are a relevance ranking system.

    Query:
    {query}

    Documents:
    {docs_text}

    Select the 3 most relevant documents for answering the query.

    Return ONLY the numbers (e.g., 1,3,5)
"""

    response = llm.invoke(prompt).content

    # extract indices
    selected_docs = []
    try:
        # Expecting comma separated numbers
        idx_matches = re.findall(r"\d+", response)
        indices = [int(i) - 1 for i in idx_matches]
        selected_docs = [docs[i] for i in indices if 0 <= i < len(docs)]
    except:
        selected_docs = docs[:1] if docs else [] # Minimal fallback

    # Threshold Logic: If we have no relevant docs, switch to direct mode
    route = state.get("route", "rag")
    if not selected_docs:
        route = "direct"

    return {
        **state,
        "retrieved_docs": selected_docs,
        "route": route
    }