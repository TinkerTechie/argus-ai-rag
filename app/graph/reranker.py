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
    try:
        indices = [int(x.strip()) - 1 for x in response.split(",")]
        selected_docs = [docs[i] for i in indices if i < len(docs)]
    except:
        selected_docs = docs[:3]  # fallback

    return {
        **state,
        "retrieved_docs": selected_docs
    }