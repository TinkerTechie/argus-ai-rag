from app.llm import llm

def query_expander_node(state):
    query = state["query"]

    prompt = f"""
        You are a query expansion system.
    Generate 3 different versions of this query to improve search results.
    Original Query:
    {query}

Return only the 3 queries as a list.
"""

    response = llm.invoke(prompt).content

    # simple split (can improve later)
    queries = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

    return {
        **state,
        "sub_queries": queries
    }