from app.llm import llm

def router_node(state):
    query = state["query"]
    
    prompt = f"""
    Analyze the user query and classify it into one of two categories:
    1. 'direct': For general greetings, broad common knowledge, or conversational filler.
    2. 'rag': For specific, technical, or factual queries that likely require searching through a specialized knowledge base (PDFs, research papers, etc.).

    Query: {query}
    
    Return ONLY the category name ('direct' or 'rag').
    """
    
    response = llm.invoke(prompt).content.strip().lower()
    
    # Fallback and cleaning
    category = "rag"
    if "direct" in response:
        category = "direct"
    elif "rag" in response:
        category = "rag"
        
    return {**state, "route": category}