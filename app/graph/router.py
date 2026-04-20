def router_node(state):
    query = state["query"].lower()

    if any(word in query for word in ["hi", "hello", "hey"]):
        return {**state, "route": "direct"}
    
    return {**state, "route": "rag"}