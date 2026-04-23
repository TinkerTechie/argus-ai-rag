from langgraph.graph import StateGraph, END
from app.graph.reranker import reranker_node
from app.state.graph_state import GraphState
from app.graph.router import router_node
from app.graph.nodes import retriever_node, analyst_node, critic_node
from app.graph.query_expander import query_expander_node


def should_continue(state):
    if state["critique_score"] < 0.6 and state["revision_count"] < 2:
        return "retry"
    return "end"


def build_graph():
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("expander", query_expander_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reranker", reranker_node)

    # Entry
    workflow.set_entry_point("router")

    # Router → Expander or Analyst
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "expander",
            "direct": "analyst"
        }
    )

    # Expander → Retriever
    workflow.add_edge("expander", "retriever")

    # Retriever → Analyst
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "analyst")

    # Analyst → Critic
    workflow.add_edge("analyst", "critic")

    # Critic → Retry or End
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "retry": "analyst",
            "end": END
        }
    )

    return workflow.compile()