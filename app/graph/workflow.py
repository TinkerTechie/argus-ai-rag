from langgraph.graph import StateGraph, END

from app.state.graph_state import GraphState
from app.graph.router import router_node
from app.graph.nodes import retriever_node, analyst_node, critic_node


def should_continue(state):
    if state["critique_score"] < 0.6 and state["revision_count"] < 2:
        return "retry"
    return "end"


def build_graph():
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)

    # Flow
    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "retriever",
            "direct": "analyst"
        }
    )

    workflow.add_edge("retriever", "analyst")
    workflow.add_edge("analyst", "critic")

    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "retry": "analyst",
            "end": END
        }
    )

    return workflow.compile()