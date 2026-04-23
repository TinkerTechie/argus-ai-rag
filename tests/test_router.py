import pytest
from app.graph.nodes import router_node

def test_router_greeting():
    """Test that simple greetings bypass the RAG pipeline."""
    state = {"query": "Hello there!", "revision_count": 0}
    # router_node returns a dictionary with the next node to visit
    result = router_node(state)
    assert result["next_node"] == "analyst"
    assert "retrieval_required" in result
    assert result["retrieval_required"] is False

def test_router_research_query():
    """Test that research queries trigger the RAG pipeline."""
    state = {"query": "What is the impact of inflation on GDP?", "revision_count": 0}
    result = router_node(state)
    assert result["next_node"] == "query_expander"
    assert result["retrieval_required"] is True
