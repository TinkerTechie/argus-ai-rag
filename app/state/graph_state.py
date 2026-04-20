from typing import TypedDict, List

class GraphState(TypedDict):
    query: str
    route: str

    sub_queries: List[str]

    retrieved_docs: List[str]
    reranked_docs: List[str]

    draft_answer: str
    critique_feedback: str
    critique_score: float

    revision_count: int