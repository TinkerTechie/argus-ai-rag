import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.graph.workflow import build_graph

app = build_graph()

# query = "What is Retrieval-Augmented Generation?"
# 
# result = app.invoke({
#     "query": query,
#     "revision_count": 0
# })
# 
# print("\nFINAL OUTPUT:\n")
# print(result["draft_answer"])
# print("\nCRITIC SCORE:", result["critique_score"])
# print("FEEDBACK:", result["critique_feedback"])