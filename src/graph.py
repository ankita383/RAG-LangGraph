from langgraph.graph import END, StateGraph, START
from src.state import GraphState
from src.nodes import retrieve, generate

def create_rag_graph(retriever):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()