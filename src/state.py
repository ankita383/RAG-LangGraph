from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    