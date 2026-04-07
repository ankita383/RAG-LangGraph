import os
import re
from typing import List, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

load_dotenv()

def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def retrieve(state, retriever_bundle):
    print("---NODE: HYBRID RETRIEVAL---")
    question: str = state["question"]
    qdrant = retriever_bundle["qdrant"]
    bm25 = retriever_bundle["bm25"]
    docs: List[Document] = retriever_bundle["documents"]

    qdrant_docs: List[Document] = qdrant.similarity_search(question, k=5)

    tokenized_query = bm25_tokenize(question)
    bm25_scores = bm25.get_scores(tokenized_query)

    top_n = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]

    bm25_docs = [docs[i] for i in top_n]

    combined_docs = qdrant_docs + bm25_docs

    seen = set()
    unique_docs: List[Document] = []

    for d in combined_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    if not unique_docs:
        print("No documents retrieved!")
        return {
            "documents": [],
            "question": question
        }

    reranker = get_reranker()

    pairs: List[Tuple[str, str]] = [
        (question, d.page_content) for d in unique_docs
    ]

    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(unique_docs, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_docs = [doc for doc, _ in ranked[:5]]

    print(f"Retrieved {len(final_docs)} documents after reranking")

    return {
        "documents": final_docs,
        "question": question
    }


def generate(state):
    print("---NODE: GENERATING---")

    question: str = state["question"]
    documents: List[Document] = state["documents"]

    if not documents:
        return {"generation": "No relevant context found."}

    context = "\n\n".join([d.page_content for d in documents])

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
    )

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return {"generation": response}
