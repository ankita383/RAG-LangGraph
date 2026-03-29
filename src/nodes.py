import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve(state, retriever_bundle):
    print("---NODE: HYBRID RETRIEVAL---")

    question = state["question"]

    faiss = retriever_bundle["faiss"]
    bm25 = retriever_bundle["bm25"]
    docs = retriever_bundle["documents"]

    faiss_docs = faiss.similarity_search(question, k=5)

    tokenized_query = question.split()
    scores = bm25.get_scores(tokenized_query)

    top_n = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:5]

    bm25_docs = [docs[i] for i in top_n]

    combined_docs = faiss_docs + bm25_docs

    seen = set()
    unique_docs = []

    for d in combined_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)

    pairs = [(question, d.page_content) for d in unique_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(unique_docs, scores),
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

    question = state["question"]
    documents = state["documents"]

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