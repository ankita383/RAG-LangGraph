import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def retrieve(state, retriever):
    print("---NODE: RETRIEVING---")
    question = state["question"]
    documents = retriever.invoke(question) 
    return {"documents": documents, "question": question}

def generate(state):
    print("---NODE: GENERATING---")
    question = state["question"]
    documents = state["documents"]
    
    context = "\n\n".join([d.page_content for d in documents])
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": question})
    
    return {"generation": response}
