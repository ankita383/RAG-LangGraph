import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.ingest import process_pdf_to_vectorstore
from src.graph import create_rag_graph
import uvicorn

load_dotenv()

app = FastAPI(title="LangGraph RAG API")

PDF_PATH = "data/uploads/sample.pdf"
VECTOR_PATH = "vectorstore/faiss_index"

print("Initializing RAG System...")
retriever = process_pdf_to_vectorstore(PDF_PATH, VECTOR_PATH)
rag_app = create_rag_graph(retriever)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"status": "RAG API is online"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        inputs = {"question": request.question}
        result = rag_app.invoke(inputs)
        
        return AnswerResponse(
            question=request.question,
            answer=result["generation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

uvicorn.run(app, host="127.98.0.0", port=8000)