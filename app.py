import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.ingest import process_pdf_to_vectorstore
from src.graph import create_rag_graph

load_dotenv()

app = FastAPI(title="LangGraph RAG API")

PDF_PATH = "data/uploads/sample.pdf"
VECTOR_PATH = "vectorstore/faiss_index"


rag_app = None

@app.on_event("startup")
async def startup_event():
    global rag_app
    print("--- SYSTEM STARTUP ---")
    
    if not os.path.exists(PDF_PATH):
        print(f"CRITICAL: {PDF_PATH} not found. Please add the PDF.")
        return

    if not os.path.exists(os.path.join(VECTOR_PATH, "index.faiss")):
        print("Index not found in vectorstore/. Running ingestion...")
        retriever = process_pdf_to_vectorstore(PDF_PATH, VECTOR_PATH)
    else:
        print("Index found. Loading existing vectorstore...")
        retriever = process_pdf_to_vectorstore(PDF_PATH, VECTOR_PATH)

    rag_app = create_rag_graph(retriever)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"status": "Online", "mode": "RAG-Enabled"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if rag_app is None:
        raise HTTPException(status_code=503, detail="System initializing or PDF missing.")
    
    try:
        inputs = {"question": request.question}
        result = rag_app.invoke(inputs)
        
        return AnswerResponse(
            question=request.question,
            answer=result["generation"]
        )
    except Exception as e:
        print(f"Error during invocation: {e}")
        raise HTTPException(status_code=500, detail="Internal AI Error")

uvicorn.run(app, host="127.0.0.1", port=8000)