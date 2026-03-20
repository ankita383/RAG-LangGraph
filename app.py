import os
import shutil
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from src.ingest import process_pdf_to_vectorstore
from src.graph import create_rag_graph

load_dotenv()

app = FastAPI(title="LangGraph RAG API")

PDF_DIR = "data/uploads"
VECTOR_PATH = "vectorstore/faiss_index"

rag_app = None


@app.on_event("startup")
async def startup_event():
    global rag_app
    print("--- SYSTEM STARTUP ---")

    default_pdf = os.path.join(PDF_DIR, "sample.pdf")

    if not os.path.exists(default_pdf):
        print(f"No default PDF found at {default_pdf}. Waiting for upload via /upload.")
        return

    print("Default PDF found. Ingesting...")
    retriever = process_pdf_to_vectorstore(default_pdf, VECTOR_PATH)
    rag_app = create_rag_graph(retriever)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str


class UploadResponse(BaseModel):
    filename: str
    message: str


@app.get("/")
def read_root():
    return {"status": "Online", "mode": "RAG-Enabled"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file to replace the active knowledge base.
    The system will re-ingest the document and rebuild the vector index.
    """
    global rag_app

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    os.makedirs(PDF_DIR, exist_ok=True)

    save_path = os.path.join(PDF_DIR, file.filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        file.file.close()

    print(f"--- Uploaded: {file.filename}. Re-ingesting... ---")

    try:
        retriever = process_pdf_to_vectorstore(save_path, VECTOR_PATH)
        rag_app = create_rag_graph(retriever)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        filename=file.filename,
        message="PDF uploaded and indexed successfully. You can now use /ask."
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if rag_app is None:
        raise HTTPException(
            status_code=503,
            detail="No PDF has been loaded yet. Please upload one via POST /upload."
        )

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