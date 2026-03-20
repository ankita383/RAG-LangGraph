import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_fireworks import FireworksEmbeddings

def process_pdf_to_vectorstore(file_path: str, storage_path: str):
    embeddings = FireworksEmbeddings(
        model="fireworks/qwen3-embedding-8b",
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(f"Extracted {len(docs)} pages")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(storage_path)
    
    return vectorstore.as_retriever()