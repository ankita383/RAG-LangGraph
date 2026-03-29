import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_fireworks import FireworksEmbeddings
from rank_bm25 import BM25Okapi

def process_pdf_to_vectorstore(file_path: str, storage_path: str):
    embeddings = FireworksEmbeddings(
        model="fireworks/qwen3-embedding-8b",
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(f"Extracted {len(docs)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    vectorstore.save_local(storage_path)

    texts = [doc.page_content for doc in splits]
    tokenized_corpus = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "faiss": vectorstore,
        "bm25": bm25,
        "documents": splits
    }