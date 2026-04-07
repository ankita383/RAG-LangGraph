import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_fireworks import FireworksEmbeddings
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def process_pdf_to_vectorstore(file_path: str):
    embeddings = FireworksEmbeddings(
        model="fireworks/qwen3-embedding-8b",
        fireworks_api_key=os.getenv("FIREWORKS_API_KEY")
    )
    
    collection_name = "pdf_docs"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(f"Extracted {len(docs)} pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60
    )

    splits: List[Document] = text_splitter.split_documents(docs)

    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url="http://localhost:6333",
        api_key="EmailAI_VectorKey_123",
        collection_name=collection_name,
    )

    print(f"Stored {len(splits)} chunks in Qdrant")

    texts = [doc.page_content for doc in splits]

    tokenized_corpus = [
        bm25_tokenize(text) for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    print(f"BM25 index built with {len(tokenized_corpus)} documents")

    return {
        "qdrant": vectorstore,
        "bm25": bm25,
        "documents": splits
    }