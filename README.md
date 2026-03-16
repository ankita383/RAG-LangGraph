# LangGraph RAG System with FastAPI

A professional Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **FastAPI**, and **Groq**. This project enables high-speed, accurate Q&A over PDF documents using state-of-the-art embedding models and a graph-based orchestration logic.

## 🚀 Key Features
- **Graph-Based Workflow:** Powered by LangGraph for modular and controllable AI reasoning.
- **FastAPI Interface:** Ready-to-use API endpoints for seamless integration with frontends or external services.
- **High-Precision Embeddings:** Utilizes `thenlper/gte-large` (1024 dimensions) for superior sentence-level semantic similarity.
- **Self-Healing Vector Store:** Automatically detects if the FAISS index is missing and builds it from the source PDF on startup.
- **Optimized Chunking:** Uses a 600-character chunk size with a 60-character overlap to maintain context integrity.

## 🛠️ Technology Stack
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration:** [LangGraph](https://www.langchain.com/langgraph)
- **LLM:** Llama 3.3-70B (via [Groq](https://groq.com/))
- **Embeddings:** `thenlper/gte-large`
- **Vector Database:** FAISS (Local)
- **Package Manager:** [uv](https://github.com/astral-sh/uv)

## 🚀 System Architecture
Here is the visual flow of how your PDF is processed and how questions are answered:

  ```mermaid
  flowchart TD
    subgraph Initialization [Initialization & Ingestion]
        A[sample.pdf] --> B[PyPDFLoader Extraction]
        B --> C[Recursive Character Splitting - 600 chars]
        C --> D[gte-large Embedding Model]
        D --> E[(FAISS Vector Database)]
    end

    subgraph API_Flow [FastAPI & LangGraph Logic]
        F[User Question via /ask] --> G[LangGraph: Retrieve Node]
        G --> H[Similarity Search in FAISS]
        H --> I[Relevant Context Chunks]
        I --> J[LangGraph: Generate Node]
        J --> K[Groq: Llama 3.3-70B]
    end

    K --> L[Final Answer Response]

    %% Styling
    style E fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
    style A fill:#dfd,stroke:#333
  ```
## 📋 Prerequisites
Ensure you have the following environment variables set in a `.env` file in the project root:
```text
GROQ_API_KEY=your_groq_api_key
FIREWORKS_API_KEY=your_fireworks_api_key

```

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone <your-repository-url>
cd New-project

```


2. **Install dependencies:**
Using `uv` (recommended):
```bash
uv sync

```


3. **Prepare Data:**
Place your source PDF in `data/uploads/sample.pdf`.

## 🏃 Running the Application

Start the FastAPI server:

```bash
uv run python app.py

```

The server will be available at `http://127.0.0.1:8000`.

## 🧪 Testing the API

1. Navigate to the interactive documentation: `http://127.0.0.1:8000/docs`
2. Open the **POST /ask** endpoint.
3. Click **"Try it out"** and use the following JSON format:
```json
{
  "question": "Who gave the welcome speech at the event?"
}

```



## 📁 Project Structure

```text
├── app.py              # FastAPI server & Self-healing logic
├── src/
│   ├── ingest.py       # PDF processing & VectorStore creation
│   ├── graph.py        # LangGraph workflow definition
│   └── nodes.py        # Retrieval & Generation node logic
├── data/
│   └── uploads/        # Source PDFs
├── vectorstore/        # Local FAISS index (Git ignored)
├── .env                # API Keys (Git ignored)
└── pyproject.toml      # Project dependencies

```

## 🛡️ Git Strategy

The `vectorstore/` folder and `.env` file are ignored by Git to ensure security and repository cleanliness. The system will recreate the vector store locally upon the first run based on the files in `data/uploads/`.

```

---

### How to use this:
1. Create a new file named `README.md` in your `New-project` folder.
2. Paste the text above into it.
3. Save it.
