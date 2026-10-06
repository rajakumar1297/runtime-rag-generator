# Runtime RAG Generator

An agent-built RAG (Retrieval-Augmented Generation) application that creates a searchable knowledge base from documents uploaded at runtime. Upload any combination of documents, and instantly ask AI-powered questions with source-grounded answers — no code changes needed for different document sets.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (Single-Page App)                  │
│         Upload Zone │ Collections Panel │ Chat Interface     │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API
┌──────────────────────────┴──────────────────────────────────┐
│                  FastAPI Backend                             │
│  ┌──────────┐  ┌────────────┐  ┌─────────────────────────┐ │
│  │ Document  │  │  Vector    │  │     RAG Chain           │ │
│  │ Ingestion │──│  Store     │──│ (Retrieve → Prompt →    │ │
│  │ Pipeline  │  │ (ChromaDB) │  │  LLM → Answer)         │ │
│  └──────────┘  └────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Server** | FastAPI | REST API + static file serving |
| **Document Parsing** | PyPDF, python-docx | Extract text from PDF, DOCX, TXT, CSV, MD |
| **Text Splitting** | LangChain RecursiveTextSplitter | Chunk documents with overlap for better retrieval |
| **Embeddings** | Sentence-Transformers (local) or OpenAI | Convert text chunks to vectors |
| **Vector Store** | ChromaDB | Persistent vector storage & similarity search |
| **LLM** | OpenAI GPT-4o-mini / Google Gemini | Generate grounded answers from retrieved context |
| **Orchestration** | LangChain | RAG chain composition |
| **Frontend** | Vanilla HTML/CSS/JS | Drag-drop upload, collection management, chat UI |

## Features

- **Runtime document upload** — PDF, DOCX, TXT, CSV, Markdown
- **Named collections** — organize documents into separate knowledge bases
- **Source-grounded answers** — every answer cites which document chunks were used
- **Multiple LLM support** — swap between OpenAI and Google Gemini via config
- **Local embeddings** — default uses free sentence-transformers (no API key needed for embeddings)
- **Persistent storage** — ChromaDB stores vectors on disk, survives restarts
- **Beautiful web UI** — dark theme, drag-drop uploads, real-time chat

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Google API key
```

**Minimum required:** Get a **free** Google Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and set `GOOGLE_API_KEY` in `.env`. The default config uses Google Gemini (free tier) with local embeddings (no cost).

Alternatively, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` to use OpenAI GPT-4o-mini.

### 3. Run

```bash
python run.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### 4. Use

1. **Upload** — Drag documents into the upload zone, name your collection, click "Upload & Index"
2. **Select** — Click on a collection in the sidebar to activate it
3. **Ask** — Type your question and get AI answers grounded in your documents

## Configuration

All settings are configured via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `google` | LLM backend: `google` or `openai` |
| `GOOGLE_API_KEY` | — | Your Google AI API key (free at aistudio.google.com) |
| `GOOGLE_MODEL` | `gemini-2.0-flash` | Google model to use |
| `OPENAI_API_KEY` | — | Your OpenAI API key (alternative) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `EMBEDDING_PROVIDER` | `local` | `local` (free) or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `CHUNK_SIZE` | `1000` | Text chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/api/upload` | Upload documents (multipart form) |
| `POST` | `/api/query` | Ask a question (JSON body) |
| `GET` | `/api/collections` | List all collections |
| `GET` | `/api/collections/{name}` | Get collection details |
| `DELETE` | `/api/collections/{name}` | Delete a collection |
| `GET` | `/api/health` | Health check |

## Design Decisions

1. **ChromaDB over FAISS/Pinecone** — Zero setup, runs locally, persists to disk, good enough for the use case
2. **Local embeddings by default** — Sentence-transformers requires no API key and runs on CPU; keeps the barrier to entry low
3. **LangChain for orchestration** — Provides clean abstractions for the RAG pipeline while keeping individual components swappable
4. **Single-file frontend** — No build step needed; keeps deployment simple while still delivering a polished UI
5. **Collection-based architecture** — Users can maintain separate knowledge bases and switch between them without code changes

## Project Structure

```
runtime-rag-generator/
├── app/
│   ├── __init__.py
│   ├── config.py         # Settings from environment variables
│   ├── ingest.py         # Document parsing & chunking
│   ├── vectorstore.py    # ChromaDB operations & embeddings
│   ├── rag_chain.py      # LLM chain for Q&A
│   └── main.py           # FastAPI routes & application
├── static/
│   └── index.html        # Web frontend
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt
├── run.py                 # Application entry point
└── README.md
```
