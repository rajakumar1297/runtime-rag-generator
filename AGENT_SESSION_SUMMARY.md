# Agentic Coding Assessment — Option A: RAG Generator

## Executive Summary & Candidate Brief Alignment

This project implements **Option A: RAG Generator** for the Senior AI Engineer evaluation. The objective was to build a production-grade, flexible Retrieval-Augmented Generation (RAG) system that:
1. **Accepts documents at runtime** (PDF, DOCX, TXT, CSV, Markdown)
2. **Dynamically creates a RAG application** over those documents
3. **Allows users to ask questions and receive grounded answers with citations**
4. **Works seamlessly with different document sets without code changes**
5. **Provides full agent transcripts and clear documentation of the session**

---

## 1. Problem Framing & Strategic Decisions

### A. Runtime Multi-Tenancy via Named Collections
Instead of a monolithic or static vector index, the system uses a **Collection-Based Vector Architecture** backed by ChromaDB. 
- Users can upload different document sets under arbitrary collection names (e.g., `company-policies`, `q3-earnings`, `tech-architecture`).
- Each collection is isolated in vector space.
- Switching between document sets or domains requires **zero code modifications or server restarts**.

### B. Free & Open-Source First Stack
To ensure frictionless execution and zero mandatory cloud API costs for basic usage:
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) runs completely locally on CPU with normalized dense embeddings. No OpenAI or cloud embedding credits required.
- **LLM Tier**: Configured for **Google Gemini 2.0 Flash** via Google AI Studio's generous free tier, with modular drop-in support for **OpenAI GPT-4o-mini** via the `.env` configuration file.
- **Vector DB**: ChromaDB embedded persistent store on disk (`./chroma_db`) requiring zero Docker containers or external cloud databases.

### C. Grounded Answers & Source Attribution
To eliminate hallucinations:
- Strict system prompt directives enforce that answers are derived *solely* from retrieved context.
- When context is insufficient, the system explicitly reports lack of supporting data rather than inventing answers.
- Every response returns source metadata: filename, chunk index, and an excerpt of the text chunk used for generation.

---

## 2. Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (Single-Page App)                  │
│       • Drag-and-drop runtime upload zone                   │
│       • Dynamic collection management & switcher            │
│       • Real-time chat interface with source inspection     │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / REST
┌──────────────────────────┴──────────────────────────────────┐
│                  FastAPI Backend Server                      │
│                                                             │
│  ┌───────────────────────┐     ┌────────────────────────┐  │
│  │ Document Parsers       │     │ Vector Store (Chroma)  │  │
│  │ • PDF (pypdf)         │     │ • HNSW indexing        │  │
│  │ • DOCX (python-docx)  │────▶│ • Persistent storage   │  │
│  │ • CSV, TXT, Markdown  │     │ • all-MiniLM-L6-v2     │  │
│  └───────────────────────┘     └───────────┬────────────┘  │
│                                            │                │
│                                    Top-K Retrieval          │
│                                            ▼                │
│                                ┌────────────────────────┐   │
│                                │ LangChain RAG Pipeline │   │
│                                │ • Grounded Prompting   │   │
│                                │ • Google Gemini/OpenAI │   │
│                                └────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
runtime-rag-generator/
├── app/
│   ├── __init__.py
│   ├── config.py           # Pydantic Settings & environment variable validation
│   ├── ingest.py           # Multi-format document parser & semantic chunker
│   ├── vectorstore.py      # ChromaDB client & embedding abstraction
│   ├── rag_chain.py        # LangChain LCEL RAG prompt & generation chain
│   └── main.py             # FastAPI REST endpoints & static file server
├── static/
│   └── index.html          # Polished, responsive dark-mode UI with vanilla JS
├── sample_docs/            # Pre-packaged test datasets for immediate evaluation
│   ├── company_policies.txt
│   ├── ai_architecture_guide.md
│   └── quarterly_earnings_report.csv
├── .env.example            # Environment configuration template
├── .gitignore              # Ignores .env, chroma_db/, uploads/, caches
├── requirements.txt        # Pinned dependencies
├── run.py                  # Server entry point
├── README.md               # User manual & architectural documentation
└── AGENT_SESSION_SUMMARY.md # This document
```

---

## 4. How to Test and Run the System

### Prerequisites
- Python 3.10+
- A free Google Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (or an OpenAI key)

### Step 1: Clone and Configure
```bash
git clone https://github.com/rajakumar1297/runtime-rag-generator.git
cd runtime-rag-generator

# Copy template and add your API key
cp .env.example .env
```
In `.env`, set:
```ini
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### Step 2: Install and Launch
```bash
pip install -r requirements.txt
python run.py
```
Open **`http://localhost:8000`** in any web browser.

### Step 3: Run the Test Flow
1. **Upload Dataset A**: Enter `hr-policies` in the Collection Name input, drag-and-drop `sample_docs/company_policies.txt`, and click **Upload & Index**.
2. **Ask Questions**: Ask *"How many PTO days do employees receive?"* or *"What is the equipment refresh cycle?"* Verify grounded answers with source citations.
3. **Upload Dataset B (Zero Code Changes)**: Enter `financial-reports`, upload `sample_docs/quarterly_earnings_report.csv`, and index it.
4. **Switch Collections**: Select `financial-reports` in the sidebar and ask *"What was the operating margin of the Cloud AI Solutions division in Q2-2026?"*
5. Notice that both collections remain independently queryable without restarts or code updates.

---

## 5. Agentic Coding Methodology & Evaluation Criteria

| Evaluation Dimension | Agentic Action Taken |
|----------------------|----------------------|
| **Problem Framing** | Decomposed the brief into modular layers: Ingestion, Vector Persistence, Generation Chain, REST Layer, and Frontend. Identified the core differentiator: runtime collections without code changes. |
| **Architectural Decisions** | Selected ChromaDB + local sentence-transformers to minimize setup friction and guarantee zero cloud embedding charges. Isolated configuration into Pydantic models. |
| **Directing the Agent** | Steered the tech stack toward a free open-source tier (Google Gemini free API + local embeddings), requested sample datasets for turnkey grading, and maintained complete transparency. |
| **Output Validation** | Validated document parser coverage across 5 file extensions (`.pdf`, `.docx`, `.txt`, `.csv`, `.md`), ensured source metadata tracking down to chunk index, and generated end-to-end documentation. |
| **Transcript Preservation** | Included complete, unedited agent interaction logs in the repository for full auditability. |
