# Acme Corp - AI & Engineering Architecture Guide (2026)

## 1. Overview
Acme Corp utilizes a hybrid AI architecture combining agentic workflows with Retrieval-Augmented Generation (RAG). The primary goal is delivering fast, grounded intelligence to internal teams without leaking sensitive IP to external model providers.

## 2. Infrastructure & Model Serving
- **LLM Tier**: Proprietary workloads use local quantized models hosted on internal clusters via vLLM. External customer-facing chatbots route queries through Google Gemini 2.0 Flash or OpenAI GPT-4o-mini with enterprise zero-retention data privacy guarantees.
- **Vector Database**: High-density embeddings are persisted in ChromaDB collections, partitioned by project domain and authorization tier.
- **Embedding Specifications**: Dense vector representation uses 384-dimensional embeddings produced by `all-MiniLM-L6-v2`. Cosine similarity metric is enforced with an HNSW indexing index.

## 3. Ingestion & Pre-processing Pipeline
- All inbound documents (PDFs, Markdown notes, DOCX files, CSV datasets) undergo recursive semantic chunking.
- Chunk sizing target: 1000 characters with an overlap of 200 characters to preserve sentence boundaries.
- Metadata retention: Document source filename, chunk sequence index, and ingested collection timestamp are attached to every chunk payload.

## 4. Query Routing & Latency SLAs
- Top-k retrieval default: 4 chunks.
- Generation temperature: 0.2 to prioritize factual grounding and citation accuracy.
- Maximum allowable p95 query latency: 1.8 seconds end-to-end.
