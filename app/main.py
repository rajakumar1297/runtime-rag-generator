"""
FastAPI application for the Runtime RAG Generator.
Provides REST API endpoints for document upload, collection management, and Q&A.
"""

import os
import re
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.ingest import ingest_file, SUPPORTED_EXTENSIONS
from app.vectorstore import (
    add_documents,
    list_collections,
    delete_collection,
    get_collection_info,
)
from app.rag_chain import query_rag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Runtime RAG Generator",
    description=(
        "Upload documents at runtime and ask questions with "
        "retrieval-augmented generation."
    ),
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------- Pydantic Models ----------


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    collection_name: str
    question: str
    num_results: int = 4


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: list
    collection_name: str


class UploadResponse(BaseModel):
    """Response model for document uploads."""
    message: str
    collection_name: str
    files_processed: List[str]
    total_chunks: int


class CollectionInfo(BaseModel):
    """Response model for collection information."""
    name: str
    document_count: int
    sources: Optional[List[str]] = None


# ---------- API Routes ----------


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>Frontend not found. Place index.html in /static/</h1>")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(...),
):
    """
    Upload one or more documents and ingest them into a named collection.

    - Supports: PDF, DOCX, TXT, CSV, MD
    - Documents are chunked and embedded into the vector store
    - The collection_name groups related documents together
    """
    # Sanitize collection name (ChromaDB requirements)
    sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", collection_name.strip())
    if len(sanitized_name) < 3:
        sanitized_name = sanitized_name + "_collection"
    if len(sanitized_name) > 63:
        sanitized_name = sanitized_name[:63]

    processed_files = []
    total_chunks = 0
    errors = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            errors.append(
                f"Skipped '{file.filename}': unsupported format ({ext})"
            )
            continue

        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())[:8]
        temp_path = os.path.join(
            settings.upload_dir, f"{file_id}_{file.filename}"
        )

        try:
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)

            # Ingest the file
            documents = ingest_file(temp_path, file.filename, sanitized_name)
            add_documents(sanitized_name, documents)

            processed_files.append(file.filename)
            total_chunks += len(documents)
            logger.info(
                f"Ingested '{file.filename}' -> {len(documents)} chunks "
                f"into '{sanitized_name}'"
            )

        except Exception as e:
            errors.append(f"Error processing '{file.filename}': {str(e)}")
            logger.error(f"Error ingesting {file.filename}: {e}", exc_info=True)

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not processed_files:
        detail = "No files were processed successfully."
        if errors:
            detail += " Errors: " + "; ".join(errors)
        raise HTTPException(status_code=400, detail=detail)

    message = f"Successfully processed {len(processed_files)} file(s) into collection '{sanitized_name}'."
    if errors:
        message += f" Warnings: {'; '.join(errors)}"

    return UploadResponse(
        message=message,
        collection_name=sanitized_name,
        files_processed=processed_files,
        total_chunks=total_chunks,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question against a document collection.

    The system retrieves relevant document chunks and uses an LLM
    to generate a grounded answer with source citations.
    """
    # Validate collection exists
    info = get_collection_info(request.collection_name)
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_name}' not found. "
            f"Upload documents first.",
        )

    if info["document_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{request.collection_name}' is empty.",
        )

    try:
        result = query_rag(
            collection_name=request.collection_name,
            question=request.question,
            k=request.num_results,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            collection_name=request.collection_name,
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@app.get("/api/collections", response_model=List[CollectionInfo])
async def get_collections():
    """List all available document collections."""
    try:
        collections = list_collections()
        return [CollectionInfo(**col) for col in collections]
    except Exception as e:
        logger.error(f"Error listing collections: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection(collection_name: str):
    """Get detailed information about a specific collection."""
    info = get_collection_info(collection_name)
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found.",
        )
    return CollectionInfo(**info)


@app.delete("/api/collections/{collection_name}")
async def remove_collection(collection_name: str):
    """Delete a document collection and all its embeddings."""
    try:
        delete_collection(collection_name)
        return {"message": f"Collection '{collection_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
    }
