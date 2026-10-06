"""
Entry point for the Runtime RAG Generator.
Run with: python run.py
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("=" * 60)
    print("  Runtime RAG Generator")
    print("=" * 60)
    print(f"  LLM Provider:       {settings.llm_provider}")
    print(f"  Embedding Provider:  {settings.embedding_provider}")
    print(f"  Embedding Model:     {settings.embedding_model}")
    print(f"  Chunk Size:          {settings.chunk_size}")
    print(f"  Chunk Overlap:       {settings.chunk_overlap}")
    print("=" * 60)
    print(f"  Open http://localhost:{settings.port} in your browser")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
