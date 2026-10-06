"""
Vector store module.
Manages ChromaDB collections for storing and retrieving document embeddings.
"""

import logging
from typing import List, Optional, Dict

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level cache for the embedding function
_embedding_function = None
_chroma_client = None


def get_embedding_function():
    """
    Get the embedding function based on configuration.
    Cached at module level for reuse.
    """
    global _embedding_function
    if _embedding_function is not None:
        return _embedding_function

    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _embedding_function = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    else:
        # Default: local sentence-transformers (free, no API key)
        _embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    logger.info(
        f"Initialized embedding function: {settings.embedding_provider} "
        f"({settings.embedding_model})"
    )
    return _embedding_function


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
        logger.info(f"ChromaDB client initialized at {settings.chroma_persist_dir}")
    return _chroma_client


def get_vectorstore(collection_name: str) -> Chroma:
    """
    Get a LangChain Chroma vectorstore for a specific collection.

    Args:
        collection_name: Name of the ChromaDB collection.

    Returns:
        LangChain Chroma vectorstore instance.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_documents(collection_name: str, documents: List[Document]) -> int:
    """
    Add documents to a ChromaDB collection.

    Args:
        collection_name: Target collection name.
        documents: List of LangChain Document objects to add.

    Returns:
        Number of documents added.
    """
    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(documents)
    logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
    return len(documents)


def similarity_search(
    collection_name: str, query: str, k: int = 4
) -> List[Document]:
    """
    Perform similarity search against a collection.

    Args:
        collection_name: Collection to search.
        query: Search query string.
        k: Number of results to return.

    Returns:
        List of most relevant Document objects.
    """
    vectorstore = get_vectorstore(collection_name)
    results = vectorstore.similarity_search(query, k=k)
    logger.info(
        f"Similarity search in '{collection_name}' for '{query[:50]}...' "
        f"returned {len(results)} results"
    )
    return results


def list_collections() -> List[Dict]:
    """
    List all available collections with their document counts.

    Returns:
        List of dicts with collection info.
    """
    client = get_chroma_client()
    collections = client.list_collections()
    result = []
    for col in collections:
        collection = client.get_collection(col.name)
        result.append(
            {
                "name": col.name,
                "document_count": collection.count(),
            }
        )
    return result


def delete_collection(collection_name: str) -> bool:
    """
    Delete a collection from ChromaDB.

    Args:
        collection_name: Name of the collection to delete.

    Returns:
        True if deleted successfully.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection '{collection_name}': {e}")
        raise


def get_collection_info(collection_name: str) -> Optional[Dict]:
    """
    Get detailed information about a specific collection.

    Args:
        collection_name: Name of the collection.

    Returns:
        Dict with collection details or None if not found.
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        # Get unique sources from metadata
        all_data = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return {
            "name": collection_name,
            "document_count": collection.count(),
            "sources": list(sources),
        }
    except Exception:
        return None
