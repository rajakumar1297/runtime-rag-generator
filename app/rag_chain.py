"""
RAG chain module.
Builds the retrieval-augmented generation chain using LangChain.
"""

import logging
from typing import Dict, List

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from app.config import settings
from app.vectorstore import similarity_search

logger = logging.getLogger(__name__)

# Module-level cache for the LLM
_llm = None


def get_llm():
    """
    Get the LLM based on configuration.
    Supports OpenAI and Google Gemini.
    """
    global _llm
    if _llm is not None:
        return _llm

    if settings.llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        _llm = ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
            convert_system_message_to_human=True,
        )
    else:
        # Default: OpenAI
        from langchain_openai import ChatOpenAI

        _llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.2,
        )

    logger.info(f"Initialized LLM: {settings.llm_provider}")
    return _llm


# RAG Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions based on the provided context documents. 

Rules:
1. Answer ONLY based on the provided context. Do not use prior knowledge.
2. If the context doesn't contain enough information to answer, say so clearly.
3. Cite which source document(s) your answer comes from when possible.
4. Be concise but thorough in your answers.
5. If the question is ambiguous, interpret it in the most reasonable way given the context.""",
        ),
        (
            "human",
            """Context from documents:
{context}

Question: {question}

Answer based on the context above:""",
        ),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a context string."""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        formatted_parts.append(
            f"[Source {i}: {source} (chunk {chunk_idx})]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_parts)


def query_rag(
    collection_name: str, question: str, k: int = 4
) -> Dict:
    """
    Run a RAG query against a document collection.

    Args:
        collection_name: The collection to query.
        question: The user's question.
        k: Number of context documents to retrieve.

    Returns:
        Dict with 'answer', 'sources', and 'context' keys.
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = similarity_search(collection_name, question, k=k)

    if not retrieved_docs:
        return {
            "answer": "No relevant documents were found in the collection to answer your question. Please upload documents first.",
            "sources": [],
            "context": "",
        }

    # Step 2: Format context
    context = format_docs(retrieved_docs)

    # Step 3: Build and invoke the chain
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({"context": context, "question": question})

    # Step 4: Extract source information
    sources = []
    seen_sources = set()
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(
                {
                    "filename": source,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                }
            )

    logger.info(
        f"RAG query on '{collection_name}': '{question[:50]}...' -> "
        f"{len(sources)} sources used"
    )

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }
