"""
Initializes the Retrieval-Augmented Generation (RAG) pipeline.

This pipeline integrates document loading, text chunking, embedding generation,
vector storage, reranking, and large language model (LLM) inference to provide
a complete workflow for answering user queries with contextually relevant
documents. It also supports context relevance checking and optional caching
through `rag_cache`.

Attributes:
    loader (FileLoader): Loads documents from various sources.
    chunker (TextChunker): Splits documents into parent and child chunks.
    dense_embedder (DenseEmbedder): Generates dense embeddings for text.
    sparse_embedder (SparseEmbedder): Generates sparse embeddings for text.
    vectorstore (QdrantStore): Stores and retrieves vector embeddings.
    reranker (Rerank): Reranks retrieved documents using a cross-encoder model.
    llm (LLM): Large language model used for query rewriting and response generation.
    system_prompt (str): System prompt to guide the LLM behavior.
    context_relevance_prompt (str): Prompt used to assess context relevance.
    context_relevance (bool): Whether to check context relevance before response generation.
    session (AsyncSessionLocal): Database session factory for retrieving parent chunks.
    rag_cache (SemanticCache, optional): Semantic cache for storing and retrieving previous queries.
"""
from src.rag import RagPipeline
from src.builder import *

# Initialize Rag pipeline
pipeline = RagPipeline(
        loader=loader,
        chunker=chunker,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        vectorstore=qdrant_store,
        reranker=reranker,
        llm=llm,
        system_prompt=system_prompt,
        context_relevance_prompt=context_relevance_prompt,
        context_relevance=True,
        session=AsyncSessionLocal,
        rag_cache=rag_cache
    )
