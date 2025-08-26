from rag_pipeline import RagPipeline
from docstore import SessionLocal
from config import (
    API_KEY,
    MODEL,
    COLLECTION,
    PARENT_CHUNKS_FILE, 
    PARENT_CHUNKS_DIR, 
    PARENT_CHUNK_SIZE, 
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    UPSERT_BATCH_SIZE,
    GET_NEIGHBORS,
    COLLECTION_RESOURCES,
    SYSTEM_PROMPT_PATH,
    REWRITE_QUERY_PROMPT_PATH,
    DENSE_EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    CROSS_ENCODER_MODEL
)

rag_pipeline = RagPipeline(
        
        collection_name=COLLECTION,
        llm_api_key=API_KEY,
        llm_model=MODEL,
        system_prompt_path=SYSTEM_PROMPT_PATH,
        rewrite_query_prompt_path=REWRITE_QUERY_PROMPT_PATH,
        dense_embedding_model_name=DENSE_EMBEDDING_MODEL,
        sparse_embedding_model_name=SPARSE_EMBEDDING_MODEL,
        cross_encoder_model_name=CROSS_ENCODER_MODEL, 
        upsert_batch_size=UPSERT_BATCH_SIZE,  
        top_k=50,
        rerank_top_k=3,
        parent_chunks_dir=PARENT_CHUNKS_DIR,
        parent_chunks_file=PARENT_CHUNKS_FILE,
        max_seq_len_embedding=512,
        parent_chunk_size=PARENT_CHUNK_SIZE,
        parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
        child_chunk_size=CHILD_CHUNK_SIZE,
        child_chunk_overlap=CHILD_CHUNK_OVERLAP,
        docstore_session=SessionLocal,
        get_neighbors=GET_NEIGHBORS,
        retriever_eval=False,
        collection_resources=COLLECTION_RESOURCES
    )