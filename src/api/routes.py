import logging
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, Depends, HTTPException, File as fastapi_file
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict
from src.docstore.models import File
from src.docstore.files_crud import (async_list_all_files, 
                                     async_list_files, 
                                     async_ensure_unique_filenames, 
                                     async_stage_file_rows, 
                                     async_delete_file_row
                                    )
from src.docstore.docstore_crud import async_upsert_parents_to_docstore
from src.docstore.session import get_async_session
from src.minio_utils import async_upload_file_to_minio, async_download_file
from src.builder import (
    llm,
    system_prompt,
    rewrite_query_prompt,
    context_relevance_prompt,
    minio_client,
    pipeline
    )
from src.docstore.session import AsyncSessionLocal
from src.rag_utils import rewrite_query
from src.config import MINIO_BUCKET as minio_bucket, TEMP_FILE_DOWNLOAD_DIR as temp_dir

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class FileInfo(BaseModel):
    id: int
    name: str

class UploadFileResponse(BaseModel):
    status: str
    file: FileInfo 

class FilesResponse(BaseModel):
    count: int
    files: List[str]

router = APIRouter()

@router.get("/get_all_files", response_model=FilesResponse)
async def get_all_files(session: AsyncSession = Depends(get_async_session))-> Dict[str, str]:
    try:
        files = await async_list_all_files(session)
        if files:
            logger.debug(f"Found {len(files)} files")
            return {"count": len(files), "files": files}
        else:
            return {"count": 0, "files": []}
    except Exception as e:
        logger.exception(f"Unable to get the list of files due to {e}")

@router.get("/get_collection_files", response_model=FilesResponse)
async def get_collection_files(session: AsyncSession = Depends(get_async_session))-> Dict[str, str]:
    try:
        files = await async_list_files(session)
        if files:
            logger.debug(f"Found {len(files)} files")
            return {"count": len(files), "files": files}
        else:
            return {"count": 0, "files": []}
    except Exception as e:
        logger.exception(f"Unable to get the list of files due to {e}")


@router.post("/upload_files/", response_model=UploadFileResponse)
async def upload_files(file: UploadFile = fastapi_file(...), 
                       session: AsyncSession = Depends(get_async_session)) -> Optional[File]:
    try:
        await async_ensure_unique_filenames(session=session, file=file)
        minio_meta = await async_upload_file_to_minio(minio_client, file, minio_bucket)
        logger.debug(f"uploaded {file.filename} to minio")
        staged_file = await async_stage_file_rows(session=session, minio_metadata=minio_meta)
        logger.debug(f"Staged {staged_file.name, staged_file.id}")
        save_file_path = await async_download_file(minio_client=minio_client, object_name=staged_file.name, local_dir=temp_dir, minio_bucket=minio_bucket)
        logger.debug(f"uploaded {file.filename} to minio")
        texts = pipeline.load(path=save_file_path)
        parent_chunks, child_chunks = pipeline.split(texts=texts, base_metadata=None)
        await pipeline.embed_and_index(texts=child_chunks, doc_type="documents")
        file_id_map = {staged_file.name: staged_file.id}
        saved = await async_upsert_parents_to_docstore(parent_chunks, session, file_id_map=file_id_map)
        logger.debug(f"Saved {saved} parents in docstore")
        await session.commit()
        logger.debug(f"Committed pipeline for file: {staged_file.name}, id: {staged_file.id} to the docstore database")
        return UploadFileResponse(
            status="success",
            file=FileInfo(id=staged_file.id, name=staged_file.name)
        )

    # catch any known HPPT exceptions
    except HTTPException:
        session.rollback()
        raise
    
    except Exception as e:
        await session.rollback()
        logger.exception(f"Failed to commit pipeline for file: {getattr(file, 'filename', None)} to the docstore database")
        raise RuntimeError(f"DB commit failed for file {getattr(file, 'filename', None)}: {str(e)}")

@router.delete("/delete_file/async/{filename}")
async def delete_file(filename: str, session: AsyncSession = Depends(get_async_session)):
    try:
        res = await async_delete_file_row(filename, session)
        if res:
            return {"message": f"File '{filename}' and related docstore entries deleted."}
        else:
            return {"message": f"File '{filename}' is not present in the docstore."}
    except Exception as e:
        logger.exception(f"File not deleted. Exception while deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

@router.post("/generate_reponse")
async def generate(user_query: str, 
                 session: AsyncSession = Depends(get_async_session),
                 system_prompt: str = system_prompt,
                 rewrite_query_prompt: str = rewrite_query_prompt,
                 context_relevance_prompt: str = context_relevance_prompt,
                 context_relevance: bool = True,
                 is_rewrite_query: bool = True,
                 rerank: bool = True,
                 top_k: int = 50,
                 rerank_top_k: int = 5,
                 retrieve_neighbors: bool = True
                 ):
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty or whitespace.")
    try:
        query = user_query
        if is_rewrite_query:
                query, sources = await rewrite_query(user_query=user_query, rewrite_query_prompt=rewrite_query_prompt, llm=llm)
                logger.debug(f"Actual query: {user_query}\nRewritten query: {query}\n Sources: {sources}")
        # TODO: check cache here before retrieval

        retrieved_docs = await pipeline.retrieve(query=query, 
                                            top_k=top_k, 
                                            rerank=rerank, 
                                            rerank_top_k=rerank_top_k,
                                            retrieve_neighbors=retrieve_neighbors,
                                            session=session
                                            )
        logger.debug(f"Retrieved {len(retrieved_docs)} docs")
        context = pipeline.build_context(retrieved_docs)
        stream, contexts = await pipeline.generate_response(query, context, system_prompt, context_relevance=True, context_relevance_prompt=context_relevance_prompt)
        logger.debug(f"Response generated successfully")
        return {"response": stream, "contexts": contexts}
    except Exception as e:
        logger.debug(f"Failed to generate response: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

    



    





