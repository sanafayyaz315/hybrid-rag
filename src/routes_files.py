import os, shutil, logging, asyncio
from fastapi import APIRouter, UploadFile, Depends, HTTPException, File as fastapi_file
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from init_pipeline import rag_pipeline as pipeline
from config import TEMP_FILE_DOWNLOAD_DIR
from docstore import SessionLocal, AsyncSessionLocal
from file import (
                    ensure_unique_filenames,
                    upload_all_to_minio, 
                    stage_file_rows,
                    download_file_from_minio,
                    run_indexing_pipeline,
                    delete_from_docstore,
                    list_files,
                    async_delete_from_docstore
)
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Dependency to get DB session per request
def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Async dependency to get a session
async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

router = APIRouter()

# this func is not being used by the FE currently. The FE directly runs the query on the docstore to get files
@router.get("/get_files")
def get_files(session: Session = Depends(get_db)) -> Dict[str, Any]:
    try:
        files = list_files(session)
        if files:
            return {"count": len(files), "files": files}
        else:
            return {"count": 0, "files": []}
    except Exception as e:
        logger.exception(f"Unable to get the list of files due to {e}")

@router.post("/upload_files/")
async def upload_files(file: UploadFile = fastapi_file(...), 
                       session: Session = Depends(get_db)) -> Dict[str, Any]:
                    #    session: AsyncSession = Depends(get_async_db)) -> Dict[str, Any]:
                
    upload_dir = TEMP_FILE_DOWNLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    temp_dir = TEMP_FILE_DOWNLOAD_DIR

    try:
        # 1) Guard: no dupes (bulk check)
        ensure_unique_filenames(session, file)
        # await async_ensure_unique_filenames(session, file)
        # 2) Upload to MinIO (pure IO, no DB yet)
        meta = await upload_all_to_minio(file)
        # 3) Stage DB rows (add+flush, still uncommitted)
        staged_file = stage_file_rows(session, meta)
        # 4) Pull back to local temp
        download_file_from_minio(meta, temp_dir)
        # 5) Index (chunks + embeddings)
        num_parent_chunks, num_child_chunks = run_indexing_pipeline(session, temp_dir, staged_file, rag_pipeline=pipeline)
        # Commit once
        session.commit()
        logger.info(f"Saved {num_parent_chunks} parent chunks in docstore and {num_child_chunks} in vectorDB")
        
        return {
            "status": "success",
            "file": [{"id": staged_file.id, "filename": staged_file.name}],
        }
    # catch any known HPPT exceptions
    except HTTPException:
        session.rollback()
        raise

    # catch general exceptions
    except Exception as e:
        session.rollback()
        logger.exception("Upload+Index failed — rolled back")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@router.delete("/delete_file/{filename}")

def delete_file(filename: str, session: Session = Depends(get_db)):
    try:
        res = delete_from_docstore(filename, db=session)
        if res:
            return {"message": f"File '{filename}' and related docstore entries deleted."}
        else:
            return {"message": f"File '{filename}' is not present in the docstore."}
    except Exception as e:
        logger.exception(f"File not deleted. The following exception occured while deleting file {filename}: {e}")
        raise e
    
@router.delete("/delete_file/async/{filename}")
async def async_delete_file(filename: str, session: AsyncSession = Depends(get_async_db)):
    try:
        res = await async_delete_from_docstore(filename, session)
        
        if res:
            return {"message": f"File '{filename}' and related docstore entries deleted."}
        else:
            return {"message": f"File '{filename}' is not present in the docstore."}
    except Exception as e:
        logger.exception(f"File not deleted. Exception while deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")


    

    







