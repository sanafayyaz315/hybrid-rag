import os, shutil, logging, asyncio
from typing import List, Dict
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import File 
from minio_utils import upload_file_to_minio, download_file
from config import COLLECTION

logger = logging.getLogger(__name__)

# func to check if a file name already exists in the files table
def ensure_unique_filenames(session: Session, file: UploadFile) -> None:
    file_name =  file.filename
    if not file_name:
        raise HTTPException(status_code=400, detail="No file provided.")
       
    # check if file already exists
    existing_file = session.query(File).filter(File.name == file_name).first()
    logger.debug(f"File {file_name} ALREADY exists")
    if existing_file:
        raise HTTPException(
            status_code=400,
            detail=f"FILE {file_name} ALREADY EXISTS. Delete it first before re-uploading."
            )   

async def async_ensure_unique_filenames(session: AsyncSession, file: UploadFile) -> None:
    file_name = file.filename
    if not file_name:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Async select
    result = await session.execute(select(File).where(File.name == file_name))
    existing_file = result.scalar_one_or_none()

    if existing_file:
        raise HTTPException(
            status_code=400,
            detail=f"FILE {file_name} ALREADY EXISTS. Delete it first before re-uploading."
        )    
        
# Upload file to Minio
async def upload_all_to_minio(file: UploadFile) -> Dict:
    """Return a metadata dict; include filename for later."""
    try:
        minio_metadata = await upload_file_to_minio(file)
        return minio_metadata
    except Exception as e:
        raise RuntimeError((f"Failed to save {file.filename} to MinIO: {str(e)}"))
    
# Stage file names in files table
def stage_file_rows(session: Session, minio_metadata: Dict) -> "File":
    """Add File rows and flush to get IDs. Not visible to others until commit."""
    try:
        new_file = File(
                        name=minio_metadata["object_name"],
                        file_metadata=minio_metadata,
                        qdrant_collection=COLLECTION
                    )
        session.add(new_file)
        session.flush()
        return new_file
    except Exception as e:
        raise RuntimeError((f"Failed to add a row for {minio_metadata['object_name']} in table files: {str(e)}"))

async def async_stage_file_rows(session: AsyncSession, minio_metadata: dict) -> File:
    """Add File rows and flush to get IDs. Async version for AsyncSession"""
    try:
        new_file = File(
            name=minio_metadata["object_name"],
            file_metadata=minio_metadata
        )
        session.add(new_file)
        await session.flush()  # async flush
        return new_file
    except Exception as e:
        raise RuntimeError(
            f"Failed to add a row for {minio_metadata['object_name']} in table files: {str(e)}"
        )
    
# Download new file to local dir for indexing
def download_file_from_minio(minio_metadata: Dict, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    try:
        download_file(minio_metadata['object_name'], dest_dir)
    except:
        logger.error(f"Failed to download file {minio_metadata['object_name']} from Minio")
        raise HTTPException(status_code=500, detail="Download from MinIO failed")

async def async_download_file_from_minio(minio_metadata: dict, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    try:
        # Wrap the sync download function in a thread
        await asyncio.to_thread(download_file, minio_metadata['object_name'], dest_dir)
    except Exception as e:
        logger.error(f"Failed to download file {minio_metadata['object_name']} from Minio: {str(e)}")
        raise HTTPException(status_code=500, detail="Download from MinIO failed")

#  Run pipeline (chunk + index)
def run_indexing_pipeline(session: Session, dest_dir: str, new_file: "File", rag_pipeline) -> None:
    print(f"new file: {new_file}")
    file_id_map = {new_file.name: new_file.id}
    try:
        rag_pipeline.process_file(session=session, save_file_path=dest_dir, file_id_map=file_id_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

async def async_run_indexing_pipeline(session: AsyncSession, dest_dir: str, new_file: File, rag_pipeline):
    print(f"new file: {new_file}")
    file_id_map = {new_file.name: new_file.id}
    try:
        # Assuming process_file is sync, wrap it in a thread
        await asyncio.to_thread(
            rag_pipeline.process_file,
            session,
            dest_dir,
            file_id_map
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    
# Delete files from docstore
def delete_from_docstore(filename: str, db: Session):
        file = db.query(File).filter_by(name=filename).first()
        if file:
            # Cascade delete all related docstore rows
            db.delete(file)
            db.commit()
            return True
        else:
            return False
        
async def async_delete_from_docstore(filename: str, db: AsyncSession):
    # Fetch the file row
    result = await db.execute(select(File).where(File.name == filename))
    file = result.scalar_one_or_none()  # gets the single object or None  
    if file:
        await db.delete(file)      # schedule deletion
        await db.commit()          # commit to DB
        return True
    else:
        return False

# List files
def list_files(session):
    from config import COLLECTION
    # rows = session.query(File).all()
    # filter on collection 
    rows = session.query(File).filter(File.qdrant_collection == COLLECTION).all()
    print(f"files: {rows}")
    files = []
    for r in rows:
        files.append(r.name)
    return files






     
    
    
    


    
