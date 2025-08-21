import os
import logging
import shutil
from fastapi import FastAPI, UploadFile, Depends, HTTPException, File as fastapi_file
from sqlalchemy.orm import Session
from rag_pipeline import RagPipeline
from typing import List
from docstore import SessionLocal, engine
from models import File, Docstore
from minio_utils import upload_file_to_minio, download_file_from_minio
from config import (
    API_KEY,
    COLLECTION,
    PARENT_CHUNKS_FILE, 
    PARENT_CHUNKS_DIR, 
    PARENT_CHUNK_SIZE, 
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    UPSERT_BATCH_SIZE,
    TEMP_FILE_DOWNLOAD_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI()

pipeline = RagPipeline(
    llm_api_key=API_KEY,
    collection_name=COLLECTION,
    rerank_top_k=3,
    parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
    parent_chunk_size=PARENT_CHUNK_SIZE,
    child_chunk_size=CHILD_CHUNK_SIZE,
    child_chunk_overlap=CHILD_CHUNK_OVERLAP,
    parent_chunks_dir=PARENT_CHUNKS_DIR,
    parent_chunks_file=PARENT_CHUNKS_FILE,
    upsert_batch_size=UPSERT_BATCH_SIZE,
    docstore_session=SessionLocal
)

# Dependency to get DB session per request
def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@app.post("/upload-files/")
async def upload_files(files: List[UploadFile] = fastapi_file(...), session: Session= Depends(get_db)):
    upload_dir = TEMP_FILE_DOWNLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = []
    staged_files_db = []
    file_id_map = {}

    try:
        # loading and saving files to a local path
        for file in files:
            # check if file already exists
            existing_file = session.query(File).filter(File.name == file.filename).first()
            logger.debug(f"File {file.filename} already exists")
            if existing_file:
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' already exists. Delete it first before re-uploading."
                )    
            # to save files locally
            # try:
                 # file_path = save_files_locally(upload_dir, file)
                 # saved_files.append(file_path)
                 # saved
                 
            # try except will raise error if any file fails to save
            # Save file to minio
            try:
                minio_metadata = await upload_file_to_minio(file)
                
                # Create a row on the files table
                new_file = File(
                    name=file.filename,
                    # file_metadata={"path":file_path}
                    file_metadata=minio_metadata
                )
                session.add(new_file)
                staged_files_db.append(new_file)
                # session.commit()

                logger.debug(f"{file.filename} saved  in {upload_dir}")
            except Exception as e:
                raise RuntimeError((f"Failed to save {file.filename}: {str(e)}"))
            
        # commit to close the session
        session.commit()
        # Build mapping: filename -> id
        for f in staged_files_db:
            file_id_map[f.name] = f.id
          
        # download files from Minio to local storage TEMP_FILE_DOWNLOAD_DIR
        for f in staged_files_db:
            try:
                download_file_from_minio(f.file_metadata["object_name"], TEMP_FILE_DOWNLOAD_DIR)
            except:
                logger.error(f"Failed to download file {f.file_metadata['object_name']} from Minio")
                raise 
        # chunking in indexing pipeline
        try:
            pipeline.process_file(save_file_path=upload_dir, file_id_map=file_id_map)
        except Exception as e:
            raise RuntimeError(f"Pipeline processing failed: {str(e)}")
        
        return {"status": "success", "files_uploaded": len(files)}
    
    except Exception as e:
        logger.exception("Error during file upload or processing — rolling back")
        session.rollback()  # rollback DB if anything fails
        return {"status": "failed", "error": str(e)}
    # finally block will always run. In case of successful file processing and any exceptions.
    finally:
        # --- 5. Cleanup local temp folder ---
        shutil.rmtree(upload_dir, ignore_errors=True)
    
@app.delete("/delete-file/{filename}")
def delete_file(filename: str, db: Session = Depends(get_db)):
    file = db.query(File).filter_by(name=filename).first()
    if not file:
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found."
        )

    # This will cascade delete all related docstore rows
    db.delete(file)
    db.commit()

    return {"message": f"File '{filename}' and related docstore entries deleted."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "upload_service:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
