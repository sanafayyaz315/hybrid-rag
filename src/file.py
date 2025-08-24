import os, shutil, logging
from typing import List, Dict
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from models import File 
from minio_utils import upload_file_to_minio, download_file

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
                        file_metadata=minio_metadata
                    )
        session.add(new_file)
        session.flush()
        return new_file
    except Exception as e:
        raise RuntimeError((f"Failed to add a row for {minio_metadata['object_name']} in table files: {str(e)}"))
    
# Download new file to local dir for indexing
def download_file_from_minio(minio_metadata: Dict, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    try:
        download_file(minio_metadata['object_name'], dest_dir)
    except:
        logger.error(f"Failed to download file {minio_metadata['object_name']} from Minio")
        raise HTTPException(status_code=500, detail="Download from MinIO failed")

#  Run pipeline (chunk + index)
def run_indexing_pipeline(session: Session, dest_dir: str, new_file: "File", rag_pipeline) -> None:
    file_id_map = {new_file.name: new_file.id}
    try:
        rag_pipeline.process_file(session=session, save_file_path=dest_dir, file_id_map=file_id_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    
# Delete files from docstore
def delete_from_docstore(filename: str, db: Session):
    try:
        file = db.query(File).filter_by(name=filename).first()
        if not file:
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found."
            )

        # This will cascade delete all related docstore rows
        db.delete(file)
        db.commit()
        return True
    except Exception as e:
        logger.exception(f"File {filename} failed due to: {e}")
        raise e
        
# List files
def list_files(session):
    rows = session.query(File).all()
    files = []
    for r in rows:
        files.append(r.name)
    return files


    

    




     
    
    
    


    
