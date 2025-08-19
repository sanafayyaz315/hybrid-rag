from fastapi import FastAPI, UploadFile, File
import os
from rag_pipeline import RagPipeline
from config import API_KEY
import logging
import shutil
from typing import List
from config import (
    COLLECTION,
    PARENT_CHUNKS_FILE, 
    PARENT_CHUNKS_DIR, 
    PARENT_CHUNK_SIZE, 
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
    UPSERT_BATCH_SIZE
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
    upsert_batch_size=UPSERT_BATCH_SIZE
)

@app.post("/upload-files/")
async def upload_files(files: List[UploadFile] = File(...)):
    upload_dir = "/tmp/uploaded_docs"
    os.makedirs(upload_dir, exist_ok=True)

    saved_files = []
    try:
        # loading and saving files to a local path
        for file in files:
            # try except will raise error if any file fails to save
            try:
                file_path = os.path.join(upload_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                    saved_files.append(file_path)
                logger.debug(f"{file.filename} saved  in {upload_dir}")
            except Exception as e:
                raise RuntimeError((f"Failed to save {file.filename}: {str(e)}"))
    
        # chunking in indexing pipeline
        try:
            pipeline.upload_file(save_file_path=upload_dir)
        except Exception as e:
            raise RuntimeError(f"Pipeline processing failed: {str(e)}")
        
        return {"status": "success", "files_uploaded": len(files)}
    
    except Exception as e:
        logger.exception("Error during file upload or processing — rolling back")
        # delete any successfully saved files to get a clean dir
        try:
            for file_path in saved_files:
                os.remove(file_path)
                logger.debug(f"deleted files {file_path}")
        except OSError as cleanup_error:
                logger.warning(f"Failed to delete {file_path}: {cleanup_error}")
        
        # delete and recreate the upload_dir directory to get a clean slate
        try:
            shutil.rmtree(upload_dir)
            os.makedirs(upload_dir, exist_ok=True)  # recreate empty folder
        except OSError as cleanup_error:
            logger.warning(f"Failed to reset upload directory: {cleanup_error}")

        return {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "upload_service:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
